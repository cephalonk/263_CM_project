"""
Waikite Geyser: VERIFICATION of the pressure ODE solver.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator import solve_pressure, pressure_rhs
from calibration import pars_initial

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
})

# ---------------------------------------------------------------------------
# 0. Setup: parameters and derived constants (all SI: Pa, kg/s, s)
# ---------------------------------------------------------------------------
pars = dict(pars_initial)          # copy so we never modify the original
P0 = pars["P0"]                    # Pa, natural pressure
YEAR = 365.25 * 24 * 3600          # s. Used ONLY to convert time for plotting

a = pars["g"] / (pars["A_res"] * pars["phi"])
lam = a * pars["rho"] * pars["k"] * pars["A"] / (pars["mu"] * pars["L"])
tau_res = 1.0 / lam                # natural relaxation time of the reservoir

# Run length: 10 years, which is several reservoir time scales (see printout).
t0, t1 = 0.0, 10.0 * YEAR
t_eval = np.linspace(t0, t1, 600)

# max_step is capped at one month, like the monthly spacing of real data.
# This stops the adaptive solver from taking one giant step across a kink in q.
MAX_STEP = YEAR / 12

# Default solver tolerances used for the pass/fail checks (same as simulator).
RTOL, ATOL = 1e-8, 1e-6

print(f"lambda = {lam:.3e} 1/s  ->  time scale 1/lambda = {tau_res/YEAR:.2f} years")
print(f"run length = {(t1 - t0)/YEAR:.1f} years\n")


# ZERO-FORCING BENCHMARK
# remove extraction rate to verify. thus q = 0

def q_zero(t):
    """No extraction at all."""
    return 0.0


def run_zero_forcing():
    # (a) Start exactly at equilibrium, no extraction: nothing should happen.
    #     RHS = a*(-0 - lam/a*(P0 - P0)) = 0 exactly, so we expect
    #     an error of exactly zero (or at worst round-off).
    # exact model is just line at P0
    t, P = solve_pressure(q_zero, t0, t1, pars, t_eval, MAX_STEP,
                          rtol=RTOL, atol=ATOL)
    err_a = np.abs(P - P0)

    # (b) Pressure starts away from eq. and converges to the equilibrium
    # model using solve pressures function
    dP0 = 2.0e5                                  # Pa, 0.2 MPa offset (~14% of P0)
    t_b, P_b = solve_pressure(q_zero, t0, t1, pars, t_eval, MAX_STEP,
                              P_init=P0 + dP0, rtol=RTOL, atol=ATOL)

    # exact model (see derivation)
    P_exact_b = P0 + dP0 * np.exp(-lam * (t_b - t0))
    err_b = np.abs(P_b - P_exact_b)

    return dict(t=t, P_a=P, err_a=err_a,
                t_b=t_b, P_b=P_b, P_exact_b=P_exact_b, err_b=err_b, dP0=dP0)


def plot_zero_forcing(r):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: test (a). The simulated line should lie exactly on P0.
    ax[0].plot(r["t"] / YEAR, r["P_a"] / 1e6, lw=2,label="simulator")
    ax[0].axhline(P0 / 1e6,ls="--",  color='r', label="exact: $P_0$")
    ax[0].set(xlabel="time (years)", ylabel="P (MPa)",
              title="(a) q = 0, start at $P_0$")
    ax[0].legend()
    ax[0].text(0.5, 0.15, f"max |P - P0| = {r['err_a'].max():.1e} Pa",
               transform=ax[0].transAxes, ha="center")

    # Panel 2: test (b). Simulator should overlay the exact exponential decay.
    ax[1].plot(r["t_b"] / YEAR, r["P_b"] / 1e6, "C0", lw=3, label="simulator")
    ax[1].plot(r["t_b"] / YEAR, r["P_exact_b"] / 1e6, "k--", label="exact")
    ax[1].axhline(P0 / 1e6, color="grey", ls=":", lw=1)
    ax[1].set(xlabel="time (years)", ylabel="P (MPa)",
              title="(b) q = 0, start at $P_0$ + 0.2 MPa")
    ax[1].legend()
    fig.tight_layout()

    fig.savefig("verif_zero_forcing.png", dpi=150)
    return fig

# 2. MMS BENCHMARK
# ---------------------------------------------------------------------------
# Manufactured solution: pressure DRAWDOWN D(t) = P0 - P_m(t) that grows from
# zero towards dP_M and wobbles with a seasonal-style oscillation:
#     D(t)   = dP_M * (1 - exp(-t / TAU_R)) * (1 + M * sin(W t))
#     P_m(t) = P0 - D(t)
# Why this shape?
#   - smooth and analytic, so the exact derivative is available;
#   - starts at P0 (D(0)=0), the same initial condition as the real model;
#   - pressure falls BELOW P0, as it does when fluid is extracted. Extraction
#     (q > 0) can only push pressure down, so a positive q_m requires this;
#   - the sine makes dP/dt change sign, and the exponential makes the response
#     build up like the real reservoir, with TAU_R comparable to 1/lam;
#   - the wobble is kept gentle (M = 0.2, 4-year period) so that the
#     oscillation never outruns the recharge term. Algebraically
#         a*q_m = D' + lam*D,
#     so q_m > 0 as long as lam*D > |D'|. This is checked by an assert below.
dP_M = 1.0e5                      # Pa, steady drawdown scale (~7% of P0)
TAU_R = 2.0 * YEAR                # s, build-up time (about 1/lam)
W = 2 * np.pi / (4.0 * YEAR)      # rad/s, 4-year period
M = 0.2                           # relative size of the oscillation
N_PTS = 40                        # output points for the MMS comparison


def D(t):
    return dP_M * (1.0 - np.exp(-t / TAU_R)) * (1.0 + M * np.sin(W * t))


def P_m(t):
    return P0 - D(t)


def dP_m_dt(t):
    """Exact analytic derivative of P_m (product rule). No finite differences."""
    g = 1.0 - np.exp(-t / TAU_R)
    dg = np.exp(-t / TAU_R) / TAU_R
    h = 1.0 + M * np.sin(W * t)
    dh = M * W * np.cos(W * t)
    return -dP_M * (dg * h + g * dh)

def q_manufactured(t):
    return (-dP_m_dt(t) - lam * (P_m(t) - P0)) / a


def run_mms(n_pts=N_PTS):
    # Solve with the simulator's own RK45 solver, driven by the manufactured q
    t_m = np.linspace(t0, t1, n_pts)
    t, P = solve_pressure(q_manufactured, t0, t1, pars, t_m, MAX_STEP,
                          P_init=P_m(t0), rtol=RTOL, atol=ATOL)
    return t, P, np.abs(P - P_m(t))


def plot_mms(t, P, err):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    circ = dict(marker="o", color="red", ls="none", ms=4, mfc="none")
    tf = np.linspace(t0, t1, 600)

    # Panel 1: numerical (red circles) vs manufactured (black line).
    ax[0].plot(tf / YEAR, P_m(tf) / 1e6, "k-", label="manufactured $P_m$")
    ax[0].plot(t / YEAR, P / 1e6, label="Simulator (RK45)", **circ)
    ax[0].set(xlabel="time (years)", ylabel="P (MPa)",
              title="MMS: numerical vs manufactured")
    ax[0].legend()


    # Panel 3: the manufactured extraction rate that drives the test.
    ax[1].plot(tf / YEAR, q_manufactured(tf), "C2")
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].set(xlabel="time (years)", ylabel="q (kg/s)",
              title="Manufactured extraction $q_m(t)$")
    fig.tight_layout()
    fig.savefig("verif_mms.png", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# 4. Run everything and print a pass/fail table
# ---------------------------------------------------------------------------
def main():
    results = []

    zf = run_zero_forcing()
    plot_zero_forcing(zf)
    results.append(("Zero-forcing (a): max |P-P0| [Pa]",
                    zf["err_a"].max(), 1e-6))
    results.append(("Zero-forcing (b): max err / dP0",
                    zf["err_b"].max() / zf["dP0"], 1e-6))

    t, P, err = run_mms()
    plot_mms(t, P, err)
    pressure_range = np.ptp(P_m(t))
    results.append(("MMS: max err / pressure range",
                    err.max() / pressure_range, 1e-6))


    print(f"{'test':<40}{'value':>12}   result")
    for name, val, lim in results:
        print(f"{name:<40}{val:>12.2e}  "
              f"{'PASS' if val < lim else 'FAIL'}")
    print("Figures saved: verif_zero_forcing.png, verif_mms.png")

    plt.show()


if __name__ == "__main__":
    main()