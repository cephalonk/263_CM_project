"""
Waikite Geyser: VERIFICATION of the pressure ODE solver.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulator import solve_pressure, pressure_rhs
from calibration import pars_initial

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

    fig.savefig("verif_zero_forcing.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 2. MMS BENCHMARK
# ---------------------------------------------------------------------------
# Manufactured solution: an oscillation that decays slowly, starting at P0.
#     P_m(t) = P0 + dP * sin(w t) * exp(-t / tau_d)
# Why this shape?
#   - smooth and analytic, so the exact derivative is available;
#   - starts at P0 (sin(0)=0) so the initial condition is the same P0 as in
#     the real model;
#   - oscillation period (2 yr) and decay (4 yr) are comparable to 1/lam, so
#     it exercises the same time scales as the real problem.
dP_M = 2.0e5                      # Pa, amplitude (a sensible fraction of P0)
W = 2 * np.pi / (2.0 * YEAR)      # rad/s, 2-year period
TAU_D = 4.0 * YEAR                # s, decay time


def P_m(t):
    return P0 + dP_M * np.sin(W * t) * np.exp(-t / TAU_D)


def dP_m_dt(t):
    """Exact analytic derivative of P_m (product rule). No finite differences."""
    e = np.exp(-t / TAU_D)
    return dP_M * e * (W * np.cos(W * t) - np.sin(W * t) / TAU_D)


# Smooth, analytic extraction rate (kg/s). We deliberately do NOT use the
# interpolated data here: interpolated q has kinks, which would test the data
# handling, not the solver. Kink handling is controlled by MAX_STEP instead.
Q0 = 500.0                        # kg/s, mean extraction


def q_mms(t):
    return Q0 * (1.0 + 0.5 * np.sin(2 * np.pi * t / YEAR))   # annual cycle


def source_mms(t):
    """s(t) = dP_m/dt - f(t, P_m).
    We call the SIMULATOR'S OWN pressure_rhs so the source is consistent with
    whatever the code actually does. This makes the test sensitive to errors
    in the solver loop and in how `source` is added, rather than relying on a
    second hand-typed copy of the equation."""
    return dP_m_dt(t) - pressure_rhs(t, P_m(t), q_mms, pars)


def run_mms(rtol=RTOL, atol=ATOL, max_step=MAX_STEP):
    t, P = solve_pressure(q_mms, t0, t1, pars, t_eval, max_step,
                          P_init=P_m(t0), source=source_mms,
                          rtol=rtol, atol=atol)
    return t, P, np.abs(P - P_m(t))


def plot_mms(t, P, err):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(t / YEAR, P / 1e6, "C0", lw=3, label="simulator")
    ax[0].plot(t / YEAR, P_m(t) / 1e6, "k--", label="manufactured $P_m$")
    ax[0].set(xlabel="time (years)", ylabel="P (MPa)",
              title="MMS: numerical vs manufactured")
    ax[0].legend()

    ax[1].semilogy(t / YEAR, np.maximum(err, 1e-16), "C3")
    ax[1].set(xlabel="time (years)", ylabel="|P - $P_m$| (Pa)",
              title="MMS: error vs time")

    # Show the forcing terms so the reader sees the test is non-trivial:
    # the source must be comparable in size to the physical terms.
    ax[2].plot(t / YEAR, -a * q_mms(t), label="$-a\\,q(t)$ (extraction)")
    ax[2].plot(t / YEAR, source_mms(t), label="$s(t)$ (manufactured source)")
    ax[2].set(xlabel="time (years)", ylabel="Pa / s",
              title="Terms in the right-hand side")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig("verif_mms.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 3. CONVERGENCE (the "time step convergence" item)
# ---------------------------------------------------------------------------
# solve_ivp is ADAPTIVE: you do not set a step size, you set a tolerance and
# it picks steps to meet it. So the honest convergence test is
# "max error vs rtol". If the code is right, tightening rtol must shrink the
# error roughly in proportion, until round-off (about 1e-16 * P0 ~ 1e-10 Pa)
# takes over.
#
# max_step is set to infinity here so that rtol, not the step cap, controls
# the step size. Otherwise the error would flatten at the cap.
def run_convergence():
    rtols = 10.0 ** np.arange(-3, -12, -1.0)
    max_errs = []
    for rt in rtols:
        # atol is tiny so rtol dominates the error control.
        _, _, e = run_mms(rtol=rt, atol=1e-10, max_step=np.inf)
        max_errs.append(e.max())
    return rtols, np.array(max_errs)


def plot_convergence(rtols, max_errs):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(rtols, max_errs, "o-", label="max error (MMS)")
    ax.loglog(rtols, rtols * dP_M, "k--", lw=1, label="slope 1 reference")
    ax.invert_xaxis()                       # tighter tolerance to the right
    ax.set(xlabel="rtol", ylabel="max |P - $P_m$| (Pa)",
           title="MMS: error vs solver tolerance")
    ax.legend()
    fig.tight_layout()
    fig.savefig("verif_convergence.png", dpi=150)
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

    rtols, max_errs = run_convergence()
    plot_convergence(rtols, max_errs)

    # Fit the slope only where the error is above the round-off floor.
    good = max_errs > 1e-7
    slope = np.polyfit(np.log10(rtols[good]), np.log10(max_errs[good]), 1)[0]

    print(f"{'test':<40}{'value':>12}{'limit':>10}   result")
    for name, val, lim in results:
        print(f"{name:<40}{val:>12.2e}{lim:>10.0e}   "
              f"{'PASS' if val < lim else 'FAIL'}")
    print(f"\nconvergence slope (log error vs log rtol): {slope:.2f} "
          f"(about 1 expected)")
    print("Figures saved: verif_zero_forcing.png, verif_mms.png, "
          "verif_convergence.png")

    plt.show()


if __name__ == "__main__":
    main()