import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from simulator import make_q, solve_model

# ---- settings you may want to change --------------------------------------
DATA_FILE = "data/rotorua_data.csv"
free = ["A_res", "phi", "k", "A", "L", "P0", "M0", "Rth", "Thot"]
bounds = {}     # physical units, e.g. {"phi": (0.1, 0.3), "L": (100, 300)}
SIGMA_P = 0.002  # MPa, assumed pressure scatter (sets the weight of pressure)
SIGMA_T = 3.0    # degC, assumed temperature scatter (sets the weight of T)

SEC_PER_YEAR = 365.25 * 86400
TDAY_TO_KGS = 1000.0 / 86400.0      # tonnes/day -> kg/s

# ---- data (converted to SI: s since first year, Pa, kg/s, degC) ------------
df = pd.read_csv(DATA_FILE)
year = df["year"].to_numpy(float)
t_data = (year - year[0]) * SEC_PER_YEAR
q = make_q(t_data, df["extraction_[tonnes/day]"].to_numpy() * TDAY_TO_KGS)

P_obs = df["pressure_[MPa]"].to_numpy() * 1e6        # NaN where not measured
T_obs = df["temperature_[degC]"].to_numpy()          # NaN where not measured
has_P, has_T = ~np.isnan(P_obs), ~np.isnan(T_obs)

# Intermediate constants, only used to build the dict below
k_therm = 2.5       # W m^-1 K^-1, thermal conductivity (table)
L_cond = 200.0      # m, conduction distance, table range 100-300
A_area = 2.0e7     # m^2, reservoir area, table range 1.5-2.8e7 (midpoint)
 
pars_initial = {
    # --- given in the model PDF ---
    "Tc":    30.0,        # C, shallow cold-water temperature
 
    # --- from the check-in 1 table ---
    "g":     9.81,        # m s^-2
    "A_res": A_area,      # m^2, reservoir area, range 1.5-2.8e7
    "A":     A_area,      # m^2, recharge area, taken equal to A_res
    "rho":   995.65,      # kg m^-3, water at 30 C (rho_cold)
    "mu":    7.97e-4,     # Pa s, water at 30 C (mu_cold)
    "c":     3750.0,      # J kg^-1 K^-1, specific heat of brine (FLAG 2)
    "phi":   0.2,         # porosity, range 0.1-0.3
    "k":     1.0e-13,     # m^2, permeability, range 15-200e-15 (geometric mean)
    "L":     200.0,       # m, cold recharge distance, range 100-300
    "P0":    1.4e6,       # Pa, natural pressure, table range 1.4-2e6
    "M0":    5.0e11,      # kg, reservoir mass, table range 1e11-1e12
 
    # --- read from the data / derived (FLAGS) ---
    "Thot":  150.0,                         # C (FLAG 3)
    "Rth":   L_cond / (k_therm * A_area),   # K W^-1, ~3.7e-6 (FLAG 1)
}

def simulate(pars, t_eval):
    # tight tolerances keep the numerical noise below the finite-difference step
    return solve_model(q, t_eval[0], t_eval[-1], pars, t_eval,
                       max_step=SEC_PER_YEAR, rtol=1e-10, atol=1e-8)

# ---- optimisation in log-space: theta = log(parameter / initial value) -----
def unpack(theta):
    pars = dict(pars_initial)
    for name, val in zip(free, theta):
        pars[name] = pars_initial[name] * np.exp(val)
    return pars


def residuals(theta):
    t, P, T = simulate(unpack(theta), t_data)
    if len(t) != len(t_data) or not (np.isfinite(P).all() and np.isfinite(T).all()):
        return np.full(has_P.sum() + has_T.sum(), 1e3)   # failed solve: penalise
    r_P = (P[has_P] - P_obs[has_P]) / 1e6 / SIGMA_P
    r_T = (T[has_T] - T_obs[has_T]) / SIGMA_T
    return np.concatenate([r_P, r_T])


def jacobian(theta, h=1e-3):
    """Forward differences with an absolute step h in log-space (scipy's
    built-in relative step breaks down when theta is near zero)."""
    r0 = residuals(theta)
    J = np.zeros((len(r0), len(theta)))
    for i in range(len(theta)):
        th = theta.copy()
        th[i] += h
        J[:, i] = (residuals(th) - r0) / h
    return J


lo = np.full(len(free), -np.inf)
hi = np.full(len(free), np.inf)
for i, name in enumerate(free):
    if name in bounds:
        lo[i] = np.log(bounds[name][0] / pars_initial[name])
        hi[i] = np.log(bounds[name][1] / pars_initial[name])
theta0 = np.clip(np.zeros(len(free)), lo, hi)

fit = least_squares(residuals, theta0, jac=jacobian, bounds=(lo, hi),
                    method="trf", max_nfev=200, verbose=1)
cal = unpack(fit.x)

# ---- report ---------------------------------------------------------------
print("\nsuccess:", fit.success, "-", fit.message)
print(f"final cost: {fit.cost:.3f}   (evaluations: {fit.nfev})")
print(f"\n{'param':<7}{'initial':>13}{'calibrated':>13}   at bound?")
for i, name in enumerate(free):
    flag = {-1: "lower", 1: "upper"}.get(int(fit.active_mask[i]), "")
    print(f"{name:<7}{pars_initial[name]:>13.4g}{cal[name]:>13.4g}   {flag}")
sv = np.linalg.svd(fit.jac, compute_uv=False)
print("\nJacobian singular values (relative to largest):")
print(np.array2string(sv / sv[0], precision=2))
print("Values near zero mean directions the data cannot constrain.")

# ---- plot calibrated model vs data ----------------------------------------
t_plot = np.linspace(0, t_data[-1], 4 * (len(t_data) - 1) + 1)
tp, P_cal, T_cal = simulate(cal, t_plot)
yr_plot = year[0] + tp / SEC_PER_YEAR

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(yr_plot, P_cal / 1e6, label="calibrated model")
ax1.plot(year[has_P], P_obs[has_P] / 1e6, "k.", label="data")
ax1.set_ylabel("reservoir pressure [MPa]")
ax1.legend()
ax2.plot(yr_plot, T_cal, label="calibrated model")
ax2.plot(year[has_T], T_obs[has_T], "k.", label="data")
ax2.set_ylabel("reservoir temperature [°C]")
ax2.set_xlabel("year")
ax2.legend()
ax1.set_xlim(year[0], year[-1])
plt.tight_layout()
plt.show()