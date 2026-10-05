"""
Waikite Geyser / Rotorua geothermal field: simulator (solve_ivp).

  dP/dt = g / (A_res * phi) * [ -q(t) - (rho*k*A / (mu*L)) * (P - P0) ]

  dT/dt = rho*k*A / (mu*L*M0) * (P0 - P) * (Tc - T)
          + 1 / (M0 * c * Rth) * (Thot - T)

Initial conditions: P(t0) = P0, T(t0) = Thot.

UNITS: no unit conversion is done here. Pass everything in one consistent
set (e.g. SI: P in Pa, q in kg/s, t in s) and convert for plotting.

PARAMETERS: a dict `pars` with keys
  g, A_res, phi, rho, k, A, mu, L, P0, M0, c, Rth, Tc, Thot

INTERFACE
  make_q(t_data, q_data)                     -> callable q(t)
  solve_pressure(q, t0, t1, pars, t_eval, max_step, ...)     -> (t, P)
  solve_temperature(P, t0, t1, pars, t_eval, max_step, ...)  -> (t, T)
  solve_model(q, t0, t1, pars, t_eval, max_step, ...)        -> (t, P, T)

  solve_pressure / solve_temperature accept an optional source(t), which is
  added to the right-hand side of the ODE (used for MMS verification) and
  optional initial values P_init / T_init (default P0 / Thot).
  max_step should be <= the spacing of the q data so kinks are not skipped.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Right-hand sides (the equations exactly as given)
# ---------------------------------------------------------------------------
def pressure_rhs(t, P, q, pars):
    """dP/dt. q is a callable q(t). pars is a dict of parameters."""
    recharge = pars["rho"] * pars["k"] * pars["A"] / (pars["mu"] * pars["L"]) \
        * (P - pars["P0"])
    return pars["g"] / (pars["A_res"] * pars["phi"]) * (-q(t) - recharge)


def temperature_rhs(t, T, P, pars):
    """dT/dt. P is the pressure at time t (a number)."""
    cold_recharge = pars["rho"] * pars["k"] * pars["A"] \
        / (pars["mu"] * pars["L"] * pars["M0"]) \
        * (pars["P0"] - P) * (pars["Tc"] - T)
    hot_conduction = 1.0 / (pars["M0"] * pars["c"] * pars["Rth"]) \
        * (pars["Thot"] - T)
    return cold_recharge + hot_conduction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_q(t_data, q_data):
    """Linear interpolation of extraction data into a callable q(t)."""
    return lambda t: np.interp(t, t_data, q_data)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def solve_pressure(q, t0, t1, pars, t_eval, max_step, P_init=None,
                   source=None, rtol=1e-8, atol=1e-6):
    """Pressure ODE on its own. Returns (t, P)."""
    P_init = pars["P0"] if P_init is None else P_init

    def f(t, y):
        return pressure_rhs(t, y[0], q, pars) + (source(t) if source else 0.0)

    sol = solve_ivp(f, (t0, t1), [P_init], t_eval=t_eval, method="RK45",
                    max_step=max_step, rtol=rtol, atol=atol)
    return sol.t, sol.y[0]


def solve_temperature(P, t0, t1, pars, t_eval, max_step, T_init=None,
                      source=None, rtol=1e-8, atol=1e-6):
    """Temperature ODE on its own, given P as a callable P(t) or a constant.
    Returns (t, T)."""
    T_init = pars["Thot"] if T_init is None else T_init

    def f(t, y):
        Pt = P(t) if callable(P) else P
        return temperature_rhs(t, y[0], Pt, pars) + (source(t) if source else 0.0)

    sol = solve_ivp(f, (t0, t1), [T_init], t_eval=t_eval, method="RK45",
                    max_step=max_step, rtol=rtol, atol=atol)
    return sol.t, sol.y[0]


def solve_model(q, t0, t1, pars, t_eval, max_step, rtol=1e-8, atol=1e-6):
    """Full model: P and T solved together as one system y = [P, T].
    Returns (t, P, T)."""
    def f(t, y):
        return [pressure_rhs(t, y[0], q, pars),
                temperature_rhs(t, y[1], y[0], pars)]

    sol = solve_ivp(f, (t0, t1), [pars["P0"], pars["Thot"]], t_eval=t_eval,
                    method="RK45", max_step=max_step, rtol=rtol, atol=atol)
    return sol.t, sol.y[0], sol.y[1]
