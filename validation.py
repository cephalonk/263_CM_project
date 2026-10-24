import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulator import make_q, solve_pressure

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
})

HERE = Path(__file__).resolve().parent
DATA = HERE / "benchmark_reservoir.csv"

SECONDS_PER_YEAR = 365.25 * 24 * 3600
ML_PER_DAY_TO_KG_PER_S = 1e6 / 86400.0
MPA_TO_PA = 1e6

# Benchmark parameters (given in the brief)
pars = {
    "g":     9.81,
    "A_res": 5.0e7,
    "phi":   0.14,
    "rho":   1000.0,
    "k":     6.9e-14,
    "A":     5.0e7,
    "mu":    1.0e-3,
    "L":     1.0e3,
    "P0":    0.050e6,
    "M0":    5e11,
    "c":     3750.0,
    "Rth":   3.7e-6,
    "Tc":    30.0,
    "Thot":  150.0,
}

P_INIT = 0.060e6   # P(1985), given separately from P0

# Load benchmark data
raw = np.genfromtxt(DATA, delimiter=",", skip_header=1)
years = raw[:, 0]
extraction_ml_day = raw[:, 1]
pressure_mpa = raw[:, 2]

t_data = (years - years[0]) * SECONDS_PER_YEAR
q = make_q(t_data, extraction_ml_day * ML_PER_DAY_TO_KG_PER_S)

has_p = ~np.isnan(pressure_mpa)

# Run simulator with benchmark parameters
t_eval = np.linspace(t_data[0], t_data[-1], 2000)
t_sim, P_sim = solve_pressure(q, t_data[0], t_data[-1], pars,
                              t_eval=t_eval, max_step=SECONDS_PER_YEAR,
                              P_init=P_INIT)
P_sim_mpa = P_sim / MPA_TO_PA

# Misfit at the measured years
P_sim_at_obs = np.interp(t_data[has_p], t_sim, P_sim) / MPA_TO_PA
rms = np.sqrt(np.mean((P_sim_at_obs - pressure_mpa[has_p]) ** 2))
print(f"RMS misfit: {rms:.4f} MPa")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(years[0] + t_sim / SECONDS_PER_YEAR, P_sim_mpa,
        color="tab:orange", label="Simulated (benchmark parameters)")
ax.scatter(years[has_p], pressure_mpa[has_p], color="tab:blue",
           zorder=3, label="Benchmark data")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("Year")
ax.set_ylabel("Reservoir pressure (MPa)")
ax.set_title("Reservoir model validation")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "validation_pressure.png", dpi=120)
plt.show()