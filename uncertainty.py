"""
Uncertainty analysis for the calibrated geyser model.

1. Summarise the parameter uncertainty (variances and correlations from Gamma).
2. Sample parameter sets from the Gaussian approximation of the posterior.
3. Plot the Gaussian posterior contours: every parameter pair (2x3 grid), then
   just a vs b and c vs d (1x2 grid).
4. Forecast under each extraction scenario with every sample, and plot the ensembles
   with all scenarios overlaid (one colour each): one pressure plot, one temperature plot.
5. Report 90% prediction intervals (5th to 95th percentile of the samples).

The pressure and temperature stages were calibrated separately, so their
parameters are sampled independently.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from simulator import make_q
from calibration import (df, year, t_data, SEC_PER_YEAR, TDAY_TO_KGS,
                                P_obs, T_obs, has_P, has_T,
                                simulate, to_sim_pars, compound_initial,
                                free_P, free_T, fit_P, cov_P, fit_T, cov_T, cal)

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
})

# ---- settings you may want to change --------------------------------------
N_SAMPLES = 100
END_YEAR = 2050             # forecast horizon (must be after the last data year)
REPORT_YEAR = END_YEAR      # year at which the prediction intervals are reported
SCENARIOS = {               # future extraction as a multiple of the last recorded rate
    "no extraction": 0.0,   # edit these to match the scenarios in your brief
    "current rate": 1.0,
    "increased rate (1.5x)": 1.5,
}

# ---- 1. parameter uncertainty ---------------------------------------------
def summarise(label, names, theta_star, cov):
    """Standard deviations of theta = log(param) are approximately the
    fractional uncertainties of the parameters."""
    sd = np.sqrt(np.diag(cov))
    print(f"\n{label}")
    print(f"{'param':<7}{'best fit':>12}{'± %':>8}")
    for n, th, s in zip(names, theta_star, sd):
        print(f"{n:<7}{compound_initial[n] * np.exp(th):>12.4g}{100 * s:>8.1f}")
    print("correlation matrix:")
    print(np.round(cov / np.outer(sd, sd), 2))


summarise("Pressure parameters", free_P, fit_P.x, cov_P)
summarise("Temperature parameters", free_T, fit_T.x, cov_T)

# ---- 2. sample the posterior ----------------------------------------------
# Sampling in log-space means every sampled parameter is positive.
rng = np.random.default_rng(0)
th_P = rng.multivariate_normal(fit_P.x, cov_P, N_SAMPLES)
th_T = rng.multivariate_normal(fit_T.x, cov_T, N_SAMPLES)

samples = []
for tp, tt in zip(th_P, th_T):
    cp = {n: compound_initial[n] * np.exp(v) for n, v in zip(free_P, tp)}
    cp.update({n: compound_initial[n] * np.exp(v) for n, v in zip(free_T, tt)})
    samples.append(to_sim_pars(cp))

# ---- 3. posterior contours ------------------------------------------------
ALL_PAIRS = [(0, 1), (0, 2), (1, 2)]


def plot_pairs(names, theta_star, cov, axes_row, pairs=ALL_PAIRS):
    """Gaussian posterior contours, marginalised onto each pair of parameters."""
    for ax, (i, j) in zip(axes_row, pairs):
        ni, nj = names[i], names[j]
        mean = theta_star[[i, j]]
        c2 = cov[np.ix_([i, j], [i, j])]
        sd = np.sqrt(np.diag(c2))
        gi = np.linspace(mean[0] - 3 * sd[0], mean[0] + 3 * sd[0], 100)
        gj = np.linspace(mean[1] - 3 * sd[1], mean[1] + 3 * sd[1], 100)
        GI, GJ = np.meshgrid(gi, gj)
        pdf = multivariate_normal(mean, c2).pdf(np.dstack((GI, GJ)))
        ax.contourf(compound_initial[ni] * np.exp(GI), compound_initial[nj] * np.exp(GJ),
                    pdf, levels=15, cmap="Blues")
        ax.plot(compound_initial[ni] * np.exp(mean[0]), compound_initial[nj] * np.exp(mean[1]),
                marker="*", color="gold", markeredgecolor="k", markersize=14,
                linestyle="none", label="best fit")
        ax.set_xlim(compound_initial[ni] * np.exp(gi[0]), compound_initial[ni] * np.exp(gi[-1]))
        ax.set_ylim(compound_initial[nj] * np.exp(gj[0]), compound_initial[nj] * np.exp(gj[-1]))
        ax.set_xlabel(ni)
        ax.set_ylabel(nj)


# every parameter pair: pressure stage on the top row, temperature stage below
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
plot_pairs(free_P, fit_P.x, cov_P, axes1[0])
plot_pairs(free_T, fit_T.x, cov_T, axes1[1])
axes1[0, 0].legend()
fig1.suptitle("Posterior (Gaussian approximation): all parameter pairs")
fig1.tight_layout()

# a vs b and c vs d only
fig1b, axes1b = plt.subplots(1, 2, figsize=(10, 4.5))
plot_pairs(free_P, fit_P.x, cov_P, [axes1b[0]], pairs=[(0, 1)])
plot_pairs(free_T, fit_T.x, cov_T, [axes1b[1]], pairs=[(0, 1)])
axes1b[0].legend()
fig1b.suptitle("Posterior (Gaussian approximation): a vs b and c vs d")
fig1b.tight_layout()

FONT_SIZE = 14
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE + 3,
    "axes.titlesize": FONT_SIZE + 6,
    "xtick.labelsize": FONT_SIZE + 4,
    "ytick.labelsize": FONT_SIZE + 4,
    "legend.fontsize": FONT_SIZE - 2,
})

# ---- 4. forecast each scenario with every sample --------------------------
extraction_tday = df["extraction_[tonnes/day]"].to_numpy()
extraction_kgs = extraction_tday * TDAY_TO_KGS
q_last_tday = extraction_tday[-1]

t_end = (END_YEAR - year[0]) * SEC_PER_YEAR
t_f = np.linspace(0, t_end, 400)
yr_f = year[0] + t_f / SEC_PER_YEAR
i_rep = np.argmin(np.abs(yr_f - REPORT_YEAR))


def scenario_q(factor):
    """Recorded extraction, then `factor` times the last recorded rate.
    (Check how make_q interpolates: the extra point one day after the last
    record makes the new rate take effect almost immediately.)"""
    t_s = np.concatenate([t_data, [t_data[-1] + 86400.0, t_end]])
    q_s = np.concatenate([extraction_kgs, [factor * q_last_tday * TDAY_TO_KGS] * 2])
    return make_q(t_s, q_s)


SCENARIO_COLOURS = ["tab:blue", "tab:orange", "tab:red"]   # one per scenario, in order

# Forecast portion starts one point before the last data year so the lines join the history.
i0 = max(np.searchsorted(yr_f, year[-1]) - 1, 0)
fut = slice(i0, None)
hist = slice(0, i0 + 1)

fig2, (axP, axT) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
results = {}
for k, (name, factor) in enumerate(SCENARIOS.items()):
    colour = SCENARIO_COLOURS[k % len(SCENARIO_COLOURS)]
    q_f = scenario_q(factor)
    runs = [simulate(s, t_f, q_f) for s in samples]
    runs = [r for r in runs if len(r[0]) == len(t_f)]          # drop failed solves
    P_ens = np.array([r[1] for r in runs]) / 1e6               # MPa
    T_ens = np.array([r[2] for r in runs])                     # degC
    _, P_best, T_best = simulate(cal, t_f, q_f)
    results[name] = (P_ens, T_ens)
    label = f"{name}"

    for ax, ens, best in [(axP, P_ens, P_best / 1e6), (axT, T_ens, T_best)]:
        ax.plot(yr_f[fut], ens[:, fut].T, color=colour, linewidth=0.5, alpha=0.15)
        ax.plot(yr_f[fut], best[fut], color=colour, linewidth=2, label=label)
        if k == 0:   # the history is the same in every scenario, so draw it once
            ax.plot(yr_f[hist], best[hist], color="k", linewidth=1.5,
                    label="best fit (history)")

axP.plot(year[has_P], P_obs[has_P] / 1e6, "k.", label="data")
axT.plot(year[has_T], T_obs[has_T], "k.", label="data")
for ax in (axP, axT):
    ax.axvline(year[-1], color="grey", linestyle=":")

axP.set_ylabel("reservoir pressure [MPa]")
axT.set_ylabel("reservoir temperature [°C]")
axT.set_xlabel("year")
axP.legend()
fig2.tight_layout()

# ---- 5. 95% prediction intervals ------------------------------------------
print(f"\n95% prediction intervals in {yr_f[i_rep]:.0f} "
      f"(2.5th to 97.5th percentile of the samples)")
for name, (P_ens, T_ens) in results.items():
    P_lo, P_hi = np.percentile(P_ens[:, i_rep], [2.5, 97.5])
    T_lo, T_hi = np.percentile(T_ens[:, i_rep], [2.5, 97.5])
    print(f"{name:<15} pressure: {P_lo:.3f} to {P_hi:.3f} MPa   "
          f"temperature: {T_lo:.1f} to {T_hi:.1f} °C")

plt.show()