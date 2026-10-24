import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- Settings ----
DATA_FILE = "data/rotorua_data.csv"
OUTPUT_DIR = "Plots"
COLOUR = "tab:blue"
CLOSURE_YEAR = 1986      # start of borehole closure programme
SHOW_CLOSURE_LINE = True

# Width matches the text width of an A4 Word page with standard 2.54 cm
# margins (~6.3 in), so fonts appear at their true size in the document.
# Height is larger than before to fit three stacked panels.
FIG_SIZE = (6.3, 6.5)

FONT_SIZE = 14
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE + 3,
    "axes.titlesize": FONT_SIZE + 6,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv(DATA_FILE)
df = df[df["year"] >= 1950]

# Pressure and temperature have gaps (different sampling frequencies),
# so drop missing values separately for each quantity.
pressure = df.dropna(subset=["pressure_[MPa]"])
temperature = df.dropna(subset=["temperature_[degC]"])

# ---- Stacked plot: extraction, pressure, temperature ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True)

# Extraction (annual, continuous line)
ax1.plot(df["year"], df["extraction_[tonnes/day]"], color=COLOUR, linewidth=2.5)
ax1.set_ylabel("Extraction\n[tonnes/day]")

# Pressure (annual from 1985, line with markers)
ax2.plot(pressure["year"], pressure["pressure_[MPa]"], color=COLOUR,
         linewidth=2.5, marker="o", markersize=5)
ax2.set_ylabel("Pressure\n[MPa]")

# Temperature (5-yearly, line with markers)
ax3.plot(temperature["year"], temperature["temperature_[degC]"], color=COLOUR,
         linewidth=2.5, marker="o", markersize=5)
ax3.set_ylabel("Temperature\n[°C]")
ax3.set_xlabel("Year")

for ax, label in zip((ax1, ax2, ax3), ("(a)", "(b)", "(c)")):
    ax.grid(True, alpha=0.3)
    ax.text(0.01, 0.95, label, transform=ax.transAxes,
            va="top", ha="left", fontweight="bold")
    if SHOW_CLOSURE_LINE:
        ax.axvline(CLOSURE_YEAR, color="grey", linestyle="--", linewidth=1.5)

ax3.set_xlim(left=1950)

fig.align_ylabels([ax1, ax2, ax3])
fig.tight_layout()
fig.subplots_adjust(hspace=0.12)

plt.show()