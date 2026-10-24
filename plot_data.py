import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- Settings ----
DATA_FILE = "data/rotorua_data.csv"
OUTPUT_DIR = "Plots"
COLOUR = "tab:blue"
FONT_SIZE = 13

# Figure sized to the text width of an A4 Word page with standard 2.54 cm
# margins (~6.3 in), so fonts appear at their true size in the document.
FIG_SIZE = (6.3, 2.9)

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv(DATA_FILE)
df = df[df["year"] >= 1950]

# Pressure has gaps, so drop missing values for that plot only
pressure = df.dropna(subset=["pressure_[MPa]"])

# ---- Plot 1: Extraction ----
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
ax1.plot(df["year"], df["extraction_[tonnes/day]"], color=COLOUR, linewidth=2.5)
ax1.set_xlabel("Year")
ax1.set_ylabel("Extraction [tonnes/day]")
ax1.set_title("Reservoir Extraction Over Time")
ax1.set_xlim(left=1950)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "extraction_plot.png"), dpi=300)

# ---- Plot 2: Pressure ----
fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
ax2.plot(pressure["year"], pressure["pressure_[MPa]"], color=COLOUR,
         linewidth=2.5, marker="o", markersize=5)
ax2.set_xlabel("Year")
ax2.set_ylabel("Pressure [MPa]")
ax2.set_title("Reservoir Pressure Over Time")
ax2.set_xlim(left=1950)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "pressure_plot.png"), dpi=300)

plt.show()