"""
Figures of Section 5.2 into Thesis/figures/.

All figures show MULTIFIT with different subroutines on the same dataset -- the
comparison of the subroutines in use, not of the subroutines on their own.
Style and localization as in thesis_figures.py.

  abb_hm_guete.pdf      empirical distribution function of C_MF/OPT per arm
  abb_hm_paare.pdf      win/tie/loss of the arms against MULTIFIT+FFD
  abb_hm_laufzeit.pdf   runtime per arm over d (box plot, log y)
  abb_hm_skalierung.pdf runtime per arm over n at fixed n/m (log-log)

Usage:
  python Implementation/Empirical_Tests/thesis_figures_hm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Empirical_Tests.thesis_figures import localize, de_log_formatter

CSV_ROOT = Path(__file__).parent
OUT_DIR = Path(__file__).parents[2] / "Thesis" / "figures"
DATASET = "jansen_hm"
SCALE_DATASET = "jansen_hm_scale"
BOUND = 13 / 11
TEXT_WIDTH = 5.9

ARM_ORDER = ["ffd", "mffd", "jansen", "jansen_dual", "gg_lp", "gg_ip"]
ARM_LABEL = {"ffd": "FFD", "mffd": "MFFD", "jansen": "Jansen OPT+1",
             "jansen_dual": "Jansen $\\varepsilon$-dual",
             "gg_lp": "GG-LP", "gg_ip": "GG-IP (exakt)"}
ARM_COLOR = {"ffd": "#4C72B0", "mffd": "#DD8452", "jansen": "#55A868",
             "jansen_dual": "#8172B3", "gg_lp": "#C44E52", "gg_ip": "#937860"}
COL_WIN = "#2CA02C"
COL_TIE = "#B8C4D4"
COL_LOSS = "#C44E52"
COL_LINE = "#333333"
BASELINE = "ffd"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.linestyle": ":", "grid.linewidth": 0.5,
    "grid.alpha": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load():
    df = pd.read_csv(CSV_ROOT / DATASET / "benchmark_hm.csv")
    df["opt"] = pd.to_numeric(df["opt"], errors="coerce")
    arms = [a for a in ARM_ORDER if f"makespan_{a}" in df.columns]
    for a in arms:
        df[f"makespan_{a}"] = pd.to_numeric(df[f"makespan_{a}"], errors="coerce")
        df[f"ratio_opt_{a}"] = df[f"makespan_{a}"] / df["opt"]
    return df, arms


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    localize(fig)
    fig.savefig(OUT_DIR / name)
    plt.close(fig)
    print(f"  -> Thesis/figures/{name}")


def fig_guete(df, arms):
    d_opt = df[df.opt.notna() & ~df.degeneriert]
    fig, ax = plt.subplots(figsize=(0.78 * TEXT_WIDTH * 1.28, 3.0))
    for a in arms:
        v = np.sort(d_opt[f"ratio_opt_{a}"].dropna().values)
        if len(v) == 0:
            continue
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                label=ARM_LABEL[a], color=ARM_COLOR[a], linewidth=1.2)
    ax.axvline(BOUND, color=COL_LINE, linestyle="--", linewidth=0.8)
    # label left of the line, otherwise it runs into the legend
    ax.annotate("$13/11$", xy=(BOUND, 0.30), xytext=(-4, 0), ha="right",
                textcoords="offset points", fontsize=8, color=COL_LINE)
    ax.set_xlabel("$C^{\\mathrm{MF}}_{\\max}\\,/\\,C^{*}_{\\max}$")
    ax.set_ylabel("Anteil der Instanzen")
    ax.set_ylim(0, 1.04)
    ax.legend(loc="center right", framealpha=0.95)
    save(fig, "abb_hm_guete.pdf")


def fig_paare(df, arms):
    candidates = [a for a in arms if a != BASELINE]
    fig, ax = plt.subplots(figsize=(0.72 * TEXT_WIDTH * 1.35, 2.6))
    y = np.arange(len(candidates))
    wins, ties, losses = [], [], []
    for a in candidates:
        sub = df.dropna(subset=[f"makespan_{a}", f"makespan_{BASELINE}"])
        w = int((sub[f"makespan_{a}"] < sub[f"makespan_{BASELINE}"]).sum())
        l = int((sub[f"makespan_{a}"] > sub[f"makespan_{BASELINE}"]).sum())
        wins.append(w); losses.append(l); ties.append(len(sub) - w - l)
    ax.barh(y, wins, color=COL_WIN, label="besser als FFD")
    ax.barh(y, ties, left=wins, color=COL_TIE, label="gleich wie FFD")
    ax.barh(y, losses, left=np.array(wins) + np.array(ties), color=COL_LOSS,
            label="schlechter als FFD")
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABEL[a] for a in candidates])
    ax.invert_yaxis()
    ax.set_xlabel("Instanzen")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", visible=False)
    save(fig, "abb_hm_paare.pdf")


def fig_laufzeit(df, arms):
    ds = sorted(df.d.unique())
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.2))
    width = 0.8 / len(arms)
    for j, a in enumerate(arms):
        data = [df[df.d == d][f"time_{a}_ms"].dropna().values for d in ds]
        pos = np.arange(len(ds)) + (j - (len(arms) - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                        showfliers=False, patch_artist=True,
                        medianprops={"color": COL_LINE, "linewidth": 0.9})
        for box in bp["boxes"]:
            box.set(facecolor=ARM_COLOR[a], alpha=0.85, linewidth=0.6)
        ax.plot([], [], color=ARM_COLOR[a], linewidth=4, alpha=0.85,
                label=ARM_LABEL[a])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(de_log_formatter())
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("Anzahl verschiedener Aufgabengrößen $d$")
    ax.set_ylabel("Laufzeit je Instanz [ms]")
    # legend below the axis: inside the plot it covers the d=6 boxes
    ax.legend(ncol=min(5, len(arms)), loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False, handlelength=1.4,
              columnspacing=1.2)
    save(fig, "abb_hm_laufzeit.pdf")


def fig_skalierung():
    # Runtime along the multiplicity axis, log-log.
    # n/m is fixed, so the capacity stays constant and only the multiplicities
    # grow. FFD shows up as a line with slope ~1, the high-multiplicity methods
    # stay nearly flat.
    path = CSV_ROOT / SCALE_DATASET / "benchmark_hm.csv"
    if not path.exists():
        print("  (jansen_hm_scale missing, abb_hm_skalierung skipped)")
        return
    df = pd.read_csv(path)
    arms = [a for a in ARM_ORDER if f"time_{a}_ms" in df.columns]
    ns = sorted(df.n.unique())
    fig, ax = plt.subplots(figsize=(0.8 * TEXT_WIDTH * 1.25, 3.1))
    for a in arms:
        med = [df[df.n == n][f"time_{a}_ms"].median() for n in ns]
        ax.plot(ns, med, marker="o", markersize=3.5, linewidth=1.2,
                color=ARM_COLOR[a], label=ARM_LABEL[a])
    ref = [df[df.n == ns[0]][f"time_{BASELINE}_ms"].median() * (n / ns[0]) for n in ns]
    ax.plot(ns, ref, linestyle="--", linewidth=0.8, color=COL_LINE, label="linear in $n$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(de_log_formatter())
    ax.yaxis.set_major_formatter(de_log_formatter())
    ax.set_xlabel("Anzahl der Aufgaben $n$ (bei festem $d \\in \\{3, 4\\}$ und $n/m = 8$)")
    ax.set_ylabel("Laufzeit je Instanz [ms]")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, handlelength=1.6, columnspacing=1.2)
    save(fig, "abb_hm_skalierung.pdf")


def main():
    df, arms = load()
    print(f"{len(df)} instances, arms: {' '.join(arms)}")
    fig_guete(df, arms)
    fig_paare(df, arms)
    fig_laufzeit(df, arms)
    fig_skalierung()


if __name__ == "__main__":
    main()
