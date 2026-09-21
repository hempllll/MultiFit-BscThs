"""
Generates the figures of Section 5.1 of the thesis into Thesis/figures/.

Style: serif font, font size 9, no plot titles (the caption carries the
description), German labels, width matched to \textwidth.

Usage:
  python Implementation/Empirical_Tests/thesis_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from binpacking.binpacking_algorithms import ffd_nlogn, mffd_nlogn
from Empirical_Tests.friesen_case_study import M as FRIESEN_M, OPT as FRIESEN_OPT, build_instance

DATASETS = ["franca", "frangioni", "lawrinenko", "berndt", "planted"]
LABELS = {"franca": "Franca", "frangioni": "Frangioni", "lawrinenko": "Lawrinenko",
          "berndt": "Berndt", "planted": "Planted"}
CSV_ROOT = Path(__file__).parent
OUT_DIR = Path(__file__).parents[2] / "Thesis" / "figures"
K_CSV = CSV_ROOT / "k_sensitivity.csv"

BOUND = 13 / 11
RHO_THRESHOLD = 2.0
NM_THRESHOLD = 12.0

COL_FFD = "#4C72B0"
COL_MFFD = "#DD8452"
COL_WIN = "#2CA02C"
COL_TIE = "#B8C4D4"
COL_LOSS = "#C44E52"
COL_LINE = "#333333"

TEXT_WIDTH = 5.9   # inches, corresponds to \textwidth with scrreprt/A4


def de_log_formatter():
    # log axis labels without exponential notation, with decimal comma
    def fmt(x, _pos):
        if x >= 1:
            return f"{x:.0f}"
        return f"{x:g}".replace(".", ",")
    return ticker.FuncFormatter(fmt)


class DeScalarFormatter(ticker.ScalarFormatter):
    # Like ScalarFormatter, but with decimal comma. How many decimal places an
    # axis needs is still decided by matplotlib.

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def get_offset(self):
        return super().get_offset().replace(".", ",")


def localize(fig):
    # Decimal comma on all numeric linear axes of a figure.
    # Only axes with a ScalarFormatter are replaced; categorical axes
    # (FixedFormatter, e.g. the dataset names in the runtime box plot) and log
    # axes (LogFormatter or de_log_formatter) stay untouched.
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ticker.ScalarFormatter):
                axis.set_major_formatter(DeScalarFormatter())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load_all() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        df = pd.read_csv(CSV_ROOT / dataset / "benchmark.csv")
        df["dataset"] = dataset
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["opt"] = pd.to_numeric(df["opt"], errors="coerce")
    df["rho"] = df["sum_tasks"] / (df["m"] * df["max_task"])
    df["n_over_m"] = df["n"] / df["m"]
    df["ratio"] = df["makespan_mffd"] / df["makespan_ffd"]
    df["time_ratio"] = df["time_mffd_ms"] / df["time_ffd_ms"]
    df["outcome"] = np.where(df["makespan_mffd"] < df["makespan_ffd"], "win",
                    np.where(df["makespan_mffd"] > df["makespan_ffd"], "loss", "tie"))
    return df


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    localize(fig)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(OUT_DIR.parents[1])}")


# --- Figures -----------------------------------------------------------------

def fig_kollaps(df):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9), sharey=True)

    specs = [
        (axes[0], "rho", RHO_THRESHOLD,
         r"$\rho = \sum_j p_j \,/\, (m \cdot p_{\max})$", r"$\rho = 2$"),
        (axes[1], "n_over_m", NM_THRESHOLD, r"$n/m$", r"$n/m = 12$"),
    ]
    style = {"win": (COL_WIN, 0.85, 3, "MFFD besser"),
             "tie": (COL_TIE, 0.35, 1, "gleich"),
             "loss": (COL_LOSS, 0.85, 3, "MFFD schlechter")}

    for ax, xcol, threshold, xlabel, tlabel in specs:
        for outcome in ("tie", "win", "loss"):
            sub = df[df["outcome"] == outcome]
            color, alpha, z, label = style[outcome]
            ax.scatter(sub[xcol], sub["ratio"], s=5, c=color, alpha=alpha,
                       linewidths=0, zorder=z, label=label)
        ax.axvline(threshold, color=COL_LINE, linestyle="--", linewidth=1,
                   zorder=4, label=tlabel)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(de_log_formatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.legend(loc="upper right", markerscale=2.2, framealpha=0.9)

    axes[0].set_ylabel(r"$C^{\mathrm{MFFD}}_{\max} / C^{\mathrm{FFD}}_{\max}$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "abb_kollaps.pdf")


def fig_ratio_verteilung(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.72, 2.7))
    ax.hist(df["ratio"], bins=60, color=COL_FFD, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color=COL_LINE, linestyle="--", linewidth=1,
               label=r"$C^{\mathrm{MFFD}}_{\max} = C^{\mathrm{FFD}}_{\max}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$C^{\mathrm{MFFD}}_{\max} / C^{\mathrm{FFD}}_{\max}$")
    ax.set_ylabel("Instanzen (log.)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "abb_ratio_verteilung.pdf")


def fig_guete_opt(df):
    known = df[df["opt"].notna()]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.78, 2.9))

    for alg, color, label in (("ffd", COL_FFD, "MULTIFIT+FFD"),
                              ("mffd", COL_MFFD, "MULTIFIT+MFFD")):
        r = np.sort(known[f"makespan_{alg}"] / known["opt"])
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(r, 100 * y, where="post", color=color, linewidth=1.4, label=label)

    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.9)
    # The bound is proven only for MULTIFIT+FFD (Theorem 3.12 with r_m <= 13/11);
    # for MULTIFIT+MFFD it is a reference line, and the legend has to say so.
    ax.axvline(BOUND, color=COL_LOSS, linestyle="--", linewidth=1.1,
               label=r"$13/11$ (bewiesen für MULTIFIT+FFD)")
    ax.set_xlabel(r"$C^{\mathrm{MF}}_{\max} / C^{*}_{\max}$")
    # usetex is off, so no LaTeX escape: "\\%" would be rendered as \%.
    ax.set_ylabel("Anteil der Instanzen (%)")
    ax.set_xlim(0.995, 1.20)
    ax.set_ylim(0, 101)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "abb_guete_opt.pdf")


def fig_laufzeit(df):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))

    ax = axes[0]
    data = [df[df["dataset"] == d]["time_ratio"].values for d in DATASETS]
    bp = ax.boxplot(data, tick_labels=[LABELS[d] for d in DATASETS],
                    showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(COL_MFFD)
        patch.set_alpha(0.55)
        patch.set_linewidth(0.8)
    for element in ("medians", "whiskers", "caps"):
        for item in bp[element]:
            item.set_color(COL_LINE)
            item.set_linewidth(0.9)
    ax.axhline(1.0, color=COL_LINE, linestyle="--", linewidth=1)
    ax.set_ylabel(r"$t_{\mathrm{MFFD}} / t_{\mathrm{FFD}}$")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Right panel on Planted only: the only dataset whose n spans three orders
    # of magnitude (8 to 20000). A slope estimate pooled over all datasets would
    # mix different generation procedures.
    ax = axes[1]
    sub = df[df["dataset"] == "planted"]
    for alg, color, label in (("ffd", COL_FFD, "FFD"), ("mffd", COL_MFFD, "MFFD")):
        ax.scatter(sub["n"], sub[f"time_{alg}_ms"], s=3, c=color, alpha=0.25,
                   linewidths=0, label=label)
        slope, intercept = np.polyfit(np.log(sub["n"]), np.log(sub[f"time_{alg}_ms"]), 1)
        xs = np.array([sub["n"].min(), sub["n"].max()])
        ax.plot(xs, np.exp(intercept) * xs ** slope, color=color, linewidth=1.2,
                linestyle="--", label=f"Steigung {slope:.2f}".replace(".", ","))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ (Planted)")
    ax.set_ylabel("Laufzeit (ms)")
    ax.xaxis.set_major_formatter(de_log_formatter())
    ax.legend(loc="upper left", markerscale=3)

    fig.tight_layout(w_pad=1.6)
    save(fig, "abb_laufzeit.pdf")


def fig_k_sensitivitaet():
    if not K_CSV.exists():
        print(f"  (skipped: {K_CSV.name} missing -- run k_sensitivity.py first)")
        return
    df = pd.read_csv(K_CSV)
    df = df[df["opt"].notna()].copy()
    for alg in ("ffd", "mffd"):
        df[f"r_{alg}"] = df[f"makespan_{alg}"] / df["opt"]
    ks = sorted(df["k"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))

    for alg, color, label in (("ffd", COL_FFD, "MULTIFIT+FFD"),
                              ("mffd", COL_MFFD, "MULTIFIT+MFFD")):
        means = [df[df["k"] == k][f"r_{alg}"].mean() for k in ks]
        maxes = [df[df["k"] == k][f"r_{alg}"].max() for k in ks]
        axes[0].plot(ks, means, marker="o", markersize=3.5, color=color,
                     linewidth=1.3, label=label)
        axes[1].plot(ks, maxes, marker="o", markersize=3.5, color=color,
                     linewidth=1.3, label=label)

    theory = [BOUND + 2.0 ** (-k) for k in ks]
    axes[1].plot(ks, theory, color=COL_LOSS, linestyle="--", linewidth=1.2,
                 label="$13/11 + 2^{-k}$\n(für MULTIFIT+FFD)")

    for ax in axes:
        ax.set_xlabel("Binärsuchtiefe $k$")
        ax.set_xticks(ks)
    axes[0].set_ylabel(r"arithm. Mittel von $C^{\mathrm{MF}}_{\max} / C^{*}_{\max}$")
    axes[1].set_ylabel(r"maximales $C^{\mathrm{MF}}_{\max} / C^{*}_{\max}$")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.6)
    save(fig, "abb_k_sensitivitaet.pdf")


def fig_friesen_binzahl():
    # Bin counts of FFD and MFFD on the Friesen instance for all integer C
    # (Section 5.1.8). The range starts at C*_max = 66 and extends beyond the
    # non-monotonicity witness 79 -> 80 until FFD drops back to 11 bins.
    tasks = build_instance()
    cs = np.arange(FRIESEN_OPT, 91)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.72, 2.6))
    # Where both curves coincide, FFD must stay visible: FFD as a solid line
    # with open circles, MFFD dashed with small squares.
    styles = ((ffd_nlogn, COL_FFD, "FFD", "-", 1.6, dict(marker="o", markersize=4.5,
                                                         markerfacecolor="none")),
              (mffd_nlogn, COL_MFFD, "MFFD", "--", 1.1, dict(marker="s", markersize=2.5)))
    for alg, color, label, ls, lw, mk in styles:
        bins = [len(alg(tasks, int(c))) for c in cs]
        ax.plot(cs, bins, drawstyle="steps-mid", color=color, linestyle=ls, linewidth=lw)
        ax.plot(cs, bins, linestyle="none", color=color, label=label, **mk)
    ax.axhline(FRIESEN_M, color=COL_LINE, linestyle="--", linewidth=0.9,
               label=f"$m = {FRIESEN_M}$")
    ax.set_xlabel("Kapazität $C$")
    ax.set_ylabel("Anzahl Bins")
    ax.set_xticks(np.arange(FRIESEN_OPT, 91, 2))
    ax.set_yticks(np.arange(10, 16))
    ax.set_ylim(10.5, 14.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "abb_friesen_binzahl.pdf")


def main():
    df = load_all()
    print(f"{len(df)} instances loaded. Figures:")
    fig_kollaps(df)
    fig_ratio_verteilung(df)
    fig_guete_opt(df)
    fig_laufzeit(df)
    fig_k_sensitivitaet()
    fig_friesen_binzahl()


if __name__ == "__main__":
    main()
