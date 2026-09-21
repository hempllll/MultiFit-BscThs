"""
Generates the LaTeX table bodies and number macros of Section 5.1 from the benchmark CSVs.

Output to Thesis/tables/:
  tab_datensaetze.tex      overview of the five datasets
  tab_win_tie_loss.tex     win/tie/loss and ratio statistics MFFD vs. FFD
  tab_kollaps.tex          deviations binned by rho = sum p / (m * p_max)
  tab_ratio_opt.tex        makespan/OPT per dataset for FFD and MFFD
  tab_abdeckung.tex        coverage of the 13/11 statement (OPT known / lower bound / open)
  tab_laufzeit.tex         runtimes and pairwise ratio
  tab_k_sensitivitaet.tex  makespan/OPT over k (needs k_sensitivity.csv)
  kennzahlen.tex           \newcommand macros for numbers in the running text

All tables are plain tabular environments (booktabs); the chapter embeds them via
\input into a table environment with caption. Table contents stay German, since
they are part of the thesis.

Usage:
  python Implementation/Empirical_Tests/summary_tables.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["franca", "frangioni", "lawrinenko", "berndt", "planted"]
LABELS = {"franca": "Fran\u00e7a", "frangioni": "Frangioni", "lawrinenko": "Lawrinenko",
          "berndt": "Berndt", "planted": "Planted"}
CSV_ROOT = Path(__file__).parent
K_CSV = CSV_ROOT / "k_sensitivity.csv"
OUT_DIR = Path(__file__).parents[2] / "Thesis" / "tables"
BOUND = 13 / 11
RHO_THRESHOLD = 2.0
NM_THRESHOLD = 12.0


# --- Formatting --------------------------------------------------------------

def num(x, digits=0):
    # German number format: comma as decimal separator, \, as thousands separator.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if digits == 0:
        # floor(x + 0.5) instead of round(): Python's banker's rounding rounds .5
        # to the even number, 80.5 would become 80.
        return f"{math.floor(x + 0.5):,}".replace(",", "\\,")
    s = f"{x:,.{digits}f}"
    intpart, _, frac = s.partition(".")
    # {,} instead of , so that the decimal comma adds no space in math mode either
    return intpart.replace(",", "\\,") + "{,}" + frac


def pct(x, digits=1):
    return num(x, digits) + "\\,\\%"


def write(name: str, lines: list):
    path = OUT_DIR / name
    path.write_text("\n".join(lines) + "\n")
    print(f"  -> {path.relative_to(OUT_DIR.parents[1])}")


# --- Data --------------------------------------------------------------------

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
    df["lb"] = np.maximum(np.ceil(df["sum_tasks"] / df["m"]), df["max_task"])
    df["ratio_mffd_ffd"] = df["makespan_mffd"] / df["makespan_ffd"]
    df["differs"] = df["makespan_ffd"] != df["makespan_mffd"]
    df["time_ratio"] = df["time_mffd_ms"] / df["time_ffd_ms"]
    for alg in ("ffd", "mffd"):
        df[f"ratio_opt_{alg}"] = df[f"makespan_{alg}"] / df["opt"]
        df[f"ratio_lb_{alg}"] = df[f"makespan_{alg}"] / df["lb"]
    return df


def groups(df):
    # iterates over the datasets and finally over the union of all of them
    for dataset in DATASETS:
        yield LABELS[dataset], df[df["dataset"] == dataset]
    yield None, df


# --- Tables ------------------------------------------------------------------

def tab_datensaetze(df):
    def rng(series, digits=0):
        lo, hi = series.min(), series.max()
        return f"{num(lo, digits)}--{num(hi, digits)}"

    lines = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r"Datensatz & Instanzen & $n$ & $m$ & $n/m$ & $\rho$ & $C^{*}_{\max}$ bekannt \\",
             r"\midrule"]
    for label, sub in groups(df):
        row = (f"{num(len(sub))} & {rng(sub['n'])} & {rng(sub['m'])} & "
               f"{rng(sub['n_over_m'], 1)} & {rng(sub['rho'], 2)} & "
               f"{num(int(sub['opt'].notna().sum()))}")
        if label is None:
            lines += [r"\midrule", r"\textbf{Gesamt} & " + row + r" \\"]
        else:
            lines.append(f"{label} & " + row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_datensaetze.tex", lines)


def tab_win_tie_loss(df):
    lines = [r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"Datensatz & $N$ & MFFD besser & gleich & MFFD schlechter"
             r" & \multicolumn{3}{c}{$C^{\mathrm{MFFD}}_{\max}/C^{\mathrm{FFD}}_{\max}$} \\",
             r"\cmidrule(lr){6-8}",
             r" & & & & & Min. & Median & Max. \\", r"\midrule"]
    for label, sub in groups(df):
        w = int((sub["makespan_mffd"] < sub["makespan_ffd"]).sum())
        l = int((sub["makespan_mffd"] > sub["makespan_ffd"]).sum())
        t = len(sub) - w - l
        r = sub["ratio_mffd_ffd"]
        row = (f"{num(len(sub))} & {num(w)} & {num(t)} & {num(l)} & "
               f"{num(r.min(), 4)} & {num(r.median(), 4)} & {num(r.max(), 4)}")
        if label is None:
            lines += [r"\midrule", r"\textbf{Alle} & " + row + r" \\"]
        else:
            lines.append(f"{label} & " + row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_win_tie_loss.tex", lines)


def tab_kollaps(df):
    edges = [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, np.inf]
    labels = [r"$\rho < 0{,}5$", r"$0{,}5 \le \rho < 1{,}0$", r"$1{,}0 \le \rho < 1{,}25$",
              r"$1{,}25 \le \rho < 1{,}5$", r"$1{,}5 \le \rho < 1{,}75$",
              r"$1{,}75 \le \rho < 2{,}0$", r"$\rho \ge 2{,}0$"]
    cut = pd.cut(df["rho"], bins=edges, labels=labels, right=False)
    agg = df.groupby(cut, observed=False).agg(
        n=("differs", "size"), diff=("differs", "sum"))
    lines = [r"\begin{tabular}{l r r r}", r"\toprule",
             r"Bereich & Instanzen & Abweichungen & Anteil \\", r"\midrule"]
    for label, row in agg.iterrows():
        share = 100 * row["diff"] / row["n"] if row["n"] else 0.0
        cells = f"{label} & {num(row['n'])} & {num(row['diff'])} & {pct(share)}"
        if label == labels[-1]:
            lines.append(r"\midrule")
            cells = (f"{label} & \\textbf{{{num(row['n'])}}} & "
                     f"\\textbf{{{num(row['diff'])}}} & \\textbf{{{pct(share)}}}")
        lines.append(cells + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_kollaps.tex", lines)

    # Numbers on the transition: the strongest bin below the threshold and the
    # last bin before it. Both are used in the running text of Section 5.1.3.
    below = agg.loc[labels[:-1]].copy()
    below["share"] = 100 * below["diff"] / below["n"].replace(0, np.nan)
    top = below["share"].idxmax()
    return {
        "EvalKollapsMaxBereich": str(top),
        "EvalKollapsMaxAnteil": pct(below.loc[top, "share"]),
        "EvalKollapsLetzterAnteil": pct(below.loc[labels[-2], "share"]),
    }


def tab_ratio_opt(df):
    lines = [r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"Datensatz & $C^{*}_{\max}$ bekannt & \multicolumn{3}{c}{MULTIFIT+FFD}"
             r" & \multicolumn{3}{c}{MULTIFIT+MFFD} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r" & & Mittel & Max. & $=C^{*}_{\max}$ & Mittel & Max. & $=C^{*}_{\max}$ \\",
             r"\midrule"]
    for label, sub in groups(df):
        o = sub[sub["opt"].notna()]
        cells = [num(len(o))]
        for alg in ("ffd", "mffd"):
            r = o[f"ratio_opt_{alg}"]
            cells += [num(r.mean(), 4), num(r.max(), 4), pct(100 * (r == 1).mean())]
        row = " & ".join(cells)
        if label is None:
            lines += [r"\midrule", r"\textbf{Alle} & " + row + r" \\"]
        else:
            lines.append(f"{label} & " + row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ratio_opt.tex", lines)


def tab_abdeckung(df):
    lines = [r"\begin{tabular}{l r r r r}", r"\toprule",
             r"Datensatz & $N$ & $C^{*}_{\max}$ bekannt & über $\mathrm{LB}$ belegt & unbestimmt \\",
             r"\midrule"]
    for label, sub in groups(df):
        known = int(sub["opt"].notna().sum())
        unknown = sub[sub["opt"].isna()]
        cert = int(((unknown["ratio_lb_ffd"] <= BOUND) & (unknown["ratio_lb_mffd"] <= BOUND)).sum())
        row = f"{num(len(sub))} & {num(known)} & {num(cert)} & {num(len(unknown) - cert)}"
        if label is None:
            lines += [r"\midrule", r"\textbf{Gesamt} & " + row + r" \\"]
        else:
            lines.append(f"{label} & " + row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_abdeckung.tex", lines)


def tab_laufzeit(df):
    lines = [r"\begin{tabular}{l r r r r}", r"\toprule",
             r"Datensatz & Median FFD & Median MFFD"
             r" & \multicolumn{2}{c}{$t_{\mathrm{MFFD}}/t_{\mathrm{FFD}}$} \\",
             r"\cmidrule(lr){4-5}",
             r" & (ms) & (ms) & Median & 95.\ Perzentil \\", r"\midrule"]
    for label, sub in groups(df):
        row = (f"{num(sub['time_ffd_ms'].median(), 3)} & {num(sub['time_mffd_ms'].median(), 3)} & "
               f"{num(sub['time_ratio'].median(), 3)} & {num(sub['time_ratio'].quantile(0.95), 3)}")
        if label is None:
            lines += [r"\midrule", r"\textbf{Alle} & " + row + r" \\"]
        else:
            lines.append(f"{label} & " + row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_laufzeit.tex", lines)


def tab_k_sensitivitaet():
    # table and numbers on the binary search depth; needs k_sensitivity.csv
    if not K_CSV.exists():
        print(f"  (skipped: {K_CSV.name} missing -- run k_sensitivity.py first)")
        return {}
    df = pd.read_csv(K_CSV)
    ks = sorted(df["k"].unique())
    k_max = max(ks)

    known = df[df["opt"].notna()].copy()

    # The column "stabil" is also computed over the instances with known optimum
    # only -- otherwise the table would mix two populations and the caption would
    # be wrong for one of them.
    stable = {}
    for alg in ("ffd", "mffd"):
        piv = known.pivot_table(index=["dataset", "instance_file"], columns="k",
                                values=f"makespan_{alg}")
        stable[alg] = {k: 100 * (piv[k] == piv[k_max]).mean() for k in ks}

    for alg in ("ffd", "mffd"):
        known[f"r_{alg}"] = known[f"makespan_{alg}"] / known["opt"]
    agg = known.groupby("k")[["r_ffd", "r_mffd"]].agg(["mean", "max"])

    lines = [r"\begin{tabular}{r r r r r r r}", r"\toprule",
             r"$k$ & \multicolumn{2}{c}{MULTIFIT+FFD} & \multicolumn{2}{c}{MULTIFIT+MFFD}"
             r" & $13/11 + 2^{-k}$ & stabil \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & Mittel & Max. & Mittel & Max. & & \\", r"\midrule"]
    for k in ks:
        lines.append(
            f"{k} & {num(agg.loc[k, ('r_ffd', 'mean')], 4)} & {num(agg.loc[k, ('r_ffd', 'max')], 4)}"
            f" & {num(agg.loc[k, ('r_mffd', 'mean')], 4)} & {num(agg.loc[k, ('r_mffd', 'max')], 4)}"
            f" & {num(BOUND + 2.0 ** -k, 4)} & {pct(stable['ffd'][k])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_k_sensitivitaet.tex", lines)

    # smallest k from which on the maximum no longer decreases
    max_ffd = {k: agg.loc[k, ("r_ffd", "max")] for k in ks}
    k_stable_max = min(k for k in ks if all(
        abs(max_ffd[j] - max_ffd[k_max]) < 1e-12 for j in ks if j >= k))

    # Non-monotonicity of the makespan in k: transitions k -> k+1 in which the
    # makespan increases.
    nm = df.sort_values(["dataset", "instance_file", "k"])
    non_monotone = {}
    worst = None
    for alg in ("ffd", "mffd"):
        col = f"makespan_{alg}"
        prev = nm.groupby(["dataset", "instance_file"])[col].shift(1)
        worse = nm[prev.notna() & (nm[col] > prev + 1e-9)].copy()
        worse["prev"] = prev[worse.index]
        non_monotone[alg] = len(worse)
        if len(worse):
            # The worst case is searched over both algorithms; otherwise the text
            # would name an MFFD value even when FFD has the larger jump.
            worse["rel"] = worse[col] / worse["prev"] - 1
            worse["alg"] = alg
            worse["makespan"] = worse[col]
            candidate = worse.sort_values("rel", ascending=False).iloc[0]
            if worst is None or candidate["rel"] > worst["rel"]:
                worst = candidate

    extra_nm = {
        "EvalKNichtMonotonFFD": num(non_monotone["ffd"]),
        "EvalKNichtMonotonMFFD": num(non_monotone["mffd"]),
    }
    if worst is not None:
        extra_nm.update({
            "EvalKNichtMonotonMax": pct(100 * worst["rel"]),
            "EvalKNichtMonotonAlg": "MULTIFIT+" + worst["alg"].upper(),
            "EvalKNichtMonotonInstanz": worst["instance_file"].replace("_", r"\_"),
            "EvalKNichtMonotonVorher": num(worst["prev"]),
            "EvalKNichtMonotonNachher": num(worst["makespan"]),
            "EvalKNichtMonotonKVorher": str(int(worst["k"]) - 1),
            "EvalKNichtMonotonKNachher": str(int(worst["k"])),
        })

    return {
        **extra_nm,
        "EvalKMax": str(k_max),
        "EvalKMittelEins": num(agg.loc[1, ("r_ffd", "mean")], 4),
        "EvalKMaxEins": num(agg.loc[1, ("r_ffd", "max")], 4),
        "EvalKMittelZehn": num(agg.loc[k_max, ("r_ffd", "mean")], 4),
        "EvalKMaxZehn": num(agg.loc[k_max, ("r_ffd", "max")], 4),
        "EvalKStabilMax": str(k_stable_max),
        "EvalKStabilFuenf": pct(stable["ffd"][5]),
        "EvalKStabilNeun": pct(stable["ffd"][k_max - 1]),
        "EvalKSchrankeStabil": num(BOUND + 2.0 ** -k_stable_max, 4),
        # R_m(MF(k)) <= r_m + 2^-k (Theorem 3.12), so the bound depends on k;
        # 13/11 alone only holds in the limit.
        "EvalKSchrankeEins": num(BOUND + 2.0 ** -1, 4),
        "EvalSchrankeK": num(BOUND + 2.0 ** -k_max, 4),
    }


def dataset_macros(df):
    # Per dataset the numbers that Section 5.1.1 mentions in the running text.
    # Without these macros, measured values would be typed in by hand there --
    # and silently diverge from the generated tables after the next measurement.
    tag = {"franca": "Franca", "frangioni": "Frangioni", "lawrinenko": "Lawrinenko",
           "berndt": "Berndt", "planted": "Planted"}
    macros = {}
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        t = tag[ds]
        rho_two = int((sub["rho"] >= 2).sum())
        macros[f"EvalInstanzen{t}"] = num(len(sub))
        macros[f"EvalOptBekannt{t}"] = num(int(sub["opt"].notna().sum()))
        macros[f"EvalRhoZwei{t}"] = num(rho_two)
        macros[f"EvalRhoUnterZwei{t}"] = num(len(sub) - rho_two)
        macros[f"EvalRhoZweiAnteil{t}"] = pct(100 * rho_two / len(sub))
        macros[f"EvalRhoMin{t}"] = num(sub["rho"].min(), 2)
        macros[f"EvalRhoMax{t}"] = num(sub["rho"].max(), 2)
        macros[f"EvalNMMin{t}"] = num(sub["n_over_m"].min(), 1)
        macros[f"EvalNMMax{t}"] = num(sub["n_over_m"].max(), 1)

    # Frangioni splits into uniform (U) and bimodal (NU) instances;
    # Section 5.1.1 compares the two halves.
    fr = df[df["dataset"] == "frangioni"]
    bimodal = fr[fr["instance_file"].str.startswith("NU")]
    uniform = fr[fr["instance_file"].str.startswith("U")]
    for part, sub in (("Bimodal", bimodal), ("Gleich", uniform)):
        if len(sub):
            macros[f"EvalRhoMedianFrangioni{part}"] = num(sub["rho"].median(), 1)
            macros[f"EvalRhoZweiAnteilFrangioni{part}"] = pct(100 * (sub["rho"] >= 2).mean())

    # Planted splits into the converted originals and the self-generated
    # instances; only the latter carry a seed in the file name. The perturbation r
    # is part of the name as well: r = 0 means OPT = U by construction.
    pl = df[df["dataset"] == "planted"]
    generated = pl["instance_file"].str.contains("seed")
    r_zero = pl["instance_file"].str.extract(r"perturb([0-9]+(?:\.[0-9]+)?)")[0].astype(float) == 0
    macros["EvalPlantedGeneriert"] = num(int(generated.sum()))
    macros["EvalPlantedOriginale"] = num(int((~generated).sum()))
    macros["EvalPlantedOriginaleOpt"] = num(int(pl[~generated]["opt"].notna().sum()))
    macros["EvalPlantedRNull"] = num(int(r_zero.sum()))
    macros["EvalPlantedRPositiv"] = num(int(pl[~r_zero]["opt"].notna().sum()))
    return macros


def write_macros(df, extra=None):
    hi_rho = df[df["rho"] >= RHO_THRESHOLD]
    hi_nm = df[df["n_over_m"] >= NM_THRESHOLD]
    differing = df[df["differs"]]
    known = df[df["opt"].notna()]
    unknown = df[df["opt"].isna()]
    cert = unknown[(unknown["ratio_lb_ffd"] <= BOUND) & (unknown["ratio_lb_mffd"] <= BOUND)]
    both = pd.concat([known["ratio_opt_ffd"], known["ratio_opt_mffd"]])

    macros = {
        "EvalInstanzen": num(len(df)),
        "EvalOptBekannt": num(len(known)),
        "EvalOhneOpt": num(len(unknown)),
        "EvalLbBelegt": num(len(cert)),
        "EvalUnbestimmt": num(len(unknown) - len(cert)),
        "EvalWins": num(int((df["makespan_mffd"] < df["makespan_ffd"]).sum())),
        "EvalTies": num(int((~df["differs"]).sum())),
        "EvalLosses": num(int((df["makespan_mffd"] > df["makespan_ffd"]).sum())),
        "EvalWinsProzent": pct(100 * (df["makespan_mffd"] < df["makespan_ffd"]).mean()),
        "EvalTiesProzent": pct(100 * (~df["differs"]).mean()),
        "EvalLossesProzent": pct(100 * (df["makespan_mffd"] > df["makespan_ffd"]).mean()),
        "EvalAbweichungen": num(len(differing)),
        "EvalRhoZwei": num(len(hi_rho)),
        "EvalRhoZweiProzent": pct(100 * len(hi_rho) / len(df)),
        "EvalRhoZweiAbw": num(int(hi_rho["differs"].sum())),
        "EvalRhoMaxAbw": num(differing["rho"].max(), 4),
        "EvalNMZwoelf": num(len(hi_nm)),
        "EvalNMZwoelfAbw": num(int(hi_nm["differs"].sum())),
        "EvalNMMaxAbw": num(differing["n_over_m"].max(), 1),
        "EvalMaxRatioOpt": num(both.max(), 4),
        "EvalMaxRatioOptFFD": num(known["ratio_opt_ffd"].max(), 4),
        "EvalMaxRatioOptMFFD": num(known["ratio_opt_mffd"].max(), 4),
        "EvalMaxRatioMffdFfd": num(df["ratio_mffd_ffd"].max(), 4),
        "EvalMinRatioMffdFfd": num(df["ratio_mffd_ffd"].min(), 4),
        "EvalOptimalFFD": pct(100 * (known["ratio_opt_ffd"] == 1).mean()),
        "EvalZeitMedianMin": num(df.groupby("dataset")["time_ratio"].median().min(), 3),
        "EvalZeitMedianMax": num(df.groupby("dataset")["time_ratio"].median().max(), 3),
        "EvalSchranke": num(BOUND, 4),
        "EvalRhoUnterZwei": num(len(df) - len(hi_rho)),
        "EvalMaxRatioLb": num(pd.concat([cert["ratio_lb_ffd"], cert["ratio_lb_mffd"]]).max(), 4),
    }

    # Slopes of the regression lines on a log-log scale, on Planted only -- the only
    # dataset whose n spans three orders of magnitude.
    # Must match the right half of abb_laufzeit.pdf (thesis_figures.py).
    planted = df[df["dataset"] == "planted"]
    for alg, key in (("ffd", "EvalSteigungFFD"), ("mffd", "EvalSteigungMFFD")):
        slope = np.polyfit(np.log(planted["n"]), np.log(planted[f"time_{alg}_ms"]), 1)[0]
        macros[key] = num(slope, 2)
    macros.update(extra or {})
    lines = [r"% Automatisch erzeugt von Implementation/Empirical_Tests/summary_tables.py",
             r"% Nicht von Hand bearbeiten."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    write("kennzahlen.tex", lines)
    return macros


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    print(f"{len(df)} instances loaded from {len(DATASETS)} datasets.")
    print("Tables:")
    tab_datensaetze(df)
    tab_win_tie_loss(df)
    kollaps_macros = tab_kollaps(df)
    tab_ratio_opt(df)
    tab_abdeckung(df)
    tab_laufzeit(df)
    k_macros = tab_k_sensitivitaet()
    macros = write_macros(df, {**kollaps_macros, **k_macros, **dataset_macros(df)})
    print("\nMacros:")
    for k, v in macros.items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
