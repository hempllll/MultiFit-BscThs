"""
Tables and number macros of Section 5.2 from benchmark_hm.csv.

The core is the comparison of the subroutines *inside* MULTIFIT: which makespan
does MULTIFIT return if FFD is replaced by Jansen's OPT+1 algorithm, its eps-dual
variant or the configuration LP.

Output to Thesis/tables/:
  tab_hm_datensatz.tex   structure of the dataset (cells, n, m, d, rho)
  tab_hm_paare.tex       win/tie/loss of the arms against each other
  tab_hm_ratio_opt.tex   makespan/OPT per arm, binned by d
  tab_hm_laufzeit.tex    runtime per arm, binned by d
  tab_hm_akzeptanz.tex   accepted and rejected capacities per arm,
                         breakdown of the GG rejections
  tab_hm_skalierung.tex  runtime along the n axis (dataset HM-Skalierung)
  kennzahlen_hm.tex      \newcommand macros with prefix \Hm

The arms are detected from the makespan_* columns of the CSV, so the script also
runs when only some of them have been measured. Table contents stay German,
since they are part of the thesis.

Usage:
  python Implementation/Empirical_Tests/summary_tables_hm.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

CSV_ROOT = Path(__file__).parent
OUT_DIR = Path(__file__).parents[2] / "Thesis" / "tables"
DATASET = "jansen_hm"
SCALE_DATASET = "jansen_hm_scale"
BOUND = 13 / 11

# Order and labels of the arms; FFD is the baseline, gg_ip the exact reference.
ARM_ORDER = ["ffd", "mffd", "jansen", "jansen_dual", "gg_lp", "gg_ip"]
ARM_LABEL = {"ffd": "FFD", "mffd": "MFFD", "jansen": "Jansen OPT+1",
             "jansen_dual": "Jansen $\\varepsilon$-dual",
             "gg_lp": "GG-LP", "gg_ip": "GG-IP (exakt)"}
BASELINE = "ffd"
REFERENCE = "gg_ip"


# --- Formatting (identical to summary_tables.py) -----------------------------

def num(x, digits=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if digits == 0:
        # floor(x + 0.5) instead of round(): Python's banker's rounding turns
        # 80.5 into 80.
        return f"{math.floor(x + 0.5):,}".replace(",", "\\,")
    s = f"{x:,.{digits}f}"
    intpart, _, frac = s.partition(".")
    return intpart.replace(",", "\\,") + "{,}" + frac


def pct(x, digits=1):
    return num(x, digits) + "\\,\\%"


def write(name, lines):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text("\n".join(lines) + "\n")
    print(f"  -> Thesis/tables/{name}")


# --- Data --------------------------------------------------------------------

def load_trace(df):
    # The capacity trace, enriched with OPT[Gamma,C] from the exact arm.
    path = CSV_ROOT / DATASET / "capacities_hm.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path)
    # opt is needed for the column "zu Unrecht": by Lemma 2.5, OPT[Gamma,C] <= m
    # holds if and only if C >= C*_max(Gamma). The detour via opt_bins_at_c only
    # covers the capacities that the exact arm has visited as well -- but the
    # binary search runs differently for each arm.
    t = t.merge(df[["instance_file", "m", "d", "opt"]].rename(columns={"d": "d_inst"}),
                on="instance_file", how="left")
    if REFERENCE in set(t.arm):
        ref = (t[t.arm == REFERENCE][["instance_file", "capacity", "bins"]]
               .rename(columns={"bins": "opt_bins_at_c"}))
        t = t.merge(ref, on=["instance_file", "capacity"], how="left")
    else:
        t["opt_bins_at_c"] = np.nan
    return t


def tab_akzeptanz(t):
    # What does the subroutine do in the binary search -- and what does MULTIFIT
    # learn from it? For FFD a rejection carries no information; the configuration
    # LP splits it into a certified one (ceil(LP) > m, so C < C*_max(Gamma) is
    # proven) and an undecided one (only the rounding has failed). The last column
    # counts how often a rejection discarded an actually feasible capacity
    # C >= C*_max(Gamma) -- the price of the rounding.
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& & \multicolumn{2}{c}{Entscheidung} & \multicolumn{2}{c}{davon} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Subroutine & Aufrufe & angen. & abgel. & zertifiziert & zu Unrecht \\",
             r"\midrule"]
    wrongly_rejected = {}
    for a in ARM_ORDER:
        s = t[t.arm == a]
        if s.empty:
            continue
        acc = int(s.accepted.sum())
        rej = len(s) - acc
        if "accept_status" in s and s.accept_status.notna().any():
            certified = int((s.accept_status == "zertifiziert_abgelehnt").sum())
            certified_s = num(certified)
        else:
            certified_s = "---"
        lost = s[(~s.accepted) & (s.capacity >= s.opt)]
        wrongly_rejected[a] = len(lost)
        lines.append(f"{ARM_LABEL[a]} & {num(len(s))} & {num(acc)} & {num(rej)} & "
                     f"{certified_s} & {num(len(lost))} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_akzeptanz.tex", lines)
    return {"HmZuUnrecht" + a.replace("_", "").capitalize(): num(v)
            for a, v in wrongly_rejected.items()}


def non_monotonicity_witness(df, t):
    # Does the jump J[C+d] = J[C]+1 admitted by Lemma 4.4 actually occur?
    # The binary search visits only logarithmically many, widely spaced
    # capacities per instance; nothing shows up there. The search therefore
    # looks, first, at the trace (frequency over all instances) and, second, at
    # the contiguous capacity range in case_study_hm.csv, which
    # run_hm_benchmark.py --case-study generates.
    macros = {}

    in_trace = 0
    ja = t[t.arm == "jansen"]
    for _, g in ja.groupby("instance_file"):
        b = g.sort_values("capacity").bins.values
        in_trace += sum(1 for i in range(len(b)) for j in range(i + 1, len(b))
                        if b[j] > b[i])
    macros["HmJansenNichtMonotonSpur"] = num(in_trace)

    path = CSV_ROOT / DATASET / "case_study_hm.csv"
    if not path.exists():
        return macros
    f = pd.read_csv(path)
    js = (f[f.arm == "jansen"].set_index("capacity").bins.sort_index())
    pairs = [(c1, c2) for c1, c2 in zip(js.index, js.index[1:]) if js[c2] > js[c1]]
    if not pairs:
        return macros
    c1, c2 = pairs[0]
    inst = f.instance_file.iloc[0]
    macros.update({
        "HmNichtMonotonInstanz": r"\texttt{" + inst.replace("_", r"\_").replace(".txt", "") + "}",
        "HmNichtMonotonC": num(c1), "HmNichtMonotonCPlus": num(c2),
        "HmNichtMonotonJ": num(js[c1]), "HmNichtMonotonJPlus": num(js[c2])})
    return macros


def rounding_macros(t):
    # How expensive is the rounding of the configuration LP (variant B, Lemma 4.13)?
    # The additive term of the bound is |F| - 1, i.e. the number of fractional
    # columns. Measured: whether the support bound |F| <= d binds at all, and
    # how many bins the residual demand actually costs.
    gg = t[t.arm == "gg_lp"]
    if gg.empty:
        return {}
    rest = gg[gg.leftover_items > 0]
    m = {"HmFMax": num(gg.fractional.max()),
         "HmFMedian": num(gg.fractional.median()),
         "HmFGleichD": pct(100 * (gg.fractional == gg.d).mean(), 1),
         "HmRestAufrufe": num(len(rest)),
         "HmRestBinsMedian": num(rest.leftover_bins.median()),
         "HmRestBinsMax": num(rest.leftover_bins.max()),
         "HmRestUnterF": num(int((rest.leftover_bins < rest.fractional).sum())),
         "HmRestUnterFAnteil": pct(
             100 * (rest.leftover_bins < rest.fractional).mean(), 1)}
    source = gg.leftover_source.value_counts()
    for k, tag in (("ffd", "Ffd"), ("patterns", "Muster"), ("none", "Ohne")):
        m["HmRestQuelle" + tag] = num(int(source.get(k, 0)))
    return m


def overload_macros(df, t):
    # Does the acceptance decision alone determine the makespan? The eps-dual arm
    # and the exact configuration IP are compared because they accept the same
    # capacities -- whatever makespan difference remains cannot come from the
    # decision. It comes from property (ii) of Definition 4.6: the eps-dual arm
    # may fill a bin up to (1+eps)*C, so the returned packing can exceed the
    # capacity it was accepted at.
    if "jansen_dual" not in set(t.arm) or REFERENCE not in set(t.arm):
        return {}
    acc = {}
    for arm in ("jansen_dual", REFERENCE):
        a = t[(t.arm == arm) & t.accepted.astype(bool)]
        acc[arm] = a.groupby("instance_file").capacity.apply(frozenset)
    same = sum(1 for i in acc["jansen_dual"].index
               if acc["jansen_dual"][i] == acc[REFERENCE].get(i))
    worse = df[df.makespan_jansen_dual > df[f"makespan_{REFERENCE}"]]
    # the returned packing belongs to the smallest accepted capacity
    smallest = acc["jansen_dual"].apply(min)
    factor = (worse.set_index("instance_file").makespan_jansen_dual
              / smallest.reindex(worse.instance_file).values)
    return {"HmEpsGleicheAkzeptanz": num(same),
            "HmEpsAkzeptanzInstanzen": num(len(acc["jansen_dual"])),
            "HmEpsSchlechterRef": num(len(worse)),
            "HmEpsUeberlastMax": num(factor.max(), 4)}


def degenerate_macros(df, arms):
    # The instances with rho < 1, which are excluded from the quality tables.
    # rho < 1 means p_max > sum p / m, so OPT >= p_max by Lemma 2.2. Whether the
    # bound is attained is counted by HmDegOptPMax -- it does not always hold.
    deg = df[df.degeneriert]
    if deg.empty:
        return {}
    optimal = all((deg[f"makespan_{a}"] == deg.opt).all() for a in arms)
    return {"HmDegOptPMax": num(int((deg.opt == deg.max_task).sum())),
            "HmDegAlleOptimal": "alle" if optimal else "nicht alle",
            "HmDegRhoMax": num(deg.rho.max(), 2)}


def d_benchmarks():
    # d = number of distinct sizes on the five datasets of Section 5.1.
    # Supports the statement in 5.2.7 that the high-multiplicity methods are out
    # of their scope there. Silently returns nothing if the instance
    # directories are not available locally (see Test_Instances/fetch_pcmax.py).
    root = Path(__file__).parents[1] / "Test_Instances"
    medians, largest, skipped = [], 0, 0
    for ds in ["franca", "frangioni", "lawrinenko", "berndt", "planted"]:
        values = []
        for f in sorted((root / ds).glob("*.txt")) if (root / ds).is_dir() else []:
            if f.name.startswith("opt_") or f.name == "readme.txt":
                continue
            # Token-wise like run_benchmark.load_instance: first m, then n, then
            # the n sizes. Accessing lines[2] would silently swallow a different
            # line layout.
            try:
                tok = f.read_text().split()
                n = int(tok[1])
                values.append(len(set(int(x) for x in tok[2:2 + n])))
            except (IndexError, ValueError):
                skipped += 1
                continue
        if values:
            medians.append(float(np.median(values)))
            largest = max(largest, max(values))
    if skipped:
        print(f"  (warning: {skipped} instance files not readable, skipped)")
    if not medians:
        return {}
    return {"HmDBenchMedianMin": num(min(medians)),
            "HmDBenchMedianMax": num(max(medians)),
            "HmDBenchMax": num(largest)}


def scale_d_budget(draws=2000):
    # Why HM-Skalierung only contains d in {3, 4}.
    # The generator discards size combinations whose saturated bound
    # prod_i (floor(C/p_i) + 1) exceeds Q_BUDGET, and gives up after MAX_DRAWS
    # failed attempts. Here we count per (d, scenario) how many draws respect
    # the budget -- with a fixed seed, so that the macros are reproducible.
    import importlib.util
    path = Path(__file__).parents[1] / "Test_Instances" / SCALE_DATASET / "generate.py"
    spec = importlib.util.spec_from_file_location("hm_scale_generate", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    share = {}
    for d in (3, 4, 5, 6):
        for sc in gen.SCENARIOS:
            rng = np.random.default_rng(0)
            ok = sum(gen._saturated_box(gen.SIZE_FN[sc](d, rng)) <= gen.Q_BUDGET
                     for _ in range(draws))
            share[d, sc] = ok / draws
    d_max = max(d for d in (3, 4, 5, 6)
                if all(share[dd, sc] > 0 for dd in range(3, d + 1) for sc in gen.SCENARIOS))
    gap = [sc for sc in gen.SCENARIOS if share[d_max + 1, sc] == 0]
    return {"HmSkalBudget": num(gen.Q_BUDGET),
            "HmSkalZuege": num(draws),
            "HmSkalDMaxAlle": num(d_max),
            "HmSkalDLuecke": num(d_max + 1),
            "HmSkalSzenarioLuecke": ", ".join(sc.capitalize() for sc in gap)}


def tab_skalierung():
    # Runtime along the multiplicity axis.
    # On jansen_hm_scale, n grows at fixed n/m, so the capacity stays constant
    # and only the multiplicities u_i grow. FFD has to handle every task
    # individually, the high-multiplicity methods work on the d sizes --
    # exactly the difference that jansen_hm with n <= 160 cannot show.
    path = CSV_ROOT / SCALE_DATASET / "benchmark_hm.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    arms = [a for a in ARM_ORDER if f"time_{a}_ms" in df.columns]
    ns = sorted(df.n.unique())
    header = " & ".join(ARM_LABEL[a] for a in arms)
    lines = [r"\begin{tabular}{rr" + "r" * len(arms) + "}", r"\toprule",
             r"& & \multicolumn{%d}{c}{Median der Laufzeit [ms]} \\" % len(arms),
             r"\cmidrule(lr){3-%d}" % (2 + len(arms)),
             f"$n$ & $m$ & {header} \\\\", r"\midrule"]
    for n in ns:
        sub = df[df.n == n]
        vals = " & ".join(num(sub[f"time_{a}_ms"].median(), 1) for a in arms)
        lines.append(f"{num(n)} & {num(sub.m.iloc[0])} & {vals} \\\\")
    first, last = df[df.n == ns[0]], df[df.n == ns[-1]]
    factors = " & ".join(
        num(last[f"time_{a}_ms"].median() / first[f"time_{a}_ms"].median(), 1)
        for a in arms)
    lines += [r"\midrule",
              r"\multicolumn{2}{l}{Faktor} & " + factors + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_skalierung.tex", lines)

    # Where does the configuration LP overtake FFD? Log-linear interpolation
    # between the two sample points around the change of sign.
    crossing = None
    if BASELINE in arms and "gg_lp" in arms:
        med = {a: [df[df.n == n][f"time_{a}_ms"].median() for n in ns] for a in arms}
        for i in range(len(ns) - 1):
            d0 = np.log(med[BASELINE][i]) - np.log(med["gg_lp"][i])
            d1 = np.log(med[BASELINE][i + 1]) - np.log(med["gg_lp"][i + 1])
            if d0 < 0 <= d1:
                lam = -d0 / (d1 - d0)
                crossing = np.exp(np.log(ns[i]) + lam * (np.log(ns[i + 1]) - np.log(ns[i])))
                break

    macros = {"HmSkalInstanzen": num(len(df)), "HmSkalNMin": num(ns[0]),
              "HmSkalNMax": num(ns[-1]), "HmSkalMMax": num(df.m.max()),
              "HmSkalRatio": num(df.ratio.iloc[0]),
              "HmSkalNFaktor": num(ns[-1] / ns[0])}
    if crossing is not None:
        macros["HmSkalSchnittpunkt"] = num(round(crossing, -2))
    growth = {}
    for a in arms:
        tag = a.replace("_", "")
        growth[a] = last[f"time_{a}_ms"].median() / first[f"time_{a}_ms"].median()
        macros[f"HmSkalFaktor{tag}"] = num(growth[a], 1)
        macros[f"HmSkalZeitMax{tag}"] = num(last[f"time_{a}_ms"].median(), 1)
    # The range over the HM arms gets its own macros: which arm attains the
    # minimum changes between measurements -- a reference to a fixed arm would
    # silently make the range in the text wrong after the next measurement.
    hm = [a for a in arms if a != "ffd"]
    if hm:
        macros["HmSkalFaktorMin"] = num(min(growth[a] for a in hm), 1)
        macros["HmSkalFaktorMax"] = num(max(growth[a] for a in hm), 1)
    return macros


def load():
    df = pd.read_csv(CSV_ROOT / DATASET / "benchmark_hm.csv")
    df["opt"] = pd.to_numeric(df["opt"], errors="coerce")
    df["lb"] = np.maximum(np.ceil(df["sum_tasks"] / df["m"]), df["max_task"])
    arms = [a for a in ARM_ORDER if f"makespan_{a}" in df.columns]
    for a in arms:
        df[f"makespan_{a}"] = pd.to_numeric(df[f"makespan_{a}"], errors="coerce")
        df[f"ratio_opt_{a}"] = df[f"makespan_{a}"] / df["opt"]
        df[f"ratio_lb_{a}"] = df[f"makespan_{a}"] / df["lb"]
    return df, arms


def wtl(df, a, b):
    # win/tie/loss of b against a: 'win' means b returns the smaller makespan
    sub = df.dropna(subset=[f"makespan_{a}", f"makespan_{b}"])
    win = int((sub[f"makespan_{b}"] < sub[f"makespan_{a}"]).sum())
    loss = int((sub[f"makespan_{b}"] > sub[f"makespan_{a}"]).sum())
    return win, len(sub) - win - loss, loss, len(sub)


# --- Tables ------------------------------------------------------------------

def tab_datensatz(df):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Szenario & Instanzen & $n$ & $m$ & $d$ & $\rho$ \\", r"\midrule"]
    for scen in ["dense", "mixed", "asymmetric"]:
        s = df[df.scenario == scen]
        if s.empty:
            continue
        lines.append(
            f"{scen.capitalize()} & {num(len(s))} & "
            f"{num(s.n.min())}--{num(s.n.max())} & "
            f"{num(s.m.min())}--{num(s.m.max())} & "
            f"{num(s.d.min())}--{num(s.d.max())} & "
            f"{num(s.rho.min(), 2)}--{num(s.rho.max(), 2)} \\\\")
    lines += [r"\midrule",
              f"Gesamt & {num(len(df))} & {num(df.n.min())}--{num(df.n.max())} & "
              f"{num(df.m.min())}--{num(df.m.max())} & "
              f"{num(df.d.min())}--{num(df.d.max())} & "
              f"{num(df.rho.min(), 2)}--{num(df.rho.max(), 2)} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_datensatz.tex", lines)


def tab_paare(df, arms):
    # the arms directly against each other -- the actual pairwise comparison
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]
             if BASELINE not in (a, b) or a == BASELINE]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Referenz $A$ & Kandidat $B$ & $B$ besser & gleich & "
             r"$B$ schlechter & Instanzen \\", r"\midrule"]
    for a, b in pairs:
        win, tie, loss, tot = wtl(df, a, b)
        if tot == 0:
            continue
        lines.append(f"{ARM_LABEL[a]} & {ARM_LABEL[b]} & {num(win)} & "
                     f"{num(tie)} & {num(loss)} & {num(tot)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_paare.tex", lines)


def tab_ratio_opt(df, arms):
    # quality against OPT, binned by d -- the parameter the theory depends on
    d_opt = df[df.opt.notna() & ~df.degeneriert]
    header = " & ".join(ARM_LABEL[a] for a in arms)
    lines = [r"\begin{tabular}{lr" + "r" * len(arms) + "}", r"\toprule",
             r"& & \multicolumn{%d}{c}{Maximum von $C^{\mathrm{MF}}_{\max}/C^{*}_{\max}$} \\"
             % len(arms),
             r"\cmidrule(lr){3-%d}" % (2 + len(arms)),
             f"$d$ & Instanzen & {header} \\\\", r"\midrule"]
    for d in sorted(d_opt.d.unique()):
        s = d_opt[d_opt.d == d]
        vals = " & ".join(num(s[f"ratio_opt_{a}"].max(), 4) for a in arms)
        lines.append(f"{num(d)} & {num(len(s))} & {vals} \\\\")
    vals = " & ".join(num(d_opt[f"ratio_opt_{a}"].max(), 4) for a in arms)
    lines += [r"\midrule", f"alle $d$ & {num(len(d_opt))} & {vals} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_ratio_opt.tex", lines)


def tab_laufzeit(df, arms):
    header = " & ".join(ARM_LABEL[a] for a in arms)
    lines = [r"\begin{tabular}{lr" + "r" * len(arms) + "}", r"\toprule",
             r"& & \multicolumn{%d}{c}{Median der Laufzeit [ms]} \\" % len(arms),
             r"\cmidrule(lr){3-%d}" % (2 + len(arms)),
             f"$d$ & Instanzen & {header} \\\\", r"\midrule"]
    for d in sorted(df.d.unique()):
        s = df[df.d == d]
        vals = " & ".join(num(s[f"time_{a}_ms"].median(), 1) for a in arms)
        lines.append(f"{num(d)} & {num(len(s))} & {vals} \\\\")
    vals = " & ".join(num(df[f"time_{a}_ms"].median(), 1) for a in arms)
    lines += [r"\midrule", f"alle $d$ & {num(len(df))} & {vals} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_hm_laufzeit.tex", lines)


def write_macros(df, arms, extra=None):
    d_opt = df[df.opt.notna() & ~df.degeneriert]
    macros = {
        "HmInstanzen": num(len(df)),
        "HmInstanzenSzenario": num(len(df) // df.scenario.nunique()),
        "HmArme": num(len(arms)),
        "HmOptBekannt": num(int(df.opt.notna().sum())),
        "HmOhneOpt": num(int(df.opt.isna().sum())),
        "HmDegeneriert": num(int(df.degeneriert.sum())),
        "HmDAbweichung": num(int((df.d != df.d_nominal).sum())),
        "HmDMin": num(df.d.min()), "HmDMax": num(df.d.max()),
        "HmNMin": num(df.n.min()), "HmNMax": num(df.n.max()),
        "HmMMin": num(df.m.min()), "HmMMax": num(df.m.max()),
        "HmRhoMin": num(df.rho.min(), 2), "HmRhoMax": num(df.rho.max(), 2),
        "HmSchranke": num(BOUND, 4),
        # R_m(MF(k)) <= r_m + 2^-k (Theorem 3.12); for the measured runs this is
        # the bound that applies, 13/11 alone only holds in the limit.
        "HmSchrankeK": num(BOUND + 2.0 ** -int(df.k_iters.max()), 4),
        "HmDMinusEins": num(df.d.max() - 1),
    }
    for a in arms:
        tag = a.replace("_", "")
        macros[f"HmMaxRatioOpt{tag}"] = num(d_opt[f"ratio_opt_{a}"].max(), 4)
        macros[f"HmOptimal{tag}"] = pct(
            100 * (d_opt[f"ratio_opt_{a}"] == 1.0).mean(), 1)
        macros[f"HmZeitMedian{tag}"] = num(df[f"time_{a}_ms"].median(), 1)
        if a != BASELINE:
            win, tie, loss, _ = wtl(df, BASELINE, a)
            macros[f"HmWins{tag}"] = num(win)
            macros[f"HmTies{tag}"] = num(tie)
            macros[f"HmLosses{tag}"] = num(loss)
    if REFERENCE in arms:
        macros["HmMaxRatioOptReferenz"] = num(d_opt[f"ratio_opt_{REFERENCE}"].max(), 4)
    macros.update(extra or {})

    lines = ["% automatisch erzeugt von Empirical_Tests/summary_tables_hm.py",
             "% nicht von Hand bearbeiten"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    write("kennzahlen_hm.tex", lines)
    return macros


def main():
    df, arms = load()
    print(f"{len(df)} instances, arms: {' '.join(arms)}")
    tab_datensatz(df)
    tab_paare(df, arms)
    tab_ratio_opt(df, arms)
    tab_laufzeit(df, arms)
    t = load_trace(df)
    extra = {}
    if t is not None:
        extra.update(tab_akzeptanz(t) or {})
        extra.update(non_monotonicity_witness(df, t))
        extra.update(rounding_macros(t))
        extra.update(overload_macros(df, t))
        extra["HmJansenSchlechterFFD"] = num(int((df.makespan_jansen > df.makespan_ffd).sum()))
        extra["HmJansenBesserFFD"] = num(int((df.makespan_jansen < df.makespan_ffd).sum()))
        gg = t[t.arm == "gg_lp"]
        if not gg.empty:
            extra["HmGGAufrufe"] = num(len(gg))
            extra["HmGGZertifiziert"] = num(int((gg.accept_status == "zertifiziert_abgelehnt").sum()))
            extra["HmGGUnentschieden"] = num(int((gg.accept_status == "unentschieden").sum()))
            extra["HmGGUnentschiedenAnteil"] = pct(
                100 * (gg.accept_status == "unentschieden").mean(), 2)
            extra["HmGGRundungsverlustMax"] = num((gg.bins - gg.lp_bound).max())
            extra["HmGGOhneVerlust"] = pct(100 * (gg.bins == gg.lp_bound).mean(), 1)
            extra["HmQMax"] = num(gg.q.max())
            extra["HmQMedian"] = num(gg.q.median())
        if not gg.empty and gg.opt_bins_at_c.notna().any():
            dgg = (gg.bins - gg.opt_bins_at_c).dropna()
            extra["HmGGGleichOpt"] = num(int((dgg == 0).sum()))
            extra["HmGGPlusEins"] = num(int((dgg == 1).sum()))
            extra["HmGGMaxUeberOpt"] = num(dgg.max())
            extra["HmGGVergleiche"] = num(len(dgg))
        ja = t[t.arm == "jansen"]
        if not ja.empty and ja.opt_bins_at_c.notna().any():
            diff = (ja.bins - ja.opt_bins_at_c).dropna()
            extra["HmJansenVergleiche"] = num(len(diff))
            extra["HmJansenGleichOpt"] = num(int((diff == 0).sum()))
            extra["HmJansenPlusEins"] = num(int((diff == 1).sum()))
            extra["HmJansenUeberOpt"] = pct(100 * (diff == 1).mean(), 1)
            extra["HmJansenMaxUeberOpt"] = num(diff.max())
            extra["HmJansenMilpMax"] = num(ja.milp_steps.max())
            extra["HmJansenMilpMedian"] = num(ja.milp_steps.median())
            # binary search over {1, ..., n} (start value m* = n): at most
            # floor(log2 n) + 1 steps, each step one MILP
            extra["HmJansenMilpSchranke"] = num(math.floor(math.log2(df.n.max())) + 1)
            # Does one model of the OPT+1 algorithm cost more than one of the
            # configuration program, or is the difference only the number of
            # calls? Time per solved model instead of per capacity.
            extra["HmMsProMilpJansen"] = num((ja.t_ms / ja.milp_steps).median(), 2)
            ip = t[t.arm == REFERENCE]
            if not ip.empty:
                extra["HmMsProModellGgip"] = num(ip.t_ms.median(), 2)
        # Only the calls that actually produced a packing: for an infeasible
        # MILP the module returns a sentinel of empty bins, and achieved_eps
        # and last_faktor are 0 then.
        jd = t[t.arm == "jansen_dual"]
        if "unzulaessig" in jd:
            jd = jd[~jd.unzulaessig.astype(bool)]
        if not jd.empty and jd.achieved_eps.notna().any():
            extra["HmEpsMax"] = num(jd.achieved_eps.max(), 4)
            extra["HmEpsTheorieMin"] = num(jd.eps_theorie.min(), 4)
            extra["HmEpsTheorieMax"] = num(jd.eps_theorie.max(), 4)
            extra["HmLastFaktorMax"] = num(jd.last_faktor.max(), 4)
            extra["HmEpsPositiv"] = num(int((jd.achieved_eps > 0).sum()))
            extra["HmEpsAufrufe"] = num(len(jd))
    extra.update(degenerate_macros(df, arms))
    extra.update(d_benchmarks())
    extra.update(scale_d_budget())
    extra.update(tab_skalierung() or {})
    macros = write_macros(df, arms, extra)
    print(f"  -> {len(macros)} macros")


if __name__ == "__main__":
    main()
