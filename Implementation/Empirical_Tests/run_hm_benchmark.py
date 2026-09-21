"""Measurement of the high-multiplicity subroutines inside MULTIFIT (Section 5.2).

The core of the evaluation is the comparison of the makespans MULTIFIT returns
with the different subroutines -- the same design as the FFD/MFFD comparison in
Section 5.1, only on a high-multiplicity dataset.

Writes two files to Empirical_Tests/{dataset}/:
  benchmark_hm.csv   one row per instance (makespan, runtime, status per arm)
  capacities_hm.csv  one row per (instance, capacity, arm) -- the trace

Some column names are German (degeneriert, opt_quelle, eps_theorie, ...); they
are the schema of the measured CSVs from which the thesis tables are generated.

Usage:
  python Implementation/Empirical_Tests/run_hm_benchmark.py \
    --instance-dir Implementation/Test_Instances/jansen_hm/ --k 10 --workers 4 --time-limit 60
"""
import argparse
import csv
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multifit import multifit
from binpacking.hm_wrappers import ARMS
from Empirical_Tests.run_benchmark import load_instance

DEFAULT_ARMS = ["ffd", "jansen", "jansen_dual", "gg_lp", "gg_ip"]

# Columns of the trace. Not every arm fills every column; missing ones stay empty.
TRACE_FIELDS = ["instance_file", "arm", "capacity", "bins", "t_ms", "accepted", "status",
                "lp_value", "lp_bound", "q", "support", "fractional",
                "floor_bins", "leftover_items", "leftover_bins",
                "leftover_source", "accept_status", "d",
                "m_star", "milp_steps", "kB", "kAll", "alpha",
                "achieved_eps", "eps_theorie", "max_load", "last_faktor",
                "unzulaessig"]


def instance_meta(path, tasks, m):
    counts = Counter(tasks)
    n = len(tasks)
    max_task = max(tasks)
    sum_tasks = sum(tasks)
    rho = sum_tasks / (m * max_task)
    # d from the content, not from the file name: the generator can round a
    # multiplicity down to 0, and then the instance has fewer sizes than its
    # name claims (two cases in jansen_hm).
    meta = {"instance_file": path.name, "m": m, "n": n, "max_task": max_task,
            "sum_tasks": sum_tasks, "rho": round(rho, 6),
            "d": len(counts), "d_nominal": "", "scenario": "",
            "ratio": round(n / m, 4),
            # rho < 1 means p_max > sum p / m. This implies OPT >= p_max
            # (Lemma 2.2), not OPT = p_max -- the lower bound is attained here
            # mostly, but not always (measured: 23 of 24).
            "degeneriert": rho < 1}
    parts = path.stem.split("_")
    if len(parts) >= 6 and parts[0] == "HM":
        meta["scenario"] = parts[1]
        meta["d_nominal"] = int(parts[2][1:])
    return meta


def run_instance(job):
    path, m, tasks, arm_names, k, time_limit = job
    if time_limit:
        import binpacking.gilmore_gomory_lp as gg
        import binpacking.jansen_high_multiplicity_binpacking as jn
        import binpacking.jansen_epsilon_dual_bp as jd
        gg.TIME_LIMIT = jn.TIME_LIMIT = jd.TIME_LIMIT = time_limit

    row = instance_meta(path, tasks, m)
    row["k_iters"] = k
    trace = []

    for name in arm_names:
        arm = ARMS[name](m)
        t0 = time.perf_counter()
        try:
            bins = multifit(tasks, m, k, arm, int_capacity=True)
            if not bins or all(b["load"] == 0 for b in bins):
                # sentinel of the eps-dual arm: infeasible at c_high
                raise RuntimeError("no valid packing at c_high")
            makespan = max(b["load"] for b in bins)
            status = "ok"
        except Exception as exc:                      # noqa: BLE001
            makespan, status = "", f"{type(exc).__name__}: {exc}"[:120]
        row[f"makespan_{name}"] = makespan
        row[f"time_{name}_ms"] = round((time.perf_counter() - t0) * 1000, 4)
        row[f"calls_{name}"] = len(arm.trace)
        row[f"status_{name}"] = status
        if name == "jansen_dual" and arm.trace:
            last = arm.trace[-1]
            row["eps_achieved_jansen_dual"] = last.get("achieved_eps", "")
            row["last_faktor_jansen_dual"] = last.get("last_faktor", "")
        for r in arm.trace:
            trace.append({"instance_file": path.name, **r})

    return row, trace


def case_study(path, m, tasks, arm_names, lo, hi, time_limit):
    # All arms on a contiguous range of capacities [lo, hi]. The binary search
    # visits different capacities per arm; a case study needs the same ones,
    # otherwise the comparison has gaps.
    if time_limit:
        import binpacking.gilmore_gomory_lp as gg
        import binpacking.jansen_high_multiplicity_binpacking as jn
        import binpacking.jansen_epsilon_dual_bp as jd
        gg.TIME_LIMIT = jn.TIME_LIMIT = jd.TIME_LIMIT = time_limit

    rows = []
    for C in range(lo, hi + 1):
        for name in arm_names:
            arm = ARMS[name](m)
            try:
                arm(tasks, C)
                r = arm.trace[-1]
                r["status"] = "ok"
            except Exception as exc:                  # noqa: BLE001
                r = {"arm": name, "capacity": C, "bins": "", "t_ms": "",
                     "accepted": "", "status": f"{type(exc).__name__}"[:60]}
            rows.append({"instance_file": path.name, **r})
    return rows


def build_fieldnames(arm_names):
    base = ["instance_file", "scenario", "d", "d_nominal", "m", "n", "ratio",
            "max_task", "sum_tasks", "rho", "degeneriert", "opt", "opt_quelle",
            "k_iters"]
    for name in arm_names:
        base += [f"makespan_{name}", f"time_{name}_ms",
                 f"calls_{name}", f"status_{name}"]
    if "jansen_dual" in arm_names:
        base += ["eps_achieved_jansen_dual", "last_faktor_jansen_dual"]
    return base


def stratified(paths, limit):
    # one instance per (scenario, d, m) -- so a pilot run hits every cell of the
    # grid instead of only the cheap start of the alphabetical order
    cells = {}
    for p in paths:
        parts = p.stem.split("_")
        key = (parts[1], parts[2], parts[4]) if len(parts) >= 6 else (p.stem,)
        cells.setdefault(key, []).append(p)
    picked = [sorted(v)[0] for v in cells.values()]
    picked.sort()
    return picked[:limit] if limit < len(picked) else picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                    choices=list(ARMS), metavar="ARM")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pilot", type=int, default=0,
                    help="stratified sample instead of all instances")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds per solver model")
    ap.add_argument("--resume", action="store_true",
                    help="skip instances that are already in the CSV")
    ap.add_argument("--case-study", metavar="FILE",
                    help="instead of the benchmark: all arms on a contiguous "
                         "capacity range of this instance -> case_study_hm.csv")
    ap.add_argument("--case-range", type=int, nargs=2, metavar=("LO", "HI"),
                    help="capacity range for --case-study")
    args = ap.parse_args()

    inst_dir = args.instance_dir
    out_dir = Path(__file__).parent / inst_dir.name
    out_dir.mkdir(exist_ok=True)
    bench_csv = out_dir / "benchmark_hm.csv"
    trace_csv = out_dir / "capacities_hm.csv"

    if args.case_study:
        path = inst_dir / args.case_study
        m, n, tasks = load_instance(path)
        lo, hi = args.case_range
        print(f"Case study {path.name}: m={m}, n={n}, C = {lo}..{hi}, "
              f"arms {' '.join(args.arms)}")
        rows = case_study(path, m, tasks, args.arms, lo, hi, args.time_limit)
        out = out_dir / "case_study_hm.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"-> {out}  ({len(rows)} rows)")
        return

    # certified OPT values from solve_opt_gg.py (configuration IP)
    opt_gg = {}
    gg_file = inst_dir / "opt_known_gg.txt"
    if gg_file.exists():
        for line in gg_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                opt_gg[parts[0]] = float(parts[1])

    paths = sorted(p for p in inst_dir.glob("*.txt")
                   if p.name not in ("opt_known_gg.txt", "readme.txt"))
    if args.pilot:
        paths = stratified(paths, args.pilot)

    done = set()
    if args.resume and bench_csv.exists():
        with open(bench_csv) as f:
            done = {r["instance_file"] for r in csv.DictReader(f)}
        paths = [p for p in paths if p.name not in done]

    jobs = []
    for p in paths:
        try:
            m, n, tasks = load_instance(p)
        except (ValueError, IndexError):
            continue
        jobs.append((p, m, tasks, args.arms, args.k, args.time_limit))

    print(f"{len(jobs)} instances, arms: {' '.join(args.arms)}, k={args.k}, "
          f"{args.workers} workers" + (f", {len(done)} skipped" if done else ""))

    fieldnames = build_fieldnames(args.arms)
    mode = "a" if (args.resume and bench_csv.exists()) else "w"
    t_start = time.perf_counter()
    with open(bench_csv, mode, newline="") as fb, open(trace_csv, mode, newline="") as ft:
        wb = csv.DictWriter(fb, fieldnames=fieldnames, extrasaction="ignore")
        wt = csv.DictWriter(ft, fieldnames=TRACE_FIELDS, extrasaction="ignore")
        if mode == "w":
            wb.writeheader()
            wt.writeheader()

        runner = Pool(args.workers).imap(run_instance, jobs, chunksize=1) \
            if args.workers > 1 else map(run_instance, jobs)
        for i, (row, trace) in enumerate(runner, 1):
            opt = opt_gg.get(row["instance_file"])
            row["opt"] = opt if opt is not None else ""
            row["opt_quelle"] = "gg_ip" if opt is not None else ""
            wb.writerow(row)
            wt.writerows(trace)
            fb.flush(); ft.flush()
            if i % 25 == 0 or i == len(jobs):
                el = time.perf_counter() - t_start
                print(f"  {i}/{len(jobs)}  {el:.0f}s  "
                      f"({el / i:.2f}s per instance)", flush=True)

    errors = 0
    with open(bench_csv) as f:
        for r in csv.DictReader(f):
            errors += sum(1 for a in args.arms
                          if r.get(f"status_{a}", "ok") not in ("ok", ""))
    print(f"\n-> {bench_csv}")
    print(f"-> {trace_csv}")
    if errors:
        print(f"WARNING: {errors} arm runs with error status")


if __name__ == "__main__":
    main()
