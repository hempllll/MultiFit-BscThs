"""
Benchmark script: compare MULTIFIT with different bin-packing subroutines
on existing test instance sets.

One CSV is written per dataset (--instance-dir):
  Empirical_Tests/{dataset_name}/benchmark.csv

Usage:
  python Implementation/Empirical_Tests/run_benchmark.py \
    --algorithms ffd mffd \
    --instance-dir Implementation/Test_Instances/franca/ \
    --k 10
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binpacking.binpacking_algorithms import ffd_nlogn, mffd_nlogn
from multifit import multifit

ALGORITHM_MAP = {
    "ffd": ffd_nlogn,
    "mffd": mffd_nlogn,
}


def load_instance(path: Path):
    tokens = path.read_text().split()
    m = int(tokens[0])
    n = int(tokens[1])
    tasks = [int(x) for x in tokens[2:2 + n]]
    return m, n, tasks


def load_opt_known(directory: Path) -> dict:
    opt_file = directory / "opt_known.txt"
    if not opt_file.exists():
        return {}
    result = {}
    for line in opt_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            result[parts[0]] = int(parts[1])
    return result


def build_fieldnames(algorithm_names):
    base = ["instance_file", "m", "n", "max_task", "sum_tasks", "opt", "k_iters"]
    per_alg = [col for a in algorithm_names for col in (f"makespan_{a}", f"time_{a}_ms")]
    return base + per_alg


def run_benchmark_for_dir(algorithm_names, instance_dir, k):
    dataset = instance_dir.name
    out_path = Path(__file__).parent / dataset / "benchmark.csv"

    fieldnames = build_fieldnames(algorithm_names)
    rows = []
    opt_violations = []

    opt_known = load_opt_known(instance_dir)
    files = sorted(f for f in instance_dir.glob("*.txt") if f.name != "opt_known.txt")

    skipped = 0
    for i, filepath in enumerate(files, 1):
        if i % 100 == 0 or i == len(files):
            print(f"[{dataset}] {i}/{len(files)}", flush=True)

        try:
            m, n, tasks = load_instance(filepath)
        except (ValueError, IndexError):
            skipped += 1
            continue
        opt = opt_known.get(filepath.name)

        row = {
            "instance_file": filepath.name,
            "m": m,
            "n": n,
            "max_task": max(tasks),
            "sum_tasks": sum(tasks),
            "opt": opt if opt is not None else "",
            "k_iters": k,
        }

        for alg_name in algorithm_names:
            subroutine = ALGORITHM_MAP[alg_name]
            t0 = time.perf_counter()
            bins = multifit(tasks, m, k, subroutine)
            t1 = time.perf_counter()
            makespan = max(b["load"] for b in bins)
            row[f"makespan_{alg_name}"] = makespan
            row[f"time_{alg_name}_ms"] = round((t1 - t0) * 1000, 4)

            if opt is not None and makespan < opt:
                opt_violations.append(
                    f"  {filepath.name}: {alg_name} makespan={makespan} < opt={opt}"
                )

        rows.append(row)

    if skipped:
        print(f"  [{dataset}] {skipped} non-instance file(s) skipped")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if opt_violations:
        print(f"\nWARNING: {len(opt_violations)} makespan < opt violation(s) -- possibly wrong OPT values:")
        for msg in opt_violations:
            print(msg)
    print(f"Done. {len(rows)} instances -> {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MULTIFIT subroutines on test instance sets."
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=list(ALGORITHM_MAP), required=True,
        metavar="ALG", help=f"one or more of: {', '.join(ALGORITHM_MAP)}"
    )
    parser.add_argument(
        "--instance-dir", dest="instance_dirs", action="append", type=Path, required=True,
        metavar="DIR", help="path to a test instance directory (repeatable)"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="MULTIFIT binary search iterations (default: 10)"
    )
    args = parser.parse_args()

    for instance_dir in args.instance_dirs:
        run_benchmark_for_dir(args.algorithms, instance_dir, args.k)


if __name__ == "__main__":
    main()
