"""
Sensitivity of MULTIFIT to the binary search depth k (Section 5.1.7).

Runs MULTIFIT with FFD and MFFD for k = 1..K_MAX on all instances of the given
datasets and writes one row per (instance, k) to
Empirical_Tests/k_sensitivity.csv.

Background: R_m(MF(k)) <= r_m + 2^{-k} (Theorem 3.12). The CSV allows comparing
the measured course of makespan/OPT over k with this bound.

Usage:
  python Implementation/Empirical_Tests/k_sensitivity.py
  python Implementation/Empirical_Tests/k_sensitivity.py --datasets franca frangioni
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binpacking.binpacking_algorithms import ffd_nlogn, mffd_nlogn
from multifit import multifit
from Empirical_Tests.run_benchmark import load_instance, load_opt_known

DATASETS = ["franca", "frangioni", "lawrinenko", "berndt", "planted"]
K_MAX = 10
ALGORITHMS = {"ffd": ffd_nlogn, "mffd": mffd_nlogn}

INSTANCE_ROOT = Path(__file__).parents[1] / "Test_Instances"
OUT_PATH = Path(__file__).parent / "k_sensitivity.csv"


def run(datasets, k_max):
    fieldnames = ["dataset", "instance_file", "m", "n", "opt", "k",
                  "makespan_ffd", "makespan_mffd"]
    rows = []

    for dataset in datasets:
        instance_dir = INSTANCE_ROOT / dataset
        opt_known = load_opt_known(instance_dir)
        files = sorted(f for f in instance_dir.glob("*.txt") if f.name != "opt_known.txt")

        t0 = time.perf_counter()
        for i, filepath in enumerate(files, 1):
            if i % 250 == 0 or i == len(files):
                print(f"[{dataset}] {i}/{len(files)}", flush=True)
            try:
                m, n, tasks = load_instance(filepath)
            except (ValueError, IndexError):
                continue
            opt = opt_known.get(filepath.name)

            for k in range(1, k_max + 1):
                row = {
                    "dataset": dataset,
                    "instance_file": filepath.name,
                    "m": m,
                    "n": n,
                    "opt": opt if opt is not None else "",
                    "k": k,
                }
                for name, subroutine in ALGORITHMS.items():
                    bins = multifit(tasks, m, k, subroutine)
                    row[f"makespan_{name}"] = max(b["load"] for b in bins)
                rows.append(row)
        print(f"[{dataset}] done in {time.perf_counter() - t0:.1f} s", flush=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. {len(rows)} rows -> {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of MULTIFIT to the binary search depth k.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, metavar="DS")
    parser.add_argument("--k-max", type=int, default=K_MAX)
    args = parser.parse_args()
    run(args.datasets, args.k_max)


if __name__ == "__main__":
    main()
