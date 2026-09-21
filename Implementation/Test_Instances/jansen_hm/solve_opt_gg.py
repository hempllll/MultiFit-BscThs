"""Certified OPT values via the configuration IP.

By Lemma 2.5, C*_max(Gamma) <= C  <=>  OPT[Gamma,C] <= m, and OPT[Gamma,C'] <=
OPT[Gamma,C] for C' >= C. Hence C*_max(Gamma) = min{C : OPT[Gamma,C] <= m}, which
a binary search over C finds as long as OPT[Gamma,C] can be computed exactly --
the configuration IP does that (gilmore_gomory_bp with integer=True).

The optima are proven: gilmore_gomory_bp raises a RuntimeError for every status
other than OPTIMAL. The size of the IP depends only on d and C, not on n or m --
which is why the method also handles instances with very many machines.

Writes opt_known_gg.txt (lines "<instance file> <OPT>") and opt_certified.csv.

Usage:
  python Implementation/Test_Instances/jansen_hm/solve_opt_gg.py --workers 4
  python Implementation/Test_Instances/jansen_hm/solve_opt_gg.py \\
    --dir Implementation/Test_Instances/jansen_hm_scale --workers 4
"""
import argparse
import csv
import sys
import time
from math import ceil
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multifit import multifit
from binpacking.binpacking_algorithms import ffd_nlogn, group_by_size
from binpacking.gilmore_gomory_lp import gilmore_gomory_bp
from Empirical_Tests.run_benchmark import load_instance

HERE = Path(__file__).parent


def solve(path):
    m, _, tasks = load_instance(path)
    p, u = group_by_size(tasks)

    lo = max(ceil(sum(tasks) / m), max(tasks))
    # MULTIFIT+FFD yields a valid upper bound on OPT and narrows the search
    # interval -- usually just two or three IP calls per instance.
    bins = multifit(tasks, m, 10, ffd_nlogn, int_capacity=True)
    hi = max(b["load"] for b in bins)

    t0 = time.perf_counter()
    calls, q_max = 0, 0
    best = hi
    try:
        while lo < hi:
            c = (lo + hi) // 2
            packing, stats = gilmore_gomory_bp(p, u, c, integer=True)
            calls += 1
            q_max = max(q_max, stats["columns"])
            if len(packing) <= m:
                hi = best = c
            else:
                lo = c + 1
        status = "optimal"
    except RuntimeError as exc:
        return {"file": path.name, "opt": "", "status": str(exc)[:80],
                "ip_calls": calls, "q_max": q_max,
                "t_ms": round((time.perf_counter() - t0) * 1000, 1)}

    return {"file": path.name, "opt": best, "status": status,
            "ip_calls": calls, "q_max": q_max,
            "t_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=HERE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(p for p in args.dir.glob("*.txt")
                   if p.name not in ("opt_known_gg.txt", "readme.txt"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} instances, {args.workers} workers")

    t0 = time.perf_counter()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap(solve, paths, chunksize=4), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(paths):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(paths)}  {el:.0f}s", flush=True)

    rows.sort(key=lambda r: r["file"])
    with open(args.dir / "opt_certified.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "opt", "status", "ip_calls",
                                          "q_max", "t_ms"])
        w.writeheader()
        w.writerows(rows)
    with open(args.dir / "opt_known_gg.txt", "w") as f:
        for r in rows:
            if r["opt"] != "":
                f.write(f"{r['file']} {r['opt']}\n")

    solved = sum(1 for r in rows if r["opt"] != "")
    print(f"\n{solved}/{len(rows)} proven optimal, "
          f"{sum(r['ip_calls'] for r in rows)} IP calls, "
          f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
