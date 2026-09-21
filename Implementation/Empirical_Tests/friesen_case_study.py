"""
Case study (Section 5.1.8): the Friesen instance, on which MULTIFIT+FFD attains
its worst-case ratio 13/11.

Instance (Friesen 1984): 8x(40,13,13) + 3x(25,25,16) + 2x(25,24,17), m = 13.
Sum 858, OPT = 66 (13 machines with a load of exactly 66 each).

Output:
  - makespan of MULTIFIT+FFD and MULTIFIT+MFFD for k = 1..K_MAX
  - bin counts of both subroutines at fixed capacities (shows why the binary
    search converges differently, and the non-monotonicity of FFD)
  - MFFD class boundaries at the relevant capacities

Usage:
  python Implementation/Empirical_Tests/friesen_case_study.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binpacking.binpacking_algorithms import ffd_nlogn, mffd_nlogn
from multifit import multifit

M = 13
OPT = 66
K_MAX = 10
CAPACITIES = [66, 74, 75, 76, 77, 78, 79, 80]


def build_instance() -> list:
    tasks = []
    tasks += [40, 13, 13] * 8
    tasks += [25, 25, 16] * 3
    tasks += [25, 24, 17] * 2
    return tasks


def makespan(bins) -> float:
    return max(b["load"] for b in bins)


def classify(tasks, capacity):
    # MFFD classes (Johnson & Garey 1985): large > C/2, medium > C/3, small > C/6, otherwise tiny.
    classes = {"large": set(), "medium": set(), "small": set(), "tiny": set()}
    for t in tasks:
        if t > capacity / 2:
            classes["large"].add(t)
        elif t > capacity / 3:
            classes["medium"].add(t)
        elif t > capacity / 6:
            classes["small"].add(t)
        else:
            classes["tiny"].add(t)
    return {k: sorted(v, reverse=True) for k, v in classes.items()}


def main():
    tasks = build_instance()
    total = sum(tasks)
    rho = total / (M * max(tasks))

    print(f"n = {len(tasks)}, m = {M}, sum = {total}, p_max = {max(tasks)}, OPT = {OPT}")
    print(f"rho = sum / (m * p_max) = {rho:.4f}"
          f"  ->  {'< 2, the subroutines may differ' if rho < 2 else '>= 2'}")
    print()

    print("Makespan depending on the binary search depth k:")
    print(f"  {'k':>3}  {'MF+FFD':>8}  {'MF+MFFD':>8}")
    for k in range(1, K_MAX + 1):
        ms_ffd = makespan(multifit(tasks, M, k, ffd_nlogn))
        ms_mffd = makespan(multifit(tasks, M, k, mffd_nlogn))
        print(f"  {k:>3}  {ms_ffd:>8.0f}  {ms_mffd:>8.0f}")
    print()

    print(f"Bin counts at fixed capacity (m = {M}):")
    print(f"  {'C':>4}  {'FFD':>5}  {'MFFD':>5}   fits into m bins?")
    for c in CAPACITIES:
        n_ffd = len(ffd_nlogn(tasks, c))
        n_mffd = len(mffd_nlogn(tasks, c))
        flags = []
        if n_ffd <= M:
            flags.append("FFD")
        if n_mffd <= M:
            flags.append("MFFD")
        print(f"  {c:>4}  {n_ffd:>5}  {n_mffd:>5}   {', '.join(flags) if flags else '-'}")
    print()

    for c in (75, 78):
        cls = classify(tasks, c)
        print(f"MFFD classes at C = {c}  (C/2 = {c/2:g}, C/3 = {c/3:.4g}, C/6 = {c/6:g}):")
        for name in ("large", "medium", "small", "tiny"):
            print(f"  {name:>6}: {cls[name]}")
        print()

    ms_ffd = makespan(multifit(tasks, M, K_MAX, ffd_nlogn))
    ms_mffd = makespan(multifit(tasks, M, K_MAX, mffd_nlogn))
    print(f"Result at k = {K_MAX}:")
    print(f"  MF+FFD :  {ms_ffd:.0f}   ratio to OPT: {ms_ffd / OPT:.4f}  (13/11 = {13/11:.4f})")
    print(f"  MF+MFFD:  {ms_mffd:.0f}   ratio to OPT: {ms_mffd / OPT:.4f}")


if __name__ == "__main__":
    main()
