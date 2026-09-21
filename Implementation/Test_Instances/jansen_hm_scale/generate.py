"""Multiplicity axis for the high-multiplicity evaluation (HM-Skalierung, Section 5.2.6).

The dataset jansen_hm varies d, m and n/m, but keeps n at most 160. That does
not measure what distinguishes Jansen's algorithm in the first place: its
running time is polynomial in log n for fixed d. This dataset therefore varies
only n and keeps everything else fixed.

The key is to scale m with n. The capacities MULTIFIT tests lie around
C ~ sum p / m + p_max = ratio * p_mean + p_max; with fixed ratio = n/m, C stays
constant while n grows. Only then do the multiplicities u_i really grow without
the configuration set growing along -- with fixed m, C would grow linearly with
n and the enumeration would explode.

Size ranges and seed scheme are taken from jansen_hm/generate.py; the rounding
defect in _counts (a multiplicity could drop to 0) is fixed here.

The generated files are part of the repository; this script documents how they
were created. summary_tables_hm.py imports SCENARIOS, SIZE_FN, Q_BUDGET and
_saturated_box from here.

Usage:
  python Implementation/Test_Instances/jansen_hm_scale/generate.py
"""
import hashlib
import math
import os

import numpy as np

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
C_REF     = 100
D_VALUES  = [3, 4]
N_VALUES  = [40, 400, 4000, 40000]
RATIO     = 8                 # n/m fixed -> C stays constant along the n axis
N_SEEDS   = 5
SCENARIOS = ["dense", "mixed", "asymmetric"]

# Upper bound on the configuration set. Along the n axis q does not stay
# constant: for small n the multiplicity u_i binds, for large n floor(C/p_i).
# As soon as u_i >= floor(C/p_i) for all i, q saturates -- and this saturated
# value depends on the smallest drawn size. With p_min = 1 there are several
# million columns, which makes one LP per capacity useless.
#
# The check therefore uses the *saturated* value prod_i (floor(C/p_i) + 1), not
# the one actually reached at this n. Only then does the acceptance of a size
# combination not depend on n -- otherwise the size distribution would be
# shifted upwards systematically for large n, and the comparison along the
# n axis would mix two effects.
Q_BUDGET  = 200_000
MAX_DRAWS = 500


def _large_range(d):
    return C_REF // (2 * d - 1) + 1, C_REF // 2


def _small_range(d):
    return 1, max(1, C_REF // (2 * d - 1) - 1)


def _sizes_dense(d, rng):
    lo, hi = _large_range(d)
    return sorted(rng.choice(np.arange(lo, hi + 1), size=d,
                             replace=False).tolist(), reverse=True)


def _sizes_mixed(d, rng):
    n_large = math.ceil(d / 2)
    lo_l, hi_l = _large_range(d)
    lo_s, hi_s = _small_range(d)
    large = rng.choice(np.arange(lo_l, hi_l + 1), size=n_large, replace=False).tolist()
    small = rng.choice(np.arange(lo_s, hi_s + 1), size=d - n_large, replace=False).tolist()
    return sorted(large + small, reverse=True)


def _sizes_asymmetric(d, rng):
    _, hi_l = _large_range(d)
    lo_s, hi_s = _small_range(d)
    dominant = int(rng.integers(C_REF // 3, hi_l + 1))
    small = rng.choice(np.arange(lo_s, hi_s + 1), size=d - 1, replace=False).tolist()
    return sorted([dominant] + small, reverse=True)


SIZE_FN = {"dense": _sizes_dense, "mixed": _sizes_mixed,
           "asymmetric": _sizes_asymmetric}


def _saturated_box(sizes):
    # prod_i (floor(C/p_i) + 1) with C ~ RATIO * p_mean + p_max, i.e. the value
    # that |K(Gamma, C)| converges to as n grows. Depends only on the sizes,
    # not on n.
    C = RATIO * (sum(sizes) // len(sizes)) + max(sizes)
    box = 1
    for s in sizes:
        box *= C // s + 1
    return box


def _draw(d, n, scenario, rng):
    for _ in range(MAX_DRAWS):
        sizes = SIZE_FN[scenario](d, rng)
        if _saturated_box(sizes) <= Q_BUDGET:
            return sizes, _counts(d, n, scenario, rng)
    raise RuntimeError(f"no draw within Q_BUDGET for d={d}, {scenario}")


def _counts(d, n, scenario, rng):
    # Every size gets at least one job, the rest is distributed by a Dirichlet draw.
    extra_n = n - d
    alpha = np.ones(d)
    if scenario == "asymmetric":
        alpha[0] = 4.0
    extra = np.round(rng.dirichlet(alpha) * extra_n).astype(int)
    diff = extra_n - int(extra.sum())
    # Settle the rounding remainder only where it does not go below zero --
    # otherwise a multiplicity drops to 0 and the instance has fewer than d sizes.
    while diff != 0:
        i = int(rng.integers(d))
        if diff > 0:
            extra[i] += 1
            diff -= 1
        elif extra[i] > 0:
            extra[i] -= 1
            diff += 1
    counts = (1 + extra).tolist()
    assert min(counts) >= 1 and sum(counts) == n
    return counts


def _seed(scenario, d, n, m, k):
    key = f"jansen_hm_scale_{scenario}_d{d}_n{n}_m{m}_k{k}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


def generate():
    written = 0
    for d in D_VALUES:
        for scenario in SCENARIOS:
            for n in N_VALUES:
                m = n // RATIO
                for k in range(N_SEEDS):
                    rng = np.random.default_rng(_seed(scenario, d, n, m, k))
                    sizes, counts = _draw(d, n, scenario, rng)
                    assert len(set(sizes)) == d

                    jobs = []
                    for sz, cnt in zip(sizes, counts):
                        jobs.extend([sz] * cnt)
                    rng.shuffle(jobs)

                    name = f"HMS_{scenario}_d{d}_n{n:06d}_m{m:05d}_{k}.txt"
                    with open(os.path.join(OUT_DIR, name), "w") as f:
                        f.write(f"{m}\n{n}\n")
                        f.write(" ".join(str(int(j)) for j in jobs) + "\n")
                    written += 1
    return written


if __name__ == "__main__":
    n_files = generate()
    print(f"{n_files} instances written to {OUT_DIR}")
