"""
High-multiplicity benchmark generator (HM grid, Section 5.2.1)

Instances with d in {3,4,5,6} distinct job sizes, built for MULTIFIT with the
high-multiplicity subroutines. Three structurally different scenarios per d:

  dense       -- all d sizes in the large range (> eps*C_ref), several fit into
                 one bin. Tests Jansen at maximal MILP effort.

  mixed       -- ceil(d/2) large sizes + floor(d/2) small sizes. The typical mix
                 in which the MILP handles the large items and the small items
                 are placed greedily.

  asymmetric  -- 1 dominant large size (~60% of all jobs) + d-1 small sizes.
                 The MILP solves only for the dominant size; many small
                 leftovers. Tests Jansen's greedy step 3.

Sizes relative to C_ref = 100 (approximate capacity for MULTIFIT):

  large sizes: [floor(C_ref/(2d-1))+1, C_ref//2]
  small sizes: [1, floor(C_ref/(2d-1))-1]

  d=3 (eps=1/5) : large=[21,50], small=[1,19]
  d=4 (eps=1/7) : large=[15,50], small=[1,13]
  d=5 (eps=1/9) : large=[12,50], small=[1,10]
  d=6 (eps=1/11): large=[10,50], small=[1, 8]

Note on the actual classification in Jansen:
  C_actual ~ avg_size * n/m. At ratio=3 almost all dense items are large
  (n/m < 2d-1 = 5 for d=3). At ratio=8 they can fall into the small range for
  small d -- the behaviour in this regime is explicitly part of the experiment.

n = ratio * m, so that the ratio n/m is controlled:
  ratio in {3, 5, 8},  m in {5, 10, 20}

Every combination (d, scenario, m, ratio) is instantiated with 10 deterministic
seeds (MD5-based, reproducible).

Total: 4 * 3 * 3 * 3 * 10 = 1080 instances

File name: HM_{scenario}_d{d}_n{n:04d}_m{m:02d}_{k}.txt
Format:
  line 1: m
  line 2: n  (= sum of u_i, all jobs)
  line 3: p[1] ... p[n]  (in random order)

Known defect, kept so that the instances stay reproducible: _counts can round a
multiplicity down to 0. Two instances therefore have fewer sizes than their
name claims (HM_asymmetric_d5_n0080_m10_8 has d=4, HM_asymmetric_d6_n0025_m05_1
has d=5); the evaluation reads d from the content. jansen_hm_scale/generate.py
fixes the rounding.

The generated files are part of the repository; this script documents how they
were created (numpy's Generator does not guarantee the same stream across versions).
"""

import hashlib
import math
import os
import numpy as np

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
C_REF    = 100
D_VALUES = [3, 4, 5, 6]
M_VALUES = [5, 10, 20]
RATIOS   = [3, 5, 8]       # n = ratio * m
N_SEEDS  = 10
SCENARIOS = ["dense", "mixed", "asymmetric"]


# ---------------------------------------------------------------------------
# Helpers: size ranges
# ---------------------------------------------------------------------------

def _large_range(d: int):
    lo = C_REF // (2 * d - 1) + 1
    hi = C_REF // 2
    return lo, hi   # [lo, hi] inclusive


def _small_range(d: int):
    hi = max(1, C_REF // (2 * d - 1) - 1)
    return 1, hi    # [1, hi] inclusive


# ---------------------------------------------------------------------------
# Size selection per scenario
# ---------------------------------------------------------------------------

def _sizes_dense(d: int, rng: np.random.Generator) -> list[int]:
    lo, hi = _large_range(d)
    return sorted(
        rng.choice(np.arange(lo, hi + 1), size=d, replace=False).tolist(),
        reverse=True
    )


def _sizes_mixed(d: int, rng: np.random.Generator) -> list[int]:
    n_large = math.ceil(d / 2)
    n_small = d - n_large
    lo_l, hi_l = _large_range(d)
    lo_s, hi_s = _small_range(d)
    large = rng.choice(np.arange(lo_l, hi_l + 1), size=n_large, replace=False).tolist()
    small = rng.choice(np.arange(lo_s, hi_s + 1), size=n_small, replace=False).tolist()
    return sorted(large + small, reverse=True)


def _sizes_asymmetric(d: int, rng: np.random.Generator) -> list[int]:
    _, hi_l    = _large_range(d)
    lo_s, hi_s = _small_range(d)
    # dominant size: upper part of the large range
    dom_lo = C_REF // 3
    dom_hi = hi_l
    dominant = int(rng.integers(dom_lo, dom_hi + 1))
    n_small   = d - 1
    small = rng.choice(np.arange(lo_s, hi_s + 1), size=n_small, replace=False).tolist()
    return sorted([dominant] + small, reverse=True)


# ---------------------------------------------------------------------------
# Multiplicities (intended: at least 1 job per size, see the known defect above)
# ---------------------------------------------------------------------------

def _counts(d: int, n: int, scenario: str, rng: np.random.Generator) -> list[int]:
    # Guaranteed 1 per type, then distribute the rest
    extra_n = n - d
    if extra_n < 0:
        raise ValueError(f"n={n} < d={d}: cannot use every size at least once")
    if scenario == "asymmetric":
        alpha = np.ones(d)
        alpha[0] = 4.0   # the dominant size gets ~60% of the remaining jobs
    else:
        alpha = np.ones(d)
    fracs = rng.dirichlet(alpha)
    extra  = np.round(fracs * extra_n).astype(int)
    diff   = extra_n - extra.sum()
    extra[rng.integers(d)] += diff
    return (1 + extra).tolist()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def _seed(scenario: str, d: int, n: int, m: int, k: int) -> int:
    key = f"jansen_hm_{scenario}_d{d}_n{n}_m{m}_k{k}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

SIZE_FN = {
    "dense":      _sizes_dense,
    "mixed":      _sizes_mixed,
    "asymmetric": _sizes_asymmetric,
}


def generate_jansen_hm():
    written = 0

    for d in D_VALUES:
        for scenario in SCENARIOS:
            for m in M_VALUES:
                for ratio in RATIOS:
                    n = ratio * m
                    if n < d:
                        continue   # too few jobs for d sizes (does not occur here)
                    for k in range(N_SEEDS):
                        s = _seed(scenario, d, n, m, k)
                        rng = np.random.default_rng(s)

                        sizes  = SIZE_FN[scenario](d, rng)
                        counts = _counts(d, n, scenario, rng)
                        assert sum(counts) == n
                        assert len(set(sizes)) == d

                        # build the job list and shuffle it
                        jobs = []
                        for sz, cnt in zip(sizes, counts):
                            jobs.extend([sz] * cnt)
                        rng.shuffle(jobs)

                        filename = f"HM_{scenario}_d{d}_n{n:04d}_m{m:02d}_{k}.txt"
                        with open(os.path.join(OUT_DIR, filename), "w") as f:
                            f.write(f"{m}\n{n}\n")
                            f.write(" ".join(map(str, jobs)) + "\n")
                        written += 1

    _write_readme()
    print(f"Jansen HM: {written} instances written to {OUT_DIR}/")
    assert written == 1080, f"expected 1080, got {written}"


def _write_readme():
    eps = {d: f"1/{2*d-1}" for d in D_VALUES}
    lo_large = {d: C_REF // (2*d-1) + 1 for d in D_VALUES}
    hi_small = {d: max(1, C_REF // (2*d-1) - 1) for d in D_VALUES}

    with open(os.path.join(OUT_DIR, "readme.txt"), "w") as f:
        f.write(f"""\
---------------------------------------------------------------------------------
HM GRID (directory jansen_hm, called "HM-Gitter" in the thesis, Section 5.2.1)
Test set for MULTIFIT with the high-multiplicity subroutines (Jansen OPT+1,
its eps-dual variant, configuration LP/IP). Generated by generate.py.

Motivation: the algorithm of Jansen & Solis-Oba (2010) works on d distinct
sizes; its configuration enumeration is only practical for small d.
The three scenarios cover structurally different packing situations.

PARAMETERS
  d        : {D_VALUES}
  scenario : dense, mixed, asymmetric (see below)
  m        : {M_VALUES}
  ratio    : {RATIOS}  ->  n = ratio * m
  seeds    : 10 per combination (deterministic, MD5-based)
  total    : {len(D_VALUES)} * 3 * {len(M_VALUES)} * {len(RATIOS)} * 10 = 1080

SIZE RANGES (C_ref = {C_REF})
  d  eps      large=[lo, 50]  small=[1, hi]
""")
        for d in D_VALUES:
            f.write(f"  {d}  {eps[d]:5s}   lo={lo_large[d]:3d}          hi={hi_small[d]:3d}\n")

        f.write(f"""
SCENARIOS
  dense       all d sizes in the large range [lo, 50].
              Several items fit into one bin; maximal MILP complexity.
              For n/m < 2d-1 all items are actually large in Jansen's
              sense (threshold = C/(2d-1) < lo).

  mixed       ceil(d/2) large + floor(d/2) small sizes.
              Typical instance: the MILP covers the large items, the small
              ones are distributed greedily in step 3.

  asymmetric  1 dominant size from [33, 50] with ~60% of all jobs,
              d-1 small sizes from [1, hi].
              Tests Jansen's greedy placement with many leftovers.

MULTIPLICITIES
  dense/mixed : Dirichlet(1,...,1) -- balanced at random
  asymmetric  : Dirichlet(4,1,...,1) -- size 1 dominant
  Intended: every size gets >= 1 job. The rounding can set a multiplicity
  to 0; this happens in HM_asymmetric_d5_n0080_m10_8 (d=4) and
  HM_asymmetric_d6_n0025_m05_1 (d=5).

FORMAT
  line 1: m (number of machines)
  line 2: n (total number of jobs = sum of u_i)
  line 3: p[1] ... p[n] (processing times, shuffled)

FILE NAME: HM_<scenario>_d<D>_n<NNNN>_m<MM>_<K>.txt

NOTE ON THE RATIO n/m
  C_actual ~ avg_size * n/m. Jansen classifies an item as large if
  size >= C/(2d-1). For dense at ratio=3 almost all items lie above this
  threshold. At ratio=8, small d can slip into the range where all items
  become small -- this transition is explicitly part of the experiment.

OPT
  opt_known_gg.txt / opt_certified.csv: certified optimal makespans of all
  1080 instances, computed by solve_opt_gg.py (binary search over C with the
  configuration IP, Lemma 2.5).
---------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    generate_jansen_hm()
