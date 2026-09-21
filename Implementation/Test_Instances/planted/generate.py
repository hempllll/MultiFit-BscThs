"""
Planted instance generator

Reimplements the generator of the planted benchmark instances from the P||Cmax
repository of Akram, Maas, Sanders, Schreiber (2025):
  https://github.com/anon495351/pcmax  (benchmarks/planted/generate-planted.py)

This script only generates the 540 self-generated instances (file names ending in
-seed{k}.txt). The 1064 original instances (-exact- in the file name) come from
that repository and are downloaded by ../fetch_pcmax.py.

Algorithm:
  Start with an occupancy matrix job_matrix[i] = [i]*U for every machine
  i in 0..m-1, i.e. every machine has exactly U time slots, all belonging to
  job i. New jobs are created iteratively by splitting off a contiguous block
  of an existing job. The result is an instance with optimal makespan U.

  Perturbation: ceil(perturb_ratio * n) jobs are increased by 1, so the total
  load exceeds m*U and the optimal makespan is at least U+1.

Deterministic seeds:
  seed = MD5("planted_{n}_{m}_{U}_{r}_{k}") mod 2**31

Parameter space:
  N_M_PAIRS      : (n, m) combinations
  U_VALUES       : capacity values (= OPT for perturb = 0)
  PERTURB_VALUES : perturbation rate
  N_SEEDS        : seeds per configuration (k = 0 .. N_SEEDS-1)

Format (identical to all other datasets):
  line 1: m
  line 2: n
  line 3: p[1] p[2] ... p[n]

File name:
  p_cmax-n{n}-m{m}-planted-U{U}-perturb{r}-seed{k}.txt

The generated files are part of the repository; this script documents how they
were created.
"""

import hashlib
import math
import os
import random

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
N_M_PAIRS = [(10, 4), (20, 5), (50, 10), (100, 20), (200, 50), (500, 100)]
U_VALUES = [100, 300, 1000]
PERTURB_VALUES = [0, 0.05, 0.1]
N_SEEDS = 10


def _seed(n: int, m: int, U: int, r: float, k: int) -> int:
    key = f"planted_{n}_{m}_{U}_{r}_{k}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def _generate(n: int, m: int, U: int, perturb_ratio: float, seed: int) -> list[int]:
    # generates one planted instance and returns the job sizes in descending order
    random.seed(seed)

    # initialization: job_matrix[i] = [i]*U
    job_matrix = [[i] * U for i in range(m)]
    nb_jobs = m

    while nb_jobs < n:
        # every machine is cut at least once (the first m splits)
        if nb_jobs < 2 * m:
            rand_i = nb_jobs - m
        else:
            rand_i = random.randrange(0, len(job_matrix))

        rand_t = random.randrange(0, len(job_matrix[rand_i]))

        # invalid cut: hit the first slot of the job
        if rand_t == 0 or job_matrix[rand_i][rand_t - 1] != job_matrix[rand_i][rand_t]:
            continue

        # reassign the block starting at rand_t to the new job
        t = rand_t
        oldjob = job_matrix[rand_i][rand_t]
        while t < len(job_matrix[rand_i]) and job_matrix[rand_i][t] == oldjob:
            job_matrix[rand_i][t] = nb_jobs
            t += 1

        nb_jobs += 1

    # compute the sizes
    sizes: dict[int, int] = {}
    for row in job_matrix:
        for j in row:
            sizes[j] = sizes.get(j, 0) + 1

    # Perturbation
    if perturb_ratio > 0:
        jlist = list(sizes.keys())
        random.shuffle(jlist)
        choice = jlist[: math.ceil(perturb_ratio * len(sizes))]
        for j in choice:
            sizes[j] += 1

    return sorted(sizes.values(), reverse=True)


def _perturb_str(r: float) -> str:
    # formats perturb_ratio for the file name (0 -> '0', 0.05 -> '0.05')
    if r == 0:
        return "0"
    return str(r)


def generate_planted():
    written = 0
    total = len(N_M_PAIRS) * len(U_VALUES) * len(PERTURB_VALUES) * N_SEEDS

    for n, m in N_M_PAIRS:
        for U in U_VALUES:
            for r in PERTURB_VALUES:
                for k in range(N_SEEDS):
                    seed = _seed(n, m, U, r, k)
                    jobs = _generate(n, m, U, r, seed)

                    r_str = _perturb_str(r)
                    filename = f"p_cmax-n{n}-m{m}-planted-U{U}-perturb{r_str}-seed{k}.txt"
                    filepath = os.path.join(OUT_DIR, filename)

                    with open(filepath, "w") as f:
                        f.write(f"{m}\n{n}\n")
                        f.write(" ".join(map(str, jobs)) + "\n")

                    written += 1

    print(f"Planted: {written} instances written to {OUT_DIR}/")
    assert written == total, f"expected {total}, got {written}"


if __name__ == "__main__":
    generate_planted()
