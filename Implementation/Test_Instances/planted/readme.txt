---------------------------------------------------------------------------------
PLANTED BENCHMARK
Source: Akram, Maas, Sanders, Schreiber (2025), "Engineering Optimal Parallel
Task Scheduling", ALENEX 2025,
repository https://github.com/anon495351/pcmax, commit 7ce7488 (2024-07-18),
directory benchmarks/planted (original generator: generate-planted.py).

CONSTRUCTION
  Every one of the m machines starts with U time slots, all belonging to a
  single job. Contiguous blocks of existing jobs are split off as new jobs
  until there are n jobs. No time slot is lost, so the optimal makespan is U.
  Perturbation r > 0: ceil(r * n) jobs are increased by 1, so the total load
  exceeds m*U and the optimal makespan is at least U+1.

CONTENTS
1) 1064 original instances, file names with "-exact-":
     p_cmax-n{n}-m{m}-planted-exact-U{U}-perturb{r}.txt
   NOT part of this repository (the source has no license). Download and
   convert them with
     python Implementation/Test_Instances/fetch_pcmax.py
   which checks every file against pcmax_manifest.sha256.

2) 540 self-generated instances, file names ending in "-seed{k}.txt":
     p_cmax-n{n}-m{m}-planted-U{U}-perturb{r}-seed{k}.txt
   Part of this repository, created by generate.py (a reimplementation of the
   original generator with deterministic MD5 seeds).
     (n, m)  : (10,4), (20,5), (50,10), (100,20), (200,50), (500,100)
     U       : 100, 300, 1000
     r       : 0, 0.05, 0.1
     k       : 0-9
     total   : 6 * 3 * 3 * 10 = 540

FORMAT
  line 1: m (number of machines)
  line 2: n (number of jobs)
  line 3: p[1] p[2] ... p[n] (descending)

opt_known.txt: 1218 known optimal makespans
  461 with r = 0 (OPT = U by construction; 281 original, 180 generated)
  757 original instances with r > 0 (exact solver of Akram et al. 2025)
---------------------------------------------------------------------------------
