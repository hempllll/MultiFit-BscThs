---------------------------------------------------------------------------------
FRANGIONI BENCHMARK
Source: Frangioni, Necciari, Scutellà (2004), J. Combinatorial Optimization 8(2).
Instance files and OPT values: Akram, Maas, Sanders, Schreiber (2025),
"Engineering Optimal Parallel Task Scheduling", ALENEX 2025,
repository https://github.com/anon495351/pcmax, commit 7ce7488 (2024-07-18),
directory benchmarks/frangioni/INSTANCES.

The instance files are NOT part of this repository (the source has no license).
Download and convert them with
  python Implementation/Test_Instances/fetch_pcmax.py
which checks every file against pcmax_manifest.sha256.

File name: "{U|NU}_{R}_{n:04d}_{m:02d}_{k}.txt"
  U  : sizes uniform in {1, ..., R}
  NU : bimodal, with probability 1/2 uniform in {1, ..., floor(R/3)},
       otherwise uniform in {floor(2R/3), ..., R}
  R  : 1 -> 100, 2 -> 1000, 3 -> 10000
  n  : 10, 50, 100, 500, 1000
  m  : 5, 10, 25
  k  : index 0-9

6 * 15 * 10 = 900 combinations; the original contains the 780 with n > m
(the pairs (n, m) = (10, 10) and (10, 25) are missing). Five further files in
the source directory with m >= 92 are not used.

Format (after conversion):
  line 1: m (number of machines)
  line 2: n (number of jobs)
  line 3: p[1] p[2] ... p[n]

opt_known.txt: 747 known optimal makespans from Akram et al. (2025).
---------------------------------------------------------------------------------
