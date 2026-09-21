---------------------------------------------------------------------------------
LAWRINENKO BENCHMARK
Source: Lawrinenko; used in Mrad & Souayah (2018), "An Arc-Flow Model for the
Makespan Minimization Problem on Identical Parallel Machines", IEEE Access 6.
Instance files and OPT values: Akram, Maas, Sanders, Schreiber (2025),
"Engineering Optimal Parallel Task Scheduling", ALENEX 2025,
repository https://github.com/anon495351/pcmax, commit 7ce7488 (2024-07-18),
directory benchmarks/lawrinenko.

The instance files are NOT part of this repository (the source has no license).
Download and convert them with
  python Implementation/Test_Instances/fetch_pcmax.py
which checks every file against pcmax_manifest.sha256.

File name: "p_cmax-<class>-n<N>-m<M>-<params>-seed<S>.txt"
  (S is the random seed of the original generator script)

Classes:
  class1  uniform {1, ..., 100}
  class2  uniform {20, ..., 100}
  class3  uniform {50, ..., 100}
  class4  normal  N(100, 20^2)
  class5  normal  N(100, 50^2)
  class6  uniform {n, ..., 4n}
  class7  normal  N(4n, n^2)
  (normal values rounded to integers, non-positive values redrawn)

(n, m) pairs (50 in total):
  n in {20, 40, ..., 200}:  m = n/2 and m = floor(2n/5)
  n in {36, 54, ..., 198}:  m = n/3 and m = floor(4n/9)
  n in {22, 44, ..., 220}:  m = floor(4n/11)

50 * 7 * 10 = 3500 combinations, of which 3499 are in the original.

Format (after conversion):
  line 1: m (number of machines)
  line 2: n (number of jobs)
  line 3: p[1] p[2] ... p[n]

opt_known.txt: 3488 known optimal makespans from Akram et al. (2025).
---------------------------------------------------------------------------------
