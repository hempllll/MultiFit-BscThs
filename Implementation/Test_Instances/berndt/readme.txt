---------------------------------------------------------------------------------
BERNDT BENCHMARK
Source: Berndt, Deppert, Jansen, Rohwedder (2022), "Load Balancing: The Long Road
from Theory to Practice", ALENEX 2022.
Instance files and OPT values: Akram, Maas, Sanders, Schreiber (2025),
"Engineering Optimal Parallel Task Scheduling", ALENEX 2025,
repository https://github.com/anon495351/pcmax, commit 7ce7488 (2024-07-18),
directory benchmarks/berndt.

The instance files are NOT part of this repository (the source has no license).
Download and convert them with
  python Implementation/Test_Instances/fetch_pcmax.py
which checks every file against pcmax_manifest.sha256.

File name: "p_cmax-<TAG>-n<N>-m<M>-minsize<LO>-maxsize<HI>-seed<S>.txt"
  (S is the random seed of the original generator script)

Categories (sizes uniform in {LO, ..., HI}):
  E1   m in {3,4,5}, n in {2m, 3m, 5m};                 {1..20} and {20..50}
  E2   m in {2,3} with n in {10,30,50,100};
       m in {4,6,8,10} with n in {30,50,100};         {100..800}
  E3   m in {3,5,8,10}, n = mu*m + delta
       (mu in {3,4,5}, delta in {1,2});                {1..100} and {100..200}
  E4   (m,n) in {(2,10), (3,9)}; six ranges {1..20}, {20..50}, {1..100},
       {50..100}, {100..200}, {100..800}
  BIG  m in {25,50,75,100}, n = 4m;                     {1..1000}

30 * 102 = 3060 combinations, of which 3059 are in the original.

Format (after conversion):
  line 1: m (number of machines)
  line 2: n (number of jobs)
  line 3: p[1] p[2] ... p[n]

opt_known.txt: 3056 known optimal makespans from Akram et al. (2025).
---------------------------------------------------------------------------------
