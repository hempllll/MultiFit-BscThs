# Analyse und Optimierung von MULTIFIT Algorithmen

Code, data and LaTeX sources of the bachelor's thesis *Analyse und Optimierung von
MULTIFIT Algorithmen* (Fridjoff Hempel, Kiel University, 2026; written in German).
The thesis itself is [`Thesis/bachelorarbeit.pdf`](Thesis/bachelorarbeit.pdf).

MULTIFIT schedules n tasks on m identical machines (P||C_max) by a binary search over
a bin capacity C, calling First Fit Decreasing (FFD) as a bin-packing subroutine. The
thesis works through the performance analysis of MULTIFIT (bound 13/11, Cao 1995) and
replaces FFD by other subroutines: Modified First Fit Decreasing (MFFD), the OPT+1
algorithm of Jansen & Solis-Oba, an ε-dual variant of it and the configuration LP of
Gilmore & Gomory. This repository contains the implementations, the instance
generators and all scripts that produce the numbers, tables and figures of
Chapter 5.

## Layout

```
Implementation/
  multifit.py                   MULTIFIT: binary search over the capacity
  binpacking/
    binpacking_algorithms.py    FFD (O(n log n), segment tree), MFFD (Johnson & Garey 1985)
    jansen_high_multiplicity_binpacking.py   OPT+1 algorithm (Section 4.2)
    jansen_epsilon_dual_bp.py   ε-dual variant (Section 4.2.4)
    gilmore_gomory_lp.py        configuration LP/IP with rounding (Section 4.3)
    hm_wrappers.py              subroutines with call trace for the Section 5.2 pipeline
  Empirical_Tests/              measurement and evaluation scripts, measured CSVs
  Test_Instances/               datasets, instance generators, OPT solvers
  requirements.txt
Thesis/                         LaTeX sources; tables/ and figures/ are generated
```

## Setup

Python 3.11, dependencies from `requirements.txt`:

```bash
python3 -m venv Implementation/.venv
source Implementation/.venv/bin/activate
pip install -r Implementation/requirements.txt
```

The Jansen and configuration-LP modules, the certified OPT computation
and the Section 5.2 measurements use Gurobi and need a Gurobi license (free for
academic use). FFD, MFFD, the Section 5.1 measurements and all table and figure
scripts run without Gurobi.

## Datasets

All instance files share one format: line 1 `m`, line 2 `n`, line 3 the `n` task sizes.

| Directory | Name in the thesis | Instances | Origin | In this repository |
|---|---|---|---|---|
| `franca` | França | 150 | `generate.py` (parameters of França et al. 1994) | yes |
| `frangioni` | Frangioni | 780 | original files, Akram et al. 2025 | no, `fetch_pcmax.py` |
| `lawrinenko` | Lawrinenko | 3 499 | original files, Akram et al. 2025 | no, `fetch_pcmax.py` |
| `berndt` | Berndt | 3 059 | original files, Akram et al. 2025 | no, `fetch_pcmax.py` |
| `planted` | Planted | 1 064 + 540 | original files (`-exact-`) + `generate.py` (`-seed{k}`) | the 540 generated ones |
| `jansen_hm` | HM-Gitter | 1 080 | `generate.py` | yes |
| `jansen_hm_scale` | HM-Skalierung | 120 | `generate.py` | yes |

The original instances come from the repository of Akram, Maas, Sanders, Schreiber,
*Engineering Optimal Parallel Task Scheduling* (ALENEX 2025),
<https://github.com/anon495351/pcmax>, which also provides their known optimal
makespans. That repository has no license, so its files are not redistributed here.
`fetch_pcmax.py` downloads the fixed commit `7ce7488` (about 29 MB), converts the
files and checks each one against `pcmax_manifest.sha256`:

```bash
python Implementation/Test_Instances/fetch_pcmax.py
```

The self-generated instances are committed as files; the `generate.py` scripts
document how they were created (they reproduce them bit for bit with the pinned
versions, but NumPy does not guarantee its random streams across versions).
Every dataset directory has a `readme.txt` with the parameters. Known optimal
makespans are in `opt_known.txt` (França: three instances solved with
`franca/solve_opt.py`; Frangioni, Lawrinenko, Berndt, Planted: from Akram et al. or by
construction) and `opt_known_gg.txt` (HM datasets: certified with
`jansen_hm/solve_opt_gg.py`).

## Reproducing Chapter 5

All numbers in Chapter 5 are generated; the text uses the macros from
`Thesis/tables/kennzahlen.tex` and `Thesis/tables/kennzahlen_hm.tex`. The measured
CSVs are committed, so the tables and figures can be rebuilt without measuring again
(the original instances are still needed for one statistic in Section 5.2.7):

```bash
python Implementation/Test_Instances/fetch_pcmax.py
python Implementation/Empirical_Tests/summary_tables.py      # -> Thesis/tables/
python Implementation/Empirical_Tests/thesis_figures.py      # -> Thesis/figures/
python Implementation/Empirical_Tests/summary_tables_hm.py
python Implementation/Empirical_Tests/thesis_figures_hm.py
```

Runtimes depend on the machine, so a new measurement changes the runtime columns;
all other numbers are deterministic.

### Section 5.1: FFD vs. MFFD (9 092 instances, real-valued capacities)

```bash
python Implementation/Empirical_Tests/run_benchmark.py --algorithms ffd mffd --k 10 \
  --instance-dir Implementation/Test_Instances/franca/ \
  --instance-dir Implementation/Test_Instances/frangioni/ \
  --instance-dir Implementation/Test_Instances/lawrinenko/ \
  --instance-dir Implementation/Test_Instances/berndt/ \
  --instance-dir Implementation/Test_Instances/planted/
#   -> Empirical_Tests/<dataset>/benchmark.csv
python Implementation/Empirical_Tests/k_sensitivity.py       # k = 1..10, ~13 min -> k_sensitivity.csv
python Implementation/Empirical_Tests/friesen_case_study.py  # console output for Section 5.1.8
python Implementation/Empirical_Tests/summary_tables.py
python Implementation/Empirical_Tests/thesis_figures.py
```

### Section 5.2: high-multiplicity subroutines (1 080 + 120 instances, integer capacities, Gurobi)

```bash
# certified optimal makespans (binary search over C with the configuration IP)
python Implementation/Test_Instances/jansen_hm/solve_opt_gg.py --workers 4
python Implementation/Test_Instances/jansen_hm/solve_opt_gg.py --workers 4 \
  --dir Implementation/Test_Instances/jansen_hm_scale

# MULTIFIT with ffd, jansen, jansen_dual, gg_lp, gg_ip
#   -> Empirical_Tests/<dataset>/benchmark_hm.csv and capacities_hm.csv
python Implementation/Empirical_Tests/run_hm_benchmark.py --k 10 --workers 4 --time-limit 60 \
  --instance-dir Implementation/Test_Instances/jansen_hm/
python Implementation/Empirical_Tests/run_hm_benchmark.py --k 10 --workers 4 --time-limit 60 \
  --instance-dir Implementation/Test_Instances/jansen_hm_scale/

# all subroutines on a contiguous capacity range (non-monotonicity of J, Lemma 4.4)
#   -> Empirical_Tests/jansen_hm/case_study_hm.csv
python Implementation/Empirical_Tests/run_hm_benchmark.py \
  --instance-dir Implementation/Test_Instances/jansen_hm/ \
  --case-study HM_dense_d5_n0080_m10_7.txt --case-range 320 340

python Implementation/Empirical_Tests/summary_tables_hm.py
python Implementation/Empirical_Tests/thesis_figures_hm.py
```

`run_hm_benchmark.py` also accepts `--pilot N` (one instance per cell of the grid) and
`--resume`.

## Building the thesis

```bash
cd Thesis && latexmk    # -> Thesis/build/main.pdf
```
