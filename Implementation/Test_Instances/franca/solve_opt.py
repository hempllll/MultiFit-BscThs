"""
Exact OPT for single P||Cmax instances with Gurobi (assignment ILP).

Background: for instances without known OPT, makespan/OPT is bounded from above
via the lower bound LB = max(ceil(sum p / m), p_max). For three França instances
this bound is not enough to certify makespan/OPT <= 13/11 -- they are solved
exactly here and added to opt_known.txt.

Usage:
  python Implementation/Test_Instances/franca/solve_opt.py                    # the three open instances
  python Implementation/Test_Instances/franca/solve_opt.py U_1_0010_05_1.txt  # a specific one
"""
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gurobipy as gb
from gurobipy import GRB

from Empirical_Tests.run_benchmark import load_instance

INSTANCE_DIR = Path(__file__).parent
OPT_FILE = INSTANCE_DIR / "opt_known.txt"

# instances for which neither OPT is known nor the lower bound certifies makespan/OPT <= 13/11
DEFAULT_INSTANCES = [
    "U_1_0010_05_1.txt",
    "U_1_0010_05_7.txt",
    "U_1_0010_05_8.txt",
]

TIME_LIMIT = 300


def solve_exact(m: int, tasks: list, time_limit: int = TIME_LIMIT):
    # Minimizes the makespan exactly. Returns (opt, status); opt is None on timeout.
    n = len(tasks)
    model = gb.Model("p_cmax")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    cmax = model.addVar(lb=max(sum(tasks) / m, max(tasks)), ub=sum(tasks), name="cmax")

    # every task on exactly one machine
    for i in range(n):
        model.addConstr(gb.quicksum(x[i, j] for j in range(m)) == 1)

    loads = [gb.quicksum(tasks[i] * x[i, j] for i in range(n)) for j in range(m)]

    for j in range(m):
        model.addConstr(loads[j] <= cmax)

    # symmetry breaking: machine loads in descending order
    for j in range(m - 1):
        model.addConstr(loads[j] >= loads[j + 1])

    model.setObjective(cmax, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return int(round(model.ObjVal)), "OPTIMAL"
    return None, f"STATUS_{model.Status}"


def main():
    names = sys.argv[1:] or DEFAULT_INSTANCES

    existing = {}
    if OPT_FILE.exists():
        for line in OPT_FILE.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                existing[parts[0]] = int(parts[1])

    results = {}
    for name in names:
        path = INSTANCE_DIR / name
        m, n, tasks = load_instance(path)
        lb = max(ceil(sum(tasks) / m), max(tasks))
        opt, status = solve_exact(m, tasks)
        results[name] = opt
        print(f"{name}  m={m} n={n}  LB={lb}  OPT={opt}  ({status})")

    solved = {k: v for k, v in results.items() if v is not None}
    if not solved:
        print("\nNo instance solved exactly -- opt_known.txt unchanged.")
        return

    existing.update(solved)
    OPT_FILE.write_text("".join(f"{k} {v}\n" for k, v in sorted(existing.items())))
    print(f"\n{len(solved)} value(s) written -> {OPT_FILE} ({len(existing)} entries in total)")


if __name__ == "__main__":
    main()
