from math import ceil, floor

import gurobipy as gb
from gurobipy import GRB

from binpacking.binpacking_algorithms import ffd_nlogn, group_by_size

# Gurobi returns 2.9999999996 instead of 3, so every rounding needs a tolerance.
TOL = 1e-6

# Seconds per model, None = unlimited. Can be set from outside for batch runs; a
# run that hits the limit ends in a RuntimeError instead of a wrong result.
TIME_LIMIT = None


def enumerate_configurations(p: list, u: list, capacity: int,
                             only_maximal: bool = False) -> list:
    # Depth-first search with a capacity cut: generates only feasible vectors
    # instead of spanning the full box and filtering afterwards.
    d = len(p)
    configurations = []
    counts = [0]*d

    def extend(i: int, space: int):
        if i == d:
            # The zero configuration costs a bin and covers nothing.
            if not any(counts):
                return
            # only_maximal: if something still fits, a + e_i dominates this configuration
            if only_maximal and any(counts[j] < u[j] and p[j] <= space for j in range(d)):
                return
            configurations.append(tuple(counts))
            return
        for c in range(min(space // p[i], u[i]), -1, -1):
            counts[i] = c
            extend(i+1, space - c*p[i])
        counts[i] = 0

    extend(0, capacity)
    return configurations


def _fill_bin(p: list, configuration: tuple, remaining: list) -> dict:
    # Takes from remaining what the configuration asks for. The >= form of the
    # program may over-cover; the surplus slots of the configuration then stay empty.
    tasks = []
    for i, count in enumerate(configuration):
        take = min(count, remaining[i])
        remaining[i] -= take
        tasks.extend([p[i]]*take)
    return {"load": sum(tasks), "tasks": tasks}


def gilmore_gomory_bp(p: list, u: list, capacity: int, integer: bool = False,
                      only_maximal: bool = False):
    # STEP 1: enumerate all configurations for the capacity
    configurations = enumerate_configurations(p, u, capacity, only_maximal)

    # STEP 2: solve the configuration LP (integer=True: the corresponding IP,
    # whose optimal value is OPT[Gamma, C]).
    # Method=1 (dual simplex) is required: only a vertex has at most d nonzero
    # entries (Lemma 4.12), and the whole rounding bound (Lemma 4.13) depends
    # on it. An interior point method without crossover ends inside an optimal
    # face and may return arbitrarily many.
    model = gb.Model("gilmore_gomory")
    model.Params.OutputFlag = 0
    model.Params.Method = 1
    if TIME_LIMIT is not None:
        model.Params.TimeLimit = TIME_LIMIT
    model.ModelSense = GRB.MINIMIZE

    x = model.addVars(len(configurations), lb=0.0, obj=1.0,
                      vtype=GRB.INTEGER if integer else GRB.CONTINUOUS, name="x")

    # one demand row per size:  sum_a a_i * x_a >= u_i
    for i in range(len(p)):
        covering = [j for j, a in enumerate(configurations) if a[i]]
        model.addConstr(gb.LinExpr([configurations[j][i] for j in covering],
                                   [x[j] for j in covering]) >= u[i])
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        # Infeasible means capacity < max(p): no configuration contains the
        # largest task. A time limit ends up here as well -- the rounding bound
        # only holds for a true optimal vertex, an aborted solution is useless.
        raise RuntimeError(f"configuration program not solved (status {model.Status})")
    lp_value = model.ObjVal
    solution = [x[j].X for j in range(len(configurations))]

    # STEP 3: round down and pack the residual demand (variant B, Section 4.3.3).
    # Rounding up is never better: it uses one bin per fractional column, and
    # the residual demand needs at most that many.
    floored = [int(floor(v + TOL)) for v in solution]
    fractional = [j for j, v in enumerate(solution) if v - floored[j] > TOL]

    remaining = list(u)
    bins = []
    for j, count in enumerate(floored):
        for _ in range(count):
            bin = _fill_bin(p, configurations[j], remaining)
            if bin["tasks"]:
                bins.append(bin)
    floor_bins = len(bins)

    leftover = [p[i] for i in range(len(p)) for _ in range(remaining[i])]
    leftover_source = "none"
    if leftover:
        # One bin per fractional column always covers the residual demand,
        # because r_i <= sum_{a fractional} a_i; FFD is often more economical, though.
        left = list(remaining)
        by_pattern = [_fill_bin(p, configurations[j], left) for j in fractional]
        by_pattern = [b for b in by_pattern if b["tasks"]]
        assert not any(left), "residual demand not covered by the fractional columns"

        leftover_desc = sorted(leftover, reverse=True)
        by_ffd = [{"load": b["load"], "tasks": [leftover_desc[i] for i in b["tasks"]]}
                  for b in ffd_nlogn(leftover_desc, capacity)]

        if len(by_pattern) <= len(by_ffd):
            bins, leftover_source = bins + by_pattern, "patterns"
        else:
            bins, leftover_source = bins + by_ffd, "ffd"

    # Statistics for the evaluation: lp_bound is the lower bound ceil(LP), against
    # which the rounding loss len(bins) - lp_bound is measured.
    stats = {"lp_value": lp_value,
             "lp_bound": ceil(lp_value - TOL),
             "columns": len(configurations),
             "support": sum(1 for v in solution if v > TOL),
             "fractional": len(fractional),
             "floor_bins": floor_bins,
             "leftover_items": len(leftover),
             "leftover_bins": len(bins) - floor_bins,
             "leftover_source": leftover_source,
             "d": len(p)}
    return bins, stats


def gilmore_gomory_bin_packing(tasks: list, capacity, integer: bool = False,
                               only_maximal: bool = False):
    assert tasks != []
    assert capacity >= max(tasks)

    p, u = group_by_size(tasks)
    return gilmore_gomory_bp(p, u, int(floor(capacity)), integer, only_maximal)


# The two MULTIFIT subroutines. Same signature as ffd_nlogn/mffd_nlogn, but
# multifit runs with int_capacity=True here: for integer sizes, C admits the
# same configurations as floor(C), and the integer search never tests a
# capacity twice (every LP call is expensive).
# The bins hold task sizes in "tasks" instead of indices -- as in the Jansen
# modules and unlike ffd_nlogn/mffd_nlogn.

def gilmore_gomory_lp_bp(tasks: list, capacity) -> list:
    return gilmore_gomory_bin_packing(tasks, capacity)[0]


def gilmore_gomory_ip_bp(tasks: list, capacity) -> list:
    return gilmore_gomory_bin_packing(tasks, capacity, integer=True)[0]
