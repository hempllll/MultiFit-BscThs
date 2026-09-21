from math import floor
import gurobipy as gb
from gurobipy import GRB

from binpacking.binpacking_algorithms import group_by_size

# Seconds per MILP, None = unlimited. Can be set from outside for batch runs.
TIME_LIMIT = None


def enumerate_configurations(p: list, u: list, capacity: int) -> list:
    # The maximum count for each size p_i: the bin holding only copies of p_i
    # stays within the capacity, and no bin can use more copies than exist in total.
    # The cap at u_i is not merely a speed-up, it tightens the relaxation:
    # without it, p = [10], u = [2], capacity = 30 admits the column (3), which
    # corresponds to no packing -- the MILP uses it with x = 2/3 and reaches
    # 0.667 instead of the exact value 1. The cap can therefore change the LP
    # value; the acceptance decision stays the same, because a real packing
    # never puts more than u_i copies of a size into one bin and thus remains
    # representable.
    max_counts = [min(capacity // p_i, u_i) for p_i, u_i in zip(p, u)]

    # Depth-first search with a capacity cut instead of spanning the full box
    # and filtering afterwards. The order is identical to the former
    # product(*[range(m + 1) for m in max_counts]) -- counts ascending, last
    # index fastest --, so the MILP columns keep their indices; only the
    # infeasible part of the box is never built. That part dominates: for
    # d = 6, m = 20 the box has 16.5 million entries, of which a few hundred
    # thousand are feasible.
    d = len(p)
    configurations = []

    def extend(i, space, prefix):
        if i == d:
            configurations.append(tuple(prefix))
            return
        p_i = p[i]
        for c in range(max_counts[i] + 1):
            rest = space - c * p_i
            if rest < 0:
                break          # any larger c exceeds the capacity as well
            prefix.append(c)
            extend(i + 1, rest, prefix)
            prefix.pop()

    extend(0, capacity, [])
    return configurations


def _jansen_hm_bp_impl(p: list, u: list, capacity: int):
    # STEP 1: Set eps = 1/(2d-1) and partition the sizes into big (at least
    # eps*capacity) and small (smaller than eps*capacity). Set m* = n.
    d = len(p)
    eps = 1 / (2 * d - 1)

    # sort descending by size so big items come first
    paired = sorted(zip(p, u), reverse=True)
    p_sorted = [size for size, _ in paired]
    u_sorted = [mult for _, mult in paired]

    # alpha: number of big sizes, i.e. the first index that is no longer big
    alpha = sum(1 for pi in p_sorted if pi >= capacity * eps)

    p_B = p_sorted[:alpha]

    # STEP 2: binary search for the smallest m* for which the MILP is feasible.
    # c_B: configurations of the big items, c_all: all configurations.
    c_B = enumerate_configurations(p_B, u_sorted[:alpha], capacity)
    c_all = enumerate_configurations(p_sorted, u_sorted, capacity)

    # group c_all once by big-item part, reused across all binary-search iterations
    big_config_groups = {}
    for i, c in enumerate(c_all):
        big_config_groups.setdefault(c[:alpha], []).append(i)

    m_lo = 1
    m_hi = sum(u)
    best = None
    milp_steps = 0
    while m_lo <= m_hi:
        milp_steps += 1
        m = (m_lo + m_hi) // 2
        model = gb.Model("milp")
        model.Params.OutputFlag = 0  # silence output
        if TIME_LIMIT is not None:
            model.Params.TimeLimit = TIME_LIMIT

        # variables y[ell] (big configurations) and x[i] (all configurations)
        y = model.addVars(len(c_B),    lb=0, vtype=GRB.INTEGER,    name="y")
        x = model.addVars(len(c_all), lb=0, vtype=GRB.CONTINUOUS, name="x")

        # constraint one: the configurations whose big part is c_B[ell]
        #   are used at most y[ell] times in total
        for ell, big_config in enumerate(c_B):
            matching = big_config_groups.get(big_config, [])
            model.addConstr(gb.quicksum(x[i] for i in matching) <= y[ell])

        # constraint two: total bins <= m  (inequality so minimisation has room)
        model.addConstr(gb.quicksum(y[ell] for ell in range(len(c_B))) <= m)

        # constraint three: every big item type j must be covered >= u_sorted[j] times
        for j in range(alpha):
            model.addConstr(gb.quicksum(y[ell] * big_config[j]
                            for ell, big_config in enumerate(c_B)) >= u_sorted[j])

        # constraint four: every small item type j must be covered >= u_sorted[j] times
        for j in range(alpha, d):
            model.addConstr(gb.quicksum(x[i] * c_all[i][j]
                            for i in range(len(c_all))) >= u_sorted[j])

        model.setObjective(0)
        model.optimize()

        # A time limit must not count as "infeasible": that would push the
        # binary search upwards and return an m* that is too large.
        feasible = (model.Status == GRB.OPTIMAL
                    or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0))
        if feasible:
            best = {"m": m,
                    "y": [y[ell].X for ell in range(len(c_B))],
                    "x": [x[i].X for i in range(len(c_all))]}
            m_hi = m - 1
        elif model.Status == GRB.INFEASIBLE:
            m_lo = m + 1
        else:
            raise RuntimeError(f"MILP undecided (status {model.Status})")

    if best is None:
        # No m in [1, n] was feasible, i.e. capacity < max(p): no configuration
        # holds the largest task.
        raise RuntimeError(f"no feasible m* for capacity={capacity}")

    # STEP 3: floor the x variables, assign each big config their bins,
    # place the small items inside their respective big item bins.
    # place the remaining small items (lost due to rounding) into one remaining
    #   bin.
    # bin_pool: one dictionary linking each big_config with their respective
    #   unfilled bin indices.

    # flooring the x variables
    # and converting the y values to int (gurobi returns float)
    best["x"] = [int(floor(x)) for x in best["x"]]
    best["y"] = [int(round(y)) for y in best["y"]]

    # assign each big config c_B[ell] y[ell] bins
    # clamp to actual multiplicities: MILP uses >= so may over-cover
    big_remaining = list(u_sorted[:alpha])
    bin_pool = {}
    bins = []
    for ell, big_config in enumerate(c_B):
        bin_pool[big_config] = []
        for _ in range(best["y"][ell]):
            tasks = []
            for j, cnt in enumerate(big_config):
                actual = min(cnt, big_remaining[j])
                big_remaining[j] -= actual
                for _ in range(actual):
                    tasks.append(p_sorted[j])
            bins.append({"load": sum(tasks), "tasks": tasks})
            bin_pool[big_config].append(len(bins) - 1)

    # placing the small items into their respective big item bins
    remaining = list(u_sorted[alpha:])

    for i, count in enumerate(best["x"]):
        big_part = c_all[i][:alpha]
        small_part = c_all[i][alpha:]
        for _ in range(count):
            bin_idx = bin_pool[big_part].pop()
            for j_local, c in enumerate(small_part):
                for _ in range(c):
                    if remaining[j_local] > 0:   # clamp: MILP may over-cover
                        bins[bin_idx]["tasks"].append(
                            p_sorted[alpha + j_local])
                        bins[bin_idx]["load"] += p_sorted[alpha + j_local]
                        remaining[j_local] -= 1

    # Collect leftover small items (not covered by floor(x) configs)
    leftover = []
    for j_local, count in enumerate(remaining):
        for _ in range(count):
            leftover.append(p_sorted[alpha + j_local])

    # Greedy First Fit: pack leftover into existing bins' remaining capacity.
    # Jansen's analysis guarantees that the remaining capacity plus one extra bin suffices.
    leftover.sort(reverse=True)
    extra = []
    for item in leftover:
        placed = False
        for b in bins:
            if b["load"] + item <= capacity:
                b["tasks"].append(item)
                b["load"] += item
                placed = True
                break
        if not placed:
            extra.append(item)

    if extra:
        bins.append({"load": sum(extra), "tasks": extra})

    return bins, {"m_star": best["m"], "milp_steps": milp_steps,
                  "kB": len(c_B), "kAll": len(c_all), "alpha": alpha}


def jansen_high_multiplicity_bp(p: list, u: list, capacity: int):
    return _jansen_hm_bp_impl(p, u, capacity)[0]


def jansen_high_multiplicity_bp_stats(p: list, u: list, capacity: int):
    # as above, but also returns m*, the number of MILP steps and the sizes of
    # both configuration sets
    return _jansen_hm_bp_impl(p, u, capacity)


def jansen_bin_packing(tasks: list, capacity: int):
    p, u = group_by_size(tasks)
    return _jansen_hm_bp_impl(p, u, capacity)[0]


def jansen_bin_packing_stats(tasks: list, capacity: int):
    # as jansen_bin_packing, but returns (bins, stats)
    p, u = group_by_size(tasks)
    return _jansen_hm_bp_impl(p, u, capacity)
