from math import floor

import gurobipy as gb
from gurobipy import GRB

from binpacking.binpacking_algorithms import group_by_size
# The same enumeration as in the OPT+1 module, not just the same set: otherwise
# the runtime comparison of the two Jansen variants would measure the difference
# between depth-first search and product(...)+filter instead of the one between
# the algorithms. For d = 6 that is 28.7 ms instead of 0.6 ms per call.
from binpacking.jansen_high_multiplicity_binpacking import enumerate_configurations

# Seconds per MILP, None = unlimited. Can be set from outside for batch runs.
TIME_LIMIT = None


def _jansen_eps_dual_impl(p: list, u: list, capacity: int, n_bins: int = None):
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

    # STEP 2: find m* (binary search, or a single MILP if n_bins is given).
    # c_B: configurations of the big items, c_all: all configurations.
    c_B = enumerate_configurations(p_B, u_sorted[:alpha], capacity)
    c_all = enumerate_configurations(p_sorted, u_sorted, capacity)

    # group c_all once by big-item part, reused across all _build_and_solve calls
    big_config_groups = {}
    for i, c in enumerate(c_all):
        big_config_groups.setdefault(c[:alpha], []).append(i)

    def _build_and_solve(m_candidate):
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
        model.addConstr(gb.quicksum(y[ell] for ell in range(len(c_B))) <= m_candidate)

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

        # A time limit must not count as "infeasible".
        if (model.Status == GRB.OPTIMAL
                or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0)):
            return {"m": m_candidate,
                    "y": [y[ell].X for ell in range(len(c_B))],
                    "x": [x[i].X for i in range(len(c_all))]}
        if model.Status != GRB.INFEASIBLE:
            raise RuntimeError(f"MILP undecided (status {model.Status})")
        return None

    if n_bins is not None:
        # MULTIFIT mode: n_bins (= m machines) is known from outside.
        # A single MILP call suffices -- the binary search for m* is unnecessary,
        # because MULTIFIT only needs feasibility with m bins, not the minimum.
        # The eps-dual guarantee still holds: MILP feasibility gives
        # sum_i p_i*u_i <= n_bins*capacity, and the same contradiction argument applies.
        milp_steps = 1
        best = _build_and_solve(n_bins)
        if best is None:
            # Infeasible: capacity too small for n_bins bins -> signal to MULTIFIT
            return ([{"load": 0, "tasks": []} for _ in range(n_bins + 1)], 0.0,
                    {"m_star": None, "milp_steps": 1, "kB": len(c_B),
                     "kAll": len(c_all), "alpha": alpha, "eps": eps,
                     "unzulaessig": True})
    else:
        m_lo = 1
        m_hi = sum(u)
        best = None
        milp_steps = 0
        while m_lo <= m_hi:
            m = (m_lo + m_hi) // 2
            milp_steps += 1
            result = _build_and_solve(m)
            if result is not None:
                best = result
                m_hi = m - 1
            else:
                m_lo = m + 1
        if best is None:
            raise RuntimeError(f"no feasible m* for capacity={capacity}")

    # STEP 3: floor the x variables, assign each big config their bins,
    # place the small items inside their respective big item bins.
    # eps-dual variant: instead of opening an extra bin for the leftover
    # small items, distribute them over the existing bins, allowing each bin
    # to reach load (1+eps)*capacity (Lemma 4.8).
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

    # eps-dual distribution: place every leftover into an existing bin (no new bin).
    # Lemma 4.8 guarantees, without any condition on m*, that a bin with load
    # < capacity exists: if all bins had load >= capacity, the total load would
    # be >= m*·capacity, contradicting MILP feasibility
    # (sum_i p_i u_i <= capacity·sum_b y_b). Inserting an item < eps·capacity
    # then gives load < (1+eps)·capacity.
    # The argument needs sum(y) == m*. In MULTIFIT mode (n_bins = m), m is not
    # minimal, so the MILP may get by with fewer bins -- then the fallback
    # below applies. Measured: 22 of 5595 feasible calls, all with
    # achieved_eps = 0.
    leftover.sort(reverse=True)
    for item in leftover:
        candidate = min(
            (b for b in bins if b["load"] < capacity),
            key=lambda b: b["load"],
            default=None,
        )
        if candidate is None:
            # sum(y) < n_bins: no bin with load < capacity; achieved_eps may
            # exceed eps. Shows up in the trace as last_faktor.
            candidate = min(bins, key=lambda b: b["load"])
        candidate["tasks"].append(item)
        candidate["load"] += item

    # achieved_eps: actual worst-case overflow ratio max(load/capacity - 1, 0).
    # At most eps by Lemma 4.8, as long as sum(y) == m*.
    achieved_eps = max(0.0, max((b["load"] / capacity - 1 for b in bins), default=0.0))
    return bins, achieved_eps, {"m_star": best["m"], "milp_steps": milp_steps,
                                "kB": len(c_B), "kAll": len(c_all),
                                "alpha": alpha, "eps": eps, "unzulaessig": False}


def jansen_epsilon_dual_bp(p: list, u: list, capacity: int, n_bins: int = None):
    bins, achieved_eps, _ = _jansen_eps_dual_impl(p, u, capacity, n_bins=n_bins)
    return bins, achieved_eps


def jansen_epsilon_dual_bp_stats(p: list, u: list, capacity: int, n_bins: int = None):
    # as above, but also returns m*, MILP steps, configuration counts and eps
    return _jansen_eps_dual_impl(p, u, capacity, n_bins=n_bins)


def jansen_epsilon_dual_bin_packing(tasks: list, capacity: int, n_bins: int = None):
    p, u = group_by_size(tasks)
    bins, achieved_eps, _ = _jansen_eps_dual_impl(p, u, capacity, n_bins=n_bins)
    return bins, achieved_eps


def jansen_epsilon_dual_bin_packing_stats(tasks: list, capacity: int, n_bins: int = None):
    # as jansen_epsilon_dual_bin_packing, but returns (bins, achieved_eps, stats)
    p, u = group_by_size(tasks)
    return _jansen_eps_dual_impl(p, u, capacity, n_bins=n_bins)
