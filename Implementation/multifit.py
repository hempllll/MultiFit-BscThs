from typing import Callable


def multifit(tasks: list, m: int, k: int, bp_subroutine: Callable,
             int_capacity: bool = False) -> list:
    tasks_sum = sum(tasks)
    # c_low: lower bound on the optimal makespan (average load, largest task; Lemma 2.2).
    # c_high: upper bound at which FFD needs at most m bins (Lemma 2.6(b)),
    # so the search starts from a known feasible partition.
    c_low = max(tasks_sum/m, max(tasks))
    c_high = tasks_sum/m + max(tasks)

    if int_capacity:
        c_low  = int(c_low)
        # Rounded down, hence strictly below the bound of Lemma 2.6(b). For
        # integer p_j it still holds: int(sum p/m + p_max) =
        # floor(sum p/m) + p_max, and if FFD opened an (m+1)-th bin there, each
        # of the m previous bins would have integer load >= floor(sum p/m)+1
        # > sum p/m, so the total load would exceed sum p -- contradiction.
        c_high = int(tasks_sum/m + max(tasks))

    cur_best_partition = bp_subroutine(tasks, c_high)
    # The result is an assignment to m machines only if the initial call
    # already yields <= m bins. For FFD/MFFD this is Lemma 2.6(b); for
    # subroutines without that guarantee (e.g. J = OPT+1) it is unproven, but
    # it never fails on any of the evaluated datasets.
    assert len(cur_best_partition) <= m, (
        f"initial call at C={c_high} yields {len(cur_best_partition)} > {m} bins")

    # Binary search over capacity: shrink the interval k times.
    # cur_best_partition always holds the result for the current c_high (the last capacity that packed into ≤ m bins).
    for _ in range(k):
        candidate_c = (c_low + c_high) / 2
        if int_capacity:
            candidate_c = int(candidate_c)
            if c_low >= c_high:   # integer interval collapsed — done
                break
        partition_bp = bp_subroutine(tasks, candidate_c)
        if len(partition_bp) <= m:
            c_high = candidate_c
            cur_best_partition = partition_bp
        else:
            c_low = candidate_c + 1 if int_capacity else candidate_c

    return cur_best_partition
