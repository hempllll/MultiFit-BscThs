import time

from binpacking.binpacking_algorithms import ffd_nlogn, mffd_nlogn
from binpacking.gilmore_gomory_lp import gilmore_gomory_bin_packing
from binpacking.jansen_high_multiplicity_binpacking import jansen_bin_packing_stats
from binpacking.jansen_epsilon_dual_bp import jansen_epsilon_dual_bin_packing_stats

# The subroutines ("arms") handed to MULTIFIT. Each is a callable with the
# signature (tasks, capacity) -> bins that multifit expects, and records one row
# per call in self.trace. Without this trace all statistics would be lost,
# because multifit only evaluates len(bins).
# Some trace keys and status values are German (unzulaessig, eps_theorie,
# last_faktor, angenommen, ...): they are the column schema of capacities_hm.csv,
# from which the thesis tables are generated.


class Arm:
    name = None

    def __init__(self, m):
        self.m = m
        self.trace = []

    def _run(self, tasks, capacity):
        # returns (bins, extra); extra ends up as columns in the trace
        raise NotImplementedError

    def __call__(self, tasks, capacity):
        t0 = time.perf_counter()
        bins, extra = self._run(tasks, capacity)
        t_ms = (time.perf_counter() - t0) * 1000
        row = {"arm": self.name, "capacity": capacity, "bins": len(bins),
               "t_ms": round(t_ms, 4), "accepted": len(bins) <= self.m}
        row.update(extra)
        self.trace.append(row)
        return bins

    def total_ms(self):
        return round(sum(r["t_ms"] for r in self.trace), 4)


class FFDArm(Arm):
    name = "ffd"

    def _run(self, tasks, capacity):
        return ffd_nlogn(tasks, capacity), {}


class MFFDArm(Arm):
    name = "mffd"

    def _run(self, tasks, capacity):
        return mffd_nlogn(tasks, capacity), {}


class JansenArm(Arm):
    name = "jansen"

    def _run(self, tasks, capacity):
        bins, stats = jansen_bin_packing_stats(tasks, int(capacity))
        return bins, {"m_star": stats["m_star"], "milp_steps": stats["milp_steps"],
                      "kB": stats["kB"], "kAll": stats["kAll"],
                      "alpha": stats["alpha"]}


class JansenDualArm(Arm):
    name = "jansen_dual"

    # n_bins = m is known from outside, which saves the binary search for m*:
    # one MILP call per capacity instead of log n.
    def _run(self, tasks, capacity):
        bins, achieved_eps, stats = jansen_epsilon_dual_bin_packing_stats(
            tasks, int(capacity), n_bins=self.m)
        max_load = max((b["load"] for b in bins), default=0)
        return bins, {"m_star": stats["m_star"], "milp_steps": stats["milp_steps"],
                      "kB": stats["kB"], "kAll": stats["kAll"],
                      "alpha": stats["alpha"],
                      "achieved_eps": round(achieved_eps, 6),
                      "eps_theorie": round(stats["eps"], 6),
                      "max_load": max_load,
                      "last_faktor": round(max_load / capacity, 6) if capacity else None,
                      "unzulaessig": stats["unzulaessig"]}


class GGArm(Arm):
    # common base of the LP and the IP arm; integer switches between them
    integer = False

    def _run(self, tasks, capacity):
        bins, stats = gilmore_gomory_bin_packing(tasks, int(capacity),
                                                 integer=self.integer)
        extra = {"lp_value": round(stats["lp_value"], 6),
                 "lp_bound": stats["lp_bound"],
                 "q": stats["columns"],
                 "support": stats["support"],
                 "fractional": stats["fractional"],
                 "floor_bins": stats["floor_bins"],
                 "leftover_items": stats["leftover_items"],
                 "leftover_bins": stats["leftover_bins"],
                 "leftover_source": stats["leftover_source"],
                 "d": stats["d"]}
        extra["accept_status"] = self._accept_status(len(bins), stats["lp_bound"])
        return bins, extra

    def _accept_status(self, gg, lp_bound):
        # The three-valued criterion of Section 4.3.4. For the binary search only
        # accepted/rejected matters -- the difference between the two kinds of
        # rejection is diagnostics: ceil(LP) > m proves OPT[Gamma,C] > m, hence
        # C < C*_max(Gamma); otherwise only the rounding has failed.
        if gg <= self.m:
            return "angenommen"
        if lp_bound > self.m:
            return "zertifiziert_abgelehnt"
        return "unentschieden"


class GGLpArm(GGArm):
    name = "gg_lp"
    integer = False


class GGIpArm(GGArm):
    name = "gg_ip"
    integer = True

    def _accept_status(self, gg, lp_bound):
        # The IP arm returns OPT[Gamma, C] exactly; a rejection is always
        # certified, an undecided case cannot occur.
        return "angenommen" if gg <= self.m else "zertifiziert_abgelehnt"


ARMS = {"ffd": FFDArm, "mffd": MFFDArm, "jansen": JansenArm,
        "jansen_dual": JansenDualArm, "gg_lp": GGLpArm, "gg_ip": GGIpArm}

# These arms need Gurobi; without a license only ffd/mffd can be measured.
SOLVER_ARMS = {"jansen", "jansen_dual", "gg_lp", "gg_ip"}
