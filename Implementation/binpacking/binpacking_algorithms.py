import math
from sortedcontainers import SortedList


# Segment tree where each leaf stores remaining capacity of one bin,
# and each internal node stores the max of its subtree.
# This lets search_leftmost find the leftmost bin that fits a task in O(log n).
class SegmentTree():
    def __init__(self, n: int, initial_value: float = 0):
        # Round up to the next power of two so the tree is always complete.
        # Leaves occupy indices [n, 2n), internal nodes [1, n).
        # Note: math.ceil(math.log2(1)) == 0, so n=1 works correctly (tree size 2, root = leaf at index 1).
        self.n = 2**(math.ceil(math.log2(n)))
        self.st = [initial_value]*(2*self.n)

    def update(self, i: int, value: float):
        i += self.n
        self.st[i] = value

        # Propagate the new max up to the root.
        while i > 1:
            neighbor = i-1 if i % 2 == 1 else i+1
            parent = i // 2

            self.st[parent] = max(self.st[neighbor], self.st[i])
            i = parent

    def search_leftmost(self, value):
        # Root's value is the global max remaining capacity; early-exit if no bin fits.
        if value > self.st[1]:
            return -1
        i = 1
        # Greedily descend left whenever the left subtree contains a fitting bin.
        while i < self.n:
            child_left = 2*i
            child_right = 2*i + 1

            i = child_left if self.st[child_left] >= value else child_right

        return i - self.n

    def get_list(self):
        return self.st[self.n:]

    def __getitem__(self, i):
        return self.st[self.n + i]


def group_by_size(tasks: list) -> tuple:
    # High-multiplicity encoding: distinct sizes p (descending) and their multiplicities u.
    tasks_sorted = sorted(tasks, reverse=True)
    p = [tasks_sorted[0]]
    u = [1]
    for t in tasks_sorted[1:]:
        if p[-1] == t:
            u[-1] += 1
        else:
            p.append(t)
            u.append(1)
    return p, u


def first_fit(tasks: list, capacity: float) -> list:
    bins = [{"load": 0, "tasks": []}]
    assert tasks != []
    assert capacity >= max(tasks)

    for task_index, task in enumerate(tasks):
        i = 0
        fitting_bin_found = False
        while i < len(bins) and not fitting_bin_found:
            if bins[i]["load"] + task <= capacity:
                bins[i]["tasks"].append(task_index)
                bins[i]["load"] = bins[i]["load"] + task
                fitting_bin_found = True
            i += 1
        if not fitting_bin_found:
            bins.append({"load": task, "tasks": [task_index]})

    return bins


def first_fit_nlogn(tasks: list, capacity: float) -> list:
    # Pre-allocate one bin per task (worst case: each task gets its own bin).
    # The segment tree tracks remaining capacity; bin_counter tracks how many are in use.
    bins_st = SegmentTree(n=len(tasks), initial_value=capacity)

    bins = [{"load": 0, "tasks": []} for _ in range(len(tasks))]

    assert tasks != []
    assert capacity >= max(tasks)
    bin_counter = 0
    for task_index, task in enumerate(tasks):
        bin = bins_st.search_leftmost(task)
        bins_st.update(bin, bins_st[bin] - task)
        bins[bin]["load"] += task
        bins[bin]["tasks"].append(task_index)
        bin_counter = max(bin+1, bin_counter)
    return bins[:bin_counter]


def ffd(tasks: list, capacity: float) -> list:
    tasks_sorted = sorted(tasks, reverse=True)
    bins = first_fit(tasks_sorted, capacity)

    return bins


def ffd_nlogn(tasks: list, capacity: float) -> list:
    tasks_sorted = sorted(tasks, reverse=True)
    bins = first_fit_nlogn(tasks_sorted, capacity)

    return bins


def mffd_nlogn(tasks: list, capacity: float) -> list:
    tasks_sorted = sorted(tasks, reverse=True)
    # Assign each item a class relative to the capacity.
    tasks_sorted_in_classes = {"large": [],
                               "medium": [],
                               "small": [],
                               "tiny": []}

    for i in range(len(tasks_sorted)):
        if tasks_sorted[i] > capacity / 2:
            item_class = "large"
        elif tasks_sorted[i] > capacity / 3:
            item_class = "medium"
        elif tasks_sorted[i] > capacity / 6:
            item_class = "small"
        else:
            item_class = "tiny"

        tasks_sorted_in_classes[item_class].append((tasks_sorted[i], i))

    # Phase 1: give each large item its own bin, in descending order.
    bins = []
    for large_task in tasks_sorted_in_classes["large"]:
        bins.append({"load": large_task[0], "tasks": [large_task[1]]})

    # Phase 2: scan bins left to right. For each bin, try to place the largest medium item that still fits alongside the
    # large item already there. Each bin gets at most one medium item.
    bins_that_didnt_receive_medium_item = []
    medium_items_asc = SortedList(tasks_sorted_in_classes["medium"])
    for bin_index, bin in enumerate(bins):
        # largest medium item that still fits into this bin
        space = capacity - bin["load"]
        largest_medium_fitting_idx = medium_items_asc.bisect_right(
            (space, float('inf'))) - 1

        # no medium item fits
        if largest_medium_fitting_idx == -1:
            bins_that_didnt_receive_medium_item.append(bin_index)
        else:
            bin["load"] += medium_items_asc[largest_medium_fitting_idx][0]
            bin["tasks"].append(
                medium_items_asc[largest_medium_fitting_idx][1])
            medium_items_asc.pop(largest_medium_fitting_idx)

    # Phase 3: scan bins right to left, but only bins that did not receive a medium item in Phase 2.
    # Johnson & Garey (1985): skip the bin unless the TWO smallest remaining small items fit into it
    # together — the phase exists to place *pairs* of small items alongside a large one. Only then:
    # place the smallest remaining small item, then the largest remaining small item that still fits.
    small_items_asc = SortedList(tasks_sorted_in_classes["small"])
    for bin_index in reversed(bins_that_didnt_receive_medium_item):
        if len(small_items_asc) < 2:
            continue
        if bins[bin_index]["load"] + small_items_asc[0][0] + small_items_asc[1][0] > capacity:
            continue
        bins[bin_index]["load"] += small_items_asc[0][0]
        bins[bin_index]["tasks"].append(small_items_asc[0][1])
        small_items_asc.pop(0)
        # second item: largest remaining small item that still fits
        space = capacity - bins[bin_index]["load"]
        largest_small_fitting_idx = small_items_asc.bisect_right(
            (space, float('inf'))) - 1
        if largest_small_fitting_idx != -1:
            bins[bin_index]["load"] += small_items_asc[largest_small_fitting_idx][0]
            bins[bin_index]["tasks"].append(
                small_items_asc[largest_small_fitting_idx][1])
            small_items_asc.pop(largest_small_fitting_idx)

    # Phase 4: top-up pass. Make another forward pass over all bins.
    # For each bin, greedily add the largest remaining item (of any class)
    # that still fits, one at a time, until nothing more fits in that bin.
    remaining_items = SortedList(tasks_sorted_in_classes["tiny"])
    remaining_items.update(small_items_asc)
    remaining_items.update(medium_items_asc)
    for bin in bins:
        no_item_fits = False
        while not no_item_fits:
            space = capacity - bin["load"]
            largest_item_fitting_idx = remaining_items.bisect_right(
                (space, float('inf'))) - 1
            # is there a remaining item that fits?
            if largest_item_fitting_idx != -1:
                bin["load"] += remaining_items[largest_item_fitting_idx][0]
                bin["tasks"].append(
                    remaining_items[largest_item_fitting_idx][1])
                remaining_items.pop(largest_item_fitting_idx)
            else:
                no_item_fits = True

    # Phase 5: FFD on the leftovers.
    if remaining_items:
        # Sort descending as in ffd_nlogn, but keep the (size, index) pairs:
        # first_fit_nlogn numbers its input list by position, so the returned
        # indices can be looked up directly in leftovers_desc.
        leftovers_desc = sorted(remaining_items, key=lambda item: item[0], reverse=True)
        leftover_bins = first_fit_nlogn([value for value, _ in leftovers_desc], capacity)
        for b in leftover_bins:
            b["tasks"] = [leftovers_desc[idx][1] for idx in b["tasks"]]
        bins = bins + leftover_bins
    return bins
