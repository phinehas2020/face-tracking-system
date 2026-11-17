"""Attendance as a range, not a single number.

The physical entry count is observed directly. The number of distinct people is
not: some entries never yield a usable face. Two statements are always true and
are reported as hard bounds:

* at least as many people entered as there are distinct resolved people;
* at most that many plus one new person per unresolved entry.

Between the bounds, an estimate is possible under one explicit assumption: each
entry is resolved independently with the same probability r, whoever the person
is. If person i truly entered k_i times and m_i of those were resolved, then
E[C(m_i, j)] = r^j C(k_i, j), and since sum_{j>=1} (-1)^(j+1) C(k, j) = 1 for
every k >= 1, the number of people is

    N = E - S_2 / r^2 + S_3 / r^3 - ...,   S_j = sum_i C(m_i, j),

where E is the entry count. The assumption is false whenever the same people are
systematically unreadable (hats, children below the camera, people who look
down), so the estimate is labelled as conditional and the bounds are not.
"""

from __future__ import annotations

from math import comb

import numpy as np

MIN_RESOLUTION_RATE = 0.2
MIN_PEOPLE = 5


def _thinned(entries: int, resolved: np.ndarray, rate: float, order: int) -> float:
    estimate = float(entries)
    for j in range(2, order + 1):
        pairs = sum(comb(int(m), j) for m in resolved)
        estimate += (-1) ** (j + 1) * pairs / rate**j
    return estimate


def estimate_attendance(
    entries: int, resolved_entries_per_person: list[int], *, replicates: int = 500, seed: int = 0
) -> dict:
    resolved = np.asarray([m for m in resolved_entries_per_person if m > 0], dtype=np.int64)
    linked = int(resolved.sum())
    people = len(resolved)
    if linked > entries:
        raise ValueError("More resolved entries than entries")
    unresolved = entries - linked
    lower = people if (people or not entries) else 1
    result = {
        "entries": entries,
        "linked_entries": linked,
        "resolution_rate": round(linked / entries, 4) if entries else None,
        "lower_bound": lower,
        "upper_bound": people + unresolved,
        "estimate": None,
        "interval_90": None,
        "method": None,
        "note": None,
    }
    if not unresolved:
        result.update(estimate=people, interval_90=[people, people], method="exact")
        return result
    rate = linked / entries
    if rate < MIN_RESOLUTION_RATE or people < MIN_PEOPLE:
        result["note"] = (
            "Too few entries have a face match to estimate between the bounds; "
            "only the bounds are reported."
        )
        return result
    # The third-order term is only worth its variance when most entries resolve.
    order = 3 if rate >= 0.5 and resolved.max() >= 3 else 2
    low, high = result["lower_bound"], result["upper_bound"]
    point = min(max(_thinned(entries, resolved, rate, order), low), high)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sample = rng.choice(resolved, size=people, replace=True)
        # Posterior for the resolution rate under a uniform prior.
        sampled_rate = max(rng.beta(linked + 1, unresolved + 1), 1e-3)
        draws.append(min(max(_thinned(entries, sample, sampled_rate, order), low), high))
    result.update(
        estimate=round(point),
        interval_90=[round(float(np.percentile(draws, 5))), round(float(np.percentile(draws, 95)))],
        method=f"binomial thinning, order {order}",
        note="Assumes every entry is equally likely to yield a face match. "
        "The bounds do not depend on that assumption.",
    )
    return result
