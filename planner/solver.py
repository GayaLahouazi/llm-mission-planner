"""Classical planners used as references.

    exact_plan       : depth-first branch and bound -> the shortest valid route (optimal)
    nearest_neighbour: greedy heuristic, ignores time windows and zones
    deadline_first   : sort by time-window closing time, then nearest

`simulate` computes the earliest feasible schedule of a fixed order. Starting
every leg as early as possible is optimal here (waiting never helps later
stops), except in front of an active temporary zone, where we wait until it
closes. It is an independent implementation from the Z3 verifier: the tests
check that both always agree.
"""
from __future__ import annotations

import math

from .scenario import Scenario, dist, segment_hits_circle


def simulate(s: Scenario, order: list[str]):
    """Return (feasible, total_distance, end_time) with earliest departures."""
    wp = s.by_name()
    pos, t, total = s.depot, 0.0, 0.0
    for name in order + ["DEPOT"]:
        nxt = s.point(name)
        travel = dist(pos, nxt) / s.speed
        for _ in range(len(s.zones) + 1):           # repeat: waiting for one zone may hit another
            moved = False
            for z in s.zones:
                if segment_hits_circle(pos, nxt, (z.x, z.y), z.r + s.zone_margin):
                    if z.end == math.inf:
                        return False, math.inf, math.inf
                    if t < z.end and t + travel > z.start:   # overlaps the active period
                        t, moved = z.end, True
            if not moved:
                break
        total += dist(pos, nxt)
        t += travel
        if name != "DEPOT":
            w = wp[name]
            t = max(t, w.earliest)
            if t > w.latest:
                return False, math.inf, math.inf
            t += w.service
        pos = nxt
    ok = t <= s.deadline and total <= s.usable_range + 1e-9
    return ok, total, t


def exact_plan(s: Scenario):
    """Optimal (minimum distance) valid order by depth-first branch and bound."""
    names = [w.name for w in s.waypoints]
    best = {"order": None, "dist": math.inf}

    def dfs(prefix, remaining, partial):
        here = s.point(prefix[-1]) if prefix else s.depot
        if partial + dist(here, s.depot) >= best["dist"]:      # admissible bound
            return
        if not remaining:
            ok, total, _ = simulate(s, prefix)
            if ok and total < best["dist"]:
                best.update(order=list(prefix), dist=total)
            return
        ok, _, _ = simulate_prefix(s, prefix)
        if not ok:
            return
        for n in sorted(remaining, key=lambda n: dist(here, s.point(n))):
            dfs(prefix + [n], remaining - {n}, partial + dist(here, s.point(n)))

    dfs([], set(names), 0.0)
    return best["order"], best["dist"]


def simulate_prefix(s: Scenario, prefix: list[str]):
    """Feasibility of a partial route (windows only; deadline/battery checked at the end)."""
    if not prefix:
        return True, 0.0, 0.0
    tmp = Scenario(s.depot, s.waypoints, s.zones, s.speed, math.inf, 0.0, s.zone_margin, math.inf)
    ok, total, t = simulate(tmp, prefix)
    return ok, total, t


def nearest_neighbour(s: Scenario) -> list[str]:
    pos, left, order = s.depot, {w.name for w in s.waypoints}, []
    while left:
        n = min(sorted(left), key=lambda n: dist(pos, s.point(n)))
        order.append(n)
        left.remove(n)
        pos = s.point(n)
    return order


def deadline_first(s: Scenario) -> list[str]:
    windowed = sorted([w for w in s.waypoints if w.latest != math.inf], key=lambda w: w.latest)
    order = [w.name for w in windowed]
    free = [w.name for w in s.waypoints if w.latest == math.inf]
    pos = s.point(order[-1]) if order else s.depot
    while free:   # then the remaining ones by nearest neighbour
        n = min(free, key=lambda n: dist(pos, s.point(n)))
        order.append(n)
        free.remove(n)
        pos = s.point(n)
    return order


def generate_feasible(seed: int, **kw):
    """Draw random scenarios until one has at least one valid route."""
    from .scenario import generate
    k = 0
    while True:
        s = generate(seed * 1000 + k, **kw)
        order, d = exact_plan(s)
        if order is not None:
            return s, order, d
        k += 1
