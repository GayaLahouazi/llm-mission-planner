"""Formal verification of a plan with the Z3 SMT solver.

A plan is an ordered list of waypoint names. The verifier asks Z3:
    "does there exist a schedule (departure / waiting times) such that
     every constraint holds for this visiting order?"
Each constraint is *tracked* with a label. If the answer is UNSAT, Z3 returns
an (approximately minimal) unsat core: the labels of constraints that cannot
hold together. They are turned into short sentences that are sent back to
the LLM so that it can repair its plan.

Note on the zone constraint: a temporary no-fly zone gives a disjunction
(fly the whole leg before it activates OR take off after it closes), which is
exactly where an SMT solver is more convenient than hand-written checks.

Waiting is allowed in two places, as in the procedures: on site before an
inspection, until the time window opens (DRN-02), and after an inspection,
before taking off, in front of an active zone (DRN-04). Service start and
take-off times are therefore separate variables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import z3

from .scenario import Scenario, dist, segment_hits_circle


@dataclass
class Verdict:
    valid: bool
    problems: list[str] = field(default_factory=list)
    schedule: dict[str, float] = field(default_factory=dict)   # service start time per waypoint
    distance: float = 0.0
    end_time: float | None = None

    def feedback(self) -> str:
        return "Plan is valid." if self.valid else "Plan rejected by the verifier:\n" + \
            "\n".join(f"- {p}" for p in self.problems)


def verify(s: Scenario, order: list[str]) -> Verdict:
    names = [w.name for w in s.waypoints]
    wp = s.by_name()
    problems = []

    # ---- structural checks (no solver needed) -------------------------------------
    unknown = [n for n in order if n not in wp]
    missing = [n for n in names if n not in order]
    dupes = sorted({n for n in order if order.count(n) > 1})
    if unknown:
        problems.append(f"unknown waypoints {unknown}: only {names} exist")
    if missing:
        problems.append(f"waypoints never visited: {missing} (every waypoint must be inspected)")
    if dupes:
        problems.append(f"waypoints visited more than once: {dupes}")
    if problems:
        return Verdict(False, problems)

    stops = ["DEPOT"] + order + ["DEPOT"]
    pts = [s.point(n) for n in stops]
    legs = [dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    total = sum(legs)

    # ---- SMT encoding ------------------------------------------------------------------
    solver = z3.Solver()
    solver.set("core.minimize", True)
    start = [z3.Real(f"start_{n}_{k}") for k, n in enumerate(stops)]      # service start (depot: mission start)
    depart = [z3.Real(f"depart_{n}_{k}") for k, n in enumerate(stops[:-1])]   # take-off from each stop
    explain: dict[str, str] = {}

    def track(label: str, constraint, message: str):
        explain[label] = message
        solver.assert_and_track(constraint, z3.Bool(label))

    solver.add(start[0] >= 0)
    for k in range(len(stops) - 1):
        a, b = stops[k], stops[k + 1]
        service = 0.0 if a == "DEPOT" else wp[a].service
        travel = legs[k] / s.speed
        # take off after the inspection (hovering allowed), start service at b after arriving (waiting allowed)
        solver.add(depart[k] >= start[k] + service, start[k + 1] >= depart[k] + travel)
        for z in s.zones:
            if not segment_hits_circle(pts[k], pts[k + 1], (z.x, z.y), z.r + s.zone_margin):
                continue
            label = f"zone_{z.name}_{a}_{b}"
            if z.end == math.inf:
                track(label, z3.BoolVal(False),
                      f"leg {a} -> {b} crosses no-fly zone {z.name} (permanent, incl. {s.zone_margin} km margin)")
            else:
                track(label, z3.Or(depart[k] + travel <= z.start, depart[k] >= z.end),
                      f"leg {a} -> {b} crosses no-fly zone {z.name}, which is active during "
                      f"[{z.start:g}, {z.end:g}) min: fly it entirely outside that period")
    for k, n in enumerate(stops[1:-1], start=1):
        w = wp[n]
        track(f"early_{n}", start[k] >= w.earliest, f"{n} cannot be inspected before t = {w.earliest:g}")
        if w.latest != math.inf:
            track(f"window_{n}", start[k] <= w.latest,
                  f"{n} must be inspected by t = {w.latest:g} min (time window [{w.earliest:g}, {w.latest:g}])")
    track("deadline", start[-1] <= s.deadline, f"the drone must be back at the depot by t = {s.deadline:g} min")
    track("battery", z3.BoolVal(total <= s.usable_range + 1e-9),
          f"route length {total:.1f} km exceeds the usable range {s.usable_range:.1f} km "
          f"({s.battery_range:g} km minus {s.reserve:.0%} reserve)")

    if solver.check() == z3.sat:
        m = solver.model()
        val = lambda v: float(m.eval(v, model_completion=True).as_fraction())
        sched = {n: round(val(start[k]), 2) for k, n in enumerate(stops[1:-1], start=1)}
        return Verdict(True, [], sched, total, round(val(start[-1]), 2))
    core = [str(c) for c in solver.unsat_core()]
    # waypoint-early constraints are never the "cause" alone; keep the informative ones first
    core.sort(key=lambda c: c.startswith("early_"))
    return Verdict(False, [explain[c] for c in core], {}, total)
