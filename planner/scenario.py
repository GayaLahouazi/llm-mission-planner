"""Drone inspection missions: data model, random generator and text brief.

A drone leaves the depot, must inspect every waypoint once and come back.
Constraints:
    - time windows      : some waypoints can only be inspected in [earliest, latest] (minutes)
    - battery           : total distance <= usable range (range minus the reserve, see docs/)
    - no-fly zones      : circles that must not be crossed while they are active
    - mission deadline  : back at the depot before `deadline`
Units: kilometres, minutes, km/min.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    earliest: float = 0.0
    latest: float = math.inf
    service: float = 2.0          # inspection time on site (min)


@dataclass
class NoFlyZone:
    name: str
    x: float
    y: float
    r: float
    start: float = 0.0            # active during [start, end)
    end: float = math.inf


@dataclass
class Scenario:
    depot: tuple[float, float]
    waypoints: list[Waypoint]
    zones: list[NoFlyZone] = field(default_factory=list)
    speed: float = 0.8            # km/min (48 km/h)
    battery_range: float = 60.0   # km on a full battery
    reserve: float = 0.15         # fraction kept for emergencies (procedure DRN-03)
    zone_margin: float = 0.3      # lateral safety margin around no-fly zones, km (procedure DRN-05)
    deadline: float = 120.0       # minutes

    @property
    def usable_range(self) -> float:
        return self.battery_range * (1 - self.reserve)

    def point(self, name: str) -> tuple[float, float]:
        if name == "DEPOT":
            return self.depot
        w = self.by_name()[name]
        return (w.x, w.y)

    def by_name(self) -> dict[str, Waypoint]:
        return {w.name: w for w in self.waypoints}

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda v: None if v == math.inf else v)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_hits_circle(a, b, c, r) -> bool:
    """True if segment [a, b] passes at distance < r from centre c."""
    ax, ay = a; bx, by = b; cx, cy = c
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy) < r


def generate(seed: int, n_waypoints: int = 6, n_zones: int = 2, size: float = 20.0) -> Scenario:
    """Random mission. Feasibility is NOT guaranteed here (see solver.generate_feasible)."""
    rng = np.random.default_rng(seed)
    depot = (size / 2, size / 2)
    zones = []
    for i in range(n_zones):
        cx, cy = rng.uniform(3, size - 3, size=2)
        if i % 2 == 0:
            zones.append(NoFlyZone(f"Z{i + 1}", round(cx, 1), round(cy, 1), round(rng.uniform(1.5, 3.0), 1)))
        else:   # temporary restriction (e.g. helicopter activity)
            start = float(rng.integers(10, 50))
            zones.append(NoFlyZone(f"Z{i + 1}", round(cx, 1), round(cy, 1), round(rng.uniform(1.5, 3.0), 1),
                                   start, start + float(rng.integers(20, 40))))
    wps = []
    while len(wps) < n_waypoints:
        x, y = rng.uniform(1, size - 1, size=2)
        if any(math.hypot(x - z.x, y - z.y) < z.r + 0.5 for z in zones):
            continue
        w = Waypoint(f"W{len(wps) + 1}", round(x, 1), round(y, 1))
        if rng.random() < 0.4:                      # 40 % of waypoints have a time window
            e = float(rng.integers(0, 60))
            w.earliest, w.latest = e, e + float(rng.integers(15, 35))
        wps.append(w)
    return Scenario(depot, wps, zones)


def brief(s: Scenario) -> str:
    """Natural-language mission brief given to the LLM (rules come from the RAG documents)."""
    lines = [f"Inspection mission. Depot at ({s.depot[0]}, {s.depot[1]}) km. "
             f"Drone speed {s.speed} km/min, battery range {s.battery_range} km, "
             f"mission must end at the depot before t = {s.deadline} min.",
             "Waypoints to inspect (inspection takes the service time on site):"]
    for w in s.waypoints:
        tw = "" if w.latest == math.inf else f", time window [{w.earliest:g}, {w.latest:g}] min"
        lines.append(f"- {w.name} at ({w.x}, {w.y}), service {w.service:g} min{tw}")
    if s.zones:
        lines.append("No-fly zones (circles):")
        for z in s.zones:
            when = "always active" if z.end == math.inf else f"active during [{z.start:g}, {z.end:g}) min"
            lines.append(f"- {z.name} centre ({z.x}, {z.y}), radius {z.r} km, {when}")
    return "\n".join(lines)
