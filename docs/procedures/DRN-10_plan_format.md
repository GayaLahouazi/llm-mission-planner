# DRN-10 Plan format
A plan is returned as JSON: {"order": ["W3", "W1", ...]} listing every waypoint name exactly once, in visiting order, without the depot. Plans that are not valid JSON or reference unknown waypoints are rejected automatically.
