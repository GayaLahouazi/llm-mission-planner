import math

from planner.llm import Scripted, parse_order, plan
from planner.rag import retrieve_rules
from planner.scenario import NoFlyZone, Scenario, Waypoint, brief
from planner.solver import exact_plan, generate_feasible, nearest_neighbour, simulate
from planner.verifier import verify


def tiny():
    return Scenario(depot=(0, 0), waypoints=[Waypoint("W1", 4, 0), Waypoint("W2", 0, 4, 0, 6)],
                    zones=[], speed=1.0, battery_range=100, deadline=100)


def test_valid_plan_and_schedule():
    v = verify(tiny(), ["W2", "W1"])
    assert v.valid and v.schedule["W2"] <= 6


def test_time_window_violation_is_explained():
    v = verify(tiny(), ["W1", "W2"])          # W2 reached at t = 4 + 2 + 5.66 > 6
    assert not v.valid and any("W2 must be inspected by" in p for p in v.problems)


def test_missing_and_duplicate_waypoints():
    v = verify(tiny(), ["W1", "W1"])
    assert not v.valid and len(v.problems) == 2


def test_permanent_zone_blocks_leg():
    s = tiny()
    s.zones = [NoFlyZone("Z1", 2, 0, 0.5)]
    v = verify(s, ["W1", "W2"])
    assert not v.valid and any("Z1" in p for p in v.problems)


def test_temporary_zone_can_be_avoided_by_waiting():
    s = Scenario(depot=(0, 0), waypoints=[Waypoint("W1", 4, 0)], speed=1.0, battery_range=100,
                 deadline=100, zones=[NoFlyZone("Z1", 2, 0, 0.5, start=0, end=10)])
    v = verify(s, ["W1"])
    assert v.valid and v.schedule["W1"] >= 14       # waits for the zone to close at t = 10


def test_battery_reserve_is_enforced():
    s = tiny()
    s.battery_range = 15                             # route ~13.7 km > 15 * 0.85 = 12.75 usable
    v = verify(s, ["W2", "W1"])
    assert not v.valid and any("usable range" in p for p in v.problems)


def test_simulator_and_z3_agree_on_random_missions():
    for seed in range(8):
        s, order, _ = generate_feasible(seed, n_waypoints=5)
        for o in (order, nearest_neighbour(s), list(reversed(order))):
            assert verify(s, o).valid == simulate(s, o)[0]


def test_exact_plan_is_optimal_on_small_case():
    order, d = exact_plan(tiny())
    assert order == ["W2", "W1"] and math.isclose(d, 4 + 4 * math.sqrt(2) + 4)


def test_parse_order_handles_prose_and_fences():
    assert parse_order('Sure!\n```json\n{"order": ["W2", "W1"]}\n```') == ["W2", "W1"]
    assert parse_order("I would go W3 then W1") == ["W3", "W1"]


def test_repair_loop_uses_verifier_feedback():
    llm = Scripted(['{"order": ["W1", "W2"]}', '{"order": ["W2", "W1"]}'])
    r = plan(tiny(), llm, use_rag=False, max_repairs=2)
    assert r.verdict.valid and r.iterations == 1 and not r.first_valid
    assert "W2 must be inspected by" in r.transcript[3]["content"]   # feedback sent to the LLM


def test_retrieval_finds_battery_and_zone_rules():
    s, _, _ = generate_feasible(0, n_waypoints=5)
    ids, text = retrieve_rules(brief(s))
    assert "DRN-03_battery_reserve" in ids and "DRN-05_no_fly_zones" in ids
    assert "15 %" in text
