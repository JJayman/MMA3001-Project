"""Tests for treeskel.tune_cost."""

import numpy as np

from test_guided_gt import FORK, LEFT_TIP, RIGHT_TIP, ROOT, synthetic_rgbd
from treeskel.guided_gt import Clicks, GuidedParams, cost_cues
from treeskel.tune_cost import (
    GRID,
    base_clicks,
    evaluate,
    grid_sample,
    leave_one_out,
    pooled,
    summarise,
    waypoint_distances,
    waypoints,
)


def test_waypoints_and_base_clicks():
    c = Clicks(trunk=[(1, 1), (2, 2), (3, 3)], left=[(4, 4), (5, 5)], right=[(6, 6)])
    assert base_clicks(c) == Clicks(trunk=[(1, 1), (3, 3)], left=[(5, 5)], right=[(6, 6)])
    assert waypoints(c, "trunk") == [(2, 2)]
    assert waypoints(c, "left") == [(4, 4)]
    assert waypoints(c, "right") == []


def test_waypoint_distances():
    path = np.stack([np.arange(10, 90), np.full(80, 50)], 1)
    assert waypoint_distances(path, [(40, 50), (40, 53), (0, 50)]) == [0.0, 3.0, 10.0]


def test_summarise():
    s = summarise([0, 2, 4, 20], tolerance=8)
    assert s == {"mean_px": 6.5, "median_px": 3.0, "still_needed": 0.25, "n": 4}
    assert summarise([], 8)["n"] == 0


def test_grid_sample_includes_default_and_is_distinct():
    s = grid_sample(20, seed=1)
    assert s[0] == GuidedParams() and len(s) == len(set(s))
    assert all(p.sigmas in GRID["sigmas"] for p in s[1:])
    assert grid_sample(20, seed=1) == s            # reproducible


def test_leave_one_out_never_uses_the_held_out_tree():
    # Waypoint distances per setting and tree:   A       B       C
    results = [{"A": [0.0], "B": [5.0], "C": [5.0], "D": []},   # setting 0
               {"A": [9.0], "B": [1.0], "C": [1.0], "D": []}]   # setting 1
    rows, held = leave_one_out(results)
    chosen = {t: k for t, k, _ in rows}
    # Hold out A: on B, C setting 1 is better. Hold out B: on A, C setting 0
    # is better (2.5 vs 5). Same for C. D has no waypoints and is not scored.
    assert chosen == {"A": 1, "B": 0, "C": 0}
    assert held == [9.0, 5.0, 5.0]
    assert pooled(results[1], exclude="A") == [1.0, 1.0]


def test_evaluate_zero_when_waypoints_on_the_trace():
    rgb, depth, valid = synthetic_rgbd()
    p = GuidedParams()
    cues = {p.sigmas: cost_cues(rgb, depth, valid, p)}
    # Waypoints placed on the branches: the main-clicks trace passes through them.
    clicks = Clicks(trunk=[ROOT, (400, 320), FORK], left=[(175, 250), LEFT_TIP], right=[RIGHT_TIP])
    res = evaluate({"T": {"cues": cues, "clicks": clicks}}, p)
    assert len(res["T"]) == 2 and max(res["T"]) <= 3
    # A waypoint off the branch is far from the trace.
    off = Clicks(trunk=[ROOT, FORK], left=[(175, 400), LEFT_TIP], right=[RIGHT_TIP])
    assert evaluate({"T": {"cues": cues, "clicks": off}}, p)["T"][0] > 50
