"""Tests for treeskel.guided_gt and the click tool's logic (treeskel.annotate)."""

import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from test_winter_gt import FORK, LEFT_TIP, RIGHT_TIP, synthetic_tree
from treeskel.annotate import Annotator
from treeskel.guided_gt import (
    Clicks,
    GuidedParams,
    branch_cost,
    guided_gt,
    insert_waypoint,
    load_clicks,
    main,
    check_clicks,
    save_clicks,
    trace,
)
from treeskel.io import index_dataset
from treeskel.winter_gt import render_main_branches

ROOT = (468, 320)


def synthetic_rgbd(**kw):
    """Synthetic V tree as (rgb, depth, valid): pale bark on green grass.

    The bark is shaded like a cylinder (brightest along the centre line), as
    real branches are; a flat-shaded bar would make the tube filter respond
    at its edges instead.
    """
    depth, valid = synthetic_tree(**kw)
    tree = depth == 150
    edt = ndi.distance_transform_edt(tree)
    local = ndi.maximum_filter(edt, size=15)
    shade = np.divide(edt, local, out=np.zeros_like(edt), where=local > 0)
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:] = (60, 120, 50)
    grey = (150 + 80 * np.sqrt(shade)).astype(np.uint8)
    rgb[tree] = np.stack([grey, grey, grey - 5], -1)[tree]
    return rgb, depth, valid


def good_clicks():
    return Clicks(trunk=[ROOT, FORK], left=[LEFT_TIP], right=[RIGHT_TIP])


def near(path, point, tol=4):
    return np.hypot(*(path - np.asarray(point)).T).min() <= tol


def test_cost_is_low_on_branches():
    cost, wood = branch_cost(*synthetic_rgbd())
    assert cost.shape == wood.shape == (480, 640)
    assert 0 <= wood.min() and wood.max() <= 1
    assert cost[400, 320] < cost[400, 100]         # trunk cheaper than ground
    assert cost[150, 250] > cost[150, 232]          # off vs on the left leader (x≈232 at row 150)
    assert cost.min() >= GuidedParams().cost_floor


def test_trace_passes_through_points_and_is_connected():
    cost, _ = branch_cost(*synthetic_rgbd())
    pts = [ROOT, FORK, LEFT_TIP]
    p = trace(cost, pts)
    assert tuple(p[0]) == ROOT and tuple(p[-1]) == LEFT_TIP
    assert all(near(p, q, 0) for q in pts)
    steps = np.abs(np.diff(p, axis=0)).max(1)
    assert steps.max() == 1                         # 8-connected, no jumps


def test_trace_follows_branch_not_straight_line():
    cost, _ = branch_cost(*synthetic_rgbd())
    p = trace(cost, [FORK, LEFT_TIP])
    rgb, depth, _ = synthetic_rgbd()
    on_tree = depth[p[:, 0], p[:, 1]] == 150
    assert on_tree.mean() > 0.95


def test_trace_crosses_depth_gap():
    rgb, depth, valid = synthetic_rgbd(gap=True)
    cost, _ = branch_cost(rgb, depth, valid)
    p = trace(cost, [FORK, RIGHT_TIP])
    assert tuple(p[-1]) == RIGHT_TIP


def test_guided_gt_result():
    gt = guided_gt(*synthetic_rgbd(), good_clicks())
    assert gt.ok and gt.root == ROOT and gt.fork == FORK
    assert tuple(gt.left[-1]) == LEFT_TIP and tuple(gt.right[-1]) == RIGHT_TIP
    assert tuple(gt.left[0]) == tuple(gt.right[0]) == FORK
    m = render_main_branches(gt)
    assert m[400, 320] and not m[400, 100]
    assert 9 <= m[400].sum() <= 17                 # trunk is 13 px wide
    trunk_row = [c for r, c in gt.trunk if r == 400]
    assert abs(trunk_row[0] - 320) <= 2            # traced down the centre
    assert gt.left_shape["radius"] > 0


def test_guided_gt_rejects_incomplete_or_skipped():
    with pytest.raises(ValueError):
        guided_gt(*synthetic_rgbd(), Clicks(trunk=[ROOT, FORK], left=[LEFT_TIP]))
    with pytest.raises(ValueError):
        guided_gt(*synthetic_rgbd(), Clicks(skip=True))


def test_waypoint_goes_to_nearest_segment_in_order():
    rgb, depth, valid = synthetic_rgbd()
    c = good_clicks()
    gt = guided_gt(rgb, depth, valid, c)
    # A point on the right leader, half way up: inserted before the right tip.
    c2 = insert_waypoint(c, gt, (175, 390))
    assert c2.right == [(175, 390), RIGHT_TIP] and c2.left == c.left and c2.trunk == c.trunk
    # A point on the trunk: between root and fork.
    c3 = insert_waypoint(c2, gt, (400, 320))
    assert c3.trunk == [ROOT, (400, 320), FORK]
    # A second right waypoint lower down goes before the first.
    c4 = insert_waypoint(c3, guided_gt(rgb, depth, valid, c3), (280, 350))
    assert c4.right == [(280, 350), (175, 390), RIGHT_TIP]
    assert c.right == [RIGHT_TIP]                  # original unchanged


def test_waypoint_reroutes_trace():
    rgb, depth, valid = synthetic_rgbd()
    c = Clicks(trunk=[ROOT, FORK], left=[LEFT_TIP], right=[(100, 600)])
    detour = (200, 600)                           # force the right trace via a far point
    c.right = [detour, (100, 600)]
    gt = guided_gt(rgb, depth, valid, c)
    assert near(gt.right, detour, 0)


def test_clicks_json_roundtrip(tmp_path):
    c = {"R01N01": good_clicks(), "R01N02": Clicks(skip=True, note="post")}
    path = tmp_path / "sub" / "clicks.json"
    save_clicks(path, c)
    assert load_clicks(path) == c
    data = json.loads(path.read_text())
    assert data["R01N01"]["trunk"] == [list(ROOT), list(FORK)]
    assert load_clicks(tmp_path / "missing.json") == {}


# --- click tool logic (no display needed) ----------------------------------

@pytest.fixture
def sample_trees(sample_dir):
    return index_dataset(sample_dir).trees[:3]


def test_annotator_full_flow(sample_trees, tmp_path):
    path = tmp_path / "clicks.json"
    ann = Annotator(sample_trees, str(path))
    first = ann.tree.tree_id
    assert "trunk base" in ann.prompt()
    for p in [(460, 330), (330, 310), (10, 110), (10, 450)]:
        ann.click(*p)
    assert ann.gt is not None and "Enter" in ann.prompt()
    ann.click(200, 420)                           # a correcting waypoint
    assert len(ann.clicks.left) + len(ann.clicks.right) + len(ann.clicks.trunk) == 5
    ann.key("u")                                  # undo the waypoint
    assert len(ann.clicks.left) + len(ann.clicks.right) + len(ann.clicks.trunk) == 4
    ann.key("enter")
    saved = load_clicks(path)
    assert saved[first].complete and saved[first].trunk[0] == (460, 330)
    assert ann.tree.tree_id != first              # moved on

    second = ann.tree.tree_id
    ann.click(400, 300)
    ann.key("u")
    assert ann.base == [] and ann.clicks is None
    ann.key("s")                                  # skip
    assert load_clicks(path)[second].skip

    ann.key("d")
    assert ann.show_depth
    ann.key("q")
    assert ann.done


def test_annotator_resumes_and_redo(sample_trees, tmp_path):
    path = tmp_path / "clicks.json"
    save_clicks(path, {sample_trees[0].tree_id: Clicks(trunk=[(460, 330), (330, 310)],
                                                        left=[(10, 110)], right=[(10, 450)])})
    ann = Annotator(sample_trees, str(path))
    assert [t.tree_id for t in ann.trees] == [t.tree_id for t in sample_trees[1:]]
    redo = Annotator(sample_trees, str(path), redo=True)
    assert redo.tree.tree_id == sample_trees[0].tree_id
    assert redo.gt is not None                    # existing clicks traced on load


def test_annotator_finishes(sample_trees, tmp_path):
    ann = Annotator(sample_trees[:1], str(tmp_path / "c.json"))
    ann.key("s")
    assert ann.done and "done" in ann.prompt()


def test_render_cli(sample_dir, tmp_path, capsys):
    trees = index_dataset(sample_dir).trees
    clicks = {trees[0].tree_id: Clicks(trunk=[(460, 330), (330, 310)], left=[(10, 110)], right=[(10, 450)]),
              trees[1].tree_id: Clicks(skip=True),
              "R99N99": Clicks(trunk=[(1, 1), (2, 2)], left=[(0, 0)], right=[(0, 3)])}
    save_clicks(tmp_path / "c.json", clicks)
    out = tmp_path / "out"
    assert main([str(sample_dir), "--clicks", str(tmp_path / "c.json"), "--out", str(out), "--overlays"]) == 0
    text = capsys.readouterr().out
    assert "Rendered 1 trees in" in text and "1 skipped or incomplete; 1 not found" in text
    assert (out / f"{trees[0].tree_id}_winter_gt.png").exists()
    assert (out / f"{trees[0].tree_id}_overlay.jpg").exists()
    assert (out / "summary.csv").exists()


def test_leaders_normalised_by_tip_position():
    c = Clicks(trunk=[ROOT, FORK], left=[(200, 400), RIGHT_TIP], right=[LEFT_TIP], note="x")
    n = c.normalised()
    assert n.left == [LEFT_TIP] and n.right == [(200, 400), RIGHT_TIP] and n.note == "x"
    assert good_clicks().normalised() == good_clicks()
    gt = guided_gt(*synthetic_rgbd(), c)
    assert tuple(gt.left[-1]) == LEFT_TIP and tuple(gt.right[-1]) == RIGHT_TIP


def test_annotator_normalises_swapped_tips(sample_trees, tmp_path):
    ann = Annotator(sample_trees, str(tmp_path / "c.json"))
    for p in [(460, 330), (330, 310), (10, 450), (10, 110)]:   # right tip clicked first
        ann.click(*p)
    assert ann.clicks.left == [(10, 110)] and ann.clicks.right == [(10, 450)]


def test_annotator_reset_and_undo_reset(sample_trees, tmp_path):
    path = tmp_path / "c.json"
    save_clicks(path, {sample_trees[0].tree_id: Clicks(trunk=[(460, 330), (330, 310)],
                                                        left=[(10, 110)], right=[(10, 450)])})
    ann = Annotator(sample_trees[:1], str(path), redo=True)
    old = ann.clicks
    ann.key("r")
    assert ann.clicks is None and ann.base == [] and "trunk base" in ann.prompt()
    ann.key("u")                                   # undo the reset
    assert ann.clicks == old and ann.gt is not None
    ann.key("r")
    for p in [(450, 320), (320, 300), (5, 100), (5, 460)]:
        ann.click(*p)
    ann.key("enter")
    assert load_clicks(path)[sample_trees[0].tree_id].trunk[0] == (450, 320)


def test_check_clicks():
    assert check_clicks(good_clicks()) == []
    assert check_clicks(Clicks(skip=True)) == []
    swapped_order = Clicks(trunk=[FORK, ROOT], left=[LEFT_TIP], right=[RIGHT_TIP])
    assert any("fork only" in i for i in check_clicks(swapped_order))
    double = Clicks(trunk=[(474, 356), (471, 355)], left=[LEFT_TIP], right=[RIGHT_TIP])
    assert any("fork only 3 px" in i for i in check_clicks(double))
    low_tip = Clicks(trunk=[ROOT, FORK], left=[(320, 200)], right=[RIGHT_TIP])
    assert any("left tip" in i for i in check_clicks(low_tip))
    close = Clicks(trunk=[ROOT, FORK], left=[(20, 300)], right=[(20, 320)])
    assert "leader tips are close together" in check_clicks(close)


def test_render_cli_filters_and_parallel(sample_dir, tmp_path, capsys):
    trees = index_dataset(sample_dir).trees
    c = Clicks(trunk=[(460, 330), (330, 310)], left=[(10, 110)], right=[(10, 450)])
    save_clicks(tmp_path / "c.json", {t.tree_id: c for t in trees[:3]})
    out = tmp_path / "out"
    first_row = int(trees[0].tree_id[1:3])
    # Row filter, run in parallel with 2 worker processes.
    assert main([str(sample_dir), "--clicks", str(tmp_path / "c.json"), "--out", str(out),
                 "--rows", str(first_row), "--jobs", "2"]) == 0
    text = capsys.readouterr().out
    n_row = sum(int(t.tree_id[1:3]) == first_row for t in trees[:3])
    assert f"Rendered {n_row} trees" in text and f"{n_row}/{n_row} trees" in text
    # Tree filter, single process.
    assert main([str(sample_dir), "--clicks", str(tmp_path / "c.json"), "--out", str(out),
                 "--trees", trees[2].tree_id, "--jobs", "1"]) == 0
    assert "Rendered 1 trees" in capsys.readouterr().out
    assert (out / f"{trees[2].tree_id}_winter_gt.png").exists()
