"""Tests for treeskel.guided_gt and the click tool's logic (treeskel.annotate)."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import polygon

from test_winter_gt import FORK, LEFT_TIP, RIGHT_TIP, synthetic_tree, thick_line
from treeskel.annotate import Annotator
from treeskel.guided_gt import (
    Clicks,
    GuidedParams,
    arc_length,
    branch_cost,
    branch_edges,
    clear_of,
    guided_gt,
    insert_waypoint,
    load_clicks,
    main,
    check_clicks,
    fit_leader_radius,
    fit_taper,
    huber_line,
    huber_location,
    outward_normal,
    path_normals,
    profile_edges,
    scale_space_radius,
    shoulder_fill,
    smooth_segment,
    save_clicks,
    straighten,
    trace,
)
from treeskel.io import index_dataset
from treeskel.winter_gt import render_main_branches

ROOT = (468, 320)


def bark_rgb(tree):
    """Colour image of a bool tree mask: pale bark on green grass.

    The bark is shaded like a cylinder (brightest along the centre line), as
    real branches are; a flat-shaded bar would make the tube filter respond
    at its edges instead.
    """
    edt = ndi.distance_transform_edt(tree)
    local = ndi.maximum_filter(edt, size=15)
    shade = np.divide(edt, local, out=np.zeros_like(edt), where=local > 0)
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:] = (60, 120, 50)
    grey = (150 + 80 * np.sqrt(shade)).astype(np.uint8)
    rgb[tree] = np.stack([grey, grey, grey - 5], -1)[tree]
    return rgb


def synthetic_rgbd(**kw):
    """Synthetic V tree as (rgb, depth, valid), with bark from :func:`bark_rgb`."""
    depth, valid = synthetic_tree(**kw)
    return bark_rgb(depth == 150), depth, valid


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
    double.note = "splits at ground (checked)"
    assert check_clicks(double) == []
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


def test_huber_line_resists_outliers():
    x = np.arange(100.0)
    y = 10 - 0.05 * x
    y_bad = y.copy()
    y_bad[::10] += 15                              # 10% large outliers (e.g. ground blobs)
    a, b = huber_line(x, y_bad)
    assert abs(a - 10) < 0.3 and abs(b + 0.05) < 0.005
    ols_b = np.polyfit(x, y_bad, 1)[0]
    assert abs(b + 0.05) < abs(ols_b + 0.05)       # better than ordinary least squares
    assert huber_line(np.zeros(5), np.array([1.0, 2, 3, 4, 100])) == (3.0, 0.0)


def test_fit_taper_ignores_gaps_and_blobs():
    path = np.stack([np.arange(300, 0, -1), np.full(300, 200)], 1)
    true = np.linspace(6, 4, 300)
    meas = true.copy()
    meas[40:80] = 0.0                              # depth gap -> would draw a line
    meas[200:215] = 20.0                           # merged background -> blob
    r = fit_taper(path, meas, "left")
    assert np.abs(r - true).max() < 0.5
    assert r.min() >= 2.5


def test_fit_taper_never_widens_upwards_and_falls_back():
    path = np.stack([np.arange(100, 0, -1), np.full(100, 50)], 1)
    widening = np.linspace(4, 8, 100)
    r = fit_taper(path, widening, "left")
    assert np.ptp(r) == 0 and abs(r[0] - np.median(widening)) < 1e-9
    assert np.all(fit_taper(path, np.zeros(100), "trunk") == 10.0)   # fallback
    assert np.all(fit_taper(path, np.full(100, 40.0), "left") == 15.0)  # clipped


def test_guided_masks_have_no_thin_lines_across_depth_holes():
    rgb, depth, valid = synthetic_rgbd()
    depth = depth.copy()
    depth[150:165, 350:470] = 0                    # hole in the right leader's depth
    gt = guided_gt(rgb, depth, depth > 0, good_clicks())
    m = render_main_branches(gt)
    # Across the hole the leader is still drawn at full width (about 9 px).
    row = m[157]
    assert row[380:470].sum() >= 6
    assert gt.radii["left"][0] <= gt.radii["trunk"][-1]   # leaders no wider than trunk at the fork


def _connected(p):
    return np.abs(np.diff(p, axis=0)).max() <= 1


def test_smooth_segment_removes_zigzag_keeps_ends():
    rows = np.arange(200, 0, -1)
    zig = np.stack([rows, 300 + (rows % 2) * 2], 1)      # 2 px zig-zag on a straight line
    zig[-1] = (1, 300)
    sm = smooth_segment(zig, 6)
    assert tuple(sm[0]) == tuple(zig[0]) and tuple(sm[-1]) == tuple(zig[-1])
    assert _connected(sm)
    assert np.ptp(sm[5:-5, 1]) <= 1                         # zig-zag gone
    assert smooth_segment(zig, 0) is zig


def test_smooth_segment_keeps_real_bends():
    # A branch that bows 80 px sideways over 190 rows (half a sine wave).
    t = np.linspace(0, np.pi, 300)
    arc = np.stack([200 - 60 * t, 300 + 80 * np.sin(t)], 1).round().astype(int)
    sm = smooth_segment(arc, 6)
    # The 80 px bulge is kept to within a couple of pixels.
    assert abs(sm[:, 1].max() - arc[:, 1].max()) <= 2
    assert tuple(sm[0]) == tuple(arc[0]) and tuple(sm[-1]) == tuple(arc[-1]) and _connected(sm)


def test_smoothed_trace_still_passes_through_clicks():
    cost, _ = branch_cost(*synthetic_rgbd())
    pts = [ROOT, (400, 320), FORK, LEFT_TIP]
    p = trace(cost, pts, smooth_sigma=6)
    assert all(near(p, q, 0) for q in pts) and _connected(p)


def test_scale_space_radius_on_known_bars():
    rng = np.random.default_rng(0)
    x = np.arange(200) - 100
    for a in (3, 5, 8, 11):
        prof = np.where(np.abs(x) <= a, 0.5 + 0.35 * np.sqrt(np.clip(1 - (x / a) ** 2, 0, 1)), 0)
        img = 0.3 + 0.02 * rng.standard_normal((200, 200)) + prof[None, :]
        pts = np.stack([np.arange(60, 140), np.full(80, 102)], 1)   # 2 px off-centre
        r = scale_space_radius(img, pts)
        assert abs(np.median(r) - a) <= 1.0, (a, np.median(r))
    dark = 0.8 - np.where(np.abs(x) <= 5, 0.4, 0)[None, :] + np.zeros((200, 1))   # dark on light (sky)
    assert abs(np.median(scale_space_radius(dark, np.stack([np.arange(60, 140), np.full(80, 100)], 1))) - 5.5) <= 1


def test_leader_width_options_on_the_synthetic_tree():
    # The leaders' 9 px horizontal runs are 8.2 px across their 24° lean (half-width 4.1).
    gt = guided_gt(*synthetic_rgbd(), good_clicks())
    assert 3.5 <= np.median(gt.radii["left"]) <= 5.5
    assert 3.5 <= np.median(gt.radii["right"]) <= 5.5
    for params in (GuidedParams(leader_width_source="depth"),
                   GuidedParams(leader_width_source="scale_space", leader_fit="taper")):   # the old masks
        other = guided_gt(*synthetic_rgbd(), good_clicks(), params)
        assert len(other.radii["left"]) == len(other.left)
    with pytest.raises(ValueError, match="leader_width_source"):
        guided_gt(*synthetic_rgbd(), good_clicks(), GuidedParams(leader_width_source="colour"))
    with pytest.raises(ValueError, match="leader_fit"):
        guided_gt(*synthetic_rgbd(), good_clicks(), GuidedParams(leader_fit="spline"))


# --- leader widths from colour cross-sections --------------------------------

def lit_cylinder(h, centre=0.0, light=0.8, half=20, background=(110, 125, 95)):
    """Cross-section (``2·half + 1`` samples, RGB) of a matt branch lit from one side.

    Brightness is ``200 · (0.2 + max(0, L·n))`` for surface normal ``n`` and a
    light whose sideways component is ``light`` (negative: from the other
    side), as for a Lambertian cylinder: bright on the lit side, dark on the
    other, and darker at the edges than in the middle.
    """
    x = np.arange(-half, half + 1)
    u = (x - centre) / h
    inside = np.abs(u) <= 1
    facing = np.sqrt(np.clip(1 - u ** 2, 0, 1))
    shade = 0.2 + np.clip(-light * u + np.sqrt(1 - light ** 2) * facing, 0, None)
    prof = np.tile(np.array(background, float), (len(x), 1))
    prof[inside] = 200 * shade[inside, None] * np.array([1, 0.95, 0.97])
    return prof


def side_lit_rgb(tree, light=0.8, seed=0):
    """Colour image of a bool tree mask, each branch shaded as a matt cylinder lit from the left.

    Each horizontal run of the mask is shaded across its width as in
    :func:`lit_cylinder`, on grass.
    """
    rng = np.random.default_rng(seed)
    rgb = np.zeros(tree.shape + (3,))
    rgb[:] = (70, 120, 55)
    for y in range(tree.shape[0]):
        lab, n = ndi.label(tree[y])
        for i in range(1, n + 1):
            xs = np.nonzero(lab == i)[0]
            u = (xs - (xs[0] + xs[-1]) / 2) / ((xs[-1] - xs[0] + 1) / 2)
            shade = 0.2 + np.clip(-light * u + np.sqrt(1 - light ** 2) * np.sqrt(np.clip(1 - u ** 2, 0, 1)), 0, None)
            rgb[y, xs] = 200 * shade[:, None] * np.array([1, 0.95, 0.97])
    return np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)


def test_profile_edges_keep_both_halves_of_a_side_lit_branch():
    rng = np.random.default_rng(0)
    errors = []
    for h in (3, 5, 7, 9):
        for light in (-0.9, 0.0, 0.6, 0.9):
            for centre in (-2.0, 0.0, 1.5):
                profiles = lit_cylinder(h, centre, light)[None] + rng.normal(0, 6, (20, 41, 3))
                first, last = profile_edges(profiles)
                errors.append(np.median((last - first + 1) / 2) - h)
                assert abs(np.median((first + last) / 2) - centre) <= 0.5, (h, light, centre)
    # Within the pixel grid (a bar centred on a sample covers 2h + 1 samples),
    # except where the dark rim of a front-lit branch is as bright as the
    # background (h = 9, light 0), which loses one sample on each side.
    errors = np.abs(errors)
    assert errors.max() <= 1.0 and np.mean(errors <= 0.5) >= 0.9
    grey = profile_edges(lit_cylinder(5)[None].mean(-1))            # one channel works too
    assert (grey[1] - grey[0] + 1) / 2 == pytest.approx(5.5)
    assert [len(v) for v in profile_edges(np.zeros((0, 41, 3)))] == [0, 0]
    with pytest.raises(ValueError, match="too short"):
        profile_edges(np.zeros((2, 7, 3)))                          # no room for 3 background samples a side


def test_scale_space_sees_only_the_sunlit_half():
    # Why cross-section widths replaced the scale-space widths (session 8).
    rng = np.random.default_rng(0)
    for h in (5, 7, 9):
        img = np.tile(lit_cylinder(h, 0, 0.9, half=60).mean(1), (120, 1)) + rng.normal(0, 6, (120, 121))
        r = scale_space_radius(img / 255, np.stack([np.arange(20, 100), np.full(80, 60)], 1))
        assert np.median(r) < 0.75 * h


def test_straighten_samples_across_the_path():
    img = np.zeros((50, 60))
    img[:, 30:33] = 1.0                                             # a vertical stripe, columns 30-32
    path = np.stack([np.arange(40, 10, -1), np.full(30, 31)], 1)    # up column 31
    assert np.allclose(path_normals(path), [0, -1])                 # going up, the normal points left
    s = straighten(img, path, 5)
    assert s.shape == (30, 11)
    assert np.allclose(s[:, 4:7], 1) and np.allclose(s[:, :4], 0) and np.allclose(s[:, 7:], 0)
    assert straighten(np.dstack([img] * 3), path, 5).shape == (30, 11, 3)
    assert np.all(path_normals(np.array([[5, 5]])) == 0)            # a single point has no direction


def test_branch_edges_on_a_leaning_side_lit_branch():
    rng = np.random.default_rng(2)
    h, lean = 6.0, np.radians(25)
    along = np.array([-np.cos(lean), -np.sin(lean)])                # up and to the left
    across = np.array([-along[1], along[0]])
    yy, xx = np.indices((300, 300)).astype(float)
    d = (yy - 250) * across[0] + (xx - 200) * across[1]             # signed distance from the axis
    u = d / h
    shade = 0.2 + np.clip(-0.8 * u + 0.6 * np.sqrt(np.clip(1 - u ** 2, 0, 1)), 0, None)
    rgb = np.where((np.abs(u) <= 1)[..., None], 200 * shade[..., None] * np.array([1, 0.95, 0.97]), (70, 120, 55))
    rgb = np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255)
    t = np.arange(200)
    path = np.round(np.array([250, 200]) + t[:, None] * along + 1.0 * across).astype(int)   # 1 px off the axis
    first, last = branch_edges(rgb, path)
    assert abs(np.median((last - first + 1) / 2) - h) <= 0.6
    assert abs(np.median((first + last) / 2) + 1.0) <= 0.7          # sees the path is 1 px off
    assert [len(v) for v in branch_edges(rgb, np.zeros((0, 2), int))] == [0, 0]


def test_huber_location_resists_outliers():
    rng = np.random.default_rng(0)
    y = 6 + rng.normal(0, 0.5, 200)
    y[:30] = 3.0                                                    # cross-sections that lost the shaded half
    assert abs(huber_location(y) - 6) < 0.3
    assert abs(np.mean(y) - 6) > 0.4                                # the plain mean is pulled down
    assert huber_location(np.full(5, 2.0)) == 2.0


def test_fit_leader_radius_recovers_width_and_widening():
    rng = np.random.default_rng(1)
    path = np.stack([300 - np.arange(300), np.full(300, 200)], 1)  # straight up from the fork
    R, c = 15.0, 6.0
    s = arc_length(path)
    shape = 1 + 0.35 * np.exp(-s / (4 * R))
    meas = c * shape + rng.normal(0, 0.7, len(s))
    meas[s < 1.5 * R] = 2 * R            # the cross-section spans the top of the trunk: not used
    meas[100:140] = 0.0                  # failed edges: ignored
    meas[200:230] = c / 2                # lost shaded half: down-weighted
    r = fit_leader_radius(path, meas, R)
    assert np.abs(r - c * shape).max() < 0.4
    assert np.allclose(r / r[-1], shape / shape[-1])                # the shared shape, scaled
    flat = fit_leader_radius(path, meas, R, widening=0)
    assert np.ptp(flat) == 0 and abs(flat[0] - c) < 0.5
    assert np.allclose(fit_leader_radius(path, np.zeros(300), R), 5.0 * shape)   # fallback width
    assert fit_leader_radius(path, np.full(300, 40.0), R).max() == 15.0         # clipped
    assert len(fit_leader_radius(np.zeros((0, 2), int), np.zeros(0), R)) == 0


def test_leaders_on_a_side_lit_tree_are_drawn_full_width():
    depth, valid = synthetic_tree()
    rgb = side_lit_rgb(depth == 150)
    gt = guided_gt(rgb, depth, valid, good_clicks())
    for k in ("left", "right"):
        assert abs(np.median(gt.radii[k]) - 4.1) <= 0.5            # 8.2 px across the lean
        assert gt.radii[k][0] == pytest.approx(1.35 * gt.radii[k][-1], rel=0.02)   # widened at the fork
        assert gt.radii[k].max() <= gt.radii["trunk"][-1]


def test_clear_of():
    left, right = diagonal_leaders()
    d = np.hypot(*(left - right[0]).T)                              # the paths only meet at the fork
    clear = clear_of(left, right, 30.0)
    assert not clear[0] and clear[-1]
    assert np.all(clear == (np.hypot(*(left[:, None] - right[None]).T).min(0) > 30))   # brute force
    assert np.all(clear_of(left, np.zeros((0, 2), int), 30.0))
    assert d[0] == 0


def test_cross_sections_near_the_other_leader_are_not_used():
    # Near the fork a cross-section across one leader also takes in the other, so
    # on the synthetic tree (leaders of constant width) the raw widths there are
    # far too wide; the leader's own width comes only from cross-sections clear of it.
    depth, valid = synthetic_tree()
    rgb = side_lit_rgb(depth == 150)
    gt = guided_gt(rgb, depth, valid, good_clicks(), GuidedParams(fork_widening=0))
    first, last = branch_edges(rgb, gt.left)
    widths = (last - first + 1) / 2
    near = ~clear_of(gt.left, gt.right, GuidedParams().profile_half_width + 10)
    s = arc_length(gt.left) / gt.radii["trunk"][-1]
    merging = (s > 1.5) & (s < 3)
    assert np.median(widths[merging]) > 6 and near[merging].all()   # merged with the right leader, left out
    assert np.median(widths[~near]) == pytest.approx(4.0, abs=0.5)  # clear of it: the true width
    assert abs(gt.radii["left"][0] - 4.1) <= 0.3                    # own width: 8.2 px across


# --- where the leaders leave the trunk (shoulder fill) ----------------------

def diagonal_leaders(n=150, fork=(300, 300)):
    """Two straight leaders from ``fork``, up-left and up-right at 45°."""
    i = np.arange(n)
    return np.stack([fork[0] - i, fork[1] - i], 1), np.stack([fork[0] - i, fork[1] + i], 1)


def test_arc_length():
    assert np.allclose(arc_length(np.array([[0, 0], [0, 3], [4, 3]])), [0, 3, 7])
    assert np.allclose(arc_length(np.array([[5, 5], [6, 6]])), [0, np.sqrt(2)])
    assert arc_length(np.array([[5, 5]])).tolist() == [0.0]
    assert len(arc_length(np.zeros((0, 2), int))) == 0


def test_outward_normal_points_away_from_the_other_leader():
    left, right = diagonal_leaders()
    h = np.sqrt(0.5)
    assert np.allclose(outward_normal(left, right, 20), [h, -h])      # down and left
    assert np.allclose(outward_normal(right, left, 20), [h, h])       # down and right
    level = np.stack([np.full(100, 300), 300 - np.arange(100)], 1)    # runs straight left
    assert np.allclose(outward_normal(level, right, 20), [1, 0])      # outer side is below
    assert np.all(outward_normal(left[:1], right, 20) == 0)          # no direction yet


def test_shoulder_fill_is_a_straight_edge_on_the_outer_side_only():
    left, right = diagonal_leaders()
    R, r, L = 15.0, 4.0, 21.0                                          # trunk, leader radius; reach
    f = shoulder_fill((480, 640), left, right, np.full(len(left), r), R, L)
    q = np.argwhere(f) - np.array([300, 300])
    across = q @ outward_normal(left, right, L)                       # distance out from the leader's line
    along = q @ np.array([-np.sqrt(0.5), -np.sqrt(0.5)])              # distance up the leader
    assert across.min() >= -1                                         # nothing on the crotch side
    assert R - 1 <= across.max() <= R + 1                             # as wide as the trunk at the fork
    assert np.all(along <= L + r + 1)                                 # ends where the reach does
    # Between the trunk's disc and the last leader disc within reach (at
    # 14 diagonal steps, 19.8 px), the edge is their straight outer tangent:
    # it narrows linearly, with slope k / sqrt(1 - k^2), k = (R - r) / 19.8.
    a = np.arange(9.0, 21.0)
    edge = np.array([across[np.abs(along - v) <= 0.5].max() for v in a])
    slope, icpt = np.polyfit(a, edge, 1)
    k = (R - r) / (14 * np.sqrt(2))
    assert abs(slope + k / np.sqrt(1 - k * k)) <= 0.1, slope
    assert np.abs(edge - (icpt + slope * a)).max() <= 1.0            # straight: no bulge, no notch
    assert not shoulder_fill((480, 640), left, right, np.full(len(left), r), R, 0).any()


def wide_fork_rgbd():
    """Synthetic tree whose trunk is much wider than its leaders, as in the orchard.

    The trunk is 31 px wide and the leaders 9 px. On each outer side, a
    straight chamfer fills the shoulder from the trunk's edge at the fork to
    the leader's outer edge 30 px up it, as the wood does in the winter depth
    images; between the leaders, the crotch is a sharp V.

    Returns:
        ``(rgb, depth, valid, truth)``, where ``truth`` is the tree's mask.
    """
    d = np.zeros((480, 640), np.uint8)
    d[300:] = 100                                                      # ground
    thick_line(d, (470, 320), FORK, 31, 150)
    thick_line(d, FORK, LEFT_TIP, 9, 150)
    thick_line(d, FORK, RIGHT_TIP, 9, 150)
    f = np.array(FORK, float)
    for tip, side in ((LEFT_TIP, -1), (RIGHT_TIP, 1)):
        t = (np.array(tip) - f) / np.hypot(*(np.array(tip) - f))       # up the leader
        n = np.array([abs(t[1]), side * abs(t[0])])                    # its outer side
        edge = (f[0], f[1] + side * 15.5)                              # trunk's edge at the fork
        pts = [(f[0] + 5, edge[1]), edge, f + 30 * t + 4.5 * n, f + 30 * t, f]
        rr, cc = polygon(*np.array(pts).T, shape=d.shape)
        d[rr, cc] = 150
    truth = d == 150
    return bark_rgb(truth), d, d > 0, truth


FILL = GuidedParams(shoulder_length=1.4)
"""The shoulder fill as chosen in session 6 (off by default since session 8)."""


def test_shoulder_fill_on_a_known_fork():
    rgb, depth, valid, truth = wide_fork_rgbd()
    gt = guided_gt(rgb, depth, valid, good_clicks(), FILL)
    filled = render_main_branches(gt)
    unfilled = render_main_branches(replace(gt, shoulders=None))
    added = filled & ~unfilled
    R = gt.radii["trunk"][-1]
    assert 13 <= R <= 18                                               # trunk measured about 15.5 px
    yy, xx = np.indices(truth.shape)
    dist = np.hypot(yy - FORK[0], xx - FORK[1])
    # Only adds pixels, only near the fork, and all of them are wood.
    assert added.sum() >= 20
    assert dist[added].max() <= (FILL.shoulder_length + 1) * R
    assert (added & ~truth).sum() == 0
    # Nothing is added in the crotch: between the leaders' inner edges, above the fork.
    rise = FORK[0] - yy
    half_gap = rise * (140 / 310) - 4.5 / np.cos(np.arctan(140 / 310))  # centre-line offset minus half-width
    assert not (added & (rise > 0) & (np.abs(xx - FORK[1]) < half_gap)).any()
    # Closer to the true shape around the fork than without the fill.
    zone = dist <= 2.5 * R

    def iou(m):
        return (m & truth & zone).sum() / ((m | truth) & zone).sum()

    assert iou(filled) >= iou(unfilled) + 0.01, (iou(filled), iou(unfilled))


def test_shoulder_fill_is_off_by_default_and_can_be_turned_off():
    rgb, depth, valid, _ = wide_fork_rgbd()
    off = guided_gt(rgb, depth, valid, good_clicks(), GuidedParams(shoulder_length=0))
    assert not off.shoulders.any()
    assert np.array_equal(render_main_branches(off), render_main_branches(replace(off, shoulders=None)))
    # Since session 8 (leaders drawn at their true width) the default draws no fill.
    default = guided_gt(rgb, depth, valid, good_clicks())
    assert GuidedParams().shoulder_length == 0 and not default.shoulders.any()
    assert np.array_equal(render_main_branches(default), render_main_branches(off))


def test_shoulder_fill_when_the_tree_splits_at_the_ground():
    # A fork 4 px above the root: too little trunk to measure, so its width
    # falls back to the default; the fill still stays near the fork.
    rgb, depth, valid = synthetic_rgbd()
    gt = guided_gt(rgb, depth, valid, Clicks(trunk=[(470, 320), (466, 320)], left=[LEFT_TIP], right=[RIGHT_TIP]),
                   FILL)
    added = render_main_branches(gt) & ~render_main_branches(replace(gt, shoulders=None))
    assert np.all(gt.radii["trunk"] == 10.0) and added.any()             # fallback width, still filled
    reach = (FILL.shoulder_length + 1) * gt.radii["trunk"][-1]
    yy, xx = np.nonzero(added)
    assert np.hypot(yy - 466, xx - 320).max() <= reach + 1
