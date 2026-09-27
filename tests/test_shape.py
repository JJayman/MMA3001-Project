"""Tests for treeskel.shape: shape measures on synthetic trees with known angles."""

import numpy as np
import pytest
from skimage.draw import polygon

from treeskel.guided_gt import Clicks
from treeskel.shape import (
    MASK_ZOOM,
    MEASURES,
    ROLL_FREE,
    UNIT_PX,
    ShapeParams,
    click_paths,
    crop_top,
    framing_change,
    line_direction,
    mask_branches,
    mirror_measures,
    perspective_change,
    polyline,
    shape_measures,
    tilt,
    wrap,
)

FORK = (360, 320)


def ray(start, tilt_deg, length):
    """End point ``(row, col)`` of a straight stretch from ``start`` at a tilt from vertical, to a whole pixel.

    Rounding here, not in :func:`treeskel.shape.polyline`, lets a test take the
    exact tilt of what is drawn from the rounded points (:func:`drawn_tilt`):
    over 60-100 px, rounding alone moves a line by up to about 0.5 degrees.
    """
    t = np.radians(tilt_deg)
    return (round(start[0] - length * np.cos(t)), round(start[1] + length * np.sin(t)))


def drawn_tilt(a, b):
    """Tilt from vertical (degrees, positive right) of the straight stretch from point ``a`` to ``b``."""
    return float(np.degrees(np.arctan2(b[1] - a[1], a[0] - b[0])))


def v_tree(trunk_tilt=5.0, left_tilt=-25.0, right_tilt=20.0, left_turn=0.0, right_turn=0.0,
           fork=FORK, trunk_len=100.0, leader_len=320.0):
    """Paths of a V tree with known tilts (degrees from vertical, positive right).

    Each leader runs straight at its tilt for half its length, then turns by
    ``*_turn`` degrees (positive outwards) for the other half.

    Returns:
        ``(trunk, left, right)`` paths.
    """
    root = ray(fork, trunk_tilt + 180.0, trunk_len)            # straight down the trunk's axis
    mid_l = ray(fork, left_tilt, leader_len / 2)
    mid_r = ray(fork, right_tilt, leader_len / 2)
    return (polyline([root, fork]),
            polyline([fork, mid_l, ray(mid_l, left_tilt - left_turn, leader_len / 2)]),
            polyline([fork, mid_r, ray(mid_r, right_tilt + right_turn, leader_len / 2)]))


def test_tilt_and_line_direction():
    assert tilt(np.array([0.0, 1.0])) == 0.0
    assert tilt(np.array([1.0, 0.0])) == pytest.approx(90.0)
    d = line_direction(np.array([[5.0, 0.0], [4.0, 1.0], [3.0, 2.0]]))   # points up and to the left
    assert tilt(d) == pytest.approx(-45.0)
    assert wrap(190.0) == -170.0 and wrap(-180.0) == 180.0


def test_known_angles_of_a_straight_v():
    trunk, left, right = v_tree()
    m = shape_measures(trunk, left, right, fork_radius=15.0)
    t, lt, rt = drawn_tilt(trunk[0], trunk[-1]), drawn_tilt(left[0], left[-1]), drawn_tilt(right[0], right[-1])
    assert (t, lt, rt) == pytest.approx((5.0, -25.0, 20.0), abs=0.5)     # as asked, to a pixel
    assert m["trunk_tilt"] == pytest.approx(t, abs=0.15)
    assert m["left_tilt"] == pytest.approx(lt, abs=0.15)
    assert m["right_tilt"] == pytest.approx(rt, abs=0.15)
    assert m["v_opening"] == pytest.approx(rt - lt, abs=0.2)
    assert m["left_angle"] == pytest.approx(t - lt, abs=0.2)             # outwards is positive
    assert m["right_angle"] == pytest.approx(rt - t, abs=0.2)
    assert m["lean_v"] == pytest.approx((lt + rt) / 2, abs=0.2)
    assert m["lean"] == pytest.approx((lt + rt) / 2 - t, abs=0.2)        # about -7.5
    assert abs(m["left_bend"]) < 0.3 and abs(m["right_bend"]) < 0.3
    # The documented relations hold exactly.
    assert m["v_opening"] == pytest.approx(m["left_angle"] + m["right_angle"])
    assert m["lean"] == pytest.approx((m["right_angle"] - m["left_angle"]) / 2)


WHOLE = ShapeParams(start=0.0, reach=float("inf"))
"""Lines through the whole of each leader, so the halves split at a synthetic leader's corner."""


def test_bends_are_measured_outwards():
    m = shape_measures(*v_tree(left_turn=10.0, right_turn=-8.0), fork_radius=15.0, params=WHOLE)
    # The halves split at the corner, so each half is one straight stretch.
    assert m["left_bend"] == pytest.approx(10.0, abs=0.5)      # turns outwards (further left)
    assert m["right_bend"] == pytest.approx(-8.0, abs=0.5)     # turns inwards (back towards vertical)
    # The whole-leader line lies between the two stretches.
    assert -35.0 < m["left_tilt"] < -25.0 and 12.0 < m["right_tilt"] < 20.0


def test_bend_of_a_circular_arc_is_half_its_turning():
    # A right leader turning outwards by 24 degrees in total, along a circle.
    t = np.radians(np.linspace(10.0, 34.0, 200))
    radius = 300.0 / np.radians(24.0)                           # arc length 300 px
    cx, cy = FORK[1] - radius * np.cos(t[0]), FORK[0] - radius * np.sin(t[0])
    arc = np.stack([cy + radius * np.sin(t), cx + radius * np.cos(t)], 1)
    right = polyline(arc)
    trunk, left, _ = v_tree()
    m = shape_measures(trunk, left, right, fork_radius=5.0, params=WHOLE)
    # The halves' lines point along the arc at 1/4 and 3/4 of its length: 12 degrees apart.
    assert m["right_bend"] == pytest.approx(12.0, abs=0.7)


def test_only_the_top_of_the_trunk_sets_its_axis():
    # A trunk that is bent near its base: only the top 4 units (66 px) count.
    top = 4 * UNIT_PX
    knee = ray(FORK, 180.0 + 10.0, top)                        # the top tilted 10 degrees
    root = ray(knee, 180.0 - 30.0, 60.0)                       # base leaning the other way
    trunk = polyline([root, knee, FORK])
    _, left, right = v_tree()
    m = shape_measures(trunk, left, right, fork_radius=15.0)
    assert drawn_tilt(knee, FORK) == pytest.approx(10.0, abs=0.5)
    assert m["trunk_tilt"] == pytest.approx(drawn_tilt(knee, FORK), abs=0.3)


def test_scale_shift_and_roll_leave_the_roll_free_measures_unchanged():
    paths = v_tree(left_turn=6.0, right_turn=-4.0)
    for kw in ({"scale": 1.3}, {"roll": 5.0}, {"roll": -8.0, "scale": 0.8}):
        change = perspective_change(*paths, 15.0, **kw)
        for k in ROLL_FREE:
            assert abs(change[k]) < 0.3, (kw, k, change[k])
        for k in ("trunk_tilt", "left_tilt", "right_tilt", "lean_v"):     # turn with the camera
            assert change[k] == pytest.approx(kw.get("roll", 0.0), abs=0.3), (kw, k)


def test_pitch_and_yaw_change_the_angles_only_slightly():
    paths = v_tree()
    for kw in ({"pitch": 5.0}, {"yaw": 5.0}, {"pitch": -5.0, "yaw": 5.0}):
        change = perspective_change(*paths, 15.0, **kw)
        assert max(abs(change[k]) for k in ROLL_FREE) < 1.5, (kw, change)


def test_mirror_image():
    trunk, left, right = v_tree(trunk_tilt=7.0, left_tilt=-18.0, right_tilt=27.0, left_turn=5.0, right_turn=-3.0)
    m = shape_measures(trunk, left, right, 15.0)
    flip = lambda p: np.stack([p[:, 0], 639 - p[:, 1]], 1)
    f = shape_measures(flip(trunk), flip(right), flip(left), 15.0)        # the leaders swap sides
    expected = mirror_measures(m)
    for k in MEASURES:
        assert f[k] == pytest.approx(expected[k], abs=0.3), k


def test_tree_split_at_the_ground_has_no_trunk_axis():
    trunk = polyline([(FORK[0] + 4, FORK[1]), FORK])            # 4 px of trunk
    _, left, right = v_tree()
    m = shape_measures(trunk, left, right, 15.0)
    for k in ("trunk_tilt", "left_angle", "right_angle", "lean"):
        assert np.isnan(m[k])
    assert m["v_opening"] == pytest.approx(45.0, abs=0.3)


def test_short_leader_gives_nan():
    trunk, left, _ = v_tree()
    right = polyline([FORK, ray(FORK, 20.0, 25.0)])            # shorter than 1.5 trunk radii + 10 points
    m = shape_measures(trunk, left, right, 15.0)
    assert np.isnan(m["right_tilt"]) and np.isnan(m["v_opening"]) and np.isnan(m["right_bend"])
    assert m["left_tilt"] == pytest.approx(-25.0, abs=0.3)


def test_framing_changes_nothing_while_the_fixed_stretch_is_in_view():
    # Fork at row 360: keeping 0.7 of the height leaves 252 rows, more than 16 trunk radii of 15 px.
    for turn in (0.0, 10.0):
        change = framing_change(*v_tree(right_turn=turn), fork_radius=15.0, keep=0.7)
        assert max(abs(v) for v in change.values() if np.isfinite(v)) < 1e-9
    # Lines through all of each leader in view follow the framing on a bent leader.
    whole = ShapeParams(reach=float("inf"))
    assert max(abs(v) for v in framing_change(*v_tree(), 15.0, 0.7, whole).values() if np.isfinite(v)) < 0.3
    bent = framing_change(*v_tree(right_turn=10.0), 15.0, 0.7, whole)
    assert bent["right_tilt"] < -1.0              # less of the outward-turned top is in view


def test_lines_stop_at_the_reach():
    trunk, left, right = v_tree(right_turn=25.0)            # the corner is 160 px up the leader
    short = shape_measures(trunk, left, right, 15.0, ShapeParams(reach=10.0))   # 150 px: before the corner
    assert short["right_tilt"] == pytest.approx(drawn_tilt(right[0], ray(FORK, 20.0, 160.0)), abs=0.3)
    # The whole leader in view (320 px) is reported, not the stretch; an 8-connected path runs a few % long.
    assert short["right_length"] == pytest.approx(320.0 / 15.0, rel=0.06)


def test_crop_top():
    leader = polyline([FORK, (FORK[0] - 200, FORK[1] + 50)])
    kept = crop_top(leader, FORK[0] - 100)
    assert kept[-1][0] >= FORK[0] - 100 and kept[0].tolist() == list(FORK)
    assert len(crop_top(leader, -5)) == len(leader)


def test_click_paths_match_the_known_tree():
    trunk, left, right = v_tree()
    c = Clicks(trunk=[tuple(trunk[0]), FORK], left=[tuple(left[-1])], right=[tuple(right[-1])])
    p = click_paths(c)
    m = shape_measures(p["trunk"], p["left"], p["right"], 15.0)
    assert m["v_opening"] == pytest.approx(45.0, abs=0.5)
    assert m["trunk_tilt"] == pytest.approx(5.0, abs=0.7)
    # Tips clicked right-then-left are normalised by position, as for tracing.
    swapped = click_paths(Clicks(trunk=c.trunk, left=c.right, right=c.left))
    assert swapped["left"][-1].tolist() == p["left"][-1].tolist()


def v_mask(trunk_tilt=0.0, left_tilt=-24.0, right_tilt=22.0, trunk_half=9, leader_half=4):
    """A 256x256 hand-mask-like V: a trunk bar and two leader bars to the top edge."""
    m = np.zeros((256, 256), bool)
    fork = np.array([180.0, 128.0])

    def bar(a, b, half):
        d = (b - a) / np.hypot(*(b - a))
        n = np.array([-d[1], d[0]]) * half
        rr, cc = polygon(*np.stack([a + n, b + n, b - n, a - n], 1), shape=m.shape)
        m[rr, cc] = True

    bar(np.array(ray(fork, trunk_tilt + 180, 90.0)), fork, trunk_half)
    bar(fork, np.array(ray(fork, left_tilt, 200.0)), leader_half)
    bar(fork, np.array(ray(fork, right_tilt, 200.0)), leader_half)
    return m


def test_mask_branches_recovers_the_drawn_v():
    b = mask_branches(v_mask())
    assert b is not None
    assert b["fork_radius"] == pytest.approx(9.0, abs=1.5)
    m = shape_measures(b["trunk"], b["left"], b["right"], b["fork_radius"])
    assert m["left_tilt"] == pytest.approx(-24.0, abs=1.5)
    assert m["right_tilt"] == pytest.approx(22.0, abs=1.5)
    assert m["v_opening"] == pytest.approx(46.0, abs=2.0)
    assert m["trunk_tilt"] == pytest.approx(0.0, abs=2.0)


def test_mask_branches_on_an_empty_mask():
    assert mask_branches(np.zeros((256, 256), bool)) is None


# --- the run over the development trees (sample data) -------------------------

def test_cli_measures_development_trees_only(sample_dir, tmp_path, capsys):
    from treeskel.shape import load_features, main

    out = tmp_path / "shape"
    fig = tmp_path / "lines.jpg"
    assert main([str(sample_dir), "--out", str(out), "--features", "none", "--jobs", "2",
                 "--figure", str(fig), "--figure-trees", "R01N01"]) == 0
    text = capsys.readouterr().out
    assert "test-row trees left out" in text and "v_opening" in text
    import csv as _csv
    with open(out / "shape_measures.csv") as f:
        rows = list(_csv.DictReader(f))
    # The sample's development trees; test rows 3 and 12 and excluded row 4 are never measured.
    assert [r["tree"] for r in rows] == ["R01N01", "R01N02", "R15N05", "R20N06"]
    r = rows[0]
    assert abs(float(r["winter_v_opening"]) - float(r["hand_v_opening"])) < 15      # the same tree, two sources
    assert float(r["winter_fork_radius"]) > 5 and float(r["winter_left_length"]) > 16
    summary = __import__("json").loads((out / "shape_summary.json").read_text())
    assert summary["n_trees"] == 4 and summary["test_rows_excluded"] == [3, 12, 16]
    assert fig.exists() and not (out / "summer_resnet18.npz").exists()
    from treeskel.shape import load_traces
    traces = load_traces(out / "winter_traces.npz")
    assert list(traces) == [r["tree"] for r in rows] and len(traces["R01N01"]["left"]) > 100
    assert "winter_radii_v_opening" in rows[0] and "hand_radii_left_bend" in rows[0]
    assert set(summary["radii_unit"]) == set(summary["measures"])


def test_cli_features_with_random_weights(sample_dir, tmp_path):
    pytest.importorskip("torchvision")
    from treeskel.shape import GRID, load_features, main

    out = tmp_path / "shape"
    assert main([str(sample_dir), "--out", str(out), "--random-weights", "--device", "cpu", "--jobs", "2"]) == 0
    ids, arrays, meta = load_features(out / "summer_resnet18.npz")
    assert ids == ["R01N01", "R01N02", "R15N05", "R20N06"] and meta["weights"] == "random"
    assert arrays["rgb_global"].shape == (4, 2, 512) and arrays["depth_global"].shape == (4, 2, 512)
    assert arrays["rgb_cells"].shape == (4, 2, GRID * GRID, 512)
    # The mirror image is a different input, so its features differ from the photo's.
    assert not np.allclose(arrays["rgb_cells"][:, 0], arrays["rgb_cells"][:, 1])


def test_network_input_mirrors_and_normalises():
    from treeskel.shape import IMAGENET_MEAN, IMAGENET_STD, network_input

    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :320] = 255                                   # left half white
    x, xm = network_input(img, (24, 32)), network_input(img, (24, 32), mirror=True)
    white = (1 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(x[:, :, 2], white[:, None], atol=1e-5) and np.allclose(xm[:, :, -3], white[:, None], atol=1e-5)
    assert np.allclose(x[:, :, ::-1], xm, atol=1e-5)
    depth = network_input(np.full((480, 640), 128, np.uint8), (24, 32))
    assert depth.shape == (3, 24, 32) and depth.dtype == np.float32


def test_fitted_lines_and_drawing():
    from treeskel.shape import draw_shape, fitted_lines, mask_to_image

    trunk, left, right = v_tree()
    lines = fitted_lines(trunk, left, right, 15.0)
    assert set(lines) == {"trunk", "left", "right", "left_lower", "left_upper", "right_lower", "right_upper"}
    (r0, c0), (r1, c1) = lines["left"]
    assert r1 < r0 and c1 < c0                           # from near the fork, up and to the left
    assert np.degrees(np.arctan2(c1 - c0, r0 - r1)) == pytest.approx(drawn_tilt(left[0], left[-1]), abs=0.3)
    im = draw_shape(np.zeros((480, 640, 3), np.uint8), (trunk, left, right), lines, "x")
    assert im.size == (640, 480) and np.asarray(im).any()
    assert mask_to_image(np.array([[0, 0], [256, 256]])).tolist() == [[0.0, 80.0], [480.0, 560.0]]


def test_similarity_fit_recovers_a_known_move():
    from treeskel.align import apply_transform, describe
    from treeskel.shape import similarity_fit

    trunk, left, right = v_tree()
    dst = np.concatenate([trunk, left, right])[:, ::-1].astype(float)          # (x, y) with y down
    a = np.radians(3.0)
    move = np.array([[1.1 * np.cos(a), -1.1 * np.sin(a), 20.0], [1.1 * np.sin(a), 1.1 * np.cos(a), -30.0], [0, 0, 1]])
    src = apply_transform(np.linalg.inv(move), dst)[40:-40]                   # shows less than dst
    m, residual = similarity_fit(src, dst)
    d = describe(m, src.mean(0))
    assert d["rotation"] == pytest.approx(3.0, abs=0.2) and residual < 1.0
    # The scale is not recovered: straight lines through the fork map onto themselves under any
    # scaling about the fork, so a V pins down the rotation but hardly the scale (1.01 here, not 1.1).
    assert 0.95 < d["scale"] < 1.12


def test_hand_frame_of_a_mask_drawn_in_a_rolled_view():
    from treeskel.shape import hand_frame

    # A hand mask like v_mask, its V rolled 4 degrees clockwise, against the unrolled V traced in winter.
    mb = mask_branches(v_mask(trunk_tilt=4.0, left_tilt=-24.0 + 4.0, right_tilt=22.0 + 4.0))
    fork = (340, 320)
    winter = {"trunk": polyline([ray(fork, 180.0, 150.0), fork]),
              "left": polyline([fork, ray(fork, -24.0, 330.0)]), "right": polyline([fork, ray(fork, 22.0, 330.0)])}
    f = hand_frame(winter, mb)
    assert f["rotation"] == pytest.approx(4.0, abs=1.0) and f["residual"] < 3.0


def test_cli_draws_chosen_trees_without_touching_the_table(sample_dir, tmp_path, monkeypatch, capsys):
    from treeskel.shape import main

    monkeypatch.chdir(tmp_path)                            # the default output folder is relative
    split = tmp_path / "splits.json"
    split.write_text(__import__("json").dumps({"test_rows": [3, 12, 16]}))
    fig = tmp_path / "trees.jpg"
    clicks = __import__("pathlib").Path(__file__).resolve().parents[1] / "annotations" / "winter_clicks.json"
    assert main([str(sample_dir), "--trees", "R01N02", "R12N02", "--figure", str(fig), "--split", str(split),
                 "--clicks", str(clicks), "--jobs", "1"]) == 0
    text = capsys.readouterr().out
    assert "left out: R12N02" in text                      # a test-row tree is never measured
    assert (tmp_path / "outputs" / "shape_trees" / "shape_measures.csv").exists() and fig.exists()
    assert not (tmp_path / "data").exists() and "Computing summer features" not in text


# --- the unit of the stretch --------------------------------------------------

def test_the_stretch_is_in_pixels_not_in_the_trees_trunk_radius():
    paths = v_tree(left_turn=12.0, right_turn=-10.0)
    thin, thick = shape_measures(*paths, 10.0), shape_measures(*paths, 22.0)
    assert all(thin[k] == thick[k] for k in MEASURES)       # the trunk's radius no longer moves the stretch
    radii = ShapeParams(unit=None)
    a, b = shape_measures(*paths, 10.0, radii), shape_measures(*paths, 22.0, radii)
    assert abs(a["right_tilt"] - b["right_tilt"]) > 1.0     # in trunk radii it did, on curved leaders
    # Lengths are in units; an 8-connected path runs up to 8% long at these slopes.
    assert thin["left_length"] == pytest.approx(320.0 / UNIT_PX, rel=0.1)


def test_zoom_measures_the_same_stretch_in_a_larger_view():
    # The same bent tree in a photo 1.12 times larger, as a summer photo shows it.
    trunk, left, right = v_tree(left_turn=12.0, right_turn=-10.0)
    big = [polyline(FORK + (np.asarray(q, float) - FORK) * 1.12) for q in (trunk, left, right)]
    same = shape_measures(*big, 15.0, zoom=1.12)
    plain = shape_measures(*big, 15.0)                        # the winter stretch in the larger view
    orig = shape_measures(trunk, left, right, 15.0)
    for k in ("v_opening", "left_bend", "right_bend", "left_tilt", "right_tilt"):
        assert same[k] == pytest.approx(orig[k], abs=0.5), k
    assert abs(plain["right_tilt"] - orig["right_tilt"]) > 1.0
    assert MASK_ZOOM == pytest.approx(1.12 * 256 / 480)


def test_saved_traces_load_back_exactly(tmp_path):
    from treeskel.shape import load_traces, save_traces

    trunk, left, right = v_tree()
    traces = [{"trunk": trunk, "left": left, "right": right, "fork_radius": 14.5},
              {"trunk": trunk[:5], "left": left[:9], "right": right, "fork_radius": 20.0}]
    save_traces(tmp_path / "t.npz", ["R01N01", "R02N02"], traces)
    back = load_traces(tmp_path / "t.npz")
    assert list(back) == ["R01N01", "R02N02"]
    for t, orig in zip(back.values(), traces):
        for part in ("trunk", "left", "right"):
            assert np.array_equal(t[part], orig[part])
        assert t["fork_radius"] == pytest.approx(orig["fork_radius"])
