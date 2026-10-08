"""Tests for treeskel.tree3d, on synthetic trees in 3D with a known opening, lean and turn."""

import json
from pathlib import Path

import numpy as np
import pytest

from test_depth3d import ground_depth
from treeskel.depth3d import Camera
from treeskel.tree3d import (
    SETTINGS,
    TURNED,
    angle,
    depth_along,
    fit_plane,
    ground_up,
    main,
    on_plane,
    paired,
    plane_against_photo,
    scales,
    stretch,
    summarise,
    to_metres,
    tree_measures,
    turned,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_tree(opening=50.0, lean=10.0, turn=-5.0, fork=(0.0, 40.0, 180.0), camera=Camera(),
                   leader=100.0, trunk=25.0, radius=5):
    """A flat V tree in 3D (units of the scale) rendered as a depth photo through ``camera``.

    The tree's plane leans its top ``lean`` degrees towards the camera and turns ``turn`` degrees (its
    right side farther); its leaders open ``opening`` degrees in the plane. Returns ``(paths, depth,
    valid)``: the skeleton's pixel paths (trunk root to fork, fork to each tip) and the depth image.
    """
    phi, tau, half = np.radians(lean), np.radians(turn), np.radians(opening / 2)
    up = np.array([0.0, -np.cos(phi), -np.sin(phi)])            # y is down; the top nearer when leaning
    side = np.array([np.cos(tau), 0.0, np.sin(tau)])            # the right side farther when turned
    side -= up * (side @ up)
    side /= np.linalg.norm(side)
    f = np.asarray(fork, float)
    parts = {"trunk": f[None] - np.linspace(trunk, 0, 400)[:, None] * up,
             "left": f + np.linspace(0, leader, 1500)[:, None] * (np.cos(half) * up - np.sin(half) * side),
             "right": f + np.linspace(0, leader, 1500)[:, None] * (np.cos(half) * up + np.sin(half) * side)}
    depth = np.zeros((480, 640), np.uint8)
    paths = {}
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = np.hypot(yy, xx) <= radius
    for k, p in parts.items():
        rc = np.stack([camera.centre[0] + camera.focal * p[:, 1] / p[:, 2],
                       camera.centre[1] + camera.focal * p[:, 0] / p[:, 2]], 1)
        px = np.rint(rc).astype(int)
        inside = (px[:, 0] >= radius) & (px[:, 0] < 480 - radius) & (px[:, 1] >= radius) & (px[:, 1] < 640 - radius)
        assert inside.all(), f"{k} leaves the photo"
        keep = np.r_[True, np.any(np.diff(px, axis=0) != 0, axis=1)]
        paths[k] = px[keep]
        for (r, c), z in zip(px[keep], p[keep, 2]):
            sl = (slice(r - radius, r + radius + 1), slice(c - radius, c + radius + 1))
            patch = depth[sl]
            patch[disc] = np.clip(np.rint(camera.zero - z), 1, 255)
    return paths, depth, depth > 0


def test_fit_plane_recovers_a_plane_despite_outliers():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 50, (400, 2))
    n_true = np.array([0.1, 0.2, -1.0]) / np.linalg.norm([0.1, 0.2, -1.0])
    z = 200 + (n_true[0] * xy[:, 0] + n_true[1] * xy[:, 1]) / -n_true[2]
    pts = np.c_[xy, z] + rng.normal(0, 0.3, (400, 3))
    pts[:60, 2] += rng.uniform(15, 40, 60)                      # 15% on the background behind
    plane = fit_plane(pts)
    assert angle(plane["normal"], n_true) < 1.0
    assert plane["normal"][2] < 0                               # it faces the camera
    assert plane["median_abs"] < 1.0 < plane["rms"]


def test_on_plane_inverts_the_projection():
    cam = Camera()
    plane = {"normal": np.array([0.0, 0.3, -1.0]) / np.hypot(0.3, 1.0), "centre": np.array([5.0, 30.0, 170.0])}
    rc = np.array([[100.0, 200.0], [400.0, 500.0]])
    p = on_plane(cam.rays(rc), plane)
    assert np.allclose((p - plane["centre"]) @ plane["normal"], 0, atol=1e-9)
    back = np.stack([cam.centre[0] + cam.focal * p[:, 1] / p[:, 2], cam.centre[1] + cam.focal * p[:, 0] / p[:, 2]], 1)
    assert np.allclose(back, rc)


def test_depth_along_reads_the_branch_and_skips_gaps():
    d = np.zeros((50, 50), np.uint8)
    d[20:30, :] = 120
    d[20:30, 30:40] = 0                                         # a gap in the depth
    v = depth_along(d, d > 0, np.array([[25, 10], [25, 35], [25, 30]]))
    assert v[0] == 120 and np.isnan(v[1]) and v[2] == 120


@pytest.mark.parametrize("opening, lean, turn", [(50.0, 10.0, -5.0), (35.0, 0.0, 0.0), (70.0, -8.0, 6.0)])
def test_a_synthetic_tree_is_measured_right(opening, lean, turn):
    cam = Camera()
    paths, depth, valid = synthetic_tree(opening, lean, turn, camera=cam)
    m = tree_measures(paths, depth, valid, cam, fork_radius=10.0)
    assert m["v_opening_3d"] == pytest.approx(opening, abs=1.0)
    assert m["lean_camera"] == pytest.approx(lean, abs=1.0)
    assert m["turn"] == pytest.approx(turn, abs=1.0)
    assert m["flatness_median"] < 0.6                           # rounding to whole depth units only
    assert m["correction"] == pytest.approx(m["v_opening_3d"] - m["v_opening_2d"])
    assert m["fork_distance"] == pytest.approx(180.0, rel=0.02)
    assert m["trunk_diameter"] == pytest.approx(2 * 10.0 * m["fork_distance"] / cam.focal)
    assert m["depth_share"] > 0.95


def test_a_tree_facing_the_camera_needs_no_correction():
    cam = Camera()
    paths, depth, valid = synthetic_tree(50.0, 0.0, 0.0, fork=(20.0, 40.0, 180.0), camera=cam)
    m = tree_measures(paths, depth, valid, cam, 10.0)
    # A plane facing the camera is seen without distortion of its angles.
    assert abs(m["correction"]) < 0.5


def test_a_wrong_zero_changes_the_lean_but_not_the_opening_of_a_facing_tree():
    true = Camera(zero=300.0)
    paths, depth, valid = synthetic_tree(50.0, 10.0, 0.0, camera=true)
    right = tree_measures(paths, depth, valid, true, 10.0)
    low = tree_measures(paths, depth, valid, Camera(zero=255.0), 10.0)
    # With a smaller zero every distance shrinks, depth differences stay: the lean looks larger.
    assert low["lean_camera"] > right["lean_camera"] + 2
    assert abs(low["v_opening_3d"] - right["v_opening_3d"]) < 1.5


def test_the_ground_gives_the_camera_pitch_and_a_vertical_tree_no_lean():
    cam = Camera(zero=300.0)
    for pitch in (8.0, 15.0):
        depth, valid = ground_depth(zero=300.0, pitch=pitch, roll=0.0, tree=False, far=6.0)
        up = ground_up(depth, valid, cam)
        assert np.degrees(np.arcsin(-up[2])) == pytest.approx(pitch, abs=1.0)
        # A vertical tree seen by this camera leans towards it by the pitch, and not at all from the vertical.
        paths, d, v = synthetic_tree(50.0, pitch, 0.0, camera=cam)
        m = tree_measures(paths, d, v, cam, 10.0, up=up)
        assert m["camera_pitch"] == pytest.approx(pitch, abs=1.0)
        assert m["lean_vertical"] == pytest.approx(0.0, abs=1.5)


def test_stretch_follows_the_2d_measure():
    path = np.stack([np.arange(400.0, 0, -1), np.full(400, 300.0)], 1)
    m = stretch(path, 16.5)
    s = np.arange(400.0)
    assert m.sum() == ((s >= 1.5 * 16.5) & (s <= 16 * 16.5)).sum()


def _rows():
    rows = []
    for i, (fd, v3, lean) in enumerate(((170.0, 50.0, 8.0), (180.0, 55.0, 10.0), (190.0, 60.0, 12.0))):
        for setting in SETTINGS:
            base = {"tree": f"R01N{i:02d}", "row": 1, "setting": setting, "fork_distance": fd, "v_opening_3d": v3,
                    "v_opening_2d": v3 + 1, "correction": -1.0, "lean_camera": lean, "turn": 0.0,
                    "lean_vertical": float("nan"), "camera_pitch": float("nan"), "flatness_rms": 3.0,
                    "flatness_median": 2.0, "trunk_diameter": 9.0, "left_length": 100.0, "right_length": 110.0,
                    "depth_points": 500.0, "depth_share": 0.95}
            rows.append({**base, "source": "winter"})
            rows.append({**base, "source": "U-Net", "v_opening_3d": v3 + 2, "lean_camera": lean - 1})
    return rows


def test_the_scale_puts_the_median_fork_at_two_metres():
    rows = _rows()
    per = scales(rows)
    assert per["main"] == pytest.approx(2.0 / 180.0)
    m = to_metres(rows, per)
    assert np.median([r["fork_distance"] for r in m if r["source"] == "winter" and r["setting"] == "main"]) == \
        pytest.approx(2.0)


def test_paired_and_summary():
    rows = _rows()
    p = paired(rows, "U-Net", "v_opening_3d")
    assert p["n"] == 3 and p["mae"] == pytest.approx(2.0) and p["bias"] == pytest.approx(2.0)
    rep = summarise(rows)
    assert rep["winter"]["main"]["v_opening_3d"]["median"] == 55.0
    assert rep["summer"]["U-Net"]["against_winter"]["lean_camera"]["bias"] == pytest.approx(-1.0)
    assert rep["summer"]["hand masks"]["n_trees"] == 0
    assert p["spearman"] == pytest.approx(1.0)                  # same order in both seasons


def test_rank_correlation_resists_a_failed_fit():
    rows = _rows()
    for i in range(3, 40):                                      # more trees, summer = winter + small noise
        for src, v in (("winter", 40.0 + i), ("U-Net", 40.0 + i + (0.5 if i % 2 else -0.5))):
            rows.append({**rows[0], "tree": f"R02N{i:02d}", "source": src, "setting": "main", "v_opening_3d": v})
    good = paired(rows, "U-Net", "v_opening_3d")
    rows.append({**rows[0], "tree": "R02N99", "source": "winter", "setting": "main", "v_opening_3d": 50.0})
    rows.append({**rows[0], "tree": "R02N99", "source": "U-Net", "setting": "main", "v_opening_3d": 150.0})
    bad = paired(rows, "U-Net", "v_opening_3d")
    assert bad["r"] < good["r"] - 0.3                           # one failed fit wrecks Pearson's r ...
    assert bad["spearman"] > 0.9                                # ... not the rank correlation
    assert bad["median_abs"] == pytest.approx(good["median_abs"], abs=0.5)


def test_plane_against_photo_counts_trees_where_the_plane_agrees_better():
    rows = []
    for i, (w3, s3, w2, s2) in enumerate(((50, 51, 52, 55), (40, 44, 42, 43), (60, 60.5, 62, 64), (45, 46, 47, 49))):
        for src, v3, v2 in (("winter", w3, w2), ("hand masks", s3, s2)):
            rows.append({"tree": f"R01N{i:02d}", "row": 1, "source": src, "setting": "main",
                         "v_opening_3d": float(v3), "v_opening_2d": float(v2)})
    pp = plane_against_photo(rows, "hand masks")
    assert pp["n"] == 4 and pp["plane_closer"] == 0.75
    assert pp["median_plane"] == pytest.approx(1.0) and pp["median_photo"] == pytest.approx(2.0)


def test_turned_lists_planes_far_from_facing_the_camera():
    rows = [{"tree": t, "source": "winter", "setting": "main", "turn": v}
            for t, v in (("R01N01", -4.0), ("R01N02", TURNED + 5), ("R01N03", -(TURNED + 1)), ("R01N04", float("nan")))]
    assert turned(rows, "winter") == ["R01N02", "R01N03"]


def test_cli_on_the_sample(sample_dir, tmp_path, capsys):
    out = tmp_path / "t3d"
    assert main([str(sample_dir), "--out", str(out), "--jobs", "1", "--split", str(ROOT / "splits.json"),
                 "--traces", str(ROOT / "data/shape/winter_traces.npz"),
                 "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"),
                 "--unet", str(ROOT / "docs/results/segment_lined_up_joined/predicted_masks.npz"),
                 "--figure", str(tmp_path / "fig.png")]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_trees"] == 4                                  # R01N01, R01N02, R15N05, R20N06: no test rows
    assert rep["summer"]["U-Net"]["n_trees"] == 4
    text = capsys.readouterr().out
    assert "V opening in the plane" in text and "Sensitivity" in text and "plane against photo" in text
    assert (tmp_path / "fig.png").exists()
    assert main(["--from-saved", str(out), "--out", str(tmp_path / "again"), "--figure", ""]) == 0
    assert json.loads((tmp_path / "again" / "report.json").read_text()) == rep
