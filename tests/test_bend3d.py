"""Tests for treeskel.bend3d, on synthetic trees in 3D with known bends rendered as noisy 8-bit depth."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from treeskel.bend3d import (
    GRID,
    BendParams,
    behind_plane,
    bend_measures,
    blocks,
    block_cv,
    choose,
    cv_table,
    fit_bends,
    load_csv,
    main,
    out_of_fold,
    params_of,
    per_unit,
    posterior_sd,
    prepare,
    prior_terms,
    precision,
    report_lines,
    ridge_contrast,
    save_csv,
    setting_key,
    summarise,
)
from treeskel.depth3d import Camera
from treeskel.tree3d import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
PARAMS = BendParams(spread=0.02, length=0.2)


def bent_tree(lean=9.0, turn=-5.0, opening=50.0, trunk_tilt=0.0, bow=0.0, fork=(0.0, 40.0, 180.0), noise=0.7,
              seed=0, radii=(8, 5), camera=Camera(), trunk=25.0, leader=100.0, band=False):
    """A V tree in 3D (units of the scale) whose trunk and left leader may leave the V's plane, rendered as a
    noisy 8-bit depth photo of its near surfaces.

    The V's plane leans its top ``lean`` degrees towards the camera and turns ``turn`` degrees. The trunk
    runs down from the fork ``trunk_tilt`` degrees out of that plane (positive: the root behind); the left
    leader bows ``bow`` units behind it (a half sine over its first 60%, about the stretch its line is fitted to). The depth is the branches' near
    surface (a cylinder of the given radii in px), with smoothed noise of ``noise`` units, rounded. ``band``
    lays a horizontal branch 15 units nearer across the left leader.

    Returns:
        ``(paths, radii, depth, valid, true_z)``: the skeleton's pixel paths (trunk root to fork, fork to
        each tip), each point's radius (px), the depth image, and each path point's true axis distance (units).
    """
    rng = np.random.default_rng(seed)
    phi, tau, half = np.radians(lean), np.radians(turn), np.radians(opening / 2)
    up = np.array([0.0, -np.cos(phi), -np.sin(phi)])
    side = np.array([np.cos(tau), 0.0, np.sin(tau)])
    side -= up * (side @ up)
    side /= np.linalg.norm(side)
    back = np.cross(side, up)
    back = back if back[2] > 0 else -back                       # out of the plane, away from the camera
    f = np.asarray(fork, float)
    t = np.radians(trunk_tilt)
    s_t, s_l = np.linspace(trunk, 0, 1200)[:, None], np.linspace(0, leader, 4000)[:, None]
    parts = {"trunk": f + s_t * (-np.cos(t) * up + np.sin(t) * back),
             "left": f + s_l * (np.cos(half) * up - np.sin(half) * side)
             + bow * np.sin(np.pi * np.minimum(s_l / (0.6 * leader), 1)) * back,
             "right": f + s_l * (np.cos(half) * up + np.sin(half) * side)}
    value = np.zeros((480, 640))
    paths, true_z, rad = {}, {}, {}
    for k, p in parts.items():
        r_px = radii[0] if k == "trunk" else radii[1]
        rc = np.stack([camera.centre[0] + camera.focal * p[:, 1] / p[:, 2],
                       camera.centre[1] + camera.focal * p[:, 0] / p[:, 2]], 1)
        px = np.rint(rc).astype(int)
        keep = np.r_[True, np.any(np.diff(px, axis=0) != 0, axis=1)]
        paths[k], true_z[k] = px[keep], p[keep, 2]
        rad[k] = np.full(len(paths[k]), float(r_px))
        yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
        rho = np.hypot(yy, xx)
        disc = rho <= r_px
        for (r, c), z in zip(paths[k], true_z[k]):
            # The near surface of a cylinder of radius r_px around the axis at distance z.
            surf = z - np.sqrt(np.maximum((r_px * z / camera.focal) ** 2 - (rho * z / camera.focal) ** 2, 0))
            v = np.where(disc, camera.zero - surf, 0)
            sl = (slice(r - r_px, r + r_px + 1), slice(c - r_px, c + r_px + 1))
            value[sl] = np.maximum(value[sl], v)
    painted = value > 0
    if band:
        row = paths["left"][len(paths["left"]) // 2, 0]
        col = paths["left"][len(paths["left"]) // 2, 1]
        near = camera.zero - (true_z["left"][len(paths["left"]) // 2] - 15)
        value[row - 3:row + 4, col - 30:col + 31] = near
        painted[row - 3:row + 4, col - 30:col + 31] = True
    n = ndi.gaussian_filter(rng.normal(0, 1, value.shape), 1.5)
    n *= noise / n.std()
    depth = np.where(painted, np.clip(np.rint(value + n), 1, 255), 0).astype(np.uint8)
    return paths, rad, depth, depth > 0, true_z


def true_measures(paths, rad, depth, valid, true_z, params=PARAMS):
    """The tree's measures from its true 3D curve, against the plane fitted to its depth."""
    prep = prepare(paths, depth, valid, rad)
    z = np.zeros(len(prep.z_axis))
    for p in ("trunk", "left", "right"):
        z[prep.nodes[p]] = true_z[p] * prep.per_unit
    return prep, bend_measures(prep, z - prep.z_plane, params=params)


def test_per_unit_matches_the_3d_runs_scales():
    report = json.loads((ROOT / "docs/results/tree3d/report.json").read_text())
    for name, camera in SETTINGS.items():
        assert per_unit(camera) == pytest.approx(report["settings"][name]["metres_per_unit"], rel=2e-3)


def test_the_prior_has_the_stated_spread():
    paths, rad, depth, valid, _ = bent_tree(noise=0.0)
    prep = prepare(paths, depth, valid, rad)
    q = precision(prior_terms(prep), PARAMS)
    sd = np.sqrt(np.diag(np.linalg.inv(q.toarray())))
    middle = prep.chains["left"][len(prep.chains["left"]) // 2]
    assert sd[middle] == pytest.approx(PARAMS.spread, rel=0.1)


def test_the_parts_are_joined_at_the_fork():
    paths, rad, depth, valid, _ = bent_tree()
    prep = prepare(paths, depth, valid, rad)
    assert prep.nodes["trunk"][-1] == prep.nodes["left"][0] == prep.nodes["right"][0] == 0
    assert all(prep.chains[p][0] == 0 for p in prep.chains)
    nodes = np.concatenate([prep.nodes[p][1:] if p != "trunk" else prep.nodes[p] for p in prep.nodes])
    assert len(np.unique(nodes)) == len(nodes) == len(prep.z_axis)
    for p in ("trunk", "left", "right"):                       # nodes follow the caller's points
        assert np.array_equal(prep.paths[p], paths[p])


def test_a_flat_leaning_tree_shows_no_bends():
    paths, rad, depth, valid, true_z = bent_tree()
    prep = prepare(paths, depth, valid, rad)
    fit = fit_bends(prep, PARAMS)
    bent, flat = bend_measures(prep, fit["h"], params=PARAMS), bend_measures(prep, 0 * fit["h"], params=PARAMS)
    assert np.percentile(np.abs(fit["h"]), 95) < 0.01          # within 1 cm of the plane
    assert abs(bent["trunk_out"]) < 2.0
    assert abs(bent["left_bow"]) < 0.5 and abs(bent["right_bow"]) < 0.5
    assert abs(bent["v_opening"] - flat["v_opening"]) < 0.5     # the bends do not open the V
    assert flat["trunk_out"] == pytest.approx(0, abs=1e-6)


def test_the_surface_is_moved_to_the_axis():
    paths, rad, depth, valid, true_z = bent_tree(noise=0.0)
    prep = prepare(paths, depth, valid, rad)
    z = prep.z_axis[prep.nodes["trunk"]] / prep.per_unit
    ok = np.isfinite(z)
    assert np.median(np.abs(z[ok] - true_z["trunk"][ok])) < 0.6     # units; the surface is 2–3 units nearer
    no_radius = prepare(paths, depth, valid, {p: 0 * r for p, r in rad.items()})
    z0 = no_radius.z_axis[no_radius.nodes["trunk"]] / prep.per_unit
    assert np.median(true_z["trunk"][ok] - z0[ok]) > 1.5


@pytest.mark.parametrize("tilt,bow", [(20.0, 5.0), (-15.0, -4.0)])
def test_known_bends_are_recovered(tilt, bow):
    paths, rad, depth, valid, true_z = bent_tree(trunk_tilt=tilt, bow=bow)
    prep, truth = true_measures(paths, rad, depth, valid, true_z)
    assert np.sign(truth["trunk_out"]) == np.sign(tilt) and abs(truth["trunk_out"]) > 10
    fit = fit_bends(prep, PARAMS)
    got = bend_measures(prep, fit["h"], posterior_sd(fit), PARAMS)
    assert got["trunk_out"] == pytest.approx(truth["trunk_out"], abs=3.0)
    assert got["left_bow"] == pytest.approx(truth["left_bow"], abs=1.0)
    assert abs(truth["left_bow"]) > 3
    assert got["right_bow"] == pytest.approx(truth["right_bow"], abs=1.0)
    assert got["v_opening"] == pytest.approx(truth["v_opening"], abs=1.0)
    # Each point lies on its own ray: the bent skeleton still projects onto the trace.
    pts = prep.rays * (prep.z_plane + fit["h"])[:, None]
    assert np.allclose(pts[:, :2] / pts[:, 2:], prep.rays[:, :2])


def test_a_branch_in_front_gets_no_weight():
    paths, rad, depth, valid, _ = bent_tree(band=True)
    prep = prepare(paths, depth, valid, rad)
    fit = fit_bends(prep, PARAMS)
    k = prep.nodes["left"]
    d = prep.d[k]
    hit = np.isfinite(d) & (d < -0.1)                          # the band reads 15 units (18 cm) nearer
    assert hit.sum() >= 3
    assert np.all(fit["weights"][k][hit] == 0)
    assert np.max(np.abs(fit["h"][k])) < 0.015


def test_a_gap_stays_near_the_plane_and_is_uncertain():
    paths, rad, depth, valid, _ = bent_tree()
    prep = prepare(paths, depth, valid, rad)
    k = prep.nodes["left"]
    gap = np.zeros(len(prep.z_axis), bool)
    gap[k[150:300]] = True                                      # about 45 cm of the leader unread
    fit = fit_bends(prep, PARAMS, hold=gap)
    sd = posterior_sd(fit)
    assert sd[k[225]] > PARAMS.uncertain * PARAMS.spread > sd[k[100]]
    assert abs(fit["h"][k[225]]) < 0.01
    m = bend_measures(prep, fit["h"], sd, PARAMS)
    assert 0.1 < m["left_uncertain"] < 0.6 and m["right_uncertain"] == 0


def test_nothing_seen_is_the_prior_and_all_uncertain():
    paths, rad, depth, valid, _ = bent_tree()
    prep = prepare(paths, depth, valid, rad)
    fit = fit_bends(prep, PARAMS, hold=np.ones(len(prep.z_axis), bool))
    sd = posterior_sd(fit)
    assert np.all(fit["h"] == 0) and np.all(sd > PARAMS.uncertain * PARAMS.spread)
    m = bend_measures(prep, fit["h"], sd, PARAMS)
    assert m["trunk_uncertain"] == m["left_uncertain"] == m["right_uncertain"] == 1.0


def test_the_plane_is_no_fit():
    paths, rad, depth, valid, _ = bent_tree(trunk_tilt=20)
    prep = prepare(paths, depth, valid, rad)
    fit = fit_bends(prep, BendParams(spread=0.0))
    assert np.all(fit["h"] == 0) and fit["system"] is None
    assert np.all(posterior_sd(fit) == 0)


def test_block_cv_finds_bends_only_where_there_are_some():
    grid = [(0.02, 0.2), (0.04, 0.4)]
    flat, bent = (prepare(p, d, v, r) for p, r, d, v, _ in (bent_tree(seed=1), bent_tree(trunk_tilt=20, bow=5, seed=1)))

    def mae(cv, key):
        return sum(v[0] for v in cv[key].values()) / sum(v[2] for v in cv[key].values())

    cv_flat, cv_bent = block_cv(flat, grid), block_cv(bent, grid)
    assert mae(cv_bent, "tau 2 L 20") < 0.6 * mae(cv_bent, "plane")
    assert mae(cv_flat, "tau 2 L 20") > 0.9 * mae(cv_flat, "plane")
    # Every reading is scored once; the fork with the trunk.
    scored = np.sum([b[1] for b in blocks(flat)], axis=0)
    assert np.all(scored == 1)


def test_ridges_count_and_canopy_does_not():
    camera = Camera()
    depth = np.zeros((100, 100), np.uint8)
    depth[:, 45:56] = 130                                       # a branch in front of nothing
    path = np.stack([np.arange(10, 90), np.full(80, 50)], 1).astype(float)
    radii = np.full(80, 5.0)
    assert np.all(ridge_contrast(depth, depth > 0, path, radii) == np.inf)
    canopy = np.full((100, 100), 128, np.uint8)                 # the same branch inside foliage 2 units behind
    canopy[:, 45:56] = 130
    assert np.all(ridge_contrast(canopy, canopy > 0, path, radii) < 4)
    paths = {"trunk": np.stack([np.arange(90, 49, -1), np.full(41, 50)], 1),
             "left": np.stack([np.arange(50, 9, -1), np.arange(50, 9, -1)], 1),
             "right": np.stack([np.arange(50, 9, -1), np.arange(50, 91)], 1)}
    img = np.full((100, 100), 128, np.uint8)
    for p in paths.values():
        img[p[:, 0], p[:, 1]] = 130
    radii = {p: np.full(len(v), 1.0) for p, v in paths.items()}
    prep = prepare(paths, img, img > 0, radii, camera, ridge=True)
    assert not prep.anchor.any() and np.isfinite(prep.z_axis).all()   # all read, none count


def test_signs_of_the_measures():
    paths, rad, depth, valid, true_z = bent_tree(trunk_tilt=20, bow=5, noise=0.0)
    prep, truth = true_measures(paths, rad, depth, valid, true_z)
    assert truth["trunk_out"] > 0 and truth["left_bow"] > 0     # the root and the bow behind
    flat = true_measures(*bent_tree(noise=0.0))[1]               # leaders in the tree's plane: no tilt
    assert all(abs(flat[k]) < 0.1 for k in ("left_tilt", "right_tilt", "left_tilt_whole", "right_tilt_whole"))
    tilted = true_measures(*bent_tree(trunk_tilt=20, noise=0.0))[1]   # the leaders, not the trunk, hold the plane
    assert tilted["trunk_out"] > 19 and all(abs(tilted[k]) < 0.5 for k in ("left_tilt", "right_tilt"))
    pts = prep.rays * prep.z_plane[:, None]
    assert np.allclose(behind_plane(prep, pts), 0, atol=1e-9)
    assert behind_plane(prep, pts * 1.01).mean() > 0             # farther along the rays: behind


def test_settings_round_trip():
    for spread, length in GRID:
        p = params_of(setting_key(BendParams(spread=spread, length=length)))
        assert (p.spread, p.length) == pytest.approx((spread, length))
    assert params_of("plane").spread == 0 and setting_key(BendParams(spread=0)) == "plane"


def _cv_rows():
    rows = []
    for i, row in enumerate([1, 1, 2, 2, 5, 5]):
        cv = {"plane": {"trunk": [0.04, 0, 1], "left": [0.04, 0, 1], "right": [0.04, 0, 1]},
              "tau 2 L 20": {"trunk": [0.02, 0, 1], "left": [0.02 + 0.001 * i, 0, 1], "right": [0.02, 0, 1]},
              "tau 4 L 40": {"trunk": [0.03, 0, 1], "left": [0.03, 0, 1], "right": [0.03, 0, 1]}}
        rows += cv_table([{"tree": f"R{row:02d}N{i:02d}", "row": row, "cv": cv}])
    return rows


def test_choose_and_out_of_fold():
    rows = _cv_rows()
    assert choose(rows) == "tau 2 L 20"
    oof = out_of_fold(rows, [[1], [2], [5]])
    assert oof["chosen"] == ["tau 2 L 20"] * 3
    assert len(oof["trees"]) == 6
    assert all(b < a for a, b in oof["trees"].values())
    assert out_of_fold(rows, [[1, 2, 5], [7]])["trees"] == {}   # no other folds to choose on: skipped


def test_csv_round_trip_and_summary(tmp_path):
    cv_rows = _cv_rows()
    rows = []
    for i, r in enumerate(sorted({r["tree"] for r in cv_rows})):
        for source in ("winter", "winter ridges", "hand masks", "U-Net"):
            for setting in (SETTINGS if source == "winter" else ["main"]):
                for model in ("plane", "bent"):
                    tilt = 0.0 if model == "plane" else 10.0 + i + (2.0 if source != "winter" else 0.0)
                    rows.append({"tree": r, "row": int(r[1:3]), "source": source, "setting": setting, "model": model,
                                 "error": "", "trunk_out": tilt, "left_bow": 1.0 if model == "bent" else 0.0,
                                 "right_bow": 0.0, "v_opening": 50.0 + i, "cv_wide_mae": 4.0 if model == "plane" else 3.0,
                                 **{k: 0.5 for k in ("left_tilt", "right_tilt", "left_tilt_whole",
                                                     "right_tilt_whole", "left_out_max", "right_out_max", "trunk_out_max", "fork_behind",
                                                     "trunk_anchors", "left_anchors", "right_anchors", "sigma",
                                                     "left_length", "right_length", "trunk_out_rms", "left_out_rms",
                                                     "right_out_rms", "trunk_uncertain", "left_uncertain",
                                                     "right_uncertain")}})
    save_csv(tmp_path / "cv.csv", cv_rows)
    save_csv(tmp_path / "trees.csv", rows)
    assert load_csv(tmp_path / "cv.csv") == cv_rows
    back = load_csv(tmp_path / "trees.csv")
    rep = summarise(load_csv(tmp_path / "cv.csv"), back, [[1], [2], [5]])
    assert rep["chosen"] == "tau 2 L 20"
    assert rep["out_of_fold"]["better"] == 1.0
    tr = rep["summer"]["U-Net"]["against_winter"]["trunk_out"]
    assert tr["median_bent"] == pytest.approx(2.0) and tr["median_plane"] > 10 and tr["bent_closer"] == 1.0
    assert rep["winter"]["trunk_out"]["median"] == pytest.approx(12.5)
    assert rep["winter_ridges"]["median_abs_diff"] == pytest.approx(2.0) and rep["winter_ridges"]["spearman"] == 1.0
    assert rep["failed"]["winter"] == [] and set(rep["failed_winter_by_setting"]) == set(SETTINGS)


def test_failed_planes_are_reported_per_setting():
    rows = [{"tree": "R01N01", "row": 1, "source": "winter", "setting": "zero 262", "model": "bent", "error": "turned"},
            {"tree": "R01N02", "row": 1, "source": "winter", "setting": "main", "model": "bent", "error": "turned"},
            {"tree": "R01N02", "row": 1, "source": "winter", "setting": "zero 262", "model": "bent", "error": "turned"}]
    rep = summarise(_cv_rows(), rows, [[1], [2], [5]])
    assert rep["failed"]["winter"] == ["R01N02"]
    assert rep["failed_winter_by_setting"]["zero 262"] == ["R01N01", "R01N02"]
    assert "zero 262 2" in "\n".join(report_lines(rep))


def test_cli_on_the_sample(sample_dir, tmp_path, capsys):
    out = tmp_path / "bend3d"
    assert main([str(sample_dir), "--trees", "R15N05", "R20N06", "--out", str(out), "--jobs", "2",
                 "--figure", str(tmp_path / "bend3d.png"), "--profiles", "R15N05",
                 "--profiles-figure", str(tmp_path / "profiles.png")]) == 0
    text = capsys.readouterr().out
    assert "Chosen prior" in text and "out of fold" not in text   # two trees: too few to score out of fold
    for f in ("cv.csv", "trees.csv", "report.json", "output.txt"):
        assert (out / f).exists()
    assert (tmp_path / "bend3d.png").exists() and (tmp_path / "profiles.png").exists()
    rows = load_csv(out / "trees.csv")
    assert {r["source"] for r in rows} == {"winter", "winter ridges", "hand masks", "U-Net"}
    assert {r["tree"] for r in rows} == {"R15N05", "R20N06"}       # never a test-row tree
    assert main(["--from-saved", str(out), "--figure", ""]) == 0
