"""Tests for treeskel.measure_registration, with depth shifted by known amounts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk as draw_disk

from test_guided_gt import bark_rgb, good_clicks
from test_winter_gt import synthetic_tree
from treeskel import io
from treeskel.guided_gt import guided_gt
from treeskel.io import index_dataset, shift_image
from treeskel.measure_registration import (
    MI_REACH,
    TRIAL_MOVES,
    centre_difference,
    depth_runs,
    edge_overshoots,
    fit_affine,
    fit_shift,
    leader_sections,
    main,
    measure_tree,
    mi_peak,
    mi_surface,
    silhouette,
    split_half,
    zero_move,
)

ROOT = Path(__file__).resolve().parents[1]


def shifted_tree(shift):
    """The synthetic V tree with its bark in colour where the tree is, and its depth moved by ``shift``."""
    depth, _ = synthetic_tree()
    rgb = bark_rgb(depth == 150)
    moved = shift_image(depth, shift)
    return rgb, moved, moved > 0


@pytest.mark.parametrize("shift", [(0, 0), (4, -1), (-3, 2)])
def test_leaders_recover_a_known_shift(shift):
    rgb, depth, valid = shifted_tree(shift)
    gt = guided_gt(rgb, depth, valid, good_clicks())
    sec = leader_sections(rgb, gt)
    assert len(sec["side"]) > 300 and set(sec["side"].tolist()) == {0, 1}
    d0, d1, closed = depth_runs(silhouette(depth, valid), gt, sec)
    assert closed.mean() > 0.9
    dy, dx = fit_shift(sec["n_row"][closed], sec["n_col"][closed],
                       centre_difference(sec["first"], sec["last"], d0, d1)[closed])
    assert dy == pytest.approx(shift[0], abs=0.35) and dx == pytest.approx(shift[1], abs=0.35)


def test_the_move_that_leaves_no_shift():
    rgb, depth, valid = shifted_tree((3, -1))
    gt = guided_gt(rgb, depth, valid, good_clicks())
    sec = leader_sections(rgb, gt)
    fits = []
    for move in TRIAL_MOVES:
        moved = shift_image(depth, move)
        d0, d1, closed = depth_runs(silhouette(moved, moved > 0), gt, sec)
        fits.append(fit_shift(sec["n_row"][closed], sec["n_col"][closed],
                              centre_difference(sec["first"], sec["last"], d0, d1)[closed]))
    zm = zero_move(TRIAL_MOVES, fits)
    assert zm["move"] == pytest.approx([-3, 1], abs=0.35)
    assert np.allclose(zm["response"], np.eye(2), atol=0.15)           # a clean tree follows a move one for one


def test_zero_move_undoes_a_measure_that_follows_moves_less_than_one_for_one():
    moves = [(dy, dx) for dy in range(-5, 2) for dx in (-1, 0, 1)]
    jac = np.array([[0.85, 0.02], [0.0, 0.9]])
    start = np.array([3.2, -0.6])
    shifts = [start + jac @ np.array(m, float) for m in moves]
    shifts[3] = (np.nan, np.nan)                                       # a failed fit is skipped
    zm = zero_move(moves, shifts)
    assert zm["move"] == pytest.approx(list(-np.linalg.solve(jac, start)), abs=1e-9)
    assert zm["at_no_move"] == pytest.approx(list(start))
    assert zero_move(moves[:2], shifts[:2]) is None


def test_fit_shift_resists_outliers_and_needs_normals_both_ways():
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 600)
    n = np.c_[np.sin(angle), np.cos(angle)]
    delta = n @ np.array([3.5, -0.7]) + rng.normal(0, 1.0, 600)
    delta[:60] += 15                                                   # merged runs
    dy, dx = fit_shift(n[:, 0], n[:, 1], delta)
    assert dy == pytest.approx(3.5, abs=0.2) and dx == pytest.approx(-0.7, abs=0.2)
    assert np.isnan(fit_shift([1, 0], [0, 1], [1.0, 2.0])[0])           # too few cross-sections


def test_fit_affine_tells_a_shift_from_a_scale():
    rng = np.random.default_rng(1)
    rows, cols = rng.uniform(0, 480, 2000), rng.uniform(0, 640, 2000)
    angle = rng.uniform(0, 2 * np.pi, 2000)
    n_row, n_col = np.sin(angle), np.cos(angle)
    pure = fit_affine(n_row, n_col, rows, cols, 3 * n_row - 0.5 * n_col)
    assert pure["dy"] == pytest.approx(3) and pure["dx"] == pytest.approx(-0.5)
    assert max(abs(pure[k]) for k in ("dy_to_bottom", "dy_to_right", "dx_to_bottom", "dx_to_right")) < 1e-6
    # The depth 1% taller than the colour about the centre: 2.4 px more at the bottom edge.
    scaled = fit_affine(n_row, n_col, rows, cols, n_row * (3 + 0.01 * (rows - 240)))
    assert scaled["dy"] == pytest.approx(3) and scaled["dy_to_bottom"] == pytest.approx(2.4)
    assert abs(scaled["dy_to_right"]) < 1e-6


def test_edge_overshoots_follow_the_normal():
    # Bark from -3 to +3; depth from -5 to +4 along the normal.
    up, low = edge_overshoots([-0.8, 0.8], [-3, -3], [3, 3], [-5, -5], [4, 4])
    assert up.tolist() == [1, 2] and low.tolist() == [2, 1]           # normal up: +offsets are the upper side
    assert centre_difference(-3, 3, -5, 4) == -0.5


def test_split_half():
    rng = np.random.default_rng(2)
    real = rng.normal(3, 1.5, 400)
    same = split_half(real + rng.normal(0, 0.5, 400), real + rng.normal(0, 0.5, 400))
    assert same["pearson"] > 0.85 and same["real_sd"] == pytest.approx(1.5, abs=0.2)
    assert same["noise_sd"] == pytest.approx(0.5, abs=0.1)
    noise = split_half(3 + rng.normal(0, 1, 400), 3 + rng.normal(0, 1, 400))
    assert abs(noise["pearson"]) < 0.15 and noise["real_sd"] < 0.4
    assert split_half([1, 2], [np.nan, 1])["pearson"] is None


def blobs_scene(shift, seed=0, overshoot=1):
    """Colour and depth of the same random discs on a background, the depth moved by ``shift``.

    The depth reaches ``overshoot`` px past every object, as real depth does (about 2 px).
    """
    rng = np.random.default_rng(seed)
    label = np.zeros((480, 640), int)
    for k in range(60):
        rr, cc = draw_disk((rng.integers(0, 480), rng.integers(0, 640)), rng.integers(8, 40), shape=label.shape)
        label[rr, cc] = k % 5 + 1
    colours = np.array([[70, 120, 60], [200, 190, 180], [120, 80, 50], [60, 60, 70], [220, 60, 50], [150, 150, 40]])
    rgb = np.clip(colours[label] + rng.normal(0, 12, (480, 640, 3)), 0, 255).astype(np.uint8)
    depth = np.where(label > 0, 90 + 20 * label, 0).astype(np.uint8)
    if overshoot:
        y, x = np.mgrid[-overshoot:overshoot + 1, -overshoot:overshoot + 1]
        depth = ndi.grey_dilation(depth, footprint=y ** 2 + x ** 2 <= overshoot ** 2 + 0.5)
    depth = shift_image(depth, shift)
    return rgb, depth, depth > 0


@pytest.mark.parametrize("shift,overshoot", [((0, 0), 0), ((3, -2), 1), ((-2, 1), 2), ((1, 3), 2)])
def test_mutual_information_finds_a_known_shift(shift, overshoot):
    rgb, depth, valid = blobs_scene(shift, overshoot=overshoot)
    surf = mi_surface(rgb, depth, valid, reach=6)
    assert surf.shape == (5, 13, 13)
    dy, dx, edge = mi_peak(surf[0])
    assert (dy, dx) == pytest.approx(shift, abs=0.25) and not edge
    for band in surf[1:]:                                             # each band of rows on its own
        assert mi_peak(band)[:2] == pytest.approx(shift, abs=0.5)


def test_an_overshooting_depth_flattens_the_peak():
    # Every shift that keeps the colour's objects inside the depth's scores about the same, so the
    # highest point alone is decided by noise; the fitted centre is not.
    flat = mi_surface(*blobs_scene((0, 0), overshoot=2), reach=4)[0]
    y, x = np.mgrid[-4:5, -4:5]
    near, beyond = flat[y ** 2 + x ** 2 <= 4], flat[(y ** 2 + x ** 2 > 4) & (y ** 2 + x ** 2 <= 8)]
    assert np.ptp(near) < 0.002 * flat[4, 4]                           # 13 shifts within 2 px: a plateau
    assert beyond.max() < near.min() - 0.005 * flat[4, 4]              # it falls away beyond
    assert mi_peak(flat)[:2] == pytest.approx((0, 0), abs=0.15)


def test_mi_peak_refines_between_pixels_and_flags_the_edge():
    y, x = np.mgrid[-8:9, -8:9]
    tilted = -((y - 1.3) ** 2) - (x + 0.6) ** 2 - 0.3 * (y - 1.3) * (x + 0.6)      # vertex (1.3, -0.6)
    assert mi_peak(tilted) == pytest.approx((1.3, -0.6, False), abs=1e-6)
    dy, dx, edge = mi_peak(-((y - 9) ** 2) - x ** 2)                  # beyond the search
    assert edge and dy > 5
    assert mi_peak(-(np.abs(y - 6.0)) - np.abs(x))[2]                   # within the window of the edge
    assert mi_peak(y * 0.0 + x)[2]                                      # no maximum at all


def test_measure_tree_loads_raw_depth_unless_asked(sample_dir, monkeypatch):
    rec = {t.tree_id: t for t in index_dataset(sample_dir).trees}["R20N06"]
    clicks = io_clicks()["R20N06"]
    monkeypatch.setattr(io, "DEPTH_SHIFT", {"summer": (0, 0), "winter": (-4, 1)})
    raw = measure_tree((rec, clicks, False))
    assert raw["d0"].shape == (len(TRIAL_MOVES), len(raw["sections"]["side"]))
    assert raw["mi"]["winter"].shape == raw["mi"]["summer"].shape == (5, 2 * MI_REACH + 1, 2 * MI_REACH + 1)
    moved = measure_tree((rec, clicks, True))
    # The depth moved on loading, 4 rows up and 1 column right: on the same colour pixels (a margin
    # beyond the move keeps its uncovered edge out), the information at shift s' is the stored
    # depth's at s' + (4, -1), wherever both were searched.
    np.testing.assert_allclose(moved["mi"]["winter"][:, :-4, 1:], raw["mi"]["winter"][:, 4:, :-1], rtol=1e-6)
    np.testing.assert_array_equal(moved["mi"]["summer"], raw["mi"]["summer"])


def io_clicks():
    from treeskel.guided_gt import load_clicks
    return load_clicks(ROOT / "annotations" / "winter_clicks.json")


def test_cli_measures_development_rows_only(sample_dir, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    for p in sample_dir.glob("R*N0[12]_*.png"):         # R01N01/02 (development), R04N01 (excluded), R12N02 (test)
        shutil.copy(p, data)
    rc = main([str(data), "--clicks", str(ROOT / "annotations" / "winter_clicks.json"), "--split",
               str(ROOT / "splits.json"), "--out", str(tmp_path / "out"), "--jobs", "1", "--boot", "5",
               "--plot", str(tmp_path / "rows.png")])
    out = capsys.readouterr().out
    assert rc == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["trees"] == ["R01N01", "R01N02"] and report["test_rows_excluded"] == [3, 12, 16]
    assert "test-row trees left out" in out and "Mutual information" in out
    assert len(report["grid"]) == len(TRIAL_MOVES) and report["fixed_set"]["ci_dy"] is not None
    assert [r["row"] for r in report["by_row"]] == [1]
    rows = (tmp_path / "out" / "trees.csv").read_text().splitlines()
    assert rows[0].startswith("tree,row,time") and len(rows) == 3
    assert (tmp_path / "rows.png").stat().st_size > 10_000
    assert "NaN" not in (tmp_path / "out" / "report.json").read_text()   # standard JSON
