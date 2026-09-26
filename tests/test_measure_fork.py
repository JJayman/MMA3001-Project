"""Tests for treeskel.measure_fork."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_guided_gt import good_clicks, synthetic_rgbd, wide_fork_rgbd
from test_winter_gt import FORK
from treeskel.guided_gt import GuidedParams, guided_gt, load_clicks
from treeskel.measure_fork import (
    BASE,
    coverage,
    crotch_height,
    crotch_zone,
    fit_decay_length,
    fork_variants,
    main,
    matching_length,
    shoulder_triangle,
    variant_label,
)
from treeskel.winter_gt import render_main_branches

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wide_fork():
    rgb, depth, valid, truth = wide_fork_rgbd()
    gt = guided_gt(rgb, depth, valid, good_clicks(), BASE)
    return gt, truth


def test_base_masks_have_the_round_top_and_no_fill():
    assert (BASE.shoulder_length, BASE.junction_reach, BASE.flat_root) == (0, 0, False)


def test_shoulder_triangle_on_a_known_fork(wide_fork):
    gt, truth = wide_fork
    R = gt.radii["trunk"][-1]
    for side, sign in (("left", -1), ("right", 1)):
        tri = shoulder_triangle(gt, side, 1.5)
        assert np.allclose(tri[0], FORK)
        assert np.allclose(tri[1], (FORK[0], FORK[1] + sign * R), atol=1.5)   # trunk's edge beside the fork
        assert 1.2 * R <= np.hypot(*(tri[2] - FORK)) <= 1.8 * R             # 1.5 R along the (wiggly) path
        assert sign * (tri[2][1] - FORK[1]) > 0                              # on this leader's side
        # The true shoulder is filled (straight chamfer up to 30 px); the round-top mask has a notch,
        # which the junction fills.
        assert coverage(truth, tri) >= 0.97
        assert coverage(render_main_branches(gt), tri) <= 0.95
        assert coverage(fork_variants(gt, (), (1.5,))["j1.5"], tri) >= 0.97


def test_fork_variants(wide_fork):
    gt, _ = wide_fork
    masks = fork_variants(gt, (0.0, 1.4), (1.0, 1.5))
    base = render_main_branches(gt)
    assert list(masks) == ["k0.0", "k1.4", "j1.0", "j1.5"]
    assert np.array_equal(masks["k0.0"], base)
    assert (masks["k1.4"] & ~base).any() and not (base & ~masks["k1.4"]).any()    # the fill only adds
    assert (base & ~masks["j1.5"]).any()                                          # the junction also removes
    assert not (masks["j1.0"] & ~masks["j1.5"]).any()                             # more reach, more join
    assert [variant_label(n) for n in masks] == ["round top", "fill 1.4", "junction 1", "junction 1.5"]


def test_crotch_zone_lies_between_the_leaders_above_the_fork():
    gt = guided_gt(*synthetic_rgbd(), good_clicks())
    zone = crotch_zone(gt)
    rows, cols = np.nonzero(zone)
    assert zone.sum() > 50
    assert rows.max() < FORK[0]                                        # above the fork
    for r in np.unique(rows):                                          # strictly between the two paths
        lc, rc = gt.left[gt.left[:, 0] == r, 1].max(), gt.right[gt.right[:, 0] == r, 1].min()
        assert cols[rows == r].min() > lc and cols[rows == r].max() < rc
    assert np.hypot(rows - FORK[0], cols - FORK[1]).max() <= 2.5 * gt.radii["trunk"][-1]


def test_crotch_height_on_a_known_fork(wide_fork):
    gt, truth = wide_fork
    R = gt.radii["trunk"][-1]
    # The leaders' inner edges meet about 10 px above the fork (see test_junction_on_a_known_fork).
    assert crotch_height(truth, gt) * R == pytest.approx(10, abs=1.5)
    assert crotch_height(render_main_branches(gt), gt) >= 1.0                     # the round top: R or more
    assert crotch_height(fork_variants(gt, (), (1.5,))["j1.5"], gt) * R == pytest.approx(10, abs=2.5)
    assert np.isnan(crotch_height(np.ones_like(truth), gt))                      # no gap at all
    assert crotch_height(np.zeros_like(truth), gt) * R <= 3                       # a gap as soon as the paths part


def test_coverage():
    img = np.zeros((50, 50), bool)
    tri = np.array([[10.0, 10], [10, 40], [40, 10]])
    assert coverage(img, tri) == 0.0
    img[:, :25] = True
    assert 0.6 < coverage(img, tri) < 0.9
    assert coverage(~np.zeros_like(img), tri) == 1.0
    assert np.isnan(coverage(img, np.array([[-9.0, -9], [-9, -3], [-3, -9]])))   # no pixels inside the image


def test_fit_decay_length_recovers_a_known_length():
    rng = np.random.default_rng(0)
    s = np.arange(0.0, 150.0)
    R, L = 15.0, 22.0
    u = np.clip(s / L, 0, 1)
    width = 5 + (14 - 5) * 0.5 * (1 + np.cos(np.pi * u)) + rng.normal(0, 0.5, len(s))
    width[40:45] = 0                                          # a depth hole is ignored
    assert abs(fit_decay_length(s, width, R) - L) <= 2
    assert np.isnan(fit_decay_length(s, np.full(len(s), 6.0), R))      # no fall: nothing to measure
    assert np.isnan(fit_decay_length(s[:5], width[:5], R))              # too short


def test_matching_length_interpolates_and_leaves_one_out():
    def tree(name, depth, covs):
        row = {"depth@1.5": depth, **{f"k{k}@1.5": c for k, c in zip((0.0, 1.0, 2.0), covs)}}
        return {"tree": name, "leaders": [row]}

    results = [tree("A", 0.9, (0.8, 0.9, 1.0)), tree("B", 0.9, (0.8, 0.9, 1.0)), tree("C", 1.0, (0.8, 0.9, 1.0))]
    # Mean depth coverage 0.933 lies a third of the way from k = 1 (0.9) to k = 2 (1.0).
    assert matching_length(results, (0.0, 1.0, 2.0), 1.5, "depth") == pytest.approx(4 / 3)
    assert matching_length(results, (0.0, 1.0, 2.0), 1.5, "depth", exclude="C") == pytest.approx(1.0)
    assert np.isnan(matching_length([tree("D", 0.5, (0.8, 0.9, 1.0))], (0.0, 1.0, 2.0), 1.5, "depth"))


def test_cli_measures_development_rows_only(sample_dir, tmp_path, capsys):
    clicks = tmp_path / "clicks.json"
    shutil.copy(ROOT / "annotations" / "winter_clicks.json", clicks)
    out = tmp_path / "out"
    figure = tmp_path / "fig.jpg"
    assert main([str(sample_dir), "--clicks", str(clicks), "--split", str(ROOT / "splits.json"), "--out", str(out),
                 "--jobs", "2", "--figure", str(figure), "--figure-lengths", "1.4", "--figure-reaches", "1.5"]) == 0
    text = capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    dev = sorted(t for t in load_clicks(clicks) if t in ("R01N01", "R01N02", "R15N05", "R20N06"))
    assert report["trees"] == dev and report["test_rows_excluded"] == [3, 12, 16]
    assert "4 test-row trees left out" in text
    assert report["n_leaders"] == 2 * len(dev)
    for reach in ("1.5", "2.0"):
        cov = report["coverage"][reach]
        assert cov["k0.0"] <= cov["k1.4"] <= cov["k2.5"] <= 1     # more reach, more of the shoulder filled
        assert cov["k0.0"] <= cov["j1.0"] <= cov["j1.5"] <= cov["j2.5"] <= 1
    masks = report["masks"]
    assert masks["k0.0"]["added_px"] == masks["k0.0"]["removed_px"] == 0
    assert masks["k1.4"]["added_px"] > 0 and masks["k1.4"]["removed_px"] == 0   # the fill only adds
    assert masks["j1.5"]["removed_px"] > 0                                      # the round top goes
    heights = report["crotch_height"]
    assert heights["shrunk"] <= heights["depth"]                               # shrinking opens the crotch lower
    assert masks["j1.5"]["crotch_height"] <= masks["k0.0"]["crotch_height"]      # no round top in the crotch
    assert masks["j1.5"]["crotch_px"] <= masks["k0.0"]["crotch_px"]             # less of the crotch filled
    assert "junction 1.5" in text and "The crotch" in text
    assert (out / "leaders.csv").read_text().count("\n") == 1 + 2 * len(dev)
    assert figure.exists()


def test_cli_rejects_figure_settings_that_are_not_measured(sample_dir, tmp_path):
    with pytest.raises(SystemExit):
        main([str(sample_dir), "--out", str(tmp_path), "--figure", str(tmp_path / "f.jpg"), "--figure-reaches", "1.7"])
