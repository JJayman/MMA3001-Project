"""Tests for treeskel.measure_fork."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_guided_gt import good_clicks, wide_fork_rgbd
from test_winter_gt import FORK
from treeskel.guided_gt import GuidedParams, guided_gt, load_clicks
from treeskel.measure_fork import (
    coverage,
    fit_decay_length,
    main,
    matching_length,
    shoulder_triangle,
)
from treeskel.winter_gt import render_main_branches

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wide_fork():
    rgb, depth, valid, truth = wide_fork_rgbd()
    gt = guided_gt(rgb, depth, valid, good_clicks(), GuidedParams(shoulder_length=0))
    return gt, truth


def test_shoulder_triangle_on_a_known_fork(wide_fork):
    gt, truth = wide_fork
    R = gt.radii["trunk"][-1]
    for side, sign in (("left", -1), ("right", 1)):
        tri = shoulder_triangle(gt, side, 1.5)
        assert np.allclose(tri[0], FORK)
        assert np.allclose(tri[1], (FORK[0], FORK[1] + sign * R), atol=1.5)   # trunk's edge beside the fork
        assert 1.2 * R <= np.hypot(*(tri[2] - FORK)) <= 1.8 * R             # 1.5 R along the (wiggly) path
        assert sign * (tri[2][1] - FORK[1]) > 0                              # on this leader's side
        # The true shoulder is filled (straight chamfer up to 30 px); the unfilled mask has a notch.
        assert coverage(truth, tri) >= 0.97
        assert coverage(render_main_branches(gt), tri) <= 0.95


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
                 "--jobs", "2", "--figure", str(figure), "--figure-lengths", "1.4", "2"]) == 0
    text = capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    dev = sorted(t for t in load_clicks(clicks) if t in ("R01N01", "R01N02", "R15N05", "R20N06"))
    assert report["trees"] == dev and report["test_rows_excluded"] == [3, 12, 16]
    assert "4 test-row trees left out" in text
    assert report["n_leaders"] == 2 * len(dev)
    for reach in ("1.5", "2.0"):
        cov = report["coverage"][reach]
        assert cov["k0.0"] <= cov["k1.4"] <= cov["k2.5"] <= 1     # more reach, more of the shoulder filled
    assert report["masks"]["0.0"]["added_px"] == 0 and report["masks"]["1.4"]["added_px"] > 0
    assert (out / "leaders.csv").read_text().count("\n") == 1 + 2 * len(dev)
    assert figure.exists()
