"""Tests for treeskel.depth3d: the depth map's zero, on synthetic ground with a known answer."""

import json
from pathlib import Path

import numpy as np
import pytest

from treeskel.depth3d import (
    CENTRE,
    FOCAL_PX,
    SESSION2_ZERO,
    ZEROS,
    affine_error,
    calibrate,
    fit_zero,
    ground_blocks,
    main,
    pooled_zero,
    zero_errors,
)

ROOT = Path(__file__).resolve().parents[1]


def ground_depth(zero=300.0, scale=0.016, height=0.9, pitch=8.0, roll=2.0, far=4.2, noise=0.0, tree=True, seed=0):
    """A synthetic 8-bit depth photo: flat ground seen from a camera ``height`` m up, pitched down and rolled
    (degrees), mapped by ``v = zero − Z / scale``; beyond ``far`` m no reading. Optionally a nearer vertical
    trunk in the middle (which the ground blocks must leave out)."""
    h, w = 480, 640
    r, u = np.mgrid[0:h, 0:w].astype(float)
    x, y = (u - CENTRE[1]) / FOCAL_PX, (r - CENTRE[0]) / FOCAL_PX          # ray (x, y, 1), y down
    p, q = np.radians(pitch), np.radians(roll)
    # Height of the ray's direction below the camera, in the world: roll about the axis, then pitch down.
    yr = x * np.sin(q) + y * np.cos(q)
    down = yr * np.cos(p) + np.sin(p)
    z = np.where(down > 0, height / np.where(down > 0, down, 1), np.inf)     # distance along the axis
    v = zero - z / scale
    if noise:
        v = v + np.random.default_rng(seed).normal(0, noise, v.shape)
    depth = np.where((z < far) & (v >= 1), np.clip(np.round(v), 1, 255), 0)
    if tree:
        # A trunk 2 m away stands on the ground: it hides the ground behind it, and the nearer ground hides it.
        zt, cols = 2.0, slice(300, 340)
        behind = z[:, cols] > zt
        depth[:, cols] = np.where(behind, round(zero - zt / scale), depth[:, cols])
    depth = depth.astype(np.uint8)
    return depth, depth > 0


def test_zero_errors_find_a_known_zero():
    depth, valid = ground_depth(zero=300.0)
    bare, _ = ground_depth(zero=300.0, tree=False)
    blocks = ground_blocks(depth, valid)
    assert len(blocks) > 50
    # Every kept block is ground: the trunk, nearer than the ground behind it, is left out.
    truth = np.array([np.median(bare[int(r) - 7:int(r) + 9, int(u) - 15:int(u) + 17]) for u, r, _ in blocks])
    assert np.all(np.abs(blocks[:, 2] - truth) <= 1)
    # ... and without the outlier removal, trunk blocks would be in (so the check above can fail).
    assert ground_blocks(depth, valid, cut=1e9).shape[0] > len(blocks)
    fit = fit_zero(blocks)
    assert fit["zero"] == pytest.approx(300.0, abs=10)
    assert fit["rms"] < fit["rms_disparity"] and fit["rms"] < 1.0
    assert not fit["at_edge"]


@pytest.mark.parametrize("zero", [255.0, 350.0])
def test_the_zero_does_not_depend_on_the_scale_height_or_pose(zero):
    for scale, height, pitch, roll in ((0.012, 0.7, 5.0, -3.0), (0.02, 1.2, 12.0, 4.0)):
        depth, valid = ground_depth(zero=zero, scale=scale, height=height, pitch=pitch, roll=roll, far=6.0)
        fit = fit_zero(ground_blocks(depth, valid))
        assert fit["zero"] == pytest.approx(zero, rel=0.05), (scale, height, pitch, roll)


def test_a_map_linear_in_disparity_fits_the_affine_model():
    # v = a + b / Z: on flat ground v is affine in (u, r), the limit of a very large zero.
    r, u = np.meshgrid(np.arange(250.0, 480, 16), np.arange(16.0, 640, 32), indexing="ij")
    inv_z = 0.25 + 0.0001 * u + 0.002 * (r - 250)
    blocks = np.stack([u.ravel(), r.ravel(), (20 + 150 * inv_z).ravel()], 1)
    assert affine_error(blocks) < 1e-9
    fit = fit_zero(blocks)
    assert fit["zero"] >= 2000 or fit["at_edge"]


def test_noise_and_rounding_still_give_the_zero():
    depth, valid = ground_depth(zero=320.0, noise=1.5)
    assert fit_zero(ground_blocks(depth, valid))["zero"] == pytest.approx(320.0, rel=0.08)


def test_pooled_zero_and_summary():
    e = np.array([[3.0, 1.0, 2.0], [2.5, 1.5, 1.0], [np.inf, 1.0, 1.0]])
    assert pooled_zero(e, (250.0, 300.0, 350.0)) == 300.0        # the third tree rules 250 out
    assert np.isnan(pooled_zero(np.full((2, 3), np.inf), (250.0, 300.0, 350.0)))
    results = []
    for i, z in enumerate((280.0, 300.0, 320.0)):
        depth, valid = ground_depth(zero=z, seed=i)
        blocks = ground_blocks(depth, valid)
        results.append({"tree": f"R01N{i:02d}", "row": 1, "fit": fit_zero(blocks), "fit_all": fit_zero(blocks),
                        "errors": zero_errors(blocks).tolist(), "errors_all": zero_errors(blocks).tolist()})
    s = calibrate(results)
    assert s["kept"]["zero_median"] == pytest.approx(300.0, abs=10)
    assert 270 <= s["kept"]["pooled_zero"] <= 330
    assert s["kept"]["better_than_disparity"] == 1.0
    assert s["rows"]["1"]["n"] == 3


def test_session2_zero_is_in_the_grid():
    assert SESSION2_ZERO in ZEROS and min(ZEROS) < SESSION2_ZERO < max(ZEROS)


def test_cli_on_the_sample_leaves_out_the_test_rows(sample_dir, tmp_path, capsys):
    assert main([str(sample_dir), "--calibrate", "--out", str(tmp_path), "--jobs", "1",
                 "--split", str(ROOT / "splits.json")]) == 0
    s = json.loads((tmp_path / "calibration.json").read_text())
    split = json.loads((ROOT / "splits.json").read_text())
    rows = {int(r) for r in s["rows"]}
    assert rows == {1, 15, 20} and not rows & set(split["test_rows"])       # row 4 is excluded, 3 and 12 are test rows
    assert (tmp_path / "calibration_trees.csv").read_text().count("\n") == s["kept"]["n_trees"] + 1
    assert "The depth map's zero" in capsys.readouterr().out
