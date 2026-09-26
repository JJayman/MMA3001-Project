"""Tests for treeskel.summer_gt (summer masks from clicked winter-summer pairs)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from treeskel import summer_gt
from treeskel.align import apply_transform, warp_mask
from treeskel.summer_gt import (
    PairLimits,
    SeasonPairs,
    check_pairs,
    load_pairs,
    loo_errors,
    loo_table,
    pair_transform,
    save_pairs,
)

CLICKS = Path(__file__).resolve().parents[1] / "annotations" / "winter_clicks.json"
WINTER = [(300.0, 200.0), (350.0, 450.0), (150.0, 300.0), (420.0, 330.0)]    # (row, col)


def similarity(scale=1.0, deg=0.0, tx=0.0, ty=0.0):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1.0]])


def make_pairs(matrix, winter=WINTER, noise=0.0, seed=0):
    """Pairs whose summer points are the winter points moved by ``matrix`` (plus click noise)."""
    w = np.asarray(winter, float)
    s = apply_transform(matrix, w[:, ::-1])[:, ::-1]
    s += np.random.default_rng(seed).normal(0, noise, s.shape)
    return SeasonPairs(winter=[tuple(p) for p in w], summer=[tuple(p) for p in s])


TRUTH = similarity(1.05, 2.0, 15.0, -20.0)


def test_pairs_json_round_trip(tmp_path):
    p = SeasonPairs(winter=[(1.234, 5.678)], summer=[(9.87, 6.54)], note="x")
    assert p.to_json() == {"winter": [[1.2, 5.7]], "summer": [[9.9, 6.5]], "skip": False, "note": "x"}
    path = tmp_path / "a" / "pairs.json"
    save_pairs(path, {"R01N01": p, "R01N02": SeasonPairs(skip=True)})
    back = load_pairs(path)
    assert back["R01N01"].winter == [(1.2, 5.7)] and back["R01N02"].skip
    assert load_pairs(tmp_path / "missing.json") == {}
    with pytest.raises(ValueError):
        SeasonPairs(winter=[(1, 2)], summer=[])


def test_pair_transform_uses_row_col_points():
    """Points are (row, col); the transform is (x, y): a pure shift must come out right."""
    p = SeasonPairs(winter=[(100, 200), (300, 250), (200, 400), (350, 500)],
                    summer=[(90, 215), (290, 265), (190, 415), (340, 515)])
    for model in summer_gt.PAIR_MODELS:
        m = pair_transform(p, model)
        assert np.allclose(apply_transform(m, [(200, 100)]), [(215, 90)])


def test_exact_pairs_have_no_leave_one_out_error():
    p = make_pairs(TRUTH)
    m = pair_transform(p)
    grid = np.random.default_rng(1).uniform([0, 0], [640, 480], (20, 2))
    assert np.abs(apply_transform(m, grid) - apply_transform(TRUTH, grid)).max() < 1e-6
    assert np.nanmax(loo_errors(p)) < 1e-6
    assert check_pairs(p) == []


def test_leave_one_out_singles_out_a_wrong_click():
    p = make_pairs(TRUTH)
    r, c = p.summer[2]
    p.summer[2] = (r + 24.0, c - 18.0)                  # 30 px off
    loo = loo_errors(p)
    assert int(np.argmax(loo)) == 2 and loo[2] > 25
    assert any("pair 3" in s for s in check_pairs(p))


def test_leave_one_out_needs_enough_pairs():
    p = make_pairs(TRUTH, winter=WINTER[:2])
    assert np.isnan(loo_errors(p, "similarity")).all()          # 2 pairs fix it exactly: nothing left to check
    assert np.isfinite(loo_errors(p, "translation")).all()
    assert check_pairs(p) == ["only 2 pair(s)"]


def test_leave_one_out_error_reflects_click_noise():
    """With 1 px of click noise per coordinate, the leave-one-out error is about 1.2 px.

    That is the median length of 2D Gaussian noise with sigma = 1 (a Rayleigh
    distribution: sigma * sqrt(2 ln 2)), plus a little error of the fit itself.
    """
    rng = np.random.default_rng(2)
    winter = [tuple(x) for x in rng.uniform([50, 50], [430, 590], (30, 2))]
    loo = loo_errors(make_pairs(TRUTH, winter, noise=1.0, seed=3))
    assert 1.0 < np.median(loo) < 1.8


def test_check_pairs_flags_implausible_transforms_and_ignores_skipped():
    assert check_pairs(make_pairs(similarity(1.6))) == ["scale 1.60"]
    assert check_pairs(make_pairs(similarity(1.0, 12.0))) == ["rotation 12.0 deg"]
    assert check_pairs(make_pairs(similarity(1.6)), limits=PairLimits(scale_range=(0.5, 2))) == []
    assert check_pairs(SeasonPairs(skip=True)) == []


def test_loo_table_compares_models_on_the_same_trees():
    exact_shift = make_pairs(similarity(1.0, 0.0, 10, 5))
    scaled = make_pairs(similarity(1.2, 0.0, 10, 5))
    too_few = make_pairs(TRUTH, winter=WINTER[:3])
    rows = {r["model"]: r for r in loo_table({"a": exact_shift, "b": scaled, "c": too_few},
                                             summer_gt.PAIR_MODELS[:3])}
    assert all(r["trees"] == 2 and r["pairs"] == 8 for r in rows.values())   # "c" has too few for affine
    assert rows["similarity"]["median"] < 1e-6
    assert rows["translation"]["mean"] > 10                                   # cannot model the scale
    with_h = loo_table({"a": exact_shift, "d": make_pairs(TRUTH, winter=WINTER + [(60.0, 560.0)])})
    assert [r["model"] for r in with_h] == list(summer_gt.PAIR_MODELS)
    assert all(r["trees"] == 1 for r in with_h)                               # only "d" has 5 pairs


def test_cli_writes_summer_masks(sample_dir, tmp_path, capsys):
    pairs_path = tmp_path / "pairs.json"
    save_pairs(pairs_path, {"R01N01": make_pairs(TRUTH), "R01N02": SeasonPairs(skip=True)})
    out = tmp_path / "out"
    args = [str(sample_dir), "--pairs", str(pairs_path), "--out", str(out), "--winter-gt", "",
            "--clicks", str(CLICKS), "--overlays"]
    assert summer_gt.main(args) == 0
    text = capsys.readouterr().out
    assert "1 trees with pairs (4 pairs; 1 skipped" in text and "similarity" in text
    with open(out / "summary.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert row["tree_id"] == "R01N01" and row["ok"] == "True" and float(row["loo_max"]) < 0.3
    assert float(row["scale"]) == pytest.approx(1.05, abs=1e-3)
    # The saved summer mask is the traced winter mask moved by the fitted transform.
    from treeskel.align import load_winter_mask
    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset, load_tree

    rec = next(t for t in index_dataset(sample_dir).trees if t.tree_id == "R01N01")
    mask = load_winter_mask("R01N01", load_tree(rec, "winter"), load_clicks(CLICKS), None)
    expect, covered = warp_mask(mask, np.array(json.loads(row["matrix"])))
    saved = np.asarray(Image.open(out / "R01N01_summer_gt.png")) > 127
    assert saved.sum() > 1000 and (saved != expect).mean() < 1e-4
    assert np.array_equal(np.asarray(Image.open(out / "R01N01_summer_covered.png")) > 127, covered)
    assert (out / "R01N01_overlay.jpg").exists() and (out / "sheet_01.jpg").exists()
    assert not (out / "R01N02_summer_gt.png").exists()


def test_cli_without_pairs_says_so(sample_dir, tmp_path, capsys):
    assert summer_gt.main([str(sample_dir), "--pairs", str(tmp_path / "none.json"), "--out", str(tmp_path)]) == 1
    assert "No clicked pairs" in capsys.readouterr().out


def pitch_homography(deg, f=615.0):
    """How the image of a plane changes when the camera tilts (pitches) by ``deg``: K R K^-1."""
    k = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
    a = np.radians(deg)
    r = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    h = k @ r @ np.linalg.inv(k)
    return h / h[2, 2]


SPREAD = [(60.0, 80.0), (60.0, 560.0), (420.0, 80.0), (420.0, 560.0), (240.0, 320.0), (150.0, 200.0),
          (330.0, 450.0)]


def test_a_camera_pitch_needs_the_homography():
    """An 8 degree tilt between the seasons: the similarity is off by about 15 px, the homography exact."""
    p = make_pairs(pitch_homography(8.0), winter=SPREAD)
    assert np.median(loo_errors(p, "similarity")) > 10
    assert np.max(loo_errors(p, "homography")) < 1e-6
    assert any("from the fit of the others" in s for s in check_pairs(p, "similarity"))
    assert check_pairs(p, "homography") == []


def test_homography_needs_five_pairs_to_be_checked():
    p = make_pairs(pitch_homography(4.0), winter=SPREAD[:4])
    assert pair_transform(p, "homography") is not None                      # 4 pairs fix it exactly...
    assert np.isnan(loo_errors(p, "homography")).all()                        # ...but leave nothing to check
    assert check_pairs(p, "homography") == ["only 4 pair(s)"]


def test_a_homography_that_folds_the_image_is_rejected():
    fold = np.array([[1.0, 0, 0], [0, 1, 0], [-1 / 300, 0, 1]])              # infinite at x = 300
    p = make_pairs(fold, winter=[(50.0, 50.0), (400.0, 60.0), (60.0, 250.0), (420.0, 280.0)])
    assert pair_transform(p, "homography") is None
