"""Tests for growing predicted masks into the U-Net's less sure pixels (treeskel.segment.grow_mask and its
choice by clDice against the hand masks). No PyTorch needed."""

import csv
import json

import numpy as np
import pytest

from treeskel.segment import (
    GROW_LOWS,
    JOIN_GAP,
    choose_low,
    cl_dice,
    grow_mask,
    grown_run,
    join_pieces,
    load_probabilities,
    mask_shape,
    save_masks,
    save_probabilities,
)
from treeskel.shape import MEASURES

from test_segment_join import YY, XX, fake_run, v_mask


def faint_v(cut=(20, 45), faint=0.3, noise=None):
    """Probabilities of a V whose right leader fades to ``faint`` from ``cut[0]`` to ``cut[1]`` px along it
    (the network unsure under leaves); ``noise=(value, row, col)`` adds a patch of that value beside it."""
    whole, broken = v_mask(), v_mask(cut=cut)
    p = np.where(broken, 0.9, 0.02).astype(np.float32)
    p[whole & ~broken] = faint
    if noise is not None:
        value, r, c = noise
        p[(np.hypot(YY - r, XX - c) < 12) & (p < value)] = value
    return p


def test_growing_keeps_the_faint_pixels_connected_to_the_mask_only():
    p = faint_v()
    p[(np.hypot(YY - 60, XX - 30) < 8)] = 0.3                   # a faint patch touching nothing
    seed = p > 0.5
    grown = grow_mask(p, 0.2, seed)
    assert np.array_equal(grown & seed, seed)                   # the mask is kept exactly
    assert grown[v_mask() & ~v_mask(cut=(20, 45))].all()        # the faint stretch of the leader is back
    assert not grown[(np.hypot(YY - 60, XX - 30) < 8)].any()   # the lone patch is not
    assert np.array_equal(grow_mask(p, 0.5, seed), seed)        # 0.5 grows nothing
    assert np.array_equal(grow_mask(p, 0.2), grown)             # the seed defaults to p > 0.5
    assert not grow_mask(np.zeros((8, 8)), 0.1).any()


def test_a_leader_broken_beyond_the_joining_gap_is_measured_again_after_growing():
    p = faint_v(cut=(20, 45))                                   # 25 px: too far for joining (19.3)
    joined, n = join_pieces(p > 0.5)
    assert n == 0
    whole, _ = mask_shape(v_mask())
    grown, _ = join_pieces(grow_mask(p, 0.2))
    assert mask_shape(grown)[0]["v_opening"] == pytest.approx(whole["v_opening"], abs=0.5)


def test_centre_line_dice():
    m = v_mask()
    assert cl_dice(m, m) == pytest.approx(1.0)
    short = cl_dice(v_mask(cut=(60, 200)), m)                   # the right leader stops 60 px up
    assert 0.6 < short < 0.95
    wide = m | ((np.abs(XX - 128) < 60) & (YY > 100) & (YY < 200))   # a blob of leaves across the V
    assert cl_dice(wide, m) < 0.95
    assert np.isnan(cl_dice(np.zeros_like(m), m))


def test_the_lower_threshold_is_chosen_on_the_hand_masks():
    # Leaders fade to 0.3 where leaves hide them; leaves beside the V read 0.1. Growing to 0.3 restores the
    # leaders; going below 0.1 also takes in the leaves. Chosen: the largest threshold under 0.3.
    probs, seeds, hands = {}, {}, {}
    for i, cut in enumerate([(20, 45), (40, 80), (100, 150)]):
        p = faint_v(cut=cut, noise=(0.1, 150, 60))
        p[(np.abs(XX - 90) < 10) & (YY > 120) & (YY < 200) & (p < 0.1)] = 0.1   # leaves touching the left leader
        probs[f"T{i}"], seeds[f"T{i}"], hands[f"T{i}"] = p, p > 0.5, v_mask()
    low, table = choose_low(probs, seeds, hands, lows=(0.5, 0.4, 0.25, 0.2, 0.05), progress=lambda m: None)
    assert low == 0.25                                         # 0.25 and 0.2 tie: the higher wins
    assert table["0.25"] > table["0.5"] and table["0.05"] < table["0.25"]
    assert GROW_LOWS[0] == 0.5 and list(GROW_LOWS) == sorted(GROW_LOWS, reverse=True)


def test_probabilities_are_saved_to_within_a_255th(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.random((3, 256, 256)).astype(np.float32)
    save_probabilities(tmp_path / "p.npz", ["A", "B", "C"], p)
    got = load_probabilities(tmp_path / "p.npz")
    assert list(got) == ["A", "B", "C"] and np.abs(got["B"] - p[1]).max() <= 0.5 / 255 + 1e-6


def test_a_saved_run_is_scored_again_with_its_masks_grown(tmp_path):
    probs = {"T0": faint_v(cut=(1000, 1001)), "T1": faint_v(cut=(20, 45)), "T2": faint_v(cut=(60, 90)),
             "T3": faint_v(cut=(5, 20)), "T4": faint_v(cut=(20, 45), faint=0.1), "T5": faint_v(cut=(30, 36))}
    src = tmp_path / "results" / "segment_lined_up_rerun"
    fake_run(src, {t: p > 0.5 for t, p in probs.items()})
    report0 = json.loads((src / "report.json").read_text())
    (src / "report.json").write_text(json.dumps({**report0, "params": {"threshold": 0.5}}))
    save_probabilities(src / "probabilities.npz", list(probs), np.stack(list(probs.values())))
    out, failed = grown_run(src, 0.2, {"0.5": 0.9, "0.2": 0.95}, figures=tmp_path / "figures", progress=lambda m: None)
    assert failed == [] and out.name == "segment_lined_up_rerun_grown"
    report = json.loads((out / "report.json").read_text())
    g = report["grow"]
    assert g["low"] == 0.2 and g["cldice"] == {"0.5": 0.9, "0.2": 0.95} and g["masks_grown"] == 4   # T1, T2, T3, T5
    assert g["v_opening_joined_only"] == 4 and g["v_opening_grown"] == 5   # T1 regrown; T4 (0.1) stays short
    assert report["join"]["gap"] == JOIN_GAP and report["masks_note"].endswith("before the change")
    rows = {(r["tree"], r["measure"]): r for r in csv.DictReader(open(out / "predictions.csv"))}
    assert float(rows[("T1", "v_opening")]["segmentation"]) == pytest.approx(40.0, abs=1.0)
    assert rows[("T4", "v_opening")]["segmentation"] == ""
    assert "segment_lined_up_rerun" in report["against_runs"]
    assert (tmp_path / "figures" / "segment_lined_up_rerun_grown_skill.png").exists()
    assert (tmp_path / "figures" / "segment_lined_up_rerun_grown_masks.png").exists()
    # With hand masks, the pixel scores are the grown masks' own.
    out2, _ = grown_run(src, 0.2, hands={t: v_mask() for t in probs}, out=tmp_path / "g2", progress=lambda m: None)
    rep2 = json.loads((out2 / "report.json").read_text())
    assert rep2["masks"]["dice"]["n"] == 6 and rep2["masks_note"].endswith("against the hand masks")
