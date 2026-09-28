"""Tests for continuing short leaders to pieces on their line and pruning to the trunk and two leaders
(treeskel.segment.continue_leaders, prune_to_branches, continued_run). No PyTorch needed."""

import csv
import json

import numpy as np
import pytest

from treeskel.segment import (
    CONTINUE_CONE,
    continue_leaders,
    continued_run,
    join_pieces,
    mask_shape,
    prune_to_branches,
)

from test_segment_join import XX, YY, fake_run, v_mask


def test_the_cone_is_how_far_the_hand_masks_leaders_turn():
    assert CONTINUE_CONE == 25.0                               # the 95th percentile of |bend|: 24.8 degrees


@pytest.mark.parametrize("cut", [(20, 60), (60, 120), (20, 45)])
def test_a_leader_cut_beyond_the_joining_gap_is_continued(cut):
    whole, _ = mask_shape(v_mask())
    m, _ = join_pieces(v_mask(cut=cut))
    out, n = continue_leaders(m)
    assert n == 1
    assert mask_shape(out)[0]["v_opening"] == pytest.approx(whole["v_opening"], abs=1.0)


def test_nothing_to_continue():
    m = v_mask()
    out, n = continue_leaders(m)                                # whole leaders: only pruned, which keeps the V
    assert n == 0 and (out & m).sum() / m.sum() > 0.98
    short = v_mask(cut=(100, 200))                              # cut to its end: no piece further on
    assert continue_leaders(short)[1] == 0
    empty, n = continue_leaders(np.zeros((256, 256), bool))
    assert n == 0 and not empty.any()


def test_pieces_off_the_line_or_too_small_are_not_joined():
    m, _ = join_pieces(v_mask(cut=(60, 200)))                  # the right leader stops 60 px up
    off = np.hypot(YY - 120, XX - 190) < 6                     # a blob well off its line
    assert continue_leaders(m | off)[1] == 0
    speck = np.zeros_like(m)                                    # a 2 px speck on its line: too little centre line
    fork, a = np.array([200.0, 128.0]), np.radians(20.0)
    r, c = (fork + 100 * np.array([-np.cos(a), np.sin(a)])).astype(int)
    speck[r, c:c + 2] = True
    assert continue_leaders(m | speck)[1] == 0


def test_pruning_keeps_the_trunk_and_leaders_only():
    branch = (np.abs(YY - 230) < 3) & (XX > 128) & (XX < 170)   # a side branch off the trunk
    m = v_mask() | branch
    p = prune_to_branches(m, mask_shape(m)[1])
    assert not (p & branch & (XX > 140)).any()
    assert (p & v_mask()).sum() / v_mask().sum() > 0.98 and not (p & ~m).any()


def test_a_saved_joined_run_is_scored_again_with_its_leaders_continued(tmp_path):
    trees = {"T0": v_mask(), "T1": join_pieces(v_mask(cut=(20, 60)))[0], "T2": join_pieces(v_mask(cut=(60, 120)))[0],
             "T3": v_mask(cut=(100, 200)), "T4": v_mask(15.0), "T5": v_mask(25.0)}
    src = tmp_path / "results" / "segment_lined_up_rerun_joined"
    fake_run(src, trees)
    report0 = json.loads((src / "report.json").read_text())
    (src / "report.json").write_text(json.dumps({**report0, "join": {"gap": 19.3}}))
    out, failed = continued_run(src, figures=tmp_path / "figures", progress=lambda m: None)
    assert failed == [] and out.name == "segment_lined_up_rerun_continued"
    report = json.loads((out / "report.json").read_text())
    c = report["continue"]
    assert c["cone"] == CONTINUE_CONE and c["masks_continued"] == 2 and c["pieces_joined"] == 2   # T1, T2
    assert c["v_opening_before"] == 5 and c["v_opening_after"] == 6          # T1 measured again
    assert report["join"] == {"gap": 19.3} and "segment_lined_up_rerun_joined" in report["against_runs"]
    rows = {(r["tree"], r["measure"]): r for r in csv.DictReader(open(out / "predictions.csv"))}
    assert float(rows[("T1", "v_opening")]["segmentation"]) == pytest.approx(40.0, abs=1.5)
    assert (tmp_path / "figures" / "segment_lined_up_rerun_continued_masks.png").exists()
