"""Tests for treeskel.segment.join_pieces: joining a broken predicted mask before it is measured.

No PyTorch needed (the rest of treeskel.segment's tests are in test_segment.py)."""

import numpy as np
import pytest

from treeskel.segment import JOIN_GAP, join_pieces, mask_shape

YY, XX = np.mgrid[:256, :256]


def bar(width=9):
    """A vertical bar ``width`` px wide (odd), rows 21-239."""
    return (np.abs(XX - 128) <= width // 2) & (YY > 20) & (YY < 240)


def v_mask(half_opening=20.0, cut=None):
    """A 256x256 V (as in test_segment.py): a trunk 9 px wide and two leaders 7 px wide, ``half_opening``
    degrees either side of the vertical; ``cut=(a, b)`` removes the right leader from ``a`` to ``b`` px
    along it from the fork."""
    fork = np.array([200.0, 128.0])
    m = (np.abs(XX - fork[1]) < 5) & (YY >= fork[0])
    for sign in (-1, 1):
        a = np.radians(sign * half_opening)
        d = np.array([-np.cos(a), np.sin(a)])
        rel = np.stack([YY - fork[0], XX - fork[1]], -1)
        along, across = rel @ d, np.abs(rel @ np.array([-d[1], d[0]]))
        leader = (along > 0) & (along < 190) & (across < 3.5)
        if cut is not None and sign > 0:
            leader &= ~((along > cut[0]) & (along < cut[1]))
        m |= leader
    return m


def test_the_gap_is_about_a_trunks_width():
    # Twice the hand masks' median trunk radius (9.67 mask px): 36 px of the photo.
    assert JOIN_GAP == pytest.approx(2 * 9.67, abs=0.05)
    assert JOIN_GAP * 480 / 256 == pytest.approx(36, abs=0.5)


def test_a_mask_in_one_piece_is_unchanged():
    m = v_mask()
    out, n = join_pieces(m)
    assert n == 0 and np.array_equal(out, m) and out is not m
    empty, n = join_pieces(np.zeros((256, 256), bool))
    assert n == 0 and not empty.any()


def test_a_cut_bar_is_joined_at_its_own_width():
    whole = bar()
    cut = whole & ~((YY >= 100) & (YY < 106))                # 6 rows of gap
    out, n = join_pieces(cut)
    assert n == 1
    assert (out & ~cut).sum() > 0 and np.array_equal(out | cut, out)          # pixels only added
    gap_rows = out[100:106]
    assert np.all(gap_rows.sum(axis=1) >= 9) and np.all(gap_rows.sum(axis=1) <= 11)   # 9 px wide, +-1
    assert (out ^ whole).sum() <= 20                          # the bar again, but for the band's slight slant


@pytest.mark.parametrize("cut", [(20, 26), (20, 35), (60, 66), (5, 20)])
def test_a_leader_cut_from_the_fork_is_measured_again(cut):
    whole, _ = mask_shape(v_mask())
    m = v_mask(cut=cut)
    before, _ = mask_shape(m)
    out, n = join_pieces(m)
    after, br = mask_shape(out)
    assert n == 1 and br is not None
    if cut[0] < 40:                                           # the fork was lost with the leader
        assert np.isnan(before["v_opening"])
    assert after["v_opening"] == pytest.approx(whole["v_opening"], abs=0.5)   # 40 degrees, as the whole V


def test_pieces_beyond_the_gap_stay_apart():
    m = v_mask(cut=(20, 45))                                  # a 25 px cut along the leader: gap > 19.3
    out, n = join_pieces(m)
    assert n == 0 and np.array_equal(out, m)
    assert join_pieces(m, gap=30.0)[1] == 1                   # a larger limit would join it
    assert join_pieces(v_mask(cut=(20, 26)), gap=0)[1] == 0   # 0 joins nothing


def test_a_piece_can_come_within_reach_through_another():
    m = bar() & ~((YY >= 100) & (YY < 115)) & ~((YY >= 60) & (YY < 75))
    # Three pieces: rows 21-59, 75-99 and 115-239 (the largest). The top one is 56 px from the largest,
    # but 16 px from the middle one, so it is joined once the middle one is.
    out, n = join_pieces(m)
    assert n == 2
    assert out[21:240, 128].all()
    assert join_pieces(m & (YY >= 75), gap=JOIN_GAP)[1] == 1   # without the top piece: one join


def test_a_piece_is_joined_to_the_nearest_centre_line():
    m = bar()
    blob = (np.hypot(YY - 120, XX - 150) < 4)                 # a small blob 13 px right of the bar's edge
    out, n = join_pieces(m | blob)
    assert n == 1
    band = out & ~(m | blob)
    rows = np.nonzero(band.any(axis=1))[0]
    assert rows.min() >= 115 and rows.max() <= 125            # a short horizontal band at the blob's height


# --- scoring a saved run again with its masks joined -----------------------------------------

def fake_run(folder, trees):
    """A saved run of broken and whole V masks, with winter measures of 40 degrees' opening for each."""
    import csv
    import json

    from treeskel.segment import save_masks
    from treeskel.shape import MEASURES

    folder.mkdir(parents=True)
    whole, _ = mask_shape(v_mask())
    save_masks(folder / "predicted_masks.npz", list(trees), np.stack(list(trees.values())))
    with open(folder / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", "fold", "measure", "winter", "mean shape", "hand masks", "segmentation"])
        for i, t in enumerate(trees):
            m, _ = mask_shape(trees[t])
            for k in MEASURES:
                truth = whole[k] + (i - 2)                    # the winter truth varies a little by tree
                seg = m[k] if np.isfinite(m[k]) else 0.0
                w.writerow([t, 1 + i % 3, i % 3, k, f"{truth:.2f}", f"{whole[k] + 5:.2f}", f"{truth:.2f}",
                            f"{seg:.2f}"])
    with open(folder / "mask_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", "fold", "fork_found", "iou", "dice", "precision", "recall"])
        for i, t in enumerate(trees):
            w.writerow([t, 1 + i % 3, i % 3, 1, 0.5, 0.6, 0.7, 0.8])
    (folder / "report.json").write_text(json.dumps({"variant": folder.name, "n_masks": len(trees),
                                                    "masks": {"dice": {"median": 0.6}}, "epochs": [], "scores": {}}))


def test_a_saved_run_is_scored_again_with_its_masks_joined(tmp_path):
    import json

    from treeskel.segment import joined_run, load_masks, main

    trees = {"T0": v_mask(), "T1": v_mask(cut=(20, 26)), "T2": v_mask(15.0), "T3": v_mask(cut=(5, 20)),
             "T4": v_mask(cut=(20, 60)), "T5": v_mask(25.0, cut=(60, 66))}
    src = tmp_path / "results" / "segment_lined_up"
    fake_run(src, trees)
    out, failed = joined_run(src, figures=tmp_path / "figures", progress=lambda m: None)
    assert failed == [] and out == tmp_path / "results" / "segment_lined_up_joined"
    report = json.loads((out / "report.json").read_text())
    j = report["join"]
    assert j["masks_joined"] == 3 and j["pieces_joined"] == 3            # T1, T3, T5; T4's 40 px cut is too wide
    # T3 lost its fork with the cut leader; T1 and T4 kept theirs, with a stub too short for a leader line.
    assert j["no_fork_before"] == 1 and j["no_fork_after"] == 0
    assert j["forks_found_by_joining"] == ["T3"] and j["forks_lost_by_joining"] == []
    assert j["v_opening_before"] == 3 and j["v_opening_after"] == 5      # T1 and T3 measured again; T4 not
    assert report["no_fork"] == 0 and report["variant"] == "segment_lined_up_joined"
    assert report["masks_note"] and report["masks"] == {"dice": {"median": 0.6}}
    v = report["scores"]["v_opening"]
    assert v["segmentation"]["n"] == 5 and v["segmentation, forks found"]["n"] == 5   # T4's opening is unknown
    # The source run is compared tree by tree; the joined masks are saved; the figures drawn.
    assert "segment_lined_up" in report["against_runs"]
    assert np.array_equal(load_masks(out / "predicted_masks.npz")["T0"], trees["T0"])
    assert (tmp_path / "figures" / "segment_lined_up_joined_skill.png").exists()
    assert (tmp_path / "figures" / "segment_lined_up_joined_masks.png").exists()
    # Masks not joined keep their measures; T1's opening is the whole V's again.
    import csv

    rows = {(r["tree"], r["measure"]): r for r in csv.DictReader(open(out / "predictions.csv"))}
    old = {(r["tree"], r["measure"]): r for r in csv.DictReader(open(src / "predictions.csv"))}
    assert rows[("T0", "v_opening")]["segmentation"] == old[("T0", "v_opening")]["segmentation"]
    assert float(rows[("T1", "v_opening")]["segmentation"]) == pytest.approx(40.0, abs=1.0)
    assert rows[("T4", "v_opening")]["segmentation"] == ""                # a stub leader: not measured
    # The command line does the same, and another gap gets its own folder.
    assert main(["--join", str(src), "--join-gap", "45", "--figures", str(tmp_path / "f2")]) == 0
    wide = json.loads((tmp_path / "results" / "segment_lined_up_joined_45" / "report.json").read_text())
    assert wide["join"]["v_opening_after"] == 6 and wide["join"]["gap"] == 45.0
