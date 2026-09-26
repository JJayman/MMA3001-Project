"""Tests for treeskel.measure_width."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from test_guided_gt import good_clicks, synthetic_rgbd
from test_winter_gt import FORK, thick_line
from treeskel.guided_gt import GuidedParams, guided_gt, load_clicks
from treeskel.measure_width import (
    LENGTHS,
    OLD,
    WIDENINGS,
    centre_runs,
    choose_shape,
    crotch_zone,
    edge_overshoot,
    free_taper,
    hand_mask_profiles,
    main,
    widening_costs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_centre_runs():
    strip = np.zeros((4, 11), bool)
    strip[0, 3:8] = True                  # closed run from -2 to +2
    strip[1, 0:7] = True                  # reaches the left end: not closed
    strip[2, 6:9] = True                  # misses the centre
    strip[3, 2:10] = True                 # closed, -3 to +4
    first, last, closed = centre_runs(strip)
    assert closed.tolist() == [True, False, False, True]
    assert (first[0], last[0]) == (-2, 2) and (first[3], last[3]) == (-3, 4)


def test_edge_overshoot_measures_how_far_depth_reaches_past_clear_edges():
    n, half = 30, 20
    x = np.arange(-half, half + 1)
    grey = np.where(np.abs(x) <= 5, 200.0, 100.0)[None].repeat(n, 0)   # bark -5..5 on grass
    first, last = np.full(n, -5), np.full(n, 5)
    d_first, d_last = np.full(n, -7), np.full(n, 8)                   # depth 2 and 3 px past the edges
    closed = np.ones(n, bool)
    closed[:10] = False                                                # merged depth: not used
    over = edge_overshoot(grey, first, last, d_first, d_last, closed)
    assert len(over) == 40 and sorted(set(over)) == [2.0, 3.0]
    faint = np.where(np.abs(x) <= 5, 130.0, 100.0)[None].repeat(n, 0)  # a 30-level change is not a clear edge
    assert len(edge_overshoot(faint, first, last, d_first, d_last, closed)) == 0
    far = np.full(n, -15)                                              # 10 px away: merged, left out
    assert len(edge_overshoot(grey, first, last, far, d_last, closed)) == 20


def y_mask(trunk_half=8, leader_half=4):
    """256x256 hand-drawn-style mask: a vertical trunk and two straight leaders."""
    m = np.zeros((256, 256), np.uint8)
    thick_line(m, (255, 128), (180, 128), 2 * trunk_half + 1, 1)
    for tip in ((10, 40), (10, 216)):
        thick_line(m, (180, 128), tip, 2 * leader_half + 1, 1)
    return m.astype(bool)


def test_hand_mask_profiles_give_leader_width_relative_to_the_trunk():
    prof = hand_mask_profiles(y_mask())
    for side in ("left", "right"):
        s, w = prof[side]
        assert np.all(np.diff(s) > 0) and s[0] == 0
        far = (s > 4) & (s < s[-1] - 2)
        # Horizontal runs of 9 px on leaders leaning 25°: 8.2 px across, half-width 4.1; trunk 8.5.
        assert np.median(w[far]) == pytest.approx(4.1 / 8.5, abs=0.08)
        assert w[0] > np.median(w[far])                               # wider where the leaders join
    assert hand_mask_profiles(np.zeros((256, 256), bool)) is None


def test_widening_costs_recover_a_known_shape():
    s = np.linspace(0, 30, 400)                                        # in trunk radii
    at_start = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        widths = 6.0 * (1 + 0.35 * np.exp(-s / 4.0)) + rng.normal(0, 0.4, len(s))
        # Lost shaded halves, scattered as on the development trees (7-12% of points
        # at every distance). A solid block of them inside the widening would bias the shape.
        widths[rng.random(len(s)) < 0.1] = 3.0
        costs = widening_costs(s, widths)
        assert costs.shape == (len(WIDENINGS), len(LENGTHS))
        i, j = np.unravel_index(np.argmin(costs), costs.shape)
        assert costs[i, j] < costs[0, 0]                               # beats a constant width
        at_start.append(1 + WIDENINGS[i] * np.exp(-1.5 / LENGTHS[j]))
    # The widening and its length trade off (more widening fading faster draws
    # nearly the same curve beyond 1.5 trunk radii), so compare the curve there.
    # One noisy leader can be off by about 0.04; the average should not be.
    assert np.mean(at_start) == pytest.approx(1 + 0.35 * np.exp(-1.5 / 4.0), abs=0.03)
    assert np.isnan(widening_costs(s[:5], widths[:5])).all()           # too few points


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


def test_choose_shape_holds_out_each_group():
    costs = np.ones((4, len(WIDENINGS), len(LENGTHS)))                  # constant width costs 1 everywhere
    costs[:2, 3, 4] = 0.5                                              # group "a" is fitted best by (w3, L4)
    costs[2:, 6, 8] = 0.2                                              # group "b" by (w6, L8),
    costs[:2, 6, 8] = 0.8                                              # which suits "a" a little too
    chosen = choose_shape(costs, np.array(["a", "a", "b", "b"]))
    assert chosen["groups"] == ["a", "b"]
    # "a" held out: "b" chooses (w6, L8), which costs "a" 0.8, less than a constant width.
    assert chosen["picks"][0] == (WIDENINGS[6], LENGTHS[8]) and chosen["held_out"][0] == pytest.approx(0.8)
    # "b" held out: "a" chooses (w3, L4), which does nothing for "b".
    assert chosen["picks"][1] == (WIDENINGS[3], LENGTHS[4]) and chosen["held_out"][1] == pytest.approx(1.0)
    assert chosen["held_out_constant"] == [1.0, 1.0]
    assert choose_shape(costs[:2], np.array(["a", "a"]))["groups"] == []     # nothing left to choose with


def test_old_settings_differ_only_in_what_session_8_changed():
    # "Before" must stay the session-7 masks when today's defaults change: only the
    # leader widths, the leader fit and the shoulder fill differ.
    changed = {k for k, v in asdict(OLD).items() if v != asdict(GuidedParams())[k]}
    assert changed == {"leader_width_source", "leader_fit", "shoulder_length"}
    assert (OLD.leader_width_source, OLD.leader_fit, OLD.shoulder_length) == ("scale_space", "taper", 1.4)


def test_free_taper():
    s = np.linspace(0, 300, 301)
    assert free_taper(s, np.linspace(8, 6, 301), 10.0) == pytest.approx(-0.25, abs=0.01)
    assert free_taper(s, np.linspace(6, 8, 301), 10.0) > 0             # a leader "widening upwards"
    assert np.isnan(free_taper(s[:5], np.full(5, 6.0), 10.0))


def test_cli_measures_development_rows_only(sample_dir, tmp_path, capsys):
    clicks = tmp_path / "clicks.json"
    shutil.copy(ROOT / "annotations" / "winter_clicks.json", clicks)
    out, figure = tmp_path / "out", tmp_path / "fig.jpg"
    assert main([str(sample_dir), "--clicks", str(clicks), "--split", str(ROOT / "splits.json"), "--out", str(out),
                 "--jobs", "2", "--figure", str(figure), "--figure-trees", "R01N01"]) == 0
    text = capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    dev = sorted(t for t in load_clicks(clicks) if t in ("R01N01", "R01N02", "R15N05", "R20N06"))
    assert report["trees"] == dev and report["test_rows_excluded"] == [3, 12, 16]
    assert "4 test-row trees left out" in text and "Shared widening" in text
    assert report["n_leaders"] == 2 * len(dev)
    far = report["far_from_fork"]["median_half_width"]
    assert far["new"] > 1.5 * far["old"]                               # the leaders are drawn wider now
    assert report["overshoot"]["n_edges"] > 0 and report["profile"]["n_hand_leaders"] == 2 * len(dev)
    wd = report["widening"]
    assert np.array(wd["grid"]["cost"]).shape == (len(WIDENINGS), len(LENGTHS))
    assert wd["default"]["cost"] is not None and wd["cost_best"] <= wd["default"]["cost"]
    # Rows 1 and 15 are in the first fold of splits.json, row 20 in the second.
    assert wd["row_folds"]["n_groups"] == 2 and wd["leave_one_out"]["n_groups"] == len(dev)
    assert "Row-grouped folds" in text
    crotch = report["crotch_off_depth"]
    # Wider leaders reach further into the crotch.
    assert crotch["before"][1] <= crotch["after_no_widening"][1] <= crotch["after"][1]
    assert report["off_centre"]["n_cross_sections"] > 0 and "Traced path" in text
    assert (out / "leaders.csv").read_text().count("\n") == 1 + 2 * len(dev)
    assert figure.exists()
