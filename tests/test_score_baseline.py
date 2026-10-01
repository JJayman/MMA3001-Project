"""Tests for treeskel.score_baseline, on synthetic trees with a known answer and the committed results."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_guided_gt import bark_rgb, good_clicks, synthetic_rgbd
from test_winter_gt import synthetic_tree
from treeskel.guided_gt import guided_gt
from treeskel.io import index_dataset
from treeskel.score_baseline import (
    PREDICTIONS,
    SESSION4_ROWS,
    SESSION4_WRONG,
    TRACED,
    baseline_shape,
    example_trees,
    issue_type,
    load_rows,
    main,
    path_scores,
    quality_check,
    read_predictions,
    save_rows,
    score_tree,
    session4_check,
    shape_scores,
    summarise,
)
from treeskel.shape import MEASURES, gt_shape
from treeskel.winter_gt import GTParams, WinterGT, extract_winter_gt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tree():
    """A synthetic V tree: its images, the guided result from good clicks, and the baseline's."""
    rgb, depth, valid = synthetic_rgbd()
    return {"guided": guided_gt(rgb, depth, valid, good_clicks()),
            "baseline": extract_winter_gt(depth, valid), "depth": depth, "valid": valid}


def shifted(gt: WinterGT, dx: int) -> WinterGT:
    """The same paths moved ``dx`` columns (a baseline beside the branch)."""
    move = np.array([0, dx])
    return replace(gt, trunk=gt.trunk + move, left=gt.left + move, right=gt.right + move,
                   root=(gt.root[0], gt.root[1] + dx), fork=(gt.fork[0], gt.fork[1] + dx))


def test_the_guided_trace_scores_perfectly_against_itself(tree):
    g = tree["guided"]
    s = path_scores(g, g)
    assert s["found"] == s["on_branch"] == s["f1"] == 1.0
    assert all(s[f"found_{p}"] == 1.0 and s[f"on_branch_{p}"] == 1.0 for p in ("trunk", "left", "right"))
    assert s["fork_px"] == s["root_px"] == s["left_tip_px"] == s["right_tip_px"] == 0.0
    assert s["to_guided_px"] == s["to_baseline_px"] == 0.0


def test_the_baseline_traces_a_clean_synthetic_tree(tree):
    b, g = tree["baseline"], tree["guided"]
    assert b.ok
    s = path_scores(b, g)
    assert s["f1"] >= TRACED[0]
    assert s["fork_px"] < 15 and s["left_tip_px"] < 15 and s["right_tip_px"] < 15
    # Its V is measured like the guided one's (the synthetic leaders are straight).
    assert abs(baseline_shape(b)["v_opening"] - gt_shape(g)["v_opening"]) < 2.0


def test_a_baseline_beside_the_branch_is_neither_found_nor_on_it(tree):
    g = tree["guided"]
    # The synthetic leaders are drawn about 5 px wide (radius below 5), the trunk about 7: 12 px beside
    # them is off every branch, but within the 20 px tolerance.
    s = path_scores(shifted(g, 12), g)
    # Only near the fork, where the leaders are close together, does a moved leader touch the other one.
    assert s["found_left"] < 0.1 and s["found_right"] < 0.1
    assert s["on_branch_left"] < 0.1 and s["on_branch_right"] < 0.1
    assert s["f1"] < 0.2
    assert s["found_10px"] < 0.1 and s["found_20px"] > 0.95
    assert s["fork_px"] == pytest.approx(12.0)


def test_a_leader_that_stops_half_way_is_half_found(tree):
    g = tree["guided"]
    half = replace(g, left=g.left[: len(g.left) // 2])
    s = path_scores(half, g)
    assert s["found_left"] == pytest.approx(0.5, abs=0.05)
    assert s["found_right"] == s["found_trunk"] == 1.0
    assert s["on_branch"] == 1.0                     # what it did draw is right
    assert s["left_tip_px"] > 100


def test_no_tree_found_scores_zero(tree):
    g = tree["guided"]
    empty = extract_winter_gt(np.zeros_like(tree["depth"]), np.zeros_like(tree["valid"]))
    s = path_scores(empty, g)
    assert s["found"] == 0.0 and s["f1"] == 0.0
    assert np.isnan(s["on_branch"]) and np.isnan(s["fork_px"]) and np.isnan(s["left_tip_px"])
    assert np.isnan(baseline_shape(empty)["v_opening"])
    row = score_tree(empty, g)
    assert row["no_fork"] and not row["ok"] and row["f1"] == 0.0


def test_a_post_taken_for_a_leader_is_off_the_branch():
    """With its post rule off, the baseline follows a post's top beside the right leader (as in
    test_winter_gt): the right leader's points leave the branch."""
    depth, _ = synthetic_tree()
    depth[:230, 372:414] = 150
    valid = depth > 0
    g = guided_gt(bark_rgb(depth == 150), depth, valid, good_clicks())
    b = extract_winter_gt(depth, valid, GTParams(max_leader_radius=1e9))
    s = path_scores(b, g)
    # It climbs the leader to the post's lower end (row 230), about a third of the way, then the post.
    assert s["right_tip_px"] > 50
    assert s["on_branch_right"] < 0.75 and s["found_right"] < 0.75
    assert s["found_left"] == s["found_trunk"] == 1.0
    assert s["f1"] < TRACED[0]
    # With the rule on, the leader is found.
    assert path_scores(extract_winter_gt(depth, valid), g)["f1"] >= TRACED[0]


def test_score_tree_columns(tree):
    row = score_tree(tree["baseline"], tree["guided"])
    assert row["ok"] and not row["no_fork"] and row["issues"] == ""
    assert 0 < row["mask_iou"] <= row["mask_dice"] <= 1
    assert all(f"baseline_{k}" in row and f"guided_{k}" in row for k in MEASURES)


def test_issue_types_drop_the_numbers():
    assert issue_type("right leader rises 3 px (< 140)") == "right leader rises too little"
    assert issue_type("no tree region found") == "no tree region found"


def _row(tree, f1, ok, row=None, fork=5.0, issues=""):
    return {"tree": tree, "row": row if row is not None else int(tree[1:3]), "f1": f1, "ok": ok,
            "no_fork": False, "issues": issues, "fork_px": fork}


def test_quality_check_counts():
    rows = [_row("R01N01", 0.95, True), _row("R01N02", 0.5, True), _row("R01N03", 0.85, True),
            _row("R01N04", 0.92, False, issues="left leader rises 3 px (< 140)"),
            _row("R01N05", 0.1, False, issues="left leader rises 3 px (< 140); right leader rises 9 px (< 140)")]
    q = quality_check(rows)
    assert (q["passed"], q["flagged"]) == (3, 2)
    assert q["at_0.9"]["passed_not_traced"] == 2 and q["at_0.9"]["flagged_traced"] == 1
    assert q["at_0.8"]["passed_not_traced"] == 1
    assert q["issues"] == {"left leader rises too little": 2, "right leader rises too little": 1}


def test_session4_list_is_in_the_inspected_development_rows():
    split = json.loads((ROOT / "splits.json").read_text())
    rows = {int(t[1:3]) for t in SESSION4_WRONG}
    assert rows <= set(SESSION4_ROWS) and not rows & set(split["test_rows"])
    assert len(SESSION4_WRONG) == 38


def test_session4_check_splits_the_inspected_rows():
    rows = [_row("R01N01", 0.5, True, fork=80.0), _row("R01N05", 0.0, False), _row("R01N02", 0.95, True),
            _row("R07N12", 0.91, True), _row("R02N01", 0.2, True)]          # row 2 was not inspected
    s = session4_check(rows)
    assert s["listed"]["n"] == 2 and s["others"]["n"] == 2
    assert s["listed"]["traced"] == 0.0 and s["others"]["traced"] == 1.0
    assert s["by_type"]["wrong fork"]["fork_px_median"] == 80.0


def test_example_trees_cover_passed_and_flagged():
    rows = [_row(f"R01N{i:02d}", i / 20, i % 3 != 0) for i in range(1, 21)]
    picks = example_trees(rows, 4)
    groups = [g for _, g in picks]
    assert groups.count("passed QC, lowest F1") == 4 and groups.count("flagged") == 4
    trees = [t for t, _ in picks]
    assert len(trees) == len(set(trees))
    worst = sorted((r["f1"], r["tree"]) for r in rows if r["ok"])[:4]
    assert trees[:4] == [t for _, t in worst]


def test_rows_round_trip(tmp_path, tree):
    row = {"tree": "R01N01", "row": 1, **score_tree(tree["baseline"], tree["guided"]), "baseline_s": 0.2,
           "guided_s": 1.3}
    save_rows(tmp_path / "trees.csv", [row])
    back = load_rows(tmp_path / "trees.csv")[0]
    assert back["ok"] is True and back["no_fork"] is False and back["row"] == 1
    assert back["f1"] == pytest.approx(row["f1"], abs=1e-4)


def _rows_from_predictions(change=None):
    """One row per development tree whose baseline equals the winter measures (rounded as saved)."""
    pred = read_predictions(ROOT / PREDICTIONS)
    rows = []
    for t, m in pred.items():
        r = {"tree": t, "row": int(t[1:3]), "ok": True, "no_fork": False, "issues": ""}
        for k in MEASURES:
            r[f"baseline_{k}"] = r[f"guided_{k}"] = m[k]["winter"]
        rows.append(change(r) if change else r)
    return rows


def test_shape_scores_reproduce_the_named_runs_skill():
    """On the committed predictions: a baseline equal to winter scores 1, and the U-Net's column gives
    the skill its run reported."""
    sh = shape_scores(_rows_from_predictions(), ROOT / PREDICTIONS, ROOT / "data/shape/shape_measures.csv")
    run = json.loads((ROOT / PREDICTIONS.parent / "report.json").read_text())["scores"]
    for k in MEASURES:
        assert sh["scores"][k]["winter baseline"]["skill"] == pytest.approx(1.0)
        # The saved predictions are rounded to 2 decimals; the run scored them unrounded.
        assert sh["scores"][k]["U-Net (named)"]["skill"] == pytest.approx(run[k]["segmentation"]["skill"], abs=2e-4)
        assert sh["scores"][k]["U-Net (named)"]["n"] == run[k]["segmentation"]["n"]
        assert sh["scores"][k]["hand masks"]["skill"] == pytest.approx(run[k]["hand masks"]["skill"], abs=2e-4)
        assert sh["scores"][k]["mean shape"]["skill"] == 0.0
    assert sh["guided_check"]["max_abs_diff"] == 0.0
    assert sh["against_unet"]["v_opening"]["baseline_closer"] > 0.5


def test_shape_scores_count_no_fork_and_gaps_as_the_mean_shape():
    def no_fork_or_gap(r):
        if r["tree"].endswith("1"):                     # no fork: every measure missing
            for k in MEASURES:
                r[f"baseline_{k}"] = float("nan")
            r["ok"] = False
        elif r["tree"].endswith("2"):                   # a short left leader: its measures missing
            for k in ("v_opening", "left_angle", "lean", "left_bend", "left_tilt", "lean_v"):
                r[f"baseline_{k}"] = float("nan")
        return r

    rows = _rows_from_predictions(no_fork_or_gap)
    sh = shape_scores(rows, ROOT / PREDICTIONS, ROOT / "data/shape/shape_measures.csv")
    v = sh["scores"]["v_opening"]
    n_all = v["mean shape"]["n"]
    n_gap = sum(r["tree"].endswith("2") for r in rows)
    assert sh["baseline_no_fork"] == sum(r["tree"].endswith("1") for r in rows)
    assert v["winter baseline"]["n"] == n_all - n_gap           # no fork filled, gaps left out
    assert v["baseline, gaps as mean"]["n"] == n_all            # both filled
    assert v["baseline, gaps as mean"]["skill"] < v["winter baseline"]["skill"] < 1.0
    assert v["winter baseline, passed QC"]["skill"] == pytest.approx(1.0)   # passed trees only, gaps left out


def test_cli_on_the_sample_leaves_out_the_test_rows(sample_dir, tmp_path, capsys):
    out = tmp_path / "res"
    code = main([str(sample_dir), "--out", str(out), "--figure", str(tmp_path / "fig.jpg"), "--jobs", "1",
                 "--clicks", str(ROOT / "annotations/winter_clicks.json"), "--split", str(ROOT / "splits.json"),
                 "--predictions", str(ROOT / PREDICTIONS), "--table", str(ROOT / "data/shape/shape_measures.csv")])
    assert code == 0
    rows = load_rows(out / "trees.csv")
    split = json.loads((ROOT / "splits.json").read_text())
    assert {r["tree"] for r in rows} == {"R01N01", "R01N02", "R15N05", "R20N06"}   # R04 excluded, test rows left out
    assert not {r["row"] for r in rows} & set(split["test_rows"])
    assert (tmp_path / "fig.jpg").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["n_trees"] == 4 and "shape" in report
    text = capsys.readouterr().out
    assert "test-row trees left out" in text and "Shape: skill over the mean shape" in text
    # Re-scored from the saved rows, the report is the same.
    assert main(["--from-saved", str(out), "--out", str(tmp_path / "again"),
                 "--predictions", str(ROOT / PREDICTIONS), "--table", str(ROOT / "data/shape/shape_measures.csv")]) == 0
    assert json.loads((tmp_path / "again" / "report.json").read_text()) == report


def test_a_partial_run_does_not_overwrite_the_saved_results(sample_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main([str(sample_dir), "--trees", "R15N05", "--figure", "", "--jobs", "1",
                 "--clicks", str(ROOT / "annotations/winter_clicks.json"), "--split", str(ROOT / "splits.json"),
                 "--predictions", str(ROOT / PREDICTIONS), "--table", str(ROOT / "data/shape/shape_measures.csv")])
    assert code == 0
    assert (tmp_path / "outputs/score_baseline_partial/trees.csv").exists()
    assert not (tmp_path / "docs").exists()


def test_summarise_without_predictions_has_no_shape(tree):
    row = {"tree": "R01N01", "row": 1, **score_tree(tree["baseline"], tree["guided"]), "baseline_s": 0.2,
           "guided_s": 1.3}
    report = summarise([row], predictions=None)
    assert "shape" not in report and report["paths"]["all"]["n_trees"] == 1
