"""Tests for treeskel.final_test (random weights, CPU, small images).

The real test rows are never touched here: a split whose "test rows" are development rows 15 and 20 (R15N05
and R20N06 in data/sample) stands in for splits.json, and the development measures table has no test-row
tree, so the sample's test-row trees (rows 3 and 12) are neither trained on nor scored."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from treeskel.final_test import best_shift_dice, bootstrap_skill, main, named_recipe  # noqa: E402
from treeskel.segment import JOIN_GAP, SegParams, load_masks  # noqa: E402
from treeskel.shape import MEASURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def named_run(tmp_path, size=(64, 96), join=True, grow=None, cont=None):
    """A saved development run (its report only) and a model.json naming it."""
    run = tmp_path / "results" / "segment_lined_up_64x96_joined"
    run.mkdir(parents=True)
    params = {**SegParams().__dict__, "size": list(size), "batch": 4}
    report = {"params": params, "mask_overlay": str(ROOT / "annotations/mask_overlay.json"),
              "epochs": [{"fold": i, "epochs": e} for i, e in enumerate([2, 1, 1, 3, 1])],
              "scores": {k: {"segmentation": {"skill": 0.5}} for k in MEASURES}}
    if join:
        report["join"] = {"gap": JOIN_GAP}
    if grow is not None:
        report["grow"] = {"low": grow}
    if cont is not None:
        report["continue"] = {"cone": cont}
        report["complete"] = {"coef_all": [-2.8, 0.7]}
    (run / "report.json").write_text(json.dumps(report))
    out = tmp_path / "final"
    out.mkdir()
    completed = tmp_path / "results" / "segment_lined_up_completed"
    completed.mkdir()
    (completed / "report.json").write_text(json.dumps({"complete": {"coef_all": [-2.8, 0.7]}}))
    (out / "model.json").write_text(json.dumps({"run": str(run), "why": "named in a test",
                                                "secondary": {"complete_from": str(completed)}}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_rows": [15, 20]}))
    return out, split


def test_the_named_recipe_is_taken_as_it_stands(tmp_path):
    out, _ = named_run(tmp_path)
    r = named_recipe(out / "model.json")
    assert r["params"].size == (64, 96) and r["params"].batch == 4 and r["params"].encoder == "resnet18"
    assert r["epochs"] == 1 and r["fold_epochs"] == [2, 1, 1, 3, 1]      # the median of the folds' choices
    assert r["join_gap"] == JOIN_GAP and r["mask_overlay"].endswith("mask_overlay.json")
    out2, _ = named_run(tmp_path / "b", size=(256, 256), join=False)
    assert named_recipe(out2 / "model.json")["join_gap"] is None and named_recipe(out2 / "model.json")["grow_low"] is None
    out3, _ = named_run(tmp_path / "c", grow=0.2)
    assert named_recipe(out3 / "model.json")["grow_low"] == 0.2
    assert named_recipe(out3 / "model.json")["continue_cone"] is None
    assert named_recipe(out3 / "model.json")["complete_coef"] is None


def test_bootstrap_interval_of_the_skill():
    rng = np.random.default_rng(0)
    truth = rng.normal(50, 10, 200)
    base = np.full(200, 50.0)
    good = truth + rng.normal(0, 3, 200)                      # skill about 1 - 9/100
    lo, hi = bootstrap_skill(truth, good, base)
    assert lo < 0.91 < hi and hi - lo < 0.1
    assert np.isnan(bootstrap_skill(truth[:2], good[:2], base[:2])[0])


def test_the_best_shift_finds_a_moved_mask():
    m = np.zeros((256, 256), bool)
    m[100:200, 120:130] = True
    assert best_shift_dice(np.roll(m, 16, 1), m) == pytest.approx(1.0)


def test_the_final_evaluation_runs_once_on_the_local_trees(tmp_path, capsys):
    out, split = named_run(tmp_path, grow=0.2, cont=25.0)
    args = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning"), "--out", str(out), "--split", str(split),
            "--figures", str(tmp_path / "figures"), "--device", "cpu", "--random-weights", "--jobs", "2",
            "--examples", "2", "--table", str(ROOT / "data/shape/shape_measures.csv"),
            "--clicks", str(ROOT / "annotations/winter_clicks.json")]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "2 test trees" in text and "median of the folds' [2, 1, 1, 3, 1]" in text
    assert "grown into pixels above 0.2" in text and "short leaders continued within 25 degrees" in text
    assert "short leader lines completed" in text
    report = json.loads((out / "report.json").read_text())
    assert report["test_rows"] == [15, 20] and report["n_trees"] == 2
    assert report["n_dev"] == 19                              # the 21 local development trees minus rows 15 and 20
    assert report["epochs"] == 1 and report["seed"] == 1000 and report["join_gap"] == JOIN_GAP
    assert report["grow_low"] == 0.2
    assert report["secondary"]["complete_coef"] == [-2.8, 0.7]
    assert "model, leaders completed" in report["scores"]["v_opening"]
    assert (out / "predicted_masks_completed.npz").exists()
    assert set(report["scores"]["v_opening"]) == {"mean shape", "hand masks", "model", "model, forks found",
                                                  "development", "model, leaders completed"}
    assert report["scores"]["v_opening"]["development"] == 0.5
    assert set(load_masks(out / "predicted_masks.npz")) == {"R15N05", "R20N06"}
    rows = (out / "test_shape_measures.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[1].startswith("R15N05")
    for name in ("predictions.csv", "mask_scores.csv", "test_winter_traces.npz", "output.txt"):
        assert (out / name).exists()
    for name in ("final_test_skill.png", "final_test_examples.jpg"):
        assert (tmp_path / "figures" / name).exists()
    # Once only; and never without a named model.
    assert main(args) == 1 and "run once" in capsys.readouterr().out
    (out / "model.json").unlink()
    (out / "report.json").unlink()
    assert main(args) == 1 and "name the model first" in capsys.readouterr().out
