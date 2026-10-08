"""Tests for treeskel.benchmark (the network parts need PyTorch and are skipped without it)."""

import json
from pathlib import Path

import numpy as np
import pytest

from treeskel.benchmark import (
    estimated_minutes,
    final_training,
    ground_truths,
    mask_post_processing,
    peak_python_mb,
    recorded_training,
    report_lines,
    timed,
)
from treeskel.guided_gt import load_clicks
from treeskel.io import index_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_timed_reports_the_median_of_the_repeats():
    calls = []
    t = timed(lambda: calls.append(1), repeats=4, warmup=2)
    assert len(calls) == 6 and t["repeats"] == 4 and 0 <= t["min_s"] <= t["median_s"]


def test_peak_memory_sees_numpy_arrays_and_nothing_else():
    def both():
        a = np.zeros((2, 1_000_000))                   # 16 MB
        b = a + 1                                       # 16 MB more (a is still held, so not added in place)
        return a, b

    assert 31.5 < peak_python_mb(both) < 34
    assert peak_python_mb(lambda: None) < 0.1


def test_estimated_minutes():
    assert estimated_minutes(0.06, 8, 34) == pytest.approx(0.06 * 57 * 34 / 60, abs=1e-3)
    assert estimated_minutes(None, 8, 34) is None and estimated_minutes(0.1, None, 34) is None


def test_recorded_training_reads_the_runs():
    rec = recorded_training("U-Net 256x256 (named)", ROOT / "docs/results")
    assert rec["run"] == "segment_lined_up" and rec["minutes_5_folds"] == 20.4 and rec["device"] == "cuda"
    assert 30 <= rec["median_epochs"] <= 50
    assert recorded_training("ResNet-18 features (ridge)", ROOT / "docs/results") is None
    fin = final_training(ROOT / "docs/results")
    assert (fin["minutes"], fin["epochs"], fin["batch"], fin["n_trees"]) == (2.0, 34, 8, 449)


def test_ground_truths_on_two_sample_trees(sample_dir):
    clicks = load_clicks(ROOT / "annotations/winter_clicks.json")
    recs = [r for r in index_dataset(sample_dir).trees if r.tree_id in ("R01N02", "R15N05")]
    gt = ground_truths(recs, clicks, repeats=1)
    assert set(gt) == {"winter baseline", "guided tracing"}
    for g in gt.values():
        assert g["n_trees"] == 2 and 0 < g["seconds"] <= g["seconds_max"] and g["peak_mb"] > 1
    # The guided tracing computes a cost map with a tube filter at five scales: the slower of the two.
    assert gt["guided tracing"]["seconds"] > gt["winter baseline"]["seconds"]


def test_mask_post_processing_on_saved_masks():
    from treeskel.segment import load_masks

    masks = load_masks(ROOT / "docs/results/segment_lined_up_joined/predicted_masks.npz")
    pp = mask_post_processing(masks, n=2, repeats=1)
    assert pp["n_masks"] == 2 and pp["seconds"] > 0 and pp["peak_mb"] > 0


def _report(networks):
    return {"machine": {"cpu": "x86_64", "gpu": None, "torch": "2.x", "threads": 4},
            "ground_truths": {"winter baseline": {"seconds": 0.2, "seconds_max": 0.3, "peak_mb": 40.0, "n_trees": 4}},
            "networks": networks,
            "training": {"U-Net 256x256 (named)": {"minutes_5_folds": 20.4, "device": "cuda", "median_epochs": 34,
                                                   "estimated_minutes": None, "estimated_on": "cpu"}},
            "final_training": {"minutes": 2.0, "epochs": 34, "batch": 8, "n_trees": 449, "estimated_minutes": 1.9},
            "post_processing": {"seconds": 0.01, "seconds_max": 0.02, "peak_mb": 3.0, "n_masks": 20},
            "mean_shape": {"train_s": 1e-5, "parameters": 10}}


def test_report_lines_handle_missing_values():
    net = {"parameters": 11_000_000, "weights_mb": 44.0, "predict_s": 0.05, "predict_peak_mb": None,
           "train_step_s": None, "train_peak_mb": None}
    text = "\n".join(report_lines(_report({"cpu": {"angle CNN 240x320": net}})))
    assert "11,000,000" in text and "no GPU" in text and "recorded   20.4" in text


def test_measure_network_on_the_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from treeskel.benchmark import measure_network, n_parameters, network_specs

    spec = next(s for s in network_specs() if s["name"] == "U-Net 256x256 (named)")
    m = measure_network(dict(spec, batch=2), "cpu", repeats=1)
    assert m["parameters"] == n_parameters(spec["build"]()) > 1_000_000
    assert m["weights_mb"] == pytest.approx(m["parameters"] * 4 / 1e6)
    assert m["predict_s"] > 0 and m["train_step_s"] > 0
    assert m["predict_peak_mb"] is None and m["train_peak_mb"] is None          # GPU only
    feats = next(s for s in network_specs() if s["kind"] == "features")
    assert measure_network(feats, "cpu", repeats=1)["train_step_s"] is None   # no training step
    assert torch.get_num_threads() >= 1


def test_quick_run_writes_the_report(sample_dir, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from treeskel.benchmark import main

    monkeypatch.chdir(ROOT)
    assert main([str(sample_dir), "--trees", "1", "--quick", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    nets = report["networks"]["cpu"]
    assert "U-Net 256x256 (named)" in nets and "angle CNN 240x320" in nets
    assert report["ground_truths"]["winter baseline"]["n_trees"] == 1
    assert report["final_training"]["estimated_minutes"] > 0
    assert (tmp_path / "output.txt").read_text().startswith("Machine:")
