"""Tests for treeskel.benchmark (the network and FLOP parts need PyTorch and are skipped without it)."""

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


# --------------------------------------------------------------------------
# FLOPs, memory traffic and the profiler (--flops)
# --------------------------------------------------------------------------

def _tiny_net(torch):
    """A 3x3 convolution (3 -> 8 channels, no bias), an in-place ReLU and a flatten (a view)."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1, bias=False), torch.nn.ReLU(inplace=True),
                               torch.nn.Flatten()).eval()


def test_count_flops_of_a_convolution():
    torch = pytest.importorskip("torch")
    from treeskel.benchmark import count_flops, prediction

    net, x = _tiny_net(torch), torch.zeros(1, 3, 16, 16)
    one = 2 * (8 * 16 * 16) * (3 * 3 * 3)                   # a multiply-add per output pixel and kernel weight
    assert count_flops(lambda: net(x)) == one
    assert count_flops(prediction(net, x)) == 2 * one       # the photo and its mirror image; ReLU not counted


def test_memory_traffic_counts_inputs_weights_and_outputs_but_not_views():
    torch = pytest.importorskip("torch")
    from treeskel.benchmark import memory_traffic

    net, x = _tiny_net(torch), torch.zeros(1, 3, 16, 16)
    t = memory_traffic(lambda: net(x))
    photo, weights, features = 4 * 3 * 16 * 16, 4 * 8 * 3 * 3 * 3, 4 * 8 * 16 * 16    # float32 bytes
    # The convolution reads the photo and its weights and writes the features; the in-place ReLU reads and
    # writes the features; the flatten is a view and moves nothing.
    assert t["read_bytes"] == photo + weights + features and t["written_bytes"] == 2 * features
    assert t["operations"] == 2 and "view.default" not in t["by_operation"]


def test_roofline_names_the_limit():
    from treeskel.benchmark import roofline

    r = roofline(21.6e9, 626e6, 11.3e12, 484e9)              # the report's figures for the named U-Net
    assert r["bound"] == "compute" and r["intensity"] == pytest.approx(34.5, abs=0.05)
    assert r["compute_s"] == pytest.approx(1.91e-3, abs=1e-5) and r["memory_s"] == pytest.approx(1.29e-3, abs=1e-5)
    low = roofline(1e9, 1e9, 10e12, 500e9)                   # 1 FLOP per byte, ridge at 20
    assert low["bound"] == "memory" and low["memory_s"] > low["compute_s"] and low["ridge"] == 20


def test_summarise_events_counts_outermost_operators_and_gpu_work():
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from torch.autograd import DeviceType

    from treeskel.benchmark import summarise_events

    def ev(name, device=DeviceType.CPU, parent=None, us=0.0):
        return SimpleNamespace(name=name, device_type=device, cpu_parent=parent,
                               time_range=SimpleNamespace(elapsed_us=lambda: us))

    conv = ev("aten::conv2d")
    events = [conv, ev("aten::convolution", parent=conv), ev("aten::relu_"), ev("ProfilerStep#0"),
              ev("cudnn::conv_kernel", DeviceType.CUDA, us=100), ev("relu_kernel", DeviceType.CUDA, us=50),
              ev("Memcpy HtoD (Pageable -> Device)", DeviceType.CUDA, us=10)]
    s = summarise_events(events)
    assert (s["operators"], s["kernels"], s["copies"]) == (2, 2, 1)          # convolution inside conv2d not counted
    assert s["kernel_s"] == pytest.approx(150e-6) and s["copy_s"] == pytest.approx(10e-6)


def test_profile_prediction_on_the_cpu():
    torch = pytest.importorskip("torch")
    from treeskel.benchmark import prediction, profile_prediction

    p = profile_prediction(prediction(_tiny_net(torch), torch.zeros(1, 3, 16, 16)), "cpu")
    # Per pass: conv2d, relu_, flatten; and one flip for the mirror image. No GPU, so no kernels.
    assert p == {"operators": 7, "kernels": 0, "kernel_s": 0.0, "copies": 0, "copy_s": 0.0}


def test_flops_lines_handle_unknown_peak_and_profiler_errors():
    from treeskel.benchmark import flops_lines

    traffic = {"read_bytes": 2e8, "written_bytes": 1e8, "operations": 106}
    net = {"flops_pass": 1.08e10, "traffic_pass": traffic, "flops_photo": 2.16e10, "traffic_photo": traffic,
           "roofline": None, "recorded": {},
           "devices": {"cuda": {"predict_s": 0.01, "flop_per_s": 2.16e12, "profile": {"error": "profiler failed: x"}},
                       "cpu": {"predict_s": 0.1, "flop_per_s": 2.16e11, "profile": {"operators": 215, "kernels": 0,
                                                                                  "kernel_s": 0.0, "copies": 0,
                                                                                  "copy_s": 0.0}}}}
    report = {"machine": {"cpu": "x86_64", "gpu": "Some GPU", "torch": "2.x", "threads": 4}, "repeats": 3,
              "gpu": "Some GPU", "peak": None, "networks": {"U-Net 256x256 (named)": net}}
    text = "\n".join(flops_lines(report))
    assert "No roofline: the peak rates of Some GPU are unknown" in text
    assert "profiler failed: x" in text and "cpu: 215 operators" in text and "10.80" in text
    assert "report.json" not in text                          # nothing recorded, so no such section


def test_flops_mode_reproduces_the_reported_figures(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import shutil

    from treeskel.benchmark import main

    monkeypatch.chdir(ROOT)
    shutil.copy(ROOT / "docs/results/benchmark/report.json", tmp_path)         # Table 5's times, the GTX 1080 Ti
    assert main(["--flops", "--quick", "--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "flops.json").read_text())
    unet = r["networks"]["U-Net 256x256 (named)"]
    # The report: 10.8 GFLOP per pass, about 313 MB moved, 34 FLOP per byte, compute-bound on the GTX 1080 Ti
    # (ridge about 23), 20% of its peak at the measured 9.7 ms; the CPU about 190 GFLOP/s.
    assert unet["flops_pass"] == 10_779_361_280 and unet["flops_photo"] == 2 * unet["flops_pass"]
    moved = unet["traffic_pass"]["read_bytes"] + unet["traffic_pass"]["written_bytes"]
    assert moved / 1e6 == pytest.approx(313.1, abs=0.1) and unet["flops_pass"] / moved == pytest.approx(34.4, abs=0.1)
    assert r["gpu"] == "NVIDIA GeForce GTX 1080 Ti" and unet["roofline"]["bound"] == "compute"
    assert unet["roofline"]["ridge"] == pytest.approx(23.4, abs=0.05)
    assert unet["recorded"]["cuda"]["flop_per_s"] / r["peak"][0] == pytest.approx(0.196, abs=0.002)
    assert unet["recorded"]["cpu"]["flop_per_s"] / 1e9 == pytest.approx(186, abs=1)
    assert unet["devices"]["cpu"]["profile"]["operators"] == 215 and "480x640" not in " ".join(r["networks"])
    assert (tmp_path / "flops_output.txt").read_text().startswith("Machine:")
    assert (tmp_path / "report.json").read_bytes() == (ROOT / "docs/results/benchmark/report.json").read_bytes()
    assert main(["--flops", "--quick", "--out", str(tmp_path), "--peak-tflops", "11"]) == 2      # both or neither
