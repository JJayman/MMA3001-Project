"""Run time and memory of every method: the two ground truths and the summer models.

The brief asks for methods to be compared on accuracy, run time and memory. The
accuracies are in the results folders; this module measures the rest, on one
machine, the same way for every method:

- **Ground truths (winter, CPU):** the automatic baseline
  (:func:`treeskel.winter_gt.extract_winter_gt`) and the guided tracing
  (:func:`treeskel.guided_gt.guided_gt`: cost map, tracing, widths), on the
  development trees of a dataset folder (by default the committed sample).
  The guided ground truth also took the student's clicks, about 15 s per tree
  (session 5).
- **Mean shape:** the training trees' mean of each measure.
- **Ridge** (:mod:`treeskel.shape_model`): the summer photo's fixed ResNet-18
  features (photo and mirror image at 240x320), then ridge regression; its
  5-fold training (nested choice of the penalty) is timed on the committed
  features.
- **Angle CNN** (:mod:`treeskel.shape_cnn`): ResNet-18 fine-tuned to the ten
  measures, at 240x320 and 480x640.
- **U-Net** (:mod:`treeskel.segment`): the named model at 256x256, and the
  full-size variant at 480x640; then joining its mask's pieces and measuring
  it (:func:`treeskel.segment.join_pieces`, :func:`treeskel.segment.mask_shape`),
  timed on the named run's saved masks.

For each network: its parameters and the memory they take (32-bit), the time
to predict one photo (with its mirror image, as every model here predicts),
on the GPU and the CPU, and one training step at its recipe's batch size,
with the GPU memory each needs at its peak. Timing and memory do not depend
on the weights' values, so the networks are built with random weights (no
download) and fed random photos of the right size. Each time is the median of
repeated runs after warm-up runs (the GPU is synchronised before reading the
clock). The CPU methods' memory is the peak of what Python and NumPy allocate
(``tracemalloc``), measured in a separate run, since tracing slows the code.

The training times of the real runs (5 folds with their inner folds, and the
final test's one network) are read from their ``report.json``, beside an
estimate from the measured step time: steps per epoch times epochs.

``--flops`` explains the times instead of measuring the methods (session 18):
each network's floating-point operations (PyTorch's FLOP counter), the bytes
it moves if nothing were fused or cached, hence its arithmetic intensity, and
where that puts it on the roofline of the GPU the times came from; then
PyTorch's profiler over one prediction counts the operators and, on a GPU,
the kernels it launches and how long the GPU is busy with them. It writes
``flops.json`` and ``flops_output.txt`` beside ``report.json``, which it
leaves alone.

Usage (on the PC with the GPU):

```
python -m treeskel.benchmark                 # -> docs/results/benchmark/
python -m treeskel.benchmark data/raw --trees 8
python -m treeskel.benchmark --flops         # -> docs/results/benchmark/flops.json, flops_output.txt
```
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

RESULTS = Path("docs/results")
"""Where the runs' results are."""

RECORDED = {
    "angle CNN 240x320": "shape_cnn",
    "angle CNN 480x640": "shape_cnn_480x640",
    "U-Net 256x256 (named)": "segment_lined_up",
    "U-Net 480x640": "segment_lined_up_480x640",
}
"""The results folder of each network's development run (5 folds, each with its inner fold)."""

N_DEV = 449
"""Development trees (the final network trains on all of them; a fold on about four fifths)."""

GPU_PEAKS = {"NVIDIA GeForce GTX 1080 Ti": (11.34e12, 484.4e9)}
"""Rated 32-bit peak (FLOP/s) and memory bandwidth (bytes/s) of known GPUs, from NVIDIA's specifications.
The GTX 1080 Ti: 3584 cores x 2 FLOP (a multiply-add) x 1582 MHz boost clock; 11 Gbit/s x 352-bit memory."""


def timed(fn: Callable[[], object], repeats: int = 5, warmup: int = 1, sync: Callable[[], None] | None = None) -> dict:
    """Seconds one call of ``fn`` takes: the median and the fastest of ``repeats`` calls after ``warmup`` calls.

    Args:
        fn: The work to time.
        repeats: Timed calls.
        warmup: Untimed calls first (caches, the GPU's kernels, memory pools).
        sync: Called before each clock reading (``torch.cuda.synchronize`` on the GPU, whose calls
            return before the work is done).

    Example:
        >>> t = timed(lambda: sum(range(1000)), repeats=3)
        >>> sorted(t) == ["median_s", "min_s", "repeats"] and t["min_s"] <= t["median_s"]
        True
    """
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(times), "min_s": min(times), "repeats": repeats}


def peak_python_mb(fn: Callable[[], object]) -> float:
    """The peak memory (MB) Python and NumPy allocate while ``fn`` runs, above what was in use before.

    Example:
        >>> 7.5 < peak_python_mb(lambda: np.ones(1_000_000)) < 8.5        # 8 MB of float64
        True
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        fn()
        return (tracemalloc.get_traced_memory()[1] - before) / 1e6
    finally:
        tracemalloc.stop()


def n_parameters(model) -> int:
    """Number of parameters of a PyTorch model."""
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# The ground truths and the CPU steps
# --------------------------------------------------------------------------

def ground_truths(records: Sequence, clicks: dict, repeats: int = 3) -> dict:
    """Seconds and peak memory per tree of the automatic baseline and of the guided tracing.

    Args:
        records: Development trees (:class:`treeskel.io.TreeRecord`) with complete clicks.
        clicks: :func:`treeskel.guided_gt.load_clicks`.
        repeats: Timed runs per tree.

    Returns:
        ``{"winter baseline": {...}, "guided tracing": {...}}``, each with the median over the trees of
        each tree's median seconds, the largest peak memory (MB), and the number of trees.
    """
    from treeskel.guided_gt import guided_gt
    from treeskel.io import load_tree
    from treeskel.winter_gt import extract_winter_gt

    per = {"winter baseline": ([], []), "guided tracing": ([], [])}
    for rec in records:
        d = load_tree(rec, "winter")
        jobs = {"winter baseline": lambda: extract_winter_gt(d["depth"], d["valid"]),
                "guided tracing": lambda: guided_gt(d["rgb"], d["depth"], d["valid"], clicks[rec.tree_id])}
        for name, fn in jobs.items():
            per[name][0].append(timed(fn, repeats)["median_s"])
            per[name][1].append(peak_python_mb(fn))
    return {name: {"seconds": statistics.median(s), "seconds_max": max(s), "peak_mb": max(m), "n_trees": len(s)}
            for name, (s, m) in per.items() if s}


def mask_post_processing(masks: dict[str, np.ndarray], n: int = 20, repeats: int = 3) -> dict:
    """Seconds and peak memory to join a predicted mask's pieces and measure its shape (per mask).

    Args:
        masks: Saved 256x256 masks (:func:`treeskel.segment.load_masks`); the first ``n`` (by tree) are used.
    """
    from treeskel.segment import join_pieces, mask_shape

    def post(m):
        """Join the mask's pieces, then measure its shape."""
        return mask_shape(join_pieces(m)[0])

    chosen = [masks[t] for t in sorted(masks)[:n]]
    secs = [timed(lambda m=m: post(m), repeats)["median_s"] for m in chosen]
    return {"seconds": statistics.median(secs), "seconds_max": max(secs),
            "peak_mb": max(peak_python_mb(lambda m=m: post(m)) for m in chosen), "n_masks": len(chosen)}


def ridge_training(features: Path, table: Path, split: Path) -> dict:
    """Seconds to train and cross-validate the ridge model as its run did: 5 row-grouped folds, the penalty
    chosen in each by inner folds, on the committed summer features; and the size of one fitted model."""
    from treeskel.shape import MEASURES, load_features
    from treeskel.shape_model import cross_validate, feature_matrix, load_table

    ids, arrays, _ = load_features(features)
    by_id = {r["tree"]: r for r in load_table(table)}
    keep = [i for i, t in enumerate(ids) if t in by_id]
    x2 = feature_matrix(arrays, "rgb")[keep]
    rows = np.array([by_id[ids[i]]["row"] for i in keep])
    y = np.array([[by_id[ids[i]][f"winter_{k}"] for k in MEASURES] for i in keep], float)
    fold_rows = json.loads(Path(split).read_text())["fold_val_rows"]
    t0 = time.perf_counter()
    cross_validate(x2, y, rows, fold_rows)
    d = x2.shape[-1]
    # A fitted model: the features' mean and SD, and per measure a weight per feature and an intercept.
    return {"seconds_5_folds": time.perf_counter() - t0, "n_trees": len(keep), "n_features": d,
            "parameters": 2 * d + len(MEASURES) * (d + 1)}


# --------------------------------------------------------------------------
# The networks
# --------------------------------------------------------------------------

def network_specs() -> list[dict]:
    """Each network as measured: name, input size, the recipe's training batch, and how to build it."""
    from treeskel.segment import SegParams, UNet
    from treeskel.shape import MEASURES, make_resnet18
    from treeskel.shape_cnn import CNNParams, make_model

    cnn, seg = CNNParams(), SegParams()
    return [
        {"name": "ResNet-18 features (ridge)", "size": (240, 320), "batch": None, "kind": "features",
         "build": lambda: make_resnet18(pretrained=False, device="cpu")[0]},
        {"name": "angle CNN 240x320", "size": tuple(cnn.size), "batch": cnn.batch, "kind": "regression",
         "build": lambda: make_model(len(MEASURES), pretrained=False)},
        {"name": "angle CNN 480x640", "size": (480, 640), "batch": cnn.batch, "kind": "regression",
         "build": lambda: make_model(len(MEASURES), pretrained=False)},
        {"name": "U-Net 256x256 (named)", "size": (seg.size, seg.size), "batch": seg.batch, "kind": "segmentation",
         "build": lambda: UNet(pretrained=False)},
        {"name": "U-Net 480x640", "size": (480, 640), "batch": seg.batch, "kind": "segmentation",
         "build": lambda: UNet(pretrained=False)},
    ]


def prediction(model, x) -> Callable[[], None]:
    """One prediction, as every model here makes it: ``model`` on the photo ``x`` and on its mirror image."""
    import torch

    def predict():
        """The photo, then its mirror image, without gradients."""
        with torch.inference_mode():
            model(x)
            model(x.flip(-1))                                   # every model averages in the mirror image

    return predict


def _loss(kind: str, out, torch):
    """A loss of the right form for a training step (its value does not matter for the timing)."""
    if kind == "segmentation":
        from treeskel.segment import seg_loss
        return seg_loss(out, (torch.rand_like(out) > 0.9).float())
    return ((out - torch.randn_like(out)) ** 2).mean()


def measure_network(spec: dict, device: str, repeats: int = 5, train: bool = True) -> dict:
    """Parameters, prediction time per photo (with its mirror image) and one training step, on ``device``.

    Returns:
        ``{"parameters", "weights_mb", "predict_s", "predict_peak_mb", "train_step_s", "train_peak_mb"}``;
        the peaks are the GPU's (None on the CPU), the training entries None if not measured.
    """
    import torch

    torch.manual_seed(0)
    model = spec["build"]().to(device)
    h, w = spec["size"]
    cuda = device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else None
    out = {"parameters": n_parameters(model), "weights_mb": n_parameters(model) * 4 / 1e6,
           "predict_s": None, "predict_peak_mb": None, "train_step_s": None, "train_peak_mb": None}

    x = torch.randn(1, 3, h, w, device=device)
    model.eval()
    predict = prediction(model, x)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    out["predict_s"] = timed(predict, repeats, warmup=2, sync=sync)["median_s"]
    if cuda:
        out["predict_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6

    if train and spec["batch"]:
        xb = torch.randn(spec["batch"], 3, h, w, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()

        def step():
            """One training step on the random batch: forward, backward, update."""
            loss = _loss(spec["kind"], model(xb), torch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # A full-size step on the CPU can take tens of seconds, so fewer repeats there.
        out["train_step_s"] = timed(step, repeats if cuda else 2, warmup=1, sync=sync)["median_s"]
        if cuda:
            out["train_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    del model
    if cuda:
        torch.cuda.empty_cache()
    return out


def recorded_training(name: str, results: Path = RESULTS) -> dict | None:
    """The development run's training time and chosen epochs (``report.json``), or None if not found."""
    folder = RECORDED.get(name)
    path = results / folder / "report.json" if folder else None
    if path is None or not path.exists():
        return None
    r = json.loads(path.read_text())
    epochs = [e["epochs"] for e in r.get("epochs", []) if isinstance(e, dict)]
    return {"run": folder, "minutes_5_folds": r.get("minutes"), "device": r.get("device"),
            "median_epochs": statistics.median(epochs) if epochs else None}


def final_training(results: Path = RESULTS) -> dict | None:
    """The final test's one training of the named U-Net on all development trees (``report.json``)."""
    path = results / "final_test" / "report.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return {"minutes": r.get("minutes"), "epochs": r.get("epochs"), "batch": r.get("params", {}).get("batch"),
            "n_trees": N_DEV}


def estimated_minutes(step_s: float | None, batch: int | None, epochs: float | None, n_trees: int = N_DEV) -> float | None:
    """Training minutes from a measured step: ``ceil(n_trees / batch)`` steps per epoch, times ``epochs``.

    Example:
        >>> estimated_minutes(0.06, 8, 34)                  # 57 steps per epoch
        1.938
    """
    if step_s is None or not batch or not epochs:
        return None
    return round(step_s * math.ceil(n_trees / batch) * epochs / 60, 3)


# --------------------------------------------------------------------------
# FLOPs, memory traffic and the profiler (--flops)
# --------------------------------------------------------------------------

def count_flops(fn: Callable[[], object]) -> int:
    """Floating-point operations ``fn`` performs, by PyTorch's FLOP counter.

    The counter (``torch.utils.flop_counter.FlopCounterMode``) counts convolutions and matrix products,
    two operations per multiply-add, from the tensors' shapes, so the count is the same on every device.
    Batch norm, ReLU, additions, pooling and interpolation are not counted: a few operations per element
    they write, about 0.5% of the convolutions' count in the named U-Net.

    For example, a 3x3 convolution from 3 to 8 channels on a 16x16 image counts
    2 x (8 x 16 x 16 outputs) x (3 x 3 x 3 weights each) = 110,592 (tested in ``tests/test_benchmark.py``).

    Args:
        fn: The work, e.g. one forward pass or :func:`prediction`.
    """
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return int(counter.get_total_flops())


def memory_traffic(fn: Callable[[], object]) -> dict:
    """Bytes ``fn`` reads and writes if nothing were fused or cached.

    Every operation PyTorch dispatches (convolution, batch norm, ReLU, addition, interpolation,
    concatenation, pooling) is taken to read all its input tensors, weights included, from memory and to
    write all its outputs back. Views (``select``, ``view``, ...) move nothing and are left out. Real kernels
    can fuse steps or find data still in cache, so this is an upper bound on the traffic, and the arithmetic
    intensity from it (:func:`roofline`) a lower bound. It is the same on every device.

    For example, that convolution of :func:`count_flops` in 32-bit floats reads its image and weights,
    4 x (3 x 16 x 16 + 8 x 3 x 3 x 3) bytes, and writes 4 x 8 x 16 x 16; an in-place ReLU after it reads and
    writes those 8,192 bytes again (tested in ``tests/test_benchmark.py``).

    Returns:
        ``{"read_bytes", "written_bytes", "operations", "by_operation"}``: the totals, the number of
        operations counted, and how many of each.
    """
    import collections

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    def nbytes(tree) -> int:
        """Bytes of all the tensors in a nest of arguments or results."""
        return sum(t.numel() * t.element_size() for t in tree_flatten(tree)[0] if isinstance(t, torch.Tensor))

    class Traffic(TorchDispatchMode):
        """Adds up the bytes each dispatched operation reads and writes."""

        def __init__(self):
            """Start from zero."""
            super().__init__()
            self.read = self.written = 0
            self.ops = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            """Run the operation, then count its inputs and outputs unless its result is a view."""
            out = func(*args, **(kwargs or {}))
            # A view returns an alias of its input that it does not write; in-place operations (relu_, add_)
            # return an alias they write, and read and write that memory, so they are counted.
            if any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns):
                return out
            self.read += nbytes((args, kwargs))
            self.written += nbytes(out)
            self.ops[func.__name__] += 1
            return out

    mode = Traffic()
    with mode:
        fn()
    return {"read_bytes": mode.read, "written_bytes": mode.written, "operations": sum(mode.ops.values()),
            "by_operation": dict(mode.ops.most_common())}


def roofline(flops: float, traffic_bytes: float, peak_flops: float, bandwidth: float) -> dict:
    """Where work of ``flops`` operations moving ``traffic_bytes`` sits on a device's roofline.

    A device can do at most ``peak_flops`` operations and move at most ``bandwidth`` bytes per second, so
    the work takes at least ``flops / peak_flops`` (compute) and ``traffic_bytes / bandwidth`` (memory)
    seconds. Its arithmetic intensity (operations per byte) above the device's ridge point
    (``peak_flops / bandwidth``) means the arithmetic, not the memory, sets that lower limit.

    Returns:
        ``{"intensity", "ridge", "compute_s", "memory_s", "bound"}``, ``bound`` "compute" or "memory".

    Example:
        >>> r = roofline(21.6e9, 626e6, 11.3e12, 484e9)       # a photo of the named U-Net on a GTX 1080 Ti
        >>> round(r["intensity"], 1), round(r["ridge"], 1), r["bound"], round(r["compute_s"] * 1e3, 2)
        (34.5, 23.3, 'compute', 1.91)
    """
    intensity, ridge = flops / traffic_bytes, peak_flops / bandwidth
    return {"intensity": intensity, "ridge": ridge, "compute_s": flops / peak_flops,
            "memory_s": traffic_bytes / bandwidth, "bound": "compute" if intensity >= ridge else "memory"}


def summarise_events(events: Sequence) -> dict:
    """Operators and GPU work in the profiler's events of one prediction.

    Args:
        events: ``torch.profiler.profile(...).events()``, or objects with the same ``name``,
            ``device_type``, ``cpu_parent`` and ``time_range.elapsed_us()``.

    Returns:
        ``{"operators", "kernels", "kernel_s", "copies", "copy_s"}``: the PyTorch operators called (the
        outermost ``aten::`` calls, not the ones they call in turn), and the GPU's kernels and memory
        copies or fills with their summed durations (zero without a GPU). Kernels on one stream run one
        after another, so ``kernel_s`` is the time the GPU was busy computing.
    """
    from torch.autograd import DeviceType

    def outermost(e) -> bool:
        """An ``aten::`` operator not called by another one."""
        return e.name.startswith("aten::") and not (e.cpu_parent is not None
                                                    and e.cpu_parent.name.startswith("aten::"))

    gpu = [e for e in events if e.device_type == DeviceType.CUDA]
    copies = [e for e in gpu if e.name.startswith(("Memcpy", "Memset"))]
    kernels = [e for e in gpu if not e.name.startswith(("Memcpy", "Memset"))]
    return {"operators": sum(1 for e in events if e.device_type == DeviceType.CPU and outermost(e)),
            "kernels": len(kernels), "kernel_s": sum(e.time_range.elapsed_us() for e in kernels) / 1e6,
            "copies": len(copies), "copy_s": sum(e.time_range.elapsed_us() for e in copies) / 1e6}


def profile_prediction(predict: Callable[[], None], device: str, warmup: int = 2) -> dict:
    """PyTorch's profiler over one prediction, after warm-up calls: :func:`summarise_events` of its events.

    On a GPU it records the kernels too (through NVIDIA's CUPTI); if the profiler fails, or sees no kernels
    on a GPU, the result says so in ``"error"`` instead of stopping the benchmark.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    cuda = device.startswith("cuda")
    for _ in range(warmup):
        predict()
    if cuda:
        torch.cuda.synchronize()
    try:
        with profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])) as prof:
            predict()
            if cuda:
                torch.cuda.synchronize()
        out = summarise_events(prof.events())
    except Exception as e:                                       # e.g. CUPTI missing: keep the other numbers
        return {"error": f"profiler failed: {e}"}
    if cuda and out["kernels"] == 0:
        out["error"] = "the profiler saw no GPU kernels (is CUPTI available?)"
    return out


def flop_report(specs: Sequence[dict], devices: Sequence[str], repeats: int = 10, recorded: dict | None = None,
                gpu: str | None = None, peak: tuple[float, float] | None = None) -> dict:
    """FLOPs, memory traffic, roofline and profile of each network, and the speed its times imply.

    Args:
        specs: :func:`network_specs` (random weights, as the timings).
        devices: Where to time and profile one prediction (photo and mirror image, batch 1).
        repeats: Timed predictions per network and device (median).
        recorded: The benchmark's ``report.json`` (Table 5's times), whose prediction times are turned into
            FLOP/s too; None to skip.
        gpu: The GPU whose roofline to draw: by default the one present, else the one ``recorded`` ran on.
        peak: ``(FLOP/s, bytes/s)`` of that GPU, when :data:`GPU_PEAKS` does not know it.

    Returns:
        ``{"gpu", "peak", "networks"}``; per network ``flops_pass`` and ``traffic_pass`` (one forward pass,
        counted on the CPU), ``flops_photo`` and ``traffic_photo`` (a prediction, with the mirror image),
        ``roofline`` (per photo, on ``gpu``; None if its peak is unknown), and per device ``predict_s``,
        ``flop_per_s`` and ``profile``; and ``recorded`` with the FLOP/s of ``report.json``'s times.
    """
    import torch

    gpu = gpu or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None) \
        or (recorded or {}).get("machine", {}).get("gpu")
    peak = peak or GPU_PEAKS.get(gpu)
    out = {"gpu": gpu, "peak": list(peak) if peak else None, "networks": {}}
    for spec in specs:
        torch.manual_seed(0)
        model = spec["build"]().eval()
        h, w = spec["size"]
        x = torch.randn(1, 3, h, w)

        def one_pass():
            """One forward pass, without gradients."""
            with torch.inference_mode():
                model(x)

        n = {"size": [h, w], "flops_pass": count_flops(one_pass), "traffic_pass": memory_traffic(one_pass),
             "flops_photo": count_flops(prediction(model, x))}
        n["traffic_photo"] = memory_traffic(prediction(model, x))
        moved = n["traffic_photo"]["read_bytes"] + n["traffic_photo"]["written_bytes"]
        n["roofline"] = roofline(n["flops_photo"], moved, *peak) if peak else None
        n["devices"] = {}
        for dev in devices:
            model.to(dev)
            xd = x.to(dev)
            predict = prediction(model, xd)
            sync = torch.cuda.synchronize if dev.startswith("cuda") else None
            secs = timed(predict, repeats, warmup=2, sync=sync)["median_s"]
            n["devices"][dev] = {"predict_s": secs, "flop_per_s": n["flops_photo"] / secs,
                                 "profile": profile_prediction(predict, dev)}
        rec = {dev: nets[spec["name"]]["predict_s"] for dev, nets in (recorded or {}).get("networks", {}).items()
               if nets.get(spec["name"], {}).get("predict_s")}
        n["recorded"] = {dev: {"predict_s": s, "flop_per_s": n["flops_photo"] / s} for dev, s in rec.items()}
        out["networks"][spec["name"]] = n
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def _f(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def report_lines(report: dict) -> list[str]:
    """The report as text, for the console and ``output.txt``."""
    m = report["machine"]
    out = [f"Machine: {m['cpu']}; {m['gpu'] or 'no GPU'}; PyTorch {m['torch']}; {m['threads']} CPU threads.", ""]
    gt = report.get("ground_truths", {})
    if gt:
        out.append(f"Ground truths, per winter tree ({next(iter(gt.values()))['n_trees']} development trees; "
                   "seconds median / slowest, peak Python+NumPy memory):")
        for name, g in gt.items():
            out.append(f"  {name:22s} {g['seconds']:7.3f} / {g['seconds_max']:.3f} s   {g['peak_mb']:7.1f} MB")
        out.append("  guided tracing also needs the clicks: about 15 s per tree (session 5)")
        out.append("")
    nets = report["networks"]
    devices = list(nets)
    out.append("Networks (random weights; prediction = photo + mirror image, batch 1; training step at the recipe's "
               "batch; GPU peaks include the weights):")
    head = f"  {'network':28s} {'params':>10s} {'MB':>6s}"
    for dev in devices:
        head += f" | {dev + ' predict ms':>17s} {'peak MB':>8s} {'train step s':>12s} {'peak MB':>8s}"
    out.append(head)
    for name in nets[devices[0]]:
        first = nets[devices[0]][name]
        line = f"  {name:28s} {first['parameters']:>10,d} {first['weights_mb']:6.1f}"
        for dev in devices:
            n = nets[dev][name]
            line += (f" | {_f(n['predict_s'] and n['predict_s'] * 1e3, '17.1f')} {_f(n['predict_peak_mb'], '8.0f')}"
                     f" {_f(n['train_step_s'], '12.3f')} {_f(n['train_peak_mb'], '8.0f')}")
        out.append(line)
    out.append("")
    tr = report.get("training", {})
    if tr:
        out.append("Training time (minutes): the development run (5 folds with inner folds, as recorded), and one "
                   f"network on all {N_DEV} trees estimated from the measured step at the run's median epochs:")
        for name, t in tr.items():
            out.append(f"  {name:28s} recorded {_f(t.get('minutes_5_folds'), '6.1f')} ({t.get('device') or '-'}), "
                       f"median epochs {_f(t.get('median_epochs'), '4.1f')}, estimated one network "
                       f"{_f(t.get('estimated_minutes'), '6.2f')} on {t.get('estimated_on') or '-'}")
    fin = report.get("final_training")
    if fin:
        out.append(f"  final test: the named U-Net trained once on {fin['n_trees']} trees, {fin['epochs']} epochs: "
                   f"recorded {_f(fin['minutes'], '.1f')} min; estimated {_f(fin.get('estimated_minutes'), '.2f')} min")
    rd = report.get("ridge")
    if rd:
        out += ["", f"Ridge: 5-fold training with nested penalty choice {rd['seconds_5_folds']:.1f} s on "
                    f"{rd['n_trees']} trees; {rd['parameters']:,d} parameters ({rd['n_features']} features)"]
        feats = nets[devices[0]].get("ResNet-18 features (ridge)", {}).get("predict_s")
        if feats:
            out.append(f"  its features (photo and mirror image) for {rd['n_trees']} trees: about "
                       f"{feats * rd['n_trees']:.0f} s on the {devices[0].upper()}; a prediction is the features "
                       f"({feats * 1e3:.1f} ms) and one product with the weights")
    pp = report.get("post_processing")
    if pp:
        out.append(f"U-Net mask: joining pieces and measuring the shape {pp['seconds'] * 1e3:.1f} ms per mask "
                   f"(slowest {pp['seconds_max'] * 1e3:.1f}), peak {pp['peak_mb']:.1f} MB ({pp['n_masks']} saved masks)")
    ms = report.get("mean_shape")
    if ms:
        out.append(f"Mean shape: training {ms['train_s'] * 1e6:.0f} us, {ms['parameters']} parameters; "
                   "prediction is a constant")
    return out


def flops_lines(report: dict) -> list[str]:
    """The ``--flops`` report as text, for the console and ``flops_output.txt``."""
    m = report["machine"]
    nets = report["networks"]
    out = [f"Machine: {m['cpu']}; {m['gpu'] or 'no GPU'}; PyTorch {m['torch']}; {m['threads']} CPU threads.", "",
           "One forward pass, batch 1 (PyTorch's FLOP counter; bytes read and written if nothing were fused or "
           "cached, so FLOP/byte is a lower bound):",
           f"  {'network':28s} {'GFLOP':>7s} {'read MB':>8s} {'written MB':>10s} {'FLOP/byte':>9s} {'operations':>10s}"]
    for name, n in nets.items():
        t = n["traffic_pass"]
        moved = t["read_bytes"] + t["written_bytes"]
        out.append(f"  {name:28s} {n['flops_pass'] / 1e9:7.2f} {t['read_bytes'] / 1e6:8.1f} "
                   f"{t['written_bytes'] / 1e6:10.1f} {n['flops_pass'] / moved:9.1f} {t['operations']:10d}")
    peak = report.get("peak")
    out.append("")
    if peak:
        out.append(f"A prediction (photo + mirror image) on the roofline of the {report['gpu']} "
                   f"({peak[0] / 1e12:.2f} TFLOP/s, {peak[1] / 1e9:.0f} GB/s; ridge {peak[0] / peak[1]:.1f} FLOP/byte):")
        out.append(f"  {'network':28s} {'GFLOP':>7s} {'MB moved':>9s} {'FLOP/byte':>9s} {'bound':>8s} "
                   f"{'compute ms':>10s} {'memory ms':>9s}  (at the peak rates)")
        for name, n in nets.items():
            r, t = n["roofline"], n["traffic_photo"]
            out.append(f"  {name:28s} {n['flops_photo'] / 1e9:7.2f} {(t['read_bytes'] + t['written_bytes']) / 1e6:9.1f} "
                       f"{r['intensity']:9.1f} {r['bound']:>8s} {r['compute_s'] * 1e3:10.2f} {r['memory_s'] * 1e3:9.2f}")
    else:
        out.append(f"No roofline: the peak rates of {report['gpu'] or 'the GPU'} are unknown "
                   "(give --peak-tflops and --bandwidth-gbs).")

    def speed(d: dict, gpu: bool) -> str:
        """Milliseconds, GFLOP/s and, on the GPU, the share of its peak."""
        share = f" {d['flop_per_s'] / peak[0]:6.1%}" if gpu and peak else ""
        return f"{d['predict_s'] * 1e3:8.1f} ms {d['flop_per_s'] / 1e9:7.0f} GFLOP/s{share}"

    for key, title in (("devices", f"timed in this run (median of {report['repeats']})"),
                       ("recorded", "as timed by the benchmark (report.json)")):
        rows = [(name, n[key]) for name, n in nets.items() if n.get(key)]
        if not rows:
            continue
        out += ["", f"Speed of a prediction, {title}:"]
        for name, by_dev in rows:
            out.append(f"  {name:28s} " + " | ".join(f"{dev}: {speed(d, dev.startswith('cuda'))}"
                                                      for dev, d in by_dev.items()))
    out += ["", "Profile of one prediction (torch.profiler; GPU busy = summed kernel time / timed prediction):"]
    for name, n in nets.items():
        parts = []
        for dev, d in n["devices"].items():
            p = d["profile"]
            if "operators" not in p:
                parts.append(f"{dev}: {p['error']}")
            elif dev.startswith("cuda"):
                parts.append(f"{dev}: {p['operators']} operators, {p['kernels']} kernels taking "
                             f"{p['kernel_s'] * 1e3:.2f} ms, {p['copies']} copies; GPU busy "
                             f"{p['kernel_s'] / d['predict_s']:.0%}" + (f" ({p['error']})" if "error" in p else ""))
            else:
                parts.append(f"{dev}: {p['operators']} operators")
        out.append(f"  {name:28s} " + " | ".join(parts))
    return out


def machine() -> dict:
    """The machine the benchmark ran on."""
    import os

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"cpu": platform.processor() or platform.machine(), "system": platform.platform(),
            "threads": torch.get_num_threads(), "cpu_count": os.cpu_count(), "gpu": gpu,
            "torch": torch.__version__, "python": platform.python_version()}


def main(argv: Sequence[str] | None = None) -> int:
    """Measure everything and save ``report.json`` and ``output.txt``."""
    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset
    from treeskel.segment import load_masks
    from treeskel.shape import MEASURES

    ap = argparse.ArgumentParser(prog="python -m treeskel.benchmark", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", nargs="?", default="data/sample", help="winter photos for the ground truths")
    ap.add_argument("--trees", type=int, default=4, help="development trees timed for the ground truths")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", default="docs/results/benchmark")
    ap.add_argument("--repeats", type=int, default=10, help="timed runs per network measurement")
    ap.add_argument("--no-cpu-training", action="store_true", help="skip the (slow) training steps on the CPU")
    ap.add_argument("--quick", action="store_true", help="few repeats and small parts only (for tests)")
    ap.add_argument("--flops", action="store_true",
                    help="only FLOPs, memory traffic, roofline and profiler: flops.json and flops_output.txt")
    ap.add_argument("--peak-tflops", type=float, help="the GPU's rated 32-bit peak, if not in GPU_PEAKS")
    ap.add_argument("--bandwidth-gbs", type=float, help="the GPU's memory bandwidth, if not in GPU_PEAKS")
    args = ap.parse_args(argv)

    import torch

    repeats = 2 if args.quick else args.repeats
    if args.flops:
        return flops_main(args, repeats)
    split = json.loads(Path(args.split).read_text())
    report: dict = {"machine": machine(), "repeats": repeats}
    print(f"Benchmark on {report['machine']['gpu'] or 'the CPU only'} ...", flush=True)

    # The ground truths (automatic baseline, guided tracing) on a few annotated development trees.
    clicks = load_clicks(args.clicks)
    records = [r for r in index_dataset(args.data_dir).trees
               if r.row not in split["test_rows"] and r.tree_id in clicks and clicks[r.tree_id].complete
               and not clicks[r.tree_id].skip][: args.trees]
    if records:
        print(f"  ground truths on {len(records)} development trees ...", flush=True)
        report["ground_truths"] = ground_truths(records, clicks, 1 if args.quick else 3)

    # Every network on every device: parameters, prediction and training-step time, peak memory.
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    specs = network_specs()
    if args.quick:
        specs = [s for s in specs if "480x640" not in s["name"]]
    report["networks"] = {}
    for dev in devices:
        report["networks"][dev] = {}
        for spec in specs:
            print(f"  {spec['name']} on the {dev.upper()} ...", flush=True)
            train = dev != "cpu" or not args.no_cpu_training
            report["networks"][dev][spec["name"]] = measure_network(spec, dev, repeats, train)

    # Training time: the recorded epochs of each saved run times the measured step time (on the GPU if
    # there is one), and the same for the final test's one network.
    fast = devices[0]
    report["training"] = {}
    for spec in specs:
        rec = recorded_training(spec["name"])
        if rec is None:
            continue
        step = report["networks"][fast][spec["name"]]["train_step_s"]
        rec["estimated_minutes"] = estimated_minutes(step, spec["batch"], rec["median_epochs"])
        rec["estimated_on"] = fast
        report["training"][spec["name"]] = rec
    fin = final_training()
    named = "U-Net 256x256 (named)"
    if fin and named in report["networks"][fast]:
        fin["estimated_minutes"] = estimated_minutes(report["networks"][fast][named]["train_step_s"], fin["batch"],
                                                     fin["epochs"])
        report["final_training"] = fin

    # The cheap methods: ridge on the committed features, the U-Net's post-processing, the mean shape.
    features, table = Path("data/shape/summer_resnet18.npz"), Path("data/shape/shape_measures.csv")
    if features.exists() and table.exists() and not args.quick:
        print("  ridge: 5-fold training on the committed features ...", flush=True)
        report["ridge"] = ridge_training(features, table, Path(args.split))
    saved = RESULTS / "segment_lined_up_joined" / "predicted_masks.npz"
    if saved.exists():
        print("  U-Net masks: joining and measuring ...", flush=True)
        report["post_processing"] = mask_post_processing(load_masks(saved), 3 if args.quick else 20,
                                                         1 if args.quick else 3)
    y = np.random.default_rng(0).normal(size=(360, len(MEASURES)))
    report["mean_shape"] = {"train_s": timed(lambda: np.nanmean(y, axis=0), 20)["median_s"],
                            "parameters": len(MEASURES)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = report_lines(report)
    print("\n" + "\n".join(lines))
    failed = []
    try:
        (out / "output.txt").write_text("\n".join(lines) + "\n")
    except OSError as e:
        failed.append(f"{out / 'output.txt'}: {e.strerror or e}")
    print(f"\nSaved {out / 'report.json'} and output.txt")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


def flops_main(args: argparse.Namespace, repeats: int) -> int:
    """``--flops``: measure :func:`flop_report` and save ``flops.json`` and ``flops_output.txt`` in ``--out``."""
    import torch

    out = Path(args.out)
    recorded_path = out / "report.json"
    recorded = json.loads(recorded_path.read_text()) if recorded_path.exists() else None
    if (args.peak_tflops is None) != (args.bandwidth_gbs is None):
        print("Give both --peak-tflops and --bandwidth-gbs, or neither.")
        return 2
    peak = (args.peak_tflops * 1e12, args.bandwidth_gbs * 1e9) if args.peak_tflops else None
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    specs = network_specs()
    if args.quick:
        specs = [s for s in specs if "480x640" not in s["name"]]
    print(f"FLOPs, traffic and profile on {', '.join(d.upper() for d in devices)} ...", flush=True)
    report = {"machine": machine(), "repeats": repeats,
              **flop_report(specs, devices, repeats, recorded, peak=peak)}
    out.mkdir(parents=True, exist_ok=True)
    lines = flops_lines(report)
    print("\n" + "\n".join(lines))
    failed = []
    for name, text in (("flops.json", json.dumps(report, indent=2)), ("flops_output.txt", "\n".join(lines) + "\n")):
        try:
            (out / name).write_text(text)
        except OSError as e:                                    # e.g. the file is open in a viewer on Windows
            failed.append(f"{out / name}: {e.strerror or e}")
    print(f"\nSaved {out / 'flops.json'} and flops_output.txt")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
