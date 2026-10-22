"""Fine-tune a CNN to predict each tree's winter shape from its summer photo, with the row-grouped folds.

Fixed ImageNet features (:mod:`treeskel.shape_model`) predicted the winter
shape barely better than the mean shape: skill at most about 0.1. The hand
masks showed that a person can read the V from the same photos (skill +0.87
on the V opening). So here the network itself is trained (fine-tuned):
ResNet-18, pretrained on ImageNet, with its classifier replaced by a linear
layer that outputs the ten shape measures of :mod:`treeskel.shape`. It learns
from the summer photos of the training rows and their winter measures only;
the hand masks stay for validation (the student's choice, session 10).

**Targets.** Each measure is centred on the training trees' mean and divided
by a spread it shares with its mirror partner (:func:`target_scale`), so that
a mirror image's targets are exactly the mirrored standardised targets. The
loss is the squared error averaged over the measures a tree has
(:func:`masked_mse`): a tree that splits at the ground has no trunk axis.

**Augmentation** (:func:`augment`, every training step): a random crop of
80–100% of the photo's area, resized back (a change of scale or position
changes no angle); jitter in brightness, contrast and saturation; and, half
of the time, the mirror image with the mirrored targets. A prediction
averages the photo's and the mirror of its mirror image's (:func:`predict`).

**Validation** (:func:`cross_validate_cnn`): the 5 row-grouped folds of
``splits.json``.

1. Within each fold's training rows, the next fold's rows are held out, and
   the number of epochs is chosen where their loss is lowest.
2. The network is trained again, from the ImageNet weights, on all the
   training rows for that many epochs.
3. It predicts the fold's own rows.

The rows being predicted never influence a choice, and the recipe
(:class:`CNNParams`) was fixed before any result.

**Variants** (the student's choice after the first run, session 10): the
summer depth as a fourth input channel (``--depth``; the first layer's
weights for it start as the mean of its colour weights, and the colour
jitter leaves it alone), and the photos at full size (``--size 480 640``).
Each variant saves to its own folder. They are all reported, since choosing
the best on the same folds would flatter it.

Usage, from the command line:

```
python -m treeskel.shape_cnn data/raw                  # the student's PC (GPU); docs/results/shape_cnn/
python -m treeskel.shape_cnn data/raw --depth          # docs/results/shape_cnn_depth/
python -m treeskel.shape_cnn data/raw --size 480 640   # docs/results/shape_cnn_480x640/
python -m treeskel.shape_cnn --replot docs/results/shape_cnn docs/results/shape_cnn_depth   # figures only
```
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from treeskel.shape import IMAGENET_MEAN, IMAGENET_STD, MEASURES
from treeskel.shape_model import HEADLINE, TRUNK_RELATIVE, VERTICAL, evaluate, load_table, mirror_matrix, print_scores


@dataclass(frozen=True)
class CNNParams:
    """The training recipe, fixed before any result.

    Attributes:
        size: ``(height, width)`` the summer photos are resized to (half size,
            as for the fixed features of :mod:`treeskel.shape`).
        epochs: Most epochs tried when choosing the number of epochs.
        batch: Photos per training step.
        lr_backbone: Learning rate of the pretrained layers (AdamW).
        lr_head: Learning rate of the new output layer.
        weight_decay: AdamW's weight decay.
        crop: Smallest random crop, as a fraction of the photo's area.
        jitter: Brightness, contrast and saturation are scaled by up to
            ``1 ± jitter``.
        seed: Random seed (each training run adds its own offset).
        depth: Also give the network the summer depth, as a fourth channel.
    """

    size: tuple[int, int] = (240, 320)
    epochs: int = 40
    batch: int = 16
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    crop: float = 0.8
    jitter: float = 0.2
    seed: int = 0
    depth: bool = False


DEPTH_MEAN, DEPTH_STD = float(IMAGENET_MEAN.mean()), float(IMAGENET_STD.mean())
"""Normalisation of the depth channel (8-bit depth scaled to [0, 1]): the ImageNet colour statistics
averaged over the three channels, the scale the pretrained weights it starts from expect."""


def target_scale(y: np.ndarray, names: Sequence[str] = MEASURES) -> tuple[np.ndarray, np.ndarray]:
    """Centre and spread for standardising the targets, the spread shared by mirror partners.

    With the left and right bends (and the other pairs) sharing one spread,
    standardising commutes with mirroring: ``((y − μ) @ M) / s = ((y − μ) / s) @ M``.

    Args:
        y: ``(n, k)`` training targets (NaN = unknown).
        names: The measures of the columns.

    Returns:
        ``(mu, s)``, each ``(k,)``.
    """
    with warnings.catch_warnings():                       # a measure no training tree has stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(y, axis=0)
        sd = np.nanstd(y, axis=0)
    partner = np.abs(mirror_matrix(names)).argmax(axis=0)  # the measure each one's mirror comes from
    s = (sd + sd[partner]) / 2
    s[~np.isfinite(s) | (s < 1e-6)] = 1.0
    return mu, s


def mirror_columns(names: Sequence[str] = MEASURES) -> tuple[np.ndarray, np.ndarray]:
    """How to mirror a row of measures by moving and negating columns: ``mirrored = y[:, src] * sign``.

    Unlike multiplying by the mirror matrix, this keeps an unknown (NaN)
    measure in its own place instead of spreading it (NaN × 0 is NaN).

    Args:
        names: The measures, in column order.

    Returns:
        ``(src, sign)``: the column each mirrored column comes from, and its sign (±1).

    Example:
        >>> src, sign = mirror_columns(("left_bend", "right_bend", "lean"))
        >>> (np.array([[1.0, np.nan, 3.0]])[:, src] * sign).tolist()
        [[nan, 1.0, -3.0]]
    """
    m = mirror_matrix(names)
    src = np.abs(m).argmax(axis=0)
    return src, m[src, np.arange(len(names))]


def masked_mse(pred, target):
    """Mean squared error over the known targets (NaN in ``target`` is skipped).

    Args:
        pred: ``(B, k)`` predictions (tensor).
        target: ``(B, k)`` targets, NaN where unknown.

    Returns:
        The loss, a scalar tensor (0 if no target is known).
    """
    import torch

    known = torch.isfinite(target)
    diff = torch.where(known, pred - torch.nan_to_num(target), torch.zeros_like(pred))
    return (diff ** 2).sum() / known.sum().clamp(min=1)


def normalise(x):
    """``(B, 3, H, W)`` floats in [0, 1] to the ImageNet normalisation; a fourth (depth) channel uses
    :data:`DEPTH_MEAN` and :data:`DEPTH_STD`.

    Args:
        x: ``(B, 3 or 4, H, W)`` tensor in [0, 1].

    Returns:
        The normalised tensor.
    """
    import torch

    mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    if x.shape[1] == 4:
        mean, std = mean + [DEPTH_MEAN], std + [DEPTH_STD]
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


def augment(x, generator, params: CNNParams = CNNParams(), flip_p: float = 0.5):
    """Random crop, mirror image and colour jitter for a batch of photos.

    The crop and the mirror are one affine resampling per photo: a square
    fraction ``f`` of each side (``f² ∈ [crop, 1]`` of the area) at a random
    position, with the x axis reversed for a mirror image.

    Args:
        x: ``(B, 3, H, W)`` floats in [0, 1], or ``(B, 4, H, W)`` with the
            depth last (cropped and mirrored with the photo, but not jittered).
        generator: A ``torch.Generator`` on ``x``'s device, for repeatable runs.
        params: ``crop`` and ``jitter``.
        flip_p: Chance of the mirror image.

    Returns:
        ``(images, flipped)``: normalised images of ``x``'s shape and a
        ``(B,)`` bool tensor, True where the mirror image was taken.
    """
    import torch
    import torch.nn.functional as F

    b, dev = x.shape[0], x.device

    def uniform(lo, hi):
        """``b`` random numbers between ``lo`` and ``hi``, one per image."""
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=generator)

    f = torch.sqrt(uniform(params.crop, 1.0))                    # side fraction
    tx, ty = uniform(-1.0, 1.0) * (1 - f), uniform(-1.0, 1.0) * (1 - f)
    flipped = torch.rand(b, device=dev, generator=generator) < flip_p
    sx = torch.where(flipped, -f, f)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = sx, tx, f, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    j = params.jitter
    if j > 0:
        rgb, rest = x[:, :3], x[:, 3:]                                          # depth is not jittered
        rgb = rgb * uniform(1 - j, 1 + j).view(b, 1, 1, 1)                     # brightness
        mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
        rgb = (rgb - mean) * uniform(1 - j, 1 + j).view(b, 1, 1, 1) + mean      # contrast
        grey = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).unsqueeze(1)
        rgb = grey + (rgb - grey) * uniform(1 - j, 1 + j).view(b, 1, 1, 1)      # saturation
        x = torch.cat([rgb.clamp(0, 1), rest], dim=1)
    return normalise(x), flipped


def make_model(n_out: int, pretrained: bool = True, channels: int = 3):
    """ResNet-18 with its classifier replaced by a linear layer of ``n_out`` outputs.

    The new layer starts with small weights and zero bias, so the first
    predictions are near the training mean (0 in standardised units). With
    ``channels=4`` the first convolution takes a depth channel too; its
    weights for it start as the mean of its (pretrained) colour weights.

    Args:
        n_out: Outputs (one per measure).
        pretrained: Load the ImageNet weights.
        channels: Input channels: 3 (colour) or 4 (colour and depth).

    Returns:
        The model, a ``torch.nn.Module``.

    Raises:
        ValueError: If ``channels`` is not 3 or 4.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if channels == 4:
        old = net.conv1
        net.conv1 = torch.nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            net.conv1.weight[:, :3] = old.weight
            net.conv1.weight[:, 3] = old.weight.mean(dim=1)
    elif channels != 3:
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    net.fc = torch.nn.Linear(net.fc.in_features, n_out)
    torch.nn.init.normal_(net.fc.weight, std=0.01)
    torch.nn.init.zeros_(net.fc.bias)
    return net


def _as_float(x_uint8):
    """``(B, 3, H, W)`` uint8 tensor to floats in [0, 1]."""
    return x_uint8.float().div_(255.0)


def predict(model, x_uint8, mirror: tuple, batch: int = 32) -> np.ndarray:
    """Standardised predictions, each the average of the photo's and its mirror image's.

    Args:
        model: The network.
        x_uint8: ``(N, 3, H, W)`` uint8 tensor on the model's device.
        mirror: ``(src, sign)`` from :func:`mirror_columns`, as tensors on
            the same device.
        batch: Photos per forward pass.

    Returns:
        ``(N, k)`` array.
    """
    import torch

    src, sign = mirror
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x_uint8), batch):
            x = normalise(_as_float(x_uint8[i:i + batch]))
            out.append(((model(x) + model(x.flip(-1))[:, src] * sign) / 2).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(src)))


def train(model, x_uint8, z: np.ndarray, epochs: int, params: CNNParams, mirror, seed: int,
          val: tuple | None = None, on_epoch: Callable[[dict], None] | None = None) -> list[dict]:
    """Train ``model`` in place: AdamW with a cosine learning-rate decay over ``epochs``.

    Args:
        model: The network, on its device.
        x_uint8: ``(N, 3, H, W)`` uint8 tensor of training photos on the device.
        z: ``(N, k)`` standardised targets (NaN = unknown).
        epochs: Epochs to train.
        params: The recipe.
        mirror: ``(src, sign)`` from :func:`mirror_columns`, as tensors on the device.
        seed: Seed for the batch order and the augmentation.
        val: Optional ``(x_uint8, z)`` held out; its loss (with the mirror
            averaging of :func:`predict`) is recorded after each epoch.
        on_epoch: Called with each epoch's record.

    Returns:
        One record per epoch: ``{"epoch", "train_loss", "val_loss"}``.
    """
    import torch

    dev = x_uint8.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    target = torch.as_tensor(z, dtype=torch.float32, device=dev)
    head = list(model.fc.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": params.lr_backbone}, {"params": head, "lr": params.lr_head}],
                            weight_decay=params.weight_decay)
    n = len(x_uint8)
    steps = max(1, math.ceil(n / params.batch)) * max(1, epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, device=dev, generator=gen)
        total, count = 0.0, 0
        for i in range(0, n, params.batch):
            idx = order[i:i + params.batch]
            if len(idx) < 2:                                   # batch norm needs two photos
                continue
            x, flipped = augment(_as_float(x_uint8[idx]), gen, params)
            t = target[idx]
            t = torch.where(flipped.unsqueeze(1), t[:, mirror[0]] * mirror[1], t)   # an unknown stays in its place
            loss = masked_mse(model(x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, count = total + loss.item() * len(idx), count + len(idx)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": None}
        if val is not None:
            pv = torch.as_tensor(predict(model, val[0], mirror), device=dev)
            record["val_loss"] = float(masked_mse(pv, torch.as_tensor(val[1], dtype=torch.float32, device=dev)))
        log.append(record)
        if on_epoch:
            on_epoch(record)
    return log


def _inner_rows(i: int, fold_rows: Sequence[Sequence[int]], train_rows: set[int], rows: np.ndarray) -> list[int] | None:
    """The first following fold whose rows leave at least two training trees and hold at least one (or None)."""
    k = len(fold_rows)
    for step in range(1, k):
        cand = [r for r in fold_rows[(i + step) % k] if r in train_rows]
        held = np.isin(rows, cand)
        if held.any() and (np.isin(rows, list(train_rows)) & ~held).sum() >= 2:
            return cand
    return None


def _n(count, word: str) -> str:
    """``"1 tree"``, ``"2 trees"``."""
    return f"{int(count)} {word}{'' if int(count) == 1 else 's'}"


def cross_validate_cnn(images: np.ndarray, y: np.ndarray, rows: np.ndarray, fold_rows: Sequence[Sequence[int]],
                       params: CNNParams = CNNParams(), device: str | None = None, pretrained: bool = True,
                       names: Sequence[str] = MEASURES, progress: Callable[[str], None] = print) -> dict:
    """Fine-tune and predict every tree from the fold that holds its row out (see the module docstring).

    Args:
        images: ``(n, H, W, 3)`` uint8 summer photos, already at ``params.size``
            (``(n, H, W, 4)`` with the depth last, for ``params.depth``).
        y: ``(n, k)`` winter measures (NaN = unknown).
        rows: ``(n,)`` orchard rows.
        fold_rows: Each fold's held-out rows (``splits.json``).
        params: The recipe.
        device: ``"cuda"``, ``"cpu"``, or None for the GPU if there is one.
        pretrained: Start from the ImageNet weights (False: random, for tests).
        names: The measures of ``y``'s columns.
        progress: Where progress lines go.

    Returns:
        ``{"cnn": (n, k), "mean": (n, k), "fold": (n,), "epochs": [...], "log": [...], "device": str}``.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params.seed)                          # the new output layer's first weights
    base = make_model(len(names), pretrained, channels=images.shape[3]).to(device)
    start = copy.deepcopy(base.state_dict())
    # A mirrored photo has mirrored measures: left and right swap, and signed angles change sign.
    src, sign = mirror_columns(names)
    mirror = (torch.as_tensor(src, device=device), torch.as_tensor(sign, dtype=torch.float32, device=device))
    x_all = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(device)
    n = len(y)
    pred, mean, fold = np.full((n, len(names)), np.nan), np.full((n, len(names)), np.nan), np.full(n, -1)
    chosen, logs = [], []

    def fresh():
        """The network reset to its starting weights (one copy reused, to save memory)."""
        base.load_state_dict(start)
        return base

    for i, held in enumerate(fold_rows):
        # This fold's rows are predicted; the network trains on the other rows.
        va = np.isin(rows, list(held))
        tr = ~va
        if not va.any() or tr.sum() < 2:
            continue
        t0 = time.time()
        # The number of epochs is chosen inside the training rows: the next fold's rows are held out, and the
        # epoch with the lowest loss on them is kept. The fold's own rows play no part in the choice.
        train_rows = set(rows[tr].tolist())
        inner = _inner_rows(i, fold_rows, train_rows, rows)
        best = params.epochs
        if inner is not None:
            iv = tr & np.isin(rows, inner)
            it = tr & ~iv
            # The measures are standardised with the training trees' own mean and spread.
            mu, s = target_scale(y[it], names)
            z = (y - mu) / s
            progress(f"Fold {i + 1}/{len(fold_rows)} (rows {sorted(map(int, held))}, {_n(va.sum(), 'tree')}): "
                     f"choosing the epochs on rows {sorted(map(int, inner))} ({_n(iv.sum(), 'tree')})...")
            log = train(fresh(), x_all[torch.from_numpy(np.nonzero(it)[0]).to(device)], z[it], params.epochs, params,
                        mirror, params.seed + 100 * i, val=(x_all[torch.from_numpy(np.nonzero(iv)[0]).to(device)], z[iv]),
                        on_epoch=lambda r: (progress(f"  epoch {r['epoch']:3d}: train {r['train_loss']:.3f}, "
                                                     f"held out {r['val_loss']:.3f}")
                                            if r["epoch"] % 5 == 0 or r["epoch"] == 1 else None))
            logs += [{"fold": i, "stage": "choose", **r} for r in log]
            best = int(min(log, key=lambda r: r["val_loss"])["epoch"])
            progress(f"  lowest held-out loss after {_n(best, 'epoch')}")
        else:
            progress(f"Fold {i + 1}/{len(fold_rows)}: no rows to hold out inside the training rows; "
                     f"training for {_n(best, 'epoch')}")
        # Then a fresh network trains on all the training rows for that many epochs and predicts the fold.
        mu, s = target_scale(y[tr], names)
        z = (y - mu) / s
        idx = torch.from_numpy(np.nonzero(tr)[0]).to(device)
        log = train(fresh(), x_all[idx], z[tr], best, params, mirror, params.seed + 100 * i + 50)
        logs += [{"fold": i, "stage": "final", **r} for r in log]
        zp = predict(base, x_all[torch.from_numpy(np.nonzero(va)[0]).to(device)], mirror)
        pred[va] = mu + s * zp                              # back from standardised units to degrees
        # The baseline: the training trees' mean shape (a measure unknown on every tree stays NaN).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean[va] = np.nanmean(y[tr], axis=0)
        fold[va] = i
        chosen.append({"fold": i, "epochs": best, "inner_rows": inner, "seconds": round(time.time() - t0, 1)})
        progress(f"  trained on {_n(tr.sum(), 'tree')} for {_n(best, 'epoch')}; fold done in {time.time() - t0:.0f} s")
    return {"cnn": pred, "mean": mean, "fold": fold, "epochs": chosen, "log": logs, "device": device}


def load_photos(records: dict, ids: Sequence[str], size: tuple[int, int], progress: Callable[[str], None] = print,
                depth: bool = False) -> np.ndarray:
    """The summer photos of ``ids``, resized (bilinear) to ``size`` = ``(height, width)``.

    Args:
        records: Tree records by id.
        ids: The trees to load.
        size: ``(height, width)``.
        progress: Prints a progress line.
        depth: Add the summer depth as a fourth channel.

    Returns:
        ``(n, H, W, 3)`` uint8, or ``(n, H, W, 4)`` with the summer depth last if ``depth``.
    """
    from PIL import Image

    from treeskel.io import load_tree

    out = np.zeros((len(ids), size[0], size[1], 4 if depth else 3), np.uint8)
    step = max(1, len(ids) // 10)
    for i, t in enumerate(ids):
        summer = load_tree(records[t], "summer")
        out[i, ..., :3] = np.asarray(Image.fromarray(summer["rgb"]).resize(size[::-1], Image.BILINEAR))
        if depth:
            out[i, ..., 3] = np.asarray(Image.fromarray(summer["depth"]).resize(size[::-1], Image.BILINEAR))
        if (i + 1) % step == 0 or i + 1 == len(ids):
            progress(f"  {i + 1}/{len(ids)} photos loaded")
    return out


def _ridge_predictions(path: Path, trees: Sequence[str], names: Sequence[str]) -> np.ndarray | None:
    """The ridge model's predictions (``docs/results/shape/predictions.csv``) for comparison, or None."""
    if not path.exists():
        return None
    got = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "ridge rgb" in r and r["ridge rgb"] != "":
                got[(r["tree"], r["measure"])] = float(r["ridge rgb"])
    out = np.array([[got.get((t, k), np.nan) for k in names] for t in trees], float)
    return out if np.isfinite(out).any() else None


def variant(params: CNNParams) -> str:
    """The run's name, which is also its results folder: ``shape_cnn``, ``shape_cnn_depth``, ``shape_cnn_480x640``, ...

    Args:
        params: The run's settings.

    Returns:
        The name.
    """
    name = "shape_cnn" + ("_depth" if params.depth else "")
    return name if tuple(params.size) == CNNParams.size else f"{name}_{params.size[0]}x{params.size[1]}"


SERIES_STYLE = {"hand masks": ("#2a78d6", "o"), "ridge rgb": ("#eb6834", "s"), "shape_cnn": ("#1baf7a", "^"),
                "shape_cnn_depth": ("#eda100", "D"), "shape_cnn_480x640": ("#e87ba4", "v")}
"""Colour and marker of each model in the comparison figure (the project's validated categorical slots
in order, with a different marker each, so no model is told apart by colour alone)."""

SERIES_NAME = {"hand masks": "Hand masks (a person's reading of the summer photo)",
               "ridge rgb": "Ridge on fixed ResNet-18 features",
               "shape_cnn": "ResNet-18 fine-tuned (photo, 240x320)",
               "shape_cnn_depth": "ResNet-18 fine-tuned (photo and depth)",
               "shape_cnn_480x640": "ResNet-18 fine-tuned (photo, 480x640)"}


def plot_comparison(series: Sequence[tuple[str, dict]], path: Path, failed: list[str],
                    title: str = "Fine-tuning reads the trunk and some of the V; a person reading the photo still "
                                 "does far better",
                    subtitle: str = "449 development trees, each predicted by the fold that held its row out",
                    names: dict[str, str] | None = None, styles: dict[str, tuple[str, str]] | None = None) -> None:
    """Skill over the mean shape per measure, one marker per model, whiskers over the 5 folds.

    Args:
        series: ``(key, scores)`` pairs in drawing order; ``scores`` is a
            report's ``{"measure": {model: evaluate(...)}}`` slice for that
            model (``{measure: evaluate(...)}``).
        path: Where to save the figure.
        failed: Collects paths that could not be written.
        title: The figure's title.
        subtitle: The line under it.
        names: Legend names for keys beyond :data:`SERIES_NAME` (or to override them).
        styles: ``(colour, marker)`` for keys beyond :data:`SERIES_STYLE` (or to override them).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.shape_model import AXIS, GRID, INK, LABELS, MUTED, SECONDARY, SURFACE, _save, _style

    names, styles = {**SERIES_NAME, **(names or {})}, {**SERIES_STYLE, **(styles or {})}
    groups = (("No reference", HEADLINE), ("From the trunk", TRUNK_RELATIVE), ("From the vertical", VERTICAL))
    rows_y, labels, pos = {}, [], 0.0
    for _, members in groups:
        for k in members:
            rows_y[k] = -pos
            labels.append(LABELS[k])
            pos += 1.0
        pos += 0.6
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    n = len(series)
    for i, (key, scores) in enumerate(series):
        colour, marker = styles.get(key, (MUTED, "o"))
        off = (i - (n - 1) / 2) * 0.16
        for k, y0 in rows_y.items():
            sc = scores.get(k, {})
            if sc.get("n", 0) < 3:
                continue
            fs = [v for v in sc.get("fold_skill", []) if v is not None]
            if fs:
                ax.plot([min(fs), max(fs)], [y0 - off] * 2, color=colour, lw=0.9, alpha=0.7, zorder=2,
                        solid_capstyle="round")
            ax.plot(sc["skill"], y0 - off, marker=marker, ms=5.5, color=colour, mec=SURFACE, mew=0.8, ls="none",
                    zorder=3, label=names.get(key, key) if k == HEADLINE[0] else None)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(list(rows_y.values()), labels, fontsize=8, color=SECONDARY)
    ax.set_xlim(-0.25, 1.0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    gy = 0.0
    for group, members in groups:
        ax.text(-0.25, gy + 0.55, group, fontsize=7.5, color=MUTED, ha="left", va="bottom", style="italic")
        gy -= len(members) + 0.6
    ax.set_xlabel("Skill over the mean shape: 1 − MSE / MSE(mean shape)   (lines: the range over the 5 folds)",
                  fontsize=8, color=SECONDARY)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=8, color=SECONDARY)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2,
               frameon=False, fontsize=7.5, labelcolor=SECONDARY)
    fig.tight_layout(rect=(0, 0.07 + 0.025 * max(0, (n - 1) // 2), 1, 1))
    _save(fig, path, failed)
    plt.close(fig)


def plot_training(log: Sequence[dict], chosen: Sequence[dict], path: Path, failed: list[str], title: str,
                  ylabel: str = "Loss (standardised squared error)") -> None:
    """Loss against epoch for each fold's choice of epochs: the training loss and the held-out rows' loss.

    A falling training loss with a flat held-out loss is the network learning its training photos
    rather than the trees; the dashed line marks the epochs chosen.

    Args:
        log: The run's training log: one row per fold, stage and epoch, with its losses.
        chosen: The epochs each fold chose.
        path: The figure file to write.
        failed: Files that could not be written are added to it (a viewer can lock a file on Windows).
        title: The figure's title.
        ylabel: The loss axis's label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.shape_model import GRID, INK, MUTED, SECONDARY, SURFACE, _save, _style

    folds = sorted({int(r["fold"]) for r in log if r["stage"] == "choose"})
    if not folds:
        return
    fig, axes = plt.subplots(1, len(folds), figsize=(2.2 * len(folds) + 0.6, 2.8), dpi=150, sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    best = {int(c["fold"]): c["epochs"] for c in chosen}
    top = max(float(r["train_loss"]) for r in log if r["stage"] == "choose")
    top = max(top, max(float(r["val_loss"]) for r in log if r["stage"] == "choose" and r["val_loss"] not in ("", None)))
    for ax, f in zip(axes[0], folds):
        _style(ax)
        rows = [r for r in log if int(r["fold"]) == f and r["stage"] == "choose"]
        ep = [int(r["epoch"]) for r in rows]
        ax.plot(ep, [float(r["train_loss"]) for r in rows], color=MUTED, lw=1.2, label="training photos")
        ax.plot(ep, [float(r["val_loss"]) for r in rows], color="#2a78d6", lw=1.6, label="held-out rows")
        if f in best:
            ax.axvline(best[f], color=SECONDARY, lw=0.8, ls=(0, (3, 3)))
        ax.set_ylim(0, top * 1.05)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(f"Fold {f + 1}: {best.get(f, '?')} epochs chosen", loc="left", fontsize=7.5, color=SECONDARY)
        ax.set_xlabel("Epoch", fontsize=7.5, color=SECONDARY)
    axes[0][0].set_ylabel(ylabel, fontsize=7.5, color=SECONDARY)
    axes[0][0].legend(frameon=False, fontsize=7, labelcolor=SECONDARY, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, path, failed)
    plt.close(fig)


def compare_runs(dirs: Sequence[Path], names: Sequence[str] = MEASURES) -> dict:
    """Each run against the first, tree by tree, on the same trees and folds.

    The folds' skills swing a lot (the V opening's from −0.15 to +0.44 in one
    run), so differences between runs are judged tree by tree: the fraction
    of trees a run predicts more closely than the first, a Wilcoxon
    signed-rank test on the paired absolute errors, and the number of folds
    whose skill rose.

    Args:
        dirs: Results folders (``predictions.csv`` and ``report.json`` each);
            the first is the reference.
        names: The measures.

    Returns:
        ``{run: {measure: {"skill", "reference_skill", "closer", "wilcoxon_p",
        "folds_better", "n_folds"}}}`` for every run after the first.
    """
    from scipy import stats

    def load(d):
        """A run's ``{(tree, measure): (winter, predicted)}`` and its report."""
        rows = {}
        with open(Path(d) / "predictions.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["winter"] and r["cnn fine-tuned"]:
                    rows[(r["tree"], r["measure"])] = (float(r["winter"]), float(r["cnn fine-tuned"]))
        return rows, json.loads((Path(d) / "report.json").read_text())

    ref, ref_report = load(dirs[0])
    out = {}
    for d in dirs[1:]:
        run, report = load(d)
        out[Path(d).name] = {}
        for k in names:
            keys = [key for key in ref if key[1] == k and key in run]
            if len(keys) < 3:
                continue
            e0 = np.array([abs(ref[key][1] - ref[key][0]) for key in keys])
            e1 = np.array([abs(run[key][1] - run[key][0]) for key in keys])
            diff = e0 - e1
            f0 = ref_report["scores"][k]["cnn fine-tuned"].get("fold_skill", [])
            f1 = report["scores"][k]["cnn fine-tuned"].get("fold_skill", [])
            pairs = [(a, b) for a, b in zip(f0, f1) if a is not None and b is not None]
            out[Path(d).name][k] = {
                "skill": report["scores"][k]["cnn fine-tuned"]["skill"],
                "reference_skill": ref_report["scores"][k]["cnn fine-tuned"]["skill"],
                "closer": float(np.mean(diff > 0)),
                "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                "folds_better": int(sum(b > a for a, b in pairs)), "n_folds": len(pairs)}
    return out


def print_comparison(comparison: dict, reference: str) -> None:
    """Print :func:`compare_runs`'s result.

    Args:
        comparison: From :func:`compare_runs`.
        reference: The run the others are compared with.
    """
    for run, measures in comparison.items():
        print(f"\n{run} against {reference} (skill: reference -> run; trees predicted more closely; "
              "Wilcoxon p; folds whose skill rose):")
        for k, c in measures.items():
            print(f"  {k:12s} {c['reference_skill']:+.2f} -> {c['skill']:+.2f}   {c['closer']:4.0%}   "
                  f"p {c['wilcoxon_p']:.1e}   {c['folds_better']}/{c['n_folds']}")


def replot(dirs: Sequence[Path], figures: Path) -> list[str]:
    """The figures from saved results: a comparison of every run in ``dirs`` and each run's learning curves.

    Args:
        dirs: Saved runs' results folders.
        figures: The folder for the figures.

    Returns:
        Paths that could not be written.
    """
    failed: list[str] = []
    figures.mkdir(parents=True, exist_ok=True)
    reports = {Path(d).name: json.loads((Path(d) / "report.json").read_text()) for d in dirs}
    first = next(iter(reports.values()))
    series = [("hand masks", {k: v["hand masks"] for k, v in first["scores"].items()})]
    series += [(name, {k: v["cnn fine-tuned"] for k, v in rep["scores"].items()}) for name, rep in reports.items()]
    if all("ridge rgb" in v for v in first["scores"].values()):
        series.append(("ridge rgb", {k: v["ridge rgb"] for k, v in first["scores"].items()}))
    plot_comparison(series, figures / "shape_cnn_skill.png", failed)
    for d in dirs:
        with open(Path(d) / "training_log.csv", newline="") as f:
            log = list(csv.DictReader(f))
        name = Path(d).name
        plot_training(log, reports[name]["epochs"], figures / f"{name}_training.png", failed,
                      f"{SERIES_NAME.get(name, name)}: loss while choosing the epochs")
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    """Fine-tune on the development rows, fold by fold, and save the scores (or, with ``--replot``, only redraw
    the figures from saved results).

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if the table holds test-row trees, too few trees were found, PyTorch
        is missing, the GPU ran out of memory, or an output file could not be written.
    """
    ap = argparse.ArgumentParser(prog="python -m treeskel.shape_cnn", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s) with the summer photos")
    ap.add_argument("--table", default="data/shape/shape_measures.csv", help="winter measures (treeskel.shape)")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", help="results folder (default: docs/results/<the run's name, e.g. shape_cnn_depth>)")
    ap.add_argument("--figures", default="docs/figures", help="folder for the figures")
    ap.add_argument("--ridge", default="docs/results/shape/predictions.csv", help="ridge predictions, for comparison")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    ap.add_argument("--epochs", type=int, default=CNNParams.epochs, help="most epochs tried")
    ap.add_argument("--size", type=int, nargs=2, default=list(CNNParams.size), metavar=("H", "W"))
    ap.add_argument("--depth", action="store_true", help="also give the network the summer depth")
    ap.add_argument("--batch", type=int, default=CNNParams.batch,
                    help="photos per training step (smaller if the GPU runs out of memory; recorded in the report)")
    ap.add_argument("--seed", type=int, default=CNNParams.seed)
    ap.add_argument("--replot", nargs="+", metavar="DIR",
                    help="only redraw the figures from these results folders (and compare them with the first)")
    ap.add_argument("--random-weights", action="store_true", help="skip the ImageNet weights (tests only)")
    args = ap.parse_args(argv)

    if args.replot:
        dirs = [Path(d) for d in args.replot]
        failed = replot(dirs, Path(args.figures))
        print(f"Saved the figures in {args.figures}")
        if len(dirs) > 1:
            comparison = compare_runs(dirs)
            print_comparison(comparison, dirs[0].name)
            path = dirs[0].parent / "shape_cnn_runs.json"
            path.write_text(json.dumps(comparison, indent=2))
            print(f"Saved {path}")
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if not args.data_dirs:
        ap.error("give the dataset folder(s), or --replot")

    from treeskel.io import index_dataset

    params = CNNParams(size=tuple(args.size), epochs=args.epochs, seed=args.seed, depth=args.depth, batch=args.batch)
    split = json.loads(Path(args.split).read_text())
    test_rows = set(split["test_rows"])
    table = load_table(args.table)
    if any(r["row"] in test_rows for r in table):
        print("The table holds test-row trees; they are never used here. Stopping.")
        return 1
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    by_id = {r["tree"]: r for r in table}
    trees = sorted(t for t in by_id if t in records)
    if len(trees) < 3:
        print(f"Only {len(trees)} trees have both measures and photos; nothing to train.")
        return 1
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        print(f"PyTorch and torchvision are needed: {e}")
        return 1
    what = "photos and depth" if params.depth else "photos"
    print(f"{len(trees)} development trees; loading the summer {what} at {params.size[0]}x{params.size[1]}...",
          flush=True)
    images = load_photos(records, trees, params.size, lambda m: print(m, flush=True), depth=params.depth)
    rows = np.array([by_id[t]["row"] for t in trees])
    y = np.array([[by_id[t][f"winter_{k}"] for k in MEASURES] for t in trees], float)
    hand = np.array([[by_id[t][f"hand_{k}"] for k in MEASURES] for t in trees], float)
    t0 = time.time()
    try:
        cv = cross_validate_cnn(images, y, rows, split["fold_val_rows"], params, args.device,
                                pretrained=not args.random_weights, progress=lambda m: print(m, flush=True))
    except (OSError, RuntimeError) as e:                   # a failed weights download, or out of GPU memory
        print(f"Training stopped: {type(e).__name__}: {e}\nIf the weights could not be downloaded, see the note "
              "printed by python -m treeskel.shape when that happens (the file resnet18-f37072fd.pth). If the GPU ran "
              "out of memory, close other programs using it (nvidia-smi lists them) and try again, or add --batch 8.")
        return 1
    ridge = _ridge_predictions(Path(args.ridge), trees, MEASURES)
    report = {"n_trees": len(trees), "variant": variant(params), "params": asdict(params), "device": cv["device"],
              "weights": "random" if args.random_weights else "IMAGENET1K_V1", "epochs": cv["epochs"],
              "minutes": round((time.time() - t0) / 60, 1), "scores": {}}
    for j, k in enumerate(MEASURES):
        base = cv["mean"][:, j]
        mean_shape = evaluate(y[:, j], base, base, cv["fold"])
        mean_shape["r"] = None
        sc = {"mean shape": mean_shape, "hand masks": evaluate(y[:, j], hand[:, j], base, cv["fold"]),
              "cnn fine-tuned": evaluate(y[:, j], cv["cnn"][:, j], base, cv["fold"])}
        if ridge is not None:
            sc["ridge rgb"] = evaluate(y[:, j], ridge[:, j], base, cv["fold"])
        report["scores"][k] = sc
    for title, names in (("Headline (no reference direction)", HEADLINE), ("From the trunk's axis", TRUNK_RELATIVE),
                         ("From the photo's vertical (camera roll adds to these)", VERTICAL)):
        print(f"\n{title}:")
        print_scores(report, names)
    out = Path(args.out or f"docs/results/{variant(params)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", "fold", "measure", "winter", "mean shape", "hand masks", "cnn fine-tuned"])
        for i, t in enumerate(trees):
            for j, k in enumerate(MEASURES):
                w.writerow([t, int(rows[i]), int(cv["fold"][i]), k]
                           + ["" if not np.isfinite(v) else f"{v:.2f}"
                              for v in (y[i, j], cv["mean"][i, j], hand[i, j], cv["cnn"][i, j])])
    with open(out / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "stage", "epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(cv["log"])
    print(f"\nTrained in {report['minutes']} min on {cv['device']}. Saved {out / 'report.json'}, "
          f"{out / 'predictions.csv'} and {out / 'training_log.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
