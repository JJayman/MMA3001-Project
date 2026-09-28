"""Segment the main branches in the summer photo, trained on the hand masks and scored against winter.

The angle CNN of :mod:`treeskel.shape_cnn` learned ten numbers per tree and
overfitted (training loss 0.12, held-out 0.7–0.8); the hand masks, a person's
reading of the same photos, predict the winter shape far better (skill +0.87
on the V opening against +0.16). The hand masks are in the summer view and
have the winter shape (448 of 448 laid onto their winter traces within 8 px).
So here a network learns the masks themselves, pixel by pixel, and its
predicted masks are measured with the same code as the hand masks
(:func:`treeskel.shape.mask_branches`, :func:`treeskel.shape.shape_measures`).

**What changed** (the student's decision, session 11): the hand masks of the
development rows are now training labels, not only a validation reference.
The winter traces stay fully independent: they never train or choose
anything here, and the predicted shape is scored against them. The test rows'
masks stay unused until the final evaluation.

**Caveat: the hand masks are not all on the branch.** Session 3 found the
sample's masks 15–45 px from the visible summer trunk. On the 21 development
trees in the repository (session 11, by eye) most sit on the trunk or
within about its width, but several are 20–60 px off to either side, and
R20N06's about 90 px (on the winter position, session 7). So the network
learns the drawn masks, offsets included (unless they are lined up first,
``--mask-overlay``). The pixel scores against the hand
masks mix its error with theirs and are not directly comparable with Chen et
al.'s. The shape measures do not depend on where a mask sits, so the
headline score against the winter traces is unaffected.

**The network** (:class:`UNet`): a U-Net whose encoder is ResNet-18,
pretrained on ImageNet (Chen et al. (2020) used ResNet-34, ``--encoder
resnet34``), and whose decoder upsamples by two five times, joining the
encoder's features at each scale. It sees the masks' frame: the centre
480x480 of the photo, resized to ``size`` x ``size`` (256, the masks' own
resolution, by default).

**Loss** (:func:`seg_loss`): binary cross-entropy plus the soft Dice loss;
branch pixels are only about 6% of the frame, and the Dice term keeps the
thin leaders from being traded for background.

**Augmentation** (:func:`augment_pair`): as for the angle CNN, a random crop
of 80–100% of the area, the mirror image half of the time and colour jitter;
here also a rotation of up to ±10° (the camera's roll between rows reaches a
few degrees). The mask is resampled with the photo, so it stays on the
branch. A prediction averages the photo's and its mirror image's
probabilities (:func:`predict_masks`).

**Validation** (:func:`cross_validate_seg`): the 5 row-grouped folds of
``splits.json``, as :func:`treeskel.shape_cnn.cross_validate_cnn`: the number
of epochs is chosen on the next fold's rows inside the training rows (their
loss against the hand masks), the network is trained again on all training
rows, and it predicts the fold's own rows. The recipe (:class:`SegParams`)
was fixed before any result.

**Scores** (:func:`main`), for every development tree from the fold that held
its row out:

- the predicted mask against the hand mask: IoU, Dice, precision, recall
  (:func:`mask_scores`; comparable with Chen et al.'s pixel scores);
- the shape of the predicted mask against the winter traces, with the skill
  over the mean shape as in :mod:`treeskel.shape_model`. A predicted mask in
  which no fork is found counts as the mean shape (the fold's training mean),
  so failures are not dropped from the score; their number is reported.
- tree by tree against the angle CNN's runs and, for a run on lined-up
  masks, the run on the masks as drawn (:func:`compare_with`).

**Lined-up masks** (``--mask-overlay``, session 11, the student's choice
after the first run): each hand mask moved onto its summer photo with
:mod:`treeskel.annotate_mask`. Only trees lined up (and not skipped) train,
and the pixel scores are against the lined-up masks. The shape scores need
no change: moving a mask does not change its angles.

**Joined masks** (:func:`join_pieces`, :func:`joined_run`, session 12, as
agreed at the end of session 11): the measuring keeps only a mask's largest
piece, so a leader cut off by a few pixels of leaves takes the fork with it
(14 of the lined-up run's 449 masks had no fork). Before measuring, pieces
within :data:`JOIN_GAP` (about a trunk's width, fixed from the hand masks'
geometry, never tuned on winter) are joined to the largest. Every run is
also scored this way, into ``<run>_joined``, and both are reported.

Usage::

    python -m treeskel.segment data/raw                      # the student's PC (GPU); docs/results/segment/
    python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json   # docs/results/segment_lined_up/
    python -m treeskel.segment data/raw --depth              # docs/results/segment_depth/
    python -m treeskel.segment --replot docs/results/segment # figures only
    python -m treeskel.segment --join docs/results/segment_lined_up   # its masks joined: segment_lined_up_joined/
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

from treeskel.shape import MASK_ZOOM, MEASURES, ShapeParams, mask_branches, shape_measures
from treeskel.shape_cnn import _inner_rows, _n, normalise
from treeskel.shape_model import HEADLINE, TRUNK_RELATIVE, VERTICAL, evaluate, load_table, print_scores


@dataclass(frozen=True)
class SegParams:
    """The training recipe, fixed before any result.

    Attributes:
        size: Side of the square input: the photo's centre 480x480, resized.
            256 is the hand masks' own resolution; it must divide by 32.
        encoder: ``"resnet18"`` or ``"resnet34"``.
        epochs: Most epochs tried when choosing the number of epochs.
        batch: Photos per training step.
        lr_encoder: Learning rate of the pretrained encoder (AdamW).
        lr_decoder: Learning rate of the new decoder.
        weight_decay: AdamW's weight decay.
        crop: Smallest random crop, as a fraction of the frame's area.
        rotate: Largest random rotation, degrees either way.
        jitter: Brightness, contrast and saturation are scaled by up to
            ``1 ± jitter``.
        threshold: Probability above which a pixel is branch.
        seed: Random seed (each training run adds its own offset).
        depth: Also give the network the summer depth, as a fourth channel.
    """

    size: int = 256
    encoder: str = "resnet18"
    epochs: int = 60
    batch: int = 8
    lr_encoder: float = 1e-4
    lr_decoder: float = 1e-3
    weight_decay: float = 1e-4
    crop: float = 0.8
    rotate: float = 10.0
    jitter: float = 0.2
    threshold: float = 0.5
    seed: int = 0
    depth: bool = False


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

def _block(cin: int, cout: int):
    """Two 3x3 convolutions, each with batch norm and ReLU."""
    import torch.nn as nn

    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def make_encoder(name: str = "resnet18", pretrained: bool = True, channels: int = 3):
    """A torchvision ResNet (18 or 34) for the encoder; with ``channels=4`` its first convolution takes a
    depth channel too, whose weights start as the mean of the colour weights (as
    :func:`treeskel.shape_cnn.make_model`)."""
    import torch
    from torchvision import models

    if name not in ("resnet18", "resnet34"):
        raise ValueError(f"encoder must be resnet18 or resnet34, got {name!r}")
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet34": models.ResNet34_Weights.IMAGENET1K_V1}[name]
    net = getattr(models, name)(weights=weights if pretrained else None)
    if channels == 4:
        old = net.conv1
        net.conv1 = torch.nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            net.conv1.weight[:, :3] = old.weight
            net.conv1.weight[:, 3] = old.weight.mean(dim=1)
    elif channels != 3:
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    return net


def _unet_class():
    """The :class:`UNet` class, built on first use so that importing this module needs no PyTorch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _UNet(nn.Module):
        """U-Net with a ResNet encoder: features at 1/2, 1/4, 1/8, 1/16 and 1/32 of the input, and a
        decoder that doubles the resolution five times, joining the encoder's features at each scale.
        Outputs one logit per pixel (branch or not) at the input's size."""

        def __init__(self, encoder: str = "resnet18", pretrained: bool = True, channels: int = 3):
            super().__init__()
            net = make_encoder(encoder, pretrained, channels)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # 1/2, 64
            self.pool = net.maxpool
            self.enc = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])   # 1/4 ... 1/32
            skips = (256, 128, 64, 64)                                        # layer3, layer2, layer1, stem
            outs = (256, 128, 64, 32)
            ins = (512,) + outs[:-1]
            self.dec = nn.ModuleList([_block(i + s, o) for i, s, o in zip(ins, skips, outs)])
            self.last = _block(outs[-1], 16)                                  # back to the input's size
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            if x.shape[-1] % 32 or x.shape[-2] % 32:
                raise ValueError(f"the input's sides must divide by 32, got {tuple(x.shape[-2:])}")
            s = self.stem(x)
            feats = [s]
            e = self.pool(s)
            for layer in self.enc:
                e = layer(e)
                feats.append(e)
            d = feats[-1]
            for block, skip in zip(self.dec, feats[-2::-1]):
                d = F.interpolate(d, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                d = block(torch.cat([d, skip], dim=1))
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(self.last(d))[:, 0]

        def groups(self):
            """``(encoder parameters, decoder parameters)``, for their own learning rates."""
            enc = list(self.stem.parameters()) + list(self.enc.parameters())
            ids = {id(p) for p in enc}
            return enc, [p for p in self.parameters() if id(p) not in ids]

    return _UNet


def UNet(encoder: str = "resnet18", pretrained: bool = True, channels: int = 3):
    """A U-Net with a ResNet encoder (see the module docstring); ``pretrained`` loads the ImageNet weights.

    The model maps ``(B, channels, H, W)`` normalised images (sides divisible
    by 32) to ``(B, H, W)`` logits.
    """
    return _unet_class()(encoder, pretrained, channels)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def seg_loss(logits, target):
    """Binary cross-entropy plus soft Dice loss (``1 − 2|P∩T| / (|P| + |T|)``, per photo, smoothed by 1).

    Args:
        logits: ``(B, H, W)`` network outputs.
        target: ``(B, H, W)`` masks in [0, 1] (soft at resampled edges).
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


def augment_pair(x, m, generator, params: SegParams = SegParams(), flip_p: float = 0.5):
    """Random crop, rotation, mirror image and colour jitter, with the mask resampled alike.

    The crop, rotation and mirror are one affine resampling per photo (of a
    square frame, so a rotation keeps its angles): a square fraction ``f`` of
    each side (``f² ∈ [crop, 1]`` of the area) at a random position.

    Args:
        x: ``(B, C, S, S)`` floats in [0, 1] (C = 3, or 4 with the depth last,
            which is not jittered).
        m: ``(B, S, S)`` masks in [0, 1].
        generator: A ``torch.Generator`` on ``x``'s device.
        params: ``crop``, ``rotate`` and ``jitter``.
        flip_p: Chance of the mirror image.

    Returns:
        ``(images, masks)``: normalised images and masks in [0, 1], same shapes.
    """
    import torch
    import torch.nn.functional as F

    b, dev = x.shape[0], x.device

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=generator)

    f = torch.sqrt(uniform(params.crop, 1.0))
    a = torch.deg2rad(uniform(-params.rotate, params.rotate))
    tx, ty = uniform(-1.0, 1.0) * (1 - f), uniform(-1.0, 1.0) * (1 - f)
    sx = torch.where(torch.rand(b, device=dev, generator=generator) < flip_p, -f, f)
    theta = torch.zeros(b, 2, 3, device=dev)
    cos, sin = torch.cos(a), torch.sin(a)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos * sx, -sin * f, tx     # rotation after the mirror and scale
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * sx, cos * f, ty
    both = torch.cat([x, m.unsqueeze(1)], dim=1)
    grid = F.affine_grid(theta, list(both.shape), align_corners=False)
    both = F.grid_sample(both, grid, mode="bilinear", padding_mode="border", align_corners=False)
    x, m = both[:, :-1], both[:, -1].clamp(0, 1)
    j = params.jitter
    if j > 0:
        rgb, rest = x[:, :3], x[:, 3:]
        rgb = rgb * uniform(1 - j, 1 + j).view(b, 1, 1, 1)                     # brightness
        mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
        rgb = (rgb - mean) * uniform(1 - j, 1 + j).view(b, 1, 1, 1) + mean      # contrast
        grey = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).unsqueeze(1)
        rgb = grey + (rgb - grey) * uniform(1 - j, 1 + j).view(b, 1, 1, 1)      # saturation
        x = torch.cat([rgb.clamp(0, 1), rest], dim=1)
    return normalise(x), m


def _float(x_uint8):
    return x_uint8.float().div_(255.0)


def predict_masks(model, x_uint8, batch: int = 16) -> np.ndarray:
    """Branch probabilities, each the average of the photo's and (flipped back) its mirror image's.

    Args:
        model: The network.
        x_uint8: ``(N, C, S, S)`` uint8 tensor on the model's device.
        batch: Photos per forward pass.

    Returns:
        ``(N, S, S)`` float32 probabilities.
    """
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x_uint8), batch):
            x = normalise(_float(x_uint8[i:i + batch]))
            p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
            out.append(p.float().cpu().numpy())
    s = x_uint8.shape[-1]
    return np.concatenate(out) if out else np.zeros((0, s, s), np.float32)


def train_seg(model, x_uint8, m_uint8, epochs: int, params: SegParams, seed: int, val: tuple | None = None,
              on_epoch: Callable[[dict], None] | None = None) -> list[dict]:
    """Train ``model`` in place: AdamW with a cosine learning-rate decay over ``epochs``.

    Args:
        model: The :class:`UNet`, on its device.
        x_uint8: ``(N, C, S, S)`` uint8 tensor of training photos on the device.
        m_uint8: ``(N, S, S)`` uint8 tensor of their masks (0–255).
        epochs: Epochs to train.
        params: The recipe.
        seed: Seed for the batch order and the augmentation.
        val: Optional ``(x_uint8, m_uint8)`` held out; its loss (on the
            mirror-averaged probabilities) is recorded after each epoch.
        on_epoch: Called with each epoch's record.

    Returns:
        One record per epoch: ``{"epoch", "train_loss", "val_loss"}``.
    """
    import torch

    dev = x_uint8.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    enc, dec = model.groups()
    opt = torch.optim.AdamW([{"params": enc, "lr": params.lr_encoder}, {"params": dec, "lr": params.lr_decoder}],
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
            x, m = augment_pair(_float(x_uint8[idx]), _float(m_uint8[idx]), gen, params)
            loss = seg_loss(model(x), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, count = total + loss.item() * len(idx), count + len(idx)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": None}
        if val is not None:
            p = torch.as_tensor(predict_masks(model, val[0])).clamp(1e-6, 1 - 1e-6)
            record["val_loss"] = float(seg_loss(torch.logit(p), _float(val[1]).cpu()))
        log.append(record)
        if on_epoch:
            on_epoch(record)
    return log


def cross_validate_seg(images: np.ndarray, masks: np.ndarray, has_mask: np.ndarray, rows: np.ndarray,
                       fold_rows: Sequence[Sequence[int]], params: SegParams = SegParams(), device: str | None = None,
                       pretrained: bool = True, progress: Callable[[str], None] = print) -> dict:
    """Train and predict every tree from the fold that holds its row out (see the module docstring).

    Args:
        images: ``(n, S, S, C)`` uint8 summer photos in the masks' frame, at ``params.size``.
        masks: ``(n, S, S)`` uint8 hand masks (0–255) at the same size.
        has_mask: ``(n,)`` bool; trees without a hand mask are predicted but never trained on.
        rows: ``(n,)`` orchard rows.
        fold_rows: Each fold's held-out rows (``splits.json``).
        params: The recipe.
        device: ``"cuda"``, ``"cpu"``, or None for the GPU if there is one.
        pretrained: Start from the ImageNet weights (False: random, for tests).
        progress: Where progress lines go.

    Returns:
        ``{"prob": (n, S, S) float32, "fold": (n,), "epochs": [...], "log": [...], "device": str}``.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params.seed)                          # the decoder's first weights
    base = UNet(params.encoder, pretrained, channels=images.shape[3]).to(device)
    start = copy.deepcopy(base.state_dict())
    x_all = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(device)
    m_all = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
    n, s = len(images), images.shape[1]
    prob, fold = np.full((n, s, s), np.nan, np.float32), np.full(n, -1)
    chosen, logs = [], []

    def fresh():
        base.load_state_dict(start)
        return base

    def take(sel):
        idx = torch.from_numpy(np.nonzero(sel)[0]).to(device)
        return x_all[idx], m_all[idx]

    for i, held in enumerate(fold_rows):
        va = np.isin(rows, list(held))
        tr = ~va & has_mask
        if not va.any() or tr.sum() < 2:
            continue
        t0 = time.time()
        inner = _inner_rows(i, fold_rows, set(rows[tr].tolist()), rows[has_mask])
        best = params.epochs
        if inner is not None:
            iv = tr & np.isin(rows, inner)
            it = tr & ~iv
            progress(f"Fold {i + 1}/{len(fold_rows)} (rows {sorted(map(int, held))}, {_n(va.sum(), 'tree')}): "
                     f"choosing the epochs on rows {sorted(map(int, inner))} ({_n(iv.sum(), 'tree')})...")
            log = train_seg(fresh(), *take(it), params.epochs, params, params.seed + 100 * i, val=take(iv),
                            on_epoch=lambda r: (progress(f"  epoch {r['epoch']:3d}: train {r['train_loss']:.3f}, "
                                                         f"held out {r['val_loss']:.3f}")
                                                if r["epoch"] % 5 == 0 or r["epoch"] == 1 else None))
            logs += [{"fold": i, "stage": "choose", **r} for r in log]
            best = int(min(log, key=lambda r: r["val_loss"])["epoch"])
            progress(f"  lowest held-out loss after {_n(best, 'epoch')}")
        else:
            progress(f"Fold {i + 1}/{len(fold_rows)}: no rows to hold out inside the training rows; "
                     f"training for {_n(best, 'epoch')}")
        log = train_seg(fresh(), *take(tr), best, params, params.seed + 100 * i + 50)
        logs += [{"fold": i, "stage": "final", **r} for r in log]
        prob[va] = predict_masks(base, take(va)[0])
        fold[va] = i
        chosen.append({"fold": i, "epochs": best, "inner_rows": inner, "seconds": round(time.time() - t0, 1)})
        progress(f"  trained on {_n(tr.sum(), 'tree')} for {_n(best, 'epoch')}; fold done in {time.time() - t0:.0f} s")
    return {"prob": prob, "fold": fold, "epochs": chosen, "log": logs, "device": device}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def to_mask_size(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """``(S, S)`` probabilities to a 256x256 bool mask (the hand masks' frame), resized bilinearly first."""
    from PIL import Image

    from treeskel.io import MASK_SHAPE

    p = np.asarray(prob, np.float32)
    if p.shape != MASK_SHAPE:
        p = np.asarray(Image.fromarray(p).resize(MASK_SHAPE[::-1], Image.BILINEAR))
    return p > threshold


def mask_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """IoU, Dice, precision and recall of a predicted mask against a hand mask (NaN where undefined)."""
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = float((p & t).sum())
    fp, fn = float((p & ~t).sum()), float((~p & t).sum())

    def ratio(a, b):
        return a / b if b > 0 else float("nan")

    return {"iou": ratio(tp, tp + fp + fn), "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def mask_shape(mask: np.ndarray, params: ShapeParams = ShapeParams()) -> tuple[dict[str, float], dict | None]:
    """The shape measures of a 256x256 mask, measured as the hand masks are.

    Returns:
        ``(measures, branches)``: :func:`treeskel.shape.shape_measures` (NaN
        throughout if no fork is found) and :func:`treeskel.shape.mask_branches`'s paths (or None).
    """
    br = mask_branches(np.asarray(mask, bool))
    if br is None:
        return dict.fromkeys(MEASURES, float("nan")), None
    return shape_measures(br["trunk"], br["left"], br["right"], br["fork_radius"], params, MASK_ZOOM), br


JOIN_GAP = 19.3
"""Largest gap (mask px, nearest pixel centre to nearest pixel centre) that :func:`join_pieces` bridges:
about a trunk's width. It is twice the development hand masks' median trunk radius at the fork (9.67
mask px as :func:`treeskel.shape.mask_branches` measures it, the distance from the centre line to the
nearest background pixel; 448 trees), so up to about 18 background pixels, 36 px of the photo. Fixed
from that geometry before any joined mask was scored (session 12); never tuned on the winter traces."""


def join_pieces(mask: np.ndarray, gap: float = JOIN_GAP) -> tuple[np.ndarray, int]:
    """Join the pieces of a broken predicted mask to its largest piece across gaps of at most ``gap``.

    :func:`treeskel.shape.mask_branches` keeps only the largest piece, so a
    leader cut from the trunk by a few pixels of leaves is lost, and with it
    the fork. Here, starting from the largest piece, the nearest other piece
    is joined if its nearest pixel is within ``gap`` of the joined mask, and
    so on until none is (a piece may come within reach through one joined
    before it). The join is a straight band between the two pieces' nearest
    centre-line (skeleton) points, as wide as the narrower of the two near
    there (the widest point of each centre line within ``gap`` of the join),
    so the centre lines run on through it. Pixels are only added, and a mask
    in one piece is returned unchanged.

    Args:
        mask: ``(H, W)`` bool predicted mask.
        gap: Largest gap bridged (px, nearest pixel centre to nearest pixel
            centre); 0 joins nothing.

    Returns:
        ``(joined mask, number of pieces joined)``.
    """
    from scipy import ndimage as ndi
    from skimage.draw import line as draw_line
    from skimage.morphology import skeletonize

    out = np.asarray(mask, bool).copy()
    filled = ndi.binary_fill_holes(out)
    lab, n = ndi.label(filled, np.ones((3, 3)))
    if n < 2 or gap <= 0:
        return out, 0
    sizes = ndi.sum(filled, lab, range(1, n + 1))
    first = int(np.argmax(sizes)) + 1
    main = lab == first
    rest = [k for k in range(1, n + 1) if k != first]
    half_width = ndi.distance_transform_edt(filled)        # at a centre-line point: the piece's half-width
    joined = 0
    while rest:
        to_main = ndi.distance_transform_edt(~main)         # distance to the nearest pixel of the joined mask
        near = {k: float(to_main[lab == k].min()) for k in rest}
        k = min(near, key=near.get)
        if near[k] > gap:
            break
        piece = lab == k
        # The two nearest centre-line points: one on the piece, one on the joined mask.
        sk_main, sk_piece = skeletonize(main), skeletonize(piece)
        d, (iy, ix) = ndi.distance_transform_edt(~sk_main, return_indices=True)
        ys, xs = np.nonzero(sk_piece)
        j = int(np.argmin(d[ys, xs]))
        a, b = (int(ys[j]), int(xs[j])), (int(iy[ys[j], xs[j]]), int(ix[ys[j], xs[j]]))
        line = np.zeros_like(out)
        line[draw_line(*a, *b)] = True
        width = min(_widest_near(half_width, sk_piece, a, gap), _widest_near(half_width, sk_main, b, gap))
        band = ndi.distance_transform_edt(~line) < max(1.0, width)
        out |= band
        main |= piece | band
        rest.remove(k)
        joined += 1
    return out, joined


def _widest_near(half_width: np.ndarray, skeleton: np.ndarray, point: tuple[int, int], reach: float) -> float:
    """The largest half-width on a centre line within ``reach`` of ``point``: the branch's own half-width
    near a join, not the smaller one where the centre line ends short of a cut face."""
    ys, xs = np.nonzero(skeleton)
    near = np.hypot(ys - point[0], xs - point[1]) <= max(reach, 1.0)
    return float(half_width[ys[near], xs[near]].max())


def fill_failures(pred: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Predictions with each tree whose mask gave no shape at all replaced by the mean shape.

    A tree whose predicted mask has a fork but, say, no trunk axis keeps NaN on
    the trunk measures, as the winter trace and the hand mask of such a tree do.
    """
    out = pred.copy()
    failed = ~np.isfinite(pred).any(axis=1)
    out[failed] = mean[failed]
    return out


def compare_with(seg: dict[tuple[str, str], tuple[float, float]], other: dict[tuple[str, str], tuple[float, float]],
                 names: Sequence[str] = MEASURES) -> dict:
    """The segmentation against another model, tree by tree, on the trees both predict.

    Args:
        seg, other: ``{(tree, measure): (winter, prediction)}``.

    Returns:
        ``{measure: {"n", "closer", "wilcoxon_p", "mae_seg", "mae_other"}}``: the fraction of trees the
        segmentation predicts more closely, and a Wilcoxon signed-rank test on the paired absolute errors.
    """
    from scipy import stats

    out = {}
    for k in names:
        keys = [key for key in seg if key[1] == k and key in other]
        if len(keys) < 3:
            continue
        e_s = np.array([abs(seg[key][1] - seg[key][0]) for key in keys])
        e_o = np.array([abs(other[key][1] - other[key][0]) for key in keys])
        diff = e_o - e_s
        out[k] = {"n": len(keys), "closer": float(np.mean(diff > 0)),
                  "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                  "mae_seg": float(e_s.mean()), "mae_other": float(e_o.mean())}
    return out


def read_predictions(path: Path, column: str) -> dict[tuple[str, str], tuple[float, float]]:
    """``{(tree, measure): (winter, prediction)}`` from a ``predictions.csv`` (this module's or
    :mod:`treeskel.shape_cnn`'s), where both are known."""
    got = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("winter") and r.get(column):
                got[(r["tree"], r["measure"])] = (float(r["winter"]), float(r[column]))
    return got


# --------------------------------------------------------------------------
# Data and figures
# --------------------------------------------------------------------------

def hand_mask(rec, overlays: dict | None = None) -> np.ndarray | None:
    """A tree's 256x256 hand mask: as drawn, or, given the line-up file's ``overlays``
    (:mod:`treeskel.annotate_mask`), moved onto the summer photo. With ``overlays``, a tree that was
    skipped or not lined up yet has none (None), so moved and unmoved masks are never mixed."""
    from treeskel.io import load_mask

    if not rec.has_mask:
        return None
    if overlays is None:
        return load_mask(rec.paths[("summer", "mask")])
    ov = overlays.get(rec.tree_id)
    if ov is None or ov.skip:
        return None
    from treeskel.annotate_mask import move_mask

    return move_mask(load_mask(rec.paths[("summer", "mask")]), ov)


def load_frames(records: dict, ids: Sequence[str], size: int, depth: bool = False,
                progress: Callable[[str], None] = print,
                overlays: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The summer photos and hand masks of ``ids`` in the masks' frame (centre 480x480), at ``size``.

    With ``overlays`` (the mask line-up file), each mask is moved onto its photo
    (:func:`hand_mask`), and trees not lined up count as having none.

    Returns:
        ``(images, masks, has_mask)``: ``(n, size, size, 3 or 4)`` uint8 (depth
        last, nearest-neighbour so no-reading zeros are not blended),
        ``(n, size, size)`` uint8 masks (0–255, bilinear, so soft at the edges;
        zero where there is none) and ``(n,)`` bool.
    """
    from PIL import Image

    from treeskel.io import CROP_COLS, load_tree

    images = np.zeros((len(ids), size, size, 4 if depth else 3), np.uint8)
    masks = np.zeros((len(ids), size, size), np.uint8)
    has = np.zeros(len(ids), bool)
    step = max(1, len(ids) // 10)
    for i, t in enumerate(ids):
        summer = load_tree(records[t], "summer")
        crop = slice(CROP_COLS[0], CROP_COLS[1])
        images[i, ..., :3] = np.asarray(Image.fromarray(np.ascontiguousarray(summer["rgb"][:, crop]))
                                        .resize((size, size), Image.BILINEAR))
        if depth:
            images[i, ..., 3] = np.asarray(Image.fromarray(np.ascontiguousarray(summer["depth"][:, crop]))
                                           .resize((size, size), Image.NEAREST))
        hm = hand_mask(records[t], overlays)
        if hm is not None:
            masks[i] = np.asarray(Image.fromarray(hm.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR))
            has[i] = True
        if (i + 1) % step == 0 or i + 1 == len(ids):
            progress(f"  {i + 1}/{len(ids)} photos loaded")
    return images, masks, has


def save_masks(path: Path, ids: Sequence[str], masks: np.ndarray) -> None:
    """The predicted 256x256 masks of many trees, bit-packed, in one ``.npz``."""
    np.savez_compressed(path, ids=np.array(ids), bits=np.packbits(np.asarray(masks, bool), axis=-1),
                        shape=np.array(masks.shape[1:]))


def load_masks(path: Path) -> dict[str, np.ndarray]:
    """Inverse of :func:`save_masks`: ``{tree: (256, 256) bool}``."""
    with np.load(path) as z:
        w = int(z["shape"][1])
        bits = np.unpackbits(z["bits"], axis=-1)[..., :w].astype(bool)
        return {str(t): bits[i] for i, t in enumerate(z["ids"])}


def example_trees(dice: dict[str, float], k: int = 6) -> list[str]:
    """Trees at evenly spaced quantiles of the Dice score, worst to best, so the figure shows the range
    rather than the best cases."""
    ranked = sorted((v, t) for t, v in dice.items() if np.isfinite(v))
    if not ranked:
        return []
    picks = np.unique(np.round(np.linspace(0, len(ranked) - 1, min(k, len(ranked)))).astype(int))
    return [ranked[i][1] for i in picks]


def draw_example(rec, pred: np.ndarray, trace: dict | None, label: str, scale: float = 0.5,
                 overlays: dict | None = None):
    """One tree as a row of three: the winter trace, the hand mask and the predicted mask, each with the
    lines :func:`treeskel.shape.shape_measures` fits (PIL image, resized by ``scale``, then labelled). With
    ``overlays``, the hand mask is shown lined up (:func:`hand_mask`)."""
    from PIL import Image, ImageDraw, ImageFont

    from treeskel.io import load_tree, mask_to_image_frame
    from treeskel.shape import MASK_ZOOM as ZOOM, draw_shape, fitted_lines, mask_to_image

    params = ShapeParams()
    s = load_tree(rec, "summer")["rgb"]

    def on_summer(mask, name, colour):
        img = s.copy()
        full, _ = mask_to_image_frame(np.asarray(mask, bool))
        img[full] = (0.45 * img[full] + 0.55 * np.array(colour)).astype(np.uint8)
        br = mask_branches(mask)
        if br is None:
            return draw_shape(img, (), {}), f"{name}: no fork found"
        lines = {k: tuple(mask_to_image(np.array(v)))
                 for k, v in fitted_lines(br["trunk"], br["left"], br["right"], br["fork_radius"], params, ZOOM).items()}
        return draw_shape(img, [mask_to_image(br[p]) for p in ("trunk", "left", "right")], lines), name

    tiles = []                                             # (image, label)
    if trace is not None:
        w = load_tree(rec, "winter")["rgb"]
        lines = fitted_lines(trace["trunk"], trace["left"], trace["right"], trace["fork_radius"], params)
        tiles.append((draw_shape(w, (trace["trunk"], trace["left"], trace["right"]), lines), f"{label}: winter trace"))
    hand = hand_mask(rec, overlays)
    tiles.append(on_summer(hand, "hand mask, lined up" if overlays is not None else "hand mask", (60, 120, 255))
                 if hand is not None else (draw_shape(s, (), {}), "no hand mask used"))
    tiles.append(on_summer(pred, "predicted mask", (0, 200, 0)))
    size = (round(tiles[0][0].width * scale), round(tiles[0][0].height * scale))
    out = Image.new("RGB", (size[0] * len(tiles), size[1]))
    dr = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=14)
    for i, (im, text) in enumerate(tiles):
        out.paste(im.resize(size, Image.BILINEAR), (i * size[0], 0))
        dr.text((i * size[0] + 6, 5), text, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


SEGMENT_STYLE = ("#008300", "P")
"""Colour and marker of the segmentation in the comparison figure: the next categorical slot after the
angle CNN's runs (checked with the palette validator: adjacent colours stay apart under colour-vision
deficiencies) and its own marker."""

LINED_UP_STYLE = ("#4a3aa7", "X")
"""The run on the lined-up masks: the slot after it, and its own marker."""

JOINED_STYLE = ("#e34948", "h")
"""The run on the lined-up masks with the pieces of its masks joined (:func:`joined_run`): the last slot, and
its own marker (validated beside the lined-up run's violet; the markers tell the two apart)."""

RUN_STYLES = {"segment": SEGMENT_STYLE, "segment_lined_up": LINED_UP_STYLE, "segment_lined_up_joined": JOINED_STYLE}

RUN_NAMES = {"segment": "U-Net on the hand masks as drawn, its masks measured",
             "segment_lined_up": "U-Net on the lined-up hand masks, its masks measured",
             "segment_lined_up_joined": "The same, pieces of its masks within a trunk's width joined"}


def plot_skill(report: dict, path: Path, failed: list[str], others: Sequence[tuple[str, dict]] = ()) -> None:
    """The comparison figure of :func:`treeskel.shape_cnn.plot_comparison`, with the segmentation added."""
    from treeskel.shape_cnn import plot_comparison

    scores = report["scores"]
    key = report.get("variant") or ("segment_lined_up" if report.get("mask_overlay") else "segment")
    series = [("hand masks", {k: v["hand masks"] for k, v in scores.items()})]
    series += [o for o in others if o[0] != key]
    series.append((key, {k: v["segmentation"] for k, v in scores.items()}))
    what = f"masks joined, {report['no_fork']}" if report.get("join") else _n(report["no_fork"], "U-Net mask")
    plot_comparison(series, path, failed, names=RUN_NAMES, styles=RUN_STYLES,
                    title="A U-Net trained on the hand masks, its masks measured against the winter traces",
                    subtitle=f"{report['n_trees']} trees, each from the fold that held its row out; "
                             f"{what} with no fork count as the mean shape")


def plot_learning(log: Sequence[dict], chosen: Sequence[dict], path: Path, failed: list[str]) -> None:
    """Loss against epoch while choosing the epochs (:func:`treeskel.shape_cnn.plot_training`)."""
    from treeskel.shape_cnn import plot_training

    plot_training(log, chosen, path, failed, "U-Net on the hand masks: loss while choosing the epochs",
                  ylabel="Loss (cross-entropy + Dice)")


def _other_runs(results: Path, source: str | None = None) -> list[tuple[str, dict]]:
    """The angle CNN's saved runs and the segmentation on the masks as drawn, for the comparison figure;
    and ``source``, the run a joined run was scored from."""
    out = []
    runs = [("shape_cnn", "cnn fine-tuned"), ("shape_cnn_480x640", "cnn fine-tuned"), ("segment", "segmentation")]
    if source and source != "segment":
        runs.append((source, "segmentation"))
    for name, model in runs:
        p = results / name / "report.json"
        if p.exists():
            rep = json.loads(p.read_text())
            out.append((name, {k: v[model] for k, v in rep["scores"].items()}))
    return out


def replot(out: Path, figures: Path) -> list[str]:
    """The skill and learning-curve figures from a saved run. Returns paths that could not be written."""
    failed: list[str] = []
    figures.mkdir(parents=True, exist_ok=True)
    report = json.loads((out / "report.json").read_text())
    source = (report.get("join") or {}).get("source")
    plot_skill(report, figures / f"{out.name}_skill.png", failed,
               _other_runs(out.parent, Path(source).name if source else None))
    if (out / "training_log.csv").exists():                 # a joined run has none: it trained nothing
        with open(out / "training_log.csv", newline="") as f:
            plot_learning(list(csv.DictReader(f)), report["epochs"], figures / f"{out.name}_training.png", failed)
    return failed


def variant(params: SegParams, lined_up: bool = False) -> str:
    """The run's name, which is also its results folder: ``segment``, ``segment_depth``, ``segment_resnet34``,
    ``segment_lined_up`` (trained on the lined-up masks), ..."""
    name = "segment" + ("_lined_up" if lined_up else "") + ("_depth" if params.depth else "")
    if params.encoder != SegParams.encoder:
        name += f"_{params.encoder}"
    return name if params.size == SegParams.size else f"{name}_{params.size}"


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

MASK_COLUMNS = ("iou", "dice", "precision", "recall")

OTHER_RUNS = (("shape_cnn", "cnn fine-tuned"), ("shape_cnn_480x640", "cnn fine-tuned"), ("segment", "segmentation"),
              ("segment_lined_up", "segmentation"))
"""The saved runs a new run is compared with tree by tree (results folder, predictions column)."""


def against_runs(out: Path, others: Sequence[tuple[str, str]] = OTHER_RUNS) -> dict:
    """A run's predictions (``out/predictions.csv``) against each other saved run beside it that exists,
    tree by tree (:func:`compare_with`)."""
    mine = read_predictions(out / "predictions.csv", "segmentation")
    got = {}
    for name, column in dict(others).items():
        p = out.parent / name / "predictions.csv"
        if p.exists() and p.resolve() != (out / "predictions.csv").resolve():
            got[name] = compare_with(mine, read_predictions(p, column))
    return got


def print_against(runs: dict) -> None:
    """Print :func:`against_runs`."""
    for name, measures in runs.items():
        print(f"\nAgainst {name}, tree by tree (MAE: {name} -> segmentation; trees predicted more closely; "
              "Wilcoxon p):")
        for k, c in measures.items():
            print(f"  {k:12s} {c['mae_other']:5.1f} -> {c['mae_seg']:5.1f}   {c['closer']:4.0%}   "
                  f"p {c['wilcoxon_p']:.1e}")


def _value(text: str) -> float:
    return float(text) if text else float("nan")


def joined_run(src: Path, gap: float = JOIN_GAP, out: Path | None = None, figures: Path | None = None,
               progress: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """Score a saved run's masks again after joining their pieces (:func:`join_pieces`).

    Everything comes from the run's own files, so no photos are needed:
    ``predicted_masks.npz``, and ``predictions.csv`` for each tree's row, fold,
    winter and hand-mask measures and its fold's mean shape. The result is a
    run like any other (``report.json``, ``predictions.csv``,
    ``mask_scores.csv``, ``predicted_masks.npz``), in ``<run>_joined`` (or
    ``<run>_joined_<gap>`` for another gap), compared tree by tree with the
    run it came from and the other saved runs. Its pixel scores are the
    run's own, of the masks before joining (the hand masks are not needed
    here); ``mask_scores.csv`` lists the pieces joined and pixels added per tree.

    Args:
        src: The saved run's folder.
        gap: Largest gap joined (:data:`JOIN_GAP`, fixed from geometry).
        out: Results folder (default: beside ``src``, as above).
        figures: Where the skill figure and the joins' contact sheet go (None: no figures).
        progress: Where the printed summary goes.

    Returns:
        ``(out, paths that could not be written)``.
    """
    src = Path(src)
    source = json.loads((src / "report.json").read_text())
    masks = load_masks(src / "predicted_masks.npz")
    info: dict[str, dict] = {}
    with open(src / "predictions.csv", newline="") as f:
        for r in csv.DictReader(f):
            t = info.setdefault(r["tree"], {"row": int(r["row"]), "fold": int(r["fold"])})
            t[r["measure"]] = (_value(r["winter"]), _value(r["mean shape"]), _value(r["hand masks"]))
    trees = [t for t in info if t in masks]
    out = Path(out) if out else src.parent / (f"{src.name}_joined" if gap == JOIN_GAP else f"{src.name}_joined_{gap:g}")
    y, mean, hand = (np.array([[info[t][k][c] for k in MEASURES] for t in trees]) for c in range(3))
    folds = np.array([info[t]["fold"] for t in trees])
    seg, seg0 = np.full((len(trees), len(MEASURES)), np.nan), np.full((len(trees), len(MEASURES)), np.nan)
    joined, pieces, added = {}, np.zeros(len(trees), int), np.zeros(len(trees), int)
    for i, t in enumerate(trees):
        joined[t], pieces[i] = join_pieces(masks[t], gap)
        added[i] = int((joined[t] & ~masks[t]).sum())
        m, _ = mask_shape(joined[t])
        seg[i] = [m[k] for k in MEASURES]
        seg0[i] = [mask_shape(masks[t])[0][k] for k in MEASURES] if pieces[i] else seg[i]   # unchanged: once
    before, after = np.isfinite(seg0).any(axis=1), np.isfinite(seg).any(axis=1)   # as main: no measure = no fork
    v = MEASURES.index("v_opening")
    seg_filled = fill_failures(seg, mean)

    report = {k: source[k] for k in ("n_masks", "mask_overlay", "params", "device", "weights", "epochs", "minutes")
              if k in source}
    report.update(n_trees=len(trees), variant=out.name, no_fork=int((~after).sum()),
                  masks=source.get("masks", {}), masks_note="pixel scores of the masks before joining",
                  join={"source": str(src), "gap": gap, "masks_joined": int((pieces > 0).sum()),
                        "pieces_joined": int(pieces.sum()), "pixels_added": int(added.sum()),
                        "no_fork_before": int((~before).sum()), "no_fork_after": int((~after).sum()),
                        "v_opening_before": int(np.isfinite(seg0[:, v]).sum()),
                        "v_opening_after": int(np.isfinite(seg[:, v]).sum()),
                        "forks_found_by_joining": [t for i, t in enumerate(trees) if after[i] and not before[i]],
                        "forks_lost_by_joining": [t for i, t in enumerate(trees) if before[i] and not after[i]]},
                  scores={}, against_runs={})
    for j, k in enumerate(MEASURES):
        base = mean[:, j]
        mean_shape = evaluate(y[:, j], base, base, folds)
        mean_shape["r"] = None
        report["scores"][k] = {"mean shape": mean_shape, "hand masks": evaluate(y[:, j], hand[:, j], base, folds),
                               "segmentation": evaluate(y[:, j], seg_filled[:, j], base, folds),
                               "segmentation, forks found": evaluate(y[:, j], seg[:, j], base, folds)}
    jn = report["join"]
    progress(f"Joined pieces within {gap:g} px in {_n(jn['masks_joined'], 'mask')} of {len(trees)} "
             f"({_n(jn['pieces_joined'], 'piece')}, {jn['pixels_added']} pixels added); masks with no fork "
             f"{jn['no_fork_before']} -> {jn['no_fork_after']}; with a V opening {jn['v_opening_before']} -> "
             f"{jn['v_opening_after']}")
    for title, names in (("Headline (no reference direction)", HEADLINE), ("From the trunk's axis", TRUNK_RELATIVE),
                         ("From the photo's vertical (camera roll adds to these)", VERTICAL)):
        progress(f"\n{title}:")
        print_scores(report, names)

    failed: list[str] = []
    out.mkdir(parents=True, exist_ok=True)

    def write(path: Path, fn) -> None:
        try:
            fn(path)
        except OSError as e:                                # e.g. open in a viewer on Windows
            failed.append(f"{path}: {e.strerror or e}")

    def predictions(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "fold", "measure", "winter", "mean shape", "hand masks", "segmentation"])
            for i, t in enumerate(trees):
                for j, k in enumerate(MEASURES):
                    w.writerow([t, info[t]["row"], info[t]["fold"], k]
                               + ["" if not np.isfinite(v) else f"{v:.2f}"
                                  for v in (y[i, j], mean[i, j], hand[i, j], seg_filled[i, j])])

    pixel = {}
    if (src / "mask_scores.csv").exists():
        with open(src / "mask_scores.csv", newline="") as f:
            pixel = {r["tree"]: r for r in csv.DictReader(f)}

    def mask_table(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "fold", "fork_found", "fork_found_before", "pieces_joined", "pixels_added",
                        *(f"{c}_before_joining" for c in MASK_COLUMNS)])
            for i, t in enumerate(trees):
                w.writerow([t, info[t]["row"], info[t]["fold"], int(after[i]), int(before[i]), int(pieces[i]),
                            int(added[i]), *(pixel.get(t, {}).get(c, "") for c in MASK_COLUMNS)])

    write(out / "predictions.csv", predictions)
    write(out / "mask_scores.csv", mask_table)
    write(out / "predicted_masks.npz", lambda p: save_masks(p, trees, np.stack([joined[t] for t in trees])))
    if (out / "predictions.csv").exists():
        report["against_runs"] = against_runs(out, OTHER_RUNS + ((src.name, "segmentation"),))
        print_against(report["against_runs"])
    write(out / "report.json", lambda p: p.write_text(json.dumps(report, indent=2)))
    if figures is not None:
        figures = Path(figures)
        figures.mkdir(parents=True, exist_ok=True)
        if (out / "report.json").exists():
            failed += replot(out, figures)
        changed = [t for i, t in enumerate(trees) if pieces[i] > 0]
        if changed:
            write(figures / f"{out.name}_masks.png",
                  lambda p: draw_joins({t: masks[t] for t in changed}, {t: joined[t] for t in changed}).save(p))
    progress(f"\nSaved {out}/ (report.json, predictions.csv, mask_scores.csv, predicted_masks.npz)")
    return out, failed


def draw_joins(before: dict[str, np.ndarray], after: dict[str, np.ndarray], columns: int = 8, scale: float = 0.5):
    """A contact sheet of joined masks: each tile the mask (grey), the pixels the join added (magenta) and the
    centre lines and fitted lines found after joining; masks whose V opening could be measured only after
    joining come first, labelled. A PIL image; needs no photos."""
    from PIL import Image, ImageDraw, ImageFont

    from treeskel.shape import draw_shape, fitted_lines

    params = ShapeParams()
    order = sorted(before, key=lambda t: (np.isfinite(mask_shape(before[t])[0]["v_opening"]), t))
    size = round(256 * scale)
    rows = math.ceil(len(order) / columns)
    sheet = Image.new("RGB", (columns * size, rows * size), (20, 20, 20))
    font = ImageFont.load_default(size=11)
    for i, t in enumerate(order):
        b, a = np.asarray(before[t], bool), np.asarray(after[t], bool)
        rgb = np.full(b.shape + (3,), 20, np.uint8)
        rgb[b] = (150, 150, 150)
        rgb[a & ~b] = (235, 80, 235)
        br = mask_branches(a)
        if br is None:
            im = draw_shape(rgb, (), {})
        else:
            lines = fitted_lines(br["trunk"], br["left"], br["right"], br["fork_radius"], params, MASK_ZOOM)
            im = draw_shape(rgb, [br[p] for p in ("trunk", "left", "right")], lines)
        tile = im.resize((size, size), Image.BILINEAR)
        found = not np.isfinite(mask_shape(b)[0]["v_opening"]) and np.isfinite(mask_shape(a)[0]["v_opening"])
        ImageDraw.Draw(tile).text((4, 3), t + ("  V measured" if found else ""), fill=(255, 255, 0), font=font,
                                  stroke_width=1, stroke_fill=(0, 0, 0))
        sheet.paste(tile, ((i % columns) * size, (i // columns) * size))
    return sheet


def main(argv: Sequence[str] | None = None) -> int:
    """Train on the development rows' hand masks, fold by fold, and score the masks and their shape
    (or, with ``--replot``, only redraw the figures from a saved run)."""
    ap = argparse.ArgumentParser(prog="python -m treeskel.segment", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s) with the summer photos and hand masks")
    ap.add_argument("--table", default="data/shape/shape_measures.csv", help="winter measures (treeskel.shape)")
    ap.add_argument("--traces", default="data/shape/winter_traces.npz", help="winter traces, for the example figure")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", help="results folder (default: docs/results/<the run's name>)")
    ap.add_argument("--figures", default="docs/figures", help="folder for the figures")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    ap.add_argument("--epochs", type=int, default=SegParams.epochs, help="most epochs tried")
    ap.add_argument("--size", type=int, default=SegParams.size, help="side of the input (divisible by 32)")
    ap.add_argument("--encoder", default=SegParams.encoder, choices=("resnet18", "resnet34"))
    ap.add_argument("--depth", action="store_true", help="also give the network the summer depth")
    ap.add_argument("--mask-overlay", metavar="JSON",
                    help="train on the hand masks lined up with treeskel.annotate_mask (e.g. "
                         "annotations/mask_overlay.json); only lined-up trees train")
    ap.add_argument("--batch", type=int, default=SegParams.batch,
                    help="photos per training step (smaller if the GPU runs out of memory; recorded in the report)")
    ap.add_argument("--seed", type=int, default=SegParams.seed)
    ap.add_argument("--examples", type=int, default=6, help="trees in the example figure")
    ap.add_argument("--replot", metavar="DIR", help="only redraw the figures from this results folder")
    ap.add_argument("--join", metavar="DIR",
                    help="only score a saved run's masks again with their pieces joined (writes DIR_joined)")
    ap.add_argument("--join-gap", type=float, default=JOIN_GAP,
                    help=f"largest gap joined, mask px (default {JOIN_GAP:g}: about a trunk's width, fixed)")
    ap.add_argument("--random-weights", action="store_true", help="skip the ImageNet weights (tests only)")
    args = ap.parse_args(argv)

    if args.replot:
        failed = replot(Path(args.replot), Path(args.figures))
        print(f"Saved the figures in {args.figures}")
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if args.join:
        out, failed = joined_run(Path(args.join), args.join_gap, Path(args.out) if args.out else None,
                                 Path(args.figures))
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if not args.data_dirs:
        ap.error("give the dataset folder(s), or --replot or --join")
    if args.size % 32:
        ap.error(f"--size must divide by 32, got {args.size}")

    from treeskel.io import index_dataset

    params = SegParams(size=args.size, encoder=args.encoder, epochs=args.epochs, batch=args.batch, seed=args.seed,
                       depth=args.depth)
    split = json.loads(Path(args.split).read_text())
    test_rows = set(split["test_rows"])
    table = load_table(args.table)
    if any(r["row"] in test_rows for r in table):
        print("The table holds test-row trees; they are never used here. Stopping.")
        return 1
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    by_id = {r["tree"]: r for r in table}
    trees = sorted(t for t in by_id if t in records and records[t].row not in test_rows)
    if len(trees) < 3:
        print(f"Only {len(trees)} trees have both measures and photos; nothing to train.")
        return 1
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        print(f"PyTorch and torchvision are needed: {e}")
        return 1
    what = "photos, depth and hand masks" if params.depth else "photos and hand masks"
    print(f"{len(trees)} development trees; loading the summer {what} at {params.size}x{params.size}...", flush=True)
    overlays = None
    if args.mask_overlay:
        from treeskel.summer_gt import load_overlays

        overlays = load_overlays(args.mask_overlay)
        if not overlays:
            print(f"No lined-up masks in {args.mask_overlay}. Stopping.")
            return 1
    images, masks, has = load_frames(records, trees, params.size, params.depth, lambda m: print(m, flush=True),
                                     overlays)
    print(f"{int(has.sum())} of them have a hand mask to train on"
          + (f" (lined up; {sum(1 for t in trees if t in overlays and overlays[t].skip)} skipped, "
             f"{sum(1 for t in trees if records[t].has_mask and t not in overlays)} not lined up yet)"
             if overlays is not None else ""), flush=True)
    rows = np.array([by_id[t]["row"] for t in trees])
    t0 = time.time()
    try:
        cv = cross_validate_seg(images, masks, has, rows, split["fold_val_rows"], params, args.device,
                                pretrained=not args.random_weights, progress=lambda m: print(m, flush=True))
    except (OSError, RuntimeError) as e:                   # a failed weights download, or out of GPU memory
        print(f"Training stopped: {type(e).__name__}: {e}\nIf the GPU ran out of memory, close other programs "
              "using it (nvidia-smi lists them) and try again, or add --batch 4.")
        return 1
    minutes = round((time.time() - t0) / 60, 1)

    # Score every predicted tree: its mask against the hand mask, its shape against the winter trace.
    from treeskel.io import MASK_SHAPE

    done = cv["fold"] >= 0
    pred_masks = np.zeros((len(trees),) + MASK_SHAPE, bool)
    seg = np.full((len(trees), len(MEASURES)), np.nan)
    scores = {t: dict.fromkeys(MASK_COLUMNS, float("nan")) for t in trees}
    for i, t in enumerate(trees):
        if not done[i]:
            continue
        pred_masks[i] = to_mask_size(cv["prob"][i], params.threshold)
        m, _ = mask_shape(pred_masks[i])
        seg[i] = [m[k] for k in MEASURES]
        if has[i]:
            scores[t] = mask_scores(pred_masks[i], hand_mask(records[t], overlays))   # the mask it was trained on
    y = np.array([[by_id[t][f"winter_{k}"] for k in MEASURES] for t in trees], float)
    hand = np.array([[by_id[t][f"hand_{k}"] for k in MEASURES] for t in trees], float)
    mean = np.full_like(y, np.nan)
    for f in sorted(set(cv["fold"][done].tolist())):
        va = cv["fold"] == f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean[va] = np.nanmean(y[~va], axis=0)          # the fold's training rows' mean shape, as shape_cnn
    no_fork = int((done & ~np.isfinite(seg).any(axis=1)).sum())
    seg_filled = fill_failures(seg, mean)

    report = {"n_trees": int(done.sum()), "n_masks": int((done & has).sum()), "variant": variant(params, overlays is not None),
              "mask_overlay": args.mask_overlay,
              "params": asdict(params), "device": cv["device"],
              "weights": "random" if args.random_weights else "IMAGENET1K_V1", "epochs": cv["epochs"],
              "minutes": minutes, "no_fork": no_fork, "masks": {}, "scores": {}, "against_runs": {}}
    for c in MASK_COLUMNS:
        v = np.array([scores[t][c] for t in trees])[done & has]
        v = v[np.isfinite(v)]
        report["masks"][c] = {"median": float(np.median(v)) if len(v) else None,
                              "mean": float(v.mean()) if len(v) else None, "n": int(len(v))}
    folds = cv["fold"]
    for j, k in enumerate(MEASURES):
        base = mean[:, j]
        mean_shape = evaluate(y[:, j], base, base, folds)
        mean_shape["r"] = None
        report["scores"][k] = {"mean shape": mean_shape, "hand masks": evaluate(y[:, j], hand[:, j], base, folds),
                               "segmentation": evaluate(y[:, j], seg_filled[:, j], base, folds),
                               "segmentation, forks found": evaluate(y[:, j], seg[:, j], base, folds)}

    print(f"\nPredicted masks against the hand masks ({report['n_masks']} trees; median, mean):")
    for c in MASK_COLUMNS:
        s = report["masks"][c]
        if s["n"]:
            print(f"  {c:9s} {s['median']:.3f}  {s['mean']:.3f}")
    print(f"No fork found in {_n(no_fork, 'predicted mask')} (scored as the mean shape)")
    for title, names in (("Headline (no reference direction)", HEADLINE), ("From the trunk's axis", TRUNK_RELATIVE),
                         ("From the photo's vertical (camera roll adds to these)", VERTICAL)):
        print(f"\n{title}:")
        print_scores(report, names)

    out = Path(args.out or f"docs/results/{variant(params, overlays is not None)}")
    out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    def write(path: Path, fn) -> None:
        try:
            fn(path)
        except OSError as e:                                # e.g. open in a viewer on Windows
            failed.append(f"{path}: {e.strerror or e}")

    def predictions(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "fold", "measure", "winter", "mean shape", "hand masks", "segmentation"])
            for i, t in enumerate(trees):
                if done[i]:
                    for j, k in enumerate(MEASURES):
                        w.writerow([t, int(rows[i]), int(folds[i]), k]
                                   + ["" if not np.isfinite(v) else f"{v:.2f}"
                                      for v in (y[i, j], mean[i, j], hand[i, j], seg_filled[i, j])])

    def mask_table(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "fold", "fork_found", *MASK_COLUMNS])
            for i, t in enumerate(trees):
                if done[i]:
                    w.writerow([t, int(rows[i]), int(folds[i]), int(np.isfinite(seg[i]).any())]
                               + ["" if not np.isfinite(scores[t][c]) else f"{scores[t][c]:.4f}" for c in MASK_COLUMNS])

    def training_log(path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "stage", "epoch", "train_loss", "val_loss"])
            w.writeheader()
            w.writerows(cv["log"])

    write(out / "predictions.csv", predictions)
    write(out / "mask_scores.csv", mask_table)
    write(out / "training_log.csv", training_log)
    write(out / "predicted_masks.npz", lambda p: save_masks(p, [t for i, t in enumerate(trees) if done[i]],
                                                            pred_masks[done]))
    # Tree by tree against the other runs, on the same trees and folds.
    if (out / "predictions.csv").exists():
        report["against_runs"] = against_runs(out)
        print_against(report["against_runs"])
    write(out / "report.json", lambda p: p.write_text(json.dumps(report, indent=2)))

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists() and (out / "training_log.csv").exists():
        failed += replot(out, figures)
    if args.examples > 0:
        from PIL import Image

        from treeskel.shape import load_traces

        traces = load_traces(args.traces) if Path(args.traces).exists() else {}
        picks = example_trees({t: scores[t]["dice"] for i, t in enumerate(trees) if done[i]}, args.examples)
        tiles = []
        for t in picks:
            i = trees.index(t)
            label = f"{t}  Dice {scores[t]['dice']:.2f}"
            tiles.append(draw_example(records[t], pred_masks[i], traces.get(t), label, overlays=overlays))
        if tiles:
            sheet = Image.new("RGB", (max(im.width for im in tiles), sum(im.height for im in tiles)))
            ytop = 0
            for im in tiles:
                sheet.paste(im, (0, ytop))
                ytop += im.height
            write(figures / f"{out.name}_examples.jpg", lambda p: sheet.save(p, quality=85))
    print(f"\nTrained in {minutes} min on {cv['device']}. Saved {out}/ (report.json, predictions.csv, "
          f"mask_scores.csv, training_log.csv, predicted_masks.npz) and the figures in {figures}/")
    if (out / "predictions.csv").exists() and (out / "predicted_masks.npz").exists():
        print(f"\n--- The same masks with their pieces joined (gaps up to {args.join_gap:g} px) ---")
        failed += joined_run(out, args.join_gap, figures=figures)[1]
    for f in failed:
        print(f"Could not write {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
