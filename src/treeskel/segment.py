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

**Full size** (``--size 480 640``, session 12, the one variant agreed at the
end of session 11): the whole photo at its own resolution, as the angle
CNN's best run. The hand masks cover only the photo's centre 480 columns, so
the side columns are unknown to the loss (:func:`known_columns`,
:func:`seg_loss`), and the predictions are measured in the masks' frame
(:func:`to_mask_size`), exactly as the square runs'. The rotation of the
augmentation is made in pixels, so it keeps angles in the wider frame.

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

**Grown masks** (:func:`grow_mask`, :func:`grown_run`, session 12, the
student's idea): where leaves hide a leader, the U-Net's probability often
fades below 0.5 before the leader ends, so the mask stops short or breaks.
On the lined-up run, 33 of 444 trees had a leader shorter than the stretch
the angles are fitted over, and they carried 47% of the V opening's squared
error. Each mask is grown into the less sure pixels connected to it
(hysteresis thresholding), then joined. The lower threshold is the one of
:data:`GROW_LOWS` whose masks best match the lined-up hand masks' centre
lines (:func:`cl_dice`); the winter traces play no part. Training runs save
their probabilities in the masks' frame (``probabilities.npz``) and write
``<run>_grown`` beside ``<run>_joined``. Not adopted: on the re-run it grew
every mask (a lower threshold of 0.1 chosen), added side branches, and was
no closer to winter on the V opening.

**Completed leader lines** (:func:`complete_measures`, :func:`completed_run`,
session 12, the student's shape-prior idea): a short leader's line over the
whole fitted stretch is predicted from its visible part, by a model fitted on
complete leaders cut short on purpose (:func:`fit_completion`): over the
missing part, lines turn more upright and keep turning the way the visible
part does. Fitted per fold on the other folds' trees; never on winter.

**Continued leaders** (:func:`continue_leaders`, :func:`continued_run`,
session 12, the student's next idea): a short leader is joined to a piece
lying on its line further on (within :data:`CONTINUE_CONE` of its direction
at the tip, and within the part of the fitted stretch still missing), and
each mask is then cut down to its trunk and two leaders
(:func:`prune_to_branches`). The settings come from the hand masks and the
measures' own stretch, never from winter.

Usage::

    python -m treeskel.segment data/raw                      # the student's PC (GPU); docs/results/segment/
    python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json   # docs/results/segment_lined_up/
    python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --size 480 640
                                                             # docs/results/segment_lined_up_480x640/ (+ _joined/)
    python -m treeskel.segment data/raw --depth              # docs/results/segment_depth/
    python -m treeskel.segment --replot docs/results/segment # figures only
    python -m treeskel.segment --join docs/results/segment_lined_up   # its masks joined: segment_lined_up_joined/
    python -m treeskel.segment data/raw --grow docs/results/segment_lined_up_rerun   # grown, threshold chosen again
    python -m treeskel.segment --continue docs/results/segment_lined_up_rerun_joined  # leaders continued, pruned
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
        size: One number: the side of a square input, the photo's centre
            480x480 (the hand masks' frame) resized; 256 is the masks' own
            resolution. Two, ``(height, width)``: the whole photo resized,
            e.g. ``(480, 640)`` at full size (session 12), where the hand
            masks cover only the centre columns and the rest is unknown to
            the loss (:func:`known_columns`). Each side must divide by 32.
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

    size: int | tuple[int, int] = 256
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


def whole_photo(params: SegParams) -> bool:
    """True if the network sees the whole photo (``size`` is ``(height, width)``), not the masks' square frame."""
    return not isinstance(params.size, int)


def input_shape(params: SegParams) -> tuple[int, int]:
    """``(height, width)`` of the network's input."""
    return (params.size, params.size) if not whole_photo(params) else (int(params.size[0]), int(params.size[1]))


def known_columns(width: int) -> tuple[int, int]:
    """The columns ``[start, stop)`` of a whole photo resized to ``width`` that the hand masks describe (the
    photo's columns 80-559, :data:`treeskel.io.CROP_COLS`); outside them a mask says nothing.

    Example:
        >>> known_columns(640), known_columns(320)
        ((80, 560), (40, 280))
    """
    from treeskel.io import CROP_COLS, IMAGE_SHAPE

    return round(CROP_COLS[0] * width / IMAGE_SHAPE[1]), round(CROP_COLS[1] * width / IMAGE_SHAPE[1])


def known_map(params: SegParams) -> np.ndarray | None:
    """``(height, width)`` float32, 1 where the hand masks say something and 0 where they do not; None for the
    masks' own square frame, where they describe every pixel."""
    if not whole_photo(params):
        return None
    h, w = input_shape(params)
    out = np.zeros((h, w), np.float32)
    a, b = known_columns(w)
    out[:, a:b] = 1.0
    return out


def mask_to_photo(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A 256x256 hand mask placed on the whole photo (:func:`treeskel.io.mask_to_image_frame`, but resized
    bilinearly, so soft at the edges as in the square frame) and resized to ``shape`` = ``(height, width)``:
    uint8 0-255, zero outside the columns it describes."""
    from PIL import Image

    from treeskel.io import CROP_COLS, IMAGE_SHAPE

    side = IMAGE_SHAPE[0]
    big = np.asarray(Image.fromarray(np.asarray(mask, bool).astype(np.uint8) * 255).resize((side, side),
                                                                                            Image.BILINEAR))
    full = np.zeros(IMAGE_SHAPE, np.uint8)
    full[:, CROP_COLS[0]:CROP_COLS[1]] = big
    if tuple(shape) == IMAGE_SHAPE:
        return full
    return np.asarray(Image.fromarray(full).resize((shape[1], shape[0]), Image.BILINEAR))


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

def seg_loss(logits, target, known=None):
    """Binary cross-entropy plus soft Dice loss (``1 − 2|P∩T| / (|P| + |T|)``, per photo, smoothed by 1).

    Args:
        logits: ``(B, H, W)`` network outputs.
        target: ``(B, H, W)`` masks in [0, 1] (soft at resampled edges).
        known: Optional ``(B, H, W)`` or ``(H, W)`` weights in [0, 1]: where
            the masks say something (1) or nothing (0, the whole photo's side
            columns). Pixels of weight 0 count in neither term; None counts
            every pixel, exactly as before.
    """
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits)
    if known is None:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    else:
        known = known.expand_as(target)
        bce = ((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * known).sum()
               / known.sum().clamp_min(1.0))
        p, target = p * known, target * known
    inter = (p * target).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()


def augment_pair(x, m, generator, params: SegParams = SegParams(), flip_p: float = 0.5, known=None):
    """Random crop, rotation, mirror image and colour jitter, with the mask resampled alike.

    The crop, rotation and mirror are one affine resampling per photo: a
    fraction ``f`` of each side (``f² ∈ [crop, 1]`` of the area) at a random
    position. The rotation is made in pixels, so it keeps angles in a
    non-square frame too (in a square one the arithmetic is unchanged).

    Args:
        x: ``(B, C, H, W)`` floats in [0, 1] (C = 3, or 4 with the depth last,
            which is not jittered).
        m: ``(B, H, W)`` masks in [0, 1].
        generator: A ``torch.Generator`` on ``x``'s device.
        params: ``crop``, ``rotate`` and ``jitter``.
        flip_p: Chance of the mirror image.
        known: Optional ``(H, W)`` or ``(B, H, W)`` weights (:func:`seg_loss`),
            resampled with the mask; pixels brought in from past the edge are
            the edge's.

    Returns:
        ``(images, masks)``: normalised images and masks in [0, 1], same
        shapes; with ``known``, ``(images, masks, known)``.
    """
    import torch
    import torch.nn.functional as F

    b, dev = x.shape[0], x.device
    aspect = x.shape[-1] / x.shape[-2]                                       # width / height

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=generator)

    f = torch.sqrt(uniform(params.crop, 1.0))
    a = torch.deg2rad(uniform(-params.rotate, params.rotate))
    tx, ty = uniform(-1.0, 1.0) * (1 - f), uniform(-1.0, 1.0) * (1 - f)
    sx = torch.where(torch.rand(b, device=dev, generator=generator) < flip_p, -f, f)
    theta = torch.zeros(b, 2, 3, device=dev)
    cos, sin = torch.cos(a), torch.sin(a)
    # Rotation after the mirror and scale, in pixels: in the grid's [-1, 1] coordinates the off-diagonal
    # terms carry the frame's aspect ratio (1 for a square frame).
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos * sx, -sin * f / aspect, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin * sx * aspect, cos * f, ty
    parts = [x, m.unsqueeze(1)]
    if known is not None:
        parts.append(known.expand(b, *m.shape[-2:]).unsqueeze(1).to(x.dtype))
    both = torch.cat(parts, dim=1)
    grid = F.affine_grid(theta, list(both.shape), align_corners=False)
    both = F.grid_sample(both, grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = x.shape[1]
    x, m = both[:, :c], both[:, c].clamp(0, 1)
    if known is not None:
        known = both[:, c + 1].clamp(0, 1)
    j = params.jitter
    if j > 0:
        rgb, rest = x[:, :3], x[:, 3:]
        rgb = rgb * uniform(1 - j, 1 + j).view(b, 1, 1, 1)                     # brightness
        mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
        rgb = (rgb - mean) * uniform(1 - j, 1 + j).view(b, 1, 1, 1) + mean      # contrast
        grey = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).unsqueeze(1)
        rgb = grey + (rgb - grey) * uniform(1 - j, 1 + j).view(b, 1, 1, 1)      # saturation
        x = torch.cat([rgb.clamp(0, 1), rest], dim=1)
    return (normalise(x), m) if known is None else (normalise(x), m, known)


def _float(x_uint8):
    return x_uint8.float().div_(255.0)


def predict_masks(model, x_uint8, batch: int = 16) -> np.ndarray:
    """Branch probabilities, each the average of the photo's and (flipped back) its mirror image's.

    Args:
        model: The network.
        x_uint8: ``(N, C, S, S)`` uint8 tensor on the model's device.
        batch: Photos per forward pass.

    Returns:
        ``(N, H, W)`` float32 probabilities.
    """
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x_uint8), batch):
            x = normalise(_float(x_uint8[i:i + batch]))
            p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
            out.append(p.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + tuple(x_uint8.shape[-2:]), np.float32)


def train_seg(model, x_uint8, m_uint8, epochs: int, params: SegParams, seed: int, val: tuple | None = None,
              on_epoch: Callable[[dict], None] | None = None) -> list[dict]:
    """Train ``model`` in place: AdamW with a cosine learning-rate decay over ``epochs``.

    Args:
        model: The :class:`UNet`, on its device.
        x_uint8: ``(N, C, H, W)`` uint8 tensor of training photos on the device.
        m_uint8: ``(N, H, W)`` uint8 tensor of their masks (0–255).
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
    km = known_map(params)                                  # None in the masks' own frame
    known = None if km is None else torch.from_numpy(km).to(dev)
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
            if known is None:
                x, m = augment_pair(_float(x_uint8[idx]), _float(m_uint8[idx]), gen, params)
                loss = seg_loss(model(x), m)
            else:
                x, m, k = augment_pair(_float(x_uint8[idx]), _float(m_uint8[idx]), gen, params, known=known)
                loss = seg_loss(model(x), m, k)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, count = total + loss.item() * len(idx), count + len(idx)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": None}
        if val is not None:
            p = torch.as_tensor(predict_masks(model, val[0])).clamp(1e-6, 1 - 1e-6)
            record["val_loss"] = float(seg_loss(torch.logit(p), _float(val[1]).cpu(),
                                               None if known is None else known.cpu()))
        log.append(record)
        if on_epoch:
            on_epoch(record)
    return log


def cross_validate_seg(images: np.ndarray, masks: np.ndarray, has_mask: np.ndarray, rows: np.ndarray,
                       fold_rows: Sequence[Sequence[int]], params: SegParams = SegParams(), device: str | None = None,
                       pretrained: bool = True, progress: Callable[[str], None] = print) -> dict:
    """Train and predict every tree from the fold that holds its row out (see the module docstring).

    Args:
        images: ``(n, H, W, C)`` uint8 summer photos at ``params.size`` (:func:`load_frames`).
        masks: ``(n, H, W)`` uint8 hand masks (0–255) in the same frame.
        has_mask: ``(n,)`` bool; trees without a hand mask are predicted but never trained on.
        rows: ``(n,)`` orchard rows.
        fold_rows: Each fold's held-out rows (``splits.json``).
        params: The recipe.
        device: ``"cuda"``, ``"cpu"``, or None for the GPU if there is one.
        pretrained: Start from the ImageNet weights (False: random, for tests).
        progress: Where progress lines go.

    Returns:
        ``{"prob": (n, H, W) float32, "fold": (n,), "epochs": [...], "log": [...], "device": str}``.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params.seed)                          # the decoder's first weights
    base = UNet(params.encoder, pretrained, channels=images.shape[3]).to(device)
    start = copy.deepcopy(base.state_dict())
    x_all = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(device)
    m_all = torch.from_numpy(np.ascontiguousarray(masks)).to(device)
    n = len(images)
    prob, fold = np.full((n,) + images.shape[1:3], np.nan, np.float32), np.full(n, -1)
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

def prob_frame(prob: np.ndarray, whole: bool = False) -> np.ndarray:
    """Probabilities in the hand masks' 256x256 frame (float32), resized bilinearly.

    Args:
        prob: ``(S, S)`` probabilities in the masks' frame, or, with ``whole``,
            ``(H, W)`` over the whole photo, whose centre columns
            (:func:`known_columns`) are the masks' frame.
        whole: ``prob`` covers the whole photo.
    """
    from PIL import Image

    from treeskel.io import MASK_SHAPE

    p = np.asarray(prob, np.float32)
    if whole:
        a, b = known_columns(p.shape[1])
        p = np.ascontiguousarray(p[:, a:b])
    if p.shape != MASK_SHAPE:
        p = np.asarray(Image.fromarray(p).resize(MASK_SHAPE[::-1], Image.BILINEAR))
    return p


def to_mask_size(prob: np.ndarray, threshold: float = 0.5, whole: bool = False) -> np.ndarray:
    """Probabilities to a 256x256 bool mask in the hand masks' frame (:func:`prob_frame`), thresholded.

    Args:
        prob: As :func:`prob_frame`.
        threshold: Probability above which a pixel is branch.
        whole: ``prob`` covers the whole photo.
    """
    return prob_frame(prob, whole) > threshold


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
        width = min(_widest_near(half_width, sk_piece, a, gap), _widest_near(half_width, sk_main, b, gap))
        band = _band(out.shape, a, b, width)
        out |= band
        main |= piece | band
        rest.remove(k)
        joined += 1
    return out, joined


def _band(shape: tuple[int, int], a: tuple[int, int], b: tuple[int, int], width: float) -> np.ndarray:
    """The pixels within ``width`` (at least 1) of the straight line from ``a`` to ``b``: a join's band."""
    from scipy import ndimage as ndi
    from skimage.draw import line as draw_line

    line = np.zeros(shape, bool)
    line[draw_line(*a, *b)] = True
    return ndi.distance_transform_edt(~line) < max(1.0, width)


def _widest_near(half_width: np.ndarray, skeleton: np.ndarray, point: tuple[int, int], reach: float) -> float:
    """The largest half-width on a centre line within ``reach`` of ``point``: the branch's own half-width
    near a join, not the smaller one where the centre line ends short of a cut face."""
    ys, xs = np.nonzero(skeleton)
    near = np.hypot(ys - point[0], xs - point[1]) <= max(reach, 1.0)
    return float(half_width[ys[near], xs[near]].max())


GROW_LOWS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)
"""The lower thresholds :func:`choose_low` tries for :func:`grow_mask`, fixed before any run (session 12);
0.5 grows nothing."""


def grow_mask(prob: np.ndarray, low: float, seed: np.ndarray | None = None, high: float = 0.5) -> np.ndarray:
    """Hysteresis thresholding: the confident mask plus every less sure pixel connected to it.

    Where leaves hide a leader, the U-Net's probability often fades below the
    threshold before the leader ends, so the mask stops short or breaks. Here
    pixels above ``low`` are kept wherever they connect (8-connected, through
    other such pixels) to the confident mask; faint patches that touch no
    confident pixel stay out. It uses only the network's own output.

    Args:
        prob: ``(H, W)`` probabilities.
        low: Lower threshold; ``low >= high`` grows nothing.
        seed: The confident mask (default ``prob > high``); pass the saved
            mask so that it is kept exactly.
        high: The threshold the mask was made with.

    Returns:
        ``(H, W)`` bool: ``seed`` and the grown pixels.
    """
    from scipy import ndimage as ndi

    p = np.asarray(prob, np.float32)
    seed = p > high if seed is None else np.asarray(seed, bool)
    candidates = seed | (p > low)
    lab, n = ndi.label(candidates, np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    return keep[lab]


def cl_dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Centre-line Dice (Shit et al., 2021): how well two masks' centre lines lie in each other.

    The harmonic mean of the fraction of the prediction's centre line inside
    the truth (topology precision) and of the truth's centre line inside the
    prediction (topology sensitivity). A leader that stops short or breaks
    loses sensitivity; growth into leaves beside the branch loses precision.
    Unlike the pixel Dice, it is not dominated by the wide trunk. NaN if
    either mask is empty.
    """
    from skimage.morphology import skeletonize

    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    sp, st = skeletonize(p), skeletonize(t)
    if not sp.any() or not st.any():
        return float("nan")
    prec, sens = (sp & t).sum() / sp.sum(), (st & p).sum() / st.sum()
    return float(2 * prec * sens / (prec + sens)) if prec + sens > 0 else 0.0


def choose_low(probs: dict[str, np.ndarray], seeds: dict[str, np.ndarray], hands: dict[str, np.ndarray],
               lows: Sequence[float] = GROW_LOWS, gap: float = JOIN_GAP,
               progress: Callable[[str], None] = print) -> tuple[float, dict[str, float]]:
    """The lower threshold of :func:`grow_mask` whose masks, grown and joined, best match the hand masks'
    centre lines (mean :func:`cl_dice` over the trees with a hand mask). The winter traces play no part.

    Args:
        probs: ``{tree: (256, 256) probabilities}`` in the masks' frame, each from the fold that held it out.
        seeds: ``{tree: the saved mask}``.
        hands: ``{tree: its (lined-up) hand mask}``.
        lows: The thresholds tried; on a tie the higher (less growth) wins.
        gap: The joining gap applied after growing, as in the scored masks.

    Returns:
        ``(chosen low, {str(low): mean clDice})``.
    """
    trees = [t for t in probs if t in hands]
    table = {}
    for low in lows:
        vals = [cl_dice(join_pieces(grow_mask(probs[t], low, seeds[t]), gap)[0], hands[t]) for t in trees]
        table[f"{low:g}"] = float(np.nanmean(vals))
        progress(f"  lower threshold {low:<4g}: mean clDice {table[f'{low:g}']:.4f} ({len(trees)} trees)")
    best = max(lows, key=lambda v: (round(table[f"{v:g}"], 6), v))
    return float(best), table


def save_probabilities(path: Path, ids: Sequence[str], probs: np.ndarray) -> None:
    """Probability maps of many trees in the masks' frame, as uint8 (``round(255 p)``) in one ``.npz``."""
    q = np.round(np.clip(np.asarray(probs, np.float32), 0, 1) * 255).astype(np.uint8)
    np.savez_compressed(path, ids=np.array(ids), prob=q)


def load_probabilities(path: Path) -> dict[str, np.ndarray]:
    """Inverse of :func:`save_probabilities`: ``{tree: (256, 256) float32 in [0, 1]}``."""
    with np.load(path) as z:
        return {str(t): z["prob"][i].astype(np.float32) / 255 for i, t in enumerate(z["ids"])}


CONTINUE_CONE = 25.0
"""Largest angle (degrees) between a short leader's direction at its tip and a piece that
:func:`continue_leaders` joins to it: the development hand masks' leaders turn by at most 24.8 degrees
between the two halves of the fitted stretch in 95% of cases (896 leaders), so a piece further off that
line is more likely another branch than the same leader. Fixed before any continued mask was scored
(session 12); never tuned on the winter traces."""


def _tip_direction(path: np.ndarray, length: float) -> np.ndarray:
    """Unit ``(row, col)`` direction of a leader at its tip: a line through its last ``length`` px, pointing out."""
    from treeskel.guided_gt import arc_length
    from treeskel.shape import line_direction, to_xy

    p = np.asarray(path, float)
    s = arc_length(p)
    end = p[s >= s[-1] - length]
    if len(end) < 2:
        end = p[-2:]
    d = line_direction(to_xy(end))                          # (x, y), y up
    d = np.array([-d[1], d[0]])                            # (row, col)
    return d if np.dot(d, end[-1] - end[0]) >= 0 else -d


def continue_leaders(mask: np.ndarray, cone: float = CONTINUE_CONE, params: ShapeParams = ShapeParams(),
                     rounds: int = 5) -> tuple[np.ndarray, int]:
    """Join pieces lying on a short leader's way on, then keep only the trunk and the two leaders.

    The student's idea (session 12): where leaves hide a leader, the mask
    often stops, and a piece of the same leader lies further along its line,
    too far for :func:`join_pieces`. Here, for each leader shorter than the
    stretch the angles are fitted over (``params.reach`` units at
    :data:`treeskel.shape.MASK_ZOOM`), the pieces of the mask not joined to
    the tree are searched: one is joined if some pixel of it lies within the
    part of the stretch still missing (``reach`` minus the leader's length)
    of the tip and within ``cone`` degrees of the leader's direction at its
    tip (a line through its last ``params.trunk_reach`` units, the length of
    the trunk's axis), and it has at least ``params.min_points`` px of
    centre line. The nearest such piece is joined by a band from the tip to
    its nearest centre-line point, as :func:`join_pieces` joins; the leaders
    are found again and the search repeats (a chain of pieces), up to
    ``rounds`` times. Finally the mask is cut down to the trunk and the two
    leaders (:func:`prune_to_branches`).

    Args:
        mask: ``(256, 256)`` bool mask, already joined (:func:`join_pieces`).
        cone: :data:`CONTINUE_CONE`.
        params: The shape measures' stretch.
        rounds: Most joins per leader search.

    Returns:
        ``(the trunk and leaders, pieces joined)``; a mask with no fork is returned unchanged.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    from treeskel.guided_gt import arc_length
    from treeskel.shape import MASK_ZOOM, unit_px

    out = np.asarray(mask, bool).copy()
    unit = unit_px(0.0, params, MASK_ZOOM)
    reach, tip_len = params.reach * unit, params.trunk_reach * unit
    joined = 0
    br = mask_branches(out)
    if br is None:
        return out, 0
    for _ in range(rounds):
        filled = ndi.binary_fill_holes(out)
        lab, n = ndi.label(filled, np.ones((3, 3)))
        tree = lab[tuple(br["trunk"][-1])]
        half_width = ndi.distance_transform_edt(filled)
        best = None                                         # (distance, side, piece label)
        for side in ("left", "right"):
            path = br[side]
            length = float(arc_length(path)[-1]) if len(path) > 1 else 0.0
            if length >= reach or len(path) < 2:
                continue
            tip, d = np.asarray(path[-1], float), _tip_direction(path, tip_len)
            for k in range(1, n + 1):
                if k == tree:
                    continue
                ys, xs = np.nonzero(lab == k)
                v = np.stack([ys, xs], 1) - tip
                dist = np.hypot(v[:, 0], v[:, 1])
                ok = (dist <= reach - length) & (v @ d >= np.cos(np.radians(cone)) * dist)
                if not ok.any() or skeletonize(lab == k).sum() < params.min_points:
                    continue
                if best is None or dist[ok].min() < best[0]:
                    best = (float(dist[ok].min()), side, k)
        if best is None:
            break
        _, side, k = best
        piece = lab == k
        sk_piece = skeletonize(piece)
        ys, xs = np.nonzero(sk_piece)
        tip = tuple(int(v) for v in br[side][-1])
        j = int(np.argmin(np.hypot(ys - tip[0], xs - tip[1])))
        b = (int(ys[j]), int(xs[j]))
        tree_line = np.zeros_like(out)
        tree_line[tuple(np.asarray(br[side]).T)] = True
        width = min(_widest_near(half_width, sk_piece, b, JOIN_GAP), _widest_near(half_width, tree_line, tip, JOIN_GAP))
        out |= _band(out.shape, tip, b, width)
        joined += 1
        found = mask_branches(out)
        if found is None:                                   # cannot happen when joining to the tree; be safe
            break
        br = found
    return prune_to_branches(out, br), joined


def prune_to_branches(mask: np.ndarray, br: dict) -> np.ndarray:
    """The mask cut down to the trunk and the two leaders: its pixels within each path point's own half-width
    (the distance to the mask's edge there) of the paths :func:`treeskel.shape.mask_branches` found."""
    from scipy import ndimage as ndi

    filled = ndi.binary_fill_holes(np.asarray(mask, bool))
    half_width = ndi.distance_transform_edt(filled)
    pts = np.concatenate([np.asarray(br[p]).reshape(-1, 2) for p in ("trunk", "left", "right")])
    seeds = np.zeros(filled.shape, bool)
    seeds[pts[:, 0], pts[:, 1]] = True
    dist, (iy, ix) = ndi.distance_transform_edt(~seeds, return_indices=True)
    return filled & (dist < half_width[iy, ix])


COMPLETE_FRACTIONS = tuple(round(0.40 + 0.05 * i, 2) for i in range(12))
"""Where :func:`completion_samples` cuts complete leaders (fractions of the fitted stretch, 0.40-0.95): the
lengths short predicted leaders stop at (median 0.75 of the stretch, quartiles 0.54-0.89)."""


def _outward(path: np.ndarray, side: str, params: ShapeParams = ShapeParams()) -> tuple[float, float]:
    """A leader's line from the vertical, positive outwards, and its bend (as :func:`treeskel.shape.shape_measures`
    measures them in a mask), or NaN where the path is too short."""
    from treeskel.shape import MASK_ZOOM, halves, leader_part, line_direction, tilt, to_xy, unit_px, wrap

    part = leader_part(path, unit_px(0.0, params, MASK_ZOOM), params)
    if len(part) < params.min_points:
        return float("nan"), float("nan")
    t = tilt(line_direction(to_xy(part)))
    low, up = halves(part)
    bend = float("nan")
    if min(len(low), len(up)) >= params.min_points:
        turn = wrap(tilt(line_direction(to_xy(up))) - tilt(line_direction(to_xy(low))))
        bend = turn if side == "right" else -turn
    return (t if side == "right" else -t), bend


def completion_samples(masks: dict[str, np.ndarray], params: ShapeParams = ShapeParams(),
                       fractions: Sequence[float] = COMPLETE_FRACTIONS) -> np.ndarray:
    """Complete leaders cut short on purpose: ``(n, 4)`` rows of ``(missing fraction g, outward line of the
    cut leader, its bend, outward line of the whole)``, for :func:`fit_completion`.

    Every leader of ``masks`` that spans the whole fitted stretch is cut at each of ``fractions`` of it.
    """
    from treeskel.guided_gt import arc_length
    from treeskel.shape import MASK_ZOOM, unit_px

    reach = params.reach * unit_px(0.0, params, MASK_ZOOM)
    rows = []
    for m in masks.values():
        _, br = mask_shape(m, params)
        if br is None:
            continue
        for side in ("left", "right"):
            s = arc_length(br[side]) if len(br[side]) > 1 else np.zeros(1)
            if s[-1] < reach:
                continue
            whole, _ = _outward(br[side], side, params)
            for f in fractions:
                cut, bend = _outward(br[side][s <= f * reach], side, params)
                if np.isfinite(cut) and np.isfinite(whole):
                    rows.append((1 - f, cut, bend, whole))
    return np.array(rows, float).reshape(-1, 4)


def fit_completion(samples: np.ndarray) -> np.ndarray:
    """Least squares for the change of a leader's line over its missing part:
    ``whole − cut = a·g + c·g·bend`` (``g`` the missing fraction; a missing bend counts as 0). Returns ``(a, c)``."""
    g, cut, bend, whole = samples.T
    x = np.stack([g, g * np.nan_to_num(bend)], 1)
    coef, *_ = np.linalg.lstsq(x, whole - cut, rcond=None)
    return coef


def complete_measures(mask: np.ndarray, coef: Sequence[float], params: ShapeParams = ShapeParams()
                      ) -> tuple[dict[str, float], int]:
    """:func:`mask_shape` with each short leader's line predicted over the whole fitted stretch.

    A leader shorter than the stretch has its line from the vertical moved
    by :func:`fit_completion`'s model (``a·g + c·g·bend``, outwards); the
    measures built on the leader lines (the V opening, the angles from the
    trunk, the V's lean, the tilts) follow. The bends stay as measured.

    Returns:
        ``(measures, leaders completed)``.
    """
    from treeskel.guided_gt import arc_length
    from treeskel.shape import MASK_ZOOM, unit_px, wrap

    m, br = mask_shape(mask, params)
    if br is None:
        return m, 0
    reach = params.reach * unit_px(0.0, params, MASK_ZOOM)
    a, c = coef
    tilts, done = {}, 0
    for side in ("left", "right"):
        tilts[side] = m[f"{side}_tilt"]
        length = float(arc_length(br[side])[-1]) if len(br[side]) > 1 else 0.0
        if length >= reach or not np.isfinite(tilts[side]):
            continue
        g = 1 - length / reach
        _, bend = _outward(br[side], side, params)
        shift = a * g + c * g * (0.0 if not np.isfinite(bend) else bend)
        tilts[side] += shift if side == "right" else -shift
        done += 1
    if not done:
        return m, 0
    out = dict(m)
    out["left_tilt"], out["right_tilt"] = tilts["left"], tilts["right"]
    trunk = m["trunk_tilt"]
    if np.isfinite(tilts["left"]) and np.isfinite(tilts["right"]):
        out["v_opening"] = wrap(tilts["right"] - tilts["left"])
        out["lean_v"] = (tilts["left"] + tilts["right"]) / 2
        if np.isfinite(trunk):
            out["lean"] = wrap(out["lean_v"] - trunk)
    if np.isfinite(trunk):
        out["left_angle"] = wrap(trunk - tilts["left"]) if np.isfinite(tilts["left"]) else float("nan")
        out["right_angle"] = wrap(tilts["right"] - trunk) if np.isfinite(tilts["right"]) else float("nan")
    return out, done


def draw_completion(mask: np.ndarray, coef: Sequence[float], params: ShapeParams = ShapeParams()
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The mask with each short leader drawn on over the missing part of the stretch, as :func:`complete_measures`
    predicts it: from the tip, along the leader's predicted line, as wide as the leader near its tip.

    For showing the prediction (a leader that goes on through the leaves, not a stub); the scores come
    from :func:`complete_measures`.

    Returns:
        ``(mask with the continuations, the continuations alone)``.
    """
    from scipy import ndimage as ndi

    from treeskel.guided_gt import arc_length
    from treeskel.shape import MASK_ZOOM, unit_px

    out = np.asarray(mask, bool).copy()
    added = np.zeros_like(out)
    m, br = mask_shape(out, params)
    if br is None:
        return out, added
    reach = params.reach * unit_px(0.0, params, MASK_ZOOM)
    a, c = coef
    half_width = ndi.distance_transform_edt(ndi.binary_fill_holes(out))
    h, w = out.shape
    for side in ("left", "right"):
        path = np.asarray(br[side])
        length = float(arc_length(path)[-1]) if len(path) > 1 else 0.0
        if length >= reach or not np.isfinite(m[f"{side}_tilt"]):
            continue
        g = 1 - length / reach
        line, bend = _outward(path, side, params)
        whole = line + a * g + c * g * (0.0 if not np.isfinite(bend) else bend)
        angle = np.radians(whole if side == "right" else -whole)
        d = np.array([-np.cos(angle), np.sin(angle)])     # (row, col), upwards
        tip = path[-1].astype(float)
        steps = np.arange(0.0, reach - length, 0.5)
        pts = tip + steps[:, None] * d
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= h - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= w - 1)
        if not inside.any():
            continue
        end = tuple(int(round(v)) for v in pts[np.nonzero(inside)[0][-1]])
        leader = np.zeros_like(out)
        leader[path[:, 0], path[:, 1]] = True
        width = _widest_near(half_width, leader, tuple(int(v) for v in tip), JOIN_GAP)
        added |= _band(out.shape, tuple(int(v) for v in tip), end, width) & ~out
    return out | added, added


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


def load_frames(records: dict, ids: Sequence[str], size: int | tuple[int, int], depth: bool = False,
                progress: Callable[[str], None] = print,
                overlays: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The summer photos and hand masks of ``ids``: in the masks' frame (centre 480x480) at ``size`` x ``size``,
    or, for ``size = (height, width)``, the whole photo at that size with each mask on its centre columns
    (:func:`mask_to_photo`).

    With ``overlays`` (the mask line-up file), each mask is moved onto its photo
    (:func:`hand_mask`), and trees not lined up count as having none.

    Returns:
        ``(images, masks, has_mask)``: ``(n, H, W, 3 or 4)`` uint8 (depth
        last, nearest-neighbour so no-reading zeros are not blended),
        ``(n, H, W)`` uint8 masks (0–255, bilinear, so soft at the edges;
        zero where there is none, and on a whole photo's side columns) and ``(n,)`` bool.
    """
    from PIL import Image

    from treeskel.io import CROP_COLS, load_tree

    whole = not isinstance(size, int)
    h, w = (int(size[0]), int(size[1])) if whole else (size, size)
    images = np.zeros((len(ids), h, w, 4 if depth else 3), np.uint8)
    masks = np.zeros((len(ids), h, w), np.uint8)
    has = np.zeros(len(ids), bool)
    step = max(1, len(ids) // 10)
    for i, t in enumerate(ids):
        summer = load_tree(records[t], "summer")
        crop = slice(0, None) if whole else slice(CROP_COLS[0], CROP_COLS[1])
        images[i, ..., :3] = np.asarray(Image.fromarray(np.ascontiguousarray(summer["rgb"][:, crop]))
                                        .resize((w, h), Image.BILINEAR))
        if depth:
            images[i, ..., 3] = np.asarray(Image.fromarray(np.ascontiguousarray(summer["depth"][:, crop]))
                                           .resize((w, h), Image.NEAREST))
        hm = hand_mask(records[t], overlays)
        if hm is not None:
            masks[i] = (mask_to_photo(hm, (h, w)) if whole else
                        np.asarray(Image.fromarray(hm.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)))
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
                 overlays: dict | None = None, continuation: np.ndarray | None = None):
    """One tree as a row of three: the winter trace, the hand mask and the predicted mask, each with the
    lines :func:`treeskel.shape.shape_measures` fits (PIL image, resized by ``scale``, then labelled). With
    ``overlays``, the hand mask is shown lined up (:func:`hand_mask`); ``continuation`` (from
    :func:`draw_completion`) is shown in orange on the predicted mask."""
    from PIL import Image, ImageDraw, ImageFont

    from treeskel.io import load_tree, mask_to_image_frame
    from treeskel.shape import MASK_ZOOM as ZOOM, draw_shape, fitted_lines, mask_to_image

    params = ShapeParams()
    s = load_tree(rec, "summer")["rgb"]

    def on_summer(mask, name, colour, extra=None):
        img = s.copy()
        full, _ = mask_to_image_frame(np.asarray(mask, bool))
        img[full] = (0.45 * img[full] + 0.55 * np.array(colour)).astype(np.uint8)
        if extra is not None and np.any(extra):
            more, _ = mask_to_image_frame(np.asarray(extra, bool))
            img[more] = (0.3 * img[more] + 0.7 * np.array((255, 150, 0))).astype(np.uint8)
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
    shown = continuation is not None and np.any(continuation)
    tiles.append(on_summer(pred, "predicted mask" + (" (orange: predicted on)" if shown else ""), (0, 200, 0),
                           continuation))
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

RUN_STYLES = {"segment": SEGMENT_STYLE, "segment_lined_up": LINED_UP_STYLE, "segment_lined_up_joined": JOINED_STYLE,
              "segment_lined_up_480x640": ("#eb6834", "p"), "segment_lined_up_480x640_joined": ("#eda100", "H"),
              "segment_lined_up_rerun": ("#1baf7a", "X"), "segment_lined_up_rerun_joined": ("#eda100", "h"),
              "segment_lined_up_rerun_grown": ("#1baf7a", "*"), "segment_lined_up_rerun_continued": ("#1baf7a", "d"),
              "segment_lined_up_completed": ("#1baf7a", "d"), "segment_lined_up_rerun_completed": ("#1baf7a", "d")}
"""The full-size runs (session 12) take the two slots not used by these figures, orange and yellow; their figure
draws them between the square runs (:data:`FIGURE_RUNS`) so that no two warm colours are adjacent (checked with
the palette validator)."""

RUN_NAMES = {"segment": "U-Net on the hand masks as drawn, its masks measured",
             "segment_lined_up": "U-Net on the lined-up hand masks, its masks measured",
             "segment_lined_up_joined": "U-Net on the lined-up hand masks, pieces of its masks joined",
             "segment_lined_up_480x640": "U-Net on the lined-up hand masks, whole photo at 480x640",
             "segment_lined_up_480x640_joined": "U-Net, whole photo at 480x640, pieces of its masks joined",
             "segment_lined_up_rerun": "The lined-up U-Net trained again (probabilities saved)",
             "segment_lined_up_rerun_joined": "The same run again, pieces of its masks joined",
             "segment_lined_up_rerun_grown": "The same run again, masks grown into less sure pixels, then joined",
             "segment_lined_up_rerun_continued": "The same run again, joined, short leaders continued, pruned",
             "segment_lined_up_completed": "U-Net on the lined-up masks, joined, short leaders' lines completed",
             "segment_lined_up_rerun_completed": "The same run again, joined, short leaders' lines completed"}

FIGURE_RUNS = {"segment_lined_up_rerun": ("segment_lined_up", "segment_lined_up_rerun"),
               "segment_lined_up_rerun_joined": ("segment_lined_up_joined", "segment_lined_up_rerun_joined"),
               "segment_lined_up_rerun_grown": ("segment_lined_up_joined", "segment_lined_up_rerun_joined",
                                                "segment_lined_up_rerun_grown"),
               "segment_lined_up_rerun_continued": ("segment_lined_up_joined", "segment_lined_up_rerun_joined",
                                                    "segment_lined_up_rerun_continued"),
               "segment_lined_up_completed": ("segment_lined_up_joined", "segment_lined_up_completed"),
               "segment_lined_up_rerun_completed": ("segment_lined_up_rerun_joined", "segment_lined_up_rerun_completed"),
               "segment_lined_up_480x640": ("segment_lined_up_480x640", "segment_lined_up"),
               "segment_lined_up_480x640_joined": ("segment_lined_up_480x640", "segment_lined_up",
                                                   "segment_lined_up_480x640_joined", "segment_lined_up_joined")}
"""The runs drawn (after the hand masks, in this order) in a run's skill figure, where not the default: the
angle CNN's two runs, the U-Net on the masks as drawn, the lined-up run and its joined one, up to the run
itself. The full-size figures compare the U-Net variants only, so that no more than six series share a row."""


def plot_skill(report: dict, path: Path, failed: list[str], others: Sequence[tuple[str, dict]] = ()) -> None:
    """The comparison figure of :func:`treeskel.shape_cnn.plot_comparison`, with the segmentation added."""
    from treeskel.shape_cnn import plot_comparison

    scores = report["scores"]
    key = report.get("variant") or ("segment_lined_up" if report.get("mask_overlay") else "segment")
    series = [("hand masks", {k: v["hand masks"] for k, v in scores.items()})]
    have = dict(others)
    have[key] = {k: v["segmentation"] for k, v in scores.items()}
    series += [(name, have[name]) for name in figure_runs(key) if name in have]
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


def figure_runs(key: str) -> tuple[str, ...]:
    """The runs drawn in ``key``'s skill figure, in order (:data:`FIGURE_RUNS`), ``key`` included."""
    if key in FIGURE_RUNS:
        return FIGURE_RUNS[key]
    base = ("shape_cnn", "shape_cnn_480x640", "segment", "segment_lined_up", "segment_lined_up_joined")
    return base[:base.index(key) + 1] if key in base else base[:3] + (key,)


def _other_runs(results: Path, key: str) -> list[tuple[str, dict]]:
    """The saved runs beside ``key``'s that its skill figure draws (:func:`figure_runs`)."""
    out = []
    for name in figure_runs(key):
        p = results / name / "report.json"
        if name != key and p.exists():
            rep = json.loads(p.read_text())
            model = "cnn fine-tuned" if name.startswith("shape_cnn") else "segmentation"
            out.append((name, {k: v[model] for k, v in rep["scores"].items()}))
    return out


def replot(out: Path, figures: Path) -> list[str]:
    """The skill and learning-curve figures from a saved run. Returns paths that could not be written."""
    failed: list[str] = []
    figures.mkdir(parents=True, exist_ok=True)
    report = json.loads((out / "report.json").read_text())
    key = report.get("variant") or ("segment_lined_up" if report.get("mask_overlay") else "segment")
    plot_skill(report, figures / f"{out.name}_skill.png", failed, _other_runs(out.parent, key))
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
    if whole_photo(params):
        return f"{name}_{input_shape(params)[0]}x{input_shape(params)[1]}"
    return name if params.size == SegParams.size else f"{name}_{params.size}"


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

MASK_COLUMNS = ("iou", "dice", "precision", "recall")

OTHER_RUNS = (("shape_cnn", "cnn fine-tuned"), ("shape_cnn_480x640", "cnn fine-tuned"), ("segment", "segmentation"),
              ("segment_lined_up", "segmentation"), ("segment_lined_up_joined", "segmentation"))
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


def _read_run(src: Path) -> tuple[dict, dict[str, dict]]:
    """A saved run's report, and per tree its row, fold and ``(winter, mean shape, hand masks)`` per measure
    (from its ``predictions.csv``, in its order)."""
    source = json.loads((Path(src) / "report.json").read_text())
    info: dict[str, dict] = {}
    with open(Path(src) / "predictions.csv", newline="") as f:
        for r in csv.DictReader(f):
            t = info.setdefault(r["tree"], {"row": int(r["row"]), "fold": int(r["fold"])})
            t[r["measure"]] = (_value(r["winter"]), _value(r["mean shape"]), _value(r["hand masks"]))
    return source, info


def _measure_all(masks: dict[str, np.ndarray], trees: Sequence[str]) -> np.ndarray:
    """``(n, len(MEASURES))`` shape measures of each tree's mask (:func:`mask_shape`)."""
    out = np.full((len(trees), len(MEASURES)), np.nan)
    for i, t in enumerate(trees):
        m, _ = mask_shape(masks[t])
        out[i] = [m[k] for k in MEASURES]
    return out


def _rescored_run(src: Path, source: dict, info: dict[str, dict], trees: Sequence[str], final: dict[str, np.ndarray],
                  seg: np.ndarray, out: Path, extra: dict, mask_header: Sequence[str], mask_rows: Sequence[Sequence],
                  figures: Path | None, sheet: tuple[str, dict, dict] | None, versus: Sequence[tuple[str, str]],
                  progress: Callable[[str], None]) -> list[str]:
    """Score, print and save a saved run's masks after a change (:func:`joined_run`, :func:`grown_run`).

    Args:
        src, source, info: The run's folder, report and :func:`_read_run` measures.
        trees: The trees, in order.
        final: ``{tree: the changed mask}``; ``seg`` their measures (:func:`_measure_all`).
        out: The new run's folder.
        extra: Report entries of the change (its settings and counts, and the pixel scores).
        mask_header, mask_rows: ``mask_scores.csv``.
        figures: Where the skill figure and the contact sheet go (None: none).
        sheet: ``(name, masks before, masks after)`` for :func:`draw_joins`, or None.
        versus: Extra saved runs to compare with tree by tree, beside :data:`OTHER_RUNS`.
        progress: Where the printed summary goes.

    Returns:
        Paths that could not be written.
    """
    y, mean, hand = (np.array([[info[t][k][c] for k in MEASURES] for t in trees]) for c in range(3))
    folds = np.array([info[t]["fold"] for t in trees])
    seg_filled = fill_failures(seg, mean)
    report = {k: source[k] for k in ("n_masks", "mask_overlay", "params", "device", "weights", "epochs", "minutes")
              if k in source}
    report.update(n_trees=len(trees), variant=out.name, no_fork=int((~np.isfinite(seg).any(axis=1)).sum()))
    report.update(extra)
    report.update(scores={}, against_runs={})
    for j, k in enumerate(MEASURES):
        base = mean[:, j]
        mean_shape = evaluate(y[:, j], base, base, folds)
        mean_shape["r"] = None
        report["scores"][k] = {"mean shape": mean_shape, "hand masks": evaluate(y[:, j], hand[:, j], base, folds),
                               "segmentation": evaluate(y[:, j], seg_filled[:, j], base, folds),
                               "segmentation, forks found": evaluate(y[:, j], seg[:, j], base, folds)}
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

    def mask_table(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(mask_header)
            w.writerows(mask_rows)

    write(out / "predictions.csv", predictions)
    write(out / "mask_scores.csv", mask_table)
    write(out / "predicted_masks.npz", lambda p: save_masks(p, trees, np.stack([final[t] for t in trees])))
    if (out / "predictions.csv").exists():
        report["against_runs"] = against_runs(out, tuple(OTHER_RUNS) + tuple(versus))
        print_against(report["against_runs"])
    write(out / "report.json", lambda p: p.write_text(json.dumps(report, indent=2)))
    if figures is not None:
        figures = Path(figures)
        figures.mkdir(parents=True, exist_ok=True)
        if (out / "report.json").exists():
            failed += replot(out, figures)
        if sheet is not None and sheet[1]:
            write(figures / f"{out.name}_{sheet[0]}.png", lambda p: draw_joins(sheet[1], sheet[2]).save(p))
    progress(f"\nSaved {out}/ (report.json, predictions.csv, mask_scores.csv, predicted_masks.npz)")
    return failed


def _pixel_columns(src: Path, trees: Sequence[str], final: dict[str, np.ndarray],
                   hands: dict[str, np.ndarray] | None) -> tuple[dict, str, dict[str, list]]:
    """Pixel scores for a re-scored run: against the hand masks if given, else the run's own (of the masks
    before the change). Returns ``(report summary, note, {tree: row values})``."""
    rows: dict[str, list] = {}
    if hands is not None:
        per = {t: mask_scores(final[t], hands[t]) for t in trees if t in hands}
        summary = {}
        for c in MASK_COLUMNS:
            v = np.array([per[t][c] for t in per], float)
            v = v[np.isfinite(v)]
            summary[c] = {"median": float(np.median(v)) if len(v) else None,
                          "mean": float(v.mean()) if len(v) else None, "n": int(len(v))}
        for t in trees:
            rows[t] = ["" if t not in per or not np.isfinite(per[t][c]) else f"{per[t][c]:.4f}" for c in MASK_COLUMNS]
        return summary, "pixel scores of the changed masks against the hand masks", rows
    pixel = {}
    if (Path(src) / "mask_scores.csv").exists():
        with open(Path(src) / "mask_scores.csv", newline="") as f:
            pixel = {r["tree"]: r for r in csv.DictReader(f)}
    for t in trees:
        rows[t] = [pixel.get(t, {}).get(c, "") for c in MASK_COLUMNS]
    return json.loads((Path(src) / "report.json").read_text()).get("masks", {}), "pixel scores of the masks before the change", rows


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
    source, info = _read_run(src)
    masks = load_masks(src / "predicted_masks.npz")
    trees = [t for t in info if t in masks]
    out = Path(out) if out else src.parent / (f"{src.name}_joined" if gap == JOIN_GAP else f"{src.name}_joined_{gap:g}")
    joined, pieces, added = {}, np.zeros(len(trees), int), np.zeros(len(trees), int)
    for i, t in enumerate(trees):
        joined[t], pieces[i] = join_pieces(masks[t], gap)
        added[i] = int((joined[t] & ~masks[t]).sum())
    seg = _measure_all(joined, trees)
    seg0 = seg.copy()
    changed = [i for i in range(len(trees)) if pieces[i]]
    seg0[changed] = _measure_all(masks, [trees[i] for i in changed])            # unchanged masks: measured once
    before, after = np.isfinite(seg0).any(axis=1), np.isfinite(seg).any(axis=1)   # as main: no measure = no fork
    v = MEASURES.index("v_opening")
    join = {"source": str(src), "gap": gap, "masks_joined": int((pieces > 0).sum()),
            "pieces_joined": int(pieces.sum()), "pixels_added": int(added.sum()),
            "no_fork_before": int((~before).sum()), "no_fork_after": int((~after).sum()),
            "v_opening_before": int(np.isfinite(seg0[:, v]).sum()), "v_opening_after": int(np.isfinite(seg[:, v]).sum()),
            "forks_found_by_joining": [t for i, t in enumerate(trees) if after[i] and not before[i]],
            "forks_lost_by_joining": [t for i, t in enumerate(trees) if before[i] and not after[i]]}
    progress(f"Joined pieces within {gap:g} px in {_n(join['masks_joined'], 'mask')} of {len(trees)} "
             f"({_n(join['pieces_joined'], 'piece')}, {join['pixels_added']} pixels added); masks with no fork "
             f"{join['no_fork_before']} -> {join['no_fork_after']}; with a V opening {join['v_opening_before']} -> "
             f"{join['v_opening_after']}")
    masks_summary, _, pixel = _pixel_columns(src, trees, joined, None)
    header = ["tree", "row", "fold", "fork_found", "fork_found_before", "pieces_joined", "pixels_added",
              *(f"{c}_before_joining" for c in MASK_COLUMNS)]
    rows = [[t, info[t]["row"], info[t]["fold"], int(after[i]), int(before[i]), int(pieces[i]), int(added[i]),
             *pixel[t]] for i, t in enumerate(trees)]
    moved = [t for i, t in enumerate(trees) if pieces[i]]
    failed = _rescored_run(src, source, info, trees, joined, seg, out,
                           {"masks": masks_summary, "masks_note": "pixel scores of the masks before joining",
                            "join": join},
                           header, rows, figures, ("masks", {t: masks[t] for t in moved}, {t: joined[t] for t in moved}),
                           ((src.name, "segmentation"),), progress)
    return out, failed


def grown_run(src: Path, low: float, choice: dict | None = None, hands: dict[str, np.ndarray] | None = None,
              gap: float = JOIN_GAP, out: Path | None = None, figures: Path | None = None,
              progress: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """Score a saved run's masks again after growing them (:func:`grow_mask`) and joining their pieces.

    Like :func:`joined_run`, from the run's own files, plus its
    ``probabilities.npz``: each saved mask is grown into the pixels above
    ``low`` connected to it, then joined (:func:`join_pieces`), then measured.
    The result goes to ``<run>_grown`` and is compared tree by tree with
    ``<run>_joined`` (the same network, joined only) and the other saved runs.

    Args:
        src: The saved run's folder (with ``probabilities.npz``).
        low: The lower threshold (chosen by :func:`choose_low`, never on winter).
        choice: How ``low`` was chosen (``{str(low): mean clDice}``), for the report.
        hands: ``{tree: hand mask}`` for pixel scores of the grown masks (else the run's own, before growing).
        gap: The joining gap after growing.
        out: Results folder (default ``<run>_grown``).
        figures: Where the skill figure and the contact sheet of the most grown masks go (None: none).
        progress: Where the printed summary goes.

    Returns:
        ``(out, paths that could not be written)``.
    """
    src = Path(src)
    source, info = _read_run(src)
    masks = load_masks(src / "predicted_masks.npz")
    probs = load_probabilities(src / "probabilities.npz")
    trees = [t for t in info if t in masks and t in probs]
    out = Path(out) if out else src.parent / f"{src.name}_grown"
    high = float(source.get("params", {}).get("threshold", 0.5))
    final, grown_px, pieces = {}, np.zeros(len(trees), int), np.zeros(len(trees), int)
    for i, t in enumerate(trees):
        grown = grow_mask(probs[t], low, masks[t], high)
        grown_px[i] = int((grown & ~masks[t]).sum())
        final[t], pieces[i] = join_pieces(grown, gap)
    seg = _measure_all(final, trees)
    joined_only = {t: join_pieces(masks[t], gap)[0] for t in trees}
    seg0 = _measure_all(joined_only, trees)
    v = MEASURES.index("v_opening")
    before, after = np.isfinite(seg0).any(axis=1), np.isfinite(seg).any(axis=1)
    change = np.abs(seg - seg0)                             # a measure gained or lost counts as changed
    change[np.isnan(seg) != np.isnan(seg0)] = np.inf
    moved = np.nan_to_num(change, nan=0.0).max(axis=1)
    grow = {"source": str(src), "low": low, "high": high, "chosen_by": "mean clDice against the lined-up hand masks",
            "cldice": choice or {}, "masks_grown": int((grown_px > 0).sum()), "pixels_added": int(grown_px.sum()),
            "no_fork_joined_only": int((~before).sum()), "no_fork_grown": int((~after).sum()),
            "v_opening_joined_only": int(np.isfinite(seg0[:, v]).sum()),
            "v_opening_grown": int(np.isfinite(seg[:, v]).sum()),
            "masks_whose_measures_changed": int((moved > 0.5).sum())}
    progress(f"Grown into pixels above {low:g} connected to each mask: {_n(grow['masks_grown'], 'mask')} of "
             f"{len(trees)} grew ({grow['pixels_added']} pixels); then joined within {gap:g} px. Masks with no fork "
             f"(joined only -> grown) {grow['no_fork_joined_only']} -> {grow['no_fork_grown']}; with a V opening "
             f"{grow['v_opening_joined_only']} -> {grow['v_opening_grown']}; measures moved by more than 0.5 degrees "
             f"in {grow['masks_whose_measures_changed']}")
    masks_summary, note, pixel = _pixel_columns(src, trees, final, hands)
    header = ["tree", "row", "fold", "fork_found", "fork_found_joined_only", "pixels_grown", "pieces_joined",
              "largest_change_deg", *(MASK_COLUMNS if hands is not None else
                                      tuple(f"{c}_before_growing" for c in MASK_COLUMNS))]
    rows = [[t, info[t]["row"], info[t]["fold"], int(after[i]), int(before[i]), int(grown_px[i]), int(pieces[i]),
             f"{moved[i]:.2f}", *pixel[t]] for i, t in enumerate(trees)]
    top = [trees[i] for i in np.argsort(-moved)[:48] if moved[i] > 0.5]
    failed = _rescored_run(src, source, info, trees, final, seg, out,
                           {"masks": masks_summary, "masks_note": note, "grow": grow, "join": {"gap": gap}},
                           header, rows, figures, ("masks", {t: masks[t] for t in top}, {t: final[t] for t in top}),
                           ((src.name, "segmentation"), (f"{src.name}_joined", "segmentation")), progress)
    return out, failed


def continued_run(src: Path, cone: float = CONTINUE_CONE, out: Path | None = None, figures: Path | None = None,
                  progress: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """Score a saved joined run's masks again after continuing short leaders and pruning (:func:`continue_leaders`).

    Like :func:`joined_run`, from the run's own files (no photos). ``src`` is a
    joined run (e.g. ``segment_lined_up_rerun_joined``); the result goes to
    ``<run>_continued`` beside it (``_joined`` dropped from the name) and is
    compared tree by tree with ``src`` and the other saved runs. Pixel scores
    are the run's own, of the masks before the change.

    Returns:
        ``(out, paths that could not be written)``.
    """
    src = Path(src)
    source, info = _read_run(src)
    masks = load_masks(src / "predicted_masks.npz")
    trees = [t for t in info if t in masks]
    out = Path(out) if out else src.parent / f"{src.name.removesuffix('_joined')}_continued"
    final, pieces = {}, np.zeros(len(trees), int)
    for i, t in enumerate(trees):
        final[t], pieces[i] = continue_leaders(masks[t], cone)
    seg, seg0 = _measure_all(final, trees), _measure_all(masks, trees)
    v = MEASURES.index("v_opening")
    change = np.abs(seg - seg0)
    change[np.isnan(seg) != np.isnan(seg0)] = np.inf
    moved = np.nan_to_num(change, nan=0.0).max(axis=1)
    removed = np.array([int((masks[t] & ~final[t]).sum()) for t in trees])
    cont = {"source": str(src), "cone": cone, "masks_continued": int((pieces > 0).sum()),
            "pieces_joined": int(pieces.sum()), "pixels_pruned": int(removed.sum()),
            "v_opening_before": int(np.isfinite(seg0[:, v]).sum()), "v_opening_after": int(np.isfinite(seg[:, v]).sum()),
            "masks_whose_measures_changed": int((moved > 0.5).sum())}
    progress(f"Short leaders continued to pieces on their line (within {cone:g} degrees and the missing stretch) in "
             f"{_n(cont['masks_continued'], 'mask')} of {len(trees)} ({_n(cont['pieces_joined'], 'piece')}); every mask "
             f"cut down to its trunk and two leaders ({cont['pixels_pruned']} pixels removed). With a V opening "
             f"{cont['v_opening_before']} -> {cont['v_opening_after']}; measures moved by more than 0.5 degrees in "
             f"{cont['masks_whose_measures_changed']}")
    masks_summary, _, pixel = _pixel_columns(src, trees, final, None)
    header = ["tree", "row", "fold", "fork_found", "pieces_joined", "pixels_pruned", "largest_change_deg",
              *(f"{c}_before" for c in MASK_COLUMNS)]
    rows = [[t, info[t]["row"], info[t]["fold"], int(np.isfinite(seg[i]).any()), int(pieces[i]), int(removed[i]),
             f"{moved[i]:.2f}", *pixel[t]] for i, t in enumerate(trees)]
    shown = [t for i, t in enumerate(trees) if pieces[i]] or [trees[i] for i in np.argsort(-moved)[:24]]
    failed = _rescored_run(src, source, info, trees, final, seg, out,
                           {"masks": masks_summary, "masks_note": "pixel scores of the masks before continuing and pruning",
                            "continue": cont, "join": source.get("join", {})},
                           header, rows, figures, ("masks", {t: masks[t] for t in shown}, {t: final[t] for t in shown}),
                           ((src.name, "segmentation"),), progress)
    return out, failed


def completed_run(src: Path, out: Path | None = None, figures: Path | None = None,
                  progress: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """Score a saved joined run again with its short leaders' lines predicted over the whole stretch.

    The model (:func:`fit_completion`) is fitted on the run's own complete
    leaders cut short on purpose (:func:`completion_samples`), for each fold
    on the other folds' trees only, so no tree's prediction uses its own
    leaders; the fit on all trees is kept for the final test. The masks are
    not changed. The result goes to ``<run>_completed`` (``_joined`` dropped)
    and is compared tree by tree with ``src`` and the other saved runs.

    Returns:
        ``(out, paths that could not be written)``.
    """
    src = Path(src)
    source, info = _read_run(src)
    masks = load_masks(src / "predicted_masks.npz")
    trees = [t for t in info if t in masks]
    out = Path(out) if out else src.parent / f"{src.name.removesuffix('_joined')}_completed"
    folds = sorted({info[t]["fold"] for t in trees})
    samples = {f: completion_samples({t: masks[t] for t in trees if info[t]["fold"] != f}) for f in folds}
    coef = {f: fit_completion(samples[f]) for f in folds}
    coef_all = fit_completion(completion_samples({t: masks[t] for t in trees}))
    seg = np.full((len(trees), len(MEASURES)), np.nan)
    done = np.zeros(len(trees), int)
    for i, t in enumerate(trees):
        m, done[i] = complete_measures(masks[t], coef[info[t]["fold"]])
        seg[i] = [m[k] for k in MEASURES]
    comp = {"source": str(src), "model": "whole - cut = a*g + c*g*bend (outward line from the vertical)",
            "fractions": list(COMPLETE_FRACTIONS), "coef_all": [float(v) for v in coef_all],
            "coef_by_fold": {str(f): [float(v) for v in c] for f, c in coef.items()},
            "leaders_completed": int(done.sum()), "trees_completed": int((done > 0).sum())}
    progress(f"Short leaders' lines predicted over the whole stretch in {_n(comp['trees_completed'], 'tree')} "
             f"({_n(comp['leaders_completed'], 'leader')}); model fitted per fold on the other folds' complete leaders "
             f"(all trees: a = {coef_all[0]:+.2f}, c = {coef_all[1]:+.3f})")
    masks_summary, _, pixel = _pixel_columns(src, trees, masks, None)
    header = ["tree", "row", "fold", "fork_found", "leaders_completed", *MASK_COLUMNS]
    rows = [[t, info[t]["row"], info[t]["fold"], int(np.isfinite(seg[i]).any()), int(done[i]), *pixel[t]]
            for i, t in enumerate(trees)]
    failed = _rescored_run(src, source, info, trees, masks, seg, out,
                           {"masks": masks_summary, "masks_note": "the masks are not changed", "complete": comp,
                            "join": source.get("join", {})},
                           header, rows, figures, None, ((src.name, "segmentation"),), progress)
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


def _grow_and_score(run: Path, hands: dict[str, np.ndarray], gap: float, figures: Path | None,
                    low: float | None = None) -> list[str]:
    """Choose the lower threshold on the hand masks (unless given) and score the run's grown masks."""
    print("\n--- The same masks grown into the less sure pixels connected to them, then joined ---")
    if low is None:
        print("Choosing the lower threshold by the hand masks' centre lines (mean clDice; the winter traces play "
              "no part):", flush=True)
        probs = load_probabilities(run / "probabilities.npz")
        seeds = load_masks(run / "predicted_masks.npz")
        low, choice = choose_low(probs, seeds, hands, gap=gap, progress=lambda m: print(m, flush=True))
        print(f"Chosen: {low:g}", flush=True)
    else:
        choice = None
    return grown_run(run, low, choice, hands or None, gap, figures=figures)[1]


def _grow_command(args) -> int:
    """``--grow DIR``: grow a saved run's masks, with ``--grow-low`` or a threshold chosen on the hand masks."""
    run = Path(args.grow)
    if args.grow_low is None and not args.data_dirs:
        print("Choosing the lower threshold needs the hand masks: give the dataset folder(s), or --grow-low.")
        return 1
    hands: dict[str, np.ndarray] = {}
    if args.data_dirs:
        from treeskel.io import index_dataset
        from treeskel.summer_gt import load_overlays

        source = json.loads((run / "report.json").read_text())
        overlays = load_overlays(source["mask_overlay"]) if source.get("mask_overlay") else None
        records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
        for t in load_masks(run / "predicted_masks.npz"):
            hm = hand_mask(records[t], overlays) if t in records else None
            if hm is not None:
                hands[t] = hm
    failed = _grow_and_score(run, hands, args.join_gap, Path(args.figures), args.grow_low)
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


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
    ap.add_argument("--size", type=int, nargs="+", default=[SegParams.size], metavar="N",
                    help="one number: side of the square input, the masks' frame (default 256); two, H W: the "
                         "whole photo at that size, e.g. 480 640 (each divisible by 32)")
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
    ap.add_argument("--grow", metavar="DIR",
                    help="only score a saved run's masks again, grown into the less sure pixels connected to them "
                         "and joined (writes DIR_grown); the lower threshold is --grow-low, or chosen on the hand "
                         "masks if the dataset folder(s) are given")
    ap.add_argument("--grow-low", type=float, help="the lower threshold for --grow (default: chosen by clDice)")
    ap.add_argument("--complete", metavar="DIR",
                    help="only score a saved joined run again with its short leaders' lines predicted over the "
                         "whole stretch from complete leaders' shapes (writes <run>_completed)")
    ap.add_argument("--continue", dest="continue_run", metavar="DIR",
                    help="only score a saved joined run's masks again with short leaders continued to pieces on their "
                         "line, and cut down to the trunk and two leaders (writes <run>_continued)")
    ap.add_argument("--overwrite", action="store_true", help="train even if the results folder holds a run")
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
    if args.grow:
        return _grow_command(args)
    if args.complete:
        out, failed = completed_run(Path(args.complete), out=Path(args.out) if args.out else None,
                                    figures=Path(args.figures))
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if args.continue_run:
        out, failed = continued_run(Path(args.continue_run), out=Path(args.out) if args.out else None,
                                    figures=Path(args.figures))
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if not args.data_dirs:
        ap.error("give the dataset folder(s), or --replot, --join, --grow, --continue or --complete")
    if len(args.size) > 2 or any(v % 32 for v in args.size):
        ap.error(f"--size takes one or two numbers, each divisible by 32; got {args.size}")
    size = args.size[0] if len(args.size) == 1 else (args.size[0], args.size[1])

    from treeskel.io import index_dataset

    params = SegParams(size=size, encoder=args.encoder, epochs=args.epochs, batch=args.batch, seed=args.seed,
                       depth=args.depth)
    out = Path(args.out or f"docs/results/{variant(params, args.mask_overlay is not None)}")
    if (out / "report.json").exists() and not args.overwrite:
        print(f"{out} already holds a run. Give another --out (e.g. {out}_2), or --overwrite to replace it.")
        return 1
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
    frame = "whole photos" if whole_photo(params) else "masks' frame"
    print(f"{len(trees)} development trees; loading the summer {what} ({frame}) at "
          f"{input_shape(params)[0]}x{input_shape(params)[1]}...", flush=True)
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
    probs = np.zeros((len(trees),) + MASK_SHAPE, np.float32)          # in the masks' frame, for growing
    seg = np.full((len(trees), len(MEASURES)), np.nan)
    scores = {t: dict.fromkeys(MASK_COLUMNS, float("nan")) for t in trees}
    for i, t in enumerate(trees):
        if not done[i]:
            continue
        probs[i] = prob_frame(cv["prob"][i], whole_photo(params))
        pred_masks[i] = probs[i] > params.threshold                  # as to_mask_size
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

    report = {"n_trees": int(done.sum()), "n_masks": int((done & has).sum()),
              "variant": out.name, "recipe": variant(params, overlays is not None),   # its folder; its settings' name
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
    write(out / "probabilities.npz", lambda p: save_probabilities(p, [t for i, t in enumerate(trees) if done[i]],
                                                                  probs[done]))
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
    if (out / "predictions.csv").exists() and (out / "probabilities.npz").exists():
        hands = {t: hand_mask(records[t], overlays) for i, t in enumerate(trees) if done[i] and has[i]}
        failed += _grow_and_score(out, hands, args.join_gap, figures)
    for f in failed:
        print(f"Could not write {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
