"""Measure the leaders' widths in the winter photos, to set and check the width model.

Until session 8 the leaders were drawn about half as wide as their bark:
their widths came from scale-space ridge detection, which responds to the
sunlit half of a branch lit from one side. They are now measured across
the branch in the colour image (:func:`treeskel.guided_gt.branch_edges`)
and drawn at each leader's own width, widened near the fork by a shared
amount (:func:`treeskel.guided_gt.fit_leader_radius`). This module
measures, on **development-row trees only**, and only with cross-sections
that cannot reach the other leader (near the fork they would take it in
too, see :func:`treeskel.guided_gt.clear_of`):

1. **Depth overshoot.** Depth is a second, independent sensor, but its
   silhouette reaches past a branch's edges. Where the depth run through a
   cross-section ends in "no reading" on both sides (nothing behind to
   merge with) and the colour changes sharply at an edge, the depth edge
   minus the colour edge measures the overshoot (:func:`edge_overshoot`).
2. **Agreement per leader.** Each leader's drawn width far from the fork
   (at least 8 trunk radii), before and after, against the depth width
   there minus the overshoot.
3. **Width along the leaders.** Pooled medians by distance from the fork,
   relative to the trunk's radius at the fork: the new cross-section
   widths, the old drawn radii, the raw depth width, and the provided hand
   masks (:func:`hand_mask_profiles`), which are drawn on the summer photos
   but whose ratios need no alignment.
4. **The fork widening.** Each leader's widths beyond 1.5 trunk radii are
   fitted as ``c · (1 + w · exp(−s / (L · R)))`` with ``c`` per leader and
   ``(w, L)`` shared, over a grid (:func:`widening_costs`). The shared
   values are chosen with each tree, and each fold of the row-grouped
   cross-validation in ``splits.json``, held out in turn
   (:func:`choose_shape`), and compared with a constant width (``w = 0``).
   The whole cost surface is saved, and the free straight tapers used
   before are summarised.
5. **The crotch.** Mask pixels between the leaders near the fork that are
   off the depth silhouette (:func:`treeskel.measure_fork.crotch_zone`),
   before and after, and after without the widening: wider leaders fill the
   bottom of the V. "Before" has the trunk's round top (and the shoulder
   fill), "after" the current fork junction; :mod:`treeskel.measure_fork`
   compares the two ways of drawing the fork on the same leaders.
6. **Centring.** How far the traced path is from the bark's centre (the
   middle of the two edges found across it); the mask is centred on the
   path.

Usage, from the command line:

```
python -m treeskel.measure_width data/raw --clicks annotations/winter_clicks.json \\
    --split splits.json --out outputs/measure_width --figure outputs/leader_width.jpg \\
    --plot outputs/leader_width_profiles.png
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from treeskel.guided_gt import (
    NEIGHBOUR_MARGIN,
    GuidedParams,
    arc_length,
    branch_cost,
    branch_edges,
    clear_of,
    guided_gt,
    huber_line,
    huber_location,
    load_clicks,
    straighten,
)
from treeskel.measure_fork import crotch_zone
from treeskel.winter_gt import find_main_branches, render_main_branches, report_failed, save_image

OLD = GuidedParams(leader_width_source="scale_space", leader_fit="taper", shoulder_length=1.4,
                  junction_reach=0, flat_root=False)
"""The settings of the masks before session 8 (scale-space widths, free straight tapers, the
session-6 shoulder fill on the trunk's round top); they reproduce those masks pixel for pixel."""

BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0)
"""Distance from the fork, in trunk radii, for the pooled width profiles."""

WIDENINGS = tuple(np.round(np.arange(0.0, 1.01, 0.05), 2))
"""Widening at the fork (fraction) tried by :func:`widening_costs`."""

LENGTHS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
"""Widening lengths (trunk radii) tried by :func:`widening_costs`. The first grid stopped at
12; on all 449 development trees every leave-one-tree-out fit chose 12, so it was extended."""

FIT_START = 1.5
"""Nearest distance from the fork (trunk radii) of the widths used in the fits."""

FAR = 8.0
"""Leaders are compared with depth at least this many trunk radii from the fork."""

CONTRAST = 60.0
"""A colour edge is clear if the mean grey level changes by more than this across it."""

HUBER_SCALE = 1.5
"""Fixed robust scale (px) of the residuals, so fits of different shapes cost alike."""


def centre_runs(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The run of set samples through the middle of each row of a straightened mask.

    Args:
        strip: ``(N, 2X + 1)`` bool, e.g. the depth foreground sampled
            across a path (:func:`treeskel.guided_gt.straighten`).

    Returns:
        ``(first, last, closed)``: offsets (px) of the run's ends, and
        whether the run is set at the centre and ends inside the row on
        both sides (so nothing next to it merged into it beyond the row).
    """
    half = strip.shape[1] // 2
    left, right = ~strip[:, half::-1], ~strip[:, half:]           # outwards from the centre
    steps_l = np.where(left.any(1), left.argmax(1), half + 1)
    steps_r = np.where(right.any(1), right.argmax(1), half + 1)
    closed = strip[:, half] & left.any(1) & right.any(1)
    return 1 - steps_l, steps_r - 1, closed


def edge_overshoot(grey: np.ndarray, first: np.ndarray, last: np.ndarray, d_first: np.ndarray,
                   d_last: np.ndarray, closed: np.ndarray, contrast: float = CONTRAST,
                   max_gap: int = 6) -> np.ndarray:
    """How far the depth silhouette reaches past clear colour edges (px, outwards positive).

    Args:
        grey: ``(N, 2X + 1)`` straightened grey levels (0–255).
        first, last: Colour edges per cross-section (offsets, px).
        d_first, d_last: Depth run ends per cross-section.
        closed: Cross-sections whose depth run is closed (:func:`centre_runs`).
        contrast: Least change in mean grey level across a colour edge
            (3 samples each side) for it to count as clear.
        max_gap: Edges further apart than this are left out (depth merged
            with something, or a depth hole).

    Returns:
        1-D array of overshoots, one per clear edge.
    """
    half = grey.shape[1] // 2
    out = []
    for i in np.nonzero(closed)[0]:
        g = grey[i]
        for edge, d_edge, outward in ((first[i], d_first[i], -1), (last[i], d_last[i], 1)):
            inside = g[np.clip(half + edge - outward * np.arange(3), 0, len(g) - 1)].mean()
            outside = g[np.clip(half + edge + outward * np.arange(1, 4), 0, len(g) - 1)].mean()
            gap = outward * (d_edge - edge)
            if abs(inside - outside) > contrast and abs(gap) <= max_gap:
                out.append(gap)
    return np.array(out, float)


def hand_mask_profiles(mask: np.ndarray) -> dict | None:
    """Width along the trunk and leaders of a hand-drawn mask, relative to its trunk.

    The mask is thinned to its centre lines, the root, fork and two leaders
    are picked as for the winter depth (:func:`treeskel.winter_gt.find_main_branches`),
    and the half-width at each centre-line point is its distance to the
    mask's edge. Everything is divided by the trunk's half-width ``T`` (the
    median 3–40 mask px below the fork), so the mask's scale does not matter.

    Returns:
        ``{"left": (s/T, w/T), "right": (s/T, w/T)}`` with ``s`` measured
        from the fork, or None if no fork is found.
    """
    m = ndi.binary_fill_holes(mask)
    sk = skeletonize(m)
    lab, n = ndi.label(sk, np.ones((3, 3)))
    if n > 1:
        sk = lab == (np.argmax(ndi.sum(sk, lab, range(1, n + 1))) + 1)
    br = find_main_branches(m, sk)
    if br["fork"] is None:
        return None
    edt = ndi.distance_transform_edt(m)
    trunk = br["trunk"][::-1]                                       # from the fork down
    s, w = arc_length(trunk), edt[trunk[:, 0], trunk[:, 1]]
    near = (s > 3) & (s < 40)
    t = float(np.median(w[near])) if near.sum() > 3 else float(np.median(w))
    return {side: (arc_length(br[side]) / t, edt[br[side][:, 0], br[side][:, 1]] / t) for side in ("left", "right")}


def widening_costs(s_over_r: np.ndarray, widths: np.ndarray, widenings: Sequence[float] = WIDENINGS,
                   lengths: Sequence[float] = LENGTHS, start: float = FIT_START, min_valid: float = 2.5) -> np.ndarray:
    """Robust misfit of one leader's widths for each shared widening shape.

    For each ``(w, L)``, the leader's own width ``c`` is the Huber average of
    ``widths / (1 + w · exp(−s/(L·R)))`` (as in
    :func:`treeskel.guided_gt.fit_leader_radius`) and the cost is the mean
    Huber loss (scale :data:`HUBER_SCALE`) of the residuals. Only points at
    least ``start`` trunk radii from the fork with widths above
    ``min_valid`` are used.

    Args:
        s_over_r: ``(N,)`` distance from the fork, in trunk radii.
        widths: ``(N,)`` measured half-widths (px).

    Returns:
        ``(len(widenings), len(lengths))`` costs (nan if too few points).
    """
    use = (s_over_r >= start) & (widths > min_valid)
    out = np.full((len(widenings), len(lengths)), np.nan)
    if use.sum() < 8:
        return out
    s, y = s_over_r[use], widths[use]
    k = 1.345 * HUBER_SCALE
    for i, w in enumerate(widenings):
        for j, length in enumerate(lengths):
            g = 1 + w * np.exp(-s / length)
            res = np.abs(y - huber_location(y / g) * g)
            out[i, j] = float(np.mean(np.where(res <= k, 0.5 * res ** 2, k * (res - 0.5 * k))))
    return out


def free_taper(s: np.ndarray, widths: np.ndarray, fork_radius: float, start: float = FIT_START) -> float:
    """Relative change of a free straight taper from ``start`` trunk radii to the tip (nan if unfit)."""
    use = (s >= start * fork_radius) & (widths > 2.5)
    if use.sum() < 8:
        return float("nan")
    a, b = huber_line(s[use], widths[use])
    base = a + b * s[use].min()
    return float(b * (s[-1] - s[use].min()) / base) if base > 0 else float("nan")


def choose_shape(costs: np.ndarray, groups: np.ndarray) -> dict:
    """Choose the shared widening with each group held out in turn.

    For each group, the ``(w, L)`` with the lowest mean cost over the other
    groups' leaders is picked, and its mean cost on the held-out group's
    leaders is compared with a constant width (``w = 0``).

    Args:
        costs: ``(n_leaders, len(WIDENINGS), len(LENGTHS))`` costs from
            :func:`widening_costs` (no nan).
        groups: ``(n_leaders,)`` group of each leader: its tree (to hold out
            one tree at a time) or its fold of row-grouped cross-validation.

    Returns:
        ``{"groups", "picks", "held_out", "held_out_constant"}``, one entry per
        group that has leaders both in and out of it; ``picks`` are
        ``(w, L)`` pairs.
    """
    out: dict = {"groups": [], "picks": [], "held_out": [], "held_out_constant": []}
    for g in sorted(set(groups.tolist())):
        train, test = costs[groups != g], costs[groups == g]
        if not len(train) or not len(test):
            continue
        a, b = np.unravel_index(np.argmin(train.mean(0)), train.shape[1:])
        out["groups"].append(g)
        out["picks"].append((float(WIDENINGS[a]), float(LENGTHS[b])))
        out["held_out"].append(float(test[:, a, b].mean()))
        out["held_out_constant"].append(float(test[:, 0, 0].mean()))
    return out


def _held_out_summary(chosen: dict) -> dict:
    """Ranges of the picks and the mean held-out costs of :func:`choose_shape`."""
    held, const = np.array(chosen["held_out"]), np.array(chosen["held_out_constant"])
    return {"widening": [min(p[0] for p in chosen["picks"]), max(p[0] for p in chosen["picks"])],
            "length": [min(p[1] for p in chosen["picks"]), max(p[1] for p in chosen["picks"])],
            "picks": [list(p) for p in chosen["picks"]],
            "held_out_cost": float(held.mean()), "held_out_cost_constant": float(const.mean()),
            "better_than_constant": int((held < const).sum()), "n_groups": len(held)}


def _crop(image: np.ndarray, centre: Sequence[int], half: int) -> np.ndarray:
    r0 = int(np.clip(centre[0] - half, 0, image.shape[0] - 2 * half))
    c0 = int(np.clip(centre[1] - half, 0, image.shape[1] - 2 * half))
    return image[r0:r0 + 2 * half, c0:c0 + 2 * half]


def _outline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    im = rgb.copy()
    im[mask & ~ndi.binary_erosion(mask)] = (255, 255, 0)
    return im


def _paint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask on the image: a red wash with a yellow outline."""
    im = rgb.astype(float)
    im[mask] = 0.55 * im[mask] + 0.45 * np.array([255, 0, 60])
    return _outline(im.astype(np.uint8), mask)


def measure_tree(job: tuple) -> dict:
    """All measurements for one tree (top-level so worker processes can run it).

    Args:
        job: ``(tree_record, clicks, make_tiles)``.

    Returns:
        Per-leader arrays and summaries, the hand-mask profiles, the depth
        overshoots and, if asked, crops for the figure.
    """
    from treeskel.io import load_mask, load_tree

    rec, clicks, make_tiles = job
    d = load_tree(rec, "winter")
    cost, _ = branch_cost(d["rgb"], d["depth"], d["valid"])
    new = guided_gt(d["rgb"], d["depth"], d["valid"], clicks, cost=cost)
    old = guided_gt(d["rgb"], d["depth"], d["valid"], clicks, OLD, cost=cost)
    flat = guided_gt(d["rgb"], d["depth"], d["valid"], clicks, GuidedParams(fork_widening=0), cost=cost)
    radius = float(new.radii["trunk"][-1])
    crotch = crotch_zone(new)
    crotch_px = {name: [int((m & crotch & ~new.tree_mask).sum()), int((m & crotch).sum())]
                 for name, m in (("before", render_main_branches(old)), ("after", render_main_branches(new)),
                                 ("after_no_widening", render_main_branches(flat)))}
    grey = d["rgb"].astype(float).mean(2)
    half = GuidedParams().profile_half_width
    leaders, overshoot = [], []
    for side, other in (("left", new.right), ("right", new.left)):
        path = getattr(new, side)
        s = arc_length(path)
        first, last = branch_edges(d["rgb"], path, half, GuidedParams().profile_along)
        # Cross-sections that could take in the other leader (near the fork) are not used.
        clear = clear_of(path, other, half + NEIGHBOUR_MARGIN)
        widths = np.where(clear, (last - first + 1) / 2.0, np.nan)
        d_first, d_last, closed = centre_runs(straighten(new.tree_mask.astype(float), path, half, order=0) > 0.5)
        closed &= clear
        overshoot.append(edge_overshoot(straighten(grey, path, half), first, last, d_first, d_last, closed))
        depth_w = np.where(closed, (d_last - d_first + 1) / 2.0, np.nan)
        far = s >= FAR * radius
        leaders.append({
            "side": side, "s_over_r": s / radius, "widths": widths, "depth": depth_w,
            "first_clear": float(s[clear].min() / radius) if clear.any() else float("nan"),
            # How far the traced path is from the bark's centre (px), where the edges were found.
            "off_centre": np.abs((first + last) / 2.0)[clear & (widths > 2.5)],
            "old": old.radii[side], "new": new.radii[side],
            "old_far": float(np.median(old.radii[side][far])) if far.any() else float("nan"),
            "new_far": float(np.median(new.radii[side][far])) if far.any() else float("nan"),
            "depth_far": float(np.nanmedian(depth_w[far])) if (far & closed).any() else float("nan"),
            "free_taper": free_taper(s, widths, radius),
            "costs": widening_costs(s / radius, widths),
        })
    hand = None
    if rec.has_mask:
        hand = hand_mask_profiles(load_mask(rec.paths[("summer", "mask")]))
    tiles = None
    if make_tiles:
        before, after = render_main_branches(old), render_main_branches(new)
        tiles = {"whole": [_paint(d["rgb"], before), _paint(d["rgb"], after)], "crops": []}
        for side in ("left", "right"):
            path = getattr(new, side)
            for frac in (0.0, 0.3, 0.6):
                c = path[int(frac * (len(path) - 1))]
                tiles["crops"].append([_crop(im, c, 32) for im in (d["rgb"], _outline(d["rgb"], before),
                                                                  _outline(d["rgb"], after))])
    return {"tree": rec.tree_id, "row": rec.row, "fork_radius": radius, "leaders": leaders, "hand": hand,
            "overshoot": np.concatenate(overshoot), "crotch": crotch_px, "tiles": tiles}


def _binned(s_over_r: np.ndarray, values: np.ndarray) -> list[float]:
    """Median of ``values`` in each :data:`BINS` interval (nan if fewer than 3)."""
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        v = values[(s_over_r >= lo) & (s_over_r < hi) & np.isfinite(values)]
        out.append(float(np.median(v)) if len(v) >= 3 else float("nan"))
    return out


def _nanmedian_rows(rows: list[list[float]]) -> list[float | None]:
    a = np.array(rows, float)
    return [None if np.all(np.isnan(col)) else float(np.nanmedian(col)) for col in a.T]


def summarise(results: list[dict], folds: Sequence[Sequence[int]] | None = None) -> dict:
    """Pool :func:`measure_tree` results into the report.

    Args:
        results: One :func:`measure_tree` result per tree.
        folds: Rows in each fold of row-grouped cross-validation (the
            ``fold_val_rows`` of ``splits.json``), to choose the widening with
            whole folds held out as well as single trees.
    """
    leaders = [(r["tree"], r["fork_radius"], ld) for r in results for ld in r["leaders"]]
    over = np.concatenate([r["overshoot"] for r in results])
    bleed = float(np.median(over)) if len(over) else float("nan")
    first_clear = np.array([ld["first_clear"] for _, _, ld in leaders], float)
    first_clear = first_clear[~np.isnan(first_clear)]
    out = {"trees": [r["tree"] for r in results], "n_leaders": len(leaders),
           "first_clear": {"median": float(np.median(first_clear)) if len(first_clear) else None,
                           "iqr": [float(v) for v in np.percentile(first_clear, [25, 75])] if len(first_clear)
                           else None},
           "overshoot": {"n_edges": int(len(over)), "median": bleed,
                         "iqr": [float(v) for v in np.percentile(over, [25, 75])] if len(over) else None,
                         "mean": float(np.mean(over)) if len(over) else None}}
    # Agreement per leader, far from the fork.
    rows = np.array([[ld["old_far"], ld["new_far"], ld["depth_far"] - bleed] for _, _, ld in leaders], float)
    ok = ~np.isnan(rows).any(1)
    diff_new, diff_old = rows[ok, 1] - rows[ok, 2], rows[ok, 0] - rows[ok, 2]
    out["far_from_fork"] = {
        "n_leaders": int(ok.sum()), "min_trunk_radii": FAR,
        "median_half_width": {"old": float(np.median(rows[ok, 0])), "new": float(np.median(rows[ok, 1])),
                              "depth_minus_overshoot": float(np.median(rows[ok, 2]))},
        "new_minus_depth": {"median": float(np.median(diff_new)),
                            "iqr": [float(v) for v in np.percentile(diff_new, [25, 75])]},
        "old_minus_depth": {"median": float(np.median(diff_old)),
                            "iqr": [float(v) for v in np.percentile(diff_old, [25, 75])]}} if ok.any() else None
    # Width along the leaders, relative to the trunk's radius at the fork.
    prof = {"bins": list(BINS)}
    for key in ("widths", "depth", "old", "new"):
        prof[key] = _nanmedian_rows([_binned(ld["s_over_r"], ld[key] / rad) for _, rad, ld in leaders])
    prof["depth_corrected"] = _nanmedian_rows([_binned(ld["s_over_r"], (ld["depth"] - bleed) / rad)
                                               for _, rad, ld in leaders])
    hands = [h[side] for r in results if (h := r["hand"]) for side in ("left", "right")]
    prof["hand"] = _nanmedian_rows([_binned(s, w) for s, w in hands]) if hands else None
    prof["n_hand_leaders"] = len(hands)
    out["profile"] = prof
    # The shared widening: the whole cost surface, its best point, and the point chosen with
    # each tree, and each row-grouped fold, held out in turn, against a constant width.
    costs = np.array([ld["costs"] for _, _, ld in leaders])
    trees = np.array([t for t, _, _ in leaders])
    rows_of = np.array([r["row"] for r in results for _ in r["leaders"]])
    fine = ~np.isnan(costs).any((1, 2))
    costs, trees, rows_of = costs[fine], trees[fine], rows_of[fine]
    total = costs.mean(0)
    i, j = np.unravel_index(np.argmin(total), total.shape)
    default = GuidedParams()
    on_grid = default.fork_widening in WIDENINGS and default.widening_length in LENGTHS
    by_tree = _held_out_summary(choose_shape(costs, trees))
    by_fold = None
    if folds:
        fold_of = np.array([next((k for k, f in enumerate(folds) if row in f), -1) for row in rows_of])
        chosen = choose_shape(costs[fold_of >= 0], fold_of[fold_of >= 0])
        by_fold = _held_out_summary(chosen) if chosen["groups"] else None
    out["crotch_off_depth"] = {name: [sum(r["crotch"][name][k] for r in results) for k in (0, 1)]
                               for name in ("before", "after", "after_no_widening")}
    off = np.concatenate([ld["off_centre"] for _, _, ld in leaders])
    per_leader = [float(np.median(ld["off_centre"])) for _, _, ld in leaders if len(ld["off_centre"])]
    out["off_centre"] = {"n_cross_sections": int(len(off)), "median": float(np.median(off)) if len(off) else None,
                         "over_3px": float(np.mean(off > 3)) if len(off) else None,
                         "per_leader_median_range": [min(per_leader), max(per_leader)] if per_leader else None}
    tapers = np.array([ld["free_taper"] for _, _, ld in leaders], float)
    tapers = tapers[~np.isnan(tapers)]
    out["widening"] = {
        "best": {"widening": float(WIDENINGS[i]), "length": float(LENGTHS[j]),
                 "at_fit_start": float(WIDENINGS[i] * np.exp(-FIT_START / LENGTHS[j])),
                 "at_grid_edge": bool(i == len(WIDENINGS) - 1 or j == len(LENGTHS) - 1)},
        "cost_best": float(total[i, j]), "cost_constant": float(total[0, 0]),
        "default": {"widening": default.fork_widening, "length": default.widening_length,
                    "cost": float(total[WIDENINGS.index(default.fork_widening),
                                        LENGTHS.index(default.widening_length)]) if on_grid else None},
        "grid": {"widenings": list(WIDENINGS), "lengths": list(LENGTHS), "cost": total.round(5).tolist()},
        "leave_one_out": by_tree,
        "row_folds": by_fold,
        "free_taper": {"n_leaders": int(len(tapers)), "median": float(np.median(tapers)) if len(tapers) else None,
                       "iqr": [float(v) for v in np.percentile(tapers, [25, 75])] if len(tapers) else None,
                       "widening_upwards": float(np.mean(tapers > 0)) if len(tapers) else None}}
    return out


def _figure(results: list[dict], scale: int = 2) -> Image.Image:
    """Per tree: the whole tree before and after, then crops at the fork, 30% and 60% up each leader."""
    def titled(img: np.ndarray, text: str, width: int) -> np.ndarray:
        """The image with a black title strip above it, padded to ``width``."""
        pil = Image.new("RGB", (width, img.shape[0] + 14), "black")
        pil.paste(Image.fromarray(img), (0, 14))
        ImageDraw.Draw(pil).text((3, 1), text, fill="white")
        return np.asarray(pil)

    rows = []
    for r in results:
        crop_rows = []
        for tiles in (r["tiles"]["crops"][:3], r["tiles"]["crops"][3:]):
            parts = []
            for trio in tiles:
                strip = np.concatenate([np.pad(t, ((0, 0), (0, 2), (0, 0))) for t in trio], 1)
                parts.append(np.pad(strip, ((0, 0), (0, 8), (0, 0))))
            row = np.concatenate(parts, 1)
            crop_rows.append(np.asarray(Image.fromarray(row).resize((row.shape[1] * scale, row.shape[0] * scale),
                                                                    Image.NEAREST)))
        width = max(crop_rows[0].shape[1], 2 * 640 + 8)
        panel = ((width - 8) // 2, (width - 8) // 2 * 3 // 4)                # each whole tree, 4:3
        before, after = (np.asarray(Image.fromarray(im).resize(panel, Image.LANCZOS)) for im in r["tiles"]["whole"])
        rows.append(titled(np.concatenate([before, np.zeros((panel[1], 8, 3), np.uint8), after], 1),
                           f"{r['tree']}: masks before (left) and after (right)", width))
        for side, row in zip(("left", "right"), crop_rows):
            rows.append(titled(row, f"{r['tree']} {side} leader: fork, 30%, 60% up (winter RGB | before | after)",
                               width))
        rows.append(np.zeros((10, width, 3), np.uint8))
    return Image.fromarray(np.concatenate(rows, 0))


def plot_profiles(report: dict, path: Path, failed: list[str]) -> None:
    """Line chart of the width along the leaders: before and after, against the two references.

    One axis (half-width ÷ trunk radius at the fork) against distance from
    the fork; the masks before (orange) and after (blue) are the story, the
    hand masks and the overshoot-corrected depth (greys) are the context.
    Each line is labelled at its end as well as in the legend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, secondary, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    prof = report["profile"]
    mid = [(lo + hi) / 2 for lo, hi in zip(BINS[:-1], BINS[1:])]
    series = [("hand", "Hand masks (summer)", secondary), ("depth_corrected", "Depth, minus its overshoot", muted),
              ("new", "Masks after", "#2a78d6"), ("old", "Masks before", "#eb6834")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axvspan(0, FIT_START, color="#f0efec", lw=0, zorder=0)
    ax.text(FIT_START + 0.4, 1.05,
            "Shaded: too near the fork to measure (a cross-section spans the trunk or the other leader)",
            ha="left", va="center", fontsize=7, color=muted)
    ends = []
    for key, name, colour in series:
        values = prof.get(key)
        if not values:
            continue
        x = [m for m, v in zip(mid, values) if v is not None]
        y = [v for v in values if v is not None]
        ax.plot(x, y, color=colour, lw=1.0, solid_capstyle="round", solid_joinstyle="round", marker="o", ms=4,
                mec=surface, mew=1.0, label=name, zorder=3)
        ends.append((y[-1], x[-1], name))
    # End labels in ink, nudged apart only where they would overlap.
    ends.sort()
    placed: list[float] = []
    for y, x, name in ends:
        y_text = max(y, placed[-1] + 0.055) if placed else y
        placed.append(y_text)
        ax.annotate(name, (x, y), xytext=(x + 0.6, y_text), fontsize=7.5, color=secondary, va="center",
                    arrowprops={"arrowstyle": "-", "color": axis, "lw": 0.6} if abs(y_text - y) > 0.01 else None)
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 1.5, 5, 10, 15, 20], ["0", "1.5", "5", "10", "15", "20"])
    ax.set_xlabel("Distance along the leader from the fork (trunk radii)", fontsize=8.5, color=secondary)
    ax.set_ylabel("Half-width ÷ trunk radius at the fork", fontsize=8.5, color=secondary)
    ax.grid(axis="y", color=grid, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    n = report["n_leaders"]
    fig.suptitle("The leaders were drawn about half as wide as their bark", x=0.07, ha="left", fontsize=11,
                 color=ink, fontweight="bold")
    ax.set_title(f"Median over {n} leaders of {len(report['trees'])} development-row trees", loc="left",
                 fontsize=8.5, color=secondary)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=7.5, labelcolor=secondary)
    fig.tight_layout()
    try:
        fig.savefig(path, facecolor=surface)
    except OSError as e:
        failed.append(f"{path}: {e.strerror or e}")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure development-row trees; print and save the results."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.measure_width", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json", help="its test rows are never measured")
    ap.add_argument("--out", default="outputs/measure_width")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--figure", help="also save crops (winter RGB, masks before and after) as this image")
    ap.add_argument("--figure-trees", nargs="+", help="trees for the figure (default: the first three)")
    ap.add_argument("--plot", help="also save a chart of the width along the leaders as this image")
    args = ap.parse_args(argv)

    # Every annotated development-row tree; the test rows are never measured.
    split = json.loads(Path(args.split).read_text())
    test_rows = set(split["test_rows"])
    clicks = load_clicks(args.clicks)
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    ids = sorted(t for t, c in clicks.items()
                 if t in records and c.complete and not c.skip and records[t].row not in test_rows)
    held_out = sorted(t for t in records if records[t].row in test_rows)
    if not ids:
        print("No annotated development-row trees found; nothing to measure.")
        return 1
    fig_ids = set(args.figure_trees or ids[:3]) if args.figure else set()
    jobs = [(records[t], clicks[t], t in fig_ids) for t in ids]

    print(f"Measuring {len(ids)} development-row trees ({len(held_out)} test-row trees left out)...", flush=True)
    t0 = time.time()
    step = max(1, min(25, len(jobs) // 10))
    results = []
    # One tree per worker process, with progress about every 10% of the trees.
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
        for i, r in enumerate(ex.map(measure_tree, jobs, chunksize=2), 1):
            results.append(r)
            if i % step == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                      flush=True)

    # Pool the trees (the widening chosen on all trees, by leave-one-tree-out and by the row folds of
    # splits.json), save, then print each part of the report.
    report = summarise(results, split.get("fold_val_rows"))
    report["test_rows_excluded"] = sorted(test_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "leaders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "side", "fork_radius", "old_far", "new_far", "depth_far", "new_at_fork", "free_taper"])
        for r in results:
            for ld in r["leaders"]:
                w.writerow([r["tree"], ld["side"], round(r["fork_radius"], 2)]
                           + [round(float(ld[k]), 3) for k in ("old_far", "new_far", "depth_far")]
                           + [round(float(ld["new"][0]), 3), round(float(ld["free_taper"]), 4)])
    (out / "report.json").write_text(json.dumps(report, indent=2))

    ov, far, wd, fc = report["overshoot"], report["far_from_fork"], report["widening"], report["first_clear"]
    print(f"\n{report['n_leaders']} leaders on {len(ids)} trees.")
    if fc["median"] is not None:
        print(f"Cross-sections that could reach the other leader are left out: the first one used is a median "
              f"{fc['median']:.1f} trunk radii from the fork (IQR {fc['iqr'][0]:.1f}-{fc['iqr'][1]:.1f}).")
    if ov["n_edges"]:
        print(f"Depth reaches past clear colour edges by a median {ov['median']:+.1f} px "
              f"(IQR {ov['iqr'][0]:+.1f} to {ov['iqr'][1]:+.1f}, mean {ov['mean']:+.2f}; {ov['n_edges']} edges).")
    if far:
        m = far["median_half_width"]
        print(f"Leader half-width at least {FAR:g} trunk radii from the fork ({far['n_leaders']} leaders): "
              f"before {m['old']:.1f} px, after {m['new']:.1f} px, "
              f"depth minus overshoot {m['depth_minus_overshoot']:.1f} px")
        for key, name in (("new_minus_depth", "after"), ("old_minus_depth", "before")):
            v = far[key]
            print(f"  {name} minus depth: median {v['median']:+.1f} px (IQR {v['iqr'][0]:+.1f} to {v['iqr'][1]:+.1f})")
    prof = report["profile"]
    print("Half-width / trunk radius at the fork, by distance from the fork (trunk radii):")
    print("  " + "from" + " " * 13 + " ".join(f"{lo:>5g}" for lo in BINS[:-1]))
    for key, name in (("hand", "hand masks"), ("depth", "depth (raw)"), ("depth_corrected", "depth-overshoot"),
                      ("widths", "measured"), ("new", "after"), ("old", "before")):
        if prof.get(key):
            print(f"  {name:16s}" + " ".join("  -  " if v is None else f"{v:5.2f}" for v in prof[key]))
    b, dflt = wd["best"], wd["default"]
    print(f"Shared widening, best on all trees: {b['widening']:.2f} at the fork ({b['at_fit_start']:.2f} at "
          f"{FIT_START:g} trunk radii), fading over {b['length']:g} trunk radii; cost {wd['cost_best']:.4f} "
          f"(constant width {wd['cost_constant']:.4f}"
          + (f"; current default {dflt['widening']:g} and {dflt['length']:g}: {dflt['cost']:.4f}"
             if dflt["cost"] is not None else "") + ").")
    if b["at_grid_edge"]:
        print("  The best point is at the edge of the grid searched: the true best may lie beyond it.")
    for key, name, unit in (("leave_one_out", "Leave-one-tree-out", "trees"),
                            ("row_folds", "Row-grouped folds (splits.json)", "folds")):
        h = wd[key]
        if h:
            print(f"  {name}: picks {h['widening'][0]:.2f}-{h['widening'][1]:.2f} and {h['length'][0]:g}-"
                  f"{h['length'][1]:g}; held-out cost {h['held_out_cost']:.4f} vs {h['held_out_cost_constant']:.4f}"
                  f" for a constant width (better on {h['better_than_constant']} of {h['n_groups']} {unit}).")
    ft = wd["free_taper"]
    if ft["n_leaders"]:
        median = ft["median"] if abs(ft["median"]) >= 0.005 else 0.0      # print 0%, not -0%
        print(f"Free straight tapers (the old fit): median change {median:+.0%} from {FIT_START:g} trunk radii "
              f"to the tip (IQR {ft['iqr'][0]:+.0%} to {ft['iqr'][1]:+.0%}); "
              f"{ft['widening_upwards']:.0%} widen upwards.")
    cr = report["crotch_off_depth"]
    print("Mask pixels in the crotch that are off the depth silhouette (off / all): "
          + ", ".join(f"{name.replace('_', ' ')} {v[0]}/{v[1]}" for name, v in cr.items()))
    oc = report["off_centre"]
    if oc["median"] is not None:
        print(f"Traced path to the bark's centre: median {oc['median']:.1f} px over {oc['n_cross_sections']} "
              f"cross-sections, {oc['over_3px']:.0%} more than 3 px (per leader {oc['per_leader_median_range'][0]:.1f}-"
              f"{oc['per_leader_median_range'][1]:.1f} px).")
    failed: list[str] = []
    if args.figure:
        save_image(_figure([r for r in results if r["tiles"] is not None]), Path(args.figure), failed, quality=85)
    if args.plot:
        plot_profiles(report, Path(args.plot), failed)
    saved = [str(out / "leaders.csv"), str(out / "report.json")] + [p for p in (args.figure, args.plot) if p]
    print(f"Saved {', '.join(saved)}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
