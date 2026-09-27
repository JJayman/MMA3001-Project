"""Measure how far the depth images sit from the colour images (their registration).

Session 8 found that the winter depth images sit about 4 px below the colour
images (``docs/results/depth_registration/``). The tracing follows depth, so
the traced paths sit about 1 px off the bark on sloping leaders, and every
check of the masks against depth compared them with a shifted silhouette.
This module measures the shift on **development-row trees only** (the test
rows of ``splits.json`` are left out), so that the correction
(:data:`treeskel.io.DEPTH_SHIFT`) is not chosen on test data. The depth is
loaded as stored (``register=False``), unless ``--registered`` asks for a
check of the correction itself. Shifts are written ``(dy, dx)``: where the
depth sits relative to the colour, so positive ``dy`` means the depth is
lower. The correction moves the depth by about ``(−dy, −dx)``.

1. **Across the leaders (winter; the reference).** Along each traced leader,
   colour gives the bark's two edges (:func:`treeskel.guided_gt.branch_edges`)
   and the depth silhouette gives the ends of its run through the same
   cross-section (:func:`treeskel.measure_width.centre_runs`). If the images
   were registered, the run's centre would sit on the bark's centre, whatever
   the branch's direction; a shift ``s`` of the depth moves it by ``n · s``
   along the cross-section's normal ``n``. The difference of the centres is
   therefore fitted as ``n · s`` by robust (Huber) least squares
   (:func:`fit_shift`), on cross-sections clear of the other leader whose
   depth run is closed on both sides (:func:`leader_sections`). Depth's
   overshoot past the bark's edges cancels, being the same on both sides.
   The fit is made pooled over the trees, per orchard row, per tree, in bands
   across the image, and with an affine term (:func:`fit_affine`: does the
   shift change across the image?).

   The depth is then moved by each shift on a grid (:data:`TRIAL_MOVES`),
   with the traced paths kept, and the shift is fitted again. On the same
   cross-sections a move changes the fitted shift one for one (checked on 21
   trees: slope 0.985), but *which* cross-sections are usable changes with
   the move: a misregistered run can merge with something beside it, or miss
   the path. At the depth as loaded that selection hides part of the shift
   (on 21 trees the fit reads 3.19 px, the same cross-sections usable at a
   4-row move read 3.43 px). So three estimates of the correction are
   reported: the move that leaves no fitted shift, interpolated over the
   grid (:func:`zero_move`); the fit on the cross-sections usable at every
   move on the grid; and the move at which depth reaches equally far past
   the bark's upper and lower edges on sloping leaders (mean overshoots).
   Each tree's leaders are also split into the halves near and far from the
   fork, fitted separately: if trees really differ in their shift, the two
   halves agree tree by tree (:func:`split_half`).
2. **Mutual information (both seasons).** The summer images have no traced
   branches, so (1) cannot be used there, and a check along single depth
   edges was tried and failed on winter (thin, shaded bark gives no clear
   colour step at one edge; it read 0.9 px where (1) reads 3.2, and moved
   0.2–1 px for a 3–4 px move). Mutual information, the usual measure for
   registering images from different sensors, needs neither: for each shift
   within :data:`MI_REACH` px, the information the depth (no reading, then
   bins of :data:`DEPTH_STEP` levels) carries about the colour (each channel
   in :data:`COLOUR_LEVELS` levels) is computed over the image and over
   bands of rows (:func:`mi_surface`), and the shift with the most, refined
   between pixels (:func:`mi_peak`), is taken. Surfaces are summed over
   trees for pooled estimates. It is checked against (1) on winter: on 21
   trees the whole-image estimates were 4.0 and 3.2–3.8 px down, but the
   top band of rows read 1.5 px where (1) read 3.5, single trees did not
   agree with (1) at all (r = +0.01), and a tree's surface can rise
   towards the edge of the search (the far ground's range limit is a
   depth edge with no colour edge). So mutual information is reported by
   band and pooled over many trees only.

**Caveat (session 9, not resolved).** On real leaders the colour centre found
by :func:`treeskel.guided_gt.branch_edges` follows the window it is measured
in: moving the window 1 px along the normal moves the mean centre about
0.55–0.7 px (the median does not move: most cross-sections are unaffected,
some jump as clutter enters or leaves the window's ends; on synthetic bark the
pull is at most 0.2). The windows sit on the traced paths, which follow the
depth, so the fitted shift depends on the paths: re-traced with the depth
moved up 4 rows, the same images read about 0.9 px more. On cross-sections
whose two edges are both clear the pull is smaller (0.43), and the move that
leaves no shift is 4.2–4.4 rows up (21 trees), against 3.7 here. The
estimates of (1) are therefore low; see ``docs/results/depth_registration/``.

Usage::

    python -m treeskel.measure_registration data/raw --clicks annotations/winter_clicks.json \\
        --split splits.json --out outputs/measure_registration --plot outputs/depth_registration_rows.png
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
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import disk

from treeskel.guided_gt import (
    NEIGHBOUR_MARGIN,
    GuidedParams,
    arc_length,
    branch_edges,
    clear_of,
    guided_gt,
    load_clicks,
    path_normals,
    straighten,
)
from treeskel.io import parse_filename, shift_image
from treeskel.measure_width import centre_runs
from treeskel.winter_gt import WinterGT, foreground_mask, report_failed

TRIAL_MOVES = tuple((dy, dx) for dy in range(2, -7, -1) for dx in (-1, 0, 1, 2))
"""Moves ``(rows, cols)`` of the depth tried with the traced paths kept (negative rows: up)."""

MIN_HALF_WIDTH = 2.5
"""Cross-sections whose bark is narrower than this (px, half-width) are not used (failed edges)."""

SLOPING = 0.4
"""A cross-section is sloping if its normal's row component is larger than this; only sloping
cross-sections tell the bark's upper edge from its lower edge."""

ROW_BANDS = ((0, 120), (120, 240), (240, 360), (360, 480))
"""Bands of image rows, for the shift across the image and for :func:`mi_surface`."""

COL_BANDS = ((0, 160), (160, 320), (320, 480), (480, 640))
"""Bands of image columns, for the shift across the image."""

MI_REACH = 10
"""Mutual information is computed for shifts up to this many pixels each way."""

PEAK_WINDOW = 3
"""Half-size of the window of shifts to which :func:`mi_peak` fits a quadratic; peaks up to
``MI_REACH − PEAK_WINDOW`` px out can be measured."""

COLOUR_LEVELS = 4
"""Levels per colour channel for mutual information (4³ = 64 colours)."""

DEPTH_STEP = 12
"""Depth levels per bin for mutual information (with one more bin for no reading)."""

N_BOOT = 200
"""Bootstrap resamples of the trees, for the confidence intervals."""


# --- the leaders' cross-sections --------------------------------------------------------------

def silhouette(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The depth foreground as :func:`treeskel.guided_gt.guided_gt` builds it (closed by a disc of 2)."""
    return ndi.binary_closing(foreground_mask(depth, valid), disk(2))


def leader_sections(rgb: np.ndarray, gt: WinterGT, half_width: int | None = None) -> dict[str, np.ndarray]:
    """The leaders' cross-sections that can be compared with depth, and their colour edges.

    Cross-sections that could reach the other leader
    (:func:`treeskel.guided_gt.clear_of`) or whose bark is narrower than
    :data:`MIN_HALF_WIDTH` are left out.

    Args:
        rgb: The winter colour image.
        gt: The traced tree (:func:`treeskel.guided_gt.guided_gt`).
        half_width: Reach of each cross-section either side of the path (px);
            default :attr:`treeskel.guided_gt.GuidedParams.profile_half_width`.

    Returns:
        Arrays, one entry per cross-section: ``side`` (0 left, 1 right),
        ``index`` (the point on that leader's path), ``frac`` (its distance
        along the leader, as a fraction of the leader's length), ``row`` and
        ``col``, the normal ``n_row`` and ``n_col``
        (:func:`treeskel.guided_gt.path_normals`), and the colour edges
        ``first`` and ``last`` (offsets along the normal, px).
    """
    half = half_width or GuidedParams().profile_half_width
    parts = []
    for k, (side, other) in enumerate((("left", gt.right), ("right", gt.left))):
        path = getattr(gt, side)
        if len(path) == 0:
            continue
        first, last = branch_edges(rgb, path, half, GuidedParams().profile_along)
        use = clear_of(path, other, half + NEIGHBOUR_MARGIN) & ((last - first + 1) / 2.0 > MIN_HALF_WIDTH)
        s, n = arc_length(path), path_normals(path)
        i = np.nonzero(use)[0]
        parts.append({"side": np.full(len(i), k), "index": i, "frac": s[i] / max(s[-1], 1.0),
                      "row": path[i, 0], "col": path[i, 1], "n_row": n[i, 0], "n_col": n[i, 1],
                      "first": first[i], "last": last[i]})
    keys = ("side", "index", "frac", "row", "col", "n_row", "n_col", "first", "last")
    return {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in keys}


def depth_runs(fg: np.ndarray, gt: WinterGT, sections: dict[str, np.ndarray],
               half_width: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ends of the depth silhouette's run through each cross-section.

    Args:
        fg: ``(H, W)`` bool depth silhouette (:func:`silhouette`).
        gt: The traced tree whose leaders the sections lie on.
        sections: From :func:`leader_sections`.
        half_width: As in :func:`leader_sections`.

    Returns:
        ``(d0, d1, closed)`` per cross-section, as in
        :func:`treeskel.measure_width.centre_runs`.
    """
    half = half_width or GuidedParams().profile_half_width
    d0 = np.zeros(len(sections["side"]), int)
    d1 = np.zeros(len(sections["side"]), int)
    closed = np.zeros(len(sections["side"]), bool)
    for k, side in enumerate(("left", "right")):
        mine = sections["side"] == k
        if not mine.any():
            continue
        path = getattr(gt, side)
        a, b, c = centre_runs(straighten(fg.astype(float), path, half, order=0) > 0.5)
        i = sections["index"][mine].astype(int)
        d0[mine], d1[mine], closed[mine] = a[i], b[i], c[i]
    return d0, d1, closed


def centre_difference(first, last, d0, d1) -> np.ndarray:
    """The depth run's centre minus the bark's centre, along each cross-section's normal (px)."""
    return (np.asarray(d0) + np.asarray(d1)) / 2.0 - (np.asarray(first) + np.asarray(last)) / 2.0


def edge_overshoots(n_row, first, last, d0, d1) -> tuple[np.ndarray, np.ndarray]:
    """How far the depth run reaches past the bark's upper and lower edges (px, outwards positive).

    The normal points along increasing offsets, so where it points up
    (``n_row < 0``) the ``last`` edge is the upper one.
    """
    past_first = np.asarray(first) - np.asarray(d0)
    past_last = np.asarray(d1) - np.asarray(last)
    up = np.asarray(n_row) < 0
    return np.where(up, past_last, past_first), np.where(up, past_first, past_last)


# --- fits ------------------------------------------------------------------------------------

def huber_fit(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 40) -> np.ndarray:
    """Robust linear least squares ``y ≈ X @ c`` by Huber's M-estimator.

    Iteratively reweighted: residuals within ``k`` robust standard deviations
    (1.4826 × the median absolute deviation of the residuals from their
    median) get full weight, larger ones ``k·σ/|r|``.

    Returns:
        ``c``; zeros if ``y`` is empty.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(y) == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 1)
    w = np.ones(len(y))
    c = np.zeros(X.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        r = y - X @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        w = np.where(np.abs(r) <= k * s, 1.0, k * s / np.maximum(np.abs(r), 1e-12))
    return c


def fit_shift(n_row, n_col, delta) -> tuple[float, float]:
    """The fixed image shift ``(dy, dx)`` of the depth that best explains the centre differences.

    Each cross-section's centre difference (depth minus colour, along its
    normal) is modelled as ``n_row · dy + n_col · dx`` and fitted by
    :func:`huber_fit`. Normals in both directions are needed for both parts:
    vertical branches say nothing about ``dy``.

    Returns:
        ``(dy, dx)`` in px; nan if there are fewer than 3 cross-sections.
    """
    if len(delta) < 3:
        return float("nan"), float("nan")
    dy, dx = huber_fit(np.c_[n_row, n_col], delta)
    return float(dy), float(dx)


def fit_affine(n_row, n_col, row, col, delta, shape: tuple[int, int] = (480, 640)) -> dict[str, float]:
    """A shift that may change linearly across the image: ``s + A · (p − centre)``.

    ``A`` holds scale, rotation and shear. It is reported as the change of
    each part of the shift from the image's centre to its bottom and right
    edges (px), so a pure shift gives zeros.

    Returns:
        ``{"dy", "dx", "dy_to_bottom", "dy_to_right", "dx_to_bottom", "dx_to_right"}``.
    """
    h, w = shape
    y, x = (np.asarray(row) - h / 2) / (h / 2), (np.asarray(col) - w / 2) / (w / 2)
    X = np.c_[n_row, n_col, n_row * y, n_row * x, n_col * y, n_col * x]
    c = huber_fit(X, delta)
    return dict(zip(("dy", "dx", "dy_to_bottom", "dy_to_right", "dx_to_bottom", "dx_to_right"), map(float, c)))


def zero_move(moves: Sequence[tuple[int, int]], shifts: Sequence[tuple[float, float]]) -> dict:
    """The move of the depth that leaves no fitted shift, from fits at several moves.

    The remaining shift is modelled as linear in the move,
    ``r(m) = r0 + J · m``, fitted by least squares over the moves; for a
    measure that follows a move one for one, ``J`` is the identity. The move
    that leaves none is ``−J⁻¹ · r0``.

    Args:
        moves: ``(rows, cols)`` the depth was moved by.
        shifts: The fitted ``(dy, dx)`` remaining at each move (nan ones are skipped).

    Returns:
        ``{"move": [rows, cols], "response": J, "at_no_move": r0}``, or None
        with fewer than 3 usable moves.
    """
    m, r = np.asarray(moves, float), np.asarray(shifts, float)
    ok = np.isfinite(r).all(1)
    if ok.sum() < 3:
        return None
    X = np.c_[np.ones(ok.sum()), m[ok]]
    coef, *_ = np.linalg.lstsq(X, r[ok], rcond=None)           # rows: r0, then J's columns
    r0, jac = coef[0], coef[1:].T
    return {"move": [float(v) for v in -np.linalg.solve(jac, r0)], "response": jac.round(3).tolist(),
            "at_no_move": [float(v) for v in r0]}


def split_half(first: np.ndarray, second: np.ndarray) -> dict:
    """How much of the spread of per-tree estimates is real, from two independent halves per tree.

    If each tree has its own shift ``t`` and each half measures it with
    independent noise, the halves' covariance across trees is the variance of
    ``t`` and their correlation is the share of the spread that is real.

    Args:
        first, second: One estimate per tree from each half (nan skipped).

    Returns:
        ``{"n_trees", "pearson", "spearman", "real_sd", "noise_sd"}``: the
        standard deviation across trees of the real shift (from the
        covariance; 0 if it is negative) and of one half's noise.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return {"n_trees": int(len(a)), "pearson": None, "spearman": None, "real_sd": None, "noise_sd": None}
    cov = float(np.cov(a, b)[0, 1])
    noise = float(np.sqrt(max(np.var(a - b, ddof=1) / 2, 0.0)))
    return {"n_trees": int(len(a)), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(spearmanr(a, b).statistic), "real_sd": float(np.sqrt(max(cov, 0.0))),
            "noise_sd": noise}


# --- mutual information ----------------------------------------------------------------------

def _mutual_information(joint: np.ndarray) -> float:
    """Mutual information (nats) of a joint histogram."""
    total = joint.sum()
    if total == 0:
        return 0.0
    p = joint / total
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def mi_surface(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, reach: int = MI_REACH,
               bands: Sequence[tuple[int, int]] = ROW_BANDS, levels: int = COLOUR_LEVELS,
               step: int = DEPTH_STEP, margin: int | None = None) -> np.ndarray:
    """Mutual information between the colour image and the depth, for each shift of the depth.

    Entry ``[b, reach + dy, reach + dx]`` compares the colour at ``x`` with the
    depth at ``x + (dy, dx)``, so the shift with the most information is where
    the depth sits relative to the colour. Every shift uses the same colour
    pixels: all but a margin. The colour is coded with ``levels`` levels per
    channel; the depth as no reading, or in bins of ``step`` levels.

    Args:
        rgb, depth, valid: One tree's images for one season.
        reach: Largest shift tried each way (px).
        bands: Row ranges measured on their own as well.
        levels, step: Coding of the colour and the depth.
        margin: Colour pixels this close to the edge are not compared (at
            least ``reach``, the default). A depth moved on loading has an
            edge of "no reading" where the move uncovered it; paired with the
            colour there it makes a peak of its own, so the margin should
            then exceed ``reach`` by the move.

    Returns:
        ``(1 + len(bands), 2·reach + 1, 2·reach + 1)``: the whole image, then each band.
    """
    h, w = depth.shape
    m = max(reach, margin or 0)
    q = rgb.astype(int) * levels // 256
    colour = ((q[..., 0] * levels + q[..., 1]) * levels + q[..., 2])[m:h - m, m:w - m]
    n_depth = 255 // step + 2
    coded = np.where(valid, 1 + depth.astype(int) // step, 0)
    # Rows of each band inside the margin, relative to the compared region.
    spans = [(max(lo, m) - m, max(min(hi, h - m) - m, max(lo, m) - m)) for lo, hi in bands]
    # If the bands tile the compared rows, the whole image's histogram is the sum of theirs.
    tiled = bool(spans) and spans[0][0] == 0 and spans[-1][1] == h - 2 * m and all(
        a[1] == b[0] for a, b in zip(spans, spans[1:]))
    size = levels ** 3 * n_depth
    out = np.zeros((1 + len(bands), 2 * reach + 1, 2 * reach + 1))
    for i, dy in enumerate(range(-reach, reach + 1)):
        for j, dx in enumerate(range(-reach, reach + 1)):
            joint = colour * n_depth + coded[m + dy:h - m + dy, m + dx:w - m + dx]
            parts = [np.bincount(joint[a:b].ravel(), minlength=size) for a, b in spans]
            whole = sum(parts) if tiled else np.bincount(joint.ravel(), minlength=size)
            out[0, i, j] = _mutual_information(whole.reshape(-1, n_depth))
            for k, part in enumerate(parts):
                out[1 + k, i, j] = _mutual_information(part.reshape(-1, n_depth))
    return out


def mi_peak(surface: np.ndarray, window: int = PEAK_WINDOW, iters: int = 5) -> tuple[float, float, bool]:
    """The centre of the peak of mutual information, between pixels.

    Depth reaches past the edges of what it sees (by about 2 px), so every
    shift that keeps the colour's objects inside the depth's scores about the
    same: the peak is a plateau, and its highest point is decided by noise.
    On synthetic scenes whose depth overshoots by 2 px, the highest point
    missed the true shift by up to 1.5 px (a parabola through it and its
    neighbours did not help). So a quadratic surface is fitted by least
    squares to the ``(2·window + 1)²`` values around the highest point, and
    the window is moved to the vertex and fitted again until it settles;
    the vertex is the estimate. On the same scenes it was within 0.15 px.

    Args:
        surface: ``(2·reach + 1, 2·reach + 1)``, one entry of :func:`mi_surface` (or a sum).
        window: Half-size of the fitted window.
        iters: Most refits.

    Returns:
        ``(dy, dx, at_edge)``; ``at_edge`` is True if the window could not be
        centred on the peak (it lies within ``window`` of the edge of the
        search, or beyond it) or the fit has no maximum, when the peak may lie
        further out than was searched.
    """
    size = surface.shape[0]
    reach = size // 2
    w = min(window, reach)
    # Start from the top of the lightly smoothed surface: the highest single value wanders on a plateau.
    smooth = ndi.gaussian_filter(np.asarray(surface, float), 1.0, mode="nearest")
    peak = np.array(np.unravel_index(int(np.argmax(smooth)), surface.shape), float)
    offsets = np.mgrid[-w:w + 1, -w:w + 1].reshape(2, -1).T.astype(float)
    design = np.c_[np.ones(len(offsets)), offsets, offsets ** 2, offsets[:, 0] * offsets[:, 1]]
    ok = True
    for _ in range(iters):
        centre = np.clip(np.round(peak).astype(int), w, size - 1 - w)
        z = surface[centre[0] - w:centre[0] + w + 1, centre[1] - w:centre[1] + w + 1].ravel()
        c, *_ = np.linalg.lstsq(design, z, rcond=None)
        hess = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
        if np.linalg.eigvalsh(hess).max() >= 0:                        # no maximum: keep the highest point
            ok = False
            break
        new = centre + np.linalg.solve(hess, -c[1:3])
        if np.allclose(new, peak, atol=1e-3):
            peak = new
            break
        peak = new
    edge = not ok or bool(np.any(np.abs(peak - reach) > reach - w))
    return float(peak[0] - reach), float(peak[1] - reach), edge


# --- one tree --------------------------------------------------------------------------------

def measure_tree(job: tuple) -> dict:
    """All measurements for one tree (top-level so worker processes can run it).

    Args:
        job: ``(tree_record, clicks, registered)``; ``registered`` loads the
            depth moved by :data:`treeskel.io.DEPTH_SHIFT` (to check the
            correction) instead of as stored.

    Returns:
        The tree's cross-sections (:func:`leader_sections`), the depth runs
        through them at every move in :data:`TRIAL_MOVES`, the median depth
        along the leaders, the capture time, and the mutual-information
        surfaces of both seasons.
    """
    from treeskel.io import DEPTH_SHIFT, load_tree

    rec, clicks, registered = job
    winter = load_tree(rec, "winter", register=registered)
    gt = guided_gt(winter["rgb"], winter["depth"], winter["valid"], clicks)
    sections = leader_sections(winter["rgb"], gt)
    runs = []
    for move in TRIAL_MOVES:
        depth = shift_image(winter["depth"], move)
        runs.append(depth_runs(silhouette(depth, depth > 0), gt, sections))
    d0 = np.array([r[0] for r in runs], np.int8)
    d1 = np.array([r[1] for r in runs], np.int8)
    closed = np.array([r[2] for r in runs], bool)
    along = np.concatenate([gt.left, gt.right])
    values = winter["depth"][along[:, 0], along[:, 1]]
    summer = load_tree(rec, "summer", register=registered)
    # The same colour pixels with the depth as stored or moved: a margin beyond the largest move.
    margin = MI_REACH + max(abs(v) for shift in DEPTH_SHIFT.values() for v in shift)
    return {
        "tree": rec.tree_id, "row": rec.row,
        "time": parse_filename(rec.paths[("winter", "depth")]).time,
        "sections": {k: (v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int16))
                     for k, v in sections.items()},
        "d0": d0, "d1": d1, "closed": closed,
        "depth_value": float(np.median(values[values > 0])) if (values > 0).any() else float("nan"),
        "mi": {season: mi_surface(d["rgb"], d["depth"], d["valid"], margin=margin).astype(np.float32)
               for season, d in (("winter", winter), ("summer", summer))},
    }


# --- pooling ---------------------------------------------------------------------------------

def _pool(results: list[dict], move: int, keep=None) -> dict[str, np.ndarray]:
    """All trees' cross-sections usable at one move (index into :data:`TRIAL_MOVES`)."""
    parts = []
    for t, r in enumerate(results):
        s = r["sections"]
        use = r["closed"][move] if keep is None else keep[t]
        parts.append({"tree": np.full(int(use.sum()), t), "n_row": s["n_row"][use], "n_col": s["n_col"][use],
                      "row": s["row"][use], "col": s["col"][use], "frac": s["frac"][use],
                      "delta": centre_difference(s["first"][use], s["last"][use], r["d0"][move][use],
                                                 r["d1"][move][use]),
                      "upper_lower": np.stack(edge_overshoots(s["n_row"][use], s["first"][use], s["last"][use],
                                                              r["d0"][move][use], r["d1"][move][use]), -1)})
    keys = ("tree", "n_row", "n_col", "row", "col", "frac", "delta")
    out = {k: np.concatenate([p[k] for p in parts]).astype(float) for k in keys}
    out["upper_lower"] = np.concatenate([p["upper_lower"] for p in parts]).reshape(-1, 2).astype(float)
    return out


def _fit(p: dict[str, np.ndarray], sel=None) -> tuple[float, float]:
    sel = np.ones(len(p["delta"]), bool) if sel is None else sel
    return fit_shift(p["n_row"][sel], p["n_col"][sel], p["delta"][sel])


def _ci(values: np.ndarray) -> list[float] | None:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else None


def _quartiles(values) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "median": None, "iqr": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "iqr": [float(x) for x in np.percentile(v, [25, 75])]}


def _mi_estimates(surfaces: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Pooled peak (and its bootstrap interval over trees) for the whole image and each band."""
    total = surfaces.sum(0)
    out = []
    for b in range(surfaces.shape[1]):
        dy, dx, edge = mi_peak(total[b])
        boots = np.array([mi_peak(surfaces[rng.integers(0, len(surfaces), len(surfaces)), b].sum(0))[:2]
                          for _ in range(n_boot)]) if n_boot else np.zeros((0, 2))
        out.append({"dy": dy, "dx": dx, "at_edge": edge, "ci_dy": _ci(boots[:, 0]) if n_boot else None,
                    "ci_dx": _ci(boots[:, 1]) if n_boot else None})
    return {"whole": out[0], "bands": out[1:]}


def summarise(results: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Pool :func:`measure_tree` results into the report (see the module docstring)."""
    rng = np.random.default_rng(seed)
    moves = list(TRIAL_MOVES)
    home = moves.index((0, 0))
    trees = [r["tree"] for r in results]
    rows = np.array([r["row"] for r in results])
    report: dict = {"trees": trees, "n_trees": len(results)}

    # Fitted shift at every move (the usable cross-sections at each), and the move that leaves none.
    grid = []
    for m in range(len(moves)):
        p = _pool(results, m)
        dy, dx = _fit(p)
        slope = np.abs(p["n_row"]) > SLOPING
        up, low = p["upper_lower"][slope].T
        grid.append({"move": list(moves[m]), "n": int(len(p["delta"])), "dy": dy, "dx": dx,
                     "upper": float(up.mean()) if len(up) else None, "lower": float(low.mean()) if len(low) else None})
    loaded = _pool(results, home)
    everywhere = [r["closed"].all(0) for r in results]              # usable at every move
    fixed = _pool(results, home, everywhere)
    fdy, fdx = _fit(fixed)
    of_tree = [np.nonzero(fixed["tree"] == t)[0] for t in range(len(results))]
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([of_tree[t] for t in rng.integers(0, len(results), len(results))])
        boots.append(fit_shift(fixed["n_row"][sel], fixed["n_col"][sel], fixed["delta"][sel]))
    boots = np.array(boots).reshape(-1, 2)
    report["grid"] = grid
    report["loaded"] = {"n": int(len(loaded["delta"])), "dy": grid[home]["dy"], "dx": grid[home]["dx"]}
    report["fixed_set"] = {"n": int(len(fixed["delta"])), "dy": fdy, "dx": fdx,
                           "ci_dy": _ci(boots[:, 0]) if n_boot else None, "ci_dx": _ci(boots[:, 1]) if n_boot else None}
    report["zero_move"] = zero_move(moves, [(g["dy"], g["dx"]) for g in grid])
    # Equal overshoot at the upper and lower edges, along the rows moved (at the whole column move
    # nearest the zero move's): where the mean difference changes sign.
    zm = report["zero_move"]
    col_move = int(np.clip(np.round(zm["move"][1]), -1, 2)) if zm else 0
    line = sorted((g["move"][0], g["upper"] - g["lower"]) for g in grid
                  if g["move"][1] == col_move and g["upper"] is not None)
    equal = None
    for (m0, v0), (m1, v1) in zip(line, line[1:]):
        if v0 == 0 or (v0 < 0) != (v1 < 0):
            equal = float(m0 + (m1 - m0) * v0 / (v0 - v1)) if v0 != v1 else float(m0)
            break
    report["edges_equal"] = {"col_move": col_move, "row_move": equal,
                             "upper_minus_lower": [[float(m), float(v)] for m, v in line]}
    # Across the image: bands and an affine term (at the depth as loaded, fixed set).
    report["row_bands"] = [dict(zip(("rows", "n", "dy", "dx"), (list(b), int(s.sum()), *_fit(fixed, s))))
                           for b in ROW_BANDS for s in [(fixed["row"] >= b[0]) & (fixed["row"] < b[1])]]
    report["col_bands"] = [dict(zip(("cols", "n", "dy", "dx"), (list(b), int(s.sum()), *_fit(fixed, s))))
                           for b in COL_BANDS for s in [(fixed["col"] >= b[0]) & (fixed["col"] < b[1])]]
    report["affine"] = fit_affine(fixed["n_row"], fixed["n_col"], fixed["row"], fixed["col"], fixed["delta"])
    # Per tree (fixed set): the shift, the two halves of the leaders, and the depth along them.
    per_tree, near_far = [], []
    for t in range(len(results)):
        own = fixed["tree"] == t
        per_tree.append(_fit(fixed, own) if own.sum() >= 40 else (float("nan"), float("nan")))
        near_far.append([_fit(fixed, own & half)[0] if (own & half).sum() >= 20 else float("nan")
                         for half in (fixed["frac"] < 0.5, fixed["frac"] >= 0.5)])
    per_tree, near_far = np.array(per_tree), np.array(near_far)
    depth_values = np.array([r["depth_value"] for r in results])
    ok = np.isfinite(per_tree[:, 0]) & np.isfinite(depth_values)
    report["per_tree"] = {
        "dy": _quartiles(per_tree[:, 0]), "dx": _quartiles(per_tree[:, 1]),
        "dy_positive": int((per_tree[:, 0] > 0).sum()),
        "split_half": split_half(near_far[:, 0], near_far[:, 1]),
        "depth_trend": {"spearman": float(spearmanr(depth_values[ok], per_tree[ok, 0]).statistic)
                        if ok.sum() > 5 else None,
                        "slope_per_10_levels": float(10 * huber_fit(np.c_[np.ones(ok.sum()), depth_values[ok]],
                                                                    per_tree[ok, 0])[1]) if ok.sum() > 5 else None,
                        "depth_levels": _quartiles(depth_values)}}
    # Mutual information, pooled and per tree, in both seasons.
    surfaces = {s: np.stack([r["mi"][s] for r in results]).astype(float) for s in ("winter", "summer")}
    report["mi"] = {}
    for season, surf in surfaces.items():
        peaks = np.array([mi_peak(s[0]) for s in surf])
        inside = ~peaks[:, 2].astype(bool)
        report["mi"][season] = {**_mi_estimates(surf, rng, n_boot),
                                "per_tree": {"dy": _quartiles(peaks[:, 0]), "dx": _quartiles(peaks[:, 1]),
                                             "at_edge": int((~inside).sum())}}
        # Do trees that the leaders put lower also come out lower by mutual information? The two
        # measures share no error source but the images, so agreement means the trees really differ.
        both = inside & np.isfinite(per_tree[:, 0])
        report["mi"][season]["per_tree"]["vs_leaders"] = split_half(per_tree[both, 0], peaks[both, 0])
    # Per orchard row: the fit (fixed set), the move that leaves no shift, and mutual information.
    by_row = []
    for row in sorted(set(rows.tolist())):
        members = np.nonzero(rows == row)[0]
        sel = np.isin(fixed["tree"], members)
        row_grid = []
        for m in range(len(moves)):
            p = _pool([results[t] for t in members], m)
            row_grid.append(_fit(p))
        zm_row = zero_move(moves, row_grid)
        dy, dx = _fit(fixed, sel)
        # Bootstrap over the row's trees (a row of one tree has no interval).
        picks = [rng.choice(members, len(members)) for _ in range(n_boot if len(members) > 1 else 0)]
        boot = np.array([fit_shift(*(fixed[k][np.concatenate([of_tree[t] for t in pick])]
                                     for k in ("n_row", "n_col", "delta"))) for pick in picks]).reshape(-1, 2)
        entry = {"row": int(row), "n_trees": int(len(members)), "n": int(sel.sum()),
                 "time": min(results[t]["time"] for t in members), "dy": dy, "dx": dx,
                 "ci_dy": _ci(boot[:, 0]) if len(boot) else None, "ci_dx": _ci(boot[:, 1]) if len(boot) else None,
                 "zero_move": zm_row["move"] if zm_row else None}
        for season, surf in surfaces.items():
            dy, dx, edge = mi_peak(surf[members, 0].sum(0))
            boot = np.array([mi_peak(surf[pick, 0].sum(0))[:2] for pick in picks]).reshape(-1, 2)
            entry[f"mi_{season}"] = {"dy": dy, "dx": dx, "at_edge": edge,
                                     "ci_dy": _ci(boot[:, 0]) if len(boot) else None,
                                     "ci_dx": _ci(boot[:, 1]) if len(boot) else None}
        by_row.append(entry)
    report["by_row"] = by_row
    report["per_tree_rows"] = [{"tree": results[t]["tree"], "row": int(rows[t]), "time": results[t]["time"],
                                "n": int((fixed["tree"] == t).sum()), "dy": per_tree[t, 0], "dx": per_tree[t, 1],
                                "dy_near": near_far[t, 0], "dy_far": near_far[t, 1],
                                "depth_value": float(depth_values[t]),
                                **{f"mi_{s}_{k}": v for s in ("winter", "summer")
                                   for k, v in zip(("dy", "dx", "at_edge"), mi_peak(surfaces[s][t, 0]))}}
                               for t in range(len(results))]
    return report


# --- output ----------------------------------------------------------------------------------

def _fmt(v, spec: str = "+.2f") -> str:
    return "  -  " if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def _no_nan(value):
    """The report with nan replaced by None, so the JSON file is standard."""
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_nan(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def print_report(report: dict) -> None:
    """Print the main numbers of :func:`summarise`'s report."""
    ld, fx, zm, eq = report["loaded"], report["fixed_set"], report["zero_move"], report["edges_equal"]
    print(f"\n{report['n_trees']} trees. Shifts are where the depth sits relative to the colour (dy > 0: lower); "
          "the correction moves the depth the other way.")
    print("1. Across the leaders (winter)")
    print(f"   depth as loaded, {ld['n']} cross-sections: dy {ld['dy']:+.2f}, dx {ld['dx']:+.2f} px")
    ci = (f" (95% CI over trees {fx['ci_dy'][0]:+.2f} to {fx['ci_dy'][1]:+.2f}; dx {fx['ci_dx'][0]:+.2f} to "
          f"{fx['ci_dx'][1]:+.2f})") if fx["ci_dy"] else ""
    print(f"   the {fx['n']} cross-sections usable at every move: dy {fx['dy']:+.2f}, dx {fx['dx']:+.2f} px{ci}")
    if zm:
        print(f"   the move that leaves no fitted shift: rows {zm['move'][0]:+.2f}, cols {zm['move'][1]:+.2f} "
              f"(the fit follows a move by {zm['response'][0][0]:.2f} rows per row and {zm['response'][1][1]:.2f} "
              "cols per col)")
    if eq["row_move"] is not None:
        print(f"   depth past the upper and lower edges equal (sloping cross-sections, cols moved {eq['col_move']:+d}):"
              f" rows moved {eq['row_move']:+.2f}")
    print("   remaining shift, and depth past the upper / lower edge on sloping cross-sections (means), by move:")
    for g in report["grid"]:
        if g["move"][1] in (0, 1):
            print(f"     rows {g['move'][0]:+d}, cols {g['move'][1]:+d}: dy {_fmt(g['dy'])}, dx {_fmt(g['dx'])}; "
                  f"upper {_fmt(g['upper'])}, lower {_fmt(g['lower'])}  ({g['n']} cross-sections)")
    print("   across the image (cross-sections usable at every move):")
    for b in report["row_bands"]:
        print(f"     rows {b['rows'][0]:3d}-{b['rows'][1]:3d}: dy {_fmt(b['dy'])}, dx {_fmt(b['dx'])} ({b['n']})")
    for b in report["col_bands"]:
        print(f"     cols {b['cols'][0]:3d}-{b['cols'][1]:3d}: dy {_fmt(b['dy'])}, dx {_fmt(b['dx'])} ({b['n']})")
    a = report["affine"]
    print(f"     affine: dy {a['dy']:+.2f} at the centre, changing by {a['dy_to_bottom']:+.2f} to the bottom edge "
          f"and {a['dy_to_right']:+.2f} to the right edge; dx {a['dx']:+.2f}, {a['dx_to_bottom']:+.2f} and "
          f"{a['dx_to_right']:+.2f}")
    pt = report["per_tree"]
    iqr = pt["dy"]["iqr"] or [None, None]
    print(f"   per tree: dy median {_fmt(pt['dy']['median'])} (IQR {_fmt(iqr[0])} to {_fmt(iqr[1])}), lower than "
          f"the colour on {pt['dy_positive']} of {pt['dy']['n']} trees; dx median {_fmt(pt['dx']['median'])}")
    sh = pt["split_half"]
    if sh["pearson"] is not None:
        print(f"   near and far halves of the leaders, tree by tree ({sh['n_trees']} trees): r {sh['pearson']:+.2f}; "
              f"spread of the real shift across trees {sh['real_sd']:.2f} px, noise of one half {sh['noise_sd']:.2f} px")
    dt = pt["depth_trend"]
    if dt["spearman"] is not None:
        print(f"   against the tree's depth level (median {dt['depth_levels']['median']:.0f}): Spearman "
              f"{dt['spearman']:+.2f}, {dt['slope_per_10_levels']:+.2f} px per 10 levels nearer")
    print("2. Mutual information (whole image; then bands of rows 0-120, 120-240, 240-360, 360-480)")
    for season, m in report["mi"].items():
        w = m["whole"]
        ci = f" (95% CI {w['ci_dy'][0]:+.2f} to {w['ci_dy'][1]:+.2f})" if w["ci_dy"] else ""
        bands = ", ".join(f"{b['dy']:+.2f}" + ("!" if b["at_edge"] else "") for b in m["bands"])
        iqr = m["per_tree"]["dy"]["iqr"] or [None, None]
        print(f"   {season}: dy {w['dy']:+.2f}{ci}, dx {w['dx']:+.2f}; by band dy {bands} (! = at the edge of the "
              f"search); per tree dy median {_fmt(m['per_tree']['dy']['median'])} (IQR {_fmt(iqr[0])} to "
              f"{_fmt(iqr[1])}), {m['per_tree']['at_edge']} at the edge of the search")
        vs = m["per_tree"]["vs_leaders"]
        if vs["pearson"] is not None:
            print(f"     tree by tree against the leaders' dy ({vs['n_trees']} trees): r {vs['pearson']:+.2f}, "
                  f"Spearman {vs['spearman']:+.2f}")
    print("3. By orchard row (capture order): dy, dx across the leaders; the move that leaves none; "
          "mutual information winter / summer")
    for r in report["by_row"]:
        zm_row = r["zero_move"]
        mi = {s: f"{r[f'mi_{s}']['dy']:+.2f}, {r[f'mi_{s}']['dx']:+.2f}" + ("!" if r[f"mi_{s}"]["at_edge"] else "")
              for s in ("winter", "summer")}
        print(f"   row {r['row']:2d} ({r['n_trees']:2d} trees, from {r['time']}): {_fmt(r['dy'])}, {_fmt(r['dx'])}; "
              f"move {_fmt(zm_row[0] if zm_row else None)}, {_fmt(zm_row[1] if zm_row else None)}; "
              f"MI winter {mi['winter']}; summer {mi['summer']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Measure development-row trees; print and save the results."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import DEPTH_SHIFT, index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.measure_registration",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json", help="its test rows are never measured")
    ap.add_argument("--out", default="outputs/measure_registration")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--registered", action="store_true",
                    help=f"load the depth moved by treeskel.io.DEPTH_SHIFT ({DEPTH_SHIFT}), to check the correction")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples of the trees (0: none)")
    ap.add_argument("--plot", help="also save a chart of the shift by orchard row as this image")
    args = ap.parse_args(argv)

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
    jobs = [(records[t], clicks[t], args.registered) for t in ids]
    state = f"moved by {DEPTH_SHIFT}" if args.registered else "as stored"
    print(f"Measuring {len(ids)} development-row trees ({len(held_out)} test-row trees left out), depth {state}...",
          flush=True)
    t0 = time.time()
    step = max(1, min(25, len(jobs) // 10))
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
        for i, r in enumerate(ex.map(measure_tree, jobs, chunksize=2), 1):
            results.append(r)
            if i % step == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                      flush=True)
    print("Pooling (with a bootstrap over trees)...", flush=True)
    report = summarise(results, n_boot=args.boot)
    report["depth"] = state
    report["test_rows_excluded"] = sorted(test_rows)
    print_report(report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    saved = []
    for name, write in (("report.json", lambda f: json.dump(_no_nan({k: v for k, v in report.items()
                                                                     if k != "per_tree_rows"}), f, indent=2)),
                        ("trees.csv", lambda f: _write_trees(f, report["per_tree_rows"]))):
        try:
            with open(out / name, "w", newline="") as f:
                write(f)
            saved.append(str(out / name))
        except OSError as e:
            failed.append(f"{out / name}: {e.strerror or e}")
    if args.plot:
        plot_rows(report, Path(args.plot), failed)
        saved.append(args.plot)
    print(f"Saved {', '.join(saved)}")
    report_failed(failed)
    return 1 if failed else 0


def _write_trees(f, rows: list[dict]) -> None:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def plot_rows(report: dict, path: Path, failed: list[str]) -> None:
    """Dot chart of the shift by orchard row, in capture order, with 95% bootstrap intervals over trees.

    Two panels share the rows: the vertical part of the shift (the story) and
    the sideways part. Three series, dodged within each row: across the
    winter leaders (the reference), and mutual information on winter and on
    summer. A hollow dot is at the edge of the search (the peak may lie
    beyond it). A line marks the pooled estimate across the leaders.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, secondary, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    series = [("leaders", "Across the winter leaders", "#2a78d6"),
              ("mi_winter", "Mutual information, winter", "#eb6834"),
              ("mi_summer", "Mutual information, summer", "#1baf7a")]
    rows = sorted(report["by_row"], key=lambda r: r["time"])
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(surface)
    pooled = report["fixed_set"]
    for ax, part in zip(axes, ("dy", "dx")):
        ax.set_facecolor(surface)
        ax.axhline(0, color=axis, lw=0.8, zorder=1)
        ax.axhline(pooled[part], color=secondary, lw=0.8, zorder=1,
                   label=f"All rows across the leaders ({pooled['dy']:+.1f} px lower, {pooled['dx']:+.1f} px right)")
        for k, (key, name, colour) in enumerate(series):
            labelled = False
            for xi, r in zip(x + (k - 1) * 0.22, rows):
                e = r if key == "leaders" else r[key]
                v, ci = e[part], e[f"ci_{part}"]
                if v is None or not np.isfinite(v):
                    continue
                if ci:
                    ax.plot([xi, xi], ci, color=colour, lw=1.2, solid_capstyle="round", zorder=2)
                hollow = key != "leaders" and e.get("at_edge")
                ax.plot(xi, v, "o", ms=6, color=colour, mfc=surface if hollow else colour, mec=colour if hollow
                        else surface, mew=1.4 if hollow else 1.2, zorder=3, label=None if labelled else name)
                labelled = True
        ax.grid(axis="y", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(axis)
        ax.tick_params(colors=muted, labelsize=8, length=0)
    axes[0].set_ylabel("Depth lower than colour (px)", fontsize=8.5, color=secondary)
    axes[1].set_ylabel("Depth right of colour (px)", fontsize=8.5, color=secondary)
    axes[1].set_xticks(x, [f"{r['row']}\n{r['n_trees']}" for r in rows])
    axes[1].set_xlabel("Orchard row, in winter capture order (number of trees below)", fontsize=8.5, color=secondary)
    fig.suptitle("Where the depth image sits relative to the colour image, by orchard row", x=0.07, ha="left",
                 fontsize=11, color=ink, fontweight="bold")
    axes[0].set_title(f"{report['n_trees']} development-row trees; dots are row estimates, bars 95% bootstrap "
                      "intervals over trees; hollow: at the edge of the search", loc="left", fontsize=7.5,
                      color=secondary)
    axes[0].set_xlim(-0.5, len(rows) - 0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for _, name, _ in series if name in labels] + [0]     # the line was added first
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=7.5, labelcolor=secondary, ncol=2, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    try:
        fig.savefig(path, facecolor=surface)
    except OSError as e:
        failed.append(f"{path}: {e.strerror or e}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
