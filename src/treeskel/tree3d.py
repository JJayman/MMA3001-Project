"""The tree in 3D: its plane, the V in that plane, its lean and its sizes, from the 8-bit depth.

The trees are trained flat (a 2D open V on a trellis), so each tree is
modelled as a plane in front of the camera. The depth map and the camera are
those of :mod:`treeskel.depth3d` (zero 300, focal length 616 px; the scale set so
that the median development tree's fork is 2.0 m away). For each tree:

1. **Depth along the skeleton** (:func:`depth_along`): at every point of the
   trunk and leaders, the median of the depth readings within 3 px (the
   centre of a branch; depth reaches about 2 px past each edge). Points with
   fewer than 5 readings get none.
2. **The tree's plane** (:func:`fit_plane`): fitted to those 3D points by
   total least squares (perpendicular distances), made robust with Huber
   weights, so depth that falls on the background or a post counts little.
   How far the points lie from it says how flat the tree is.
3. **The skeleton on the plane** (:func:`on_plane`): every traced point is
   moved along its pixel's ray to the plane, so the 3D skeleton is complete
   even where the depth has gaps, and depth noise does not make it jagged.
4. **Measures** (:func:`tree_measures`):

   - *V opening in the plane*: the angle between straight lines fitted to the
     same stretch of each leader as the 2D measure (25–264 px from the fork in
     the winter photo, :mod:`treeskel.shape`), beside the 2D V opening of
     that stretch; their difference is the **3D correction** to the 2D
     headline;
   - *the plane's lean towards the camera* (positive: the top nearer) and its
     *turn* (positive: the right side farther), relative to the camera;
   - *lean from the vertical* and the *camera's pitch*, where the ground
     behind the tree gives the vertical (:func:`ground_up`; winter only, as
     the summer canopy hides the ground);
   - *sizes*: leader lengths in view, the trunk's diameter at the fork (the
     winter trunk radius, measured from depth, at the fork's distance), and
     the flatness, in metres.

Angles need only the zero and the focal length; sizes need the scale too
(:mod:`treeskel.depth3d`). Every measure is also computed at the ends of the
zero's range (262 and 366), at session 2's 255, and at ±5% focal length
(:data:`SETTINGS`); sizes scale with the assumed 1.8–2.2 m.

**Summer against winter, tree by tree** (development rows): the same from
the summer depth, along the skeletons of the lined-up hand masks and of the
named U-Net's masks. The V opening in the plane needs no common frame, so it
can be compared directly, and tree by tree against the V in the photo
(:func:`plane_against_photo`); the lean and turn relative to the camera also
carry the change of the camera's pose between the seasons. A few plane fits
fail (planes turned far from the camera, :data:`TURNED`), so the summaries
are medians and rank correlations.

Usage (development rows only)::

    python -m treeskel.tree3d data/raw      # -> docs/results/tree3d/
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

from treeskel.depth3d import (
    DISTANCE_M,
    DISTANCE_RANGE,
    FOCAL_RANGE,
    MIN_VALUE,
    SESSION2_ZERO,
    ZERO_RANGE,
    Camera,
    ground_blocks,
    metres_per_unit,
)
from treeskel.guided_gt import arc_length
from treeskel.shape import SUMMER_SCALE, UNIT_PX, ShapeParams, line_direction, to_xy

PARTS = ("trunk", "left", "right")

SETTINGS = {
    "main": Camera(),
    f"zero {ZERO_RANGE[0]:g}": Camera(zero=ZERO_RANGE[0]),
    f"zero {ZERO_RANGE[1]:g}": Camera(zero=ZERO_RANGE[1]),
    f"zero {SESSION2_ZERO:g} (session 2)": Camera(zero=SESSION2_ZERO),
    f"focal {FOCAL_RANGE[0]:g}": Camera(focal=FOCAL_RANGE[0]),
    f"focal {FOCAL_RANGE[1]:g}": Camera(focal=FOCAL_RANGE[1]),
}
"""The camera settings every measure is computed with: the chosen one, and the sensitivity range."""

SAMPLE_RADIUS = 3
"""Depth is read within this many px of each skeleton point (a branch's centre)."""

MIN_SAMPLES = 5
"""Fewest depth readings within :data:`SAMPLE_RADIUS` for a point to get a depth."""

MIN_GROUND_BLOCKS = 20
"""Fewest ground blocks for the ground to give the vertical."""

SOURCES = ("winter", "hand masks", "U-Net")
"""Where a tree's skeleton comes from: the winter trace (the truth), and two summer ones."""

MEASURES = ("v_opening_3d", "v_opening_2d", "correction", "lean_camera", "turn", "lean_vertical",
            "camera_pitch", "flatness_rms", "flatness_median", "trunk_diameter", "left_length", "right_length",
            "fork_distance", "depth_points", "depth_share")
"""Columns of a tree's measures. Sizes (``flatness_*``, ``trunk_diameter``, ``*_length``,
``fork_distance``) are in units of the scale until :func:`to_metres`."""

SIZES = ("flatness_rms", "flatness_median", "trunk_diameter", "left_length", "right_length", "fork_distance")


# --------------------------------------------------------------------------
# One tree
# --------------------------------------------------------------------------

def depth_along(depth: np.ndarray, valid: np.ndarray, path: np.ndarray, radius: int = SAMPLE_RADIUS,
                min_valid: int = MIN_SAMPLES) -> np.ndarray:
    """The depth value at each point of a path: the median of the readings within ``radius`` px.

    Args:
        depth: ``(480, 640)`` uint8 depth.
        valid: Boolean mask of its readings.
        path: ``(N, 2)`` ``(row, col)`` points.
        radius: Reach of the median (px).
        min_valid: Fewest readings within reach.

    Returns:
        ``(N,)`` floats, NaN where fewer than ``min_valid`` pixels within reach have a reading.

    Example:
        >>> d = np.zeros((20, 20), np.uint8); d[5:15, 5:15] = 100
        >>> depth_along(d, d > 0, np.array([[10, 10], [10, 4], [0, 0]])).tolist()
        [100.0, 100.0, nan]
    """
    p = np.rint(np.asarray(path, float).reshape(-1, 2)).astype(int)
    dy, dx = np.nonzero(np.hypot(*np.mgrid[-radius:radius + 1, -radius:radius + 1]) <= radius)
    dy, dx = dy - radius, dx - radius
    h, w = depth.shape
    rr, cc = p[:, :1] + dy[None], p[:, 1:] + dx[None]
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr, cc = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
    ok = valid[rr, cc] & inside
    v = np.where(ok, depth[rr, cc].astype(float), np.nan)
    out = np.full(len(p), np.nan)
    enough = ok.sum(axis=1) >= min_valid
    if enough.any():
        out[enough] = np.nanmedian(v[enough], axis=1)
    return out


def fit_plane(points: np.ndarray, k: float = 1.345, iters: int = 50) -> dict:
    """A plane fitted to 3D points by total least squares, made robust with Huber weights.

    Each round fits the plane through the weighted centre whose normal is the weighted points' direction
    of least spread (the last right singular vector), then weights each point by ``min(1, k s / |d|)``,
    where ``d`` is its distance from the plane and ``s`` the distances' robust standard deviation
    (1.4826 × the median absolute deviation).

    Args:
        points: ``(N, 3)`` points.
        k: Huber's tuning constant (in robust SDs).
        iters: Most reweighting rounds.

    Returns:
        ``{"normal", "centre", "distances", "weights", "rms", "median_abs", "n"}``: the unit normal points
        towards the camera (at the origin); ``rms`` and ``median_abs`` are of all points' distances.

    Raises:
        ValueError: If there are fewer than 3 points.
    """
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        raise ValueError(f"a plane needs at least 3 points, got {len(p)}")
    w = np.ones(len(p))
    for _ in range(iters):
        c = (w[:, None] * p).sum(0) / w.sum()
        _, _, vt = np.linalg.svd((p - c) * np.sqrt(w)[:, None], full_matrices=False)
        n = vt[-1]
        d = (p - c) @ n
        s = 1.4826 * np.median(np.abs(d - np.median(d)))
        new = np.minimum(1.0, k * max(s, 1e-9) / np.maximum(np.abs(d), 1e-12))
        if np.allclose(new, w, atol=1e-6):
            break
        w = new
    if n @ c > 0:                                     # face the camera
        n = -n
    return {"normal": n, "centre": c, "distances": d, "weights": w, "rms": float(np.sqrt(np.mean(d ** 2))),
            "median_abs": float(np.median(np.abs(d))), "n": int(len(p))}


def on_plane(rays: np.ndarray, plane: dict) -> np.ndarray:
    """Where each ray from the camera meets the plane: ``(N, 3)`` points.

    Args:
        rays: ``(N, 3)`` ray directions from the camera.
        plane: From :func:`fit_plane` (``normal`` and ``centre``).

    Returns:
        The ``(N, 3)`` points.

    Example:
        >>> plane = {"normal": np.array([0.0, 0.0, -1.0]), "centre": np.array([0.0, 0.0, 5.0])}
        >>> on_plane(np.array([[0.5, -0.2, 1.0]]), plane).tolist()
        [[2.5, -1.0, 5.0]]
    """
    n, c = plane["normal"], plane["centre"]
    return rays * ((n @ c) / (rays @ n))[:, None]


def angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions (degrees).

    Args:
        a: A direction (any length).
        b: Another direction.

    Returns:
        The angle in [0, 180].
    """
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def stretch(path: np.ndarray, unit: float, params: ShapeParams = ShapeParams()) -> np.ndarray:
    """Which points of a leader (fork to tip, image px) lie on the stretch its line is fitted to: from
    ``start`` to ``reach`` units along it, as :func:`treeskel.shape.leader_part`.

    Args:
        path: ``(N, 2)`` ``(row, col)`` leader path from the fork.
        unit: The length unit in the path's pixels.
        params: ``start`` and ``reach``.

    Returns:
        ``(N,)`` boolean.
    """
    s = arc_length(np.asarray(path, float))
    return (s >= params.start * unit) & (s <= params.reach * unit)


def ground_up(depth: np.ndarray, valid: np.ndarray, camera: Camera) -> np.ndarray | None:
    """The upward vertical in the camera's frame, from the ground behind the tree, or None.

    The ground blocks of :func:`treeskel.depth3d.ground_blocks` (values of at least
    :data:`treeskel.depth3d.MIN_VALUE`) are put in 3D and a plane fitted (:func:`fit_plane`); its
    normal, which faces the camera above it, points up.

    Args:
        depth: ``(480, 640)`` uint8 depth.
        valid: Boolean mask of its readings.
        camera: The depth map's zero and the focal length.

    Returns:
        The ``(3,)`` unit vector, or None if too few ground blocks are found.
    """
    b = ground_blocks(depth, valid)
    b = b[b[:, 2] >= MIN_VALUE]
    if len(b) < MIN_GROUND_BLOCKS:
        return None
    return fit_plane(camera.points(b[:, [1, 0]], b[:, 2]))["normal"]


def tree_measures(paths: dict, depth: np.ndarray, valid: np.ndarray, camera: Camera, fork_radius: float,
                  zoom: float = 1.0, up: np.ndarray | None = None, values: dict | None = None,
                  params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """One tree's 3D measures (see the module docstring).

    Args:
        paths: ``{"trunk", "left", "right"}`` ``(N, 2)`` ``(row, col)`` photo points: the trunk from the
            root to the fork, each leader from the fork to its tip.
        depth: The photo's ``(480, 640)`` uint8 depth.
        valid: Boolean mask of the depth's readings.
        camera: The depth map and camera.
        fork_radius: The trunk's radius at the fork (photo px).
        zoom: Photo px per winter px (1 for winter; :data:`treeskel.shape.SUMMER_SCALE` for summer), for
            the leader stretch.
        up: The upward vertical (:func:`ground_up`), for the lean from the vertical and the camera's pitch.
        values: Depth values along each path (:func:`depth_along`), to reuse across camera settings.
        params: The leader stretch.

    Returns:
        :data:`MEASURES` (degrees; sizes in units of the scale), NaN where undefined.
    """
    out = dict.fromkeys(MEASURES, float("nan"))
    if values is None:
        values = {k: depth_along(depth, valid, paths[k]) for k in PARTS}
    rc = np.concatenate([np.asarray(paths[k], float).reshape(-1, 2) for k in PARTS])
    v = np.concatenate([values[k] for k in PARTS])
    ok = np.isfinite(v)
    out["depth_points"], out["depth_share"] = float(ok.sum()), float(ok.mean()) if len(v) else 0.0
    if ok.sum() < 20:
        return out
    plane = fit_plane(camera.points(rc[ok], v[ok]))
    n = plane["normal"]
    on = {k: on_plane(camera.rays(paths[k]), plane) for k in PARTS}
    unit = UNIT_PX * zoom
    lines2, lines3 = {}, {}
    for side in ("left", "right"):
        m = stretch(paths[side], unit, params)
        if m.sum() >= params.min_points:
            lines2[side] = line_direction(to_xy(np.asarray(paths[side], float)[m]))
            lines3[side] = line_direction(on[side][m])
    if len(lines3) == 2:
        out["v_opening_2d"] = angle(lines2["left"], lines2["right"])
        out["v_opening_3d"] = angle(lines3["left"], lines3["right"])
        out["correction"] = out["v_opening_3d"] - out["v_opening_2d"]
    out["lean_camera"] = float(np.degrees(np.arctan2(n[1], -n[2])))      # y is down: the top nearer
    out["turn"] = float(np.degrees(np.arctan2(n[0], -n[2])))             # the right side farther
    if up is not None:
        out["lean_vertical"] = float(np.degrees(np.arcsin(np.clip(-n @ up, -1, 1))))
        out["camera_pitch"] = float(np.degrees(np.arcsin(np.clip(-up[2], -1, 1))))   # positive: looking down
    out["flatness_rms"], out["flatness_median"] = plane["rms"], plane["median_abs"]
    fork = on["trunk"][-1] if len(on["trunk"]) else on["left"][0]
    out["fork_distance"] = float(fork[2])
    out["trunk_diameter"] = 2 * fork_radius * float(fork[2]) / camera.focal
    for side in ("left", "right"):
        out[f"{side}_length"] = float(arc_length(on[side])[-1]) if len(on[side]) > 1 else float("nan")
    return out


def mask_paths(mask: np.ndarray) -> dict | None:
    """A 256x256 summer mask's trunk and leaders in photo px, and its trunk radius at the fork, or None.

    Args:
        mask: A 256x256 boolean branch mask.

    Returns:
        ``{"trunk", "left", "right", "fork_radius"}`` in photo px, or None if no fork is found.
    """
    from treeskel.io import IMAGE_SHAPE, MASK_SHAPE
    from treeskel.shape import mask_branches, mask_to_image

    br = mask_branches(np.asarray(mask, bool))
    if br is None:
        return None
    return {**{k: mask_to_image(br[k]) for k in PARTS}, "fork_radius": br["fork_radius"] * IMAGE_SHAPE[0] / MASK_SHAPE[0]}


def measure_tree(job: tuple) -> list[dict]:
    """All of one tree's rows (top-level for worker processes): every source and camera setting.

    Args:
        job: ``(record, winter_trace, hand_mask or None, unet_mask or None)``; the trace as
            :func:`treeskel.shape.load_traces` gives it, the masks 256x256 (the hand mask lined up).

    Returns:
        One row per source and camera setting: the tree's 3D measures, or the ``error`` that stopped them.
    """
    from treeskel.io import load_tree

    rec, trace, hand, unet = job
    rows = []
    winter = load_tree(rec, "winter")
    sources = [("winter", {k: trace[k] for k in PARTS}, trace["fork_radius"], winter, 1.0)]
    summer = None
    for name, mask in (("hand masks", hand), ("U-Net", unet)):
        mp = mask_paths(mask) if mask is not None else None
        if mp is not None:
            summer = summer or load_tree(rec, "summer")
            sources.append((name, {k: mp[k] for k in PARTS}, mp["fork_radius"], summer, SUMMER_SCALE))
    for name, paths, radius, images, zoom in sources:
        values = {k: depth_along(images["depth"], images["valid"], paths[k]) for k in PARTS}
        for setting, camera in SETTINGS.items():
            up = ground_up(images["depth"], images["valid"], camera) if name == "winter" else None
            m = tree_measures(paths, images["depth"], images["valid"], camera, radius, zoom, up, values)
            rows.append({"tree": rec.tree_id, "row": rec.row, "source": name, "setting": setting, **m})
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def scales(rows: Sequence[dict]) -> dict[str, float]:
    """Metres per unit for each setting: the winter forks' median distance put at :data:`DISTANCE_M`.

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).

    Returns:
        ``{setting: metres per unit}``.
    """
    out = {}
    for setting in SETTINGS:
        f = [r["fork_distance"] for r in rows if r["source"] == "winter" and r["setting"] == setting
             and np.isfinite(r["fork_distance"])]
        out[setting] = metres_per_unit(float(np.median(f))) if f else float("nan")
    return out


def to_metres(rows: Sequence[dict], per_unit: dict[str, float]) -> list[dict]:
    """The rows with their sizes in metres (each setting's own scale).

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
        per_unit: From :func:`scales`.

    Returns:
        New rows; the sizes in metres.
    """
    return [{**r, **{k: r[k] * per_unit[r["setting"]] for k in SIZES}} for r in rows]


def _q(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return {"n": 0, "median": None, "q1": None, "q3": None, "p95_abs": None}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "median": float(med), "q1": float(q1), "q3": float(q3),
            "p95_abs": float(np.percentile(np.abs(v), 95))}


TURNED = 20.0
"""A plane turned more than this (degrees) from facing the camera is listed as a likely failed fit (median
turn −4.4°, quartiles −6.3° to −2.4°). Set after seeing the 449-tree run, so it only describes: no
summary leaves those trees out; the summaries are medians and rank correlations, which a few failed fits
barely move."""


def paired(rows: Sequence[dict], source: str, measure: str, setting: str = "main") -> dict:
    """A summer source against winter, tree by tree, on one measure: error and agreement.

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
        source: The summer source (``"hand masks"`` or ``"U-Net"``).
        measure: The measure compared.
        setting: The camera setting (:data:`SETTINGS`).

    Returns:
        ``{"n", "mae", "median_abs", "bias", "r", "spearman"}``: the mean and median absolute difference,
        the mean difference (summer − winter), and the Pearson and Spearman (rank) correlations. A few
        failed plane fits dominate the mean and Pearson's r, so the median and the rank correlation are
        the ones to read.
    """
    from scipy import stats

    w = {r["tree"]: r[measure] for r in rows if r["source"] == "winter" and r["setting"] == setting}
    s = {r["tree"]: r[measure] for r in rows if r["source"] == source and r["setting"] == setting}
    t = [k for k in s if k in w and np.isfinite(s[k]) and np.isfinite(w[k])]
    if len(t) < 3:
        return {"n": len(t)}
    a, b = np.array([w[k] for k in t]), np.array([s[k] for k in t])
    e = b - a
    varied = np.std(b) > 0 and np.std(a) > 0
    return {"n": len(t), "mae": float(np.mean(np.abs(e))), "median_abs": float(np.median(np.abs(e))),
            "bias": float(np.mean(e)), "r": float(stats.pearsonr(a, b)[0]) if varied else None,
            "spearman": float(stats.spearmanr(a, b)[0]) if varied else None}


def plane_against_photo(rows: Sequence[dict], source: str, setting: str = "main") -> dict:
    """Does summer agree with winter better in the tree's plane than in the photo? Tree by tree.

    For each tree with both: ``|summer − winter|`` of the V opening in the plane against the same in the
    photo. The in-plane V does not depend on where the camera stood, which changed between the seasons.

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
        source: The summer source (``"hand masks"`` or ``"U-Net"``).
        setting: The camera setting (:data:`SETTINGS`).

    Returns:
        ``{"n", "plane_closer", "wilcoxon_p", "median_plane", "median_photo"}``.
    """
    from scipy import stats

    by = {(r["tree"], r["source"]): r for r in rows if r["setting"] == setting}
    e3, e2 = [], []
    for (tree, src), r in by.items():
        w = by.get((tree, "winter"))
        if src != source or w is None:
            continue
        vals = (r["v_opening_3d"], w["v_opening_3d"], r["v_opening_2d"], w["v_opening_2d"])
        if all(np.isfinite(v) for v in vals):
            e3.append(abs(vals[0] - vals[1]))
            e2.append(abs(vals[2] - vals[3]))
    if len(e3) < 3:
        return {"n": len(e3)}
    e3, e2 = np.array(e3), np.array(e2)
    d = e2 - e3
    return {"n": int(len(e3)), "plane_closer": float(np.mean(d > 0)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "median_plane": float(np.median(e3)), "median_photo": float(np.median(e2))}


def turned(rows: Sequence[dict], source: str, setting: str = "main") -> list[str]:
    """The trees whose plane is turned more than :data:`TURNED` from the camera (likely failed fits).

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
        source: The source (``"winter"`` or a summer source).
        setting: The camera setting (:data:`SETTINGS`).

    Returns:
        The tree ids, sorted.
    """
    return sorted(r["tree"] for r in rows if r["source"] == source and r["setting"] == setting
                  and np.isfinite(r["turn"]) and abs(r["turn"]) > TURNED)


def summarise(rows: Sequence[dict]) -> dict:
    """The report: winter's 3D measures under every setting (sizes in metres), and summer against winter.

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).

    Returns:
        The report (``report.json``).
    """
    per_unit = scales(rows)
    rows_m = to_metres(rows, per_unit)
    report = {"settings": {k: {"zero": c.zero, "focal": c.focal, "metres_per_unit": per_unit[k]}
                           for k, c in SETTINGS.items()},
              "distance_m": DISTANCE_M, "distance_range_m": list(DISTANCE_RANGE),
              "n_trees": len({r["tree"] for r in rows}), "winter": {}, "summer": {}}
    for setting in SETTINGS:
        sub = [r for r in rows_m if r["source"] == "winter" and r["setting"] == setting]
        report["winter"][setting] = {k: _q([r[k] for r in sub]) for k in MEASURES}
    for source in SOURCES[1:]:
        sub = [r for r in rows_m if r["source"] == source and r["setting"] == "main"]
        report["summer"][source] = {
            "n_trees": len(sub),
            "measures": {k: _q([r[k] for r in sub]) for k in MEASURES},
            "against_winter": {k: paired(rows_m, source, k) for k in
                               ("v_opening_3d", "v_opening_2d", "correction", "lean_camera", "turn")},
            "plane_against_photo": plane_against_photo(rows_m, source),
            "turned": turned(rows_m, source),
        }
    report["winter_turned"] = turned(rows_m, "winter")
    return report


def _f(x, spec=".1f") -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


LABELS = {"v_opening_3d": ("V opening in the plane", "deg", 1), "v_opening_2d": ("V opening in the photo", "deg", 1),
          "correction": ("3D correction (plane - photo)", "deg", 1),
          "lean_camera": ("lean towards the camera", "deg", 1), "turn": ("turn (right side farther)", "deg", 1),
          "lean_vertical": ("lean from the vertical (ground)", "deg", 1),
          "camera_pitch": ("camera pitch down (ground)", "deg", 1),
          "flatness_rms": ("distance from the plane, RMS", "cm", 100), "flatness_median": ("  median", "cm", 100),
          "trunk_diameter": ("trunk diameter at the fork", "cm", 100),
          "left_length": ("left leader in view", "m", 1), "right_length": ("right leader in view", "m", 1),
          "fork_distance": ("fork distance", "m", 1), "depth_share": ("skeleton points with depth", "%", 100)}


def report_lines(report: dict) -> list[str]:
    """The report as text.

    Args:
        report: From :func:`summarise`.

    Returns:
        The lines of text.
    """
    out = [f"3D measures of {report['n_trees']} development trees. Winter traces (the truth), median "
           "[quartiles]; sizes with the median fork at "
           f"{report['distance_m']:g} m (they scale with it: {report['distance_range_m'][0]:g}-"
           f"{report['distance_range_m'][1]:g} m is -10%/+10%):"]
    main = report["winter"]["main"]
    for k, (label, unit, mult) in LABELS.items():
        q = main[k]
        if q["n"]:
            out.append(f"  {label:34s} {_f(q['median'] * mult):>7s} [{_f(q['q1'] * mult)}, {_f(q['q3'] * mult)}] "
                       f"{unit}  (n {q['n']})")
    corr = main["correction"]
    out.append(f"  |3D correction|, 95th percentile: {_f(corr['p95_abs'])} deg")
    out.append(f"  planes turned more than {TURNED:g} deg from the camera (likely failed fits; not left out): "
               f"{len(report['winter_turned'])} ({', '.join(report['winter_turned']) or 'none'})")
    out += ["", "Sensitivity (winter medians under each setting):"]
    keys = ("v_opening_3d", "correction", "lean_camera", "lean_vertical", "flatness_rms", "trunk_diameter",
            "left_length")
    out.append(f"  {'setting':22s} {'m/unit':>8s} " + " ".join(f"{LABELS[k][0][:14]:>15s}" for k in keys))
    for setting, s in report["winter"].items():
        cells = []
        for k in keys:
            med = s[k]["median"]
            cells.append(_f(None if med is None else med * LABELS[k][2], ".2f" if k == "correction" else ".1f"))
        out.append(f"  {setting:22s} {report['settings'][setting]['metres_per_unit']:8.4f} "
                   + " ".join(f"{c:>15s}" for c in cells))
    out += ["", "Summer against winter, tree by tree (main setting; mean and median |summer - winter|, mean of "
                "summer - winter, Pearson r, Spearman rank correlation; a few failed fits dominate the mean and r):"]
    for source, s in report["summer"].items():
        out.append(f"  {source} ({s['n_trees']} trees; skeleton points with depth "
                   f"{_f(s['measures']['depth_share']['median'] * 100 if s['measures']['depth_share']['median'] is not None else None, '.0f')}%, "
                   f"distance from the plane {_f(s['measures']['flatness_rms']['median'] * 100 if s['measures']['flatness_rms']['median'] is not None else None)} cm RMS):")
        for k, a in s["against_winter"].items():
            if a.get("n", 0) >= 3:
                out.append(f"    {LABELS[k][0]:34s} n {a['n']:3d}  MAE {a['mae']:5.1f}  median {a['median_abs']:5.1f}  "
                           f"bias {a['bias']:+5.1f}  r {_f(a['r'], '.2f')}  rank {_f(a['spearman'], '.2f')}")
        pp = s["plane_against_photo"]
        if pp.get("n", 0) >= 3:
            out.append(f"    V opening, plane against photo: |summer - winter| median {pp['median_plane']:.2f} against "
                       f"{pp['median_photo']:.2f} deg; the plane closer on {pp['plane_closer']:.0%} of {pp['n']} trees "
                       f"(Wilcoxon p {pp['wilcoxon_p']:.1e})")
        out.append(f"    planes turned more than {TURNED:g} deg: {len(s['turned'])}")
    return out


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------

SUMMER_STYLE = {"hand masks": ("#2a78d6", "o"), "U-Net": ("#e34948", "h")}
"""The summer sources' colours and markers, as in the project's other figures."""


def plot_figure(rows: Sequence[dict], path: Path, failed: list[str]) -> None:
    """Three panels: winter's V opening in the plane against the photo's; the 3D correction under every
    camera setting (median and quartiles); and summer's correction against winter's, tree by tree.

    Args:
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
        path: The figure file to write.
        failed: Files that could not be written are added to it (a viewer can lock a file on Windows).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    from treeskel.shape_model import AXIS, INK, MUTED, SECONDARY, SURFACE, _save, _style

    win = {r["tree"]: r for r in rows if r["source"] == "winter" and r["setting"] == "main"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1.05, 1]})
    fig.patch.set_facecolor(SURFACE)

    # A: the V in the plane against the V in the photo.
    ax = axes[0]
    v2 = np.array([r["v_opening_2d"] for r in win.values()])
    v3 = np.array([r["v_opening_3d"] for r in win.values()])
    ok = np.isfinite(v2) & np.isfinite(v3)
    lo, hi = 15.0, 100.0
    shown = ok & (v3 >= lo) & (v3 <= hi) & (v2 >= lo) & (v2 <= hi)
    med = float(np.median(v3[ok] - v2[ok]))
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1.2, zorder=1)
    ax.plot([lo, hi], [lo + med, hi + med], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(v2[shown], v3[shown], s=10, color=SECONDARY, alpha=0.55, linewidths=0, zorder=2)
    ax.text(hi - 1, lo + 3, f"dashed: {med:+.1f}° (median)\n{int((ok & ~shown).sum())} trees off the axes",
            ha="right", va="bottom", fontsize=7.5, color=SECONDARY)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("V opening in the photo (°)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("V opening in the tree's plane (°)", fontsize=8, color=SECONDARY)
    ax.set_title("Winter: the V in 3D is a little narrower", fontsize=9, color=INK, loc="left")

    # B: the correction under every camera setting.
    ax = axes[1]
    names = list(SETTINGS)
    for i, name in enumerate(names):
        c = np.array([r["correction"] for r in rows if r["source"] == "winter" and r["setting"] == name])
        c = c[np.isfinite(c)]
        q1, m, q3 = np.percentile(c, [25, 50, 75])
        y = len(names) - 1 - i
        ax.plot([q1, q3], [y, y], color=SECONDARY, lw=2, solid_capstyle="round")
        ax.scatter([m], [y], s=46, color=INK if name == "main" else SURFACE, edgecolors=INK, linewidths=1.2, zorder=3)
        ax.text(q3 + 0.15, y, f"{m:+.1f}", va="center", fontsize=7.5, color=SECONDARY)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("main", "chosen: zero 300, focal 616") for n in names[::-1]], fontsize=7.5)
    ax.set_xlim(-4.5, 0.8)
    ax.set_xlabel("3D correction: plane − photo (°), median and quartiles", fontsize=8, color=SECONDARY)
    ax.set_title("Same sign for every calibration", fontsize=9, color=INK, loc="left")

    # C: summer's correction against winter's, tree by tree.
    ax = axes[2]
    lim = (-7.0, 3.0)
    ax.plot(lim, lim, color=AXIS, lw=1.2, zorder=1)
    for source, (colour, marker) in SUMMER_STYLE.items():
        s = {r["tree"]: r["correction"] for r in rows if r["source"] == source and r["setting"] == "main"}
        t = [k for k in s if k in win and np.isfinite(s[k]) and np.isfinite(win[k]["correction"])]
        x, y = np.array([win[k]["correction"] for k in t]), np.array([s[k] for k in t])
        r = stats.spearmanr(x, y)[0] if len(t) > 2 else float("nan")
        inside = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[0]) & (y <= lim[1])
        ax.scatter(x[inside], y[inside], s=12, color=colour, marker=marker, alpha=0.5, linewidths=0,
                   label=f"{source} (rank correlation {r:.2f})", zorder=2)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("winter correction (°)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("summer correction (°)", fontsize=8, color=SECONDARY)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=SECONDARY)
    ax.set_title("Summer repeats it in part, tree by tree", fontsize=9, color=INK, loc="left")

    for ax in axes:
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
    fig.suptitle(f"The trees in 3D ({len(win)} development trees; winter traces, summer masks)",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, failed)
    plt.close(fig)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def save_rows(path: Path, rows: Sequence[dict]) -> None:
    """Write the per-tree rows (sizes in units of the scale), floats to 4 decimals.

    Args:
        path: The CSV file.
        rows: One row per tree, source and camera setting (:func:`measure_tree`).
    """
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def load_rows(path: Path) -> list[dict]:
    """Read :func:`save_rows`'s file back.

    Args:
        path: A file from :func:`save_rows`.

    Returns:
        The rows.
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ("tree", "source", "setting") else int(v) if k == "row"
                             else float(v) if v not in ("", "nan") else float("nan")) for k, v in r.items()})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the development trees in 3D; print and save the results.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if no development tree with a winter trace was found or an output
        file could not be written.
    """
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset
    from treeskel.segment import hand_mask, load_masks
    from treeskel.shape import load_traces
    from treeskel.summer_gt import load_overlays

    ap = argparse.ArgumentParser(prog="python -m treeskel.tree3d", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s)")
    ap.add_argument("--traces", default="data/shape/winter_traces.npz", help="the winter traces")
    ap.add_argument("--mask-overlay", default="annotations/mask_overlay.json", help="the hand masks' line-up")
    ap.add_argument("--unet", default="docs/results/segment_lined_up_joined/predicted_masks.npz",
                    help="the named U-Net's development masks")
    ap.add_argument("--split", default="splits.json", help="its test rows are never loaded")
    ap.add_argument("--out", default="docs/results/tree3d")
    ap.add_argument("--trees", nargs="+", help="only these trees")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--from-saved", metavar="DIR", help="re-summarise a saved run's trees.csv (no images)")
    ap.add_argument("--figure", default="docs/figures/tree3d.png", help="the figure ('' for none)")
    args = ap.parse_args(argv)

    if args.from_saved:                                  # only the summary again, from the saved rows
        rows = load_rows(Path(args.from_saved) / "trees.csv")
        out = Path(args.out if args.out != "docs/results/tree3d" else args.from_saved)
    else:
        if not args.data_dirs:
            ap.error("give the dataset folder(s), or --from-saved")
        # Each development tree's three skeletons: the winter trace, the lined-up hand mask, the U-Net's mask.
        test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])
        traces = load_traces(args.traces)
        overlays = load_overlays(args.mask_overlay)
        unet = load_masks(Path(args.unet)) if Path(args.unet).exists() else {}
        records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees if t.row not in test_rows}
        ids = sorted(t for t in records if t in traces and (not args.trees or t in args.trees))
        if not ids:
            print("No development-row trees with winter traces found; nothing to measure.")
            return 1
        # A run on chosen trees goes to outputs/, so it never overwrites the committed full run.
        out = Path(args.out if not args.trees or args.out != "docs/results/tree3d" else "outputs/tree3d_partial")
        jobs = [(records[t], traces[t], hand_mask(records[t], overlays), unet.get(t)) for t in ids]
        print(f"Measuring {len(ids)} development-row trees in 3D (winter, hand masks, U-Net; "
              f"{len(SETTINGS)} camera settings)...", flush=True)
        t0, step, rows = time.time(), max(1, min(25, len(jobs) // 10)), []
        # One tree per worker process; each returns a row per skeleton and camera setting.
        with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
            for i, r in enumerate(ex.map(measure_tree, jobs, chunksize=2), 1):
                rows += r
                if i % step == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                          flush=True)
        out.mkdir(parents=True, exist_ok=True)
        save_rows(out / "trees.csv", rows)
        rows = load_rows(out / "trees.csv")              # the report from the saved rows, so it reproduces
    report = summarise(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = report_lines(report)
    print("\n".join(lines))
    try:
        (out / "output.txt").write_text("\n".join(lines) + "\n")
    except OSError as e:
        print(f"Could not write {out / 'output.txt'}: {e.strerror or e}")
        return 1
    failed: list[str] = []
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot_figure(rows, Path(args.figure), failed)
    saved = "report.json and output.txt" if args.from_saved else f"{out / 'trees.csv'}, report.json and output.txt"
    print(f"\nSaved {saved} in {out}" + (f", and {args.figure}" if args.figure else ""))
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
