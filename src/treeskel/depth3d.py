"""The 3D estimate from the 8-bit depth: how the depth values map to distance, and the camera.

The depth images hold 8-bit values (brighter = nearer, 0 = no reading), not
distances, and no calibration came with the data. Session 2 guessed a linear
map, ``Z = 4 m · (1 − v / 255)``, from the canopy's values and the documented
1.8–2.2 m from camera to canopy (Chen et al., 2020).

**What the 3D measures need.** Write the map as ``Z = c · (v0 − v)``: ``v0`` is
the value at zero distance (its *zero*, 255 in session 2's guess) and ``c``
the metres per unit (its *scale*). A pixel ``(u, r)`` at distance ``Z`` lies at
``X = (u − cx) Z / f`` and ``Y = (r − cy) Z / f``. Then:

- **angles** (the V opening in the tree's plane, the plane's lean) compare
  depth differences with sideways distances, ``dZ / dX = −dv · f /
  ((v0 − v) du)``: the scale ``c`` cancels, and only the zero ``v0`` and the
  focal length ``f`` count;
- **sizes** (leader lengths, the trunk's diameter) need the scale ``c`` too.

**The zero, from the ground** (:func:`ground_blocks`, :func:`fit_zero`). Seen
from a camera at any height, pitch and roll, a flat ground satisfies
``1 / Z = α + β u + γ r`` (its inverse distance is affine in the pixel's
coordinates). So for the right zero, ``1 / (v0 − v)`` is affine in ``(u, r)``
on the ground, whatever the scale; a map linear in disparity instead (``v``
affine in ``1 / Z``) would make ``v`` itself affine, the limit of a very
large ``v0``. The ground behind each tree (below the rows where most pixels
have no reading) is summarised in blocks of 16 rows by 32 columns (medians, so
grass, leaves and the drip line count little); trees and posts are removed as
blocks far from a smooth surface through the rest, which assumes no zero. For
each zero on a grid the affine model is fitted by least squares and
its root-mean-square error is measured in depth units. The zero with the
smallest error is each tree's estimate; the pooled error over all trees gives
one estimate, and its spread over the orchard rows shows how steady it is.
Real ground is not quite flat (the trees stand on a raised strip), which can
bias this; the residual error says how well a plane fits at all.

**The focal length** (:data:`FOCAL_PX`): the D435's colour camera, to which
the depth is aligned, has a 69.4° x 42.5° field of view at 1920x1080; its
640x480 images are the centre 1440x1080 scaled by 4/9, so ``f ≈ 616 px``
(fields of view 55° x 43°), centred at (320, 240). Its range of sensitivity
is :data:`FOCAL_RANGE`.

**The camera** (:class:`Camera`): the zero chosen from the ground
(:data:`ZERO`), the focal length, and pixel-to-3D. The tree's 3D shape is in
:mod:`treeskel.tree3d`.

Usage (development rows only)::

    python -m treeskel.depth3d data/raw --calibrate      # -> docs/results/depth3d/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FOCAL_PX = 616.0
"""Nominal focal length (px) of the D435's colour camera at 640x480 (see the module docstring)."""

FOCAL_RANGE = (585.0, 647.0)
"""±5% around :data:`FOCAL_PX`, for the sensitivity of the 3D measures."""

CENTRE = (240.0, 320.0)
"""Principal point ``(row, col)``: the image's centre (nominal)."""

SESSION2_ZERO = 255.0
"""Session 2's guess at the zero: ``Z = 4 m · (1 − v / 255)``."""

ZERO = 300.0
"""The zero used for the 3D estimate: the median of the per-tree estimates from the ground on the 449
development trees (session 13, ``docs/results/depth3d/``; quartiles 262–366, pooled 325)."""

ZERO_RANGE = (262.0, 366.0)
"""The quartiles of the per-tree estimates, for the sensitivity of the 3D measures."""

DISTANCE_M = 2.0
"""Metres from the camera to the median development tree's fork, which sets the scale: the middle of the
1.8–2.2 m from camera to canopy documented for this orchard and camera (Chen et al., 2020). Nothing in the
photos fixes the scale; it affects sizes only, never angles."""

DISTANCE_RANGE = (1.8, 2.2)
"""The documented range, for the sensitivity of the sizes."""

METRES_PER_UNIT = 0.01197
"""The scale with :data:`ZERO` and :data:`DISTANCE_M`: the 449 development trees' winter forks put at a
median 2.0 m (``docs/results/tree3d/report.json``, setting ``main``: 0.011972). For a single photo, where
no median can be taken (:mod:`treeskel.model3d`)."""


@dataclass(frozen=True)
class Camera:
    """The camera and depth map: pixel ``(row, col)`` and depth value ``v`` to a 3D point.

    Distances are in *units of the scale* (``Z = zero − v``); multiply by the metres per unit
    (:func:`metres_per_unit`) for metres. Angles do not depend on the scale.

    Attributes:
        zero: The depth value at zero distance.
        focal: Focal length (px).
        centre: Principal point ``(row, col)``.
    """

    zero: float = ZERO
    focal: float = FOCAL_PX
    centre: tuple[float, float] = CENTRE

    def distance(self, values: np.ndarray) -> np.ndarray:
        """Distance along the optical axis (units) of depth values (NaN stays NaN)."""
        return self.zero - np.asarray(values, float)

    def rays(self, rc: np.ndarray) -> np.ndarray:
        """``(N, 3)`` ray directions ``(x, y, 1)`` through ``(N, 2)`` ``(row, col)`` pixels (``y`` down).

        Example:
            >>> Camera(focal=100.0).rays(np.array([[240.0, 420.0]])).tolist()
            [[1.0, 0.0, 1.0]]
        """
        p = np.asarray(rc, float).reshape(-1, 2)
        return np.stack([(p[:, 1] - self.centre[1]) / self.focal, (p[:, 0] - self.centre[0]) / self.focal,
                         np.ones(len(p))], 1)

    def points(self, rc: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``(N, 3)`` points ``(X, Y, Z)`` (units; ``Y`` down) of pixels with depth values."""
        return self.rays(rc) * self.distance(values)[:, None]


def metres_per_unit(fork_distance_units: float, distance_m: float = DISTANCE_M) -> float:
    """The scale: metres per unit that put a fork ``fork_distance_units`` away at ``distance_m``.

    Example:
        >>> metres_per_unit(180.0)
        0.011111111111111112
    """
    return distance_m / fork_distance_units

ZEROS = tuple(float(v) for v in np.concatenate([np.arange(190, 400, 5), np.arange(400, 1000, 25),
                                                 np.arange(1000, 5001, 250)]))
"""Zeros tried by :func:`fit_zero`; the largest is near a map linear in disparity."""

BLOCK = (16, 32)
"""Rows and columns of a ground block (:func:`ground_blocks`)."""

MIN_VALUE = 10
"""Ground blocks with a median below this are left out: near the end of the range, where values may
be clipped. (A setting fixed before the run; :func:`calibrate` also reports the zeros without it.)"""


def ground_blocks(depth: np.ndarray, valid: np.ndarray, block: tuple[int, int] = BLOCK, min_fill: float = 0.9,
                  cut: float = 3.5, floor: float = 2.0, iters: int = 10) -> np.ndarray:
    """Median depth of the ground in blocks: ``(n, 3)`` rows of ``(col, row, value)`` at block centres.

    The ground is in the rows where most pixels have a reading (as in
    :func:`treeskel.winter_gt.foreground_mask`). A block counts if at least
    ``min_fill`` of it has a reading, so blocks cut by the end of the range,
    whose median leans towards the nearer pixels, are left out. Objects
    standing on the ground (trees, posts) are removed without assuming any
    zero: a smooth surface (cubic in the row, with the column and their
    product for the camera's roll) is fitted to the blocks' values, and blocks
    more than ``cut`` robust standard deviations (at least ``floor`` units)
    from it are dropped, repeating until none are. (A rule by rows, "nearer
    than the row's median", would also cut away ground: when the camera rolls,
    one image row spans several distances.)
    """
    ground_rows = valid.mean(axis=1) >= 0.5
    h, w = depth.shape
    out = []
    for r0 in range(0, h - block[0] + 1, block[0]):
        if not ground_rows[r0:r0 + block[0]].all():
            continue
        for c0 in range(0, w - block[1] + 1, block[1]):
            ok = valid[r0:r0 + block[0], c0:c0 + block[1]]
            if ok.mean() >= min_fill:
                v = depth[r0:r0 + block[0], c0:c0 + block[1]][ok]
                out.append((c0 + block[1] / 2 - 0.5, r0 + block[0] / 2 - 0.5, float(np.median(v))))
    b = np.asarray(out, float).reshape(-1, 3)
    keep = np.ones(len(b), bool)
    if len(b) < 12:
        return b
    u, r = (b[:, 0] - w / 2) / w, (b[:, 1] - h / 2) / h
    a = np.stack([np.ones_like(u), u, r, r ** 2, r ** 3, u * r], 1)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(a[keep], b[keep, 2], rcond=None)
        res = b[:, 2] - a @ coef
        sd = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) <= max(cut * sd, floor)
        if np.array_equal(new, keep) or new.sum() < 12:
            break
        keep = new
    return b[keep]


def zero_errors(blocks: np.ndarray, zeros: Sequence[float] = ZEROS) -> np.ndarray:
    """RMS error (depth units) of the flat-ground model for each zero: ``1 / (v0 − v)`` affine in ``(u, r)``.

    For each zero, ``α, β, γ`` are fitted by least squares to ``1 / (v0 − v)``; the fitted inverse is
    turned back into values (``v0 − 1 / fit``) and compared with the blocks' values. A zero at or below
    a block's value, or a fit that is not positive everywhere, gives ``inf``.

    Example:
        >>> u, r = np.meshgrid(np.arange(16.0, 640, 32), np.arange(250.0, 480, 16))
        >>> v = 300 - 1 / (0.002 + 0.00001 * u + 0.00004 * (r - 250))      # zero 300, tilted ground
        >>> b = np.stack([u.ravel(), r.ravel(), v.ravel()], 1)
        >>> e = zero_errors(b, (250.0, 300.0, 400.0))
        >>> bool(e[1] < 1e-9 and e[0] > e[1] and e[2] > e[1])
        True
    """
    u, r, v = blocks[:, 0], blocks[:, 1], blocks[:, 2]
    a = np.stack([np.ones_like(u), u, r], 1)
    out = np.full(len(zeros), np.inf)
    for i, v0 in enumerate(zeros):
        if len(v) < 4 or v0 <= v.max():
            continue
        y = 1.0 / (v0 - v)
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        fit = a @ coef
        if np.all(fit > 0):
            out[i] = float(np.sqrt(np.mean((v0 - 1.0 / fit - v) ** 2)))
    return out


def affine_error(blocks: np.ndarray) -> float:
    """RMS error (depth units) of ``v`` itself affine in ``(u, r)``: a map linear in disparity."""
    if len(blocks) < 4:
        return float("inf")
    a = np.stack([np.ones(len(blocks)), blocks[:, 0], blocks[:, 1]], 1)
    coef, *_ = np.linalg.lstsq(a, blocks[:, 2], rcond=None)
    return float(np.sqrt(np.mean((a @ coef - blocks[:, 2]) ** 2)))


def fit_zero(blocks: np.ndarray, zeros: Sequence[float] = ZEROS) -> dict:
    """One tree's zero: the grid value with the smallest :func:`zero_errors`, refined by a parabola through
    it and its neighbours.

    Returns:
        ``{"zero", "rms", "rms_session2", "rms_disparity", "n_blocks", "at_edge"}``; ``zero`` is NaN
        if no zero fits, and ``at_edge`` is True if the best lies at either end of the grid.
    """
    e = zero_errors(blocks, zeros)
    out = {"n_blocks": int(len(blocks)), "rms_disparity": affine_error(blocks),
           "rms_session2": float(zero_errors(blocks, (SESSION2_ZERO,))[0])}
    if not np.isfinite(e).any():
        return {**out, "zero": float("nan"), "rms": float("nan"), "at_edge": False}
    i = int(np.nanargmin(np.where(np.isfinite(e), e, np.nan)))
    z = zeros[i]
    if 0 < i < len(zeros) - 1 and np.isfinite(e[i - 1]) and np.isfinite(e[i + 1]):
        x = np.array(zeros[i - 1:i + 2], float)
        p = np.polyfit(x, e[i - 1:i + 2], 2)
        if p[0] > 0:
            z = float(np.clip(-p[1] / (2 * p[0]), x[0], x[2]))
    return {**out, "zero": float(z), "rms": float(e[i]), "at_edge": i in (0, len(zeros) - 1)}


def pooled_zero(errors: np.ndarray, zeros: Sequence[float] = ZEROS) -> float:
    """The zero with the smallest summed squared error over trees (``errors`` is ``(trees, zeros)``).

    One zero must suit every tree, so a zero that some tree rules out (infinite error: at or below its
    values) is not a candidate; trees that no zero fits are left out first.

    Example:
        >>> pooled_zero(np.array([[3.0, 1.0, 2.0], [2.5, 1.5, 1.0], [np.inf, 1.0, 1.0]]), (250.0, 300.0, 350.0))
        300.0
    """
    e = np.asarray(errors, float)
    e = e[np.isfinite(e).any(axis=1)]
    cols = np.isfinite(e).all(axis=0) if len(e) else np.zeros(len(zeros), bool)
    if not cols.any():
        return float("nan")
    total = np.where(cols, (np.where(np.isfinite(e), e, 0) ** 2).sum(axis=0), np.inf)
    return float(zeros[int(np.argmin(total))])


def _calibrate_one(job: tuple) -> dict:
    """One tree's ground blocks and zero fits (top-level for worker processes)."""
    from treeskel.io import load_tree

    rec, zeros = job
    d = load_tree(rec, "winter")
    blocks = ground_blocks(d["depth"], d["valid"])
    kept = blocks[blocks[:, 2] >= MIN_VALUE]
    return {"tree": rec.tree_id, "row": rec.row, "fit": fit_zero(kept, zeros), "fit_all": fit_zero(blocks, zeros),
            "errors": zero_errors(kept, zeros).tolist(), "errors_all": zero_errors(blocks, zeros).tolist()}


def calibrate(results: Sequence[dict], zeros: Sequence[float] = ZEROS) -> dict:
    """Summary of the per-tree zero fits: the per-tree estimates' median and quartiles, the pooled estimate
    (with and without the blocks below :data:`MIN_VALUE`), per orchard row, and how session 2's zero and
    the disparity-like map compare."""
    def summary(key: str, err_key: str) -> dict:
        """The zero's median, quartiles, pooled estimate and fit quality for one block selection (``key``)."""
        z = np.array([r[key]["zero"] for r in results], float)
        fin = np.isfinite(z) & ~np.array([r[key]["at_edge"] for r in results])
        errs = np.array([r[err_key] for r in results], float)
        rms = np.array([r[key]["rms"] for r in results], float)
        s2 = np.array([r[key]["rms_session2"] for r in results], float)
        disp = np.array([r[key]["rms_disparity"] for r in results], float)
        ok = np.isfinite(rms) & np.isfinite(s2) & np.isfinite(disp)
        q = np.percentile(z[fin], [25, 50, 75]).tolist() if fin.any() else [None] * 3
        return {"n_trees": len(results), "n_fitted": int(fin.sum()),
                "at_edge": int(sum(r[key]["at_edge"] for r in results)),
                "zero_median": q[1], "zero_q1": q[0], "zero_q3": q[2], "pooled_zero": pooled_zero(errs, zeros),
                "rms_median": float(np.median(rms[ok])) if ok.any() else None,
                "rms_session2_median": float(np.median(s2[ok])) if ok.any() else None,
                "rms_disparity_median": float(np.median(disp[ok])) if ok.any() else None,
                "better_than_disparity": float(np.mean(rms[ok] < disp[ok])) if ok.any() else None}

    out = {"zeros_tried": [min(zeros), max(zeros)], "min_value": MIN_VALUE,
           "kept": summary("fit", "errors"), "all_blocks": summary("fit_all", "errors_all"), "rows": {}}
    for row in sorted({r["row"] for r in results}):
        sub = [r for r in results if r["row"] == row]
        z = np.array([r["fit"]["zero"] for r in sub if not r["fit"]["at_edge"]], float)
        z = z[np.isfinite(z)]
        out["rows"][str(row)] = {"n": len(sub), "zero_median": float(np.median(z)) if len(z) else None,
                                 "pooled_zero": pooled_zero(np.array([r["errors"] for r in sub], float), zeros)}
    return out


def calibration_lines(s: dict) -> list[str]:
    """The calibration summary as text."""
    def f(x, spec=".0f"):
        """A number as text, or ``-`` if missing."""
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)

    out = [f"The depth map's zero from the ground (flat ground: 1/(v0 - v) affine in the pixel's position); "
           f"zeros tried {f(s['zeros_tried'][0])}-{f(s['zeros_tried'][1])}:"]
    for name, k in (("blocks of value >= " + str(s["min_value"]), "kept"), ("all ground blocks", "all_blocks")):
        g = s[k]
        out.append(f"  {name}: {g['n_fitted']} of {g['n_trees']} trees fitted ({g['at_edge']} at the grid's edge); "
                   f"zero median {f(g['zero_median'])} [{f(g['zero_q1'])}, {f(g['zero_q3'])}], pooled "
                   f"{f(g['pooled_zero'])}")
        out.append(f"    RMS error (depth units, median over trees): best zero {f(g['rms_median'], '.2f')}, session 2's "
                   f"255 {f(g['rms_session2_median'], '.2f')}, linear in disparity {f(g['rms_disparity_median'], '.2f')}; "
                   f"the best zero fits better than disparity on {f(g['better_than_disparity'], '.0%')} of trees")
    out.append("  by orchard row (median of trees / pooled): " + ", ".join(
        f"{row} {f(v['zero_median'])}/{f(v['pooled_zero'])}" for row, v in s["rows"].items()))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run the calibration on the development rows; print and save the results."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.depth3d", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--calibrate", action="store_true", help="estimate the depth map's zero from the ground")
    ap.add_argument("--split", default="splits.json", help="its test rows are never loaded")
    ap.add_argument("--out", default="docs/results/depth3d")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    if not args.calibrate:
        ap.error("nothing to do: give --calibrate")

    # Every development-row tree (the ground needs no clicks); the test rows are never loaded.
    test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])
    records = sorted((t for d in args.data_dirs for t in index_dataset(d).trees if t.row not in test_rows),
                     key=lambda t: t.tree_id)
    print(f"Fitting the depth map's zero on the ground of {len(records)} development-row trees...", flush=True)
    t0, step, results = time.time(), max(1, min(25, len(records) // 10)), []
    # Each tree's zero is fitted in a worker process: the zero that makes its ground blocks come out flattest.
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(records)))) as ex:
        for i, r in enumerate(ex.map(_calibrate_one, [(rec, ZEROS) for rec in records], chunksize=4), 1):
            results.append(r)
            if i % step == 0 or i == len(records):
                el = time.time() - t0
                print(f"  {i}/{len(records)} trees, {el:.0f} s elapsed, about {el / i * (len(records) - i):.0f} s left",
                      flush=True)
    # Pool the trees (median, quartiles, pooled fit, per row), then save the per-tree table and the summary.
    summary = calibrate(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibration_trees.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tree", "row", "zero", "rms", "rms_session2", "rms_disparity", "n_blocks", "at_edge",
                    "zero_all_blocks", "at_edge_all_blocks"])
        for r in results:
            a, b = r["fit"], r["fit_all"]
            w.writerow([r["tree"], r["row"], *(round(a[k], 3) for k in ("zero", "rms", "rms_session2", "rms_disparity")),
                        a["n_blocks"], a["at_edge"], round(b["zero"], 3), b["at_edge"]])
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    lines = calibration_lines(summary)
    print("\n".join(lines))
    (out / "calibration_output.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved {out / 'calibration_trees.csv'}, calibration.json and calibration_output.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
