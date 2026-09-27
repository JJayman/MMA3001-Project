"""Reproduce the session-8 evidence that the winter depth images sit about 4 px below the colour images.

A stand-alone analysis script, not part of the package: session 9 is to
turn the measurement into a ``treeskel`` module that runs on all
development trees. It uses the development-row trees in ``data/sample`` and
``data/tuning`` (test rows from ``splits.json`` are left out), traces them
with the default settings, and compares the leaders' edges found in colour
(``branch_edges``) with the edges of the depth silhouette along the same
cross-sections (``centre_runs``). It also re-traces with the depth moved,
and records the two fixes that were tried and rejected (re-centring the
leaders on the colour bark centre, and trunk widths from colour).

Run from the repository root (takes about 10 minutes)::

    python docs/results/depth_registration/reproduce.py > docs/results/depth_registration/output_21_dev_trees.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line
from skimage.morphology import disk

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from treeskel.guided_gt import (  # noqa: E402
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
from treeskel.io import index_dataset, load_tree  # noqa: E402
from treeskel.measure_width import centre_runs  # noqa: E402
from treeskel.winter_gt import foreground_mask  # noqa: E402

HALF = GuidedParams().profile_half_width
SHIFTS = (0, -2, -3, -4, -5)
"""Vertical depth shifts tried (rows; negative moves the depth up), each with 1 column to the right."""
REGISTERED = (-4, 1)
"""The shift used to re-trace: the depth moved up 4 rows and right 1 column."""


def register(d: dict, shift: tuple[int, int]) -> dict:
    """A copy of ``load_tree``'s output with depth and valid moved by ``shift`` (rows, cols).

    The uncovered edge gets no reading.
    """
    out = dict(d)
    dy, dx = shift
    for k in ("depth", "valid"):
        a = np.roll(d[k], shift, axis=(0, 1))
        if dy < 0:
            a[dy:] = 0
        elif dy > 0:
            a[:dy] = 0
        if dx < 0:
            a[:, dx:] = 0
        elif dx > 0:
            a[:, :dx] = 0
        out[k] = a
    return out


def huber_fit(X: np.ndarray, y: np.ndarray, iters: int = 40) -> np.ndarray:
    """Robust (Huber) least squares ``y ≈ X @ c``."""
    w = np.ones(len(y))
    c = np.zeros(X.shape[1])
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        c, *_ = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)
        r = y - X @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        w = np.where(np.abs(r) <= 1.345 * s, 1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
    return c


def silhouette(d: dict) -> np.ndarray:
    """The depth foreground as ``guided_gt`` builds it."""
    return ndi.binary_closing(foreground_mask(d["depth"], d["valid"]), disk(2))


def cross_sections(rgb, fg, gt) -> list[dict]:
    """Colour and depth edges on every leader cross-section clear of the other leader."""
    out = []
    for side, other in (("left", gt.right), ("right", gt.left)):
        p = getattr(gt, side)
        f, l = branch_edges(rgb, p, HALF)
        ok = clear_of(p, other, HALF + NEIGHBOUR_MARGIN) & ((l - f + 1) / 2.0 > 2.5)
        d0, d1, closed = centre_runs(straighten(fg.astype(float), p, HALF, order=0) > 0.5)
        grey = ndi.uniform_filter1d(straighten(rgb.astype(float).mean(2), p, HALF), 7, axis=0, mode="nearest")
        n = path_normals(p)
        for i in np.nonzero(ok)[0]:
            a, b = int(f[i]) + HALF, int(l[i]) + HALF
            mid = (a + b) // 2
            lit = 1.0 if grey[i, mid:b + 1].mean() > grey[i, a:mid + 1].mean() else -1.0
            out.append({"side": side, "row": p[i, 0], "col": p[i, 1], "n_row": n[i, 0], "n_col": n[i, 1],
                        "first": f[i], "last": l[i], "closed": bool(closed[i]),
                        "d0": d0[i], "d1": d1[i], "lit": lit})
    return out


def arrays(rows: list[dict], closed_only: bool = True) -> dict:
    rows = [r for r in rows if r["closed"] or not closed_only]
    return {k: np.array([r[k] for r in rows], float) for k in rows[0]
            if not isinstance(rows[0][k], str)} if rows else {}


def shift_fit(a: dict) -> np.ndarray:
    """(dy, dx): the depth silhouette's centre minus the colour centre, as a fixed image shift."""
    delta = (a["d0"] + a["d1"]) / 2.0 - (a["first"] + a["last"]) / 2.0
    return huber_fit(np.c_[a["n_row"], a["n_col"]], delta)


def edge_overshoot(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Depth past colour at the upper and the lower edge of each cross-section (px)."""
    neg, pos = a["first"] - a["d0"], a["d1"] - a["last"]          # at the edge along −normal / +normal
    upper = np.where(a["n_row"] < 0, pos, neg)
    lower = np.where(a["n_row"] < 0, neg, pos)
    return upper, lower


def shift_path(path: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Move each point along its normal and reconnect the points into an 8-connected path."""
    pts = np.round(path + offsets[:, None] * path_normals(path)).astype(int)
    out = [tuple(pts[0])]
    for b in pts[1:]:
        if tuple(b) != out[-1]:
            rr, cc = draw_line(out[-1][0], out[-1][1], int(b[0]), int(b[1]))
            out.extend(zip(rr[1:].tolist(), cc[1:].tolist()))
    return np.array(out, int)


def colour_centre_offsets(rgb, gt, side, other) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed offset of the colour bark centre from the path (the re-centring that was rejected)."""
    p = getattr(gt, side)
    f, l = branch_edges(rgb, p, HALF)
    centre = (f + l) / 2.0
    ok = clear_of(p, other, HALF + NEIGHBOUR_MARGIN) & ((l - f + 1) / 2.0 > 2.5)
    idx = np.nonzero(ok)[0]
    med = np.full(len(p), np.nan)
    for i in range(len(p)):
        sel = idx[np.abs(idx - i) <= 10]
        if len(sel) >= 5:
            med[i] = np.median(centre[sel])
    med[0] = 0.0                                                    # the fork stays put
    x = np.arange(len(p))
    good = np.isfinite(med)
    off = ndi.gaussian_filter1d(np.interp(x, x[good], med[good]), 5.0, mode="nearest")
    off[0] = 0.0
    return np.clip(off, -6.0, 6.0), centre


def main() -> None:
    test_rows = set(json.loads((ROOT / "splits.json").read_text())["test_rows"])
    clicks = load_clicks(ROOT / "annotations" / "winter_clicks.json")
    recs = {t.tree_id: t for d in ("data/sample", "data/tuning") for t in index_dataset(ROOT / d).trees}
    trees = sorted(t for t in recs if recs[t].row not in test_rows and t in clicks
                   and clicks[t].complete and not clicks[t].skip)
    print(f"{len(trees)} development-row trees: {' '.join(trees)}\n")

    loaded, gts, rows = {}, {}, {}
    for t in trees:
        loaded[t] = load_tree(recs[t], "winter")
        d = loaded[t]
        gts[t] = guided_gt(d["rgb"], d["depth"], d["valid"], clicks[t])
        rows[t] = cross_sections(d["rgb"], gts[t].tree_mask, gts[t])
    allrows = [r | {"tree": t} for t in trees for r in rows[t]]
    a = arrays(allrows)

    print("1. Depth centre minus colour centre across the leaders, as a fixed image shift")
    dy, dx = shift_fit(a)
    print(f"   {len(a['n_row'])} cross-sections clear of the other leader and closed in depth: "
          f"dy {dy:+.2f} px (depth lower), dx {dx:+.2f} px")
    delta = (a["d0"] + a["d1"]) / 2.0 - (a["first"] + a["last"]) / 2.0
    c = huber_fit(np.c_[a["n_row"], a["n_col"], a["lit"]], delta)
    print(f"   with a lighting term (which side of the bark is brighter): dy {c[0]:+.2f}, dx {c[1]:+.2f}, "
          f"lighting {c[2]:+.2f} px")
    lit_dir = np.c_[a["n_row"] * a["lit"], a["n_col"] * a["lit"]].mean(0)
    print(f"   mean direction of the lit side (row, col): ({lit_dir[0]:+.2f}, {lit_dir[1]:+.2f}); "
          "negative row = from above, positive col = from the right")
    upper, lower = edge_overshoot(a)
    slope = np.abs(a["n_row"]) > 0.4
    print(f"   sloping cross-sections (|n_row| > 0.4, {slope.sum()}): depth reaches past the upper colour edge by "
          f"{np.median(upper[slope]):+.1f} px and past the lower edge by {np.median(lower[slope]):+.1f} px (medians)")
    for name, sel in (("upper half of the image", a["row"] < 240), ("lower half", a["row"] >= 240),
                      ("left half", a["col"] < 320), ("right half", a["col"] >= 320)):
        s = {k: v[sel] for k, v in a.items()}
        sy, sx = shift_fit(s)
        print(f"   {name:24s} dy {sy:+.2f}, dx {sx:+.2f} ({sel.sum()} cross-sections)")
    per = []
    for t in trees:
        s = arrays([r for r in rows[t]])
        if not s:
            continue
        keep = np.abs(s["n_row"]) > 0.3
        if keep.sum() >= 40:
            per.append(shift_fit({k: v[keep] for k, v in s.items()})[0])
    per = np.array(per)
    print(f"   per tree (sloping cross-sections, {len(per)} trees): dy median {np.median(per):+.2f} "
          f"(IQR {np.percentile(per, 25):+.2f} to {np.percentile(per, 75):+.2f}), {int((per > 0).sum())} positive")

    print("\n2. The same traced paths, with the depth moved up (and 1 column right)")
    for sy in SHIFTS:
        shift = (sy, 1 if sy else 0)
        moved = []
        for t in trees:
            d = register(loaded[t], shift)
            fg = silhouette(d)
            for side, other in (("left", gts[t].right), ("right", gts[t].left)):
                p = getattr(gts[t], side)
                d0, d1, closed = centre_runs(straighten(fg.astype(float), p, HALF, order=0) > 0.5)
                sel = [r for r in rows[t] if r["side"] == side]
                idx = {(r["row"], r["col"]): r for r in sel}
                for i, (y, x) in enumerate(p):
                    r = idx.get((y, x))
                    if r is not None and closed[i]:
                        moved.append(r | {"d0": d0[i], "d1": d1[i], "closed": True})
        m = arrays(moved)
        ry, rx = shift_fit(m)
        up, lo = edge_overshoot(m)
        sl = np.abs(m["n_row"]) > 0.4
        print(f"   depth moved {shift}: remaining shift dy {ry:+.2f}, dx {rx:+.2f}; sloping cross-sections: "
              f"past the upper edge {np.mean(up[sl]):+.2f}, past the lower edge {np.mean(lo[sl]):+.2f} px (means)")

    print(f"\n3. Does the traced path follow the depth? Re-traced with the depth moved by {REGISTERED}")
    moved_gts = {}
    for shift in ((0, 0), REGISTERED):
        pts = []
        for t in trees:
            d = register(loaded[t], shift)
            gt = gts[t] if shift == (0, 0) else guided_gt(d["rgb"], d["depth"], d["valid"], clicks[t])
            if shift != (0, 0):
                moved_gts[t] = gt
            for side, other in (("left", gt.right), ("right", gt.left)):
                p = getattr(gt, side)
                f, l = branch_edges(d["rgb"], p, HALF)
                ok = clear_of(p, other, HALF + NEIGHBOUR_MARGIN) & ((l - f + 1) / 2.0 > 2.5)
                n = path_normals(p)
                pts += [(n[i, 0], n[i, 1], (f[i] + l[i]) / 2.0) for i in np.nonzero(ok)[0]]
        v = np.array(pts)
        cy, cx = huber_fit(v[:, :2], v[:, 2])
        print(f"   depth moved {shift}: colour bark centre minus path = {cy:+.2f}*n_row {cx:+.2f}*n_col; "
              f"|colour centre - path| median {np.median(np.abs(v[:, 2])):.2f} px over {len(v)} cross-sections")
    trunk = np.array([moved_gts[t].radii["trunk"][-1] - gts[t].radii["trunk"][-1] for t in trees])
    leader = np.array([np.median(moved_gts[t].radii[k]) - np.median(gts[t].radii[k])
                       for t in trees for k in ("left", "right")])
    print(f"   drawn radii after re-tracing: trunk at the fork {np.median(trunk):+.2f} px (median; range "
          f"{trunk.min():+.1f} to {trunk.max():+.1f}), leaders {np.median(leader):+.2f} px (range "
          f"{leader.min():+.1f} to {leader.max():+.1f})")

    print("\n4. Rejected: re-centring the leaders on the colour bark centre (depth as loaded)")
    colour_c = (a["first"] + a["last"]) / 2.0
    depth_c = (a["d0"] + a["d1"]) / 2.0
    print(f"   colour centre vs depth centre, relative to the path: Pearson r {np.corrcoef(colour_c, depth_c)[0, 1]:+.2f}; "
          f"median |colour - depth| {np.median(np.abs(colour_c - depth_c)):.2f} px")
    better = worse = capped = 0
    for t in trees:
        d, gt = loaded[t], gts[t]
        for side, other in (("left", gt.right), ("right", gt.left)):
            p = getattr(gt, side)
            off, _ = colour_centre_offsets(d["rgb"], gt, side, other)
            q = shift_path(p, off)
            capped += np.abs(off).max() >= 6.0 - 1e-9
            before, after = [], []
            for path, store in ((p, before), (q, after)):
                f, l = branch_edges(d["rgb"], path, HALF)
                ok = clear_of(path, other, HALF + NEIGHBOUR_MARGIN) & ((l - f + 1) / 2.0 > 2.5)
                d0, d1, closed = centre_runs(straighten(gt.tree_mask.astype(float), path, HALF, order=0) > 0.5)
                store.extend(np.abs((d0 + d1) / 2.0)[ok & closed])
            if before and after:
                better += np.median(after) < np.median(before)
                worse += np.median(after) > np.median(before)
    print(f"   after re-centring, the path is nearer the depth centre on {better} leaders and further on {worse} "
          f"(of {2 * len(trees)}); the shift reached its 6 px cap on {capped} leaders")

    print("\n5. Rejected: trunk widths from colour. Trunk half-width over the upper 70% of the trunk (px)")
    print("   tree    drawn (depth EDT, taper)  colour  depth run - 2")
    table = []
    for t in trees:
        d, gt = loaded[t], gts[t]
        p = gt.trunk
        s = arc_length(p)
        if s[-1] < 30:
            continue
        f, l = branch_edges(d["rgb"], p, 40)
        colour = (l - f + 1) / 2.0
        d0, d1, closed = centre_runs(straighten(gt.tree_mask.astype(float), p, 40, order=0) > 0.5)
        depth = np.where(closed, (d1 - d0 + 1) / 2.0 - 2.0, np.nan)
        up = s >= 0.3 * s[-1]
        row = (t, float(np.median(gt.radii["trunk"])), float(np.median(colour[up])),
               float(np.nanmedian(depth[up])) if np.isfinite(depth[up]).any() else float("nan"))
        table.append(row)
        print(f"   {row[0]} {row[1]:8.1f} {row[2]:24.1f} {row[3]:8.1f}")
    v = np.array([r[1:] for r in table], float)
    print(f"   medians: drawn {np.median(v[:, 0]):.1f}, colour {np.median(v[:, 1]):.1f}, depth run - 2 "
          f"{np.nanmedian(v[:, 2]):.1f}; drawn - colour per tree: median {np.median(v[:, 0] - v[:, 1]):+.1f} "
          f"(IQR {np.percentile(v[:, 0] - v[:, 1], 25):+.1f} to {np.percentile(v[:, 0] - v[:, 1], 75):+.1f})")


if __name__ == "__main__":
    main()
