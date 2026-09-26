"""Measure the masks at the fork against the winter depth, to set the shoulder fill.

Where a leader leaves the trunk, the mask used to show a notch (a
"shoulder") beside it, because each part's width is one taper along its
whole length: a thin leader stuck out of the round end of a much wider
trunk. :func:`treeskel.guided_gt.shoulder_fill` fills these notches, out
to the straight line from the trunk's edge to the leader's edge
``shoulder_length`` trunk radii up the leader. This module measures, on
**development-row trees only**, how far that fill should reach:

1. **Shoulder coverage.** For each leader, the triangle between the fork,
   the trunk's edge level with the fork and the leader's outer edge
   ``reach`` trunk radii up the leader (:func:`shoulder_triangle`). The
   fraction of it that is covered (:func:`coverage`) by the winter depth
   silhouette (the wood actually there) is compared with the fraction
   covered by the masks for each ``shoulder_length``. The silhouette is also
   shrunk by 2 px, because depth "bleeds" past the edges of thin branches,
   so the true coverage lies between the two. The setting whose mean
   coverage matches the depth's is found for each reach and reference, and
   checked by leave-one-tree-out.
2. **Precision.** The fraction of the pixels the fill adds that lie on the
   depth silhouette, next to the precision and IoU of the whole mask
   against the silhouette around the fork (:func:`fork_zone`).
3. **Width decay (an independent check).** Along each leader, the depth
   silhouette's half-width falls from its value at the fork to the leader's
   own width. A smooth step fitted to it (:func:`fit_decay_length`) gives
   the length of that transition in trunk radii.

Usage::

    python -m treeskel.measure_fork data/raw --clicks annotations/winter_clicks.json \\
        --split splits.json --out outputs/measure_fork --figure outputs/fork_shoulders.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar
from skimage.draw import polygon
from skimage.morphology import disk

from treeskel.guided_gt import (
    GuidedParams,
    arc_length,
    fill_shoulders,
    guided_gt,
    load_clicks,
    outward_normal,
)
from treeskel.winter_gt import WinterGT, render_main_branches, report_failed, save_image

LENGTHS = (0.0, 1.0, 1.4, 1.7, 2.0, 2.5)
"""``shoulder_length`` settings measured (0 = no fill, the old masks)."""

REACHES = (1.5, 2.0)
"""How far up the leader the shoulder triangle reaches, in trunk radii.
Depth bleeds least near the fork, so the shorter reach is the more reliable."""

BLEED_PX = 2
"""Shrink applied to the depth silhouette for the bleed-corrected reference."""


def shoulder_triangle(gt: WinterGT, side: str, reach: float) -> np.ndarray:
    """The shoulder beside one leader, as a triangle of ``(row, col)`` vertices.

    Vertices: the fork; the trunk's edge level with the fork, on the leader's
    outer side (the trunk's direction is taken over its last ``R`` px, where
    ``R`` is its drawing radius at the fork); and the leader's outer edge
    ``reach·R`` px up the leader. Without a shoulder fill, a notch cuts into
    this triangle.

    Args:
        gt: A guided result with fitted radii.
        side: ``"left"`` or ``"right"``.
        reach: Distance up the leader, in trunk radii.

    Returns:
        ``(3, 2)`` float array.
    """
    radius = float(gt.radii["trunk"][-1])
    fork = np.asarray(gt.fork, float)
    back = arc_length(gt.trunk)
    back = back[-1] - back
    below = gt.trunk[int(np.argmin(np.abs(back - radius)))]
    up = fork - below
    up /= np.hypot(*up) or 1.0
    leader, other = (gt.left, gt.right) if side == "left" else (gt.right, gt.left)
    s = arc_length(leader)
    j = min(int(np.searchsorted(s, reach * radius)), len(leader) - 1)
    n = outward_normal(leader, other, max(reach * radius, 10.0))
    across = np.array([-up[1], up[0]])
    across = across if across @ n >= 0 else -across          # the trunk's edge on the leader's side
    return np.array([fork, fork + radius * across, leader[j] + n * min(gt.radii[side][j], radius)])


def coverage(obj: np.ndarray, triangle: np.ndarray) -> float:
    """Fraction of the pixels inside ``triangle`` that are set in ``obj`` (nan if none)."""
    rr, cc = polygon(triangle[:, 0], triangle[:, 1], shape=obj.shape)
    return float(obj[rr, cc].mean()) if len(rr) else float("nan")


def fork_zone(gt: WinterGT, zone: float = 2.5) -> np.ndarray:
    """Pixels within ``zone`` trunk radii of the fork and one trunk radius of a path.

    The second condition keeps side shoots, posts and ground further from the
    main branches out of the comparison with the depth silhouette.
    """
    radius = float(gt.radii["trunk"][-1])
    paths = np.zeros(gt.tree_mask.shape, bool)
    for p in (gt.trunk, gt.left, gt.right):
        paths[p[:, 0], p[:, 1]] = True
    yy, xx = np.indices(paths.shape)
    near_fork = np.hypot(yy - gt.fork[0], xx - gt.fork[1]) <= zone * radius
    return near_fork & (ndi.distance_transform_edt(~paths) <= radius)


def fit_decay_length(s: np.ndarray, width: np.ndarray, radius: float) -> float:
    """Length (px) over which a leader's depth half-width falls from the fork.

    ``width`` is modelled as ``far + (w0 − far)·½(1 + cos(π·s/L))`` for
    ``s < L`` and ``far`` beyond, where ``w0`` is the median width within
    3 px of the fork and ``far`` the median between 4 and 8 trunk radii
    along; ``L`` is fitted by least squares over the first 5 trunk radii.
    Depth holes (width 0) are ignored.

    Args:
        s: Distance along the leader from the fork (px).
        width: Depth half-width at each point (px).
        radius: The trunk's radius at the fork (px).

    Returns:
        ``L`` in px, or nan if the width does not fall by more than 2 px
        (nothing to measure) or too few points remain.
    """
    ok = width > 0
    near, far_sel, fit_sel = ok & (s < 3), ok & (s >= 4 * radius) & (s < 8 * radius), ok & (s <= 5 * radius)
    if not near.any() or not far_sel.any() or fit_sel.sum() < 10:
        return float("nan")
    w0, far = np.median(width[near]), np.median(width[far_sel])
    if w0 <= far + 2:
        return float("nan")
    x, y = s[fit_sel], width[fit_sel]

    def loss(length: float) -> float:
        u = np.clip(x / length, 0, 1)
        return float(np.mean((y - far - (w0 - far) * 0.5 * (1 + np.cos(np.pi * u))) ** 2))

    return float(minimize_scalar(loss, bounds=(2.0, 8 * radius), method="bounded").x)


def _crop(image: np.ndarray, centre: Sequence[int], half: int) -> np.ndarray:
    r0 = int(np.clip(centre[0] - half, 0, image.shape[0] - 2 * half))
    c0 = int(np.clip(centre[1] - half, 0, image.shape[1] - 2 * half))
    return image[r0:r0 + 2 * half, c0:c0 + 2 * half]


def _paint(rgb: np.ndarray, mask: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Mask on the image: red fill, added pixels cyan, outline yellow."""
    im = rgb.astype(float)
    im[mask] = 0.55 * im[mask] + 0.45 * np.array([255, 0, 60])
    added = mask & ~base
    im[added] = 0.4 * im[added] + 0.6 * np.array([0, 255, 255])
    im[mask & ~ndi.binary_erosion(mask)] = (255, 255, 0)
    return im.astype(np.uint8)


def measure_tree(job: tuple) -> dict:
    """All measurements for one tree (top-level so worker processes can run it).

    Args:
        job: ``(tree_record, clicks, lengths, figure_lengths or None)``.

    Returns:
        ``{"tree", "fork_radius", "leaders": [...], "masks": {...}, "tiles"}``.
    """
    from treeskel.io import load_tree

    rec, clicks, lengths, figure_lengths = job
    d = load_tree(rec, "winter")
    gt = guided_gt(d["rgb"], d["depth"], d["valid"], clicks, GuidedParams(shoulder_length=0))
    base = render_main_branches(gt)
    masks = {k: base | fill_shoulders(gt, k) for k in lengths}
    depth = gt.tree_mask                                   # the depth silhouette (closed foreground)
    shrunk = ndi.binary_erosion(depth, disk(BLEED_PX))
    zone = fork_zone(gt)
    radius = float(gt.radii["trunk"][-1])
    dt = ndi.distance_transform_edt(depth)
    leaders = []
    for side in ("left", "right"):
        path = getattr(gt, side)
        row = {"side": side, "decay_ratio": fit_decay_length(arc_length(path), dt[path[:, 0], path[:, 1]],
                                                              radius) / radius}
        for reach in REACHES:
            tri = shoulder_triangle(gt, side, reach)
            row[f"depth@{reach}"] = coverage(depth, tri)
            row[f"shrunk@{reach}"] = coverage(shrunk, tri)
            for k in lengths:
                row[f"k{k}@{reach}"] = coverage(masks[k], tri)
        leaders.append(row)
    stats = {}
    for k, m in masks.items():
        added = m & ~base
        both = (m & depth & zone).sum()
        stats[k] = {"added": int(added.sum()), "added_on_depth": int((added & depth).sum()),
                    "zone_precision": both / max((m & zone).sum(), 1),
                    "zone_iou": both / max(((m | depth) & zone).sum(), 1)}
    tiles = None
    if figure_lengths is not None:
        half = 40
        tiles = [_crop(d["rgb"], gt.fork, half)] + [
            _crop(_paint(d["rgb"], base | fill_shoulders(gt, k) if k > 0 else base, base), gt.fork, half)
            for k in (0.0, *figure_lengths)]
    return {"tree": rec.tree_id, "fork_radius": radius, "leaders": leaders, "masks": stats, "tiles": tiles}


def matching_length(results: list[dict], lengths: Sequence[float], reach: float, ref: str,
                    exclude: str | None = None) -> float:
    """``shoulder_length`` at which the masks' mean coverage equals the reference's.

    Linear interpolation of mean coverage against the measured lengths
    (coverage only grows with length). nan if the reference lies outside
    the measured range.

    Args:
        results: :func:`measure_tree` results.
        lengths: The measured lengths, increasing.
        reach: Which shoulder triangle.
        ref: ``"depth"`` or ``"shrunk"``.
        exclude: A tree id to leave out.
    """
    rows = [row for r in results if r["tree"] != exclude for row in r["leaders"]]
    target = np.nanmean([row[f"{ref}@{reach}"] for row in rows])
    cov = np.array([np.nanmean([row[f"k{k}@{reach}"] for row in rows]) for k in lengths])
    if not cov[0] <= target <= cov[-1]:
        return float("nan")
    return float(np.interp(target, cov, lengths))


def _or_none(x: float) -> float | None:
    """``x``, or None if it is nan (JSON has no nan)."""
    return None if np.isnan(x) else float(x)


def summarise(results: list[dict], lengths: Sequence[float], seed: int = 0) -> dict:
    """Pool :func:`measure_tree` results into the report."""
    rows = [row for r in results for row in r["leaders"]]
    out = {"trees": [r["tree"] for r in results], "n_leaders": len(rows), "lengths": list(lengths),
           "reaches": list(REACHES), "bleed_px": BLEED_PX, "coverage": {}, "matching_length": {}, "masks": {}}
    for reach in REACHES:
        cov = {ref: float(np.nanmean([row[f"{ref}@{reach}"] for row in rows])) for ref in ("depth", "shrunk")}
        cov.update({f"k{k}": float(np.nanmean([row[f"k{k}@{reach}"] for row in rows])) for k in lengths})
        # Share of leaders whose shoulder is clearly notched (less than 80% covered).
        cov["notched"] = {name: float(np.mean([row[f"{name}@{reach}"] < 0.8 for row in rows]))
                          for name in ("depth", "shrunk", *(f"k{k}" for k in lengths))}
        out["coverage"][str(reach)] = cov
        for ref in ("depth", "shrunk"):
            loo = np.array([matching_length(results, lengths, reach, ref, exclude=r["tree"]) for r in results])
            found = loo[~np.isnan(loo)]
            out["matching_length"][f"{ref}@{reach}"] = {
                "all_trees": _or_none(matching_length(results, lengths, reach, ref)),
                "leave_one_out_min": float(found.min()) if len(found) else None,
                "leave_one_out_max": float(found.max()) if len(found) else None}
    for k in lengths:
        added = sum(r["masks"][k]["added"] for r in results)
        on = sum(r["masks"][k]["added_on_depth"] for r in results)
        out["masks"][str(k)] = {
            "added_px": added, "added_on_depth": on / added if added else None,
            "zone_precision": float(np.mean([r["masks"][k]["zone_precision"] for r in results])),
            "zone_iou": float(np.mean([r["masks"][k]["zone_iou"] for r in results]))}
    ratios = np.array([row["decay_ratio"] for row in rows], float)
    tree_of = [r["tree"] for r in results for _ in r["leaders"]]
    ok = ~np.isnan(ratios)
    decay = {"n_measured": int(ok.sum())}
    if ok.any():
        rng = np.random.default_rng(seed)
        ids = sorted(set(tree_of))
        by_tree = {t: ratios[[i for i, u in enumerate(tree_of) if u == t]] for t in ids}
        boot = [np.nanmedian(np.concatenate([by_tree[t] for t in rng.choice(ids, len(ids))]))
                for _ in range(2000)]
        loo = [np.nanmedian(np.concatenate([by_tree[u] for u in ids if u != t])) for t in ids]
        decay.update({"median": float(np.median(ratios[ok])),
                      "iqr": [float(v) for v in np.percentile(ratios[ok], [25, 75])],
                      "bootstrap_95ci": [float(v) for v in np.nanpercentile(boot, [2.5, 97.5])],
                      "leave_one_out": [float(np.nanmin(loo)), float(np.nanmax(loo))]})
    out["decay_ratio"] = decay
    return out


def _figure(results: list[dict], labels: Sequence[str], scale: int = 2) -> Image.Image:
    """Rows of fork crops: winter RGB, then the mask for each setting."""
    rows = []
    for r in results:
        size = r["tiles"][0].shape[0] * scale
        row = Image.new("RGB", (len(r["tiles"]) * (size + 4), size + 14), "black")
        for i, (tile, label) in enumerate(zip(r["tiles"], [r["tree"], *labels])):
            row.paste(Image.fromarray(tile).resize((size, size), Image.NEAREST), (i * (size + 4), 14))
            ImageDraw.Draw(row).text((i * (size + 4) + 3, 1), label, fill="white")
        rows.append(np.asarray(row))
    return Image.fromarray(np.concatenate(rows, 0))


def main(argv: Sequence[str] | None = None) -> int:
    """Measure development-row trees; print and save the results."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.measure_fork", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json", help="its test rows are never measured")
    ap.add_argument("--out", default="outputs/measure_fork")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--figure", help="also save fork crops (winter RGB, masks) as this image")
    ap.add_argument("--figure-trees", nargs="+", help="trees for the figure (default: the first six)")
    ap.add_argument("--figure-lengths", type=float, nargs="+", default=[GuidedParams().shoulder_length],
                    help="shoulder_length settings shown in the figure, after 0")
    args = ap.parse_args(argv)

    test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])
    clicks = load_clicks(args.clicks)
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    ids = sorted(t for t, c in clicks.items()
                 if t in records and c.complete and not c.skip and records[t].row not in test_rows)
    held_out = sorted(t for t in records if records[t].row in test_rows)
    if not ids:
        print("No annotated development-row trees found; nothing to measure.")
        return 1
    fig_ids = set(args.figure_trees or ids[:6]) if args.figure else set()
    jobs = [(records[t], clicks[t], LENGTHS, args.figure_lengths if t in fig_ids else None) for t in ids]

    print(f"Measuring {len(ids)} development-row trees ({len(held_out)} test-row trees left out)...", flush=True)
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

    report = summarise(results, LENGTHS)
    report["test_rows_excluded"] = sorted(test_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "leaders.csv", "w", newline="") as f:
        rows = [{"tree": r["tree"], "fork_radius": round(r["fork_radius"], 2),
                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}}
                for r in results for row in r["leaders"]]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{report['n_leaders']} leaders on {len(ids)} trees. Shoulder coverage (mean fraction of the triangle):")
    for reach in REACHES:
        c = report["coverage"][str(reach)]
        masks = ", ".join(f"{k:g}: {c[f'k{k}']:.3f}" for k in LENGTHS)
        print(f"  reach {reach:g} R: depth {c['depth']:.3f}, depth shrunk {BLEED_PX} px {c['shrunk']:.3f} | "
              f"masks by shoulder_length {masks}")
        notched = ", ".join(f"{k}: {v:.0%}" for k, v in c["notched"].items())
        print(f"    leaders with less than 80% covered: {notched}")
    print("shoulder_length matching the depth coverage (leave-one-tree-out range):")
    for key, v in report["matching_length"].items():
        rng = (f"{v['leave_one_out_min']:.2f}-{v['leave_one_out_max']:.2f}"
               if v["leave_one_out_min"] is not None else "n/a")
        best = f"{v['all_trees']:.2f}" if v["all_trees"] is not None else "outside the measured range"
        print(f"  {key:12s} {best} ({rng})")
    print("Fill precision (pixels added that are on the depth silhouette) and fork-zone agreement:")
    for k in LENGTHS:
        m = report["masks"][str(k)]
        prec = f"{m['added_on_depth']:.2f}" if m["added_on_depth"] is not None else "  - "
        print(f"  shoulder_length {k:g}: added {m['added_px']:6d} px, {prec} on depth; "
              f"zone precision {m['zone_precision']:.3f}, IoU {m['zone_iou']:.3f}")
    dr = report["decay_ratio"]
    if dr["n_measured"]:
        print(f"Width decay from the fork (independent check): median {dr['median']:.2f} trunk radii over "
              f"{dr['n_measured']} leaders, bootstrap 95% CI {dr['bootstrap_95ci'][0]:.2f}-"
              f"{dr['bootstrap_95ci'][1]:.2f}, leave-one-tree-out {dr['leave_one_out'][0]:.2f}-"
              f"{dr['leave_one_out'][1]:.2f}")
    failed: list[str] = []
    if args.figure:
        shown = [r for r in results if r["tiles"] is not None]
        labels = ["no fill (before)"] + [f"shoulder_length {k:g}" for k in args.figure_lengths]
        save_image(_figure(shown, labels), Path(args.figure), failed, quality=90)
    print(f"Saved {out / 'leaders.csv'} and {out / 'report.json'}" + (f" and {args.figure}" if args.figure else ""))
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
