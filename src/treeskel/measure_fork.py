"""Measure the masks at the fork against the winter depth: the shoulder fill and the junction.

Drawn as discs, the trunk ends at the fork in a round top of its own radius,
and the much thinner leaders stick out of it. That left a notch (a
"shoulder") on each leader's outer side, and a round knob between and
beside the leaders. Two ways of drawing the fork are measured here, on
**development-row trees only**, against masks with the round top and no
fill (:data:`BASE`):

- the **shoulder fill** (:func:`treeskel.guided_gt.shoulder_fill`, session
  6; off since session 8): out to the straight line from the trunk's edge
  to the leader's edge ``shoulder_length`` trunk radii up the leader,
  keeping the round top;
- the **junction** (:func:`treeskel.guided_gt.fork_junction`, session 8):
  the trunk cut flat at the fork, and each leader joined to the trunk's
  edge on its outer side by a straight line ``junction_reach`` trunk radii
  up the leader; the crotch is left as the leaders draw it.

1. **Shoulder coverage.** For each leader, the triangle between the fork,
   the trunk's edge level with the fork and the leader's outer edge
   ``reach`` trunk radii up the leader (:func:`shoulder_triangle`). The
   fraction of it that is covered (:func:`coverage`) by the winter depth
   silhouette (the wood actually there) is compared with the fraction
   covered by each mask. The silhouette is also
   shrunk by 2 px, because depth "bleeds" past the edges of thin branches,
   so the true coverage lies between the two. The fill length and the
   junction reach whose mean coverage matches the depth's are found for
   each reach and reference, and checked by leave-one-tree-out.
2. **Precision.** The pixels each mask adds to or removes from the base,
   and the fraction of them on the depth silhouette, next to the precision
   and IoU of the whole mask against the silhouette around the fork
   (:func:`fork_zone`).
3. **The crotch.** Mask pixels between the leaders near the fork that are
   off the depth silhouette (:func:`crotch_zone`), and the height of the
   crotch above the fork (:func:`crotch_height`) against the depth
   silhouette's, raw and shrunk.
4. **Width decay (an independent check).** Along each leader, the depth
   silhouette's half-width falls from its value at the fork to the leader's
   own width. A smooth step fitted to it (:func:`fit_decay_length`) gives
   the length of that transition in trunk radii. It describes the whole
   junction: along the leader's centre line the crotch limits the width as
   much as the outer shoulder does.

Usage, from the command line:

```
python -m treeskel.measure_fork data/raw --clicks annotations/winter_clicks.json \\
    --split splits.json --out outputs/measure_fork --figure outputs/fork_junction.jpg
```
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
    end_cut,
    end_direction,
    fill_shoulders,
    fork_junction,
    guided_gt,
    load_clicks,
    outward_normal,
)
from treeskel.winter_gt import WinterGT, render_main_branches, report_failed, save_image

BASE = GuidedParams(shoulder_length=0, junction_reach=0, flat_root=False)
"""The masks the fill and the junction are measured against: the trunk's round top, no fill."""

LENGTHS = (0.0, 1.0, 1.4, 1.7, 2.0, 2.5)
"""``shoulder_length`` settings measured (0 = no fill, the base masks)."""

JUNCTION_REACHES = (0.5, 1.0, 1.5, 2.0, 2.5)
"""``junction_reach`` settings measured (the base masks have the round top instead)."""

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
    up = end_direction(gt.trunk, radius)
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


def crotch_zone(gt: WinterGT) -> np.ndarray:
    """The crotch: pixels between the two leaders' paths, above the fork and in its zone.

    The zone is :func:`fork_zone` (within 2.5 trunk radii of the fork and
    one trunk radius of a path). Mask pixels here that are off the depth
    silhouette are almost certainly not wood, because depth reaches past
    the bark's edges.
    """
    out = np.zeros(gt.tree_mask.shape, bool)
    radius = float(gt.radii["trunk"][-1])
    top = max(int(gt.fork[0] - 2.5 * radius), 0)
    for row in range(top, gt.fork[0]):
        lc, rc = gt.left[gt.left[:, 0] == row, 1], gt.right[gt.right[:, 0] == row, 1]
        if len(lc) and len(rc) and lc.max() + 1 < rc.min():
            out[row, lc.max() + 1:rc.min()] = True
    return out & fork_zone(gt)


def crotch_height(obj: np.ndarray, gt: WinterGT, zone: float = 2.5) -> float:
    """Height of the crotch above the fork, in trunk radii: where ``obj`` first has a gap between the leaders.

    Rows are scanned upwards from the fork; in each, the pixels strictly
    between the two leaders' paths are checked, and the first row where
    any of them is not in ``obj`` is the bottom of the crotch.

    Args:
        obj: Bool mask (a drawn mask or the depth silhouette).
        gt: The traced tree (its fork, leaders and trunk radius).
        zone: How far up to look, in trunk radii.

    Returns:
        The height divided by the trunk's radius at the fork, or nan if no
        gap is found within ``zone``.
    """
    radius = float(gt.radii["trunk"][-1])
    fy = gt.fork[0]
    for row in range(fy - 1, max(int(fy - zone * radius), 0) - 1, -1):
        lc, rc = gt.left[gt.left[:, 0] == row, 1], gt.right[gt.right[:, 0] == row, 1]
        if len(lc) and len(rc) and lc.max() + 1 < rc.min() and not obj[row, lc.max() + 1:rc.min()].all():
            return (fy - row) / radius
    return float("nan")


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
        """Mean squared error of a cosine decay over ``length`` px."""
        u = np.clip(x / length, 0, 1)
        return float(np.mean((y - far - (w0 - far) * 0.5 * (1 + np.cos(np.pi * u))) ** 2))

    return float(minimize_scalar(loss, bounds=(2.0, 8 * radius), method="bounded").x)


def _crop(image: np.ndarray, centre: Sequence[int], half: int) -> np.ndarray:
    r0 = int(np.clip(centre[0] - half, 0, image.shape[0] - 2 * half))
    c0 = int(np.clip(centre[1] - half, 0, image.shape[1] - 2 * half))
    return image[r0:r0 + 2 * half, c0:c0 + 2 * half]


def _paint(rgb: np.ndarray, mask: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Mask on the image: red fill, pixels added to ``base`` cyan, removed ones blue, outline yellow."""
    im = rgb.astype(float)
    im[mask] = 0.55 * im[mask] + 0.45 * np.array([255, 0, 60])
    added = mask & ~base
    im[added] = 0.4 * im[added] + 0.6 * np.array([0, 255, 255])
    removed = base & ~mask
    im[removed] = 0.5 * im[removed] + 0.5 * np.array([40, 80, 255])
    im[mask & ~ndi.binary_erosion(mask)] = (255, 255, 0)
    return im.astype(np.uint8)


def variant_label(name: str) -> str:
    """``"k1.4"`` → ``"fill 1.4"``, ``"j1.5"`` → ``"junction 1.5"``, ``"k0.0"`` → ``"round top"``."""
    value = float(name[1:])
    if name.startswith("k"):
        return f"fill {value:g}" if value > 0 else "round top"
    return f"junction {value:g}"


def fork_variants(gt: WinterGT, lengths: Sequence[float], reaches: Sequence[float]) -> dict[str, np.ndarray]:
    """The masks measured, by name: ``"k<length>"`` with the shoulder fill, ``"j<reach>"`` with the junction.

    Args:
        gt: A tree traced with :data:`BASE` (round top, no fill), so that
            ``"k0.0"`` is its own mask.
        lengths: ``shoulder_length`` settings.
        reaches: ``junction_reach`` settings.

    Returns:
        ``{name: bool mask}``, fill settings first.
    """
    base = render_main_branches(gt)
    out = {f"k{k}": base | fill_shoulders(gt, k) for k in lengths}
    top = [c for c in [end_cut(gt.trunk, gt.radii["trunk"][-1])] if c is not None]
    for r in reaches:
        out[f"j{r}"] = render_main_branches(replace(gt, trunk_cuts=top, junction=fork_junction(gt, r)))
    return out


def measure_tree(job: tuple) -> dict:
    """All measurements for one tree (top-level so worker processes can run it).

    Args:
        job: ``(tree_record, clicks, lengths, reaches, figure_names or None)``,
            where ``figure_names`` are the :func:`fork_variants` to crop for
            the figure, after the base.

    Returns:
        ``{"tree", "fork_radius", "leaders": [...], "masks": {...},
        "crotch_height": {...}, "tiles"}``.
    """
    from treeskel.io import load_tree

    rec, clicks, lengths, reaches, figure_names = job
    # Trace the tree once; draw the base mask and each fork variant (fill lengths, junction reaches) from it.
    d = load_tree(rec, "winter")
    gt = guided_gt(d["rgb"], d["depth"], d["valid"], clicks, BASE)
    base = render_main_branches(gt)
    masks = fork_variants(gt, lengths, reaches)
    depth = gt.tree_mask                                   # the depth silhouette (closed foreground)
    shrunk = ndi.binary_erosion(depth, disk(BLEED_PX))
    zone, crotch = fork_zone(gt), crotch_zone(gt)
    radius = float(gt.radii["trunk"][-1])
    dt = ndi.distance_transform_edt(depth)
    # Per leader: how far the depth width takes to fall to the leader's own, and how much of each shoulder
    # triangle depth (as seen and shrunk by its bleed) and each mask cover.
    leaders = []
    for side in ("left", "right"):
        path = getattr(gt, side)
        row = {"side": side, "decay_ratio": fit_decay_length(arc_length(path), dt[path[:, 0], path[:, 1]],
                                                              radius) / radius}
        for reach in REACHES:
            tri = shoulder_triangle(gt, side, reach)
            row[f"depth@{reach}"] = coverage(depth, tri)
            row[f"shrunk@{reach}"] = coverage(shrunk, tri)
            for name, m in masks.items():
                row[f"{name}@{reach}"] = coverage(m, tri)
        leaders.append(row)
    # Per mask: the pixels it adds and removes against the base mask and whether they lie on depth, its
    # precision and IoU against depth around the fork, and what it draws in the crotch.
    stats = {}
    for name, m in masks.items():
        added, removed = m & ~base, base & ~m
        both = (m & depth & zone).sum()
        stats[name] = {"added": int(added.sum()), "added_on_depth": int((added & depth).sum()),
                       "added_on_shrunk": int((added & shrunk).sum()),
                       "removed": int(removed.sum()), "removed_on_depth": int((removed & depth).sum()),
                       "removed_on_shrunk": int((removed & shrunk).sum()),
                       "zone_precision": both / max((m & zone).sum(), 1),
                       "zone_iou": both / max(((m | depth) & zone).sum(), 1),
                       "crotch_off": int((m & crotch & ~depth).sum()), "crotch_px": int((m & crotch).sum()),
                       "crotch_height": crotch_height(m, gt)}
    tiles = None
    if figure_names is not None:
        half = 40
        tiles = [_crop(d["rgb"], gt.fork, half), _crop(_paint(d["rgb"], base, base), gt.fork, half)] + [
            _crop(_paint(d["rgb"], masks[name], base), gt.fork, half) for name in figure_names]
    return {"tree": rec.tree_id, "fork_radius": radius, "leaders": leaders, "masks": stats,
            "crotch_height": {"depth": crotch_height(depth, gt), "shrunk": crotch_height(shrunk, gt)},
            "tiles": tiles}


def matching_length(results: list[dict], lengths: Sequence[float], reach: float, ref: str,
                    exclude: str | None = None, prefix: str = "k") -> float:
    """Setting at which the masks' mean coverage equals the reference's.

    Linear interpolation of mean coverage against the measured settings
    (coverage only grows with them). nan if the reference lies outside
    the measured range.

    Args:
        results: :func:`measure_tree` results.
        lengths: The measured settings, increasing.
        reach: Which shoulder triangle.
        ref: ``"depth"`` or ``"shrunk"``.
        exclude: A tree id to leave out.
        prefix: ``"k"`` for ``shoulder_length``, ``"j"`` for ``junction_reach``
            (see :func:`fork_variants`).
    """
    rows = [row for r in results if r["tree"] != exclude for row in r["leaders"]]
    target = np.nanmean([row[f"{ref}@{reach}"] for row in rows])
    cov = np.array([np.nanmean([row[f"{prefix}{k}@{reach}"] for row in rows]) for k in lengths])
    if not cov[0] <= target <= cov[-1]:
        return float("nan")
    return float(np.interp(target, cov, lengths))


def _or_none(x: float) -> float | None:
    """``x``, or None if it is nan (JSON has no nan)."""
    return None if np.isnan(x) else float(x)


def _median(values: Sequence[float]) -> float | None:
    """Median ignoring nan, or None if nothing is left."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else None


def summarise(results: list[dict], lengths: Sequence[float], reaches: Sequence[float] = (), seed: int = 0) -> dict:
    """Pool :func:`measure_tree` results into the report."""
    rows = [row for r in results for row in r["leaders"]]
    names = [f"k{k}" for k in lengths] + [f"j{r}" for r in reaches]
    out = {"trees": [r["tree"] for r in results], "n_leaders": len(rows), "lengths": list(lengths),
           "junction_reaches": list(reaches), "reaches": list(REACHES), "bleed_px": BLEED_PX, "coverage": {},
           "matching_length": {}, "matching_reach": {}, "masks": {}}
    for reach in REACHES:
        cov = {ref: float(np.nanmean([row[f"{ref}@{reach}"] for row in rows])) for ref in ("depth", "shrunk")}
        cov.update({name: float(np.nanmean([row[f"{name}@{reach}"] for row in rows])) for name in names})
        # Share of leaders whose shoulder is clearly notched (less than 80% covered).
        cov["notched"] = {name: float(np.mean([row[f"{name}@{reach}"] < 0.8 for row in rows]))
                          for name in ("depth", "shrunk", *names)}
        out["coverage"][str(reach)] = cov
        for ref in ("depth", "shrunk"):
            for key, settings, prefix in (("matching_length", lengths, "k"), ("matching_reach", reaches, "j")):
                if len(settings) < 2:
                    continue
                loo = np.array([matching_length(results, settings, reach, ref, exclude=r["tree"], prefix=prefix)
                                for r in results])
                found = loo[~np.isnan(loo)]
                out.setdefault(key, {})[f"{ref}@{reach}"] = {
                    "all_trees": _or_none(matching_length(results, settings, reach, ref, prefix=prefix)),
                    "leave_one_out_min": float(found.min()) if len(found) else None,
                    "leave_one_out_max": float(found.max()) if len(found) else None}
    ref_heights = {ref: np.array([r["crotch_height"][ref] for r in results], float) for ref in ("depth", "shrunk")}
    out["crotch_height"] = {ref: _median(h) for ref, h in ref_heights.items()}
    for name in names:
        total = {k: sum(r["masks"][name][k] for r in results)
                 for k in ("added", "added_on_depth", "added_on_shrunk", "removed", "removed_on_depth",
                           "removed_on_shrunk", "crotch_off", "crotch_px")}
        heights = np.array([r["masks"][name]["crotch_height"] for r in results], float)
        out["masks"][name] = {
            "added_px": total["added"],
            "added_on_depth": total["added_on_depth"] / total["added"] if total["added"] else None,
            "added_on_shrunk": total["added_on_shrunk"] / total["added"] if total["added"] else None,
            "removed_px": total["removed"],
            "removed_on_depth": total["removed_on_depth"] / total["removed"] if total["removed"] else None,
            "removed_on_shrunk": total["removed_on_shrunk"] / total["removed"] if total["removed"] else None,
            "zone_precision": float(np.mean([r["masks"][name]["zone_precision"] for r in results])),
            "zone_iou": float(np.mean([r["masks"][name]["zone_iou"] for r in results])),
            "crotch_off_px": total["crotch_off"], "crotch_px": total["crotch_px"],
            "crotch_height": _median(heights),
            # Paired by tree: how much higher the mask's crotch is than the silhouette's (trunk radii).
            "crotch_height_minus_depth": _median(heights - ref_heights["depth"]),
            "crotch_height_minus_shrunk": _median(heights - ref_heights["shrunk"])}
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


def _num(x: float | None, spec: str = ".2f", missing: str = "n/a") -> str:
    """``x`` formatted with ``spec``, or ``missing`` if it is None."""
    return format(x, spec) if x is not None else missing


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
    ap.add_argument("--figure-lengths", type=float, nargs="*", default=[],
                    help="shoulder_length settings shown in the figure, after the round top (default: none; "
                         "1.4 was the session-6 choice)")
    ap.add_argument("--figure-reaches", type=float, nargs="*", default=[1.0],
                    help="junction_reach settings shown in the figure, after the fills (default: 1.0, "
                         "the session-8 choice)")
    args = ap.parse_args(argv)
    bad = [v for v in args.figure_lengths if v not in LENGTHS] + [v for v in args.figure_reaches
                                                                   if v not in JUNCTION_REACHES]
    if bad:
        ap.error(f"figure settings must be among the measured ones ({LENGTHS}, {JUNCTION_REACHES}): {bad}")

    # Every annotated development-row tree; the test rows are never measured.
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
    fig_names = [f"k{float(k)}" for k in args.figure_lengths] + [f"j{float(r)}" for r in args.figure_reaches]
    jobs = [(records[t], clicks[t], LENGTHS, JUNCTION_REACHES, fig_names if t in fig_ids else None) for t in ids]

    print(f"Measuring {len(ids)} development-row trees ({len(held_out)} test-row trees left out)...", flush=True)
    t0 = time.time()
    step = max(1, min(25, len(jobs) // 10))
    results = []
    # One tree per worker process (tracing is the slow part), with progress about every 10% of the trees.
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
        for i, r in enumerate(ex.map(measure_tree, jobs, chunksize=2), 1):
            results.append(r)
            if i % step == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                      flush=True)

    # Pool the trees, then save the per-leader table and the report before printing it.
    report = summarise(results, LENGTHS, JUNCTION_REACHES)
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

    names = [f"k{k}" for k in LENGTHS] + [f"j{r}" for r in JUNCTION_REACHES]
    print(f"\n{report['n_leaders']} leaders on {len(ids)} trees. Shoulder coverage (mean fraction of the triangle):")
    for reach in REACHES:
        c = report["coverage"][str(reach)]
        fills = ", ".join(f"{k:g}: {c[f'k{k}']:.3f}" for k in LENGTHS)
        joins = ", ".join(f"{r:g}: {c[f'j{r}']:.3f}" for r in JUNCTION_REACHES)
        print(f"  reach {reach:g} R: depth {c['depth']:.3f}, depth shrunk {BLEED_PX} px {c['shrunk']:.3f} | "
              f"masks by shoulder_length {fills} | by junction_reach {joins}")
        notched = ", ".join(f"{k}: {v:.0%}" for k, v in c["notched"].items())
        print(f"    leaders with less than 80% covered: {notched}")
    for key, setting in (("matching_length", "shoulder_length"), ("matching_reach", "junction_reach")):
        print(f"{setting} matching the depth coverage (leave-one-tree-out range):")
        for ref, v in report[key].items():
            rng = (f"{v['leave_one_out_min']:.2f}-{v['leave_one_out_max']:.2f}"
                   if v["leave_one_out_min"] is not None else "n/a")
            best = f"{v['all_trees']:.2f}" if v["all_trees"] is not None else "outside the measured range"
            print(f"  {ref:12s} {best} ({rng})")
    print(f"Pixels added to and removed from the round-top masks (share on depth / on depth shrunk {BLEED_PX} px), "
          "and fork-zone agreement:")
    for name in names:
        m = report["masks"][name]
        print(f"  {variant_label(name):13s} added {m['added_px']:6d} px ({_num(m['added_on_depth'], missing=' -  ')} / "
              f"{_num(m['added_on_shrunk'], missing=' -  ')}), removed {m['removed_px']:6d} px ({_num(m['removed_on_depth'], missing=' -  ')} / "
              f"{_num(m['removed_on_shrunk'], missing=' -  ')}); zone precision {m['zone_precision']:.3f}, IoU {m['zone_iou']:.3f}")
    ch = report["crotch_height"]
    print(f"The crotch: mask pixels between the leaders off the depth silhouette (off / all), and its height "
          f"above the fork in trunk radii (median; depth {_num(ch['depth'])}, depth shrunk {BLEED_PX} px "
          f"{_num(ch['shrunk'])}), with the difference from each, paired by tree:")
    for name in names:
        m = report["masks"][name]
        print(f"  {variant_label(name):13s} off {m['crotch_off_px']:6d} / {m['crotch_px']:6d}; height "
              f"{_num(m['crotch_height'])} ({_num(m['crotch_height_minus_depth'], '+.2f')}, "
              f"{_num(m['crotch_height_minus_shrunk'], '+.2f')})")
    dr = report["decay_ratio"]
    if dr["n_measured"]:
        print(f"Width decay from the fork (independent check): median {dr['median']:.2f} trunk radii over "
              f"{dr['n_measured']} leaders, bootstrap 95% CI {dr['bootstrap_95ci'][0]:.2f}-"
              f"{dr['bootstrap_95ci'][1]:.2f}, leave-one-tree-out {dr['leave_one_out'][0]:.2f}-"
              f"{dr['leave_one_out'][1]:.2f}")
    failed: list[str] = []
    if args.figure:
        shown = [r for r in results if r["tiles"] is not None]
        labels = ["round top"] + [variant_label(n) for n in fig_names]
        save_image(_figure(shown, labels), Path(args.figure), failed, quality=90)
    print(f"Saved {out / 'leaders.csv'} and {out / 'report.json'}" + (f" and {args.figure}" if args.figure else ""))
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
