"""Summer ground truth: the winter masks moved into the summer view with clicked point pairs.

The winter masks (:mod:`treeskel.guided_gt`) show where the trunk and leaders
are, but in the winter image, and the summer model needs them in the summer
image of the same tree. Automatic matching between the seasons failed (see
``docs/results/align/``): SIFT found only chance matches, and the learned
matcher DISK+LightGlue gave a plausible fit on about 1% of the trees, because
in summer the leaves hide most of what the two images have in common. So the
correspondence is clicked by hand (:mod:`treeskel.annotate_pairs`): for each
tree, a few points that are clearly the same physical point in both images,
on the plane of the tree's row (trunk features, posts, where the drip line
crosses them).

From the pairs:

1. **Transform** (:func:`pair_transform`): a least-squares fit, by default of
   a similarity (scale, rotation and shift), because the tree lies close to
   one plane that roughly faces the camera in both seasons (see
   :mod:`treeskel.align`). The translation, similarity and affine models are
   compared by leave-one-out error on the development rows (:func:`main`).
2. **Error** (:func:`loo_errors`): leave-one-out. Each pair in turn is left
   out, the transform is fitted to the others, and the error is the distance
   between where that transform sends the left-out winter point and where the
   point was clicked in summer. It measures click noise, how well the model
   fits and whether the points really lie on one plane, on points the fit has
   not seen.
3. **Warp** (:func:`treeskel.align.warp_mask`): the winter mask is moved into
   the summer view, together with the area the winter image covers; outside
   it the label is unknown, not "no branch".

Usage::

    python -m treeskel.summer_gt data/raw --dev-only          # choose the model
    python -m treeskel.summer_gt data/raw --overlays          # write the summer masks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from treeskel.align import (
    MIN_SAMPLES,
    apply_transform,
    contact_sheet,
    describe,
    fit_transform,
    load_winter_mask,
    outline,
    warp_mask,
)
from treeskel.winter_gt import report_failed, save_image

PAIR_MODELS = ("translation", "similarity", "affine")
"""Models compared on the clicked pairs. A homography (8 parameters) would
need at least 5 pairs per tree for a leave-one-out check, so it is not used."""

Point = tuple[float, float]


@dataclass
class SeasonPairs:
    """Clicked winter-summer point pairs for one tree, as ``(row, col)`` pixels.

    Attributes:
        winter: Points in the winter image.
        summer: The same physical points in the summer image, in the same order.
        skip: True if the tree could not be aligned (it then gets no summer mask).
        note: Free-text comment.
    """

    winter: list[Point] = field(default_factory=list)
    summer: list[Point] = field(default_factory=list)
    skip: bool = False
    note: str = ""

    def __post_init__(self):
        if len(self.winter) != len(self.summer):
            raise ValueError("winter and summer need the same number of points")

    def __len__(self) -> int:
        return len(self.winter)

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """``(winter, summer)`` as ``(N, 2)`` ``(x, y)`` arrays, the order :mod:`treeskel.align` uses."""
        w = np.asarray(self.winter, float).reshape(-1, 2)[:, ::-1]
        s = np.asarray(self.summer, float).reshape(-1, 2)[:, ::-1]
        return w, s

    def to_json(self) -> dict:
        """Plain-data form for the JSON file (coordinates to 0.1 px)."""
        d = asdict(self)
        for k in ("winter", "summer"):
            d[k] = [[round(float(r), 1), round(float(c), 1)] for r, c in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SeasonPairs":
        """Inverse of :meth:`to_json`."""
        return cls(winter=[tuple(p) for p in d.get("winter", [])], summer=[tuple(p) for p in d.get("summer", [])],
                   skip=bool(d.get("skip", False)), note=str(d.get("note", "")))


def load_pairs(path: str | Path) -> dict[str, SeasonPairs]:
    """Read a pairs file; a missing file gives an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    return {k: SeasonPairs.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_pairs(path: str | Path, pairs: dict[str, SeasonPairs]) -> None:
    """Write a pairs file atomically (a crash never leaves it half-written)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: pairs[k].to_json() for k in sorted(pairs)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def pair_transform(pairs: SeasonPairs, model: str = "similarity") -> np.ndarray | None:
    """Least-squares winter-to-summer transform (3x3, ``(x, y)``) from the pairs.

    Returns ``None`` if there are too few pairs for the model or they are
    degenerate (see :func:`treeskel.align.fit_transform`).

    Example:
        >>> p = SeasonPairs(winter=[(100, 200), (300, 250)], summer=[(90, 215), (290, 265)])
        >>> m = pair_transform(p, "translation")
        >>> [round(v) for v in m[:2, 2]]          # (dx, dy): 15 px right, 10 px up
        [15, -10]
    """
    w, s = pairs.xy()
    return fit_transform(w, s, model)


def loo_errors(pairs: SeasonPairs, model: str = "similarity") -> np.ndarray:
    """Leave-one-out error (px) of each pair.

    For pair ``i``, the transform is fitted to all the other pairs, and the
    error is the distance between the transformed winter point ``i`` and the
    summer point ``i``. A wrong click shows up as a large error on its own
    pair (and smaller rises on the others, whose fits it takes part in).

    Returns:
        ``(N,)`` errors; ``nan`` for a pair whose fit without it is impossible
        (fewer than ``MIN_SAMPLES[model] + 1`` pairs in all, or degenerate).
    """
    w, s = pairs.xy()
    n = len(w)
    out = np.full(n, np.nan)
    if n < MIN_SAMPLES[model] + 1:
        return out
    for i in range(n):
        keep = np.arange(n) != i
        m = fit_transform(w[keep], s[keep], model)
        if m is not None:
            out[i] = float(np.hypot(*(apply_transform(m, w[i:i + 1])[0] - s[i])))
    return out


@dataclass(frozen=True)
class PairLimits:
    """Limits used by :func:`check_pairs`.

    Attributes:
        min_pairs: Fewest pairs for a checked fit (2 fix a similarity exactly,
            so a third is needed to check it).
        max_loo: Largest acceptable leave-one-out error of any pair (px).
        scale_range: Plausible scale between the seasons.
        max_rotation: Largest plausible rotation (degrees).
    """

    min_pairs: int = 3
    max_loo: float = 10.0
    scale_range: tuple[float, float] = (0.8, 1.25)
    max_rotation: float = 8.0


def check_pairs(pairs: SeasonPairs, model: str = "similarity", limits: PairLimits = PairLimits()) -> list[str]:
    """Reasons to look at a tree's pairs again (empty if none, or if skipped).

    Flags too few pairs, a pair far from the fit of the others (probably a
    wrong click, or a point off the tree's plane) and an implausible scale or
    rotation.
    """
    if pairs.skip:
        return []
    issues = []
    if len(pairs) < limits.min_pairs:
        issues.append(f"only {len(pairs)} pair(s)")
    m = pair_transform(pairs, model)
    if m is None:
        return issues + ["no transform could be fitted"]
    loo = loo_errors(pairs, model)
    if np.isfinite(loo).any() and np.nanmax(loo) > limits.max_loo:
        issues.append(f"pair {int(np.nanargmax(loo)) + 1} is {np.nanmax(loo):.0f} px from the fit of the others")
    d = describe(m, (320, 240))
    lo, hi = limits.scale_range
    if not lo <= d["scale"] <= hi:
        issues.append(f"scale {d['scale']:.2f}")
    if abs(d["rotation"]) > limits.max_rotation:
        issues.append(f"rotation {d['rotation']:.1f} deg")
    return issues


def loo_table(pairs: dict[str, SeasonPairs], models: Sequence[str] = PAIR_MODELS) -> list[dict]:
    """Leave-one-out errors of each model, pooled over trees, for choosing the model.

    Only trees with enough pairs for every model's leave-one-out check are
    used (``max(MIN_SAMPLES) + 1`` pairs), so all models are compared on the
    same points.

    Returns:
        One dict per model: ``model``, ``trees``, ``pairs``, and the median,
        mean and 90th percentile of the errors (px) and the fraction above 5
        and 10 px.
    """
    need = max(MIN_SAMPLES[m] for m in models) + 1
    use = {t: p for t, p in pairs.items() if not p.skip and len(p) >= need}
    rows = []
    for m in models:
        e = np.concatenate([loo_errors(p, m) for p in use.values()]) if use else np.zeros(0)
        e = e[np.isfinite(e)]
        rows.append({"model": m, "trees": len(use), "pairs": len(e),
                     "median": float(np.median(e)) if len(e) else float("nan"),
                     "mean": float(e.mean()) if len(e) else float("nan"),
                     "p90": float(np.percentile(e, 90)) if len(e) else float("nan"),
                     "over_5": float((e > 5).mean()) if len(e) else float("nan"),
                     "over_10": float((e > 10).mean()) if len(e) else float("nan")})
    return rows


def draw_summer_gt(rgb_summer: np.ndarray, warped: np.ndarray, pairs: SeasonPairs, loo: np.ndarray,
                   label: str = "") -> Image.Image:
    """The summer image with the warped mask outlined (magenta) and the clicked summer points.

    Each point is numbered with its leave-one-out error; red if above 10 px.
    """
    im = rgb_summer.copy()
    im[outline(warped, 3)] = (255, 0, 255)
    pil = Image.fromarray(im)
    dr = ImageDraw.Draw(pil)
    for i, (r, c) in enumerate(pairs.summer):
        bad = np.isfinite(loo[i]) and loo[i] > 10
        col = (255, 60, 60) if bad else (0, 255, 255)
        dr.ellipse([c - 6, r - 6, c + 6, r + 6], outline=col, width=2)
        txt = f"{i + 1}" + (f": {loo[i]:.1f} px" if np.isfinite(loo[i]) else "")
        dr.text((c + 8, r - 14), txt, fill=col)
    dr.text((8, 8), label, fill="yellow")
    return pil


def main(argv: Sequence[str] | None = None) -> int:
    """Check the clicked pairs, compare the models, and write the summer masks.

    Usage: ``python -m treeskel.summer_gt DATA_DIR [--pairs annotations/season_pairs.json]
    [--model similarity] [--dev-only] [--rows 5 6] [--trees R05N12] [--overlays]
    [--out outputs/summer_gt]``.

    Prints the leave-one-out table of every model (choose the model on the
    development rows only: ``--dev-only``) and the trees whose pairs should be
    looked at again. ``summary.csv`` gives each tree's scale, rotation and
    shift at its fork click. For every aligned tree it writes
    ``<tree>_summer_gt.png`` (the winter mask in the summer view) and
    ``<tree>_summer_covered.png`` (white where the winter image covers the
    summer view), plus ``summary.csv`` and contact sheets of the summer images
    with the masks; ``--overlays`` also saves a full-size picture per tree.
    """
    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset, load_tree
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.summer_gt", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--pairs", default="annotations/season_pairs.json")
    ap.add_argument("--model", choices=PAIR_MODELS, default="similarity")
    ap.add_argument("--rows", type=int, nargs="+", help="only trees in these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true", help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--winter-gt", default="outputs/winter_gt_guided",
                    help="folder of <tree>_winter_gt.png masks; traced from the clicks if missing")
    ap.add_argument("--out", default="outputs/summer_gt")
    ap.add_argument("--overlays", action="store_true", help="also save a full-size picture per tree")
    args = ap.parse_args(argv)

    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    pairs = load_pairs(args.pairs)
    skipped = [t.tree_id for t in trees if t.tree_id in pairs and pairs[t.tree_id].skip]
    trees = [t for t in trees if t.tree_id in pairs and not pairs[t.tree_id].skip]
    if not trees:
        print(f"No clicked pairs in {args.pairs} for the selected trees.")
        return 1
    chosen = {t.tree_id: pairs[t.tree_id] for t in trees}

    table = loo_table(chosen)
    print(f"{len(trees)} trees with pairs ({sum(len(p) for p in chosen.values())} pairs; "
          f"{len(skipped)} skipped as unalignable).")
    print(f"Leave-one-out error (px), on the {table[0]['trees']} trees with at least "
          f"{max(MIN_SAMPLES[m] for m in PAIR_MODELS) + 1} pairs:")
    print(f"{'model':<12}{'median':>8}{'mean':>8}{'p90':>8}{'> 5 px':>9}{'> 10 px':>9}")
    for r in table:
        print(f"{r['model']:<12}{r['median']:>8.2f}{r['mean']:>8.2f}{r['p90']:>8.2f}"
              f"{100 * r['over_5']:>8.0f}%{100 * r['over_10']:>8.0f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clicks = load_clicks(args.clicks)
    gt_dir = Path(args.winter_gt) if args.winter_gt else None
    rows, tiles, failed, flagged = [], [], [], {}
    t0 = time.time()
    step = max(1, min(25, len(trees) // 10))
    print(f"Writing the summer masks ({args.model})...", flush=True)
    for i, rec in enumerate(trees, 1):
        p = chosen[rec.tree_id]
        issues = check_pairs(p, args.model)
        if issues:
            flagged[rec.tree_id] = issues
        m = pair_transform(p, args.model)
        loo = loo_errors(p, args.model)
        row = {"tree_id": rec.tree_id, "pairs": len(p), "ok": not issues, "issues": "; ".join(issues),
               "loo_median": round(float(np.nanmedian(loo)), 2) if np.isfinite(loo).any() else "",
               "loo_max": round(float(np.nanmax(loo)), 2) if np.isfinite(loo).any() else ""}
        if m is not None:
            c = clicks.get(rec.tree_id)
            at = c.fork[::-1] if c is not None and c.complete and not c.skip else (320, 240)
            row.update({k: round(v, 4) for k, v in describe(m, at).items()})      # shift at the fork
            row["matrix"] = json.dumps(np.round(m, 6).tolist())
            winter = load_tree(rec, "winter")
            mask = load_winter_mask(rec.tree_id, winter, clicks, gt_dir)
            if mask is not None:
                summer_rgb = load_tree(rec, "summer")["rgb"]
                warped, covered = warp_mask(mask, m, summer_rgb.shape[:2])
                save_image(Image.fromarray(warped.astype(np.uint8) * 255), out / f"{rec.tree_id}_summer_gt.png",
                           failed)
                save_image(Image.fromarray(covered.astype(np.uint8) * 255),
                           out / f"{rec.tree_id}_summer_covered.png", failed)
                flag = "  FLAG" if issues else ""
                pic = draw_summer_gt(summer_rgb, warped, p, loo, f"{rec.tree_id} {args.model}{flag}")
                if args.overlays:
                    save_image(pic, out / f"{rec.tree_id}_overlay.jpg", failed, quality=85)
                tiles.append(np.asarray(pic.resize((320, 240))))
        rows.append(row)
        if i % step == 0 or i == len(trees):
            el = time.time() - t0
            print(f"  {i}/{len(trees)} trees, {el:.0f} s elapsed, about {el / i * (len(trees) - i):.0f} s left",
                  flush=True)

    try:
        with open(out / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            wr.writeheader()
            wr.writerows(rows)
    except OSError as e:
        failed.append(f"{out / 'summary.csv'}: {e.strerror or e}")
    for i in range(0, len(tiles), 24):
        save_image(contact_sheet(tiles[i:i + 24]), out / f"sheet_{i // 24 + 1:02d}.jpg", failed, quality=85)
    if flagged:
        print(f"\nLook at these trees' pairs again ({len(flagged)}; python -m treeskel.annotate_pairs DATA "
              "--redo --trees ...):")
        for tid, why in sorted(flagged.items()):
            print(f"  {tid}: {'; '.join(why)}")
    print(f"\nResults in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
