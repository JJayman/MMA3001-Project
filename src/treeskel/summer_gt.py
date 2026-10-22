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

1. **Transform** (:func:`pair_transform`): a least-squares fit. The tree
   lies close to one plane, and the camera stood at a different height,
   distance and angle in each season. For a plane, every such change maps
   one image to the other by a homography (8 parameters). A change of height
   or a sideways move shifts the tree, a change of distance scales it and
   roll rotates it: a similarity (4 parameters) covers these. Pitch and yaw
   add perspective, which only the homography models exactly. More
   parameters need more pairs and follow click errors more, so the models
   (translation, similarity, affine, homography) are compared by
   leave-one-out error on the development rows (:func:`main`).
2. **Error** (:func:`loo_errors`): leave-one-out. Each pair in turn is left
   out, the transform is fitted to the others, and the error is the distance
   between where that transform sends the left-out winter point and where the
   point was clicked in summer. It measures click noise, how well the model
   fits and whether the points really lie on one plane, on points the fit has
   not seen.
3. **Warp** (:func:`treeskel.align.warp_mask`): the winter mask is moved into
   the summer view, together with the area the winter image covers; outside
   it the label is unknown, not "no branch".

Usage, from the command line:

```
python -m treeskel.summer_gt data/raw --dev-only          # choose the model
python -m treeskel.summer_gt data/raw --overlays          # write the summer masks
```
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
    keeps_orientation,
    load_winter_mask,
    outline,
    warp_mask,
)
from treeskel.winter_gt import report_failed, save_image

PAIR_MODELS = ("translation", "similarity", "affine", "homography")
"""Models compared on the clicked pairs, from the fewest parameters to the
most. A leave-one-out check needs one pair more than the model's minimum
(:data:`treeskel.align.MIN_SAMPLES`): 3 pairs for a similarity, 4 for an
affine transform and 5 for a homography."""

IMAGE_CORNERS = np.array([[0, 0], [639, 0], [0, 479], [639, 479]], float)
"""Corners of a 640x480 image, ``(x, y)``."""

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
        """``(winter, summer)`` as ``(N, 2)`` ``(x, y)`` arrays, the order :mod:`treeskel.align` uses.

        Returns:
            ``(winter, summer)``, each ``(N, 2)`` float.
        """
        w = np.asarray(self.winter, float).reshape(-1, 2)[:, ::-1]
        s = np.asarray(self.summer, float).reshape(-1, 2)[:, ::-1]
        return w, s

    def to_json(self) -> dict:
        """Plain-data form for the JSON file (coordinates to 0.1 px).

        Returns:
            ``{"winter", "summer", "skip", "note"}`` with the points as lists.
        """
        d = asdict(self)
        for k in ("winter", "summer"):
            d[k] = [[round(float(r), 1), round(float(c), 1)] for r, c in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "SeasonPairs":
        """Inverse of :meth:`to_json`.

        Args:
            d: A dict as written by :meth:`to_json` (missing keys default to empty).

        Returns:
            The :class:`SeasonPairs`.
        """
        return cls(winter=[tuple(p) for p in d.get("winter", [])], summer=[tuple(p) for p in d.get("summer", [])],
                   skip=bool(d.get("skip", False)), note=str(d.get("note", "")))


def load_pairs(path: str | Path) -> dict[str, SeasonPairs]:
    """Read a pairs file; a missing file gives an empty dict.

    Args:
        path: The JSON file (``annotations/season_pairs.json``).

    Returns:
        The pairs per tree id.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return {k: SeasonPairs.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_pairs(path: str | Path, pairs: dict[str, SeasonPairs]) -> None:
    """Write a pairs file atomically (a crash never leaves it half-written).

    Args:
        path: The JSON file to write.
        pairs: The pairs per tree id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: pairs[k].to_json() for k in sorted(pairs)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def pair_transform(pairs: SeasonPairs, model: str = "similarity") -> np.ndarray | None:
    """Least-squares winter-to-summer transform (3x3, ``(x, y)``) from the pairs.

    Returns ``None`` if there are too few pairs for the model, they are
    degenerate (see :func:`treeskel.align.fit_transform`), or the fitted
    homography would turn part of the image inside out (:func:`_fit`).

    Args:
        pairs: The tree's clicked pairs.
        model: One of :data:`treeskel.align.MODELS` (``"translation"``, ``"similarity"``, ``"affine"``,
            ``"homography"``).

    Returns:
        The 3x3 transform, or None.

    Example:
        >>> p = SeasonPairs(winter=[(100, 200), (300, 250)], summer=[(90, 215), (290, 265)])
        >>> m = pair_transform(p, "translation")
        >>> [round(v) for v in m[:2, 2]]          # (dx, dy): 15 px right, 10 px up
        [15, -10]
    """
    w, s = pairs.xy()
    return _fit(w, s, model)


def _fit(w: np.ndarray, s: np.ndarray, model: str) -> np.ndarray | None:
    """:func:`treeskel.align.fit_transform`, rejecting homographies that fold the image.

    A homography fitted to a few noisy clicks can put the line where its
    perspective denominator is zero inside the image, sending part of it to
    infinity. No camera move does that, so such a fit counts as failed.
    """
    m = fit_transform(w, s, model)
    if m is not None and not keeps_orientation(m, IMAGE_CORNERS):
        return None
    return m


def loo_errors(pairs: SeasonPairs, model: str = "similarity") -> np.ndarray:
    """Leave-one-out error (px) of each pair.

    For pair ``i``, the transform is fitted to all the other pairs, and the
    error is the distance between the transformed winter point ``i`` and the
    summer point ``i``. A wrong click shows up as a large error on its own
    pair (and smaller rises on the others, whose fits it takes part in).

    Args:
        pairs: The tree's clicked pairs.
        model: One of :data:`treeskel.align.MODELS` (``"translation"``, ``"similarity"``, ``"affine"``,
            ``"homography"``).

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
        m = _fit(w[keep], s[keep], model)
        if m is not None:
            out[i] = float(np.hypot(*(apply_transform(m, w[i:i + 1])[0] - s[i])))
    return out


@dataclass(frozen=True)
class PairLimits:
    """Limits used by :func:`check_pairs`.

    Attributes:
        min_pairs: Fewest pairs for a checked fit (2 fix a similarity exactly,
            so a third is needed to check it). Models with more parameters
            need more: one more than :data:`treeskel.align.MIN_SAMPLES`.
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

    Args:
        pairs: The tree's clicked pairs.
        model: One of :data:`treeskel.align.MODELS` (``"translation"``, ``"similarity"``, ``"affine"``,
            ``"homography"``).
        limits: The thresholds of the checks.

    Returns:
        Human-readable reasons; empty if the pairs pass or the tree is skipped.
    """
    if pairs.skip:
        return []
    issues = []
    if len(pairs) < max(limits.min_pairs, MIN_SAMPLES[model] + 1):
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

    Args:
        pairs: The pairs per tree id.
        models: The models compared.

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


FOCAL_PX = 615.0
"""Assumed focal length (px) of the D435 colour camera at 640x480. It only
sets how a tilt in degrees turns into perspective in :meth:`Overlay.matrix`;
a different value would give the same transforms with different angles."""


@dataclass
class Overlay:
    """A winter-to-summer transform set by hand in :mod:`treeskel.annotate_overlay`.

    It is built from the camera changes it stands for, so each number means
    something. A change of height or a sideways move shifts the tree
    (``dx``, ``dy``), a change of distance scales it (``scale``), and roll
    rotates it (``roll``). Pitch and yaw also add perspective (``pitch``,
    ``yaw``): the rotation of a pinhole camera, ``K R K⁻¹``, with the image
    centre held in place so that tilting does not also move the picture.
    With both tilts at 0 this is a similarity; with them, a homography with
    6 of its 8 degrees of freedom.

    Attributes:
        dx: Rightward shift (px) of the image centre, winter to summer.
        dy: Downward shift (px) of the image centre, winter to summer.
        scale: Size of the tree in summer relative to winter.
        roll: Rotation about the image centre (degrees; positive is clockwise).
        pitch: Perspective tilt (degrees) about the image's horizontal axis.
        yaw: Perspective tilt (degrees) about the image's vertical axis.
        skip: The tree could not be aligned (it then gets no summer mask).
        mask_flag: The winter mask looked wrong while aligning, to fix later.
        note: Free-text comment.
        seconds: Time spent aligning the tree.
    """

    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    skip: bool = False
    mask_flag: bool = False
    note: str = ""
    seconds: float = 0.0

    PARAMS = ("dx", "dy", "scale", "roll", "pitch", "yaw")

    def matrix(self, focal: float = FOCAL_PX, centre: tuple[float, float] = (320.0, 240.0)) -> np.ndarray:
        """The 3x3 winter-to-summer ``(x, y)`` transform.

        Args:
            focal: Focal length (px) of the pinhole camera for the tilts.
            centre: The image centre ``(x, y)``, held in place.

        Returns:
            The 3x3 transform, scaled so its last entry is 1.

        Example:
            >>> m = Overlay(dx=12, dy=-30).matrix()
            >>> [round(float(v), 6) for v in apply_transform(m, [(100, 200)])[0]]
            [112.0, 170.0]
        """
        cx, cy = centre
        k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])
        p, y, r = np.radians([self.pitch, self.yaw, self.roll])
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1.0]])
        h = k @ rz @ rx @ ry @ np.linalg.inv(k)
        c = apply_transform(h, [centre])[0]                  # a tilt also moves the centre: undo that
        h = np.array([[1, 0, cx - c[0]], [0, 1, cy - c[1]], [0, 0, 1.0]]) @ h
        a = np.array([[self.scale, 0, cx * (1 - self.scale) + self.dx],
                      [0, self.scale, cy * (1 - self.scale) + self.dy], [0, 0, 1.0]])
        m = a @ h
        return m / m[2, 2]

    def params(self) -> dict[str, float]:
        """The six transform parameters.

        Returns:
            ``{"dx", "dy", "scale", "roll", "pitch", "yaw"}``.
        """
        return {k: float(getattr(self, k)) for k in self.PARAMS}

    def to_json(self) -> dict:
        """Plain-data form for the JSON file.

        Returns:
            The fields as plain data.
        """
        d = asdict(self)
        for k in (*self.PARAMS, "seconds"):
            d[k] = round(float(d[k]), 4)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Overlay":
        """Inverse of :meth:`to_json` (missing fields take their defaults).

        Args:
            d: A dict as written by :meth:`to_json`; unknown keys are ignored.

        Returns:
            The :class:`Overlay`.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_overlays(path: str | Path) -> dict[str, Overlay]:
    """Read an overlay file; a missing file gives an empty dict.

    Args:
        path: The JSON file (e.g. ``annotations/season_overlay.json``).

    Returns:
        The transforms per tree id.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return {k: Overlay.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_overlays(path: str | Path, overlays: dict[str, Overlay]) -> None:
    """Write an overlay file atomically.

    Args:
        path: The JSON file to write.
        overlays: The transforms per tree id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: overlays[k].to_json() for k in sorted(overlays)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def interior_pairs(pairs: SeasonPairs, margin: float = 15.0) -> SeasonPairs:
    """The pairs whose points are both at least ``margin`` px from the top and bottom edges.

    Where a branch leaves the edge of the photo is not the same physical
    point in the two seasons, because they frame the tree differently (the
    student's finding in session 7), so such pairs cannot check an alignment.

    Args:
        pairs: A tree's clicked pairs.
        margin: Distance from the top and bottom edges (px).

    Returns:
        The pairs kept.
    """
    keep = [i for i in range(len(pairs))
            if all(margin <= r <= 479 - margin for r in (pairs.winter[i][0], pairs.summer[i][0]))]
    return SeasonPairs([pairs.winter[i] for i in keep], [pairs.summer[i] for i in keep])


def pair_distances(matrix: np.ndarray, pairs: SeasonPairs) -> np.ndarray:
    """Distance (px) between each clicked summer point and where ``matrix`` puts its winter point.

    Args:
        matrix: 3x3 winter-to-summer transform.
        pairs: A tree's clicked pairs.

    Returns:
        ``(N,)`` distances (px).
    """
    w, s = pairs.xy()
    return np.hypot(*(apply_transform(matrix, w) - s).T) if len(w) else np.zeros(0)


def mask_displacement(m1: np.ndarray, m2: np.ndarray, mask: np.ndarray, step: int = 3) -> float:
    """Mean distance (px) the winter mask's pixels move between two transforms.

    Measures how much two alignments of a tree differ where it matters, on
    the branches (every ``step``-th mask pixel is used, for speed).

    Args:
        m1: One 3x3 transform.
        m2: The other.
        mask: The winter branch mask.
        step: Use every ``step``-th mask pixel.

    Returns:
        The mean distance (px); NaN for an empty mask.
    """
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys][::step].astype(float)
    if not len(pts):
        return float("nan")
    return float(np.mean(np.hypot(*(apply_transform(m1, pts) - apply_transform(m2, pts)).T)))


def draw_summer_gt(rgb_summer: np.ndarray, warped: np.ndarray, pairs: SeasonPairs, loo: np.ndarray,
                   label: str = "") -> Image.Image:
    """The summer image with the warped mask outlined (magenta) and the clicked summer points.

    Each point is numbered with its leave-one-out error; red if above 10 px.

    Args:
        rgb_summer: ``(480, 640, 3)`` uint8 summer photo.
        warped: The winter mask in the summer view (boolean).
        pairs: The tree's clicked pairs.
        loo: Each pair's leave-one-out error (px).
        label: Text written in the corner.

    Returns:
        The PIL image.
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


def _print_pair_tables(chosen: dict[str, SeasonPairs]) -> None:
    """Leave-one-out tables of the models on the clicked pairs (see :func:`loo_table`)."""
    print(f"{len(chosen)} trees with pairs ({sum(len(p) for p in chosen.values())} pairs).")
    # Each comparison uses only trees with enough pairs to check every model in it, so the
    # models are judged on the same points: first without the homography, then with it.
    for models in (PAIR_MODELS[:3], PAIR_MODELS):
        table = loo_table(chosen, models)
        need = max(MIN_SAMPLES[m] for m in models) + 1
        if not table[0]["trees"]:
            print(f"No tree has the {need} pairs needed to compare {', '.join(models)}.")
            continue
        print(f"Leave-one-out error (px) on the {table[0]['trees']} trees with at least {need} pairs:")
        print(f"{'model':<12}{'median':>8}{'mean':>8}{'p90':>8}{'> 5 px':>9}{'> 10 px':>9}")
        for r in table:
            print(f"{r['model']:<12}{r['median']:>8.2f}{r['mean']:>8.2f}{r['p90']:>8.2f}"
                  f"{100 * r['over_5']:>8.0f}%{100 * r['over_10']:>8.0f}%")


def _stats(v: Sequence[float]) -> str:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return "none"
    return f"median {np.median(v):.1f}, 90th percentile {np.percentile(v, 90):.1f}, max {v.max():.1f} (n = {len(v)})"


def _print_overlay_summary(chosen: dict[str, Overlay]) -> None:
    """Time spent, the camera changes and the flagged masks of the overlay transforms."""
    secs = [o.seconds for o in chosen.values() if o.seconds > 0]
    print(f"{len(chosen)} trees aligned by overlay." + (
        f" Time per tree: median {np.median(secs):.0f} s, total {sum(secs) / 60:.0f} min." if secs else ""))
    for k, unit in (("dx", "px"), ("dy", "px"), ("scale", ""), ("roll", "deg"), ("pitch", "deg"), ("yaw", "deg")):
        v = np.array([getattr(o, k) for o in chosen.values()])
        print(f"  {k:<6} median {np.median(v):+8.3f} {unit:<4} range {v.min():+.3f} to {v.max():+.3f}")
    flagged = sorted(t for t, o in chosen.items() if o.mask_flag)
    if flagged:
        print(f"Winter masks flagged as wrong ({len(flagged)}): {' '.join(flagged)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Check the alignments and write the summer masks.

    Usage: ``python -m treeskel.summer_gt DATA_DIR [--source overlay|pairs]
    [--overlay-file annotations/season_overlay.json] [--repeat FILE]
    [--pairs annotations/season_pairs.json] [--model similarity] [--dev-only]
    [--rows 5 6] [--trees R05N12] [--overlays] [--out outputs/summer_gt]``.

    With ``--source overlay`` (the default), the transforms come from the
    overlay tool (:mod:`treeskel.annotate_overlay`). It prints the time spent,
    the camera changes, the flagged winter masks and two checks: the distance
    between each clicked pair's summer point and where the overlay transform
    puts its winter point (pairs at the photos' top and bottom edges left
    out, :func:`interior_pairs`), and, with ``--repeat`` (a second alignment
    of some trees), how far the winter masks move between the two alignments
    (:func:`mask_displacement`). With ``--source pairs``, the transforms are
    fitted to the clicked pairs (``--model``), and it prints the leave-one-out
    tables of the models (choose the model on the development rows only:
    ``--dev-only``) and the trees whose pairs to look at again.

    For every aligned tree it writes ``<tree>_summer_gt.png`` (the winter mask
    in the summer view) and ``<tree>_summer_covered.png`` (white where the
    winter image covers the summer view), plus ``summary.csv`` (each tree's
    scale, rotation and shift at its fork click) and contact sheets;
    ``--overlays`` also saves a full-size picture per tree.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if no aligned tree was selected or an output file could not be
        written.
    """
    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset, load_tree
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.summer_gt", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--source", choices=("overlay", "pairs"), default="overlay")
    ap.add_argument("--overlay-file", default="annotations/season_overlay.json")
    ap.add_argument("--repeat", help="a second overlay file for some of the same trees (repeatability)")
    ap.add_argument("--pairs", default="annotations/season_pairs.json")
    ap.add_argument("--model", choices=PAIR_MODELS, default="similarity", help="with --source pairs")
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

    # The trees asked for (with --dev-only, never the test rows).
    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    # The alignments: the overlay tool's transforms, or the clicked pairs (always loaded, as a check).
    pairs = load_pairs(args.pairs)
    source = load_overlays(args.overlay_file) if args.source == "overlay" else pairs
    src_file = args.overlay_file if args.source == "overlay" else args.pairs
    skipped = [t.tree_id for t in trees if t.tree_id in source and source[t.tree_id].skip]
    trees = [t for t in trees if t.tree_id in source and not source[t.tree_id].skip]
    if not trees:
        print(f"No aligned trees in {src_file} for the selected trees.")
        return 1
    chosen = {t.tree_id: source[t.tree_id] for t in trees}
    # First the summary of the alignments themselves.
    if args.source == "pairs":
        _print_pair_tables(chosen)
    else:
        _print_overlay_summary(chosen)
    if skipped:
        print(f"{len(skipped)} trees skipped as unalignable: {' '.join(skipped)}")
    repeat = load_overlays(args.repeat) if args.repeat else {}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clicks = load_clicks(args.clicks)
    gt_dir = Path(args.winter_gt) if args.winter_gt else None
    rows, tiles, failed, flagged = [], [], [], {}
    pair_err, moved = [], []
    t0 = time.time()
    step = max(1, min(25, len(trees) // 10))
    print(f"Writing the summer masks...", flush=True)
    for i, rec in enumerate(trees, 1):
        tid = rec.tree_id
        row = {"tree_id": tid}
        # The tree's winter-to-summer transform, and its check: leave-one-out errors of the pairs it was
        # fitted to, or the distance to the clicked pairs it was not.
        if args.source == "pairs":
            p = chosen[tid]
            m = pair_transform(p, args.model)
            shown, errs = p, loo_errors(p, args.model)
            issues = check_pairs(p, args.model)
            row.update({"pairs": len(p), "loo_median": round(float(np.nanmedian(errs)), 2) if np.isfinite(errs).any()
                        else "", "loo_max": round(float(np.nanmax(errs)), 2) if np.isfinite(errs).any() else ""})
        else:
            o = chosen[tid]
            m = o.matrix()
            row.update({**{k: round(v, 4) for k, v in o.params().items()}, "seconds": round(o.seconds, 1),
                        "mask_flag": o.mask_flag})
            shown = interior_pairs(pairs[tid]) if tid in pairs and not pairs[tid].skip else SeasonPairs()
            errs = pair_distances(m, shown)
            pair_err += list(errs)
            issues = ["winter mask flagged"] if o.mask_flag else []
            if len(shown):
                row["pair_error_median"] = round(float(np.median(errs)), 2)
        row.update({"ok": not issues, "issues": "; ".join(issues)})
        if issues:
            flagged[tid] = issues
        if m is not None:
            # The camera change at the fork, then the winter mask moved into the summer view.
            c = clicks.get(tid)
            at = c.fork[::-1] if c is not None and c.complete and not c.skip else (320, 240)
            row.update({f"at_fork_{k}": round(v, 4) for k, v in describe(m, at).items()})
            row["matrix"] = json.dumps(np.round(m, 6).tolist())
            winter = load_tree(rec, "winter")
            mask = load_winter_mask(tid, winter, clicks, gt_dir)
            if mask is not None:
                if tid in repeat and not repeat[tid].skip:
                    row["repeat_displacement"] = round(mask_displacement(m, repeat[tid].matrix(), mask), 2)
                    moved.append(row["repeat_displacement"])
                summer_rgb = load_tree(rec, "summer")["rgb"]
                warped, covered = warp_mask(mask, m, summer_rgb.shape[:2])
                save_image(Image.fromarray(warped.astype(np.uint8) * 255), out / f"{tid}_summer_gt.png", failed)
                save_image(Image.fromarray(covered.astype(np.uint8) * 255), out / f"{tid}_summer_covered.png", failed)
                label = f"{tid} {args.source}" + ("  FLAG" if issues else "")
                pic = draw_summer_gt(summer_rgb, warped, shown, errs, label)
                if args.overlays:
                    save_image(pic, out / f"{tid}_overlay.jpg", failed, quality=85)
                tiles.append(np.asarray(pic.resize((320, 240))))
        rows.append(row)
        if i % step == 0 or i == len(trees):
            el = time.time() - t0
            print(f"  {i}/{len(trees)} trees, {el:.0f} s elapsed, about {el / i * (len(trees) - i):.0f} s left",
                  flush=True)

    # The checks over all trees.
    if args.source == "overlay":
        print(f"Clicked pairs (away from the photos' edges) against the overlay transforms: "
              f"distance (px) {_stats(pair_err)}")
        if args.repeat:
            print(f"Repeat alignment: the winter masks move (px) {_stats(moved)}")
    try:
        with open(out / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            wr.writeheader()
            wr.writerows(rows)
    except OSError as e:
        failed.append(f"{out / 'summary.csv'}: {e.strerror or e}")
    for i in range(0, len(tiles), 24):
        save_image(contact_sheet(tiles[i:i + 24]), out / f"sheet_{i // 24 + 1:02d}.jpg", failed, quality=85)
    if flagged and args.source == "pairs":
        print(f"\nLook at these trees' pairs again ({len(flagged)}; python -m treeskel.annotate_pairs DATA "
              "--redo --trees ...):")
        for tid, why in sorted(flagged.items()):
            print(f"  {tid}: {'; '.join(why)}")
    print(f"\nResults in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
