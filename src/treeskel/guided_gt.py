"""Click-guided main-branch ground truth from winter images.

The automatic method (:mod:`treeskel.winter_gt`) is kept as a baseline. It
failed on many trees: posts, straight unbranched trees, gaps in depth, support
wires, and branches that cross. Here a person supplies a few clicks per tree,
and the branches are traced between them automatically:

1. **Clicks** (:class:`Clicks`): the trunk base, the fork, and the tip of each
   of the two V leaders, plus optional waypoints where a trace needs guiding.
2. **Cost map** (:func:`branch_cost`): each pixel gets a "woodiness" in
   [0, 1] from three cues, and a cost ``floor + (1 - woodiness)^power``:

   - depth: the pixel stands out from the ground or background
     (:func:`treeskel.winter_gt.foreground_mask`, smoothed), weighted by how
     central it is within the object (distance to the edge divided by the
     largest such distance nearby), so paths run down the middle of thick
     trunks rather than along an edge;
   - tube shape: the Frangi vesselness filter, which uses the eigenvalues of
     the image's Hessian (second derivatives) to respond to bright, elongated
     structures at the chosen widths;
   - colour: bark is pale and unsaturated, unlike grass and soil.

3. **Tracing** (:func:`trace`): between consecutive clicks, the path with the
   smallest total cost is found by Dijkstra's algorithm over the 8-connected
   pixel grid (diagonal steps cost √2 × the pixel cost). Cheap woody pixels
   pull the path onto the branch, and short gaps are crossed where needed.
4. **Result** (:func:`guided_gt`): a :class:`treeskel.winter_gt.WinterGT`, so
   the same rendering, shape measures and overlays apply to both methods.

Clicks are stored in a JSON file keyed by tree id (see :func:`save_clicks`),
so the ground truth can be regenerated exactly with
``python -m treeskel.guided_gt DATA_DIR --clicks annotations/winter_clicks.json``.
Clicks are made with :mod:`treeskel.annotate`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import frangi
from skimage.graph import route_through_array
from skimage.morphology import disk

from treeskel.winter_gt import (
    WinterGT,
    foreground_mask,
    overlay,
    path_shape,
    render_main_branches,
    report_failed,
    save_image,
)

Point = tuple[int, int]


@dataclass(frozen=True)
class GuidedParams:
    """Settings for the cost map.

    Attributes:
        depth_weight: Weight of the depth cue in woodiness.
        tube_weight: Weight of the Frangi tube cue.
        colour_weight: Weight of the pale-bark colour cue.
        sigmas: Frangi filter scales (Gaussian sigma, px), roughly half the
            branch widths to respond to.
        cost_floor: Smallest pixel cost; keeps paths from taking long detours
            through very woody areas.
        cost_power: Exponent on (1 − woodiness); higher values stick more
            tightly to woody pixels.
        centre_window: Half-size (px) of the neighbourhood used to judge how
            central a pixel is within its object; should exceed the widest
            trunk's half-width.
    """

    depth_weight: float = 0.5
    tube_weight: float = 0.3
    colour_weight: float = 0.2
    sigmas: tuple[float, ...] = (2, 4, 6, 8, 10)
    cost_floor: float = 0.02
    cost_power: float = 2.0
    centre_window: int = 15


@dataclass
class Clicks:
    """Clicked points for one tree, as ``(row, col)`` pixels.

    Attributes:
        trunk: ``[root, waypoints..., fork]`` (at least two points).
        left: ``[waypoints..., tip]`` for the image-left leader; traced from
            the fork.
        right: ``[waypoints..., tip]`` for the image-right leader.
        skip: True if the tree cannot be annotated (it is then excluded).
        note: Free-text reason or comment.
    """

    trunk: list[Point] = field(default_factory=list)
    left: list[Point] = field(default_factory=list)
    right: list[Point] = field(default_factory=list)
    skip: bool = False
    note: str = ""

    @property
    def complete(self) -> bool:
        """True if there is enough to trace: root, fork and both tips."""
        return len(self.trunk) >= 2 and bool(self.left) and bool(self.right)

    @property
    def fork(self) -> Point:
        """The fork (last trunk point)."""
        return self.trunk[-1]

    def to_json(self) -> dict:
        """Plain-data form for the JSON file."""
        d = asdict(self)
        for k in ("trunk", "left", "right"):
            d[k] = [list(map(int, p)) for p in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Clicks":
        """Inverse of :meth:`to_json`."""
        return cls(
            trunk=[tuple(p) for p in d.get("trunk", [])],
            left=[tuple(p) for p in d.get("left", [])],
            right=[tuple(p) for p in d.get("right", [])],
            skip=bool(d.get("skip", False)),
            note=str(d.get("note", "")),
        )


def load_clicks(path: str | Path) -> dict[str, Clicks]:
    """Read a clicks file; a missing file gives an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    return {k: Clicks.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_clicks(path: str | Path, clicks: dict[str, Clicks]) -> None:
    """Write a clicks file atomically (a crash never leaves it half-written)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: clicks[k].to_json() for k in sorted(clicks)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def branch_cost(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                params: GuidedParams = GuidedParams()) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel woodiness and tracing cost for a winter image.

    Args:
        rgb: ``(H, W, 3)`` uint8 colour image.
        depth: ``(H, W)`` 8-bit depth (brighter = nearer).
        valid: ``(H, W)`` bool validity mask.
        params: Cue weights and cost shape.

    Returns:
        ``(cost, woodiness)``, both ``(H, W)`` float arrays; ``woodiness`` is in
        [0, 1] and ``cost`` in ``[cost_floor, cost_floor + 1]``.
    """
    mask = foreground_mask(depth, valid)
    edt = ndi.distance_transform_edt(mask)
    local_max = ndi.maximum_filter(edt, size=2 * params.centre_window + 1)
    centred = np.divide(edt, local_max, out=np.zeros_like(edt), where=local_max > 0)
    fg = ndi.gaussian_filter(mask.astype(float), 1.0) * (0.5 + 0.5 * centred)
    grey = rgb.astype(float).mean(2) / 255.0
    tube = frangi(grey, sigmas=params.sigmas, black_ridges=False)
    scale = np.percentile(tube, 99.5)
    tube = np.clip(tube / scale, 0, 1) if scale > 0 else np.zeros_like(tube)
    hsv = rgb2hsv(rgb)
    pale = np.clip((hsv[..., 2] - 0.35) / 0.5, 0, 1) * np.clip(1 - hsv[..., 1] / 0.45, 0, 1)
    wood = np.clip(params.depth_weight * fg + params.tube_weight * tube
                   + params.colour_weight * pale, 0, 1)
    return params.cost_floor + (1 - wood) ** params.cost_power, wood


def trace(cost: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    """Cheapest 8-connected path through ``points`` in order.

    Args:
        cost: ``(H, W)`` positive cost map.
        points: Two or more ``(row, col)`` points.

    Returns:
        ``(N, 2)`` int array of path pixels from the first to the last point,
        passing through every point.
    """
    h, w = cost.shape
    pts = [(min(max(int(r), 0), h - 1), min(max(int(c), 0), w - 1)) for r, c in points]
    out: list[tuple[int, int]] = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if a == b:
            continue
        seg, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
        out.extend(tuple(p) for p in seg[1:])
    return np.array(out, int).reshape(-1, 2)


def guided_gt(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, clicks: Clicks,
              params: GuidedParams = GuidedParams(), cost: np.ndarray | None = None) -> WinterGT:
    """Trace the trunk and both leaders through the clicked points.

    Args:
        rgb, depth, valid: The tree's winter images.
        clicks: Complete :class:`Clicks` (not skipped).
        params: Cost-map settings (ignored if ``cost`` is given).
        cost: Precomputed cost map, to avoid recomputing it while clicking.

    Returns:
        A :class:`treeskel.winter_gt.WinterGT` with ``ok=True``. Its
        ``tree_mask`` is the depth foreground, used for branch widths.

    Raises:
        ValueError: If the clicks are skipped or incomplete.
    """
    if clicks.skip or not clicks.complete:
        raise ValueError("clicks must be complete and not skipped")
    if cost is None:
        cost, _ = branch_cost(rgb, depth, valid, params)
    fg = ndi.binary_closing(foreground_mask(depth, valid), disk(2))
    trunk = trace(cost, clicks.trunk)
    left = trace(cost, [clicks.fork, *clicks.left])
    right = trace(cost, [clicks.fork, *clicks.right])
    radius = ndi.distance_transform_edt(fg)
    skel = np.zeros(fg.shape, bool)
    for p in (trunk, left, right):
        skel[p[:, 0], p[:, 1]] = True
    return WinterGT(
        tree_mask=fg, skeleton=skel,
        root=tuple(map(int, trunk[0])), fork=tuple(map(int, trunk[-1])),
        trunk=trunk, left=left, right=right,
        trunk_shape=path_shape(trunk, radius), left_shape=path_shape(left, radius),
        right_shape=path_shape(right, radius), ok=True,
    )


def insert_waypoint(clicks: Clicks, gt: WinterGT, point: Point) -> Clicks:
    """Add a correcting click to the part of the trace it is nearest to.

    The point joins the trunk, left or right control points, whichever traced
    path passes closest to it, at the position matching its place along that
    path. The trunk's root and fork, and the leaders' tips, stay fixed.

    Args:
        clicks: Current complete clicks.
        gt: The current trace (from :func:`guided_gt`).
        point: The new ``(row, col)`` click.

    Returns:
        New :class:`Clicks` with the waypoint inserted.
    """
    p = np.asarray(point)
    best = None
    for name in ("trunk", "left", "right"):
        path = getattr(gt, name)
        if len(path):
            d = np.hypot(*(path - p).T)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (d[i], name, i, path)
    _, name, idx, path = best
    ctrl = list(getattr(clicks, name))

    def pos(q: Point) -> int:
        return int(np.argmin(np.hypot(*(path - np.asarray(q)).T)))

    # Insert before the first control point that lies further along the path.
    # The trunk's first point (root) is fixed, so start after it.
    start = 1 if name == "trunk" else 0
    k = next((j for j in range(start, len(ctrl)) if pos(ctrl[j]) > idx), len(ctrl))
    if name == "trunk":
        k = min(max(k, 1), len(ctrl) - 1)   # keep root first and fork last
    else:
        k = min(k, len(ctrl) - 1)           # keep the tip last
    ctrl.insert(k, (int(point[0]), int(point[1])))
    new = Clicks(**{**clicks.__dict__})
    setattr(new, name, ctrl)
    return new


def main(argv: Sequence[str] | None = None) -> int:
    """Render guided ground truth for every annotated tree.

    Usage: ``python -m treeskel.guided_gt DATA_DIR --clicks annotations/winter_clicks.json
    --out outputs/winter_gt_guided [--overlays]``.
    """
    from treeskel.io import index_dataset, load_tree

    ap = argparse.ArgumentParser(prog="python -m treeskel.guided_gt", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--out", default="outputs/winter_gt_guided", help="output folder (git-ignored)")
    ap.add_argument("--overlays", action="store_true", help="also save <tree>_overlay.jpg")
    args = ap.parse_args(argv)

    clicks = load_clicks(args.clicks)
    trees = {t.tree_id: t for t in index_dataset(args.data_dir).trees}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = GuidedParams()
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    rows, skipped, missing, failed = [], [], [], []
    for tid, c in sorted(clicks.items()):
        if c.skip or not c.complete:
            skipped.append(tid)
            continue
        if tid not in trees:
            missing.append(tid)
            continue
        d = load_tree(trees[tid], "winter")
        gt = guided_gt(d["rgb"], d["depth"], d["valid"], c, params)
        save_image(Image.fromarray(render_main_branches(gt).astype(np.uint8) * 255),
                   out / f"{tid}_winter_gt.png", failed)
        if args.overlays:
            save_image(overlay(d["rgb"], gt, tid), out / f"{tid}_overlay.jpg", failed, quality=85)
        rows.append({"tree_id": tid, **gt.summary()})
    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    print(f"Rendered {len(rows)} trees; {len(skipped)} skipped or incomplete; "
          f"{len(missing)} not found in {args.data_dir}. Results in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
