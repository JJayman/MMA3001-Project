"""Automatic main-branch ground truth from winter depth images.

In winter the trees are bare, so the main branches can be seen directly. This
module extracts them from the winter depth image in five steps:

1. **Foreground** (:func:`foreground_mask`). Above the ground line, rows are
   mostly "no reading" because the rows behind are out of range, so any valid
   pixel is foreground. In rows that are mostly ground, a pixel is foreground
   if it is clearly nearer (brighter) than that row's median, which is the
   ground.
2. **Tree selection** (:func:`select_tree`). Choose the connected region that
   is large, reaches low in the image and is centred at its base: the tree
   being photographed rather than a neighbour or a post.
3. **Skeleton and gap linking** (:func:`link_gaps`). Thin the tree to
   one-pixel centre lines. Depth often misses sections of branch, so a branch
   end that points upwards is joined to a nearby skeleton piece lying ahead of
   it, within a distance and angle limit.
4. **Main branches** (:func:`find_main_branches`). The trunk root is the
   skeleton point at the bottom centre of the tree. The trees are trained as
   a 2D open V (Tatura trellis), so one leader is chosen on each side of the
   root: the tip that climbs highest, minus a penalty for wandering sideways.
   Candidates whose upper part is wider than a leader can be (trellis posts)
   are skipped. The fork is where the two leader paths separate.
5. **Quality check.** The tree is flagged, so it can be left out rather than
   given a wrong label, if a leader rises less than a set fraction of the
   root's height (usually a depth gap), or if no non-post leader exists on a
   side. (The trunk's width is recorded but not checked: near the ground,
   ground pixels merge into the trunk and make every trunk look post-wide.)

:func:`render_main_branches` turns the result into a mask, with the width at
each point taken from the tree's thickness there.

Run ``python -m treeskel.winter_gt DATA_DIR --out outputs/winter_gt`` to
process a whole dataset folder (add ``--figure overview.png`` for a visual
check).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.draw import disk as draw_disk
from skimage.draw import line as draw_line
from skimage.morphology import disk, skeletonize

_K8 = np.ones((3, 3), bool)  # 8-connectivity


@dataclass(frozen=True)
class GTParams:
    """Settings for :func:`extract_winter_gt`. Units are pixels unless noted.

    Attributes:
        ground_row_fraction: A row whose fraction of valid pixels is at least
            this is treated as a ground row.
        ground_margin: In ground rows, how much brighter (nearer) than the row
            median a pixel must be to count as foreground (8-bit depth units).
        min_tree_area: Smallest region considered as the tree.
        min_piece_area: Smallest separate region that gap linking may join.
        link_gap: Largest gap a branch end may be joined across.
        link_max_angle: Largest angle (degrees) between a branch end's direction
            and the gap for a join.
        link_min_up: A branch end is only extended if the upward component of
            its unit direction is at least this (0.5 = within 60° of vertical).
        wander_penalty: Weight on (path length − rise) when scoring leaders.
        min_leader_rise: A leader must rise at least this fraction of the
            root's height above the image top to pass the quality check.
        max_leader_radius: A candidate leader whose upper part has a larger
            median half-width than this is treated as a post and skipped.
            On the sample, real leaders measured 6.0–8.2 and a post 17.7.
        shape_top_fraction: Width, straightness and tilt are measured on the
            top part of a leader, covering this fraction of its rise, so the
            thick trunk and fork do not dominate.
    """

    ground_row_fraction: float = 0.5
    ground_margin: float = 6.0
    min_tree_area: int = 200
    min_piece_area: int = 30
    link_gap: float = 35.0
    link_max_angle: float = 35.0
    link_min_up: float = 0.5
    wander_penalty: float = 0.5
    min_leader_rise: float = 0.3
    max_leader_radius: float = 12.0
    shape_top_fraction: float = 0.6


@dataclass
class WinterGT:
    """Result of :func:`extract_winter_gt`.

    Paths are ``(N, 2)`` integer arrays of ``(row, col)`` pixel coordinates.

    Attributes:
        tree_mask: The selected tree region (before branch selection).
        skeleton: Centre lines of the tree, after gap linking.
        root: Trunk root ``(row, col)``, or ``None`` if nothing was found.
        fork: Point where the two leaders separate, or ``None``.
        trunk: Path from ``root`` to ``fork``.
        left: Path from ``fork`` to the left leader's tip.
        right: Path from ``fork`` to the right leader's tip.
        trunk_shape: :func:`path_shape` of the trunk (empty if none).
        left_shape: :func:`path_shape` of the left leader (empty if none).
        right_shape: :func:`path_shape` of the right leader (empty if none).
        posts_skipped: Number of candidate leaders skipped as too wide.
        radii: Optional per-point drawing radius for each part
            (``{"trunk": array, "left": ..., "right": ...}``, same lengths as
            the paths). If absent, :func:`render_main_branches` measures
            widths from ``tree_mask``.
        shoulders: Optional bool mask of the pixels that fill the notches
            ("shoulders") beside the fork, where each leader leaves the trunk
            (guided ground truth only, see
            :func:`treeskel.guided_gt.fill_shoulders`);
            :func:`render_main_branches` adds it to the drawn branches.
        trunk_cuts: Optional flat ends for the trunk, as a list of
            ``(point, direction)`` pairs of ``(2,)`` arrays: trunk pixels
            ``p`` beyond a cut, ``(p − point)·direction > 0``, are not drawn.
            Without cuts, the trunk's first and last discs make its ends
            round (guided ground truth only, see
            :func:`treeskel.guided_gt.end_cut`).
        junction: Optional bool mask of the pixels joining the trunk's flat
            top to each leader (guided ground truth only, see
            :func:`treeskel.guided_gt.fork_junction`);
            :func:`render_main_branches` adds it to the drawn branches.
        ok: True if both leaders passed the quality check.
        issues: Reasons the quality check failed (empty if ``ok``).
    """

    tree_mask: np.ndarray
    skeleton: np.ndarray
    root: tuple[int, int] | None = None
    fork: tuple[int, int] | None = None
    trunk: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    left: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    right: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    trunk_shape: dict = field(default_factory=dict)
    left_shape: dict = field(default_factory=dict)
    right_shape: dict = field(default_factory=dict)
    posts_skipped: int = 0
    radii: dict | None = None
    shoulders: np.ndarray | None = None
    trunk_cuts: list[tuple[np.ndarray, np.ndarray]] | None = None
    junction: np.ndarray | None = None
    ok: bool = False
    issues: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Plain-data summary (for CSV/JSON): points, rises and QC result."""
        def rise(p: np.ndarray) -> int:
            return int(p[0, 0] - p[-1, 0]) if len(p) else 0
        return {
            "ok": self.ok,
            "issues": "; ".join(self.issues),
            "root": list(self.root) if self.root else None,
            "fork": list(self.fork) if self.fork else None,
            "trunk_rise": rise(self.trunk),
            "left_rise": rise(self.left),
            "right_rise": rise(self.right),
            "posts_skipped": self.posts_skipped,
            **{f"{side}_{k}": round(v, 2)
               for side in ("trunk", "left", "right")
               for k, v in getattr(self, f"{side}_shape").items()},
        }


def foreground_mask(depth: np.ndarray, valid: np.ndarray, params: GTParams = GTParams()) -> np.ndarray:
    """Find pixels that stand out from the ground or the empty background.

    Args:
        depth: ``(H, W)`` 8-bit depth (brighter = nearer).
        valid: ``(H, W)`` bool, True where depth has a reading.
        params: Settings.

    Returns:
        ``(H, W)`` bool foreground mask.
    """
    d = depth.astype(float)
    fg = np.zeros(valid.shape, bool)
    for y in range(d.shape[0]):
        v = valid[y]
        if v.mean() < params.ground_row_fraction:
            fg[y] = v  # above the ground line: anything in range is foreground
        else:
            fg[y] = v & (d[y] > np.median(d[y][v]) + params.ground_margin)
    return fg


def select_tree(fg: np.ndarray, params: GTParams = GTParams()) -> np.ndarray:
    """Choose the region most likely to be the photographed tree.

    Each region is scored by ``area × (lowest row / height) × centring²``,
    where centring is 1 when the base of the region (its lowest 10%) is at the
    image centre and 0 at an edge.

    Args:
        fg: Foreground mask from :func:`foreground_mask`.
        params: Settings (``min_tree_area``).

    Returns:
        Bool mask of the chosen region (all False if none is large enough).
    """
    lab, n = ndi.label(fg, structure=_K8)
    h, w = fg.shape
    best, best_score = 0, -1.0
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < params.min_tree_area:
            continue
        base_x = np.median(xs[ys >= np.percentile(ys, 90)])
        centring = 1 - abs(base_x - w / 2) / (w / 2)
        score = len(ys) * (ys.max() / h) * centring**2
        if score > best_score:
            best, best_score = i, score
    return lab == best if best else np.zeros_like(fg)


def _graph(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Pixel graph of a skeleton: node coordinates and 8-neighbour edge lengths."""
    ys, xs = np.nonzero(skel)
    idx = -np.ones(skel.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        j = np.full(len(ys), -1)
        j[ok] = idx[y2[ok], x2[ok]]
        m = j >= 0
        rows.append(np.nonzero(m)[0])
        cols.append(j[m])
        wts.append(np.full(int(m.sum()), np.hypot(dy, dx)))
    n = len(ys)
    a = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return ys, xs, (a + a.T).tocsr()


def _end_directions(skel: np.ndarray, back: float = 12.0) -> list[tuple[int, int, np.ndarray]]:
    """Skeleton end points with the unit direction the branch is heading."""
    ys, xs, a = _graph(skel)
    out = []
    for e in np.nonzero(np.diff(a.indptr) == 1)[0]:
        dist = dijkstra(a, indices=e, limit=back)
        f = int(np.argmax(np.where(np.isinf(dist), -1, dist)))
        d = np.array([ys[e] - ys[f], xs[e] - xs[f]], float)
        n = np.hypot(*d)
        if n > 0:
            out.append((int(ys[e]), int(xs[e]), d / n))
    return out


def link_gaps(main: np.ndarray, others: np.ndarray, params: GTParams = GTParams(),
              iters: int = 15) -> np.ndarray:
    """Join skeleton pieces that continue a branch across a depth gap.

    A branch end heading upwards is joined by a straight line to the nearest
    pixel of another piece within ``link_gap`` and within ``link_max_angle``
    of the end's direction; that whole piece is then added. Repeats until
    nothing more joins.

    Args:
        main: Skeleton of the tree.
        others: Skeletons of other foreground pieces.
        params: Settings.
        iters: Maximum number of passes.

    Returns:
        The extended skeleton.
    """
    lab, _ = ndi.label(others, structure=_K8)
    oy, ox = np.nonzero(others)
    skel = main.copy()
    cos_min = np.cos(np.radians(params.link_max_angle))
    for _ in range(iters):
        added = False
        for y, x, d in _end_directions(skel):
            if -d[0] < params.link_min_up:  # row index decreases going up
                continue
            dy, dx = oy - y, ox - x
            dist = np.hypot(dy, dx)
            ok = (dist > 0) & (dist <= params.link_gap) & ~skel[oy, ox]
            ok &= (dy * d[0] + dx * d[1]) >= cos_min * dist
            if not ok.any():
                continue
            k = np.nonzero(ok)[0][np.argmin(dist[ok])]
            rr, cc = draw_line(y, x, int(oy[k]), int(ox[k]))
            skel[rr, cc] = True
            skel |= lab == lab[oy[k], ox[k]]
            added = True
        if not added:
            break
    return skel


def path_shape(path: np.ndarray, radius: np.ndarray, top_fraction: float = 0.6) -> dict:
    """Width, straightness and tilt of the upper part of a path.

    The upper part is the points within the top ``top_fraction`` of the
    path's rise. A straight line is fitted to it by total least squares (the
    principal axis from an SVD of the centred points), which, unlike ordinary
    regression, treats rows and columns alike and so also works for
    near-vertical lines.

    Args:
        path: ``(N, 2)`` ``(row, col)`` points, starting at the bottom.
        radius: Half-width image (distance from each pixel to the edge of the
            tree region).
        top_fraction: Fraction of the rise, from the top, to measure.

    Returns:
        ``{"radius": median half-width (px), "straightness": RMS distance from
        the fitted line (px), "tilt": angle from vertical (degrees)}``, or an
        empty dict if the path has fewer than 3 points.
    """
    p = np.asarray(path)
    if len(p) < 3:
        return {}
    rise = p[0, 0] - p[:, 0].min()
    q = p[p[:, 0] <= p[0, 0] - (1 - top_fraction) * rise]
    if len(q) < 3:
        q = p
    c = q - q.mean(0)
    _, _, vt = np.linalg.svd(c.astype(float), full_matrices=False)
    off = c @ vt[1]  # signed distance from the fitted line
    return {
        "radius": float(np.median(radius[q[:, 0], q[:, 1]])),
        "straightness": float(np.sqrt(np.mean(off**2))),
        "tilt": float(np.degrees(np.arctan2(abs(vt[0][1]), abs(vt[0][0])))),
    }


def find_main_branches(tree: np.ndarray, skel: np.ndarray, params: GTParams = GTParams(),
                       radius: np.ndarray | None = None) -> dict:
    """Pick the root, fork and the two leaders from a tree skeleton.

    Args:
        tree: Tree region (used to locate the trunk base).
        skel: Skeleton of the tree (connected).
        params: Settings (``wander_penalty``, ``max_leader_radius``).
        radius: Half-width image used for the post check; defaults to the
            distance transform of ``tree``. Pass one computed from the full
            foreground so leaders joined across gaps are measured too.

    Returns:
        Dict with ``root``, ``fork`` (``(row, col)`` or ``None``), ``trunk``,
        ``left``, ``right`` paths (``(N, 2)`` arrays, possibly empty), their
        ``trunk_shape``/``left_shape``/``right_shape`` and ``posts_skipped``.
    """
    empty = np.zeros((0, 2), int)
    out = {"root": None, "fork": None, "trunk": empty, "left": empty, "right": empty,
           "trunk_shape": {}, "left_shape": {}, "right_shape": {}, "posts_skipped": 0}
    if skel.sum() < 2:
        return out
    ys, xs, a = _graph(skel)
    ty, tx = np.nonzero(tree)
    base_x = np.median(tx[ty >= np.percentile(ty, 97)])
    # Root: skeleton point closest to the bottom centre of the trunk (sideways
    # distance counts double, so stray ground pixels at the base are avoided).
    root = int(np.argmin(np.hypot(ys - ty.max(), 2 * (xs - base_x))))
    out["root"] = (int(ys[root]), int(xs[root]))
    dist, pred = dijkstra(a, indices=root, return_predecessors=True)
    ends = np.nonzero((np.diff(a.indptr) == 1) & np.isfinite(dist))[0]
    ends = ends[ends != root]

    def path(i: int) -> list[int]:
        p = [i]
        while p[-1] != root:
            p.append(int(pred[p[-1]]))
        return p[::-1]

    def score(e: int) -> float:
        rise = ys[root] - ys[e]
        return rise - params.wander_penalty * (dist[e] - rise)

    coords = np.stack([ys, xs], 1)
    if radius is None:
        radius = ndi.distance_transform_edt(tree)

    def best(candidates: list[int]) -> tuple[list[int] | None, dict]:
        """Highest-scoring candidate that is not post-like."""
        for e in sorted(candidates, key=score, reverse=True):
            p = path(e)
            shape = path_shape(coords[p], radius, params.shape_top_fraction)
            if shape and shape["radius"] > params.max_leader_radius:
                out["posts_skipped"] += 1
                continue
            return p, shape
        return None, {}

    pl, sl = best([e for e in ends if xs[e] < xs[root]])
    pr, sr = best([e for e in ends if xs[e] >= xs[root]])
    if pl is None or pr is None:
        return out
    shared = set(pr)
    k = max(i for i, v in enumerate(pl) if v in shared)  # last shared node = fork
    fork = pl[k]
    kr = pr.index(fork)
    out.update(
        trunk_shape=path_shape(coords[pl[: k + 1]], radius, params.shape_top_fraction),
        left_shape=sl,
        right_shape=sr,
        fork=(int(ys[fork]), int(xs[fork])),
        trunk=coords[pl[: k + 1]],
        left=coords[pl[k:]],
        right=coords[pr[kr:]],
    )
    return out


def extract_winter_gt(depth: np.ndarray, valid: np.ndarray, params: GTParams = GTParams()) -> WinterGT:
    """Extract the trunk and the two V leaders from a winter depth image.

    Args:
        depth: ``(H, W)`` 8-bit depth (brighter = nearer).
        valid: ``(H, W)`` bool validity mask.
        params: Settings.

    Returns:
        A :class:`WinterGT`. ``ok`` is False (with reasons in ``issues``) if no
        tree was found or a leader failed the quality check.
    """
    fg = foreground_mask(depth, valid, params)
    tree = select_tree(fg, params)
    if not tree.any():
        return WinterGT(tree, np.zeros_like(tree), issues=["no tree region found"])
    tree = ndi.binary_fill_holes(ndi.binary_closing(tree, disk(2)))

    # Other pieces that gap linking may join.
    others = fg & ~ndi.binary_dilation(tree, disk(1))
    lab, n = ndi.label(others, structure=_K8)
    areas = ndi.sum(others, lab, range(1, n + 1))
    others = np.isin(lab, np.nonzero(areas >= params.min_piece_area)[0] + 1)

    # Keep the largest connected piece of the tree's own skeleton.
    sk = skeletonize(tree)
    lab, n = ndi.label(sk, structure=_K8)
    if n > 1:
        sk = lab == (np.argmax(ndi.sum(sk, lab, range(1, n + 1))) + 1)
    sk = link_gaps(sk, skeletonize(others), params)

    radius = ndi.distance_transform_edt(ndi.binary_closing(fg, disk(2)) | tree)
    br = find_main_branches(tree, sk, params, radius)
    gt = WinterGT(tree, sk, **br)
    if gt.fork is None:
        gt.issues.append("could not find a (non-post) leader on each side")
    else:
        need = params.min_leader_rise * gt.root[0]
        for side in ("left", "right"):
            p = getattr(gt, side)
            rise = p[0, 0] - p[-1, 0]
            if rise < need:
                gt.issues.append(f"{side} leader rises {rise} px (< {need:.0f})")
    gt.ok = not gt.issues
    return gt


def render_main_branches(gt: WinterGT, min_radius: float = 2.0, max_radius: float = 25.0) -> np.ndarray:
    """Draw the trunk and leaders as a mask with realistic widths.

    If ``gt.radii`` is set (guided ground truth, fitted tapers), those radii
    are used. Otherwise the radius at each path point is the distance from
    that point to the edge of the tree region, clipped to
    ``[min_radius, max_radius]``; linked gaps (outside the tree region) get
    ``min_radius``. If ``gt.trunk_cuts`` is set, the trunk is cut flat
    there; if ``gt.shoulders`` or ``gt.junction`` is set, those pixels are
    added.

    Args:
        gt: Result of :func:`extract_winter_gt`.
        min_radius: Smallest drawn radius in pixels.
        max_radius: Largest drawn radius in pixels.

    Returns:
        Bool mask the same shape as ``gt.tree_mask``.
    """
    shape = gt.tree_mask.shape
    out = np.zeros(shape, bool)
    if gt.radii is None:
        radius = np.clip(ndi.distance_transform_edt(gt.tree_mask), min_radius, max_radius)
    for name in ("trunk", "left", "right"):
        p = getattr(gt, name)
        rs = gt.radii[name] if gt.radii is not None else radius[p[:, 0], p[:, 1]] if len(p) else []
        part = np.zeros(shape, bool)
        for (y, x), r in zip(p, rs):
            rr, cc = draw_disk((y, x), r, shape=shape)
            part[rr, cc] = True
        if name == "trunk" and gt.trunk_cuts:
            yy, xx = np.indices(shape)
            for point, direction in gt.trunk_cuts:
                part &= (yy - point[0]) * direction[0] + (xx - point[1]) * direction[1] <= 0
        out |= part
    for extra in (gt.shoulders, gt.junction):
        if extra is not None:
            out |= extra
    return out


def overlay(rgb: np.ndarray, gt: WinterGT, label: str = "") -> Image.Image:
    """Draw the result on the winter RGB image for visual checking.

    Skeleton in blue, left leader red, right leader yellow, trunk white, fork
    circled; the label and PASS/FLAG status are written top-left.
    """
    im = (rgb * 0.5).astype(np.uint8)
    im[gt.skeleton] = (0, 160, 255)
    pil = Image.fromarray(im)
    dr = ImageDraw.Draw(pil)
    for p, col in ((gt.trunk, "white"), (gt.left, "red"), (gt.right, "yellow")):
        if len(p) > 1:
            dr.line([(int(x), int(y)) for y, x in p], fill=col, width=4)
    if gt.fork:
        y, x = gt.fork
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline="white", width=3)
    dr.text((8, 8), f"{label}  {'PASS' if gt.ok else 'FLAG'}", fill="lime" if gt.ok else "orange")
    return pil


def save_image(image: Image.Image, path: Path, failed: list[str], **kwargs) -> None:
    """Save an image, recording (not raising) a failure to write it.

    On Windows a file that is open in another program (an image viewer, the
    Explorer preview pane, a sync client) cannot be overwritten. One such file
    should not stop a run over hundreds of trees, so the error is added to
    ``failed`` and reported at the end instead.
    """
    try:
        image.save(path, **kwargs)
    except OSError as e:
        failed.append(f"{path}: {e.strerror or e}")


def report_failed(failed: list[str]) -> None:
    """Print files that could not be written, with the likely cause."""
    if failed:
        print(f"Could not write {len(failed)} file(s) - are they open in another "
              "program (image viewer, Explorer preview pane, OneDrive)?")
        for f in failed[:20]:
            print(f"  {f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the extraction over a dataset folder.

    Writes ``<tree>_winter_gt.png`` (rendered main branches) for every tree
    and ``summary.csv`` with the quality-check result per tree. Usage:
    ``python -m treeskel.winter_gt DATA_DIR --out outputs/winter_gt``.
    """
    from treeskel.io import index_dataset, load_tree

    ap = argparse.ArgumentParser(prog="python -m treeskel.winter_gt", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="outputs/winter_gt", help="output folder (git-ignored)")
    ap.add_argument("--limit", type=int, help="only process the first N trees")
    ap.add_argument("--figure", metavar="PNG",
                    help="also save an overview of up to 24 trees drawn on their winter RGB")
    ap.add_argument("--overlays", action="store_true",
                    help="also save <tree>_overlay.jpg for every tree (for visual checking)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trees = index_dataset(args.data_dir).trees[: args.limit]
    params = GTParams()
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    rows, tiles, failed = [], [], []
    for t in trees:
        d = load_tree(t, "winter")
        gt = extract_winter_gt(d["depth"], d["valid"], params)
        if args.overlays or (args.figure and len(tiles) < 24):
            ov = overlay(d["rgb"], gt, t.tree_id)
            if args.overlays:
                save_image(ov, out / f"{t.tree_id}_overlay.jpg", failed, quality=85)
            if args.figure and len(tiles) < 24:
                tiles.append(np.asarray(ov.resize((320, 240))))
        save_image(Image.fromarray(render_main_branches(gt).astype(np.uint8) * 255),
                   out / f"{t.tree_id}_winter_gt.png", failed)
        rows.append({"tree_id": t.tree_id, **gt.summary()})
    with open(out / "summary.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r)) or ["tree_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if args.figure and tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        save_image(Image.fromarray(grid), Path(args.figure), failed)
    n_ok = sum(r["ok"] for r in rows)
    print(f"Processed {len(rows)} trees: {n_ok} passed the quality check, "
          f"{len(rows) - n_ok} flagged. Results in {out}")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['tree_id']}: {r['issues']}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
