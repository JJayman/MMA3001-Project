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
   The fork is where the two leader paths separate.
5. **Quality check.** A leader that rises less than a set fraction of the
   root's height is marked as failed (usually a depth gap), and the tree is
   flagged so it can be left out rather than giving a wrong label.

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


def find_main_branches(tree: np.ndarray, skel: np.ndarray, params: GTParams = GTParams()) -> dict:
    """Pick the root, fork and the two leaders from a tree skeleton.

    Args:
        tree: Tree region (used to locate the trunk base).
        skel: Skeleton of the tree (connected).
        params: Settings (``wander_penalty``).

    Returns:
        Dict with ``root``, ``fork`` (``(row, col)`` or ``None``) and ``trunk``,
        ``left``, ``right`` paths (``(N, 2)`` arrays, possibly empty).
    """
    empty = np.zeros((0, 2), int)
    out = {"root": None, "fork": None, "trunk": empty, "left": empty, "right": empty}
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

    left = [e for e in ends if xs[e] < xs[root]]
    right = [e for e in ends if xs[e] >= xs[root]]
    if not left or not right:
        return out
    pl, pr = path(max(left, key=score)), path(max(right, key=score))
    shared = set(pr)
    k = max(i for i, v in enumerate(pl) if v in shared)  # last shared node = fork
    fork = pl[k]
    kr = pr.index(fork)
    coords = np.stack([ys, xs], 1)
    out.update(
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

    br = find_main_branches(tree, sk, params)
    gt = WinterGT(tree, sk, **br)
    if gt.fork is None:
        gt.issues.append("could not find one leader on each side")
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

    The radius at each path point is the distance from that point to the edge
    of the tree region, clipped to ``[min_radius, max_radius]``. Linked gaps
    (outside the tree region) get ``min_radius``.

    Args:
        gt: Result of :func:`extract_winter_gt`.
        min_radius: Smallest drawn radius in pixels.
        max_radius: Largest drawn radius in pixels.

    Returns:
        Bool mask the same shape as ``gt.tree_mask``.
    """
    shape = gt.tree_mask.shape
    radius = np.clip(ndi.distance_transform_edt(gt.tree_mask), min_radius, max_radius)
    out = np.zeros(shape, bool)
    for p in (gt.trunk, gt.left, gt.right):
        for y, x in p:
            rr, cc = draw_disk((y, x), radius[y, x], shape=shape)
            out[rr, cc] = True
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
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trees = index_dataset(args.data_dir).trees[: args.limit]
    params = GTParams()
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))
    rows, tiles = [], []
    for t in trees:
        d = load_tree(t, "winter")
        gt = extract_winter_gt(d["depth"], d["valid"], params)
        if args.figure and len(tiles) < 24:
            tiles.append(np.asarray(overlay(d["rgb"], gt, t.tree_id).resize((320, 240))))
        Image.fromarray(render_main_branches(gt).astype(np.uint8) * 255).save(
            out / f"{t.tree_id}_winter_gt.png")
        rows.append({"tree_id": t.tree_id, **gt.summary()})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["tree_id"])
        w.writeheader()
        w.writerows(rows)
    if args.figure and tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        Image.fromarray(grid).save(args.figure)
    n_ok = sum(r["ok"] for r in rows)
    print(f"Processed {len(rows)} trees: {n_ok} passed the quality check, "
          f"{len(rows) - n_ok} flagged. Results in {out}")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['tree_id']}: {r['issues']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
