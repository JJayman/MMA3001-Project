"""Figures for the session-8 depth-registration finding (stand-alone, like ``reproduce.py``).

Run from the repository root::

    python docs/results/depth_registration/figures.py

Writes three figures to ``docs/figures/``:

- ``depth_registration_edges.jpg``: sloping leaders with the colour edges
  (red dots) and the depth silhouette's outline (cyan), as loaded and with
  the depth moved up 4 rows and right 1 column;
- ``trunk_width_colour_vs_depth.jpg``: trunks where colour and depth
  disagree (photo, mask, colour edges, depth edges);
- ``recentre_colour_rejected.jpg``: close-ups of leaders re-centred on the
  colour bark centre (red) against the traced path (cyan).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce import HALF, ROOT, REGISTERED, colour_centre_offsets, register, shift_path, silhouette  # noqa: E402

from treeskel.guided_gt import branch_edges, guided_gt, load_clicks, path_normals, straighten  # noqa: E402
from treeskel.io import index_dataset, load_tree  # noqa: E402
from treeskel.measure_width import centre_runs  # noqa: E402
from treeskel.winter_gt import render_main_branches  # noqa: E402

OUT = ROOT / "docs" / "figures"
clicks = load_clicks(ROOT / "annotations" / "winter_clicks.json")
recs = {t.tree_id: t for d in ("data/sample", "data/tuning") for t in index_dataset(ROOT / d).trees}


def label(arr: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(arr)
    ImageDraw.Draw(pil).text((5, 3), text, fill="yellow")
    return np.asarray(pil)


def stack(rows: list[np.ndarray], gap: int = 6) -> Image.Image:
    width = max(r.shape[1] for r in rows)
    return Image.fromarray(np.concatenate([np.pad(r, ((0, gap), (0, width - r.shape[1]), (0, 0))) for r in rows], 0))


def mark_edges(rgb, path, offsets, colour, step=2):
    """Dots at ``path + offset · normal`` for the given per-point offsets (nan = none)."""
    im = rgb.copy()
    n = path_normals(path)
    for i in range(0, len(path), step):
        for off in offsets:
            if np.isfinite(off[i]):
                y, x = np.round(path[i] + off[i] * n[i]).astype(int)
                if 0 <= y < im.shape[0] and 0 <= x < im.shape[1]:
                    im[y, x] = colour
    return im


def edges_figure(specs, scale=5):
    """Leader crops: colour edges (red dots) and the depth silhouette's outline (cyan), as loaded and registered."""
    rows = []
    for tree, side, frac in specs:
        base = load_tree(recs[tree], "winter")
        gt = guided_gt(base["rgb"], base["depth"], base["valid"], clicks[tree])
        p = getattr(gt, side)
        f, l = branch_edges(base["rgb"], p, HALF)
        tiles = []
        for shift, name in (((0, 0), "as loaded"), (REGISTERED, f"depth moved {REGISTERED}")):
            fg = silhouette(register(base, shift))
            im = base["rgb"].copy()
            im[fg & ~ndi.binary_erosion(fg)] = (0, 255, 255)           # the depth silhouette's outline
            im = mark_edges(im, p, (f.astype(float), l.astype(float)), (255, 0, 0), step=1)
            c = p[int(frac * (len(p) - 1))]
            h = 22
            r0 = int(np.clip(c[0] - h, 0, 480 - 2 * h)); c0 = int(np.clip(c[1] - h, 0, 640 - 2 * h))
            crop = im[r0:r0 + 2 * h, c0:c0 + 2 * h]
            big = np.asarray(Image.fromarray(crop).resize((2 * h * scale, 2 * h * scale), Image.NEAREST))
            tiles.append(np.pad(label(big, f"{tree} {side}: {name}"), ((0, 0), (0, 6), (0, 0))))
        rows.append(np.concatenate(tiles, 1))
    return stack(rows)


def trunk_figure(trees, height=260):
    rows = []
    for t in trees:
        w = load_tree(recs[t], "winter")
        gt = guided_gt(w["rgb"], w["depth"], w["valid"], clicks[t])
        p = gt.trunk
        f, l = branch_edges(w["rgb"], p, 40)
        d0, d1, closed = centre_runs(straighten(gt.tree_mask.astype(float), p, 40, order=0) > 0.5)
        m = render_main_branches(gt)
        outline = w["rgb"].copy()
        outline[m & ~ndi.binary_erosion(m)] = (255, 255, 0)
        panels = [w["rgb"], outline, mark_edges(w["rgb"], p, (f.astype(float), l.astype(float)), (255, 0, 0)),
                  mark_edges(w["rgb"], p, (np.where(closed, d0, np.nan), np.where(closed, d1, np.nan)), (0, 255, 255))]
        lo = np.maximum(p.min(0) - 50, 0)
        hi = np.minimum(p.max(0) + 50, (480, 640))
        row = np.concatenate([np.pad(x[lo[0]:hi[0], lo[1]:hi[1]], ((0, 0), (0, 3), (0, 0))) for x in panels], 1)
        s = height / row.shape[0]
        row = np.asarray(Image.fromarray(row).resize((int(row.shape[1] * s), height), Image.NEAREST))
        rows.append(label(row, f"{t}: photo | mask | colour edges (red) | depth edges (cyan)"))
    return stack(rows)


def recentre_figure(specs, scale=6):
    rows = []
    for tree, side, picks in specs:
        w = load_tree(recs[tree], "winter")
        gt = guided_gt(w["rgb"], w["depth"], w["valid"], clicks[tree])
        other = gt.right if side == "left" else gt.left
        p = getattr(gt, side)
        off, _ = colour_centre_offsets(w["rgb"], gt, side, other)
        q = shift_path(p, off)
        tiles = []
        for frac in picks:
            i = int(frac * (len(p) - 1))
            h = 16
            r0 = int(np.clip(p[i][0] - h, 0, 480 - 2 * h)); c0 = int(np.clip(p[i][1] - h, 0, 640 - 2 * h))
            crop = w["rgb"][r0:r0 + 2 * h, c0:c0 + 2 * h].copy()
            marked = crop.copy()
            for path, col in ((p, (0, 255, 255)), (q, (255, 0, 0))):
                sel = (path[:, 0] >= r0) & (path[:, 0] < r0 + 2 * h) & (path[:, 1] >= c0) & (path[:, 1] < c0 + 2 * h)
                marked[path[sel, 0] - r0, path[sel, 1] - c0] = col
            pair = np.concatenate([crop, np.zeros((2 * h, 1, 3), np.uint8), marked], 1)
            big = np.asarray(Image.fromarray(pair).resize((pair.shape[1] * scale, pair.shape[0] * scale), Image.NEAREST))
            tiles.append(np.pad(label(big, f"{tree} {side}, shift {off[i]:+.1f} px"), ((0, 0), (0, 8), (0, 0))))
        rows.append(np.concatenate(tiles, 1))
    return stack(rows)


if __name__ == "__main__":
    edges_figure([("R15N05", "right", 0.35), ("R09N18", "left", 0.4), ("R07N25", "right", 0.45),
                  ("R07N09", "left", 0.5)], scale=4).save(OUT / "depth_registration_edges.jpg", quality=90)
    trunk_figure(["R01N05", "R20N06", "R01N18", "R09N34"]).save(OUT / "trunk_width_colour_vs_depth.jpg", quality=90)
    recentre_figure([("R15N05", "right", (0.3, 0.5, 0.8)), ("R07N26", "right", (0.15, 0.6, 0.75))]).save(
        OUT / "recentre_colour_rejected.jpg", quality=90)
    print("Saved the three figures in", OUT)
