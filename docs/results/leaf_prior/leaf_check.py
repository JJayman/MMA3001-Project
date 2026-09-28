"""Are the leaders in the middle of the leaves? (the student's idea, session 12)

For each development tree with a lined-up hand mask: leaves are the pixels of the summer photo whose
greenness 2G - R - B exceeds 20; the leaders are the hand mask's, over the fitted stretch of the shape
measures (the part the angles come from), in the photo's pixels. At each point of a leader the photo is
sampled at sideways offsets of -160 to +160 px (positive: outwards, away from the other leader), and the
leaf fraction at each offset is averaged over all leader points of all trees.

If the leaves gather around the leaders, the profile peaks near 0 (apart, perhaps, from a dip on the
leader's own bark) and falls away to either side; if it is flat, the leaves do not tell where a leader
is. The hand masks are a person's reading of the same photos, so where they had to guess they may have
drawn through the leaves; a peak would be weak evidence, a flat profile strong evidence against.

Run on the student's PC: python docs/results/leaf_prior/leaf_check.py data/raw
(writes leaf_check_output.txt beside this file)
"""

import json
import sys
from pathlib import Path

import numpy as np

from treeskel.io import index_dataset, load_tree
from treeskel.segment import hand_mask
from treeskel.shape import MASK_ZOOM, ShapeParams, leader_part, mask_branches, mask_to_image, unit_px
from treeskel.shape_model import load_table
from treeskel.summer_gt import load_overlays

HERE = Path(__file__).parent
OFFSETS = np.arange(-160, 161, 10)
lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


data_dirs = sys.argv[1:] or ["data/raw"]
test_rows = set(json.loads(Path("splits.json").read_text())["test_rows"])
dev = {r["tree"] for r in load_table("data/shape/shape_measures.csv") if r["row"] not in test_rows}
records = {t.tree_id: t for d in data_dirs for t in index_dataset(d).trees if t.tree_id in dev}
overlays = load_overlays("annotations/mask_overlay.json")
params = ShapeParams()
unit = unit_px(0.0, params, MASK_ZOOM)
total, count = np.zeros(len(OFFSETS)), np.zeros(len(OFFSETS))
peaks, used = [], 0
for i, t in enumerate(sorted(records)):
    hm = hand_mask(records[t], overlays)
    br = mask_branches(hm) if hm is not None else None
    if br is None:
        continue
    rgb = load_tree(records[t], "summer")["rgb"].astype(np.int16)
    leaves = (2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) > 20
    h, w = leaves.shape
    tree_total, tree_count = np.zeros(len(OFFSETS)), np.zeros(len(OFFSETS))
    for side, out in (("left", -1), ("right", 1)):
        pts = np.round(mask_to_image(leader_part(br[side], unit, params))).astype(int)
        for j, dx in enumerate(OFFSETS):
            cols = pts[:, 1] + out * dx
            ok = (cols >= 0) & (cols < w) & (pts[:, 0] >= 0) & (pts[:, 0] < h)
            tree_total[j] += leaves[pts[ok, 0], cols[ok]].sum()
            tree_count[j] += ok.sum()
    total += tree_total
    count += tree_count
    frac = tree_total / np.maximum(tree_count, 1)
    away = np.abs(OFFSETS) > 10                                  # leave out the leader's own bark
    peaks.append(int(OFFSETS[away][np.argmax(frac[away])]))
    used += 1
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(records)} trees", flush=True)

profile = total / np.maximum(count, 1)
say(f"Leaf fraction (greenness 2G-R-B > 20) at sideways offsets from the hand-mask leaders, {used} development trees")
say("  offset px (+ outwards)  leaf fraction")
for dx, v in zip(OFFSETS, profile):
    say(f"  {dx:+5d}                 {v:.3f}  {'#' * int(round(v * 100))}")
near = profile[(np.abs(OFFSETS) >= 20) & (np.abs(OFFSETS) <= 40)].mean()
far = profile[np.abs(OFFSETS) >= 120].mean()
say(f"\nMean leaf fraction 20-40 px from the leader {near:.3f}; 120-160 px away {far:.3f} (ratio {near / far:.2f})")
peaks = np.array(peaks)
say(f"Each tree's leafiest offset (beyond 10 px): within 40 px of the leader in {np.mean(np.abs(peaks) <= 40):.0%} "
    f"of trees (by chance, with a flat profile, about {np.mean(np.abs(OFFSETS[np.abs(OFFSETS) > 10]) <= 40):.0%})")
(HERE / "leaf_check_output.txt").write_text("\n".join(lines) + "\n")
