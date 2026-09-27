"""The hand masks' line-up (treeskel.annotate_mask): the shifts, and how repeatable they are.

The student lined up all 449 development masks (annotations/mask_overlay.json), then redid 20
randomly chosen trees (seed 11) into a separate file, starting from no shift and without the saved
shifts (annotations/mask_overlay_repeat.json). The difference between the two is the line-up error.

Run from the repository root: python docs/results/segment/lineup.py
"""

import json
from collections import defaultdict

import numpy as np

from treeskel.summer_gt import load_overlays

first = load_overlays("annotations/mask_overlay.json")
repeat = load_overlays("annotations/mask_overlay_repeat.json")
test_rows = set(json.load(open("splits.json"))["test_rows"])
ok = {t: v for t, v in first.items() if not v.skip}
dx, dy = np.array([v.dx for v in ok.values()]), np.array([v.dy for v in ok.values()])
mag = np.hypot(dx, dy)
secs = np.array([v.seconds for v in first.values()])
print(f"Lined up: {len(first)} masks ({sum(int(t[1:3]) in test_rows for t in first)} from test rows), "
      f"{len(first) - len(ok)} skipped, flagged: {[t for t, v in first.items() if v.mask_flag]}")
print(f"Time: {secs.sum() / 60:.0f} min, median {np.median(secs):.1f} s per tree")
print(f"Across (dx): median {np.median(dx):+.1f} px, middle half {np.percentile(dx, 25):+.1f} to "
      f"{np.percentile(dx, 75):+.1f}, range {dx.min():+.0f} to {dx.max():+.0f}")
print(f"Down (dy):   median {np.median(dy):+.1f} px, middle half {np.percentile(dy, 25):+.1f} to "
      f"{np.percentile(dy, 75):+.1f}, range {dy.min():+.0f} to {dy.max():+.0f}")
print(f"Size of the shift: median {np.median(mag):.1f} px; over 3 px {np.mean(mag > 3):.0%}, over 20 px "
      f"{np.mean(mag > 20):.0%}, over 40 px {np.mean(mag > 40):.0%}; scaled: {np.mean([v.scale != 1 for v in ok.values()]):.0%}")
by = defaultdict(list)
for t, v in ok.items():
    by[int(t[1:3])].append((v.dx, v.dy))
print("Per row: trees, median dx, SD dx (px)")
for r in sorted(by):
    a = np.array(by[r])
    print(f"  row {r:2d}: {len(a):3d}  {np.median(a[:, 0]):+6.1f}  {a[:, 0].std(ddof=1):5.1f}")

pairs = [(first[t], v) for t, v in sorted(repeat.items()) if not v.skip and not first[t].skip]
ddx = np.array([b.dx - a.dx for a, b in pairs])
ddy = np.array([b.dy - a.dy for a, b in pairs])
d = np.hypot(ddx, ddy)
print(f"\nRepeatability ({len(pairs)} trees redone from no shift):")
print(f"  difference across: median |.| {np.median(np.abs(ddx)):.1f} px, mean {ddx.mean():+.1f}; "
      f"down: median |.| {np.median(np.abs(ddy)):.1f} px")
print(f"  distance: median {np.median(d):.1f} px, 90th percentile {np.percentile(d, 90):.1f}, max {d.max():.1f}")
print(f"  SD of one line-up (difference SD / sqrt 2): across {ddx.std(ddof=1) / np.sqrt(2):.1f} px, "
      f"down {ddy.std(ddof=1) / np.sqrt(2):.1f} px")
print(f"  correlation of the two shifts across: r = {np.corrcoef([a.dx for a, _ in pairs], [b.dx for _, b in pairs])[0, 1]:.3f}")
