"""Two checks on a U-Net run's saved outputs (the first run, on the masks as drawn, by default).

1. Does it reproduce the angles of the hand masks it was trained on? The correlation of its measures
   with the hand masks' and with the winter traces', and the spread of each (all 449 development trees,
   from predictions.csv).
2. How much of its low Dice against the hand masks is the masks' offset? Dice as it is, and after the
   best shift of the prediction onto the hand mask (within ±40 mask px across, ±20 down), on the
   development trees whose hand masks are in the repository. Picking the best of 861 shifts flatters
   the second number, so it is an upper estimate.

Run from the repository root: python docs/results/segment/diagnostics.py [results folder]
(e.g. docs/results/segment_lined_up; its Dice is then against the drawn masks, as the first run's.)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from treeskel.io import index_dataset, load_mask
from treeskel.segment import load_masks, mask_scores

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
MEASURES = ("v_opening", "left_bend", "right_bend", "trunk_tilt", "left_tilt", "right_tilt", "left_angle",
            "right_angle")

by = defaultdict(dict)
for r in csv.DictReader(open(HERE / "predictions.csv")):
    by[r["measure"]][r["tree"]] = r


def column(m, col):
    return np.array([float(x[col]) if x[col] else np.nan for x in by[m].values()])


print("1. Spread (SD, degrees) and correlations, trees with all three known")
print(f"   {'measure':12s} {'winter':>7s} {'hand':>6s} {'U-Net':>6s}   r(U-Net, winter)  r(U-Net, hand)  r(hand, winter)")
for m in MEASURES:
    w, h, s = column(m, "winter"), column(m, "hand masks"), column(m, "segmentation")
    ok = np.isfinite(w) & np.isfinite(h) & np.isfinite(s)
    r = lambda a, b: np.corrcoef(a[ok], b[ok])[0, 1]                    # noqa: E731
    print(f"   {m:12s} {w[ok].std():7.1f} {h[ok].std():6.1f} {s[ok].std():6.1f}   {r(s, w):+16.2f}  {r(s, h):+14.2f}"
          f"  {r(h, w):+15.2f}")

pred = load_masks(HERE / "predicted_masks.npz")
test_rows = set(json.loads(Path("splits.json").read_text())["test_rows"])
recs = [t for d in ("data/sample", "data/tuning") for t in index_dataset(d).trees
        if t.tree_id in pred and t.has_mask and t.row not in test_rows]
print(f"\n2. Dice against the hand mask, as predicted and after the best shift ({len(recs)} trees)")
before, after = [], []
for rec in recs:
    h, p = load_mask(rec.paths[("summer", "mask")]), pred[rec.tree_id]
    d0 = mask_scores(p, h)["dice"]
    d1, dy, dx = max((mask_scores(np.roll(np.roll(p, dy, 0), dx, 1), h)["dice"], dy, dx)
                     for dy in range(-20, 21, 2) for dx in range(-40, 41, 2))
    before.append(d0)
    after.append(d1)
    print(f"   {rec.tree_id}  {d0:.2f} -> {d1:.2f}  shift ({dx * 480 / 256:+4.0f}, {dy * 480 / 256:+4.0f}) photo px")
print(f"   median {np.median(before):.2f} -> {np.median(after):.2f}")
