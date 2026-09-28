"""What joining the pieces changed, tree by tree, and how much the result depends on the gap limit.

1. On the masks the join changed only (the other trees are identical in both runs): how many trees each
   measure moved closer to or farther from winter, the MAE before and after, and a Wilcoxon signed-rank
   test on the paired absolute errors. Then the trees whose V opening could be measured only after
   joining, and the trees whose V opening moved most either way.
2. The skill at half and double the chosen gap (9.65 and 38.6 mask px), scored into a temporary folder.
   The gap was fixed before any of these were computed (commit 4103006); they show the sensitivity and
   are not used to choose.

Run from the repository root: python docs/results/segment_lined_up_joined/diagnostics.py
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

from treeskel.segment import JOIN_GAP, joined_run
from treeskel.shape import MEASURES

HERE = Path(__file__).parent
SOURCE = HERE.parent / "segment_lined_up"


def read(path):
    return {(r["tree"], r["measure"]): r for r in csv.DictReader(open(path))}


def value(text):
    return float(text) if text else np.nan


before, after = read(SOURCE / "predictions.csv"), read(HERE / "predictions.csv")
masks = {r["tree"]: r for r in csv.DictReader(open(HERE / "mask_scores.csv"))}
changed = sorted(t for t, r in masks.items() if int(r["pieces_joined"]) > 0)
print(f"1. The {len(changed)} masks the join changed (of {len(masks)})")
print(f"   {'measure':12s} {'n':>3s} {'closer':>6s} {'farther':>7s} {'same':>4s}   MAE before -> after   Wilcoxon p")
for m in MEASURES:
    rows = []
    for t in changed:
        w, a, b = (value(r[(t, m)][c]) for r, c in ((before, "winter"), (before, "segmentation"),
                                                     (after, "segmentation")))
        if np.isfinite(w) and np.isfinite(a) and np.isfinite(b):
            rows.append((abs(a - w), abs(b - w)))
    ea, eb = np.array(rows).T
    d = ea - eb
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"   {m:12s} {len(d):3d} {np.sum(d > 0.05):6d} {np.sum(d < -0.05):7d} {np.sum(abs(d) <= 0.05):4d}"
          f"   {ea.mean():5.2f} -> {eb.mean():5.2f}        {p:.1e}")
print("   (closer / farther: by more than 0.05 degrees; 'no fork' trees count as the fold's mean shape before)")

print("\n   V opening measured only after joining (winter; before: the mean shape or not measured; after):")
for t in changed:
    r0, r1 = before[(t, "v_opening")], after[(t, "v_opening")]
    was = "mean shape" if masks[t]["fork_found_before"] == "0" else "not measured"
    if not r0["segmentation"] or masks[t]["fork_found_before"] == "0":
        if r1["segmentation"] and masks[t]["fork_found"] == "1":
            print(f"   {t}  winter {value(r1['winter']):5.1f}  before {r0['segmentation'] or '-':>5s} ({was})"
                  f"  after {value(r1['segmentation']):5.1f}")

moved = []
for t in changed:
    w, a, b = (value(r[(t, "v_opening")][c]) for r, c in ((before, "winter"), (before, "segmentation"),
                                                           (after, "segmentation")))
    if np.isfinite(w) and np.isfinite(a) and np.isfinite(b):
        moved.append((abs(b - w) - abs(a - w), t, w, a, b))
moved.sort()
print("\n   V opening, the largest changes of error (winter, before, after, change of |error|):")
for d, t, w, a, b in moved[:5] + moved[-5:]:
    print(f"   {t}  fold {masks[t]['fold']}  {w:5.1f}  {a:5.1f}  {b:5.1f}  {d:+5.1f}")

print(f"\n2. Skill over the mean shape at other gaps (chosen: {JOIN_GAP:g} mask px, fixed before scoring)")
runs = [("none", json.loads((SOURCE / "report.json").read_text()))]
with tempfile.TemporaryDirectory() as tmp:
    for gap in (JOIN_GAP / 2, JOIN_GAP, JOIN_GAP * 2):
        out, _ = joined_run(SOURCE, gap, out=Path(tmp) / f"g{gap:g}", progress=lambda m: None)
        runs.append((f"{gap:g}", json.loads((out / "report.json").read_text())))
print(f"   {'gap':>5s} {'joined':>6s} {'no fork':>7s} " + " ".join(f"{m[:11]:>11s}" for m in MEASURES))
for name, rep in runs:
    j = rep.get("join", {})
    print(f"   {name:>5s} {j.get('masks_joined', 0):6d} {rep['no_fork']:7d} "
          + " ".join(f"{rep['scores'][m]['segmentation']['skill']:+11.3f}" for m in MEASURES))
print("   V opening per fold:")
for name, rep in runs:
    print(f"   {name:>5s} " + " ".join(f"{s:+.2f}" for s in rep["scores"]["v_opening"]["segmentation"]["fold_skill"]))
