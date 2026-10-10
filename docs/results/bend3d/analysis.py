"""Follow-up analysis of the 449-tree bend3d run, from its saved rows (no images; nothing is chosen here).

1. The V opening: the bent leaders' V against the plane's and the photo's (session 13's
   ``docs/results/tree3d/trees.csv``, winter, main setting).
2. The trunks more than 40 degrees out of the V's plane, and the trunk's angle by orchard row.
3. Summer's trunk against winter's, split by whether model3d draws the summer trunk as seen (less than half
   of it uncertain, bark colour) or as unseen (pale): does the marking say which summer bends to trust?
4. How far summer's trunk angle is shrunk towards the plane.
5. The leaders' bows in summer: bends against the plane.

Run from the repository root:

```
python docs/results/bend3d/analysis.py      # -> docs/results/bend3d/analysis_output.txt
```
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from treeskel.bend3d import load_csv  # noqa: E402

rows = load_csv(ROOT / "docs/results/bend3d/trees.csv")


def col(source, model, measure, setting="main"):
    """One source's and model's measure per tree (fitted, finite)."""
    return {r["tree"]: r[measure] for r in rows if r["source"] == source and r["setting"] == setting
            and r["model"] == model and not r.get("error") and np.isfinite(r.get(measure, np.nan))}


def q(v):
    """Median [quartiles] as text."""
    v = np.asarray(list(v), float)
    return f"{np.median(v):+.2f} [{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}] (n {len(v)})"


out = []

# 1. The V opening: bends, plane, photo.
with open(ROOT / "docs/results/tree3d/trees.csv", newline="") as f:
    t3 = {r["tree"]: r for r in csv.DictReader(f) if r["source"] == "winter" and r["setting"] == "main"}
vb, vp = col("winter", "bent", "v_opening"), col("winter", "plane", "v_opening")
trees = [t for t in vb if t in vp and t in t3 and t3[t]["v_opening_2d"] not in ("", "nan")]
photo = {t: float(t3[t]["v_opening_2d"]) for t in trees}
plane13 = {t: float(t3[t]["v_opening_3d"]) for t in trees}
out += ["1. V opening (winter, main setting; degrees), tree by tree:",
        f"  plane (this run, axis depth) - plane (session 13, surface depth): {q(vp[t] - plane13[t] for t in trees)}",
        f"  plane - photo (session 13's 3D correction):                        {q(vp[t] - photo[t] for t in trees)}",
        f"  bends - plane:                                                     {q(vb[t] - vp[t] for t in trees)}",
        f"  bends - photo:                                                     {q(vb[t] - photo[t] for t in trees)}",
        f"  bends narrower than the photo on {np.mean([vb[t] < photo[t] for t in trees]):.0%} of trees "
        f"(the plane on {np.mean([vp[t] < photo[t] for t in trees]):.0%})"]

# 2. Trunks far out of the V's plane.
tb, tr = col("winter", "bent", "trunk_out"), col("winter ridges", "bent", "trunk_out")
anch = col("winter", "bent", "trunk_anchors")
big = sorted(t for t, v in tb.items() if v > 40)
out += ["", f"2. Trunks more than 40 deg out of the V's plane (winter): {len(big)} of {len(tb)}",
        f"  with ridge readings only, still above 40 deg: {sum(tr.get(t, 0) > 40 for t in big)}; above 20 deg: "
        f"{sum(tr.get(t, 0) > 20 for t in big)}",
        f"  trees: {', '.join(big)}"]

by_row = {}
for r in rows:
    if r["source"] == "winter" and r["setting"] == "main" and r["model"] == "bent" and r["tree"] in tb:
        by_row.setdefault(r["row"], []).append(tb[r["tree"]])
out.append("  by orchard row (median trunk angle, n): " + ", ".join(
    f"{row}: {np.median(v):.1f} ({len(v)})" for row, v in sorted(by_row.items())))

# 3. Summer's trunk against winter's, by whether model3d draws it as seen.
out += ["", "3. Summer's trunk angle against winter's, by how model3d draws the summer trunk "
        "(seen: less than half uncertain):"]
for source in ("U-Net", "hand masks"):
    sb, sp = col(source, "bent", "trunk_out"), col(source, "plane", "trunk_out")
    unc = col(source, "bent", "trunk_uncertain")
    for name, keep in (("seen", lambda u: u < 0.5), ("pale", lambda u: u >= 0.5)):
        k = [t for t in sb if t in tb and t in sp and t in unc and keep(unc[t])]
        if len(k) < 3:
            continue
        w, b, p = (np.array([m[t] for t in k]) for m in (tb, sb, sp))
        eb, ep = np.abs(b - w), np.abs(p - w)
        d = ep - eb
        out.append(f"  {source:10s} {name}: n {len(k):3d}  median |summer - winter| bends {np.median(eb):5.2f}, plane "
                   f"{np.median(ep):5.2f}; bends closer on {np.mean(d > 0):.0%} (Wilcoxon p "
                   f"{stats.wilcoxon(d).pvalue:.1e}); rank correlation {stats.spearmanr(w, b)[0]:.2f}")

# 4. Shrinkage towards the plane.
out += ["", "4. The trunk's angle out of the V's plane (degrees, main setting):",
        f"  winter:              {q(tb.values())}"]
for source in ("hand masks", "U-Net"):
    sb = col(source, "bent", "trunk_out")
    k = [t for t in sb if t in tb]
    slope = stats.theilslopes([sb[t] for t in k], [tb[t] for t in k])[0]
    out.append(f"  summer, {source:10s} {q(sb[t] for t in k)}; Theil-Sen slope on winter's {slope:.2f}")

# 5. The leaders' bows in summer.
out += ["", "5. Leaders' bows (cm): |summer - winter| median, bends against plane, and the summer bends' size:"]
for source in ("hand masks", "U-Net"):
    for side in ("left", "right"):
        wb, sb, sp = (col(s, m, f"{side}_bow") for s, m in (("winter", "bent"), (source, "bent"), (source, "plane")))
        k = [t for t in sb if t in wb and t in sp]
        eb = np.median([abs(sb[t] - wb[t]) for t in k])
        ep = np.median([abs(sp[t] - wb[t]) for t in k])
        out.append(f"  {source:10s} {side:5s}: bends {eb:.2f}, plane {ep:.2f} (difference {eb - ep:+.2f}); "
                   f"|summer bow| median {np.median([abs(sb[t]) for t in k]):.2f}")

print("\n".join(out))
(Path(__file__).parent / "analysis_output.txt").write_text("\n".join(out) + "\n")
