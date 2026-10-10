# The typical winter shape as the prior's mean

Session 15, the student's question after the bends became `model3d`'s default: the summer models' leaders
look nearly straight, although the winter trees show a clear pattern (the leaders go away from the camera
low down and come back towards it near the top; the trunk stands about 14° out of the V's plane). Two
reasons were found (`docs/results/bend3d/`):

1. **Summer does not see the leaders** (2–3% of their readings count), and where nothing is read the bends
   return to the prior's mean, which was the flat plane. The pattern lives only in the winter data.
2. **The real bends are small:** a winter leader gets a median 4.1–4.3 cm from the plane along 1.4 m, which
   looks nearly straight at true scale.

The fix for the first: make the prior's mean the **typical winter shape** instead of the plane, so that
where summer sees nothing the model draws the typical tree. For the second, `model3d`'s viewer gets a
display-only slider that exaggerates the bends.

**Status: done on the 449 development trees** (the student's PC, 10 October). The method, the checks and the
rule were committed before the run (31bc667, merged in PR #58); the run's files are `trees.csv`,
`typical_shape.json`, `profiles.npz`, `report.json`, `output.txt` and `docs/figures/bend3d_typical.png`, and
`--from-saved` reproduces `output.txt` and `report.json` exactly. **The rule passed: `model3d build` now uses
the typical shape as the prior's mean.**

## Results (449 development trees)

**The typical winter shape** (median bend behind the tree's plane; quartiles in the figure):

| Part | At the fork | 0.2 m | 0.4 m | 0.6 m | 0.8 m | 1.0 m | 1.2 m | 1.4 m | 1.6 m |
|---|---|---|---|---|---|---|---|---|---|
| Trunk (downwards) | −0.9 cm | +2.8 | +3.1 | | | | | | |
| Left leader | −0.9 cm | −2.1 | −1.2 | +0.7 | +1.5 | +0.8 | −0.1 | −0.8 | −1.1 |
| Right leader | −0.9 cm | −3.2 | −1.9 | +0.0 | +0.9 | +0.9 | +0.3 | −0.2 | −0.4 |

The typical leader is an **S**: just above the fork it comes 2–3 cm towards the camera, then goes away to
1–1.5 cm behind the plane around 0.8–1.0 m, and comes back towards the camera near the top. The trunk's root end
lies about 3 cm behind the plane.

**Summer against winter's bends** (median |summer − winter|; the shape learnt without the tree's own fold):

| U-Net masks | Plane | Bends, plane as mean | Typical shape alone | **Bends, typical shape as mean** | Nearer than the plane-mean bends |
|---|---|---|---|---|---|
| Trunk out of the V's plane | 14.5° | 12.0° | 7.0° | **8.5°** | 78% (p 9e-26) |
| Leader tilts on the stretch (both) | 4.2° | 4.5° | 2.7° | **3.0°** | 76% (p 3e-49) |
| Whole leaders' tilts (left / right) | 1.0° / 1.6° | 1.5° / 1.9° | 1.2° / 1.3° | **1.4° / 1.4°** | 60% / 68% |
| V opening | 3.00° | 3.01° | 2.76° | **2.79°** | 64% (p 6e-8) |
| Bows (left / right) | 1.73 / 1.63 cm | 1.85 / 1.71 | 1.78 / 1.63 | 1.80 / 1.82 | 49% / 45% (n.s.) |

The hand masks agree (trunk 9.2° against 12.5°, 75%; tilts 2.9° against 4.6°, 77%).

- **The rule passes** on both tests, by a wide margin; the V opening improves too, and the bows do not change.
- **The typical shape alone is nearer still on the angles** (trunk 7.0°, tilts 2.7°; nearer than the adopted
  model on 52–55% of trees, p 0.02 to 3e-4). Summer's readings carry information about the individual tree (the
  trunk's angle follows winter's with a rank correlation of 0.38 with them, 0.49 where the trunk is seen,
  against 0.10 for the typical shape, which is the same for every tree), but they also pull the angles low (trunk
  median 10.9° against winter's 14.3°; tilts 2.8–3.3° against 3.7–4.2°), whose cause is not known (possibly the
  summer plane, fitted through the leaves). The adopted model is the one that shows each tree's own shape; the
  typical shape alone would show every tree as the average one. The rule compared the adopted model with the
  plane-mean bends, as fixed before the run; the comparison with the shape alone is reported, not used.
- **Failed planes:** as in `docs/results/bend3d/` (4 winter, 1 hand mask, 9 U-Net masks).



## Method (`treeskel.bend3d`, `--typical`)

- **The winter profiles.** Every development tree's winter bends are fitted as in `docs/results/bend3d/`
  (prior τ 2 cm, L 20 cm) and sampled every 2 cm along each part from the fork (`profile`): metres along the
  part on the tree's plane, the trunk downwards to 0.8 m, each leader upwards to 2 m.
- **The typical shape** (`typical_shape`): at each station, the median over the trees that reach it; a
  part's stations stop before the first one fewer than 20 trees reach. The three parts share the fork's
  median. Left and right leaders keep their own profiles (they differ: the right tilts more).
- **As the prior's mean** (`prior_mean`, `fit_bends(mean=...)`): each node gets its part's typical bend at
  its distance from the fork (the last value held beyond the stations), and the bends solve
  `(Q + W/σ²) h = W d/σ² + Q m`. Where summer reads the wood the data lead as before; where it reads
  nothing the bend is the typical shape, still marked uncertain (the posterior standard deviation does not
  depend on the mean).
- **Out of fold:** each tree's summer is fitted with the shape learnt from the other four folds' trees only,
  so no tree's own winter shape reaches its summer model.
- **Four summer models per tree** (the U-Net's mask and the lined-up hand mask, ridge readings only):
  `plane` (flat), `bent` (the bends with the plane as the mean: `model3d` until now), `typical` (the typical
  shape alone, ignoring summer's readings) and `bent typical` (the bends with the typical shape as the mean).
  Each is measured as in `docs/results/bend3d/` and compared with the same tree's winter bends.

## Checks with known answers (`tests/test_bend3d.py`)

- With nothing read, the bends equal the mean exactly; with every reading, a tree's own shape as the mean
  still leaves the data in charge (within 1 cm of the truth).
- The typical shape is the per-station median, stops where too few trees reach, and shares the fork.
- Synthetic trees sharing a bent leader, with only the trunk read (as summer): the typical shape as the mean
  recovers the unread bow at less than half the plane's error.

## The rule, fixed before the 449-tree run

Committed with this README, before any of the 449 trees was fitted for it. Development rows only; the test
rows are not loaded.

`model3d` uses the typical shape as the prior's mean if, with the **U-Net's masks** (what `model3d` uses) and
the shape learnt **out of fold**, the bends with it (`bent typical`) are nearer the tree's winter bends than
the bends with the plane as the mean (`bent`) on more than half of the trees, with a Wilcoxon signed-rank
p < 0.05, on **both**:

1. the trunk's angle out of the V's plane, and
2. the leaders' tilts on their stretch (left and right pooled, one pair per leader).

`summarise_typical` applies it (`report.json`: `rule`), and `model3d build` reads the verdict: the typical
shape is used only if it passed (`--plane-prior` keeps the plane either way). Every other measure (the
whole leaders' tilts, the V opening, the bows) and the hand masks are reported whatever they show.

**Why these two:** they are where the typical shape should help (the estimate from the saved 449-tree rows,
with the typical value alone, was 7.0° against 12.0° for the trunk and 2.7° against 4.4–4.6° for the tilts).
The bows vary from tree to tree and the estimate showed no gain, so a typical shape may make them worse; that
is reported, not tested.

## Smoke run (21 trees in the repository; not a result; as written before the run)

The command on `data/sample` and `data/tuning` with the 20-tree minimum lowered to 5 (each fold's shape then
comes from about 10 trees), only to check that it runs on real photos. The learnt shape already shows the
pattern: the leaders start 2.3 cm in front of the plane at the fork, 2.4–3.3 cm in front 20 cm up,
1.1–1.9 cm behind around 0.8 m, and back to 1.5–2.9 cm in front at 1.5 m (right and left). With the U-Net's masks the trunk improved
(8.5° against 12.8°, nearer on 74%, p 0.007), the leader tilts improved without significance (3.7° against
4.9°, 68%, p 0.11), and the left bow got worse (4.2 against 2.7 cm). These trees are mostly the session-4
problem trees and row 1, which misled before; the 449-tree run decides.

## The run (student's PC)

```bash
python -m treeskel.bend3d data/raw --typical
python -m treeskel.bend3d --typical --from-saved docs/results/bend3d_typical --figure ""   # the summary again
```

About 5–10 minutes (winter and summer fits only, no cross-validation over priors; the last estimate here
was half the real time). It writes `trees.csv`,
`typical_shape.json`, `profiles.npz`, `report.json` and `output.txt` here and `docs/figures/bend3d_typical.png`.
Push those files; `python -m treeskel.bend3d --typical --from-saved docs/results/bend3d_typical` remakes the
summary without images.
