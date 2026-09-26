# Fork shoulder-fill results

Produced by

```
python -m treeskel.measure_fork data/tuning data/sample --out docs/results/fork_shoulders \
    --figure docs/figures/fork_shoulders_before_after.jpg \
    --figure-trees R07N17 R07N25 R07N29 R09N12 R01N32 R01N05 --figure-lengths 1.4 2
```

on the 21 annotated **development-row** trees available locally (the 17
tuning trees plus R01N01, R01N02, R15N05 and R20N06). The four test-row trees
in `data/sample` were left out automatically (test rows 3, 12 and 16 are read
from `splits.json`).

## The problem

The masks looked like "a big cylinder with two branches coming off it". Each
part's width is one straight taper fitted along its whole length, so at the
fork the trunk ends in a round cap (radius R = 11–22 px) and the leaders
(radius 3–5 px, a median of 0.27 R) stick out of the top of it. That leaves a
notch, a "shoulder", on the outer side of each leader. The winter depth
images show wood there: the trunk's edge runs into the leader's outer edge.

## Measure

For each leader, the **shoulder triangle** has corners at the fork, at the
trunk's edge level with the fork, and at the leader's outer edge `reach`
trunk radii up the leader. Coverage is the fraction of the triangle that is
wood (depth silhouette) or mask. Depth "bleeds" past the edges of thin
branches (session 5), which inflates its coverage more the further up the
leader the triangle reaches. So two reaches are used, and the silhouette is
also shrunk by 2 px. The true coverage should lie between the two
silhouettes.

| Mean coverage of the shoulder triangle | reach 1.5 R | reach 2 R |
|---|---|---|
| Depth silhouette (wood) | 0.980 | 0.966 |
| Depth silhouette shrunk 2 px | 0.947 | 0.927 |
| **Masks before (no fill)** | **0.861** | **0.782** |
| Fill, `shoulder_length` 1.0 | 0.892 | 0.804 |
| **Fill, `shoulder_length` 1.4 (chosen)** | **0.987** | **0.885** |
| Fill, `shoulder_length` 1.7 | 1.000 | 0.950 |
| Fill, `shoulder_length` 2.0 | 1.000 | 0.998 |

Leaders whose shoulder is clearly notched (less than 80% covered) at reach
1.5 R: 24% before, 0% with the fill (depth: 2%). At reach 2 R: 62% before, 17%
with the fill (depth: 2%; shrunk depth: 12%).

**Where the fill goes** (pixels added, and the fraction on the depth silhouette):

| `shoulder_length` | Pixels added (21 trees) | On depth | Fork-zone precision | Fork-zone IoU |
|---|---|---|---|---|
| 0 (before) | 0 | – | 0.955 | 0.729 |
| 1.0 | 207 | 0.93 | 0.955 | 0.732 |
| **1.4** | **939** | **0.94** | **0.955** | **0.744** |
| 1.7 | 1654 | 0.93 | 0.954 | 0.754 |
| 2.0 | 2368 | 0.93 | 0.954 | 0.765 |
| 2.5 | 3697 | 0.89 | 0.950 | 0.781 |

The fork zone is the part of the image within 2.5 R of the fork and within R
of a traced path. The added pixels are as precise as the rest of the mask
there (0.94 against 0.955). IoU against depth keeps rising with the fill
only because depth bleeds, so it cannot be used to choose the setting.

## Choosing `shoulder_length`

| Setting at which the fill covers the shoulder as much as | Value | Leave-one-tree-out |
|---|---|---|
| depth, reach 1.5 R | 1.37 | 1.36–1.52 |
| shrunk depth, reach 1.5 R | 1.23 | 1.21–1.29 |
| depth, reach 2 R | 1.80 | 1.78–1.92 |
| shrunk depth, reach 2 R | 1.59 | 1.56–1.68 |
| **Independent check:** the depth width along a leader falls from the fork to the leader's own over | **1.38** R (median of 35 leaders; bootstrap 95% CI 1.05–1.97) | 1.35–1.43 |

**Chosen: 1.4.** It agrees with the width decay, which does not depend on the
fill's shape, and lies inside the 1.23–1.80 range of matching settings. No
single setting matches both reaches. The fill's edge is straight, while the
real edge curves: near the fork the fill covers slightly more than depth
(0.987 against 0.980 at 1.5 R), and further up slightly less (0.885 against
0.927–0.966 at 2 R). Settings up to 2.0 keep 93% of the added pixels on depth,
so the choice is not critical. 2.0 fills visibly more (last column of the
figure) but covers the shoulder more fully than depth does at both reaches.

## Alternatives tried and rejected

Prototyped in the same session (the prototypes are not in the package; the
numbers are from the same 21 trees):

| Method | Result | Why rejected |
|---|---|---|
| Smooth flare on both sides: each leader's radius blended from the trunk's at the fork to its own over 1.4 R (cosine) | Of the pixels added between the leaders (the crotch), only 62% were on depth; 87% on the outer sides | Fills the crotch, which depth shows is a notch |
| The same flare on the outer side only | Synthetic fork of known shape: 41% of added pixels were wood, IoU 0.931 → 0.925 (worse) | Discs as wide as the trunk, centred on a leader that leans outwards, bulge past a straight trunk's edge |
| Round fillet (morphological closing, radius 1.5 R, outer sides) | Synthetic: 100% wood, IoU 0.941; real trees: 89% on depth | Only rounds the sharp corner; the trunk's round cap still shows |
| Convex hull of the trunk's last R px (not just its end disc) and the leader's base | Real trees: 86% on depth, large wedges under wide-angle leaders | The straight edge then starts below the fork |
| **Chosen: convex hull of the trunk's end disc and the leader's first 1.4 R, outer side only** | Synthetic: 100% wood, IoU 0.931 → 0.945; real: 94% on depth | Never reaches outside the hull, so it cannot bulge; crotch untouched |

`docs/figures/fork_fill_alternatives_synthetic.png` shows the synthetic
comparison. `docs/figures/fork_shoulders_before_after.jpg` shows six
development trees: winter RGB, before, the chosen fill (1.4), and 2.0.
Cyan marks the added pixels.

## Related finding, not changed

The trunk is drawn with a **constant width in 15 of 21 trees**. For the first
5–20% of the trunk above the root click, the depth width is near zero, because
the trunk meets the ground there and depth cannot separate them. Those points
tilt the straight-line fit upwards, and the "no widening upwards" rule then
falls back to a constant width. Ignoring widths below half the median makes 6
more trunks taper (9 stay constant). But it did not improve agreement with depth (trunk
precision 0.976 → 0.974) and barely changes the width at the fork (median 15.5
→ 15.2 px), so it was left for a separate decision.

## Caveats

- 21 trees. Confirm on all development rows (about 450 trees) with
  `python -m treeskel.measure_fork data/raw --out outputs/measure_fork`.
- Depth bleeding is only bracketed (raw against 2 px shrink), not modelled.
- The fill's edge is a straight line (a chamfer); real forks curve.
- The synthetic known-answer tree also fills its shoulders with a straight
  chamfer. It shows that the fill adds no false wood and leaves the crotch
  alone, not that a straight edge is the true shape.
- The masks are only filled where a leader leaves the trunk. The crotch
  (between the leaders) keeps the rounded top of the trunk's end disc. The
  crotch height matched depth to within 1.5 px (median) before and after.
