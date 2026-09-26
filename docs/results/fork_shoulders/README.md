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
from `splits.json`). The chosen setting was then checked on all 449
development-row trees on the student's PC (see [Check on all development
rows](#check-on-all-development-rows-449-trees)).

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
of a traced path. On these 21 trees the added pixels were as precise as the
rest of the mask there (0.94 against 0.955); on all development trees they
are slightly less so (0.91 against 0.966, see below). IoU against depth keeps
rising with the fill only because depth bleeds, so it cannot be used to
choose the setting.

## Choosing `shoulder_length`

| Setting at which the fill covers the shoulder as much as | Value | Leave-one-tree-out |
|---|---|---|
| depth, reach 1.5 R | 1.37 | 1.36–1.52 |
| shrunk depth, reach 1.5 R | 1.23 | 1.21–1.29 |
| depth, reach 2 R | 1.80 | 1.78–1.92 |
| shrunk depth, reach 2 R | 1.59 | 1.56–1.68 |
| **Independent check:** the depth width along a leader falls from the fork to the leader's own over | **1.38** R (median of 35 leaders; bootstrap 95% CI 1.05–1.97) | 1.35–1.43 |

**Chosen on these 21 trees: 1.4.** It agreed with the width decay here (1.38),
which does not depend on the fill's shape, and lies inside the 1.23–1.80 range
of matching settings. (On all development trees the width decay came out
higher; see the next section.) No single setting matches both reaches. The fill's edge is straight, while the
real edge curves: near the fork the fill covers slightly more than depth
(0.987 against 0.980 at 1.5 R), and further up slightly less (0.885 against
0.927–0.966 at 2 R). Settings up to 2.0 keep 93% of the added pixels on depth,
so the choice is not critical. 2.0 fills visibly more (last column of the
figure) but covers the shoulder more fully than depth does at both reaches.

## Check on all development rows (449 trees)

Run on the student's PC after 1.4 had been chosen, with
`python -m treeskel.measure_fork data/raw --out outputs/measure_fork` (console
output in `all_dev_rows_output.txt`; test rows left out as before):

| | Coverage, reach 1.5 R | reach 2 R | Notched (< 80%), 1.5 R | 2 R |
|---|---|---|---|---|
| Depth silhouette (wood) | 0.965 | 0.950 | 5% | 8% |
| Depth silhouette shrunk 2 px | 0.924 | 0.898 | 13% | 19% |
| Masks before (no fill) | 0.850 | 0.764 | 30% | 66% |
| **Fill, `shoulder_length` 1.4** | **0.984** | **0.872** | **0%** | **23%** |
| Fill, `shoulder_length` 1.7 | 1.000 | 0.946 | 0% | 0% |
| Fill, `shoulder_length` 2.0 | 1.000 | 0.997 | 0% | 0% |

- Settings at which the fill covers the shoulder as much as depth: 1.33
  (depth, 1.5 R), 1.17 (shrunk, 1.5 R), 1.72 (depth, 2 R) and 1.51 (shrunk,
  2 R); their mean is 1.43. Every leave-one-tree-out range is within 0.02.
- The fill adds 22,650 px, 91% of them on the depth silhouette. Around the
  fork the mask's precision against depth is 0.966 before and 0.965 after;
  IoU rises from 0.739 to 0.754.
- Width decay: 1.59 trunk radii (median of 811 leaders; bootstrap 95% CI
  1.50–1.68).

Two statements made from the 21 trees did not hold, and are corrected here:

- The width decay (1.38 on 21 trees) is 1.59 on all development trees, and
  its confidence interval excludes 1.4. So 1.4 no longer agrees with it;
  with more data it suggests the transition is slightly longer.
- The added pixels (94% on depth on 21 trees) are 91% on depth, a little
  below the 96.6% of the rest of the mask around the fork.

**Kept at 1.4** (the student's decision). It is still near the centre of the
direct measure of the notch (the four matching settings range from 1.17 to
1.72, mean 1.43), and 1.5–1.6 would reach only 1–3 px further up the leader.
The setting was chosen on 21 trees and then checked on all 449 development
trees (which include those 21); no test-row tree was used.

## Re-check with the corrected leader widths (session 8)

In session 8 the leaders turned out to have been drawn about half as wide as
their bark (see `docs/results/leader_width/`). With their true widths
(median 0.27 R before, about 0.5 R now) most of the notch the fill was made
for has gone. `measure_fork` re-run on the same 21 trees with the new widths
(`new_leader_widths/`, figure `docs/figures/fork_shoulders_new_leader_widths.jpg`):

| Mean coverage of the shoulder triangle | reach 1.5 R | reach 2 R |
|---|---|---|
| Depth silhouette (wood) | 0.954 | 0.943 |
| Depth silhouette shrunk 2 px | 0.896 | 0.879 |
| **New masks, no fill** | **0.936** | **0.912** |
| New masks, fill 1.0 | 0.967 | 0.935 |
| New masks, fill 1.4 (current) | 0.998 | 0.964 |

- Without any fill the new masks lie between the two depth references, the
  criterion used above. Session 8 also measured the depth overshoot
  directly: a median 2.0 px past clear bark edges, so the shrunk silhouette is
  the better reference, and against it no fill is needed.
- Matching settings: 0.58 (depth, 1.5 R; leave-one-out 0.36–1.03) and 1.12
  (depth, 2 R; 1.00–1.36); against the shrunk depth, below 0.
- The pixels the 1.4 fill adds are 81% on depth (94% before on these trees).
- **All 449 development trees** (student's PC, `new_leader_widths/all_dev_rows_output.txt`):
  the same picture. No fill: 0.914 (1.5 R) and 0.879 (2 R), between depth
  (0.938, 0.914) and shrunk depth (0.875, 0.840); fill 1.4: 0.996 and 0.948,
  above depth at both reaches. Matching settings 0.66 and 1.08 against depth
  (leave-one-out 0.64–0.72 and 1.07–1.10), below 0 against the shrunk depth.
  The added pixels are 79% on depth (91% in the session-6 check).
- **Decision (the student's, session 8): the fill is turned off "for the
  moment"** (`shoulder_length = 0`). Of the three options offered (keep 1.4,
  about 1.0, or off), off matches the depth silhouette corrected for its
  measured overshoot, the reference now trusted; 1.4 remains available in
  the code and restores the fill.

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

The trunk is drawn with a **constant width in 15 of 21 trees** (measured on
the 21 local trees only). For the first
5–20% of the trunk above the root click, the depth width is near zero, because
the trunk meets the ground there and depth cannot separate them. Those points
tilt the straight-line fit upwards, and the "no widening upwards" rule then
falls back to a constant width. Ignoring widths below half the median makes 6
more trunks taper (9 stay constant). But it did not improve agreement with depth (trunk
precision 0.976 → 0.974) and barely changes the width at the fork (median 15.5
→ 15.2 px), so it was left for a separate decision.

## Caveats

- The setting was chosen on 21 trees and checked on 449. The per-leader data
  of the full check stay on the student's PC (`outputs/measure_fork`); only
  its console output is in the repository.
- Depth bleeding is only bracketed (raw against 2 px shrink), not modelled.
- The fill's edge is a straight line (a chamfer); real forks curve.
- The synthetic known-answer tree also fills its shoulders with a straight
  chamfer. It shows that the fill adds no false wood and leaves the crotch
  alone, not that a straight edge is the true shape.
- The masks are only filled where a leader leaves the trunk. The crotch
  (between the leaders) keeps the rounded top of the trunk's end disc. The
  crotch height matched depth to within 1.5 px (median) before and after.
