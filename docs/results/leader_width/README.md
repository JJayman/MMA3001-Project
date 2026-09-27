# Leader widths (session 8)

Produced by

```
python -m treeskel.measure_width data/tuning data/sample --out docs/results/leader_width \
    --figure docs/figures/leader_width_cross_sections_before_after.jpg --figure-trees R07N25 R01N32 \
    --plot docs/figures/leader_width_profiles.png
```

on the 21 annotated **development-row** trees available locally (the 17
tuning trees plus R01N01, R01N02, R15N05 and R20N06). The four test-row trees
in `data/sample` were left out automatically (test rows 3, 12 and 16 are read
from `splits.json`). Console output: `output_21_dev_trees.txt`; per-leader
numbers: `leaders.csv`; everything else: `report.json`. These files and the
figures are from the **final settings** (the widening fading over 12 trunk
radii, no shoulder fill); where the first design (4 trunk radii, fill 1.4)
matters, the text says so. The checks on all 449 development trees are
further down. They were made before the fork junction replaced the trunk's
round top (`docs/results/fork_junction/`): re-run now, the widths come out
the same, but the crotch counts ("after") no longer include the round top.

## The problem (seen by the student)

The branches were "slightly too thin", and the masks looked odd because they
went "from really thick at the trunk to two quite thin leaders". In the
provided hand masks the width changes from the trunk to the leaders more
naturally, and the student expected the leaders to narrow slightly along
their length, "not much".

## Diagnosis

- **Cross-sections across the leaders** (colour sampled along the normal to
  the traced path): the bark was 13–17 px wide where the mask drew 7.4 px.
  The drawn width covered the **sunlit half** of each leader; the shaded half
  was left out.
- **Cause.** The leader widths came from scale-space ridge detection
  (`scale_space_radius`, session 5). It responds to the bright ridge of a
  branch. Its known-answer test used bark that was brighter than the
  background right to its edges. Real bark is lit from one side, and its
  edges are darker than its middle. On synthetic branches shaded that way
  (a matt, Lambertian cylinder) it returns 0.55–0.6 of the half-width.
- **In proportion to the trunk**, which needs no winter–summer alignment
  (leader half-width ÷ trunk half-width at the fork, 3–5 trunk radii up the
  leaders, 42 leaders): hand masks 0.55–0.56, depth silhouette 0.55–0.57,
  masks before 0.25–0.26.

## The new measurement

`treeskel.guided_gt.branch_edges`: the colour image is sampled across each
leader (±20 px, `straighten`), each cross-section is averaged with its 3
neighbours on either side, and it is split into **background | branch |
background** by least squares (`profile_edges`): a constant colour on each
side and a quadratic across the branch, which follows the shading from the lit
to the shaded side. Every pair of edges is tried; each piece's squared error
comes from cumulative sums, so all pairs are scored at once (about 80 ms per
leader).

- **Known answer:** on synthetic side-lit branches (half-width 3–9 px, lit
  from the front or either side, path up to 2 px off-centre) the half-width
  was right to within 0.5 px in 47 of 48 cases, and within 1 px in the last
  (a front-lit branch whose dark rim was as bright as the background).
- **Neighbour rule:** near the fork a cross-section across one leader also
  takes in the other. On a synthetic tree whose leaders have constant width,
  this made them look 1.7–2.4 times too wide up to 3 trunk radii from the
  fork. Cross-sections that could reach the other leader (within 30 px of its
  path) are therefore not used (`clear_of`). On the development trees the
  first usable one is a median 2.1 trunk radii from the fork (IQR 1.8–2.7).
- Each leader is drawn at its **own width**, a robust (Huber) average of its
  cross-sections at least 1.5 trunk radii from the fork, times a **widening
  near the fork** shared by all leaders (`fit_leader_radius`, next section).
- The trunk is unchanged (depth, straight taper).

## Checks against independent references

**Depth overshoot.** Depth is an independent sensor, but its silhouette
reaches past branch edges. Where a depth run ends in "no reading" on both
sides and the colour edge is clear (a change of more than 60 grey levels),
the depth edge lies a median **+2.0 px** outside the colour edge (IQR 0 to
+4; 4,394 edges). Session 6 bracketed this with a 2 px shrink; this is the
first direct measurement.

**Each leader, at least 8 trunk radii from the fork** (42 leaders, half-width in px):

| | Median | Minus depth (minus its overshoot) |
|---|---|---|
| Masks before | 3.1 | −3.8 (IQR −4.6 to −3.1) |
| **Masks after** | **6.7** | **−0.5 (IQR −1.7 to +0.8)** |
| Depth minus overshoot | 7.2 | – |

**Along the leaders** (`docs/figures/leader_width_profiles.png`; half-width ÷
trunk radius at the fork, medians over leaders; "–": too near the fork to
measure without the other leader):

| Trunk radii from the fork | 0–0.5 | 1–1.5 | 1.5–2 | 3–4 | 5–7 | 10–15 | 15–25 |
|---|---|---|---|---|---|---|---|
| Hand masks (summer) | 0.88 | 0.61 | 0.60 | 0.56 | 0.55 | 0.46 | 0.39 |
| Depth minus overshoot | – | – | 0.43 | 0.45 | 0.45 | 0.52 | 0.46 |
| Measured cross-sections | – | – | 0.45 | 0.52 | 0.47 | 0.44 | 0.42 |
| **Masks after** | 0.54 | 0.53 | 0.53 | 0.51 | 0.49 | 0.45 | 0.43 |
| Masks before | 0.26 | 0.26 | 0.26 | 0.26 | 0.25 | 0.23 | 0.22 |

The hand masks are drawn on the summer photos and are a little wider than
the winter bark measured by either sensor; their proportions need no
alignment, which is why they can be compared at all.

**By eye:** `docs/figures/leader_width_cross_sections_before_after.jpg`
(R07N25 and R01N32: the whole tree before and after, then crops at the fork,
30% and 60% up each leader). All 21 trees were also checked on a contact
sheet: every leader covers its branch, none is drawn wider than its bark
by more than a pixel or two, and the joins at the fork look natural.

## Along the leader: taper and widening

*The first two statements below hold for these 21 trees but not for all 449
development trees; see the next section.*

- **A straight taper per leader is not reliable.** Fitted freely to each
  leader (as before), the median change from 1.5 trunk radii to the tip is 0%
  (IQR −29% to +22%), and **50% of leaders come out widening upwards**, which
  a branch does not do. The old code replaced a widening fit by a constant
  width, which is why so many leaders had no taper at all.
- **Pooled over leaders**, the width is about constant from 5 trunk radii
  on, and a little wider nearer the fork. So one shape is shared by all
  leaders: `r(s) = c · (1 + w · exp(−s / (L · R)))`, with `c` each leader's
  own width. The grid search (`widening_costs`) chose **w = 0.35 at the fork,
  L = 4 trunk radii** (+24% at 1.5 R, +13% at 4 R, +5% at 8 R).
  Leave-one-tree-out picked 0.30–0.45 and 3–5.
- **The evidence is weak** (on these 21 trees; on all 449 it is clear). On
  held-out trees the shape beats a constant
  width only slightly (cost 2.426 against 2.438) and on 12 of 21 trees. The
  colour cross-sections and the hand masks show the widening; depth (minus
  its overshoot) does not show it near the fork (0.43–0.45 from 1.5 to 5
  trunk radii). Kept because it matches the student's observation of a
  slight narrowing away from the fork; `fork_widening=0` gives leaders of
  constant width.
- **Not the rejected flare of session 6.** That blended each leader from the
  trunk's radius (R ≈ 15 px) at the fork; this widening starts at 1.35 times
  the leader's own width (about 9 px, 0.6 R).
- **Cost at the crotch.** Mask pixels between the two leaders near the fork
  that are off the depth silhouette (so, given the overshoot, not wood;
  `crotch_zone`): 856 before, 1,561 with the new widths and no widening,
  2,097 with the final widening (21 trees; about 25 px per tree from the
  widening; 2,219 with the first fit, fading over 4 trunk radii). By eye the
  bottom of the V sits a few pixels higher and still follows the depth
  outline.

## Check on all development rows (449 trees)

Run on the student's PC with
`python -m treeskel.measure_width data/raw --out outputs/measure_width --plot outputs/leader_width_profiles.png`
after the change was merged (console output in `all_dev_rows_output.txt`;
test rows left out as before):

| | 21 trees | 449 trees |
|---|---|---|
| First usable cross-section (trunk radii from the fork) | 2.1 | 2.2 |
| Depth overshoot past clear edges (median) | +2.0 px (4,394 edges) | +2.0 px (115,789 edges) |
| Leader half-width, ≥ 8 trunk radii from the fork: before / after / depth minus overshoot | 3.1 / 6.8 / 7.2 px | 3.2 / 6.9 / 6.5 px |
| After minus depth (median, IQR) | −0.4 px (−1.7 to +0.9) | +0.1 px (−0.8 to +1.4) |
| Before minus depth | −3.8 px | −3.2 px |
| **Free straight taper per leader: median change, share widening upwards** | **0%, 50%** | **−16%, 28%** |
| **Shared widening: best at the fork, length** | **0.35, 4 trunk radii** | **0.35, 12 trunk radii (edge of the grid)** |
| Leave-one-tree-out picks | 0.30–0.45, 3–5 | 0.30–0.35, 12 |
| Held-out trees where the shape beats a constant width | 12 of 21 | 297 of 449 |
| Crotch pixels off depth: before / after / after without widening | 856 / 2,219 / 1,561 | 10,567 / 34,234 / 23,262 |
| Traced path to the bark's centre: median, share over 3 px | 2.5 px, 39% | 2.5 px, 41% |

Both columns are from the first settings (the widening fading over 4 trunk
radii, fill 1.4).

**Held:** the width correction itself (the new leaders agree with depth
minus its overshoot to +0.1 px), the 2 px depth overshoot, the neighbour
rule, the off-centre paths, and the cost at the crotch (about 24 px per tree
from the widening).

**Three statements made from the 21 trees did not hold:**

- *"A straight taper along the whole leader is not supported (median change
  0%, half of the leaders widen upwards)."* On all development trees the
  median free taper is **−16%** from 1.5 trunk radii to the tip, and only
  28% of leaders widen. The leaders do narrow along their length, as the
  student expected.
- *"The widening fades over 3–5 trunk radii."* The best length is 12, the
  longest in the grid, and every leave-one-tree-out fit chose it: the
  narrowing continues along the leader rather than stopping near the fork.
- *"The evidence for the widening is weak."* It beats a constant width on
  297 of 449 held-out trees (66%), cost 2.458 against 2.511.

**Why the 21 trees misled:** 17 of them are the tuning trees, chosen in
session 4 *because* they were problem trees for the automatic method (posts,
depth gaps, straight unbranched trees, three main branches). They were never
a representative sample of leader shapes. Session 6 had the same lesson (a
setting from 21 trees needs the full development set); here the small sample
was also selected for being unusual.

**Depth cannot settle the taper.** Along the leaders, depth minus its
overshoot stays flat (0.40–0.44 of the trunk radius), while the colour
cross-sections narrow (0.50 at 2–3 trunk radii to 0.39 at 15–25) and the
hand masks narrow more (0.55 to 0.35). Raw depth *widens* towards the top
(0.50 to 0.56), where the leaders are seen against the next row's canopy and
the silhouette merges more often, which would hide a taper.

**Consequence:** the first default (0.35 fading over 4 trunk radii) put the
narrowing too close to the fork: near the fork the drawn leaders were wider
than the pooled measurements (0.53 against 0.44 of the trunk radius at
1.5–2 trunk radii, 0.43 against 0.39 at 15–25). So the grid was extended to
48 trunk radii and a widening of 1.0, the shape was also chosen with the
row-grouped folds of `splits.json` held out, and the whole cost surface was
saved.

## The final shape, chosen on all 449 trees

Run on the student's PC with the extended grid,
`python -m treeskel.measure_width data/raw --out outputs/measure_width2`
(console output in `all_dev_rows_output_extended_grid.txt`):

| Shared widening | At the fork | Length (trunk radii) | Cost |
|---|---|---|---|
| **Best on all 449 trees** (inside the grid) | **0.35** | **12** | **2.4573** |
| First default | 0.35 | 4 | 2.4661 |
| Constant width | 0 | – | 2.5114 |

- **Leave-one-tree-out** picked 0.30–0.35 and 12 (held-out cost 2.4576
  against 2.5114 for a constant width; better on 297 of 449 trees).
- **Row-grouped folds** of `splits.json` picked 0.30–0.35 and 8–12 (held-out
  cost 2.4620 against 2.5137; better on **5 of 5** folds).
- **Set as the default:** `fork_widening = 0.35`, `widening_length = 12`. A
  leader is drawn 31% wider than its own width 1.5 trunk radii from the fork,
  15% at 10 and 3% at 30, so it narrows gradually along its length, as the
  student expected.
- The 21 local trees alone still prefer 4 (cost 2.4100 against 2.4297 at 12),
  as expected of a sample chosen for being unusual. On them the final masks
  are 0.5 px narrower than depth minus its overshoot far from the fork
  (tables above); on all 449 they agree with it (next section).

## The final masks on all 449 trees

Run on the student's PC with the final settings (0.35, 12; no fill),
`python -m treeskel.measure_width data/raw --out outputs/measure_width3`
(console output in `all_dev_rows_output_final.txt`). These masks still had
the trunk's round top at the fork, which the fork junction replaced
afterwards (`docs/results/fork_junction/`). The junction changes no drawn
radius, only pixels within about 1.5 trunk radii of the fork, so the widths
below are unaffected; the crotch counts are not.

| Leader half-width, at least 8 trunk radii from the fork (897 leaders) | Median | Minus depth (median, IQR) |
|---|---|---|
| Depth minus its 2 px overshoot | 6.5 px | – |
| **Final masks** | **6.9 px** | **+0.0 px (−0.9 to +1.3)** |
| First session-8 settings (0.35 fading over 4) | 6.9 px | +0.1 px (−0.8 to +1.4) |
| Before session 8 | 3.2 px | −3.2 px (−4.3 to −2.4) |

| Half-width / trunk radius, by distance from the fork (trunk radii) | 1 | 1.5 | 2 | 3 | 5 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| Measured across the bark (pooled cross-sections) | 0.42 | 0.44 | 0.50 | 0.47 | 0.45 | 0.43 | 0.39 |
| Depth minus its overshoot | 0.40 | 0.40 | 0.42 | 0.41 | 0.41 | 0.43 | 0.41 |
| **Final masks** | **0.52** | **0.51** | **0.51** | **0.50** | **0.48** | **0.45** | **0.42** |
| First session-8 settings | 0.54 | 0.53 | 0.51 | 0.49 | 0.46 | 0.44 | 0.43 |
| Hand masks (summer) | 0.56 | 0.55 | 0.55 | 0.53 | 0.50 | 0.42 | 0.35 |
| Before session 8 | 0.26 | 0.26 | 0.26 | 0.26 | 0.25 | 0.22 | 0.20 |

- Far from the fork the masks now match the corrected depth width (median
  difference 0.0 px), and the leaders narrow from 0.52 to 0.42 of the trunk's
  radius between 1 and 15 trunk radii from the fork, as the cross-sections do
  (0.50 at 2 to 0.39 at 15).
- Within 1.5 trunk radii of the fork the drawn leaders (0.51–0.52) are wider
  than the few cross-sections that can be measured there (0.42–0.44; the
  first usable one is a median 2.2 trunk radii out). The widening was fitted
  from 1.5 trunk radii outwards, so this is an extrapolation; the fork
  junction now shapes that stretch on the outer side.
- Mask pixels in the crotch off the depth silhouette: 10,567 of 171,998
  before session 8 (6.1%), 32,666 of 258,992 with the final widths (12.6%),
  23,262 of 233,851 without the widening (9.9%). Wider leaders fill more of the
  bottom of the V; the fork junction then took the trunk's round top out of it.

## Effect on the shoulder fill (decided: off for now)

With the leaders at their true width the notch that `shoulder_fill` was made
for (session 6) has mostly gone. Re-run of `measure_fork` with the new widths
(`docs/results/fork_shoulders/new_leader_widths/`):

| Mean shoulder coverage | reach 1.5 R | reach 2 R |
|---|---|---|
| Depth silhouette | 0.954 | 0.943 |
| Depth shrunk 2 px (≈ the measured overshoot) | 0.896 | 0.879 |
| **New masks, no fill** | **0.936** | **0.912** |
| New masks, fill 1.4 (session 6) | 0.998 | 0.964 |

Without any fill the new masks already lie between the two depth references,
the criterion used in session 6. The matching setting is 0.58 (reach 1.5 R)
or 1.12 (reach 2 R) against raw depth, and below 0 against the 2 px-corrected
depth. The pixels the 1.4 fill adds are now 81% on depth (91–94% before).
`shoulder_length` is a fixed decision, so it was left at 1.4 for the
student to decide between keeping it, lowering it to about 1.0, or turning
it off (0).
`docs/figures/fork_shoulders_new_leader_widths.jpg` shows six forks with no
fill, 1.4 and 2.0.

**On all 449 development trees** (`docs/results/fork_shoulders/new_leader_widths/all_dev_rows_output.txt`)
the same holds:

| Mean shoulder coverage | reach 1.5 R | reach 2 R |
|---|---|---|
| Depth silhouette | 0.938 | 0.914 |
| Depth shrunk 2 px | 0.875 | 0.840 |
| **New masks, no fill** | **0.914** | **0.879** |
| New masks, fill 1.0 | 0.951 | 0.906 |
| New masks, fill 1.4 (session 6) | 0.996 | 0.948 |

Matching settings: 0.66 (depth, 1.5 R) and 1.08 (depth, 2 R), below 0 against
the shrunk depth. The pixels the 1.4 fill adds are 79% on depth (91% in
session 6, with the thin leaders).

**Decision (the student's): the fill is turned off "for the moment"**
(`shoulder_length = 0`), which matches the depth silhouette corrected for its
measured overshoot. `shoulder_length = 1.4` restores it. The figures and
the 21-tree outputs in this folder have since been regenerated with the
final settings (no fill). The fill only touches each leader's outer side near
the fork: turning it off left the widths, the crotch counts and the centring
unchanged (checked on the 21 trees).

## Found, not changed

- **The traced path is off the bark's centre** by a median 2.5 px (14,640
  cross-sections; 39% more than 3 px; per leader 1.0–4.0 px), measured from
  the cross-sections' two edges. *Looked into at the end of session 8
  (`docs/results/depth_registration/`):* this is mostly noise in the colour
  centre of single cross-sections. The systematic part (about 1.3 px on
  sloping leaders) comes from the winter depth sitting about 4 px below the
  colour image, which the tracing follows. Re-centring on the colour centre
  was tried and rejected; correcting the depth is planned for session 9.
- **Trunk widths.** The trunk still comes from depth. Its median half-width
  agrees with the cross-sections (15.5 against 15.0 px on the 21 trees), but
  some trees differ a lot: R01N25 (21.9 against 13.5 px: a swelling low on the
  trunk, drawn along its whole length because the taper fit falls back to a
  constant), R01N05 (17.3 against 11.2: depth includes what looks like a tree
  guard), R20N06 (12.0 against 19.0), R09N32 and R09N34. *Looked into at the
  end of session 8:* colour is not reliably better (it catches only the
  sunlit stripe of R01N18's trunk and takes in a white object beside R09N34's),
  so the trunk was left as it is (`docs/results/depth_registration/`).

## Caveats

- The shared widening was first fitted on 21 trees and failed the check on
  all 449 development trees; the final values were chosen on all 449, by
  leave-one-tree-out and by the row-grouped folds. No test-row tree was
  used.
- The cross-section model assumes one piece of bark between two backgrounds.
  Where a shaded half is as dark as the background next to it, it can be left
  out (7–12% of cross-sections came out below 0.6 times their leader's
  width, spread evenly along the leaders); where a branch crosses behind, the
  cross-section can come out too wide. The robust average absorbs both.
- The hand masks come from another season and another person's judgement;
  they are a guide to proportions, not ground truth.
- Rendering takes about 1.1 s per tree (1.4 s before).

## To do

Done: the final masks' numbers on all development trees are recorded above
([The final masks on all 449 trees](#the-final-masks-on-all-449-trees)).
The chart `leader_width_profiles.png` made on the student's PC with them is
not in the repository; `docs/figures/leader_width_profiles.png` is the
21-tree version.
