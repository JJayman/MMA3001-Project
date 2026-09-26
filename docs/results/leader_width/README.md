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
numbers: `leaders.csv`; everything else: `report.json`. The check on all 449
development trees is still to run (see [To do](#to-do)).

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
| **Masks after** | **6.8** | **−0.4 (IQR −1.7 to +0.9)** |
| Depth minus overshoot | 7.2 | – |

**Along the leaders** (`docs/figures/leader_width_profiles.png`; half-width ÷
trunk radius at the fork, medians over leaders; "–": too near the fork to
measure without the other leader):

| Trunk radii from the fork | 0–0.5 | 1–1.5 | 1.5–2 | 3–4 | 5–7 | 10–15 | 15–25 |
|---|---|---|---|---|---|---|---|
| Hand masks (summer) | 0.88 | 0.61 | 0.60 | 0.56 | 0.55 | 0.46 | 0.39 |
| Depth minus overshoot | – | – | 0.43 | 0.45 | 0.45 | 0.52 | 0.46 |
| Measured cross-sections | – | – | 0.45 | 0.52 | 0.47 | 0.44 | 0.42 |
| **Masks after** | 0.59 | 0.55 | 0.54 | 0.50 | 0.47 | 0.45 | 0.44 |
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
- **The evidence is weak.** On held-out trees the shape beats a constant
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
  2,219 with the widening (21 trees; about 30 px per tree from the
  widening). By eye the bottom of the V sits a few pixels higher and still
  follows the depth outline.

## Effect on the shoulder fill (open decision)

With the leaders at their true width the notch that `shoulder_fill` was made
for (session 6) has mostly gone. Re-run of `measure_fork` with the new widths
(`docs/results/fork_shoulders/new_leader_widths/`):

| Mean shoulder coverage | reach 1.5 R | reach 2 R |
|---|---|---|
| Depth silhouette | 0.954 | 0.943 |
| Depth shrunk 2 px (≈ the measured overshoot) | 0.896 | 0.879 |
| **New masks, no fill** | **0.936** | **0.912** |
| New masks, fill 1.4 (current) | 0.998 | 0.964 |

Without any fill the new masks already lie between the two depth references,
the criterion used in session 6. The matching setting is 0.58 (reach 1.5 R)
or 1.12 (reach 2 R) against raw depth, and below 0 against the 2 px-corrected
depth. The pixels the 1.4 fill adds are now 81% on depth (91–94% before).
`shoulder_length` is a fixed decision, so it is **left at 1.4** for the
student to decide: keep 1.4, lower it to about 1.0, or turn it off (0).
`docs/figures/fork_shoulders_new_leader_widths.jpg` shows six forks with no
fill, 1.4 and 2.0.

## Found, not changed

- **The traced path is off the bark's centre** by a median 2.5 px (14,640
  cross-sections; 39% more than 3 px; per leader 1.0–4.0 px), measured from
  the cross-sections' two edges. The mask is centred on the path, so its edges
  are shifted by that much. Re-centring the leaders on the measured bark
  centre (smoothed along the leader) would change the centre lines, a fixed
  decision.
- **Trunk widths.** The trunk still comes from depth. Its median half-width
  agrees with the cross-sections (15.5 against 15.0 px on the 21 trees), but
  some trees differ a lot: R01N25 (21.9 against 13.5 px: a swelling low on the
  trunk, drawn along its whole length because the taper fit falls back to a
  constant), R01N05 (17.3 against 11.2: depth includes what looks like a tree
  guard), R20N06 (12.0 against 19.0), R09N32 and R09N34.

## Caveats

- 21 trees. The numbers, the shared widening and the fill decision should be
  checked on all 449 development trees, as in session 6.
- The cross-section model assumes one piece of bark between two backgrounds.
  Where a shaded half is as dark as the background next to it, it can be left
  out (7–12% of cross-sections came out below 0.6 times their leader's
  width, spread evenly along the leaders); where a branch crosses behind, the
  cross-section can come out too wide. The robust average absorbs both.
- The hand masks come from another season and another person's judgement;
  they are a guide to proportions, not ground truth.
- Rendering takes about 1.1 s per tree (1.4 s before).

## To do

On the student's PC:

```
python -m treeskel.measure_width data/raw --out outputs/measure_width --plot outputs/leader_width_profiles.png
python -m treeskel.measure_fork data/raw --out outputs/measure_fork
python -m treeskel.guided_gt data/raw --overlays
```

The first two print the checks above for all development rows (test rows are
left out); the third re-renders the ground truth with the new widths.
