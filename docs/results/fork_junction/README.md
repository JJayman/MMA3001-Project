# Fork junction results (session 8)

Produced by

```
python -m treeskel.measure_fork data/tuning data/sample --out docs/results/fork_junction \
    --figure docs/figures/fork_junction_before_after.jpg \
    --figure-trees R01N18 R01N02 R01N25 R01N13 R07N29 R01N32 --figure-reaches 1.0 1.5
```

on the 21 annotated **development-row** trees available locally (the 17
tuning trees plus R01N01, R01N02, R15N05 and R20N06). The four test-row trees
in `data/sample` were left out automatically (test rows 3, 12 and 16 are read
from `splits.json`). Console output: `output_21_dev_trees.txt`; per-leader
numbers: `leaders.csv`; everything else: `report.json`. The check on all 449
development trees is still to run (see [To do](#to-do)).

## The problem (seen by the student)

After the leader widths were corrected, the student found that where the
trunk meets the leaders the mask "kind of bulges and just looks like a
circle", unlike the hand masks, and asked for a fix "not with the fill".

Each part of the mask is a union of discs along its path, so the trunk ended
at the fork in a half disc of its own radius R (11–22 px here). That round
top showed in two places:

- **between the leaders**, as a round floor to the crotch: the crotch sat a
  median 1.00 R above the fork click, against 0.77 R in the depth silhouette
  (0.66 R with its 2 px overshoot removed);
- **beside them**, as a notch on each leader's outer side where it left the
  circle (the notch the session-6 shoulder fill closed, keeping the circle).

On trees with a short trunk the round end at the root adds a second half
disc, and the whole trunk reads as a circle (R01N25 had a trunk about one
radius long; the student's R01N17 looked the same). The trunk's width itself
was checked first and is not inflated: its drawn half-width agrees with the
colour cross-sections (a median difference of 0.0 px on the 21 trees).

## What changed

- **The trunk is cut flat at the fork**, square to its direction over its
  last R px (`end_cut`).
- **Each leader is joined to the trunk on its outer side** (`fork_junction`):
  the convex hull of the half of the flat top on that side and the leader's
  discs up to `junction_reach` trunk radii along it, kept to the outer side of
  the leader. Its edge is a straight line from the trunk's edge at the fork to
  the leader's edge. One hull per leader, and only on the outer side, so the
  **crotch is exactly what the leaders' own discs draw: a V**.
- **The root is cut flat too** (`flat_root`), where the trunk's round end
  would lie wholly inside the image, on the ground below the root click: a
  root click more than the trunk's radius above the bottom edge. That is 4 of
  the 21 local trees and, judging by where the clicks are, about 10% of the
  449 development trees. Most root clicks are a median 4 px above the bottom
  edge, where the trunk leaves the frame; there the round end reaches the
  edge and is kept, since a flat cut would leave a strip of background under
  the trunk.
- `junction_reach = 0` and `flat_root = False` give back the masks before
  this change pixel for pixel (a test checks that they are then just the
  union of the discs).

Two ways of joining were prototyped. The first took one convex hull per side
without keeping it to the outer side. On trees whose leaders curve, it also
filled 1–61 px of the crotch per tree, and the longer the reach, the higher
the crotch (median 0.90, 0.92 and 1.04 R for reaches 1.0, 1.5 and 2.0).
Keeping the hull to the outer side fixes the crotch at 0.86 R for every
reach.

## Measures (21 local development trees)

`measure_fork` now compares, against the round-top masks without fill
(`BASE`), both the old shoulder fill and the junction at several reaches.

**The shoulders.** As in session 6: the triangle between the fork, the
trunk's edge level with the fork and the leader's outer edge 1.5 or 2 trunk
radii up the leader. The true coverage should lie between the raw depth
silhouette (which reaches about 2 px past the bark) and the silhouette shrunk
by 2 px.

| Mean coverage of the shoulder triangle | reach 1.5 R | reach 2 R | Notched (< 80%), 1.5 R | 2 R |
|---|---|---|---|---|
| Depth silhouette (wood) | 0.955 | 0.943 | 7% | 10% |
| Depth silhouette shrunk 2 px | 0.900 | 0.880 | 21% | 19% |
| **Round top (the masks before)** | **0.929** | **0.898** | **2%** | **24%** |
| Round top + fill 1.4 (session 6, off) | 0.998 | 0.958 | 0% | 2% |
| Junction 0.5 | 0.904 | 0.879 | 24% | 29% |
| **Junction 1.0 (chosen)** | **0.958** | **0.917** | **0%** | **12%** |
| Junction 1.5 | 1.000 | 0.961 | 0% | 0% |
| Junction 2.0 | 1.000 | 0.998 | 0% | 0% |

| Junction reach at which the joins cover the shoulder as much as | Value | Leave-one-tree-out |
|---|---|---|
| depth, reach 1.5 R | 0.98 | 0.92–1.16 |
| shrunk depth, reach 1.5 R | below 0.5 | 0.50–0.61 |
| depth, reach 2 R | 1.30 | 1.22–1.51 |
| shrunk depth, reach 2 R | 0.51 | 0.50–0.74 |

**The crotch.** Mask pixels between the leaders within 2.5 R of the fork
that are off the depth silhouette, and the height of the crotch's bottom
above the fork click (the first row between the leaders' paths with a gap).
The height is the same for every reach, because the joins stay on the outer
sides.

| | Mask pixels off depth / all | Crotch height (trunk radii, median) | Minus depth's (paired by tree) | Minus shrunk depth's |
|---|---|---|---|---|
| Depth silhouette | – | 0.77 | – | – |
| Depth shrunk 2 px | – | 0.66 | – | – |
| Round top | 2,097 / 11,793 | 1.00 | +0.11 | +0.25 |
| **Junction (any reach)** | **1,970 / 11,233** | **0.86** | **+0.03** | **+0.17** |

**Pixels changed, against depth.**

| Masks | Pixels added (share on depth / shrunk depth) | Pixels removed (on depth / shrunk depth) | Fork-zone precision | IoU |
|---|---|---|---|---|
| Round top | 0 | 0 | 0.930 | 0.830 |
| Round top + fill 1.4 | 836 (0.83 / 0.70) | 0 | 0.928 | 0.839 |
| Junction 0.5 | 11 (0.64 / 0.64) | 833 (0.82 / 0.68) | 0.931 | 0.824 |
| **Junction 1.0** | **203 (0.91 / 0.76)** | **638 (0.79 / 0.64)** | **0.931** | **0.829** |
| Junction 1.5 | 626 (0.87 / 0.73) | 606 (0.79 / 0.64) | 0.931 | 0.834 |
| Junction 2.0 | 1,115 (0.86 / 0.69) | 591 (0.78 / 0.64) | 0.930 | 0.840 |

(The fork zone is the part of the image within 2.5 R of the fork and within
R of a traced path. As in session 6, IoU against depth grows with anything
added near the fork, because depth reaches past the bark, so it cannot
choose a setting.)

## Reading the numbers

- **Depth does not decide between the round top and the junction.** The
  removed round top is 79% on the depth silhouette (64% on the shrunk one),
  and precision and IoU around the fork barely move (0.930 → 0.931, 0.830 →
  0.829). Depth blurs a junction: its silhouette merges the two leaders and
  the top of the trunk into one mass. So the change is made for the shape
  the student asked for, and depth shows it is no worse.
- **The crotch is closer to depth's.** Its bottom drops from 1.00 R to 0.86
  R above the fork (depth: 0.77 R); the crotch lost 560 mask pixels, 127 of
  them off depth. In the figure the removed pixels (blue) lie on background
  in the photos (on R01N02 in deep shadow). On trees whose trunk leans, the
  flat top is tilted with the trunk, and the round top also went where it
  bulged past a leader's outer edge onto the ground (R01N25).
- **The shoulders.** With reach 1.0 the joins cover the shoulder as much as
  the raw silhouette at 1.5 R (0.958 against 0.955) and between the two
  references at 2 R (0.917, between 0.880 and 0.943). Leaders with a notched
  shoulder at 2 R: 12%, between depth (10%) and shrunk depth (19%).

## Choosing `junction_reach`

**1.0** (chosen). The reaches that match the depth silhouette's coverage of
the shoulder are 0.98 and 1.30 against the raw silhouette and 0.5 or less
against the shrunk one; 1.0 lies within that range, at the raw-depth end.

**1.5** was the first proposal. It rested on the session-6 width decay:
along each leader the depth silhouette's width falls from the trunk's to the
leader's over a median 1.59 trunk radii (449 trees). But that decay is
measured along the leader's centre line, where the crotch limits the width as
much as the outer shoulder: depth's crotch opens at 0.77 R, about the middle
of that transition. So the decay describes the whole junction, not how far
the outer join should reach. Measured directly, 1.5 covers more of the
shoulder than even the raw silhouette at both reaches (1.000 and 0.961),
which is why the 1.4 fill was turned off. On the photos, 1.0 and 1.5 differ by
a few pixels on each outer shoulder (figure, last two columns).

**Other options for the student.** About 0.75 would keep the shoulder
coverage of the masks the student accepted when turning the fill off (0.929
and 0.898), changing only the shape: the flat top and the V crotch. 0.5
leaves deeper notches than the round top did (24–29% of leaders notched).

**The hand masks do not settle it.** A scratch check (not in the package)
measured the width along the centre lines relative to the trunk, as for the
hand masks in `docs/results/leader_width/`, on these 21 trees. The hand masks,
the round top and junctions 0.5 to 2.0 all come within 0.1 of each other at
every distance from the fork. The round top changes the outline beside the
centre line more than the width across it.

## A known answer

`tests/test_guided_gt.py` builds a synthetic tree whose 31 px trunk ends
flat at the fork, with straight chamfers from the trunk's edge to each 9 px
leader and a sharp V crotch. The leaders' inner edges meet 10 px above the
fork. With the junction the mask's crotch bottom on the trunk's axis is
within 2 px of that; with the round top it is at R (16 px). IoU with the true
shape within 2.5 R of the fork: 0.942 with the round top, 0.943 with junction
1.0 (the remaining errors are the leaders' widening near the fork, which this
synthetic tree does not have, and the trunk's radius, 16 px against 15.5).

## Figures

- `docs/figures/fork_junction_before_after.jpg`: six forks (the student's
  R01N18 among them): winter RGB, the round top, junction 1.0 and junction
  1.5. Cyan marks pixels added, blue pixels removed.

## Caveats

- Chosen on 21 trees, most of them the unusual trees used for tuning; the
  check on all 449 development trees is still to run.
- The reach is set by the shoulder coverage against a bracketed reference
  (raw against shrunk depth), not a model of the depth overshoot at a fork.
- The join's edge is straight (a chamfer). Real forks curve: depth covers
  95.5% of the 1.5 R triangle, not all of it.
- The flat root cut takes the root click as the ground line. Where the click
  is above the ground, the trunk below it is not drawn (the round end used to
  draw up to R px of it).
- The trunk's width comes from depth, as before. Trees whose depth width is
  off (R01N25's swelling low on the trunk, R01N05's tree guard, see
  `docs/results/leader_width/`) still show a trunk that is too wide or too
  narrow.

## To do

On the student's PC:

```
python -m treeskel.guided_gt data/raw --overlays
python -m treeskel.measure_fork data/raw --out outputs/measure_fork_junction \
    --figure outputs/fork_junction.jpg --figure-trees R01N17 R01N18 R07N25 R01N32 R09N12 R01N05
```

The first re-renders the ground truth for all 541 trees (check R01N17 and
R01N18 against the hand masks); the second measures the junction on all
development trees.
