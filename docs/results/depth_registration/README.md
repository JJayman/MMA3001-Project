# The winter depth sits about 4 px below the colour image (session 8)

Found at the end of session 8 while checking two proposed mask fixes: moving
the leaders' centre lines onto the bark's centre found in colour, and taking
the trunk's width from colour. Neither fix held up. The checks showed instead
that the winter depth images are shifted relative to the colour images.
**Nothing in the package was changed**; correcting the shift is the first job
of session 9.

Produced by `python docs/results/depth_registration/reproduce.py` (console
output in `output_21_dev_trees.txt`) and `figures.py`, on the 21 annotated
development-row trees available locally. These are stand-alone analysis
scripts, not package modules; the measurement is to become a `treeskel`
module that runs on all development trees.

## The finding

Across each leader, colour gives the bark's two edges (`branch_edges`) and
depth gives the edges of its silhouette along the same cross-section. If the
two images were registered, the silhouette's centre would sit on the bark's
centre, whatever the branch's direction. Instead, over 9,988 cross-sections
(clear of the other leader, closed in depth), the silhouette's centre minus
the bark's centre fits a fixed image shift:

| | Shift of the depth relative to colour |
|---|---|
| **All cross-sections (robust fit)** | **3.2 px down, 0.6 px left** |
| Upper half of the image / lower half | 3.2 / 3.2 px down |
| Left half / right half | 2.9 / 2.7 px down |
| Per tree (sloping cross-sections) | median 3.3 px down (IQR 2.0–4.3); down on 19 of 21 trees |

- **The edges show it directly.** On sloping leaders, depth reaches a median
  0 px past the bark's upper edge and 4 px past its lower edge.
- **It is not the lighting.** The sun lights the branches from the right
  (mean lit-side direction: row −0.03, col +0.39), which would bias the
  colour edges sideways, not up. With a term for which side of the bark is
  brighter, the shift stays 3.1 px down; the lighting term is 0.35 px.
- **A pure shift, not a scale.** It is the same in every part of the image.
- **Moving the depth undoes it.** With the traced paths fixed and the depth
  moved up by 0, 2, 3, 4 and 5 rows (and right by 1 column), the remaining
  shift is +3.19, +1.53, +0.62, −0.20 and −1.06 px. It reaches zero at about
  **3.8 rows up**. At 3–4 rows the depth reaches past both edges alike (means
  2.0 and 2.4 px at 3 rows; 2.5 and 1.9 at 4), matching the 2 px overshoot
  measured earlier in session 8.
- `docs/figures/depth_registration_edges.jpg`: four sloping leaders, with
  the colour edges (red dots) and the depth silhouette's outline (cyan), as
  loaded and with the depth moved up 4 rows and right 1 column.

## What it affects

- **The traced paths.** The cost map pulls paths towards the centre of the
  depth silhouette, so the paths sit off the bark's centre along the shift:
  the bark's centre minus the path fits −1.32 px × the normal's row
  component. Re-traced with the depth moved up 4 rows and right 1 column,
  this is +0.17 px. The masks, drawn around the paths, move with them:
  about 1 px on sloping leaders.
- **The measurements against depth.** The shoulder coverage and crotch
  height at the fork (`docs/results/fork_junction/`) compared masks placed
  on the colour image, and the fork click, with a silhouette that sits
  about 4 px lower. The depth crotch, for example, looks about 4 px (0.25
  trunk radii) lower than it is. The junction reach and the crotch
  comparison need re-checking after the correction.
- **Hardly the widths.** A shift does not change a width: the leader widths
  and their 2 px overshoot (the median over both edges) stand. Re-traced
  with the depth moved, the drawn radii changed by a median +0.02 px for the
  leaders (at most 0.4 px) and +0.04 px for the trunk at the fork (−3.3 to
  +1.8 px on single trees, as the trunk's path moved).
- **Summer:** not checked. These checks need branches traced in the image,
  and summer has none. A correlation of edge maps over the whole image, tried
  as a check that does not need them, failed on winter (its peak hit the
  search limit on 14 of 21 trees), so it cannot be used for summer either.

## Rejected: re-centring the leaders on the colour bark centre

The session-8 width check reported the traced path "a median 2.5 px off the
bark's centre". Re-centring each leader on the colour centre, smoothed along
the leader (running median over ±10 points, then a Gaussian, at most 6 px),
was tried:

- The colour centre and the depth centre disagree about which side the path
  is off: correlation +0.03, median difference 3.0 px. After re-centring, the
  path was nearer the depth centre on 2 leaders and further on 28 (of 42);
  the shift reached its 6 px cap on 10 leaders.
- Up close (`docs/figures/recentre_colour_rejected.jpg`), R07N26's right
  leader moved off the pale bark onto a dark band beside it, which looks like
  a post behind the leader. Where a leader looks like a pale band next to a
  dark one (R15N05), the colour model cannot tell a shaded half of the bark
  from something behind it.
- The 2.5 px is mostly noise in the colour centre of single cross-sections:
  it stayed 2.5 px when the paths were re-traced with the depth moved. The
  systematic part is the misregistration above (−1.32 px × the normal's row
  component), which moving the depth removes.

## Rejected: trunk widths from colour

The trunk is drawn from depth (the distance to the silhouette's edge at the
path, fitted as a straight taper). Its width was compared with colour
cross-sections (±40 px) and with the depth silhouette's width across the
trunk minus the 2 px overshoot, over the upper 70% of the trunk:

| Trunk half-width (px), median of 21 trees | |
|---|---|
| Drawn (depth, taper) | 15.9 |
| Colour cross-sections | 17.2 |
| Depth width minus overshoot | 19.8 |

Drawn and colour agree on the median tree (difference +0.0 px) but differ a
lot tree by tree (IQR −3.3 to +2.0 px), and each cue fails somewhere
(`docs/figures/trunk_width_colour_vs_depth.jpg`):

- **R01N05:** colour follows the white trunk (11.2 px); depth, and so the
  drawn trunk (17.3), also takes in a dark object beside it, probably a tree
  guard.
- **R20N06:** colour and depth agree (19.0, 19.5) that the trunk is wider
  than drawn (12.0).
- **R01N18:** colour catches only the bright stripe of a side-lit trunk
  (13.5); depth (18.5) and the drawn trunk (17.1) cover the shaded side too.
- **R09N34:** colour takes in a white object beside the trunk (20.5 against
  12.8 drawn); depth has no clean edges there.
- Depth's width is too wide where it merges with weeds or the ground (R01N08:
  22.2 against 13.8 in colour, which follows the bark in the photo; R01N13).

No cue is reliably better, so the trunk was left as it is.

## Session 9: measured in the package, and why the estimate was low

`python -m treeskel.measure_registration` now makes the measurement for all
development trees (test rows left out). On the 21 local trees it reproduces
the numbers above exactly (9,988 cross-sections, dy +3.19, dx −0.63, and the
same table of moves). The numbers below are from those 21 trees; **the run on
all 449 development trees is still to come**, and will replace them.

### Why the fit reads low: the colour centre follows its window

- **Selection is not the cause.** On the cross-sections usable both before
  and after moving the depth up 4 rows, each difference changes by exactly
  the move along its normal (slope 0.985), and the fit by 3.97 px. The fit on
  its own usable set changes by only 3.39 (0.87 per row), because the set
  changes. The cross-sections usable at every move read 3.50 px (95% CI over
  trees 2.97–4.09), not 3.19.
- **The colour estimate follows its window.** Moving each cross-section's
  window along its normal, with the images unchanged, moves the depth run's
  centre by nothing (±0.04 px) but the bark's colour centre (`branch_edges`)
  by a mean of −1.47, −0.67, +0.56 and +0.99 px for moves of −2, −1, +1 and
  +2 px. The median moves by 0.00: most cross-sections are unaffected, and
  some jump as clutter enters or leaves the window's ends. On synthetic bark
  the pull is 0.04–0.22 (sharp or blurred edges, grass or a background as
  dark as the shaded side), so it comes from real clutter.
- **So the result depends on the paths.** The windows sit on the traced
  paths, and the paths follow the depth. Traced again with the depth moved
  up 4 rows (the paths then sit 0.86 px higher on sloping leaders), the same
  images read +0.68 px instead of −0.20.

| Tried to remove the dependence (depth moved up 4 / up 5, 1 col right) | Paths kept | Traced again |
|---|---|---|
| Huber fit (as before) | −0.20 / −1.06 | +0.68 / +0.18 |
| Tukey's biweight | −0.19 / −1.06 | +0.70 / +0.18 |
| Least absolute deviations | −0.01 / −0.84 | +0.70 / +0.02 |
| Only cross-sections whose centre stays within 0.5 px when the window moves ±2 px (48%) | −0.24 / −1.14 | +1.01 / +0.53 |
| **Only cross-sections with both edges clear** (grey change over 60; 9%) | **+0.21 / −0.77** | **+0.41 / −0.58** |

Re-centring each window on its own estimate did not converge: the pull fell
only from 0.64 to 0.37 over six rounds while the estimates wandered (median
change 0.6 → 3.1 px). On clear-edged cross-sections the pull is 0.43
(0.71 on the others), and the paths matter least.

### Estimates of the correction (21 trees; rows up, columns right)

| Estimate | Rows | Cols |
|---|---|---|
| Paths kept, all cross-sections (session 8's method) | −3.74 | +0.74 |
| Paths kept: equal overshoot past the upper and lower edges | −3.38 | |
| Mutual information, winter, whole image | −4.04 | +0.92 |
| Paths kept, both edges clear | −4.18 | +1.70 |
| Traced again, both edges clear (interpolated) | −4.42 | |
| Colour measured on the depth run, both edges clear | −5.11 | +1.76 |
| Traced again, all cross-sections (extrapolated) | −5.35 | |
| Colour measured on the depth run, all cross-sections | −5.77 | +0.92 |

Measured on the depth run's centre instead of the path, the colour estimate
is pulled towards the depth, so the difference shrinks (by about half) but
is zero exactly where the depth lines up; it magnifies any bias of the
colour centre of its own, though. All estimates put the depth lower than the
colour; they differ on how far (3.4 to 5.8 rows). Session 8's (−4, +1) is
at the low end.

- **A shift, not a scale** (cross-sections usable at every move): dy +3.37 px
  at the image's centre, changing by −0.23 px to the bottom edge and +0.10 to
  the right; by bands of rows +3.45, +3.71, +3.71 (+1.86 in the bottom band,
  353 cross-sections).
- **Per tree:** median +3.95 px (IQR +2.81 to +4.54), lower on 20 of 21. The
  halves of the leaders near and far from the fork agree tree by tree
  (r = +0.47), but mutual information and the leaders do not (r = +0.01), so
  whether trees really differ is open.

### Mutual information, and summer

- **Its highest point is not the estimate.** Depth reaches about 2 px past
  what it sees, so every shift that keeps the colour's objects inside the
  depth's scores about the same (on a synthetic scene, all 13 shifts within
  2 px of the true one within 0.2%). The highest point, refined by a
  parabola, missed known shifts by up to 1.54 px; the vertex of a quadratic
  fitted to the 7×7 values around it misses by at most 0.15 px, and is used.
- **Winter:** 4.04 px down (95% CI 3.27–5.47), 0.92 left. By bands of rows
  1.48, 2.77, 3.58 and 1.95, where the leaders read 3.45–3.71: it reads low
  near the top. Single trees are unreliable (6 of 21 peak at the edge of the
  search; one surface rises steadily towards larger shifts, likely the far
  ground's range limit, a depth edge with no colour edge).
- **Summer:** 2.13 px down (95% CI 1.52–2.85), 1.09 left; by bands 1.85,
  2.40, 3.08. It cannot set a summer correction on its own.

### Rejected in session 9

- **A check along single depth edges** (the colour step along each depth
  edge's normal, fitted as overshoot plus shift). Exact on a synthetic scene,
  but on winter it read 0.9 px where the leaders read 3.2, and moved 0.2–1 px
  for known moves of 3–4 px: across thin, shaded bark there is no single
  colour step to find.
- **The correction clicks already in `annotations/winter_clicks.json`** as a
  human reference: of 71 leader clicks on the 21 trees, 24 lie on a closed
  depth run, mostly tips at the image's edge, and the waypoints were placed
  where depth misled the trace. The fit swung (+8.5 px, +6.4 after moving
  the depth up 4 rows).
- **A visual check** of crops with the depth outline moved 0, 4, 5 and 6 rows
  up: where one bark edge is shaded into a dark background, or depth merges
  the leader with a branch beside it, the eye cannot place the edge to 1 px
  either.

### All 449 development trees (session 9, the student's PC)

Console output in `all_dev_rows_output.txt`, the numbers in `report.json` and
`trees.csv`, and `depth_registration_rows.png` (redrawn from `report.json`
after the run; the first chart plotted a mutual-information peak extrapolated
to 113 px, which is now kept within the search).

- **The depth sits lower than the colour** on 431 of 447 trees.
- **One shift for every tree.** The halves of the leaders near and far from
  the fork do not agree tree by tree (r = +0.03): the real spread across
  trees is about 0.4 px, against 2.8 px of noise in each half. The shift does
  not depend on the tree's distance either (Spearman −0.03): a fixed offset
  of the camera, not parallax.
- **Steady across the image and the capture.** By bands of rows (session 8's
  measure, cross-sections usable at every move): +2.60, +3.13, +3.01, +2.80
  px. By orchard row, in capture order from 11:23 to 13:26: 2.1 to 3.9 rows
  (mean 3.11 ± 0.25). The bands of columns and the affine fit's sideways
  terms are not reliable: the leaders' normals come in two families, one per
  leader, and a band of columns holds mostly one leader, which cannot tell
  dy from dx.

| Estimate of the correction (449 trees) | Rows | Cols | Mean of the 16 rows ± 2 SE |
|---|---|---|---|
| Session 8's measure: colour on the kept paths | −3.10 | +0.81 | −3.11 ± 0.25 |
| The same: equal overshoot past both edges | −2.80 | | |
| Colour measured on the depth run | **−5.06** | +0.87 | −5.35 ± 0.61 |
| Traced again (left: +0.56 px at 4 rows up, +0.04 at 5) | **−5.09** | | −5.31 ± 0.48 |
| Mutual information, winter | −4.75 (CI −5.12 to −4.38) | +0.89 | not usable by row |
| Both edges clear: kept paths / depth run / traced again | −2.46 / −3.54 / −3.29 | +2.63 / +2.55 | |

- **The two estimates that take the pull out agree** (5.06 and 5.09 rows),
  and lie above session 8's measure in all 16 rows, by about 2 rows
  (`depth_registration_rows.png`).
- **The clear-edge subset is not the best-behaved after all.** On 21 trees it
  looked best; on 449 its sideways estimate is +2.6 columns where every
  other estimate gives +0.8 to +0.9. Which cross-sections have both edges
  clear depends on the light (from the right), so the subset is biased;
  traced again, it reads −0.47 and −1.12 px (dy) with −1.5 px (dx).
- **Mutual information** is not reliable in winter: 184 of 449 trees' surfaces
  peak at the edge of the search, and so do 7 of the 16 rows (all those
  captured after 12:55). In summer it reads 2.28 px down (CI 2.13–2.41),
  0.67 px left; by bands of rows 2.09, 2.22 and 3.15; 28 trees at the edge.

### Decision: the winter depth moves 5 rows up and 1 column right (the student's)

`treeskel.io.DEPTH_SHIFT = {"summer": (0, 0), "winter": (-5, 1)}`, applied by
`load_tree` (`register=False` loads the depth as stored). The student chose
it from three options (5 up; 4 up, midway between 3.1 and 5.1; or a hand
check first), on these grounds:

- the colour pull was measured directly, and the two estimates that take it
  out agree (5.06 and 5.09 rows), above session 8's measure in every row;
- at 5 rows up, the trees traced again and the depth line up by session 8's
  own measure (+0.04 px left), and that is the state the ground truth is now
  in;
- mutual information, which fits no colour edges, gives 4.75.

The risk: a bias of the colour-edge fit of its own, which those estimates
would magnify; the lower estimates (2.5–3.1) cannot be ruled out entirely.

**Summer is not moved.** Mutual information puts it about 2.3 px low, but in
winter it was too unreliable to set a correction on its own.

**Checks on the 21 local trees**, with the correction loaded
(`--registered`): session 8's measure now reads +0.18 px, and the moves that
would leave no shift are −0.21 rows (session 8's measure) and −0.04 rows
(colour on the depth run): both near zero, where before they differed by
2 rows. Mutual information reads +0.96 rows (it gave 4.04 on these trees).
The student's correction clicks were tried as a human check: the 24 near a
leader's trace (traced through the main clicks only) sit 4.4 px above the
traces made with the depth as stored (fitted as an image shift), falling to
1.4 px at 5 rows up, the smallest of the moves tried (0 to 7 rows up); but
they stay 1.5–2 px above at every move from 4 rows up, likely because they
were placed where traces cut the inside of curves. Too few, and biased, to
decide.

**What changes in the masks** (21 local trees, the same clicks): the leaders
move a median 0.9 px across themselves, all upwards along their normals
(more than 1 px on a third of their points), and the masks before and after
overlap at IoU 0.86 (lowest 0.76); the widths do not change (−0.05 px).
`docs/figures/depth_registration_masks_before_after.jpg` shows six trees
(outlines traced with the depth as stored in orange, corrected in blue).

**Re-checks previewed on the 21 local trees** (`recheck_fork_21_dev_trees.txt`
and `recheck_width_21_dev_trees.txt`, against session 8's outputs on the same
trees); the runs on all 449 are to come:

- *At the fork*, the masks agree better with depth: precision around the fork
  0.930 → 0.949, IoU 0.830 → 0.842. Depth's crotch moves up from 0.77 to
  1.11 trunk radii above the fork (0.34 R, as 5 px should). The junction
  reach whose joins cover the shoulder as much as depth does falls from
  0.98 / 1.30 to 0.83 / 1.02 (shoulders reaching 1.5 / 2 R), so the chosen
  1.0 now sits at the generous end; and the crotch now favours the round top
  (1.01 R, against depth's 1.11 and 0.99 shrunk) over the leaders' V (0.93).
- *The leaders' widths* barely change (6.7 → 6.8 px at 8 trunk radii or more
  from the fork). The mask's crotch pixels off depth halve (18% → 10%).
  Depth reaches further past clear bark edges: median 2.0 → 3.0 px, mean
  1.79 → 2.19, probably because many lower-edge gaps were beyond the 6 px
  cut-off while the depth sat low, which held session 8's figure down. The
  shared widening (0.35, 12) now costs slightly more than a constant width
  on these trees; they are 17 problem trees and 4 others, and the 21-tree
  choice was overturned on 449 in session 8.

**Re-checked on all 449 development trees** (the student's PC; masks
re-rendered for all 541 trees, `rerender_all_trees.txt`; outputs in
`recheck_fork_all_dev_rows.txt` and `recheck_width_all_dev_rows.txt`), against
session 8's runs with the depth as stored:

| 449 development trees | Depth as stored (session 8) | Corrected (5 up, 1 right) |
|---|---|---|
| Fork-zone precision / IoU (masks with the junction) | 0.942 / 0.831 | 0.949 / 0.836 |
| Crotch pixels off the depth silhouette | 12.5% | 8.4% |
| Junction reach matching depth, 1.5 R / 2 R (shrunk 2 px) | 1.04 / 1.23 (0.62 / 0.59) | 1.05 / 1.26 (0.62 / 0.65) |
| Shoulder coverage at 1.5 R, no fill (depth, shrunk) | 0.905 (0.940, 0.878) | 0.892 (0.925, 0.852) |
| Widening: best / default (0.35, 12) / constant width | (0.35, 12) 2.457 / 2.457 / 2.511 | (0.30, 12) 2.498 / 2.498 / 2.546 |
| Leader half-width at 8+ trunk radii: drawn / depth minus overshoot | 6.9 / 6.5 px | 6.9 / 6.5 px |
| Depth past clear edges: median (IQR), mean | +2.0 (0 to +4), 1.69 | +2.0 (+1 to +4), 2.09 |
| Crotch above the fork: depth / round top / the leaders' V | 1.07 / 1.00 / 0.89 R | 1.35 / 1.05 / 1.04 R |

- **Every session-8 setting holds.** The junction reach that matches depth
  still brackets 1.0 (the 21-tree drop to 0.83 did not hold); the masks
  without a shoulder fill still lie between depth and depth shrunk 2 px; the
  widening still fades over 12 trunk radii, and 0.30 at the fork is within
  0.0005 of the default 0.35 (leave-one-tree-out picks 0.30 and 12; the
  folds 0.30–0.35 and 8–16, better than a constant width on 5 of 5).
- **The masks agree a little better with depth**, around the fork and in the
  crotch, and depth's overshoot past the two edges is more even (fewer
  zeros: in session 8 the upper edges read about 0, the lower ones 4).
- **New:** depth's crotch now sits 0.23–0.33 trunk radii above the masks'
  (0.10–0.21 against depth shrunk 2 px): the wood between the leaders reaches
  higher than either the round top or the leaders' V draws it.
- **The student's look** at the re-rendered masks: "not sure they look that
  much better". That is the size expected: the leaders moved a median 0.9 px.

## Next (session 8's list)

1. Turn `reproduce.py`'s measurement into a package module and run it on all
   449 development trees on the student's PC: is the shift the same for every
   row and tree?
2. Find a way to check the summer images, which the model will use.
3. If the shift holds, move the depth when it is loaded (about 4 rows up and
   1 column right, estimated on development rows only), with a test.
   Re-trace, re-render, re-check the fork measurements against the corrected
   depth, and re-measure the leader widths.
4. Only then look again at the trunk width and at re-centring: both may
   change once depth and colour agree.

## Next (session 9)

1. Run `python -m treeskel.measure_registration` on all 449 development trees
   (the student's PC) and see which estimates agree, overall and row by row.
2. Choose the winter correction from them and set it in
   `treeskel.io.DEPTH_SHIFT`; check it with `--registered`; re-trace and
   re-render; re-check the fork measurements and the leader widths.
3. Check whether the session-8 leader widths depend on where their window
   sits: moved 1 px off the path, a cross-section's measured width grew by
   0.24 px on average.
4. Leave the summer depth as it is unless a better check is found.
