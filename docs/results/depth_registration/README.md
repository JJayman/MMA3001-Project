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
