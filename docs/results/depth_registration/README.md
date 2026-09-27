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

## Next (session 9)

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
