# The trees in 3D

Session 13, the second agreed item. `python -m treeskel.tree3d data/raw` on the student's PC (449
development trees; winter traces, lined-up hand masks and the named U-Net's masks on the summer depth;
six camera settings). Per tree and setting: `trees.csv` (sizes in units of the scale). The summary
(`report.json`, `output.txt`) and the figure (`docs/figures/tree3d.png`) were made from `trees.csv` by
`python -m treeskel.tree3d --from-saved docs/results/tree3d`, after the robust summaries below were
added (the per-tree numbers are the run's own). The test rows were not loaded.

## Method

- **Settings fixed before the run** (committed in 337b33d): the depth map's zero 300 (from the ground,
  `docs/results/depth3d/`), focal length 616 px, and the scale that puts the median fork 2.0 m from the
  camera. Every measure is also computed at zeros 262 and 366 (the quartiles), 255 (session 2's guess)
  and focal lengths 585 and 647.
- **Per tree:** the depth along the trace (median within 3 px of each point), a robust plane through
  those 3D points (total least squares with Huber weights), and every traced point moved along its ray
  onto the plane, so the 3D skeleton is complete where the depth has gaps.
- **Measures:** the V opening in the plane, on the same stretch of each leader as the 2D measure, and
  its difference from the 2D V (the **3D correction**); how far the depth points lie from the plane
  (flatness); the plane's lean and turn relative to the camera; the lean from the vertical, with the
  ground behind the tree as the vertical; the trunk's diameter at the fork; the leaders' length in view.
- **Checks with known answers:** synthetic V trees rendered through the camera are measured to within
  1° (opening, lean, turn); a tree facing the camera needs no correction; a wrong zero changes the lean
  but not the opening; the ground gives a known camera pitch, and a vertical tree no lean.

## Results: winter (449 trees, chosen settings)

| Measure | Median [quartiles] | Across the calibration range |
|---|---|---|
| Distance of the depth from the tree's plane | **1.7 cm** [1.4, 2.2] (RMS 3.3 cm) | RMS 2.4–4.3 cm |
| V opening in the tree's plane | 51.1° [43.0, 58.1] | 50.1–51.8° |
| V opening in the photo (same stretch) | 53.1° [45.3, 60.6] | – |
| **3D correction** (plane − photo) | **−2.3°** [−3.1, −1.6]; negative on 97% of trees | −1.5 to −3.4°, always negative |
| Lean towards the camera (top nearer) | 9.2° [6.9, 11.2] | 6.4–12.8° |
| Turn (right side farther) | −4.4° [−6.3, −2.4] | – |
| Lean from the vertical (ground) | −12.9° [−15.5, −10.1] | **−3.1 to −23.8°** |
| Trunk diameter at the fork | 10.7 cm [9.2, 12.2] | 10.2–11.3 cm (focal); ±10% with the distance |
| Each leader's length in view | 1.4 m [1.3, 1.5] | ±10% with the distance |

- **The trees are flat:** on the median tree, half the depth along the skeleton lies within 1.7 cm of
  the tree's plane (within 2.8 cm on 90% of trees). The "2D" trellis is two-dimensional to within a
  couple of centimetres.
- **The 2D V opening is about 2° wider than the V in the tree's own plane,** on 97% of trees, for every
  calibration tried (1.5–3.4°). The cause is the lean: a synthetic tree with the median lean (9.2°) and
  turn (−4.4°), placed where the median fork is, gives −2.1°, and the same tree facing the camera 0°.
  Apart from a few failed planes, the correction varies by only ±1.1° (SD) from tree to tree, so the
  project's 2D V opening is the 3D one plus a near-constant 2°: the 2D comparisons and skills stand.
- **The lean towards the camera** is about 9° (6–13° over the calibration range). The **lean from the
  vertical cannot be told:** it needs the camera's pitch, which the ground behind the tree gives as 22°,
  but that depends strongly on the zero (−3° to −24° of lean) and on the ground being level, which the
  raised tree strip is not.
- **Sizes rest on the assumed 2.0 m** to the fork (Chen et al.'s 1.8–2.2 m) and scale with it. The trunk
  diameter comes from the depth's width, which reaches about 2 px past each edge (session 8); without
  that the median would be about 9.4 cm.
- **Failed planes:** 6 trees (1%) have planes turned more than 20° from the camera (median turn −4.4°):
  R01N30, R07N26, R08N22, R09N33, R11N15, R18N12. R08N22 has depth on only 11% of its skeleton. They
  are kept in; every summary here is a median or a rank, which they barely move. (The 20° was set after
  seeing the run, so it only describes.)

## Results: summer against winter, tree by tree

| | Hand masks (448 trees) | U-Net (442–447 trees) |
|---|---|---|
| V opening: \|summer − winter\| median, in the plane / in the photo | **2.31° / 2.64°** | **2.67° / 2.80°** |
| ... the plane closer, share of trees (Wilcoxon p) | **61%** (2e-6) | **58%** (4e-4) |
| V opening, rank correlation with winter: plane / photo | 0.88 / 0.92 | 0.83 / 0.88 |
| 3D correction: rank correlation; median \|difference\| | 0.52; 0.7° | 0.48; 0.8° |
| Lean towards the camera: rank correlation; median \|difference\| | 0.39; 2.8° | 0.32; 3.2° |
| Turn: rank correlation | 0.10 | 0.06 |
| Summer planes turned more than 20° | 4 | 14 |

- **In 3D, summer agrees better with winter:** the V in the tree's plane does not depend on where the
  camera stood, and the camera stood differently in each season (session 7), so on most trees the 3D V
  removes part of that difference. The gain is small (median 0.1–0.3°), and the summer planes that fail
  (leaves in front of hidden branches) make a few trees worse, so the overall rank correlation is a
  little lower than in the photo.
- **The per-tree correction is partly real:** it repeats between the seasons (rank correlation about 0.5)
  although the summer depth is read through leaves.
- **The lean and the turn relative to the camera do not repeat well** (rank correlations 0.39 and 0.10):
  they include the camera's own pitch and yaw, which changed between the seasons.

## An error caught on the way

The first summary reported Pearson's r: 0.04 for the summer correction against winter's, which read as
"the per-tree correction does not repeat". The figure showed the points along the diagonal. A handful of
failed planes (corrections of ±80°) had swamped r; the rank correlation is 0.52. The summary now reports
medians and rank correlations, and a test checks that one failed fit cannot swamp them.

## Limits

- The depth map's zero is known only to its quartiles (262–366), the scale only from the documented
  camera distance, and the focal length is the camera's nominal one; the table gives every measure
  across these ranges.
- The summer depth is read through the leaves; the summer masks' trunk widths are not measured.
- Development rows only. The test rows were already used for the 2D result; the 3D summary is a
  property of the orchard and the camera rather than a model, so it was not repeated there.
