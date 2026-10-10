# Bends out of the tree's plane

Session 15, the student's request: the 3D model should let the trunk and the leaders bend out of the
tree's plane. Until now (`treeskel.tree3d`, `treeskel.model3d`) every skeleton point was moved along its
ray onto one plane, so the depth along the branches was used only to place that plane.

The student's plan, followed here:

1. Start with the winter trees: their branches are in view, so they test whether the depth holds useful
   bends out of the plane.
2. Fit a smooth 3D curve along each branch: reject background readings and depth jumps, keep the three
   curves joined at the fork, and do not follow every noisy pixel.
3. Make the plane a preference, not a requirement: well-observed sections may leave it, poorly observed
   ones stay near it.
4. Then summer: visible wood as depth anchors, hidden sections interpolated and marked uncertain. The
   obstacle: at a predicted branch pixel the depth may be a leaf in front, so a branch fitted to every
   reading would follow the foliage.

**Status: method, checks and rules committed; the 449-tree run is still to be made on the student's PC**
(the command is at the end). The 21 development trees in the repository give a preview in
[`../bend3d_preview/`](../bend3d_preview/); it is not a result (see below).

## Method (`treeskel.bend3d`)

- **The depth of each branch's axis.** The depth reads a branch's near surface, so its axis lies one radius
  farther: the radius in px from the 2D width (the guided trace's widths in winter, the mask's half-width
  in summer), at the point's distance. Without it the trunk (radius about 5 cm) would read about 3 cm
  nearer than the leaders (about 2 cm), a false step at the fork. On the preview trees it changes the
  trunk's angle out of the V's plane by 3.5° (median) and the V opening by 0.03°.
- **Which readings count.** Winter: every reading. Summer: only where the branch is a **depth ridge**,
  with both sides (6 px beyond its edge) reading at least 4 units (about 5 cm) farther, or nothing. Inside
  the canopy a reading at a branch pixel is usually a leaf in front, and the depth there is a smooth
  surface with no ridge; the trunk below the canopy and wood seen against a gap pass. On the preview,
  summer keeps 29–45% of the trunk's readings and 4% of the leaders'.
- **The plane**: the robust plane of `treeskel.tree3d` through every reading's axis point; the bends are
  measured from it. A plane turned more than 60° from the camera is a failed fit (R07N26 on the preview, one
  of session 13's six).
- **The bends.** Along each part, the distance to the axis along the pixel's ray is `Z_plane + h(s)`. The
  bend `h` has a Gaussian-process prior with standard deviation τ and length L, precision
  `(1 + L⁴ d⁴/ds⁴) / (2√2 τ² L)`, so its marginal standard deviation is τ and its mean is the plane:
  sections with readings follow them, sections without stay near the plane. The three parts share the
  fork's value (joined) and each may kink there. Each reading counts as `Δs / 7 px` independent ones
  (neighbouring readings share most of their 7 px disc). Robust: Huber weights with the noise re-estimated,
  then Tukey's biweight with it fixed, so background readings and depth jumps get no weight.
- **Uncertainty**: the posterior standard deviation of `h` at each point (a lower bound: it treats the
  prior, the noise and the plane as known). A point is **uncertain** where it exceeds τ/2, so where the
  readings have not halved the prior's uncertainty.
- **The 3D curves**: each point moves along its own ray, so the bent skeleton still projects exactly onto
  the 2D trace (a test checks this).
- **Choosing τ and L (block cross-validation).** Each part is cut into 40 px blocks (about 13 cm). Each
  block is left out in turn with 10 px either side (so correlated noise cannot leak in), the plane is
  fitted again without them, and the block's readings are predicted from the rest: by the plane alone
  (τ = 0) and by each prior on the grid τ ∈ {1, 2, 4, 8, 16} cm × L ∈ {5, 10, 20, 40} cm. Bends that are
  smooth over tens of centimetres predict a left-out block from its neighbours; noise does not.
- **Measures**, alignment-free, so that winter and summer can be compared tree by tree:
  - `trunk_out`: the angle of the trunk's top (66 winter px below the fork, as `treeskel.shape`) out of
    the **V's plane**, the plane through both leaders' lines (positive: the root behind);
  - `left_bow`, `right_bow`: the middle of each leader's stretch (25–264 winter px from the fork, as the
    V opening) against the middle of its chord, out of the V's plane (cm; positive: behind);
  - `left_tilt`, `right_tilt`: each leader's line on its stretch out of the tree's plane, and
    `*_tilt_whole` for the whole leader in view (positive: the upper end behind);
  - `v_opening`: the angle between the bent leaders' lines on the stretch, beside the plane's;
  - how far each part gets from the plane, the leaders' lengths, the shares of uncertain points and of
    readings that count, and the fitted noise.

  With the plane, `trunk_out` and the bows are 0 by construction.

## Checks with known answers

Synthetic trees in 3D, rendered as 8-bit depth of their near surfaces with smoothed noise and rounding
(`tests/test_bend3d.py`; the table is `synthetic_check.py`, output in `synthetic_check_output.txt`, 10 noise
seeds per case, prior τ 2 cm, L 20 cm):

| Case | V opening, bends − plane (true) | Trunk-out error | Bow error |
|---|---|---|---|
| Flat, leaning 9°, turned −5°, noise 0.7 units | −0.03 ± 0.06° (−0.00°) | +0.07 ± 1.03° | +0.07 ± 0.19 cm |
| Flat, noise 1.2 units | −0.05 ± 0.10° (−0.01°) | −0.22 ± 1.50° | +0.05 ± 0.29 cm |
| Trunk 20° out, leaders flat | +0.07 ± 0.06° (+0.09°) | −0.69 ± 0.59° | +0.07 ± 0.19 cm |
| Trunk 20° out (true 19.4°), left leader bowed (true bow 4.9 cm) | +0.12 ± 0.06° (+0.15°) | −0.68 ± 0.70° | −0.07 ± 0.31 cm |
| Trunk 15° out the other way (true −14.4°), bow −3.8 cm | −0.12 ± 0.08° (−0.14°) | +0.23 ± 1.27° | +0.05 ± 0.15 cm |

- **No false bends:** on flat trees the bends stay within 1 cm of the plane, and they do not open the V.
- **Known bends recovered:** trunk angles to about 1°, bows to a few millimetres.
- **A branch crossing 18 cm in front** gets no weight; about half a metre of leader with no readings stays within 1 cm
  of the plane and is marked uncertain; the prior's marginal standard deviation is τ (to 10%).
- **The cross-validation finds bends only where there are some:** on a bent tree the bends cut the
  left-out error by more than 40%; on a flat tree they gain nothing.

## Rules fixed before the 449-tree run

Committed with this README, before any of the 449 trees was fitted. Development rows only; the test rows
are not loaded (`splits.json`).

1. **The prior** is the one on the grid with the smallest pooled mean absolute error of the left-out
   readings over all development trees' winter depth (`bend3d.choose`). Its gain over the plane is reported
   **out of fold**: for each of the 5 row-grouped folds, the prior is chosen on the other folds' trees and
   scored on the fold's own.
2. **Step 1, does the winter depth hold useful bends?** Yes if, out of fold, the bends predict the left-out
   blocks better than the plane on more than half of the trees with a Wilcoxon signed-rank p < 0.05, **and**
   the same holds with 30 px buffers (10 cm either side of each block).
3. **Step 4, summer.** The summer bends are compared with winter's bends, tree by tree, against the summer
   plane (for which `trunk_out` and the bows are 0). The proposal to the student: `model3d build` bends the
   branches by default (it is `--bends` until then) only if, with the **U-Net's masks** (what `model3d`
   uses), the summer bends are closer to winter's than the summer plane on the **trunk's angle out of the
   V's plane** on more than half of the trees with Wilcoxon p < 0.05. The bows are reported but decide
   nothing: summer sees almost none of the leaders, so their bends stay near the plane either way.
4. **Reported whatever they show:** the bends' effect on the V opening, the leaders' tilts, the trunk's
   angle with ridge readings only (`winter ridges`: the trunk where it stands clear of the ground), and every
   measure under the calibration range (zeros 262, 366 and 255; focal lengths 585 and 647).

## The preview on the 21 trees in the repository (not a result)

`docs/results/bend3d_preview/` is the same command on `data/sample` and `data/tuning`. 17 of these trees were
picked in session 4 as problem trees, and claims from them failed on all 449 trees in sessions 6, 8
and 9, so it only shows what to look for:

- the bends predict left-out depth better than the plane on 19 of 20 trees (2.3 against 4.1 cm), also with
  30 px buffers;
- the trunk stands a median 25° out of the V's plane, root behind (23° with ridge readings only, so not
  the ground mixing into the depth), and summer repeats it in part (rank correlation 0.63–0.64 with winter);
- the leaders bow by about 2 cm; summer cannot see them;
- **the bent leaders' V is about 2.2° wider than the plane's**, which would cancel session 13's 3D
  correction (the 2D V 2.3° wider than the V in the plane). The leaders' lines on the stretch tilt about
  5.5° back from the tree's plane, while the whole leaders lie within 1–1.5° of it (medians): in the depth the
  leaders rise more steeply above the fork and lean towards the camera higher up. On synthetic flat trees
  the bends do not open the V (−0.03°), so this comes from the depth, not the method; whether it is the
  trees' shape or a bias of the depth near the fork, the depth alone cannot tell. The 449-tree run will say
  how general it is.

`docs/figures/bend3d_profiles.png` shows six of these trees: the readings behind the plane along each part,
the fitted bends and their ±2 SD band.

## The run (student's PC)

```bash
python -m treeskel.bend3d data/raw --profiles R01N05 R01N13 R07N29 R15N05 R09N34 R01N18
```

About 30 minutes on 8 threads, longer on fewer (phase 1, the cross-validation, about 13 s a tree; phase 2, the chosen prior
under six camera settings and three skeletons, about 18 s a tree). It writes `cv.csv`, `trees.csv`,
`report.json` and `output.txt` here, and `docs/figures/bend3d.png` and `bend3d_profiles.png`. Push those
files; `python -m treeskel.bend3d --from-saved docs/results/bend3d` remakes the summary without images.
After it, `model3d build --bends` uses the chosen prior (it reads `report.json`).
