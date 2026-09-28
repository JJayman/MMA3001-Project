# How the model for the final test is named

Fixed on 28 September 2026 (session 12), **before the full-size run's results existed**: the student
agreed to this rule as the run on the student's PC was starting. The commit that adds this file is the
record.

## The candidates

Both are development runs on the lined-up hand masks, with the pieces of their masks joined
(`treeskel.segment.join_pieces`, gap 19.3 mask px):

- `docs/results/segment_lined_up_joined`: the U-Net on the masks' 256x256 frame (V opening +0.75,
  5 folds);
- `docs/results/segment_lined_up_480x640_joined`: the same U-Net on the whole photo at 480x640 (session
  12's one variant).

## The rule

The full-size run is named if it beats the 256 run **tree by tree on the V opening**: on the trees both
predict, its mean absolute error against the winter traces is lower **and** the Wilcoxon signed-rank test
on the paired absolute errors gives **p < 0.05**. Otherwise the 256 run is named: it is the simpler and
cheaper model, and a difference the folds cannot tell from chance is no reason to prefer the larger one.

Both numbers are printed by the full-size run itself, under "Against segment_lined_up_joined, tree by
tree" (`report.json`: `against_runs.segment_lined_up_joined.v_opening`, keys `mae_seg`, `mae_other` and
`wilcoxon_p`).

## Why this rule

- **One measure, fixed in advance:** the V opening is the headline measure since session 10 (it needs no
  reference direction). Choosing on whichever measure turned out best would be choosing on the same folds
  that report the result.
- **Tree by tree:** the folds' skills swing widely (the 256 run: +0.50 to +0.87 on the V opening), so
  paired errors on the same trees are the fairer comparison (session 10's lesson).
- **Joined runs:** joining was fixed from geometry before it was scored, and the final evaluation
  applies the named run's recipe as it stands, joining included.

Then `model.json` is written in this folder naming the chosen run, committed, and
`python -m treeskel.final_test data/raw` is run once.

## Addendum, before the re-run: growing the masks

Added on 28 September 2026 (session 12), after the naming above and **before the re-run below existed**.
The final test was not run in between.

- **The student's observation:** many predicted leaders stop short or break. Measured on the named run
  (development trees only): 33 of 444 trees have a leader shorter than the stretch the angles are fitted
  over, where no hand mask does, and those 7% of trees carry 47% of the V opening's squared error.
- **The idea, the student's choice among four:** grow each mask into the U-Net's less sure pixels
  connected to it (hysteresis thresholding, `treeskel.segment.grow_mask`), then join as before.
- **The re-run:** the named recipe (256, lined-up masks) trained again with its probabilities saved,
  `--out docs/results/segment_lined_up_rerun`. It writes `_joined` (joined only, as the named model) and
  `_grown` (grown, then joined) from the same network.
- **The lower threshold, fixed without winter:** of 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05 (0.5 grows
  nothing), the one whose grown and joined masks best match the lined-up hand masks' centre lines (mean
  clDice over the 449 development trees, each predicted by the fold that held it out); on a tie, the
  higher. If 0.5 is chosen, there is nothing to adopt.
- **The rule:** growing is adopted if `segment_lined_up_rerun_grown` beats `segment_lined_up_rerun_joined`
  tree by tree on the V opening: lower MAE and Wilcoxon p < 0.05, as above. The comparison is between masks
  of the same network, so it measures the growing alone (a re-run differs from the original by the GPU's
  non-determinism; the original and the re-run side by side show how much).
- **If adopted,** `model.json` is rewritten to name `docs/results/segment_lined_up_rerun_grown` (its
  recipe: the re-run's median epochs, the chosen threshold, the same joining gap). **If not,** `model.json`
  stays as it is.

**Applied (re-run on the student's PC):** clDice chose a lower threshold of 0.1, which grew all 449
masks (604,615 pixels). Against the re-run's joined masks, on the V opening: MAE 4.105° → 4.141°, grown
closer on 42% of trees (p = 0.003, i.e. significantly *worse* more often). **Not adopted;** `model.json`
stays as it is. The re-run itself repeated the original closely (V opening +0.72, joined +0.75).

## Addendum, before scoring: continuing short leaders and pruning

Added on 28 September 2026 (session 12), after the growing result and **before any continued mask was
scored**. The final test was not run in between.

- **The student's idea:** keep only the trunk and the two leaders, and where a leader stops abruptly
  with a piece further along its path, join that piece (`treeskel.segment.continue_leaders`).
- **Applied to** the re-run's joined masks (`segment_lined_up_rerun_joined`), no growing; the result goes
  to `segment_lined_up_rerun_continued`, scored from the saved masks.
- **Settings, fixed without winter:** only leaders shorter than the fitted stretch (16 units, 158 mask
  px) are continued; a piece is joined if a pixel of it lies within the part of the stretch still missing
  and within **25°** of the leader's direction at its tip (the development hand masks' leaders turn by at
  most 24.8° over the stretch in 95% of cases); the tip's direction is a line through its last 4 units
  (the trunk axis's length); a piece needs 10 px of centre line (the fewest a line fit takes); up to 5
  pieces in a chain. Then every mask is cut down to its trunk and two leaders.
- **The rule:** adopted if `segment_lined_up_rerun_continued` beats `segment_lined_up_rerun_joined` tree by
  tree on the V opening (lower MAE and Wilcoxon p < 0.05). If adopted, `model.json` names
  `docs/results/segment_lined_up_rerun_continued` (the re-run's median epochs, 36; joining; continuing and
  pruning). If not, `model.json` stays as it is.
- This is the third change tested on the same development trees (joining adopted, growing not); each
  was fixed before it was scored, and the final test rows are the independent check of whichever is named.

**Applied (scored from the saved masks, after the commit above):** 9 masks had a piece to continue to;
pruning changed some measure by more than 0.5° in 344. Against the re-run's joined masks, on the V
opening: MAE 4.105° → 4.129°, closer on 45% of trees, p = 0.21. **Not adopted;** `model.json` stays as it
is (`docs/results/segment_lined_up_joined`). Details: `docs/results/segment_lined_up_rerun/README.md`.

## Addendum, before scoring: completing short leaders' lines from the leaders' shapes

Added on 28 September 2026 (session 12), after the continuing result and **before any completed
measure was scored**. The final test was not run in between.

- **The student's idea:** where a leader stops though the leaves go on, predict where it goes from the
  shape of the leaders the model has seen (`treeskel.segment.complete_measures`).
- **The model:** a short leader's line over the whole fitted stretch differs from its line over the
  visible part by `a·g + c·g·bend` (outwards; `g` the missing fraction of the stretch, `bend` the visible
  part's bend). `a` and `c` are fitted by least squares on the run's own **complete** leaders, cut on
  purpose at 40-95% of the stretch; for each fold only on the other folds' trees, so no tree's prediction
  uses its own leaders; the fit on all trees is kept for the final test. The masks are not changed; the
  bends stay as measured. Nothing comes from the winter traces. (Checked beforehand on the complete
  leaders themselves, row-grouped: cutting a leader moves its line by 2.72° on average, 1.95° after the
  model.)
- **Applied to the named run itself:** `segment_lined_up_joined` → `segment_lined_up_completed`. The rule:
  adopted if it beats `segment_lined_up_joined` tree by tree on the V opening (lower MAE and Wilcoxon
  p < 0.05); then `model.json` names `docs/results/segment_lined_up_completed` (the named recipe, with the
  model fitted on all its development trees). Otherwise `model.json` stays as it is.
- **The re-run** (`segment_lined_up_rerun_joined` → `segment_lined_up_rerun_completed`) is scored the same
  way as a replicate; it is reported, not used to decide.
- **The leaf check** (`docs/results/leaf_prior/leaf_check.py`, the student's PC) describes where the leaves
  lie around the hand-mask leaders; it decides nothing here.

**Applied (scored from the saved masks, after the commit above):** the named run: 31 trees had a short
leader completed (a = -2.78, c = +0.69 on all trees). On the V opening, tree by tree against
`segment_lined_up_joined`: MAE 4.100° → 4.102°, p = 0.78; on the 31 trees it changed, 8.33° → 8.36°
(12 closer, 15 farther). **Not adopted;** `model.json` stays as it is. The re-run replicate agrees (4.105°
→ 4.133°, p = 0.65).
