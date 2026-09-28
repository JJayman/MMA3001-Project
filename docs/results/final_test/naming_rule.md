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
