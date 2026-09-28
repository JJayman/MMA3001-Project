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
