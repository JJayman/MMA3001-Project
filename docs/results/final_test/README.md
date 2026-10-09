# The final test: test rows 3, 12 and 16

Session 12. The one evaluation on the test rows, run once on the student's PC
(`python -m treeskel.final_test data/raw`; code at 4d923b3; 39 s to trace and measure the 92 test trees'
winter traces, 2.0 min to train). Until this run nothing had touched the test rows.

- **The model was named first** (`model.json`, `naming_rule.md`): the U-Net on the lined-up hand masks at
  256x256, with its masks' pieces joined (`docs/results/segment_lined_up_joined`). The full-size run did
  not beat it on development, and none of three ways of completing short predicted leaders (growing, continuing,
  shape completion) was adopted; each rule was committed before its result.
- **The recipe, as it stands:** one network trained on all 449 development trees for 34 epochs (the
  median of the epochs its 5 folds chose), seed 1000. It predicted the 92 test photos; the masks were
  joined (18 of them) and measured exactly as in development. The truth is the test trees' winter traces;
  the baseline is the mean shape of the development trees.

Files: `output.txt` (the run's output), `report.json`, `predictions.csv`, `mask_scores.csv`,
`predicted_masks.npz`, `test_shape_measures.csv` and `test_winter_traces.npz` (the test trees' winter and
hand-mask measures and traces); the secondary result in `secondary_report.json`, `secondary_output.txt`
and `predicted_masks_completed.npz`. Figures: `docs/figures/final_test_skill.png`,
`final_test_examples.jpg`.

## Results

Skill over the development mean shape (92 test trees; the one mask with no fork counts as the mean shape):

| Measure | **Named U-Net** [95% interval] | per test row (3, 12, 16) | Hand masks | Development (5 folds) | Short leaders predicted on |
|---|---|---|---|---|---|
| V opening | **+0.88** [+0.81, +0.92] | +0.88 / +0.86 / +0.89 | +0.91 | +0.75 | +0.87 |
| Left bend | **+0.66** [+0.51, +0.79] | +0.63 / +0.77 / +0.61 | +0.67 | +0.60 | +0.66 |
| Right bend | **+0.62** [+0.46, +0.75] | +0.52 / +0.72 / +0.60 | +0.65 | +0.53 | +0.62 |
| Left leader from the trunk | **+0.69** [+0.56, +0.80] | +0.65 / +0.76 / +0.70 | +0.68 | +0.64 | +0.69 |
| Right leader from the trunk | **+0.70** [+0.53, +0.81] | +0.63 / +0.60 / +0.84 | +0.71 | +0.72 | +0.67 |
| V's lean from the trunk | **+0.66** [+0.50, +0.78] | +0.59 / +0.64 / +0.77 | +0.65 | +0.68 | +0.65 |
| Trunk from vertical | **+0.62** [+0.44, +0.76] | +0.60 / +0.47 / +0.79 | +0.57 | +0.64 | +0.62 |
| Left leader from vertical | **+0.82** [+0.75, +0.87] | +0.82 / +0.81 / +0.82 | +0.86 | +0.68 | +0.82 |
| Right leader from vertical | **+0.77** [+0.67, +0.84] | +0.82 / +0.52 / +0.87 | +0.81 | +0.66 | +0.76 |
| V's lean from vertical | **+0.72** [+0.60, +0.80] | +0.74 / +0.54 / +0.83 | +0.76 | +0.58 | +0.71 |

- **The V opening, the headline:** skill +0.88, MAE 3.37° (the mean shape: 10.4°), r = 0.94 with winter;
  better than the mean shape on 79% of trees. The three rows agree (+0.86 to +0.89).
- **Close to a person reading the same photos:** on every measure the U-Net is within the hand masks'
  interval, and on the trunk's tilt a little above them (+0.62 against +0.57).
- **Higher than in development** on 7 of 10 measures (V opening +0.88 against +0.75). The test trees are
  not more varied (V opening SD 12.6° against 12.4°) and the mean shape does about as badly on them (10.4°
  against 9.8°); the model's error itself is lower (3.37° against 4.10°). Three things may contribute:
  the final network trained on all 449 trees rather than about 360 per fold; fewer test trees have a
  short predicted leader (5 of 92, against 31 of the 444 development trees scored on the V opening; with
  the masks that found no fork, 6 against 33), though this explains little: those 5 test trees are not
  predicted worse than the rest (V opening MAE 2.8° against 3.4°), and without the short-leader and
  no-fork trees the test error is still lower (3.3° against 3.6°); and the test rows seem easier to read
  (the hand masks also score higher, +0.91 against +0.87). The first and the last cannot be told apart.
- **Pixels (secondary):** against the test rows' hand masks as drawn, median Dice 0.23, and 0.82 after the
  best shift of each (an upper estimate). The test masks were not lined up; they sit off the branches as
  the development masks did (0.22 → 0.80 there), and the U-Net's masks sit on the branches.

## The secondary result: short leaders predicted on

Declared before the test (`naming_rule.md`, `model.json`'s `secondary`): the same masks with short
leaders' lines completed by the model fitted on all development trees (`a = -2.78`, `c = +0.69`) and drawn
on (`predicted_masks_completed.npz`).

- Only 5 of the 92 test trees had a short predicted leader (none a stub). Their completion changed the V opening's MAE from 3.37°
  to 3.52° (1 tree closer, 4 farther; p = 0.14): as in development, no better than leaving them
  short, and no significant loss.
- **Computed after the run, from its saved files** (`python -m treeskel.final_test --secondary-from-saved`):
  the pull request that declared it was merged before its last commit, so the run used the code without
  it. The calculation is the one the run would have made (the same saved, joined masks and the same
  completion model; a test checks that the two give identical scores); nothing was trained or chosen
  again, and the headline `report.json` is the run's own.

## Examples (`final_test_examples.jpg`, worst to best V opening)

- **R03N32 (worst):** the one mask with no fork: its left leader is drawn but lies 22.4 px from the trunk
  at the fork, beyond the joining gap (19.3 px), so the tree counts as the mean shape (53° against
  winter's 69°).
- **R03N04, R12N34, R16N23, R03N15, R16N04:** within 0–5° of winter (R16N04: 47° and 47°).

## What it rests on, and its limits

- **One evaluation, named first:** every choice (the network, the lined-up labels, joining, the full-size
  and completion variants) was made on the development rows, each by a rule committed before its result.
- **Only three rows:** the intervals resample trees, and trees of one row share light and background, so
  they are narrower than intervals over rows would be. The three rows' skills are given; they agree on the
  V opening (+0.86 to +0.89) and the left leader's tilt, while row 12 is lower on the right leader's tilt
  and the V's lean from vertical, and the bends and the trunk-relative angles vary more between rows.
- **The winter traces are the truth throughout,** traced from the student's clicks and never used to train
  or choose anything; the scale between the seasons and the camera's roll limit the angles measured from
  the photo's vertical (session 10).
