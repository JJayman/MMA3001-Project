# The automatic winter baseline against the guided ground truth

Session 13, open since session 4. `python -m treeskel.score_baseline data/raw` on the student's PC
(449 development trees, about 2 minutes). Output: `output.txt`; per tree: `trees.csv`; all numbers:
`report.json`; figure: `docs/figures/winter_baseline_examples.jpg`. The test rows were not loaded.

## What was compared

Both methods work on the same winter photo, so no alignment is needed:

- **the automatic baseline** (`treeskel.winter_gt`, session 4): the tree picked out of the winter depth,
  thinned to centre lines, and the leader on each side chosen as the path that climbs highest; its own
  quality check flags a tree when a leader rises too little or no leader is found;
- **the guided ground truth** (`treeskel.guided_gt`): the traces through the student's clicks.

Scores fixed before the run (nothing was chosen with them):

- *found*: the share of the guided trunk and leaders the baseline passes through (within each point's
  drawn radius); *on the branch*: the share of the baseline's points inside the guided mask; their
  harmonic mean, **F1**. A tree counts as **traced** at F1 ≥ 0.9 (looser: 0.8);
- the distance between the two forks, and the same shares at 5, 10 and 20 px;
- the ten shape measures, scored exactly as the U-Net's (skill over the same mean shapes, 5 folds).

The guided traces measured here reproduce the committed winter measures (largest difference 0.006°).

## Results (449 development trees)

| | Trees | F1, median [quartiles] | Traced (F1 ≥ 0.9) | F1 ≥ 0.8 | Fork distance, median |
|---|---|---|---|---|---|
| All | 449 | 0.86 [0.67, 0.95] | **43%** | 60% | 11 px |
| Passed its quality check | 340 | 0.91 [0.82, 0.96] | 57% | 78% | 9 px |
| Flagged | 109 | 0.52 [0.22, 0.68] | 0% | 3% | 79 px |

- **The quality check catches failures but not all of them.** No flagged tree is traced (0 of 109), but
  **147 of the 340 passed trees (43%) are not traced** either. It flags leaders that stop short; it
  cannot see a leader that runs up the wrong branch.
- **What the baseline draws is mostly right; what it misses is the problem.** Median 0.94 of its points
  lie on the branch, but it finds a median 0.81 of the branches.
- **The student's session-4 verdicts agree with the scores.** The 38 trees judged wrong by eye in rows
  1, 7 and 9 have median F1 0.67 and 2 of 38 are traced; the other 72 trees of those rows have median
  F1 0.93 and 58% are traced (Mann-Whitney p = 5e-10). Every failure type scores low; "wrong fork" has
  the largest fork distance (median 299 px).

### Shape (skill over the mean shape; MAE in brackets)

| Measure | Baseline (no fork = mean shape) | Baseline, passed trees only | Lines through the clicks | Hand masks | **U-Net (summer)** |
|---|---|---|---|---|---|
| V opening | −1.97 (8.6°) | +0.34 (4.6°) | +0.36 (8.0°) | +0.87 (3.3°) | **+0.75 (4.1°)** |
| Left / right bend | −1.10 / −2.53 | +0.01 / −0.17 | +0.10 / −0.06 | +0.70 / +0.61 | **+0.60 / +0.53** |
| Trunk from vertical | −0.08 | +0.16 | +0.93 | +0.70 | **+0.64** |

V opening error per tree (degrees):

| | Median | Within 5° | More than 20° off | Largest |
|---|---|---|---|---|
| Baseline (winter photo) | 1.9 | 70% | **14.3%** | 227 |
| U-Net (summer photo) | 2.8 | 74% | 1.4% | 47 |
| Hand masks (summer) | 2.6 | 79% | 0.2% | 42 |
| Mean shape | 7.6 | 34% | 12.2% | 43 |

- **When it works, the baseline is the most precise:** on its 193 traced trees the V opening is within
  0.9° of winter (median), 98% within 5°. It sees the bare tree.
- **When it fails, it fails badly:** 14% of trees are more than 20° off (a side shoot taken for a
  leader, another tree, a post), so its squared error is larger than the mean shape's and its skill is
  negative. Tree by tree against the U-Net: closer on 56% of trees for the V opening (p = 0.82); worse on
  the measures from the trunk (closer on 45–48% of trees, p ≤ 0.009); better on the leaders' and the V's
  angles from the vertical (closer on 61–65%, p ≤ 0.005), as expected: it measures the winter photo
  itself, so the camera's roll between the seasons does not enter.
- **The summer U-Net has the higher skill on every measure**, although it sees the tree through the
  leaves: what it learned from the clicked ground truth fails far less badly than a hand-written rule
  applied to the bare tree.
- **The straight lines through the clicks alone** (no tracing) give the trunk's tilt well (+0.93) but no
  bends (+0.10 / −0.06) and a poor V opening (+0.36): the leaders curve, so tracing them was needed.

## What it rests on, and its limits

- The levels (F1 0.9 and 0.8) are descriptive, set before the run; the table shows both.
- Each method's run time: baseline 0.16 s per tree, guided tracing 1.2 s plus about 15 s of clicking
  (`docs/results/benchmark/`).
- The session-4 list holds examples (38 trees) of the about 45 judged wrong, made on the baseline before
  the depth registration of session 9; some of "the others" are wrong too (42% are not traced).
- **Conclusion:** the automatic method is right on about 4 trees in 10, its own check cannot say which,
  and a wrong trace gives a wildly wrong shape. That is why the ground truth was clicked (about 1 h 50 min
  for the 449 development trees, session 5).
