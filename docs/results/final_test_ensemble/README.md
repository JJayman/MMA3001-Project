# An ensemble of the named U-Net on the test rows (session 16, secondary)

**Status: specified; not run yet.** [`model.json`](model.json) and this file are committed before the run,
and `python -m treeskel.final_test data/raw --out docs/results/final_test_ensemble` runs once.

## Why

The student chose option B after session 16's development runs ([`../segment_variants/`](../segment_variants/)):

- Only one change survived the null check: averaging three trainings.
  - On all three development ensembles it beat its own members, by 0.06–0.14° on the V opening
    (p ≤ 0.006).
  - It halved the disagreement between retrainings (median 0.67° → 0.36°).
- The centre-line loss is not used: its gain on the V opening is not established.
- The rule fixed before the runs said an adopted change is reported **once** on the test rows, as a
  secondary line.

## What it is

- **The final test's recipe, unchanged.** `run` is `docs/results/segment_lined_up_joined`:
  - the U-Net with a ResNet-18 encoder at 256x256, trained on the lined-up hand masks of all 449
    development trees;
  - **34 epochs**, the final test's own number (the median of the named run's folds);
  - masks joined within 19.3 px;
  - no centre-line loss.
- **Three networks, seeds 0, 1 and 2.** Each is trained from its seed's first weights with training seed
  1000 + seed, as the final test's network was with seed 0.
  - So network 1 repeats the final test's training. On a GPU it is a new sample (session 14), not the same
    network.
  - A test checks that on a CPU it reproduces the final test tree for tree.
- **The average of their probabilities** on the 92 test photos is thresholded at 0.5, joined and measured,
  exactly as in the final test.
- **The truth:** the test trees' winter traces, measured again by the same code.
- **The development counterpart** is `segment_lined_up_ensemble_joined` (the re-run and seeds 1–2): V
  opening skill +0.73, against the named run's +0.75. The two development ensembles scored +0.73 and
  +0.75; the named run lies between them.

## What it reports (fixed now)

1. **The ensemble's skill** on all ten measures, with 95% bootstrap intervals and per test row. It is
   reported beside the final test's (+0.88 on the V opening), not instead of it.
2. **Each of the three networks on its own.** These are three more single-network samples on the test rows,
   showing how much retraining moves the test score.
3. **Tree by tree on the V opening:**
   - the ensemble against its networks' average;
   - the ensemble against the final test's network (`compare_with`).

   These comparisons are descriptive. The development check showed that one network against another is a
   weak comparison (seven trainings of one recipe ranged 4.00–4.47°).

It decides nothing. The final test stays the headline.

## Results

Not yet.
