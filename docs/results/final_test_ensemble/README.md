# An ensemble of the named U-Net on the test rows (session 16, secondary)

**Status: run once (the student's PC, 12 October 2026).** [`model.json`](model.json) and this file's
specification were committed before the run. **The ensemble showed no gain on the test rows** (see Results).

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

The run's own output is in [`output.txt`](output.txt) and [`report.json`](report.json). The like-for-like scores
are in [`like_for_like.py`](like_for_like.py) and its output,
[`like_for_like_output.txt`](like_for_like_output.txt).

**Training:** three networks in 5.8 minutes in all. No ensemble mask was without a fork, and 17 masks were
joined.

### Every network on the same 92 test trees

A mask with no fork or no V opening counts as the development mean shape:

| V opening, test rows 3, 12, 16 | Stubs | MAE | Skill [95% interval] |
|---|---|---|---|
| **The final test's network** (session 12, the headline) | 0 | **3.37°** | **+0.88** [+0.81, +0.92] |
| Network seed 0 (the same recipe and seed, retrained) | 0 | 3.63° | +0.85 [+0.77, +0.91] |
| Network seed 1 | 0 | 3.88° | +0.81 [+0.65, +0.90] |
| Network seed 2 | 2 | 3.64° | +0.86 [+0.78, +0.91] |
| **Ensemble of seeds 0–2** | 0 | **3.78°** | **+0.81** [+0.64, +0.91] |

- **No gain.** The ensemble is no closer to winter than its networks' average (3.72° → 3.78°, lower on 45% of
  trees, p 0.65), nor than the final test's network (3.37° → 3.78°, p 0.68).
- **The four single trainings range 3.37–3.88°** (skill +0.81 to +0.88). The final test's network was the best
  of the four. A network of this recipe should be expected to score about +0.81 to +0.88 on these rows, not
  exactly +0.88.
- **Where the ensemble lost: the two short-leader trees of session 14.**
  - On **R03N10** (winter 62.6°), the final test's network read 63.1°, and so did seed 0 (58.9°). Seed 1 read
    89.4°, and seed 2 left a stub. The average followed the two failures, to 91.6°.
  - On **R16N12** (winter 71.4°), it is the same: 70.4° from the final test's network and 70.0° from seed 0.
    Seed 1's mask had no fork and seed 2's was a stub, and the ensemble read 54.0°.
  - Where two of three networks fail on a tree, the average fails too. That is the "vote" the rule's
    guard was meant to catch. It did not show in development, but it does here.
- **Where it helped:** R03N32, the final test's one mask with no fork (52.8° against winter 68.8°). Two
  networks found the fork, and the ensemble read 62.7°.

### The run's printed comparisons were wrong, and are corrected here

- **What went wrong:** the first version of the code left a network's stubs out. Seed 2's printed score
  (+0.88) was therefore over 90 trees, without its two failures. The ensemble-against-networks comparison
  dropped the same two trees, and printed 3.51° → 3.34° on 90 trees.
- **What was done:** the code now counts stubs as the mean shape in these comparisons, as session 16's
  development comparisons do (a test checks this). The numbers above are recomputed from the saved files;
  nothing was run again.

### What this means

- **The final test stays the headline:** +0.88, as fixed in session 12.
- **The development gain does not carry over.** Ensembling's gain in development (0.06–0.14° on 449 trees) is
  smaller than the test rows' noise between trainings (0.5° on 92 trees). On the trees that matter most, the
  short leaders, averaging can lose a leader that one network found.
- **What ensembling does reliably** is make retrainings agree more (development: 0.67° → 0.36° apart). It
  does not make them more accurate.
- **Seed 0, retrained with the final test's own recipe and seed, scores 3.63° against 3.37°.** This is
  session 14's finding again: a GPU training is a sample, and the test rows hold 92 trees.
