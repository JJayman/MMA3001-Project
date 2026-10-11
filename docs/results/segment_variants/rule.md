# Improving the mask model: the centre-line loss and an ensemble (session 16)

**Status: confirmed by the student on 11 October 2026, before any run.** The student answered the three
open points:
1. the rule as written: yes;
2. an adopted change is reported once on the test rows: yes;
3. the second ensemble is run: yes.

The commit that adds this file is the record, and the commit that records these answers came before the
first run. Nothing here changes after a run has started. The code that applies
the rule is `treeskel.seg_compare` (`decide`, `verdict`); its tests are in `tests/test_seg_compare.py`.

## What stays as it is

- **The named U-Net and the final test stay the headline:** V opening +0.88 [+0.81, +0.92] on test rows
  3, 12 and 16 (`docs/results/final_test/`). They are the baseline here, and nothing below replaces them.
- **The test rows are not read** by anything in this experiment. They were used once, in session 12, and
  can no longer be an untouched test. Every result below is a development result: 449 trees, 5 row-grouped
  folds, each tree predicted by networks that never saw its row.
- **The test rows after the verdict (the student's decision):** if a change is adopted, it is reported
  **once** on the test rows. It is a secondary line beside the final test, with the caveat that those rows
  were already used in session 12. If nothing is adopted, the test rows are not run again.
  - **The network:** the adopted recipe trains on all 449 development trees, as the final test's did. Its
    epochs are the median of the epochs its development folds chose.
  - **An adopted ensemble** is three such trainings, seeds 0, 1 and 2, with their probabilities averaged.
  - **The scoring:** the 92 test trees, against their winter traces, exactly as in the final test.
  - **The code:** written after the verdict and before that run, and it runs once, as `final_test` does.
    It cannot change the verdict.
  - A fresh, independent test would need new data.

## The weakness being targeted

- **Short leaders:** on 31 of 444 development trees, a predicted leader stops short under the leaves.
  Those trees carry a third of the V opening's squared error (session 14).
- **Instability:** two trainings of the same recipe agree within about a degree on complete masks, but
  short-leader masks differ by a median 1.3°, and 27% of them by more than 5° (session 14).
- **Session 12's ideas did not help:** growing, continuing and completing all changed finished masks, and
  the leaf check found no usable prior. The two changes here act on the network instead: how it is
  trained, and how many networks vote.

## The candidates

### 1. The centre-line loss (`segment --cldice`)

- **The loss:** `BCE + (1 − α)·Dice + α·clDice`, with `α = 0.5`.
  - This is Shit et al.'s (2021) combination `(1 − α)·Dice + α·clDice`, at the default of their public
    implementation (`soft_dice_cldice`, checked in their repository), with the named recipe's
    cross-entropy kept.
  - `α` is fixed, not tuned (`treeskel.segment.CLDICE_WEIGHT`).
- **The soft skeleton:** 25 erosions (`CL_ITERS`).
  - A branch whose distance transform peaks at `d` px needs `d − 1` erosions, or its centre line is lost
    (tested on synthetic bars and discs).
  - The widest local development hand mask peaks at 17.0 px, which is 19.0 px at the augmentation's
    largest zoom.
  - The authors' 10 erosions, chosen for thin vessels, would leave the trunk out of the loss.
- **Everything else is the named recipe:** U-Net with a ResNet-18 encoder, 256x256, the lined-up masks,
  and epochs chosen on the inner fold. The epochs are chosen by the same loss the network trains on.
- **Expected size of the effect, measured before any run** (`loss_check.py`, `loss_check_output.txt`).
  On the 21 local development hand masks, the left leader was cut on purpose:

  | Cut | clDice penalty ÷ Dice penalty, median [range] | Overlap terms at α = 0.5, ÷ Dice alone |
  |---|---|---|
  | Leader stops halfway | 1.63 [1.17, 2.27] | **1.32×** |
  | 8 px gap | 1.21 [0.53, 2.96] | **1.11×** |

  - So clDice weights a branch by the length of its centre line, not its area. It is not a connectivity
    loss: a small break costs about as little as before.
  - The prediction is therefore a moderate effect on short leaders, and little on breaks. Joining
    (`join_pieces`) still handles the breaks.
  - The concern that clDice could also join the wrong branches is covered by the side-branch guard below.
- **Not the same as session 12.** There clDice only *chose* a threshold for growing finished masks
  (not adopted). Here it *trains* the network.

### 2. An ensemble of three trainings (`seg_compare ensemble`)

- **How:** the same recipe trained with seeds 0, 1 and 2. Each tree's out-of-fold probabilities are
  averaged over the three networks, then thresholded at 0.5 and joined (19.3 px), exactly as one
  network's would be.
- **Every member is out of fold:** each one predicted a tree with a network that never saw that tree's
  row. The code refuses to build an ensemble from members whose folds differ.
- **What averaging does:** it is close to a vote. Where one network finds a leader that the other two miss,
  the mean probability usually stays below 0.5, so the ensemble can *lose* leaders that one run had.
  That concern is covered by the guard on short or missing leaders below.

## The runs (the student's PC)

| Run | Recipe | Seed | New? |
|---|---|---|---|
| `segment_lined_up_rerun` | named | 0 | no: session 12's re-run (probabilities saved) |
| `segment_lined_up_seed1`, `_seed2` | named | 1, 2 | yes |
| `segment_lined_up_cldice`, `_cldice_seed1`, `_cldice_seed2` | centre-line loss | 0, 1, 2 | yes |

**A seed** sets the decoder's first weights, the order of the photos and the augmentation. Training on a
GPU adds its own non-determinism on top.

**The re-run counts as seed 0 of the named recipe.**
- The unchanged code trains it bit for bit as before: 182 weight tensors were identical to `main`'s on a
  CPU run, and a test pins the loss.

**Run time.** The named recipe took 20.4 and 18.3 minutes a run (sessions 11 and 12). The centre-line runs are slower: the soft
skeleton adds 25 erosions to every training step, and this has not been measured on a GPU.

**Commands.** `--no-grow` skips session 12's grown masks, which were not adopted. Each command also writes
the run's joined masks, `<run>_joined`.

```
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 1 --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 2 --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --cldice --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --cldice --seed 1 --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --cldice --seed 2 --no-grow
python -m treeskel.seg_compare ensemble docs/results/segment_lined_up_rerun docs/results/segment_lined_up_seed1 docs/results/segment_lined_up_seed2 --out docs/results/segment_lined_up_ensemble --data data/raw
python -m treeskel.seg_compare ensemble docs/results/segment_lined_up_cldice docs/results/segment_lined_up_cldice_seed1 docs/results/segment_lined_up_cldice_seed2 --out docs/results/segment_lined_up_cldice_ensemble --data data/raw
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 3 --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 4 --no-grow
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 5 --no-grow
python -m treeskel.seg_compare ensemble docs/results/segment_lined_up_seed3 docs/results/segment_lined_up_seed4 docs/results/segment_lined_up_seed5 --out docs/results/segment_lined_up_ensemble_b --data data/raw
python -m treeskel.seg_compare compare data/raw
```

The last four training commands are the second ensemble (the student's decision; it decides nothing).

Then push `docs/results/` and `docs/figures/`.

## The rule (`seg_compare.decide`)

**The measure.**
- The V opening against the winter traces, on the joined masks, tree by tree, over all 449 development
  trees.
- A mask with no fork counts as its fold's mean shape, as always.
- **New here:** a mask with a fork but no V opening (a leader too short for a line, session 14's "stubs")
  also counts as the mean shape. The runs' own reports leave such trees out. Under a paired rule that
  would let a variant drop its worst trees by making stubs. The MAE here is therefore slightly higher than
  in the reports: the named run gives 4.22° instead of 4.10°.

**A change is adopted if both of these hold:**
1. its mean absolute error on the V opening is **lower**, with a Wilcoxon signed-rank test on the paired
   per-tree errors giving **p < 0.05** (the project's naming rule since session 12); **and**
2. it does **not significantly increase** either guard. A guard fails if its per-tree mean is higher with
   p < 0.05 (Wilcoxon):
   - **side branches:** stretches of centre line that reach more than a trunk radius (9.65 px) beyond the
     trunk's and leaders' outline;
   - **incomplete masks:** no fork, or a leader shorter than the fitted stretch (157.7 mask px).

**What each change is compared with.**

| Change | Candidate | Comparator |
|---|---|---|
| Centre-line loss | its 3 runs, each tree's values averaged over the 3 | the named recipe's 3 runs, averaged the same way |
| Ensembling | the named recipe's ensemble | its own 3 members, each tree's values averaged (the error one training makes on that tree, on average) |

**The named candidate.**
- It is the recipe whose loss passed (the centre-line loss if it passed, else the named loss), ensembled
  if ensembling passed. If neither passed, the named U-Net stays.
- The two changes are decided separately. The combination, the centre-line ensemble, is reported beside
  them.
- A *recipe* is named, not a run: any later training (`model3d train`, a test-row report) is a new sample
  of it.

**Why not just compare with the named run.**
- The named run is one sample of its recipe. A single new run against it would mix the change with the
  run-to-run differences of session 14.
- Averaging each tree over three samples per recipe reduces that noise, and the ensemble's members provide
  exactly those samples. This is why the plan trains three runs of each recipe, not one.
- The comparisons with the named run itself are still reported, as descriptions.

## Reported, deciding nothing

- **Null check:** the four samples of the named recipe (the named run, the re-run, seeds 1 and 2) against
  each other: six pairs, each with the rule's test. This shows how often the test calls one recipe's samples
  different.
  - Before this session, the named run against the re-run gave p = 0.93 on the V opening.
  - Across all ten measures, on the runs' own predictions, p ranged from 0.11 to 0.99.
- **Stability:** for each recipe, how far its samples are apart tree by tree:
  - the median difference in the V opening;
  - the share of trees more than 5° apart;
  - the share whose incomplete status flips.

  The student's motivation for the ensemble is stability. One ensemble cannot show that an ensemble is
  stable; that needs a second, independent one. **Run (the student's decision):** seeds 3, 4 and 5 of the
  named recipe, ensembled into `docs/results/segment_lined_up_ensemble_b`. `compare` then reports how far
  the two ensembles are apart, against how far single runs are apart. This adds about 60 minutes of GPU
  time. Seeds 3–5 are in no comparison of the rule.
- **Disagreement as a warning:** whether the members' spread on the V opening ranks the trees the ensemble
  gets wrong.
  - It counts as useful if Spearman's ρ is above 0 with p < 0.05.
  - Also reported: the ensemble's error on the 10% most-spread trees against the rest, and the share of
    its 10% largest errors among them.
  - It uses winter only to check the flag, not to choose anything.
- **Each run's structure:**
  - forks found;
  - incomplete masks and short leaders;
  - side branches, stray pieces and the extra area;
  - against the lined-up hand masks: clDice, topology sensitivity (how much of the labelled branches the
    prediction covers), topology precision, and pixel Dice.
- **All ten shape measures** in each run's own report, as before.

## The label audit (`treeskel.label_audit`, `docs/results/label_audit/`)

The student's third item: whether the lined-up hand masks are still locally off. A mask can be moved as a
whole onto the trunk while a leader is drawn beside the wood. The line-up raised the skill from +0.06 to
+0.72, so label position matters more than anything tried since.

- **The sample:** 40 development trees, drawn at random within each row in proportion to its trees (at
  least one per row, all 16 rows), with seed 16.
  - It was drawn from the committed files only, and committed before anyone looked:
    `docs/results/label_audit/sample.csv`.
  - With 40 trees, a 95% interval is about ±10–15 points: enough to tell rare from common.
- **The verdicts (the student's, by eye):** each of trunk, left leader and right leader is judged on the
  image from `python -m treeskel.label_audit draw data/raw`:
  - **on:** where the wood is visible, the outline follows it to within about half the branch's width;
  - **off:** along a stretch at least a trunk's width long, the mask lies beside the visible wood by about
    the branch's width or more;
  - **hidden:** too little wood is visible to judge.

  The images show the photo and the hand mask's outline only, never a prediction.
- **What follows, fixed now:**
  - **Under 10% of judged leaders off:** label position is reported as a minor, measured limitation.
  - **Over 25% off:** a local correction of the masks is proposed as the next experiment, with its own
    rule.
  - **In between:** reported, and the student decides.
- **When:** the audit needs no GPU, so it can be done while the runs train.

## A larger network

Not part of this plan; the student ranked it last. Session 12's whole-photo run (480x640) and the three
ways of extending masks did not improve the geometry. Session 13 showed the named U-Net's accuracy does
not come from more computation.
