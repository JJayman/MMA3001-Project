# The mask model: the centre-line loss and an ensemble (session 16)

**Status:** the runs are done (the student's PC, 11–12 October 2026). The rule passed both changes, but its
own null check shows that its test is too lenient between trainings. **The student chose option B**
(12 October): an ensemble of the named recipe, reported once on the test rows
([`../final_test_ensemble/`](../final_test_ensemble/)). The centre-line loss is not used.

## Files

- **The plan and its rule:** [`rule.md`](rule.md), fixed and confirmed before any run.
- **Before any run:** [`loss_check.py`](loss_check.py), with its output in
  [`loss_check_output.txt`](loss_check_output.txt).
- **`python -m treeskel.seg_compare compare data/raw`** wrote three files:
  - [`output.txt`](output.txt): the rule's verdict and the pre-declared diagnostics;
  - [`report.json`](report.json);
  - [`trees.csv`](trees.csv): per tree and run, the error and the mask's structure.
- **After the runs:** [`checks_after_runs.py`](checks_after_runs.py), with its output in
  [`checks_after_runs_output.txt`](checks_after_runs_output.txt) and the figure
  [`docs/figures/segment_variants.png`](../../figures/segment_variants.png).

## The runs

- **Eight trainings**, all with batch 8 as fixed:
  - the named recipe with seeds 1–5, 17.5–23.4 minutes each;
  - the centre-line loss with seeds 0–2, 33.5–36.1 minutes each.
- **Session 12's re-run is seed 0** of the named recipe.
- **Three ensembles:**
  - the named recipe from the re-run and seeds 1–2;
  - a second, independent one from seeds 3–5;
  - the centre-line loss from its seeds 0–2.

All scores are on the 449 development trees, out of fold, with the masks joined. In the V-opening MAE, a
mask with no fork or no V opening counts as the fold's mean shape.

| Run | V opening MAE | Skill (run's own report) | Incomplete masks | Hand masks' centre line covered |
|---|---|---|---|---|
| Named run (session 11) | 4.224° | +0.750 | 38 | 0.871 |
| Named recipe, 6 more trainings | 3.997–4.471° | +0.689 to +0.807 | 38–58 | 0.862–0.868 |
| Centre-line loss, 3 trainings | 4.076–4.219° | +0.750 to +0.770 | 30–43 | 0.881–0.884 |
| Named recipe, ensemble A / B | 4.293° / 4.150° | +0.728 / +0.753 | 48 / 38 | 0.873 / 0.873 |
| Centre-line loss, ensemble | **4.004°** | **+0.803** | 31 | **0.891** |

## The rule's verdict, as fixed

`seg_compare.decide` ([`output.txt`](output.txt)):

| Change | V opening MAE, tree by tree | Guards | Verdict |
|---|---|---|---|
| Centre-line loss (3 runs) against the named recipe (re-run, seeds 1–2) | 4.375° → 4.140°, lower on 55%, p = 0.009 | side branches 0.028 → 0.022 per tree (p 0.07); incomplete 0.116 → 0.085 (p 2e-4) | **passed** |
| Ensemble against its 3 members | 4.375° → 4.293°, lower on 53%, p = 0.002 | side branches p 0.75; incomplete p 0.52 | **passed** |

The rule therefore names the **centre-line ensemble**: V opening 4.004°, skill +0.80 in its own report.

## The null check: the rule's test is too lenient between trainings

The rule's pre-declared null check compares samples of one recipe with each other. With seeds 3–5 there are
seven trainings of the named recipe, so the check can be made in exactly the rule's form
([`checks_after_runs_output.txt`](checks_after_runs_output.txt), section 1).

**Seven trainings of the same recipe range from 3.997° to 4.471°** (SD 0.17°). That is more than any
change moved the V opening.

| Same recipe, compared with the rule's test | Called different (p < 0.05) | Expected by chance |
|---|---|---|
| One training against another (21 pairs) | 5 | about 1 |
| Three trainings against the other three (10 splits) | 3 | about 0.5 |

- **The rule's own baseline triple is "significantly worse" than its other triple:** 4.375° against
  4.206°, p = 0.027, yet both are the same recipe.
- **Why:** the Wilcoxon test counts each tree as an independent piece of evidence. A training, though, is
  better or worse on many trees at once, and that shared shift is invisible to a tree-by-tree test.
  Averaging three trainings does not remove it.
- **Before the run,** a single pair (the named run against the re-run, p = 0.93) had looked reassuring.
  The rule's design was the assistant's, and this is its error (see the session notes).

## What survives the check

### The centre-line loss: a robust effect on the masks, not established on the V opening

| | Named recipe (7 trainings) | Centre-line loss (3) | Run-level p (permutation, one-sided) |
|---|---|---|---|
| V opening MAE | 4.281° | 4.140° | **0.11** |
| Incomplete masks per training | 47.6 | 38.3 | 0.09 |
| Hand masks' centre line covered | 0.866 | 0.883 | **0.008** (all 3 above all 7) |

- **Against the other triple** (seeds 3–5) instead of the rule's: 4.206° → 4.140°, p = 0.42.
- **What the loss reliably does:** it raises how much of the labelled branches the masks cover. It probably
  leaves fewer incomplete masks.
- **What is not established:** that it predicts the V opening better. The comparison is at the level of
  trainings, the level at which the evidence is honest. That test was chosen after the runs, so it is a
  check on the rule, not a new rule.
- **Cost:** a training takes about twice as long (35 against 18 minutes).

### Ensembling: a small, repeated gain, and half the disagreement between trainings

**Each ensemble against its own members.** An ensemble shares its members' training-level shifts, so they
cancel in this comparison:

| Ensemble | Members' average → ensemble | Lower on | p |
|---|---|---|---|
| Named recipe A (re-run, seeds 1–2) | 4.375° → 4.293° | 53% | 0.002 |
| Named recipe B (seeds 3–5; deciding nothing) | 4.206° → 4.150° | 54% | 0.006 |
| Centre-line loss | 4.140° → 4.004° | 56% | 0.003 |

- **The gain repeats** on all three ensembles, including the independent one: 0.06–0.14°.
- **The incomplete masks do not increase,** so the worry that averaging would lose leaders found by one run
  did not materialise.
- **Stability** (the student's reason for the ensemble):

  | | Single trainings (21 pairs) | The two ensembles |
  |---|---|---|
  | V opening apart, median | 0.67° | **0.36°** |
  | Trees more than 5° apart | 4.5% | 2.9% |
  | Where either mask is incomplete | 1.94° (28% > 5°) | **1.03°** (23%) |
  | Incomplete status differs | 8.7% | 6.2% |

  The two ensembles are one pair. Their agreement is a single observation, but a large one.
- **The members' spread as a warning:** it works for the named recipe's ensembles, though not for the
  centre-line one.

  | Ensemble | Spearman ρ with the error | MAE on the 10% most spread / the rest | Share of the 10% largest errors flagged |
  |---|---|---|---|
  | A | +0.14 (p 0.003) | 9.1° / 3.8° | 38% |
  | B | +0.15 (p 0.001) | 9.5° / 3.6° | 40% |
  | Centre-line | +0.07 (p 0.12) | 6.7° / 3.7° | 24% |

  By chance, 10% of the largest errors would be flagged.

## The decision (the student's, 12 October 2026): B

- **A. As the rule says:** the centre-line ensemble, reported once on the test rows. The report must add
  that the null check found the rule's test too lenient, so its centre-line half is not established on
  the V opening.
- **B. Only what survives the check:** an ensemble of the named recipe, reported once on the test rows.
  The centre-line loss is reported as raising coverage, with its effect on the V opening not established.
- **Either way:**
  - the final test (+0.88) stays the headline;
  - "three trainings averaged" is the change with evidence behind it;
  - the test-row report's code is written after the choice, and it runs once.

**Chosen: B.** The test-row report is specified in
[`../final_test_ensemble/README.md`](../final_test_ensemble/README.md) and its `model.json`, both committed
before it runs:
- the final test's recipe and 34 epochs;
- three networks (seeds 0–2), their probabilities averaged.
