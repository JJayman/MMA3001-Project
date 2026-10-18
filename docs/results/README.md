# Results: one folder per experiment

Each folder holds the numbers of one experiment or training run. Most have a README that says what was done,
the numbers and what was decided. The folders without one are raw training runs, explained by the README
named beside them below. The folders are grouped in the order of the work, the headline first.

The folder names are used as paths by the code and the tests (`treeskel.model3d`, `final_test`, `seg_compare`),
so they are kept as they were made.

## The headline

| Folder | What it holds |
|---|---|
| [`final_test/`](final_test/) | **The one evaluation on the test rows** (3, 12, 16; session 12): V opening skill +0.88 [+0.81, +0.92]. `naming_rule.md` is the rule that named the model before the results; `model.json` names it. |
| [`segment_lined_up_joined/`](segment_lined_up_joined/) | **The named model on the development rows** (5 row-grouped folds, pieces joined): V opening skill +0.75. Its masks are the ones `model3d` shows for development trees. |

## The ground truth (winter)

| Folder | What it holds |
|---|---|
| [`tune_cost/`](tune_cost/) | Tuning the click-guided tracer's cost map on the 17 tuning trees (session 5). |
| [`depth_registration/`](depth_registration/) | The winter depth sits about 5 px below the colour image; the shift applied on loading (sessions 8–9). |
| [`leader_width/`](leader_width/) | How wide each leader is drawn, measured across the branch in colour (session 8). |
| [`fork_junction/`](fork_junction/) | The fork drawn as a junction instead of the trunk's round top (session 8). |
| [`fork_shoulders/`](fork_shoulders/) | The earlier shoulder fill at the fork (session 6). Superseded in session 8. |
| [`winter_baseline/`](winter_baseline/) | The automatic winter tracer, kept as a baseline, scored against the guided ground truth (session 13). |

## From winter to summer

| Folder | What it holds |
|---|---|
| [`align/`](align/) | Lining up the winter photo on the summer photo pixel by pixel. **Abandoned** in session 7: too little rigid structure shows. |
| [`shape/`](shape/) | The tree's shape measured without alignment (V opening, leader angles), and the first summer models: ridge regression on CNN features and a fine-tuned angle CNN (session 10). Covers `shape_cnn/`, `shape_cnn_480x640/`, `shape_cnn_depth/` and `shape_cnn_runs.json`. |

## The summer model: a U-Net on the hand masks

| Folder | What it holds |
|---|---|
| [`segment/`](segment/) | The first U-Net, trained on the hand masks as drawn (session 11). Its README also covers `segment_lined_up/`, the run on the masks lined up by hand. |
| [`segment_lined_up_joined/`](segment_lined_up_joined/) | That run's masks with their pieces joined: the named model (above). |
| [`segment_lined_up_480x640/`](segment_lined_up_480x640/) | The whole photo at full size (session 12; with `_joined`). Not adopted, by the naming rule. |
| [`segment_lined_up_rerun/`](segment_lined_up_rerun/) | Completing short predicted leaders: a re-run of the named recipe, then growing, continuing and shape completion (session 12). Covers `segment_lined_up_rerun_grown/`, `_continued/` and `segment_lined_up_completed/`; `segment_lined_up_rerun_joined/` and `_rerun_completed/` are in `final_test/naming_rule.md`. None adopted; the completed masks were reported beside the final test as a secondary line. |
| [`leaf_prior/`](leaf_prior/) | Whether the leaves locate a hidden leader (session 12): a description, not a method. They show which side the V is on, too gently to place a leader. |
| [`benchmark/`](benchmark/) | Run time and memory of every method (session 13); the networks' FLOPs, memory traffic, roofline and profile (`--flops`, session 18: `flops.json`). |

## Improving the mask model (session 16)

| Folder | What it holds |
|---|---|
| [`segment_variants/`](segment_variants/) | **The README for all of session 16's runs:** the centre-line loss and ensembles against the named recipe, the rule fixed before the runs (`rule.md`), and the null check that showed the rule's test too lenient. Covers the raw runs below, each with its `_joined` twin. |
| `segment_lined_up_seed1/` … `segment_lined_up_seed5/` | Five more trainings of the named recipe (seeds 1–5). `segment_lined_up_rerun/` is seed 0. |
| `segment_lined_up_cldice/`, `_cldice_seed1/`, `_cldice_seed2/` | The centre-line (clDice) loss, seeds 0–2. |
| `segment_lined_up_ensemble/`, `_ensemble_b/` | Three trainings of the named recipe averaged: seeds 0–2, and seeds 3–5 (the independent second ensemble). |
| `segment_lined_up_cldice_ensemble/` | The three centre-line trainings averaged. |
| [`final_test_ensemble/`](final_test_ensemble/) | The student's choice (ensemble of the named recipe) reported once on the test rows: no gain. |
| [`label_audit/`](label_audit/) | 40 random lined-up hand masks judged by eye: 2 of 75 judged leaders off (3%). |

## 3D

| Folder | What it holds |
|---|---|
| [`depth3d/`](depth3d/) | How the 8-bit depth maps to distance, calibrated on the ground (session 13). |
| [`tree3d/`](tree3d/) | Each tree's plane, the V in that plane, its lean and sizes; summer against winter (session 13). |
| [`bend3d/`](bend3d/) | The trunk and leaders as smooth curves that may bend out of the plane (session 15). |
| [`bend3d_preview/`](bend3d_preview/) | A first look on the 21 trees in the repository. **Not a result:** 9 of its trees are from the edge row. |
| [`bend3d_typical/`](bend3d_typical/) | The typical winter shape filling in what summer cannot see (session 15). |
| [`model3d/`](model3d/) | A summer photo as a 3D model to open in a browser, with two example pages (sessions 13–17). |

## Reading a training run's folder

The names are built from parts:

| Part | Meaning |
|---|---|
| `segment_lined_up` | The U-Net trained on the hand masks lined up by hand, 256x256, out of fold on 5 row-grouped folds. |
| `_joined` | The same run's masks with pieces within 19.3 mask px joined (`segment --join`). Most runs have this twin, explained by the run's README; the reported scores are these. |
| `_rerun`, `_seedN` | Another training of the same recipe: the re-run is seed 0 (session 12), seeds 1–5 are session 16's. |
| `_cldice` | Trained with the centre-line loss added. |
| `_ensemble` | The probabilities of three trainings averaged. |
| `_480x640` | The whole photo at full size. |
| `_grown`, `_continued`, `_completed` | Session 12's attempts to complete short predicted leaders. |

The files in a run's folder:

| File | What it holds |
|---|---|
| `report.json` | Every number of the run: the recipe, epochs, run time and the scores. |
| `predictions.csv` | Per tree and measure: winter's value, the mean shape's, the hand mask's and the model's. |
| `mask_scores.csv` | Per tree: the predicted mask against the hand mask (IoU, Dice, precision, recall). |
| `predicted_masks.npz` | The predicted 256x256 masks, bit-packed (`treeskel.segment.load_masks`). |
| `probabilities.npz` | The out-of-fold probabilities that ensembles average (session 16's runs). |
| `training_log.csv` | The training and held-out loss per fold and epoch, where kept. |

Each run's figures (training curves, skill, examples) are in [`../figures/runs/`](../figures/runs/).
