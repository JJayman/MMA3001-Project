# CLAUDE.md: project guide for AI assistants

MMA3001 (Numerical Methods and Machine Learning) individual project, 25% of
the unit. Student: Jonah Adams (GitHub JJayman). The student drives the
decisions; the assistant explains, proposes, implements and verifies.
Assessment rewards validation, justified decisions, documentation and an
honest AI-use reflection (see the brief summary in `docs/progress/2026-09-23-session-1.md`).

## Start of every session

1. Read `README.md`, then **all** of `docs/progress/` in date order. Each
   file is one work session: what was done, the numbers, the decisions, the
   AI-use notes and the next steps.
2. Check that `main` is up to date and the tests pass (`pytest`).
3. Ask the student what today's goal is, if they have not said.

## The project in one paragraph

Predict the **main branches** (trunk + two leaders of the 2D open-V / Tatura
trellis) of apple trees from **summer** RGB-D images, where leaves hide them.
Ground truth comes from **winter** images of the same trees (bare branches),
traced with a click-guided tool. Data: "Dataset (With Summer GT)", Intel
RealSense D435, 640x480, 8-bit relative depth (brighter = nearer, 0 = no
reading), files `R<row>N<tree>_<Summer|Winter>_<RGB|D|GT>-H-M-S.png`. The
`GT` files are hand-drawn 256x256 summer masks: since session 11 also training
labels for the segmentation (development rows only; see the fixed decisions).
The related paper is Chen et al. (2020), arXiv:2010.06879: same orchard and
camera, different (autumn) capture.

## Fixed decisions: do not change without asking the student

- **Data:** 541 trees in 19 rows. Row 4 is excluded (its winter depth images
  are copies of R03N34's); see `EXCLUDED_ROWS` in `src/treeskel/io.py`.
- **Split:** `splits.json` holds the test rows **3, 12, 16** and 5 row-grouped
  folds. Always load it (`--load` / `load_split`); never regenerate it.
  Nothing may be tuned on test rows. The 17 tuning trees are in
  `data/tuning/` (development rows).
- **Ground truth:** click-guided (`guided_gt`), with clicks in
  `annotations/winter_clicks.json`, all 541 trees done. The automatic
  `winter_gt` is kept as a **baseline** only.
- **Depth registration** (session 9, the student's decision): the winter depth
  is moved **5 rows up and 1 column right** on loading
  (`treeskel.io.DEPTH_SHIFT`, applied by `load_tree`; `register=False` gives
  the depth as stored). Summer is not moved. Measured on all 449 development
  trees with `measure_registration`; the colour-edge fit follows its window,
  which made session 8's estimate (3.1–3.8) low. See
  `docs/results/depth_registration/`.
- **Hand masks as training labels** (session 11, the student's decision,
  replacing "validation only" from session 2): the development rows' hand
  masks train the segmentation (`treeskel.segment`); the winter traces stay
  the independent truth, never training or choosing anything there, and the
  predicted masks' shape is scored against them. The test rows' masks stay
  unused until the final evaluation. Caveat: several masks sit 20–60 px off
  the visible summer trunk (R20N06 about 90), varying by tree, so pixel
  scores against them are not the headline; the shape scores are. After the
  first run the student chose to line the masks up by hand
  (`treeskel.annotate_mask`, shifts in `annotations/mask_overlay.json`;
  `segment --mask-overlay` trains only on lined-up trees).
- **Two leaders per tree**, matching the hand masks. "Left" is the leader
  whose tip is further left in the image.
- **Masks:** the trunk width comes from depth (robust straight taper). Each
  leader's width is measured across the branch in colour (`branch_edges`,
  session 8: background | bark | background by least squares, so the shaded
  half of a side-lit branch counts; cross-sections within reach of the other
  leader are not used). Each leader is drawn at its own robust width, widened
  near the fork by a shared +35% that fades over 12 trunk radii
  (`fit_leader_radius`), so it narrows gradually along its length; chosen on
  all 449 development trees by leave-one-tree-out and the row-grouped folds
  (a first fit on 21 trees had chosen 4; see `docs/results/leader_width/`).
  Centre lines are smoothed between clicks (σ = 6).
  **No shoulder fill at the fork** (`shoulder_length = 0`, the student's
  decision in session 8, "for the moment"): the session-6 fill (1.4) closed
  the notch left by leaders drawn too thin; at their true width the masks
  without it already match depth on all 449 development trees, and 1.4
  overfilled. `shoulder_length = 1.4` restores it.
  **Fork junction instead of the trunk's round top** (session 8, the
  student's request: the fork "just looks like a circle"): the trunk is cut
  flat at the fork, and on each leader's outer side a straight edge joins the
  trunk's edge to the leader's edge 1 trunk radius up the leader
  (`junction_reach = 1.0`, matched to depth's shoulder coverage on 21 trees
  and confirmed on all 449; depth is neutral about the change as a whole, so
  it rests on the shape; see `docs/results/fork_junction/`). The crotch
  between the leaders is what the leaders' discs draw, a V, not filled. The
  root is cut flat too where its round end would lie wholly in the image
  (`flat_root`). Reach 0 and `flat_root=False` restore the round ends.

- **3D calibration** (session 13, proposed by the assistant; confirmed by the
  student in session 14): the 8-bit depth is linear in
  distance, `Z = scale × (zero − v)`, with **zero 300** (median of per-tree
  fits to the ground on all 449 development trees; quartiles 262–366, the
  sensitivity range), focal length **616 px** (D435 colour, ±5%), and the
  scale putting the median fork at **2.0 m** (Chen et al.'s 1.8–2.2 m):
  0.01197 m per unit (`treeskel.depth3d`). Angles need only the zero and focal
  length; sizes rest on the 2.0 m. See `docs/results/depth3d/`, `tree3d/`.

## Where things are

```
src/treeskel/  io.py (data), splits.py, winter_gt.py (baseline), guided_gt.py
               (tracing, widths, fork junction), annotate.py (click tool),
               tune_cost.py, measure_fork.py (fork fill and junction),
               measure_width.py (leader widths), measure_registration.py (depth-colour shift),
               align.py (transforms, matchers),
               annotate_pairs.py (pair click tool, tried first), annotate_overlay.py
               (line-up tool, chosen), annotate_mask.py (hand masks onto their summer
               photos, session 11), summer_gt.py (summer masks and checks),
               shape.py (shape measures without alignment; summer CNN features),
               shape_model.py (predicting the winter shape from summer, 5 folds),
               shape_cnn.py (fine-tuning a CNN on the winter shape; the student's GPU),
               segment.py (U-Net on the hand masks, scored against winter; the student's GPU;
               joining broken masks, session 12), final_test.py (the one test-row evaluation),
               score_baseline.py (the automatic winter baseline against the guided ground truth),
               benchmark.py (run time and memory of every method),
               depth3d.py (the depth map's zero from the ground; the camera),
               tree3d.py (the tree's plane, V in the plane, lean, sizes; summer against winter),
               model3d.py (a summer photo as a 3D model: HTML viewer and .ply; `train` saves the
               named U-Net to the git-ignored models/)
tests/         pytest (run: pytest); synthetic V-tree fixtures in test_winter_gt.py
data/sample/   9 committed trees incl. R04N01 (row-4 copy case)
data/matches/  winter-summer matches per matcher (computed on the student's PC)
data/shape/    shape measures and summer ResNet-18 features per development tree
               (computed on the student's PC by treeskel.shape; the models run anywhere)
annotations/   winter_clicks.json (ground truth), season_overlay.json (alignment),
               season_pairs.json (11 trees of clicked pairs, used as a check)
data/raw/      full dataset, git-ignored, only on the student's PC
docs/progress/ session summaries (the project's memory)
docs/figures/, docs/results/  evidence for the report
docs/html/     HTML documentation (pdoc -d google -o docs/html src/treeskel), committed
docs/test_report.html  pytest-html test report, committed
```

The assistant's environment has only `data/sample` and `data/tuning`. Runs on
the full dataset happen on the student's Windows PC (the `.venv` there, a GPU
is available). Give exact commands and ask for the output. Loading downloaded
pretrained network weights (e.g. the matchers in `align.py`) is blocked in the
assistant's environment: those models run on the student's PC, and their
outputs (e.g. `data/matches/*.npz`) are committed for analysis.

## Conventions

- **Git:** work on a branch, push, open a PR; the student merges. Never push
  to `main`. Commit messages explain why, not just what.
- **Code:** Google-style docstrings (the HTML docs are built by pdoc with
  `-d google`; command examples in docstrings go in a fenced block, since
  reST's `Usage::` shows literally), inline
  comments on intent, and a test for every behaviour. Synthetic data with a
  known answer is preferred; use the sample for integration tests.
- **Validation habits that caught real errors:**
  - measure before and after, and save figures and numbers to `docs/`;
  - look at actual images, not just summary numbers;
  - never build a reference with the method being tested;
  - use leave-one-out or grouped CV when choosing settings.
- **Windows:** output files can be locked by viewers, so CLIs report failed
  writes and continue. Long runs print progress.
- **End of session** (the student says "wrap up"): write
  `docs/progress/<date>-session-<n>.md` in the same format as the others
  (work, numbers, decisions, AI use with errors, next steps). Commit it and
  open a PR.

## Current status and next steps

See the latest file in `docs/progress/`. As of session 14:

- **Winter masks and depth registration** (sessions 8–9): settled, see the
  fixed decisions above.
- **The tree's shape, alignment-free** (session 10, `docs/results/shape/`):
  angles from the winter traces (`treeskel.shape`: V opening, each leader's
  angle from the trunk and from the vertical, the V's lean, bends), fitted
  over a fixed stretch of 25–264 px up each leader (×1.12 in the summer view).
- **The model (sessions 11–12):** a U-Net (ResNet-18 encoder) trained on the
  development rows' hand masks, lined up by the student
  (`annotations/mask_overlay.json`), at 256x256; its masks' pieces within
  19.3 mask px (about a trunk's width) are joined before measuring
  (`join_pieces`). Development (5 folds): V opening +0.75 (hand masks +0.87).
  Tried and not adopted, each by a rule committed before its result
  (`docs/results/final_test/naming_rule.md`): the whole photo at 480x640;
  growing masks; continuing short leaders and pruning; shape completion.
- **The final test is done** (session 12, `docs/results/final_test/`): V
  opening **+0.88** [+0.81, +0.92] on test rows 3, 12, 16 (hand masks +0.91).
  **The test rows are now used:** nothing more may be chosen on them; a new
  method is developed on the development rows and, at most, reported once on
  the test rows with that caveat.
- **Known limit:** on 7% of the development trees scored on the V opening
  (31 of 444) the U-Net's mask has a leader that stops short under the leaves
  ("short leaders" always means predicted leaders: no winter trace or hand
  mask has a leader shorter than the fitted stretch). They carry a third of
  the V opening's squared error, and no completion helped. (Session 12 said
  33 trees and 47%: it also counted the 2 masks with no fork, in which a
  leader is cut off from the trunk; corrected in session 14.)
- **Session 13 (all on the 449 development trees, PR #50):**
  - *Automatic winter baseline* (`score_baseline`, `docs/results/winter_baseline/`):
    traces 43% of trees (F1 ≥ 0.9); its quality check passes 147 untraced
    trees; more than 20° off on the V opening on 14% of trees, so its skill is
    −1.97 against the U-Net's +0.75. The student's session-4 verdicts agree.
  - *Run time and memory* (`benchmark`, `docs/results/benchmark/`): the named
    U-Net predicts a photo in 9.7 ms (GPU) / 116 ms (CPU) and trains in
    2.0 min, about the cost of the much weaker angle CNN.
  - *3D* (`depth3d`, `tree3d`): trees flat to a median 1.7 cm; the 2D V is
    2.3° wider than the V in the tree's plane (97% of trees, 1.5–3.4° over the
    calibration range; the 9° lean towards the camera explains it), so the 2D
    results stand; summer agrees with winter better in the plane than in the
    photo (61% of trees); the lean from the vertical cannot be determined.
    Summaries are medians and rank correlations (6 failed planes swamped
    Pearson's r at first).
  - *A summer photo as a 3D model* (`model3d`, `docs/results/model3d/`): HTML
    viewer (no internet) and `.ply`; `train` must be run once on the
    student's PC before `build` can use the network on a new photo.
- **Session 14 (PRs #51–#54):** the repository requirements met (see the
  table below); session 12's short-leader figures corrected (31 of 444, 33%;
  they had counted the 2 no-fork masks); `model3d` run on the student's PC:
  a short predicted leader can tip the 3D plane, and `train` is a second
  sample of the recipe (complete masks agree within about a degree between
  trainings, short-leader ones can differ by 10° or more), so the final
  test's numbers are the ones to report. Before pushing to an existing PR,
  check that it is not merged.
- **Next (the report is due about 19 October 2026):**
  1. ~~Train and build on the student's PC~~ (done, session 14). `train`
     makes a second sample of the recipe (GPU), not the final test's network:
     use `--mask saved` for development trees (out of fold) and
     `--saved docs/results/final_test/predicted_masks.npz` for figures that
     match the final test; `--mask unet` on a development tree is in-sample
     (`build` warns). A short predicted leader can tip the 3D plane (R01N04,
     R03N10). See `docs/results/model3d/README.md`.
  2. The report.
  3. **Before submission: the repository requirements.** All met as of
     session 14 (PR for branch `claude/compassionate-carson-ospof6`). If any
     code changes after that, rebuild the last two before submitting:
     `pdoc -d google -o docs/html src/treeskel` and
     `pytest --html=docs/test_report.html --self-contained-html`.

     | Requirement | Status (session 14) |
     |---|---|
     | ReadMe | refreshed: results at a glance, documentation links (HTML docs, test report, 3D viewer example), 9-tree sample, test rows named; the AI-use section rewritten in the student's first person (session 14) |
     | HTML documentation | `docs/html/` (pdoc, Google format), 24 modules; the package page lists them in the order of the work |
     | Licence | `LICENSE`, GNU GPL v3 |
     | Source code, functional `main` | 468 passed, 1 skipped (the LightGlue test needs kornia) |
     | Docstrings and inline comments | 440 of 440 public functions, classes and methods, and every nested helper; comments on intent added to the longest functions (cross-validation, the joined, grown and continued runs, the CLIs) |
     | Test functions | 27 test files, 409 test functions in `tests/` |
     | Test report | `docs/test_report.html` (pytest-html), made in the assistant's environment with CPU PyTorch; can be remade on the student's PC with the same command |
  4. Optional: the 3D step once on the test rows (with the caveat).
- **Winter-to-summer pixel alignment was abandoned** in session 7
  (`docs/results/align/`).
- Still open: the crotch; the colour fit's window dependence; which season
  was photographed first; the summer depth registration.
