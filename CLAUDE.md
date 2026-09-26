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
`GT` files are hand-drawn 256x256 summer masks, used **for validation only**.
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
- **Two leaders per tree**, matching the hand masks. "Left" is the leader
  whose tip is further left in the image.
- **Masks:** the trunk width comes from depth (robust straight taper). Each
  leader's width is measured across the branch in colour (`branch_edges`,
  session 8: background | bark | background by least squares, so the shaded
  half of a side-lit branch counts; cross-sections within reach of the other
  leader are not used). Each leader is drawn at its own robust width, widened
  near the fork by a shared +35% that fades over 4 trunk radii
  (`fit_leader_radius`); these two values came from 21 trees and are
  provisional, as the 449-tree check preferred a much longer fade (see
  `docs/results/leader_width/`). Centre lines are smoothed between clicks (σ = 6).
  At the fork, the notch on each leader's outer side is filled out to the
  straight line from the trunk's edge to the leader's edge 1.4 trunk radii up
  it (`shoulder_fill`, session 6; with the session-8 widths it may no longer
  be needed, see `docs/results/leader_width/`); the crotch between the
  leaders is not filled.

## Where things are

```
src/treeskel/  io.py (data), splits.py, winter_gt.py (baseline), guided_gt.py
               (tracing, widths, fork fill), annotate.py (click tool),
               tune_cost.py, measure_fork.py, measure_width.py (leader widths),
               align.py (transforms, matchers),
               annotate_pairs.py (pair click tool, tried first), annotate_overlay.py
               (line-up tool, chosen), summer_gt.py (summer masks and checks)
tests/         pytest (run: pytest); synthetic V-tree fixtures in test_winter_gt.py
data/sample/   9 committed trees incl. R04N01 (row-4 copy case)
data/matches/  winter-summer matches per matcher (computed on the student's PC)
annotations/   winter_clicks.json (ground truth), season_overlay.json (alignment),
               season_pairs.json (11 trees of clicked pairs, used as a check)
data/raw/      full dataset, git-ignored, only on the student's PC
docs/progress/ session summaries (the project's memory)
docs/figures/, docs/results/  evidence for the report
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
- **Code:** Google-style docstrings (the HTML docs are built by pdoc), inline
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

See the latest file in `docs/progress/`. As of session 7:

- **Winter-to-summer pixel alignment was tried and abandoned** (session 7,
  `docs/results/align/`). Automatic matching (SIFT; DISK+LightGlue gave a
  plausible fit on about 1% of trees), clicked point pairs (about 12 px off,
  2 min a tree) and lining the photos up by hand all failed: too little rigid
  structure is visible in both seasons, and the scale is not pinned down. The
  tools (`align.py`, `annotate_pairs.py`, `annotate_overlay.py`,
  `summer_gt.py`) and their results stay as documented evidence.
- **The student's direction:**
  1. **Improve the winter masks** (next chat). The student saw masks off the
     branch and of the wrong width; get example tree ids first, then
     measure, fix, and measure again. The constant-width trunks (session 6,
     section 6) may be part of it.
  2. **Learn the tree's shape from winter, alignment-free:** shape measures
     from the winter traces that need no alignment (leader angles relative to
     the trunk, V opening, curvature), predicted from summer RGB-D with the 5
     folds. Baselines: the mean shape, and the hand masks' shape.
  3. Optional later: train on the hand masks (roughly in the summer view)
     with the winter traces as shape truth. This would change "hand masks are
     for validation only", so ask first.
- Still open: score the automatic winter baseline against the guided ground
  truth; which season was photographed first.
