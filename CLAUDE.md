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
- **Masks:** the trunk width comes from depth; the leader widths come from
  colour (scale-space); both use a robust taper fit; centre lines are
  smoothed between clicks (σ = 6).

## Where things are

```
src/treeskel/  io.py (data), splits.py, winter_gt.py (baseline), guided_gt.py
               (tracing, widths), annotate.py (click tool), tune_cost.py
tests/         pytest (run: pytest); synthetic V-tree fixtures in test_winter_gt.py
data/sample/   9 committed trees incl. R04N01 (row-4 copy case)
data/raw/      full dataset, git-ignored, only on the student's PC
docs/progress/ session summaries (the project's memory)
docs/figures/, docs/results/  evidence for the report
```

The assistant's environment has only `data/sample` and `data/tuning`. Runs on
the full dataset happen on the student's Windows PC (the `.venv` there, a GPU
is available). Give exact commands and ask for the output.

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

See the latest file in `docs/progress/`. As of session 5:
1. Align each winter image with its summer image, and warp the winter masks
   into the summer view; measure the error on hand-clicked point pairs.
2. The same alignment is needed for the hand masks (15–45 px shift).
3. Summer model: baselines, then a U-Net on RGB-D with the 5 folds.
4. Score the automatic baseline against the guided ground truth.
