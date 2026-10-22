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
- **Bends out of the plane** (session 15, the student's request and plan;
  `treeskel.bend3d`): each branch is a smooth curve along its axis depth with
  the plane as a Gaussian-process prior, τ 2 cm and L 20 cm (chosen by block
  cross-validation on all 449 development trees' winter depth; every fold
  agreed); summer counts only depth-ridge readings. **`model3d build` bends by
  default** (the student's decision after the rule fixed before the 449-tree
  run passed; `--flat` gives the plane); unseen sections are drawn pale. See
  `docs/results/bend3d/README.md`. **The prior's mean is the typical winter
  shape** (`bend3d --typical`; its rule, fixed before its 449-tree run,
  passed: `model3d` reads the verdict; `--plane-prior` gives the plane). See
  `docs/results/bend3d_typical/README.md`.

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
               benchmark.py (run time and memory of every method; `--flops`, session 18: FLOPs, memory traffic,
               roofline, torch.profiler),
               depth3d.py (the depth map's zero from the ground; the camera),
               tree3d.py (the tree's plane, V in the plane, lean, sizes; summer against winter),
               bend3d.py (trunk and leaders as smooth 3D curves that may leave the plane: robust
               Gaussian-process fit joined at the fork, block cross-validation, summer ridge readings),
               model3d.py (a summer photo as a 3D model: HTML viewer and .ply; `build R12N02` in one step
               (session 17): the tree found in data/raw|sample|tuning, its saved out-of-sample mask (5-fold for
               development rows, the final test's for test rows; no network), opened in the browser; photos from
               elsewhere and row 4 need `train`, which saves the named U-Net to the git-ignored models/; bends by
               default, `--flat` for the plane; the typical winter shape as the prior if adopted, `--plane-prior`;
               bend exaggeration slider),
               seg_compare.py (session 16: the centre-line loss and three-training ensembles against the
               named U-Net on the development rows; ensembles from saved runs, the masks' structure, the
               rule), label_audit.py (a fixed random sample of lined-up hand masks, judged by eye
               in a key-press window, `judge`)
tests/         pytest (run: pytest); synthetic V-tree fixtures in test_winter_gt.py
data/sample/   9 committed trees incl. R04N01 (row-4 copy case)
data/matches/  winter-summer matches per matcher (computed on the student's PC)
data/shape/    shape measures and summer ResNet-18 features per development tree
               (computed on the student's PC by treeskel.shape; the models run anywhere)
annotations/   winter_clicks.json (ground truth), season_overlay.json (alignment),
               season_pairs.json (11 trees of clicked pairs, used as a check)
data/raw/      full dataset, git-ignored, only on the student's PC
docs/progress/ session summaries (the project's memory)
docs/figures/, docs/results/  evidence for the report (docs/results/README.md indexes every folder;
               docs/figures/runs/ holds the figures each training run writes, referred to by no README)
docs/html/     HTML documentation (pdoc -d google -o docs/html src/treeskel), committed
docs/test_report.html  pytest-html test report, committed
model3d.bat    Windows: drop photos on it, or double-click and type a tree id (runs model3d build; CRLF,
               kept by .gitattributes)
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
  reST's `Usage::` shows literally): every public function lists each
  argument in `Args:` (one per entry, in order), and has `Returns:` if it
  returns a value and `Raises:` if it raises (`tests/test_docstrings.py`
  checks this against the signatures); inline
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

See the latest file in `docs/progress/`. As of session 19 (the last):

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
  the test rows with that caveat. Session 16 did this once (the ensemble, no
  gain); four trainings of the named recipe score +0.81 to +0.88 there.
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
     makes a second sample of the recipe (GPU), not the final test's network.
     Since session 17 `build <tree id>` picks the out-of-sample masks by itself
     (5-fold for development trees, the final test's for test rows), so figures
     match the reported results; `--mask unet` on a development tree is
     in-sample (`build` warns). A short predicted leader can tip the 3D plane
     (R01N04, R03N10). See `docs/results/model3d/README.md`.
  2. The report: session 18's to-do list (`docs/progress/2026-10-18-session-18.md`, Next steps): the
     profiled kernel sentence, the mechanical-engineering link, the AI reflection's effect on workflow, the
     test count (610 since session 19), reference [7]'s commit after the last merge, and the 10-page limit.
     The student set the report aside in session 19 ("ignore the report").
  3. **Before submission: the repository requirements** (the brief, as the student quoted it in session 17:
     ReadMe; HTML documentation; licence; source code, functional `main`, with inline comments and docstrings;
     test functions and a test report; an accessible link; a zipped clone on Moodle). All met as of
     session 18; session 19 checked them against the marking guide's top band and closed the gaps (below). **The repository was private in session 17**: the link is accessible only once it is public
     or the markers are given access (the student's choice; making it public also publishes the 26 trees of
     the teaching team's data in `data/sample` and `data/tuning`, 135 images with the history, and the
     student's university email on their commits). The zip should be of a fresh `git clone` (no data/raw, .venv, models or
     outputs). If any code changes after that, rebuild the last two before
     submitting: `pdoc -d google -o docs/html src/treeskel` and
     `pytest --html=docs/test_report.html --self-contained-html` (on the
     student's PC: `pip install pytest-html` first if missing).

     | Requirement | Status (session 19) |
     |---|---|
     | ReadMe | results at a glance (session 16's mask experiments added), documentation links (HTML docs, test report, 3D viewer examples), 9-tree sample, test rows named and both uses of them stated; the AI-use section in the student's first person (session 14; updated in session 16; 19 sessions in session 19); session 19 added the engineering problem, the environment, the order to reproduce the results, data ownership, inputs and outputs (formats, units, invalid inputs) and the limitations |
     | HTML documentation | `docs/html/` (pdoc, Google format), 27 modules (`seg_compare`, `label_audit` added in session 16); the package page lists them in the order of the work; rebuilt after session 19's docstrings (identical to a fresh build) |
     | Licence | `LICENSE`, GNU GPL v3 |
     | Source code, functional `main` | 610 passed, 0 skipped (session 19, the assistant's environment); 504 passed, 0 skipped on the student's PC in session 15 |
     | Docstrings and inline comments | 546 of 546 public functions, classes and methods, and every nested helper (session 18: session 17's count plus 9 public and 7 nested; session 17 reported 73 of 73 nested by a narrower count, 86 of 86 counting every function and class defined inside a function); comments on intent in the longest functions. Session 19: every public function and method (526) lists its arguments in `Args:`, and has `Returns:`/`Raises:` where it returns or raises; dataclasses' `Attributes:` match their fields; checked by `tests/test_docstrings.py` |
     | Test functions | 32 test files, 511 test functions in `tests/` (session 19) |
     | Test report | `docs/test_report.html` (pytest-html), made in the assistant's environment (PyTorch and kornia from PyPI) after session 19's docstrings: 610 passed, 0 skipped |
  4. Optional: the 3D step once on the test rows (with the caveat).
- **Session 15: bends out of the plane** (PR #56 and the next; `docs/results/bend3d/`, all 449
  development trees, the student's PC, 57 min):
  - *Winter holds bends:* left-out depth 1.92 cm against the plane's 3.13 cm (median per tree, out of fold),
    better on 94% of trees (89% with 30 px buffers).
  - *The trunk* stands a median 14.2° out of the V's plane, root behind (13.3° with ridge readings only;
    row 1, the edge row, 26°). Leaders bow about 1.7 cm.
  - *The 3D V:* bending the leaders widens it by 1.6°, so it is 0.9° (not 2.6°) narrower than the 2D V; the
    depth cannot say which is right, and either way the 2D results stand.
  - *Summer:* sees the trunk (45% of readings), almost never the leaders (3%); its trunk angle is closer to
    winter's than the plane's on 60% of trees (U-Net) but only about a third of the size (slope 0.32).
  - The 21-tree preview overstated the sizes (trunk 25°, V +2.2°): 9 of its trees are row 1.
  - Code changed: rebuild `docs/test_report.html` before submission (the docs are rebuilt).
  - *Follow-up (the student saw nearly straight summer leaders):* the typical winter shape as the bends'
    prior mean (`bend3d --typical`, `docs/results/bend3d_typical/`; rule committed before its 449-tree run:
    U-Net, out of fold, nearer winter than the plane-mean bends on the trunk angle and the leader tilts,
    p < 0.05); `model3d` uses it only if the rule passed (`--plane-prior` keeps the plane). The viewer's
    "Exaggerate bends" slider (×1–×5) is display only.
  - *The typical-shape run (449 trees):* the rule passed (U-Net: trunk 8.5° against 12.0°, nearer on 78%;
    tilts 3.0° against 4.5°, 76%; the V opening too). The typical leader is an S (2–3 cm towards the camera
    just above the fork, 1–1.5 cm behind around 0.8 m, back near the top). The shape alone is nearer still on
    the angles (7.0°, 2.7°) but the same for every tree; summer's readings follow the individual tree (rank
    0.38–0.49) but pull the angles low. Reported, not used.
- **Session 16: improving the mask model** (the student's plan; `docs/results/segment_variants/rule.md`):
  - *Two changes against the named U-Net, development rows only:* the soft-clDice centre-line loss
    (`segment --cldice`: BCE + 0.5 Dice + 0.5 clDice, 25 soft-skeleton erosions) and an ensemble of three
    trainings (`seg_compare ensemble`: out-of-fold probabilities averaged, threshold 0.5, joined).
  - *Three samples of each recipe:* session 12's re-run is seed 0 of the named recipe (the unchanged code
    trains bit for bit as `main`), plus seeds 1, 2; the centre-line loss with seeds 0, 1, 2. Five new GPU
    runs. Each tree's error is averaged over a recipe's samples, because one run against the named run mixes
    the change with run-to-run noise.
  - *The rule* (`seg_compare.decide`): V opening MAE lower, Wilcoxon p < 0.05, and no significant increase in
    side branches or incomplete masks. Stubs (fork, no V opening) count as the mean shape here (the named
    run: 4.22° instead of the report's 4.10°).
  - *Label audit:* 40 trees drawn (seed 16, row-stratified) and committed before anyone looked
    (`docs/results/label_audit/sample.csv`); the student's verdicts (on, off, hidden per part).
  - *Status:* the rule was confirmed by the student before any run (11 October). An adopted change is reported
    once on the test rows, as a secondary line with the caveat; that code is written after the verdict. The
    second ensemble (seeds 3–5) is run, deciding nothing. Eight GPU runs in all. The HTML docs and the test
    report were rebuilt after the code changes (546 passed).
  - *Results* (`docs/results/segment_variants/README.md`, `checks_after_runs.py`): the rule passed both changes
    and names the centre-line ensemble (4.004° against the named run's 4.224°). **But its test is too lenient
    between trainings:** seven trainings of the named recipe range 3.997–4.471°, and the rule's test calls
    triples of the same recipe different in 3 of 10 splits (its own baseline triple p = 0.027). At the level
    of trainings, the centre-line loss's V-opening gain is not established (p 0.11); its higher coverage of
    the hand masks' centre lines is (0.866 → 0.883, p 0.008). Ensembling beats its own members on all three
    ensembles (p ≤ 0.006, 0.06–0.14°) and halves the disagreement between trainings (0.67° → 0.36°).
    **The student chose B** (12 October): an ensemble of the named recipe, the centre-line loss not used.
    The test-row secondary report is specified in `docs/results/final_test_ensemble/` (`model.json`: the
    final test's recipe and 34 epochs, seeds 0–2, probabilities averaged; `final_test --out
    docs/results/final_test_ensemble`, once). **Run once (12 October): no gain.** On the same 92 trees the
    ensemble scored +0.81 (MAE 3.78°), against four single trainings at +0.81 to +0.88 (3.37–3.88°; the final
    test's network was the best). It was no closer than its own networks (p 0.65): on R03N10 and R16N12, two
    of three networks failed and the average followed. The run's printed member comparison had left out
    stubs; corrected (`like_for_like.py`, code fixed, test added). The final test stays the headline; report
    that a network of this recipe scores about +0.81 to +0.88 there.
  - *Label audit:* the key-press window (`label_audit judge`). A first pass marked all 120 parts on, but the
    student was unsure what was asked: not used, kept in `sample_first_pass.csv`. **Second pass: 2 of 75
    judged leaders off (3% [1%, 9%]), every trunk on, 5 leaders hidden** (R02N27, R05N29 left leaders off):
    under the 10% fixed beforehand, so label position is a minor, measured limitation. The second pass revised
    the first sheet (every part started as on), noted as a possible anchoring.
- **Session 17: the 3D model in one step** (the student's request): `python -m treeskel.model3d build R12N02`
  (or `model3d.bat`, drop a photo or type an id) builds any of the 541 trees from its saved out-of-sample mask,
  with no network, GPU or PyTorch, and opens it; the old explicit forms give byte-identical files on the 8
  sample trees with saved masks. Photos from elsewhere (depth found beside them by name) and row 4 use the
  trained network; a new tree needs its summer photo and summer depth, never the winter photo. `model3d.bat`
  could not be run in the assistant's environment (no Windows); the student ran it ("it works"). Then the
  tidy-up (PR #66): `docs/results/README.md` indexes all 57 results folders; the 64 run figures no README
  refers to moved to `docs/figures/runs/`; two draft figures deleted; no evidence deleted. Open before
  submission: the repository's visibility, and Moodle's upload limit against the ~259 MB zip of a fresh clone.
- **Session 18: the report checked against the brief** (the student's request; the report is a .docx
  outside the repository). The repository was public by then. The report's FLOP figures (Section 7.0,
  Appendix A: 10.8 GFLOP, 313 MB, 34 FLOP/byte, 20% of the GPU's peak, 1.9 / 1.3 ms at the peak rates,
  190 / 220 GFLOP/s on the CPU) had not been saved anywhere. `python -m treeskel.benchmark --flops` now
  computes and saves them (`docs/results/benchmark/flops.json`, README section). The FLOP counter, a
  no-fusion traffic model and the GTX 1080 Ti's roofline reproduce every figure, and
  `tests/test_benchmark.py` checks them. It also adds `torch.profiler`. The student ran it on the PC
  (18 October, pushed to `main`; data files only, so the HTML docs and the test report still match). The
  named U-Net launches 271 kernels per photo, which run for 6.1 ms of a 10.3 ms prediction (GPU busy 60%,
  idle about 15 µs per kernel); its kernels compute at 31% of the peak. At 480x640 the GPU is busy 97% and
  its kernels reach 50%. So the report's "more than 200 kernel launches" was the operator count (215 per
  photo), and its "fixed overheads" explanation is half the story: idle time between launches, plus kernels
  too small to fill the GPU at batch 1. Other gaps found: the link to mechanical engineering is not stated; the AI
  reflection does not say directly how AI changed the student's understanding or workflow; the body ends
  exactly at the foot of page 10.
- **Session 19: the repository against the marking guide** (the student's request; the last session).
  The brief's checklist was met; the gaps were against the top band. The README gained the engineering
  problem, the environment, an ordered path to reproduce the results, data ownership (the 26 committed
  trees stay the teaching team's; the GPL covers the code and documentation), inputs and outputs, and the
  limitations; `data/tuning/README.md` is new; `data/sample/README.md` no longer calls the depth mapping a
  hypothesis. 385 public functions gained `Args`/`Returns`/`Raises` sections and 29 grouped or stale `Args`
  sections were split or completed (no code changed); `tests/test_docstrings.py` enforces it. HTML docs
  and test report rebuilt (610 passed). After PR #70, the student's four observations: the README's AI
  section gained why AI was used and how it changed the workflow and understanding (drafted from the notes,
  for the student to check); `model3d`'s introduction now says unseen sections follow the typical winter
  shape; the limitations open with what the validation supports (2D angles; 3D is an estimate) and the
  test intervals' within-row caveat; reproduction states its prerequisites (no weights committed).
  See `docs/progress/2026-10-22-session-19.md`.
- **Winter-to-summer pixel alignment was abandoned** in session 7
  (`docs/results/align/`).
- Still open: the crotch; the colour fit's window dependence; which season
  was photographed first; the summer depth registration; a depth bias with
  image height (five posts suggest none; a 449-tree post check was offered
  in session 15 and set aside); why summer's readings pull the bend angles low.
