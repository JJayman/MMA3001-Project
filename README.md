# Seeing Branches Through Leaves: Apple-Tree Skeleton Estimation from RGB-D

MMA3001 Numerical Methods and Machine Learning, individual project.

**Goal:** estimate the main branch structure (trunk and scaffold limbs) of apple
trees from a *summer* RGB-D image, where leaves hide much of the wood. The
*winter* RGB-D image of the same tree, when the branches are bare, is used to
build the ground truth. The intended use is perception for orchard robots
(pruning, harvesting, path planning), which must avoid or target branches they
cannot directly see.

**The engineering problem:** a pruning or harvesting robot is a machine whose arm must plan its motion around
the tree, avoid striking the wood, and place its cutter or gripper relative to the branches. Its motion
planner needs the positions and angles of the trunk and leaders, including the parts the leaves hide; an
error there is a collision or a missed cut. This project estimates that structure from one summer camera
photo, and measures how far the estimate is from the branches photographed bare in winter.

## Results at a glance

The headline measure is the **V opening**, the angle between the two leaders. Skill is how much a method
reduces the squared error of guessing the average tree's shape (1 is perfect, 0 no better than the
average tree, negative worse). Every summer prediction is scored against the same trees' winter photos.

| Step | Result | Details |
|---|---|---|
| Ground truth | 541 winter trees traced through 4 clicks each, plus about one correction (about 15 s a tree); the tuned tracer passes a mean 3.2 px from the correction clicks (4.8 px on trees left out) | `docs/results/tune_cost/` |
| Automatic winter baseline | traces only 43% of trees well (F1 ≥ 0.9 against the guided traces); V opening skill −1.97 | `docs/results/winter_baseline/` |
| Summer models (5 row-grouped folds, 449 development trees) | V opening skill: ridge on CNN features +0.10, fine-tuned angle CNN +0.18, **U-Net on the lined-up hand masks, pieces joined, +0.75**; the hand masks themselves +0.87 | `docs/results/shape/`, `segment_lined_up_joined/` |
| **Final test** (rows 3, 12, 16; 92 trees; run once) | **V opening skill +0.88** [95% interval +0.81, +0.92, trees resampled within the three rows; each row on its own +0.88, +0.86, +0.89], mean error 3.4° against 10.4° for the average tree; the hand masks +0.91 | `docs/results/final_test/` |
| Run time | the U-Net predicts a photo in 9.7 ms on a GPU, 116 ms on a CPU, and trains in 2 minutes | `docs/results/benchmark/` |
| Improving the masks (session 16, development rows) | seven trainings of the same recipe range 4.00–4.47° in V-opening error, which is more than any change moved it. A centre-line (clDice) loss covers more of the labelled branches, but its V-opening gain is not established. Three trainings averaged beat their own members by 0.06–0.14° and halve the disagreement between retrainings (0.67° → 0.36°). Reported once on the test rows: no gain (+0.81, against +0.81 to +0.88 for four single trainings, the final test's being the best), because on two short-leader trees two of the three networks failed. A label audit of 40 random lined-up hand masks found 2 of 75 judged leaders off (3%) | `docs/results/segment_variants/`, `final_test_ensemble/`, `label_audit/` |
| 3D | the trees are flat to a median 1.7 cm (sizes assume 2 m from the camera to the fork); the V in the photo is about 2.3° wider than in the tree's own plane | `docs/results/depth3d/`, `tree3d/` |
| Bends out of the plane | the winter depth holds bends: left-out depth predicted to 1.9 cm against the plane's 3.1 cm (94% of trees); the trunk stands a median 14° out of the V's plane; letting the leaders bend leaves the 3D V 0.9° (not 2.6°) narrower than the 2D V. In summer the typical winter shape fills in what the leaves hide: trunk angle off by 8.5° instead of 12.0°, leader tilts by 3.0° instead of 4.5° | `docs/results/bend3d/`, `bend3d_typical/` |
| 3D model of a summer photo | opens in any browser, the branches bent where summer sees the wood: [`R01N13.html`](docs/results/model3d/R01N13.html), [`R15N05.html`](docs/results/model3d/R15N05.html) | `docs/results/model3d/` |

## Try it: a 3D model of a tree

After the setup below, one command makes any tree's 3D model from its summer photo and opens it in the browser.
The 9-tree sample is enough, and no GPU, network or PyTorch is needed:

```bash
python -m treeskel.model3d build R12N02
```

On Windows, double-click `model3d.bat` and type a tree id, or drag a tree's photo onto it. A development tree
gets its 5-fold mask, from a network that never saw its row. A test-row tree gets the final test's mask, so
its model matches the reported results. A photo from outside the dataset needs the trained network
(`python -m treeskel.model3d train data/raw` once, on a GPU). See [`docs/results/model3d/`](docs/results/model3d/).

## Documentation

- **HTML documentation** of every module, function and class: open [`docs/html/index.html`](docs/html/index.html)
  in a browser (built with pdoc from the docstrings).
- **Test report:** [`docs/test_report.html`](docs/test_report.html) (pytest, every test with its result).
- **Session notes** in [`docs/progress/`](docs/progress/): what was done in each work session, the numbers,
  the decisions, how AI was used (and where it was wrong) and the next steps.
- **Results** in [`docs/results/`](docs/results/), one folder per experiment, with an index of them all in
  [`docs/results/README.md`](docs/results/README.md); figures in [`docs/figures/`](docs/figures/).

## Repository structure

```
data/sample/     9-tree sample of the dataset (full dataset not committed)
data/tuning/     17 development-row trees used to tune the guided tracing
src/treeskel/    Python package
  io.py          filename parsing, dataset indexing, image loading
  splits.py      row-grouped hold-out and k-fold cross-validation splits
  winter_gt.py   automatic main-branch extraction from winter depth (baseline)
  score_baseline.py  scores the automatic baseline against the guided ground truth (paths, quality check, shape)
  guided_gt.py   click-guided main-branch ground truth: cost map + minimal-path tracing
  annotate.py    click tool for guided ground truth (opens a window)
  tune_cost.py   tunes the tracing cost map against corrected annotations
  measure_fork.py  measures the masks at the fork against winter depth (shoulder fill, fork junction)
  measure_width.py measures the leaders' widths against winter depth and the hand masks (sets the width model)
  measure_registration.py  measures how far the depth images sit from the colour images (the depth-colour shift)
  align.py       winter-to-summer alignment: transforms, robust fits, mask warping, automatic matchers
  annotate_pairs.py  click tool for winter-summer point pairs (opens a window)
  annotate_overlay.py  line up the winter photo on the summer photo by hand (opens a window)
  annotate_mask.py     line up each provided hand mask on its own summer photo (opens a window)
  summer_gt.py   winter masks moved into the summer view (from the overlay transforms or pairs)
  shape.py       the tree's shape without alignment (V opening, leader angles, bends) from the winter traces
                 and the hand masks, and summer image features from a pretrained network
  shape_model.py predicts the winter shape from the summer features (ridge regression, row-grouped folds)
  shape_cnn.py   fine-tunes a CNN (ResNet-18) to predict the winter shape from the summer photo (GPU)
  segment.py     trains a U-Net on the hand masks to segment the main branches in summer; its masks
                 are measured like the hand masks and scored against the winter traces (GPU); joins
                 masks broken by a gap of up to a trunk's width before measuring
  final_test.py  the one final evaluation on the test rows: the named U-Net, trained on all development
                 trees, scored against the test trees' winter traces (GPU; runs once)
  benchmark.py   run time and memory of every method (ground truths, mean shape, ridge, angle CNN, U-Net);
                 with --flops, the networks' FLOPs, memory traffic, roofline and profile
  depth3d.py     how the 8-bit depth maps to distance (its zero, from the ground) and the camera
  tree3d.py      the tree in 3D: its plane, the V in that plane, its lean and sizes; summer against winter
  bend3d.py      the trunk and leaders as smooth 3D curves that may leave the plane where the depth sees the
                 wood (winter; summer through the leaves, uncertain sections marked); summer against winter
  model3d.py     a summer RGB-D photo as a 3D model to open and turn in a browser (and .ply files)
  seg_compare.py the centre-line loss and an ensemble of three trainings against the named U-Net (development
                 rows): ensembles from saved runs, the masks' structure (short leaders, side branches), the rule
  label_audit.py a representative sample of the lined-up hand masks, judged by eye for local misalignment
annotations/     clicks and alignments per tree (winter_clicks.json, season_pairs.json, season_overlay.json), committed
data/matches/    winter-summer point matches per matcher (from align.py), committed
data/shape/      shape measures and summer image features per development tree (from shape.py), committed
tests/           pytest test suite
docs/progress/   session-by-session progress notes (incl. AI use)
docs/results/    one folder per experiment: README, report.json, per-tree tables (index: docs/results/README.md)
docs/figures/    figures for the report (runs/: the figures each training run writes)
docs/html/       HTML documentation (pdoc)
docs/test_report.html  test report (pytest-html)
splits.json      fixed test rows and cross-validation folds (do not change)
model3d.bat      Windows: drop a summer photo on it, or double-click and type a tree id, for its 3D model
LICENSE          GNU GPL v3
```

## Setup

**Environment:** Python 3.10 or later. Tested on Windows 11 with Python 3.12.4, PyTorch 2.14.0 (CUDA 12.6)
and an NVIDIA GTX 1080 Ti (every GPU run), and on Linux with Python 3.13 and PyTorch 2.14.1 on the CPU (the
tests and the test report). The dependencies are listed in `pyproject.toml` (and `requirements.txt`): numpy,
scipy, scikit-learn, scikit-image, pillow and matplotlib for everything but the networks; PyTorch,
torchvision and kornia for the networks, as optional extras (below). Every tool runs from the repository's
folder.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The learned matchers for the winter-to-summer alignment also need PyTorch and
kornia: install PyTorch first with the command for your GPU from
[pytorch.org](https://pytorch.org/get-started/locally/), then
`pip install -e ".[match]"`. The summer image features for the shape model
(`treeskel.shape`) also need torchvision, built for the same PyTorch. Print the
installed version, then install torchvision from the index it came from, pinned
to it, so that `pip` does not replace your GPU build (`.[shape]` lists both):

```bash
python -c "import torch; print(torch.__version__)"      # e.g. 2.7.1+cu126
pip install torchvision "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Use the version printed and its `+cu…` part in both places; a version without
`+cu…` is the CPU-only build: `pip install torchvision "torch==<version>"`.

## Reproducing the results

**With this repository alone** (no dataset, no GPU), from the committed clicks, measures and saved masks:

```bash
pytest                                                     # 1. the tests (the PyTorch ones skip without it)
python -m treeskel.shape_model                             # 2. the ridge model on the saved summer features
python -m treeskel.segment --join docs/results/segment_lined_up   # 3. the named U-Net's saved development masks, joined and scored
python -m treeskel.model3d build R12N02                    # 4. a tree's 3D model, from its saved out-of-sample mask
```

Steps 2 and 3 rewrite their committed results with the same numbers: in session 19, on a fresh clone and
another machine, every number agreed to a relative 10⁻⁸ or better (the last digits of floating point). Only
the figures' pixels differ (fonts). `git checkout docs` restores the committed files.

**With the full dataset** in `data/raw/`, in the order of the work (the GPU steps are marked):

1. `python -m treeskel.splits data/raw --load splits.json`: checks the data against the fixed split
   (541 trees, test rows 3, 12 and 16).
2. `python -m treeskel.guided_gt data/raw --overlays`: the winter ground truth, traced through the committed
   clicks (`annotations/winter_clicks.json`), with an overlay per tree to check by eye.
3. `python -m treeskel.shape data/raw`: the winter shape measures and the summer image features
   (`data/shape/`; the features need torchvision).
4. `python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json` (GPU, 5 folds), then
   `python -m treeskel.segment --join docs/results/segment_lined_up`: the named U-Net, trained on the hand
   masks as lined up in `annotations/mask_overlay.json`, scored on the development rows (V opening skill
   +0.75).
5. `python -m treeskel.score_baseline data/raw`: the automatic winter baseline, scored against the ground
   truth and beside the U-Net.
6. `python -m treeskel.final_test data/raw` (GPU): the one evaluation on the test rows. It is done: it
   refuses to run while `docs/results/final_test/report.json` exists.
7. `python -m treeskel.benchmark`, then `--flops` (GPU and CPU): run time, memory, FLOPs and the profile.
8. `python -m treeskel.depth3d data/raw --calibrate`, `tree3d data/raw`, `bend3d data/raw` and
   `bend3d data/raw --typical`: the 3D steps.

**What each path needs.** The repository alone runs the tests and re-scores the saved results: steps 2–4
above need neither PyTorch nor a GPU (checked in session 19 with PyTorch blocked from importing). Running
the work again needs:

- the full dataset, which the MMA3001 teaching team provides and this repository does not redistribute;
- PyTorch and torchvision (`.[shape]`; kornia too for the learned matchers, `.[match]`), installed as in
  Setup;
- a GPU for the networks: the named U-Net's 5-fold run took 20 minutes on a GTX 1080 Ti, and one training
  on all development trees 2 minutes ([`docs/results/benchmark/`](docs/results/benchmark/)).

**The trained networks are not committed.** The experiments never saved their weights, and `models/`, where
`model3d train` saves one (about 57 MB), is git-ignored. What each network predicted is committed (each
run's masks, probabilities and `predictions.csv`), so every reported score can be recomputed without the
networks. But a network can only be retrained, and a new training is another sample of the same recipe,
not the same network: seven trainings ranged 4.00–4.47° in V-opening error
([`docs/results/segment_variants/`](docs/results/segment_variants/)). Each step's README in
`docs/results/` gives its command, its output and the numbers it produced.

## Usage

Every command, including the methods tried and not adopted (the section above gives the order for the
main results):


```bash
pytest                                     # run the tests
python -m treeskel.splits data/sample      # dataset summary and splits
python -m treeskel.splits data/raw --check-images --save splits.json   # make the split (once)
python -m treeskel.splits data/raw --load splits.json                   # reuse it
python -m treeskel.winter_gt data/raw --out outputs/winter_gt --figure outputs/winter_gt.png   # automatic baseline
python -m treeskel.score_baseline data/raw                 # score the baseline against the guided ground truth (development rows)
python -m treeskel.annotate data/raw --dev-only            # click tool (development rows first)
python -m treeskel.guided_gt data/raw --overlays           # render guided ground truth from the clicks
python -m treeskel.tune_cost data/tuning                   # tune the cost map on annotated trees
python -m treeskel.measure_fork data/raw                   # check the fork junction and fill (development rows only)
python -m treeskel.measure_width data/raw                  # check the leader widths (development rows only)
python -m treeskel.measure_registration data/raw           # measure the depth-colour shift (development rows only)
python -m treeskel.align data/raw --matcher lightglue --dev-only   # automatic matching (needs ".[match]"; failed, see docs/results/align)
python -m treeskel.annotate_overlay data/raw --dev-only    # line up each tree's winter photo on its summer photo
python -m treeskel.summer_gt data/raw                      # check the alignments; write the summer masks
python -m treeskel.annotate_pairs data/raw --dev-only      # (tried first) click winter-summer point pairs
python -m treeskel.shape data/raw --figure docs/figures/shape_lines.jpg   # shape measures and summer features (development rows)
python -m treeskel.shape_model                             # predict the winter shape from the summer photo (5 folds)
python -m treeskel.shape_cnn data/raw                      # fine-tune a CNN on the winter shape (5 folds, GPU)
python -m treeskel.segment data/raw                        # U-Net on the hand masks, scored against winter (5 folds, GPU)
python -m treeskel.annotate_mask data/raw --dev-only       # line up each hand mask on its summer photo
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json   # U-Net on the lined-up masks
python -m treeskel.segment --join docs/results/segment_lined_up   # a saved run's masks, pieces joined (no photos needed)
python -m treeskel.segment data/raw --grow docs/results/segment_lined_up_rerun   # masks grown into less sure pixels, joined
python -m treeskel.segment --continue docs/results/segment_lined_up_rerun_joined   # short leaders continued, pruned
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --size 480 640   # whole photo, full size
python -m treeskel.final_test data/raw                     # once, after naming the model in docs/results/final_test/model.json
python -m treeskel.final_test data/raw --out docs/results/final_test_ensemble   # session 16's ensemble, once (secondary)
python -m treeskel.benchmark                               # run time and memory of every method (GPU and CPU)
python -m treeskel.benchmark --flops                       # FLOPs, memory traffic, roofline, profiler (GPU and CPU)
python -m treeskel.depth3d data/raw --calibrate            # the depth map's zero from the ground (development rows)
python -m treeskel.tree3d data/raw                         # the trees in 3D, winter and summer (development rows)
python -m treeskel.bend3d data/raw                         # bends out of the plane: winter, then summer (development rows)
python -m treeskel.bend3d data/raw --typical               # the typical winter shape as the bends' prior (development rows)
python -m treeskel.model3d build R12N02                    # a tree's summer photo as a 3D model, opened (outputs/model3d/)
python -m treeskel.model3d build R12N02 --flat             # ... with every branch on the tree's plane
python -m treeskel.model3d train data/raw                  # once, for photos from outside the dataset (GPU, ~2 min)
python -m treeskel.model3d build path/to/my_tree_RGB.png   # such a photo, its depth my_tree_D.png beside it
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --seed 1 --no-grow     # another sample
python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --cldice --no-grow     # centre-line loss
python -m treeskel.seg_compare ensemble <run> <run> <run> --out docs/results/segment_lined_up_ensemble --data data/raw
python -m treeskel.seg_compare compare data/raw            # every variant against the named U-Net, and the rule
python -m treeskel.label_audit judge data/raw              # the label audit, key by key (the sample is committed)
python -m treeskel.label_audit draw data/raw               # or only its images, to fill in sample.csv by hand
python -m treeskel.label_audit summary                     # after the verdicts are filled in
pdoc -d google -o docs/html src/treeskel   # build the HTML documentation (docs/html/)
pytest --html=docs/test_report.html --self-contained-html   # the test report
```

## Data

Each tree (e.g. `R20N03` = row 20, tree 3 on the orchard map) has a summer
and a winter RGB image and depth image (640x480 PNG), plus a hand-labelled
256x256 summer mask of the main branches. Since session 11 the development rows' masks are also training
labels for the segmentation (`treeskel.segment`), scored against the winter traces; the test rows' masks stay
unused until the final evaluation. Depth is 8-bit relative depth (brighter = closer,
0 = no reading), not metric. The winter depth sits about 5 px below the colour image, so it is moved 5 rows
up and 1 column right when loaded ([`docs/results/depth_registration`](docs/results/depth_registration/)).
See [`data/sample/README.md`](data/sample/README.md).

The full dataset ("Dataset (With Summer GT)", 562 trees in 20 rows) comes from the MMA3001 teaching team
and is not redistributed here. Put it in `data/raw/`, which is git-ignored. 541 trees in 19 rows are used:
row 4 is excluded because its winter depth images are copies of R03N34's.

**Data ownership:** only 26 trees of it are committed, so that the tests and examples run without the full
dataset: the 9-tree sample in `data/sample/` and the 17 trees used to tune the ground-truth tracing in
[`data/tuning/`](data/tuning/README.md). These images remain the teaching team's and are not covered by this
repository's licence, which applies to the code and documentation.

## Inputs and outputs

**Inputs** (per tree; the dataset's file names are `R<row>N<tree>_<Summer|Winter>_<RGB|D|GT>-H-M-S.png`):

| Input | Format | Units and range | Use |
|---|---|---|---|
| Summer colour photo | PNG, 640x480, RGB, 8 bits | 0–255 per channel | The input to the prediction. The network sees the centre 480x480 (columns 80–559), resized to 256x256 |
| Summer depth image | PNG, 640x480, greyscale (3 equal channels), 8 bits | 1–255, brighter is nearer; 0 = no reading. Depth along the camera's axis ≈ 0.01197 m × (300 − value) (calibrated in `treeskel.depth3d`) | 3D only (the tree's plane and bends) |
| Winter colour photo and depth | As the summer ones | As above; the winter depth is moved 5 rows up and 1 column right on loading | The ground truth only, never an input to a prediction |
| Hand mask (`GT`) | PNG, 256x256, black and white | The centre 480x480 of the summer photo | Training labels on the development rows; lined up by hand (`annotations/mask_overlay.json`) |
| Clicks and alignments | JSON in `annotations/` | Pixels (row, column) in the 640x480 photos | The ground truth's 4 clicks (and corrections) per tree; the hand masks' shifts |

**Outputs:**

| Output | Format | Units | Meaning and intended use |
|---|---|---|---|
| Branch mask | 256x256 boolean, saved packed in `predicted_masks.npz` | Pixels of the centre 480x480 | Where the trunk and the two leaders are in the summer photo, including under the leaves |
| Shape measures | `predictions.csv`, `report.json` | Degrees | The V opening (angle between the leaders; the headline), each leader's angle from the trunk and from the vertical, the V's lean, the leaders' bends. The structure a planner needs, independent of image alignment |
| Scores | `report.json`, per-tree CSV | Degrees (mean absolute error), skill (1 perfect, 0 the average tree) | How far each method is from the winter truth |
| 3D model | `.html` viewer and `.ply` meshes (`outputs/model3d/`) | Metres; camera frame turned so `y` is up and `z` points at the camera, fork at the origin | The tree to look at and turn; positions for a planner. Sizes assume 2.0 m from the camera to the fork; angles do not |

**Invalid, missing and unsupported inputs:**

- A photo or depth image that is not 640x480, a mask that is not 256x256, or a "depth" image whose colour
  channels differ: an error naming the file (`treeskel.io`).
- Depth value 0 (no reading): treated as missing, never as a distance.
- A badly named, repeated or incomplete set of files: listed by the dataset index (`unrecognised`,
  `duplicates`, `incomplete`) and left out, so one bad file does not stop a run over the dataset
  (`python -m treeskel.splits data/raw --check-images` also finds identical images, as in row 4).
- `model3d build`: an unknown tree id, a photo with no depth image beside it, a mask with no fork, or too
  little depth on the skeleton gives a message for that tree; the others are still built, and the command
  exits with status 1.
- **Not detected:** photos from another camera or depth encoding, other framings, and trees that are not a
  trunk with two leaders (three leaders, or no fork). The code assumes this dataset's camera and the open-V
  trellis; on such photos its answers would be wrong without a warning.

## Validation design

Whole orchard rows are held out as a final test set (rows 3, 12 and 16: 92 of the 541 trees, 17%), and
5-fold cross-validation grouped by row is used on the remaining rows for model
selection. The split is saved in `splits.json`, committed before any model is
trained, and reloaded with `--load` so it never changes. Row 4 is excluded
because all of its winter depth images are identical. Grouping by row stops neighbouring trees, which share background,
lighting and capture session, from appearing in both training and test data. The test rows were used for
the final test in session 12, after the model had been named on the development rows
([`docs/results/final_test/naming_rule.md`](docs/results/final_test/naming_rule.md)). They were used once
more in session 16, for one secondary result declared and committed before it ran: an ensemble of three
trainings of the named model ([`docs/results/final_test_ensemble/`](docs/results/final_test_ensemble/)).
It showed no gain and changed nothing; the final test stays the headline.

## Limitations

- **What the validation supports:** held-out evidence for the tree's main structure in the photo: the
  angles of the trunk and the two leaders, scored against the bare winter photos on rows the model never
  saw. The 3D model is an estimate built on that, not a validated 3D reconstruction. Its angles rest on the
  depth calibration and its sizes on an assumed 2.0 m to the fork. Where the summer depth does not see the
  wood (most of the leaders), the bends follow the typical winter shape, not the tree. The winter depth
  checks the bends (left-out depth 1.9 cm against the plane's 3.1 cm), but there is no independent 3D
  measurement of the trees.
- **Scope:** one orchard, one camera and its 8-bit depth export, one summer and one winter capture, and
  trees trained to a trunk with two leaders. Only the main branches are estimated, not side branches or
  shoots.
- **Leaders that stop short:** on 31 of 444 development trees (7%) the U-Net's mask has a leader that stops
  under the leaves. They carry a third of the V opening's squared error, and none of the three ways tried
  to complete them helped ([`docs/results/final_test/naming_rule.md`](docs/results/final_test/naming_rule.md)).
- **The training labels:** the provided hand masks sat 20–60 px off the visible trunk, by a different amount
  per tree. They were lined up by hand; an audit of 40 random trees still found 2 of 75 leaders off
  (3%, 95% interval 1–9%; [`docs/results/label_audit/`](docs/results/label_audit/)).
- **Training to training:** seven trainings of the same recipe range 4.00–4.47° in V-opening error on the
  development rows, and four score +0.81 to +0.88 on the test rows; the final test's network was the best
  of the four. So the headline is one sample of the recipe.
- **The test rows** were used twice: the final test, and one secondary result declared before it ran. They
  are three rows (92 trees). Their 95% intervals resample trees within those rows, and trees in a row share
  light and background, so the intervals understate the uncertainty for a new row or orchard. Each row's own
  skill is reported beside them (+0.86 to +0.89 on the V opening).
- **3D:** sizes rest on an assumed 2.0 m from the camera to the fork (angles need only the depth's zero and
  the focal length, whose ranges are reported); one depth step is about 1.2 cm. Summer depth sees the trunk
  but almost never the leaders (3% of their readings), so in summer the leaders' bends mostly follow the
  typical winter shape. The tree's lean from the vertical cannot be determined.
- **The ground truth** assumes the trees did not change between the two captures; which season was
  photographed first is not known.

## How I used AI in this project

I used an AI assistant, Claude Code (Anthropic), throughout the project, in 19 work sessions from
23 September to 22 October 2026. Each session's notes in [`docs/progress/`](docs/progress/) end with what
was mine, what the AI did, and the mistakes it made and how they were caught.

**What I did:**

- chose the project and its idea: a ground truth from the bare winter photos, to judge predictions made
  from the leafy summer photos;
- made the decisions: cross-validation on whole orchard rows, excluding row 4, switching to the dataset
  version with tree numbers, training on the hand masks, the depth registration, the shape of the
  masks at the fork, and which model changes to try last (a centre-line loss and an ensemble) and what to
  keep of them; and agreed each rule for choosing between models before its result was seen;
- clicked the ground truth for all 541 trees, lined up all 449 development hand masks by hand, and
  checked a random sample of 40 of them for leaders drawn off the wood;
- ran everything that needed the full dataset or the GPU on my own computer;
- looked at the images, which found problems the numbers had missed: row 4's copied depth images, leaders
  drawn too thin, a fork that looked like a circle, and predicted leaders that stop short.

**What the AI did:**

- wrote most of the code (the `treeskel` package), the tests, the documentation and the figures;
- proposed methods and checks, explained the ideas behind them and the related paper, and analysed the
  results I sent back;
- drafted the session notes at the end of each session.

By commits, the AI made about three quarters of those that are not merges (132 of 173 at commit d199ee0):
all the code, tests and documentation. Mine are the sample data, the clicks, the line-ups, the label audit
and the results of the runs on my computer.

**Why I used AI for these tasks:**

- *The code and tests:* the project grew to about 24,000 lines of Python in 27 modules, with 8,500 lines of
  tests, in a month. With the AI writing them, my time went into what only I could do: looking at the
  images, clicking the ground truth for 541 trees, lining up 449 masks, and making the decisions.
- *Explanations when I needed them:* Git and pull requests, the terminal, virtual environments and PyTorch's
  errors were new to me, and so were several of the methods (robust fitting, minimal-path tracing,
  Gaussian-process priors, the roofline model). The AI explained them on my own data and results.
- *Reading and proposing checks:* it read the related paper against our capture (same orchard and camera,
  a different season), and proposed checks such as synthetic trees with known answers and the null check
  between trainings.

**How it changed my workflow and understanding:**

- *The project's memory moved into the repository.* The AI forgets between chats, so from session 5 each
  session started a new chat and ended with notes in `docs/progress/`, with the fixed decisions in
  `CLAUDE.md`. Every number and decision can be traced, and those notes are what this section rests on.
- *Every change went through a branch and a pull request that I merged* (more than 70), so nothing reached
  `main` without my seeing it.
- *Checking became most of the work.* Because the AI was confidently wrong often enough, I fixed each rule
  for choosing a model before its result was seen, asked for checks on all 449 development trees instead of
  a few, and looked at the images myself. Asking what a term or a result meant was a check as well: in
  sessions 14 and 17 my questions exposed a wrong count and misleading wording.
- *What I understand:* the methods through what they assume, what they were checked against and where they
  fail, more than through every line of the code, which the AI wrote.

**How the work was checked:** the AI was wrong often enough that nothing was taken on trust. Its mistakes
included judging an image offset by eye, a tuning measure biased towards the old settings, branch widths
measured from the sunlit half only, depth and colour images combined for eight sessions without checking
that they lined up, a count of short leaders that mixed two definitions, a rule for comparing trainings
whose test was too lenient (a null check I had agreed to beforehand caught it), and a comparison that left
out exactly the trees where a model failed. They were caught by my looking at the images, by synthetic
tests with known answers, by checks on samples that should not differ, and by running each check on all
449 development trees instead of a few. When I was unsure what a task asked, as with the label audit, I
said so and it was redone. Every rule for choosing a model was fixed before its result was seen; the test
rows were used for the final test and once more for one declared secondary result. 610 automated tests
cover the code ([`docs/test_report.html`](docs/test_report.html)).

I am writing the report myself, and I am responsible for the project and its results.
