# Seeing Branches Through Leaves: Apple-Tree Skeleton Estimation from RGB-D

MMA3001 Numerical Methods and Machine Learning, individual project.

**Goal:** estimate the main branch structure (trunk and scaffold limbs) of apple
trees from a *summer* RGB-D image, where leaves hide much of the wood. The
*winter* RGB-D image of the same tree, when the branches are bare, is used to
build the ground truth. The intended use is perception for orchard robots
(pruning, harvesting, path planning), which must avoid or target branches they
cannot directly see.

## Results at a glance

The headline measure is the **V opening**, the angle between the two leaders. Skill is how much a method
reduces the squared error of guessing the average tree's shape (1 is perfect, 0 no better than the
average tree, negative worse). Every summer prediction is scored against the same trees' winter photos.

| Step | Result | Details |
|---|---|---|
| Ground truth | 541 winter trees traced through 4 clicks each, plus about one correction (about 15 s a tree); the tuned tracer passes a mean 3.2 px from the correction clicks (4.8 px on trees left out) | `docs/results/tune_cost/` |
| Automatic winter baseline | traces only 43% of trees well (F1 ≥ 0.9 against the guided traces); V opening skill −1.97 | `docs/results/winter_baseline/` |
| Summer models (5 row-grouped folds, 449 development trees) | V opening skill: ridge on CNN features +0.10, fine-tuned angle CNN +0.18, **U-Net on the lined-up hand masks, pieces joined, +0.75**; the hand masks themselves +0.87 | `docs/results/shape/`, `segment_lined_up_joined/` |
| **Final test** (rows 3, 12, 16; 92 trees; run once) | **V opening skill +0.88** [95% interval +0.81, +0.92], mean error 3.4° against 10.4° for the average tree; the hand masks +0.91 | `docs/results/final_test/` |
| Run time | the U-Net predicts a photo in 9.7 ms on a GPU, 116 ms on a CPU, and trains in 2 minutes | `docs/results/benchmark/` |
| 3D | the trees are flat to a median 1.7 cm (sizes assume 2 m from the camera to the fork); the V in the photo is about 2.3° wider than in the tree's own plane | `docs/results/depth3d/`, `tree3d/` |
| Bends out of the plane | the winter depth holds bends: left-out depth predicted to 1.9 cm against the plane's 3.1 cm (94% of trees); the trunk stands a median 14° out of the V's plane; letting the leaders bend leaves the 3D V 0.9° (not 2.6°) narrower than the 2D V | `docs/results/bend3d/` |
| 3D model of a summer photo | opens in any browser, the branches bent where summer sees the wood: [`R01N13.html`](docs/results/model3d/R01N13.html), [`R15N05.html`](docs/results/model3d/R15N05.html) | `docs/results/model3d/` |

## Documentation

- **HTML documentation** of every module, function and class: open [`docs/html/index.html`](docs/html/index.html)
  in a browser (built with pdoc from the docstrings).
- **Test report:** [`docs/test_report.html`](docs/test_report.html) (pytest, every test with its result).
- **Session notes** in [`docs/progress/`](docs/progress/): what was done in each work session, the numbers,
  the decisions, how AI was used (and where it was wrong) and the next steps.
- **Results** in [`docs/results/`](docs/results/), one folder per experiment with a README, and figures in
  [`docs/figures/`](docs/figures/).

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
  benchmark.py   run time and memory of every method (ground truths, mean shape, ridge, angle CNN, U-Net)
  depth3d.py     how the 8-bit depth maps to distance (its zero, from the ground) and the camera
  tree3d.py      the tree in 3D: its plane, the V in that plane, its lean and sizes; summer against winter
  bend3d.py      the trunk and leaders as smooth 3D curves that may leave the plane where the depth sees the
                 wood (winter; summer through the leaves, uncertain sections marked); summer against winter
  model3d.py     a summer RGB-D photo as a 3D model to open and turn in a browser (and .ply files)
annotations/     clicks and alignments per tree (winter_clicks.json, season_pairs.json, season_overlay.json), committed
data/matches/    winter-summer point matches per matcher (from align.py), committed
data/shape/      shape measures and summer image features per development tree (from shape.py), committed
tests/           pytest test suite
docs/progress/   session-by-session progress notes (incl. AI use)
docs/results/    one folder per experiment: README, report.json, per-tree tables
docs/figures/    figures for the report
docs/html/       HTML documentation (pdoc)
docs/test_report.html  test report (pytest-html)
splits.json      fixed test rows and cross-validation folds (do not change)
LICENSE          GNU GPL v3
```

## Setup

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

## Usage

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
python -m treeskel.benchmark                               # run time and memory of every method (GPU and CPU)
python -m treeskel.depth3d data/raw --calibrate            # the depth map's zero from the ground (development rows)
python -m treeskel.tree3d data/raw                         # the trees in 3D, winter and summer (development rows)
python -m treeskel.bend3d data/raw                         # bends out of the plane: winter, then summer (development rows)
python -m treeskel.model3d train data/raw                  # once: save the named U-Net (GPU, ~2 min)
python -m treeskel.model3d build --tree R05N12 --data data/raw   # a summer photo as a 3D model: outputs/model3d/R05N12.html
python -m treeskel.model3d build --tree R05N12 --data data/raw --flat    # ... with every branch on the tree's plane
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

## Validation design

Whole orchard rows are held out as a final test set (rows 3, 12 and 16: 92 of the 541 trees, 17%), and
5-fold cross-validation grouped by row is used on the remaining rows for model
selection. The split is saved in `splits.json`, committed before any model is
trained, and reloaded with `--load` so it never changes. Row 4 is excluded
because all of its winter depth images are identical. Grouping by row stops neighbouring trees, which share background,
lighting and capture session, from appearing in both training and test data. The test rows were used once,
for the final test in session 12, after the model had been named on the development rows
([`docs/results/final_test/naming_rule.md`](docs/results/final_test/naming_rule.md)).


## How I used AI in this project

I used an AI assistant, Claude Code (Anthropic), throughout the project, in 14 work sessions from
23 September to 9 October 2026. Each session's notes in [`docs/progress/`](docs/progress/) end with what
was mine, what the AI did, and the mistakes it made and how they were caught.

**What I did:**

- chose the project and its idea: a ground truth from the bare winter photos, to judge predictions made
  from the leafy summer photos;
- made the decisions: cross-validation on whole orchard rows, excluding row 4, switching to the dataset
  version with tree numbers, training on the hand masks, the depth registration and the shape of the
  masks at the fork; and agreed each rule for choosing between models before its result was seen;
- clicked the ground truth for all 541 trees, and lined up all 449 development hand masks by hand;
- ran everything that needed the full dataset or the GPU on my own computer;
- looked at the images, which found problems the numbers had missed: row 4's copied depth images, leaders
  drawn too thin, a fork that looked like a circle, and predicted leaders that stop short.

**What the AI did:**

- wrote most of the code (the `treeskel` package), the tests, the documentation and the figures;
- proposed methods and checks, explained the ideas behind them and the related paper, and analysed the
  results I sent back;
- drafted the session notes at the end of each session.

**How the work was checked:** the AI was wrong often enough that nothing was taken on trust. Its mistakes
included judging an image offset by eye, a tuning measure biased towards the old settings, branch widths
measured from the sunlit half only, depth and colour images combined for eight sessions without checking
that they lined up, and a count of short leaders that mixed two definitions. They were caught by my
looking at the images, by synthetic tests with known answers, and by running each check on all 449
development trees instead of a few. Every rule for choosing a model was fixed before its result was seen,
and the test rows were used once, for the final test. 468 automated tests cover the code
([`docs/test_report.html`](docs/test_report.html)).

I am writing the report myself, and I am responsible for the project and its results.
