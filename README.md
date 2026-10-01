# Seeing Branches Through Leaves: Apple-Tree Skeleton Estimation from RGB-D

MMA3001 Numerical Methods and Machine Learning, individual project.

**Goal:** estimate the main branch structure (trunk and scaffold limbs) of apple
trees from a *summer* RGB-D image, where leaves hide much of the wood. The
*winter* RGB-D image of the same tree, when the branches are bare, is used to
build the ground truth. The intended use is perception for orchard robots
(pruning, harvesting, path planning), which must avoid or target branches they
cannot directly see.

> Work in progress. See the sections below for what currently exists.

## Repository structure

```
data/sample/     8-tree sample of the dataset (full dataset not committed)
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
  depth3d.py     the 3D estimate: first, how the 8-bit depth maps to distance (its zero, from the ground)
annotations/     clicks and alignments per tree (winter_clicks.json, season_pairs.json, season_overlay.json), committed
data/matches/    winter-summer point matches per matcher (from align.py), committed
data/shape/      shape measures and summer image features per development tree (from shape.py), committed
tests/           pytest test suite
docs/progress/   session-by-session progress notes (incl. AI use)
splits.json      fixed test rows and cross-validation folds (do not change)
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
pdoc -o docs/_build src/treeskel           # build HTML documentation
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
and is not redistributed here. Put it in `data/raw/`, which is git-ignored.

## Validation design

Whole orchard rows are held out as a final test set (about 15% of trees), and
5-fold cross-validation grouped by row is used on the remaining rows for model
selection. The split is saved in `splits.json`, committed before any model is
trained, and reloaded with `--load` so it never changes. Row 4 is excluded
because all of its winter depth images are identical. Grouping by row stops neighbouring trees, which share background,
lighting and capture session, from appearing in both training and test data.


## Acknowledgement of AI Assistance in Project Development

This project incorporates Artificial Intelligence (AI) tools as assistive technology throughout its development lifecycle. Our methodology emphasizes a "human-in-the-loop" approach, ensuring that human oversight, critical evaluation, and final decision-making remain central to every stage.

AI tools may have contributed to the following aspects:

*   **Code Generation and Refinement:** AI has aided in generating boilerplate code, suggesting improvements, and refactoring code segments. All AI-generated code undergoes thorough human review, testing, and comprehension before integration.
*   **Documentation and Explanation:** AI has assisted in drafting documentation, clarifying complex algorithms, and summarizing research. Human verification is paramount to ensure accuracy, clarity, and completeness.
*   **Test Case Generation:** AI has been utilized for suggesting potential test cases and identifying edge cases. Human developers rigorously validate the relevance and correctness of all generated tests.
*   **Conceptual Brainstorming:** AI has supported brainstorming sessions, exploring diverse approaches, and providing background information on relevant subjects. All conceptual outputs are critically evaluated by human team members.

Our commitment includes:

*   **Transparency:** Clearly documenting the specific applications and extent of AI usage.
*   **Responsibility:** Upholding ethical and responsible AI practices, with a focus on fairness, accountability, and privacy.
*   **Quality Assurance:** Maintaining rigorous standards for code quality and functionality through comprehensive human-led testing and review processes, irrespective of AI involvement.

Ultimately, the human developers bear full responsibility for the integrity, correctness, and functionality of this project.
