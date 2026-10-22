# Tuning data

Seventeen trees from "Dataset (With Summer GT)", five files each (summer colour, depth and hand mask; winter
colour and depth), all from development rows 1, 7 and 9:

R01N04, R01N05, R01N08, R01N13, R01N18, R01N25, R01N32, R07N09, R07N17, R07N25, R07N26, R07N29, R07N32,
R09N12, R09N18, R09N32, R09N34.

**Why these trees:** they were picked in session 4 as problem trees for the automatic winter ground truth
(`treeskel.winter_gt`): posts beside the tree, straight trees, a fork found too late or not at all, traces
that jump to another branch. That method was then kept only as a baseline, and the ground truth became the
click-guided tracing (`treeskel.guided_gt`).

**What they are used for:**

- tuning the guided tracing's cost map against the annotator's 64 correction clicks
  (`python -m treeskel.tune_cost data/tuning`; [`docs/results/tune_cost/`](../../docs/results/tune_cost/));
- the first checks of the masks' widths and fork shape (sessions 6 and 8), before each was confirmed on all
  449 development trees;
- tests and examples that need development-row trees with every file (`tests/test_segment.py`,
  `tests/test_model3d.py` and others), and `python -m treeskel.model3d build R01N13` without the full
  dataset.

No test-row tree is here: nothing is tuned on the test rows (3, 12 and 16). The file format is described in
[`data/sample/README.md`](../sample/README.md).

**Ownership:** these images are part of the dataset the MMA3001 teaching team provided. They remain theirs and
are not covered by the repository's licence, which applies to the code and documentation.
