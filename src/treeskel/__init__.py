"""treeskel: estimating apple-tree branch skeletons through foliage from RGB-D images.

The package predicts the main branches of apple trees (the trunk and the two leaders of the open-V
trellis) from summer RGB-D photos, where leaves hide them. The truth comes from winter photos of the
same trees, when the branches are bare. Most modules are also command-line tools,
``python -m treeskel.<module> --help``; the repository's README lists the commands and the results, and
``docs/progress/`` records each work session.

The modules, in the order of the work:

**Data**

- `treeskel.io`: file names, the dataset index, and loading the images (the winter depth registered).
- `treeskel.splits`: the row-grouped test rows and cross-validation folds (``splits.json``).

**Winter ground truth**

- `treeskel.winter_gt`: the automatic method, kept as a baseline.
- `treeskel.annotate`: the click tool for the guided ground truth.
- `treeskel.guided_gt`: tracing through the clicks, the branch widths and the fork; the ground truth.
- `treeskel.tune_cost`: tuning the tracing's cost map against the correction clicks.
- `treeskel.measure_fork`, `treeskel.measure_width`, `treeskel.measure_registration`: the checks that
  set the masks' fork and widths, and the shift between the depth and colour images.
- `treeskel.score_baseline`: the automatic baseline scored against the guided ground truth.

**Aligning the seasons pixel by pixel (abandoned in session 7)**

- `treeskel.align`, `treeskel.annotate_pairs`, `treeskel.annotate_overlay`, `treeskel.summer_gt`.

**The tree's shape, and the summer models**

- `treeskel.shape`: the shape measures (V opening, leader angles, bends), which need no alignment.
- `treeskel.shape_model`: ridge regression on fixed CNN features.
- `treeskel.shape_cnn`: a CNN fine-tuned on the shape measures.
- `treeskel.annotate_mask`: lining up the provided hand masks on their summer photos.
- `treeskel.segment`: the U-Net trained on the hand masks and scored against winter (the named model).
- `treeskel.final_test`: the one evaluation on the test rows.
- `treeskel.benchmark`: run time and memory of every method.

**3D**

- `treeskel.depth3d`: how the 8-bit depth maps to distance, and the camera.
- `treeskel.tree3d`: each tree's plane, the V in that plane, its lean and sizes.
- `treeskel.bend3d`: smooth 3D curves that may bend out of that plane where the depth sees the wood.
- `treeskel.model3d`: a summer photo as a 3D model (an HTML viewer and ``.ply`` files).
"""

__version__ = "0.1.0"
