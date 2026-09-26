# Winter-to-summer alignment

The ground-truth masks are traced on the **winter** images, but the summer
model needs them in the **summer** view of the same tree. This folder records
how that alignment is being found and checked (`treeskel.align`).

Status (session 7): automatic matching with hand-designed features failed;
the learned matchers are ready to run on the student's PC. No alignment has
been measured against hand-clicked points yet.

## The problem, as seen on the 21 development trees in the repository

The 17 tuning trees plus R01N01, R01N02, R15N05 and R20N06; test-row trees
were not used.

- **The view changes from tree to tree.** Judged by eye from the trunk base,
  the summer view is shifted by roughly −40 to +90 px sideways and 10–40 px
  vertically relative to the winter view, differently for each tree. Posts
  look about as wide in both seasons, so any change of scale is small. These
  are rough readings, to be replaced by the clicked-point measurement.
- **What both seasons show:** the lower trunk (visible below the canopy in
  almost every summer photo), the drip line, and posts where there are any.
  Leaves hide most of the leaders in summer.
- **The trunk base is usually out of frame in winter:** 18 of the 21 root
  clicks are in rows 459–477 of 480, i.e. the trunk leaves the bottom of the
  image before it reaches the ground. So the ground contact cannot serve as
  a reference point in most trees.

## Approaches that did not work

| Approach | Result on the 21 trees | Evidence |
|---|---|---|
| SIFT keypoints + RANSAC (the classical baseline, `--matcher sift`) | **0/21** usable for every model. Median inliers: 2 (translation), 3 (similarity), 4 (affine), 5 (homography), i.e. what random matches give. The similarity fits are nonsense: 20 of 21 rotate by more than 10° (up to 149°), and 13 have a scale outside 0.75–1.33 (0.0 to 5.5) | `sift_21_dev_trees_output.txt`, `docs/figures/align_sift_R09N12.jpg` |
| Orientation correlation (prototype, not kept): edge directions of the winter trunk region correlated with the summer image by FFT, over scales 0.8–1.25 and rotations ±6° | Wrong by eye on the trees checked; e.g. on R09N12 the winter trunk was placed on leaves about 100 px above the summer trunk. Most trees' best match lay at the edge of the search range | prototype run in the session scratchpad |
| Hand masks (`_GT`) as a bridge between the seasons | They fit neither season exactly: on R20N06 the mask's trunk is on the *winter* trunk's position, about 85 px left of the summer trunk, while on R09N12 its fork is at the *summer* fork's height. This fits session 3's guess that they were drawn on a third capture (the paper's autumn images) | overlays in the session scratchpad |
| A winter-depth band to keep only matches on the tree's plane | Per tree, the median depth along the traced trunk is 111–136, the leaders read 11 units nearer (median; −3 to 23), and the ground visible at the root's row only 5–60 units farther (median 15). There is no fixed band that separates the tree from the ground | numbers from the session |

**Why SIFT fails:** its descriptors summarise the local pattern of image
gradients, and between the seasons almost every local pattern changes
(leaves or no leaves, lighting, grass, fruit). Matches are then decided by
chance.

## Chosen next step: learned matchers (the student's decision)

Two matchers trained on photos of the same outdoor places taken at different
times (MegaDepth), which is the closest available training to a change of
season:

- **DISK + LightGlue** (`--matcher lightglue`): learned keypoints and
  descriptors (DISK), matched by a transformer that looks at all keypoints
  of both images together (LightGlue).
- **LoFTR** (`--matcher loftr`): matches without detecting keypoints first,
  from a coarse grid refined to sub-pixel positions, which helps where
  texture is weak.

Both need PyTorch and kornia and download their pretrained weights on first
use. Loading downloaded weights was blocked in the AI assistant's
environment, so they run on the student's PC. The matches (point coordinates
and scores only) are saved to `data/matches/<matcher>.npz` and committed, so
the fitting, checks and analysis can be done without a GPU.

**Fitting** (same for every matcher): RANSAC with MSAC scoring, then a
least-squares refit on the inliers, for four models: translation (2
parameters), similarity (4), affine (6) and homography (8). The estimators
are written out in `treeskel.align`: the closed-form least-squares
similarity (Umeyama), linear least squares for the affine transform, and
the normalised DLT for the homography. The tests check that each recovers a
known transform exactly, that the closed form agrees with a general-purpose
optimiser, and that RANSAC finds the transform with 60% wrong matches.

**Checks** (provisional limits, `FitLimits`): at least 15 inliers, inliers
spread over at least 2% of the image, scale within 0.75–1.33, rotation
within ±10°. These are to be set from the matcher results.

## How to run (student's PC)

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[match]"
python -c "import torch; print(torch.cuda.is_available())"
python -m treeskel.align data/raw --matcher lightglue --dev-only
python -m treeskel.align data/raw --matcher loftr --dev-only
```

- The first `pip` line installs PyTorch with GPU support; `True` from the
  third line means the GPU will be used. (Without the GPU line, `pip`
  installs the CPU-only build, which works but is much slower.)
- Matching is saved every 25 trees; running the same command again resumes.
- Outputs: `data/matches/<matcher>.npz` (to commit), and in
  `outputs/align/<matcher>/` a `summary.csv` (every tree × model) and contact
  sheets `sheet_similarity_NN.jpg` of the summer images with the warped
  winter masks outlined in magenta and the inliers as green dots.
  `--overlays` also saves a side-by-side picture per tree.

## Next steps

1. Run both matchers on the development rows; commit the two `.npz` files
   and share the console output.
2. Compare the matchers and models: how many trees get a plausible fit,
   where the inliers are (tree, posts, drip line, ground, background), and
   whether the warped trunk sits on the summer trunk (contact sheets, and
   the 21 local trees at full size).
3. Measure the error: hand-clicked point pairs on about 20–30 development
   trees (a click tool is still to be written), then the distance between
   each clicked summer point and the transformed winter point.
4. Warp every mask with the chosen matcher and model; check the test rows
   only at the final evaluation.
