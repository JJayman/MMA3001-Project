# Winter-to-summer alignment

The ground-truth masks are traced on the **winter** images, but the summer
model needs them in the **summer** view of the same tree. This folder records
how that alignment was chosen and checked.

**Status (session 7).** Automatic matching failed. SIFT was at chance on
all 21 trees tried. The learned matcher DISK+LightGlue gave a plausible fit
on about 1% of the 449 development trees. The student therefore chose to
align the seasons with **clicked point pairs**: `treeskel.annotate_pairs`
(the click tool) and `treeskel.summer_gt` (fit, error and summer masks). No
pairs have been clicked yet.

## The problem, as seen on the 21 development trees in the repository

The 17 tuning trees plus R01N01, R01N02, R15N05 and R20N06; test-row trees
were not used.

- **The view changes from tree to tree.** Judged by eye from the trunk base,
  the summer view is shifted by roughly −40 to +90 px sideways and 10–40 px
  vertically relative to the winter view, differently for each tree. Posts
  look about as wide and as upright in both seasons, so any change of scale
  or roll is small. These are rough readings, to be replaced by the
  clicked-pair measurement.
- **What both seasons show:** the lower trunk (visible below the canopy in
  almost every summer photo), the drip line, and posts where there are any.
  Leaves hide most of the leaders in summer.
- **The trunk base is usually out of frame in winter:** 18 of the 21 root
  clicks are in rows 459–477 of 480, i.e. the trunk leaves the bottom of the
  image before it reaches the ground. So the ground contact cannot serve as
  a reference point in most trees.
- **Geometry.** The trees are trained flat along the row, so the trunk and
  leaders lie close to one vertical plane. A change of camera position maps
  that plane between the images by a homography. With the plane roughly
  facing the camera and little roll, this is close to a similarity (scale,
  rotation, shift). Points off the plane, such as the ground and the rows
  behind, shift differently (parallax).

## Automatic matching: what was tried

| Approach | Result | Evidence |
|---|---|---|
| SIFT keypoints + RANSAC, 21 trees | **0/21** usable for every model. Median inliers: 2 (translation), 3 (similarity), 4 (affine), 5 (homography), i.e. what random matches give. The similarity fits are nonsense: 20 of 21 rotate by more than 10° (up to 149°), and 13 have a scale outside 0.75–1.33 (0.0 to 5.5) | `sift_21_dev_trees_output.txt`, `docs/figures/align_sift_R09N12.jpg` |
| Orientation correlation (prototype, not kept): edge directions of the winter trunk region correlated with the summer image by FFT, over scales 0.8–1.25 and rotations ±6° | Wrong by eye on the trees checked. On R09N12 the winter trunk was placed on leaves about 100 px above the summer trunk; most trees' best match lay at the edge of the search range | prototype run in the session |
| Hand masks (`_GT`) as a bridge between the seasons | They fit neither season exactly. On R20N06 the mask's trunk is on the *winter* trunk's position, about 85 px left of the summer trunk; on R09N12 its fork is at the *summer* fork's height. This fits session 3's guess that they were drawn on a third capture | overlays in the session |
| A winter-depth band to keep only matches on the tree's plane | Per tree, the median depth along the traced trunk is 111–136. The leaders read 11 units nearer (median; −3 to 23), and the ground visible at the root's row only 5–60 units farther (median 15). No fixed band separates the tree from the ground | numbers from the session |
| **DISK + LightGlue** (learned, pretrained on MegaDepth), 449 development trees, run on the student's PC | **Median 6 matches per tree** (maximum 72) and median confidence 0.18. Only **53 trees (12%) have 3 or more mutually consistent matches** under a plausible similarity (scale 0.8–1.25, rotation within 8°), **6 (1%) have 5 or more**, and 3 have 8 or more. On the 21 local trees the similarity is usable on 1 | `lightglue_consistency_output.txt`, `lightglue_21_dev_trees_output.txt`, `docs/figures/align_lightglue_21_dev_trees.jpg` |
| LoFTR (learned, detector-free) | Not run: its weights are only on a server whose certificate Python could not verify on the student's PC, and the browser download failed too | — |

**Why "consistent matches" and not inliers.** Any two matches define
*some* similarity, so a plain RANSAC fit always has at least 2 "inliers",
even for random matches. `consistency_table` (`python -m treeskel.align DATA
--matcher lightglue --consistency`) lets RANSAC try only physically
plausible transforms. Its inliers are then the largest set of matches that
one possible camera move explains.

**Where LightGlue works, and how well.** The trees with the most consistent
matches are R01N18 (32), R13N33 (29) and R01N32 (17). On R01N18 every inlier
joins the winter post to the summer post (`docs/figures/align_lightglue_R01N18.jpg`),
and on R01N32 winter trunk points to summer trunk points. Both are correct
matches, but they lie along one line, which fixes the shift well and the
scale and rotation poorly. On R01N18 the warped trunk lands about 20–30 px
from the summer trunk (by eye). So even the best automatic fits are only
roughly right.

**Why automatic matching fails here.** SIFT's descriptors summarise the
local pattern of image gradients, and between the seasons almost every local
pattern changes (leaves or no leaves, lighting, grass, fruit). Learned
matchers cope with changes of lighting and viewpoint. But in summer the
leaves cover most of the image, and the structure shared with winter (posts,
the lower trunk, the drip line) is small and seen against different
surroundings, so even they rarely find it.

## Chosen method: clicked point pairs (the student's decision)

`python -m treeskel.annotate_pairs data/raw --dev-only` shows each tree's
winter image (with its traced mask) next to its summer image. The student
clicks a point in one, then the same physical point in the other.

- **Which points:** points on the tree's row plane that are clearly the same
  physical point in both seasons:
  - trunk features: the graft union, knots, scars, where a side branch leaves;
  - where the drip line crosses the trunk or a post;
  - post features: the top, clips, knots;
  - the same on neighbouring trunks.

  Avoid the ground, leaves and fruit, the rows behind, and thin branches.
  Aim for 4 pairs (at least 3), spread out.
- **Live check:** from 2 pairs, the winter mask moved by the fitted
  similarity is drawn on the summer image. From 3 pairs, each pair shows its
  leave-one-out error, and above 10 px it is marked red. With only 3 pairs,
  one wrong click raises all three errors (each is checked against a fit
  that uses the other two), so a fourth pair helps.
- **Model choice:** `python -m treeskel.summer_gt data/raw --dev-only`
  compares translation, similarity and affine by leave-one-out error on the
  development rows, on the trees with at least 4 pairs. A homography would
  need 5 or more pairs per tree to be checked this way.
- **Error measure:** the leave-one-out error covers click noise, the model's
  fit and points off the plane, measured on points the fit has not seen. A
  planned second clicking of about 20 trees will measure how repeatable the
  alignment is.
- **Output:** `outputs/summer_gt/<tree>_summer_gt.png` (the winter mask in
  the summer view, the U-Net's label) and `<tree>_summer_covered.png` (where
  the winter image covers the summer view; outside it the label is unknown).

## Running the matchers (how the LightGlue result was produced)

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[match]"
python -c "import torch; print(torch.cuda.is_available())"
python -m treeskel.align data/raw --matcher lightglue --dev-only
python -m treeskel.align unused --matcher lightglue --consistency
```

- The first `pip` line installs PyTorch with GPU support (see below if it
  fails); `True` from the third line means the GPU will be used.
- Matching is saved every 25 trees to `data/matches/<matcher>.npz` (point
  coordinates and scores only, committed); running the same command again
  resumes.
- Outputs in `outputs/align/<matcher>/`:
  - `summary.csv`: every tree × model;
  - contact sheets `sheet_similarity_NN.jpg`: the summer images with the
    warped winter masks outlined in magenta and the inliers as green dots;
  - with `--overlays`, a side-by-side picture per tree.

### If PyTorch does not load

The first run on the student's PC stopped with `OSError: [WinError 193] %1
is not a valid Win32 application. Error loading "...\torch\lib\torch_cuda.dll"
or one of its dependencies`. The error comes from importing PyTorch, before
any matching. Windows found a DLL in the PyTorch install that is not a
valid 64-bit program, i.e. the install was damaged or incomplete. Here the
likely cause was disk space: the PC had about 3 GB free, which is not enough
for the GPU build. After freeing space, a clean reinstall worked.

**How much space the builds need:**

| Build | Download | Installed |
|---|---|---|
| CPU-only, Windows (torch 2.14.0 from PyPI, measured from the wheel) | 124 MB | 0.48 GB |
| GPU (CUDA), as installed in the AI assistant's Linux environment (torch 2.14.0, CUDA 13) | — | 5.2 GB (PyTorch 1.2 GB, NVIDIA's CUDA libraries 3.2 GB, Triton 0.9 GB) |

The Windows GPU build bundles the same kind of CUDA libraries, and `pip`
also needs room for the download (a few GB) while it unpacks, so about 8 GB
free is a safe margin. Its exact size could not be checked from the AI's
environment, because the PyTorch download site is blocked there.

**Steps:**

1. Check for an NVIDIA GPU and driver with `nvidia-smi`. The top right shows
   the highest CUDA version the driver supports. It must be at least the
   build's version (12.6 for `cu126`); if it is lower, update the driver. An
   RTX 50-series card needs a build for CUDA 12.8 or later (`cu128` in the
   address below), because the `cu126` build does not support it. If the
   command is not found, go to step 3.
2. Close other programs using the `.venv` (VS Code terminals, Jupyter),
   make sure about 8 GB are free, and reinstall. `pip cache purge` deletes
   any copy of an earlier download that `pip` kept:

   ```bat
   pip uninstall -y torch
   pip cache purge
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

3. If it still fails, use the CPU-only build, which has no CUDA DLLs. On
   Windows, PyTorch from PyPI is the CPU-only build:

   ```bat
   pip uninstall -y torch
   pip install --no-cache-dir torch
   ```

   Matching is then slower. Measured in the AI assistant's environment (4 CPU
   cores) with untrained networks of the same size, both matchers took about
   4 s per tree at 640x480. LightGlue ran without its early stopping, so that
   is its slowest case. That makes about 30 min per matcher for the 449
   development trees. The GPU build is still worth fixing before the U-Net
   is trained.

## Next steps

1. Click pairs on the 21 development trees in the repository first
   (`--trees R01N01 R01N02 ...`), so the protocol and the tool can be
   checked here against the images.
2. Choose the model by leave-one-out error on the development rows, and
   look at the trees flagged by `summer_gt`.
3. Click the remaining trees (development rows, then test rows), and
   re-click about 20 trees to measure repeatability.
4. Write the summer masks for all trees; they become the training labels
   for the summer model.
