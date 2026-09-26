# Winter-to-summer alignment

The ground-truth masks are traced on the **winter** images, but the summer
model needs them in the **summer** view of the same tree. This folder records
how that alignment was chosen and checked.

**Status (end of session 7): abandoned.** Pixel alignment of the winter
masks onto the summer photos was given up. Four approaches failed:

- **SIFT:** at chance on all 21 trees tried.
- **DISK+LightGlue:** a plausible fit on about 1% of the 449 development
  trees.
- **Clicked point pairs** (`treeskel.annotate_pairs`): about 2 minutes a
  tree, and the pairs disagree by 12–18 px.
- **Lining up the photos by hand** (`treeskel.annotate_overlay`): the
  student aligned 2 trees and skipped a third. Pairs and overlay disagree
  on the scale by about 15%.

Too little rigid structure is visible in both seasons, and nothing seen in
both fixes the scale. The student's next steps are to improve the winter
masks, then learn the tree's *shape* from winter, which needs no alignment
(`docs/progress/2026-09-26-session-7.md`).

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

**Found by the student when trying the pair tool:**

- **The camera stood at a different height, pitch, roll and yaw in each
  season**, so a similarity can leave the masks misaligned. Each change maps
  to part of the transform. Height and sideways moves shift the tree,
  distance scales it, and roll rotates it; a similarity covers these. Pitch
  and yaw also add perspective, which only a homography models. How much
  that matters was measured on a D435-like camera (focal length 615 px) with
  7 points spread over the image. For a pitch difference of 3°, 5°, 8° and
  10°, the similarity's leave-one-out error was 5.5, 9.2, 14.7 and 18.5 px
  (median). The homography was exact. The affine transform was worse than
  the similarity (7.6–25.9 px): its extra parameters do not describe
  perspective. See `test_a_camera_pitch_needs_the_homography` in
  `tests/test_summer_gt.py`.
- **The two photos frame the tree differently.** A winter photo may start
  halfway up the trunk while the summer photo shows the whole trunk, and the
  other way round at the top. Consistent with this, 18 of the 21 winter
  root clicks are at the image's bottom edge, and the two usable LightGlue
  fits put the summer view 72–81 px higher than the winter view. The part
  of the summer image that the winter image does not cover has no winter
  label. `summer_gt` writes it as `<tree>_summer_covered.png` so that
  training can ignore it, and the pair tool shades it. Extending the trunk
  down into it would be an assumption, not an observation; not done.
- **Capture order.** A large pale wooden structure behind some row 1 trees
  in summer (R01N01, R01N02) is missing from the same stretch of alley in
  winter. Enlarged, it looks like stacked wooden harvest bins (horizontal
  slats with gaps, stacked in layers). Bins stand in the orchard at picking
  time, which fits the ripe fruit in the summer photos, so this alone does
  not show which season was photographed first. It stays an open question
  (asked of staff in session 3).

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
  compares the models by leave-one-out error on the development rows, each
  time on the trees with enough pairs to check every model compared:
  translation, similarity and affine on trees with at least 4 pairs, and
  all four including the homography on trees with at least 5. The tool's
  live fit can use either (`--model similarity` or `--model homography`).
  The saved pairs do not depend on it.
- **Error measure:** the leave-one-out error covers click noise, the model's
  fit and points off the plane, measured on points the fit has not seen. A
  planned second clicking of about 20 trees will measure how repeatable the
  alignment is.
- **Output:** `outputs/summer_gt/<tree>_summer_gt.png` (the winter mask in
  the summer view, the U-Net's label) and `<tree>_summer_covered.png` (where
  the winter image covers the summer view; outside it the label is unknown).

## Clicked point pairs: what the 11 trees showed

The student clicked 60 pairs on 11 development trees (R01N01–R01N32,
R07N09, R07N17; `annotations/season_pairs.json`). It took about 20 minutes
for 10 trees, i.e. roughly 2 minutes a tree, or 18 hours for all 541.

| | Pairs | Similarity leave-one-out error (median) | 90th percentile | Above 10 px |
|---|---|---|---|---|
| All pairs | 60 | 17.8 px | 37.7 px | 80% |
| Pairs away from the photos' top and bottom edges | 33 | 11.8 px | 34.1 px | 59% |

The homography was worse than the similarity (28.3 px median on the 10
trees with 5 or more pairs): with noisy points, its extra parameters follow
the noise. All 11 trees had a pair more than 15 px from the fit of the
others. Output: `pairs_11_trees_output.txt`. Figure:
`docs/figures/align_pairs_R01N18.jpg` (circles: clicks; squares: where the
fit of all pairs puts each winter point).

**Why the pairs disagree** (found by looking at them, and by the student):

- **Leader tops are not the same point.** Where a leader leaves the top of
  the photo depends on the framing, which differs between the seasons
  (27 of the 60 pairs were at the top or bottom edge).
- **Points under the leaves** (e.g. R07N09's fork) had to be guessed in
  summer.
- **Featureless stretches of a post** fix the sideways position but not the
  height.

Finding enough points that are certainly the same in both seasons is the
bottleneck, not the tool. About 12 px at 2 minutes a tree is neither
accurate nor fast enough: the leaders are about 7 px wide.

## Chosen method: lining up the photos (the student's decision)

`python -m treeskel.annotate_overlay data/raw --trees ...` shows the summer
photo with the winter photo on top, half see-through, moved by a transform
the student adjusts until the trunk, posts, drip line and ground line sit on
top of each other.

- **Controls:** drag to shift, scroll to scale around the cursor, keys for
  1 px shifts, roll, scale and two perspective tilts. Space flips between
  the photos, so misalignment shows as a jump.
- **The transform** (`summer_gt.Overlay`) is built from the camera changes
  it stands for: shift (height, sideways move), scale (distance), roll, and
  pitch and yaw as the perspective of a rotated pinhole camera (`K R K⁻¹`,
  focal length 615 px assumed), with the image centre kept in place.
  Keyboard changes keep the tree's fork fixed, so fine-tuning does not undo
  a lined-up trunk.
- **Starting point:** each tree starts from the median transform of the
  trees already aligned in its row, else of all aligned trees.
- **Records:** the time spent per tree, and a flag (`f`) for winter masks
  that look wrong (the student saw masks off the branch and of the wrong
  width), so they can be fixed later.
- **Checks** (`python -m treeskel.summer_gt data/raw`):
  - the distance between the clicked pairs away from the edges and where
    the overlay transform puts their winter points (an independent method);
  - repeatability: with `--repeat`, how far the winter masks move between two
    alignments of the same trees;
  - the time per tree and the range of each camera change.

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

## Lining up by hand: what the 3 trees showed

The student aligned R01N01 (143 s) and R01N02 (24 s, starting from
R01N01's transform) and skipped R01N04 after 165 s. The alignments agreed
with the clicked pairs away from the edges to within 4–12 px. Their scales
were 1.15 and 1.12, whereas the pairs had given 0.93 and 0.96. So the scale
is not pinned down by what both photos show, and a 10% error moves the
leader tips by 20–30 px.

**Tool changes on the way.** The first version took about 400 ms per change
(a scipy warp and a full repaint of three layers). With Pillow's compiled
warp, one blended picture and blitting, it takes about 63 ms. The winter
photo now starts at 30% opacity, and an outline view shows the summer photo
with the winter depth silhouette.

## Next steps (the student's decision)

1. Improve the winter masks. The student saw masks off the branch and of the
   wrong width.
2. Learn the tree's shape from winter without alignment. Leader angles
   relative to the trunk, the V opening and curvature can be compared
   between seasons without aligning the photos.
3. Optional later: the provided hand masks are roughly in the summer view
   (on 12 local trees their trunks are within about 0–40 px of the summer
   trunk), so they could serve as training labels, with the winter traces
   as the truth for the shape.
