# The tree's shape from winter, predicted from summer (session 10)

Pixel-by-pixel alignment of the winter and summer photos was abandoned in
session 7 ([`../align/`](../align/)): too little rigid structure shows in
both seasons. Angles need no alignment. This step turns each winter trace
into a few angles that describe the tree's shape (the V opening, each leader's
angle and bend), and asks how well they can be predicted from the summer
photo. The code is `treeskel.shape` (the measures, the run on the student's
PC, the summer features) and `treeskel.shape_model` (the models and scores).

All numbers are for the **449 development trees**; the test rows (3, 12, 16)
were not measured.

## 1. The shape measures

Each part of the tree is summarised by straight lines fitted by total least
squares (`line_direction`), in degrees:

| Measure | Meaning | Camera roll between the seasons |
|---|---|---|
| V opening | angle between the two leader lines | no effect |
| Left / right bend | the upper half of a leader's stretch against its lower half, positive outwards | no effect |
| Left / right leader from trunk | each leader's line from the trunk's axis, positive outwards | no effect |
| V's lean from trunk | the V's bisector from the trunk's axis, positive to the right | no effect |
| Trunk / left / right leader from vertical, V's lean from vertical | the same lines from the photo's vertical | adds to them |

- **The trunk's axis** is its top: from the fork down 4 trunk radii.
- **The leader lines** run from 1.5 to 16 trunk radii from the fork (the
  student's choice, below). A trunk radius scales with the photo, so the same
  stretch of tree is measured in any photo of it.
- **Checks with known answers** (`tests/test_shape.py`): synthetic V trees with
  set tilts and bends are measured to within 0.15–0.5°; a bend along a circle
  is half its turning, as it should be; shifting, scaling and rolling the tree
  leave the roll-free measures unchanged (< 0.3°) and add the roll to the
  others; the mirror image gives the mirrored measures. Turning the camera by
  5–10° of pitch or yaw (the `K R K⁻¹` model of session 7) changes the angles
  by less than 0.8°.

### Choices (the student's), and the evidence

**Both references for the leader angles, with the V opening and the bends,
which need neither, as the headline.** The trunk's top leans a lot, and the
leaders do not follow it: the trellis holds them.

| 449 trees | SD across trees |
|---|---|
| Trunk from vertical | 15.4° |
| V's lean from trunk | 16.4° |
| V's lean from vertical | 6.3° |

So angles measured from the trunk mostly measure the trunk's own lean (in the
clicks alone, the trunk's lean and the V's lean from the trunk correlate at
−0.96). Angles from the vertical avoid that, but carry the camera's roll
between the seasons, which differs by row (section 3).

**A fixed stretch of each leader (1.5–16 trunk radii), not all of it in
view.** The leaders curve (the bends), so a line through everything in view
depends on where the photo's top edge cuts the leaders, and the summer photos
are framed tighter than the winter ones. How much a tighter top edge would
change the measures (median / 95th percentile, 449 trees, `framing_change`):

| Height above the fork left in view | Fixed stretch: V opening, left bend, right bend | Whole leader: V opening, left bend, right bend |
|---|---|---|
| 85% | 0.0 / 0.0°, 0.0 / 0.0°, 0.0 / 0.0° | 3.1 / 6.7°, 3.5 / 8.9°, 2.8 / 7.5° |
| 70% | 0.0 / 1.0°, 0.0 / 2.4°, 0.0 / 0.9° | 5.5 / 14.1°, 6.5 / 17.0°, 6.0 / 15.6° |

In view, the winter leaders are a median 27.7 trunk radii long and the hand
masks' 23.4; 1 of 898 winter leaders and 9 of 896 hand-mask leaders are
shorter than 16 trunk radii.

The choice was made on the framing check alone. The stretch was not tuned for
agreement with the hand masks, which would favour the baseline being scored.

## 2. The winter shape of the development trees

Measured on the winter traces (`data/shape/shape_measures.csv`), with the
provided hand masks' measures against them:

| Measure | Winter mean | SD across trees | Hand mask − winter: mean | Median abs. difference | r |
|---|---|---|---|---|---|
| V opening | 53.1° | 13.7° | −0.3° | 2.9° | 0.86 |
| Left leader from trunk | 21.8° | 17.7° | +2.8° | 4.9° | 0.86 |
| Right leader from trunk | 31.3° | 17.8° | −3.1° | 4.9° | 0.84 |
| V's lean from trunk | 4.7° | 16.4° | −3.0° | 4.5° | 0.85 |
| Left bend | −2.9° | 12.0° | −0.3° | 4.7° | 0.72 |
| Right bend | −5.6° | 12.0° | −0.5° | 4.8° | 0.67 |
| Trunk from vertical | −2.9° | 15.4° | +1.7° | 4.1° | 0.82 |
| Left leader from vertical | −24.7° | 9.5° | −1.1° | 2.8° | 0.86 |
| Right leader from vertical | 28.4° | 9.1° | −1.4° | 2.9° | 0.81 |
| V's lean from vertical | 1.8° | 6.3° | −1.3° | 2.3° | 0.82 |

- The leaders bend back towards the vertical on average (negative bends).
- R07N19, which splits at the ground, has no trunk axis; R17N23's hand mask
  gave no fork.
- `docs/figures/shape_lines.jpg` shows the fitted lines on six trees in both
  seasons.

## 3. The hand masks were drawn on the summer view

Each hand mask's centre lines were laid onto its winter trace by the rotation,
scale and shift that fit best (`hand_frame`, iterative closest points).

- **448 of 448 fit within 8 px** (median 2.2 px after the fit): the hand
  masks have the winter shape.
- **Their rotation is the camera's roll between the seasons.** On the 11 trees
  with the clicked point pairs of session 7, it follows the pairs' rotation
  tree by tree (r = +0.73; means +2.4° and +2.9°).
- **The roll differs by row:** winter to hand mask, median by row from −3.6°
  (row 11) to +2.0° (row 1); over all trees mean −0.7°, SD 2.3°. On the 11
  pair trees (rows 1 and 7) it had looked like a systematic +3°; on 449 that
  is row 1's value.
- The scale is not measured this way: lines through the fork map onto
  themselves under any scaling about it (a known-answer test shows this).

For the models this means that the angles from the vertical carry a few
degrees of roll that differ by row. The row-grouped folds cannot learn them,
which caps what any model can reach on those measures.

## 4. Predicting the shape from the summer photo

**Model** (`treeskel.shape_model`, fixed before seeing any result):
- **Features:** ResNet-18 pretrained on ImageNet, used as a fixed feature
  extractor on the summer photo (half size, 240x320). Its last feature map is
  averaged over the whole photo and over a 2x2 grid of cells: 2,560 numbers
  per tree, computed on the student's PC in 20 s
  (`data/shape/summer_resnet18.npz`).
- **Regression:** ridge regression on the standardised features, one per
  measure.
- **Mirror images:** each training photo's mirror image is added with the
  mirrored measures, after centring on the training trees' mean. A
  prediction averages the photo's and its mirror image's.
- **The penalty** is chosen within each training set by row-grouped
  cross-validation (nested).

**Validation:** the 5 row-grouped folds of `splits.json`; each tree is
predicted by the fold that held its row out.

**Baselines:**
- *mean shape*: the training trees' mean;
- *hand masks*: the provided masks' measures as they are, a person's reading
  of the summer photo.

Also shown, for comparison only: the depth image's features, both images'
features, the whole-photo features alone, and no mirror images.

Skill = 1 − MSE / MSE(mean shape): 0 is no better than the mean shape, 1 is
perfect. Each cell gives the MAE and the skill over the 449 held-out
predictions:

| Measure | Mean shape | Hand masks | **Ridge, photo (the model)** | Ridge, depth | Ridge, both | Ridge, whole photo only | Ridge, no mirror images |
|---|---|---|---|---|---|---|---|
| V opening | 10.4° | 4.4° / +0.74 | **10.0° / +0.06** | 9.9° / +0.10 | 10.0° / +0.06 | 10.1° / +0.04 | 10.0° / +0.07 |
| Left bend | 9.6° | 6.5° / +0.40 | **9.5° / +0.03** | 9.5° / +0.02 | 9.5° / +0.02 | 9.6° / +0.01 | 9.6° / +0.01 |
| Right bend | 9.4° | 6.8° / +0.36 | **9.4° / 0.00** | 9.4° / 0.00 | 9.4° / 0.00 | 9.4° / 0.00 | 9.4° / 0.00 |
| Left leader from trunk | 13.5° | 6.8° / +0.71 | **13.5° / 0.00** | 13.6° / −0.01 | 13.5° / 0.00 | 13.5° / −0.01 | 13.5° / +0.01 |
| Right leader from trunk | 13.9° | 7.0° / +0.68 | **13.8° / +0.01** | 13.7° / +0.03 | 13.8° / +0.02 | 13.8° / +0.02 | 13.5° / +0.06 |
| V's lean from trunk | 12.4° | 6.4° / +0.69 | **12.5° / −0.01** | 12.5° / 0.00 | 12.5° / 0.00 | 12.4° / −0.01 | 12.3° / +0.02 |
| Trunk from vertical | 11.2° | 6.1° / +0.66 | **11.1° / +0.02** | 11.2° / −0.01 | 11.2° / +0.02 | 11.1° / 0.00 | 10.9° / +0.06 |
| Left leader from vertical | 7.4° | 3.5° / +0.73 | **7.1° / +0.07** | 7.1° / +0.06 | 7.1° / +0.07 | 7.2° / +0.04 | 7.1° / +0.07 |
| Right leader from vertical | 7.3° | 3.8° / +0.61 | **7.1° / +0.02** | 7.2° / +0.04 | 7.2° / +0.01 | 7.3° / 0.00 | 7.2° / +0.01 |
| V's lean from vertical | 5.0° | 2.9° / +0.60 | **4.9° / +0.04** | 5.0° / −0.01 | 4.9° / +0.03 | 5.0° / −0.01 | 5.0° / −0.01 |

**V opening, in detail:**
- The model is better than the mean shape on 55% of trees (Wilcoxon
  p = 0.006), r = 0.20. Its skill per fold is −0.02, +0.04, +0.09, +0.06 and
  +0.16.
- The hand masks are better on 76% of trees, r = 0.86, with skill per fold
  +0.54 to +0.86.

Full output: [`model_output.txt`](model_output.txt); scores, penalties and
per-fold skills: [`report.json`](report.json); every tree's predictions:
[`predictions.csv`](predictions.csv). Figures:
`docs/figures/shape_model_skill.png` (skill per measure, with the range over
the folds) and `docs/figures/shape_model_scatter.png` (predictions against the
winter truth).

### Is it the model or the features?

The same features with two non-linear models, their settings chosen by the
same nested row-grouped cross-validation (exploratory, no mirror images;
[`robustness.py`](robustness.py), [`robustness_output.txt`](robustness_output.txt)):

| Skill | V opening | Left bend | Right bend | Best of the other measures |
|---|---|---|---|---|
| k nearest neighbours | +0.06 | 0.00 | 0.00 | +0.07 (trunk from vertical) |
| Kernel ridge (RBF) | +0.04 | 0.00 | 0.00 | +0.07 (trunk from vertical) |

No model does better than about +0.1, so the limit is in the features.

**Why the features carry so little:**
- **They barely tell a photo from its mirror image.** Networks trained on
  ImageNet are trained on flipped photos too. Here a photo's whole-photo
  features correlate at 0.95 with its mirror image's, so which way a leader
  leans is mostly averaged away.
- **Averaging over large cells discards where things are.** The leaders'
  directions depend on where the visible pieces of bark sit relative to one
  another.
- **ImageNet features describe objects and textures, not the geometry of thin,
  partly hidden branches.**
- **The penalty confirms it.** It is chosen large (3×10³ to 3×10⁵, fairly
  stable over the folds; for the right bend the top of the grid, 10⁷, which
  is the mean shape, in 3 of 5 folds), so the predictions stay within a few
  degrees of the mean shape (see the scatter figure).

## 5. What this means

- **The information is in the summer photo.** A person's reading of it (the
  hand masks) predicts the winter V opening to 4.4° (MAE) against 10.4° for
  the mean shape (skill +0.74), and the other measures with skill 0.36–0.73.
- **Fixed ImageNet features do not capture it,** whatever the regression on
  top: skill at most about 0.1.
- **The next model has to see where the branches are.** Two ways:
  1. Train the network itself on the winter measures (fine-tuning on the
     student's GPU): it keeps the winter traces as the only labels.
  2. Train a segmentation network on the hand masks, which session 10 showed
     are drawn in the summer view and have the winter shape (448 of 448 laid
     onto the winter traces within 8 px). Then measure its predicted masks
     with the same code and score them against the winter traces. This would
     change "hand masks are for validation only", so it is the student's
     decision.

## 6. Trees to look at

The largest disagreements between the hand masks and the winter traces may be
tracing slips near the fork, bad hand masks, or unusual trees:

| Measure | Tree: hand mask − winter (winter value) |
|---|---|
| V opening | R01N06 +53° (20°), R01N40 +43° (6°), R14N30 −43° (126°), R01N41 +30° (25°) |
| Left bend | R01N40 +57° (−13°), R17N02 −40° (−2°), R02N20 −40° (9°) |
| Right bend | R14N15 +48° (−52°), R01N43 +45° (−19°), R02N27 −40° (0°) |

8.9% of trees differ by more than 10° in the V opening. The fitted lines on
chosen trees can be drawn with
`python -m treeskel.shape data/raw --trees R01N06 R01N40 R14N30 R14N15 --figure outputs/shape_outliers.jpg`.

## 7. Reproduce

```bat
python -m treeskel.shape data/raw --figure docs/figures/shape_lines.jpg   & rem the student's PC: 2.5 min + 20 s (GPU)
python -m treeskel.shape_model --plots docs/figures                      & rem anywhere: 3 min
python docs/results/shape/robustness.py                                  & rem anywhere: 5 min
```
