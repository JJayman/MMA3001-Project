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

- **The trunk's axis** is its top: from the fork down 66 px.
- **The leader lines** run from 25 to 264 px up each leader from the fork in
  the winter photo (1.5 to 16 units of 16.5 px, the median trunk radius), and
  over the same stretch times 1.12 in the summer view (the student's choices,
  below).
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
| Trunk from vertical | 15.0° |
| V's lean from trunk | 16.2° |
| V's lean from vertical | 6.0° |

So angles measured from the trunk mostly measure the trunk's own lean (in the
clicks alone, the trunk's lean and the V's lean from the trunk correlate at
−0.96). Angles from the vertical avoid that, but carry the camera's roll
between the seasons, which differs by row (section 3).

**A fixed stretch of each leader, not all of it in view.** The leaders curve
(the bends), so a line through everything in view depends on where the
photo's top edge cuts the leaders, and the summer photos are framed tighter
than the winter ones. How much a tighter top edge would change the measures
(median / 95th percentile, 449 trees, `framing_change`):

| Height above the fork left in view | Fixed stretch: V opening, left bend, right bend | Whole leader: V opening, left bend, right bend |
|---|---|---|
| 85% | 0.0 / 0.0°, 0.0 / 0.0°, 0.0 / 0.0° | 3.1 / 6.8°, 3.5 / 8.8°, 2.8 / 7.6° |
| 70% | 0.0 / 0.0°, 0.0 / 0.0°, 0.0 / 0.0° | 5.5 / 14.1°, 6.4 / 17.1°, 6.0 / 15.7° |

**Its length in pixels, not in each tree's own trunk radius.** The stretch
was first 1.5–16 trunk radii, so that it would scale with the photo. The
student's check of the trees where the hand masks disagreed most (section 7)
led to a second look, which found three problems:

- **The trunk radius varies a lot.** The winter trunk radius at the fork
  varies by a factor of 2.2 between trees (9.9–21.9 px, 5th to 95th
  percentile), so 16 trunk radii covered anywhere from about 160 to 350 px of
  leader.
- **A hand mask's trunk is only as wide as it was drawn,** from under 0.86 to
  over 3 times the winter one.
- **On curved leaders the length matters.** A stretch 20% shorter or 25%
  longer moves the winter angles by a median 2.6–3.0° (V opening), 4.1–4.5°
  (bends) and 1.7–2.2° (tilts); 95th percentiles 9–14° (449 trees, from the
  saved traces).

A length in pixels varies only with the camera's distance (1.8–2.2 m by Chen
et al., about ±10%). In the summer view it is multiplied by 1.12. The scale
between the seasons is not pinned down (session 7), but two estimates agree:
the hand masks' trunks are drawn a median 1.12 times as wide as the winter
ones, and the student's two overlays gave 1.12 and 1.15. The winter leaders
are at least 362 px long in view (1st percentile), so the stretch is in view
on every tree in both seasons (a median 27.6 and 23.0 units long).

Measured in pixels, the hand masks correlate better with the winter traces on
every measure, and differ less on all but one:

| Hand mask against winter | Median abs. difference, trunk radii → pixels | r, trunk radii → pixels |
|---|---|---|
| V opening | 2.9° → 2.6° | 0.86 → 0.93 |
| Left leader from trunk | 4.9° → 4.7° | 0.86 → 0.88 |
| Right leader from trunk | 4.9° → 4.7° | 0.84 → 0.88 |
| V's lean from trunk | 4.5° → 4.7° | 0.85 → 0.87 |
| Left bend | 4.7° → 4.2° | 0.72 → 0.86 |
| Right bend | 4.8° → 3.8° | 0.67 → 0.80 |
| Trunk from vertical | 4.1° → 4.0° | 0.82 → 0.84 |
| Left leader from vertical | 2.8° → 2.6° | 0.86 → 0.91 |
| Right leader from vertical | 2.9° → 2.5° | 0.81 → 0.87 |
| V's lean from vertical | 2.3° → 2.2° | 0.82 → 0.86 |

The one exception is the median difference for the V's lean from the trunk
(4.5° → 4.7°). The winter V opening's spread across trees falls from 13.7° to
12.5°, as expected if the trunk radii had added variation that is not the
V's.

Both choices were made on grounds other than agreement with the hand masks:
framing, and how much the unit varies. The agreement is shown as a
consequence, not used to tune the stretch; tuning it on agreement would
favour the baseline being scored.

## 2. The winter shape of the development trees

Measured on the winter traces (`data/shape/shape_measures.csv`), with the
provided hand masks' measures against them:

| Measure | Winter mean | SD across trees | Hand mask − winter: mean | Median abs. difference | r |
|---|---|---|---|---|---|
| V opening | 52.8° | 12.5° | +0.1° | 2.6° | 0.93 |
| Left leader from trunk | 21.7° | 17.1° | +2.9° | 4.7° | 0.88 |
| Right leader from trunk | 31.0° | 17.6° | −2.8° | 4.7° | 0.88 |
| V's lean from trunk | 4.7° | 16.2° | −2.9° | 4.7° | 0.87 |
| Left bend | −2.3° | 12.1° | −0.4° | 4.2° | 0.86 |
| Right bend | −5.4° | 11.2° | −0.5° | 3.8° | 0.80 |
| Trunk from vertical | −2.7° | 15.0° | +1.4° | 4.0° | 0.84 |
| Left leader from vertical | −24.5° | 8.8° | −1.5° | 2.6° | 0.91 |
| Right leader from vertical | 28.3° | 8.5° | −1.4° | 2.5° | 0.87 |
| V's lean from vertical | 1.9° | 6.0° | −1.4° | 2.2° | 0.86 |

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
| V opening | 9.8° | 3.3° / +0.87 | **9.2° / +0.10** | 9.3° / +0.11 | 9.3° / +0.10 | 9.3° / +0.09 | 9.2° / +0.10 |
| Left bend | 9.5° | 5.1° / +0.70 | **9.4° / +0.03** | 9.5° / 0.00 | 9.4° / +0.02 | 9.5° / +0.02 | 9.5° / +0.01 |
| Right bend | 8.8° | 5.2° / +0.61 | **8.8° / −0.01** | 8.8° / 0.00 | 8.8° / 0.00 | 8.8° / −0.01 | 8.8° / −0.01 |
| Left leader from trunk | 13.0° | 6.3° / +0.75 | **13.0° / 0.00** | 13.1° / −0.02 | 13.0° / 0.00 | 13.0° / −0.03 | 13.1° / +0.01 |
| Right leader from trunk | 13.8° | 6.4° / +0.75 | **13.7° / +0.03** | 13.5° / +0.03 | 13.6° / +0.03 | 13.6° / +0.03 | 13.3° / +0.06 |
| V's lean from trunk | 12.3° | 6.1° / +0.73 | **12.4° / 0.00** | 12.3° / −0.01 | 12.4° / 0.00 | 12.2° / −0.01 | 12.1° / +0.03 |
| Trunk from vertical | 10.9° | 5.7° / +0.70 | **10.8° / +0.03** | 10.9° / −0.01 | 10.8° / +0.03 | 10.8° / 0.00 | 10.5° / +0.06 |
| Left leader from vertical | 7.1° | 3.1° / +0.80 | **6.7° / +0.09** | 6.9° / +0.05 | 6.7° / +0.08 | 6.8° / +0.06 | 6.9° / +0.05 |
| Right leader from vertical | 6.9° | 3.1° / +0.73 | **6.7° / +0.05** | 6.8° / +0.06 | 6.7° / +0.05 | 6.8° / +0.02 | 6.8° / +0.04 |
| V's lean from vertical | 4.8° | 2.7° / +0.66 | **4.8° / +0.03** | 4.8° / 0.00 | 4.8° / +0.04 | 4.9° / −0.01 | 4.8° / 0.00 |

**V opening, in detail:**
- The model is better than the mean shape on 58% of trees (Wilcoxon
  p = 0.003), r = 0.27. Its skill per fold is −0.02, +0.08, +0.11, +0.12 and
  +0.24.
- The hand masks are better on 78% of trees, r = 0.93, with skill per fold
  +0.80 to +0.91.

With the stretch in trunk radii (the first run, before the switch to pixels)
the scores were lower: V opening +0.74 for the hand masks and +0.06 for the
model, bends +0.40 and +0.36 for the hand masks. The trunk radius had added
variation to the targets that no summer photo can predict.

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
| k nearest neighbours | +0.09 | −0.02 | −0.02 | +0.06 (three measures) |
| Kernel ridge (RBF) | +0.07 | 0.00 | −0.01 | +0.09 (trunk from vertical) |

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
  stable over the folds; for the right bend up to the top of the grid, 10⁷,
  which is the mean shape), so the predictions stay within a few degrees of
  the mean shape (see the scatter figure).

## 5. Fine-tuning the network (the student's GPU)

The student chose to train the network itself on the winter measures, keeping
the hand masks for validation only (`treeskel.shape_cnn`, recipe fixed before
any result):

- **The network:** ResNet-18 from its ImageNet weights, with a linear output
  for all ten measures. It is trained on the summer photo (240x320) against
  the standardised winter measures, and the loss skips a tree's unknown
  measures.
- **Augmentation:** random crops of 80–100% of the photo (no angle changes),
  colour jitter, and the mirror image with the mirrored measures. A
  prediction averages the photo's and its mirror image's.
- **Validation:** the same 5 row-grouped folds. Within each fold's training
  rows, the next fold's rows chose the number of epochs (19, 32, 34, 25 and
  30); the network was then trained again on all the training rows. 6.2
  minutes on the student's GPU.

| Measure | Mean shape (MAE) | Ridge, fixed features | **CNN fine-tuned** | Hand masks |
|---|---|---|---|---|
| V opening | 9.8° | 9.2° / +0.10 | **8.9° / +0.16** | 3.3° / +0.87 |
| Left bend | 9.5° | 9.4° / +0.03 | **9.1° / +0.07** | 5.1° / +0.70 |
| Right bend | 8.8° | 8.8° / −0.01 | **8.6° / +0.05** | 5.2° / +0.61 |
| Left leader from trunk | 13.0° | 13.0° / 0.00 | **11.8° / +0.15** | 6.3° / +0.75 |
| Right leader from trunk | 13.8° | 13.7° / +0.03 | **11.3° / +0.28** | 6.4° / +0.75 |
| V's lean from trunk | 12.3° | 12.4° / 0.00 | **10.6° / +0.22** | 6.1° / +0.73 |
| Trunk from vertical | 10.9° | 10.8° / +0.03 | **8.8° / +0.30** | 5.7° / +0.70 |
| Left leader from vertical | 7.1° | 6.7° / +0.09 | **6.3° / +0.18** | 3.1° / +0.80 |
| Right leader from vertical | 6.9° | 6.7° / +0.05 | **6.4° / +0.13** | 3.1° / +0.73 |
| V's lean from vertical | 4.8° | 4.8° / +0.03 | **4.5° / +0.14** | 2.7° / +0.66 |

(MAE / skill over the mean shape; `docs/results/shape_cnn/`.)

- **Fine-tuning beats the fixed features on every measure,** and the mean
  shape significantly on eight of ten (Wilcoxon p from 2.5×10⁻¹² to 0.003),
  all but the bends (p = 0.11 and 0.34).
- **It reads the trunk best.** The trunk's tilt reaches +0.30 (r = 0.54,
  every fold +0.18 to +0.37), and the angles measured from the trunk +0.15 to
  +0.28. The trunk shows at the bottom of the summer photos, and the angles
  from it are mostly its own lean (section 1).
- **The V itself improves less:** the V opening +0.16 (r = 0.40; folds +0.08,
  +0.21, 0.00, +0.14, +0.33), the leaders' tilts +0.13 and +0.18, the bends
  +0.07 and +0.05.
- **It fits its training photos far better than new rows**
  (`docs/figures/shape_cnn_training.png`). The training loss falls from about
  1.0 to 0.12, while the held-out rows' loss falls a little (e.g. 1.13 to
  0.82) and then stays flat. It does not rise, so choosing the epochs works,
  but the gap stays. One row group is much harder: rows 1, 8 and 15, held out
  within fold 5, stay at 1.3 against 0.7–0.8.

`docs/figures/shape_cnn_skill.png` compares the three with the range over the
folds.

## 6. What this means

- **The information is in the summer photo.** A person's reading of it (the
  hand masks) predicts the winter V opening to 3.3° (MAE) against 9.8° for
  the mean shape (skill +0.87), and the other measures with skill 0.61–0.80.
- **Fixed ImageNet features do not capture it,** whatever the regression on
  top: skill at most about 0.1.
- **Fine-tuning the network on the winter measures captures some of it:**
  the trunk well (+0.30), the V less (V opening +0.16), the bends hardly. It
  fits its 270–360 training photos far better than new rows.
- **Still open:**
  1. more input for the same network: the summer depth, or the photo at full
     size (the student's next choice);
  2. a segmentation network trained on the hand masks, which session 10 showed
     are drawn in the summer view and have the winter shape (448 of 448 laid
     onto the winter traces within 8 px). Its predicted masks would be measured
     with the same code and scored against the winter traces. This would change
     "hand masks are for validation only", so it is the student's decision.

## 7. The trees where the hand masks and the winter traces disagree most

The 14 trees with the largest disagreements (the four largest in each of the
V opening, the bends and the leaders' tilts, with the stretch then in trunk
radii) were drawn with their fitted lines in both seasons
(`python -m treeskel.shape data/raw --trees ... --figure outputs/shape_outliers.jpg`).
The student looked at the other 13 on the PC; R09N32, one of the local trees,
was checked by the AI.

**Every winter trace was right (14 of 14).** No disagreement came from the
ground truth. The student's verdicts on the hand masks' centre lines in the
summer photos (12 of the 13; R17N02 has none):

| Verdict | Trees |
|---|---|
| Slightly off the branch | R01N06, R01N40, R01N41 (also a different view), R02N27, R07N15, R11N19, R14N15 |
| Close, but shifted sideways (left, or right on one leader and the trunk) | R01N43, R02N20 (also a different view), R14N30, R15N24 |
| Close | R05N04 |
| A different view of a correct trace (AI's check) | R09N32: the summer camera stood closer and lower, rolled about 7°, so the leader looks flatter near the fork |

A sideways shift alone does not change an angle; the known shift of the hand
masks (session 3) is harmless here. What did change the angles was partly
the unit: in these 14 trees the drawn trunk was 0.73–3.06 times the winter
one, which made the stretch in the hand mask up to three times too long
(R01N40, R15N24) or a quarter too short (R17N02). That led to the switch to
pixels (section 1).

## 8. Reproduce

```bat
python -m treeskel.shape data/raw --figure docs/figures/shape_lines.jpg   & rem the student's PC: 2.5 min + 20 s (GPU)
python -m treeskel.shape_model --plots docs/figures                      & rem anywhere: 3 min
python docs/results/shape/robustness.py                                  & rem anywhere: 5 min
python -m treeskel.shape_cnn data/raw                                    & rem the student's GPU: 6 min
python -m treeskel.shape_cnn --replot docs/results/shape_cnn             & rem the figures, anywhere
```

- The first command writes `data/shape/`: the table of measures, the summer
  features, and the winter traces (`winter_traces.npz`, `load_traces`), so a
  change to a measure can be tried without the photos. With
  `--features none` it skips the features, which do not depend on the
  measures (how the pixel-unit table was made).
- `python -m treeskel.shape data/raw --trees <ids> --figure outputs/<name>.jpg`
  draws chosen trees at full size into `outputs/`, without touching
  `data/shape/`.
