# Segmentation trained on the hand masks: first run (masks as drawn)

Session 11. `python -m treeskel.segment data/raw` on the student's PC (GPU, 17.5 min): a U-Net with
a ResNet-18 encoder (ImageNet weights), trained on the 449 development trees' hand masks as drawn,
5 row-grouped folds with the epochs chosen on an inner fold (18–28). Each predicted mask is measured
like the hand masks and scored against the winter traces. Full output: `report.json`; per tree:
`predictions.csv`, `mask_scores.csv`; masks: `predicted_masks.npz`; figures:
`docs/figures/segment_{skill,training,examples}.*`.

## Results

Skill over the mean shape (5 folds, 449 trees; 14 masks with no fork scored as the mean shape):

| Measure | Hand masks | Angle CNN, 240x320 | Angle CNN, 480x640 | U-Net, masks as drawn |
|---|---|---|---|---|
| V opening | +0.87 | +0.16 | +0.18 | +0.06 |
| Left / right bend | +0.70 / +0.61 | +0.07 / +0.05 | +0.06 / +0.05 | +0.02 / +0.03 |
| Trunk from vertical | +0.70 | +0.30 | +0.38 | +0.27 |
| Left / right leader from the trunk | +0.75 / +0.75 | | | +0.03 / +0.19 |

- Tree by tree, the U-Net is no better than the angle CNN on any measure, and worse on several
  (left leader from the trunk: MAE 11.8° → 13.0°, p = 0.008 against the 240x320 run).
- Its V opening is biased: +4.1° wider than winter on average (the hand masks: +0.1°).
- Against the hand masks: median Dice 0.41, IoU 0.25.
- The held-out loss is lowest after about 20 epochs and then rises while the training loss keeps
  falling (`segment_training.png`): the network memorises its training masks from there.

## Why (`diagnostics.py`, output in `diagnostics_output.txt`)

1. **It does not reproduce the hand masks' angles either.** Its V opening correlates with the hand
   masks' at r = 0.43, the same as with winter (0.42), while the hand masks correlate with winter at
   0.93. Its measures spread about two thirds as much as the real ones (V opening SD 7.7° against
   11.6° for the hand masks and 12.3° for winter): it draws a typical V more than it reads the tree.
2. **The hand masks' offsets cost pixel agreement.** On the 21 development trees whose hand masks are
   in the repository, the best shift of each prediction onto its hand mask raises the median Dice
   from 0.39 to 0.57 (an upper estimate: it is the best of 861 shifts). The best shifts reach about
   50 px across and 40 px up or down, in both directions.

Moving a mask does not change its angles, so lining the masks up (the student's next step,
`treeskel.annotate_mask`) mends the pixel labels; whether cleaner labels also help the network read
the angles is what the second run will show.

## The line-up of the hand masks (`lineup.py`, output in `lineup_output.txt`)

The student lined up all 449 development masks with `treeskel.annotate_mask` in 28 minutes (median
2.8 s a tree; none skipped; R01N01 flagged as looking like a different tree, R11N10 as showing no
branch or trunk feature at all).

- **Nearly every mask moved:** 96% by more than 3 px; median 26 px, 29% more than 40 px, range −81
  to +98 px across. Almost all of it is sideways (middle half of the vertical shifts −3 to 0 px), and
  no mask needed scaling.
- **Part of it is by row, most by tree:** the rows' median shifts run from −45 px (row 10) to +56 px
  (row 20), and within a row the shifts still spread by 23–41 px (SD). No correction per row or per
  crop would have done it.
- **Repeatable to about 2 px:** 20 random trees (seed 11) redone from no shift, without the saved
  shifts, differ by a median 1.0 px across and 2.3 px down (the largest difference 7 px); the SD of
  one line-up is 1.6 px across and 2.3 px down, about 1 px of the 256x256 mask, against shifts of
  tens of pixels (r = 0.997 between the two sets across).
- **Agrees with the earlier eye check:** the five trees judged clearly off by eye before the line-up
  (R20N06, R07N29, R01N32, R01N08, R07N32) were moved the way and about as far as judged (R20N06:
  +81 px against about 90).

## Second run: the lined-up masks (`docs/results/segment_lined_up/`)

`python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json` (20.4 min): the same
network, recipe, seed and folds; only the labels moved. Epochs chosen: 32–46 (first run 18–28): with
the labels on the branches the held-out loss keeps falling for longer (0.33–0.39 at its lowest,
against 0.87–0.92).

| Measure (skill over the mean shape) | Hand masks | Angle CNN 480x640 | U-Net, masks as drawn | **U-Net, lined up** | lined up, forks found |
|---|---|---|---|---|---|
| V opening | +0.87 | +0.18 | +0.06 | **+0.72** | +0.78 |
| Left / right bend | +0.70 / +0.61 | +0.06 / +0.05 | +0.02 / +0.03 | **+0.57 / +0.47** | +0.59 / +0.51 |
| Left / right leader from the trunk | +0.75 / +0.75 | | +0.03 / +0.19 | **+0.61 / +0.67** | +0.67 / +0.71 |
| Trunk from vertical | +0.70 | +0.38 | +0.27 | **+0.62** | +0.66 |
| Left / right leader from vertical | +0.80 / +0.73 | +0.24 / +0.15 | −0.08 / +0.08 | **+0.62 / +0.69** | +0.69 / +0.70 |

- **Every measure improves, tree by tree:** against the first run it predicts 65–75% of the trees
  more closely (V opening MAE 9.6° → 4.5°, Wilcoxon p = 1e-36); against the best angle CNN, 61–73%
  (V opening 8.8° → 4.5°, p = 6e-29). All ten measures, every comparison, p < 1e-9.
- **It now reads the tree:** its V opening correlates with winter at r = 0.85 and with the hand
  masks at 0.89 (first run 0.42 and 0.43), and it spreads almost as much as the real V (SD 11.1°
  against 12.4°; first run 7.7°). The V-opening bias of +4.1° is gone (−0.2°).
- **It places the branches where they are in the summer photo:** Dice against the lined-up masks
  0.79 (median; IoU 0.66). Against the masks as drawn its median Dice is 0.22, rising to 0.80 after
  the best shift (`diagnostics_output.txt` in its folder, the 21 local trees): its masks sit where
  the student put the labels, not where the drawn masks were.
- **Still short of the hand masks** (+0.72 against +0.87 on the V opening): 14 masks have no fork
  (scored as the mean shape; without them +0.78), and a person reading the photo still does better
  on every measure. The folds vary (V opening +0.56 to +0.81).
- **The flagged R01N01** (whose mask looks like a different tree) is the lowest Dice in the example
  figure (0.33).

**Why a line-up changed so much.** The offsets (median 26 px) are larger than a leader is wide in
the photo (about 10–20 px), so as drawn, many labels lay beside the branch the photo shows. A network
trained on them cannot tell where to put a leader, and it learned a blurred, typical V. The angles
were never at fault; the labels' positions were, and a translation-free score could not show it.

**What this rests on.** The labels changed, nothing else. The line-up used only the summer photos
(never the winter traces, which stay the independent truth), repeats to about 2 px, and was decided
before this run's result; the two runs are reported together, not the better one alone. The shifts
of the held-out trees never enter their own predictions.
