# The lined-up U-Net's masks with their pieces joined

Session 12, the first item agreed at the end of session 11. `python -m treeskel.segment --join
docs/results/segment_lined_up` (seconds, on any computer: it needs only the run's saved masks and
predictions, no photos). Output: `output.txt`; per tree: `predictions.csv`, `mask_scores.csv` (pieces
joined, pixels added, fork before and after); masks: `predicted_masks.npz`; figures:
`docs/figures/segment_lined_up_joined_{skill,masks}.png`. Checks: `diagnostics.py`, output in
`diagnostics_output.txt`.

## The rule, fixed before scoring

The measuring (`treeskel.shape.mask_branches`) keeps only a mask's largest piece. A leader cut off by a
few pixels of leaves is dropped, and with it the fork: 14 of the lined-up run's 449 masks gave no shape
at all and counted as the mean shape. 101 of the 449 masks have more than one piece.

`treeskel.segment.join_pieces`, applied to every mask before measuring:

- Starting from the largest piece, the nearest other piece is joined while its nearest pixel lies
  within **19.3 mask px** (36 px of the photo) of the joined mask; a piece may come within reach through
  one joined before it.
- The join is a straight band between the two pieces' centre lines where they come nearest, as wide as
  the narrower branch there. Pixels are only added; a mask in one piece is unchanged.
- **19.3 px is about a trunk's width:** twice the development hand masks' median trunk radius at the
  fork (9.67 mask px, 448 trees). It was fixed from that geometry and committed (4103006) before any
  joined mask was scored against winter.
- The gaps between pieces run on continuously (median 20 px), so there is no natural break to find; a
  limit tuned on the winter traces would have been fitted to them. Half and double the limit are shown
  below only to show how much the result depends on it.

## Results

70 masks changed (85 pieces joined, 6,620 pixels added in all). Masks with no fork: **14 → 2**; masks
whose V opening can be measured: 427 → 442.

Skill over the mean shape (5 folds, 449 trees; masks with no fork count as the mean shape):

| Measure | Hand masks | U-Net, lined up | **Pieces joined** | joined, forks found |
|---|---|---|---|---|
| V opening | +0.87 | +0.72 | **+0.75** | +0.78 |
| Left / right bend | +0.70 / +0.61 | +0.57 / +0.47 | **+0.60 / +0.53** | +0.60 / +0.53 |
| Left / right leader from the trunk | +0.75 / +0.75 | +0.61 / +0.67 | **+0.64 / +0.72** | +0.67 / +0.72 |
| V's lean from the trunk | +0.73 | +0.63 | **+0.68** | +0.69 |
| Trunk from vertical | +0.70 | +0.62 | **+0.64** | +0.66 |
| Left / right leader from vertical | +0.80 / +0.73 | +0.62 / +0.69 | **+0.68 / +0.66** | +0.71 / +0.66 |
| V's lean from vertical | +0.66 | +0.58 | **+0.58** | +0.59 |

- **Better on 8 of 10 measures,** the same on the V's lean from vertical, worse on the right leader from
  vertical (+0.69 → +0.66).
- **Tree by tree, on the 70 masks it changed** (`diagnostics_output.txt`): the V opening moved closer to
  winter on 27 trees and farther on 16 (MAE 6.5° → 4.6°, Wilcoxon p = 0.013); the right leader from the
  trunk 19 closer, 6 farther (p = 0.014); the trunk from vertical 11 closer, 1 farther (p = 0.03). The
  other seven measures move both ways about equally (p 0.11–1.0).
- **The 15 masks whose V opening is measured only after joining:** 10 had no fork and counted as the
  mean shape, 5 had a stub too short for a leader line and were left out of the V opening's score. Most
  land near winter (R02N07: winter 68.5°, the mean shape 51.3°, joined 67.4°); R05N04 does not (79.9°,
  joined 50.2°).
- **Where it goes wrong:** a join can attach a side branch or a shoot to a leader. R01N06's right leader
  took a branch outwards (V opening 58.3° → 74.2°, winter 26.7°; its mask was poor before), R06N12's a
  steeper piece (55.7° → 39.5°, winter 53.8°). R01N06 alone lowers fold 0's V-opening skill from +0.56
  to +0.50.
- **It does not depend much on the limit:**

  | Gap (mask px) | Masks joined | No fork | V opening | Right bend | Right leader from vertical |
  |---|---|---|---|---|---|
  | none | 0 | 14 | +0.720 | +0.470 | +0.687 |
  | 9.65 (half) | 39 | 6 | +0.736 | +0.519 | +0.666 |
  | **19.3** | **70** | **2** | **+0.750** | **+0.527** | **+0.660** |
  | 38.6 (double) | 85 | 1 | +0.756 | +0.423 | +0.623 |

  Doubling it reaches more side branches (the right bend falls to +0.42).
- The pixel scores are those of the masks before joining (the hand masks are not needed for this
  re-score); the joins add 6,620 pixels to about 2.3 million branch pixels.
- **One limit of the leader finding, not of the join:** R17N23's left piece is joined, but it grows
  almost straight up above the trunk's base, so the rule that picks the left leader (a tip left of the
  root, as for the hand masks and winter depth) does not see it.

The mask sheet (`segment_lined_up_joined_masks.png`) shows all 70 changed masks: grey the U-Net's mask,
magenta the joins, cyan the centre lines found, and the fitted lines (yellow trunk, red leaders, blue
halves); the 15 whose V opening became measurable come first, labelled.

**What this rests on.** The rule and its limit were fixed from geometry and committed before the first
joined score; the winter traces decided nothing. Every mask is treated alike. Both runs are reported.
Training runs of `treeskel.segment` now score their masks both ways (`<run>` and `<run>_joined`).
