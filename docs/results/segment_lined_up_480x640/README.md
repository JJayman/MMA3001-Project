# The lined-up U-Net at full size (whole photo, 480x640)

Session 12, the one variant agreed at the end of session 11. `python -m treeskel.segment data/raw
--mask-overlay annotations/mask_overlay.json --size 480 640` on the student's PC (GPU, 69.7 min): the same
network, recipe, seed, folds and lined-up masks as `segment_lined_up`, but the whole summer photo at its own
resolution instead of its centre 480x480 at 256x256. The hand masks cover only the photo's centre 480
columns, so the side columns counted in neither loss term; the predictions were cut back to the masks'
frame before measuring, so both runs are measured alike. Its masks with their pieces joined are in
`../segment_lined_up_480x640_joined/`. Figures: `docs/figures/segment_lined_up_480x640_*`.

## Results

Skill over the mean shape (5 folds, 449 trees; masks with no fork count as the mean shape):

| Measure | 256, lined up | 256, joined | **480x640** | **480x640, joined** |
|---|---|---|---|---|
| V opening | +0.72 | +0.75 | +0.67 | +0.74 |
| Left / right bend | +0.57 / +0.47 | +0.60 / +0.53 | +0.32 / +0.53 | +0.41 / +0.49 |
| Left / right leader from the trunk | +0.61 / +0.67 | +0.64 / +0.72 | +0.57 / +0.66 | +0.65 / +0.69 |
| V's lean from the trunk | +0.63 | +0.68 | +0.61 | +0.66 |
| Trunk from vertical | +0.62 | +0.64 | +0.60 | +0.66 |
| Left / right leader from vertical | +0.62 / +0.69 | +0.68 / +0.66 | +0.53 / +0.63 | +0.59 / +0.64 |
| V's lean from vertical | +0.58 | +0.58 | +0.52 | +0.51 |
| Masks with no fork | 14 | 2 | 29 | 14 |

- **No better on the V opening, tree by tree** (joined against joined, 440 trees): MAE 4.0° → 4.3°,
  51% of trees closer, Wilcoxon p = 0.89. Only the trunk from vertical (6.3° → 5.9°, p = 0.011) and the
  right leader from the trunk (6.9° → 6.8°, p = 0.016) favour the full size; no other measure differs
  at p < 0.05. The angle CNN's full-size run had also read the trunk's tilt better.
- **The same pixels:** median Dice against the lined-up hand masks 0.795 (256: 0.793).
- **More broken masks:** one piece in 303 of 449 masks (256: 348), three or more in 52 (40); with no fork
  29 (14). Joining mends about half (29 → 14, against 14 → 2 at 256).
- **One tree sinks a fold:** fold 1's left bend (−0.38 as predicted, −0.29 joined) is R01N37, whose
  full-size mask lost most of its right leader; the leader finder then took a path that hooks around the
  stub, a left bend of −98° against winter's +22.6° (256: +0.5°). A 120° error dominates the fold's
  squared error.
- **It learns faster:** the folds chose 20–32 epochs (256: 32–46), at a similar lowest held-out loss
  (0.34–0.39).

## The model for the final test

Named by the rule fixed before these results existed (`../final_test/naming_rule.md`, committed at 04:39
UTC; these results at 06:07 UTC): the full-size joined run would be named only if it beat the 256 joined
run tree by tree on the V opening (lower MAE and p < 0.05). It did not (MAE 4.0° → 4.3°, p = 0.89), so the
**256 joined run** is named (`../final_test/model.json`). Both runs are reported here; neither was picked
for having the best number on some measure.
