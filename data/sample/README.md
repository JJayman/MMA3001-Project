# Sample data

Nine trees from "Dataset (With Summer GT)", five files each, committed so the
tests and demos run without the full dataset:

| Trees | Why they are included |
|---|---|
| R01N01, R01N02 | Development row |
| R03N33, R03N34 | Test row. R03N34's winter depth is the genuine image that row 4 copies |
| R04N01 | Row 4 (excluded): its winter depth is a copy of R03N34's, and it uses the embedded-id file name |
| R12N02, R12N03 | Test row |
| R15N05 | Development row |
| R20N06 | Short row (6 trees) |

The full dataset is provided by the MMA3001 teaching team and is **not**
stored in this repository. Place it in `data/raw/` (git-ignored).

## Dataset (With Summer GT)

Five files per tree, all in one folder:

| File | Contents |
|---|---|
| `R20N03_Summer_RGB-14-21-52.png` | Summer colour image, 640x480 |
| `R20N03_Summer_D-14-21-52.png` | Summer depth image, 640x480 |
| `R20N03_Summer_GT-14-21-52.png` | Summer main-branch mask (trunk + two V leaders), 256x256 |
| `R20N03_Winter_RGB-13-26-42.png` | Winter colour image, 640x480 |
| `R20N03_Winter_D-13-26-42.png` | Winter depth image, 640x480 |

- 2,810 files (562 trees) in 20 rows.
- R and N follow the orchard map supplied with the dataset: rows run west to
  east, and N counts trees north to south. So `R20N03_Summer` and
  `R20N03_Winter` are the same physical tree.
- Masks are assumed to cover the photo's centre 480x480 (columns 80–559),
  resized to 256x256, as in Chen et al. (2020). **Checked by overlay on this
  sample:** the shape matches, but each mask sits about 15–45 px left of the
  visible trunk, and the shift varies by tree
  ([example](../../docs/figures/mask_alignment_R12N02.jpg)). Stretching the
  mask to the full width fits worse. Pixel-level comparison therefore needs
  a per-tree alignment first.
- Masks were for **validation only** until session 11. Since then the development rows' masks also
  train the segmentation (`treeskel.segment`), which is scored against the winter traces; the test
  rows' masks (rows 3 and 12 here) stay unused until the final evaluation.
- Depth images are 8-bit greyscale (three identical channels): a scaled
  visualisation, not metric distance. Brighter = closer; 0 = no reading.
- Row 4's 21 winter depth images are all copies of R03N34's winter depth
  image (R03N34 was photographed just before row 4). R03N34's image was
  checked against its own RGB image and is genuine, so R03N34 is kept and
  row 4 is excluded. That leaves 541 trees in 19 rows.

## Capture setup

The camera and orchard details come from Chen et al. (2020), who used images of
the same orchard, taken in a different (autumn) capture:

- Camera: Intel RealSense D435, 640x480, with depth aligned to RGB by the camera.
- Orchard: commercial orchard in north-eastern Melbourne, Ruby Pink apples,
  2D open-V (Tatura trellis). Rows 4.125 m apart centre to centre, trees 0.6 m
  apart in a row, rows planted east to west.
- Camera to canopy distance: 1.8–2.2 m.
- Chen et al. analysed depth below a 4 m maximum. The 8-bit values in this
  dataset are consistent with value ≈ 255·(1 − d / 4 m), but this is a hypothesis
  still to be tested, not a documented mapping.

Our summer/winter captures are **not** the autumn images used in the paper:
our summer images contain apples, and the capture times (about 11:00–14:00)
differ from the paper's (16:00–17:30).

Reference: Chen, Z., Ting, D., Newbury, R., Chen, C. (2020). *Semantic
Segmentation for Partially Occluded Apple Trees Based on Deep Learning.*
arXiv:2010.06879.
