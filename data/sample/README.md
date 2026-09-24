# Sample data

> **Note:** this sample is from the *earlier* version of the dataset, which
> numbered trees by photo order (`U`, "unmatched"). The project now uses
> "Dataset (With Summer GT)" (`N` numbering, with masks). This sample will be
> replaced with the same trees from that dataset.

A small sample of the orchard RGB-D dataset (8 trees from rows 1, 2, 6 and 17),
committed so that the tests and demos run without the full dataset.

Each tree has four 640x480 PNG files:

| File | Contents |
|---|---|
| `R01U01_Summer_RGB-12-40-11.png` | Summer colour image |
| `R01U01_Summer_D-12-40-11.png` | Summer depth image |
| `R01U01_Winter_RGB-11-23-20.png` | Winter colour image |
| `R01U01_Winter_D-11-23-20.png` | Winter depth image |

Filename format: `R<row>U<position in row>_<Season>_<RGB|D>-<H>-<M>-<S>.png`,
where `H-M-S` is the capture clock time (not zero-padded).

Row 4 winter files use a variant with an embedded original id, e.g.
`R04U01_Winter_RGB-R04N03_RGB_12-17-14.png` (the depth files also embed
`_RGB_`); the loader accepts both.

**Row 4 is excluded** (31 trees): all of its winter depth images are the same
image repeated, so there is no usable winter depth for that row. Its winter
depth files also embed `_RGB_` in their names, which suggests an export error.
The exclusion is recorded in `treeskel.io.EXCLUDED_ROWS`, leaving 513 trees in
22 rows. `python -m treeskel.splits DIR --check-images` reports groups of
identical images, so a problem like this is caught automatically.

Depth images are 8-bit greyscale (three identical channels). They are a scaled
visualisation, not metric distance: brighter = closer, and 0 = no reading.

The full dataset (544 trees, 23 rows) is provided by the MMA3001 teaching team
and is **not** stored in this repository. Place it in `data/raw/` (git-ignored).

## Dataset (With Summer GT), the version now used

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
  resized to 256x256, as in Chen et al. (2020). To be verified by overlay.
- Masks are used for **validation only**, not training.
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
