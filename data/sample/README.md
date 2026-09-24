# Sample data

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
`R04U01_Winter_RGB-R04N03_RGB_12-17-14.png`; the loader accepts both.

Depth images are 8-bit greyscale (three identical channels). They are a scaled
visualisation, not metric distance: brighter = closer, and 0 = no reading.

The full dataset (544 trees, 23 rows) is provided by the MMA3001 teaching team
and is **not** stored in this repository. Place it in `data/raw/` (git-ignored).

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
