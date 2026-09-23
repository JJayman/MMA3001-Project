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

Depth images are 8-bit greyscale (three identical channels). They are a scaled
visualisation, not metric distance: brighter = closer, and 0 = no reading.

The full dataset (544 trees, 23 rows) is provided by the MMA3001 teaching team
and is **not** stored in this repository. Place it in `data/raw/` (git-ignored).
