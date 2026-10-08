# The 3D estimate: how the 8-bit depth maps to distance

Session 13. `python -m treeskel.depth3d data/raw --calibrate` on the student's PC (449 development
trees, under a minute). Output: `calibration_output.txt`; per tree: `calibration_trees.csv`; summary:
`calibration.json`. The test rows were not loaded.

## Why the map's zero matters

The depth images hold 8-bit values, not distances. Write the map as **distance = scale × (zero − value)**
(session 2 guessed zero 255 and scale 4 m / 255). A point's sideways position is its distance times its
offset in the image divided by the focal length, so in a ratio of a depth difference to a sideways
distance the scale cancels:

- **angles** (the V opening in the tree's own plane, how far that plane leans) need only the **zero**
  and the focal length;
- **sizes** (leader lengths, the trunk's diameter) need the **scale** too.

## How the zero was measured

Seen from any camera height, pitch or roll, a flat ground has `1 / distance` changing linearly across the
image. So for the right zero, `1 / (zero − value)` is a linear function of the pixel's column and row on
the ground; for a map linear in disparity (what a stereo camera measures) the value itself would be.

- The ground behind each tree is cut into blocks of 16 x 32 px, each summarised by its median value.
- Trees and posts are removed as blocks far from a smooth surface through the rest (a surface that
  assumes no zero; a "nearer than the row's median" rule cut away real ground when the camera rolls,
  which a synthetic test with a known zero caught).
- For each zero from 190 to 5000 the linear model is fitted by least squares; the zero with the smallest
  error, in depth units, is the tree's estimate. Blocks below value 10 (the end of the range, where
  values may be clipped) are left out; keeping them changes nothing (zero 302 against 300).
- Tests on synthetic ground photos with known zeros (255–350, cameras 0.7–1.2 m high, pitched 5–12°,
  rolled ±4°, with noise) recover the zero within 5%.

## Results (449 development trees)

- **The value is linear in distance, not in disparity:** the best zero fits better than the disparity
  map on **98%** of trees (median error 3.2 against 4.5 depth units).
- **The zero is about 300:** per-tree median **300** [quartiles 262, 366]; the zero that fits all trees
  together best is 325. Session 2's 255 lies below the quartiles (median error 3.46 against 3.16).
- **Steady across the orchard except row 1:** the other rows' medians lie between 242 and 344. Row 1, the
  orchard's edge row, gives 381 (pooled 500); the ground behind it is not an alley between two rows.
- 14 trees could not be fitted (12 at either end of the range of zeros tried, 2 with too little ground).

## The choice for the 3D estimate (proposed; the student confirms)

- **Zero 300** (the per-tree median), with the **quartiles 262–366 as the sensitivity range**, and
  session 2's 255 shown beside it.
- **Scale** from the documented 1.8–2.2 m between the camera and the canopy (Chen et al., 2020): the
  median fork of the development trees at 2.0 m, with 1.8 and 2.2 as the range. It affects sizes only.
- **Focal length** 616 px (the D435 colour camera at 640x480), with ±5% as the range.

## Limits

- The ground is not quite flat: the trees stand on a raised strip, and the alley behind may slope. This
  is why the zero varies from tree to tree (quartiles 262–366); the median over 449 trees is steadier
  than any one tree.
- Nothing in the photos fixes the scale; the documented camera distance is the only reference. The
  neighbouring trunks (0.6 m apart in session 2's reading of Chen et al.) mix with posts in the depth and
  could not be used without checking each by hand.
