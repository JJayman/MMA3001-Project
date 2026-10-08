# A summer photo as a 3D model

Session 13, at the student's request: turn one summer RGB-D photo into a 3D model of the tree that can be
opened and turned around. `treeskel.model3d` joins the project's pieces end to end:

1. **Branches:** the named U-Net predicts the trunk and leaders from the summer photo, and the mask's pieces
   are joined.
2. **Skeleton:** the mask is thinned to the trunk and two leaders, with each point's half-width.
3. **3D:** the summer depth along the skeleton places the tree's plane (`treeskel.tree3d`), and every
   skeleton point is moved onto it in metres (zero 300, focal 616 px, 0.01197 m per depth unit;
   `docs/results/depth3d/`, `docs/results/tree3d/`).
4. **Model:** tubes as wide as the mask, a ball at the fork, and the summer depth within 0.75 m of the
   plane as coloured points (the leaves).

`R15N05.html` is the example (a development tree, its mask from the 5-fold run, so from a network that never
saw row 15). Open it in any browser: it needs no internet and nothing installed. Drag to turn, scroll to
zoom, right-drag or shift-drag to move, and fade or hide the leaves. Figure:
`docs/figures/model3d_R15N05.jpg`.

## Making one

```bash
python -m treeskel.model3d train data/raw                       # once (GPU, about 2 min): models/unet_named.pt
python -m treeskel.model3d build --tree R05N12 --data data/raw    # outputs/model3d/R05N12.html and .ply files
python -m treeskel.model3d build --rgb photo.png --depth depth.png --name my_tree
python -m treeskel.model3d build --tree R15N05 --data data/sample --mask saved   # no weights needed
```

- `train` repeats the final test's training: the same recipe, the 449 development trees, 34 epochs and
  the same seed. It saves the network (git-ignored, about 57 MB).
- `build` writes `<name>.html` (the viewer), `<name>_branches.ply` (the tubes) and `<name>_leaves.ply`
  (the points); the `.ply` files open in Blender, MeshLab or Windows 3D Viewer.
- `--mask saved` uses the development masks of the 5-fold run, and `--mask hand` the lined-up hand mask.

## Checks

- The numbers in the viewer are the 3D run's for the same tree: R15N05's V opening of 51.7° in the
  plane and 53.9° in the photo, and its 12.9° lean, match `docs/results/tree3d/trees.csv` (a test checks
  this).
- The tubes are closed surfaces (every edge shared by two triangles), at the mask's radius, on frames that
  do not twist; the `.ply` files read back exactly.
- The viewer was opened in Chromium and turned with the mouse, without errors.

## Limits

- **Sizes** rest on the assumed 2 m from the camera to the fork.
- **The leaders end** where the photo or the prediction ends; a leader the U-Net stops short (7% of trees)
  stops short here too.
- **The trunk's width** is the U-Net's, which learned the hand masks' widths (drawn wider than the bark;
  session 10).
- **The branches lie on one plane:** the depth can place the tree's plane, but not the few centimetres by
  which each branch leaves it.
