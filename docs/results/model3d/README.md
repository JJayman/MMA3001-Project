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
python -m treeskel.model3d build --tree R12N02 --data data/raw    # outputs/model3d/R12N02.html and .ply files
python -m treeskel.model3d build --rgb photo.png --depth depth.png --name my_tree
python -m treeskel.model3d build --tree R15N05 --data data/sample --mask saved   # no weights needed
python -m treeskel.model3d build --tree R12N02 --data data/sample --mask saved --saved docs/results/final_test/predicted_masks.npz
```

- `train` repeats the final test's training: the same recipe, the 449 development trees, 34 epochs and
  the same seed. It saves the network (git-ignored, about 57 MB).
- `build` writes `<name>.html` (the viewer), `<name>_branches.ply` (the tubes) and `<name>_leaves.ply`
  (the points); the `.ply` files open in Blender, MeshLab or Windows 3D Viewer.
- `--mask saved` uses the development masks of the 5-fold run, and `--mask hand` the lined-up hand mask.

### Which masks to show (session 14)

| Tree | Masks | Why |
|---|---|---|
| A development tree | `--mask saved` (the default `--saved`) | Each mask is from a network that never saw the tree's row, so it shows how well the model predicts. |
| A test-row tree, as in the final test | `--mask saved --saved docs/results/final_test/predicted_masks.npz` | The evaluated network's own masks: the 3D model matches the reported results (R12N02: 49.6° in the photo, as in `final_test/predictions.csv`; a test checks this). |
| A test-row tree or a new photo | `--mask unet` (the default) | The trained network has never seen them. The test rows were used once, for the final test, so building them chooses nothing; report them as a display only. |

`--mask unet` on a development tree shows what the network learned, not how well it predicts: `train` trains
on every development tree with its hand mask. On R06N12 it read 55.3° in the photo, against 39.4° from the
5-fold mask and 53.8° in winter. `build` now says so, and the viewer's caption adds "this tree among them";
`train` records its trees (a network saved before that, like the student's, is checked against the
lined-up development trees, which is the same set).

## Checks

- The numbers in the viewer are the 3D run's for the same tree: R15N05's V opening of 51.7° in the
  plane and 53.9° in the photo, and its 12.9° lean, match `docs/results/tree3d/trees.csv` (a test checks
  this).
- The tubes are closed surfaces (every edge shared by two triangles), at the mask's radius, on frames that
  do not twist; the `.ply` files read back exactly.
- The viewer was opened in Chromium and turned with the mouse, without errors.
- **The same masks as the final test** (session 14): for one network, `build`'s prediction (one photo at a
  time) and the final test's (a batch) give the same input image, the same probabilities and the same
  mask, pixel for pixel (checked with a randomly initialised network on the sample's test-row photos).
  On the CPU, the final test's training and `train` give bit-identical weights. `build --mask saved
  --saved docs/results/final_test/predicted_masks.npz` reproduces the final test's numbers.

## Limits

- **Sizes** rest on the assumed 2 m from the camera to the fork.
- **The leaders end** where the photo or the prediction ends; a leader the U-Net stops short (7% of trees)
  stops short here too.
- **A short predicted leader can tip the tree's plane** (session 14). With one leader short, the trunk and
  the other leader lie nearly in a line, which hardly fixes a plane. With the 5-fold masks the plane failed
  (turned more than 20°) on 4 of 442 trees, 3 of them with a short predicted leader (R01N04, R01N40,
  R05N04): 10% of the 31 short-leader trees against 1 of the other 411. R01N04's left leader stops after
  0.32 m, and its V reads 82° in the plane against 45° in the photo, leaning 58° towards the camera. On the
  test row, the retrained network's R03N10 (a short right leader) reads 87° and leans 50°.
- **`train` makes a second sample of the recipe, not the final test's network** (session 14). Training on a
  GPU cannot be repeated exactly (cuDNN), so the network differs a little. Session 12's re-run of the
  same recipe shows how much: on the 393 trees whose masks are complete in both runs the V opening differs
  by a median 0.4° (at most 6°), but on the 49 with a short predicted leader in either run, 27% differ by
  more than 5° (up to 39°), because a leader short in one run is often complete in the other (35–46%).
  The student's network on four test-row trees: R12N02 50.0° against the final test's 49.6°; R03N10 80.2°
  against 63.2° and R16N12 61.6° against 70.4° (both short in the final test); R03N32 found a fork the
  final test's mask lacked (its leader 22 px from the trunk, against the 19.3 px joining gap). Report the
  final test's numbers; use its masks (above) for figures that match them.
- **The trunk's width** is the U-Net's, which learned the hand masks' widths (drawn wider than the bark;
  session 10).
- **The branches lie on one plane:** the depth can place the tree's plane, but not the few centimetres by
  which each branch leaves it.
