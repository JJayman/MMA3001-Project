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
5. **Bends** (session 15, the default since the 449-tree run): each skeleton point moves along its ray to the
   bend `treeskel.bend3d` fits to the summer depth where the wood is seen; unseen sections stay near the
   plane, drawn pale (below). `--flat` keeps every point on the plane, as before.

The examples (development trees, their masks from the 5-fold run, so from networks that never saw their
rows) open in any browser, with no internet and nothing installed: drag to turn, scroll to zoom, right-drag
or shift-drag to move, and fade or hide the leaves.

- `R15N05.html`: the example since session 13, rebuilt with bends; little of its trunk is seen in summer,
  so it takes the typical winter shape, pale. Figure (the flat model of session 13): `docs/figures/model3d_R15N05.jpg`.
- `R01N13.html`: a tree whose summer trunk is seen, chosen by a rule (below). Figure of both, from the front
  and the side: `docs/figures/model3d_bends.png` (`bends_figure.py`).

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
- **The branches lie on one plane** unless `--bends` is given (session 15, below).

## Bends out of the plane (session 15; the default)

```bash
python -m treeskel.model3d build --tree R01N13 --data data/raw --mask saved          # bends (the default)
python -m treeskel.model3d build --tree R01N13 --data data/raw --mask saved --flat   # every point on the plane
python -m treeskel.model3d build --tree R01N13 --data data/raw --mask saved --plane-prior   # bends; unseen parts on the plane
```

Each skeleton point moves along its own ray to the bend that `treeskel.bend3d` fits to the summer depth
(`docs/results/bend3d/`): a smooth curve per part, joined at the fork, with the plane as its prior (τ 2 cm,
L 20 cm, chosen on the 449 development trees' winter depth). Only the summer readings where the branch is a
depth ridge count (its sides at least about 5 cm farther), so the depth of leaves in front is not followed.
**Sections whose depth was not seen are drawn pale** (`UNSURE`): there the bend's posterior standard
deviation is more than half the prior's, and the branch stays near the plane. Seen from the front the model
still looks like the photo; turned side-on, the trunk can run back out of the V's plane. The viewer's panel
adds the bent V opening, the trunk's angle out of the V's plane and how much of the trunk's and the leaders'
depth was seen.

**Why it is the default:** the rule fixed before the 449-tree run (`docs/results/bend3d/README.md`, rule 3)
passed, and the student chose to follow it. With the U-Net's masks the summer trunk's angle out of the V's
plane is closer to winter's than the plane's on 60% of the 433 trees (Wilcoxon p 4e-8).

**What to know when showing one** (449 development trees, `docs/results/bend3d/analysis_output.txt`):

- **Summer sees the trunk below the canopy, almost never the leaders:** a median 45% of the trunk's readings
  count, 3% of the leaders'. The trunk is drawn as seen (less than half of it uncertain) on 214 of 433 U-Net
  masks.
- **The summer bend is about a third of the real one.** The trunk's angle out of the V's plane is a median
  2.9° in summer against 14.2° in winter (Theil–Sen slope 0.32): with few, noisy readings the prior pulls it
  back towards the plane. It usually leans the right way, by too little.
- **Seen trunks are the ones to trust:** where the trunk is drawn as seen, its angle follows winter's with a
  rank correlation of 0.52; where it is pale, 0.25.
- **The leaders' bends are noise:** their bows come out 0.1–0.2 cm further from winter's than the plane's
  (median size 0.7–0.9 cm). They are drawn pale.

**The examples.** `R15N05.html` (the session-13 example, kept): 19% of its trunk read, none of it sure, so the
model follows the typical winter shape: its trunk 8.4° out of the V's plane (winter 6.4°; −2.4° with the plane
as the mean). `R01N13.html`: chosen by a rule, not by eye: among the development trees in the repository whose
U-Net trunk is drawn as seen, the one whose summer-against-winter trunk-angle error with the plane as the mean
(12.2°) is nearest the median of all 214 such trees (11.4°). Its summer trunk is 18.0° out of the V's plane
with the typical shape as the mean (16.3° with the plane; winter 28.5°); 85% of its trunk is read and 71%
drawn as seen.

**Two additions after the student saw nearly straight leaders** (the real bends are a few centimetres over
1.5 m, and summer does not see the leaders, so they followed the flat plane):

- **"Exaggerate bends" (×1–×5) in the viewer's panel:** each vertex moves by up to five times its bend out of
  the plane, for display only; the panel's numbers and the `.ply` files stay at true scale.
- **The typical winter shape as the prior's mean** (`docs/results/bend3d_typical/`): where summer does not see
  the wood, the branches take the typical winter tree's shape instead of the plane. The rule fixed before its
  449-tree run passed (U-Net masks, out of fold, against the plane as the mean: trunk angle 8.5° against
  12.0°, nearer winter on 78% of trees; leader tilts 3.0° against 4.5°, 76%), so `build` uses it (it says so;
  `--plane-prior` keeps the plane). The summer leaders now carry the typical S: towards the camera just above
  the fork, away around 0.8 m, back near the top.

**Checked:** the plane's own measures are unchanged by the bends; the colours reach the viewer and the
`.ply` file; the page opens in headless Chromium without errors, front and side views as expected; the
examples' bent measures equal the 449-tree runs' for the same trees (with the plane as the mean R15N05 −2.4°,
R01N13 16.3°; with the typical shape R01N13 18.0°, and R15N05 8.4° against 8.6°, because `build` uses the
shape learnt from all trees and the run each fold's). A test
caught that `build` had not been bending by default at first (both options shared one setting, whose
default came from `--bends`).
