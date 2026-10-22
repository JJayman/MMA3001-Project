"""Turn one summer RGB-D photo into a 3D model of the tree's trunk and leaders, to open and turn around.

**In one step** (session 17): ``python -m treeskel.model3d build R12N02`` makes
the model of any of the dataset's trees and opens it in the browser, with no
network, GPU or PyTorch: the tree is found in ``data/raw``, ``data/sample`` or
``data/tuning`` (:func:`find_photos`), and its mask is the saved out-of-sample
one (:func:`auto_masks`). A photo file works too (any of a dataset tree's
files, or a summer photo from elsewhere with its depth image beside it), and
on Windows ``model3d.bat`` takes dropped photos or a typed tree id.

The project's pieces, end to end, for a single photo:

1. **The branches.** By default (``--mask auto``) a dataset tree's mask is
   the saved out-of-sample one: the 5-fold run's for a development tree (each
   predicted by a network that never saw the tree's row), the final test's
   own for a test-row tree (:data:`FINAL_TEST_MASKS`), so its model matches
   the reported results. Any other photo, and row 4 (left out of the project,
   so without saved masks), goes to the network (:func:`predict_mask`): the
   named U-Net (256x256, ResNet-18 encoder, trained on the 449 development
   trees' lined-up hand masks with the final test's recipe, saved once by
   ``train``) predicts the trunk and leaders from the summer photo, its
   mirror image averaged in, and the mask's pieces are joined
   (:func:`treeskel.segment.join_pieces`). That network was trained on every
   development tree, so on one of them (``--mask unet``) it shows what it
   learned, not how well it predicts (``build`` says so); ``--mask hand``
   gives the lined-up hand mask. ``train`` on a GPU makes a second sample of
   the recipe, not the final test's network bit for bit: complete masks
   agree within about a degree, but a short predicted leader can come out
   complete, or the other way.
2. **The skeleton** (:func:`treeskel.shape.mask_branches`): trunk, left and
   right centre lines, and each point's half-width from the mask.
3. **3D** (:mod:`treeskel.tree3d`): the summer depth along the skeleton gives
   the tree's plane; every skeleton point is moved along its ray onto the
   plane, in metres (:mod:`treeskel.depth3d`: zero 300, focal length 616 px,
   :data:`treeskel.depth3d.METRES_PER_UNIT`).
4. **The model** (:func:`tube`): a tube around each smoothed centre line, as
   wide as the mask, with a ball at the fork; and the summer depth as a cloud
   of coloured points within :data:`SLAB_M` of the tree's plane (the leaves,
   which hide the branches).
   By default (since session 15; ``--flat`` for the plane alone) each point
   then moves along its ray to the bend :mod:`treeskel.bend3d` fits to the
   summer depth where the wood is a depth ridge (the trunk below the canopy,
   wood against a gap), joined at the fork. The fit's prior has the typical
   winter shape as its mean (:func:`typical_prior`: adopted in session 15 by
   the rule fixed before its 449-tree run; ``--plane-prior`` uses the plane),
   so sections whose depth was not seen follow the typical winter tree, not
   this one, and are drawn pale (:data:`UNSURE`). Summer sees almost none of
   the leaders (3% of their readings), so their bends are mostly that shape.
   On the 449 development trees the summer trunk bends the right way more
   often than not, but by about a third of winter's angle.
5. **Files:** ``<name>.html`` opens in any browser, with no internet and
   nothing to install (drag to turn, scroll to zoom, the leaves can be faded
   or hidden); ``<name>_branches.ply`` and ``<name>_leaves.ply`` open in
   Blender, MeshLab or Windows 3D Viewer.

The model is in metres, the camera's frame turned so that ``y`` is up and
``z`` points back at the camera, with the fork at the origin: seen from the
front it looks like the photo. Sizes rest on the assumed 2 m from camera to
fork, and the leaders end where the photo or the prediction ends.

Usage, from the command line:

```
python -m treeskel.model3d build R12N02             # any dataset tree: outputs/model3d/R12N02.html, opened
python -m treeskel.model3d build R05N12 R12N02 --no-open        # several; only write the files
python -m treeskel.model3d build path/to/R05N12_Summer_RGB-14-2-41.png   # any of a tree's files will do
python -m treeskel.model3d train data/raw            # once, on the GPU: models/unet_named.pt (for new photos)
python -m treeskel.model3d build path/to/my_tree_RGB.png        # a new photo, its depth my_tree_D.png beside it
python -m treeskel.model3d build --rgb photo.png --depth depth.png --name my_tree
python -m treeskel.model3d build R05N12 --mask unet  # the trained network on a development tree (in-sample)
python -m treeskel.model3d build R05N12 --flat       # every point on the plane
```
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from treeskel.depth3d import METRES_PER_UNIT, Camera
from treeskel.io import TreeRecord
from treeskel.tree3d import PARTS, depth_along, fit_plane, on_plane, tree_measures

WEIGHTS = Path("models/unet_named.pt")
"""Where ``train`` saves the network and ``build`` finds it (git-ignored)."""

SAVED_MASKS = Path("docs/results/segment_lined_up_joined/predicted_masks.npz")
"""The named run's development masks (5 folds, pieces joined)."""

FINAL_TEST_MASKS = Path("docs/results/final_test/predicted_masks.npz")
"""The final test's masks of the test rows (the evaluated network's, pieces joined)."""

MODEL_FILE = Path("docs/results/final_test/model.json")
"""The file naming the model, whose recipe ``train`` follows."""

SLAB_M = 0.75
"""Leaves are summer depth points within this many metres of the tree's plane (the rows behind are not)."""

STEP_M = 0.01
"""Spacing of the tube's rings along a branch (m)."""

SMOOTH_M = 0.03
"""The centre lines are smoothed over about this length (m): the 256x256 mask's skeleton steps in whole
pixels (about 6 mm each at 2 m)."""

MIN_RADIUS_M = 0.004
"""The thinnest branch drawn (m)."""

BARK = (0.56, 0.42, 0.30)
"""The branches' colour (RGB, 0–1)."""

UNSURE = (0.80, 0.76, 0.70)
"""The colour of branch sections whose depth was not seen (the bends): they follow the prior's mean (the typical
winter shape, or the plane with ``--plane-prior``), so their shape is uncertain."""

TYPICAL_REPORT = Path("docs/results/bend3d_typical/report.json")
"""The typical-shape run of :mod:`treeskel.bend3d` (``--typical``): ``build`` uses its shape as the prior's mean
only if its rule passed (``rule.adopt``)."""

TYPICAL_SHAPE = Path("docs/results/bend3d_typical/typical_shape.json")
"""The typical winter shape that run learnt from every development tree."""

BENDS_REPORT = Path("docs/results/bend3d/report.json")
"""The 449-tree run of :mod:`treeskel.bend3d`, whose chosen prior ``build`` bends with (τ 2 cm, L 20 cm)."""


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

def train(data_dirs: Sequence[str], weights: Path = WEIGHTS, device: str | None = None,
          random_weights: bool = False, epochs: int | None = None, split: str = "splits.json",
          model_file: Path = MODEL_FILE, progress=print) -> dict:
    """Train the named U-Net on every development tree, as the final test did, and save it.

    The recipe (network, lined-up masks, epochs: the median the 5 folds chose, seed) is the named run's
    (:func:`treeskel.final_test.named_recipe`), so the saved network is the final test's network again
    (up to the GPU's non-determinism). No choice is made here.

    Args:
        data_dirs: Dataset folders holding the development trees.
        weights: Where the network is saved (``models/``, git-ignored).
        device: ``"cuda"`` or ``"cpu"``; the GPU if one is available by default.
        random_weights: Start the encoder from random weights instead of ImageNet's (for tests; needs no
            download).
        epochs: Epochs to train; the recipe's by default.
        split: The fixed split (its test rows are never loaded).
        model_file: The file naming the model (the recipe).
        progress: Prints a progress line.

    Returns:
        What was saved besides the weights (the recipe, epochs, trees, minutes).
    """
    import torch

    from treeskel.final_test import SEED_OFFSET, named_recipe
    from treeskel.io import index_dataset
    from treeskel.segment import UNet, load_frames, train_seg
    from treeskel.summer_gt import load_overlays

    recipe = named_recipe(model_file)
    params = recipe["params"]
    test_rows = set(json.loads(Path(split).read_text())["test_rows"])
    records = {t.tree_id: t for d in data_dirs for t in index_dataset(d).trees if t.row not in test_rows}
    overlays = load_overlays(recipe["mask_overlay"]) if recipe["mask_overlay"] else None
    ids = sorted(records)
    progress(f"Loading {len(ids)} development photos and their lined-up hand masks...")
    images, masks, has = load_frames(records, ids, params.size, params.depth, lambda m: None, overlays)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(params.seed)
    model = UNet(params.encoder, not random_weights, channels=images.shape[3]).to(device)
    x = torch.from_numpy(np.ascontiguousarray(images[has].transpose(0, 3, 1, 2))).to(device)
    m = torch.from_numpy(np.ascontiguousarray(masks[has])).to(device)
    n_epochs = int(epochs or recipe["epochs"])
    progress(f"Training on {int(has.sum())} trees for {n_epochs} epochs on {device}...")
    t0 = time.time()
    train_seg(model, x, m, n_epochs, params, params.seed + SEED_OFFSET,
              on_epoch=lambda r: progress(f"  epoch {r['epoch']}: loss {r['train_loss']:.4f}")
              if r["epoch"] % 5 == 0 or r["epoch"] == n_epochs else None)
    meta = {"params": asdict(params), "epochs": n_epochs, "join_gap": recipe["join_gap"], "run": recipe["run"],
            "n_trees": int(has.sum()), "minutes": round((time.time() - t0) / 60, 2), "device": device,
            "seed": params.seed + SEED_OFFSET, "random_weights": bool(random_weights),
            "trees": [t for t, h in zip(ids, has) if h]}      # so build can tell a tree it was trained on
    weights = Path(weights)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, weights)
    progress(f"Saved {weights} ({meta['minutes']} min of training)")
    return meta


def trained_trees(meta: dict, split: str | Path = "splits.json", model_file: Path = MODEL_FILE) -> set[str]:
    """The trees the saved network was trained on.

    ``train`` records them; for a network saved before it did, they are found as ``train`` chose them:
    every tree outside the test rows whose hand mask is lined up (and not skipped) in the recipe's
    line-up file.

    Args:
        meta: What ``train`` saved besides the weights (:func:`load_network`'s third value).
        split: The split file, for the test rows (older networks only).
        model_file: The file naming the model, for the line-up file (older networks only).

    Returns:
        The tree ids.
    """
    if "trees" in meta:
        return set(meta["trees"])
    from treeskel.final_test import named_recipe
    from treeskel.summer_gt import load_overlays

    test_rows = set(json.loads(Path(split).read_text())["test_rows"])
    line_up = named_recipe(model_file)["mask_overlay"]
    overlays = load_overlays(line_up) if line_up else {}
    return {t for t, o in overlays.items() if not o.skip and int(t[1:3]) not in test_rows}


def saved_source(path: str | Path) -> str:
    """The viewer's caption for masks read from ``path`` (``build --mask saved``).

    Args:
        path: The folder of saved masks.

    Returns:
        The caption.
    """
    path = Path(path)
    if path.resolve() == SAVED_MASKS.resolve():
        return "Branches from the 5-fold development run (a network that never saw this row)"
    if path.resolve() == FINAL_TEST_MASKS.resolve():
        return "Branches from the final test's network (trained on the development rows only)"
    return f"Branches from the saved masks in {path.as_posix()}"


def load_network(weights: Path = WEIGHTS, device: str | None = None):
    """The saved network (no download: the weights replace ImageNet's), its device and what ``train`` saved.

    Args:
        weights: The file :func:`train` saved.
        device: ``"cuda"`` or ``"cpu"``; the GPU if one is available by default.

    Returns:
        ``(model, device, meta)``: the U-Net in evaluation mode, its device, and what :func:`train` saved
        (with ``params``, the recipe's :class:`~treeskel.segment.SegParams`).
    """
    import torch

    from treeskel.segment import SegParams, UNet

    saved = torch.load(Path(weights), map_location="cpu", weights_only=False)
    known = {f.name for f in fields(SegParams)}
    p = {k: v for k, v in saved["params"].items() if k in known}
    if isinstance(p.get("size"), list):
        p["size"] = tuple(p["size"])
    params = SegParams(**p)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(params.encoder, pretrained=False, channels=4 if params.depth else 3)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), device, {**saved, "params": params}


def predict_mask(model, device: str, rgb: np.ndarray, params, join_gap: float | None) -> np.ndarray:
    """The network's 256x256 branch mask for one 640x480 summer photo (the hand masks' frame: the centre
    480x480), its mirror image averaged in, its pieces joined.

    Args:
        model: The U-Net from :func:`load_network`.
        device: Its device.
        rgb: ``(480, 640, 3)`` uint8 summer photo.
        params: The recipe's :class:`~treeskel.segment.SegParams`.
        join_gap: Largest gap (mask px) between pieces that are joined; None to leave them.

    Returns:
        The 256x256 boolean mask.

    Raises:
        ValueError: If the network is not a square 256x256 one.
    """
    import torch
    from PIL import Image

    from treeskel.io import CROP_COLS
    from treeskel.segment import join_pieces, predict_masks

    size = params.size if isinstance(params.size, int) else None
    if size is None:
        raise ValueError("model3d expects the named square network (size 256)")
    crop = np.ascontiguousarray(rgb[:, CROP_COLS[0]:CROP_COLS[1]])
    x = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
    prob = predict_masks(model, torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device))[0]
    mask = prob > params.threshold
    if size != 256:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((256, 256), Image.BILINEAR)) > 127
    return join_pieces(mask, join_gap)[0] if join_gap else mask


# --------------------------------------------------------------------------
# The tree in 3D
# --------------------------------------------------------------------------

def smooth_line(points: np.ndarray, step: float = STEP_M, smooth: float = SMOOTH_M) -> tuple[np.ndarray, np.ndarray]:
    """A 3D polyline resampled every ``step`` along its length and smoothed over about ``smooth``.

    Args:
        points: ``(N, 3)`` points along the line (m).
        step: Spacing of the resampled points (m).
        smooth: Smoothing length (m).

    Returns:
        ``(points, s)``: the resampled, smoothed points and their distance along the original line. The
        first and last points stay where they were (so the leaders still meet the trunk at the fork).

    Example:
        >>> p, s = smooth_line(np.array([[0.0, 0, 0], [0.1, 0, 0]]))
        >>> len(p), p[0].tolist(), p[-1].tolist()
        (11, [0.0, 0.0, 0.0], [0.1, 0.0, 0.0])
    """
    p = np.asarray(points, float).reshape(-1, 3)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(seg) > 0]
    p, seg = p[keep], seg[keep]
    if len(p) < 2:
        return p, seg
    s = np.linspace(0.0, seg[-1], max(2, int(np.ceil(seg[-1] / step)) + 1))
    q = np.stack([np.interp(s, seg, p[:, k]) for k in range(3)], 1)
    sigma = smooth / step / 2
    if sigma > 0 and len(q) > 3:
        sm = ndi.gaussian_filter1d(q, sigma, axis=0, mode="nearest")
        # Pull the ends back where they were, fading over the smoothing length, and pin them exactly.
        d0, d1 = q[0] - sm[0], q[-1] - sm[-1]
        sm += d0[None] * np.exp(-s / smooth)[:, None] + d1[None] * np.exp(-(s[-1] - s) / smooth)[:, None]
        sm[0], sm[-1] = q[0], q[-1]
        q = sm
    return q, s


def frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangents and two normals along a line, turning as little as possible (parallel transport), so a
    tube built on them does not twist.

    Args:
        points: ``(N, 3)`` points along the line.

    Returns:
        ``(tangents, normals, binormals)``, each ``(N, 3)`` unit vectors.
    """
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0.0, 1, 0])
    n = np.zeros_like(p)
    n[0] = np.cross(t[0], ref)
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(p)):
        v = n[i - 1] - (n[i - 1] @ t[i]) * t[i]
        norm = np.linalg.norm(v)
        n[i] = v / norm if norm > 1e-12 else n[i - 1]
    return t, n, np.cross(t, n)


def tube(points: np.ndarray, radii: np.ndarray, sides: int = 16, caps: bool = True) -> dict:
    """A closed tube of the given radii around a line: ``{"vertices", "normals", "faces"}``.

    Args:
        points: ``(N, 3)`` centre-line points.
        radii: ``(N,)`` radius at each point.
        sides: Vertices around each ring.
        caps: Close the ends.

    Returns:
        The mesh.

    Example:
        >>> m = tube(np.array([[0.0, 0, 0], [0, 0, 1], [0, 0, 2]]), np.array([0.5, 0.5, 0.5]), sides=8)
        >>> m["vertices"].shape, m["faces"].shape
        ((26, 3), (48, 3))
    """
    p = np.asarray(points, float)
    r = np.broadcast_to(np.asarray(radii, float), (len(p),))
    t, n, b = frames(p)
    ang = 2 * np.pi * np.arange(sides) / sides
    ring = np.cos(ang)[None, :, None] * n[:, None] + np.sin(ang)[None, :, None] * b[:, None]   # (N, sides, 3)
    verts = (p[:, None] + r[:, None, None] * ring).reshape(-1, 3)
    norms = ring.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(len(p) - 1), np.arange(sides), indexing="ij")
    a, bb = i * sides + j, i * sides + (j + 1) % sides
    c, d = a + sides, bb + sides
    faces = np.concatenate([np.stack([a, bb, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    if caps:
        k = len(verts)
        verts = np.vstack([verts, p[0], p[-1]])
        norms = np.vstack([norms, -t[0], t[-1]])
        ring0, ring1 = np.arange(sides), (len(p) - 1) * sides + np.arange(sides)
        faces = np.vstack([faces, np.stack([np.full(sides, k), np.roll(ring0, -1), ring0], 1),
                           np.stack([np.full(sides, k + 1), ring1, np.roll(ring1, -1)], 1)])
    return {"vertices": verts, "normals": norms, "faces": faces.astype(np.int64)}


def ball(centre: np.ndarray, radius: float, rings: int = 10, sides: int = 16) -> dict:
    """A ball (UV sphere): ``{"vertices", "normals", "faces"}``.

    Args:
        centre: ``(3,)`` centre.
        radius: Radius.
        rings: Rings from pole to pole.
        sides: Vertices around each ring.

    Returns:
        The mesh.
    """
    th = np.linspace(0, np.pi, rings + 1)
    ph = 2 * np.pi * np.arange(sides) / sides
    n = np.stack([np.sin(th)[:, None] * np.cos(ph)[None], np.cos(th)[:, None] * np.ones_like(ph)[None],
                  np.sin(th)[:, None] * np.sin(ph)[None]], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(rings), np.arange(sides), indexing="ij")
    a, b = i * sides + j, i * sides + (j + 1) % sides
    faces = np.concatenate([np.stack([a, a + sides, b + sides], -1).reshape(-1, 3),
                            np.stack([a, b + sides, b], -1).reshape(-1, 3)])
    return {"vertices": np.asarray(centre)[None] + radius * n, "normals": n, "faces": faces}


def merge(meshes: Sequence[dict]) -> dict:
    """Several meshes as one (with per-vertex ``colours`` and ``bend`` if every mesh has them).

    Args:
        meshes: Meshes as made by :func:`tube` and :func:`ball`.

    Returns:
        The merged mesh.
    """
    keys = ["vertices", "normals", "faces"] + [k for k in ("colours", "bend") if all(k in m for m in meshes)]
    out = {key: [] for key in keys}
    k = 0
    for m in meshes:
        for key in keys:
            out[key].append(m[key] + k if key == "faces" else m[key])
        k += len(m["vertices"])
    return {key: np.concatenate(v) for key, v in out.items()}


def tube_colours(colours: np.ndarray, sides: int = 16, caps: bool = True) -> np.ndarray:
    """Per-vertex colours of a :func:`tube` from one colour per centre-line point (its rings and caps).

    Args:
        colours: ``(N, 3)`` colour per centre-line point.
        sides: As for :func:`tube`.
        caps: As for :func:`tube`.

    Returns:
        ``(V, 3)`` colour per vertex of the tube.
    """
    c = np.repeat(np.asarray(colours, float).reshape(-1, 3), sides, axis=0)
    return np.vstack([c, colours[:1], colours[-1:]]) if caps else c


def bend_params(report: Path = BENDS_REPORT):
    """The prior ``build`` bends with: the one :mod:`treeskel.bend3d`'s 449-tree run chose, if saved, else the
    default of :class:`treeskel.bend3d.BendParams`.

    Args:
        report: The 449-tree run's report (``docs/results/bend3d/report.json``).

    Returns:
        A :class:`treeskel.bend3d.BendParams`.
    """
    from treeskel.bend3d import BendParams, params_of

    if Path(report).exists():
        return params_of(json.loads(Path(report).read_text())["chosen"])
    return BendParams()


def typical_prior(report: Path = TYPICAL_REPORT, shape: Path = TYPICAL_SHAPE) -> dict | None:
    """The typical winter shape to use as the prior's mean, if :mod:`treeskel.bend3d`'s typical-shape run is
    saved and its rule (fixed before the run) passed; otherwise None (the plane).

    Args:
        report: The typical-shape run's report, with the rule's verdict.
        shape: The saved typical shape (``typical_shape.json``).

    Returns:
        The typical shape (:func:`treeskel.bend3d.typical_shape`), or None.
    """
    if not (Path(report).exists() and Path(shape).exists()):
        return None
    if not json.loads(Path(report).read_text()).get("rule", {}).get("adopt"):
        return None
    return json.loads(Path(shape).read_text())["all"]


VIEW_FLIP = np.array([1.0, -1.0, -1.0])
"""The camera frame's axes as the model's (:func:`to_view`), for vectors."""


def to_view(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Camera-frame points (``y`` down, ``z`` away) as the model's: ``y`` up, ``z`` towards the camera,
    ``origin`` (the fork) at 0.

    Args:
        points: ``(N, 3)`` camera-frame points (m).
        origin: ``(3,)`` the fork, camera frame.

    Returns:
        ``(N, 3)`` points in the model's frame (m).
    """
    q = np.asarray(points, float) - origin
    return q * np.array([1.0, -1.0, -1.0])


def tree_model(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, mask: np.ndarray, camera: Camera = Camera(),
               per_unit: float = METRES_PER_UNIT, slab: float = SLAB_M, leaf_step: int = 2, bends=None,
               typical: dict | None = None) -> dict:
    """The 3D model of one summer photo and its 256x256 branch mask.

    Args:
        rgb: The ``(480, 640, 3)`` uint8 summer photo.
        depth: Its ``(480, 640)`` uint8 depth.
        valid: Boolean mask of the depth's readings.
        mask: The 256x256 branch mask.
        camera: The depth map and camera.
        per_unit: Metres per depth unit.
        slab: Leaves within this many metres of the tree's plane are kept.
        leaf_step: Every this many pixels become a leaf point.
        bends: A :class:`treeskel.bend3d.BendParams` to let the branches bend out of the plane where the summer
            depth sees the wood (ridge readings, :func:`treeskel.bend3d.prepare`); None keeps them on the plane.
            Sections whose bend is uncertain (few or no readings) stay near the prior's mean and are drawn in
            :data:`UNSURE`.
        typical: With ``bends``, the typical winter shape (:func:`treeskel.bend3d.typical_shape`) as the prior's
            mean instead of the plane, so unread sections take the typical shape (:func:`typical_prior`).

    Returns:
        ``{"branches": mesh, "leaves": {"points", "colours"}, "measures", "plane"}``; the mesh (with per-vertex
        ``colours``, and ``bend``: each vertex's move out of the plane, for the viewer's exaggeration) and points
        in metres in the model's frame (:func:`to_view`), the measures as
        :func:`treeskel.tree3d.tree_measures` with sizes in metres, and with ``bends`` those of
        :func:`treeskel.bend3d.bend_measures` under ``"bends"``.

    Raises:
        ValueError: If the mask has no fork, or too few skeleton points have depth to place the plane.
    """
    from treeskel.io import IMAGE_SHAPE, MASK_SHAPE
    from treeskel.shape import SUMMER_SCALE, mask_branches, mask_to_image

    br = mask_branches(np.asarray(mask, bool))
    if br is None:
        raise ValueError("no fork found in the branch mask: the trunk and two leaders could not be picked out")
    k = IMAGE_SHAPE[0] / MASK_SHAPE[0]
    half = ndi.distance_transform_edt(ndi.binary_fill_holes(np.asarray(mask, bool)))
    paths = {p: mask_to_image(br[p]) for p in PARTS}
    widths = {p: np.maximum(half[br[p][:, 0], br[p][:, 1]], 0.5) * k for p in PARTS}       # photo px
    values = {p: depth_along(depth, valid, paths[p]) for p in PARTS}
    rc = np.concatenate([paths[p] for p in PARTS])
    v = np.concatenate([values[p] for p in PARTS])
    ok = np.isfinite(v)
    if ok.sum() < 20:
        raise ValueError(f"only {int(ok.sum())} skeleton points have depth: the tree's plane cannot be placed")
    plane = fit_plane(camera.points(rc[ok], v[ok]))
    measures = tree_measures(paths, depth, valid, camera, br["fork_radius"] * k, SUMMER_SCALE, values=values)
    for key in ("flatness_rms", "flatness_median", "trunk_diameter", "left_length", "right_length", "fork_distance"):
        measures[key] *= per_unit

    fork = on_plane(camera.rays(paths["trunk"][-1:]), plane)[0] * per_unit
    centres = {p: on_plane(camera.rays(paths[p]), plane) * per_unit for p in PARTS}       # metres
    unsure = {p: np.zeros(len(paths[p]), bool) for p in PARTS}
    moves = {p: np.zeros((len(paths[p]), 3)) for p in PARTS}          # each point's move out of the plane (m)
    if bends is not None:
        from treeskel.bend3d import bend_measures, bent_points, fit_bends, posterior_sd, prepare, prior_mean

        # Each skeleton point moves along its ray to its fitted bend; the summer depth counts only where the
        # wood is a ridge, so sections inside the canopy keep near the prior's mean (the plane, or the typical
        # winter shape), with a large uncertainty.
        prep = prepare(paths, depth, valid, widths, camera, ridge=True, scale=per_unit)
        fit = fit_bends(prep, bends, mean=prior_mean(prep, typical) if typical else None)
        sd = posterior_sd(fit)
        pts = bent_points(prep, fit["h"])
        centres = {p: pts[prep.nodes[p]] for p in PARTS}
        moves = {p: prep.rays[prep.nodes[p]] * fit["h"][prep.nodes[p], None] for p in PARTS}
        unsure = {p: sd[prep.nodes[p]] > bends.uncertain * bends.spread for p in PARTS}
        fork = pts[0]
        measures["bends"] = bend_measures(prep, fit["h"], sd, bends, SUMMER_SCALE, sigma=fit["sigma"])
        measures["prior"] = "the typical winter shape" if typical else "the plane"
    meshes, top_radius = [], None
    for p in PARTS:
        centre = centres[p]
        radius = widths[p] * centre[:, 2] / camera.focal
        line, s = smooth_line(centre)
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
        r = ndi.gaussian_filter1d(np.interp(s, seg, radius), max(1.0, 2 * SMOOTH_M / STEP_M), mode="nearest")
        r = np.maximum(r, MIN_RADIUS_M)
        if p == "trunk":
            top_radius = float(r[-1])
        colour = np.where((np.interp(s, seg, unsure[p].astype(float)) > 0.5)[:, None], UNSURE, BARK)
        # The move out of the plane along the tube, smoothed like its centre line, as a view-frame vector.
        move = np.stack([np.interp(s, seg, moves[p][:, j]) for j in range(3)], 1)
        if len(move) > 3:
            move = ndi.gaussian_filter1d(move, SMOOTH_M / STEP_M / 2, axis=0, mode="nearest")
        meshes.append({**tube(to_view(line, fork), r), "colours": tube_colours(colour),
                       "bend": tube_colours(move * VIEW_FLIP)})
    knot = ball(np.zeros(3), top_radius or MIN_RADIUS_M)
    meshes.append({**knot, "colours": np.tile(UNSURE if unsure["trunk"][-1] else BARK, (len(knot["vertices"]), 1)),
                   "bend": np.tile(moves["trunk"][-1] * VIEW_FLIP, (len(knot["vertices"]), 1))})
    branches = merge(meshes)

    # The leaves: the summer depth near the tree's plane, coloured by the photo.
    rr, cc = np.mgrid[0:depth.shape[0]:leaf_step, 0:depth.shape[1]:leaf_step]
    rr, cc = rr.ravel(), cc.ravel()
    good = valid[rr, cc]
    rr, cc = rr[good], cc[good]
    pts = camera.points(np.stack([rr, cc], 1), depth[rr, cc]) * per_unit
    dist = (pts - plane["centre"] * per_unit) @ plane["normal"]
    near = np.abs(dist) <= slab
    return {"branches": branches, "leaves": {"points": to_view(pts[near], fork), "colours": rgb[rr[near], cc[near]]},
            "measures": measures, "plane": plane}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def write_ply_mesh(path: Path, mesh: dict, colour: tuple[float, float, float] = BARK) -> None:
    """A binary PLY mesh with normals and a colour per vertex (Blender, MeshLab, Windows 3D Viewer): the mesh's
    own ``colours`` if it has them, else ``colour``.

    Args:
        path: The ``.ply`` file.
        mesh: A mesh ``{"vertices", "normals", "faces"}`` (and per-vertex ``colours``, ``bend`` if made with
            them).
        colour: RGB in [0, 1], for a mesh without colours.
    """
    v = np.asarray(mesh["vertices"], "<f4")
    n = np.asarray(mesh["normals"], "<f4")
    rgb = _rgb8(mesh, colour)
    vt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    va = np.empty(len(v), vt)
    for i, k in enumerate(("x", "y", "z")):
        va[k], va["n" + k] = v[:, i], n[:, i]
    va["red"], va["green"], va["blue"] = rgb.T
    ft = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    fa = np.empty(len(mesh["faces"]), ft)
    fa["n"], fa["i"] = 3, mesh["faces"]
    header = ("ply\nformat binary_little_endian 1.0\ncomment treeskel.model3d: trunk and leaders, metres\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(fa)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(va.tobytes())
        f.write(fa.tobytes())


def write_ply_points(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    """A binary PLY point cloud with a colour per point.

    Args:
        path: The ``.ply`` file.
        points: ``(N, 3)`` points.
        colours: ``(N, 3)`` RGB in [0, 1].
    """
    pt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    pa = np.empty(len(points), pt)
    for i, k in enumerate(("x", "y", "z")):
        pa[k] = points[:, i]
    pa["red"], pa["green"], pa["blue"] = np.asarray(colours, np.uint8).T
    header = ("ply\nformat binary_little_endian 1.0\ncomment treeskel.model3d: summer depth near the tree, metres\n"
              f"element vertex {len(pa)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pa.tobytes())


def read_ply(path: Path) -> dict:
    """Read back a PLY written by this module: ``{"vertex": structured array, "face": index array or None}``.

    Args:
        path: A ``.ply`` file from :func:`write_ply_mesh` or :func:`write_ply_points`.

    Returns:
        The vertices and the faces (None for a point cloud).
    """
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    counts = {ln.split()[1]: int(ln.split()[2]) for ln in header if ln.startswith("element")}
    props = [ln.split()[-1] for ln in header if ln.startswith("property") and "list" not in ln]
    types = {"float": "<f4", "uchar": "u1"}
    vt = np.dtype([(name, types[ln.split()[1]]) for name, ln in
                   zip(props, [ln for ln in header if ln.startswith("property") and "list" not in ln])])
    vert = np.frombuffer(data, vt, counts["vertex"], end)
    face = None
    if counts.get("face"):
        ft = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        face = np.frombuffer(data, ft, counts["face"], end + vert.nbytes)["i"]
    return {"vertex": vert, "face": face}


def _rgb8(mesh: dict, colour: tuple[float, float, float] = BARK) -> np.ndarray:
    """A mesh's per-vertex colours as 8-bit RGB (its ``colours`` if it has them, else ``colour`` everywhere)."""
    c = mesh.get("colours")
    c = np.tile(colour, (len(mesh["vertices"]), 1)) if c is None else np.asarray(c, float)
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode("ascii")


def viewer_data(model: dict, title: str, source: str) -> dict:
    """What the viewer page needs: the arrays (base64, little-endian) and the text.

    Args:
        model: From :func:`tree_model`.
        title: The page's title.
        source: Where the branches came from (shown on the page).

    Returns:
        The data, ready for JSON.
    """
    b, leaves = model["branches"], model["leaves"]
    m = model["measures"]

    def num(key, mult=1.0, spec=".1f"):
        """A measure as text (scaled by ``mult``), or ``None`` if missing."""
        x = m.get(key)
        return None if x is None or not np.isfinite(x) else format(x * mult, spec)

    rows = [["V opening in the tree's plane", num("v_opening_3d"), "°"],
            ["V opening in the photo", num("v_opening_2d"), "°"],
            ["Lean towards the camera", num("lean_camera"), "°"],
            ["Trunk diameter at the fork", num("trunk_diameter", 100, ".0f"), "cm"],
            ["Left leader in view", num("left_length", 1, ".2f"), "m"],
            ["Right leader in view", num("right_length", 1, ".2f"), "m"],
            ["Depth from the tree's plane (median)", num("flatness_median", 100), "cm"]]
    bent = m.get("bends")
    if bent:
        def bnum(key, spec=".1f"):
            """A bend measure as text, or ``None`` if missing."""
            x = bent.get(key)
            return None if x is None or not np.isfinite(x) else format(x, spec)

        rows += [["V opening, branches bent", bnum("v_opening"), "°"],
                 ["Trunk out of the V's plane (+: root behind)", bnum("trunk_out"), "°"],
                 ["Trunk's depth seen", bnum("trunk_anchors", ".0%"), ""],
                 ["Leaders' depth seen", None if not np.isfinite(bent.get("left_anchors", np.nan)) else
                  format((bent["left_anchors"] + bent["right_anchors"]) / 2, ".0%"), ""]]
    return {
        "title": title, "source": source, "bends": bool(bent), "prior": m.get("prior", "the plane"),
        "mesh": {"positions": _b64(b["vertices"], "<f4"), "normals": _b64(b["normals"], "<f4"),
                 "indices": _b64(b["faces"].ravel(), "<u4"), "colours": _b64(_rgb8(b), "u1"),
                 "bend": _b64(b.get("bend", np.zeros_like(b["vertices"])), "<f4")},
        "points": {"positions": _b64(leaves["points"], "<f4"), "colours": _b64(leaves["colours"], "u1"),
                   "count": int(len(leaves["points"]))},
        "measures": rows,
    }


def write_html(path: Path, model: dict, title: str, source: str) -> None:
    """The self-contained viewer page (WebGL 2, no libraries, no internet).

    Args:
        path: The ``.html`` file.
        model: From :func:`tree_model`.
        title: The page's title.
        source: Where the branches came from (shown on the page).
    """
    data = json.dumps(viewer_data(model, title, source)).replace("</", "<\\/")
    page = VIEWER.replace("__TITLE__", _html_escape(title)).replace("__DATA__", data)
    Path(path).write_text(page, encoding="utf-8")


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


VIEWER = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #f4f3ef; --panel: rgba(252, 252, 251, 0.92); --ink: #1b1b1a; --muted: #5f5e59; --line: #d9d7cf; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #151614; --panel: rgba(30, 31, 29, 0.92); --ink: #f1efe8; --muted: #b3b0a6; --line: #3a3a36; }
  }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
  canvas { display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #panel { position: absolute; top: 12px; left: 12px; width: 300px; max-width: calc(100vw - 48px);
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  h1 { font-size: 15px; margin: 0 0 2px; }
  .muted { color: var(--muted); }
  table { border-collapse: collapse; margin: 8px 0 6px; width: 100%; }
  td { padding: 1px 0; vertical-align: top; } td.v { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; padding-left: 10px; }
  label { display: flex; align-items: center; gap: 6px; margin: 4px 0; cursor: pointer; }
  input[type=range] { flex: 1; }
  #help { position: absolute; bottom: 10px; left: 12px; right: 12px; color: var(--muted); font-size: 12px; }
  #toggle { position: absolute; top: 12px; right: 12px; background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; cursor: pointer; font: inherit; }
  .note { font-size: 11.5px; margin-top: 6px; }
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="panel">
  <h1 id="title"></h1>
  <div class="muted" id="source"></div>
  <table id="measures"></table>
  <label><input type="checkbox" id="showBranches" checked> Branches (predicted)</label>
  <label><input type="checkbox" id="showLeaves" checked> Leaves (summer depth)</label>
  <label>Leaf opacity <input type="range" id="leafAlpha" min="0.05" max="1" step="0.05" value="1"></label>
  <label>Point size <input type="range" id="pointSize" min="0.3" max="3" step="0.1" value="1"></label>
  <label><input type="checkbox" id="spin"> Turn slowly</label>
  <label id="exaggerateRow" hidden>Exaggerate bends <input type="range" id="exaggerate" min="1" max="5" step="0.5"
    value="1"> <span id="exaggerateValue">×1.0</span></label>
  <div class="muted note">Sizes assume the fork is about 2 m from the camera; the leaders end where the photo
    or the prediction ends. Seen from the front, the model looks like the photo.</div>
  <div class="muted note" id="bendNote" hidden>The branches bend out of the tree's plane where the summer depth
    sees the wood. Pale sections: their depth was not seen (leaves in front), so they follow the prior (<span
    id="priorName">the plane</span>) and their bend is uncertain. "Exaggerate bends" stretches every bend for
    display only; the numbers and the .ply files are at true scale.</div>
</div>
<button id="toggle" type="button">Hide panel</button>
<div id="help">Drag to turn · scroll or pinch to zoom · right-drag or shift-drag to move · double-click to reset</div>
<script>
"use strict";
const DATA = __DATA__;

function decode(b64, Type) {
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

document.getElementById("title").textContent = DATA.title;
document.getElementById("source").textContent = DATA.source;
const table = document.getElementById("measures");
for (const [name, value, unit] of DATA.measures) {
  if (value === null) continue;
  const tr = table.insertRow();
  tr.insertCell().textContent = name;
  const td = tr.insertCell(); td.className = "v"; td.textContent = unit ? value + " " + unit : value;
}
if (DATA.bends) {
  document.getElementById("bendNote").hidden = false;
  document.getElementById("exaggerateRow").hidden = false;
  document.getElementById("priorName").textContent = DATA.prior;
}
document.getElementById("toggle").onclick = (e) => {
  const p = document.getElementById("panel"), hidden = p.style.display === "none";
  p.style.display = hidden ? "" : "none"; e.target.textContent = hidden ? "Hide panel" : "Show panel";
};

const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl2", { antialias: true });
if (!gl) {
  document.getElementById("help").textContent = "This viewer needs WebGL 2 (any recent Chrome, Edge, Firefox or Safari).";
  throw new Error("no WebGL 2");
}

function compile(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const meshProg = program(`#version 300 es
  in vec3 position; in vec3 normal; in vec3 colour; in vec3 bend; uniform mat4 mvp; uniform mat4 view;
  uniform float exaggerate; out vec3 vNormal; out vec3 vColour;
  void main() { gl_Position = mvp * vec4(position + (exaggerate - 1.0) * bend, 1.0); vNormal = mat3(view) * normal;
                vColour = colour; }`,
`#version 300 es
  precision highp float; in vec3 vNormal; in vec3 vColour; out vec4 outColour;
  void main() { float d = abs(normalize(vNormal).z); outColour = vec4(vColour * (0.38 + 0.62 * d), 1.0); }`);
const pointProg = program(`#version 300 es
  in vec3 position; in vec3 colour; uniform mat4 mvp; uniform float size; out vec3 vColour;
  void main() { gl_Position = mvp * vec4(position, 1.0); gl_PointSize = max(1.0, size / gl_Position.w); vColour = colour; }`,
`#version 300 es
  precision highp float; in vec3 vColour; uniform float alpha; out vec4 outColour;
  void main() { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; outColour = vec4(vColour, alpha); }`);

function buffer(data, attr, prog, size, type, normalised) {
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, attr); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, type, normalised, 0, 0);
}
const meshVao = gl.createVertexArray(); gl.bindVertexArray(meshVao);
const meshPos = decode(DATA.mesh.positions, Float32Array);
buffer(meshPos, "position", meshProg, 3, gl.FLOAT, false);
buffer(decode(DATA.mesh.normals, Float32Array), "normal", meshProg, 3, gl.FLOAT, false);
buffer(decode(DATA.mesh.colours, Uint8Array), "colour", meshProg, 3, gl.UNSIGNED_BYTE, true);
buffer(decode(DATA.mesh.bend, Float32Array), "bend", meshProg, 3, gl.FLOAT, false);
const indices = decode(DATA.mesh.indices, Uint32Array);
const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
const pointVao = gl.createVertexArray(); gl.bindVertexArray(pointVao);
buffer(decode(DATA.points.positions, Float32Array), "position", pointProg, 3, gl.FLOAT, false);
buffer(decode(DATA.points.colours, Uint8Array), "colour", pointProg, 3, gl.UNSIGNED_BYTE, true);
gl.bindVertexArray(null);

// The camera orbits a target: the middle of the branches.
let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < meshPos.length; i += 3) for (let k = 0; k < 3; k++) {
  lo[k] = Math.min(lo[k], meshPos[i + k]); hi[k] = Math.max(hi[k], meshPos[i + k]);
}
const home = { target: [0, (lo[1] + hi[1]) / 2, 0], yaw: 0, pitch: 0,
               dist: 1.25 * Math.max(hi[1] - lo[1], hi[0] - lo[0]) / (2 * Math.tan(Math.PI / 8)) };
let cam = JSON.parse(JSON.stringify(home));

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function multiply(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalise(a) { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function eyePosition() {
  const c = Math.cos(cam.pitch);
  return [cam.target[0] + cam.dist * c * Math.sin(cam.yaw), cam.target[1] + cam.dist * Math.sin(cam.pitch),
          cam.target[2] + cam.dist * c * Math.cos(cam.yaw)];
}
function lookAt(eye, target) {
  const z = normalise(sub(eye, target)), x = normalise(cross([0, 1, 0], z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

const ui = { branches: document.getElementById("showBranches"), leaves: document.getElementById("showLeaves"),
             alpha: document.getElementById("leafAlpha"), size: document.getElementById("pointSize"),
             spin: document.getElementById("spin"), exaggerate: document.getElementById("exaggerate") };
ui.exaggerate.addEventListener("input", () => {
  document.getElementById("exaggerateValue").textContent = "×" + parseFloat(ui.exaggerate.value).toFixed(1);
});
let dirty = true;
for (const el of Object.values(ui)) el.addEventListener("input", () => { dirty = true; });

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  const bg = getComputedStyle(document.body).backgroundColor.match(/\d+/g).map(Number);
  gl.clearColor(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const fov = Math.PI / 4, view = lookAt(eyePosition(), cam.target);
  const mvp = multiply(perspective(fov, w / h, 0.01, 50), view);
  if (ui.branches.checked) {
    gl.useProgram(meshProg); gl.bindVertexArray(meshVao);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, "mvp"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, "view"), false, view);
    gl.uniform1f(gl.getUniformLocation(meshProg, "exaggerate"), parseFloat(ui.exaggerate.value));
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
  }
  if (ui.leaves.checked) {
    const alpha = parseFloat(ui.alpha.value);
    gl.useProgram(pointProg); gl.bindVertexArray(pointVao);
    gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "mvp"), false, mvp);
    // A point about 7 mm across, so neighbouring depth readings touch at 2 m.
    gl.uniform1f(gl.getUniformLocation(pointProg, "size"), parseFloat(ui.size.value) * 0.007 * h / (2 * Math.tan(fov / 2)));
    gl.uniform1f(gl.getUniformLocation(pointProg, "alpha"), alpha);
    if (alpha < 1) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    gl.drawArrays(gl.POINTS, 0, DATA.points.count);
    gl.disable(gl.BLEND); gl.depthMask(true);
  }
  gl.bindVertexArray(null);
}

// Mouse, touch and wheel.
let drag = null;
const pointers = new Map();
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
  drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (pointers.size === 2) {                                   // pinch to zoom
    const prev = [...pointers.values()], d0 = Math.hypot(prev[0][0] - prev[1][0], prev[0][1] - prev[1][1]);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    const now = [...pointers.values()], d1 = Math.hypot(now[0][0] - now[1][0], now[0][1] - now[1][1]);
    if (d0 > 0 && d1 > 0) cam.dist *= d0 / d1;
    dirty = true; return;
  }
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const eye = eyePosition(), z = normalise(sub(eye, cam.target)), x = normalise(cross([0, 1, 0], z)), y = cross(z, x);
    const k = cam.dist * 0.0015;
    for (let i = 0; i < 3; i++) cam.target[i] += -dx * k * x[i] + dy * k * y[i];
  } else {
    cam.yaw -= dx * 0.008;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.008));
  }
  dirty = true;
});
const release = (e) => { pointers.delete(e.pointerId); if (pointers.size === 0) drag = null; };
canvas.addEventListener("pointerup", release);
canvas.addEventListener("pointercancel", release);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault(); cam.dist *= Math.exp(e.deltaY * 0.001); dirty = true;
}, { passive: false });
canvas.addEventListener("dblclick", () => { cam = JSON.parse(JSON.stringify(home)); dirty = true; });
window.addEventListener("resize", () => { dirty = true; });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { dirty = true; });

function frame() {
  if (ui.spin.checked && !drag) { cam.yaw += 0.004; dirty = true; }
  if (dirty) { draw(); dirty = false; }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

DATA_DIRS = ("data/raw", "data/sample", "data/tuning")
"""Where ``build`` looks for a tree id when ``--data`` is not given, in this order (the folders that exist)."""

_TREE_ID = re.compile(r"R(\d{1,2})([NU])(\d{1,2})", re.IGNORECASE)

_DEPTH_NAMES = (("RGB", "D"), ("rgb", "d"), ("RGB", "Depth"), ("rgb", "depth"),
                ("Color", "Depth"), ("color", "depth"), ("colour", "depth"))
"""How a depth image's name is made from its colour photo's (the last match in the name is replaced)."""


@dataclass
class Photo:
    """One summer photo for ``build``: a dataset tree's (``record``) or a photo from outside the dataset.

    Attributes:
        name: The tree id, or the photo's file name without its extension; names the output files.
        record: The tree's files, for a tree of the dataset.
        rgb: The colour photo, for a photo from outside the dataset.
        depth: Its 8-bit depth image.
    """

    name: str
    record: TreeRecord | None = None
    rgb: Path | None = None
    depth: Path | None = None


def tree_id(text: str) -> str | None:
    """``text`` as a tree id in the dataset's form, or None if it is not one.

    Args:
        text: Any text, e.g. a command-line argument.

    Returns:
        The id, e.g. ``"R12N02"``, or None.

    Example:
        >>> tree_id("r12n2"), tree_id("R05N12"), tree_id("photo.png")
        ('R12N02', 'R05N12', None)
    """
    m = _TREE_ID.fullmatch(text.strip())
    return f"R{int(m[1]):02d}{m[2].upper()}{int(m[3]):02d}" if m else None


def depth_for(photo: str | Path) -> Path | None:
    """The depth image beside a colour photo, found by its name, or None.

    The last ``RGB`` in the name becomes ``D`` (the dataset's own naming: ``R12N02_Summer_RGB-13-45-33.png``
    and ``R12N02_Summer_D-13-45-33.png``), or ``Depth``; ``color`` or ``colour`` becomes ``depth``.

    Args:
        photo: The colour photo's path.

    Returns:
        The depth image's path, or None if none exists.
    """
    photo = Path(photo)
    for old, new in _DEPTH_NAMES:
        i = photo.name.rfind(old)
        if i >= 0:
            candidate = photo.with_name(photo.name[:i] + new + photo.name[i + len(old):])
            if candidate != photo and candidate.is_file():
                return candidate
    return None


def _dataset_tree(path: Path) -> str | None:
    """The tree id in a file name that follows the dataset's naming, or None."""
    from treeskel.io import parse_filename

    try:
        return parse_filename(path).tree_id
    except ValueError:
        return None


def _summer_record(files: Sequence[Path], tree: str) -> TreeRecord | None:
    """The tree's record from dataset-named files, if they hold its summer photo and depth."""
    from treeskel.io import parse_filename

    rec = None
    for f in sorted(files):
        if _dataset_tree(f) != tree:
            continue
        info = parse_filename(f)
        rec = rec or TreeRecord(tree, info.row, info.position)
        rec.paths.setdefault((info.season, info.modality), f)
    return rec if rec and {("summer", "rgb"), ("summer", "depth")} <= rec.paths.keys() else None


def find_photos(items: Sequence[str], data_dirs: Sequence[str] | None = None,
                depth: str | None = None) -> tuple[list[Photo], list[str]]:
    """Turn what ``build`` was given (tree ids, or files, e.g. dropped on ``model3d.bat``) into the photos to build.

    - A **tree id** (``R12N02``, in any case) is looked up in ``data_dirs`` (default: those of
      :data:`DATA_DIRS` that exist; the first folder holding the tree wins). Row 4 counts too: its summer
      photos are sound, only its winter depth is a copy.
    - A **file named like the dataset's** (any of a tree's files: its summer or winter photo, depth or mask)
      stands for that tree's summer photo, from the same folder, or else looked up as a tree id.
    - **Any other file** is a summer photo from outside the dataset. Its depth is ``depth`` (if one such photo
      is given) or the image beside it named like it (:func:`depth_for`). A file that is another given
      photo's depth is not built on its own, so a photo can be dropped together with its depth.

    Args:
        items: Tree ids and file paths, in the order given.
        data_dirs: Dataset folders for tree ids.
        depth: The depth image of the one photo from outside the dataset.

    Returns:
        The photos, each tree once and in the order given, and a message for each item that could not be used.
    """
    from treeskel.io import index_dataset

    dirs = [Path(d) for d in data_dirs] if data_dirs else [Path(d) for d in DATA_DIRS if Path(d).is_dir()]
    known: dict[str, TreeRecord] = {}

    def lookup(tree: str) -> TreeRecord | None:
        """The tree's record in ``dirs`` (indexed at the first call)."""
        if not known:
            for d in dirs:
                if d.is_dir():
                    idx = index_dataset(d, exclude_rows=(), exclude_trees=())
                    for r in (*idx.trees, *idx.incomplete):               # only the summer files are needed
                        if {("summer", "rgb"), ("summer", "depth")} <= r.paths.keys():
                            known.setdefault(r.tree_id, r)
        return known.get(tree)

    outside = [Path(i) for i in items if tree_id(i) is None and Path(i).is_file() and _dataset_tree(Path(i)) is None]
    their_depths = {depth_for(o) for o in outside} - {None}
    photos: list[Photo] = []
    problems: list[str] = []
    for item in items:
        path, tree = Path(item), tree_id(item)
        if tree is not None:
            rec = lookup(tree)
            if rec is None:
                where = ", ".join(d.as_posix() for d in dirs) or "no data folder (none of " + ", ".join(DATA_DIRS) + ")"
                problems.append(f"{item}: no summer photo of this tree in {where}")
                continue
        elif path in outside:
            if path in their_depths or any(p.rgb == path for p in photos):
                continue                                       # dropped together with its photo, or twice
            d = Path(depth) if depth and len(outside) == 1 else depth_for(path)
            if d is None:
                problems.append(f"{path.name}: no depth image found beside it (name it like the photo with RGB "
                                "replaced by D, or give --depth)")
            else:
                photos.append(Photo(path.stem, rgb=path, depth=d))
            continue
        elif path.is_file():                                   # one of a dataset tree's files
            tree = _dataset_tree(path)
            # Its summer files beside it, or (a dataset kept in season folders) wherever the data folders hold them.
            rec = _summer_record([f for f in path.parent.iterdir() if f.suffix.lower() == ".png"], tree) or lookup(tree)
            if rec is None:
                problems.append(f"{path.name}: no summer photo and depth of tree {tree} beside it or in the data folders")
                continue
        else:
            problems.append(f"{item}: neither a tree id (like R12N02) nor a file")
            continue
        if all(p.name != tree for p in photos):
            photos.append(Photo(tree, record=rec))
    return photos, problems


def auto_masks(row: int, test_rows: set[int]) -> Path:
    """Where the saved out-of-sample mask of a dataset tree in ``row`` is (``build --mask auto``).

    A test-row tree's is the final test's (:data:`FINAL_TEST_MASKS`), so its model matches the reported
    results; a development tree's is the 5-fold run's (:data:`SAVED_MASKS`), from a network that never saw
    its row. Either way no trained network is needed. ``train``'s network learned every development tree, so
    on one of them it would show what it learned, not how well it predicts (session 14).

    Args:
        row: The tree's orchard row.
        test_rows: The test rows.

    Returns:
        The folder of saved masks.
    """
    return FINAL_TEST_MASKS if row in test_rows else SAVED_MASKS


def build(name: str, rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, mask: np.ndarray, out: Path,
          title: str, source: str, bends=None, typical: dict | None = None) -> dict:
    """Make one tree's model and write its files; returns the paths and the measures.

    ``bends`` (a :class:`treeskel.bend3d.BendParams`) lets the branches bend out of the plane, and ``typical``
    (a typical winter shape) is then the prior's mean (:func:`tree_model`).

    Args:
        name: The tree's name (the files' stem).
        rgb: ``(480, 640, 3)`` uint8 summer photo.
        depth: Its ``(480, 640)`` uint8 depth.
        valid: Boolean mask of the depth's readings.
        mask: 256x256 boolean branch mask.
        out: Output folder.
        title: The page's title.
        source: Where the branches came from (shown on the page).
        bends: The prior of the bends; None for the plane.
        typical: The typical winter shape as the prior's mean; None for the plane.

    Returns:
        ``{"files", "measures", "n_points", "n_triangles"}``: the paths of the ``html``, ``branches`` and
        ``leaves`` files, the tree's measures, and the sizes of the cloud and the mesh.
    """
    model = tree_model(rgb, depth, valid, mask, bends=bends, typical=typical)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{name}.html", "branches": out / f"{name}_branches.ply", "leaves": out / f"{name}_leaves.ply"}
    write_html(paths["html"], model, title, source)
    write_ply_mesh(paths["branches"], model["branches"])
    write_ply_points(paths["leaves"], model["leaves"]["points"], model["leaves"]["colours"])
    return {"files": paths, "measures": model["measures"], "n_points": len(model["leaves"]["points"]),
            "n_triangles": len(model["branches"]["faces"])}


def main(argv: Sequence[str] | None = None) -> int:
    """``train`` the network once, or ``build`` models from summer photos.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 if every model was made (or the network trained), 1 if an input could not be used,
        the split file is missing, the network is needed but not trained, or a model could not be made (each
        is reported).
    """
    from treeskel.io import load_depth, load_rgb, load_tree

    ap = argparse.ArgumentParser(prog="python -m treeskel.model3d", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train the named U-Net on all development trees and save it (GPU, ~2 min)")
    tr.add_argument("data_dirs", nargs="+")
    tr.add_argument("--weights", default=str(WEIGHTS))
    tr.add_argument("--device")
    tr.add_argument("--split", default="splits.json")
    tr.add_argument("--epochs", type=int, help=argparse.SUPPRESS)                     # tests only
    tr.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)  # tests only
    bu = sub.add_parser("build", help="make the 3D model of summer photos",
                        description="Make the 3D model of trees (by id, e.g. R12N02) or of summer photos (files), "
                                    "and open it in the browser.")
    bu.add_argument("items", nargs="*", metavar="TREE_OR_PHOTO",
                    help="tree ids (looked up in --data) or photo files: any of a dataset tree's files, or a summer "
                         "photo from elsewhere with its depth image beside it (RGB in the name replaced by D)")
    bu.add_argument("--tree", nargs="+", default=[], help="tree ids, as above (the form before session 17)")
    bu.add_argument("--data", nargs="+",
                    help="dataset folder(s) for tree ids (default: those of " + ", ".join(DATA_DIRS) + " that exist)")
    bu.add_argument("--rgb", help="a summer colour photo (640x480 PNG), with --depth")
    bu.add_argument("--depth", help="the depth image (640x480 8-bit PNG) of --rgb, or of the one photo given")
    bu.add_argument("--name", help="file name for --rgb (default: the photo's)")
    bu.add_argument("--mask", choices=("auto", "unet", "saved", "hand"), default="auto",
                    help="auto (default): a dataset tree's saved out-of-sample mask (5-fold for a development row, the "
                         "final test's for a test row), the trained network for any other photo; unet: the trained "
                         "network (it was trained on every development tree); saved: masks saved by a run (--saved); "
                         "hand: the lined-up hand mask")
    bu.add_argument("--weights", default=str(WEIGHTS))
    bu.add_argument("--saved", default=str(SAVED_MASKS),
                    help=f"masks for --mask saved: the 5-fold development masks (default), or {FINAL_TEST_MASKS} "
                         "for the test rows (the final test's own masks)")
    bu.add_argument("--split", default="splits.json", help=argparse.SUPPRESS)          # tests only
    bu.add_argument("--mask-overlay", default="annotations/mask_overlay.json", help="the hand masks' line-up")
    bu.add_argument("--device")
    shape = bu.add_mutually_exclusive_group()
    shape.add_argument("--bends", dest="flat", action="store_false",
                       help="(the default) let the branches bend out of the tree's plane where the summer depth sees "
                            "the wood (treeskel.bend3d); sections it does not see follow the typical winter shape "
                            "(the plane with --plane-prior), drawn pale")
    shape.add_argument("--flat", dest="flat", action="store_true",
                       help="put every skeleton point on the tree's plane (the model before session 15)")
    bu.set_defaults(flat=False)                  # both options share ``flat``: --bends' own default would be True
    bu.add_argument("--plane-prior", action="store_true",
                    help="keep the plane as the bends' prior mean even if the typical winter shape was adopted")
    bu.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                    help="open each model in the browser once it is made (default; --no-open to only write the files)")
    bu.add_argument("--out", default="outputs/model3d")
    args = ap.parse_args(argv)

    if args.command == "train":
        train(args.data_dirs, Path(args.weights), args.device, args.random_weights, args.epochs, args.split)
        return 0

    items = [*args.items, *args.tree]
    if not items and not args.rgb:
        # --data takes several folders, so a tree id written after it is read as one.
        ids = [d for d in args.data or [] if tree_id(d)]
        ap.error("give a tree id (like R12N02) or a summer photo"
                 + (f"; put {' '.join(ids)} before --data, which read it as a folder" if ids else ""))
    if args.rgb and not args.depth:
        ap.error("--rgb needs its --depth image")
    if args.rgb and args.mask not in ("auto", "unet"):
        ap.error("a photo outside the dataset needs the trained network (--mask auto or unet)")

    photos, problems = find_photos(items, args.data, None if args.rgb else args.depth)
    if args.rgb:
        photos.append(Photo(args.name or Path(args.rgb).stem, rgb=Path(args.rgb), depth=Path(args.depth)))
    for msg in problems:
        print(msg)
    if not photos:
        return 1

    # Each photo's masks, and the network only if one needs it (no PyTorch, GPU or training otherwise).
    test_rows: set[int] = set()
    if args.mask == "auto" and any(p.record is not None for p in photos):
        if not Path(args.split).exists():
            print(f"No {args.split} here: run from the repository's folder (or model3d.bat)")
            return 1
        test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])
    loaded: dict[Path, dict] = {}

    def saved_masks(path: Path) -> dict:
        """A saved run's masks, read once."""
        if path not in loaded:
            from treeskel.segment import load_masks
            loaded[path] = load_masks(path) if path.exists() else {}
        return loaded[path]

    network: list = []                                     # [model, device, meta, trained trees] once loaded

    def get_network():
        """The trained network, loaded at its first use; None if it was never trained."""
        if not network:
            if not Path(args.weights).exists():
                return None
            network.extend(load_network(Path(args.weights), args.device))
            meta = network[2]
            network.append(trained_trees(meta, args.split) if any(p.record is not None for p in photos) else set())
            print(f"Network: {args.weights} (trained on {meta['n_trees']} trees, {meta['epochs']} epochs)")
        return network

    overlays = None
    if args.mask == "hand":
        from treeskel.summer_gt import load_overlays
        overlays = load_overlays(args.mask_overlay)

    bends = typical = None
    if not args.flat:
        # Bending by default is the student's decision after the 449-tree run (rule 3 in docs/results/bend3d/).
        from treeskel.bend3d import setting_key

        bends = bend_params()
        print(f"Bends out of the plane: prior {setting_key(bends)} "
              + ("(chosen by the 449-tree run)" if BENDS_REPORT.exists() else "(provisional: no 449-tree run saved)")
              + "; --flat for every point on the plane")
        typical = None if args.plane_prior else typical_prior(TYPICAL_REPORT, TYPICAL_SHAPE)
        if typical is not None:
            print("Where summer does not see the wood, the branches take the typical winter shape (adopted by the rule "
                  f"in {TYPICAL_REPORT.parent.as_posix()}; --plane-prior for the plane)")
        elif not args.plane_prior:
            print("Where summer does not see the wood, the branches stay near the plane (the typical winter shape "
                  + ("was not adopted by its rule)" if TYPICAL_REPORT.exists() else "is not learnt yet)"))
    status = 0
    told_to_train = False
    for photo in photos:
        name, rec = photo.name, photo.record
        if rec is not None:
            s = load_tree(rec, "summer")
            rgb, depth, valid, title = s["rgb"], s["depth"], s["valid"], f"Tree {name}, summer"
        else:
            depth, valid = load_depth(photo.depth)
            rgb, title = load_rgb(photo.rgb), photo.rgb.name
        mask = None
        use_network = args.mask == "unet" or (args.mask == "auto" and rec is None)
        if args.mask == "auto" and rec is not None:
            where = auto_masks(rec.row, test_rows)
            if name in saved_masks(where):
                mask, source = saved_masks(where)[name], saved_source(where)
                print(f"{name}: " + ("a test-row tree: the final test's mask" if rec.row in test_rows
                                     else "a development tree: its 5-fold mask, from a network that never saw its row"))
            else:
                # Row 4 (left out of the project) has none, nor has a tree from a folder of new photos.
                from treeskel.io import EXCLUDED_ROWS
                why = (f" (row {rec.row} is left out of the project)" if rec.row in EXCLUDED_ROWS else
                       "" if where.exists() else f" ({where.as_posix()} not found: run from the repository's folder)")
                print(f"{name}: no saved mask{why}, so the trained network predicts it")
                use_network = True
        if use_network:
            net = get_network()
            if net is None:
                print(f"{name}: needs the trained network, and there is none at {args.weights}")
                if not told_to_train:
                    print("  Train it once: python -m treeskel.model3d train data/raw (GPU, about 2 min). The dataset's "
                          "trees need no network: give their ids (they use the saved masks).")
                    told_to_train = True
                status = 1
                continue
            model, device, meta, trained = net
            mask = predict_mask(model, device, rgb, meta["params"], meta.get("join_gap"))
            source = f"Branches predicted by the U-Net trained on {meta['n_trees']} development trees"
            if rec is not None and name in trained:
                # The network learned this tree's own hand mask, so its mask shows what it learned, not how well
                # it predicts a tree it has not seen (session 14: R06N12 read 55 deg here, 39 deg out of fold).
                source += " (this tree among them)"
                print(f"  note: {name} was in the network's training set, so this shows what it learned, not how "
                      "well it predicts; for a prediction from a network that never saw its row, leave out --mask")
        elif args.mask == "saved":
            saved = saved_masks(Path(args.saved))
            if name not in saved:
                hint = (f"test rows have none: use --saved {FINAL_TEST_MASKS.as_posix()}, or leave out --mask"
                        if Path(args.saved).resolve() == SAVED_MASKS.resolve() else "not among them")
                print(f"{name}: no mask in {Path(args.saved).as_posix()} ({hint})")
                status = 1
                continue
            mask, source = saved[name], saved_source(args.saved)
        elif args.mask == "hand":
            from treeskel.segment import hand_mask
            mask = hand_mask(rec, overlays) if rec is not None else None
            if mask is None:
                print(f"{name}: no lined-up hand mask")
                status = 1
                continue
            source = "Branches from the hand-drawn mask, lined up"
        try:
            res = build(name, rgb, depth, valid, mask, Path(args.out), title, source, bends, typical)
        except ValueError as e:
            print(f"{name}: {e}")
            status = 1
            continue
        m = res["measures"]
        print(f"{name}: V opening {m['v_opening_3d']:.1f}° in the tree's plane ({m['v_opening_2d']:.1f}° in the photo), "
              f"lean {m['lean_camera']:.1f}° towards the camera; {res['n_triangles']:,} triangles, "
              f"{res['n_points']:,} leaf points")
        if "bends" in m:
            b = m["bends"]
            print(f"  bent: V opening {b['v_opening']:.1f}°, trunk {b['trunk_out']:.1f}° out of the V's plane; depth "
                  f"seen on {b['trunk_anchors']:.0%} of the trunk, {(b['left_anchors'] + b['right_anchors']) / 2:.0%} "
                  f"of the leaders (the rest follows {m.get('prior', 'the plane')}, drawn pale)")
        html = res["files"]["html"]
        print(f"  {html} (and the .ply files, for Blender, MeshLab or 3D Viewer)")
        if args.open:
            opened = webbrowser.open(html.resolve().as_uri())
            print("  opened in the browser" if opened else "  open it in a browser")
    return 1 if problems else status


if __name__ == "__main__":
    raise SystemExit(main())
