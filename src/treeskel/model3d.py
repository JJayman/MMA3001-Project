"""Turn one summer RGB-D photo into a 3D model of the tree's trunk and leaders, to open and turn around.

The project's pieces, end to end, for a single photo:

1. **The branches** (:func:`predict_mask`): the named U-Net (256x256, ResNet-18
   encoder, trained on the 449 development trees' lined-up hand masks with the
   final test's recipe, saved once by ``train``) predicts the trunk and
   leaders from the summer photo, its mirror image averaged in, and the
   mask's pieces are joined (:func:`treeskel.segment.join_pieces`). For a
   development tree two other masks can be used without the weights: the one
   the 5-fold run saved (each predicted by a network that never saw the
   tree's row), or the lined-up hand mask.
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
python -m treeskel.model3d train data/raw                     # once, on the GPU: models/unet_named.pt
python -m treeskel.model3d build --tree R05N12 --data data/raw    # outputs/model3d/R05N12.html (+ .ply)
python -m treeskel.model3d build --rgb photo.png --depth depth.png --name my_tree
python -m treeskel.model3d build --tree R05N12 --data data/raw --mask saved   # no weights needed
```
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from treeskel.depth3d import METRES_PER_UNIT, Camera
from treeskel.tree3d import PARTS, depth_along, fit_plane, on_plane, tree_measures

WEIGHTS = Path("models/unet_named.pt")
"""Where ``train`` saves the network and ``build`` finds it (git-ignored)."""

SAVED_MASKS = Path("docs/results/segment_lined_up_joined/predicted_masks.npz")
"""The named run's development masks (5 folds, pieces joined)."""

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
            "seed": params.seed + SEED_OFFSET, "random_weights": bool(random_weights)}
    weights = Path(weights)
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, weights)
    progress(f"Saved {weights} ({meta['minutes']} min of training)")
    return meta


def load_network(weights: Path = WEIGHTS, device: str | None = None):
    """The saved network (no download: the weights replace ImageNet's), its device and what ``train`` saved."""
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
    480x480), its mirror image averaged in, its pieces joined."""
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
    tube built on them does not twist."""
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
    """A ball (UV sphere): ``{"vertices", "normals", "faces"}``."""
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
    """Several meshes as one."""
    out = {"vertices": [], "normals": [], "faces": []}
    k = 0
    for m in meshes:
        out["vertices"].append(m["vertices"])
        out["normals"].append(m["normals"])
        out["faces"].append(m["faces"] + k)
        k += len(m["vertices"])
    return {key: np.concatenate(v) for key, v in out.items()}


def to_view(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Camera-frame points (``y`` down, ``z`` away) as the model's: ``y`` up, ``z`` towards the camera,
    ``origin`` (the fork) at 0."""
    q = np.asarray(points, float) - origin
    return q * np.array([1.0, -1.0, -1.0])


def tree_model(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, mask: np.ndarray, camera: Camera = Camera(),
               per_unit: float = METRES_PER_UNIT, slab: float = SLAB_M, leaf_step: int = 2) -> dict:
    """The 3D model of one summer photo and its 256x256 branch mask.

    Returns:
        ``{"branches": mesh, "leaves": {"points", "colours"}, "measures", "plane"}``; the mesh and points in
        metres in the model's frame (:func:`to_view`), the measures as :func:`treeskel.tree3d.tree_measures`
        with sizes in metres.

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
    meshes, top_radius = [], None
    for p in PARTS:
        centre = on_plane(camera.rays(paths[p]), plane) * per_unit                          # metres
        radius = widths[p] * centre[:, 2] / camera.focal
        line, s = smooth_line(centre)
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
        r = ndi.gaussian_filter1d(np.interp(s, seg, radius), max(1.0, 2 * SMOOTH_M / STEP_M), mode="nearest")
        r = np.maximum(r, MIN_RADIUS_M)
        if p == "trunk":
            top_radius = float(r[-1])
        meshes.append(tube(to_view(line, fork), r))
    meshes.append(ball(np.zeros(3), top_radius or MIN_RADIUS_M))
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
    """A binary PLY mesh with normals and a colour per vertex (Blender, MeshLab, Windows 3D Viewer)."""
    v = np.asarray(mesh["vertices"], "<f4")
    n = np.asarray(mesh["normals"], "<f4")
    rgb = np.tile((np.asarray(colour) * 255).round().astype(np.uint8), (len(v), 1))
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
    """A binary PLY point cloud with a colour per point."""
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
    """Read back a PLY written by this module: ``{"vertex": structured array, "face": index array or None}``."""
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


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode("ascii")


def viewer_data(model: dict, title: str, source: str) -> dict:
    """What the viewer page needs: the arrays (base64, little-endian) and the text."""
    b, leaves = model["branches"], model["leaves"]
    m = model["measures"]

    def num(key, mult=1.0, spec=".1f"):
        """A measure as text (scaled by ``mult``), or ``None`` if missing."""
        x = m.get(key)
        return None if x is None or not np.isfinite(x) else format(x * mult, spec)

    return {
        "title": title, "source": source,
        "mesh": {"positions": _b64(b["vertices"], "<f4"), "normals": _b64(b["normals"], "<f4"),
                 "indices": _b64(b["faces"].ravel(), "<u4"), "colour": list(BARK)},
        "points": {"positions": _b64(leaves["points"], "<f4"), "colours": _b64(leaves["colours"], "u1"),
                   "count": int(len(leaves["points"]))},
        "measures": [["V opening in the tree's plane", num("v_opening_3d"), "°"],
                     ["V opening in the photo", num("v_opening_2d"), "°"],
                     ["Lean towards the camera", num("lean_camera"), "°"],
                     ["Trunk diameter at the fork", num("trunk_diameter", 100, ".0f"), "cm"],
                     ["Left leader in view", num("left_length", 1, ".2f"), "m"],
                     ["Right leader in view", num("right_length", 1, ".2f"), "m"],
                     ["Depth from the tree's plane (median)", num("flatness_median", 100), "cm"]],
    }


def write_html(path: Path, model: dict, title: str, source: str) -> None:
    """The self-contained viewer page (WebGL 2, no libraries, no internet)."""
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
  <div class="muted note">Sizes assume the fork is about 2 m from the camera; the leaders end where the photo
    or the prediction ends. Seen from the front, the model looks like the photo.</div>
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
  const td = tr.insertCell(); td.className = "v"; td.textContent = value + " " + unit;
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
  in vec3 position; in vec3 normal; uniform mat4 mvp; uniform mat4 view; out vec3 vNormal;
  void main() { gl_Position = mvp * vec4(position, 1.0); vNormal = mat3(view) * normal; }`,
`#version 300 es
  precision highp float; in vec3 vNormal; uniform vec3 colour; out vec4 outColour;
  void main() { float d = abs(normalize(vNormal).z); outColour = vec4(colour * (0.38 + 0.62 * d), 1.0); }`);
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
             spin: document.getElementById("spin") };
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
    gl.uniform3fv(gl.getUniformLocation(meshProg, "colour"), DATA.mesh.colour);
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

def build(name: str, rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, mask: np.ndarray, out: Path,
          title: str, source: str) -> dict:
    """Make one tree's model and write its files; returns the paths and the measures."""
    model = tree_model(rgb, depth, valid, mask)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{name}.html", "branches": out / f"{name}_branches.ply", "leaves": out / f"{name}_leaves.ply"}
    write_html(paths["html"], model, title, source)
    write_ply_mesh(paths["branches"], model["branches"])
    write_ply_points(paths["leaves"], model["leaves"]["points"], model["leaves"]["colours"])
    return {"files": paths, "measures": model["measures"], "n_points": len(model["leaves"]["points"]),
            "n_triangles": len(model["branches"]["faces"])}


def main(argv: Sequence[str] | None = None) -> int:
    """``train`` the network once, or ``build`` models from summer photos."""
    from treeskel.io import index_dataset, load_depth, load_rgb, load_tree

    ap = argparse.ArgumentParser(prog="python -m treeskel.model3d", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train the named U-Net on all development trees and save it (GPU, ~2 min)")
    tr.add_argument("data_dirs", nargs="+")
    tr.add_argument("--weights", default=str(WEIGHTS))
    tr.add_argument("--device")
    tr.add_argument("--split", default="splits.json")
    tr.add_argument("--epochs", type=int, help=argparse.SUPPRESS)                     # tests only
    tr.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)  # tests only
    bu = sub.add_parser("build", help="make the 3D model of summer photos")
    bu.add_argument("--tree", nargs="+", help="tree ids in --data (summer photo and depth)")
    bu.add_argument("--data", nargs="+", default=["data/raw"], help="dataset folder(s) for --tree")
    bu.add_argument("--rgb", help="a summer colour photo (640x480 PNG)")
    bu.add_argument("--depth", help="its 8-bit depth image (640x480 PNG)")
    bu.add_argument("--name", help="file name for --rgb (default: the photo's)")
    bu.add_argument("--mask", choices=("unet", "saved", "hand"), default="unet",
                    help="unet: the trained network (default); saved: the development masks of the 5-fold run; "
                         "hand: the lined-up hand mask")
    bu.add_argument("--weights", default=str(WEIGHTS))
    bu.add_argument("--saved", default=str(SAVED_MASKS), help="the saved development masks (--mask saved)")
    bu.add_argument("--mask-overlay", default="annotations/mask_overlay.json", help="the hand masks' line-up")
    bu.add_argument("--device")
    bu.add_argument("--out", default="outputs/model3d")
    args = ap.parse_args(argv)

    if args.command == "train":
        train(args.data_dirs, Path(args.weights), args.device, args.random_weights, args.epochs, args.split)
        return 0

    if bool(args.tree) == bool(args.rgb):
        ap.error("give --tree (with --data) or --rgb with --depth")
    if args.rgb and not args.depth:
        ap.error("--rgb needs its --depth image")
    if args.rgb and args.mask != "unet":
        ap.error("a photo outside the dataset needs the trained network (--mask unet)")

    network = None
    if args.mask == "unet":
        if not Path(args.weights).exists():
            print(f"No trained network at {args.weights}. Train it once: python -m treeskel.model3d train data/raw\n"
                  "(or, for a development tree, use --mask saved).")
            return 1
        network = load_network(Path(args.weights), args.device)
        print(f"Network: {args.weights} (trained on {network[2]['n_trees']} trees, {network[2]['epochs']} epochs)")

    jobs = []                                              # (name, rgb, depth, valid, title, record)
    if args.rgb:
        depth, valid = load_depth(args.depth)
        name = args.name or Path(args.rgb).stem
        jobs.append((name, load_rgb(args.rgb), depth, valid, Path(args.rgb).name, None))
    else:
        records = {t.tree_id: t for d in args.data for t in index_dataset(d).trees}
        missing = [t for t in args.tree if t not in records]
        if missing:
            print(f"Not found in {', '.join(args.data)}: {', '.join(missing)}")
            return 1
        for t in args.tree:
            s = load_tree(records[t], "summer")
            jobs.append((t, s["rgb"], s["depth"], s["valid"], f"Tree {t}, summer", records[t]))

    saved = overlays = None
    if args.mask == "saved":
        from treeskel.segment import load_masks
        saved = load_masks(Path(args.saved))
    if args.mask == "hand":
        from treeskel.summer_gt import load_overlays
        overlays = load_overlays(args.mask_overlay)

    status = 0
    for name, rgb, depth, valid, title, rec in jobs:
        if args.mask == "unet":
            model, device, meta = network
            mask = predict_mask(model, device, rgb, meta["params"], meta.get("join_gap"))
            source = f"Branches predicted by the U-Net trained on {meta['n_trees']} development trees"
        elif args.mask == "saved":
            if name not in saved:
                print(f"{name}: no saved development mask (test rows have none; use --mask unet)")
                status = 1
                continue
            mask = saved[name]
            source = "Branches from the 5-fold development run (a network that never saw this row)"
        else:
            from treeskel.segment import hand_mask
            mask = hand_mask(rec, overlays)
            if mask is None:
                print(f"{name}: no lined-up hand mask")
                status = 1
                continue
            source = "Branches from the hand-drawn mask, lined up"
        try:
            res = build(name, rgb, depth, valid, mask, Path(args.out), title, source)
        except ValueError as e:
            print(f"{name}: {e}")
            status = 1
            continue
        m = res["measures"]
        print(f"{name}: V opening {m['v_opening_3d']:.1f}° in the tree's plane ({m['v_opening_2d']:.1f}° in the photo), "
              f"lean {m['lean_camera']:.1f}° towards the camera; {res['n_triangles']:,} triangles, "
              f"{res['n_points']:,} leaf points")
        print(f"  open {res['files']['html']} in a browser (or the .ply files in Blender, MeshLab or 3D Viewer)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
