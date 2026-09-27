"""The shape of a tree's trunk and V, measured without aligning the seasons.

Winter-to-summer pixel alignment was abandoned in session 7
(``docs/results/align/``), so the winter traces cannot label the summer
photos pixel by pixel. Angles do not need that alignment. A change of camera
distance (scale) or position (shift) leaves every angle in the image
unchanged. Roll (rotation about the viewing axis) turns every direction by
the same amount, so the angle *between* two parts of the tree is unchanged
too. Pitch and yaw add perspective, which changes angles only slightly
(:func:`perspective_change`).

Each part is summarised by straight lines fitted by total least squares
(:func:`line_direction`), in an ``(x, y)`` frame with ``y`` pointing up:

- **Trunk axis:** the top of the trunk, from the fork down to
  ``trunk_reach`` trunk radii (all of the trunk if it is shorter). The top
  is used because that is what the leaders grow from, and because the
  trunk's base is often out of frame.
- **Leader line:** each leader from ``start`` to ``reach`` trunk radii from
  the fork (1.5 to 16). Nearer the fork, the traced path is still inside the
  top of the trunk. A fixed stretch, rather than all of the leader in view,
  is the student's choice (session 10): the leaders curve, so a line through
  everything in view depends on where the photo's top edge cuts them, and the
  summer photos are framed tighter than the winter ones. Trunk radii scale
  with the photo, so the same stretch of tree is measured in both seasons
  (:func:`framing_change`).
- **Halves:** the same stretch of each leader, split in two by length.

The measures (:func:`shape_measures`), all in degrees:

| Measure | Meaning | Camera roll |
|---|---|---|
| `v_opening` | angle between the two leader lines | no effect |
| `left_angle`, `right_angle` | each leader's line from the trunk axis, positive outwards (away from the other leader) | no effect |
| `lean` | the V's bisector from the trunk axis, positive to the right | no effect |
| `left_bend`, `right_bend` | the upper half's line from the lower half's, positive outwards: how much a leader curves | no effect |
| `trunk_tilt` | the trunk axis from the image's vertical, positive to the right | adds to it |
| `left_tilt`, `right_tilt` | each leader's line from the image's vertical, positive to the right | adds to it |
| `lean_v` | the V's bisector from the image's vertical | adds to it |

They are related: ``v_opening = left_angle + right_angle = right_tilt −
left_tilt``, ``lean = (right_angle − left_angle) / 2 = lean_v −
trunk_tilt``. The trunk axis matters only for the three trunk-relative
measures.

The same measures are taken from the provided hand masks
(:func:`mask_branches`), whose crop and resize (the photo's centre 480x480
to 256x256) scale both directions alike, so their angles need no conversion.
The hand masks were drawn on the summer view: laid onto the winter traces
(:func:`hand_frame`), they turn by the camera's roll between the seasons.

**The run** (:func:`main`), on the development rows only: each tree's winter
trace, the straight lines through its clicks (a check on the trace) and its
hand mask are measured; how much a tighter framing would change the
measures is recorded; and features of the summer photo are computed with a
pretrained network (:func:`tree_features`) for :mod:`treeskel.shape_model`.
The table and the features are committed in ``data/shape/``, so the models
can be fitted without the images::

    python -m treeskel.shape data/raw --figure docs/figures/shape_lines.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

from treeskel.guided_gt import arc_length
from treeskel.winter_gt import find_main_branches

MEASURES = ("v_opening", "left_angle", "right_angle", "lean", "left_bend", "right_bend",
            "trunk_tilt", "left_tilt", "right_tilt", "lean_v")
"""Names of the shape measures returned by :func:`shape_measures`, in degrees."""

ROLL_FREE = ("v_opening", "left_angle", "right_angle", "lean", "left_bend", "right_bend")
"""Measures that camera roll does not change (angles between parts of the tree)."""

MIRROR = {"v_opening": "v_opening", "left_angle": "right_angle", "right_angle": "left_angle",
          "lean": "-lean", "left_bend": "right_bend", "right_bend": "left_bend",
          "trunk_tilt": "-trunk_tilt", "left_tilt": "-right_tilt", "right_tilt": "-left_tilt",
          "lean_v": "-lean_v"}
"""Each measure of a tree's mirror image (flipped left to right), in terms of the original's
measures; a leading ``-`` negates. Used to check :func:`shape_measures` and to augment training
data with flipped photos."""


@dataclass(frozen=True)
class ShapeParams:
    """Where on the trunk and leaders the lines are fitted.

    Attributes:
        start: Leader lines start this many trunk radii (at the fork) from
            the fork, the distance :mod:`treeskel.guided_gt` also skips when
            fitting leader widths (``leader_fit_start``).
        reach: Leader lines end this many trunk radii from the fork (along
            the leader), or at the tip if that is nearer; ``inf`` uses the
            whole leader in view. On the 21 development trees in the
            repository, the leaders are 18.8-43 trunk radii long in the
            winter photos and 16.7-32.7 in the hand masks, so 16 is in view
            in both.
        trunk_reach: The trunk axis is fitted from the fork down to this many
            trunk radii.
        min_trunk: Shortest trunk (px of path) that gives an axis; trees that
            split at the ground have none, and their trunk-relative measures
            are NaN.
        min_points: Fewest path points a leader line or half-line needs.
    """

    start: float = 1.5
    reach: float = 16.0
    trunk_reach: float = 4.0
    min_trunk: float = 10.0
    min_points: int = 10


def to_xy(path: np.ndarray) -> np.ndarray:
    """``(row, col)`` image points as ``(x, y)`` with ``y`` pointing up.

    Example:
        >>> to_xy(np.array([[10, 3]])).tolist()
        [[3.0, -10.0]]
    """
    p = np.asarray(path, float).reshape(-1, 2)
    return np.stack([p[:, 1], -p[:, 0]], 1)


def line_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction of the total-least-squares line through ``(x, y)`` points.

    The direction is the principal axis of the centred points (the first
    right singular vector), so rows and columns are treated alike and a
    vertical line is no harder than any other. It is oriented from the first
    point towards the last.

    Example:
        >>> np.round(line_direction(np.array([[0.0, 0], [1, 1], [2, 2]])), 4).tolist()
        [0.7071, 0.7071]
    """
    p = np.asarray(points, float)
    c = p - p.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    return -d if np.dot(p[-1] - p[0], d) < 0 else d


def tilt(direction: np.ndarray) -> float:
    """Angle of an ``(x, y)`` direction from straight up, positive to the right (degrees).

    Example:
        >>> round(tilt(np.array([1.0, 1.0])), 6), round(tilt(np.array([-1.0, np.sqrt(3)])), 6)
        (45.0, -30.0)
    """
    return float(np.degrees(np.arctan2(direction[0], direction[1])))


def wrap(angle: float) -> float:
    """An angle in degrees, wrapped to ``(-180, 180]``."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else float(a)


def trunk_top(trunk: np.ndarray, fork_radius: float, params: ShapeParams = ShapeParams()) -> np.ndarray | None:
    """The ``(row, col)`` points of the trunk's top used for its axis, from the fork down, or None if too short.

    Args:
        trunk: ``(N, 2)`` ``(row, col)`` path from the root to the fork.
        fork_radius: The trunk's radius at the fork (px).
        params: ``trunk_reach`` and ``min_trunk``.
    """
    p = np.asarray(trunk)[::-1]                          # from the fork down
    if len(p) < 2 or arc_length(p)[-1] < params.min_trunk:
        return None
    return p[arc_length(p) <= max(params.trunk_reach * fork_radius, params.min_trunk)]


def trunk_axis(trunk: np.ndarray, fork_radius: float, params: ShapeParams = ShapeParams()) -> np.ndarray | None:
    """Direction of the top of the trunk (:func:`trunk_top`), pointing up towards the fork, or None if too short."""
    top = trunk_top(trunk, fork_radius, params)
    return None if top is None else -line_direction(to_xy(top))     # fitted fork -> down; flip to point up


def leader_part(leader: np.ndarray, fork_radius: float, params: ShapeParams = ShapeParams()) -> np.ndarray:
    """The ``(row, col)`` points of a leader (fork to tip) used for its line: ``start`` to ``reach`` trunk radii."""
    p = np.asarray(leader)
    s = arc_length(p)
    return p[(s >= params.start * fork_radius) & (s <= params.reach * fork_radius)]


def halves(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a path into its lower and upper half by length (the middle point is in both)."""
    s = arc_length(points)
    mid = s[-1] / 2 if len(s) else 0.0
    return points[s <= mid], points[s >= mid]


def shape_measures(trunk: np.ndarray, left: np.ndarray, right: np.ndarray, fork_radius: float,
                   params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """The shape measures of one tree (see the module docstring).

    Args:
        trunk: ``(N, 2)`` ``(row, col)`` path from the root to the fork.
        left, right: Paths from the fork to each leader's tip; "left" is the
            leader whose tip is further left.
        fork_radius: The trunk's radius at the fork (px), the unit of
            ``start``, ``reach`` and ``trunk_reach``.
        params: Where the lines are fitted.

    Returns:
        ``{measure: degrees}`` for every name in :data:`MEASURES`, NaN where
        a part is too short to fit (e.g. no trunk axis when the tree splits
        at the ground), plus ``fork_radius`` and each leader's whole length
        in view in trunk radii (``left_length``, ``right_length``); a leader
        shorter than ``reach`` is measured over less than the fixed stretch.
    """
    out = dict.fromkeys(MEASURES, float("nan"))
    r = float(fork_radius)
    tilts, bends, lengths = {}, {}, {}
    for side, path in (("left", left), ("right", right)):
        part = leader_part(path, r, params)
        lengths[side] = float(arc_length(path)[-1] / r) if len(path) > 1 else 0.0
        if len(part) < params.min_points:
            continue
        tilts[side] = tilt(line_direction(to_xy(part)))
        low, up = halves(part)
        if min(len(low), len(up)) >= params.min_points:
            turn = wrap(tilt(line_direction(to_xy(up))) - tilt(line_direction(to_xy(low))))
            bends[side] = turn if side == "right" else -turn       # outwards is positive
    axis = trunk_axis(trunk, r, params)
    if axis is not None:
        out["trunk_tilt"] = tilt(axis)
    if "left" in tilts:
        out["left_tilt"] = tilts["left"]
        out["left_bend"] = bends.get("left", float("nan"))
        if axis is not None:
            out["left_angle"] = wrap(out["trunk_tilt"] - tilts["left"])
    if "right" in tilts:
        out["right_tilt"] = tilts["right"]
        out["right_bend"] = bends.get("right", float("nan"))
        if axis is not None:
            out["right_angle"] = wrap(tilts["right"] - out["trunk_tilt"])
    if len(tilts) == 2:
        out["v_opening"] = wrap(tilts["right"] - tilts["left"])
        out["lean_v"] = (tilts["left"] + tilts["right"]) / 2
        if axis is not None:
            out["lean"] = wrap(out["lean_v"] - out["trunk_tilt"])
    out.update(fork_radius=r, left_length=lengths["left"], right_length=lengths["right"])
    return out


def gt_shape(gt, params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """:func:`shape_measures` of a traced tree (a :class:`treeskel.winter_gt.WinterGT` with radii)."""
    return shape_measures(gt.trunk, gt.left, gt.right, float(gt.radii["trunk"][-1]), params)


def mirror_measures(m: dict[str, float]) -> dict[str, float]:
    """The measures of a tree's mirror image, from the tree's own (see :data:`MIRROR`)."""
    return {k: (-m[v[1:]] if v.startswith("-") else m[v]) for k, v in MIRROR.items()}


# --------------------------------------------------------------------------
# Other sources of the same measures
# --------------------------------------------------------------------------

def polyline(points: Sequence[Sequence[float]]) -> np.ndarray:
    """A connected 8-connected ``(row, col)`` pixel path through the given points in order."""
    pts = [tuple(int(round(v)) for v in p) for p in points]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        rr, cc = draw_line(a[0], a[1], b[0], b[1])
        out.extend(zip(rr[1:].tolist(), cc[1:].tolist()))
    return np.array(out, int).reshape(-1, 2)


def click_paths(clicks) -> dict[str, np.ndarray]:
    """Straight-line paths through a tree's clicks (no tracing, so no images needed).

    Args:
        clicks: A complete :class:`treeskel.guided_gt.Clicks`.

    Returns:
        ``{"trunk", "left", "right"}`` paths like a trace's: the trunk from the
        root through its waypoints to the fork, each leader from the fork
        through its waypoints to the tip.
    """
    c = clicks.normalised()
    return {"trunk": polyline(c.trunk), "left": polyline([c.fork, *c.left]),
            "right": polyline([c.fork, *c.right])}


def mask_branches(mask: np.ndarray) -> dict | None:
    """The trunk, the two leaders and the trunk's radius, from a hand-drawn mask.

    The mask is thinned to its centre lines and the root, fork and leaders are
    picked as for the winter depth (:func:`treeskel.winter_gt.find_main_branches`),
    as :func:`treeskel.measure_width.hand_mask_profiles` does. The radius is the
    median half-width of the trunk 3–40 mask px below the fork.

    Args:
        mask: ``(256, 256)`` bool hand mask.

    Returns:
        ``{"trunk", "left", "right"}`` paths in the mask's own ``(row, col)``
        pixels, and ``"fork_radius"`` (mask px), or None if no fork is found.
    """
    m = ndi.binary_fill_holes(mask)
    if not m.any():
        return None
    sk = skeletonize(m)
    lab, n = ndi.label(sk, np.ones((3, 3)))
    if n > 1:
        sk = lab == (np.argmax(ndi.sum(sk, lab, range(1, n + 1))) + 1)
    br = find_main_branches(m, sk)
    if br["fork"] is None:
        return None
    edt = ndi.distance_transform_edt(m)
    down = br["trunk"][::-1]
    s, w = arc_length(down), edt[down[:, 0], down[:, 1]]
    near = (s > 3) & (s < 40)
    radius = float(np.median(w[near])) if near.sum() > 3 else float(np.median(w))
    return {"trunk": br["trunk"], "left": br["left"], "right": br["right"], "fork_radius": radius}


# --------------------------------------------------------------------------
# How much the measures depend on the view
# --------------------------------------------------------------------------

def crop_top(path: np.ndarray, top_row: float) -> np.ndarray:
    """A leader (fork to tip) as a photo whose top edge is at ``top_row`` would show it.

    The path is kept from its start until it first rises above ``top_row``.
    """
    p = np.asarray(path)
    above = np.nonzero(p[:, 0] < top_row)[0]
    return p if len(above) == 0 else p[:above[0]]


def framing_change(trunk: np.ndarray, left: np.ndarray, right: np.ndarray, fork_radius: float,
                   keep: float, params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """How much each measure changes if the photo's top edge were lower.

    The top edge is moved down so that only ``keep`` of the height between
    the fork and the image's top (row 0) stays in view, which cuts both
    leaders at the same row, as a tighter framing would. With the fixed
    stretch (``reach``) nothing changes while the stretch stays in view; with
    ``reach = inf`` the lines follow the curved leaders' visible part.

    Returns:
        ``{measure: cropped − full}`` in degrees.
    """
    fork_row = float(np.asarray(trunk)[-1][0])
    cut = fork_row * (1.0 - keep)
    full = shape_measures(trunk, left, right, fork_radius, params)
    crop = shape_measures(trunk, crop_top(left, cut), crop_top(right, cut), fork_radius, params)
    return {k: crop[k] - full[k] for k in MEASURES}


def transform_path(path: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 ``(x, y)`` image transform to a ``(row, col)`` path; returns float ``(row, col)``."""
    p = np.asarray(path, float).reshape(-1, 2)
    xy1 = np.concatenate([p[:, ::-1], np.ones((len(p), 1))], 1) @ np.asarray(matrix, float).T
    return (xy1[:, :2] / xy1[:, 2:3])[:, ::-1]


def perspective_change(trunk: np.ndarray, left: np.ndarray, right: np.ndarray, fork_radius: float,
                       pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0, scale: float = 1.0,
                       params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """How much each measure changes if the camera were turned, as between the seasons.

    The paths are moved by the pinhole-camera rotation of
    :class:`treeskel.summer_gt.Overlay` (``K R K⁻¹``, with the D435-like focal
    length used there) and measured again.

    Returns:
        ``{measure: turned − original}`` in degrees.
    """
    from treeskel.summer_gt import Overlay

    m = Overlay(pitch=pitch, yaw=yaw, roll=roll, scale=scale).matrix()
    full = shape_measures(trunk, left, right, fork_radius, params)
    moved = shape_measures(*(transform_path(p, m) for p in (trunk, left, right)), fork_radius * scale, params)
    return {k: wrap(moved[k] - full[k]) for k in MEASURES}


def similarity_fit(src: np.ndarray, dst: np.ndarray, search: int = 120, step: int = 8, iters: int = 50,
                   trim: float = 0.8) -> tuple[np.ndarray, float]:
    """The rotation, scale and shift that best lay one set of centre-line points onto another.

    Iterative closest points: the best whole-pixel shift on a coarse grid
    first, then, repeatedly, each ``src`` point is paired with its nearest
    ``dst`` point and a similarity is fitted by least squares
    (:func:`treeskel.align.fit_transform`) to the closest ``trim`` of the
    pairs, so the parts one set shows and the other does not are left out.
    Used to find the frame a hand mask was drawn in: a V's lines through its
    fork do not pin down the scale well, but they do pin down the rotation.

    Args:
        src, dst: ``(N, 2)`` and ``(M, 2)`` ``(x, y)`` points; ``src`` should
            be the set that shows less (every one of its points has a partner).
        search, step: Shifts tried first (px), from ``-search`` to ``search``.
        iters: Rounds of pairing and fitting.
        trim: Fraction of the pairs used in each fit.

    Returns:
        ``(matrix, residual)``: the 3x3 ``src`` to ``dst`` transform and the
        median distance (px) from the moved ``src`` points to ``dst``.
    """
    from scipy.spatial import cKDTree

    from treeskel.align import apply_transform, fit_transform

    src, dst = np.asarray(src, float), np.asarray(dst, float)
    tree = cKDTree(dst)
    shifts = np.arange(-search, search + 1, step)
    best = min((np.median(tree.query(src + [dx, dy])[0]), dx, dy) for dx in shifts for dy in shifts)
    m = np.array([[1.0, 0, best[1]], [0, 1, best[2]], [0, 0, 1]])
    for _ in range(iters):
        dist, idx = tree.query(apply_transform(m, src))
        keep = dist <= np.quantile(dist, trim)
        fit = fit_transform(src[keep], dst[idx[keep]], "similarity")
        if fit is None:
            break
        m = fit
    return m, float(np.median(tree.query(apply_transform(m, src))[0]))


def hand_frame(winter: dict, hand: dict) -> dict[str, float]:
    """The frame a hand mask was drawn in, relative to the winter photo.

    Args:
        winter: ``{"trunk", "left", "right"}`` traced paths in the winter photo.
        hand: The same from :func:`mask_branches` (mask pixels).

    Returns:
        ``{"rotation", "scale", "residual"}`` of the winter-to-hand-mask
        similarity: rotation in degrees, positive clockwise in the photo (as
        :func:`treeskel.align.describe`), scale, and the median distance (px)
        left between the centre lines. If the masks were drawn on the summer
        photos, the rotation is the camera's roll between the seasons. The
        scale is hardly determined (lines through the fork map onto
        themselves under any scaling about it) and should not be relied on.
    """
    from treeskel.align import describe

    w = np.concatenate([to_xy(winter[k]) for k in ("trunk", "left", "right")]) * [1, -1]      # (x, y down)
    h = np.concatenate([mask_to_image(hand[k])[:, ::-1] for k in ("trunk", "left", "right")])
    m, residual = similarity_fit(h, w)                     # the mask shows less: lay it on the winter trace
    d = describe(np.linalg.inv(m), w.mean(0))              # the inverse: winter to hand mask
    return {"rotation": d["rotation"], "scale": d["scale"], "residual": residual}


# --------------------------------------------------------------------------
# Drawing the fitted lines
# --------------------------------------------------------------------------

def _segment(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End points ``(row, col)`` of a line fitted to ``(row, col)`` points, spanning their extent."""
    xy = to_xy(points)
    d, c = line_direction(xy), xy.mean(0)
    t = (xy - c) @ d
    a, b = c + t.min() * d, c + t.max() * d
    return np.array([-a[1], a[0]]), np.array([-b[1], b[0]])


def fitted_lines(trunk: np.ndarray, left: np.ndarray, right: np.ndarray, fork_radius: float,
                 params: ShapeParams = ShapeParams()) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The lines :func:`shape_measures` fits, as ``(start, end)`` ``(row, col)`` points, for drawing.

    Returns:
        Some of ``"trunk"``, ``"left"``, ``"right"``, ``"left_lower"``, ``"left_upper"``,
        ``"right_lower"`` and ``"right_upper"`` (a part too short to fit is left out).
    """
    out = {}
    top = trunk_top(trunk, fork_radius, params)
    if top is not None and len(top) >= 2:
        out["trunk"] = _segment(top)
    for side, path in (("left", left), ("right", right)):
        part = leader_part(path, fork_radius, params)
        if len(part) < params.min_points:
            continue
        out[side] = _segment(part)
        low, up = halves(part)
        if min(len(low), len(up)) >= params.min_points:
            out[f"{side}_lower"], out[f"{side}_upper"] = _segment(low), _segment(up)
    return out


def mask_to_image(path: np.ndarray) -> np.ndarray:
    """``(row, col)`` points of a 256x256 mask as points of the 640x480 photo (see :func:`treeskel.io.to_mask_frame`)."""
    from treeskel.io import CROP_COLS, IMAGE_SHAPE, MASK_SHAPE

    p = np.asarray(path, float).reshape(-1, 2)
    k = IMAGE_SHAPE[0] / MASK_SHAPE[0]
    return np.stack([p[:, 0] * k, CROP_COLS[0] + p[:, 1] * k], 1)


LINE_COLOURS = {"trunk": (255, 230, 0), "left": (255, 60, 60), "right": (255, 60, 60),
                "left_lower": (60, 160, 255), "left_upper": (60, 160, 255),
                "right_lower": (60, 160, 255), "right_upper": (60, 160, 255)}
"""Colours of the fitted lines in figures: trunk axis yellow, leader lines red, halves blue."""


def draw_shape(rgb: np.ndarray, paths: Sequence[np.ndarray], lines: dict, label: str = "",
               path_colour: tuple[int, int, int] = (0, 255, 255)):
    """A photo with the traced paths (thin) and the fitted lines (thick) drawn on it, as a PIL image."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(rgb, np.uint8))
    dr = ImageDraw.Draw(im)
    for p in paths:
        p = np.asarray(p, float)
        if len(p) > 1:
            dr.line([(float(c), float(r)) for r, c in p], fill=path_colour, width=1)
    for name in ("left_lower", "left_upper", "right_lower", "right_upper", "left", "right", "trunk"):
        if name in lines:
            (r0, c0), (r1, c1) = lines[name]
            width = 2 if name.endswith(("lower", "upper")) else 3
            dr.line([(c0, r0), (c1, r1)], fill=LINE_COLOURS[name], width=width)
    if label:
        dr.text((6, 6), label, fill=(255, 255, 0))
    return im


# --------------------------------------------------------------------------
# Summer image features from a pretrained network
# --------------------------------------------------------------------------

FEATURE_SIZE = (240, 320)
"""``(height, width)`` the summer images are resized to for the network: half size, near the
224-pixel images ResNet-18 was trained on, so that one layer-4 cell sees a larger part of the tree."""

GRID = 2
"""The layer-4 features are also averaged over a ``GRID`` x ``GRID`` grid of cells, so that the left
and right (and upper and lower) parts of the photo keep their own features."""

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def network_input(image: np.ndarray, size: tuple[int, int] = FEATURE_SIZE, mirror: bool = False) -> np.ndarray:
    """An 8-bit colour or depth image as the network's input.

    A depth image (one channel) is repeated in all three channels. The image
    is flipped left to right first if ``mirror``, resized (bilinear), scaled
    to [0, 1] and normalised with the ImageNet mean and standard deviation.

    Returns:
        ``(3, h, w)`` float32 array.

    Example:
        >>> network_input(np.zeros((480, 640), np.uint8), (24, 32)).shape
        (3, 24, 32)
    """
    from PIL import Image

    a = np.asarray(image, np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if mirror:
        a = a[:, ::-1]
    x = np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize(size[::-1], Image.BILINEAR), np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()


def make_resnet18(pretrained: bool = True, device: str | None = None):
    """ResNet-18 up to its last convolutional block (layer 4), for extracting features.

    Args:
        pretrained: Load torchvision's ImageNet weights (downloaded once, about
            45 MB, into the torch cache). False gives random weights, for tests.
        device: ``"cuda"``, ``"cpu"``, or None for the GPU if there is one.

    Returns:
        ``(network, device)``; the network is in evaluation mode.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    body = torch.nn.Sequential(*list(net.children())[:-2])          # drop the pooling and the classifier
    return body.eval().to(device), device


def cnn_features(body, device: str, inputs: np.ndarray, grid: int = GRID,
                 batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Layer-4 features averaged over the whole map and over a grid of cells.

    Args:
        body: Network from :func:`make_resnet18`.
        device: Its device.
        inputs: ``(N, 3, h, w)`` float32 images from :func:`network_input`.
        grid: Cells per side.
        batch: Images per forward pass.

    Returns:
        ``(whole, cells)``: ``(N, C)`` and ``(N, grid * grid, C)``, cells in
        reading order (top-left, top-right, bottom-left, bottom-right for 2).
    """
    import torch
    import torch.nn.functional as F

    whole, cells = [], []
    with torch.inference_mode():
        for i in range(0, len(inputs), batch):
            f = body(torch.from_numpy(np.ascontiguousarray(inputs[i:i + batch])).to(device))
            whole.append(F.adaptive_avg_pool2d(f, 1).flatten(1).float().cpu().numpy())
            cells.append(F.adaptive_avg_pool2d(f, grid).flatten(2).transpose(1, 2).float().cpu().numpy())
    return np.concatenate(whole), np.concatenate(cells)


def tree_features(body, device: str, summer: dict, size: tuple[int, int] = FEATURE_SIZE) -> dict[str, np.ndarray]:
    """Features of one tree's summer photo and depth, and of their mirror images.

    Args:
        body, device: From :func:`make_resnet18`.
        summer: ``load_tree(rec, "summer")``.
        size: Network input size.

    Returns:
        ``{"rgb_global": (2, C), "rgb_cells": (2, GRID², C), "depth_global": (2, C)}``;
        the first axis is the photo, then its mirror image.
    """
    rgb = np.stack([network_input(summer["rgb"], size, m) for m in (False, True)])
    depth = np.stack([network_input(summer["depth"], size, m) for m in (False, True)])
    g, c = cnn_features(body, device, rgb)
    dg, _ = cnn_features(body, device, depth)
    return {"rgb_global": g, "rgb_cells": c, "depth_global": dg}


def save_features(path: str | Path, ids: Sequence[str], feats: Sequence[dict], meta: dict) -> None:
    """Write the features of many trees to one ``.npz`` (float16, compressed)."""
    arrays = {k: np.stack([f[k] for f in feats]).astype(np.float16) for k in feats[0]}
    np.savez_compressed(path, ids=np.array(ids), meta=json.dumps(meta), **arrays)


def load_features(path: str | Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    """Inverse of :func:`save_features`: ``(ids, arrays as float32, meta)``."""
    with np.load(path) as z:
        ids = [str(t) for t in z["ids"]]
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k].astype(np.float32) for k in z.files if k not in ("ids", "meta")}
    return ids, arrays, meta


# --------------------------------------------------------------------------
# The run over the development trees
# --------------------------------------------------------------------------

SOURCES = ("winter", "clicks", "hand")
"""Where the measures come from: the winter trace (the truth), straight lines through the clicks (a
check on the trace), and the provided hand masks (drawn on the summer view: a baseline)."""

COLUMNS = MEASURES + ("fork_radius", "left_length", "right_length")
"""Values recorded per source in the table."""

FRAMING_KEEP = (0.85, 0.7)
"""Tighter framings tried by the run: the fraction of the height above the fork left in view."""

FRAMING_MEASURES = ("v_opening", "left_bend", "right_bend")
"""Measures whose change under a tighter framing the run records."""

GOOD_FIT_PX = 8.0
"""A hand mask counts as laid onto its winter trace if their centre lines end up within this
median distance (px); on the 21 development trees in the repository, all fitted within 1.4-4.7 px."""


def measure_tree(job: tuple) -> tuple[dict, object]:
    """Measure one tree from its winter trace, its clicks and its hand mask.

    Args:
        job: ``(tree_record, clicks, params, want_tile)``.

    Returns:
        ``(row, tile)``: the table row (``{source}_{value}`` columns) and, if
        ``want_tile``, a figure tile (winter trace | hand mask on summer), else None.
    """
    from PIL import Image

    from treeskel.guided_gt import guided_gt
    from treeskel.io import load_mask, load_tree

    rec, clicks, params, want_tile = job
    w = load_tree(rec, "winter")
    gt = guided_gt(w["rgb"], w["depth"], w["valid"], clicks)
    r = float(gt.radii["trunk"][-1])
    cp = click_paths(clicks)
    sources = {"winter": (gt.trunk, gt.left, gt.right, r),
               "clicks": (cp["trunk"], cp["left"], cp["right"], r)}          # clicks have no width: the trace's
    mb = mask_branches(load_mask(rec.paths[("summer", "mask")])) if rec.has_mask else None
    if mb is not None:
        sources["hand"] = (mb["trunk"], mb["left"], mb["right"], mb["fork_radius"])
    row = {"tree": rec.tree_id, "row": rec.row, "note": clicks.note}
    for name in SOURCES:
        m = shape_measures(*sources[name], params) if name in sources else dict.fromkeys(COLUMNS, float("nan"))
        row.update({f"{name}_{k}": float(m[k]) for k in COLUMNS})
    # How much a tighter framing of the winter photo would change the headline measures, with the
    # fixed stretch of each leader and with all of it in view.
    for label, p in (("fixed", params), ("whole", replace(params, reach=float("inf")))):
        for keep in FRAMING_KEEP:
            change = framing_change(*sources["winter"], keep, p)
            row.update({f"framing_{label}_{round(keep * 100)}_{k}": float(change[k]) for k in FRAMING_MEASURES})
    frame = (hand_frame({"trunk": gt.trunk, "left": gt.left, "right": gt.right}, mb) if mb is not None
             else dict.fromkeys(("rotation", "scale", "residual"), float("nan")))
    row.update({f"hand_frame_{k}": float(v) for k, v in frame.items()})
    tile = None
    if want_tile:
        wl = fitted_lines(*sources["winter"], params)
        a = draw_shape(w["rgb"], (gt.trunk, gt.left, gt.right), wl, f"{rec.tree_id} winter trace")
        s = load_tree(rec, "summer")
        if mb is not None:
            hl = {k: tuple(mask_to_image(np.array(v))) for k, v in fitted_lines(*sources["hand"], params).items()}
            paths = [mask_to_image(mb[k]) for k in ("trunk", "left", "right")]
            b = draw_shape(s["rgb"], paths, hl, "hand mask, summer", path_colour=(255, 0, 255))
        else:
            b = draw_shape(s["rgb"], (), {}, "no hand mask")
        tile = Image.new("RGB", (a.width + b.width, a.height))
        tile.paste(a, (0, 0))
        tile.paste(b, (a.width, 0))
    return row, tile


def _stats(v: np.ndarray) -> tuple[float, float, int]:
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std(ddof=1)), len(v)) if len(v) > 1 else (float("nan"), float("nan"), len(v))


def summarise(rows: Sequence[dict], params: ShapeParams = ShapeParams()) -> dict:
    """Per measure: the winter spread, the clicks' and the hand masks' agreement with the winter trace.

    Returns:
        ``{"n_trees", "measures": {measure: {...}}, "short_leaders": {...},
        "no_trunk_axis": [...], "hand_missing": [...]}``.
    """
    def col(k):
        return np.array([r[k] for r in rows], float)

    out = {"n_trees": len(rows), "measures": {}}
    for k in MEASURES:
        w, c, h = col(f"winter_{k}"), col(f"clicks_{k}"), col(f"hand_{k}")
        mean, sd, n = _stats(w)
        dc, dh = c - w, h - w
        ok = np.isfinite(w) & np.isfinite(h)
        out["measures"][k] = {
            "winter_mean": mean, "winter_sd": sd, "n": n,
            "clicks_median_abs_diff": float(np.nanmedian(np.abs(dc))) if np.isfinite(dc).any() else None,
            "hand_mean_diff": float(np.nanmean(dh)) if ok.any() else None,
            "hand_median_abs_diff": float(np.nanmedian(np.abs(dh))) if ok.any() else None,
            "hand_r": float(np.corrcoef(w[ok], h[ok])[0, 1]) if ok.sum() > 2 else None,
            "hand_n": int(ok.sum()),
        }
    short = {}
    for src in ("winter", "hand"):
        lengths = np.concatenate([col(f"{src}_left_length"), col(f"{src}_right_length")])
        lengths = lengths[np.isfinite(lengths)]
        short[src] = {"fraction": float(np.mean(lengths < params.reach)) if len(lengths) else None,
                      "n_leaders": int(len(lengths)),
                      "median_length": float(np.median(lengths)) if len(lengths) else None}
    out["short_leaders"] = short
    out["no_trunk_axis"] = [r["tree"] for r in rows if not np.isfinite(r["winter_trunk_tilt"])]
    out["hand_missing"] = [r["tree"] for r in rows if not np.isfinite(r["hand_v_opening"])]
    framing = {}
    for label in ("fixed", "whole"):
        for keep in FRAMING_KEEP:
            for k in FRAMING_MEASURES:
                name = f"framing_{label}_{round(keep * 100)}_{k}"
                v = np.abs(col(name)) if rows and name in rows[0] else np.array([])
                v = v[np.isfinite(v)]
                framing[name[len("framing_"):]] = ({"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                                                    "over_1deg": float(np.mean(v > 1.0)), "n": int(len(v))}
                                                   if len(v) else None)
    out["framing_change"] = framing
    if rows and "hand_frame_rotation" in rows[0]:
        rot, res, scale = col("hand_frame_rotation"), col("hand_frame_residual"), col("hand_frame_scale")
        good = np.isfinite(rot) & (res <= GOOD_FIT_PX)
        by_row = {}
        for orchard_row in sorted({r["row"] for r in rows}):
            sel = good & (col("row") == orchard_row)
            if sel.any():
                by_row[int(orchard_row)] = {"median": float(np.median(rot[sel])), "n": int(sel.sum())}
        out["hand_frame"] = {
            "n": int(np.isfinite(rot).sum()), "good_fits": int(good.sum()), "good_fit_px": GOOD_FIT_PX,
            "residual_median": float(np.nanmedian(res)) if np.isfinite(res).any() else None,
            "rotation_mean": float(rot[good].mean()) if good.any() else None,
            "rotation_sd": float(rot[good].std(ddof=1)) if good.sum() > 1 else None,
            "scale_median": float(np.median(scale[good])) if good.any() else None,
            "rotation_by_row": by_row,
        }
    return out


def _num(x, spec: str = "6.1f") -> str:
    return "   n/a" if x is None or not np.isfinite(x) else format(x, spec)


def print_summary(s: dict, params: ShapeParams = ShapeParams()) -> None:
    """Print :func:`summarise`'s result as a table."""
    print(f"\nShape measures on {s['n_trees']} development trees (degrees; leader lines {params.start:g}-"
          f"{params.reach:g} trunk radii from the fork):")
    print(f"  {'measure':12s} {'winter mean':>11s} {'sd':>6s} {'n':>4s} | {'clicks-winter':>13s} | "
          f"{'hand-winter mean':>16s} {'median|d|':>9s} {'r':>5s} {'n':>4s}")
    for k, m in s["measures"].items():
        print(f"  {k:12s} {_num(m['winter_mean'], '11.1f')} {_num(m['winter_sd'])} {m['n']:4d} | "
              f"{_num(m['clicks_median_abs_diff'], '13.1f')} | {_num(m['hand_mean_diff'], '+16.1f')} "
              f"{_num(m['hand_median_abs_diff'], '9.1f')} {_num(m['hand_r'], '5.2f')} {m['hand_n']:4d}")
    for src, v in s["short_leaders"].items():
        if v["fraction"] is not None:
            print(f"  {src}: {v['fraction']:.1%} of {v['n_leaders']} leaders are shorter than {params.reach:g} "
                  f"trunk radii in view (median length {v['median_length']:.1f})")
    if s["no_trunk_axis"]:
        print(f"  No trunk axis (trunk shorter than {params.min_trunk:g} px): {', '.join(s['no_trunk_axis'])}")
    if s["hand_missing"]:
        print(f"  No fork found in the hand mask: {', '.join(s['hand_missing'])}")
    fr = s.get("framing_change", {})
    if any(fr.values()):
        print("  Change if the winter photo were framed tighter at the top (median / 95th percentile, degrees):")
        for keep in FRAMING_KEEP:
            parts = []
            for label, name in (("fixed", f"{params.start:g}-{params.reach:g} trunk radii"), ("whole", "whole leader")):
                v = [fr.get(f"{label}_{round(keep * 100)}_{k}") for k in FRAMING_MEASURES]
                parts.append(name + ": " + ", ".join(f"{k} {x['median']:.1f}/{x['p95']:.1f}"
                                                      for k, x in zip(FRAMING_MEASURES, v) if x))
            print(f"    {keep:.0%} of the height in view -- " + "; ".join(parts))
    hf = s.get("hand_frame")
    if hf and hf["good_fits"]:
        print(f"  Hand masks laid onto the winter traces (rotation, scale, shift): {hf['good_fits']} of {hf['n']} fit "
              f"within {hf['good_fit_px']:g} px (median {hf['residual_median']:.1f} px); rotation winter to hand mask "
              f"{hf['rotation_mean']:+.2f} deg (sd {_num(hf['rotation_sd'], '.2f').strip()}).")
        print("    by orchard row (median deg, trees): " + ", ".join(
            f"{r}: {v['median']:+.1f} ({v['n']})" for r, v in hf["rotation_by_row"].items()))


NETWORK_HELP = """\
The shape table was saved; only the summer features are missing. They need PyTorch and torchvision:
- torchvision missing: install it with the same command from pytorch.org that installed your PyTorch
  (with its --index-url), so that pip does not replace your GPU build with another one.
- PyTorch fails to load (e.g. WinError 193 or 126): see 'If PyTorch does not load' in
  docs/results/align/README.md.
- The weights could not be downloaded (about 45 MB, once): download
  https://download.pytorch.org/models/resnet18-f37072fd.pth in a browser and put it in the
  'checkpoints' folder of torch's cache (python -c "import torch; print(torch.hub.get_dir())"),
  then run the command again with --features resnet18 (the traces are measured again, about 2 minutes)."""


def main(argv: Sequence[str] | None = None) -> int:
    """Measure the development trees' shape and compute their summer features.

    Usage: ``python -m treeskel.shape DATA_DIR [--out data/shape] [--features resnet18|none]
    [--device cuda|cpu] [--figure docs/figures/shape_lines.jpg] [--jobs N]``.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset, load_tree
    from treeskel.winter_gt import report_failed, save_image

    ap = argparse.ArgumentParser(prog="python -m treeskel.shape", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json", help="its test rows are never measured")
    ap.add_argument("--out", default="data/shape", help="folder for the table and the features (committed)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="trees traced in parallel")
    ap.add_argument("--features", choices=("resnet18", "none"), default="resnet18",
                    help="summer image features to compute (needs PyTorch and torchvision)")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)   # tests only
    ap.add_argument("--figure", help="also save the fitted lines on a few trees as this image")
    ap.add_argument("--figure-trees", nargs="+", help="trees for the figure (default: six spread over the rows)")
    args = ap.parse_args(argv)

    split = json.loads(Path(args.split).read_text())
    test_rows = set(split["test_rows"])
    clicks = load_clicks(args.clicks)
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    ids = sorted(t for t, c in clicks.items()
                 if t in records and c.complete and not c.skip and records[t].row not in test_rows)
    n_test = sum(records[t].row in test_rows for t in records)
    if not ids:
        print("No annotated development-row trees found; nothing to measure.")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = ShapeParams()
    fig_ids = set()
    if args.figure:
        fig_ids = set(args.figure_trees or ids[:: max(1, len(ids) // 6)][:6])
    jobs = [(records[t], clicks[t].normalised(), params, t in fig_ids) for t in ids]

    print(f"Measuring {len(ids)} development-row trees ({n_test} test-row trees left out)...", flush=True)
    t0 = time.time()
    step = max(1, min(25, len(jobs) // 10))
    rows, tiles = [], []
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
        for i, (row, tile) in enumerate(ex.map(measure_tree, jobs, chunksize=2), 1):
            rows.append(row)
            if tile is not None:
                tiles.append(tile)
            if i % step == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                      flush=True)
    failed: list[str] = []
    table = out / "shape_measures.csv"
    try:
        with open(table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    except OSError as e:                                  # e.g. open in a spreadsheet on Windows
        failed.append(f"{table} ({e})")
    summary = summarise(rows, params)
    summary["params"] = asdict(params)
    summary["test_rows_excluded"] = sorted(test_rows)
    (out / "shape_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary, params)
    if args.figure and tiles:
        from PIL import Image

        sheet = Image.new("RGB", (tiles[0].width, tiles[0].height * len(tiles)))
        for i, t in enumerate(tiles):
            sheet.paste(t, (0, i * t.height))
        save_image(sheet.resize((sheet.width // 2, sheet.height // 2)), Path(args.figure), failed, quality=85)

    if args.features != "none":
        print(f"\nComputing summer features ({args.features}) for {len(ids)} trees...", flush=True)
        try:
            body, device = make_resnet18(not args.random_weights, args.device)
        except (ImportError, OSError, RuntimeError) as e:    # PyTorch missing or broken, or a failed download
            print(f"Could not load the network: {type(e).__name__}: {e}\n{NETWORK_HELP}")
            report_failed(failed)
            return 1
        t0 = time.time()
        feats = []
        for i, t in enumerate(ids, 1):
            feats.append(tree_features(body, device, load_tree(records[t], "summer")))
            if i % step == 0 or i == len(ids):
                el = time.time() - t0
                print(f"  {i}/{len(ids)} trees, {el:.0f} s elapsed, about {el / i * (len(ids) - i):.0f} s left "
                      f"(device {device})", flush=True)
        meta = {"network": "resnet18", "weights": "random" if args.random_weights else "IMAGENET1K_V1",
                "layer": "layer4", "input_size": list(FEATURE_SIZE), "grid": GRID,
                "second_axis": ["photo", "mirror image"], "cells": "reading order"}
        path = out / f"summer_{args.features}.npz"
        try:
            save_features(path, ids, feats, meta)
            print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except OSError as e:
            failed.append(f"{path} ({e})")
    print(f"Saved {table} and {out / 'shape_summary.json'}" + (f" and {args.figure}" if args.figure else ""))
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
