"""Click-guided main-branch ground truth from winter images.

The automatic method (:mod:`treeskel.winter_gt`) is kept as a baseline. It
failed on many trees: posts, straight unbranched trees, gaps in depth, support
wires, and branches that cross. Here a person supplies a few clicks per tree,
and the branches are traced between them automatically:

1. **Clicks** (:class:`Clicks`): the trunk base, the fork, and the tip of each
   of the two V leaders, plus optional waypoints where a trace needs guiding.
2. **Cost map** (:func:`branch_cost`): each pixel gets a "woodiness" in
   [0, 1] from three cues, and a cost ``floor + (1 - woodiness)^power``:

   - depth: the pixel stands out from the ground or background
     (:func:`treeskel.winter_gt.foreground_mask`, smoothed), weighted by how
     central it is within the object (distance to the edge divided by the
     largest such distance nearby), so paths run down the middle of thick
     trunks rather than along an edge;
   - tube shape: the Frangi vesselness filter, which uses the eigenvalues of
     the image's Hessian (second derivatives) to respond to bright, elongated
     structures at the chosen widths;
   - colour: bark is pale and unsaturated, unlike grass and soil.

3. **Tracing** (:func:`trace`): between consecutive clicks, the path with the
   smallest total cost is found by Dijkstra's algorithm over the 8-connected
   pixel grid (diagonal steps cost √2 × the pixel cost). Cheap woody pixels
   pull the path onto the branch, and short gaps are crossed where needed.
4. **Smoothing** (:func:`smooth_segment`): the pixel path zig-zags by a
   pixel or two (bark texture, 8-connected steps), so each stretch between
   clicks is smoothed with the clicks held fixed.
5. **Widths** (:func:`fit_taper`): the trunk's width is measured from depth,
   and the leaders' from the colour image by scale-space ridge detection
   (:func:`scale_space_radius`), because depth roughly doubles thin
   branches' widths. Point measurements are noisy, so each part's radius is
   fitted as a straight-line taper along its length by robust Huber
   regression (:func:`huber_line`), within plausible limits.
6. **Result** (:func:`guided_gt`): a :class:`treeskel.winter_gt.WinterGT`, so
   the same rendering, shape measures and overlays apply to both methods.

Clicks are stored in a JSON file keyed by tree id (see :func:`save_clicks`),
so the ground truth can be regenerated exactly with
``python -m treeskel.guided_gt DATA_DIR --clicks annotations/winter_clicks.json``.
Clicks are made with :mod:`treeskel.annotate`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.filters import frangi
from skimage.draw import line as draw_line
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.graph import route_through_array
from skimage.morphology import disk

from treeskel.winter_gt import (
    WinterGT,
    foreground_mask,
    overlay,
    path_shape,
    render_main_branches,
    report_failed,
    save_image,
)

Point = tuple[int, int]


@dataclass(frozen=True)
class GuidedParams:
    """Settings for the cost map.

    The defaults were chosen by :mod:`treeskel.tune_cost` on 17 annotated
    trees (64 correction clicks). The first defaults (depth 0.5, tube 0.3,
    colour 0.2, power 2, centre weight 0.5) made wood almost free, so paths
    wandered within thick trunks and needed many corrections: 62% of the
    correction clicks lay more than 8 px from the trace made from the main
    clicks alone. The tuned setting keeps wood only moderately cheaper than
    background, so path length still matters, and fully favours the centre
    of each branch: 6% of correction clicks were more than 8 px off (11% in
    leave-one-out cross-validation). The colour cue got weight 0.

    Attributes:
        depth_weight: Weight of the depth cue in woodiness.
        tube_weight: Weight of the Frangi tube cue.
        colour_weight: Weight of the pale-bark colour cue.
        sigmas: Frangi filter scales (Gaussian sigma, px), roughly half the
            branch widths to respond to.
        cost_floor: Smallest pixel cost; keeps paths from taking long detours
            through very woody areas.
        cost_power: Exponent on (1 − woodiness); higher values stick more
            tightly to woody pixels.
        centre_window: Half-size (px) of the neighbourhood used to judge how
            central a pixel is within its object; should exceed the widest
            trunk's half-width.
        centre_weight: How much the depth cue favours an object's centre line
            (0 = not at all, 1 = only the centre line counts fully).
        centre_source: ``"depth"`` to judge centredness on the depth
            foreground, or ``"wood"`` to judge it on the combined cues.
        wood_threshold: Woodiness above which a pixel counts as wood when
            ``centre_source="wood"``.
        smooth_sigma: Width (px) of the Gaussian that smooths each traced
            stretch between clicks (:func:`smooth_segment`); 0 = no smoothing.
            Not used by the cost map, so :mod:`treeskel.tune_cost` results
            are unaffected.
        leader_width_source: ``"colour"`` to measure leader widths from the
            RGB image by scale-space ridge detection
            (:func:`scale_space_radius`), or ``"depth"`` to use the depth
            foreground like the trunk. Depth "bleeds" a few pixels past
            branch edges, which roughly doubles thin leaders' widths, so
            colour is the default. Trunks always use depth: a trunk lit on
            one side makes the colour estimate lock onto the bright half.
        leader_scales: Range (px) of Gaussian scales searched for leaders.
        scale_window: Half-size (px) of the neighbourhood searched around
            each path point, so a slightly off-centre path still finds the
            branch.
    """

    depth_weight: float = 0.2
    tube_weight: float = 0.1
    colour_weight: float = 0.0
    sigmas: tuple[float, ...] = (2, 4, 6, 8, 10)
    cost_floor: float = 0.02
    cost_power: float = 1.0
    centre_window: int = 15
    centre_weight: float = 1.0
    centre_source: str = "depth"
    wood_threshold: float = 0.45
    smooth_sigma: float = 6.0
    leader_width_source: str = "colour"
    leader_scales: tuple[float, float] = (1.5, 12.0)
    scale_window: int = 4


@dataclass
class Clicks:
    """Clicked points for one tree, as ``(row, col)`` pixels.

    Attributes:
        trunk: ``[root, waypoints..., fork]`` (at least two points).
        left: ``[waypoints..., tip]`` for the image-left leader; traced from
            the fork. If the tips were clicked in the other order,
            :meth:`normalised` swaps the leaders.
        right: ``[waypoints..., tip]`` for the image-right leader.
        skip: True if the tree cannot be annotated (it is then excluded).
        note: Free-text reason or comment.
    """

    trunk: list[Point] = field(default_factory=list)
    left: list[Point] = field(default_factory=list)
    right: list[Point] = field(default_factory=list)
    skip: bool = False
    note: str = ""

    @property
    def complete(self) -> bool:
        """True if there is enough to trace: root, fork and both tips."""
        return len(self.trunk) >= 2 and bool(self.left) and bool(self.right)

    @property
    def fork(self) -> Point:
        """The fork (last trunk point)."""
        return self.trunk[-1]

    def normalised(self) -> "Clicks":
        """Copy with the leaders swapped if the "left" tip is right of the other.

        "Left" always means the leader whose tip is further left in the image,
        whatever order the tips were clicked in.
        """
        if self.left and self.right and self.left[-1][1] > self.right[-1][1]:
            return Clicks(trunk=list(self.trunk), left=list(self.right), right=list(self.left),
                          skip=self.skip, note=self.note)
        return self

    def to_json(self) -> dict:
        """Plain-data form for the JSON file."""
        d = asdict(self)
        for k in ("trunk", "left", "right"):
            d[k] = [list(map(int, p)) for p in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Clicks":
        """Inverse of :meth:`to_json`."""
        return cls(
            trunk=[tuple(p) for p in d.get("trunk", [])],
            left=[tuple(p) for p in d.get("left", [])],
            right=[tuple(p) for p in d.get("right", [])],
            skip=bool(d.get("skip", False)),
            note=str(d.get("note", "")),
        )


def load_clicks(path: str | Path) -> dict[str, Clicks]:
    """Read a clicks file; a missing file gives an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    return {k: Clicks.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_clicks(path: str | Path, clicks: dict[str, Clicks]) -> None:
    """Write a clicks file atomically (a crash never leaves it half-written)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: clicks[k].to_json() for k in sorted(clicks)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _centredness(mask: np.ndarray, window: int) -> np.ndarray:
    """Distance to the mask edge divided by the largest such distance nearby.

    1 on an object's centre line, falling towards 0 at its edges, whatever the
    object's width (up to ``window`` px half-width).
    """
    edt = ndi.distance_transform_edt(mask)
    local_max = ndi.maximum_filter(edt, size=2 * window + 1)
    return np.divide(edt, local_max, out=np.zeros_like(edt), where=local_max > 0)


def cost_cues(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
              params: GuidedParams = GuidedParams()) -> dict[str, np.ndarray]:
    """The expensive, weight-independent parts of the cost map.

    Computed once per image so that many weightings can be tried cheaply
    (see :func:`combine_cues` and :mod:`treeskel.tune_cost`).

    Args:
        rgb, depth, valid: The winter images.
        params: Uses ``sigmas`` and ``centre_window``.

    Returns:
        Dict of ``(H, W)`` float arrays in [0, 1]: ``"fg"`` (smoothed depth
        foreground), ``"fg_centred"`` (its centredness), ``"tube"`` (Frangi
        response) and ``"pale"`` (bark colour).
    """
    mask = foreground_mask(depth, valid)
    grey = rgb.astype(float).mean(2) / 255.0
    tube = frangi(grey, sigmas=params.sigmas, black_ridges=False)
    scale = np.percentile(tube, 99.5)
    tube = np.clip(tube / scale, 0, 1) if scale > 0 else np.zeros_like(tube)
    hsv = rgb2hsv(rgb)
    pale = np.clip((hsv[..., 2] - 0.35) / 0.5, 0, 1) * np.clip(1 - hsv[..., 1] / 0.45, 0, 1)
    return {
        "fg": ndi.gaussian_filter(mask.astype(float), 1.0),
        "fg_centred": _centredness(mask, params.centre_window),
        "tube": tube,
        "pale": pale,
    }


def combine_cues(cues: dict[str, np.ndarray],
                 params: GuidedParams = GuidedParams()) -> tuple[np.ndarray, np.ndarray]:
    """Weight the cues into woodiness and cost.

    ``woodiness = w_depth · fg · (1 − c + c · centred) + w_tube · tube +
    w_colour · pale`` (clipped to [0, 1]), where ``c`` is ``centre_weight``.
    ``centred`` is the centredness of the depth foreground
    (``centre_source="depth"``) or of the pixels whose woodiness without
    centring exceeds ``wood_threshold`` (``centre_source="wood"``). The
    latter uses colour and shape too, so ground pixels that merge into the
    trunk base in depth do not pull the centre line sideways.

    Returns:
        ``(cost, woodiness)``; cost is ``cost_floor + (1 − woodiness)^cost_power``.
    """
    p = params
    base = p.tube_weight * cues["tube"] + p.colour_weight * cues["pale"]
    if p.centre_source == "wood":
        rough = np.clip(p.depth_weight * cues["fg"] + base, 0, 1)
        centred = _centredness(rough > p.wood_threshold, p.centre_window)
    else:
        centred = cues["fg_centred"]
    depth_term = cues["fg"] * (1 - p.centre_weight + p.centre_weight * centred)
    wood = np.clip(p.depth_weight * depth_term + base, 0, 1)
    return p.cost_floor + (1 - wood) ** p.cost_power, wood


def branch_cost(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                params: GuidedParams = GuidedParams()) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel woodiness and tracing cost for a winter image.

    Args:
        rgb: ``(H, W, 3)`` uint8 colour image.
        depth: ``(H, W)`` 8-bit depth (brighter = nearer).
        valid: ``(H, W)`` bool validity mask.
        params: Cue weights and cost shape.

    Returns:
        ``(cost, woodiness)``, both ``(H, W)`` float arrays; ``woodiness`` is in
        [0, 1] and ``cost`` in ``[cost_floor, cost_floor + 1]``.
    """
    return combine_cues(cost_cues(rgb, depth, valid, params), params)


def smooth_segment(seg: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a traced segment while keeping both end points exactly fixed.

    The segment is split into the straight line between its end points plus
    a residual (the path's sideways wiggle). Only the residual is smoothed,
    with a Gaussian of width ``sigma`` (in path steps, about pixels), after
    an odd reflection at each end, which keeps the residual zero there. Real
    bends are kept, pixel-scale zig-zags are removed, and the result is
    re-drawn as a connected 8-connected pixel path.

    Args:
        seg: ``(M, 2)`` connected path from one click to the next.
        sigma: Smoothing width; 0 returns ``seg`` unchanged.

    Returns:
        ``(K, 2)`` int connected path with the same first and last points.
    """
    if sigma <= 0 or len(seg) < 5:
        return seg
    p = seg.astype(float)
    step = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(step)])
    t /= t[-1] if t[-1] > 0 else 1.0
    line = p[0] + t[:, None] * (p[-1] - p[0])
    res = p - line                                      # zero at both ends
    k = min(len(res) - 1, int(4 * sigma) + 1)
    ext = np.concatenate([-res[k:0:-1], res, -res[-2:-k - 2:-1]])  # odd reflection
    sm = ndi.gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[k:k + len(res)]
    sm[0] = sm[-1] = 0.0
    pts = np.round(line + sm).astype(int)
    out = [tuple(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        if tuple(b) != out[-1]:
            rr, cc = draw_line(out[-1][0], out[-1][1], int(b[0]), int(b[1]))
            out.extend(zip(rr[1:].tolist(), cc[1:].tolist()))
    return np.array(out, int)


def trace(cost: np.ndarray, points: Sequence[Point], smooth_sigma: float = 0.0) -> np.ndarray:
    """Cheapest 8-connected path through ``points`` in order.

    Args:
        cost: ``(H, W)`` positive cost map.
        points: Two or more ``(row, col)`` points.
        smooth_sigma: If > 0, each stretch between consecutive points is
            smoothed by :func:`smooth_segment`; the points themselves stay on
            the path.

    Returns:
        ``(N, 2)`` int array of path pixels from the first to the last point,
        passing through every point.
    """
    h, w = cost.shape
    pts = [(min(max(int(r), 0), h - 1), min(max(int(c), 0), w - 1)) for r, c in points]
    out: list[tuple[int, int]] = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if a == b:
            continue
        seg, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
        seg = smooth_segment(np.array(seg, int), smooth_sigma)
        seg[:, 0] = seg[:, 0].clip(0, h - 1)
        seg[:, 1] = seg[:, 1].clip(0, w - 1)
        out.extend(tuple(q) for q in seg[1:].tolist())
    return np.array(out, int).reshape(-1, 2)


def guided_gt(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, clicks: Clicks,
              params: GuidedParams = GuidedParams(), cost: np.ndarray | None = None) -> WinterGT:
    """Trace the trunk and both leaders through the clicked points.

    Args:
        rgb, depth, valid: The tree's winter images.
        clicks: Complete :class:`Clicks` (not skipped).
        params: Cost-map settings (ignored if ``cost`` is given).
        cost: Precomputed cost map, to avoid recomputing it while clicking.

    Returns:
        A :class:`treeskel.winter_gt.WinterGT` with ``ok=True``. Its
        ``tree_mask`` is the depth foreground, used for branch widths.

    Raises:
        ValueError: If the clicks are skipped or incomplete.
    """
    if clicks.skip or not clicks.complete:
        raise ValueError("clicks must be complete and not skipped")
    clicks = clicks.normalised()
    if cost is None:
        cost, _ = branch_cost(rgb, depth, valid, params)
    fg = ndi.binary_closing(foreground_mask(depth, valid), disk(2))
    sm = params.smooth_sigma
    trunk = trace(cost, clicks.trunk, sm)
    left = trace(cost, [clicks.fork, *clicks.left], sm)
    right = trace(cost, [clicks.fork, *clicks.right], sm)
    radius = ndi.distance_transform_edt(fg)
    skel = np.zeros(fg.shape, bool)
    for p in (trunk, left, right):
        skel[p[:, 0], p[:, 1]] = True
    measured = {k: radius[p[:, 0], p[:, 1]] for k, p in (("trunk", trunk), ("left", left), ("right", right))}
    if params.leader_width_source == "colour":
        grey = rgb.astype(float).mean(2) / 255.0
        both = np.concatenate([left, right])
        r = scale_space_radius(grey, both, params.leader_scales, params.scale_window)
        measured["left"], measured["right"] = r[:len(left)], r[len(left):]
    radii = {k: fit_taper(p, measured[k], k) for k, p in (("trunk", trunk), ("left", left), ("right", right))}
    # A leader cannot start wider than the trunk it grows from.
    fork_r = radii["trunk"][-1] if len(radii["trunk"]) else FALLBACK_RADIUS["trunk"]
    for k in ("left", "right"):
        radii[k] = np.minimum(radii[k], fork_r)
    return WinterGT(
        tree_mask=fg, skeleton=skel, radii=radii,
        root=tuple(map(int, trunk[0])), fork=tuple(map(int, trunk[-1])),
        trunk=trunk, left=left, right=right,
        trunk_shape=path_shape(trunk, radius), left_shape=path_shape(left, radius),
        right_shape=path_shape(right, radius), ok=True,
    )


RADIUS_LIMITS = {"trunk": (4.0, 25.0), "left": (2.5, 15.0), "right": (2.5, 15.0)}
"""Plausible drawing radius range (px) for each part."""

FALLBACK_RADIUS = {"trunk": 10.0, "left": 5.0, "right": 5.0}
"""Radius used when too few width measurements survive (px)."""


def huber_line(x: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 30) -> tuple[float, float]:
    """Robust straight-line fit ``y ≈ a + b·x`` by Huber regression.

    Iteratively reweighted least squares: residuals within ``k`` robust
    standard deviations (scale = 1.4826 × median absolute residual) get full
    weight, larger ones weight ``k·σ/|r|``, so outliers pull the line far less
    than in ordinary least squares.

    Returns:
        ``(a, b)``; ``b = 0`` if ``x`` has no spread.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0:
        return float(np.median(y)), 0.0
    X = np.stack([np.ones_like(x), x], 1)
    w = np.ones_like(y)
    beta = np.zeros(2)
    for _ in range(iters):
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        res = y - X @ new
        scale = 1.4826 * np.median(np.abs(res)) or 1e-9
        u = np.abs(res) / (k * scale)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        if np.allclose(new, beta, atol=1e-6):
            break
        beta = new
    return float(new[0]), float(new[1])


def scale_space_radius(grey: np.ndarray, points: np.ndarray, scales: tuple[float, float] = (1.5, 12.0),
                       window: int = 4, ratio: float = 1.2) -> np.ndarray:
    """Branch half-width at each point, by scale-space ridge detection.

    The image is smoothed with Gaussians of width σ over a geometric range of
    scales. At each scale the larger-magnitude eigenvalue of the Hessian
    (curvature across a branch), multiplied by σ² (scale normalisation), is
    computed; its sign is ignored, so bark lighter than grass and bark
    darker than sky both count. For a bar of half-width *a* this normalised
    response peaks at σ = *a* (the maximum of 2u·φ(u) is at u = a/σ = 1), so
    the scale of the strongest response within ``window`` px of each point
    is taken as its half-width. On synthetic cylinder-shaded bars of
    half-width 3–12 px the error is at most 0.5 px.

    Only a box around the points (padded by 3σ) is filtered, to save time.

    Args:
        grey: ``(H, W)`` float image.
        points: ``(N, 2)`` ``(row, col)`` points.
        scales: Smallest and largest σ searched (px).
        window: Search half-size around each point (px).
        ratio: Ratio between successive scales.

    Returns:
        ``(N,)`` half-widths (px).
    """
    if len(points) == 0:
        return np.zeros(0)
    sig = scales[0] * ratio ** np.arange(int(np.log(scales[1] / scales[0]) / np.log(ratio)) + 1)
    pad = int(3 * sig[-1]) + window + 1
    r0, c0 = np.maximum(points.min(0) - pad, 0)
    r1, c1 = np.minimum(points.max(0) + pad + 1, grey.shape)
    crop = grey[r0:r1, c0:c1].astype(float)
    best_resp = np.full(len(points), -np.inf)
    best_sig = np.full(len(points), sig[0])
    pr, pc = points[:, 0] - r0, points[:, 1] - c0
    for s in sig:
        h = hessian_matrix(crop, sigma=s, order="rc", use_gaussian_derivatives=True)
        ev = hessian_matrix_eigvals(h)
        resp = ndi.maximum_filter(s * s * np.maximum(np.abs(ev[0]), np.abs(ev[1])), size=2 * window + 1)
        v = resp[pr, pc]
        better = v > best_resp
        best_resp[better], best_sig[better] = v[better], s
    return best_sig


def fit_taper(path: np.ndarray, measured: np.ndarray, part: str,
              min_valid: float = 2.5, min_points: int = 8) -> np.ndarray:
    """Smooth, tapering radius along one part from noisy width measurements.

    Width measured from depth is unreliable point by point: depth gaps give
    near-zero widths (drawn as thin lines), and ground or background merging
    with the branch gives large blobs. Treating the measurements as noisy
    observations, radius is modelled as linear in arc length along the path
    (a branch tapers gradually) and fitted by :func:`huber_line`, ignoring
    measurements at or below ``min_valid`` (gaps). A fit that widens upwards
    is physically implausible, so it is replaced by the median width
    (constant). The result is clipped to
    :data:`RADIUS_LIMITS`. With fewer than ``min_points`` usable
    measurements, :data:`FALLBACK_RADIUS` is used.

    Args:
        path: ``(N, 2)`` path points, in order.
        measured: ``(N,)`` measured half-widths at those points (px).
        part: ``"trunk"``, ``"left"`` or ``"right"``.

    Returns:
        ``(N,)`` radii to draw.
    """
    lo, hi = RADIUS_LIMITS[part]
    if len(path) == 0:
        return np.zeros(0)
    step = np.hypot(*np.diff(path, axis=0).T) if len(path) > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(step)])
    ok = measured > min_valid
    if ok.sum() < min_points:
        return np.full(len(path), FALLBACK_RADIUS[part])
    a, b = huber_line(s[ok], measured[ok])
    if b > 0:
        # Branches do not widen upwards: use a constant (robust) width instead.
        a, b = float(np.median(measured[ok])), 0.0
    return np.clip(a + b * s, lo, hi)


def insert_waypoint(clicks: Clicks, gt: WinterGT, point: Point) -> Clicks:
    """Add a correcting click to the part of the trace it is nearest to.

    The point joins the trunk, left or right control points, whichever traced
    path passes closest to it, at the position matching its place along that
    path. The trunk's root and fork, and the leaders' tips, stay fixed.

    Args:
        clicks: Current complete clicks.
        gt: The current trace (from :func:`guided_gt`).
        point: The new ``(row, col)`` click.

    Returns:
        New :class:`Clicks` with the waypoint inserted.
    """
    p = np.asarray(point)
    best = None
    for name in ("trunk", "left", "right"):
        path = getattr(gt, name)
        if len(path):
            d = np.hypot(*(path - p).T)
            i = int(np.argmin(d))
            if best is None or d[i] < best[0]:
                best = (d[i], name, i, path)
    _, name, idx, path = best
    ctrl = list(getattr(clicks, name))

    def pos(q: Point) -> int:
        return int(np.argmin(np.hypot(*(path - np.asarray(q)).T)))

    # Insert before the first control point that lies further along the path.
    # The trunk's first point (root) is fixed, so start after it.
    start = 1 if name == "trunk" else 0
    k = next((j for j in range(start, len(ctrl)) if pos(ctrl[j]) > idx), len(ctrl))
    if name == "trunk":
        k = min(max(k, 1), len(ctrl) - 1)   # keep root first and fork last
    else:
        k = min(k, len(ctrl) - 1)           # keep the tip last
    ctrl.insert(k, (int(point[0]), int(point[1])))
    new = Clicks(**{**clicks.__dict__})
    setattr(new, name, ctrl)
    return new


def check_clicks(c: Clicks, min_rise: int = 40, min_spread: int = 40) -> list[str]:
    """Geometric sanity checks on a tree's clicks (image rows grow downwards).

    Flags click-order slips that cannot be right for a Tatura V tree: the fork
    not clearly (10 px) above the root, a tip not clearly above the fork, or the two tips
    almost on top of each other. It cannot catch an accepted trace that
    follows the wrong branch; that needs looking at the overlay.

    Returns:
        Human-readable problems (empty if none, or if skipped/incomplete).
    """
    if c.skip or not c.complete:
        return []
    c = c.normalised()
    (rr, _), (fr, _) = c.trunk[0], c.fork
    issues = []
    if rr - fr < 10 and "splits at ground" not in c.note:
        issues.append(f"fork only {rr - fr} px above the root (double click, or a tree "
                      "that splits at the ground?)")
    for side in ("left", "right"):
        tip = getattr(c, side)[-1]
        if fr - tip[0] < min_rise:
            issues.append(f"{side} tip only {fr - tip[0]} px above the fork")
    if c.right[-1][1] - c.left[-1][1] < min_spread:
        issues.append("leader tips are close together")
    return issues


def _render_one(job: tuple) -> tuple[str, dict, list[str]]:
    """Render one tree (top-level so worker processes can run it).

    Args:
        job: ``(tree_record, clicks, params, out_dir, overlays)``.

    Returns:
        ``(tree_id, summary_row, failed_writes)``.
    """
    from treeskel.io import load_tree

    rec, c, params, out, overlays = job
    failed: list[str] = []
    d = load_tree(rec, "winter")
    gt = guided_gt(d["rgb"], d["depth"], d["valid"], c, params)
    save_image(Image.fromarray(render_main_branches(gt).astype(np.uint8) * 255),
               out / f"{rec.tree_id}_winter_gt.png", failed)
    if overlays:
        save_image(overlay(d["rgb"], gt, rec.tree_id), out / f"{rec.tree_id}_overlay.jpg",
                   failed, quality=85)
    return rec.tree_id, {"tree_id": rec.tree_id, **gt.summary()}, failed


def main(argv: Sequence[str] | None = None) -> int:
    """Render guided ground truth for every annotated tree.

    Usage: ``python -m treeskel.guided_gt DATA_DIR --clicks annotations/winter_clicks.json
    --out outputs/winter_gt_guided [--overlays] [--rows 5 6] [--trees R05N12]
    [--jobs N]``.

    Most of the time per tree (about 1 s) is the Frangi filter in the cost
    map, so trees are processed in parallel, one per CPU core by default,
    and progress is printed as they finish.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.guided_gt", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--out", default="outputs/winter_gt_guided", help="output folder (git-ignored)")
    ap.add_argument("--overlays", action="store_true", help="also save <tree>_overlay.jpg")
    ap.add_argument("--rows", type=int, nargs="+", help="only trees in these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="trees processed in parallel (default: number of CPU cores)")
    args = ap.parse_args(argv)

    clicks = load_clicks(args.clicks)
    trees = {t.tree_id: t for t in index_dataset(args.data_dir).trees}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = GuidedParams()
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))

    wanted = sorted(clicks)
    if args.rows:
        wanted = [t for t in wanted if int(t[1:3]) in args.rows]
    if args.trees:
        wanted = [t for t in wanted if t in set(args.trees)]
    skipped = [t for t in wanted if clicks[t].skip or not clicks[t].complete]
    missing = [t for t in wanted if t not in skipped and t not in trees]
    jobs = [(trees[t], clicks[t], params, out, args.overlays)
            for t in wanted if t not in skipped and t not in missing]

    rows, failed = [], []
    t0 = time.time()
    step = max(1, min(25, len(jobs) // 10))

    def progress(done: int) -> None:
        if done % step == 0 or done == len(jobs):
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"  {done}/{len(jobs)} trees, {el:.0f} s elapsed, about {eta:.0f} s left", flush=True)

    print(f"Rendering {len(jobs)} trees using {max(1, min(args.jobs, len(jobs)))} process(es)...", flush=True)
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, (_, row, f) in enumerate(ex.map(_render_one, jobs, chunksize=4), 1):
                rows.append(row)
                failed += f
                progress(i)
    else:
        for i, job in enumerate(jobs, 1):
            _, row, f = _render_one(job)
            rows.append(row)
            failed += f
            progress(i)

    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    suspect = {t: check_clicks(clicks[t]) for t in wanted if check_clicks(clicks[t])}
    for tid, why in sorted(suspect.items()):
        print(f"Check clicks for {tid}: {'; '.join(why)}")
    print(f"Rendered {len(rows)} trees in {time.time() - t0:.0f} s; {len(skipped)} skipped or "
          f"incomplete; {len(missing)} not found in {args.data_dir}. Results in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
