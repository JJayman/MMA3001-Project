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
5. **Widths**: the trunk's width is measured from depth and fitted as a
   straight-line taper along its length by robust Huber regression
   (:func:`fit_taper`, :func:`huber_line`). Each leader's width is measured
   across the branch in the colour image (:func:`branch_edges`): every
   cross-section is split into background, branch and background by least
   squares, so both the sunlit and the shaded half of a branch lit from one
   side count. Depth is not used for leaders because it reaches about 2 px
   past each edge and merges with things behind. Each leader is drawn at its
   own robust width, widened near the fork by an amount shared by all
   leaders that fades with distance (:func:`fit_leader_radius`).
6. **Fork** (:func:`end_cut`, :func:`fork_junction`): drawn as discs, the
   trunk would end in a half disc of its own radius, which shows as a round
   knob where the much thinner leaders leave it. So the trunk is cut flat at
   the fork (and at a root click that is well above the image's bottom
   edge), and each leader is joined to the trunk's edge on its outer side
   by a straight line a little way up the leader; the crotch between the
   leaders is left as the leaders draw it.
   The older :func:`shoulder_fill` (off by default since session 8) closed
   the notch beside each leader but kept the round top; it was needed
   while the leaders were drawn about half as wide as their bark.
7. **Result** (:func:`guided_gt`): a :class:`treeskel.winter_gt.WinterGT`, so
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
from scipy.spatial import cKDTree
from skimage.color import rgb2hsv
from skimage.filters import frangi
from skimage.draw import disk as draw_disk
from skimage.draw import line as draw_line
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.graph import route_through_array
from skimage.morphology import convex_hull_image, disk

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
    """Settings for the cost map and for drawing the branches.

    The cost-map defaults were chosen by :mod:`treeskel.tune_cost` on 17 annotated
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
        leader_width_source: How leader widths are measured: ``"profile"``
            (default) finds both edges of the branch in colour cross-sections
            (:func:`branch_edges`); ``"scale_space"`` uses scale-space ridge
            detection (:func:`scale_space_radius`, the default until session
            8); ``"depth"`` uses the depth foreground like the trunk. On the
            21 development trees in the repository, ``"scale_space"`` gave
            about half the width of the bark: it responds to the sunlit half
            of a branch lit from one side. Depth reaches a median 2.0 px past
            clear bark edges; with that removed, it agrees with
            ``"profile"`` (median difference −0.4 px per leader), see
            :mod:`treeskel.measure_width`. Trunks always use depth.
        leader_scales: Range (px) of Gaussian scales searched for leaders
            (``"scale_space"`` only).
        scale_window: Half-size (px) of the neighbourhood searched around
            each path point, so a slightly off-centre path still finds the
            branch (``"scale_space"`` only).
        profile_half_width: How far (px) either side of the path each
            cross-section reaches (``"profile"`` only); wider than the widest
            leader plus a few pixels of background.
        profile_along: Cross-sections are averaged over this many path
            points either side (``"profile"`` only), which evens out bark
            texture and background clutter. Longer averages blurred the edges
            where leaders curve.
        leader_fit: ``"widening"`` (default) draws each leader at its own
            width, widened near the fork (:func:`fit_leader_radius`);
            ``"taper"`` fits a free straight taper per leader
            (:func:`fit_taper`, the method until session 8). Free tapers are
            unreliable: on the 21 development trees here, half of the leaders
            came out widening upwards (28% on all 449 development trees).
        fork_widening: How much wider a leader is at the fork than its own
            width further up (a fraction); 0 turns the widening off.
        widening_length: Distance (in trunk radii at the fork) over which
            the widening falls by a factor e. ``fork_widening`` and
            ``widening_length`` are shared by all leaders, and were chosen on
            all 449 development trees (:mod:`treeskel.measure_width`, grid up
            to 1.0 and 48 trunk radii): +35% at the fork fading over 12 trunk
            radii, so a leader narrows gradually along its length (+31% 1.5
            trunk radii from the fork, +15% at 10, +3% at 30). Leave-one-tree-
            out picked 0.30–0.35 and 12; the 5 row-grouped folds of
            ``splits.json`` picked 0.30–0.35 and 8–12, and on every fold the
            shape beat a constant width on held-out rows. (A first fit on 21
            trees, 17 of them the problem trees used for tuning, had chosen a
            length of 4.)
        leader_fit_start: A leader's own width is estimated from points at
            least this many trunk radii from the fork; nearer, the
            cross-section cuts through the top of the trunk. Cross-sections
            that could reach the other leader are not used either
            (:func:`clear_of`); on the development trees the first usable
            one is a median 2.1 trunk radii from the fork.
        shoulder_length: How far up each leader the shoulder fill reaches
            (:func:`shoulder_fill`), in multiples of the trunk's radius at
            the fork; 0 (the default) turns it off. Session 6 chose 1.4 with
            :mod:`treeskel.measure_fork`, when the leaders were drawn about
            half as wide as their bark and left a deep notch beside the
            trunk. With the leaders at their true width (session 8), the
            masks without any fill already cover the shoulder between the
            depth silhouette and the silhouette minus its measured 2 px
            overshoot, on all 449 development trees (0.914 against
            0.875–0.938 at 1.5 trunk radii), while 1.4 covered more than
            depth did. So the student turned it off, for the moment; 1.4
            restores the session-6 masks' fill (see
            ``docs/results/fork_shoulders/``). Not used by the cost map.
        junction_reach: How the trunk meets the leaders, in multiples of
            the trunk's radius at the fork; 0 keeps the round-ended trunk
            of the masks up to session 8. Drawn as discs, the trunk ended in
            a half disc of its own radius, and at the fork that round top
            showed between and beside the thinner leaders as a knob (the
            student's observation in session 8; the hand masks have none).
            The trunk is now cut flat at the fork (:func:`end_cut`), and on
            each leader's outer side the trunk's edge at the fork is joined
            by a straight line to the leader's edge this far up the leader
            (:func:`fork_junction`); the crotch between the leaders is left
            as the leaders draw it. 1.0 was chosen with
            :mod:`treeskel.measure_fork` on the 21 development trees in the
            repository, by the criterion that turned the shoulder fill off:
            the joins should cover the shoulder beside each leader about as
            much as the wood does, which lies between the raw depth
            silhouette and the silhouette minus its 2 px overshoot. The
            matching reach was 0.98 and 1.30 against the raw silhouette (for
            shoulders reaching 1.5 and 2 trunk radii up the leader) and 0.5
            or less against the shrunk one; 1.5 covered more than the raw
            silhouette at both. On all 449 development trees the matching
            reaches were 1.04 and 1.23 (raw) and 0.62 and 0.59 (shrunk), and
            1.0 lay between the two references at both
            (``docs/results/fork_junction/``). Not used by the cost map.
        flat_root: Cut the trunk flat at the root click too
            (:func:`end_cut`), where its round end would otherwise be drawn
            whole on the ground below the click: on trees whose root click
            is more than the trunk's radius above the image's bottom edge
            (about 1 in 10 development trees). Most root clicks are a few
            pixels above that edge (median 4 px), where the trunk leaves the
            frame; there the round end reaches the edge and is kept, since a
            flat cut would leave a strip of background under the trunk.
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
    leader_width_source: str = "profile"
    leader_scales: tuple[float, float] = (1.5, 12.0)
    scale_window: int = 4
    profile_half_width: int = 20
    profile_along: int = 3
    leader_fit: str = "widening"
    fork_widening: float = 0.35
    widening_length: float = 12.0
    leader_fit_start: float = 1.5
    shoulder_length: float = 0.0
    junction_reach: float = 1.0
    flat_root: bool = True


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

        Returns:
            A new :class:`Clicks` with the leaders swapped, or these clicks if they are already in order.
        """
        if self.left and self.right and self.left[-1][1] > self.right[-1][1]:
            return Clicks(trunk=list(self.trunk), left=list(self.right), right=list(self.left),
                          skip=self.skip, note=self.note)
        return self

    def to_json(self) -> dict:
        """Plain-data form for the JSON file.

        Returns:
            ``{"trunk", "left", "right", "skip", "note"}`` with the points as lists.
        """
        d = asdict(self)
        for k in ("trunk", "left", "right"):
            d[k] = [list(map(int, p)) for p in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Clicks":
        """Inverse of :meth:`to_json`.

        Args:
            d: A dict as written by :meth:`to_json` (missing keys default to empty).

        Returns:
            The :class:`Clicks`.
        """
        return cls(
            trunk=[tuple(p) for p in d.get("trunk", [])],
            left=[tuple(p) for p in d.get("left", [])],
            right=[tuple(p) for p in d.get("right", [])],
            skip=bool(d.get("skip", False)),
            note=str(d.get("note", "")),
        )


def load_clicks(path: str | Path) -> dict[str, Clicks]:
    """Read a clicks file; a missing file gives an empty dict.

    Args:
        path: The JSON file (``annotations/winter_clicks.json``).

    Returns:
        The clicks per tree id.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return {k: Clicks.from_json(v) for k, v in json.loads(path.read_text()).items()}


def save_clicks(path: str | Path, clicks: dict[str, Clicks]) -> None:
    """Write a clicks file atomically (a crash never leaves it half-written).

    Args:
        path: The JSON file to write.
        clicks: The clicks per tree id.
    """
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
        rgb: The winter ``(480, 640, 3)`` uint8 photo.
        depth: Its ``(480, 640)`` uint8 depth.
        valid: Boolean mask of the depth's readings.
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

    Args:
        cues: The cues from :func:`cost_cues`.
        params: The weights and the centring settings.

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
        rgb: The tree's winter ``(480, 640, 3)`` uint8 photo.
        depth: Its ``(480, 640)`` uint8 depth.
        valid: Boolean mask of the depth's readings.
        clicks: Complete :class:`Clicks` (not skipped).
        params: Settings; the cost-map ones are ignored if ``cost`` is given.
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
    if params.leader_width_source == "profile":
        for k, p, other in (("left", left, right), ("right", right, left)):
            first, last = branch_edges(rgb, p, params.profile_half_width, params.profile_along)
            widths = (last - first + 1) / 2.0
            # Near the fork a cross-section can reach the other leader and take it in too.
            widths[~clear_of(p, other, params.profile_half_width + NEIGHBOUR_MARGIN)] = np.nan
            measured[k] = widths
    elif params.leader_width_source == "scale_space":
        grey = rgb.astype(float).mean(2) / 255.0
        both = np.concatenate([left, right])
        r = scale_space_radius(grey, both, params.leader_scales, params.scale_window)
        measured["left"], measured["right"] = r[:len(left)], r[len(left):]
    elif params.leader_width_source != "depth":
        raise ValueError(f"unknown leader_width_source {params.leader_width_source!r}")
    radii = {"trunk": fit_taper(trunk, measured["trunk"], "trunk")}
    fork_r = _fork_radius(radii)
    for k, p in (("left", left), ("right", right)):
        if params.leader_fit == "widening":
            radii[k] = fit_leader_radius(p, measured[k], fork_r, params.fork_widening, params.widening_length,
                                         params.leader_fit_start, part=k)
        elif params.leader_fit == "taper":
            radii[k] = fit_taper(p, measured[k], k)
        else:
            raise ValueError(f"unknown leader_fit {params.leader_fit!r}")
        # A leader cannot start wider than the trunk it grows from.
        radii[k] = np.minimum(radii[k], fork_r)
    gt = WinterGT(
        tree_mask=fg, skeleton=skel, radii=radii,
        root=tuple(map(int, trunk[0])), fork=tuple(map(int, trunk[-1])),
        trunk=trunk, left=left, right=right,
        trunk_shape=path_shape(trunk, radius), left_shape=path_shape(left, radius),
        right_shape=path_shape(right, radius), ok=True,
    )
    # Fill the notches between the trunk and each leader's outer side (none if shoulder_length is 0).
    gt.shoulders = fill_shoulders(gt, params.shoulder_length)
    cuts = []
    if params.junction_reach > 0:
        # Square the trunk's top off, and join it to each leader with straight sides.
        cuts.append(end_cut(trunk, radii["trunk"][-1]))
        gt.junction = fork_junction(gt, params.junction_reach)
    if params.flat_root and trunk[0, 0] + radii["trunk"][0] < fg.shape[0] - 1:
        # The round end would lie wholly in the image, on the ground below the click. (Nearer the
        # bottom edge, the trunk leaves the frame, and the round end reaches the edge: kept.)
        cuts.append(end_cut(trunk[::-1], radii["trunk"][0]))
    gt.trunk_cuts = [c for c in cuts if c is not None] or None
    return gt


RADIUS_LIMITS = {"trunk": (4.0, 25.0), "left": (2.5, 15.0), "right": (2.5, 15.0)}
"""Plausible drawing radius range (px) for each part."""

FALLBACK_RADIUS = {"trunk": 10.0, "left": 5.0, "right": 5.0}
"""Radius used when too few width measurements survive (px)."""

NEIGHBOUR_MARGIN = 10.0
"""Extra distance (px) beyond a cross-section's reach that the other leader must be,
for the cross-section to be used: about a leader's half-width plus a few pixels."""


def clear_of(path: np.ndarray, other: np.ndarray, distance: float) -> np.ndarray:
    """True for the points of ``path`` further than ``distance`` px from every point of ``other``.

    Used to drop cross-sections that could include the other leader: near
    the fork the two leaders are close together, and a cross-section across
    one then takes in the other as well. On a synthetic tree with leaders of
    constant width, such cross-sections measured the leaders 1.7–2.4 times too
    wide up to 3 trunk radii from the fork.

    Args:
        path: ``(N, 2)`` points.
        other: ``(M, 2)`` points of the other branch.
        distance: Clearance (px).

    Returns:
        ``(N,)`` boolean; all True if ``other`` is empty.
    """
    if len(path) == 0 or len(other) == 0:
        return np.ones(len(path), bool)
    return cKDTree(other).query(path)[0] > distance


def huber_line(x: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 30) -> tuple[float, float]:
    """Robust straight-line fit ``y ≈ a + b·x`` by Huber regression.

    Iteratively reweighted least squares: residuals within ``k`` robust
    standard deviations (scale = 1.4826 × median absolute residual) get full
    weight, larger ones weight ``k·σ/|r|``, so outliers pull the line far less
    than in ordinary least squares.

    Args:
        x: Positions.
        y: Values, one per position.
        k: Huber's tuning constant (in robust SDs).
        iters: Most reweighting rounds.

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


def huber_location(y: np.ndarray, k: float = 1.345, iters: int = 30) -> float:
    """Robust average of ``y`` by Huber's M-estimator (see :func:`huber_line`).

    Starts from the median; values more than ``k`` robust standard
    deviations away are down-weighted, so a minority of outliers (e.g.
    cross-sections that also caught a branch behind) barely move it.

    Args:
        y: Values.
        k: Huber's tuning constant (in robust SDs).
        iters: Most reweighting rounds.

    Returns:
        The robust average (the last estimate if ``iters`` rounds do not settle it).
    """
    y = np.asarray(y, float)
    m = float(np.median(y))
    for _ in range(iters):
        res = y - m
        scale = 1.4826 * np.median(np.abs(res)) or 1e-9
        u = np.abs(res) / (k * scale)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        new = float(np.sum(w * y) / np.sum(w))
        if abs(new - m) < 1e-6:
            return new
        m = new
    return m


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
    is taken as its half-width. On synthetic bars that are brighter than the
    background right to their edges, of half-width 3–12 px, the error is at
    most 0.5 px.

    Real bark is not like that: a branch lit from one side is bright on that
    side and dark on the other, and its edges are darker than its middle.
    On such (Lambertian) bars this returns about 0.55–0.6 of the half-width,
    and on the development trees about half the bark's width, so it is no
    longer the default (see :func:`branch_edges`).

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


def path_normals(path: np.ndarray, half: int = 6) -> np.ndarray:
    """Unit normals to a path, from the chord ``half`` points either side.

    The chord ignores pixel-scale wiggles. The normal is the direction of
    travel turned by 90° to its left, ``(−d_col, d_row)``.

    Args:
        path: ``(N, 2)`` ``(row, col)`` points in order.
        half: Chord half-length in points (shortened at the ends).

    Returns:
        ``(N, 2)`` unit vectors ``(d_row, d_col)``; zero where the path does
        not move.
    """
    n = len(path)
    i = np.arange(n)
    d = (path[np.minimum(i + half, n - 1)] - path[np.maximum(i - half, 0)]).astype(float)
    size = np.hypot(d[:, 0], d[:, 1])[:, None]
    d = np.divide(d, size, out=np.zeros_like(d), where=size > 0)
    return np.stack([-d[:, 1], d[:, 0]], 1)


def straighten(image: np.ndarray, path: np.ndarray, half_width: int, order: int = 1) -> np.ndarray:
    """Sample an image across a path: one row per path point.

    Row ``i`` holds the image along the normal at point ``i``
    (:func:`path_normals`), from ``−half_width`` to ``+half_width`` px, so
    the branch runs down the middle column. Samples outside the image take
    the nearest edge value.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` array.
        path: ``(N, 2)`` ``(row, col)`` points.
        half_width: Reach either side of the path (px).
        order: Interpolation order (1 = bilinear, 0 = nearest).

    Returns:
        ``(N, 2·half_width + 1)`` or ``(N, 2·half_width + 1, C)`` float array.
    """
    normals = path_normals(path)
    off = np.arange(-half_width, half_width + 1)[None, :]
    coords = [path[:, :1] + off * normals[:, :1], path[:, 1:] + off * normals[:, 1:]]
    img = np.asarray(image, float)
    if img.ndim == 2:
        return ndi.map_coordinates(img, coords, order=order, mode="nearest")
    return np.stack([ndi.map_coordinates(img[..., k], coords, order=order, mode="nearest")
                     for k in range(img.shape[2])], -1)


def _cumsum0(values: np.ndarray, axis: int) -> np.ndarray:
    """Cumulative sum along ``axis`` with a leading zero (so sums over i0..i1-1 are c[i1] − c[i0])."""
    pad = [(0, 0)] * values.ndim
    pad[axis] = (1, 0)
    return np.cumsum(np.pad(values, pad), axis=axis)


def profile_edges(profiles: np.ndarray, min_background: int = 3, min_half: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Where the branch starts and ends in each cross-section profile.

    Each profile is modelled in three pieces and fitted by least squares:
    the background on one side (a constant colour), the branch (a quadratic
    in position, per colour channel) and the background on the other side
    (a constant). A branch lit from one side is bright on that side and dark
    on the other; the quadratic follows that shading across the whole branch,
    whereas an edge or ridge detector stops at the change from the lit to
    the shaded half. Every pair of change points is tried: the squared
    error of each piece comes from cumulative sums of the samples and their
    powers, so all pairs are scored at once. The branch must include the
    centre (the path).

    On synthetic branches lit from the front or either side, of half-width
    3–9 px with the path up to 2 px off-centre, the half-width was found to
    within 0.5 px in 47 of 48 cases, and within 1 px in the last (a
    front-lit branch whose dark rim was as bright as the background).
    Where a shaded half is as dark as the background next to it, it can be
    left out; the robust fit along the leader (:func:`fit_leader_radius`)
    absorbs such points.

    Args:
        profiles: ``(M, N)`` or ``(M, N, C)`` samples across the branch, ``N``
            odd, centred on the path (e.g. from :func:`straighten`).
        min_background: Fewest background samples on each side.
        min_half: The branch reaches at least this many samples either side
            of the centre.

    Returns:
        ``(first, last)``: ``(M,)`` int offsets from the centre (px) of the
        first and last branch samples. The half-width is
        ``(last − first + 1) / 2``.

    Raises:
        ValueError: If the profiles are too short for ``min_background`` background and ``min_half`` branch
            samples on each side.
    """
    p = np.asarray(profiles, float)
    if p.ndim == 2:
        p = p[..., None]
    m, n, _ = p.shape
    half = n // 2
    if m == 0:
        return np.zeros(0, int), np.zeros(0, int)
    x = (np.arange(n) - half) / max(half, 1)                       # in [-1, 1], for conditioning
    firsts = np.arange(min_background, half - min_half + 1)
    lasts = np.arange(half + min_half, n - min_background)
    if len(firsts) == 0 or len(lasts) == 0:
        raise ValueError(f"profiles of {n} samples are too short for {min_background} background "
                         f"and {min_half} branch samples on each side")
    a, b = (g.ravel() for g in np.meshgrid(firsts, lasts, indexing="ij"))
    sx = _cumsum0(np.stack([x ** k for k in range(5)]), 1)          # (5, n+1): sums of x^k
    sxy = _cumsum0(np.stack([x ** k for k in range(3)])[:, None, :, None] * p[None], 2)  # (3, m, n+1, C)
    syy = _cumsum0((p ** 2).sum(-1), 1)                            # (m, n+1)

    def constant_sse(i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        """Squared error of the best constant over samples i0..i1-1, per profile and pair."""
        s = sxy[0][:, i1] - sxy[0][:, i0]
        return syy[:, i1] - syy[:, i0] - (s ** 2).sum(-1) / (i1 - i0)

    # Quadratic over a..b: normal equations G β = r, error = Σy² − rᵀG⁻¹r.
    i0, i1 = a, b + 1
    gram = np.stack([np.stack([sx[j + k][i1] - sx[j + k][i0] for k in range(3)], -1) for j in range(3)], -2)
    r = np.stack([sxy[j][:, i1] - sxy[j][:, i0] for j in range(3)], 2)   # (m, pairs, 3, C)
    solved = np.einsum("pjk,mpkc->mpjc", np.linalg.inv(gram), r)
    quadratic = syy[:, i1] - syy[:, i0] - (r * solved).sum((2, 3))
    cost = constant_sse(np.zeros_like(a), a) + quadratic + constant_sse(b + 1, np.full_like(b, n))
    best = np.argmin(cost, 1)
    return a[best] - half, b[best] - half


def branch_edges(rgb: np.ndarray, path: np.ndarray, half_width: int = 20,
                 along: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Both edges of a branch at each point of its path, from the colour image.

    The image is sampled across the path (:func:`straighten`), each
    cross-section is averaged with its ``along`` neighbours either side, and
    the edges are found by :func:`profile_edges`.

    Args:
        rgb: ``(H, W, 3)`` colour image.
        path: ``(N, 2)`` points along the branch.
        half_width: Reach of each cross-section either side of the path (px).
        along: Cross-sections averaged either side of each point.

    Returns:
        ``(first, last)``: ``(N,)`` int offsets (px) of the branch's edges
        along each point's normal (:func:`path_normals`).
    """
    if len(path) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    samples = straighten(rgb, path, half_width)
    if along > 0:
        samples = ndi.uniform_filter1d(samples, 2 * along + 1, axis=0, mode="nearest")
    return profile_edges(samples)


def fit_leader_radius(path: np.ndarray, measured: np.ndarray, fork_radius: float, widening: float = 0.35,
                      length: float = 12.0, start: float = 1.5, part: str = "left",
                      min_valid: float = 2.5, min_points: int = 8) -> np.ndarray:
    """A leader's drawing radius: its own width, widened near the fork.

    ``r(s) = c · (1 + widening · exp(−s / (length · R)))``, where ``s`` is the
    distance along the leader from the fork and ``R`` the trunk's radius at
    the fork. The shape is shared by all leaders (see
    :attr:`GuidedParams.fork_widening`), because one leader's widths are
    too noisy to fit its own taper: with a free straight taper, half of
    the development trees' leaders came out widening upwards. Only the
    leader's own width ``c`` is fitted, as a robust (Huber) average of
    ``measured / (1 + widening · exp(…))`` over the points at least
    ``start · R`` from the fork. Nearer, the cross-section cuts through the
    top of the trunk. Measurements at or below ``min_valid`` (failed edges)
    or nan (no measurement, e.g. a cross-section that could include the
    other leader, see :func:`clear_of`) are ignored. With fewer than
    ``min_points`` usable measurements,
    :data:`FALLBACK_RADIUS` is used for ``c``. The result is clipped to
    :data:`RADIUS_LIMITS`.

    Args:
        path: ``(N, 2)`` leader path starting at the fork.
        measured: ``(N,)`` measured half-widths (px), nan where not measured.
        fork_radius: The trunk's radius at the fork, ``R`` (px).
        widening: Extra width at the fork, as a fraction of ``c``.
        length: Distance over which the widening falls by a factor e, in
            trunk radii.
        start: Nearest distance from the fork used to fit ``c``, in trunk
            radii.
        part: ``"left"`` or ``"right"`` (for the limits and fallback).
        min_valid: Measurements at or below this (px) are failed edges, not used.
        min_points: Fewest usable measurements for a fit; with fewer, the fallback radius.

    Returns:
        ``(N,)`` radii to draw.
    """
    lo, hi = RADIUS_LIMITS[part]
    if len(path) == 0:
        return np.zeros(0)
    s = arc_length(path)
    shape = 1 + widening * np.exp(-s / max(length * fork_radius, 1e-9))
    use = (measured > min_valid) & (s >= start * fork_radius)
    level = huber_location(measured[use] / shape[use]) if use.sum() >= min_points else FALLBACK_RADIUS[part]
    return np.clip(level * shape, lo, hi)


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
        min_valid: Measurements at or below this (px) are gaps, not used.
        min_points: Fewest usable measurements for a fit; with fewer, the fallback radius.

    Returns:
        ``(N,)`` radii to draw.
    """
    lo, hi = RADIUS_LIMITS[part]
    if len(path) == 0:
        return np.zeros(0)
    s = arc_length(path)
    ok = measured > min_valid
    if ok.sum() < min_points:
        return np.full(len(path), FALLBACK_RADIUS[part])
    a, b = huber_line(s[ok], measured[ok])
    if b > 0:
        # Branches do not widen upwards: use a constant (robust) width instead.
        a, b = float(np.median(measured[ok])), 0.0
    return np.clip(a + b * s, lo, hi)


def arc_length(path: np.ndarray) -> np.ndarray:
    """Distance (px) along ``path`` from its first point to each point.

    Args:
        path: ``(N, 2)`` points, in order.

    Returns:
        ``(N,)`` distances (px); zeros for fewer than 2 points.
    """
    if len(path) < 2:
        return np.zeros(len(path))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def outward_normal(path: np.ndarray, other: np.ndarray, reach: float) -> np.ndarray:
    """Unit normal to a leader's direction near the fork, on its outer side.

    The direction is the chord from the fork to the point ``reach`` px along
    the path, which ignores pixel-scale wiggles. Of its two normals, the one
    pointing from the other leader (at the same distance along it) towards
    this one is returned, i.e. away from the crotch between the leaders.

    Args:
        path: ``(N, 2)`` leader path starting at the fork.
        other: ``(M, 2)`` the other leader's path, also starting at the fork.
        reach: Chord length (px).

    Returns:
        ``(2,)`` unit vector ``(d_row, d_col)``, or zeros if the path does
        not move away from the fork.
    """
    i = min(int(np.searchsorted(arc_length(path), reach)), len(path) - 1)
    j = min(int(np.searchsorted(arc_length(other), reach)), len(other) - 1)
    d = (path[i] - path[0]).astype(float)
    n = np.array([-d[1], d[0]])
    if n @ (path[i] - other[j]) < 0:
        n = -n
    size = np.hypot(*n)
    return n / size if size > 0 else np.zeros(2)


def shoulder_fill(shape: tuple[int, int], leader: np.ndarray, other: np.ndarray, radii: np.ndarray,
                  fork_radius: float, length: float) -> np.ndarray:
    """Pixels that fill the notch ("shoulder") beside a leader where it leaves the trunk.

    Each leader's radius is one taper fitted along its whole length
    (:func:`fit_taper`), so a thin leader starts abruptly out of the round
    end of a much wider trunk, leaving a notch on its outer side. In the
    winter depth images the wood fills that notch: the trunk's edge runs
    into the leader's outer edge. So on the leader's outer side
    (:func:`outward_normal`), the mask is filled out to the convex hull of
    the trunk's end (its disc of radius ``R`` at the fork) and the leader's
    first ``length`` px. Its edge is the straight line from the trunk's edge
    to the leader's edge ``length`` px up the leader.

    The fill never reaches outside that hull, so it cannot widen a straight
    trunk, and the crotch (between the leaders) is left as drawn. Other
    shapes were tried on the development trees and rejected (see
    ``docs/results/fork_shoulders/``). For example, widening both sides of
    each leader also filled the crotch, where many of the added pixels were
    not wood in depth; and widening the leader's discs smoothly into the
    trunk (a flare) bulged past the trunk's straight edges on a synthetic
    tree of known shape.

    Args:
        shape: Image shape ``(H, W)``.
        leader: ``(N, 2)`` leader path starting at the fork.
        other: ``(M, 2)`` the other leader's path, to tell the outer side.
        radii: ``(N,)`` the leader's drawing radii (px).
        fork_radius: The trunk's drawing radius at the fork, ``R`` (px).
        length: How far along the leader the fill reaches (px); 0 or less
            gives an empty mask.

    Returns:
        ``(H, W)`` bool mask of the fill, to be added to the drawn branches.
    """
    out = np.zeros(shape, bool)
    if length <= 0 or len(leader) == 0:
        return out
    base = leader[arc_length(leader) <= length]       # a prefix: arc length only grows
    fork = leader[0]
    # Work in a box around the fork that holds every disc (and so their hull).
    pad = int(np.ceil(max(fork_radius, radii[:len(base)].max()))) + 2
    lo = np.maximum(np.minimum(base.min(0), fork) - pad, 0)
    hi = np.minimum(np.maximum(base.max(0), fork) + pad + 1, shape)
    box = tuple(int(v) for v in hi - lo)
    discs = np.zeros(box, bool)
    for (y, x), rad in [(fork, fork_radius), *zip(base, radii)]:
        rr, cc = draw_disk((y - lo[0], x - lo[1]), rad, shape=box)
        discs[rr, cc] = True
    n = outward_normal(leader, other, max(length, 10.0))
    yy, xx = np.indices(box)
    outer = (yy + lo[0] - fork[0]) * n[0] + (xx + lo[1] - fork[1]) * n[1] >= 0
    out[lo[0]:hi[0], lo[1]:hi[1]] = convex_hull_image(discs) & outer
    return out


def fill_shoulders(gt: WinterGT, shoulder_length: float) -> np.ndarray:
    """Both leaders' :func:`shoulder_fill` for a traced tree with fitted radii.

    Args:
        gt: Result of :func:`guided_gt` (its paths and ``radii`` are used).
        shoulder_length: Reach of the fill, in multiples of the trunk's
            radius at the fork (:attr:`GuidedParams.shoulder_length`).

    Returns:
        Bool mask the shape of ``gt.tree_mask``.
    """
    fork_r = _fork_radius(gt.radii)
    out = np.zeros(gt.tree_mask.shape, bool)
    for side, leader, other in (("left", gt.left, gt.right), ("right", gt.right, gt.left)):
        out |= shoulder_fill(out.shape, leader, other, gt.radii[side], fork_r, shoulder_length * fork_r)
    return out


def end_direction(path: np.ndarray, length: float) -> np.ndarray:
    """Unit vector pointing out of a path at its last point.

    The direction is the chord to the last point from the point about
    ``length`` px before it (the first point, if the path is shorter), which
    ignores pixel-scale wiggles.

    Args:
        path: ``(N, 2)`` path points, in order.
        length: Chord length (px).

    Returns:
        ``(2,)`` unit vector ``(d_row, d_col)``, or zeros if the path has no length.
    """
    if len(path) < 2:
        return np.zeros(2)
    back = arc_length(path)
    back = back[-1] - back                              # distance back from the last point
    d = (path[-1] - path[int(np.argmin(np.abs(back - length)))]).astype(float)
    size = np.hypot(*d)
    return d / size if size > 0 else np.zeros(2)


def end_cut(path: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray] | None:
    """A flat end for a path drawn as discs: the line through its last point, square to it.

    Drawn as a union of discs, a path ends in a half disc of its radius
    there. Pixels beyond this line (see
    :attr:`treeskel.winter_gt.WinterGT.trunk_cuts`) are not drawn, which
    squares the end off. The direction is taken over the path's last
    ``radius`` px (:func:`end_direction`).

    Args:
        path: ``(N, 2)`` path points, ending at the end to cut.
        radius: The drawing radius at that end (px).

    Returns:
        ``(point, outward direction)`` as float arrays, or None if the path
        has no direction there.
    """
    d = end_direction(path, float(radius))
    return (path[-1].astype(float), d) if d.any() else None


def fork_junction(gt: WinterGT, reach: float) -> np.ndarray:
    """Pixels joining the trunk's flat top to each leader's outer edge with a straight side.

    With the trunk cut flat at the fork (:func:`end_cut`), each leader
    is joined to the trunk on its outer side (:func:`outward_normal`): the
    join is the convex hull of the trunk's top from the fork to its edge on
    that side, and of the leader's discs up to ``reach`` trunk radii along
    it, kept to the outer side of the leader's direction. Its outer edge is
    then the straight line from the trunk's edge to the leader's edge, and
    the crotch between the leaders is left as the leaders' own discs draw
    it: a V, not the round top of the trunk. The shoulder fill
    (:func:`shoulder_fill`) is similar but takes the trunk's round end disc
    into the hull, so the round top stays.

    Args:
        gt: A guided result with fitted ``radii``; its trunk ends, and both
            leaders start, at the fork.
        reach: How far up each leader the join reaches, in trunk radii at
            the fork (:attr:`GuidedParams.junction_reach`); 0 or less gives
            an empty mask.

    Returns:
        ``(H, W)`` bool mask of the join, to be added to the drawn branches.
    """
    shape = gt.tree_mask.shape
    out = np.zeros(shape, bool)
    if reach <= 0 or len(gt.trunk) < 2 or len(gt.left) == 0 or len(gt.right) == 0:
        return out
    radius = _fork_radius(gt.radii)
    up = end_direction(gt.trunk, radius)
    if not up.any():
        return out
    fork = gt.trunk[-1]
    across = np.array([-up[1], up[0]])                   # square to the trunk, along its flat top
    for side, other in (("left", gt.right), ("right", gt.left)):
        leader, radii = getattr(gt, side), gt.radii[side]
        outer = outward_normal(leader, other, max(reach * radius, 10.0))
        if not outer.any():
            continue
        corner = fork + radius * (across if across @ outer >= 0 else -across)   # the trunk's edge on this side
        base = arc_length(leader) <= reach * radius       # a prefix: arc length only grows
        # Work in a box around the fork that holds the top and every disc (and so their hull).
        pad = int(np.ceil(radii[base].max())) + 2
        points = np.vstack([fork, corner, leader[base]])
        lo = np.maximum(np.floor(points.min(0)).astype(int) - pad, 0)
        hi = np.minimum(np.ceil(points.max(0)).astype(int) + pad + 1, shape)
        if np.any(hi <= lo):
            continue
        box = tuple(int(v) for v in hi - lo)
        parts = np.zeros(box, bool)
        for (y, x), rad in zip(leader[base], radii[base]):
            rr, cc = draw_disk((y - lo[0], x - lo[1]), rad, shape=box)
            parts[rr, cc] = True
        end = np.round(corner).astype(int)
        rr, cc = draw_line(int(fork[0] - lo[0]), int(fork[1] - lo[1]), int(end[0] - lo[0]), int(end[1] - lo[1]))
        inside = (rr >= 0) & (rr < box[0]) & (cc >= 0) & (cc < box[1])
        parts[rr[inside], cc[inside]] = True
        yy, xx = np.indices(box)
        keep = (yy + lo[0] - fork[0]) * outer[0] + (xx + lo[1] - fork[1]) * outer[1] >= 0
        out[lo[0]:hi[0], lo[1]:hi[1]] |= convex_hull_image(parts) & keep
    return out


def _fork_radius(radii: dict) -> float:
    """The trunk's drawing radius at the fork (its last point)."""
    return float(radii["trunk"][-1]) if len(radii["trunk"]) else FALLBACK_RADIUS["trunk"]


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
        """Index of the path point nearest to ``q``."""
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

    Args:
        c: The tree's clicks.
        min_rise: Least height (px) of each tip above the fork.
        min_spread: Least sideways distance (px) from the left tip to the right one.

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

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if an output file could not be written.
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
    (out / "params.json").write_text(json.dumps(asdict(params), indent=2))   # the settings, beside the masks

    # The clicked trees asked for; trees skipped in the click tool or not finished are left out.
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
        """Print progress every ``step`` trees (about 10%, at most 25), with the time left."""
        if done % step == 0 or done == len(jobs):
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"  {done}/{len(jobs)} trees, {el:.0f} s elapsed, about {eta:.0f} s left", flush=True)

    print(f"Rendering {len(jobs)} trees using {max(1, min(args.jobs, len(jobs)))} process(es)...", flush=True)
    # Trace and draw each tree, in worker processes when there is more than one core and tree.
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
    # Clicks that look like mistakes (e.g. a fork right at the root) are listed for checking by eye.
    suspect = {t: check_clicks(clicks[t]) for t in wanted if check_clicks(clicks[t])}
    for tid, why in sorted(suspect.items()):
        print(f"Check clicks for {tid}: {'; '.join(why)}")
    print(f"Rendered {len(rows)} trees in {time.time() - t0:.0f} s; {len(skipped)} skipped or "
          f"incomplete; {len(missing)} not found in {args.data_dir}. Results in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
