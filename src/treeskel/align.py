"""Winter-to-summer alignment of each tree's images.

The ground-truth masks are traced in the winter image (bare branches), but
the summer model needs them in the summer image of the same tree. The camera
was set up again for each season, so the two views differ from tree to tree,
by tens of pixels.

**Geometry.** The trees are trained flat along the row (a 2D open-V), so the
trunk and leaders lie close to one vertical plane. When the camera moves, a
plane maps from one image to the other by a homography (8 parameters). If the
camera turns only slightly and the plane roughly faces it, this is close to a
similarity: a scale (a change of distance), a rotation (camera roll) and a
shift. Points off the tree's plane, such as the ground behind the tree and
the rows behind it, shift by different amounts (parallax) and act as
outliers.

**Method.**

1. *Matching* (:func:`make_matcher`): pairs of points that show the same
   thing in both images. Hand-designed local features (SIFT) found only 3
   RANSAC inliers per tree on the 21 development trees in the repository,
   which is what chance gives. Learned matchers are trained on photos of the
   same places under different lighting and seasons: DISK features with the
   LightGlue matcher, and LoFTR, which matches without detecting keypoints
   first. They need PyTorch and kornia (``pip install -e ".[match]"``) and
   download their pretrained weights the first time they run.
2. *Robust fit* (:func:`ransac_fit`): RANSAC repeatedly fits the transform to
   a small random sample of matches and keeps the one that most matches
   agree with (MSAC score, :func:`msac_cost`); the transform is then refitted
   to all its inliers by least squares (:func:`fit_transform`). Four models,
   from the fewest parameters to the most: translation (2), similarity (4),
   affine (6) and homography (8).
3. *Checks* (:func:`fit_quality`, :func:`check_fit`): the number of inliers,
   how widely they are spread over the tree, and whether the scale and
   rotation are plausible.
4. *Warp* (:func:`warp_mask`): the winter mask is resampled into the summer
   image (nearest neighbour, so it stays binary), together with the part of
   the summer image that the winter image covers.

Matching is the only slow step and needs a GPU to be quick, so the matches
are computed once per tree and matcher and stored in
``data/matches/<matcher>.npz`` (point coordinates and scores only, see
:func:`save_matches`). Everything after matching can then be redone and
compared from that file, without PyTorch.

**Coordinates.** Points in this module are ``(x, y)`` = ``(column, row)``,
as usual for geometric transforms. The rest of the package uses
``(row, col)``; convert with ``pts[:, ::-1]``. A transform is a 3x3 matrix
``M`` in homogeneous coordinates that maps winter points to summer points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

from treeskel.winter_gt import report_failed, save_image

MODELS = ("translation", "similarity", "affine", "homography")
"""Transform models, from the fewest parameters to the most."""

MIN_SAMPLES = {"translation": 1, "similarity": 2, "affine": 3, "homography": 4}
"""Matches needed to fit each model exactly (RANSAC's sample size)."""

MATCHERS = ("lightglue", "loftr", "sift")
"""Matchers available to :func:`make_matcher`."""

MATCHES_DIR = Path("data/matches")
"""Where :func:`main` keeps the matches, one ``<matcher>.npz`` per matcher."""


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

def apply_transform(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` ``(x, y)`` points by a 3x3 transform (with perspective division).

    Points that a homography sends to infinity come back as ``inf`` or ``nan``.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    h = np.c_[pts, np.ones(len(pts))] @ np.asarray(matrix, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :2] / h[:, 2:3]


def keeps_orientation(matrix: np.ndarray, pts: np.ndarray) -> bool:
    """True if a transform keeps every point in front of the camera.

    A homography's perspective denominator ``w`` must stay positive across
    the points it is used for. Where ``w`` crosses zero, the map sends points
    to infinity and turns the image inside out, which no camera move can do,
    so RANSAC rejects such samples. (Always True for affine transforms.)
    """
    m = np.asarray(matrix, float)
    return bool(np.all(np.asarray(pts, float).reshape(-1, 2) @ m[2, :2] + m[2, 2] > 0))


def residuals(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Distance (px) between each transformed ``src`` point and its ``dst`` point."""
    return np.hypot(*(apply_transform(matrix, src) - dst).T)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares similarity by Umeyama's closed form (1991).

    With both point sets centred (``a``, ``b``), the rotation ``R`` comes
    from the SVD of the cross-covariance ``b^T a = U S V^T`` (``R = U D V^T``,
    with ``D`` fixing a reflection), the scale is ``tr(S D) / Σ|a|²`` and the
    shift maps the centroids onto each other. This minimises
    ``Σ |s R p + t − q|²`` exactly, with no iteration.
    """
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    var = (a ** 2).sum()
    if var < 1e-9:
        return None
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0])
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d)) / var
    m = np.eye(3)
    m[:2, :2] = scale * rot
    m[:2, 2] = md - scale * rot @ ms
    return m


def _affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares affine transform (a linear least-squares problem)."""
    x = np.c_[src, np.ones(len(src))]
    if np.linalg.matrix_rank(x, tol=1e-6 * max(1.0, np.abs(src).max())) < 3:
        return None                                     # collinear points
    sol, *_ = np.linalg.lstsq(x, dst, rcond=None)
    m = np.eye(3)
    m[:2] = sol.T
    return m


def _normaliser(pts: np.ndarray) -> np.ndarray | None:
    """Hartley normalisation: centre the points, mean distance √2 from the origin."""
    c = pts.mean(0)
    d = np.hypot(*(pts - c).T).mean()
    if d < 1e-9:
        return None
    k = np.sqrt(2) / d
    return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1.0]])


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Homography by the normalised direct linear transform (DLT).

    Each match gives two linear equations in the 9 entries of ``H``; the
    solution is the right singular vector of the smallest singular value
    (algebraic least squares). Normalising both point sets first
    (:func:`_normaliser`) makes this well conditioned (Hartley, 1997).
    """
    ts, td = _normaliser(src), _normaliser(dst)
    if ts is None or td is None:
        return None
    s = apply_transform(ts, src)
    d = apply_transform(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sv, vt = np.linalg.svd(np.asarray(rows))
    if len(sv) >= 8 and sv[7] < 1e-9 * sv[0]:
        return None                                     # degenerate (e.g. collinear) points
    h = np.linalg.inv(td) @ vt[-1].reshape(3, 3) @ ts
    if abs(h[2, 2]) < 1e-12:
        return None
    return h / h[2, 2]


def fit_transform(src: np.ndarray, dst: np.ndarray, model: str = "similarity") -> np.ndarray | None:
    """Least-squares transform mapping ``src`` onto ``dst``.

    Args:
        src, dst: ``(N, 2)`` matching ``(x, y)`` points.
        model: One of :data:`MODELS`. Translation, similarity and affine
            minimise the sum of squared distances exactly; the homography
            minimises an algebraic error (:func:`_homography`).

    Returns:
        The 3x3 matrix, or ``None`` if there are too few points or they are
        degenerate (e.g. all on one line for an affine transform).

    Raises:
        ValueError: If ``model`` is unknown.

    Example:
        >>> src = np.array([[0.0, 0], [10, 0], [0, 10]])
        >>> m = fit_transform(src, 2 * src + [5, 3], "similarity")
        >>> np.round(apply_transform(m, [[1, 1]]), 6)
        array([[7., 5.]])
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    if len(src) < MIN_SAMPLES[model] or len(src) != len(dst):
        return None
    if model == "translation":
        m = np.eye(3)
        m[:2, 2] = (dst - src).mean(0)
        return m
    return {"similarity": _similarity, "affine": _affine, "homography": _homography}[model](src, dst)


def msac_cost(res: np.ndarray, threshold: float) -> float:
    """MSAC score of a hypothesis: ``Σ min(r², t²)``; lower is better.

    Plain RANSAC counts inliers, so two hypotheses with the same count tie
    even if one fits its inliers much more closely. MSAC (Torr and
    Zisserman, 2000) charges each inlier its squared residual and each
    outlier a constant ``t²``.
    """
    return float(np.minimum(res ** 2, threshold ** 2).sum())


@dataclass
class Fit:
    """A robustly fitted transform.

    Attributes:
        model: One of :data:`MODELS`.
        matrix: 3x3 winter-to-summer ``(x, y)`` transform, or ``None`` if the
            fit failed (too few or degenerate matches).
        inliers: ``(N,)`` bool, the matches within ``threshold`` of the fit.
        rms: RMS residual of the inliers (px).
        trials: RANSAC samples drawn.
    """

    model: str
    matrix: np.ndarray | None
    inliers: np.ndarray
    rms: float = float("nan")
    trials: int = 0

    @property
    def n_inliers(self) -> int:
        """Number of inlier matches."""
        return int(self.inliers.sum())


def ransac_fit(src: np.ndarray, dst: np.ndarray, model: str = "similarity", threshold: float = 5.0,
               max_trials: int = 5000, confidence: float = 0.999, seed: int = 0) -> Fit:
    """Fit a transform to matches that include wrong ones (RANSAC with MSAC scoring).

    Repeatedly: draw the smallest sample that fixes the model
    (:data:`MIN_SAMPLES`), fit it exactly, and score it on all matches
    (:func:`msac_cost`). The number of samples adapts to the best inlier
    fraction ``w`` found so far: ``log(1 − confidence) / log(1 − w^k)``
    samples give that probability of drawing at least one all-inlier sample
    of size ``k``. The best hypothesis is then refitted by least squares to
    its inliers, and inliers and fit are updated until they stop changing.

    Args:
        src, dst: ``(N, 2)`` matching ``(x, y)`` points (winter, summer).
        model: One of :data:`MODELS`.
        threshold: Largest residual (px) of an inlier.
        max_trials: Upper limit on samples drawn.
        confidence: Wanted probability of drawing one all-inlier sample.
        seed: Random seed, so results repeat exactly.

    Returns:
        A :class:`Fit`; its ``matrix`` is ``None`` if no sample could be fitted.
    """
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    n, k = len(src), MIN_SAMPLES[model]
    none = Fit(model, None, np.zeros(n, bool))
    if n < k:
        return none
    rng = np.random.default_rng(seed)
    best, best_cost, needed, trials = None, np.inf, max_trials, 0
    while trials < min(needed, max_trials):
        trials += 1
        idx = rng.choice(n, k, replace=False)
        m = fit_transform(src[idx], dst[idx], model)
        if m is None or not np.all(np.isfinite(m)) or not keeps_orientation(m, src):
            continue
        res = residuals(m, src, dst)
        cost = msac_cost(res, threshold)
        if cost < best_cost:
            best, best_cost = m, cost
            p_good = float((res < threshold).mean()) ** k     # chance that a sample is all inliers
            if p_good >= 1.0:
                needed = trials
            elif p_good > 1e-12:                              # (below this, 1 − p rounds to 1)
                needed = int(np.ceil(np.log(1 - confidence) / np.log1p(-p_good)))
    if best is None:
        return Fit(model, None, np.zeros(n, bool), trials=trials)
    inl = residuals(best, src, dst) < threshold
    for _ in range(10):                                 # least-squares refit on the inliers
        m = fit_transform(src[inl], dst[inl], model)
        if m is None or not np.all(np.isfinite(m)) or not keeps_orientation(m, src):
            break
        new = residuals(m, src, dst) < threshold
        if new.sum() < k:                               # the refit lost its support: keep the last fit
            break
        best = m
        if np.array_equal(new, inl):
            break
        inl = new
    res = residuals(best, src, dst)
    inl = res < threshold
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else float("nan")
    return Fit(model, best, inl, rms, trials)


def linear_part(matrix: np.ndarray, at: Sequence[float]) -> np.ndarray:
    """The 2x2 Jacobian of a transform at the point ``at`` (``(x, y)``).

    For an affine transform this is its linear part everywhere; for a
    homography it changes across the image, so it is taken where it matters.
    """
    m = np.asarray(matrix, float)
    p = np.array([at[0], at[1], 1.0])
    w = m[2] @ p
    q = (m[:2] @ p) / w
    return (m[:2, :2] - np.outer(q, m[2, :2])) / w


def describe(matrix: np.ndarray, at: Sequence[float]) -> dict[str, float]:
    """Scale, rotation and shift of a transform at a point.

    Returns:
        ``{"scale": √|det J|, "rotation": degrees (positive = clockwise in
        the image, whose y axis points down), "anisotropy": ratio of J's
        singular values (1 for a similarity), "dx", "dy": shift of the point
        (px)}``, where ``J`` is :func:`linear_part` at ``at``.
    """
    j = linear_part(matrix, at)
    sv = np.linalg.svd(j, compute_uv=False)
    moved = apply_transform(matrix, [at])[0]
    return {
        "scale": float(np.sqrt(abs(np.linalg.det(j)))),
        "rotation": float(np.degrees(np.arctan2(j[1, 0] - j[0, 1], j[0, 0] + j[1, 1]))),
        "anisotropy": float(sv[0] / sv[1]) if sv[1] > 0 else float("inf"),
        "dx": float(moved[0] - at[0]),
        "dy": float(moved[1] - at[1]),
    }


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FitLimits:
    """Limits used by :func:`check_fit`. Provisional: to be set from the data.

    Attributes:
        min_inliers: Fewest inlier matches for a usable fit.
        min_hull: Smallest area of the inliers' convex hull, as a fraction of
            the image. Inliers bunched in one spot fix the shift there but
            leave the scale and rotation, and so everything far from the spot,
            poorly determined.
        scale_range: Plausible scale between the seasons.
        max_rotation: Largest plausible camera roll difference (degrees).
    """

    min_inliers: int = 15
    min_hull: float = 0.02
    scale_range: tuple[float, float] = (0.75, 1.33)
    max_rotation: float = 10.0


def hull_fraction(pts: np.ndarray, shape: tuple[int, int]) -> float:
    """Area of the convex hull of ``(x, y)`` points, as a fraction of an image of ``shape``."""
    pts = np.unique(np.asarray(pts, float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume) / (shape[0] * shape[1])   # "volume" is the area in 2D
    except QhullError:                                                  # collinear points
        return 0.0


def fit_quality(fit: Fit, src: np.ndarray, shape: tuple[int, int],
                anchor: Sequence[float] | None = None) -> dict[str, float]:
    """Numbers describing one fit, for the summary table and the checks.

    Args:
        fit: The fit.
        src: ``(N, 2)`` winter ``(x, y)`` points of the matches it was fitted to.
        shape: Winter image shape ``(H, W)``.
        anchor: Winter ``(x, y)`` point at which scale, rotation and shift are
            described (e.g. the tree's fork); the image centre by default.

    Returns:
        Dict with ``matches``, ``inliers``, ``inlier_ratio``, ``hull``
        (:func:`hull_fraction` of the inliers), ``rms`` and, if the fit
        succeeded, the :func:`describe` values.
    """
    n = len(src)
    q = {"matches": n, "inliers": fit.n_inliers,
         "inlier_ratio": fit.n_inliers / n if n else 0.0,
         "hull": hull_fraction(np.asarray(src)[fit.inliers], shape), "rms": fit.rms}
    if fit.matrix is not None:
        at = anchor if anchor is not None else (shape[1] / 2, shape[0] / 2)
        q.update(describe(fit.matrix, at))
    return q


def inlier_places(pts: np.ndarray, valid: np.ndarray, near_tree: np.ndarray | None) -> dict[str, float]:
    """Where a fit's inliers lie in the winter image.

    Args:
        pts: ``(N, 2)`` winter ``(x, y)`` inlier points.
        valid: Winter depth validity mask. The rows behind the tree are beyond
            the depth camera's range in winter, so an inlier without a depth
            reading is most likely on the background, off the tree's plane.
        near_tree: Bool mask of pixels near the traced branches, or ``None``.

    Returns:
        ``{"no_depth": fraction of inliers without a depth reading,
        "on_tree": fraction within ``near_tree``}`` (nan when undefined).
    """
    if len(pts) == 0:
        return {"no_depth": float("nan"), "on_tree": float("nan")}
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, valid.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, valid.shape[0] - 1)
    return {"no_depth": float((~valid[y, x]).mean()),
            "on_tree": float(near_tree[y, x].mean()) if near_tree is not None else float("nan")}


def check_fit(quality: dict, limits: FitLimits = FitLimits()) -> list[str]:
    """Reasons a fit looks unusable (empty if none), like :func:`treeskel.guided_gt.check_clicks`."""
    if "scale" not in quality:
        return ["no transform could be fitted"]
    issues = []
    if quality["inliers"] < limits.min_inliers:
        issues.append(f"only {quality['inliers']} inliers")
    if quality["hull"] < limits.min_hull:
        issues.append(f"inliers cover {100 * quality['hull']:.1f}% of the image")
    lo, hi = limits.scale_range
    if not lo <= quality["scale"] <= hi:
        issues.append(f"scale {quality['scale']:.2f}")
    if abs(quality["rotation"]) > limits.max_rotation:
        issues.append(f"rotation {quality['rotation']:.1f} deg")
    return issues


# --------------------------------------------------------------------------
# Warping
# --------------------------------------------------------------------------

def warp_mask(mask: np.ndarray, matrix: np.ndarray,
              out_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Resample a winter-frame mask into the summer frame.

    Each summer pixel is mapped back into the winter image by the inverse
    transform and takes the value of the nearest winter pixel.

    Args:
        mask: ``(H, W)`` bool mask in the winter frame.
        matrix: 3x3 winter-to-summer ``(x, y)`` transform.
        out_shape: Summer image shape; the mask's shape by default.

    Returns:
        ``(warped, covered)``: the mask in the summer frame, and True where the
        summer pixel maps inside the winter image. Outside ``covered`` the
        winter image says nothing, so those pixels are unknown rather than
        "not branch" (as for :func:`treeskel.io.mask_to_image_frame`).
    """
    h, w = out_shape or mask.shape
    yy, xx = np.mgrid[:h, :w]
    src = apply_transform(np.linalg.inv(matrix), np.c_[xx.ravel(), yy.ravel()])
    col, row = np.round(src[:, 0]), np.round(src[:, 1])
    covered = (row >= 0) & (row < mask.shape[0]) & (col >= 0) & (col < mask.shape[1])
    out = np.zeros(h * w, bool)
    out[covered] = mask[row[covered].astype(int), col[covered].astype(int)]
    return out.reshape(h, w), covered.reshape(h, w)


# --------------------------------------------------------------------------
# Matches: storage and matchers
# --------------------------------------------------------------------------

@dataclass
class Matches:
    """Matched points between a tree's winter and summer images.

    Attributes:
        winter: ``(N, 2)`` ``(x, y)`` points in the winter image.
        summer: ``(N, 2)`` ``(x, y)`` points in the summer image.
        score: ``(N,)`` the matcher's confidence (higher is better; its scale
            depends on the matcher).
    """

    winter: np.ndarray
    summer: np.ndarray
    score: np.ndarray = field(default=None)

    def __post_init__(self):
        self.winter = np.asarray(self.winter, np.float32).reshape(-1, 2)
        self.summer = np.asarray(self.summer, np.float32).reshape(-1, 2)
        if self.score is None:
            self.score = np.ones(len(self.winter), np.float32)
        self.score = np.asarray(self.score, np.float32).reshape(-1)
        if not len(self.winter) == len(self.summer) == len(self.score):
            raise ValueError("winter, summer and score must have the same length")

    def __len__(self) -> int:
        return len(self.winter)


def save_matches(path: str | Path, matches: dict[str, Matches], meta: dict | None = None) -> None:
    """Write matches for many trees to one compressed ``.npz`` file, atomically.

    Arrays are stored as ``<tree>_winter``, ``<tree>_summer`` and
    ``<tree>_score`` (float32), plus ``meta`` (a JSON string), so the file
    loads with ``allow_pickle=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"meta": np.array(json.dumps(meta or {}))}
    for tid, m in matches.items():
        arrays[f"{tid}_winter"], arrays[f"{tid}_summer"], arrays[f"{tid}_score"] = m.winter, m.summer, m.score
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_matches(path: str | Path) -> tuple[dict[str, Matches], dict]:
    """Read a file written by :func:`save_matches`; a missing file gives ``({}, {})``."""
    path = Path(path)
    if not path.exists():
        return {}, {}
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        ids = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_winter")})
        return {t: Matches(z[f"{t}_winter"], z[f"{t}_summer"], z[f"{t}_score"]) for t in ids}, meta


def sift_match(rgb_winter: np.ndarray, rgb_summer: np.ndarray, max_ratio: float = 0.8) -> Matches:
    """SIFT keypoints matched by nearest descriptor (the classical baseline).

    Matches must pass Lowe's ratio test (the nearest descriptor is clearly
    nearer than the second nearest) and be mutual nearest neighbours.
    """
    from skimage.feature import SIFT, match_descriptors

    pts = []
    for rgb in (rgb_winter, rgb_summer):
        f = SIFT()
        try:
            f.detect_and_extract(rgb.astype(float).mean(2) / 255.0)
        except RuntimeError:                           # no keypoints (e.g. a blank image)
            return Matches(np.zeros((0, 2)), np.zeros((0, 2)))
        pts.append(f)
    m = match_descriptors(pts[0].descriptors, pts[1].descriptors, max_ratio=max_ratio, cross_check=True)
    return Matches(pts[0].keypoints[m[:, 0]][:, ::-1], pts[1].keypoints[m[:, 1]][:, ::-1])


def _image_tensor(rgb: np.ndarray, device):
    """``(H, W, 3)`` uint8 image as a ``(1, 3, H, W)`` float tensor in [0, 1]."""
    import torch

    # A copy: images loaded with PIL are read-only arrays, which torch warns about.
    return torch.from_numpy(np.array(rgb, np.uint8)).permute(2, 0, 1)[None].float().div(255).to(device)


def disk_lightglue_match(disk, lightglue, rgb_winter: np.ndarray, rgb_summer: np.ndarray,
                         device="cpu", max_keypoints: int = 2048) -> Matches:
    """DISK keypoints matched by LightGlue.

    Args:
        disk: A ``kornia.feature.DISK`` model.
        lightglue: A ``kornia.feature.LightGlue`` model for DISK features.
        rgb_winter, rgb_summer: ``(H, W, 3)`` uint8 images (sides divisible by 16).
        device: Torch device of the models.
        max_keypoints: Keypoints detected per image.
    """
    import torch

    with torch.inference_mode():
        feats = [disk(_image_tensor(rgb, device), max_keypoints, pad_if_not_divisible=True)[0]
                 for rgb in (rgb_winter, rgb_summer)]
        data = {f"image{i}": {"keypoints": f.keypoints[None], "descriptors": f.descriptors[None],
                              "image_size": torch.tensor([[rgb.shape[1], rgb.shape[0]]], device=device)}
                for i, (f, rgb) in enumerate(zip(feats, (rgb_winter, rgb_summer)))}
        out = lightglue(data)
        idx, score = out["matches"][0], out["scores"][0]
        w = feats[0].keypoints[idx[:, 0]]
        s = feats[1].keypoints[idx[:, 1]]
    return Matches(w.cpu().numpy(), s.cpu().numpy(), score.float().cpu().numpy())


def loftr_match(loftr, rgb_winter: np.ndarray, rgb_summer: np.ndarray, device="cpu",
                max_matches: int = 2048) -> Matches:
    """LoFTR matches, keeping the ``max_matches`` most confident.

    Args:
        loftr: A ``kornia.feature.LoFTR`` model.
        rgb_winter, rgb_summer: ``(H, W, 3)`` uint8 images (sides divisible by 8).
        device: Torch device of the model.
        max_matches: Most matches kept (LoFTR can return thousands).
    """
    import torch
    from kornia.color import rgb_to_grayscale

    with torch.inference_mode():
        out = loftr({"image0": rgb_to_grayscale(_image_tensor(rgb_winter, device)),
                     "image1": rgb_to_grayscale(_image_tensor(rgb_summer, device))})
        conf = out["confidence"]
        keep = torch.argsort(conf, descending=True)[:max_matches]
        w, s, c = out["keypoints0"][keep], out["keypoints1"][keep], conf[keep]
    return Matches(w.cpu().numpy(), s.cpu().numpy(), c.float().cpu().numpy())


def make_matcher(name: str, device: str | None = None,
                 max_keypoints: int = 2048) -> Callable[[np.ndarray, np.ndarray], Matches]:
    """A function ``(rgb_winter, rgb_summer) -> Matches`` for the named matcher.

    ``"lightglue"`` (DISK features, LightGlue matcher) and ``"loftr"`` (outdoor
    weights) need PyTorch and kornia, and download their pretrained weights on
    first use; ``"sift"`` uses scikit-image only.

    Args:
        name: One of :data:`MATCHERS`.
        device: ``"cuda"`` or ``"cpu"``; the GPU if one is available by default.
        max_keypoints: Keypoints per image (LightGlue) or matches kept (LoFTR).

    Raises:
        ValueError: If ``name`` is unknown.
        ImportError: If a learned matcher is asked for without PyTorch/kornia.
    """
    if name == "sift":
        return sift_match
    if name not in MATCHERS:
        raise ValueError(f"unknown matcher {name!r}; expected one of {MATCHERS}")
    import torch
    import kornia.feature as KF

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if name == "lightglue":
        disk = KF.DISK.from_pretrained("depth", device=dev).eval()
        lightglue = KF.LightGlue("disk").to(dev).eval()
        return lambda w, s: disk_lightglue_match(disk, lightglue, w, s, dev, max_keypoints)
    loftr = KF.LoFTR(pretrained="outdoor").to(dev).eval()
    return lambda w, s: loftr_match(loftr, w, s, dev, max_keypoints)


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------

def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, iterations=width)


def draw_alignment(rgb_winter: np.ndarray, rgb_summer: np.ndarray, mask_winter: np.ndarray,
                   matches: Matches, fit: Fit, label: str = "", max_lines: int = 150) -> Image.Image:
    """Winter and summer side by side, with the mask and the matches.

    Left: the winter image with its mask outlined (magenta). Right: the summer
    image with the warped mask outlined. Inlier matches are joined by green
    lines (at most ``max_lines``, evenly subsampled); outliers are red dots.
    """
    h, w = rgb_winter.shape[:2]
    left = rgb_winter.copy()
    left[_outline(mask_winter)] = (255, 0, 255)
    right = rgb_summer.copy()
    if fit.matrix is not None:
        warped, _ = warp_mask(mask_winter, fit.matrix, rgb_summer.shape[:2])
        right[_outline(warped)] = (255, 0, 255)
    im = Image.fromarray(np.concatenate([left, right], 1))
    dr = ImageDraw.Draw(im)
    out = np.nonzero(~fit.inliers)[0]
    for (x, y) in matches.winter[out]:
        dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 60, 60))
    for (x, y) in matches.summer[out]:
        dr.ellipse([x + w - 1.5, y - 1.5, x + w + 1.5, y + 1.5], fill=(255, 60, 60))
    inl = np.nonzero(fit.inliers)[0]
    for i in inl[np.linspace(0, len(inl) - 1, min(len(inl), max_lines)).astype(int)]:
        (x0, y0), (x1, y1) = matches.winter[i], matches.summer[i]
        dr.line([(x0, y0), (x1 + w, y1)], fill=(60, 255, 60), width=1)
    dr.text((8, 8), label, fill="yellow")
    return im


def draw_summer(rgb_summer: np.ndarray, mask_winter: np.ndarray, matches: Matches, fit: Fit,
                label: str = "", width: int = 3) -> Image.Image:
    """The summer image with the warped winter mask outlined (magenta) and the inliers (green dots).

    For contact sheets: whether the warped trunk sits on the summer trunk can
    be judged at a glance. ``width`` is the outline's width (px), thick enough
    to survive shrinking the picture.
    """
    im = rgb_summer.copy()
    if fit.matrix is not None:
        warped, _ = warp_mask(mask_winter, fit.matrix, rgb_summer.shape[:2])
        im[_outline(warped, width)] = (255, 0, 255)
    pil = Image.fromarray(im)
    dr = ImageDraw.Draw(pil)
    for (x, y) in matches.summer[fit.inliers]:
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(60, 255, 60))
    dr.text((8, 8), label, fill="yellow")
    return pil


def contact_sheet(tiles: list[np.ndarray], cols: int = 4) -> Image.Image:
    """Grid of equally sized tiles (padded with black to fill the last row)."""
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    return Image.fromarray(np.concatenate(rows, 0))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

NEAR_TREE_PX = 20
"""An inlier within this distance (px) of the winter mask counts as on the tree."""


def _winter_mask(tree_id: str, winter: dict, clicks: dict, gt_dir: Path | None) -> np.ndarray | None:
    """The tree's rendered winter mask: from ``gt_dir`` if saved there, else traced from its clicks."""
    from treeskel.guided_gt import guided_gt
    from treeskel.winter_gt import render_main_branches

    if gt_dir is not None and (gt_dir / f"{tree_id}_winter_gt.png").exists():
        return np.asarray(Image.open(gt_dir / f"{tree_id}_winter_gt.png").convert("L")) > 127
    c = clicks.get(tree_id)
    if c is None or c.skip or not c.complete:
        return None
    return render_main_branches(guided_gt(winter["rgb"], winter["depth"], winter["valid"], c))


def main(argv: Sequence[str] | None = None) -> int:
    """Match, fit and check the winter-to-summer alignment of every selected tree.

    Usage::

        python -m treeskel.align DATA_DIR --matcher lightglue --dev-only
            [--rows 5 6] [--trees R05N12] [--model similarity] [--overlays]

    Matches are read from ``data/matches/<matcher>.npz`` and only missing trees
    are matched (``--rematch`` redoes them); the file is saved every 25 trees,
    so an interrupted run resumes. Then all four models are fitted to every
    tree, and ``outputs/align/<matcher>/summary.csv`` gets one row per tree
    and model. For ``--model`` (the similarity by default), contact sheets of
    the summer images with the warped winter masks are saved, and
    ``--overlays`` also saves a side-by-side picture per tree.
    """
    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset, load_tree
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.align", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--matcher", choices=MATCHERS, default="lightglue")
    ap.add_argument("--model", choices=MODELS, default="similarity", help="model for the pictures")
    ap.add_argument("--threshold", type=float, default=5.0, help="RANSAC inlier threshold (px)")
    ap.add_argument("--rows", type=int, nargs="+", help="only trees in these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true", help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--matches", help="matches file (default: data/matches/<matcher>.npz)")
    ap.add_argument("--rematch", action="store_true", help="match again even if saved")
    ap.add_argument("--no-match", action="store_true", help="only use saved matches (no PyTorch needed)")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if available)")
    ap.add_argument("--max-keypoints", type=int, default=2048)
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--winter-gt", default="outputs/winter_gt_guided",
                    help="folder of <tree>_winter_gt.png masks; traced from the clicks if missing")
    ap.add_argument("--out", default="outputs/align")
    ap.add_argument("--overlays", action="store_true", help="also save a picture per tree")
    args = ap.parse_args(argv)

    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    if not trees:
        print("No trees selected.")
        return 1

    mpath = Path(args.matches or MATCHES_DIR / f"{args.matcher}.npz")
    saved, meta = load_matches(mpath)
    todo = [t for t in trees if args.rematch or t.tree_id not in saved]
    if todo and not args.no_match:
        print(f"Loading the {args.matcher} matcher...", flush=True)
        try:
            matcher = make_matcher(args.matcher, args.device, args.max_keypoints)
        except (ImportError, OSError) as e:             # PyTorch missing or broken, or a failed download
            print(f"Could not load the {args.matcher} matcher: {type(e).__name__}: {e}\n"
                  "If PyTorch itself fails to load (e.g. WinError 193 or 126), see 'If PyTorch does not "
                  "load' in docs/results/align/README.md.")
            return 1
        meta = {"matcher": args.matcher, "max_keypoints": args.max_keypoints}
        t0 = time.time()
        errors = []
        for i, rec in enumerate(todo, 1):
            try:
                saved[rec.tree_id] = matcher(load_tree(rec, "winter")["rgb"], load_tree(rec, "summer")["rgb"])
            except Exception as e:                      # one bad tree must not stop a long run
                errors.append(f"{rec.tree_id}: {type(e).__name__}: {e}")
            if i % 25 == 0 or i == len(todo):
                save_matches(mpath, saved, meta)
                el = time.time() - t0
                print(f"  matched {i}/{len(todo)} trees, {el:.0f} s elapsed, "
                      f"about {el / i * (len(todo) - i):.0f} s left", flush=True)
        print(f"Matches saved to {mpath} ({mpath.stat().st_size / 1e6:.1f} MB)")
        for e in errors:
            print(f"  could not match {e}")
    trees = [t for t in trees if t.tree_id in saved]
    if not trees:
        print(f"No saved matches in {mpath} for the selected trees.")
        return 1

    out = Path(args.out) / args.matcher
    out.mkdir(parents=True, exist_ok=True)
    clicks = load_clicks(args.clicks)
    gt_dir = Path(args.winter_gt) if args.winter_gt else None
    rows, tiles, failed, flagged = [], [], [], {}
    t0 = time.time()
    for rec in trees:
        m = saved[rec.tree_id]
        winter, summer = load_tree(rec, "winter"), load_tree(rec, "summer")
        mask = _winter_mask(rec.tree_id, winter, clicks, gt_dir)
        near = ndi.distance_transform_edt(~mask) <= NEAR_TREE_PX if mask is not None else None
        c = clicks.get(rec.tree_id)
        anchor = c.fork[::-1] if c is not None and c.complete and not c.skip else None
        fits = {k: ransac_fit(m.winter, m.summer, k, args.threshold) for k in MODELS}
        quality = {k: {**fit_quality(f, m.winter, (480, 640), anchor),
                       **inlier_places(m.winter[f.inliers], winter["valid"], near)} for k, f in fits.items()}
        for k, f in fits.items():
            issues = check_fit(quality[k])
            rows.append({"tree_id": rec.tree_id, "model": k, "ok": not issues,
                         **{a: b if isinstance(b, int) else round(float(b), 4) for a, b in quality[k].items()},
                         "issues": "; ".join(issues),
                         "matrix": json.dumps(np.round(f.matrix, 6).tolist()) if f.matrix is not None else ""})
            if k == args.model and issues:
                flagged[rec.tree_id] = issues
        if mask is None:
            continue
        fit, q = fits[args.model], quality[args.model]
        flag = "  FLAG" if rec.tree_id in flagged else ""
        label = (f"{rec.tree_id} {args.matcher}/{args.model}: {fit.n_inliers}/{len(m)} inliers"
                 + (f", scale {q['scale']:.2f}, rot {q['rotation']:+.1f}" if "scale" in q else "") + flag)
        if args.overlays:
            save_image(draw_alignment(winter["rgb"], summer["rgb"], mask, m, fit, label),
                       out / f"{rec.tree_id}_{args.model}.jpg", failed, quality=85)
        tile = draw_summer(summer["rgb"], mask, m, fit).resize((320, 240))
        ImageDraw.Draw(tile).text((4, 4), f"{rec.tree_id} {fit.n_inliers}/{len(m)}{flag}", fill="yellow")
        tiles.append(np.asarray(tile))

    try:
        with open(out / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            wr.writeheader()
            wr.writerows(rows)
    except OSError as e:
        failed.append(f"{out / 'summary.csv'}: {e.strerror or e}")
    for i in range(0, len(tiles), 24):
        save_image(contact_sheet(tiles[i:i + 24]), out / f"sheet_{args.model}_{i // 24 + 1:02d}.jpg",
                   failed, quality=85)

    print(f"\nFitted {len(trees)} trees ({time.time() - t0:.0f} s). Inlier threshold {args.threshold} px.")
    print("Medians over trees. 'on tree': share of inliers within "
          f"{NEAR_TREE_PX} px of the winter mask; 'no depth': share without a winter depth reading "
          "(background); scale, rotation and shift at the fork click.")
    print(f"{'model':<12}{'usable':>9}{'inliers':>9}{'on tree':>9}{'no depth':>10}{'scale':>8}"
          f"{'|rot|':>7}{'shift x, y':>13}")
    for k in MODELS:
        r = [x for x in rows if x["model"] == k]
        ok = [x for x in r if "scale" in x]

        def med(key, rows=ok):
            v = [x[key] for x in rows if np.isfinite(x[key])]
            return float(np.median(v)) if v else float("nan")

        rot = float(np.median([abs(x["rotation"]) for x in ok])) if ok else float("nan")
        print(f"{k:<12}{sum(x['ok'] for x in r):>5}/{len(r):<3}{med('inliers'):>9.0f}{med('on_tree'):>9.2f}"
              f"{med('no_depth'):>10.2f}{med('scale'):>8.3f}{rot:>7.1f}{med('dx'):>7.0f},{med('dy'):>5.0f}")
    if flagged:
        print(f"\nFlagged for {args.model} ({len(flagged)} trees; limits are provisional):")
        for tid, why in sorted(flagged.items()):
            print(f"  {tid}: {'; '.join(why)}")
    print(f"\nResults in {out}")
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
