"""The tree in 3D with bends: smooth curves for the trunk and leaders that may leave the tree's plane.

:mod:`treeskel.tree3d` puts every point of the skeleton on one plane. The
trees are flat to a median 1.7 cm, but the depth along a winter trace shows
smooth structure beyond that: leaders that bow 5–10 cm towards or away from
the camera over tens of centimetres, and trunks that stand behind the
leaders' plane. Here each branch may bend out of the plane where the depth
says so, and stays on it where the depth says little. For each tree:

1. **Depth to each branch's axis** (:func:`axis_distance`): the depth reads
   a branch's near surface, so its axis lies about one radius farther (the
   radius from the 2D width, at the point's distance). Without this, the
   trunk (radius about 5 cm) would read about 3 cm nearer than the leaders
   (about 2 cm): a false step at the fork.
2. **Which readings count** (:func:`ridge_contrast`): in winter, every
   reading (the bare tree is in view). In summer, only where the branch is a
   *depth ridge*: both sides, :data:`RIDGE_GAP` px beyond its edge, read at
   least :data:`RIDGE_CONTRAST` units (about 5 cm) farther, or nothing. Inside
   the canopy the reading at a branch pixel is usually a leaf in front, and
   the depth there is a smooth surface with no ridge; the trunk below the
   canopy, and wood seen against a gap, pass.
3. **The plane**: :func:`treeskel.tree3d.fit_plane` through every reading's
   axis point (the reference the bends are measured from).
4. **The bends** (:func:`fit_bends`): along each part, the distance to its
   axis along the pixel's ray is ``Z = Z_plane + h(s)``. The bend ``h`` is a
   smooth function of the length ``s`` along the branch, with the plane as a
   preference: a Gaussian-process prior of standard deviation ``spread`` (τ)
   and length ``length`` (L), whose precision is
   ``(1 + L⁴ d⁴/ds⁴) / (2√2 τ² L)``. Well-observed sections follow the
   readings; sections with few or no readings stay near the plane, with a
   posterior standard deviation (:func:`posterior_sd`) that says so. The
   three parts share the fork's bend, so they stay connected, and each may
   kink there. Each reading counts as ``Δs / ℓ`` independent ones
   (:data:`CORR_PX`): neighbouring readings share most of their pixels. The
   fit is robust: Huber weights first, then Tukey's biweight, so background
   readings and depth jumps (a post or another branch) get no weight.
5. **The 3D curves** (:func:`bent_points`): every point moves along its own
   ray, so the bent skeleton still projects exactly onto the 2D trace.
6. **Is there anything to find?** (:func:`block_cv`): blocks of
   :data:`BLOCK_PX` px along each part are left out in turn (with
   :data:`BUFFER_PX` px either side, so correlated noise cannot leak in), and
   their depth is predicted from the rest, by the plane alone (τ = 0) and by
   each prior on :data:`GRID`. Smooth bends predict a left-out block from its
   neighbours; noise does not.
7. **Measures** (:func:`bend_measures`), alignment-free so that winter and
   summer can be compared tree by tree: how far each part leaves the plane;
   the trunk's angle out of the V's plane (the plane through both leader
   lines; positive when the trunk's lower end is behind it); each leader's
   bow out of that plane at the middle of its stretch; the V opening between
   the bent leaders' lines; the leaders' lengths. With the plane model the
   trunk's angle and the bows are 0.

Distances are in metres: the depth units times :func:`per_unit`, which puts
the median development fork at 2.0 m for each camera setting, as
:func:`treeskel.tree3d.scales` does.

Usage (development rows only):

```
python -m treeskel.bend3d data/raw      # -> docs/results/bend3d/
```
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from treeskel.depth3d import DISTANCE_M, METRES_PER_UNIT, ZERO, Camera
from treeskel.guided_gt import arc_length, path_normals
from treeskel.shape import UNIT_PX, ShapeParams, line_direction
from treeskel.tree3d import PARTS, angle, depth_along, fit_plane, on_plane, stretch

FORK_VALUE = ZERO - DISTANCE_M / METRES_PER_UNIT
"""The depth value of the median development tree's winter fork (about 132.9), which
:data:`treeskel.depth3d.METRES_PER_UNIT` puts at :data:`treeskel.depth3d.DISTANCE_M`."""

RIDGE_GAP = 6
"""A summer branch's sides are read this many px beyond its edge (depth reaches about 2 px past an edge)."""

RIDGE_CONTRAST = 4.0
"""Both sides must read at least this many depth units (about 5 cm) farther for a summer reading to count."""

CORR_PX = 7.0
"""Each depth reading along a path counts as ``Δs / CORR_PX`` independent ones: it is the median of a 3 px
disc (:func:`treeskel.tree3d.depth_along`), 7 px across, so neighbouring readings share most of their
pixels."""

BLOCK_PX = 40
"""Length of the blocks :func:`block_cv` leaves out (px along the branch; about 13 cm at 2 m)."""

BUFFER_PX = 10
"""Readings this many px either side of a left-out block are left out too, but not scored."""

MAX_TURN = 60.0
"""A plane turned more than this (degrees) from facing the camera is a failed fit (:func:`prepare` refuses
it): the 449 winter planes turn a median 4.4°, and :data:`treeskel.tree3d.TURNED` lists those past 20°."""

GRID = tuple((spread, length) for spread in (0.01, 0.02, 0.04, 0.08, 0.16) for length in (0.05, 0.1, 0.2, 0.4))
"""The priors :func:`block_cv` tries: ``(spread, length)`` in metres (τ and L), besides the plane (τ = 0)."""


def per_unit(camera: Camera) -> float:
    """Metres per depth unit for a camera setting: the median development fork at 2.0 m.

    The same scale as :func:`treeskel.tree3d.scales` (to 0.2%), without needing every tree.

    Example:
        >>> round(per_unit(Camera()), 5)
        0.01197
    """
    return DISTANCE_M / (camera.zero - FORK_VALUE)


@dataclass(frozen=True)
class BendParams:
    """The prior on the bends and the robust fit.

    The default prior, τ 2 cm and L 20 cm, is the one :func:`block_cv` chose on the 449 development trees
    (``docs/results/bend3d/``; every fold chose it too), which :mod:`treeskel.model3d` reads from there.

    Attributes:
        spread: τ, the prior standard deviation of a bend (m); 0 is the plane.
        length: L, the length over which a bend changes (m).
        huber: Huber's constant (robust standard deviations).
        tukey: Tukey's biweight constant (robust standard deviations).
        iters: Most iterations of each robust stage.
        corr_px: See :data:`CORR_PX`.
        min_noise: Smallest noise standard deviation used (m).
        uncertain: A point is uncertain where its posterior standard deviation exceeds this fraction
            of ``spread``.
    """

    spread: float = 0.02
    length: float = 0.2
    huber: float = 1.345
    tukey: float = 4.685
    iters: int = 30
    corr_px: float = CORR_PX
    min_noise: float = 0.003
    uncertain: float = 0.5


# --------------------------------------------------------------------------
# The readings
# --------------------------------------------------------------------------

def axis_distance(values: np.ndarray, radii_px: np.ndarray, camera: Camera) -> np.ndarray:
    """Distance (units, along the optical axis) to a branch's axis, from the depth of its near surface.

    The axis lies one radius behind the surface; a radius of ``r`` px at distance ``Z`` is ``r Z / f``.

    Example:
        >>> axis_distance(np.array([100.0]), np.array([10.0]), Camera(zero=300.0, focal=500.0)).tolist()
        [204.0]
    """
    z = camera.distance(values)
    return z * (1.0 + np.asarray(radii_px, float) / camera.focal)


def ridge_contrast(depth: np.ndarray, valid: np.ndarray, path: np.ndarray, radii_px: np.ndarray,
                   gap: float = RIDGE_GAP) -> np.ndarray:
    """How much farther than a branch its two sides read (depth units; the smaller of the two sides).

    Each side is read ``radius + gap`` px out along the path's normal (median within 2 px); a side with no
    reading counts as infinitely far. NaN where the branch itself has no reading.
    """
    p = np.asarray(path, float).reshape(-1, 2)
    normals = path_normals(np.rint(p).astype(int), 6)
    centre = depth_along(depth, valid, p)
    out = np.full(len(p), np.inf)
    for sign in (-1.0, 1.0):
        side = p + sign * (np.asarray(radii_px, float) + gap)[:, None] * normals
        v = depth_along(depth, valid, side, radius=2, min_valid=3)
        out = np.minimum(out, np.where(np.isnan(v), np.inf, centre - v))
    out[np.isnan(centre)] = np.nan
    return out


# --------------------------------------------------------------------------
# The tree as three chains from the fork
# --------------------------------------------------------------------------

@dataclass
class Prepared:
    """One tree's skeleton and readings, ready to fit: the parts as chains of nodes from the fork.

    Node 0 is the fork; each part's other points follow it outwards (the trunk reversed, so towards the
    root). Arrays are per node unless named per part.

    Attributes:
        paths: Each part's ``(row, col)`` photo points, in the caller's order (trunk root to fork).
        nodes: Per part, the node of each of its points (same order as ``paths``).
        chains: Per part, its nodes from the fork outwards (starting with 0).
        rays: ``(N, 3)`` ray directions ``(x, y, 1)``.
        z_axis: Distance to the branch's axis (m) where read, else NaN.
        anchor: Readings that count (all readings in winter; ridges in summer).
        cell_px: Each node's share of its path's length (px), for its readings' weight.
        plane: :func:`treeskel.tree3d.fit_plane` through every reading's axis point (m).
        z_plane: The plane's distance along each ray (m).
        s: Per part, the length along the chain on the plane (m), from the fork.
        s_px: Per part, the length along the chain in the photo (px), from the fork.
        per_unit: Metres per depth unit.
    """

    paths: dict
    nodes: dict
    chains: dict
    rays: np.ndarray
    z_axis: np.ndarray
    anchor: np.ndarray
    cell_px: np.ndarray
    plane: dict
    z_plane: np.ndarray
    s: dict
    s_px: dict
    per_unit: float

    @property
    def d(self) -> np.ndarray:
        """The readings' distance behind the plane along their rays (m), NaN where none count."""
        return np.where(self.anchor, self.z_axis - self.z_plane, np.nan)


def prepare(paths: dict, depth: np.ndarray, valid: np.ndarray, radii: dict, camera: Camera = Camera(),
            ridge: bool = False, scale: float | None = None) -> Prepared:
    """One tree's skeleton with its depth, plane and chains.

    Args:
        paths: ``{"trunk", "left", "right"}`` ``(N, 2)`` ``(row, col)`` photo points: the trunk from the root
            to the fork, each leader from the fork to its tip.
        depth, valid: The photo's depth and where it has readings.
        radii: Each part's radius at each point (photo px).
        camera: The depth map and camera.
        ridge: Count only the readings where the branch is a depth ridge (:func:`ridge_contrast`; summer).
        scale: Metres per unit (default :func:`per_unit` of ``camera``).

    Raises:
        ValueError: If fewer than 20 points have depth (no plane).
    """
    pu = per_unit(camera) if scale is None else scale
    paths = {p: np.asarray(paths[p], float).reshape(-1, 2) for p in PARTS}
    fork = paths["trunk"][-1]
    # The chains from the fork outwards; a leader's first point is the fork itself when it repeats it.
    own = {"trunk": paths["trunk"][::-1][1:]}
    for p in ("left", "right"):
        own[p] = paths[p][1:] if len(paths[p]) and np.allclose(paths[p][0], fork) else paths[p]
    rc = np.concatenate([fork[None], *(own[p] for p in PARTS)])
    rad = np.concatenate([np.asarray(radii["trunk"], float)[-1:],
                          *(np.asarray(radii[p], float)[::-1][1:] if p == "trunk"
                            else np.asarray(radii[p], float)[len(radii[p]) - len(own[p]):] for p in PARTS)])
    chains, nodes, k = {}, {}, 1
    for p in PARTS:
        idx = np.arange(k, k + len(own[p]))
        chains[p] = np.r_[0, idx]
        k += len(own[p])
        if p == "trunk":
            nodes[p] = chains[p][::-1]                          # root ... fork
        else:
            nodes[p] = chains[p] if len(paths[p]) > len(own[p]) else idx
    values = depth_along(depth, valid, rc)
    z_axis = axis_distance(values, rad, camera) * pu
    read = np.isfinite(z_axis)
    if read.sum() < 20:
        raise ValueError(f"only {int(read.sum())} skeleton points have depth: no plane")
    anchor = read.copy()
    if ridge:
        anchor &= ridge_contrast(depth, valid, rc, rad) >= RIDGE_CONTRAST
    rays = camera.rays(rc)
    plane = fit_plane(rays[read] * z_axis[read, None])
    facing = float(np.degrees(np.arccos(np.clip(-plane["normal"][2], -1, 1))))
    if facing > MAX_TURN:
        # Rays nearly along the plane would put the skeleton metres away: the fit has failed.
        raise ValueError(f"the tree's plane is turned {facing:.0f} degrees from the camera: a failed fit")
    z_plane = on_plane(rays, plane)[:, 2]
    on = rays * z_plane[:, None]
    s, s_px = {}, {}
    cell_px = np.zeros(len(rc))
    for p in PARTS:
        c = chains[p]
        s[p] = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(on[c], axis=0), axis=1))]
        steps = np.linalg.norm(np.diff(rc[c], axis=0), axis=1)
        s_px[p] = np.r_[0.0, np.cumsum(steps)]
        half = np.r_[0.0, steps / 2] + np.r_[steps / 2, 0.0]   # half of each neighbouring step
        cell_px[c] += half
    return Prepared(paths=paths, nodes=nodes, chains=chains, rays=rays, z_axis=z_axis, anchor=anchor,
                    cell_px=cell_px, plane=plane, z_plane=z_plane, s=s, s_px=s_px, per_unit=pu)


def prior_terms(prep: Prepared) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """The prior's two parts on the chains' nodes: ``(mass, curvature)``.

    ``hᵀ mass h ≈ ∫ h² ds`` and ``hᵀ curvature h ≈ ∫ (d²h/ds²)² ds``, summed over the three chains, with
    the second differences taken within each chain (each may kink at the fork).
    """
    n = len(prep.z_axis)
    mass = np.zeros(n)
    rows, cols, vals = [], [], []
    r = 0
    for p in PARTS:
        c, s = prep.chains[p], prep.s[p]
        ds = np.maximum(np.diff(s), 1e-4)
        mass[c] += np.r_[0.0, ds / 2] + np.r_[ds / 2, 0.0]
        for j in range(1, len(c) - 1):
            a, b = ds[j - 1], ds[j]
            w = np.sqrt((a + b) / 2)
            rows += [r, r, r]
            cols += [c[j - 1], c[j], c[j + 1]]
            vals += [w * 2 / (a * (a + b)), -w * 2 / (a * b), w * 2 / (b * (a + b))]
            r += 1
    g = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    return sparse.diags(np.maximum(mass, 1e-6)).tocsc(), (g.T @ g).tocsc()


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

def precision(terms: tuple, params: BendParams) -> sparse.csc_matrix:
    """The prior's precision matrix for ``spread`` τ and ``length`` L: ``(mass + L⁴ curvature) / (2√2 τ² L)``.

    The marginal standard deviation of a long chain under it is τ.
    """
    mass, curv = terms
    c = 1.0 / (2 * np.sqrt(2) * params.spread ** 2 * params.length)
    return (c * (mass + params.length ** 4 * curv)).tocsc()


def _robust_scale(r: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def fit_bends(prep: Prepared, params: BendParams = BendParams(), d: np.ndarray | None = None,
              hold: np.ndarray | None = None, terms: tuple | None = None, mean: np.ndarray | None = None) -> dict:
    """The bends ``h`` (m behind the plane, along each ray) at every node, fitted robustly.

    Solves ``(Q + W/σ²) h = W d/σ² + Q m`` with ``Q`` the prior's :func:`precision`, ``m`` its mean (the
    plane, 0, unless ``mean`` is given), ``d`` the readings' distance behind the plane and ``W`` each
    reading's weight (its robust weight times ``cell_px / corr_px``). Huber weights with the noise ``σ``
    re-estimated each round, then Tukey's biweight with ``σ`` fixed.

    Args:
        prep: The tree (:func:`prepare`).
        params: The prior and the robust fit; ``spread`` 0 gives the plane (``h`` = 0).
        d: Readings behind the plane to use instead of ``prep.d`` (e.g. against a refitted plane).
        hold: Nodes whose readings are left out.
        terms: :func:`prior_terms`, to reuse across fits.
        mean: The prior's mean at each node (m behind the plane), e.g. the typical winter shape
            (:func:`prior_mean`); where nothing is read the bends follow it instead of the plane.

    Returns:
        ``{"h", "weights", "sigma", "observed", "system"}``: ``system`` is the final ``Q + W/σ²`` (for
        :func:`posterior_sd`), None for the plane.
    """
    d = prep.d if d is None else d
    obs = np.isfinite(d) & (True if hold is None else ~hold)
    n = len(d)
    if params.spread <= 0:
        return {"h": np.zeros(n), "weights": obs.astype(float), "sigma": float("nan"), "observed": obs,
                "system": None}
    q = precision(terms or prior_terms(prep), params)
    m = np.zeros(n) if mean is None else np.asarray(mean, float)
    if obs.sum() == 0:                                  # nothing seen: the prior alone, its mean and its spread
        return {"h": m.copy(), "weights": np.zeros(n), "sigma": float("nan"), "observed": obs, "system": q}
    y = np.where(obs, d, 0.0)
    cell = prep.cell_px / params.corr_px
    sigma = max(_robust_scale((d - m)[obs]), params.min_noise)
    w = obs.astype(float)
    h = m.copy()
    qm = q @ m

    def solve(weights, s):
        """The bends for these readings' weights and noise level, and the system solved."""
        a = (q + sparse.diags(cell * weights / s ** 2)).tocsc()
        return spsolve(a, cell * weights * y / s ** 2 + qm), a

    a = None
    # Huber: bounded influence, the noise level from the residuals each round.
    for _ in range(params.iters):
        h_new, a = solve(w, sigma)
        r = np.where(obs, d - h_new, 0.0)
        sigma = max(_robust_scale(r[obs]), params.min_noise)
        w = np.where(obs, np.minimum(1.0, params.huber * sigma / np.maximum(np.abs(r), 1e-12)), 0.0)
        done = np.max(np.abs(h_new - h)) < 1e-5
        h = h_new
        if done:
            break
    # Tukey's biweight from there, with the noise level fixed: far readings get no weight at all.
    for _ in range(params.iters):
        r = np.where(obs, d - h, 0.0)
        u = r / (params.tukey * sigma)
        w = np.where(obs & (np.abs(u) < 1), (1 - u ** 2) ** 2, 0.0)
        h_new, a = solve(w, sigma)
        done = np.max(np.abs(h_new - h)) < 1e-5
        h = h_new
        if done:
            break
    return {"h": h, "weights": w, "sigma": sigma, "observed": obs, "system": a}


def posterior_sd(fit: dict) -> np.ndarray:
    """The posterior standard deviation of each node's bend (m), with the weights and noise as fitted.

    It treats the prior, the noise and the plane as known, so it is a lower bound; 0 for the plane.
    """
    a = fit["system"]
    n = len(fit["h"])
    if a is None:
        return np.zeros(n)
    if n <= 4000:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(a.toarray())), 0, None))
    lu = splu(a.tocsc())
    return np.sqrt(np.clip([lu.solve(np.eye(1, n, i).ravel())[i] for i in range(n)], 0, None))


def bent_points(prep: Prepared, h: np.ndarray) -> np.ndarray:
    """The nodes in 3D (m, the camera's frame): each moved along its ray to ``Z_plane + h``."""
    return prep.rays * (prep.z_plane + h)[:, None]


def behind_plane(prep: Prepared, points: np.ndarray) -> np.ndarray:
    """Signed distance of 3D points behind the tree's plane (m; positive = farther from the camera)."""
    n, c = prep.plane["normal"], prep.plane["centre"]
    return -(np.asarray(points) - c) @ n


# --------------------------------------------------------------------------
# The typical winter shape, as the prior's mean
# --------------------------------------------------------------------------

STATIONS = {"trunk": np.round(np.arange(0.0, 0.801, 0.02), 3), "left": np.round(np.arange(0.0, 2.001, 0.02), 3),
            "right": np.round(np.arange(0.0, 2.001, 0.02), 3)}
"""Where each part's bend is sampled for the typical shape: metres from the fork along the part (on the
tree's plane), every 2 cm; the trunk downwards."""

MIN_TREES = 20
"""A station counts in the typical shape only where at least this many trees reach it."""


def profile(prep: Prepared, h: np.ndarray, part: str, stations: np.ndarray | None = None) -> np.ndarray:
    """One part's bend (m behind the plane) at the stations, NaN beyond the part's end in view.

    Example:
        >>> class P: chains = {"left": np.array([0, 1, 2])}; s = {"left": np.array([0.0, 0.1, 0.2])}
        >>> profile(P, np.array([0.0, 0.01, 0.03]), "left", np.array([0.0, 0.15, 0.3])).round(3).tolist()
        [0.0, 0.02, nan]
    """
    st = STATIONS[part] if stations is None else np.asarray(stations, float)
    return np.interp(st, prep.s[part], np.asarray(h, float)[prep.chains[part]], right=np.nan)


def typical_shape(profiles: Sequence[dict], min_trees: int = MIN_TREES) -> dict:
    """The typical bend of each part: the median over trees of their winter profiles (:func:`profile`).

    Each part's stations end before the first one fewer than ``min_trees`` trees reach. The three parts
    share the fork, so their first value is set to the median fork bend.

    Args:
        profiles: Per tree, ``{part: values at STATIONS[part]}``.

    Returns:
        ``{part: {"stations": [...], "bend": [...], "q1": [...], "q3": [...], "n": [...]}}`` in metres.
    """
    out = {}
    for part in PARTS:
        a = np.array([p[part] for p in profiles], float).reshape(-1, len(STATIONS[part]))
        n = np.isfinite(a).sum(0)
        keep = int(np.argmax(n < min_trees)) if np.any(n < min_trees) else len(n)
        a = a[:, :keep]
        out[part] = {"stations": STATIONS[part][:keep].tolist(),
                     "bend": np.nanmedian(a, 0).tolist() if keep else [],
                     "q1": np.nanpercentile(a, 25, 0).tolist() if keep else [],
                     "q3": np.nanpercentile(a, 75, 0).tolist() if keep else [],
                     "n": n[:keep].astype(int).tolist()}
    fork = [np.array([p[part][0] for p in profiles], float) for part in PARTS]
    f = float(np.nanmedian(np.concatenate(fork))) if len(profiles) else 0.0
    for part in PARTS:
        if out[part]["bend"]:
            out[part]["bend"][0] = f
    return out


def prior_mean(prep: Prepared, shape: dict) -> np.ndarray:
    """The typical shape at each node of a tree (m behind its plane): each part's profile at the node's
    distance from the fork, holding its last value beyond the stations it covers."""
    m = np.zeros(len(prep.z_axis))
    for part in PARTS:
        st, b = shape[part]["stations"], shape[part]["bend"]
        if len(st):
            m[prep.chains[part]] = np.interp(prep.s[part], st, b)
    firsts = [shape[p]["bend"][0] for p in PARTS if len(shape[p]["bend"])]
    m[0] = float(np.mean(firsts)) if firsts else 0.0
    return m


# --------------------------------------------------------------------------
# Is there anything to find? Block cross-validation along the branches
# --------------------------------------------------------------------------

def setting_key(params: BendParams) -> str:
    """A prior's name in tables: ``plane`` or ``tau <cm> L <cm>``."""
    return "plane" if params.spread <= 0 else f"tau {params.spread * 100:g} L {params.length * 100:g}"


def blocks(prep: Prepared, block_px: float = BLOCK_PX, buffer_px: float = BUFFER_PX) -> list[tuple]:
    """The left-out blocks: ``(part, scored, removed)`` node masks.

    Each part is cut into blocks of ``block_px`` px from the fork outwards (the fork belongs to the trunk's
    first block). A block's readings and those within ``buffer_px`` px of it are removed from the fit; only
    the block's own are scored. Near the fork the buffer reaches into the other parts.
    """
    n = len(prep.z_axis)
    out = []
    for p in PARTS:
        c, s = prep.chains[p], prep.s_px[p]
        own = np.ones(len(c), bool) if p == "trunk" else np.r_[False, np.ones(len(c) - 1, bool)]
        for k in range(int(s[-1] // block_px) + 1):              # the last block holds the chain's end
            lo, hi = k * block_px, (k + 1) * block_px
            scored, removed = np.zeros(n, bool), np.zeros(n, bool)
            scored[c[own & (s >= lo) & (s < hi)]] = True
            removed[c[(s >= lo - buffer_px) & (s < hi + buffer_px)]] = True
            if lo - buffer_px < 0:                          # the buffer crosses the fork
                for q in PARTS:
                    removed[prep.chains[q][prep.s_px[q] < buffer_px - lo]] = True
            out.append((p, scored, removed))
    return out


def block_cv(prep: Prepared, grid: Sequence[tuple[float, float]] = GRID, base: BendParams = BendParams(),
             block_px: float = BLOCK_PX, buffer_px: float = BUFFER_PX) -> dict[str, dict]:
    """Left-out blocks' depth predicted by the plane and by each prior on ``grid``.

    For each block (:func:`blocks`) the plane is fitted again without the removed readings, and the block's
    readings are predicted by that plane (``h`` = 0) and by the bends fitted to the rest.

    Returns:
        ``{setting_key: {part: [sum of |error| (m), sum of error² (m²), readings]}}``, with ``"plane"`` first.
    """
    terms = prior_terms(prep)
    settings = [replace(base, spread=0.0)] + [replace(base, spread=t, length=length) for t, length in grid]
    out = {setting_key(s): {p: [0.0, 0.0, 0] for p in PARTS} for s in settings}
    read = np.isfinite(prep.z_axis)
    for part, scored, removed in blocks(prep, block_px, buffer_px):
        target = scored & prep.anchor
        keep = read & ~removed
        if not target.any() or keep.sum() < 20:
            continue
        plane = fit_plane(prep.rays[keep] * prep.z_axis[keep, None])
        if np.degrees(np.arccos(np.clip(-plane["normal"][2], -1, 1))) > MAX_TURN:
            continue
        d = np.where(prep.anchor, prep.z_axis - on_plane(prep.rays, plane)[:, 2], np.nan)
        for s in settings:
            h = fit_bends(prep, s, d=d, hold=removed, terms=terms)["h"]
            e = d[target] - h[target]
            acc = out[setting_key(s)][part]
            acc[0] += float(np.abs(e).sum())
            acc[1] += float((e ** 2).sum())
            acc[2] += int(len(e))
    return out


# --------------------------------------------------------------------------
# Measures that need no common frame between the seasons
# --------------------------------------------------------------------------

BEND_MEASURES = ("v_opening", "trunk_out", "left_bow", "right_bow", "left_tilt", "right_tilt", "left_tilt_whole",
                 "right_tilt_whole", "fork_behind", "trunk_out_max",
                 "left_out_max", "right_out_max", "trunk_out_rms", "left_out_rms", "right_out_rms",
                 "left_length", "right_length", "trunk_uncertain", "left_uncertain", "right_uncertain",
                 "trunk_anchors", "left_anchors", "right_anchors", "sigma")
"""Columns of :func:`bend_measures`: degrees; ``*_bow``, ``fork_behind``, ``*_out_*`` and ``sigma`` in cm;
lengths in m; ``*_uncertain`` and ``*_anchors`` shares of each part's points. ``*_tilt`` is a leader's line on
its stretch out of the tree's plane, ``*_tilt_whole`` the line through all of it in view (degrees, positive:
the upper end behind)."""


def bend_measures(prep: Prepared, h: np.ndarray, sd: np.ndarray | None = None, params: BendParams = BendParams(),
                  zoom: float = 1.0, shape: ShapeParams = ShapeParams(), sigma: float = float("nan")) -> dict:
    """One tree's measures of its 3D curves (see the module docstring).

    Args:
        prep: The tree (:func:`prepare`).
        h: The bends (:func:`fit_bends`); zeros for the plane.
        sd: Their posterior standard deviations (:func:`posterior_sd`), for the uncertain shares.
        params: The prior (its ``spread`` and ``uncertain`` set what is uncertain).
        zoom: Photo px per winter px, for the stretches (:data:`treeskel.shape.SUMMER_SCALE` in summer).
        shape: The stretches of :mod:`treeskel.shape`: the trunk's top and each leader's line.
        sigma: The fitted noise (m), reported in cm.

    Returns:
        :data:`BEND_MEASURES`, NaN where undefined (a leader or the trunk's top too short).
    """
    out = dict.fromkeys(BEND_MEASURES, float("nan"))
    pts = bent_points(prep, h)
    behind = behind_plane(prep, pts)
    unit = UNIT_PX * zoom
    out["fork_behind"] = float(behind[0] * 100)
    out["sigma"] = float(sigma * 100)
    for p in PARTS:
        k = prep.nodes[p]
        out[f"{p}_out_max"] = float(np.max(np.abs(behind[k])) * 100)
        out[f"{p}_out_rms"] = float(np.sqrt(np.mean(behind[k] ** 2)) * 100)
        out[f"{p}_anchors"] = float(np.mean(prep.anchor[k]))
        if sd is not None and params.spread > 0:
            out[f"{p}_uncertain"] = float(np.mean(sd[k] > params.uncertain * params.spread))
    lines, parts = {}, {}
    back = -prep.plane["normal"]                        # away from the camera, out of the tree's plane
    for side in ("left", "right"):
        k = prep.nodes[side]
        out[f"{side}_length"] = float(arc_length(pts[k])[-1]) if len(k) > 1 else float("nan")
        if len(k) >= shape.min_points:
            out[f"{side}_tilt_whole"] = float(np.degrees(np.arcsin(np.clip(line_direction(pts[k]) @ back, -1, 1))))
        m = stretch(prep.paths[side], unit, shape)
        if m.sum() >= shape.min_points:
            parts[side] = pts[k][m]
            lines[side] = line_direction(parts[side])
            out[f"{side}_tilt"] = float(np.degrees(np.arcsin(np.clip(lines[side] @ back, -1, 1))))
    if len(lines) < 2:
        return out
    out["v_opening"] = angle(lines["left"], lines["right"])
    normal = np.cross(lines["left"], lines["right"])
    normal /= np.linalg.norm(normal)
    if normal[2] > 0:                                   # face the camera, as the tree's plane
        normal = -normal
    for side, q in parts.items():
        # The bow: the middle of the stretch against the middle of its chord, out of the V's plane.
        sq = arc_length(q)
        mid = np.array([np.interp(sq[-1] / 2, sq, q[:, j]) for j in range(3)])
        out[f"{side}_bow"] = float(-(mid - (q[0] + q[-1]) / 2) @ normal * 100)
    trunk = prep.paths["trunk"][::-1]                    # from the fork down
    st = arc_length(trunk)
    if len(trunk) > 1 and st[-1] >= shape.min_trunk:
        top = st <= max(shape.trunk_reach * unit, shape.min_trunk)
        if top.sum() >= 3:
            down = line_direction(pts[prep.nodes["trunk"][::-1]][top])
            out["trunk_out"] = float(np.degrees(np.arcsin(np.clip(-down @ normal, -1, 1))))
    return out


# --------------------------------------------------------------------------
# One tree, every source
# --------------------------------------------------------------------------

def mask_skeleton(mask: np.ndarray) -> dict | None:
    """A 256x256 summer mask's trunk and leaders in photo px, with each point's radius (photo px), or None."""
    from scipy import ndimage as ndi

    from treeskel.io import IMAGE_SHAPE, MASK_SHAPE
    from treeskel.shape import mask_branches, mask_to_image

    m = np.asarray(mask, bool)
    br = mask_branches(m)
    if br is None:
        return None
    k = IMAGE_SHAPE[0] / MASK_SHAPE[0]
    half = ndi.distance_transform_edt(ndi.binary_fill_holes(m))
    return {"paths": {p: mask_to_image(br[p]) for p in PARTS},
            "radii": {p: np.maximum(half[br[p][:, 0], br[p][:, 1]], 0.5) * k for p in PARTS}}


def winter_skeleton(rec, clicks) -> tuple[dict, dict]:
    """A tree's winter images and its click-guided trace with each point's drawing radius (photo px)."""
    from treeskel.guided_gt import guided_gt
    from treeskel.io import load_tree

    images = load_tree(rec, "winter")
    gt = guided_gt(images["rgb"], images["depth"], images["valid"], clicks)
    return images, {"paths": {p: getattr(gt, p) for p in PARTS}, "radii": {p: gt.radii[p] for p in PARTS}}


def cv_tree(job: tuple) -> dict:
    """Phase 1 for one tree (top-level for worker processes): the winter block cross-validation.

    Args:
        job: ``(record, clicks)``.

    Returns:
        ``{"tree", "row", "radii", "cv", "error"}``; ``cv`` as :func:`block_cv`, None if the plane failed.
    """
    rec, clicks = job
    images, sk = winter_skeleton(rec, clicks)
    out = {"tree": rec.tree_id, "row": rec.row, "radii": sk["radii"], "cv": None, "error": ""}
    try:
        out["cv"] = block_cv(prepare(sk["paths"], images["depth"], images["valid"], sk["radii"]))
    except ValueError as e:
        out["error"] = str(e)
    return out


def measure_tree(job: tuple) -> list[dict]:
    """Phase 2 for one tree (top-level for worker processes): the chosen prior's 3D curves and measures.

    Winter under every camera setting of :data:`treeskel.tree3d.SETTINGS`, with the plane and the bends,
    and the block cross-validation again with :data:`BUFFER_PX` x 3 for the plane and the chosen prior;
    winter with ridge readings only (``"winter ridges"``: the trunk where it stands clear of the ground) and
    summer (the lined-up hand mask and the U-Net's mask, ridge readings only) under the main setting.

    Args:
        job: ``(record, clicks, winter radii, hand mask or None, U-Net mask or None, BendParams)``.
    """
    from treeskel.guided_gt import guided_gt
    from treeskel.io import load_tree
    from treeskel.shape import SUMMER_SCALE
    from treeskel.tree3d import SETTINGS

    rec, clicks, radii, hand, unet, params = job
    images = load_tree(rec, "winter")
    gt = guided_gt(images["rgb"], images["depth"], images["valid"], clicks)
    winter = {"paths": {p: getattr(gt, p) for p in PARTS}, "radii": radii}
    # Winter twice: every reading (the truth), and only the ridges, as summer (a check on the ground's depth).
    sources = [("winter", winter, images, False, 1.0), ("winter ridges", winter, images, True, 1.0)]
    summer = None
    for name, mask in (("hand masks", hand), ("U-Net", unet)):
        sk = mask_skeleton(mask) if mask is not None else None
        if sk is not None:
            summer = summer or load_tree(rec, "summer")
            sources.append((name, sk, summer, True, SUMMER_SCALE))
    rows = []
    for name, sk, im, ridge, zoom in sources:
        for setting, camera in (SETTINGS.items() if name == "winter" else [("main", SETTINGS["main"])]):
            base = {"tree": rec.tree_id, "row": rec.row, "source": name, "setting": setting}
            try:
                prep = prepare(sk["paths"], im["depth"], im["valid"], sk["radii"], camera, ridge=ridge)
            except ValueError as e:
                rows.append({**base, "model": "bent", "error": str(e)})
                continue
            fit = fit_bends(prep, params)
            sd = posterior_sd(fit)
            rows.append({**base, "model": "plane", "error": "",
                         **bend_measures(prep, np.zeros(len(fit["h"])), None, params, zoom)})
            rows.append({**base, "model": "bent", "error": "",
                         **bend_measures(prep, fit["h"], sd, params, zoom, sigma=fit["sigma"])})
            if name == "winter" and setting == "main":
                cv = block_cv(prep, grid=[(params.spread, params.length)], base=params, buffer_px=3 * BUFFER_PX)
                for key, model in (("plane", "plane"), (setting_key(params), "bent")):
                    acc = cv[key]
                    n = sum(acc[p][2] for p in PARTS)
                    rows[-2 if model == "plane" else -1]["cv_wide_mae"] = (
                        sum(acc[p][0] for p in PARTS) / n * 100 if n else float("nan"))
    return rows


# --------------------------------------------------------------------------
# Choosing the prior, and the summary
# --------------------------------------------------------------------------

def cv_table(results: Sequence[dict]) -> list[dict]:
    """Phase 1's results as rows: ``tree, row, setting, part, sum_abs, sum_sq, n`` (sums in m, m²)."""
    rows = []
    for r in results:
        for key, parts in (r["cv"] or {}).items():
            for p, (a, s2, n) in parts.items():
                rows.append({"tree": r["tree"], "row": r["row"], "setting": key, "part": p,
                             "sum_abs": a, "sum_sq": s2, "n": n})
    return rows


def pooled_mae(cv_rows: Sequence[dict], trees: set | None = None, parts: Sequence[str] = PARTS) -> dict[str, float]:
    """Each setting's mean absolute error over all left-out readings of ``trees`` (all if None), in cm."""
    tot: dict[str, list[float]] = {}
    for r in cv_rows:
        if (trees is None or r["tree"] in trees) and r["part"] in parts:
            a = tot.setdefault(r["setting"], [0.0, 0])
            a[0] += r["sum_abs"]
            a[1] += r["n"]
    return {k: a[0] / a[1] * 100 for k, a in tot.items() if a[1]}


def tree_mae(cv_rows: Sequence[dict], setting: str) -> dict[str, float]:
    """Each tree's mean absolute error (cm) over its left-out readings, for one setting."""
    tot: dict[str, list[float]] = {}
    for r in cv_rows:
        if r["setting"] == setting:
            a = tot.setdefault(r["tree"], [0.0, 0])
            a[0] += r["sum_abs"]
            a[1] += r["n"]
    return {t: a[0] / a[1] * 100 for t, a in tot.items() if a[1]}


def choose(cv_rows: Sequence[dict], trees: set | None = None) -> str:
    """The rule fixed before the 449-tree run: the prior on :data:`GRID` with the smallest pooled mean
    absolute error over the left-out blocks of ``trees`` (all if None)."""
    pooled = pooled_mae(cv_rows, trees)
    return min((k for k in pooled if k != "plane"), key=pooled.get)


def params_of(key: str, base: BendParams = BendParams()) -> BendParams:
    """The :class:`BendParams` a :func:`setting_key` names."""
    if key == "plane":
        return replace(base, spread=0.0)
    _, tau, _, length = key.split()
    return replace(base, spread=float(tau) / 100, length=float(length) / 100)


def out_of_fold(cv_rows: Sequence[dict], fold_rows: Sequence[Sequence[int]]) -> dict:
    """The bends against the plane with the prior chosen on the other folds' trees (row-grouped).

    Returns:
        ``{"chosen": [key per fold scored], "trees": {tree: (plane MAE, bent MAE)}}`` (cm); a fold with no
        trees, or with no trees outside it, is skipped.
    """
    row_of = {r["tree"]: r["row"] for r in cv_rows}
    plane = tree_mae(cv_rows, "plane")
    out = {"chosen": [], "trees": {}}
    for rows in fold_rows:
        inside = {t for t, r in row_of.items() if r in rows}
        if not inside or not set(row_of) - inside:          # nothing to score, or nothing to choose on
            continue
        key = choose(cv_rows, set(row_of) - inside)
        out["chosen"].append(key)
        bent = tree_mae([r for r in cv_rows if r["tree"] in inside], key)
        for t in inside:
            if t in bent and t in plane:
                out["trees"][t] = (plane[t], bent[t])
    return out


def _q(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0, "median": None, "q1": None, "q3": None}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "median": float(med), "q1": float(q1), "q3": float(q3)}


def _paired_better(a: dict, b: dict) -> dict:
    """Trees in both: how often ``b``'s value is below ``a``'s, the Wilcoxon p, and both medians."""
    from scipy import stats

    t = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    if len(t) < 3:
        return {"n": len(t)}
    x, y = np.array([a[k] for k in t]), np.array([b[k] for k in t])
    d = x - y
    return {"n": len(t), "better": float(np.mean(d > 0)), "median_a": float(np.median(x)),
            "median_b": float(np.median(y)), "median_gain": float(np.median(d)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


SUMMER_MEASURES = ("trunk_out", "left_bow", "right_bow", "v_opening")
"""The measures compared between the seasons, tree by tree."""


def summer_against_winter(rows: Sequence[dict], source: str, measure: str) -> dict:
    """A summer source's bends and its plane against winter's bends, tree by tree, on one measure.

    Returns:
        ``{"n", "median_bent", "median_plane", "bent_closer", "wilcoxon_p", "spearman"}``: the median
        ``|summer − winter|`` with the summer bends and with the summer plane, the share of trees where the
        bends are closer, the Wilcoxon p of the paired differences, and the rank correlation of the summer
        bends with winter's.
    """
    from scipy import stats

    def get(src, model):
        """One source's and model's ``measure`` per tree (main setting, fitted, finite)."""
        return {r["tree"]: r[measure] for r in rows if r["source"] == src and r["setting"] == "main"
                and r["model"] == model and not r.get("error") and np.isfinite(r.get(measure, np.nan))}

    w, b, p = get("winter", "bent"), get(source, "bent"), get(source, "plane")
    t = [k for k in w if k in b and k in p]
    if len(t) < 3:
        return {"n": len(t)}
    w_, b_, p_ = (np.array([m[k] for k in t]) for m in (w, b, p))
    eb, ep = np.abs(b_ - w_), np.abs(p_ - w_)
    d = ep - eb
    varied = np.std(b_) > 0 and np.std(w_) > 0
    return {"n": len(t), "median_bent": float(np.median(eb)), "median_plane": float(np.median(ep)),
            "bent_closer": float(np.mean(d > 0)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "spearman": float(stats.spearmanr(w_, b_)[0]) if varied else None}


def summarise(cv_rows: Sequence[dict], rows: Sequence[dict], fold_rows: Sequence[Sequence[int]]) -> dict:
    """The report: the cross-validation, winter's bends, their sensitivity, and summer against winter."""
    from treeskel.tree3d import SETTINGS

    chosen = choose(cv_rows)
    oof = out_of_fold(cv_rows, fold_rows)
    report = {"n_trees_cv": len({r["tree"] for r in cv_rows}), "grid": pooled_mae(cv_rows), "chosen": chosen,
              "parts": {p: {k: pooled_mae(cv_rows, parts=[p]).get(k) for k in ("plane", chosen)} for p in PARTS},
              "out_of_fold": {"chosen": oof["chosen"],
                              **_paired_better({t: v[0] for t, v in oof["trees"].items()},
                                               {t: v[1] for t, v in oof["trees"].items()})}}

    def col(source, model, measure, setting="main"):
        """One source's, model's and setting's ``measure`` per tree (fitted, finite)."""
        return {r["tree"]: r[measure] for r in rows if r["source"] == source and r["setting"] == setting
                and r["model"] == model and not r.get("error") and np.isfinite(r.get(measure, np.nan))}

    report["wide_buffer"] = _paired_better(col("winter", "plane", "cv_wide_mae"), col("winter", "bent", "cv_wide_mae"))
    bent = [r for r in rows if r["source"] == "winter" and r["setting"] == "main" and r["model"] == "bent"
            and not r.get("error")]
    report["winter"] = {k: _q([r[k] for r in bent]) for k in BEND_MEASURES}
    for side in ("left", "right"):
        report["winter"][f"{side}_bow_abs"] = _q([abs(r[f"{side}_bow"]) for r in bent])
    vb, vp = col("winter", "bent", "v_opening"), col("winter", "plane", "v_opening")
    report["winter"]["v_change"] = _q([vb[t] - vp[t] for t in vb if t in vp])
    for side in ("left", "right"):
        lb, lp = col("winter", "bent", f"{side}_length"), col("winter", "plane", f"{side}_length")
        report["winter"][f"{side}_length_change"] = _q([(lb[t] - lp[t]) * 100 for t in lb if t in lp])
    report["sensitivity"] = {}
    for setting in SETTINGS:
        sub = [r for r in rows if r["source"] == "winter" and r["setting"] == setting and r["model"] == "bent"
               and not r.get("error")]
        vb_, vp_ = col("winter", "bent", "v_opening", setting), col("winter", "plane", "v_opening", setting)
        report["sensitivity"][setting] = {
            "trunk_out": _q([r["trunk_out"] for r in sub])["median"],
            "bow_abs": _q([abs(r[f"{s}_bow"]) for r in sub for s in ("left", "right")])["median"],
            "out_max_leaders": _q([r[f"{s}_out_max"] for r in sub for s in ("left", "right")])["median"],
            "v_change": _q([vb_[t] - vp_[t] for t in vb_ if t in vp_])["median"]}
    ridges = summer_against_winter(rows, "winter ridges", "trunk_out")
    report["winter_ridges"] = {"trunk_out": _q(col("winter ridges", "bent", "trunk_out").values()),
                               "median_abs_diff": ridges.get("median_bent"), "spearman": ridges.get("spearman"),
                               "n": ridges.get("n", 0)}
    report["summer"] = {}
    for source in ("hand masks", "U-Net"):
        sub = [r for r in rows if r["source"] == source and r["model"] == "bent" and not r.get("error")]
        report["summer"][source] = {
            "n_trees": len(sub),
            "anchors": {p: _q([r[f"{p}_anchors"] for r in sub])["median"] for p in PARTS},
            "uncertain": {p: _q([r[f"{p}_uncertain"] for r in sub])["median"] for p in PARTS},
            "against_winter": {m: summer_against_winter(rows, source, m) for m in SUMMER_MEASURES}}
    # Failed planes under the main setting, and the winter ones under each other setting (whose medians above
    # leave them out).
    report["failed"] = {s: sorted({r["tree"] for r in rows if r["source"] == s and r["setting"] == "main"
                                   and r.get("error")}) for s in ("winter", "winter ridges", "hand masks", "U-Net")}
    report["failed_winter_by_setting"] = {
        setting: sorted({r["tree"] for r in rows if r["source"] == "winter" and r["setting"] == setting
                         and r.get("error")}) for setting in SETTINGS}
    return report


def _f(x, spec=".2f") -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def report_lines(report: dict) -> list[str]:
    """The report as text."""
    q = report["winter"]
    out = [f"Block cross-validation on {report['n_trees_cv']} development trees' winter depth "
           f"({BLOCK_PX} px blocks left out, {BUFFER_PX} px either side), mean absolute error of the left-out "
           "readings (cm):"]
    grid = report["grid"]
    out.append(f"  plane {grid['plane']:.2f}; best priors: " + ", ".join(
        f"{k} {grid[k]:.2f}" for k in sorted((k for k in grid if k != "plane"), key=grid.get)[:5]))
    out.append(f"  chosen (smallest pooled error): {report['chosen']}; by part, plane -> chosen: " + ", ".join(
        f"{p} {_f(v['plane'])} -> {_f(v[report['chosen']])}" for p, v in report["parts"].items()))
    o = report["out_of_fold"]
    if o.get("n", 0) >= 3:
        out.append(f"  out of fold (prior chosen on the other folds' rows: {', '.join(o['chosen'])}): bends better "
                   f"on {o['better']:.0%} of {o['n']} trees; median per tree {o['median_a']:.2f} -> "
                   f"{o['median_b']:.2f} cm (Wilcoxon p {o['wilcoxon_p']:.1e})")
    w = report["wide_buffer"]
    if w.get("n", 0) >= 3:
        out.append(f"  with {3 * BUFFER_PX} px buffers: bends better on {w['better']:.0%} of {w['n']} trees; median "
                   f"{w['median_a']:.2f} -> {w['median_b']:.2f} cm (Wilcoxon p {w['wilcoxon_p']:.1e})")
    out += ["", "Winter bends with the chosen prior (main camera setting), median [quartiles]:"]
    labels = (("trunk_out", "trunk out of the V's plane (deg, + = root behind)"),
              ("left_bow", "left leader's bow (cm, + = behind)"), ("right_bow", "right leader's bow (cm)"),
              ("left_bow_abs", "|left bow| (cm)"), ("right_bow_abs", "|right bow| (cm)"),
              ("left_tilt", "left leader's stretch out of the plane (deg, + = top behind)"),
              ("right_tilt", "right leader's stretch out of the plane (deg)"),
              ("left_tilt_whole", "left leader, whole, out of the plane (deg)"),
              ("right_tilt_whole", "right leader, whole, out of the plane (deg)"),
              ("fork_behind", "fork behind the plane (cm)"),
              ("trunk_out_max", "trunk's farthest from the plane (cm)"),
              ("left_out_max", "left leader's farthest (cm)"), ("right_out_max", "right leader's farthest (cm)"),
              ("v_change", "V opening, bends - plane (deg)"),
              ("left_length_change", "left leader longer by (cm)"),
              ("trunk_uncertain", "trunk points uncertain (share)"),
              ("left_uncertain", "left leader points uncertain"), ("sigma", "depth noise (cm)"))
    for k, label in labels:
        s = q.get(k)
        if s and s["n"]:
            out.append(f"  {label:62s} {s['median']:7.2f} [{s['q1']:.2f}, {s['q3']:.2f}]  (n {s['n']})")
    wr = report.get("winter_ridges", {})
    if wr.get("n", 0) >= 3:
        out.append(f"  with ridge readings only (the trunk clear of the ground): trunk out of the V's plane "
                   f"{wr['trunk_out']['median']:.2f} [{wr['trunk_out']['q1']:.2f}, {wr['trunk_out']['q3']:.2f}]; "
                   f"median |difference| from all readings {wr['median_abs_diff']:.2f} deg, rank correlation "
                   f"{_f(wr['spearman'])} (n {wr['n']})")
    out += ["", "Sensitivity (winter medians under each camera setting):",
            f"  {'setting':22s} {'trunk out':>10s} {'|bow|':>8s} {'leader max':>11s} {'V change':>9s}"]
    for setting, s in report["sensitivity"].items():
        out.append(f"  {setting:22s} {_f(s['trunk_out'], '.1f'):>10s} {_f(s['bow_abs']):>8s} "
                   f"{_f(s['out_max_leaders']):>11s} {_f(s['v_change']):>9s}")
    out += ["", "Summer (ridge readings only) against winter's bends, tree by tree: median |summer - winter| with "
                "the summer bends and with the summer plane; share of trees where the bends are closer; Wilcoxon p; "
                "rank correlation of the summer bends with winter's:"]
    for source, s in report["summer"].items():
        out.append(f"  {source} ({s['n_trees']} trees; readings that count, median share: " + ", ".join(
            f"{p} {_f(v, '.0%')}" for p, v in s["anchors"].items()) + "):")
        for m, a in s["against_winter"].items():
            if a.get("n", 0) >= 3:
                out.append(f"    {m:10s} n {a['n']:3d}  bends {a['median_bent']:6.2f}  plane {a['median_plane']:6.2f}  "
                           f"closer {a['bent_closer']:4.0%}  p {a['wilcoxon_p']:.1e}  rank {_f(a['spearman'])}")
    out += ["", "Failed planes under the main setting (turned more than "
            f"{MAX_TURN:g} deg, or too little depth): " + "; ".join(
                f"{s} {len(t)} ({', '.join(t) or 'none'})" for s, t in report["failed"].items())]
    other = {k: v for k, v in report.get("failed_winter_by_setting", {}).items() if k != "main"}
    if other:
        out.append("  winter under the other settings (left out of their medians): " + "; ".join(
            f"{k} {len(v)}" for k, v in other.items()))
    return out


# --------------------------------------------------------------------------
# The typical winter shape: learning it, and summer with it as the prior's mean
# --------------------------------------------------------------------------

TYPICAL_MEASURES = ("trunk_out", "left_tilt", "right_tilt", "left_tilt_whole", "right_tilt_whole", "v_opening",
                    "left_bow", "right_bow")
"""The measures compared with winter's in the typical-shape run (``--typical``)."""

SUMMER_MODELS = ("plane", "bent", "typical", "bent typical")
"""Summer's 3D curves in the typical-shape run: the plane; the bends with the plane as the prior's mean (as
``model3d`` until now); the typical winter shape alone, ignoring summer's readings; and the bends with the
typical shape as the prior's mean."""


def typical_winter(job: tuple) -> dict:
    """Pass 1 for one tree (top-level for worker processes): its winter bends, their profiles and measures.

    Args:
        job: ``(record, clicks, BendParams)``.

    Returns:
        ``{"tree", "row", "profiles": {part: values at STATIONS[part]}, "measures", "error"}``.
    """
    rec, clicks, params = job
    images, sk = winter_skeleton(rec, clicks)
    out = {"tree": rec.tree_id, "row": rec.row, "profiles": None, "measures": None, "error": ""}
    try:
        prep = prepare(sk["paths"], images["depth"], images["valid"], sk["radii"])
    except ValueError as e:
        out["error"] = str(e)
        return out
    fit = fit_bends(prep, params)
    out["profiles"] = {p: profile(prep, fit["h"], p) for p in PARTS}
    out["measures"] = bend_measures(prep, fit["h"], posterior_sd(fit), params, sigma=fit["sigma"])
    return out


def typical_summer(job: tuple) -> list[dict]:
    """Pass 2 for one tree (top-level for worker processes): summer's curves (:data:`SUMMER_MODELS`) and measures.

    Args:
        job: ``(record, hand mask or None, U-Net mask or None, BendParams, typical shape)``; the shape learnt
            without the tree's own fold.
    """
    from treeskel.io import load_tree
    from treeskel.shape import SUMMER_SCALE

    rec, hand, unet, params, shape = job
    rows, summer = [], None
    for name, mask in (("hand masks", hand), ("U-Net", unet)):
        sk = mask_skeleton(mask) if mask is not None else None
        if sk is None:
            continue
        summer = summer or load_tree(rec, "summer")
        base = {"tree": rec.tree_id, "row": rec.row, "source": name, "setting": "main"}
        try:
            prep = prepare(sk["paths"], summer["depth"], summer["valid"], sk["radii"], ridge=True)
        except ValueError as e:
            rows.append({**base, "model": "bent typical", "error": str(e)})
            continue
        terms = prior_terms(prep)
        mean = prior_mean(prep, shape)
        for model in SUMMER_MODELS:
            if model == "plane":
                h, sd, sigma = np.zeros(len(mean)), None, float("nan")
            elif model == "typical":
                h, sd, sigma = mean, None, float("nan")
            else:
                fit = fit_bends(prep, params, terms=terms, mean=mean if model == "bent typical" else None)
                h, sd, sigma = fit["h"], posterior_sd(fit), fit["sigma"]
            rows.append({**base, "model": model, "error": "",
                         **bend_measures(prep, h, sd, params, SUMMER_SCALE, sigma=sigma)})
    return rows


def fold_shapes(winter: Sequence[dict], fold_rows: Sequence[Sequence[int]], min_trees: int = MIN_TREES) -> dict:
    """The typical shape from every tree, and for each fold from the other folds' trees only.

    Returns:
        ``{"all": shape, "folds": [shape per fold]}`` (:func:`typical_shape`).
    """
    ok = [w for w in winter if w["profiles"] is not None]
    return {"all": typical_shape([w["profiles"] for w in ok], min_trees),
            "folds": [typical_shape([w["profiles"] for w in ok if w["row"] not in rows], min_trees)
                      for rows in fold_rows]}


def fold_of(row: int, fold_rows: Sequence[Sequence[int]]) -> int:
    """The fold whose rows include ``row`` (-1 if none)."""
    return next((i for i, rows in enumerate(fold_rows) if row in rows), -1)


def typical_against(rows: Sequence[dict], source: str, measures: Sequence[str]) -> dict:
    """Each summer model against winter's bends, tree by tree, on measures pooled together (one row per tree
    and measure): the median ``|summer − winter|`` per model, and the bends with the typical shape against the
    bends with the plane as the prior's mean.

    Returns:
        ``{"n", "median": {model: ...}, "closer", "wilcoxon_p"}``: ``closer`` is the share of pairs where
        ``bent typical`` is nearer winter than ``bent``.
    """
    from scipy import stats

    win = {(r["tree"], m): r[m] for r in rows if r["source"] == "winter" and not r.get("error") for m in measures}
    got = {}
    for r in rows:
        if r["source"] == source and not r.get("error"):
            for m in measures:
                got[(r["tree"], m, r["model"])] = r.get(m, float("nan"))
    keys = [k for k in win if np.isfinite(win[k]) and all(np.isfinite(got.get((*k, md), np.nan)) for md in SUMMER_MODELS)]
    if len(keys) < 3:
        return {"n": len(keys)}
    err = {md: np.array([abs(got[(*k, md)] - win[k]) for k in keys]) for md in SUMMER_MODELS}
    d = err["bent"] - err["bent typical"]
    return {"n": len(keys), "median": {md: float(np.median(e)) for md, e in err.items()},
            "closer": float(np.mean(d > 0)), "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


TYPICAL_RULE = {"source": "U-Net", "tests": {"trunk": ("trunk_out",), "leader tilts": ("left_tilt", "right_tilt")}}
"""The rule fixed before the 449-tree run: ``model3d`` uses the typical shape as the prior's mean if, with the
U-Net's masks and the shape learnt out of fold, the bends with it are nearer winter's than the bends with the
plane as the mean on more than half of the trees with Wilcoxon p < 0.05, on the trunk's angle out of the V's
plane **and** on the leaders' tilts on their stretch (both leaders pooled)."""


def summarise_typical(rows: Sequence[dict], shapes: dict) -> dict:
    """The typical-shape run's report: the shape, each model against winter, and the rule's verdict."""
    report = {"shape": shapes["all"], "n_trees": len({r["tree"] for r in rows if r["source"] == "winter"}),
              "summer": {}}
    for source in ("U-Net", "hand masks"):
        report["summer"][source] = {
            "measures": {m: typical_against(rows, source, (m,)) for m in TYPICAL_MEASURES},
            **{name: typical_against(rows, source, ms) for name, ms in TYPICAL_RULE["tests"].items()}}
    verdicts = {}
    for name in TYPICAL_RULE["tests"]:
        t = report["summer"][TYPICAL_RULE["source"]][name]
        verdicts[name] = bool(t.get("n", 0) >= 3 and t["closer"] > 0.5 and t["wilcoxon_p"] < 0.05)
    report["rule"] = {"tests": verdicts, "adopt": all(verdicts.values())}
    report["failed"] = {s: sorted({r["tree"] for r in rows if r["source"] == s and r.get("error")})
                        for s in ("winter", "hand masks", "U-Net")}
    return report


def typical_lines(report: dict) -> list[str]:
    """The typical-shape run's report as text."""
    out = [f"The typical winter shape ({report['n_trees']} development trees; median bend behind the tree's plane, cm):"]
    for part in PARTS:
        sh = report["shape"][part]
        st, b = np.array(sh["stations"]), np.array(sh["bend"]) * 100
        if not len(st):
            continue
        picks = [i for i in range(0, len(st), 10)] + ([len(st) - 1] if (len(st) - 1) % 10 else [])
        out.append(f"  {part:5s} (to {st[-1]:.2f} m): " + ", ".join(f"{st[i]:.1f} m {b[i]:+.1f}" for i in picks))
    out += ["", "Summer against winter's bends, tree by tree: median |summer - winter| for each model (degrees; bows "
                "cm); the bends with the typical shape against the bends with the plane as the prior's mean "
                "(share nearer winter, Wilcoxon p). The shape is learnt without the tree's own fold."]
    for source, s in report["summer"].items():
        out.append(f"  {source}:")
        out.append(f"    {'measure':18s} {'n':>4s} " + " ".join(f"{m:>13s}" for m in SUMMER_MODELS) + "   nearer      p")
        for name, a in list(s["measures"].items()) + [(k, s[k]) for k in TYPICAL_RULE["tests"]]:
            if a.get("n", 0) >= 3:
                out.append(f"    {name:18s} {a['n']:4d} " + " ".join(f"{a['median'][m]:13.2f}" for m in SUMMER_MODELS)
                           + f"   {a['closer']:5.0%}  {a['wilcoxon_p']:.1e}")
    r = report["rule"]
    out += ["", "The rule fixed before the run (U-Net masks; nearer on more than half of the trees, p < 0.05, on the "
                "trunk and on the leaders' tilts): " + ", ".join(f"{k} {'passes' if v else 'fails'}"
                                                                for k, v in r["tests"].items())
            + f" -> {'ADOPT: model3d uses the typical shape' if r['adopt'] else 'not adopted: model3d keeps the plane as the mean'}",
            "Failed planes: " + "; ".join(f"{s} {len(t)} ({', '.join(t) or 'none'})" for s, t in report["failed"].items())]
    return out


def plot_typical(report: dict, path, failed: list[str]) -> None:
    """Two panels: the typical winter shape (median and quartiles), and each summer model's error against
    winter on the angles (U-Net masks)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.shape_model import AXIS, INK, MUTED, SECONDARY, SURFACE, _save, _style

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    for part in PARTS:
        sh = report["shape"][part]
        st = np.array(sh["stations"]) * (-100 if part == "trunk" else 100)
        if not len(st):
            continue
        ax.fill_between(st, np.array(sh["q1"]) * 100, np.array(sh["q3"]) * 100, color=PART_COLOURS[part], alpha=0.12,
                        linewidth=0)
        ax.plot(st, np.array(sh["bend"]) * 100, color=PART_COLOURS[part], lw=2, label=part)
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.set_xlabel("cm along the part from the fork (trunk to the left)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("cm behind the tree's plane (median, quartiles)", fontsize=8, color=SECONDARY)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=SECONDARY)
    ax.set_title(f"The typical winter shape ({report['n_trees']} trees)", fontsize=9, color=INK, loc="left")

    ax = axes[1]
    names = [m for m in TYPICAL_MEASURES if not m.endswith("bow")]
    s = report["summer"]["U-Net"]["measures"]
    styles = {"plane": (MUTED, "o", "plane"), "bent": ("#e34948", "h", "bends, plane as mean (until now)"),
              "typical": ("#898781", "D", "typical shape alone"), "bent typical": (INK, "o", "bends, typical shape as mean")}
    for j, model in enumerate(SUMMER_MODELS):
        colour, marker, label = styles[model]
        xs = [s[m]["median"][model] if s[m].get("n", 0) >= 3 else np.nan for m in names]
        ax.scatter(xs, np.arange(len(names)) + (j - 1.5) * 0.13, s=34, color=colour, marker=marker, label=label,
                   zorder=3, edgecolors=SURFACE, linewidths=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("median |summer − winter| (°), U-Net masks", fontsize=8, color=SECONDARY)
    ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=SECONDARY)
    ax.set_title("Summer against winter, tree by tree", fontsize=9, color=INK, loc="left")
    for ax in axes:
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
    fig.suptitle("The typical winter shape as the prior's mean (development trees; shape learnt out of fold)",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, failed)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

SUMMER_STYLE = {"hand masks": ("#2a78d6", "o"), "U-Net": ("#e34948", "h")}
"""The summer sources' colours and markers, as in the project's other figures."""

PART_COLOURS = {"trunk": "#0b0b0b", "left": "#2a78d6", "right": "#e34948"}
"""Each part's colour in the profile figure."""


def plot_figure(cv_rows: Sequence[dict], rows: Sequence[dict], fold_rows: Sequence[Sequence[int]], path,
                failed: list[str]) -> None:
    """Three panels: each tree's left-out error with the plane and with the bends (out of fold); winter's
    trunk angle out of the V's plane; and summer's against winter's, tree by tree."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    from treeskel.shape_model import AXIS, INK, SECONDARY, SURFACE, _save, _style

    oof = out_of_fold(cv_rows, fold_rows)["trees"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    fig.patch.set_facecolor(SURFACE)

    # A: the left-out blocks' error, tree by tree.
    ax = axes[0]
    if oof:
        x, y = np.array(list(oof.values())).T
        hi = float(np.percentile(np.r_[x, y], 99)) * 1.05
        ax.plot([0, hi], [0, hi], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(x, y, s=10, color=SECONDARY, alpha=0.55, linewidths=0, zorder=2)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.text(hi * 0.97, hi * 0.04, f"bends better on {np.mean(y < x):.0%}\nof {len(x)} trees", ha="right",
                va="bottom", fontsize=7.5, color=SECONDARY)
    ax.set_xlabel("plane: left-out error (cm)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("bends: left-out error (cm)", fontsize=8, color=SECONDARY)
    ax.set_title("Winter depth holds bends", fontsize=9, color=INK, loc="left")

    # B: winter's trunk out of the V's plane.
    ax = axes[1]
    t = np.array([r["trunk_out"] for r in rows if r["source"] == "winter" and r["setting"] == "main"
                  and r["model"] == "bent" and not r.get("error") and np.isfinite(r.get("trunk_out", np.nan))])
    if len(t):
        ax.hist(np.clip(t, -40, 40), bins=np.arange(-40, 41, 2.5), color=SECONDARY, alpha=0.8)
        ax.axvline(0, color=AXIS, lw=1.2)
        ax.axvline(float(np.median(t)), color=INK, lw=1, ls="--")
        ax.text(0.97, 0.95, f"median {np.median(t):+.1f}°", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=SECONDARY)
    ax.set_xlabel("trunk out of the V's plane (°, + = root behind)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("trees", fontsize=8, color=SECONDARY)
    ax.set_title("Winter: the trunk leaves the V's plane", fontsize=9, color=INK, loc="left")

    # C: summer's trunk angle against winter's.
    ax = axes[2]
    lim = (-30.0, 40.0)
    ax.plot(lim, lim, color=AXIS, lw=1.2, zorder=1)
    win = {r["tree"]: r["trunk_out"] for r in rows if r["source"] == "winter" and r["setting"] == "main"
           and r["model"] == "bent" and not r.get("error")}
    for source, (colour, marker) in SUMMER_STYLE.items():
        s = {r["tree"]: r["trunk_out"] for r in rows if r["source"] == source and r["model"] == "bent"
             and not r.get("error")}
        k = [t for t in s if t in win and np.isfinite(s[t]) and np.isfinite(win[t])]
        if len(k) < 3:
            continue
        x, y = np.array([win[t] for t in k]), np.array([s[t] for t in k])
        inside = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[0]) & (y <= lim[1])
        ax.scatter(x[inside], y[inside], s=12, color=colour, marker=marker, alpha=0.5, linewidths=0, zorder=2,
                   label=f"{source} (rank correlation {stats.spearmanr(x, y)[0]:.2f})")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("winter (°)", fontsize=8, color=SECONDARY)
    ax.set_ylabel("summer, ridge readings only (°)", fontsize=8, color=SECONDARY)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=SECONDARY)
    ax.set_title("Summer against winter, tree by tree", fontsize=9, color=INK, loc="left")

    for ax in axes:
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
    fig.suptitle("Bends out of the tree's plane (development trees; winter traces, summer masks)",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, failed)
    plt.close(fig)


def plot_profiles(trees: Sequence[tuple[str, Prepared, dict, np.ndarray]], path, failed: list[str],
                  title: str = "") -> None:
    """Each tree's readings behind the plane along its parts, the fitted bends and their ±2 SD band.

    Args:
        trees: ``(name, prepared, fit, sd)`` per panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.shape_model import AXIS, INK, SECONDARY, SURFACE, _save, _style

    cols = 3
    nrow = int(np.ceil(len(trees) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(11.5, 2.9 * nrow), squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, prep, fit, sd) in zip(axes.ravel(), trees):
        d = prep.z_axis - prep.z_plane
        for p in PARTS:
            c = prep.chains[p]
            s = prep.s[p] * 100 * (-1 if p == "trunk" else 1)
            used = prep.anchor[c] & (fit["weights"][c] > 0)
            ax.plot(s[used], d[c][used] * 100, ".", ms=1.6, color=PART_COLOURS[p], alpha=0.35)
            ax.plot(s[~used], d[c][~used] * 100, ".", ms=1.6, color="#c3c2b7", alpha=0.6)
            h, e = fit["h"][c] * 100, 2 * sd[c] * 100
            ax.fill_between(s, h - e, h + e, color=PART_COLOURS[p], alpha=0.15, linewidth=0)
            ax.plot(s, h, color=PART_COLOURS[p], lw=1.4, label=p)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylim(-25, 25)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
    for ax in axes.ravel()[len(trees):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left", labelcolor=SECONDARY)
    fig.supxlabel("cm along the branch from the fork (trunk to the left)", fontsize=8, color=SECONDARY)
    fig.supylabel("behind the tree's plane (cm)", fontsize=8, color=SECONDARY)
    fig.suptitle(title or "Depth along the branches, the fitted bends and ±2 SD (grey: readings that do not count)",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path, failed)
    plt.close(fig)


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def save_csv(path, rows: Sequence[dict]) -> None:
    """Write rows, floats to 6 significant digits."""
    import csv

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


TEXT_FIELDS = ("tree", "source", "setting", "model", "part", "error")


def load_csv(path) -> list[dict]:
    """Read :func:`save_csv`'s file back (text fields as text, ``row`` and ``n`` as integers)."""
    import csv

    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in TEXT_FIELDS else int(v) if k in ("row", "n") and v != "" else
                             float(v) if v not in ("", "nan") else float("nan")) for k, v in r.items()})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the development trees' bends; choose the prior; compare the seasons; print and save."""
    import argparse
    import json
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset
    from treeskel.segment import hand_mask, load_masks
    from treeskel.summer_gt import load_overlays

    ap = argparse.ArgumentParser(prog="python -m treeskel.bend3d", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json", help="the winter clicks")
    ap.add_argument("--mask-overlay", default="annotations/mask_overlay.json", help="the hand masks' line-up")
    ap.add_argument("--unet", default="docs/results/segment_lined_up_joined/predicted_masks.npz",
                    help="the named U-Net's development masks (5 folds)")
    ap.add_argument("--split", default="splits.json", help="its test rows are never loaded; its folds group rows")
    ap.add_argument("--out", default="docs/results/bend3d")
    ap.add_argument("--trees", nargs="+", help="only these trees (results go to outputs/bend3d_partial)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--from-saved", metavar="DIR", help="re-summarise a saved run's cv.csv and trees.csv (no images)")
    ap.add_argument("--figure", default="docs/figures/bend3d.png", help="the summary figure ('' for none)")
    ap.add_argument("--profiles", nargs="+", metavar="TREE", help="also draw these trees' winter profiles")
    ap.add_argument("--profiles-figure", default="docs/figures/bend3d_profiles.png")
    ap.add_argument("--typical", action="store_true",
                    help="the typical winter shape as the prior's mean: learn it, and compare summer with it "
                         "(-> docs/results/bend3d_typical/, docs/figures/bend3d_typical.png)")
    args = ap.parse_args(argv)
    if args.typical:
        return typical_main(args)

    split = json.loads(Path(args.split).read_text())
    fold_rows = split["fold_val_rows"]
    failed: list[str] = []
    if args.from_saved:                                  # only the summary again, from the saved rows
        out = Path(args.out if args.out != "docs/results/bend3d" else args.from_saved)
        cv_rows, rows = load_csv(Path(args.from_saved) / "cv.csv"), load_csv(Path(args.from_saved) / "trees.csv")
    else:
        if not args.data_dirs:
            ap.error("give the dataset folder(s), or --from-saved")
        test_rows = set(split["test_rows"])
        clicks = load_clicks(args.clicks)
        records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees if t.row not in test_rows}
        ids = sorted(t for t in records if t in clicks and not clicks[t].skip and clicks[t].complete
                     and (not args.trees or t in args.trees))
        if not ids:
            print("No development-row trees with winter clicks found; nothing to measure.")
            return 1
        # A run on chosen trees goes to outputs/, so it never overwrites the committed full run.
        out = Path(args.out if not args.trees or args.out != "docs/results/bend3d" else "outputs/bend3d_partial")
        workers = max(1, min(args.jobs, len(ids)))

        def run(fn, jobs, what):
            """Map ``fn`` over ``jobs`` in worker processes, printing progress."""
            t0, step, res = time.time(), max(1, min(25, len(jobs) // 10)), []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, r in enumerate(ex.map(fn, jobs, chunksize=1), 1):
                    res.append(r)
                    if i % step == 0 or i == len(jobs):
                        el = time.time() - t0
                        print(f"  {what}: {i}/{len(jobs)} trees, {el:.0f} s elapsed, about "
                              f"{el / i * (len(jobs) - i):.0f} s left", flush=True)
            return res

        # Phase 1: every tree's winter block cross-validation over the grid of priors.
        print(f"Phase 1: block cross-validation of the winter depth on {len(ids)} development trees "
              f"({len(GRID)} priors and the plane)...", flush=True)
        results = run(cv_tree, [(records[t], clicks[t]) for t in ids], "phase 1")
        cv_rows = cv_table(results)
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / "cv.csv", cv_rows)
        cv_rows = load_csv(out / "cv.csv")
        # The rule fixed before the run: the prior with the smallest pooled left-out error.
        params = params_of(choose(cv_rows))
        print(f"Chosen prior: {setting_key(params)}", flush=True)

        # Phase 2: the chosen prior on every tree: winter (every camera setting), the hand mask, the U-Net's mask.
        overlays = load_overlays(args.mask_overlay)
        unet = load_masks(Path(args.unet)) if Path(args.unet).exists() else {}
        radii = {r["tree"]: r["radii"] for r in results}
        jobs = [(records[t], clicks[t], radii[t], hand_mask(records[t], overlays), unet.get(t), params) for t in ids]
        print(f"Phase 2: the chosen prior on {len(ids)} trees, winter and summer...", flush=True)
        rows = [r for rs in run(measure_tree, jobs, "phase 2") for r in rs]
        save_csv(out / "trees.csv", rows)
        rows = load_csv(out / "trees.csv")              # the report from the saved rows, so it reproduces

        if args.profiles:
            panels = []
            for t in args.profiles:
                if t not in records or t not in clicks:
                    print(f"  {t}: not a development tree with clicks; no profile")
                    continue
                images, sk = winter_skeleton(records[t], clicks[t])
                try:
                    prep = prepare(sk["paths"], images["depth"], images["valid"], sk["radii"])
                except ValueError as e:
                    print(f"  {t}: {e}")
                    continue
                fit = fit_bends(prep, params)
                panels.append((f"{t} winter", prep, fit, posterior_sd(fit)))
            if panels:
                plot_profiles(panels, Path(args.profiles_figure), failed,
                              f"Winter depth along the branches and the fitted bends ({setting_key(params)})")

    report = summarise(cv_rows, rows, fold_rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = report_lines(report)
    print("\n".join(lines))
    try:
        (out / "output.txt").write_text("\n".join(lines) + "\n")
    except OSError as e:
        failed.append(f"{out / 'output.txt'}: {e.strerror or e}")
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        plot_figure(cv_rows, rows, fold_rows, Path(args.figure), failed)
    print(f"\nSaved cv.csv, trees.csv, report.json and output.txt in {out}"
          + (f", and {args.figure}" if args.figure else ""))
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


def run_parallel(fn, jobs: Sequence, what: str, workers: int) -> list:
    """Map ``fn`` over ``jobs`` in worker processes, printing progress, in the jobs' order."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    t0, step, res = time.time(), max(1, min(25, len(jobs) // 10)), []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=1), 1):
            res.append(r)
            if i % step == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {what}: {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                      flush=True)
    return res


TYPICAL_OUT = "docs/results/bend3d_typical"
"""Where ``--typical`` writes, and :mod:`treeskel.model3d` reads the typical shape (``typical_shape.json``)."""


def typical_main(args) -> int:
    """``--typical``: learn the typical winter shape and compare summer with it (development rows only).

    Pass 1 fits every development tree's winter bends with the chosen prior (``docs/results/bend3d/report.json``)
    and samples them along each part; the typical shape is learnt from all trees and, for each fold, from the
    other folds' trees. Pass 2 fits summer (the lined-up hand mask and the U-Net's mask) with the plane and with
    the out-of-fold typical shape as the prior's mean. The rule (:data:`TYPICAL_RULE`) decides whether
    :mod:`treeskel.model3d` uses the shape.
    """
    import json
    from pathlib import Path

    from treeskel.guided_gt import load_clicks
    from treeskel.io import index_dataset
    from treeskel.segment import hand_mask, load_masks
    from treeskel.summer_gt import load_overlays

    out = Path(TYPICAL_OUT if args.out == "docs/results/bend3d" else args.out)
    figure = "docs/figures/bend3d_typical.png" if args.figure == "docs/figures/bend3d.png" else args.figure
    split = json.loads(Path(args.split).read_text())
    fold_rows = split["fold_val_rows"]
    failed: list[str] = []
    if args.from_saved:
        rows = load_csv(Path(args.from_saved) / "trees.csv")
        shapes = json.loads((Path(args.from_saved) / "typical_shape.json").read_text())
        out = Path(args.out if args.out != "docs/results/bend3d" else args.from_saved)
    else:
        if not args.data_dirs:
            print("give the dataset folder(s), or --from-saved")
            return 2
        chosen = Path("docs/results/bend3d/report.json")
        params = params_of(json.loads(chosen.read_text())["chosen"]) if chosen.exists() else BendParams()
        clicks = load_clicks(args.clicks)
        records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees
                   if t.row not in set(split["test_rows"])}
        ids = sorted(t for t in records if t in clicks and not clicks[t].skip and clicks[t].complete
                     and (not args.trees or t in args.trees))
        if not ids:
            print("No development-row trees with winter clicks found; nothing to measure.")
            return 1
        if args.trees and args.out == "docs/results/bend3d":
            out = Path("outputs/bend3d_typical_partial")     # a run on chosen trees never overwrites the full run
        print(f"Pass 1: winter bends ({setting_key(params)}) of {len(ids)} development trees...", flush=True)
        winter = run_parallel(typical_winter, [(records[t], clicks[t], params) for t in ids], "pass 1", args.jobs)
        shapes = fold_shapes(winter, fold_rows)
        # Each tree's summer is fitted with the shape learnt without its own fold.
        overlays = load_overlays(args.mask_overlay)
        unet = load_masks(Path(args.unet)) if Path(args.unet).exists() else {}
        jobs = [(records[t], hand_mask(records[t], overlays), unet.get(t), params,
                 shapes["folds"][fold_of(records[t].row, fold_rows)]) for t in ids]
        print(f"Pass 2: summer with the plane and with the typical shape as the prior's mean...", flush=True)
        rows = [{"tree": w["tree"], "row": w["row"], "source": "winter", "setting": "main", "model": "bent",
                 "error": w["error"], **(w["measures"] or {})} for w in winter]
        rows += [r for rs in run_parallel(typical_summer, jobs, "pass 2", args.jobs) for r in rs]
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / "trees.csv", rows)
        rows = load_csv(out / "trees.csv")
        shapes = {**shapes, "prior": setting_key(params), "min_trees": MIN_TREES,
                  "trees": len([w for w in winter if w["profiles"] is not None])}
        (out / "typical_shape.json").write_text(json.dumps(shapes, indent=1))
        np.savez_compressed(out / "profiles.npz", ids=np.array([w["tree"] for w in winter if w["profiles"] is not None]),
                            **{p: np.array([w["profiles"][p] for w in winter if w["profiles"] is not None])
                               for p in PARTS}, **{f"stations_{p}": STATIONS[p] for p in PARTS})
    report = summarise_typical(rows, shapes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = typical_lines(report)
    print("\n".join(lines))
    try:
        (out / "output.txt").write_text("\n".join(lines) + "\n")
    except OSError as e:
        failed.append(f"{out / 'output.txt'}: {e.strerror or e}")
    if figure:
        Path(figure).parent.mkdir(parents=True, exist_ok=True)
        plot_typical(report, Path(figure), failed)
    print(f"\nSaved trees.csv, typical_shape.json, profiles.npz, report.json and output.txt in {out}"
          + (f", and {figure}" if figure else ""))
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
