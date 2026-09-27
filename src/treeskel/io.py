"""Dataset indexing and image loading for the orchard RGB-D dataset.

The dataset used ("Dataset (With Summer GT)") has five PNG files per tree::

    R20N03_Summer_RGB-14-21-52.png   640x480 colour
    R20N03_Summer_D-14-21-52.png     640x480 depth
    R20N03_Summer_GT-14-21-52.png    256x256 main-branch mask (summer only)
    R20N03_Winter_RGB-13-26-42.png   640x480 colour
    R20N03_Winter_D-13-26-42.png     640x480 depth

The name encodes the orchard row (``R20``), the tree number within the row
(``N03``), the season, the file type (``RGB``, ``D`` for depth or ``GT`` for
the mask) and the capture clock time ``H-M-S`` (not zero-padded, e.g.
``14-22-1``). R and N follow the orchard map supplied with the dataset: rows
run west to east and N counts trees north to south, so ``R20N03_Summer`` and
``R20N03_Winter`` are the same physical tree.

An earlier version of the dataset numbered trees by photo order (``U``, for
"unmatched", e.g. ``R01U01``). Those names are still accepted, but U numbers
are not guaranteed to be the same tree in both seasons, so that version is
not used.

Masks are hand-labelled, 256x256, and assumed to use the same geometry as
Chen et al. (2020): the photo's centre 480x480 (columns 80-559) resized to
256x256. :func:`to_mask_frame` and :func:`mask_to_image_frame` convert between
the two frames.

**Masks do not line up exactly with the summer photos.** Overlays on the
sample show the right shape, but shifted left of the visible trunk by about
15-45 px, with the shift varying from tree to tree (see
``docs/figures/mask_alignment_R12N02.jpg``). Stretching the mask to the full
width fits worse. The masks may have been drawn on a slightly different view
(e.g. the paper's autumn images), so pixel-level comparisons need a per-tree
alignment first.

The row 4 winter files in the U version use a variant that also embeds an original capture id,
with an underscore before the time::

    R04U01_Winter_RGB-R04N03_RGB_12-17-14.png

The embedded id (``R04N03``) is kept as :attr:`FileInfo.source_id`. Its row
must match the outer row, or the file is rejected. Its modality is *not* trusted:
every row 4 winter depth file embeds ``_RGB_`` (e.g.
``R04U01_Winter_D-R04N03_RGB_12-17-14.png``), so the outer ``D``/``RGB`` label
decides the modality, and the embedded one is kept as
:attr:`FileInfo.source_modality` for reporting. :func:`load_depth` rejects any
"depth" file that is really a colour image, which catches a wrong outer label.

Depth images are stored as 8-bit greyscale (three identical channels). They are
a scaled visualisation of depth, not metric distance: brighter pixels are closer
to the camera and 0 means "no reading" (out of range or sensor dropout).

**The depth does not line up exactly with the colour image.** Session 8 found
the winter depth about 4 px below the colour image and about 0.6 px to its
left (``docs/results/depth_registration/``). :func:`load_tree` moves the depth
by :data:`DEPTH_SHIFT` to correct this; the shift is measured by
:mod:`treeskel.measure_registration`.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SEASONS = ("summer", "winter")
"""Seasons present in the dataset, in lowercase canonical form."""

IMAGE_SHAPE = (480, 640)
"""Expected (height, width) of every RGB and depth image in pixels."""

MASK_SHAPE = (256, 256)
"""Expected (height, width) of every mask in pixels."""

CROP_COLS = (80, 560)
"""Columns of the 640-wide photo covered by a mask (the centre 480x480)."""

DEPTH_SHIFT: dict[str, tuple[int, int]] = {"summer": (0, 0), "winter": (0, 0)}
"""Whole pixels ``(rows, cols)`` each season's depth image is moved by on loading
(:func:`load_tree`), so that it lines up with the colour image; negative rows
move it up.

Both are zero until the shift has been measured on all development trees
(:mod:`treeskel.measure_registration`); session 8 estimated the winter depth
to be about 4 rows too low (``docs/results/depth_registration/``).
"""

EXCLUDED_ROWS: dict[int, str] = {
    4: "Every winter depth image in row 4 is a copy of R03N34's winter depth "
       "image, so the row has no usable winter depth.",
}
"""Rows left out of the dataset by default, with the reason for each."""

EXCLUDED_TREES: dict[str, str] = {}
"""Individual trees left out of the dataset by default, with the reason.

Empty at present. R03N34's winter depth image is identical to row 4's, but it
was checked against R03N34's own RGB image and is genuine: row 4 holds the
copies (see :data:`EXCLUDED_ROWS`).
"""

_FILENAME_RE = re.compile(
    r"^R(?P<row>\d+)(?P<kind>[UN])(?P<pos>\d+)_(?P<season>summer|winter)_(?P<modality>RGB|D|GT)-"
    # Optional embedded original id, e.g. "R04N03_RGB_" (row 4 winter files).
    r"(?:R(?P<src_row>\d+)N(?P<src_num>\d+)_(?P<src_mod>RGB|D)_)?"
    r"(?P<h>\d{1,2})-(?P<m>\d{1,2})-(?P<s>\d{1,2})\.png$",
    re.IGNORECASE,
)


_MODALITIES = {"RGB": "rgb", "D": "depth", "GT": "mask"}


@dataclass(frozen=True)
class FileInfo:
    """Metadata parsed from a single dataset filename.

    Attributes:
        tree_id: Tree code, e.g. ``"R20N03"``.
        row: Orchard row number (the ``20`` in ``R20``).
        position: Tree number within its row (the ``03`` in ``N03``).
        season: ``"summer"`` or ``"winter"``.
        modality: ``"rgb"``, ``"depth"`` or ``"mask"``.
        time: Capture clock time as ``"HH:MM:SS"``.
        source_id: Embedded original capture id such as ``"R04N03"``, or
            ``None`` for the standard naming.
        source_modality: Modality written in the embedded id (``"rgb"`` or
            ``"depth"``), or ``None``. May disagree with ``modality``.
    """

    tree_id: str
    row: int
    position: int
    season: str
    modality: str
    time: str
    source_id: str | None = None
    source_modality: str | None = None


def parse_filename(name: str | Path) -> FileInfo:
    """Parse a dataset filename into its components.

    Args:
        name: Filename or path, e.g. ``"R20N03_Summer_D-14-21-52.png"``.

    Returns:
        The parsed :class:`FileInfo`.

    Raises:
        ValueError: If the name does not follow the dataset naming convention,
            has an impossible time, or embeds an id whose row disagrees with
            the outer name.

    Example:
        >>> parse_filename("R20N05_Summer_GT-14-22-1.png").modality
        'mask'
        >>> parse_filename("R02U10_Summer_RGB-12-52-3.png").time
        '12:52:03'
        >>> parse_filename("R04U01_Winter_RGB-R04N03_RGB_12-17-14.png").source_id
        'R04N03'
    """
    base = Path(name).name
    m = _FILENAME_RE.match(base)
    if m is None:
        raise ValueError(f"Filename does not match dataset convention: {base!r}")
    row, pos = int(m["row"]), int(m["pos"])
    h, mi, s = int(m["h"]), int(m["m"]), int(m["s"])
    if not (h < 24 and mi < 60 and s < 60):
        raise ValueError(f"Invalid capture time in filename: {base!r}")
    source_id = source_modality = None
    if m["src_row"] is not None:
        # A row mismatch would mean the file was renamed to the wrong tree.
        if int(m["src_row"]) != row:
            raise ValueError(f"Embedded row disagrees with filename: {base!r}")
        source_id = f"R{int(m['src_row']):02d}N{int(m['src_num']):02d}"
        source_modality = "rgb" if m["src_mod"].upper() == "RGB" else "depth"
    return FileInfo(
        # Rebuild the id so "r1n1" and "R01N01" style variants map to one tree.
        tree_id=f"R{row:02d}{m['kind'].upper()}{pos:02d}",
        row=row,
        position=pos,
        season=m["season"].lower(),
        modality=_MODALITIES[m["modality"].upper()],
        time=f"{h:02d}:{mi:02d}:{s:02d}",
        source_id=source_id,
        source_modality=source_modality,
    )


@dataclass
class TreeRecord:
    """All files belonging to one tree.

    Attributes:
        tree_id: Tree code, e.g. ``"R20N03"``.
        row: Orchard row number, used as the grouping key for splits.
        position: Tree number within its row.
        paths: Mapping ``(season, modality) -> Path`` of the files found.
    """

    tree_id: str
    row: int
    position: int
    paths: dict[tuple[str, str], Path] = field(default_factory=dict)

    def missing(self) -> list[tuple[str, str]]:
        """Return the ``(season, modality)`` pairs that have no file."""
        return [
            (s, mod)
            for s in SEASONS
            for mod in ("rgb", "depth")
            if (s, mod) not in self.paths
        ]

    @property
    def is_complete(self) -> bool:
        """True if all four images (2 seasons x RGB/depth) are present.

        The mask is not required: it is used for validation only, so a tree
        without one can still be used for training.
        """
        return not self.missing()

    @property
    def has_mask(self) -> bool:
        """True if the tree has a summer mask."""
        return ("summer", "mask") in self.paths


@dataclass
class DatasetIndex:
    """Result of scanning a data directory.

    Attributes:
        trees: Complete trees (all four images present), sorted by id.
        incomplete: Trees missing at least one image; excluded from ``trees``.
        unrecognised: Files whose names did not match the convention.
        duplicates: Files that repeat an already-seen ``(tree, season, modality)``.
        excluded: Trees deliberately left out (see :data:`EXCLUDED_ROWS` and
            :data:`EXCLUDED_TREES`), whether complete or not; not in
            ``trees`` or ``incomplete``.
    """

    trees: list[TreeRecord]
    incomplete: list[TreeRecord]
    unrecognised: list[Path]
    duplicates: list[Path]
    excluded: list[TreeRecord] = field(default_factory=list)


def index_dataset(
    root: str | Path,
    exclude_rows: Iterable[int] | None = None,
    exclude_trees: Iterable[str] | None = None,
) -> DatasetIndex:
    """Scan a directory (recursively) and group image files by tree.

    Invalid inputs are reported rather than raised, so a single bad file does
    not stop a run over the full dataset: badly named files go to
    ``unrecognised``, repeated files to ``duplicates`` (the first one found in
    sorted order is kept), and trees without all four images to ``incomplete``.
    Trees in excluded rows, and excluded individual trees, go to ``excluded``. Masks are optional (see
    :attr:`TreeRecord.has_mask`).

    Args:
        root: Directory containing the PNG files (sub-folders are searched).
        exclude_rows: Rows to leave out. ``None`` (default) uses
            :data:`EXCLUDED_ROWS`; pass ``()`` to keep every row.
        exclude_trees: Tree ids to leave out. ``None`` (default) uses
            :data:`EXCLUDED_TREES`; pass ``()`` to keep every tree.

    Returns:
        A :class:`DatasetIndex`.

    Raises:
        FileNotFoundError: If ``root`` does not exist or is not a directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    records: dict[str, TreeRecord] = {}
    unrecognised: list[Path] = []
    duplicates: list[Path] = []

    for path in sorted(root.rglob("*.png")):
        try:
            info = parse_filename(path)
        except ValueError:
            unrecognised.append(path)
            continue
        rec = records.setdefault(
            info.tree_id, TreeRecord(info.tree_id, info.row, info.position)
        )
        key = (info.season, info.modality)
        if key in rec.paths:
            duplicates.append(path)
        else:
            rec.paths[key] = path

    skip_rows = set(EXCLUDED_ROWS if exclude_rows is None else exclude_rows)
    skip_trees = set(EXCLUDED_TREES if exclude_trees is None else exclude_trees)

    def skipped(r: TreeRecord) -> bool:
        return r.row in skip_rows or r.tree_id in skip_trees

    ordered = [records[k] for k in sorted(records)]
    kept = [r for r in ordered if not skipped(r)]
    return DatasetIndex(
        trees=[r for r in kept if r.is_complete],
        incomplete=[r for r in kept if not r.is_complete],
        unrecognised=unrecognised,
        duplicates=duplicates,
        excluded=[r for r in ordered if skipped(r)],
    )


def find_identical_images(paths: Iterable[str | Path]) -> list[list[Path]]:
    """Find groups of files whose pixel content is exactly identical.

    Different trees should never share an image, so any group returned points
    to a copying or export error (this is how row 4's repeated winter depth
    images were found). Pixels are compared, not file bytes, so the same image
    saved with different PNG settings still counts as identical.

    Args:
        paths: Image files to compare.

    Returns:
        Groups of two or more identical files, each sorted, ordered by first
        file name.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        p = Path(p)
        arr = np.asarray(Image.open(p))
        h = hashlib.sha1(str(arr.shape).encode() + arr.tobytes()).hexdigest()
        groups[h].append(p)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1),
                  key=lambda g: g[0].name)


def _check_shape(arr: np.ndarray, path: Path) -> None:
    if arr.shape[:2] != IMAGE_SHAPE:
        raise ValueError(
            f"{path.name}: expected {IMAGE_SHAPE} image, got {arr.shape[:2]}"
        )


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image.

    Args:
        path: Path to an RGB PNG.

    Returns:
        ``uint8`` array of shape ``(480, 640, 3)``.

    Raises:
        ValueError: If the image is not 640x480.
    """
    path = Path(path)
    arr = np.asarray(Image.open(path).convert("RGB"))
    _check_shape(arr, path)
    return arr


def load_depth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an 8-bit depth image and its validity mask.

    The dataset stores depth as greyscale in all three channels. Values are
    relative (brighter = closer), not metric; converting to metres needs a
    calibrated mapping, which is estimated elsewhere.

    Args:
        path: Path to a depth PNG.

    Returns:
        ``(depth, valid)`` where ``depth`` is a ``uint8`` array of shape
        ``(480, 640)`` and ``valid`` is a boolean array that is False where the
        sensor gave no reading (value 0).

    Raises:
        ValueError: If the image is not 640x480, or if its colour channels
            differ (i.e. it is not a greyscale depth map).
    """
    path = Path(path)
    arr = np.asarray(Image.open(path))
    _check_shape(arr, path)
    if arr.ndim == 3:
        chans = arr[..., :3]
        if not (np.array_equal(chans[..., 0], chans[..., 1])
                and np.array_equal(chans[..., 0], chans[..., 2])):
            raise ValueError(f"{path.name}: depth channels differ; not a depth map")
        arr = chans[..., 0]
    depth = arr.astype(np.uint8, copy=False)
    return depth, depth > 0


def shift_image(image: np.ndarray, shift: tuple[int, int], fill=0) -> np.ndarray:
    """Move an image by whole pixels, filling the uncovered edge.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` array.
        shift: ``(rows, cols)``; positive moves the content down and right.
        fill: Value for the pixels the move uncovers (for depth, 0 = no reading).

    Returns:
        A new array the shape and dtype of ``image``: ``out[r + rows, c + cols] =
        image[r, c]`` wherever both are inside the image.

    Example:
        >>> shift_image(np.arange(1, 7).reshape(2, 3), (1, -1)).tolist()
        [[0, 0, 0], [2, 3, 0]]
    """
    dy, dx = (int(v) for v in shift)
    h, w = image.shape[:2]
    out = np.full_like(image, fill)
    if abs(dy) < h and abs(dx) < w:
        out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
            image[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def load_tree(rec: TreeRecord, season: str, register: bool = True) -> dict[str, np.ndarray]:
    """Load the RGB image, depth and validity mask for one tree and season.

    Args:
        rec: A complete :class:`TreeRecord`.
        season: ``"summer"`` or ``"winter"`` (case-insensitive).
        register: Move the depth by the season's :data:`DEPTH_SHIFT` so that it
            lines up with the colour image (pixels it uncovers get no reading).
            False returns the depth as stored, e.g. to measure the shift.

    Returns:
        Dict with keys ``"rgb"`` (480x640x3 uint8), ``"depth"`` (480x640
        uint8) and ``"valid"`` (480x640 bool).

    Raises:
        ValueError: If ``season`` is unknown.
        KeyError: If the record is missing that season's files.
    """
    season = season.lower()
    if season not in SEASONS:
        raise ValueError(f"Unknown season {season!r}; expected one of {SEASONS}")
    depth, valid = load_depth(rec.paths[(season, "depth")])
    if register and any(DEPTH_SHIFT[season]):
        depth = shift_image(depth, DEPTH_SHIFT[season])
        valid = depth > 0
    return {"rgb": load_rgb(rec.paths[(season, "rgb")]), "depth": depth, "valid": valid}


def load_mask(path: str | Path) -> np.ndarray:
    """Load a 256x256 main-branch mask as a boolean array.

    Masks are white (branch) on black. Pixels brighter than 127 count as
    branch, so light anti-aliasing at the edges does not matter.

    Args:
        path: Path to a ``_GT`` mask PNG.

    Returns:
        ``bool`` array of shape ``(256, 256)``, True on branch pixels.

    Raises:
        ValueError: If the mask is not 256x256 or is not greyscale.
    """
    path = Path(path)
    arr = np.asarray(Image.open(path))
    if arr.shape[:2] != MASK_SHAPE:
        raise ValueError(f"{path.name}: expected {MASK_SHAPE} mask, got {arr.shape[:2]}")
    if arr.ndim == 3:
        chans = arr[..., :3]
        if not (np.array_equal(chans[..., 0], chans[..., 1])
                and np.array_equal(chans[..., 0], chans[..., 2])):
            raise ValueError(f"{path.name}: mask channels differ; not a mask")
        arr = chans[..., 0]
    return arr > 127


def to_mask_frame(image: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Crop and resize a 640x480 image to the 256x256 frame of the masks.

    Takes the centre 480x480 (columns 80-559) and resizes it to 256x256, the
    geometry assumed for the masks.

    Args:
        image: ``(480, 640)`` or ``(480, 640, C)`` array (``uint8`` or ``bool``).
        nearest: Use nearest-neighbour resampling. Use this for depth (so
            invalid zeros are not blended with valid values) and for masks.
            Otherwise bilinear, which suits RGB images.

    Returns:
        Array of shape ``(256, 256)`` or ``(256, 256, C)``, same dtype.

    Raises:
        ValueError: If ``image`` is not 480x640.
    """
    if image.shape[:2] != IMAGE_SHAPE:
        raise ValueError(f"expected {IMAGE_SHAPE} image, got {image.shape[:2]}")
    crop = np.ascontiguousarray(image[:, CROP_COLS[0]:CROP_COLS[1]])
    is_bool = crop.dtype == bool
    pil = Image.fromarray(crop.astype(np.uint8) * 255 if is_bool else crop)
    out = np.asarray(pil.resize(MASK_SHAPE[::-1],
                                Image.NEAREST if nearest or is_bool else Image.BILINEAR))
    return out > 127 if is_bool else out


def mask_to_image_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 256x256 mask back onto the full 640x480 photo.

    Args:
        mask: ``bool`` array of shape ``(256, 256)``.

    Returns:
        ``(full, covered)``: ``full`` is a ``(480, 640)`` bool mask; ``covered``
        is True on the columns the mask describes (80-559). Outside
        ``covered``, the mask says nothing, so those pixels must be ignored
        in evaluation rather than treated as "not branch".

    Raises:
        ValueError: If ``mask`` is not 256x256.
    """
    if mask.shape != MASK_SHAPE:
        raise ValueError(f"expected {MASK_SHAPE} mask, got {mask.shape}")
    side = IMAGE_SHAPE[0]
    big = Image.fromarray(mask.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)
    full = np.zeros(IMAGE_SHAPE, bool)
    full[:, CROP_COLS[0]:CROP_COLS[1]] = np.asarray(big) > 127
    covered = np.zeros(IMAGE_SHAPE, bool)
    covered[:, CROP_COLS[0]:CROP_COLS[1]] = True
    return full, covered
