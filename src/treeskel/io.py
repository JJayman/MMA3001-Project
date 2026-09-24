"""Dataset indexing and image loading for the orchard RGB-D dataset.

Each tree has four 640x480 PNG files, one RGB and one depth image per season::

    R01U01_Summer_RGB-12-40-11.png
    R01U01_Summer_D-12-40-11.png
    R01U01_Winter_RGB-11-23-20.png
    R01U01_Winter_D-11-23-20.png

The name encodes the orchard row (``R01``), the tree's position in that row
(``U01``), the season, the modality (``RGB`` or ``D`` for depth) and the capture
clock time ``H-M-S`` (not zero-padded, e.g. ``12-52-3``).

The row 4 winter files use a variant that also embeds an original capture id,
with an underscore before the time::

    R04U01_Winter_RGB-R04N03_RGB_12-17-14.png

The embedded id (``R04N03``) is kept as :attr:`FileInfo.source_id`. Its row and
modality must match the outer ones, or the file is rejected.

Depth images are stored as 8-bit greyscale (three identical channels). They are
a scaled visualisation of depth, not metric distance: brighter pixels are closer
to the camera and 0 means "no reading" (out of range or sensor dropout).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

SEASONS = ("summer", "winter")
"""Seasons present in the dataset, in lowercase canonical form."""

IMAGE_SHAPE = (480, 640)
"""Expected (height, width) of every image in pixels."""

_FILENAME_RE = re.compile(
    r"^R(?P<row>\d+)U(?P<pos>\d+)_(?P<season>summer|winter)_(?P<modality>RGB|D)-"
    # Optional embedded original id, e.g. "R04N03_RGB_" (row 4 winter files).
    r"(?:R(?P<src_row>\d+)N(?P<src_num>\d+)_(?P<src_mod>RGB|D)_)?"
    r"(?P<h>\d{1,2})-(?P<m>\d{1,2})-(?P<s>\d{1,2})\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FileInfo:
    """Metadata parsed from a single dataset filename.

    Attributes:
        tree_id: Tree code, e.g. ``"R01U01"``.
        row: Orchard row number (the ``01`` in ``R01``).
        position: Tree position within its row (the ``01`` in ``U01``).
        season: ``"summer"`` or ``"winter"``.
        modality: ``"rgb"`` or ``"depth"``.
        time: Capture clock time as ``"HH:MM:SS"``.
        source_id: Embedded original capture id such as ``"R04N03"``, or
            ``None`` for the standard naming.
    """

    tree_id: str
    row: int
    position: int
    season: str
    modality: str
    time: str
    source_id: str | None = None


def parse_filename(name: str | Path) -> FileInfo:
    """Parse a dataset filename into its components.

    Args:
        name: Filename or path, e.g. ``"R01U01_Summer_D-12-40-11.png"``.

    Returns:
        The parsed :class:`FileInfo`.

    Raises:
        ValueError: If the name does not follow the dataset naming convention,
            has an impossible time, or embeds an id whose row or modality
            disagrees with the outer name.

    Example:
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
    source_id = None
    if m["src_row"] is not None:
        # A mismatch would mean the file was renamed to the wrong tree or type.
        if int(m["src_row"]) != row or m["src_mod"].upper() != m["modality"].upper():
            raise ValueError(f"Embedded id disagrees with filename: {base!r}")
        source_id = f"R{int(m['src_row']):02d}N{int(m['src_num']):02d}"
    return FileInfo(
        # Rebuild the id so "r1u1" and "R01U01" style variants map to one tree.
        tree_id=f"R{row:02d}U{pos:02d}",
        row=row,
        position=pos,
        season=m["season"].lower(),
        modality="rgb" if m["modality"].upper() == "RGB" else "depth",
        time=f"{h:02d}:{mi:02d}:{s:02d}",
        source_id=source_id,
    )


@dataclass
class TreeRecord:
    """All files belonging to one tree.

    Attributes:
        tree_id: Tree code, e.g. ``"R01U01"``.
        row: Orchard row number, used as the grouping key for splits.
        position: Tree position within its row.
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
        """True if all four images (2 seasons x RGB/depth) are present."""
        return not self.missing()


@dataclass
class DatasetIndex:
    """Result of scanning a data directory.

    Attributes:
        trees: Complete trees (all four images present), sorted by id.
        incomplete: Trees missing at least one image; excluded from ``trees``.
        unrecognised: Files whose names did not match the convention.
        duplicates: Files that repeat an already-seen ``(tree, season, modality)``.
    """

    trees: list[TreeRecord]
    incomplete: list[TreeRecord]
    unrecognised: list[Path]
    duplicates: list[Path]


def index_dataset(root: str | Path) -> DatasetIndex:
    """Scan a directory (recursively) and group image files by tree.

    Invalid inputs are reported rather than raised, so a single bad file does
    not stop a run over the full 544-tree dataset: badly named files go to
    ``unrecognised``, repeated files to ``duplicates`` (the first one found in
    sorted order is kept), and trees without all four images to ``incomplete``.

    Args:
        root: Directory containing the PNG files (sub-folders are searched).

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

    ordered = [records[k] for k in sorted(records)]
    return DatasetIndex(
        trees=[r for r in ordered if r.is_complete],
        incomplete=[r for r in ordered if not r.is_complete],
        unrecognised=unrecognised,
        duplicates=duplicates,
    )


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


def load_tree(rec: TreeRecord, season: str) -> dict[str, np.ndarray]:
    """Load the RGB image, depth and validity mask for one tree and season.

    Args:
        rec: A complete :class:`TreeRecord`.
        season: ``"summer"`` or ``"winter"`` (case-insensitive).

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
    return {"rgb": load_rgb(rec.paths[(season, "rgb")]), "depth": depth, "valid": valid}
