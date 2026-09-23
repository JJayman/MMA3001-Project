"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def sample_dir() -> Path:
    """Path to the committed sample dataset."""
    return SAMPLE_DIR


def write_png(path: Path, arr: np.ndarray) -> Path:
    """Write ``arr`` to ``path`` as a PNG and return the path."""
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def make_tree(tmp_path):
    """Factory writing a synthetic tree's files into ``tmp_path``.

    ``make_tree("R03U04", skip=[("winter", "D")])`` writes all four images
    except those listed in ``skip``.
    """
    def _make(tree_id: str, skip=()):
        for season in ("Summer", "Winter"):
            for mod in ("RGB", "D"):
                if (season.lower(), mod) in skip:
                    continue
                if mod == "RGB":
                    arr = np.full((480, 640, 3), 100, np.uint8)
                else:
                    g = np.zeros((480, 640), np.uint8)
                    g[240:, :] = 120
                    arr = np.stack([g] * 3, axis=-1)
                write_png(tmp_path / f"{tree_id}_{season}_{mod}-1-2-3.png", arr)
        return tmp_path
    return _make
