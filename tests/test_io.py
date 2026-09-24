"""Tests for treeskel.io."""

import numpy as np
import pytest

from conftest import write_png
from treeskel.io import (
    index_dataset,
    load_depth,
    load_rgb,
    load_tree,
    parse_filename,
)


class TestParseFilename:
    def test_parses_all_fields(self):
        info = parse_filename("R01U01_Summer_D-12-40-11.png")
        assert info.tree_id == "R01U01"
        assert (info.row, info.position) == (1, 1)
        assert info.season == "summer"
        assert info.modality == "depth"
        assert info.time == "12:40:11"

    def test_unpadded_time_and_rgb(self):
        info = parse_filename("R02U10_Summer_RGB-12-52-3.png")
        assert info.modality == "rgb"
        assert info.time == "12:52:03"

    def test_accepts_full_path_and_case_variants(self, tmp_path):
        info = parse_filename(tmp_path / "r17u9_WINTER_rgb-13-15-19.PNG")
        assert info.tree_id == "R17U09"
        assert info.season == "winter"

    def test_row4_embedded_id_variant(self):
        info = parse_filename("R04U01_Winter_RGB-R04N03_RGB_12-17-14.png")
        assert info.tree_id == "R04U01"
        assert info.modality == "rgb" and info.season == "winter"
        assert info.source_id == "R04N03"
        assert info.time == "12:17:14"
        assert parse_filename("R04U01_Winter_D-R04N03_D_12-17-14.png").modality == "depth"

    def test_standard_name_has_no_source_id(self):
        assert parse_filename("R01U01_Summer_D-12-40-11.png").source_id is None

    @pytest.mark.parametrize("bad", [
        "R04U01_Winter_RGB-R04N03_D_12-17-14.png",   # embedded modality differs
        "R04U01_Winter_RGB-R05N03_RGB_12-17-14.png",  # embedded row differs
        "R04U01_Winter_RGB-R04N03_RGB-12-17-14.png",  # wrong separator
    ])
    def test_rejects_inconsistent_embedded_id(self, bad):
        with pytest.raises(ValueError):
            parse_filename(bad)

    @pytest.mark.parametrize("bad", [
        "R01U01_Autumn_D-12-40-11.png",   # unknown season
        "R01U01_Summer_IR-12-40-11.png",  # unknown modality
        "R01U01_Summer_D.png",            # no time
        "R01U01_Summer_D-25-40-11.png",   # impossible hour
        "R01U01_Summer_D-12-40-11.jpg",   # wrong extension
        "notes.png",
    ])
    def test_rejects_invalid_names(self, bad):
        with pytest.raises(ValueError):
            parse_filename(bad)


class TestIndexDataset:
    def test_sample_dataset_is_complete(self, sample_dir):
        idx = index_dataset(sample_dir)
        assert len(idx.trees) == 8
        assert not idx.incomplete and not idx.unrecognised and not idx.duplicates
        assert idx.trees[0].tree_id == "R01U01"

    def test_reports_incomplete_and_unrecognised(self, make_tree):
        root = make_tree("R03U04")
        make_tree("R03U05", skip=[("winter", "D")])
        write_png(root / "junk.png", np.zeros((4, 4), np.uint8))
        idx = index_dataset(root)
        assert [t.tree_id for t in idx.trees] == ["R03U04"]
        assert [t.tree_id for t in idx.incomplete] == ["R03U05"]
        assert idx.incomplete[0].missing() == [("winter", "depth")]
        assert [p.name for p in idx.unrecognised] == ["junk.png"]

    def test_row4_style_files_complete_a_tree(self, make_tree, tmp_path):
        root = make_tree("R04U01", skip=[("winter", "RGB"), ("winter", "D")])
        img = np.full((480, 640, 3), 100, np.uint8)
        write_png(root / "R04U01_Winter_RGB-R04N03_RGB_12-17-14.png", img)
        write_png(root / "R04U01_Winter_D-R04N03_D_12-17-14.png", img)
        idx = index_dataset(root)
        assert [t.tree_id for t in idx.trees] == ["R04U01"]
        assert not idx.unrecognised

    def test_reports_duplicates(self, make_tree):
        root = make_tree("R03U04")
        (root / "sub").mkdir()
        src = root / "R03U04_Summer_D-1-2-3.png"
        (root / "sub" / src.name).write_bytes(src.read_bytes())
        idx = index_dataset(root)
        assert len(idx.trees) == 1
        assert len(idx.duplicates) == 1

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            index_dataset(tmp_path / "nope")


class TestLoaders:
    def test_sample_tree_loads(self, sample_dir):
        tree = index_dataset(sample_dir).trees[0]
        for season in ("summer", "Winter"):
            d = load_tree(tree, season)
            assert d["rgb"].shape == (480, 640, 3) and d["rgb"].dtype == np.uint8
            assert d["depth"].shape == (480, 640)
            assert d["valid"].dtype == bool
            np.testing.assert_array_equal(d["valid"], d["depth"] > 0)

    def test_depth_accepts_single_channel(self, tmp_path):
        g = np.arange(480 * 640, dtype=np.uint32).reshape(480, 640) % 256
        p = write_png(tmp_path / "d.png", g.astype(np.uint8))
        depth, valid = load_depth(p)
        np.testing.assert_array_equal(depth, g)
        assert valid.sum() == depth.size - (g == 0).sum()

    def test_depth_rejects_colour_image(self, tmp_path):
        arr = np.zeros((480, 640, 3), np.uint8)
        arr[..., 1] = 5
        with pytest.raises(ValueError, match="channels differ"):
            load_depth(write_png(tmp_path / "c.png", arr))

    def test_wrong_size_rejected(self, tmp_path):
        p = write_png(tmp_path / "s.png", np.zeros((10, 10, 3), np.uint8))
        with pytest.raises(ValueError, match="expected"):
            load_rgb(p)

    def test_unknown_season_rejected(self, sample_dir):
        tree = index_dataset(sample_dir).trees[0]
        with pytest.raises(ValueError):
            load_tree(tree, "autumn")
