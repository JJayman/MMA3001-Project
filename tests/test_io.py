"""Tests for treeskel.io."""

import numpy as np
import pytest

from conftest import write_png
from treeskel import io
from treeskel.io import (
    find_identical_images,
    index_dataset,
    load_depth,
    load_mask,
    load_rgb,
    load_tree,
    mask_to_image_frame,
    parse_filename,
    shift_image,
    to_mask_frame,
)


class TestParseFilename:
    def test_n_naming_and_masks(self):
        info = parse_filename("R20N03_Summer_GT-14-21-52.png")
        assert info.tree_id == "R20N03"
        assert (info.row, info.position) == (20, 3)
        assert info.modality == "mask"
        assert parse_filename("R20N05_Winter_D-13-26-50.png").modality == "depth"
        assert parse_filename("R20N05_Summer_RGB-14-22-1.png").time == "14:22:01"

    def test_u_and_n_ids_are_different_trees(self):
        assert parse_filename("R01U01_Summer_D-1-2-3.png").tree_id == "R01U01"
        assert parse_filename("R01N01_Summer_D-1-2-3.png").tree_id == "R01N01"

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

    def test_outer_label_decides_modality(self):
        # Real row 4 winter depth files embed "_RGB_"; the outer "D" wins.
        info = parse_filename("R04U01_Winter_D-R04N03_RGB_12-17-14.png")
        assert info.modality == "depth"
        assert info.source_modality == "rgb"

    @pytest.mark.parametrize("bad", [
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
        assert idx.trees[0].tree_id == "R01N01"
        assert all(t.has_mask for t in idx.trees)
        assert [t.tree_id for t in idx.excluded] == ["R04N01"]  # row 4

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
        write_png(root / "R04U01_Winter_D-R04N03_RGB_12-17-14.png", img)
        idx = index_dataset(root, exclude_rows=())  # row 4 is excluded by default
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

    def test_depth_is_moved_by_the_seasons_shift(self, sample_dir, monkeypatch):
        tree = index_dataset(sample_dir).trees[0]
        raw = load_tree(tree, "winter", register=False)
        # Zero shift: the depth as stored.
        monkeypatch.setattr(io, "DEPTH_SHIFT", {"summer": (0, 0), "winter": (0, 0)})
        np.testing.assert_array_equal(load_tree(tree, "winter")["depth"], raw["depth"])
        monkeypatch.setattr(io, "DEPTH_SHIFT", {"summer": (0, 0), "winter": (-4, 1)})
        moved = load_tree(tree, "winter")
        np.testing.assert_array_equal(moved["depth"][:-4, 1:], raw["depth"][4:, :-1])   # up 4 rows, right 1 col
        assert not moved["depth"][-4:].any() and not moved["depth"][:, 0].any()         # uncovered: no reading
        np.testing.assert_array_equal(moved["valid"], moved["depth"] > 0)
        np.testing.assert_array_equal(moved["rgb"], raw["rgb"])                         # colour never moves
        np.testing.assert_array_equal(load_tree(tree, "winter", register=False)["depth"], raw["depth"])
        np.testing.assert_array_equal(load_tree(tree, "summer")["depth"],
                                      load_tree(tree, "summer", register=False)["depth"])


class TestShiftImage:
    def test_moves_content_and_fills_the_uncovered_edge(self):
        a = np.arange(1, 13).reshape(3, 4)
        np.testing.assert_array_equal(shift_image(a, (0, 0)), a)
        assert shift_image(a, (0, 0)) is not a
        np.testing.assert_array_equal(shift_image(a, (-1, 2)), [[0, 0, 5, 6], [0, 0, 9, 10], [0, 0, 0, 0]])
        np.testing.assert_array_equal(shift_image(a, (2, -3), fill=-1), [[-1] * 4, [-1] * 4, [4, -1, -1, -1]])
        assert (shift_image(a, (3, 0)) == 0).all() and (shift_image(a, (0, -9)) == 0).all()   # moved out entirely

    def test_keeps_dtype_and_channels(self):
        mask = np.zeros((5, 6), bool)
        mask[2, 3] = True
        moved = shift_image(mask, (1, 1))
        assert moved.dtype == bool and moved[3, 4] and moved.sum() == 1
        rgb = np.random.default_rng(0).integers(0, 255, (5, 6, 3)).astype(np.uint8)
        np.testing.assert_array_equal(shift_image(rgb, (0, 2))[:, 2:], rgb[:, :-2])
        # A move and its reverse restore everything the first move kept.
        back = shift_image(shift_image(rgb, (2, -1)), (-2, 1))
        np.testing.assert_array_equal(back[:-2, 1:], rgb[:-2, 1:])


class TestExclusionsAndDuplicates:
    def test_row4_excluded_by_default(self, make_tree):
        root = make_tree("R04U01")
        make_tree("R05U01")
        idx = index_dataset(root)
        assert [t.tree_id for t in idx.trees] == ["R05U01"]
        assert [t.tree_id for t in idx.excluded] == ["R04U01"]

    def test_exclusions_can_be_disabled(self, make_tree):
        root = make_tree("R04U01")
        idx = index_dataset(root, exclude_rows=())
        assert [t.tree_id for t in idx.trees] == ["R04U01"]
        assert not idx.excluded

    def test_incomplete_excluded_tree_goes_to_excluded(self, make_tree):
        root = make_tree("R04U01", skip=[("winter", "D")])
        idx = index_dataset(root)
        assert not idx.incomplete
        assert [t.tree_id for t in idx.excluded] == ["R04U01"]

    def test_find_identical_images(self, tmp_path):
        a = np.zeros((480, 640), np.uint8)
        b = a.copy()
        b[0, 0] = 1
        p1 = write_png(tmp_path / "a1.png", a)
        p2 = write_png(tmp_path / "a2.png", a)
        p3 = write_png(tmp_path / "b.png", b)
        assert find_identical_images([p3, p2, p1]) == [[p1, p2]]

    def test_sample_finds_real_row4_copy(self, sample_dir):
        # The sample includes R04N01, whose winter depth is a copy of R03N34's.
        groups = find_identical_images(sorted(sample_dir.glob("*.png")))
        assert [[p.name for p in g] for g in groups] == [[
            "R03N34_Winter_D-11-39-13.png",
            "R04N01_Winter_D-R04N05_RGB_12-17-20.png",
        ]]


class TestMasks:
    def test_index_records_optional_mask(self, make_tree):
        root = make_tree("R01N01", mask=True)
        make_tree("R01N02")
        idx = index_dataset(root)
        assert [t.tree_id for t in idx.trees] == ["R01N01", "R01N02"]
        assert [t.has_mask for t in idx.trees] == [True, False]

    def test_load_mask(self, tmp_path):
        m = np.zeros((256, 256), np.uint8)
        m[10:20, 30:40] = 200
        m[0, 0] = 100  # below threshold: not branch
        out = load_mask(write_png(tmp_path / "m.png", m))
        assert out.dtype == bool and out.shape == (256, 256)
        assert out.sum() == 100

    def test_load_mask_accepts_rgb_greyscale(self, tmp_path):
        m = np.zeros((256, 256, 3), np.uint8)
        m[5, 5] = 255
        assert load_mask(write_png(tmp_path / "m.png", m)).sum() == 1

    @pytest.mark.parametrize("arr, msg", [
        (np.zeros((480, 640), np.uint8), "expected"),
        (np.dstack([np.zeros((256, 256), np.uint8), np.full((256, 256), 9, np.uint8),
                    np.zeros((256, 256), np.uint8)]), "channels differ"),
    ])
    def test_load_mask_rejects_bad_files(self, tmp_path, arr, msg):
        with pytest.raises(ValueError, match=msg):
            load_mask(write_png(tmp_path / "m.png", arr))

    def test_to_mask_frame_takes_centre_crop(self):
        img = np.zeros((480, 640), np.uint8)
        img[:, 80:560] = 200          # the centre 480 columns
        out = to_mask_frame(img, nearest=True)
        assert out.shape == (256, 256)
        assert (out == 200).all()     # nothing from outside the crop

    def test_to_mask_frame_rgb_and_bool(self):
        rgb = np.full((480, 640, 3), 50, np.uint8)
        assert to_mask_frame(rgb).shape == (256, 256, 3)
        b = np.zeros((480, 640), bool)
        b[:, 300:340] = True
        out = to_mask_frame(b)
        assert out.dtype == bool and out.any()

    def test_mask_roundtrip_through_image_frame(self):
        m = np.zeros((256, 256), bool)
        m[40:200, 100:140] = True
        full, covered = mask_to_image_frame(m)
        assert full.shape == covered.shape == (480, 640)
        assert not full[:, :80].any() and not full[:, 560:].any()
        assert covered[:, 80:560].all() and not covered[:, :80].any()
        back = to_mask_frame(full, nearest=True)
        # Nearest-neighbour both ways loses at most a pixel at the edges.
        assert (back ^ m).sum() <= 2 * (160 + 40)

    def test_wrong_sizes_rejected(self):
        with pytest.raises(ValueError):
            to_mask_frame(np.zeros((256, 256), np.uint8))
        with pytest.raises(ValueError):
            mask_to_image_frame(np.zeros((480, 640), bool))


def test_exclude_trees(make_tree):
    root = make_tree("R03N34")
    make_tree("R03N33")
    assert len(index_dataset(root).trees) == 2  # R03N34 is kept by default
    idx = index_dataset(root, exclude_trees=["R03N34"])
    assert [t.tree_id for t in idx.trees] == ["R03N33"]
    assert [t.tree_id for t in idx.excluded] == ["R03N34"]
