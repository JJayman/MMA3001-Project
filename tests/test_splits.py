"""Tests for treeskel.splits."""

import json

import pytest

from treeskel.io import TreeRecord
from treeskel.splits import (
    load_split,
    main,
    row_counts,
    row_kfold,
    select_test_rows,
    train_test_split_by_row,
)


def fake_trees(sizes):
    """Build TreeRecords with ``sizes[i]`` trees in row ``i + 1``."""
    return [
        TreeRecord(f"R{r:02d}U{p:02d}", r, p)
        for r, n in enumerate(sizes, start=1)
        for p in range(1, n + 1)
    ]


# Uneven row sizes, 23 rows, roughly like the real orchard (544 trees).
SIZES = [20, 25, 30, 18, 22, 27, 24, 19, 26, 23, 21, 28, 24, 22, 25, 20, 23, 26, 19, 24, 27, 21, 30]


def test_row_counts():
    assert row_counts(fake_trees([2, 3])) == {1: 2, 2: 3}


def test_test_rows_near_target_and_deterministic():
    trees = fake_trees(SIZES)
    rows = select_test_rows(trees, 0.15, seed=0)
    n_test = sum(SIZES[r - 1] for r in rows)
    assert abs(n_test / len(trees) - 0.15) < 0.03
    assert rows == select_test_rows(trees, 0.15, seed=0)
    assert rows != select_test_rows(trees, 0.15, seed=1)


def test_test_rows_leave_training_data():
    assert len(select_test_rows(fake_trees([5, 5]), 0.9)) == 1


@pytest.mark.parametrize("frac", [0, 1, -0.1])
def test_bad_fraction(frac):
    with pytest.raises(ValueError):
        select_test_rows(fake_trees(SIZES), frac)


def test_single_row_rejected():
    with pytest.raises(ValueError):
        select_test_rows(fake_trees([10]))


def test_no_row_in_both_train_and_test():
    trees = fake_trees(SIZES)
    dev, test = train_test_split_by_row(trees, select_test_rows(trees))
    assert len(dev) + len(test) == len(trees)
    assert not {t.row for t in dev} & {t.row for t in test}


def test_kfold_groups_rows_and_covers_every_tree_once():
    dev = fake_trees(SIZES[:19])
    folds = row_kfold(dev, 5)
    assert len(folds) == 5
    seen = []
    for train, val in folds:
        assert not {t.row for t in train} & {t.row for t in val}
        seen += [t.tree_id for t in val]
    assert sorted(seen) == sorted(t.tree_id for t in dev)
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= max(SIZES)  # balanced to within one row


def test_kfold_needs_enough_rows():
    with pytest.raises(ValueError):
        row_kfold(fake_trees([3, 3]), 5)


def test_cli_on_sample(sample_dir, capsys):
    assert main([str(sample_dir)]) == 0
    out = capsys.readouterr().out
    assert "Complete trees: 8" in out
    assert "Test rows" in out


def test_cli_usage(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_cli_saves_split(sample_dir, tmp_path):
    out = tmp_path / "splits.json"
    assert main([str(sample_dir), "--save", str(out)]) == 0
    saved = json.loads(out.read_text())
    all_rows = set(saved["test_rows"]).union(*map(set, saved["fold_val_rows"]))
    assert all_rows == {1, 2, 6, 17}
    assert not set(saved["test_rows"]) & set().union(*map(set, saved["fold_val_rows"]))


def test_cli_reports_embedded_id_offset(sample_dir, tmp_path, capsys):
    # Copy two sample trees, renaming one tree's winter files to the row 4 style.
    for src in sample_dir.glob("R01U0*.png"):
        name = src.name
        if name.startswith("R01U02_Winter"):
            mod = "RGB" if "_RGB-" in name else "D"
            # Real row 4 depth files embed "_RGB_" regardless of type.
            name = f"R01U02_Winter_{mod}-R01N05_RGB_11-23-42.png"
        (tmp_path / name).write_bytes(src.read_bytes())
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Complete trees: 2" in out
    assert "N number minus U position -> count: {3: 1}" in out
    assert "embedded RGB/D label differs: 1" in out


def test_check_images_flags_mislabelled_depth(sample_dir, tmp_path, capsys):
    for src in sample_dir.glob("R01U01*.png"):
        name = src.name
        # Swap the winter RGB image into the winter depth file's name.
        if name.startswith("R01U01_Winter_D-"):
            continue
        if name.startswith("R01U01_Winter_RGB-"):
            (tmp_path / name.replace("_RGB-", "_D-")).write_bytes(src.read_bytes())
        (tmp_path / name).write_bytes(src.read_bytes())
    assert main([str(tmp_path), "--check-images"]) == 0
    out = capsys.readouterr().out
    assert "Image check: 1 problem file(s)" in out
    assert "channels differ" in out


def test_check_images_passes_on_sample(sample_dir, capsys):
    assert main([str(sample_dir), "--check-images"]) == 0
    assert "Image check: 0 problem file(s)" in capsys.readouterr().out


def test_cli_reports_excluded_row_and_identical_images(sample_dir, tmp_path, capsys):
    # Two row 4 trees whose winter depth images are the same image, like the real data.
    depth = next(sample_dir.glob("R01U01_Winter_D-*.png")).read_bytes()
    for src in sample_dir.glob("R01U0*.png"):
        name = src.name.replace("R01", "R04", 1)
        data = depth if "_Winter_D-" in name else src.read_bytes()
        (tmp_path / name).write_bytes(data)
    for src in sample_dir.glob("R02U*.png"):
        (tmp_path / src.name).write_bytes(src.read_bytes())
    assert main([str(tmp_path), "--check-images"]) == 0
    out = capsys.readouterr().out
    assert "Complete trees: 2" in out
    assert "Excluded row 4 (2 trees): All winter depth images in row 4 are identical" in out
    assert "Identical-image groups: 1" in out
    assert "2 files: R04U01_Winter_D-" in out

    assert main([str(tmp_path), "--include-all-rows"]) == 0
    assert "Complete trees: 4" in capsys.readouterr().out


def test_split_save_and_load_roundtrip(sample_dir, tmp_path, capsys):
    out = tmp_path / "splits.json"
    assert main([str(sample_dir), "--save", str(out)]) == 0
    saved = capsys.readouterr().out
    assert main([str(sample_dir), "--load", str(out)]) == 0
    loaded = capsys.readouterr().out
    assert "Loaded split from" in loaded
    # Same test rows and folds either way.
    pick = lambda text: [l for l in text.splitlines() if l.startswith(("Test rows", "Fold"))]
    assert pick(saved) == pick(loaded)


def test_load_split_rejects_other_dataset(tmp_path):
    out = tmp_path / "splits.json"
    out.write_text(json.dumps({"all_rows": [1, 2, 3], "n_trees": 9,
                               "test_rows": [1], "fold_val_rows": [[2], [3]]}))
    with pytest.raises(ValueError, match="different|make a new split"):
        load_split(str(out), fake_trees([3, 3, 3, 3]))


def test_load_split_rejects_bad_folds(tmp_path):
    out = tmp_path / "splits.json"
    out.write_text(json.dumps({"all_rows": [1, 2, 3, 4], "n_trees": 12,
                               "test_rows": [1], "fold_val_rows": [[2], [2, 3]]}))
    with pytest.raises(ValueError, match="exactly once"):
        load_split(str(out), fake_trees([3, 3, 3, 3]))


def test_load_split_rebuilds_folds(tmp_path):
    trees = fake_trees([3, 3, 3, 3])
    out = tmp_path / "splits.json"
    out.write_text(json.dumps({"all_rows": [1, 2, 3, 4], "n_trees": 12,
                               "test_rows": [4], "fold_val_rows": [[1], [2, 3]]}))
    test_rows, folds = load_split(str(out), trees)
    assert test_rows == [4]
    assert [sorted({t.row for t in va}) for _, va in folds] == [[1], [2, 3]]
    assert [sorted({t.row for t in tr}) for tr, _ in folds] == [[2, 3], [1]]


def test_cli_counts_masks_and_checks_them(make_tree, tmp_path, capsys):
    make_tree("R01N01", mask=True)
    make_tree("R02N01")
    assert main([str(tmp_path), "--check-images"]) == 0
    out = capsys.readouterr().out
    assert "Complete trees: 2" in out
    assert "Complete trees with a summer mask: 1 of 2" in out
    assert "Image check: 0 problem file(s)" in out
