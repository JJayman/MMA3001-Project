"""Tests for treeskel.splits."""

import json

import pytest

from treeskel.io import TreeRecord
from treeskel.splits import (
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
            name = f"R01U02_Winter_{mod}-R01N05_{mod}_11-23-42.png"
        (tmp_path / name).write_bytes(src.read_bytes())
    assert main([str(tmp_path)]) == 0
    assert "N number minus U position -> count: {3: 1}" in capsys.readouterr().out
