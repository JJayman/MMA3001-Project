"""Row-grouped data splits.

Trees in the same orchard row share background, lighting and capture session,
and neighbouring trees can appear in each other's images. Splitting by
individual tree would therefore leak information between training and test
data, so every split here keeps whole rows together:

1. :func:`select_test_rows` holds out a few whole rows as the final test set,
   which is not used until the final evaluation.
2. :func:`row_kfold` splits the remaining trees into k folds by row for
   model selection (cross-validation).

Run ``python -m treeskel.splits <data_dir>`` to print a summary for a dataset.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from treeskel.io import TreeRecord, index_dataset


def row_counts(trees: Sequence[TreeRecord]) -> dict[int, int]:
    """Count trees per row.

    Args:
        trees: Tree records.

    Returns:
        Mapping ``row -> number of trees``, sorted by row.
    """
    c = Counter(t.row for t in trees)
    return dict(sorted(c.items()))


def select_test_rows(
    trees: Sequence[TreeRecord], test_fraction: float = 0.15, seed: int = 0
) -> list[int]:
    """Choose whole rows to hold out as the test set.

    Rows are visited in a seeded random order, and each row is added if doing
    so moves the held-out tree count closer to ``test_fraction`` of all trees.
    Because rows have different sizes, the final fraction is approximate.

    Args:
        trees: All tree records.
        test_fraction: Target fraction of trees in the test set, in (0, 1).
        seed: Random seed, so the same rows are chosen on every run.

    Returns:
        Sorted list of held-out row numbers (at least one row).

    Raises:
        ValueError: If ``test_fraction`` is outside (0, 1) or there are fewer
            than two rows (a row-grouped hold-out is then impossible).
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    counts = row_counts(trees)
    if len(counts) < 2:
        raise ValueError("Need trees from at least two rows to hold out a row")

    target = test_fraction * sum(counts.values())
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    total = 0
    for row in rng.permutation(list(counts)):
        row = int(row)
        # Always take the first row; then only add rows that reduce the error.
        if not chosen or abs(total + counts[row] - target) < abs(total - target):
            chosen.append(row)
            total += counts[row]
        # Never hold out every row - something must remain for training.
        if len(chosen) == len(counts) - 1:
            break
    return sorted(chosen)


def train_test_split_by_row(
    trees: Sequence[TreeRecord], test_rows: Sequence[int]
) -> tuple[list[TreeRecord], list[TreeRecord]]:
    """Split trees into (development, test) sets by row.

    Args:
        trees: All tree records.
        test_rows: Row numbers to hold out.

    Returns:
        ``(dev_trees, test_trees)``.
    """
    held = set(test_rows)
    dev = [t for t in trees if t.row not in held]
    test = [t for t in trees if t.row in held]
    return dev, test


def row_kfold(
    trees: Sequence[TreeRecord], n_splits: int = 5
) -> list[tuple[list[TreeRecord], list[TreeRecord]]]:
    """Split trees into k cross-validation folds, keeping whole rows together.

    Uses scikit-learn's ``GroupKFold`` with the row as the group, which
    balances the number of trees per fold even though rows differ in size.

    Args:
        trees: Development trees (the test rows already removed).
        n_splits: Number of folds, k.

    Returns:
        List of ``(train_trees, val_trees)`` pairs, one per fold.

    Raises:
        ValueError: If there are fewer distinct rows than ``n_splits``.
    """
    groups = np.array([t.row for t in trees])
    n_rows = len(set(groups.tolist()))
    if n_rows < n_splits:
        raise ValueError(f"Need at least {n_splits} rows for {n_splits} folds, got {n_rows}")
    folds = []
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(groups, groups=groups):
        folds.append(([trees[i] for i in tr_idx], [trees[i] for i in va_idx]))
    return folds


def main(argv: Sequence[str] | None = None) -> int:
    """Print a dataset and split summary. Usage: ``python -m treeskel.splits DIR``."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print("Usage: python -m treeskel.splits <data_dir>")
        return 2
    idx = index_dataset(argv[0])
    print(f"Complete trees: {len(idx.trees)}")
    print(f"Incomplete trees (excluded): {len(idx.incomplete)}")
    for r in idx.incomplete:
        print(f"  {r.tree_id}: missing {r.missing()}")
    print(f"Unrecognised files: {len(idx.unrecognised)}")
    print(f"Duplicate files: {len(idx.duplicates)}")
    print(f"Trees per row: {row_counts(idx.trees)}")

    if len(row_counts(idx.trees)) < 2:
        return 0
    test_rows = select_test_rows(idx.trees)
    dev, test = train_test_split_by_row(idx.trees, test_rows)
    print(f"Test rows {test_rows}: {len(test)} trees "
          f"({len(test) / len(idx.trees):.1%})")
    k = min(5, len(row_counts(dev)))
    if k >= 2:
        for i, (tr, va) in enumerate(row_kfold(dev, k)):
            print(f"Fold {i}: train {len(tr)}, val {len(va)}, "
                  f"val rows {sorted(row_counts(va))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
