"""Row-grouped data splits.

Trees in the same orchard row share background, lighting and capture session,
and neighbouring trees can appear in each other's images. Splitting by
individual tree would therefore leak information between training and test
data, so every split here keeps whole rows together:

1. :func:`select_test_rows` holds out a few whole rows as the final test set,
   which is not used until the final evaluation.
2. :func:`row_kfold` splits the remaining trees into k folds by row for
   model selection (cross-validation).

Run ``python -m treeskel.splits <data_dir>`` to print a summary for a dataset,
and add ``--save splits.json`` to record the split. From then on, use
``--load splits.json`` (and :func:`load_split` in code) so the split never
changes, even if the code that chooses it does.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from treeskel.io import (
    EXCLUDED_ROWS,
    EXCLUDED_TREES,
    TreeRecord,
    find_identical_images,
    index_dataset,
    load_depth,
    load_mask,
    load_rgb,
    parse_filename,
)


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


def split_to_dict(
    trees: Sequence[TreeRecord],
    test_rows: Sequence[int],
    folds: Sequence[tuple[list[TreeRecord], list[TreeRecord]]],
) -> dict:
    """Describe a split as plain data (rows only) for saving to JSON.

    Args:
        trees: All trees the split was made from.
        test_rows: Held-out test rows.
        folds: Output of :func:`row_kfold`.

    Returns:
        ``{"all_rows": [...], "n_trees": int, "test_rows": [...],
        "fold_val_rows": [[...], ...]}``. ``all_rows`` and ``n_trees`` let
        :func:`load_split` detect a split being used with a different dataset.
    """
    return {
        "all_rows": sorted(row_counts(trees)),
        "n_trees": len(trees),
        "test_rows": sorted(int(r) for r in test_rows),
        "fold_val_rows": [sorted(row_counts(val)) for _, val in folds],
    }


def load_split(
    path: str, trees: Sequence[TreeRecord]
) -> tuple[list[int], list[tuple[list[TreeRecord], list[TreeRecord]]]]:
    """Rebuild a saved split for the given trees.

    Args:
        path: JSON file written by ``--save``.
        trees: All trees (excluded rows already removed).

    Returns:
        ``(test_rows, folds)``, in the same form as :func:`select_test_rows`
        and :func:`row_kfold`.

    Raises:
        ValueError: If the file's rows do not match the trees' rows (the split
            was made for a different dataset), or the folds do not cover every
            development row exactly once.
    """
    with open(path) as f:
        d = json.load(f)
    rows = set(row_counts(trees))
    saved = set(d.get("all_rows", []))
    if saved != rows:
        raise ValueError(
            f"{path} was made for rows {sorted(saved)}, but the data has rows "
            f"{sorted(rows)}; make a new split with --save"
        )
    test_rows = [int(r) for r in d["test_rows"]]
    fold_rows = [set(map(int, f)) for f in d["fold_val_rows"]]
    dev_rows = rows - set(test_rows)
    flat = [r for f in fold_rows for r in f]
    if sorted(flat) != sorted(dev_rows):
        raise ValueError(f"{path}: folds do not cover each development row exactly once")
    dev, _ = train_test_split_by_row(trees, test_rows)
    folds = [([t for t in dev if t.row not in f], [t for t in dev if t.row in f])
             for f in fold_rows]
    return sorted(test_rows), folds


def check_images(trees: Sequence[TreeRecord]) -> list[tuple[str, str]]:
    """Open every image and check it loads as the type its name says.

    RGB files must be 640x480 colour images; depth files must be 640x480
    greyscale (identical channels); masks must be 256x256 greyscale. This
    catches files with a wrong ``RGB``/``D``/``GT`` label, which the filename
    alone cannot reveal.

    Args:
        trees: Tree records whose files should be checked.

    Returns:
        List of ``(filename, error message)`` for every file that failed.
    """
    bad = []
    for t in trees:
        for (_, modality), path in sorted(t.paths.items()):
            try:
                {"depth": load_depth, "mask": load_mask, "rgb": load_rgb}[modality](path)
            except Exception as e:  # report every failure; don't stop the scan
                bad.append((path.name, str(e)))
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    """Print a dataset and split summary, optionally saving the split.

    Usage: ``python -m treeskel.splits DATA_DIR [--save | --load splits.json]
    [--check-images]``.
    """
    ap = argparse.ArgumentParser(prog="python -m treeskel.splits", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", help="folder containing the dataset PNG files")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--save", metavar="JSON", help="make a new split and write it to this file")
    mode.add_argument("--load", metavar="JSON", help="use the split saved in this file")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-images", action="store_true",
                    help="open every image, check its size and type, and look for "
                         "identical images (slower)")
    ap.add_argument("--include-all-rows", action="store_true",
                    help="do not apply the default row and tree exclusions")
    args = ap.parse_args(argv)

    none = () if args.include_all_rows else None
    idx = index_dataset(args.data_dir, none, none)
    print(f"Complete trees: {len(idx.trees)}")
    ex_rows = {t.row for t in idx.excluded if t.row in EXCLUDED_ROWS}
    for row in sorted(ex_rows):
        n = sum(t.row == row for t in idx.excluded)
        print(f"Excluded row {row} ({n} trees): {EXCLUDED_ROWS[row]}")
    for t in idx.excluded:
        if t.row not in ex_rows:
            print(f"Excluded tree {t.tree_id}: {EXCLUDED_TREES.get(t.tree_id, '')}")
    print(f"Incomplete trees (excluded): {len(idx.incomplete)}")
    for r in idx.incomplete:
        print(f"  {r.tree_id}: missing {r.missing()}")
    print(f"Unrecognised files: {len(idx.unrecognised)}")
    for p in idx.unrecognised[:10]:
        print(f"  {p.name}")
    print(f"Duplicate files: {len(idx.duplicates)}")
    print(f"Trees per row: {row_counts(idx.trees)}")
    n_mask = sum(t.has_mask for t in idx.trees)
    print(f"Complete trees with a summer mask: {n_mask} of {len(idx.trees)}")

    # Files with an embedded original id (row 4 winter): show how the U
    # position relates to the embedded N number, so renaming can be checked.
    # Incomplete trees are included, since a naming problem often causes them.
    offsets = Counter()
    label_mismatch = 0
    for t in idx.trees + idx.incomplete + idx.excluded:
        for path in t.paths.values():
            info = parse_filename(path)
            if info.source_id is None:
                continue
            label_mismatch += info.source_modality != info.modality
            if info.modality == "rgb":
                offsets[int(info.source_id.split("N")[1]) - info.position] += 1
    if offsets:
        print(f"Embedded-id files: embedded number minus tree number -> count: {dict(sorted(offsets.items()))}")
        print(f"Embedded-id files whose embedded RGB/D label differs: {label_mismatch}")

    if args.check_images:
        everything = idx.trees + idx.incomplete + idx.excluded
        bad = check_images(everything)
        print(f"Image check: {len(bad)} problem file(s)")
        for name, err in bad:
            print(f"  {name}: {err}")
        same = find_identical_images(p for t in everything for p in t.paths.values())
        print(f"Identical-image groups: {len(same)}")
        for g in same[:10]:
            more = f" (+{len(g) - 3} more)" if len(g) > 3 else ""
            print(f"  {len(g)} files: {', '.join(p.name for p in g[:3])}{more}")

    if len(row_counts(idx.trees)) < 2:
        return 0
    if args.load:
        test_rows, folds = load_split(args.load, idx.trees)
        print(f"Loaded split from {args.load}")
    else:
        test_rows = select_test_rows(idx.trees, args.test_fraction, args.seed)
        dev, _ = train_test_split_by_row(idx.trees, test_rows)
        k = min(args.folds, len(row_counts(dev)))
        folds = row_kfold(dev, k) if k >= 2 else []
    _, test = train_test_split_by_row(idx.trees, test_rows)
    print(f"Test rows {test_rows}: {len(test)} trees "
          f"({len(test) / len(idx.trees):.1%})")
    for i, (tr, va) in enumerate(folds):
        print(f"Fold {i}: train {len(tr)}, val {len(va)}, "
              f"val rows {sorted(row_counts(va))}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(split_to_dict(idx.trees, test_rows, folds), f, indent=2)
        print(f"Saved split to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
