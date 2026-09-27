"""Predict each tree's winter shape from its summer photo, with the row-grouped folds.

The targets are the shape measures of the winter traces (:mod:`treeskel.shape`):
the V opening, each leader's angle from the trunk and from the vertical, the
V's lean and each leader's bend. They need no alignment between the seasons
(session 7 found none reliable). The inputs are features of the summer photo
from ResNet-18, a network pretrained on ImageNet: its last feature map
averaged over the whole photo and over a 2x2 grid of cells (computed by
``python -m treeskel.shape`` on the student's PC and committed in
``data/shape/``). With about 450 trees, a pretrained network used as a fixed
feature extractor, with a linear model on top, is the usual first step
(transfer learning).

**Model** (:func:`predict_fold`): ridge regression, ``min ||y − Xw||² +
α||w||²``, on standardised features, solved for every ``α`` at once from one
singular value decomposition of the training features. Each measure is
fitted separately. The mirror image of every training photo is added as a
second example, with the mirrored measures (the left and right leaders swap,
leans and tilts change sign; :data:`treeskel.shape.MIRROR`): a free way to
double the data, and a mirrored tree is still a real tree. The measures are
centred on the training trees' mean first, so a real difference between
left and right (or a camera tilt shared by all photos) is kept, and only the
tree-to-tree variation is learned from both. A prediction averages the
photo's and the mirror of its mirror image's.

**Choosing α without the held-out rows** (:func:`choose_alpha`): within each
training set, by row-grouped cross-validation again (nested), so the rows
being predicted never influence any choice.

**Baselines:**

- *mean shape*: each measure's mean over the training trees (a model that
  ignores the photo; any useful model must beat it);
- *hand masks*: the same measures from the provided hand-drawn masks, as they
  are (not fitted to anything): a person's reading of the summer view.

**Scores** (:func:`evaluate`), on the held-out rows of all 5 folds: mean
absolute error, root mean squared error, skill ``1 − MSE / MSE(mean shape)``
(0 = no better than the mean shape, 1 = perfect), the correlation with the
truth, and each tree's error compared with the mean shape's (the fraction
of trees improved and a Wilcoxon signed-rank test). Trees in one row share
light and camera settings, so the per-fold skills are also shown: a model
that only helps in one fold is not reliable.

Usage::

    python -m treeskel.shape_model                   # data/shape/ -> docs/results/shape/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import stats

from treeskel.shape import MEASURES, MIRROR, load_features

HEADLINE = ("v_opening", "left_bend", "right_bend")
"""The measures that need no reference direction (the student's choice, session 10)."""

TRUNK_RELATIVE = ("left_angle", "right_angle", "lean")
"""Measured from the trunk's axis: roll-free, but the trunk's top leans a lot (session 10)."""

VERTICAL = ("trunk_tilt", "left_tilt", "right_tilt", "lean_v")
"""Measured from the photo's vertical: camera roll between the seasons adds to them."""

ALPHAS = tuple(10.0 ** np.arange(0.0, 7.01, 0.5))
"""Ridge penalties tried (on standardised features)."""

FEATURE_SETS = {
    "rgb": ("rgb_global", "rgb_cells"),
    "rgb_global": ("rgb_global",),
    "depth": ("depth_global",),
    "rgb+depth": ("rgb_global", "rgb_cells", "depth_global"),
}
"""Named feature sets: which saved arrays are joined. ``rgb`` (the photo's features over the whole
photo and a 2x2 grid) is the model chosen before seeing any result; the others are for comparison."""


def feature_matrix(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    """The features of one set as ``(N, 2, D)``: axis 1 is the photo, then its mirror image.

    Example:
        >>> a = {"rgb_global": np.ones((3, 2, 4)), "rgb_cells": np.zeros((3, 2, 2, 4))}
        >>> feature_matrix(a, "rgb").shape
        (3, 2, 12)
    """
    parts = [arrays[k].reshape(arrays[k].shape[0], 2, -1) for k in FEATURE_SETS[name]]
    return np.concatenate(parts, axis=2)


def mirror_matrix(names: Sequence[str] = MEASURES) -> np.ndarray:
    """The matrix ``M`` with ``mirror(y) = y @ M`` for rows ``y`` of the measures ``names``.

    Example:
        >>> m = mirror_matrix(("left_bend", "right_bend", "lean"))
        >>> (np.array([[1.0, 2.0, 3.0]]) @ m).tolist()
        [[2.0, 1.0, -3.0]]
    """
    idx = {k: i for i, k in enumerate(names)}
    m = np.zeros((len(names), len(names)))
    for k, src in MIRROR.items():
        if k in idx:
            sign, src = (-1.0, src[1:]) if src.startswith("-") else (1.0, src)
            m[idx[src], idx[k]] = sign
    return m


def load_table(path: str | Path) -> list[dict]:
    """Read ``shape_measures.csv`` (numbers as floats, empty or ``nan`` as NaN)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k in ("tree", "note"):
                    out[k] = v
                elif k == "row":
                    out[k] = int(v)
                else:
                    out[k] = float(v) if v not in ("", "nan") else float("nan")
            rows.append(out)
    return rows


class RidgePath:
    """Ridge regression for many penalties at once, from one SVD.

    The features are standardised with the training mean and standard
    deviation, the targets centred on their training mean (so the intercept
    is not penalised). With ``X = U S Vᵀ``, the solution for penalty ``α`` is
    ``w = V diag(s / (s² + α)) Uᵀ y``.

    Args:
        x: ``(n, D)`` training features.
        y: ``(n,)`` or ``(n, k)`` training targets (already centred if the
            caller wants its own centre; the model centres them again).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.mu = x.mean(0)
        self.sd = x.std(0)
        self.sd[self.sd < 1e-6] = 1.0                       # constant features contribute nothing
        z = (x - self.mu) / self.sd
        y = y.reshape(len(y), -1)
        self.y0 = y.mean(0)
        self.u, self.s, self.vt = np.linalg.svd(z, full_matrices=False)
        self.uty = self.u.T @ (y - self.y0)

    def predict(self, x: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
        """``(len(alphas), m, k)`` predictions for ``(m, D)`` features."""
        z = (x - self.mu) / self.sd
        zv = z @ self.vt.T                                  # (m, r)
        out = [(zv * (self.s / (self.s ** 2 + a))) @ self.uty + self.y0 for a in alphas]
        return np.stack(out)


def _augment(x2: np.ndarray, y: np.ndarray, m: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Training examples: the photos with their targets, and (if ``m`` is given) the mirror images with
    the mirrored targets. ``x2`` is ``(n, 2, D)``, ``y`` is ``(n, k)`` centred targets."""
    if m is None:
        return x2[:, 0], y
    return np.concatenate([x2[:, 0], x2[:, 1]]), np.concatenate([y, y @ m])


def _combine(pred_photo: np.ndarray, pred_mirror: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    """Average a photo's prediction with the mirrored prediction for its mirror image (if mirroring)."""
    return pred_photo if m is None else (pred_photo + pred_mirror @ m) / 2


def fit_predict(x2_train: np.ndarray, y_train: np.ndarray, x2_test: np.ndarray, alphas: Sequence[float],
                names: Sequence[str], mirror: bool = True) -> np.ndarray:
    """Ridge predictions for every penalty, for measures with no missing values in ``y_train``.

    Args:
        x2_train, x2_test: ``(n, 2, D)`` features (photo, mirror image).
        y_train: ``(n, k)`` targets for the measures ``names``, no NaN.
        alphas: Penalties.
        names: The measures' names (for the mirror matrix; all of a mirrored
            pair must be present if ``mirror``).
        mirror: Train on the mirror images too, and average the two predictions.

    Returns:
        ``(len(alphas), m, k)`` predictions.
    """
    m = mirror_matrix(names) if mirror else None
    centre = y_train.mean(0)
    x, y = _augment(x2_train, y_train - centre, m)
    model = RidgePath(x, y)
    p = model.predict(np.concatenate([x2_test[:, 0], x2_test[:, 1]]), alphas)
    n = len(x2_test)
    return _combine(p[:, :n], p[:, n:], m) + centre


def _groups(names: Sequence[str]) -> list[tuple[str, ...]]:
    """The measures split into groups closed under mirroring (each is fitted with its mirror partner)."""
    left = list(names)
    groups = []
    while left:
        k = left.pop(0)
        partner = MIRROR[k].lstrip("-")
        if partner != k and partner in left:
            left.remove(partner)
            groups.append((k, partner))
        else:
            groups.append((k,))
    return groups


def predict_all(x2_train: np.ndarray, y_train: np.ndarray, x2_test: np.ndarray, alphas: Sequence[float],
                names: Sequence[str] = MEASURES, mirror: bool = True) -> np.ndarray:
    """:func:`fit_predict` for every measure, each group on the training trees where it is known.

    Returns:
        ``(len(alphas), m, len(names))`` predictions (NaN only if a group has
        fewer than 3 known training trees).
    """
    out = np.full((len(alphas), len(x2_test), len(names)), np.nan)
    for g in _groups(names):
        cols = [names.index(k) for k in g]
        ok = np.isfinite(y_train[:, cols]).all(1)
        if ok.sum() >= 3:
            out[:, :, cols] = fit_predict(x2_train[ok], y_train[ok][:, cols], x2_test, alphas, g, mirror)
    return out


def choose_alpha(x2: np.ndarray, y: np.ndarray, rows: np.ndarray, alphas: Sequence[float] = ALPHAS,
                 names: Sequence[str] = MEASURES, mirror: bool = True, n_splits: int = 4) -> np.ndarray:
    """The penalty for each measure with the smallest row-grouped cross-validated squared error.

    Args:
        x2: ``(n, 2, D)`` training features.
        y: ``(n, k)`` training targets (NaN = unknown).
        rows: ``(n,)`` orchard row of each tree (the groups).
        alphas, names, mirror: As for :func:`predict_all`.
        n_splits: Inner folds (fewer if there are fewer rows).

    Returns:
        ``(k,)`` chosen penalties.
    """
    from sklearn.model_selection import GroupKFold

    k = min(n_splits, len(set(rows.tolist())))
    sse = np.zeros((len(alphas), y.shape[1]))
    for tr, va in GroupKFold(n_splits=k).split(x2, groups=rows):
        p = predict_all(x2[tr], y[tr], x2[va], alphas, names, mirror)
        err = (p - y[va][None]) ** 2
        sse += np.nansum(err, axis=1)
    return np.asarray(alphas)[np.argmin(sse, axis=0)]


def cross_validate(x2: np.ndarray, y: np.ndarray, rows: np.ndarray, fold_rows: Sequence[Sequence[int]],
                   alphas: Sequence[float] = ALPHAS, names: Sequence[str] = MEASURES,
                   mirror: bool = True) -> dict:
    """Ridge and mean-shape predictions for every tree, each from the fold that holds its row out.

    Args:
        x2: ``(n, 2, D)`` features of the development trees.
        y: ``(n, k)`` winter measures (NaN = unknown).
        rows: ``(n,)`` orchard rows.
        fold_rows: The held-out rows of each fold (``splits.json``'s ``fold_val_rows``).
        alphas, names, mirror: As for :func:`predict_all`.

    Returns:
        ``{"ridge": (n, k), "mean": (n, k), "fold": (n,), "alpha": (folds, k),
        "alpha_at_edge": [...]}``.
    """
    n = len(y)
    ridge, mean = np.full((n, len(names)), np.nan), np.full((n, len(names)), np.nan)
    fold = np.full(n, -1)
    chosen, edge = [], []
    for i, held in enumerate(fold_rows):
        va = np.isin(rows, list(held))
        tr = ~va
        if not va.any():
            continue
        fold[va] = i
        a = choose_alpha(x2[tr], y[tr], rows[tr], alphas, names, mirror)
        chosen.append(a.tolist())
        for j, name in enumerate(names):
            if a[j] in (alphas[0], alphas[-1]):
                edge.append(f"fold {i} {name}: {a[j]:g}")
        for alpha in sorted(set(a.tolist())):
            use = [j for j in range(len(names)) if a[j] == alpha]
            p = predict_all(x2[tr], y[tr], x2[va], [alpha], names, mirror)[0]
            ridge[np.ix_(va, use)] = p[:, use]
        mean[va] = np.nanmean(y[tr], axis=0)
    return {"ridge": ridge, "mean": mean, "fold": fold, "alpha": chosen, "alpha_at_edge": edge}


def evaluate(truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray, fold: np.ndarray | None = None) -> dict:
    """Scores of one measure's predictions against the truth and the mean-shape baseline.

    Only trees where the truth, the prediction and the baseline are all known count.

    Returns:
        ``{"n", "mae", "rmse", "skill", "r", "baseline_mae", "improved", "wilcoxon_p",
        "fold_skill"}``; ``skill`` is ``1 − MSE / MSE(baseline)``.
    """
    ok = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    t, p, b = truth[ok], pred[ok], baseline[ok]
    if len(t) < 3:
        return {"n": int(len(t))}
    e, eb = p - t, b - t
    mse, mse_b = float(np.mean(e ** 2)), float(np.mean(eb ** 2))
    diff = np.abs(eb) - np.abs(e)
    out = {
        "n": int(len(t)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(mse)),
        "skill": 1 - mse / mse_b if mse_b > 0 else float("nan"),
        "r": float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else float("nan"),
        "bias": float(np.mean(e)),
        "baseline_mae": float(np.mean(np.abs(eb))),
        "improved": float(np.mean(diff > 0)),
        "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
    }
    if fold is not None:
        f = fold[ok]
        out["fold_skill"] = [1 - float(np.mean(e[f == i] ** 2)) / float(np.mean(eb[f == i] ** 2))
                             if (f == i).sum() > 2 and np.mean(eb[f == i] ** 2) > 0 else None
                             for i in range(int(f.max()) + 1)]
    return out


def _fmt(x, spec: str) -> str:
    return "    -" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def print_scores(report: dict, names: Sequence[str]) -> None:
    """Print the table of scores per measure and model."""
    print(f"\n  {'measure':12s} {'model':19s} {'n':>4s} {'MAE':>6s} {'RMSE':>6s} {'skill':>6s} {'r':>5s} "
          f"{'bias':>6s} {'better':>6s} {'p':>8s}  per-fold skill")
    for k in names:
        for model, sc in report["scores"][k].items():
            if sc.get("n", 0) < 3:
                continue
            folds = " ".join(_fmt(v, "+.2f") for v in sc.get("fold_skill", []))
            print(f"  {k:12s} {model:19s} {sc['n']:4d} {sc['mae']:6.2f} {sc['rmse']:6.2f} {_fmt(sc['skill'], '+6.2f')} "
                  f"{_fmt(sc['r'], '5.2f')} {sc['bias']:+6.2f} {sc['improved']:6.0%} {sc['wilcoxon_p']:8.1e}  {folds}")


def run(table: Sequence[dict], ids: Sequence[str], arrays: dict, fold_rows: Sequence[Sequence[int]],
        feature_sets: Sequence[str] = tuple(FEATURE_SETS), names: Sequence[str] = MEASURES,
        alphas: Sequence[float] = ALPHAS) -> tuple[dict, dict]:
    """Cross-validate every feature set (with and without mirroring for ``rgb``) and score all models.

    Returns:
        ``(report, predictions)``: the scores per measure and model, and the
        per-tree predictions (``{model: (n, k)}`` plus ``"truth"``, ``"hand"``,
        ``"fold"``, ``"tree"``, ``"row"``).
    """
    by_id = {r["tree"]: r for r in table}
    keep = [i for i, t in enumerate(ids) if t in by_id]
    trees = [ids[i] for i in keep]
    rows = np.array([by_id[t]["row"] for t in trees])
    y = np.array([[by_id[t][f"winter_{k}"] for k in names] for t in trees], float)
    hand = np.array([[by_id[t][f"hand_{k}"] for k in names] for t in trees], float)
    models = {}
    base = None
    variants = [(f, True) for f in feature_sets] + ([("rgb", False)] if "rgb" in feature_sets else [])
    for fs, mirror in variants:
        x2 = feature_matrix(arrays, fs)[keep]
        cv = cross_validate(x2, y, rows, fold_rows, alphas, names, mirror)
        label = f"ridge {fs}" + ("" if mirror else " no-mirror")
        models[label] = cv
        base = cv
    preds = {"tree": trees, "row": rows, "truth": y, "hand": hand, "fold": base["fold"], "mean": base["mean"],
             **{label: cv["ridge"] for label, cv in models.items()}}
    report = {"n_trees": len(trees), "fold_rows": [list(map(int, f)) for f in fold_rows],
              "alphas": {label: cv["alpha"] for label, cv in models.items()},
              "alpha_at_edge": {label: cv["alpha_at_edge"] for label, cv in models.items() if cv["alpha_at_edge"]},
              "scores": {}}
    for j, k in enumerate(names):
        mean_shape = evaluate(y[:, j], base["mean"][:, j], base["mean"][:, j], base["fold"])
        mean_shape["r"] = None                      # a constant per fold: its correlation says nothing
        sc = {"mean shape": mean_shape,
              "hand masks": evaluate(y[:, j], hand[:, j], base["mean"][:, j], base["fold"])}
        for label, cv in models.items():
            sc[label] = evaluate(y[:, j], cv["ridge"][:, j], cv["mean"][:, j], cv["fold"])
        report["scores"][k] = sc
    return report, preds


def save_predictions(path: Path, preds: dict, names: Sequence[str] = MEASURES) -> None:
    """One row per tree and measure: the truth, the fold, and every model's prediction."""
    models = [k for k in preds if k not in ("tree", "row", "truth", "fold")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", "fold", "measure", "winter"] + models)
        for i, t in enumerate(preds["tree"]):
            for j, k in enumerate(names):
                w.writerow([t, int(preds["row"][i]), int(preds["fold"][i]), k, _round(preds["truth"][i, j])]
                           + [_round(preds[m][i, j]) for m in models])


def _round(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.2f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Cross-validate the models on the development rows and save the scores."""
    ap = argparse.ArgumentParser(prog="python -m treeskel.shape_model", description=__doc__.splitlines()[0])
    ap.add_argument("--table", default="data/shape/shape_measures.csv")
    ap.add_argument("--features", default="data/shape/summer_resnet18.npz")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", default="docs/results/shape")
    ap.add_argument("--sets", nargs="+", choices=tuple(FEATURE_SETS), default=list(FEATURE_SETS))
    args = ap.parse_args(argv)

    split = json.loads(Path(args.split).read_text())
    table = load_table(args.table)
    ids, arrays, meta = load_features(args.features)
    test_rows = set(split["test_rows"])
    if any(r["row"] in test_rows for r in table):
        print("The table holds test-row trees; they are never used here. Stopping.")
        return 1
    print(f"{len(table)} trees in the table, {len(ids)} with features ({meta.get('network')}, "
          f"{meta.get('weights')} weights); folds: {split['fold_val_rows']}", flush=True)
    report, preds = run(table, ids, arrays, split["fold_val_rows"], args.sets)
    report["features"] = meta
    groups = (("Headline (no reference direction)", HEADLINE), ("From the trunk's axis", TRUNK_RELATIVE),
              ("From the photo's vertical (camera roll adds to these)", VERTICAL))
    for title, names in groups:
        print(f"\n{title}:")
        print_scores(report, names)
    for label, edges in report["alpha_at_edge"].items():
        print(f"  {label}: penalty at the edge of the grid for {len(edges)} fold-measures ({edges[0]}, ...)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    save_predictions(out / "predictions.csv", preds)
    print(f"\nSaved {out / 'report.json'} and {out / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
