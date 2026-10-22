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

Usage, from the command line:

```
python -m treeskel.shape_model --plots docs/figures   # data/shape/ -> docs/results/shape/
```
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

    Args:
        arrays: Feature arrays by name, from :func:`treeskel.shape.load_features`.
        name: A key of :data:`FEATURE_SETS` (e.g. ``"rgb"``: global and cell features).

    Returns:
        ``(N, 2, D)`` features.

    Example:
        >>> a = {"rgb_global": np.ones((3, 2, 4)), "rgb_cells": np.zeros((3, 2, 2, 4))}
        >>> feature_matrix(a, "rgb").shape
        (3, 2, 12)
    """
    parts = [arrays[k].reshape(arrays[k].shape[0], 2, -1) for k in FEATURE_SETS[name]]
    return np.concatenate(parts, axis=2)


def mirror_matrix(names: Sequence[str] = MEASURES) -> np.ndarray:
    """The matrix ``M`` with ``mirror(y) = y @ M`` for rows ``y`` of the measures ``names``.

    Args:
        names: The measures, in column order.

    Returns:
        The ``(k, k)`` matrix ``M``.

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
    """Read ``shape_measures.csv`` (numbers as floats, empty or ``nan`` as NaN).

    Args:
        path: The CSV file.

    Returns:
        The rows as dicts.
    """
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
        """Standardise the features, centre the targets and take the SVD once."""
        self.mu = x.mean(0)
        self.sd = x.std(0)
        self.sd[self.sd < 1e-6] = 1.0                       # constant features contribute nothing
        z = (x - self.mu) / self.sd
        y = y.reshape(len(y), -1)
        self.y0 = y.mean(0)
        self.u, self.s, self.vt = np.linalg.svd(z, full_matrices=False)
        self.uty = self.u.T @ (y - self.y0)

    def predict(self, x: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
        """``(len(alphas), m, k)`` predictions for ``(m, D)`` features.

        Args:
            x: ``(m, D)`` features.
            alphas: Penalties.

        Returns:
            The predictions.
        """
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
        x2_train: ``(n, 2, D)`` training features (photo, mirror image).
        y_train: ``(n, k)`` targets for the measures ``names``, no NaN.
        x2_test: ``(m, 2, D)`` features to predict.
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

    Args:
        x2_train: ``(n, 2, D)`` training features (photo, mirror image).
        y_train: ``(n, k)`` training targets (NaN = unknown).
        x2_test: ``(m, 2, D)`` features to predict.
        alphas: Penalties.
        names: The measures, in column order.
        mirror: Use the mirror images too.

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
        alphas: Penalties, as for :func:`predict_all`.
        names: The measures, in column order.
        mirror: Use the mirror images too, as for :func:`predict_all`.
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
        alphas: Penalties, as for :func:`predict_all`.
        names: The measures, in column order.
        mirror: Use the mirror images too, as for :func:`predict_all`.

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
        # One fit gives every penalty's predictions; each measure takes its own penalty's.
        p = predict_all(x2[tr], y[tr], x2[va], alphas, names, mirror)
        which = [list(alphas).index(v) for v in a]
        ridge[va] = p[which, :, range(len(names))].T
        mean[va] = np.nanmean(y[tr], axis=0)
    return {"ridge": ridge, "mean": mean, "fold": fold, "alpha": chosen, "alpha_at_edge": edge}


def evaluate(truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray, fold: np.ndarray | None = None) -> dict:
    """Scores of one measure's predictions against the truth and the mean-shape baseline.

    Only trees where the truth, the prediction and the baseline are all known count.

    Args:
        truth: Winter values per tree.
        pred: Predicted values per tree.
        baseline: The mean shape's values per tree.
        fold: Each tree's fold, for the per-fold skill; None to leave it out.

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
    """Print the table of scores per measure and model.

    Args:
        report: From :func:`run`.
        names: The measures shown.
    """
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

    Args:
        table: The winter shape table's rows.
        ids: Tree ids, in the order of the features.
        arrays: Feature arrays by name.
        fold_rows: The held-out rows of each fold.
        feature_sets: The sets tried.
        names: The measures.
        alphas: Penalties.

    Returns:
        ``(report, predictions)``: the scores per measure and model, and the
        per-tree predictions (``{model: (n, k)}``, the hand masks and the mean
        shape included, plus ``"truth"``, ``"fold"``, ``"tree"``, ``"row"``).
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
    preds = {"tree": trees, "row": rows, "truth": y, "fold": base["fold"], "mean shape": base["mean"],
             "hand masks": hand, **{label: cv["ridge"] for label, cv in models.items()}}
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
    """One row per tree and measure: the truth, the fold, and every model's prediction.

    Args:
        path: The CSV file.
        preds: The per-tree predictions from :func:`run`.
        names: The measures.
    """
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


LABELS = {"v_opening": "V opening", "left_bend": "Left bend", "right_bend": "Right bend",
          "left_angle": "Left leader from trunk", "right_angle": "Right leader from trunk", "lean": "V's lean from trunk",
          "trunk_tilt": "Trunk from vertical", "left_tilt": "Left leader from vertical",
          "right_tilt": "Right leader from vertical", "lean_v": "V's lean from vertical"}
"""Readable names of the measures for figures."""

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOURS = {"hand masks": "#2a78d6", "ridge rgb": "#eb6834"}
"""Series colours (the first two categorical slots of the project's validated palette)."""


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.set_axisbelow(True)


def _save(fig, path: Path, failed: list[str]) -> None:
    try:
        fig.savefig(path, facecolor=SURFACE)
    except OSError as e:                                    # e.g. open in a viewer on Windows
        failed.append(f"{path}: {e.strerror or e}")


def plot_skill(report: dict, path: Path, failed: list[str], models: Sequence[str] = ("hand masks", "ridge rgb")) -> None:
    """Horizontal bars: each model's skill over the mean shape, per measure, with the range over the folds.

    Skill 0 is the mean shape (the axis line) and 1 a perfect prediction.
    The whiskers span the five folds' skills, so a model that only helps in
    some rows shows it.

    Args:
        report: From :func:`run`.
        path: The figure file to write.
        failed: Files that could not be written are added to it (a viewer can lock a file on Windows).
        models: The models drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (("No reference", HEADLINE), ("From the trunk", TRUNK_RELATIVE), ("From the vertical", VERTICAL))
    rows, labels, y, pos = [], [], [], 0.0
    for _, names in groups:
        for k in names:
            rows.append(k)
            labels.append(LABELS[k])
            y.append(pos)
            pos += 1.0
        pos += 0.6                                           # a gap between the groups
    y = -np.array(y)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    h = 0.34
    for i, m in enumerate(models):
        off = (i - (len(models) - 1) / 2) * (h + 0.04)
        sk = [report["scores"][k][m].get("skill", np.nan) for k in rows]
        ax.barh(y - off, sk, height=h, color=COLOURS.get(m, MUTED), label=_model_name(m), zorder=2)
        for yy, k, v in zip(y - off, rows, sk):
            fs = [f for f in report["scores"][k][m].get("fold_skill", []) if f is not None]
            if fs:
                ax.plot([min(fs), max(fs)], [yy, yy], color=SECONDARY, lw=0.8, zorder=3, solid_capstyle="round")
            if k == "v_opening" and np.isfinite(v):          # label the headline only, clear of the whisker
                ax.text(max([v, 0.0, *fs]) + 0.02, yy, _signed(v), va="center", fontsize=7.5, color=SECONDARY)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(y, labels, fontsize=8, color=SECONDARY)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("Skill over the mean shape: 1 − MSE / MSE(mean shape)   (whiskers: the range over the 5 folds)",
                  fontsize=8, color=SECONDARY)
    ax.grid(axis="x", color=GRID, lw=0.5)
    gy = 0.0
    for title, names in groups:                               # group titles above each block
        ax.text(-0.2, gy + 0.55, title, fontsize=7.5, color=MUTED, ha="left", va="bottom", style="italic")
        gy -= len(names) + 0.6
    fig.suptitle("The hand masks read the V from the summer photo; the network's features barely do", x=0.02,
                 ha="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_title(f"{report['n_trees']} development trees, each predicted by the fold that held its row out",
                 loc="left", fontsize=8, color=SECONDARY)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2,
               frameon=False, fontsize=7.5, labelcolor=SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, path, failed)
    plt.close(fig)


def _signed(v: float) -> str:
    """``+0.12`` style, without a minus sign on a value that rounds to zero."""
    text = f"{v:+.2f}"
    return "+0.00" if text == "-0.00" else text


def _model_name(m: str) -> str:
    return {"hand masks": "Hand masks (a person's reading of the summer photo)",
            "ridge rgb": "Ridge regression on ResNet-18 features of the summer photo"}.get(m, m)


def plot_scatter(preds: dict, report: dict, path: Path, failed: list[str],
                 models: Sequence[str] = ("hand masks", "ridge rgb"), names: Sequence[str] = HEADLINE) -> None:
    """Small multiples: each model's prediction against the winter truth, one panel per model and measure.

    The diagonal is a perfect prediction; the dashed line is the mean shape.

    Args:
        preds: The per-tree predictions from :func:`run`.
        report: From :func:`run`.
        path: The figure file to write.
        failed: Files that could not be written are added to it (a viewer can lock a file on Windows).
        models: The models drawn.
        names: The measures drawn.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(models), len(names), figsize=(3.0 * len(names), 2.9 * len(models)), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for c, k in enumerate(names):
        j = MEASURES.index(k)
        truth = preds["truth"][:, j]
        lo, hi = np.nanpercentile(truth, [0.5, 99.5])
        pad = 0.08 * (hi - lo)
        lim = (lo - pad, hi + pad)
        for r, m in enumerate(models):
            ax = axes[r, c]
            _style(ax)
            p = preds[m][:, j]
            ok = np.isfinite(p) & np.isfinite(truth)
            ax.plot(lim, lim, color=AXIS, lw=0.8, zorder=1)
            ax.axhline(np.nanmean(truth), color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.scatter(truth[ok], p[ok], s=9, color=COLOURS.get(m, MUTED), alpha=0.55, lw=0, zorder=2)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.grid(color=GRID, lw=0.5)
            sc = report["scores"][k][m]
            ax.set_title(f"{LABELS[k]}: MAE {sc['mae']:.1f}°, skill {_signed(sc['skill'])}", loc="left",
                         fontsize=7.5, color=SECONDARY)
            if r == len(models) - 1:
                ax.set_xlabel("Winter trace (°)", fontsize=7.5, color=SECONDARY)
            if c == 0:
                ax.set_ylabel({"hand masks": "Hand mask (°)", "ridge rgb": "Ridge prediction (°)"}.get(m, m),
                              fontsize=7.5, color=SECONDARY)
    fig.suptitle("Predicted against the winter trace, per tree (diagonal: perfect; dashed: the mean shape)",
                 x=0.02, ha="left", fontsize=9.5, color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, path, failed)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Cross-validate the models on the development rows and save the scores.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if the table holds test-row trees or an output file could not be
        written.
    """
    ap = argparse.ArgumentParser(prog="python -m treeskel.shape_model", description=__doc__.splitlines()[0])
    ap.add_argument("--table", default="data/shape/shape_measures.csv")
    ap.add_argument("--features", default="data/shape/summer_resnet18.npz")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--out", default="docs/results/shape")
    ap.add_argument("--sets", nargs="+", choices=tuple(FEATURE_SETS), default=list(FEATURE_SETS))
    ap.add_argument("--plots", help="also save the two figures (skill per measure, scatters) in this folder")
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
    failed: list[str] = []
    saved = [str(out / "report.json"), str(out / "predictions.csv")]
    if args.plots and "ridge rgb" in preds:
        folder = Path(args.plots)
        folder.mkdir(parents=True, exist_ok=True)
        plot_skill(report, folder / "shape_model_skill.png", failed)
        plot_scatter(preds, report, folder / "shape_model_scatter.png", failed)
        saved += [str(folder / "shape_model_skill.png"), str(folder / "shape_model_scatter.png")]
    print(f"\nSaved {', '.join(saved)}")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
