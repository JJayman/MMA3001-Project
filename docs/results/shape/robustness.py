"""Is the weak result the linear model's fault or the features'? Non-linear models on the same features.

Exploratory check for the README (not a model choice): k-nearest neighbours and
kernel ridge regression (RBF kernel) on the same ResNet-18 features of the
summer photo (``rgb``: the whole photo and a 2x2 grid), with the same
row-grouped folds of ``splits.json``, their settings chosen inside each
training set by row-grouped cross-validation, as for the ridge model. No
mirror images, to keep it simple (mirroring changed the ridge model's
skill by at most 0.06 on any measure).

Run from the repository root::

    python docs/results/shape/robustness.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from treeskel.shape import MEASURES, load_features
from treeskel.shape_model import evaluate, feature_matrix, load_table

ROOT = Path(__file__).resolve().parents[3]


def models(d: int) -> dict:
    """The two non-linear models and their settings grids (``d`` = number of features)."""
    return {
        "k nearest neighbours": (make_pipeline(StandardScaler(), KNeighborsRegressor(metric="cosine")),
                                 {"kneighborsregressor__n_neighbors": [5, 10, 20, 40, 80]}),
        "kernel ridge (RBF)": (make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                               {"kernelridge__alpha": [0.1, 1.0, 10.0],
                                "kernelridge__gamma": [g / d for g in (0.03, 0.1, 0.3, 1.0)]}),
    }


def main() -> None:
    split = json.loads((ROOT / "splits.json").read_text())
    table = {r["tree"]: r for r in load_table(ROOT / "data/shape/shape_measures.csv")}
    ids, arrays, _ = load_features(ROOT / "data/shape/summer_resnet18.npz")
    ids = [t for t in ids if t in table]
    x = feature_matrix(arrays, "rgb")[: len(ids), 0]
    rows = np.array([table[t]["row"] for t in ids])
    print(f"{len(ids)} trees, {x.shape[1]} features; skill over the mean shape on held-out rows (per-fold range)")
    print(f"  {'measure':12s} " + " ".join(f"{name:>30s}" for name in models(x.shape[1])))
    for k in MEASURES:
        y = np.array([table[t][f"winter_{k}"] for t in ids])
        cells = []
        for name, (model, grid) in models(x.shape[1]).items():
            pred, base, fold = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), -1)
            for i, held in enumerate(split["fold_val_rows"]):
                va = np.isin(rows, held)
                tr = ~va & np.isfinite(y)
                search = GridSearchCV(model, grid, cv=GroupKFold(n_splits=4), scoring="neg_mean_squared_error")
                search.fit(x[tr], y[tr], groups=rows[tr])
                pred[va], base[va], fold[va] = search.predict(x[va]), y[tr].mean(), i
            s = evaluate(y, pred, base, fold)
            fs = [v for v in s["fold_skill"] if v is not None]
            cells.append(f"{s['skill']:+.2f} ({min(fs):+.2f} to {max(fs):+.2f})")
        print(f"  {k:12s} " + " ".join(f"{c:>30s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
