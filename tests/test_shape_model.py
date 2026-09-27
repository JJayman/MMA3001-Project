"""Tests for treeskel.shape_model on synthetic data with a known answer."""

import csv
import json

import numpy as np
import pytest

from treeskel.shape import COLUMNS, MEASURES, SOURCES, mirror_measures, save_features
from treeskel.shape_model import (
    RidgePath,
    choose_alpha,
    cross_validate,
    evaluate,
    feature_matrix,
    fit_predict,
    load_table,
    main,
    mirror_matrix,
    predict_all,
)

FOLDS = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


def test_mirror_matrix_matches_mirror_measures_and_undoes_itself():
    m = mirror_matrix()
    y = np.arange(1.0, len(MEASURES) + 1)
    expected = mirror_measures(dict(zip(MEASURES, y)))
    assert (y @ m).tolist() == [expected[k] for k in MEASURES]
    assert np.allclose(m @ m, np.eye(len(MEASURES)))


def test_ridge_path_matches_scikit_learn():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6)) * [1, 2, 3, 4, 5, 6] + 10
    y = x @ rng.normal(size=6) + rng.normal(size=40)
    z = (x - x.mean(0)) / x.std(0)
    ours = RidgePath(x, y).predict(x[:5], [0.1, 10.0])
    for i, a in enumerate((0.1, 10.0)):
        ref = Ridge(alpha=a).fit(z, y).predict(z[:5])
        assert np.allclose(ours[i, :, 0], ref)


def synthetic(n_per_row=10, rows=12, d=20, signal=True, seed=0):
    """Features whose two halves describe the left and right leaders' tilts.

    In the mirror image, the left half shows the mirrored right leader, whose
    tilt is the negative of the right leader's: its features are ``-b`` (and
    ``-a`` on the right). Features that did not change sign would describe a
    tree that is not the mirror image, and mirroring would then teach the
    model a contradiction.

    Returns ``(x2, y, rows)``: ``x2`` is ``(n, 2, 2d)`` (photo, mirror image), ``y`` the measures.
    """
    rng = np.random.default_rng(seed)
    n = n_per_row * rows
    a, b = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    w = rng.normal(size=d) / np.sqrt(d) * 8
    left_tilt = -22 + (a @ w if signal else 0) + rng.normal(size=n)
    right_tilt = 23 + (b @ w if signal else 0) + rng.normal(size=n)
    trunk_tilt = rng.normal(size=n) * 5
    y = np.full((n, len(MEASURES)), np.nan)
    col = {k: i for i, k in enumerate(MEASURES)}
    y[:, col["left_tilt"]], y[:, col["right_tilt"]], y[:, col["trunk_tilt"]] = left_tilt, right_tilt, trunk_tilt
    y[:, col["v_opening"]] = right_tilt - left_tilt
    y[:, col["lean_v"]] = (left_tilt + right_tilt) / 2
    y[:, col["left_angle"]] = trunk_tilt - left_tilt
    y[:, col["right_angle"]] = right_tilt - trunk_tilt
    y[:, col["lean"]] = y[:, col["lean_v"]] - trunk_tilt
    y[:, col["left_bend"]] = -8 + rng.normal(size=n)
    y[:, col["right_bend"]] = -9 + rng.normal(size=n)
    photo = np.concatenate([a, b], 1)
    mirror = np.concatenate([-b, -a], 1)        # the mirror image shows the mirrored right leader on the left
    return np.stack([photo, mirror], 1), y, np.repeat(np.arange(1, rows + 1), n_per_row)


def col(name):
    return MEASURES.index(name)


def test_cross_validation_learns_a_known_linear_shape():
    x2, y, rows = synthetic()
    cv = cross_validate(x2, y, rows, FOLDS)
    j = col("v_opening")
    good = evaluate(y[:, j], cv["ridge"][:, j], cv["mean"][:, j], cv["fold"])
    assert good["skill"] > 0.8 and good["improved"] > 0.7 and good["wilcoxon_p"] < 1e-6
    assert all(s > 0.6 for s in good["fold_skill"])
    # The trunk's tilt and the bends are noise here: no better than the mean shape.
    for name in ("trunk_tilt", "left_bend"):
        k = col(name)
        assert evaluate(y[:, k], cv["ridge"][:, k], cv["mean"][:, k], cv["fold"])["skill"] < 0.1
    # Every tree is predicted by the one fold that held its row out.
    assert set(cv["fold"].tolist()) == {0, 1, 2, 3}
    assert all(set(rows[cv["fold"] == i]) <= set(FOLDS[i]) for i in range(4))


def test_no_signal_means_a_large_penalty_and_no_skill():
    x2, y, rows = synthetic(signal=False, seed=1)
    alphas = choose_alpha(x2, y, rows)
    assert alphas[col("v_opening")] >= 1e3
    cv = cross_validate(x2, y, rows, FOLDS)
    j = col("v_opening")
    assert evaluate(y[:, j], cv["ridge"][:, j], cv["mean"][:, j], cv["fold"])["skill"] < 0.05


def test_mirroring_gives_mirror_symmetric_predictions():
    x2, y, rows = synthetic(seed=2)
    names = ("left_tilt", "right_tilt")
    yy = y[:, [col(k) for k in names]]
    p = fit_predict(x2[:80], yy[:80], x2[80:], [1.0], names)[0]
    flipped = fit_predict(x2[:80], yy[:80], x2[80:][:, ::-1], [1.0], names)[0]   # the test photos mirrored
    m = mirror_matrix(names)
    centre = yy[:80].mean(0)
    # Predicting the mirror image gives the mirrored prediction, about the training trees' centre.
    assert np.allclose(flipped - centre, (p - centre) @ m, atol=1e-8)


def test_mirroring_keeps_a_real_left_right_difference():
    # The left leaders lean 22 degrees, the right 23: the mean of the photos' targets is kept.
    x2, y, rows = synthetic(signal=False, seed=3)
    p = predict_all(x2, y, x2, [1e7])[0]                     # a huge penalty: only the centre is left
    assert p[:, col("left_tilt")].mean() == pytest.approx(np.nanmean(y[:, col("left_tilt")]), abs=1e-6)
    assert p[:, col("lean_v")].mean() == pytest.approx(np.nanmean(y[:, col("lean_v")]), abs=1e-6)


def test_measures_missing_for_some_trees_are_fitted_where_known():
    x2, y, rows = synthetic(seed=4)
    y[:7, col("left_angle")] = y[:7, col("right_angle")] = y[:7, col("lean")] = np.nan
    cv = cross_validate(x2, y, rows, FOLDS)
    assert np.isfinite(cv["ridge"]).all()                   # every tree still gets a prediction
    j = col("v_opening")
    assert evaluate(y[:, j], cv["ridge"][:, j], cv["mean"][:, j])["n"] == len(y)
    assert evaluate(y[:, col("lean")], cv["ridge"][:, col("lean")], cv["mean"][:, col("lean")])["n"] == len(y) - 7


def test_evaluate_on_a_small_example():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.full(4, 2.5)
    s = evaluate(truth, truth + [0.5, -0.5, 0.5, -0.5], base)
    assert s["mae"] == pytest.approx(0.5) and s["rmse"] == pytest.approx(0.5)
    assert s["skill"] == pytest.approx(1 - 0.25 / 1.25)
    assert s["baseline_mae"] == pytest.approx(1.0) and s["bias"] == pytest.approx(0.0)
    assert evaluate(truth, np.array([np.nan] * 4), base) == {"n": 0}


def test_feature_sets():
    a = {"rgb_global": np.ones((3, 2, 4)), "rgb_cells": np.zeros((3, 2, 4, 4)), "depth_global": np.ones((3, 2, 4))}
    assert feature_matrix(a, "rgb").shape == (3, 2, 20)
    assert feature_matrix(a, "rgb+depth").shape == (3, 2, 24)
    assert feature_matrix(a, "depth").shape == (3, 2, 4)


def write_inputs(tmp_path, x2, y, rows):
    """A shape table and a feature file like ``python -m treeskel.shape`` writes."""
    ids = [f"R{r:02d}N{i:02d}" for i, r in enumerate(rows, 1)]
    with open(tmp_path / "shape_measures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", "note"] + [f"{s}_{k}" for s in SOURCES for k in COLUMNS])
        for i, t in enumerate(ids):
            vals = []
            for s in SOURCES:
                for k in COLUMNS:
                    if k in MEASURES:
                        v = y[i, MEASURES.index(k)] + (1.0 if s == "hand" else 0.0)
                        vals.append("nan" if not np.isfinite(v) else f"{v:.3f}")
                    else:
                        vals.append("15.0" if k == "fork_radius" else "25.0")
            w.writerow([t, int(rows[i]), ""] + vals)
    d = x2.shape[2]
    feats = [{"rgb_global": x2[i, :, : d // 2], "rgb_cells": x2[i, :, d // 2:].reshape(2, 2, -1),
              "depth_global": x2[i, :, : d // 2]} for i in range(len(ids))]
    save_features(tmp_path / "summer_resnet18.npz", ids, feats, {"network": "synthetic", "weights": "none"})
    split = {"test_rows": [13], "fold_val_rows": FOLDS}
    (tmp_path / "splits.json").write_text(json.dumps(split))


def test_main_end_to_end(tmp_path, capsys):
    x2, y, rows = synthetic(seed=5)
    write_inputs(tmp_path, x2, y, rows)
    table = load_table(tmp_path / "shape_measures.csv")
    assert table[0]["row"] == 1 and np.isfinite(table[0]["winter_v_opening"])
    out = tmp_path / "results"
    assert main(["--table", str(tmp_path / "shape_measures.csv"), "--features", str(tmp_path / "summer_resnet18.npz"),
                 "--split", str(tmp_path / "splits.json"), "--out", str(out), "--sets", "rgb",
                 "--plots", str(tmp_path / "figs")]) == 0
    assert (tmp_path / "figs" / "shape_model_skill.png").exists()
    assert (tmp_path / "figs" / "shape_model_scatter.png").exists()
    text = capsys.readouterr().out
    assert "Headline" in text and "hand masks" in text and "ridge rgb no-mirror" in text
    report = json.loads((out / "report.json").read_text())
    sc = report["scores"]["v_opening"]
    assert sc["ridge rgb"]["skill"] > 0.8
    assert sc["hand masks"]["bias"] == pytest.approx(1.0, abs=1e-3)     # the hand values were +1 degree
    with open(out / "predictions.csv") as f:
        header = f.readline().strip().split(",")
        assert sum(1 for _ in f) == len(y) * len(MEASURES)
    assert header[:5] == ["tree", "row", "fold", "measure", "winter"]
    assert {"mean shape", "hand masks", "ridge rgb", "ridge rgb no-mirror"} <= set(header)


def test_main_refuses_test_rows(tmp_path, capsys):
    x2, y, rows = synthetic(seed=6)
    write_inputs(tmp_path, x2, y, rows)
    (tmp_path / "splits.json").write_text(json.dumps({"test_rows": [3], "fold_val_rows": FOLDS}))
    assert main(["--table", str(tmp_path / "shape_measures.csv"), "--features", str(tmp_path / "summer_resnet18.npz"),
                 "--split", str(tmp_path / "splits.json"), "--out", str(tmp_path / "r")]) == 1
    assert "test-row" in capsys.readouterr().out
