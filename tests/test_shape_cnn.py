"""Tests for treeskel.shape_cnn: fine-tuning a CNN on the winter shape (random weights, CPU, tiny images)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from treeskel.shape import IMAGENET_MEAN, IMAGENET_STD, MEASURES  # noqa: E402
from treeskel.shape_cnn import (  # noqa: E402
    CNNParams,
    augment,
    cross_validate_cnn,
    main,
    make_model,
    masked_mse,
    mirror_columns,
    normalise,
    predict,
    replot,
    target_scale,
    train,
    variant,
)
from treeskel.shape_model import mirror_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAIR = ("left_bend", "right_bend")


def test_mirror_columns_agree_with_the_mirror_matrix():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(5, len(MEASURES)))
    src, sign = mirror_columns()
    assert np.allclose(y[:, src] * sign, y @ mirror_matrix())
    y[2, 1] = np.nan                                       # one unknown measure stays one unknown measure
    mirrored = y[:, src] * sign
    assert np.isnan(mirrored).sum() == 1 and np.isfinite(mirrored[2]).sum() == len(MEASURES) - 1


def test_standardising_commutes_with_mirroring():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(40, len(MEASURES))) * np.arange(1, len(MEASURES) + 1) + 5
    y[3, 0] = np.nan
    mu, s = target_scale(y)
    src, sign = mirror_columns()
    c = y - mu
    assert np.allclose(((c[:, src] * sign) / s), ((c / s)[:, src] * sign), equal_nan=True)
    # Mirror partners share one spread.
    assert s[MEASURES.index("left_bend")] == s[MEASURES.index("right_bend")]


def test_masked_mse_skips_unknowns():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, float("nan")], [1.0, 4.0]])
    assert float(masked_mse(pred, target)) == pytest.approx(4.0 / 3)


def test_augment_without_crop_or_jitter_is_the_photo_or_its_mirror():
    x = torch.rand(4, 3, 12, 16)
    gen = torch.Generator().manual_seed(0)
    plain = CNNParams(crop=1.0, jitter=0.0)
    out, flipped = augment(x, gen, plain, flip_p=0.0)
    assert not flipped.any() and torch.allclose(out, normalise(x), atol=1e-5)
    out, flipped = augment(x, gen, plain, flip_p=1.0)
    assert flipped.all() and torch.allclose(out, normalise(x.flip(-1)), atol=1e-5)
    out, _ = augment(x, gen, CNNParams())                     # cropped and jittered: same size, in range
    lo = ((0 - IMAGENET_MEAN) / IMAGENET_STD).min()
    hi = ((1 - IMAGENET_MEAN) / IMAGENET_STD).max()
    assert out.shape == x.shape and float(out.min()) >= lo - 1e-5 and float(out.max()) <= hi + 1e-5


class HalfMeans(torch.nn.Module):
    """A 'network' that outputs the mean of the left half of the image, and nothing for the right."""

    def forward(self, x):
        w = x.shape[-1] // 2
        return torch.stack([x[..., :w].mean(dim=(1, 2, 3)), torch.zeros(len(x))], 1)


def test_predict_averages_the_photo_and_its_mirror_image():
    x = (torch.rand(3, 3, 8, 10) * 255).to(torch.uint8)
    src, sign = mirror_columns(PAIR)
    z = predict(HalfMeans(), x, (torch.as_tensor(src), torch.as_tensor(sign, dtype=torch.float32)))
    xn = normalise(x.float() / 255)
    left, right = xn[..., :5].mean(dim=(1, 2, 3)), xn[..., 5:].mean(dim=(1, 2, 3))
    # The mirror image's left half is the photo's right half, and its "left" output becomes the right one.
    assert np.allclose(z, torch.stack([left / 2, right / 2], 1).numpy(), atol=1e-5)


def bright_halves(n=24, h=32, w=32, seed=0):
    """Photos whose left and right halves have a brightness that is the target (a mirror swaps them)."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0.2, 0.8, n), rng.uniform(0.2, 0.8, n)
    img = np.zeros((n, h, w, 3), np.uint8)
    img[:, :, : w // 2] = (a * 255)[:, None, None, None]
    img[:, :, w // 2:] = (b * 255)[:, None, None, None]
    return img, np.stack([a, b], 1)


def test_training_learns_a_known_target():
    img, y = bright_halves()
    mu, s = target_scale(y, PAIR)
    x = torch.from_numpy(img.transpose(0, 3, 1, 2).copy())
    src, sign = mirror_columns(PAIR)
    mirror = (torch.as_tensor(src), torch.as_tensor(sign, dtype=torch.float32))
    torch.manual_seed(0)
    model = make_model(2, pretrained=False)
    fast = CNNParams(lr_backbone=1e-3, lr_head=1e-2, crop=1.0, jitter=0.0, batch=8)   # brightness is the signal
    log = train(model, x, (y - mu) / s, 15, fast, mirror, seed=0)
    assert log[-1]["train_loss"] < 0.5 * log[0]["train_loss"]
    z = predict(model, x, mirror)
    assert np.corrcoef(z[:, 0], ((y - mu) / s)[:, 0])[0, 1] > 0.8


def test_every_tree_is_predicted_by_the_fold_that_holds_its_row_out():
    img, y2 = bright_halves(n=30, h=24, w=24, seed=1)
    y = np.full((30, len(MEASURES)), np.nan)
    y[:, MEASURES.index("left_bend")], y[:, MEASURES.index("right_bend")] = y2[:, 0], y2[:, 1]
    y[:, MEASURES.index("v_opening")] = 50 + 10 * y2[:, 0]
    rows = np.repeat([1, 2, 3, 4, 5, 6], 5)
    folds = [[1, 2], [3], [4, 5], [6]]
    tiny = CNNParams(size=(24, 24), epochs=2, batch=8)
    cv = cross_validate_cnn(img, y, rows, folds, tiny, "cpu", pretrained=False, progress=lambda m: None)
    for i, held in enumerate(folds):
        assert set(rows[cv["fold"] == i]) == set(held)
    assert all(1 <= e["epochs"] <= 2 and e["inner_rows"] for e in cv["epochs"])
    j = MEASURES.index("v_opening")
    assert np.isfinite(cv["cnn"][:, j]).all()
    held = rows != 1                                       # the mean shape is each fold's training mean
    assert cv["mean"][0, j] == pytest.approx(np.mean(y[~np.isin(rows, [1, 2]), j]))
    assert held.any()


def test_cli_on_the_local_trees(tmp_path, capsys):
    data = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning")]
    out = tmp_path / "cnn"
    assert main([*data, "--random-weights", "--device", "cpu", "--size", "48", "64", "--epochs", "2",
                 "--out", str(out), "--table", str(ROOT / "data/shape/shape_measures.csv"),
                 "--split", str(ROOT / "splits.json"), "--ridge", str(ROOT / "docs/results/shape/predictions.csv")]) == 0
    text = capsys.readouterr().out
    assert "21 development trees" in text and "cnn fine-tuned" in text
    report = json.loads((out / "report.json").read_text())
    assert set(report["scores"]["v_opening"]) == {"mean shape", "hand masks", "cnn fine-tuned", "ridge rgb"}
    assert report["weights"] == "random" and len(report["epochs"]) == 4       # fold 3's rows are not in the sample
    assert (out / "predictions.csv").exists() and (out / "training_log.csv").exists()


# --- the depth channel, the run names and the figures ------------------------------

def test_depth_channel_starts_as_the_mean_of_the_colour_weights():
    torch.manual_seed(0)
    rgb = make_model(3, pretrained=False)
    torch.manual_seed(0)
    rgbd = make_model(3, pretrained=False, channels=4)
    w3, w4 = rgb.conv1.weight, rgbd.conv1.weight
    assert w4.shape[1] == 4 and torch.allclose(w4[:, :3], w3) and torch.allclose(w4[:, 3], w3.mean(dim=1))
    assert rgbd(torch.zeros(2, 4, 32, 32)).shape == (2, 3)
    with pytest.raises(ValueError):
        make_model(3, pretrained=False, channels=2)


def test_colour_jitter_leaves_the_depth_alone():
    x = torch.rand(4, 4, 12, 16)
    out, flipped = augment(x, torch.Generator().manual_seed(1), CNNParams(crop=1.0, jitter=0.5), flip_p=0.0)
    assert not flipped.any()
    assert torch.allclose(out[:, 3], normalise(x)[:, 3], atol=1e-5)          # depth: only normalised
    assert not torch.allclose(out[:, :3], normalise(x)[:, :3], atol=1e-3)    # colour: jittered
    out, flipped = augment(x, torch.Generator().manual_seed(1), CNNParams(crop=1.0, jitter=0.0), flip_p=1.0)
    assert torch.allclose(out[:, 3], normalise(x.flip(-1))[:, 3], atol=1e-5)  # mirrored with the photo


def test_run_names():
    assert variant(CNNParams()) == "shape_cnn"
    assert variant(CNNParams(depth=True)) == "shape_cnn_depth"
    assert variant(CNNParams(size=(480, 640))) == "shape_cnn_480x640"
    assert variant(CNNParams(size=(480, 640), depth=True)) == "shape_cnn_depth_480x640"


def test_cli_with_depth_and_the_figures(tmp_path, capsys):
    data = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning")]
    out = tmp_path / "shape_cnn_depth"
    assert main([*data, "--depth", "--random-weights", "--device", "cpu", "--size", "48", "64", "--epochs", "2",
                 "--out", str(out), "--table", str(ROOT / "data/shape/shape_measures.csv"),
                 "--split", str(ROOT / "splits.json"), "--ridge", str(ROOT / "docs/results/shape/predictions.csv")]) == 0
    assert "photos and depth" in capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    assert report["variant"] == "shape_cnn_depth_48x64" and report["params"]["depth"] is True   # the size is in it
    figures = tmp_path / "figures"
    assert replot([out], figures) == []
    assert (figures / "shape_cnn_skill.png").exists() and (figures / "shape_cnn_depth_training.png").exists()
    assert main(["--replot", str(out), "--figures", str(figures)]) == 0
