"""Tests for treeskel.segment: the U-Net trained on the hand masks (random weights, CPU, small images)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from treeskel.segment import (  # noqa: E402
    SegParams,
    UNet,
    augment_pair,
    compare_with,
    cross_validate_seg,
    example_trees,
    figure_runs,
    fill_failures,
    input_shape,
    known_map,
    load_frames,
    load_masks,
    main,
    mask_scores,
    mask_shape,
    mask_to_photo,
    predict_masks,
    save_masks,
    seg_loss,
    to_mask_size,
    train_seg,
    variant,
    whole_photo,
)
from treeskel.shape import MEASURES  # noqa: E402
from treeskel.shape_cnn import normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --- the network and the loss --------------------------------------------------------

def test_unet_gives_one_logit_per_pixel():
    net = UNet(pretrained=False)
    assert net(torch.zeros(2, 3, 64, 96)).shape == (2, 64, 96)
    rgbd = UNet(pretrained=False, channels=4)
    assert rgbd(torch.zeros(2, 4, 64, 64)).shape == (2, 64, 64)
    with pytest.raises(ValueError):
        net(torch.zeros(1, 3, 48, 64))                      # 48 does not divide by 32
    enc, dec = net.groups()                                 # every parameter in exactly one group
    assert len(enc) + len(dec) == len(list(net.parameters()))
    assert {id(p) for p in enc}.isdisjoint({id(p) for p in dec})


def test_resnet34_encoder_and_bad_names():
    assert UNet("resnet34", pretrained=False)(torch.zeros(2, 3, 64, 64)).shape == (2, 64, 64)
    with pytest.raises(ValueError):
        UNet("vgg16", pretrained=False)
    with pytest.raises(ValueError):
        UNet(pretrained=False, channels=2)


def test_loss_is_small_for_the_right_mask_and_large_for_its_opposite():
    t = torch.zeros(2, 16, 16)
    t[:, 4:12, 6:9] = 1
    right, wrong = (t * 2 - 1) * 20, (1 - 2 * t) * 20        # confident logits
    assert float(seg_loss(right, t)) < 0.01
    assert float(seg_loss(wrong, t)) > 5
    assert float(seg_loss(torch.zeros_like(t), t)) == pytest.approx(
        float(np.log(2)) + float(1 - (2 * 0.5 * 24 + 1) / (0.5 * 256 + 24 + 1)), rel=1e-4)   # p = 0.5 everywhere


# --- augmentation and prediction ---------------------------------------------------

def test_augment_without_changes_is_the_photo_or_its_mirror():
    x, m = torch.rand(3, 3, 32, 32), (torch.rand(3, 32, 32) > 0.7).float()
    plain = SegParams(crop=1.0, rotate=0.0, jitter=0.0)
    gen = torch.Generator().manual_seed(0)
    xa, ma = augment_pair(x, m, gen, plain, flip_p=0.0)
    assert torch.allclose(xa, normalise(x), atol=1e-5) and torch.allclose(ma, m, atol=1e-5)
    xa, ma = augment_pair(x, m, gen, plain, flip_p=1.0)
    assert torch.allclose(xa, normalise(x.flip(-1)), atol=1e-5) and torch.allclose(ma, m.flip(-1), atol=1e-5)


def test_the_mask_stays_on_the_branch_under_crop_rotation_and_mirror():
    # The photo's first channel is the mask itself, so after any resampling they must still agree.
    m = torch.zeros(6, 64, 64)
    m[:, 20:60, 30:36] = 1                                   # a trunk
    m[:, 10:22, 36:50] = 1                                   # a leader
    x = torch.stack([m, torch.zeros_like(m), torch.ones_like(m) * 0.5], 1)
    xa, ma = augment_pair(x, m, torch.Generator().manual_seed(3), SegParams(jitter=0.0))
    first = xa[:, 0] * 0.229 + 0.485                         # undo the normalisation of the first channel
    assert torch.allclose(first, ma, atol=1e-4)
    assert not torch.allclose(ma, m)                         # it did move


def test_the_known_part_stays_with_the_mask_in_a_wide_frame():
    m = torch.zeros(4, 64, 128)
    m[:, 20:60, 60:66] = 1
    known = torch.zeros(64, 128)
    known[:, 16:112] = 1
    x = torch.stack([known.expand(4, 64, 128), m, torch.zeros_like(m)], 1)
    xa, ma, ka = augment_pair(x, m, torch.Generator().manual_seed(5), SegParams(jitter=0.0), known=known)
    assert ka.shape == (4, 64, 128)
    assert torch.allclose(xa[:, 0] * 0.229 + 0.485, ka, atol=1e-4)                 # moved with the photo
    assert torch.allclose(xa[:, 1] * 0.224 + 0.456, ma, atol=1e-4)
    plain = SegParams(crop=1.0, rotate=0.0, jitter=0.0)
    _, m1, k1 = augment_pair(x, m, torch.Generator().manual_seed(0), plain, flip_p=1.0, known=known)
    assert torch.allclose(k1, known.flip(-1).expand(4, 64, 128)) and torch.allclose(m1, m.flip(-1))


def test_a_rotation_keeps_a_disc_round_in_a_wide_frame():
    """The rotation is made in pixels: in a frame twice as wide as high a disc must stay a disc (a rotation made
    in the grid's [-1, 1] coordinates would shear it into an ellipse)."""
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(128.0), indexing="ij")
    disc = ((yy - 32) ** 2 + (xx - 64) ** 2 < 15 ** 2).float().expand(8, 64, 128)
    x = torch.stack([disc] * 3, 1)
    _, md = augment_pair(x, disc, torch.Generator().manual_seed(2), SegParams(crop=1.0, rotate=30.0, jitter=0.0))
    for k in range(8):
        w = md[k]
        cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
        cov = torch.tensor([[(w * (yy - cy) ** 2).sum(), (w * (yy - cy) * (xx - cx)).sum()],
                            [(w * (yy - cy) * (xx - cx)).sum(), (w * (xx - cx) ** 2).sum()]]) / w.sum()
        ev = torch.linalg.eigvalsh(cov)
        assert float(ev[1] / ev[0]) < 1.05                                           # round
        assert float(w.sum()) == pytest.approx(float(disc[0].sum()), rel=0.02)       # same area


def test_the_loss_ignores_what_the_masks_do_not_describe():
    lg = torch.randn(2, 16, 24)
    t = (torch.rand(2, 16, 24) > 0.8).float()
    known = torch.zeros(16, 24)
    known[:, 4:20] = 1
    t2 = t.clone()
    t2[..., :4] = 1 - t2[..., :4]                                                    # differs only where unknown
    assert float(seg_loss(lg, t, known)) == pytest.approx(float(seg_loss(lg, t2, known)), abs=1e-6)
    assert float(seg_loss(lg, t, torch.ones(16, 24))) == pytest.approx(float(seg_loss(lg, t)), abs=1e-6)


def test_colour_jitter_leaves_the_mask_and_depth_alone():
    x, m = torch.rand(4, 4, 32, 32), (torch.rand(4, 32, 32) > 0.5).float()
    xa, ma = augment_pair(x, m, torch.Generator().manual_seed(1), SegParams(crop=1.0, rotate=0.0, jitter=0.5),
                          flip_p=0.0)
    assert torch.allclose(ma, m, atol=1e-5) and torch.allclose(xa[:, 3], normalise(x)[:, 3], atol=1e-5)
    assert not torch.allclose(xa[:, :3], normalise(x)[:, :3], atol=1e-3)


class Ramp(torch.nn.Module):
    """A 'network' whose logits rise from left to right, whatever the photo."""

    def forward(self, x):
        return torch.linspace(-3, 3, x.shape[-1]).expand(x.shape[0], x.shape[-2], x.shape[-1])


def test_prediction_averages_the_photo_and_its_mirror_flipped_back():
    x = (torch.rand(2, 3, 8, 10) * 255).to(torch.uint8)
    p = predict_masks(Ramp(), x)
    ramp = torch.sigmoid(torch.linspace(-3, 3, 10))
    assert np.allclose(p, ((ramp + ramp.flip(0)) / 2).expand(2, 8, 10).numpy(), atol=1e-6)   # symmetric: 0.5


def v_photos(n=16, s=64, seed=0):
    """Noisy photos with a bright V (random opening) and its mask: the branches are what is bright."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:s, :s]
    img = (rng.uniform(0, 0.4, (n, s, s, 3)) * 255).astype(np.uint8)
    masks = np.zeros((n, s, s), np.uint8)
    for i in range(n):
        fork, half = (s * 0.6, s / 2), np.radians(rng.uniform(15, 35))
        m = (np.abs(xx - fork[1]) < 3) & (yy >= fork[0])                       # trunk
        for sign in (-1, 1):
            d = np.array([-np.cos(half), sign * np.sin(half)])                 # (row, col) direction, upwards
            rel = np.stack([yy - fork[0], xx - fork[1]], -1)
            along = rel @ d
            across = np.abs(rel @ np.array([-d[1], d[0]]))
            m |= (along > 0) & (across < 2.5)
        masks[i] = m * 255
        img[i][m] = 230
    return img, masks


def test_training_on_whole_photos_counts_only_the_known_columns(monkeypatch):
    import treeskel.segment as seg

    img, masks = v_photos(n=8, s=64)
    wide = np.zeros((8, 64, 96, 3), np.uint8)
    wide[:, :, 16:80] = img                                    # the V photos, with 16 columns either side
    m = np.zeros((8, 64, 96), np.uint8)
    m[:, :, 16:80] = masks
    seen = []
    real = seg.seg_loss
    monkeypatch.setattr(seg, "seg_loss", lambda lg, t, known=None: seen.append(known) or real(lg, t, known))
    params = SegParams(size=(64, 96), batch=4, jitter=0.0)
    net = UNet(pretrained=False)
    x, mt = torch.from_numpy(wide.transpose(0, 3, 1, 2).copy()), torch.from_numpy(m)
    log = train_seg(net, x, mt, 2, params, seed=0, val=(x, mt))
    assert len(log) == 2 and all(np.isfinite(r["val_loss"]) for r in log)
    assert all(k is not None for k in seen)                   # every loss knew which columns count
    a, b = seg.known_columns(96)
    assert (a, b) == (12, 84) and float(seen[-1][:, :a].abs().sum()) == 0.0
    assert predict_masks(net, x).shape == (8, 64, 96)


def test_training_learns_a_known_mask():
    img, masks = v_photos()
    x = torch.from_numpy(img.transpose(0, 3, 1, 2).copy())
    m = torch.from_numpy(masks)
    torch.manual_seed(0)
    net = UNet(pretrained=False)
    fast = SegParams(size=64, lr_encoder=3e-3, lr_decoder=3e-3, batch=4, jitter=0.0)   # no pretrained start
    log = train_seg(net, x, m, 40, fast, seed=0)
    assert log[-1]["train_loss"] < 0.5 * log[0]["train_loss"]
    pred = predict_masks(net, x) > 0.5
    dice = np.median([mask_scores(p, t > 127)["dice"] for p, t in zip(pred, masks)])
    assert dice > 0.85                                        # 0.96 when written


def test_every_tree_is_predicted_by_its_fold_and_unmasked_trees_never_train(monkeypatch):
    import treeskel.segment as seg

    img, masks = v_photos(n=30, s=32, seed=2)
    rows = np.repeat([1, 2, 3, 4, 5, 6], 5)
    has = np.ones(30, bool)
    has[[0, 7, 12]] = False                                  # three trees without a hand mask
    folds = [[1, 2], [3], [4, 5], [6]]
    sizes = []
    real = seg.train_seg

    def spy(model, x, m, epochs, params, seed, val=None, on_epoch=None):
        sizes.append((len(x), None if val is None else len(val[0])))
        return real(model, x, m, epochs, params, seed, val, on_epoch)

    monkeypatch.setattr(seg, "train_seg", spy)
    cv = cross_validate_seg(img, masks, has, rows, folds, SegParams(size=32, epochs=2, batch=8), "cpu",
                            pretrained=False, progress=lambda m: None)
    for i, held in enumerate(folds):
        assert set(rows[cv["fold"] == i]) == set(held)
    assert np.isfinite(cv["prob"]).all()                     # the unmasked trees are predicted too
    # Fold 1 holds rows 1-2 out: rows 3-6 train, less tree 12 (row 3) = 19; the final run trains on all 19.
    choose, final = sizes[0], sizes[1]
    assert final == (19, None) and choose[0] + choose[1] == 19
    assert all(1 <= e["epochs"] <= 2 for e in cv["epochs"])


# --- scoring -----------------------------------------------------------------------

def test_mask_scores_known_values():
    t = np.zeros((10, 10), bool)
    t[:, :4] = True                                           # 40 px
    p = np.zeros((10, 10), bool)
    p[:, 2:6] = True                                          # 40 px, 20 shared
    s = mask_scores(p, t)
    assert s["iou"] == pytest.approx(20 / 60) and s["dice"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == pytest.approx(0.5)
    e = mask_scores(np.zeros_like(t), np.zeros_like(t))
    assert all(np.isnan(v) for v in e.values())


def v_mask(half_opening=20.0, lean=0.0):
    """A 256x256 V: a vertical trunk and two straight leaders ``half_opening`` degrees either side of
    a bisector leaning ``lean`` degrees (clockwise) from the vertical."""
    yy, xx = np.mgrid[:256, :256]
    fork = np.array([200.0, 128.0])
    m = (np.abs(xx - fork[1]) < 5) & (yy >= fork[0])
    for sign in (-1, 1):
        a = np.radians(lean + sign * half_opening)
        d = np.array([-np.cos(a), np.sin(a)])
        rel = np.stack([yy - fork[0], xx - fork[1]], -1)
        along, across = rel @ d, np.abs(rel @ np.array([-d[1], d[0]]))
        m |= (along > 0) & (along < 190) & (across < 3.5)
    return m


def test_mask_shape_measures_a_known_v():
    m, br = mask_shape(v_mask(20.0, 5.0))
    assert br is not None
    assert m["v_opening"] == pytest.approx(40.0, abs=1.0)
    assert abs(m["left_bend"]) < 1.5 and abs(m["right_bend"]) < 1.5
    assert m["trunk_tilt"] == pytest.approx(0.0, abs=1.0) and m["lean"] == pytest.approx(5.0, abs=1.0)
    empty, none = mask_shape(np.zeros((256, 256), bool))
    assert none is None and all(np.isnan(empty[k]) for k in MEASURES)


def test_small_probability_maps_come_back_to_the_masks_frame():
    m = v_mask()
    from PIL import Image

    small = np.asarray(Image.fromarray(m.astype(np.float32)).resize((64, 64), Image.BILINEAR))
    back = to_mask_size(small)
    assert back.shape == (256, 256) and mask_scores(back, m)["dice"] > 0.9
    assert np.array_equal(to_mask_size(m.astype(np.float32)), m)            # already 256: only thresholded


def test_a_mask_on_the_whole_photo_and_back():
    m = v_mask()
    photo = mask_to_photo(m, (480, 640))
    assert photo.shape == (480, 640) and photo.dtype == np.uint8
    assert not photo[:, :80].any() and not photo[:, 560:].any()                     # nothing said outside
    assert mask_scores(to_mask_size(photo / 255.0, whole=True), m)["dice"] > 0.97
    small = mask_to_photo(m, (64, 96))
    assert small.shape == (64, 96) and not small[:, :11].any() and not small[:, 85:].any()   # columns 12-83 known
    assert mask_scores(to_mask_size(small / 255.0, whole=True), m)["dice"] > 0.8
    params = SegParams(size=(480, 640))
    assert whole_photo(params) and input_shape(params) == (480, 640) and not whole_photo(SegParams())
    k = known_map(params)
    assert k[:, 80:560].all() and not k[:, :80].any() and not k[:, 560:].any() and known_map(SegParams()) is None


def test_masks_without_a_shape_count_as_the_mean():
    pred = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, np.nan]])
    mean = np.array([[9.0, 9.0], [5.0, 6.0], [7.0, 8.0]])
    out = fill_failures(pred, mean)
    assert out.tolist()[:2] == [[1.0, 2.0], [5.0, 6.0]]
    assert out[2, 0] == 3.0 and np.isnan(out[2, 1])        # a fork but no trunk axis keeps its NaN


def test_saved_masks_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    masks = rng.random((3, 256, 256)) > 0.8
    save_masks(tmp_path / "m.npz", ["A", "B", "C"], masks)
    got = load_masks(tmp_path / "m.npz")
    assert list(got) == ["A", "B", "C"] and all(np.array_equal(got[t], masks[i]) for i, t in enumerate("ABC"))


def test_examples_span_the_range_of_dice():
    dice = {f"T{i}": i / 10 for i in range(11)}
    dice["bad"] = float("nan")
    picks = example_trees(dice, 3)
    assert picks == ["T0", "T5", "T10"]                       # worst, median, best
    assert example_trees({}, 3) == []


def test_compare_with_another_model_tree_by_tree():
    seg = {(f"T{i}", "v_opening"): (50.0, p) for i, p in enumerate([51, 49, 52, 50, 55, 50])}
    other = {(f"T{i}", "v_opening"): (50.0, p) for i, p in enumerate([55, 45, 56, 44, 53, 47])}
    c = compare_with(seg, other, ("v_opening",))["v_opening"]
    assert c["n"] == 6 and c["closer"] == pytest.approx(5 / 6)
    assert c["mae_seg"] == pytest.approx(9 / 6) and c["mae_other"] == pytest.approx(28 / 6)
    assert 0 < c["wilcoxon_p"] <= 1


def test_run_names():
    assert variant(SegParams()) == "segment"
    assert variant(SegParams(depth=True)) == "segment_depth"
    assert variant(SegParams(encoder="resnet34")) == "segment_resnet34"
    assert variant(SegParams(size=384)) == "segment_384"
    assert variant(SegParams(size=(480, 640))) == "segment_480x640"
    assert variant(SegParams(size=(480, 640)), lined_up=True) == "segment_lined_up_480x640"


def test_skill_figures_draw_the_runs_that_belong_together():
    assert figure_runs("segment") == ("shape_cnn", "shape_cnn_480x640", "segment")
    assert figure_runs("segment_lined_up_joined")[-2:] == ("segment_lined_up", "segment_lined_up_joined")
    full = figure_runs("segment_lined_up_480x640_joined")
    assert len(full) == 4 and "shape_cnn" not in full and full[2] == "segment_lined_up_480x640_joined"
    assert figure_runs("segment_depth") == ("shape_cnn", "shape_cnn_480x640", "segment", "segment_depth")


# --- the local trees -----------------------------------------------------------------

def test_hand_masks_as_predictions_reproduce_their_measures():
    """The scoring path measures a mask exactly as treeskel.shape measured the hand masks."""
    from treeskel.io import index_dataset, load_mask
    from treeskel.shape_model import load_table

    recs = {t.tree_id: t for t in index_dataset(ROOT / "data" / "tuning").trees}
    table = {r["tree"]: r for r in load_table(ROOT / "data/shape/shape_measures.csv")}
    trees = sorted(set(recs) & set(table))[:4]
    assert trees
    for t in trees:
        hand = load_mask(recs[t].paths[("summer", "mask")])
        m, _ = mask_shape(hand)
        for k in MEASURES:
            if np.isfinite(table[t][f"hand_{k}"]):
                assert m[k] == pytest.approx(table[t][f"hand_{k}"], abs=0.01)
        assert mask_scores(to_mask_size(hand.astype(np.float32)), hand)["dice"] == 1.0


def test_whole_photos_of_the_local_trees(tmp_path):
    from treeskel.io import index_dataset
    from treeskel.summer_gt import load_overlays

    recs = {t.tree_id: t for t in index_dataset(ROOT / "data" / "tuning").trees}
    ids = sorted(recs)[:3]
    overlays = load_overlays(ROOT / "annotations" / "mask_overlay.json")
    img, masks, has = load_frames(recs, ids, (96, 128), progress=lambda m: None, overlays=overlays)
    assert img.shape == (3, 96, 128, 3) and masks.shape == (3, 96, 128) and has.all()
    a, b = (16, 112)                                          # columns 80-559 of 640, at 128 wide
    assert not masks[:, :, :a].any() and not masks[:, :, b:].any() and masks[:, :, a:b].any()
    sq, sq_masks, _ = load_frames(recs, ids, 96, progress=lambda m: None, overlays=overlays)
    for i in range(3):                                        # the same mask as in the square frame, placed
        wide = to_mask_size(masks[i] / 255.0, whole=True)
        assert mask_scores(wide, to_mask_size(sq_masks[i] / 255.0))["dice"] > 0.85


def test_cli_on_the_local_trees(tmp_path, capsys):
    data = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning")]
    results = tmp_path / "results"
    out, figures = results / "segment", tmp_path / "figures"
    assert main([*data, "--random-weights", "--device", "cpu", "--size", "64", "--epochs", "2", "--examples", "2",
                 "--out", str(out), "--figures", str(figures), "--table", str(ROOT / "data/shape/shape_measures.csv"),
                 "--traces", str(ROOT / "data/shape/winter_traces.npz"), "--split", str(ROOT / "splits.json")]) == 0
    text = capsys.readouterr().out
    assert "21 development trees" in text and "21 of them have a hand mask" in text   # no test-row tree
    report = json.loads((out / "report.json").read_text())
    assert set(report["scores"]["v_opening"]) == {"mean shape", "hand masks", "segmentation",
                                                  "segmentation, forks found"}
    assert report["weights"] == "random" and report["n_trees"] == 21 and len(report["epochs"]) == 4
    assert report["scores"]["v_opening"]["segmentation"]["n"] == 21     # no tree dropped: failures are the mean
    for name in ("predictions.csv", "mask_scores.csv", "training_log.csv", "predicted_masks.npz"):
        assert (out / name).exists()
    assert len(load_masks(out / "predicted_masks.npz")) == 21
    for name in ("segment_skill.png", "segment_training.png", "segment_examples.jpg"):
        assert (figures / name).exists()
    assert main(["--replot", str(out), "--figures", str(tmp_path / "again")]) == 0
    assert (tmp_path / "again" / "segment_skill.png").exists()


def test_cli_on_whole_photos_of_the_local_trees(tmp_path, capsys):
    data = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning")]
    results = tmp_path / "results"
    assert main([*data, "--random-weights", "--device", "cpu", "--size", "64", "96", "--epochs", "1", "--examples",
                 "0", "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"), "--out",
                 str(results / "segment_lined_up_64x96"), "--figures", str(tmp_path / "figures"),
                 "--table", str(ROOT / "data/shape/shape_measures.csv"), "--split", str(ROOT / "splits.json")]) == 0
    text = capsys.readouterr().out
    assert "(whole photos) at 64x96" in text and "masks with no fork" in text
    report = json.loads((results / "segment_lined_up_64x96" / "report.json").read_text())
    assert report["params"]["size"] == [64, 96] and report["variant"] == "segment_lined_up_64x96"
    assert report["n_trees"] == 21 and report["masks"]["dice"]["n"] == 21
    masks = load_masks(results / "segment_lined_up_64x96" / "predicted_masks.npz")
    assert len(masks) == 21 and next(iter(masks.values())).shape == (256, 256)   # measured in the masks' frame
    joined = json.loads((results / "segment_lined_up_64x96_joined" / "report.json").read_text())
    assert joined["variant"] == "segment_lined_up_64x96_joined" and joined["n_trees"] == 21
    grown = json.loads((results / "segment_lined_up_64x96_grown" / "report.json").read_text())
    assert grown["grow"]["low"] in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05) and len(grown["grow"]["cldice"]) == 8
    assert grown["masks"]["dice"]["n"] == 21                  # the grown masks against the lined-up hand masks
    assert "Choosing the lower threshold" in text and (results / "segment_lined_up_64x96" / "probabilities.npz").exists()
    # A results folder that holds a run is not overwritten by accident.
    again = [*data, "--random-weights", "--device", "cpu", "--size", "64", "96", "--epochs", "1", "--examples", "0",
             "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"), "--out", str(results / "segment_lined_up_64x96"),
             "--table", str(ROOT / "data/shape/shape_measures.csv"), "--split", str(ROOT / "splits.json")]
    assert main(again) == 1 and "already holds a run" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*data, "--size", "64", "100"])                  # 100 does not divide by 32
