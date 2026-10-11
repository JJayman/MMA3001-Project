"""Tests for the centre-line loss of treeskel.segment (session 16): the soft skeleton, soft clDice, the recipe's
loss and how a run is named and trained with it."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from treeskel.segment import (  # noqa: E402
    CL_ITERS,
    CLDICE_WEIGHT,
    SegParams,
    UNet,
    _loss_options,
    cl_dice,
    figure_runs,
    main,
    seg_loss,
    soft_cldice_loss,
    soft_skeleton,
    train_seg,
    variant,
)

ROOT = Path(__file__).resolve().parents[1]


def bar(half_width: int, s: int = 96) -> np.ndarray:
    """A vertical bar ``2 half_width + 1`` px wide, rows 10-85: its distance transform peaks at ``half_width + 1``."""
    img = np.zeros((s, s), np.float32)
    img[10:86, s // 2 - half_width:s // 2 + half_width + 1] = 1
    return img


def skel(img: np.ndarray, iters: int = CL_ITERS) -> np.ndarray:
    return soft_skeleton(torch.from_numpy(img)[None, None], iters)[0, 0].numpy()


def v_mask(cut_left: float | None = None) -> np.ndarray:
    """A 128x128 V: a wide trunk (half-width 10) and two thin leaders (3); ``cut_left`` keeps that fraction of
    the left one."""
    from scipy import ndimage as ndi
    from skimage.draw import line

    m = np.zeros((128, 128), bool)
    m[80:, 54:75] = True
    for side, end in (("left", (5, 20)), ("right", (5, 108))):
        rr, cc = line(80, 64, *end)
        if side == "left" and cut_left is not None:
            rr, cc = rr[:int(len(rr) * cut_left)], cc[:int(len(cc) * cut_left)]
        path = np.zeros_like(m)
        path[rr, cc] = True
        m |= ndi.distance_transform_edt(~path) <= 3
    return m.astype(np.float32)


# --- the soft skeleton ------------------------------------------------------------------

@pytest.mark.parametrize("half_width", [2, 4, 8, 12, 16])
def test_k_erosions_find_the_centre_of_a_branch_k_px_wide_each_side(half_width):
    """The claim behind CL_ITERS: a branch whose distance transform peaks at d needs d - 1 erosions."""
    img = bar(half_width)
    assert skel(img, half_width)[48, 48] == pytest.approx(1.0)
    assert skel(img, half_width - 1)[48].sum() == pytest.approx(0.0)        # one too few: no centre line at all


def test_the_soft_skeleton_of_a_mask_is_its_centre_line():
    from skimage.morphology import skeletonize

    img = v_mask()
    s = skel(img) > 0.5
    hard = skeletonize(img > 0.5)
    assert s.sum() < 0.15 * img.sum()                                         # thin, not the mask
    from scipy import ndimage as ndi
    near = ndi.distance_transform_edt(~hard)
    assert np.median(near[s]) <= 1.0                                          # on the centre line
    assert (ndi.distance_transform_edt(~s)[hard] <= 2).mean() > 0.9          # and along all of it


def test_cl_iters_covers_the_widest_trunks():
    assert CL_ITERS >= 19                                                     # 17.0 px at the largest zoom, 1.12
    assert skel(bar(CL_ITERS))[48, 48] == pytest.approx(1.0)


# --- soft clDice --------------------------------------------------------------------------

def test_soft_cldice_is_zero_for_the_mask_itself_and_matches_the_hard_cldice():
    t = torch.from_numpy(v_mask())[None]
    assert float(soft_cldice_loss(t, t)[0]) == pytest.approx(0.0, abs=1e-6)
    p = v_mask(0.5)
    soft = 1 - float(soft_cldice_loss(torch.from_numpy(p)[None], t)[0])
    assert soft == pytest.approx(cl_dice(p > 0.5, v_mask() > 0.5), abs=0.03)


def test_a_short_leader_costs_more_in_cldice_than_in_dice():
    """The reason for the loss: a leader that stops halfway is cheap in pixels (the wide trunk dominates) but
    loses a large share of the centre line."""
    t = torch.from_numpy(v_mask())[None]
    p = torch.from_numpy(v_mask(0.5))[None]
    dice = float(1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))
    cl = float(soft_cldice_loss(p, t)[0])
    assert cl > 1.4 * dice                                                    # 1.6 times when written


def test_soft_cldice_has_gradients_everywhere_it_should():
    logits = torch.zeros(2, 64, 64, requires_grad=True)
    t = torch.from_numpy(np.stack([bar(3, 64), bar(6, 64)]))
    soft_cldice_loss(torch.sigmoid(logits), t).sum().backward()
    g = logits.grad
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0


# --- the recipe's loss --------------------------------------------------------------------

def test_without_the_weight_the_loss_is_exactly_the_named_recipes():
    t = torch.from_numpy(np.stack([v_mask()[::2, ::2], v_mask(cut_left=0.4)[::2, ::2]]))
    lg = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(0))
    assert torch.equal(seg_loss(lg, t), seg_loss(lg, t, cldice=0.0))
    assert _loss_options(SegParams()) == {}                                   # train_seg's calls are unchanged
    assert _loss_options(SegParams(cldice=0.5)) == {"cldice": 0.5, "iters": CL_ITERS}


def test_with_the_weight_the_loss_is_bce_plus_shared_dice_and_cldice():
    import torch.nn.functional as F

    t = torch.from_numpy(np.stack([v_mask()[::2, ::2], v_mask(cut_left=0.4)[::2, ::2]]))
    lg = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(1))
    p = torch.sigmoid(lg)
    bce = F.binary_cross_entropy_with_logits(lg, t)
    dice = (1 - (2 * (p * t).sum(dim=(1, 2)) + 1) / (p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) + 1)).mean()
    cl = soft_cldice_loss(p, t).mean()
    a = CLDICE_WEIGHT
    assert float(seg_loss(lg, t, cldice=a)) == pytest.approx(float(bce + (1 - a) * dice + a * cl), rel=1e-6)
    known = torch.ones(64, 64)
    assert float(seg_loss(lg, t, known, cldice=a)) == pytest.approx(float(seg_loss(lg, t, cldice=a)), rel=1e-6)


def test_training_and_choosing_the_epochs_use_the_recipes_loss(monkeypatch):
    import treeskel.segment as seg

    calls = []
    real = seg.seg_loss
    monkeypatch.setattr(seg, "seg_loss", lambda *a, **k: calls.append(k) or real(*a, **k))
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 255, (4, 3, 32, 32), dtype=np.uint8))
    m = torch.from_numpy(np.stack([(bar(3, 32) * 255).astype(np.uint8)] * 4))
    torch.manual_seed(0)
    train_seg(UNet(pretrained=False), x, m, 1, SegParams(size=32, batch=2, cldice=0.5), seed=0, val=(x[:2], m[:2]))
    assert len(calls) == 3 and all(k == {"cldice": 0.5, "iters": CL_ITERS} for k in calls)   # 2 steps + held out
    calls.clear()
    train_seg(UNet(pretrained=False), x, m, 1, SegParams(size=32, batch=2), seed=0, val=(x[:2], m[:2]))
    assert len(calls) == 3 and all(k == {} for k in calls)


def test_run_names_with_the_loss_and_the_seed():
    assert variant(SegParams(cldice=0.5), lined_up=True) == "segment_lined_up_cldice"
    assert variant(SegParams(cldice=0.3), lined_up=True) == "segment_lined_up_cldice0.3"
    assert variant(SegParams(seed=1), lined_up=True) == "segment_lined_up_seed1"
    assert variant(SegParams(cldice=0.5, seed=2), lined_up=True) == "segment_lined_up_cldice_seed2"
    assert variant(SegParams(size=(480, 640), cldice=0.5), lined_up=True) == "segment_lined_up_480x640_cldice"
    assert variant(SegParams(), lined_up=True) == "segment_lined_up"                 # the named run's name stays
    assert figure_runs("segment_lined_up_cldice_joined") == ("segment_lined_up_joined",
                                                            "segment_lined_up_rerun_joined",
                                                            "segment_lined_up_cldice_joined")


def test_cli_trains_with_the_centre_line_loss_and_skips_growing(tmp_path, capsys):
    data = [str(ROOT / "data" / "sample"), str(ROOT / "data" / "tuning")]
    results = tmp_path / "results"
    out = results / "segment_lined_up_cldice_seed1"
    assert main([*data, "--random-weights", "--device", "cpu", "--size", "64", "--epochs", "1", "--examples", "0",
                 "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"), "--cldice", "--seed", "1",
                 "--no-grow", "--out", str(out), "--figures", str(tmp_path / "figures"),
                 "--table", str(ROOT / "data/shape/shape_measures.csv"), "--split", str(ROOT / "splits.json")]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["params"]["cldice"] == CLDICE_WEIGHT and report["params"]["cl_iters"] == CL_ITERS
    assert report["params"]["seed"] == 1 and report["recipe"] == "segment_lined_up_64_cldice_seed1"   # 64 px here
    assert (out / "probabilities.npz").exists() and (results / "segment_lined_up_cldice_seed1_joined").exists()
    assert not (results / "segment_lined_up_cldice_seed1_grown").exists()
    assert "Choosing the lower threshold" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*data, "--cldice", "1.5"])                                      # a weight is between 0 and 1
