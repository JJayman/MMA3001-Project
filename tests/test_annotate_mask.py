"""Tests for the mask line-up tool (treeskel.annotate_mask) and its use by treeskel.segment."""

import json
from pathlib import Path

import numpy as np
import pytest

from treeskel.align import apply_transform
from treeskel.annotate_mask import (
    HELP,
    MaskAnnotator,
    apply_mask_overlays,
    mask_fork,
    mask_transform,
    move_mask,
)
from treeskel.io import index_dataset, load_mask
from treeskel.summer_gt import Overlay, load_overlays, save_overlays

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "data" / "tuning"


def box(r0, r1, c0, c1):
    m = np.zeros((256, 256), bool)
    m[r0:r1, c0:c1] = True
    return m


# --- moving a mask -------------------------------------------------------------------

def test_a_photo_shift_is_the_same_shift_in_the_mask_frame():
    m = mask_transform(Overlay(dx=30.0, dy=-15.0))
    # 30 photo px = 16 mask px across, -15 photo px = -8 mask px down, anywhere in the frame
    assert np.allclose(apply_transform(m, [(10, 20), (200, 100)]), [(26, 12), (216, 92)])
    assert np.allclose(mask_transform(Overlay()), np.eye(3))


def test_move_mask_shifts_and_leaves_no_move_alone():
    m = box(100, 120, 50, 60)
    assert np.array_equal(move_mask(m, Overlay()), m)
    right_down = move_mask(m, Overlay(dx=30.0, dy=15.0))              # 16 and 8 mask px
    assert np.array_equal(right_down, box(108, 128, 66, 76))
    assert right_down.sum() == m.sum()


def test_a_trunk_moved_up_still_reaches_the_bottom_edge():
    trunk = box(150, 256, 120, 136)
    up = move_mask(trunk, Overlay(dy=-30.0))                          # 16 mask px up
    assert up[:134].sum() == 0 and up[134:, 120:136].all()            # the bottom rows repeat the edge
    assert up.sum() == (256 - 134) * 16


def test_a_scale_is_about_the_photo_centre_which_is_the_masks_centre():
    m = mask_transform(Overlay(scale=1.2))
    centre = ((320 - 80) * 256 / 480, 240 * 256 / 480)                # photo (320, 240) in the mask frame
    assert np.allclose(centre, (128, 128))
    assert np.allclose(apply_transform(m, [centre])[0], centre)
    assert np.allclose(apply_transform(m, [(138, 128)])[0], (140, 128))


def test_overlays_move_skip_or_leave_masks():
    masks = {"A": box(10, 20, 10, 20), "B": box(10, 20, 10, 20), "C": box(10, 20, 10, 20)}
    got = apply_mask_overlays(masks, {"A": Overlay(dx=15.0), "B": Overlay(skip=True)})
    assert got["A"][10:20, 18:28].all() and got["B"] is None and np.array_equal(got["C"], masks["C"])


def test_mask_fork_on_a_real_mask():
    rec = next(t for t in index_dataset(TUNING).trees if t.has_mask)
    x, y = mask_fork(load_mask(rec.paths[("summer", "mask")]))
    assert 80 <= x <= 560 and 0 <= y < 480
    assert np.allclose(mask_fork(np.zeros((256, 256), bool)), (320, 240))


# --- the annotator -------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def make_ann(tmp_path):
    trees = [t for t in index_dataset(TUNING).trees if t.tree_id in ("R01N04", "R01N05", "R01N08", "R07N09")]
    clock = Clock()

    def make(redo=False):
        ann = MaskAnnotator(trees, tmp_path / "mask_overlay.json", redo=redo, clock=clock)
        ann.clock_ = clock
        return ann
    return make


def test_line_up_accept_and_start_the_next_tree_from_the_row(make_ann, tmp_path):
    ann = make_ann()
    assert ann.tree.tree_id == "R01N04" and (ann.ov.dx, ann.ov.dy, ann.ov.scale) == (0, 0, 1)
    ann.key("shift+right")
    ann.key("up")
    ann.clock_.t = 7.0
    ann.key("enter")
    saved = load_overlays(tmp_path / "mask_overlay.json")["R01N04"]
    assert (saved.dx, saved.dy, saved.seconds, saved.skip) == (10.0, -1.0, 7.0, False)
    # Row 1 has one tree lined up: fewer than two, so the next starts from all trees' median (the same one).
    assert ann.tree.tree_id == "R01N05" and (ann.ov.dx, ann.ov.dy) == (10.0, -1.0)
    ann.key("s")                                                      # skip R01N05
    assert load_overlays(tmp_path / "mask_overlay.json")["R01N05"].skip
    assert ann.tree.tree_id == "R01N08"
    ann.key("left")
    ann.key("u")                                                      # undo
    assert ann.ov.dx == 10.0
    ann.key("f")
    ann.key("enter")
    assert load_overlays(tmp_path / "mask_overlay.json")["R01N08"].mask_flag
    ann.key("q")
    assert ann.done
    again = make_ann()                                                # done trees are not shown again
    assert [t.tree_id for t in again.trees] == ["R07N09"]
    assert len(make_ann(redo=True).trees) == 4


def test_the_picture_shows_the_moved_mask(make_ann):
    ann = make_ann()
    magenta = lambda pic: (pic == (255, 0, 255)).all(-1)              # noqa: E731
    pic = ann.composite()
    assert pic.shape == (480, 640, 3) and magenta(pic).any()
    cols = np.nonzero(magenta(pic).any(0))[0]
    ann.shift(20, 0)
    cols2 = np.nonzero(magenta(ann.composite()).any(0))[0]
    assert cols2.min() == pytest.approx(cols.min() + 20, abs=2)
    ann.key("m")
    assert not magenta(ann.composite()).any()                         # hidden: the photo alone
    ann.key("m")
    ann.key(" ")
    assert ann.filled and magenta(ann.composite()).sum() == 0          # filled: blended, not pure magenta


def test_scaling_keeps_the_fork_where_it_is(make_ann):
    ann = make_ann()
    before = apply_transform(ann.ov.matrix(), [ann.pin])[0]
    ann.key("=")
    ann.key("=")
    assert ann.ov.scale == pytest.approx(1.005 ** 2)
    assert np.allclose(apply_transform(ann.ov.matrix(), [ann.pin])[0], before)
    ann.rescale(1.1, about=(300.0, 300.0))
    assert not np.allclose(apply_transform(ann.ov.matrix(), [ann.pin])[0], before)   # scrolling pins the cursor
    ann.key("r")
    assert (ann.ov.dx, ann.ov.dy, ann.ov.scale) == pytest.approx((0, 0, 1))


def test_window_drives_the_mask_annotator(make_ann):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

    from treeskel.annotate_overlay import run_gui

    plt.switch_backend("Agg")
    ann = make_ann()
    fig = run_gui(ann, show=False, help_text=HELP)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert "space: filled" in fig._suptitle.get_text()

    def mouse(name, col, row, button=MouseButton.LEFT, **kw):
        x, y = ax.transData.transform((col, row))
        fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=button, **kw))

    mouse("button_press_event", 300, 300)
    mouse("motion_notify_event", 290, 305, button=None)
    mouse("button_release_event", 290, 305)
    assert (ann.ov.dx, ann.ov.dy) == pytest.approx((-10.0, 5.0))
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, "enter"))
    assert ann.tree.tree_id == "R01N05"
    plt.close(fig)


# --- the segmentation on lined-up masks -------------------------------------------------

def test_segmentation_uses_only_lined_up_masks_moved(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from treeskel.segment import hand_mask, main, variant, SegParams

    recs = {t.tree_id: t for t in index_dataset(TUNING).trees}
    ov = {"R01N04": Overlay(dx=15.0), "R01N05": Overlay(skip=True)}
    drawn = load_mask(recs["R01N04"].paths[("summer", "mask")])
    assert np.array_equal(hand_mask(recs["R01N04"], ov), move_mask(drawn, Overlay(dx=15.0)))
    assert hand_mask(recs["R01N05"], ov) is None and hand_mask(recs["R01N08"], ov) is None   # skipped, not done
    assert np.array_equal(hand_mask(recs["R01N08"]), load_mask(recs["R01N08"].paths[("summer", "mask")]))
    assert variant(SegParams(), lined_up=True) == "segment_lined_up"

    # Line up (unchanged) all local development trees but one skipped: 20 train, the skipped one is still predicted.
    table = {r.split(",")[0] for r in (ROOT / "data/shape/shape_measures.csv").read_text().splitlines()[1:]}
    local = sorted(t.tree_id for d in ("sample", "tuning") for t in index_dataset(ROOT / "data" / d).trees
                   if t.tree_id in table)
    overlays = {t: Overlay(dx=2.0) for t in local}
    overlays[local[0]] = Overlay(skip=True)
    path = tmp_path / "mask_overlay.json"
    save_overlays(path, overlays)
    results = tmp_path / "results"
    assert main([str(ROOT / "data" / "sample"), str(TUNING), "--mask-overlay", str(path), "--random-weights",
                 "--device", "cpu", "--size", "64", "--epochs", "1", "--examples", "1",
                 "--out", str(results / "segment_lined_up"), "--figures", str(tmp_path / "fig"),
                 "--table", str(ROOT / "data/shape/shape_measures.csv"),
                 "--traces", str(ROOT / "data/shape/winter_traces.npz"), "--split", str(ROOT / "splits.json")]) == 0
    report = json.loads((results / "segment_lined_up" / "report.json").read_text())
    assert report["mask_overlay"] == str(path) and report["recipe"] == "segment_lined_up_64"    # the size is in it
    assert report["variant"] == "segment_lined_up"                                              # its folder
    assert report["n_trees"] == 21 and report["n_masks"] == 20
    assert (tmp_path / "fig" / "segment_lined_up_skill.png").exists()
