"""Tests for the overlay tool (treeskel.annotate_overlay) and its transform (summer_gt.Overlay)."""

import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from treeskel.align import apply_transform, describe
from treeskel.annotate_overlay import OverlayAnnotator, median_overlay, run_gui
from treeskel.io import index_dataset
from treeskel.summer_gt import (
    Overlay,
    SeasonPairs,
    interior_pairs,
    load_overlays,
    mask_displacement,
    pair_distances,
    save_overlays,
)

CLICKS = Path(__file__).resolve().parents[1] / "annotations" / "winter_clicks.json"
CENTRE = (320.0, 240.0)


def moved(m, pt):
    return apply_transform(m, [pt])[0]


# --- the transform ------------------------------------------------------------

def test_overlay_parameters_mean_what_they_say():
    assert np.allclose(Overlay().matrix(), np.eye(3))
    m = Overlay(scale=1.1).matrix()
    assert np.allclose(moved(m, CENTRE), CENTRE) and np.allclose(moved(m, (420, 240)), (430, 240))
    m = Overlay(roll=3.0).matrix()
    d = describe(m, CENTRE)
    assert d["rotation"] == pytest.approx(3.0) and d["scale"] == pytest.approx(1.0)
    assert np.allclose(moved(m, CENTRE), CENTRE)
    assert np.allclose(Overlay(roll=2, scale=0.9, dx=5).matrix()[2], [0, 0, 1])   # no tilt: a similarity


@pytest.mark.parametrize("param", ["pitch", "yaw"])
def test_tilts_add_perspective_but_keep_the_centre(param):
    m = Overlay(**{param: 4.0}).matrix()
    assert np.allclose(moved(m, CENTRE), CENTRE)                 # tilting does not move the picture
    assert abs(m[2, 0]) + abs(m[2, 1]) > 1e-5                     # it adds perspective
    # A tilt makes one end of the photo wider (pitch) or taller (yaw) than the other.
    if param == "pitch":
        top = np.diff(apply_transform(m, [(0, 0), (639, 0)])[:, 0])[0]
        bottom = np.diff(apply_transform(m, [(0, 479), (639, 479)])[:, 0])[0]
    else:
        top = np.diff(apply_transform(m, [(0, 0), (0, 479)])[:, 1])[0]
        bottom = np.diff(apply_transform(m, [(639, 0), (639, 479)])[:, 1])[0]
    assert abs(top - bottom) > 5


def test_overlay_file_round_trip(tmp_path):
    o = {"R01N01": Overlay(dx=1.23456, dy=-2, scale=1.02, roll=0.5, pitch=1, yaw=-1, mask_flag=True, seconds=12.3),
         "R01N02": Overlay(skip=True)}
    save_overlays(tmp_path / "o.json", o)
    back = load_overlays(tmp_path / "o.json")
    assert back["R01N01"].dx == 1.2346 and back["R01N01"].mask_flag and back["R01N02"].skip
    assert np.allclose(back["R01N01"].matrix(), o["R01N01"].matrix(), atol=1e-3)
    assert Overlay.from_json({"dx": 3, "unknown": 1}).dx == 3
    assert load_overlays(tmp_path / "none.json") == {}


def test_median_overlay():
    m = median_overlay([Overlay(dx=1, scale=1.0), Overlay(dx=5, scale=1.1), Overlay(dx=2, scale=0.9)])
    assert (m.dx, m.scale) == (2, 1.0)
    assert median_overlay([]) == Overlay()


def test_checks_against_pairs_and_between_alignments():
    p = SeasonPairs(winter=[(5.0, 100.0), (200.0, 300.0), (300.0, 400.0), (478.0, 50.0)],
                    summer=[(5.0, 110.0), (190.0, 312.0), (290.0, 410.0), (470.0, 60.0)])
    inner = interior_pairs(p)
    assert inner.winter == [(200.0, 300.0), (300.0, 400.0)]      # points at the top and bottom edges dropped
    d = pair_distances(Overlay(dx=10, dy=-10).matrix(), inner)
    assert np.allclose(d, [2.0, 0.0])
    mask = np.zeros((480, 640), bool)
    mask[200:300, 300:320] = True
    assert mask_displacement(Overlay().matrix(), Overlay(dx=3, dy=4).matrix(), mask) == pytest.approx(5.0)


# --- the tool -----------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def ann_factory(sample_dir, tmp_path):
    """Annotator on the sample's development trees, with a stand-in winter mask and a fake clock."""
    mask = np.zeros((480, 640), bool)
    mask[300:480, 300:330] = True
    gt = tmp_path / "gt"
    gt.mkdir()
    trees = [t for t in index_dataset(sample_dir).trees if t.tree_id in ("R01N01", "R01N02", "R15N05")]
    for t in trees:
        Image.fromarray(mask.astype(np.uint8) * 255).save(gt / f"{t.tree_id}_winter_gt.png")
    clock = Clock()

    def make(redo=False):
        ann = OverlayAnnotator(trees, tmp_path / "overlay.json", redo=redo, clicks_path=CLICKS, gt_dir=gt, clock=clock)
        ann.clock_ = clock
        return ann
    return make


def test_keys_shift_roll_scale_and_tilt_keeping_the_fork(ann_factory):
    ann = ann_factory()
    assert ann.tree.tree_id == "R01N01" and ann.ov == Overlay()
    ann.key("right")
    ann.key("shift+up")
    assert (ann.ov.dx, ann.ov.dy) == (1.0, -10.0)
    fork = moved(ann.matrix(), ann.pin)
    for k in ("]", "}", "=", "-", "-", "i", "l"):
        ann.key(k)
        assert np.allclose(moved(ann.matrix(), ann.pin), fork, atol=1e-6), k   # the fork stays put
    assert ann.ov.roll == pytest.approx(1.2) and ann.ov.scale == pytest.approx(1 / 1.005)
    assert (ann.ov.pitch, ann.ov.yaw) == (0.5, 0.5)
    ann.key("u")
    assert ann.ov.yaw == 0.0
    ann.key("r")
    assert ann.ov == Overlay()
    ann.key("u")                                                     # a reset can be undone
    assert ann.ov.pitch == 0.5


def test_scroll_scale_keeps_the_point_under_the_cursor(ann_factory):
    ann = ann_factory()
    ann.shift(30, -20)
    cursor = (150.0, 380.0)
    winter_pt = apply_transform(np.linalg.inv(ann.matrix()), [cursor])[0]
    ann.rescale(1.01, about=cursor)
    assert ann.ov.scale == pytest.approx(1.01)
    assert np.allclose(moved(ann.matrix(), winter_pt), cursor)


def test_view_keys_and_flag(ann_factory):
    ann = ann_factory()
    ann.key(" ")
    assert ann.view == "winter"
    ann.key(" ")
    assert ann.view == "summer"
    ann.key("3")
    assert (ann.view, ann.alpha) == ("blend", 0.3)
    ann.key("m")
    ann.key("f")
    assert not ann.show_mask and ann.flag and "MASK FLAGGED" in ann.prompt()


def test_accept_records_time_flag_and_starts_the_next_from_the_row_median(ann_factory, tmp_path):
    ann = ann_factory()
    ann.shift(10, -40)
    ann.key("f")
    ann.clock_.t = 42.0
    ann.key("enter")
    saved = load_overlays(tmp_path / "overlay.json")["R01N01"]
    assert (saved.dx, saved.dy, saved.seconds, saved.mask_flag) == (10.0, -40.0, 42.0, True)
    # One tree done in row 1: the next row-1 tree starts from the median of all aligned trees (it).
    assert ann.tree.tree_id == "R01N02" and (ann.ov.dx, ann.ov.dy) == (10.0, -40.0) and not ann.flag
    ann.shift(-4, 0)
    ann.key("enter")
    # R15N05 is in another row: it starts from the median of all aligned trees.
    assert ann.tree.tree_id == "R15N05" and (ann.ov.dx, ann.ov.dy) == (8.0, -40.0)
    ann.key("s")
    assert ann.done and load_overlays(tmp_path / "overlay.json")["R15N05"].skip


def test_redo_starts_from_the_saved_transform_and_adds_the_time(ann_factory, tmp_path):
    ann = ann_factory()
    ann.shift(5, 5)
    ann.clock_.t = 10.0
    ann.key("enter")
    again = ann_factory(redo=True)
    assert (again.ov.dx, again.ov.dy) == (5.0, 5.0)
    again.clock_.t = 25.0
    again.key("enter")
    assert load_overlays(tmp_path / "overlay.json")["R01N01"].seconds == 25.0     # 10 s + 15 s


def test_fast_warp_matches_the_exact_warps():
    """What the window shows must be what is saved: masks exactly, photos to about a grey level."""
    from treeskel.align import warp_image, warp_mask
    from treeskel.annotate_overlay import fast_warp

    rng = np.random.default_rng(0)
    photo = (rng.uniform(0, 255, (60, 80, 3)).repeat(8, 0).repeat(8, 1)).astype(np.uint8)   # 480 x 640 blocks
    mask = np.zeros((480, 640), bool)
    mask[100:300, 200:230] = True
    mask[150:170, 230:400] = True
    for o in (Overlay(dx=12.3, dy=-40.7), Overlay(dx=-7, dy=25, scale=1.04, roll=2.5, pitch=3, yaw=-2)):
        m = o.matrix()
        assert np.array_equal(fast_warp(mask.astype(np.uint8) * 255, m, nearest=True) > 127, warp_mask(mask, m)[0])
        ref, covered = warp_image(photo, m)
        inner = covered & np.roll(covered, 2, 0) & np.roll(covered, -2, 0) & np.roll(covered, 2, 1) & np.roll(covered, -2, 1)
        diff = np.abs(fast_warp(photo, m).astype(int) - ref.astype(int))[inner]
        assert diff.mean() < 1.0


def test_views_show_what_they_say(ann_factory):
    from treeskel.annotate_overlay import fast_warp

    ann = ann_factory()
    ann.shift(20, -10)
    ann.key("m")                                       # mask off, to compare the photos alone
    ann.key(" ")
    winter = ann.composite()
    ann.key(" ")
    assert np.array_equal(ann.composite(), ann.summer["rgb"])
    moved = fast_warp(ann.winter["rgb"], ann.matrix())
    assert np.array_equal(winter[:460, 20:], moved[:460, 20:])        # the covered part shows winter
    assert np.array_equal(winter[:, :20], ann.summer["rgb"][:, :20])  # the rest shows summer
    ann.key("5")
    blend = ann.composite().astype(int)
    expect = (ann.summer["rgb"].astype(int) + moved.astype(int)) / 2
    assert np.abs(blend[:460, 20:] - expect[:460, 20:]).max() <= 1
    ann.key("m")
    assert (ann.composite() == (255, 0, 255)).all(-1).any()          # the mask's outline in magenta


def test_window_drag_scroll_and_keys(ann_factory):
    """The real matplotlib wiring, driven by simulated events on the Agg canvas."""
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

    plt.switch_backend("Agg")
    ann = ann_factory()
    fig = run_gui(ann, show=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    (picture,) = ax.get_images()
    before = picture.get_array().copy()

    def mouse(name, col, row, button=MouseButton.LEFT, **kw):
        x, y = ax.transData.transform((col, row))
        fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=button, **kw))

    ax.set_xlim(100, 400)                                             # zoomed in
    mouse("button_press_event", 300, 300)
    mouse("motion_notify_event", 312, 290, button=None)
    assert (ann.ov.dx, ann.ov.dy) == pytest.approx((12.0, -10.0))    # moves while dragging
    assert not np.array_equal(picture.get_array(), before)
    mouse("button_release_event", 312, 290)
    assert ax.get_xlim() == (100, 400)                                # the zoom stays
    n = len(ann.history)
    mouse("button_press_event", 200, 200)                             # a click without a drag
    mouse("button_release_event", 200, 200)
    assert len(ann.history) == n
    fig.canvas.toolbar = types.SimpleNamespace(mode="zoom rect")       # zoom on: drags ignored
    mouse("button_press_event", 300, 300)
    mouse("motion_notify_event", 350, 300, button=None)
    mouse("button_release_event", 350, 300)
    assert ann.ov.dx == pytest.approx(12.0)
    fig.canvas.toolbar = None
    mouse("scroll_event", 200, 200, button="up", step=1)
    assert ann.ov.scale == pytest.approx(1.01)
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, "u"))
    assert ann.ov.scale == pytest.approx(1.0)
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, "enter"))
    assert ann.tree.tree_id == "R01N02" and ax.get_xlim() == (-0.5, 639.5)   # new tree: full view
    plt.close(fig)


def test_depth_silhouette_keeps_large_near_objects():
    from treeskel.annotate_overlay import depth_silhouette

    depth = np.zeros((480, 640), np.uint8)
    depth[50:400, 300:320] = 150                       # a "post" in range; the rest out of range
    depth[100:105, 100:105] = 150                      # a speck
    sil = depth_silhouette(depth, depth > 0)
    assert sil[200, 310] and not sil[102, 102]
    assert sil.sum() == pytest.approx(350 * 20, rel=0.05)


def test_outlines_view_keeps_the_summer_photo_and_draws_winter_depth(ann_factory):
    ann = ann_factory()
    assert (ann.view, ann.alpha) == ("blend", 0.3)    # the winter photo starts faint
    ann.key("e")
    assert ann.view == "outlines" and "outlines" in ann.prompt()
    pic = ann.composite()
    cyan = (pic == (0, 255, 255)).all(-1)
    magenta = (pic == (255, 0, 255)).all(-1)
    assert cyan.any() and magenta.any()
    rest = ~(cyan | magenta)
    assert np.array_equal(pic[rest], ann.summer["rgb"][rest])      # the summer photo, untouched
    ann.key("b")
    assert ann.view == "blend"
