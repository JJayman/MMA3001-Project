"""Tests for the winter-summer pair tool (treeskel.annotate_pairs).

The window cannot be opened here, so the event logic is tested directly, and
the real matplotlib wiring is tested with simulated clicks and key presses on
a non-interactive (Agg) canvas.
"""

import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from treeskel.annotate_pairs import PairAnnotator, run_gui
from treeskel.io import index_dataset
from treeskel.summer_gt import load_pairs

CLICKS = Path(__file__).resolve().parents[1] / "annotations" / "winter_clicks.json"


@pytest.fixture
def ann_factory(sample_dir, tmp_path):
    """Annotator on the sample's development trees, with a simple stand-in winter mask."""
    mask = np.zeros((480, 640), bool)
    mask[300:480, 300:330] = True                       # a "trunk"
    gt = tmp_path / "gt"
    gt.mkdir()
    trees = [t for t in index_dataset(sample_dir).trees if t.tree_id in ("R01N01", "R01N02")]
    for t in trees:
        Image.fromarray(mask.astype(np.uint8) * 255).save(gt / f"{t.tree_id}_winter_gt.png")

    def make(redo=False):
        return PairAnnotator(trees, tmp_path / "pairs.json", redo=redo, clicks_path=CLICKS, gt_dir=gt)
    return make


def shift_pairs(ann, dx=20.0, dy=-10.0):
    """Click three pairs related by a pure shift (winter first, summer first, winter first)."""
    for k, (r, c) in enumerate([(320.0, 310.0), (400.0, 100.0), (200.0, 500.0)]):
        if k == 1:
            ann.click("summer", r + dy, c + dx)
            ann.click("winter", r, c)
        else:
            ann.click("winter", r, c)
            ann.click("summer", r + dy, c + dx)


def test_pairs_form_in_either_order_and_fit_live(ann_factory):
    ann = ann_factory()
    assert ann.tree.tree_id == "R01N01" and ann.matrix is None and ann.warped is None
    ann.click("winter", 320.04, 310.0)
    assert ann.pending == ("winter", (320.0, 310.0)) and len(ann.pairs) == 0
    ann.click("winter", 321.0, 311.0)                    # a second click in the same image moves it
    assert ann.pending == ("winter", (321.0, 311.0))
    ann.key("r")
    shift_pairs(ann)
    assert ann.pairs.winter[1] == (400.0, 100.0) and ann.pairs.summer[1] == (390.0, 120.0)
    assert ann.pending is None
    assert np.nanmax(ann.loo) < 1e-6
    ys, xs = np.nonzero(ann.warped)                      # the stand-in trunk moved by the shift
    assert (ys.min(), xs.min(), xs.max()) == (290, 320, 349)
    assert "leave-one-out error median 0.0 px" in ann.prompt()


def test_undo_reset_and_the_two_pair_minimum(ann_factory):
    ann = ann_factory()
    ann.click("winter", 100, 100)
    ann.click("summer", 110, 120)
    assert len(ann.pairs) == 1
    ann.key("enter")                                      # one pair is not enough
    assert ann.tree.tree_id == "R01N01"
    ann.key("u")                                          # undo the second click only
    assert len(ann.pairs) == 0 and ann.pending == ("winter", (100.0, 100.0))
    ann.key("u")
    assert ann.pending is None
    shift_pairs(ann)
    ann.key("r")
    assert len(ann.pairs) == 0 and ann.matrix is None
    ann.key("u")                                          # a reset can be undone
    assert len(ann.pairs) == 3 and ann.matrix is not None


def test_accept_skip_resume_and_redo(ann_factory, tmp_path):
    ann = ann_factory()
    shift_pairs(ann)
    ann.key("enter")
    assert ann.tree.tree_id == "R01N02"
    ann.key("s")
    assert ann.done
    saved = load_pairs(tmp_path / "pairs.json")
    assert len(saved["R01N01"]) == 3 and saved["R01N02"].skip
    assert ann_factory().done                            # both trees done: nothing left to show
    again = ann_factory(redo=True)                       # --redo shows the saved pairs
    assert again.tree.tree_id == "R01N01" and len(again.pairs) == 3 and again.matrix is not None


def test_navigation_toggles_and_quit(ann_factory, tmp_path):
    ann = ann_factory()
    ann.key("n")
    assert ann.tree.tree_id == "R01N02"
    ann.key("p")
    assert ann.tree.tree_id == "R01N01"
    ann.key("m")
    ann.key("d")
    assert not ann.show_masks and ann.show_depth
    with pytest.raises(ValueError):
        ann.click("autumn", 1, 1)
    ann.key("q")
    assert ann.done and (tmp_path / "pairs.json").exists()
    assert "All trees done" in ann.prompt()


def test_window_turns_clicks_and_keys_into_pairs(ann_factory):
    """The real matplotlib wiring, driven by simulated events on the Agg canvas."""
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

    plt.switch_backend("Agg")
    ann = ann_factory()
    fig = run_gui(ann, show=False)
    fig.canvas.draw()
    axw, axs = fig.axes[:2]

    def click(ax, row, col):
        x, y = ax.transData.transform((col, row))
        fig.canvas.callbacks.process("button_press_event",
                                     MouseEvent("button_press_event", fig.canvas, x, y, button=MouseButton.LEFT))

    def key(k):
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, k))

    click(axw, 320, 310)
    click(axs, 310, 330)
    assert ann.pairs.winter == [(320.0, 310.0)] and ann.pairs.summer == [(310.0, 330.0)]
    fig.canvas.toolbar = types.SimpleNamespace(mode="zoom rect")    # zoom switched on: clicks ignored
    click(axw, 100, 100)
    assert ann.pending is None
    fig.canvas.toolbar = None
    axw.set_xlim(200, 400)                               # zoomed in: the view stays after a click
    click(axw, 400, 300)
    assert axw.get_xlim() == (200, 400) and ann.pending == ("winter", (400.0, 300.0))
    key("u")
    assert ann.pending is None
    key("n")
    assert ann.tree.tree_id == "R01N02" and axw.get_xlim() != (200, 400)    # new tree: full view again
    plt.close(fig)


def test_homography_model_needs_four_pairs_and_shades_the_uncovered_part(ann_factory, sample_dir, tmp_path):
    ann = ann_factory()
    ann.model = "homography"
    shift_pairs(ann)                                      # 3 pairs: too few for a homography
    assert ann.matrix is None and ann.covered is None
    ann.key("enter")
    assert ann.tree.tree_id == "R01N01"                   # not accepted with 3 pairs
    ann.click("winter", 100.0, 150.0)
    ann.click("summer", 90.0, 170.0)                      # 4th pair, same shift
    assert ann.matrix is not None and "Fit (homography)" in ann.prompt()
    # The shift is 20 px right and 10 px up, so summer's left 20 columns and bottom 10 rows
    # show what the winter image does not.
    assert not ann.covered[:, :20].any() and not ann.covered[-10:].any()
    assert ann.covered[:470, 20:].all()


def test_window_shades_the_uncovered_part(ann_factory):
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    ann = ann_factory()
    shift_pairs(ann)
    fig = run_gui(ann, show=False)
    fig.canvas.draw()
    axs = fig.axes[1]
    shades = [im for im in axs.get_images() if im.get_array().ndim == 3 and im.get_array().shape[2] == 4
              and im.get_array()[..., 3].max() == pytest.approx(0.45)]
    assert len(shades) == 1                               # one translucent layer over the uncovered part
    plt.close(fig)
