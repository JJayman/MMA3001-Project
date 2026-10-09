"""Click tool for winter-summer point pairs, which align the two seasons.

Run on your own computer (it opens a window)::

    python -m treeskel.annotate_pairs data/raw --dev-only

Each tree's winter image is shown on the left, with its traced mask
outlined, and its summer image on the right. Click a point in one image, then
the same physical point in the other: that makes a numbered pair. Clicking
twice in the same image moves the waiting point.

**Which points.** Only points that are clearly the same physical point in
both seasons, and on the tree's row (the plane the tree grows in):

- features on the trunk: the graft union (the bulge near the base), knots,
  scars, where a side branch leaves it;
- where the drip line crosses the trunk or a post;
- features on posts: the top, a clip or staple, a knot;
- the same kinds of features on the neighbouring trees' trunks.

Avoid the ground (it lies in front of or behind the row), leaves and fruit,
the rows behind, and thin branches, which may have grown or bent under the
fruit. Aim for **4 pairs** (at least 3), spread out across the tree: left
and right, high and low. To compare the models (see below), click **5–6
pairs**, one near each corner of the part both images show, plus one or two
in between.

**The model.** The camera stood at a different height, distance and angle
in each season. A similarity (``--model similarity``, the default) covers a
shift, a scale and a roll. Pitch or yaw between the seasons also adds
perspective, which only a homography (``--model homography``, 4 pairs to
fit, 5 to check) models exactly. With points spread over the image, a 5°
tilt left the similarity about 9 px off (leave-one-out), while the
homography was exact (``tests/test_summer_gt.py``). But the homography's
extra parameters also follow click errors more, so which is better overall
is decided by ``python -m treeskel.summer_gt`` on the development rows.

**Checking as you go.** Once the model can be fitted, the winter mask moved
by it is outlined on the summer image: its trunk should sit on the summer
trunk. The part of the summer image that the winter image does not cover is
shaded, since the winter mask says nothing there (its label will be
"unknown"). With one pair more than the fit needs, each summer point shows
its leave-one-out error, i.e. how far from the click the fit of the other
pairs puts it (:func:`treeskel.summer_gt.loo_errors`). A large one (red,
over 10 px) is usually a wrong click: undo it, or reset. With just one pair
to spare, one wrong click raises all the errors, because each is checked
against a fit that includes it; the largest is usually the wrong one, and
another pair makes it clear.

Keys:

========  ==========================================================
Enter     accept this tree (enough pairs for the model) and go to the next
u         undo the last click
r         reset this tree's pairs
s         skip this tree (no usable points); it gets no summer mask
m         show or hide the mask outlines
d         switch between the colour and depth images
n / p     next / previous tree without saving
q         save and quit
========  ==========================================================

Zoom in with the toolbar's magnifier to click precisely. Clicks made while
zoom or pan is switched on are ignored, so switch it off to click; the zoom
stays while you click on the same tree. The pairs are saved to the JSON file
after every accepted or skipped tree, so you can stop and resume at any time.
Trees already done are skipped unless ``--redo`` is given; ``--trees R05N12
--redo`` shows a tree again with its saved pairs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from treeskel.align import MIN_SAMPLES, describe, load_winter_mask, outline, warp_mask
from treeskel.guided_gt import load_clicks
from treeskel.io import TreeRecord, index_dataset, load_tree
from treeskel.summer_gt import PAIR_MODELS, SeasonPairs, load_pairs, loo_errors, pair_transform, save_pairs

PANELS = ("winter", "summer")


class PairAnnotator:
    """State and event handling for the pair tool, independent of display.

    Args:
        trees: Trees to align, in order.
        pairs_path: JSON file to read and write.
        redo: Also show trees that already have pairs (with those pairs).
        clicks_path: Winter clicks, to trace a tree's mask if it is not in ``gt_dir``.
        gt_dir: Folder of ``<tree>_winter_gt.png`` masks (from :mod:`treeskel.guided_gt`).
        model: Transform fitted for the live check.
    """

    def __init__(self, trees: Sequence[TreeRecord], pairs_path: str | Path, redo: bool = False,
                 clicks_path: str | Path = "annotations/winter_clicks.json",
                 gt_dir: str | Path | None = "outputs/winter_gt_guided", model: str = "similarity"):
        """Load the saved pairs and the winter clicks, and open the first tree still to pair."""
        self.pairs_path = pairs_path
        self.saved = load_pairs(pairs_path)
        self.trees = [t for t in trees if redo or t.tree_id not in self.saved]
        self.winter_clicks = load_clicks(clicks_path)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.model = model
        self.i = 0
        self.show_depth = False
        self.show_masks = True
        self.done = False
        self._load()

    # --- current tree -------------------------------------------------------
    @property
    def tree(self) -> TreeRecord | None:
        """The tree being paired, or ``None`` once every tree is done."""
        return self.trees[self.i] if self.i < len(self.trees) else None

    def _load(self) -> None:
        """Load the current tree's images and mask, and any saved pairs."""
        self.pairs = SeasonPairs()
        self.pending: tuple[str, tuple[float, float]] | None = None   # first point of a pair
        self.history: list = []
        self.matrix = None
        self.loo = np.zeros(0)
        self.warped = None
        self.covered = None
        if self.tree is None:
            self.done = True
            return
        self.winter = load_tree(self.tree, "winter")
        self.summer = load_tree(self.tree, "summer")
        self.mask = load_winter_mask(self.tree.tree_id, self.winter, self.winter_clicks, self.gt_dir)
        old = self.saved.get(self.tree.tree_id)
        if old is not None and not old.skip:
            self.pairs = SeasonPairs(list(old.winter), list(old.summer), note=old.note)
        self._refit()

    def _refit(self) -> None:
        """Fit the transform, the leave-one-out errors and the moved mask to the current pairs."""
        enough = len(self.pairs) >= MIN_SAMPLES[self.model]
        self.matrix = pair_transform(self.pairs, self.model) if enough else None
        self.loo = loo_errors(self.pairs, self.model)
        self.warped = self.covered = None
        if self.matrix is not None:
            shape = self.summer["rgb"].shape[:2]
            mask = self.mask if self.mask is not None else np.zeros(shape, bool)
            self.warped, self.covered = warp_mask(mask, self.matrix, shape)
            if self.mask is None:
                self.warped = None

    def _snapshot(self) -> None:
        pairs = SeasonPairs(list(self.pairs.winter), list(self.pairs.summer), note=self.pairs.note)
        self.history.append((pairs, self.pending))

    # --- events -------------------------------------------------------------
    def prompt(self) -> str:
        """Instructions and the current fit, for the window title."""
        if self.done:
            return "All trees done - press q to save and quit"
        head = f"{self.tree.tree_id} ({self.i + 1}/{len(self.trees)}), {len(self.pairs)} pair(s). "
        if self.pending is not None:
            other = "summer" if self.pending[0] == "winter" else "winter"
            head += f"Now click the same point in the {other} image."
        else:
            head += "Click a point in either image."
        if self.matrix is not None:
            d = describe(self.matrix, (320, 240))
            head += f"\nFit ({self.model}): scale {d['scale']:.2f}, rotation {d['rotation']:+.1f} deg"
            if np.isfinite(self.loo).any():
                head += (f"; leave-one-out error median {np.nanmedian(self.loo):.1f} px, "
                         f"max {np.nanmax(self.loo):.1f} px")
            else:
                head += "; add a pair to check it"
        return head + "\nEnter accept, u undo, r reset, s skip, m masks, d depth, n/p next/previous, q quit"

    def click(self, panel: str, row: float, col: float) -> None:
        """Handle a left click at image coordinates ``(row, col)`` in ``"winter"`` or ``"summer"``.

        The first click of a pair waits (a second click in the same image moves
        it); a click in the other image completes the pair.
        """
        if self.done:
            return
        if panel not in PANELS:
            raise ValueError(f"panel must be one of {PANELS}, not {panel!r}")
        self._snapshot()
        p = (round(float(row), 1), round(float(col), 1))
        if self.pending is None or self.pending[0] == panel:
            self.pending = (panel, p)
            return
        w, s = (self.pending[1], p) if self.pending[0] == "winter" else (p, self.pending[1])
        self.pairs.winter.append(w)
        self.pairs.summer.append(s)
        self.pending = None
        self._refit()

    def key(self, key: str) -> None:
        """Handle a key press."""
        if key == "q":
            save_pairs(self.pairs_path, self.saved)
            self.done = True
        elif self.done:
            return
        elif key == "enter" and len(self.pairs) >= max(2, MIN_SAMPLES[self.model]):
            self.saved[self.tree.tree_id] = SeasonPairs(list(self.pairs.winter), list(self.pairs.summer),
                                                        note=self.pairs.note)
            save_pairs(self.pairs_path, self.saved)
            self._next(1)
        elif key == "s":
            self.saved[self.tree.tree_id] = SeasonPairs(skip=True, note="skipped in the pair tool")
            save_pairs(self.pairs_path, self.saved)
            self._next(1)
        elif key == "u" and self.history:
            self.pairs, self.pending = self.history.pop()
            self._refit()
        elif key == "r":
            self._snapshot()
            self.pairs, self.pending = SeasonPairs(), None
            self._refit()
        elif key == "m":
            self.show_masks = not self.show_masks
        elif key == "d":
            self.show_depth = not self.show_depth
        elif key in ("n", "p"):
            self._next(1 if key == "n" else -1)

    def _next(self, step: int) -> None:
        self.i = min(max(self.i + step, 0), len(self.trees))
        self._load()


def _show_outline(ax, mask: np.ndarray, colour=(1.0, 0.0, 1.0)) -> None:
    """Draw a mask's outline on an axes as a transparent overlay."""
    rgba = np.zeros((*mask.shape, 4))
    rgba[outline(mask, 2)] = (*colour, 1.0)
    ax.imshow(rgba, interpolation="nearest")


def run_gui(ann: PairAnnotator, show: bool = True):
    """Open the two-panel window and connect mouse and keyboard events.

    Args:
        ann: The annotator.
        show: Block in ``plt.show()`` (False only for tests).

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward", "keymap.pan"):
        plt.rcParams[k] = []                        # free s, q, p, etc. for the tool
    plt.rcParams["keymap.home"] = ["h", "home"]     # and r (reset); h still resets the zoom
    fig, (axw, axs) = plt.subplots(1, 2, figsize=(16, 6.8))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.86, wspace=0.01)
    shown = {"tree": None}                          # keep the zoom while on the same tree

    def draw() -> None:
        """Redraw both photos, the masks and the numbered pairs with their errors, keeping the zoom on the same tree."""
        same = not ann.done and shown["tree"] == ann.tree.tree_id
        limits = [(a.get_xlim(), a.get_ylim()) for a in (axw, axs)] if same else None
        for a in (axw, axs):
            a.clear()
            a.set_axis_off()
        fig.suptitle(ann.prompt(), fontsize=10)
        if ann.done:
            fig.canvas.draw_idle()
            return
        kind, cmap = ("depth", "gray") if ann.show_depth else ("rgb", None)
        axw.imshow(ann.winter[kind], cmap=cmap)
        axs.imshow(ann.summer[kind], cmap=cmap)
        axw.set_title("Winter", fontsize=10)
        axs.set_title("Summer (magenta: the winter mask moved by the fit)", fontsize=10)
        if ann.show_masks:
            if ann.mask is not None:
                _show_outline(axw, ann.mask)
            if ann.covered is not None:                 # shade what the winter image does not show
                shade = np.zeros((*ann.covered.shape, 4))
                shade[~ann.covered] = (0.0, 0.0, 0.0, 0.45)
                axs.imshow(shade, interpolation="nearest")
            if ann.warped is not None:
                _show_outline(axs, ann.warped)
        for k in range(len(ann.pairs)):
            colour = plt.cm.tab10(k % 10)
            (wr, wc), (sr, sc) = ann.pairs.winter[k], ann.pairs.summer[k]
            err = ann.loo[k] if k < len(ann.loo) else np.nan
            for ax, (r, c), text in ((axw, (wr, wc), f"{k + 1}"),
                                     (axs, (sr, sc), f"{k + 1}" + (f": {err:.1f} px" if np.isfinite(err) else ""))):
                ax.plot(c, r, "o", mfc="none", mec=colour, ms=11, mew=2)
                bad = ax is axs and np.isfinite(err) and err > 10
                ax.annotate(text, (c, r), xytext=(8, -10), textcoords="offset points", fontsize=10,
                            weight="bold", color="#ff4040" if bad else colour,
                            bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.6, "lw": 0})
        if ann.pending is not None:
            panel, (r, c) = ann.pending
            (axw if panel == "winter" else axs).plot(c, r, "+", color="white", ms=18, mew=2)
        if limits:
            for a, (xl, yl) in zip((axw, axs), limits):
                a.set_xlim(xl)
                a.set_ylim(yl)
        shown["tree"] = ann.tree.tree_id
        fig.canvas.draw_idle()

    def on_click(ev) -> None:
        """A left click in either photo adds a point to a pair (not while zooming or panning)."""
        if ev.button != 1 or ev.inaxes not in (axw, axs) or ev.xdata is None:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return                                  # zoom or pan is switched on
        ann.click("winter" if ev.inaxes is axw else "summer", ev.ydata, ev.xdata)
        draw()

    def on_key(ev) -> None:
        """Pass a key to the annotator; ``q`` closes the window."""
        ann.key(ev.key)
        if ev.key == "q":
            plt.close(fig)
        else:
            draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    if show:  # pragma: no cover - needs a display
        plt.show()
    return fig


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a display
    """Start the pair tool (see the module docstring for usage)."""
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.annotate_pairs", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--pairs", default="annotations/season_pairs.json")
    ap.add_argument("--rows", type=int, nargs="+", help="only these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true", help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--redo", action="store_true", help="also show trees already aligned")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--winter-gt", default="outputs/winter_gt_guided",
                    help="folder of <tree>_winter_gt.png masks; traced from the clicks if missing")
    ap.add_argument("--model", choices=PAIR_MODELS, default="similarity",
                    help="transform fitted for the live check (the saved pairs do not depend on it)")
    args = ap.parse_args(argv)

    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    ann = PairAnnotator(trees, args.pairs, redo=args.redo, clicks_path=args.clicks, gt_dir=args.winter_gt,
                        model=args.model)
    print(f"{len(ann.trees)} trees to align; pairs saved to {args.pairs}")
    run_gui(ann)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
