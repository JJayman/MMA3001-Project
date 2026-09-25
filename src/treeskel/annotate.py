"""Click tool for guided winter ground truth.

Run on your own computer (it opens a window)::

    python -m treeskel.annotate data/raw --clicks annotations/winter_clicks.json

For each tree the winter RGB image is shown. Click, in order:

1. the **trunk base** (where the trunk meets the ground),
2. the **fork** (where the trunk splits into the two leaders),
3. the **tip of the left leader** (highest visible point),
4. the **tip of the right leader**.

The trunk and leaders are then traced automatically (red/yellow/white). If a
trace goes wrong, click on the correct branch near where it strays: the point
is added as a waypoint to the nearest trace, which is re-routed through it.

Keys:

========  ==========================================================
Enter     accept this tree and go to the next
u         undo the last click
s         skip this tree (e.g. unusable image); it is excluded
d         switch between the RGB and depth image
n / p     next / previous tree without saving
q         save and quit
========  ==========================================================

Clicks are saved to the JSON file after every accepted or skipped tree, so
you can stop and resume at any time: trees already done are skipped unless
``--redo`` is given. Only the clicks are saved; the traced masks are
regenerated with ``python -m treeskel.guided_gt``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from treeskel.guided_gt import Clicks, branch_cost, guided_gt, insert_waypoint, load_clicks, save_clicks
from treeskel.io import TreeRecord, index_dataset, load_tree

STEPS = ("trunk base", "fork", "left leader tip", "right leader tip")


class Annotator:
    """State and event handling for the click tool, independent of display.

    Args:
        trees: Trees to annotate, in order.
        clicks_path: JSON file to read and write.
        redo: Also show trees that already have clicks.
    """

    def __init__(self, trees: Sequence[TreeRecord], clicks_path: str, redo: bool = False):
        self.clicks_path = clicks_path
        self.saved = load_clicks(clicks_path)
        self.trees = [t for t in trees if redo or t.tree_id not in self.saved]
        self.i = 0
        self.show_depth = False
        self.done = False
        self._load()

    # --- current tree -------------------------------------------------------
    @property
    def tree(self) -> TreeRecord | None:
        return self.trees[self.i] if self.i < len(self.trees) else None

    def _load(self) -> None:
        """Load the current tree's images and cost map; reset the clicks."""
        self.base: list[tuple[int, int]] = []   # clicks before the trace exists
        self.clicks: Clicks | None = None
        self.gt = None
        self.history: list = []
        if self.tree is None:
            self.done = True
            return
        self.data = load_tree(self.tree, "winter")
        self.cost, _ = branch_cost(self.data["rgb"], self.data["depth"], self.data["valid"])
        old = self.saved.get(self.tree.tree_id)
        if old and old.complete and not old.skip:
            self.clicks = old
            self._retrace()

    def _retrace(self) -> None:
        d = self.data
        self.gt = guided_gt(d["rgb"], d["depth"], d["valid"], self.clicks, cost=self.cost)

    # --- events -------------------------------------------------------------
    def prompt(self) -> str:
        """Instruction for the window title."""
        if self.done:
            return "All trees done - press q to save and quit"
        head = f"{self.tree.tree_id} ({self.i + 1}/{len(self.trees)}): "
        if self.clicks is None:
            return head + f"click the {STEPS[len(self.base)]}"
        return head + "Enter = accept, click to correct a trace, u = undo, s = skip"

    def click(self, row: float, col: float) -> None:
        """Handle a left click at image coordinates ``(row, col)``."""
        if self.done:
            return
        p = (int(round(row)), int(round(col)))
        if self.clicks is None:
            self.history.append(("base", list(self.base)))
            self.base.append(p)
            if len(self.base) == len(STEPS):
                root, fork, lt, rt = self.base
                self.clicks = Clicks(trunk=[root, fork], left=[lt], right=[rt]).normalised()
                self._retrace()
        else:
            self.history.append(("clicks", self.clicks))
            self.clicks = insert_waypoint(self.clicks, self.gt, p)
            self._retrace()

    def key(self, key: str) -> None:
        """Handle a key press."""
        if key == "q":
            save_clicks(self.clicks_path, self.saved)
            self.done = True
        elif self.done:
            return
        elif key == "enter" and self.clicks is not None:
            self.saved[self.tree.tree_id] = self.clicks
            save_clicks(self.clicks_path, self.saved)
            self._next(1)
        elif key == "s":
            self.saved[self.tree.tree_id] = Clicks(skip=True, note="skipped in annotator")
            save_clicks(self.clicks_path, self.saved)
            self._next(1)
        elif key == "u" and self.history:
            kind, prev = self.history.pop()
            if kind == "base":
                self.base, self.clicks, self.gt = prev, None, None
            else:
                self.clicks = prev
                self._retrace()
        elif key == "d":
            self.show_depth = not self.show_depth
        elif key in ("n", "p"):
            self._next(1 if key == "n" else -1)

    def _next(self, step: int) -> None:
        self.i = min(max(self.i + step, 0), len(self.trees))
        self._load()


def run_gui(ann: Annotator) -> None:  # pragma: no cover - needs a display
    """Open the matplotlib window and connect mouse and keyboard events."""
    import matplotlib.pyplot as plt

    for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward", "keymap.pan"):
        plt.rcParams[k] = []  # free up s, q, p, etc. for the tool
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.95)

    def draw() -> None:
        ax.clear()
        ax.set_axis_off()
        ax.set_title(ann.prompt(), fontsize=11)
        if ann.done:
            fig.canvas.draw_idle()
            return
        img = ann.data["depth"] if ann.show_depth else ann.data["rgb"]
        ax.imshow(img, cmap="gray" if ann.show_depth else None)
        if ann.gt is not None:
            for path, col in ((ann.gt.trunk, "white"), (ann.gt.left, "red"), (ann.gt.right, "yellow")):
                ax.plot(path[:, 1], path[:, 0], color=col, lw=2)
            c = ann.clicks
            pts = np.array([*c.trunk, *c.left, *c.right])
            ax.plot(pts[:, 1], pts[:, 0], "o", mfc="none", mec="cyan", ms=8, mew=2)
        elif ann.base:
            pts = np.array(ann.base)
            ax.plot(pts[:, 1], pts[:, 0], "o", mfc="none", mec="cyan", ms=8, mew=2)
        fig.canvas.draw_idle()

    def on_click(ev) -> None:
        if ev.inaxes is ax and ev.button == 1 and ev.xdata is not None:
            ann.click(ev.ydata, ev.xdata)
            draw()

    def on_key(ev) -> None:
        ann.key(ev.key)
        if ev.key == "q":
            plt.close(fig)
        else:
            draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a display
    """Start the click tool (see the module docstring for usage)."""
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.annotate", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--rows", type=int, nargs="+", help="only these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true",
                    help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--redo", action="store_true", help="also show trees already annotated")
    args = ap.parse_args(argv)

    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in args.trees]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    ann = Annotator(trees, args.clicks, redo=args.redo)
    print(f"{len(ann.trees)} trees to annotate; clicks saved to {args.clicks}")
    run_gui(ann)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
