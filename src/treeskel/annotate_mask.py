"""Line-up tool: move each provided hand mask onto the branches of its own summer photo.

The hand masks have the tree's shape (session 10: all 448 fit their winter
traces within 8 px), but not always its place in the summer photo: on the 21
development trees in the repository several sit 20–60 px to one side of the
visible trunk, and R20N06's about 90 px (session 11). A segmentation trained
on them (:mod:`treeskel.segment`) learns those offsets as noise. Lining up a
mask is easier than lining up the two seasons was (session 7): the summer
trunk is in view, and only the mask moves.

Run on your own computer (it opens a window)::

    python -m treeskel.annotate_mask data/raw --dev-only
    python -m treeskel.annotate_mask data/raw --trees R20N06 R01N32 --redo

The summer photo is shown with the hand mask's outline in magenta, moved by
the current shift and scale (a :class:`treeskel.summer_gt.Overlay` with no
roll or tilt, in photo pixels). Move it until it sits on the trunk and on
whatever of the leaders shows through the leaves:

- **drag** (left mouse button): shift it;
- **arrow keys**: shift by 1 px (with shift: 10 px);
- **scroll**: make it bigger or smaller around the cursor; **- and =**:
  by 0.5% about the mask's fork (use only if the trunk and the leaders
  cannot both be matched by a shift);
- **space**: the mask filled (half see-through) or outlined;
- **m**: hide the mask, to see the trunk underneath; press again to show it.

Other keys:

========  ==========================================================
Enter     accept this tree and go to the next
u         undo the last change
r         reset to the starting shift
s         skip this tree (cannot tell where the mask belongs); it is
          then not used for training
f         flag this tree (e.g. the mask's shape is wrong, not only its
          place); press again to unflag
n / p     next / previous tree without saving
q         save and quit
========  ==========================================================

Each tree starts from the median shift of the trees already lined up in its
row, or of all trees, so often only a nudge is needed. The shifts are saved
to ``annotations/mask_overlay.json`` after every tree, with the time spent.
:func:`move_mask` applies a saved shift to a 256x256 mask; the rows at the
photo's bottom edge are repeated, so a mask moved up keeps its trunk to the
edge (the trunk continues below the photo).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from treeskel.align import apply_transform, outline
from treeskel.annotate_overlay import median_overlay
from treeskel.io import CROP_COLS, IMAGE_SHAPE, MASK_SHAPE, TreeRecord, index_dataset, load_mask, load_tree, \
    mask_to_image_frame
from treeskel.summer_gt import Overlay, load_overlays, save_overlays

STEPS = {"shift": 1.0, "shift_big": 10.0, "scale": 1.005, "scroll": 1.01}
"""Size of one step of each control (px or a factor)."""

MASK_PER_PHOTO = MASK_SHAPE[0] / IMAGE_SHAPE[0]
"""Mask pixels per photo pixel (256 / 480): the masks show the photo's centre 480x480."""

_TO_MASK = np.array([[MASK_PER_PHOTO, 0, -CROP_COLS[0] * MASK_PER_PHOTO], [0, MASK_PER_PHOTO, 0], [0, 0, 1.0]])
"""Photo ``(x, y)`` to mask ``(x, y)``."""


def mask_transform(ov: Overlay) -> np.ndarray:
    """A photo-pixel shift and scale as a 3x3 ``(x, y)`` transform of the 256x256 mask frame."""
    return _TO_MASK @ ov.matrix() @ np.linalg.inv(_TO_MASK)


def move_mask(mask: np.ndarray, ov: Overlay) -> np.ndarray:
    """A 256x256 hand mask moved by a saved shift (and scale), nearest-neighbour.

    Pixels that come from beyond the mask's edge repeat the edge, so a trunk
    that reaches the bottom of the photo still does after a move up.

    Example:
        >>> m = np.zeros((256, 256), bool); m[100:120, 50:60] = True
        >>> moved = move_mask(m, Overlay(dx=15.0))           # 15 photo px = 8 mask px
        >>> [int(c) for c in np.nonzero(moved.any(0))[0][[0, -1]]]
        [58, 67]
    """
    m = np.asarray(mask, bool)
    if ov.dx == 0 and ov.dy == 0 and ov.scale == 1 and ov.roll == 0 and ov.pitch == 0 and ov.yaw == 0:
        return m.copy()
    inv = np.linalg.inv(mask_transform(ov))                 # output (x, y) -> input (x, y)
    inv = inv / inv[2, 2]
    matrix = np.array([[inv[1, 1], inv[1, 0]], [inv[0, 1], inv[0, 0]]])     # in (row, col) order
    offset = np.array([inv[1, 2], inv[0, 2]])
    return ndi.affine_transform(m.astype(np.uint8), matrix, offset, order=0, mode="nearest").astype(bool)


def mask_fork(mask: np.ndarray) -> np.ndarray:
    """The hand mask's fork as a photo ``(x, y)`` point (the pin for scaling), or its centre if none is found."""
    from treeskel.shape import mask_branches, mask_to_image

    br = mask_branches(np.asarray(mask, bool))
    if br is not None and len(br["trunk"]):
        r, c = mask_to_image(br["trunk"][-1:])[0]
        return np.array([c, r], float)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.array([IMAGE_SHAPE[1] / 2, IMAGE_SHAPE[0] / 2])
    r, c = mask_to_image(np.array([[rows.mean(), cols.mean()]]))[0]
    return np.array([c, r], float)


def apply_mask_overlays(masks: dict[str, np.ndarray], overlays: dict[str, Overlay]) -> dict[str, np.ndarray | None]:
    """Each tree's hand mask moved by its saved shift; None for a skipped tree, unchanged for one not lined up."""
    out = {}
    for t, m in masks.items():
        ov = overlays.get(t)
        out[t] = None if ov is not None and ov.skip else (move_mask(m, ov) if ov is not None else m)
    return out


class MaskAnnotator:
    """State and event handling for the mask line-up tool, independent of display.

    It offers what :func:`treeskel.annotate_overlay.run_gui` drives: ``done``,
    ``tree``, ``ov``, ``history``, :meth:`composite`, :meth:`rescale`,
    :meth:`key` and :meth:`prompt`.

    Args:
        trees: Trees to line up, in order (those without a hand mask are left out).
        path: JSON file to read and write.
        redo: Also show trees already lined up (starting from their shift).
        clock: Time source in seconds (for tests).
    """

    def __init__(self, trees: Sequence[TreeRecord], path: str | Path, redo: bool = False,
                 clock: Callable[[], float] = time.monotonic):
        """Load the saved shifts and open the first tree still to line up."""
        self.path = path
        self.saved = load_overlays(path)
        self.trees = [t for t in trees if t.has_mask and (redo or t.tree_id not in self.saved)]
        self.clock = clock
        self.i = 0
        self.filled = False
        self.show_mask = True
        self.done = False
        self._load()

    @property
    def tree(self) -> TreeRecord | None:
        """The tree being lined up, or ``None`` once every tree is done."""
        return self.trees[self.i] if self.i < len(self.trees) else None

    def start(self) -> Overlay:
        """Starting shift: the saved one, else the median of the row's trees, else of all trees."""
        tid = self.tree.tree_id
        old = self.saved.get(tid)
        if old is not None and not old.skip:
            return replace(old, skip=False)
        done = [o for t, o in self.saved.items() if not o.skip and t != tid]
        row = [o for t, o in self.saved.items() if not o.skip and t != tid and t[:3] == tid[:3]]
        m = median_overlay(row if len(row) >= 2 else done)
        return Overlay(dx=m.dx, dy=m.dy, scale=m.scale)          # only shift and scale are used here

    def _load(self) -> None:
        self.history: list[Overlay] = []
        if self.tree is None:
            self.done = True
            return
        tid = self.tree.tree_id
        self.summer = load_tree(self.tree, "summer")["rgb"]
        self.mask = load_mask(self.tree.paths[("summer", "mask")])
        self.pin = mask_fork(self.mask)
        self.initial = self.start()
        self.ov = replace(self.initial)
        old = self.saved.get(tid)
        self.flag = bool(old.mask_flag) if old is not None else False
        self.spent_before = old.seconds if old is not None else 0.0
        self.t0 = self.clock()

    def moved(self) -> np.ndarray:
        """The hand mask moved by the current shift, in the photo's frame ``(480, 640)``."""
        return mask_to_image_frame(move_mask(self.mask, self.ov))[0]

    def composite(self) -> np.ndarray:
        """The summer photo with the moved mask outlined (or filled, half see-through) in magenta."""
        out = self.summer.copy()
        if not self.show_mask:
            return out
        full = self.moved()
        if self.filled:
            out[full] = (0.5 * out[full] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        else:
            out[outline(full, 2)] = (255, 0, 255)
        return out

    def _change(self, pin_xy: np.ndarray | None = None, **params) -> None:
        """Set some parameters, keeping the photo point ``pin_xy`` of the mask where it was."""
        self.history.append(replace(self.ov))
        before = apply_transform(self.ov.matrix(), [pin_xy])[0] if pin_xy is not None else None
        for k, v in params.items():
            setattr(self.ov, k, float(v))
        if before is not None:
            after = apply_transform(self.ov.matrix(), [pin_xy])[0]
            self.ov.dx += before[0] - after[0]
            self.ov.dy += before[1] - after[1]

    def shift(self, ddx: float, ddy: float) -> None:
        """Move the mask by ``(ddx, ddy)`` photo px."""
        self._change(dx=self.ov.dx + ddx, dy=self.ov.dy + ddy)

    def rescale(self, factor: float, about: tuple[float, float] | None = None) -> None:
        """Scale by ``factor``, keeping the photo point ``about`` (``(x, y)``) fixed, or the mask's fork."""
        q = self.pin if about is None else apply_transform(np.linalg.inv(self.ov.matrix()), [about])[0]
        self._change(pin_xy=q, scale=self.ov.scale * factor)

    def prompt(self) -> str:
        """Current tree, shift and help, for the window title."""
        if self.done:
            return "All trees done - press q to save and quit"
        o = self.ov
        spent = self.spent_before + self.clock() - self.t0
        return (f"{self.tree.tree_id} ({self.i + 1}/{len(self.trees)})" + ("  FLAGGED" if self.flag else "")
                + f"  {spent:.0f} s\nshift {o.dx:+.1f}, {o.dy:+.1f} px   scale {o.scale:.3f}")

    def key(self, key: str) -> None:
        """Handle a key press."""
        if key == "q":
            save_overlays(self.path, self.saved)
            self.done = True
            return
        if self.done:
            return
        arrows = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
        base = key.replace("shift+", "")
        if base in arrows:
            step = STEPS["shift_big"] if key.startswith("shift+") else STEPS["shift"]
            self.shift(arrows[base][0] * step, arrows[base][1] * step)
        elif key in ("-", "=", "+"):
            self.rescale(1 / STEPS["scale"] if key == "-" else STEPS["scale"])
        elif key == " ":
            self.filled = not self.filled
        elif key == "m":
            self.show_mask = not self.show_mask
        elif key == "f":
            self.flag = not self.flag
        elif key == "u" and self.history:
            self.ov = self.history.pop()
        elif key == "r":
            self._change(**{k: getattr(self.initial, k) for k in ("dx", "dy", "scale")})
        elif key == "enter":
            self._save(Overlay(dx=self.ov.dx, dy=self.ov.dy, scale=self.ov.scale))
        elif key == "s":
            self._save(Overlay(skip=True, note="skipped in the mask line-up tool"))
        elif key in ("n", "p"):
            self._next(1 if key == "n" else -1)

    def _save(self, ov: Overlay) -> None:
        ov.mask_flag = self.flag
        ov.seconds = self.spent_before + self.clock() - self.t0
        self.saved[self.tree.tree_id] = ov
        save_overlays(self.path, self.saved)
        self._next(1)

    def _next(self, step: int) -> None:
        self.i = min(max(self.i + step, 0), len(self.trees))
        self._load()


HELP = ("drag: shift   arrows: 1 px (shift: 10)   scroll, - =: scale   space: filled / outline   m: hide the mask\n"
        "Enter: accept   u: undo   r: reset   s: skip   f: flag   n / p: next / previous   q: quit")
"""Key help shown above the picture."""


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a display
    """Start the mask line-up tool (see the module docstring for usage)."""
    from treeskel.annotate_overlay import run_gui
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.annotate_mask", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="annotations/mask_overlay.json", help="shifts file (read and written)")
    ap.add_argument("--rows", type=int, nargs="+", help="only these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true", help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--redo", action="store_true", help="also show trees already lined up")
    args = ap.parse_args(argv)

    all_trees = index_dataset(args.data_dir).trees
    trees = all_trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, all_trees)
        trees = [t for t in trees if t.row not in test_rows]
    ann = MaskAnnotator(trees, args.out, redo=args.redo)
    print(f"{len(ann.trees)} hand masks to line up; shifts saved to {args.out}")
    run_gui(ann, help_text=HELP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
