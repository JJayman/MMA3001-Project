"""Overlay tool: line up each tree's winter photo on its summer photo by hand.

Clicking exactly the same point in both seasons was slow (about 2 minutes a
tree) and still about 10 px off (session 7, ``docs/results/align/``): in
summer the leaves hide most of the tree, and the two photos are framed
differently, so few points can be matched for certain. Lining up the whole
photos is easier: the trunk, the posts, the drip line and the ground line
must all sit on top of each other at once.

Run on your own computer (it opens a window)::

    python -m treeskel.annotate_overlay data/raw --trees R01N01 R01N02

The summer photo is shown with the winter photo on top, half see-through,
moved by the current transform (:class:`treeskel.summer_gt.Overlay`). Move the
winter photo until the two line up:

- **drag** (left mouse button): shift it;
- **scroll**: make it bigger or smaller around the cursor;
- **arrow keys**: shift by 1 px (with shift: 10 px);
- **[ and ]**: rotate (roll) by 0.2° (**{ and }**: 1°);
- **- and =**: scale by 0.5%;
- **i and k**: tilt up and down (pitch) by 0.5°: one end of the photo gets
  wider than the other;
- **j and l**: tilt left and right (yaw) by 0.5°.

Keyboard scaling, rolling and tilting keep the tree's fork where it is, so
fine-tuning does not undo a lined-up trunk. To judge the fit:

- **space**: flip between the winter and the summer photo. What is lined up
  stays still; what is not jumps;
- **b**: back to the blend; **1**–**9**: blend from 10% to 90% winter;
- **m**: show or hide the winter mask (magenta).

A good order: drag until the trunk sits on the summer trunk; then match the
posts and the drip line with scale and roll; use the tilts only if the top
and the bottom still disagree. Leaves hide the leaders in summer, so match
what both photos show.

Other keys:

========  ==========================================================
Enter     accept this tree and go to the next
u         undo the last change
r         reset to the starting transform
s         skip this tree (cannot be aligned); it gets no summer mask
f         flag this tree's winter mask as wrong (off the branch, wrong
          width), to be fixed later; press again to unflag
n / p     next / previous tree without saving
q         save and quit
========  ==========================================================

Each tree starts from the median transform of the trees already aligned in
its row (the camera was handled alike within a row), or of all aligned
trees, so often only a small nudge is needed. The transforms are saved to
``annotations/season_overlay.json`` after every tree, with the time spent on
it. The toolbar's zoom works as usual: drags made while zoom or pan is
switched on are ignored. ``--redo`` shows trees already aligned, starting
from their saved transform.

The current tree, its transform and whether its mask is flagged are shown
in the window's title bar. Each change takes about 60 ms on a 4-core
machine (the first version took about 400 ms). The time grows with the
window's size in pixels, so if it lags, make the window smaller.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from treeskel.align import apply_transform, load_winter_mask, outline
from treeskel.guided_gt import load_clicks
from treeskel.io import TreeRecord, index_dataset, load_tree
from treeskel.summer_gt import Overlay, load_overlays, save_overlays

STEPS = {"shift": 1.0, "shift_big": 10.0, "roll": 0.2, "roll_big": 1.0, "scale": 1.005, "scroll": 1.01,
         "tilt": 0.5}
"""Size of one step of each control (px, degrees or a factor)."""


def median_overlay(overlays: Sequence[Overlay]) -> Overlay:
    """Parameter-wise median of some transforms (the identity if there are none)."""
    if not overlays:
        return Overlay()
    return Overlay(**{k: float(np.median([getattr(o, k) for o in overlays])) for k in Overlay.PARAMS})


_HALF = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])


def fast_warp(image: np.ndarray, matrix: np.ndarray, nearest: bool = False,
              out_size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Warp an image by a winter-to-summer transform with Pillow's compiled perspective warp.

    About 10 times faster than :func:`treeskel.align.warp_image`, for the live
    display. Pillow maps each output pixel back into the input with pixel
    centres at +0.5, so the transform is shifted by half a pixel either side;
    the result then matches :func:`treeskel.align.warp_mask` exactly for masks
    (``nearest=True``) and :func:`treeskel.align.warp_image` to within about
    one grey level. Pixels outside the input are 0 (transparent for RGBA).

    Args:
        image: ``uint8`` image, ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``.
        matrix: 3x3 winter-to-summer ``(x, y)`` transform.
        nearest: Nearest-neighbour instead of bilinear sampling.
        out_size: Output ``(width, height)``.
    """
    inv = _HALF @ np.linalg.inv(matrix) @ np.linalg.inv(_HALF)
    inv = inv / inv[2, 2]
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(image).transform(out_size, Image.Transform.PERSPECTIVE,
                                                       tuple(inv.ravel()[:8]), resample))


class OverlayAnnotator:
    """State and event handling for the overlay tool, independent of display.

    Args:
        trees: Trees to align, in order.
        path: JSON file to read and write.
        redo: Also show trees already aligned (starting from their transform).
        clicks_path: Winter clicks, for the fork (the pin point) and to trace
            a tree's mask if it is not in ``gt_dir``.
        gt_dir: Folder of ``<tree>_winter_gt.png`` masks.
        clock: Time source in seconds (for tests).
    """

    def __init__(self, trees: Sequence[TreeRecord], path: str | Path, redo: bool = False,
                 clicks_path: str | Path = "annotations/winter_clicks.json",
                 gt_dir: str | Path | None = "outputs/winter_gt_guided",
                 clock: Callable[[], float] = time.monotonic):
        self.path = path
        self.saved = load_overlays(path)
        self.trees = [t for t in trees if redo or t.tree_id not in self.saved]
        self.winter_clicks = load_clicks(clicks_path)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.clock = clock
        self.i = 0
        self.view = "blend"             # "blend", "winter" or "summer"
        self.alpha = 0.5
        self.show_mask = True
        self.done = False
        self._load()

    # --- current tree -------------------------------------------------------
    @property
    def tree(self) -> TreeRecord | None:
        return self.trees[self.i] if self.i < len(self.trees) else None

    def start(self) -> Overlay:
        """Starting transform: the saved one, else the median of the row's trees, else of all trees."""
        tid = self.tree.tree_id
        old = self.saved.get(tid)
        if old is not None and not old.skip:
            return replace(old)
        done = [o for t, o in self.saved.items() if not o.skip and t != tid]
        row = [o for t, o in self.saved.items() if not o.skip and t != tid and t[:3] == tid[:3]]
        return median_overlay(row if len(row) >= 2 else done)

    def _load(self) -> None:
        """Load the current tree's images, mask and pin point, and its starting transform."""
        self.history: list[Overlay] = []
        if self.tree is None:
            self.done = True
            return
        tid = self.tree.tree_id
        self.winter = load_tree(self.tree, "winter")
        self.summer = load_tree(self.tree, "summer")
        self.mask = load_winter_mask(tid, self.winter, self.winter_clicks, self.gt_dir)
        # Warped on every change, so prepared once: the photo with full opacity, and the mask's outline.
        self._rgba = np.dstack([self.winter["rgb"], np.full(self.winter["rgb"].shape[:2], 255, np.uint8)])
        self._outline = outline(self.mask, 2).astype(np.uint8) * 255 if self.mask is not None else None
        c = self.winter_clicks.get(tid)
        self.pin = np.array(c.fork[::-1], float) if c is not None and c.complete and not c.skip \
            else np.array([320.0, 240.0])
        self.initial = self.start()
        self.ov = replace(self.initial)
        old = self.saved.get(tid)
        self.flag = bool(old.mask_flag) if old is not None else False
        self.spent_before = old.seconds if old is not None else 0.0
        self.t0 = self.clock()

    # --- the transform ------------------------------------------------------
    def matrix(self) -> np.ndarray:
        """The current winter-to-summer transform."""
        return self.ov.matrix()

    def composite(self) -> np.ndarray:
        """The picture to show: the summer photo with the moved winter photo and mask on top.

        In the ``"blend"`` view the winter photo covers the summer one at
        ``alpha``, in ``"winter"`` fully, and in ``"summer"`` not at all; the
        part of the summer photo the winter one does not cover always shows
        through. The mask's outline is drawn in magenta if ``show_mask``.
        Blending here into one picture, rather than layering images in the
        window, keeps each update fast.

        Returns:
            ``(480, 640, 3)`` uint8 image.
        """
        m = self.matrix()
        a = {"blend": self.alpha, "winter": 1.0, "summer": 0.0}[self.view]
        out = self.summer["rgb"]
        if a > 0:
            w = fast_warp(self._rgba, m).astype(np.float32)
            k = w[..., 3:] * (a / 255.0)
            out = (out * (1 - k) + w[..., :3] * k + 0.5).astype(np.uint8)
        if self.show_mask and self._outline is not None:
            line = fast_warp(self._outline, m, nearest=True) > 0
            out = out.copy() if out is self.summer["rgb"] else out
            out[line] = (255, 0, 255)
        return out

    def _change(self, pin_xy: np.ndarray | None = None, **params) -> None:
        """Set some parameters, keeping the winter point ``pin_xy`` where it was in the summer view."""
        self.history.append(replace(self.ov))
        before = apply_transform(self.matrix(), [pin_xy])[0] if pin_xy is not None else None
        for k, v in params.items():
            setattr(self.ov, k, float(v))
        if before is not None:
            after = apply_transform(self.matrix(), [pin_xy])[0]
            self.ov.dx += before[0] - after[0]
            self.ov.dy += before[1] - after[1]

    def shift(self, ddx: float, ddy: float) -> None:
        """Move the winter photo by ``(ddx, ddy)`` px."""
        self._change(dx=self.ov.dx + ddx, dy=self.ov.dy + ddy)

    def rescale(self, factor: float, about: tuple[float, float] | None = None) -> None:
        """Scale by ``factor``, keeping the summer point ``about`` (``(x, y)``) fixed, or the fork."""
        q = self.pin if about is None else apply_transform(np.linalg.inv(self.matrix()), [about])[0]
        self._change(pin_xy=q, scale=self.ov.scale * factor)

    def rotate(self, degrees: float) -> None:
        """Roll by ``degrees`` (clockwise), keeping the fork fixed."""
        self._change(pin_xy=self.pin, roll=self.ov.roll + degrees)

    def tilt(self, pitch: float = 0.0, yaw: float = 0.0) -> None:
        """Add perspective (degrees), keeping the fork fixed."""
        self._change(pin_xy=self.pin, pitch=self.ov.pitch + pitch, yaw=self.ov.yaw + yaw)

    # --- events -------------------------------------------------------------
    def prompt(self) -> str:
        """Current tree, transform and help, for the window title."""
        if self.done:
            return "All trees done - press q to save and quit"
        o = self.ov
        spent = self.spent_before + self.clock() - self.t0
        return (f"{self.tree.tree_id} ({self.i + 1}/{len(self.trees)})  {self.view}"
                + (f" {int(100 * self.alpha)}%" if self.view == "blend" else "")
                + ("  MASK FLAGGED" if self.flag else "") + f"  {spent:.0f} s\n"
                f"shift {o.dx:+.1f}, {o.dy:+.1f} px   scale {o.scale:.3f}   roll {o.roll:+.1f}   "
                f"pitch {o.pitch:+.1f}   yaw {o.yaw:+.1f} deg\n"
                "drag shift, scroll scale, arrows, [ ] roll, - = scale, i k j l tilt, space flip, b blend, "
                "m mask | Enter accept, u undo, r reset, s skip, f flag mask, n/p, q quit")

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
        elif key in ("[", "]", "{", "}"):
            step = STEPS["roll_big"] if key in "{}" else STEPS["roll"]
            self.rotate(-step if key in "[{" else step)
        elif key in ("-", "=", "+"):
            self.rescale(1 / STEPS["scale"] if key == "-" else STEPS["scale"])
        elif key in ("i", "k"):
            self.tilt(pitch=STEPS["tilt"] if key == "i" else -STEPS["tilt"])
        elif key in ("j", "l"):
            self.tilt(yaw=-STEPS["tilt"] if key == "j" else STEPS["tilt"])
        elif key == " ":
            self.view = "summer" if self.view == "winter" else "winter"
        elif key == "b":
            self.view = "blend"
        elif key in tuple("123456789"):
            self.view, self.alpha = "blend", int(key) / 10
        elif key == "m":
            self.show_mask = not self.show_mask
        elif key == "f":
            self.flag = not self.flag
        elif key == "u" and self.history:
            self.ov = self.history.pop()
        elif key == "r":
            self._change(**self.initial.params())
        elif key == "enter":
            self._save(replace(self.ov, skip=False))
        elif key == "s":
            self._save(Overlay(skip=True, note="skipped in the overlay tool"))
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


HELP = ("drag: shift   scroll: scale   arrows: 1 px (shift: 10)   [ ]: roll   - =: scale   i k j l: tilt   "
        "space: flip   b, 1-9: blend   m: mask\n"
        "Enter: accept   u: undo   r: reset   s: skip   f: flag the winter mask   n / p: next / previous   q: quit")
"""Key help shown above the picture (the current tree and transform are in the window's title bar)."""


def run_gui(ann: OverlayAnnotator, show: bool = True):
    """Open the window and connect mouse and keyboard events.

    For speed, the window is drawn in full only when needed (a new tree, a
    zoom, a resize) and cached without the picture; each change then only
    draws :meth:`OverlayAnnotator.composite` over the cache ("blitting").
    The current tree and transform go in the window's title bar, which the
    system draws.

    Args:
        ann: The annotator.
        show: Block in ``plt.show()`` (False only for tests).

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward", "keymap.pan",
              "keymap.xscale", "keymap.yscale", "keymap.fullscreen"):
        plt.rcParams[k] = []                        # free the tool's keys (s, q, arrows, k, l, f, ...)
    plt.rcParams["keymap.home"] = ["h", "home"]     # and r (reset); h still resets the zoom
    fig, ax = plt.subplots(figsize=(10, 8.2))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.93)
    ax.set_axis_off()
    fig.suptitle(HELP, fontsize=8)
    picture = ax.imshow(np.zeros((480, 640, 3), np.uint8), interpolation="nearest", animated=True)
    canvas = fig.canvas
    st = {"tree": None, "drag": None, "bg": None}

    def on_draw(ev) -> None:
        """After a full draw: cache it (without the picture), then draw the picture on top."""
        st["bg"] = canvas.copy_from_bbox(fig.bbox)
        fig.draw_artist(picture)

    def status() -> None:
        manager = getattr(canvas, "manager", None)
        if manager is not None:
            lines = ann.prompt().splitlines()
            manager.set_window_title(lines[0] if ann.done else f"{lines[0]}   |   {lines[1]}")

    def update() -> None:
        status()
        if ann.done:
            picture.set_data(np.zeros((480, 640, 3), np.uint8))
            canvas.draw_idle()
            return
        picture.set_data(ann.composite())
        if st["tree"] != ann.tree.tree_id:          # a new tree: show all of it (a full draw)
            ax.set_xlim(-0.5, 639.5)
            ax.set_ylim(479.5, -0.5)
            st["tree"] = ann.tree.tree_id
            canvas.draw_idle()
        elif st["bg"] is None:
            canvas.draw_idle()
        else:                                        # the same view: only the picture changes
            canvas.restore_region(st["bg"])
            fig.draw_artist(picture)
            canvas.blit(fig.bbox)

    def tool_active() -> bool:
        toolbar = getattr(canvas, "toolbar", None)
        return toolbar is not None and bool(getattr(toolbar, "mode", ""))

    def on_press(ev) -> None:
        if ann.done or ev.button != 1 or ev.inaxes is not ax or ev.xdata is None or tool_active():
            return
        ann.history.append(replace(ann.ov))
        st["drag"] = (ev.xdata, ev.ydata, ann.ov.dx, ann.ov.dy)

    def on_motion(ev) -> None:
        if st["drag"] is None or ev.xdata is None:
            return
        x0, y0, dx0, dy0 = st["drag"]
        ann.ov.dx, ann.ov.dy = dx0 + ev.xdata - x0, dy0 + ev.ydata - y0
        update()

    def on_release(ev) -> None:
        if st["drag"] is None:
            return
        st["drag"] = None
        if ann.history and ann.history[-1] == ann.ov:
            ann.history.pop()                        # a click without a drag changes nothing
        update()

    def on_scroll(ev) -> None:
        if ann.done or ev.inaxes is not ax or ev.xdata is None:
            return
        ann.rescale(STEPS["scroll"] if ev.button == "up" else 1 / STEPS["scroll"], about=(ev.xdata, ev.ydata))
        update()

    def on_key(ev) -> None:
        if ev.key is None:
            return
        ann.key(ev.key)
        if ev.key == "q":
            plt.close(fig)
        else:
            update()

    for name, fn in (("draw_event", on_draw), ("button_press_event", on_press),
                     ("motion_notify_event", on_motion), ("button_release_event", on_release),
                     ("scroll_event", on_scroll), ("key_press_event", on_key)):
        canvas.mpl_connect(name, fn)
    update()
    if show:  # pragma: no cover - needs a display
        plt.show()
    return fig


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a display
    """Start the overlay tool (see the module docstring for usage)."""
    from treeskel.splits import load_split

    ap = argparse.ArgumentParser(prog="python -m treeskel.annotate_overlay", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="annotations/season_overlay.json", help="transforms file (read and written)")
    ap.add_argument("--rows", type=int, nargs="+", help="only these rows")
    ap.add_argument("--trees", nargs="+", help="only these tree ids")
    ap.add_argument("--dev-only", action="store_true", help="skip the test rows listed in --split")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--redo", action="store_true", help="also show trees already aligned")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--winter-gt", default="outputs/winter_gt_guided",
                    help="folder of <tree>_winter_gt.png masks; traced from the clicks if missing")
    args = ap.parse_args(argv)

    trees = index_dataset(args.data_dir).trees
    if args.rows:
        trees = [t for t in trees if t.row in args.rows]
    if args.trees:
        trees = [t for t in trees if t.tree_id in set(args.trees)]
    if args.dev_only:
        test_rows, _ = load_split(args.split, index_dataset(args.data_dir).trees)
        trees = [t for t in trees if t.row not in test_rows]
    ann = OverlayAnnotator(trees, args.out, redo=args.redo, clicks_path=args.clicks, gt_dir=args.winter_gt)
    print(f"{len(ann.trees)} trees to align; transforms saved to {args.out}")
    run_gui(ann)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
