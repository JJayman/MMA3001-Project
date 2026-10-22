"""Audit a representative sample of the lined-up hand masks for local misalignment (session 16).

The student lined up all 449 development hand masks in session 11 by moving
each one as a whole (:mod:`treeskel.annotate_mask`), which raised the U-Net's
V-opening skill from +0.06 to +0.72: label position was the largest gain so
far. A shift cannot fix a mask whose trunk sits on the wood while a leader is
drawn beside it. This audit measures how common such *local* misalignment is
in the labels the network trains on.

**The sample** (:func:`draw_sample`): 40 development trees, drawn at random
within each orchard row in proportion to its trees (at least one per row),
with a fixed seed, before any tree is looked at. Rows differ in light and
capture, so the sample covers all 16 development rows. It needs only the
committed files (``splits.json``, ``data/shape/shape_measures.csv``,
``annotations/mask_overlay.json``), so it is written and committed first
(``docs/results/label_audit/sample.csv``), and the verdicts are filled into
that file. 40 trees tell a rare problem (under 10% of leaders) from a common
one (over 30%): the 95% interval of a share is about ±10-15 points.

**The verdicts** (the student's, by eye; one per part: ``trunk``, ``left``,
``right``, the left leader being the one whose tip is further left):

- ``on``: where the wood is visible, the mask's outline follows it (within
  about half the branch's width);
- ``off``: along a stretch at least a trunk's width long, the mask lies beside
  the visible wood by about the branch's width or more;
- ``hidden``: too little of that part's wood is visible to judge.

**The images** (:func:`draw_panel`, the student's PC): the photo's centre
480x480 (the hand masks' frame), at full resolution, beside the same with the
lined-up mask's outline. The predicted masks are not shown, so they cannot
sway the verdicts.

**The summary** (:func:`summarise`): for each part, the share judged ``off``
among the parts that could be judged, with Wilson 95% intervals; trees with
any part off. It changes nothing by itself: what follows from it is in
``docs/results/segment_variants/rule.md``.

Usage, from the command line:

```
python -m treeskel.label_audit sample            # the sample, docs/results/label_audit/sample.csv (done once)
python -m treeskel.label_audit judge data/raw    # judge them in a window: o on, f off, h hidden (the student's PC)
python -m treeskel.label_audit draw data/raw     # or only the images, outputs/label_audit/, and fill in the table
python -m treeskel.label_audit summary           # after the verdicts are filled in: report.json
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLE_SIZE = 40
"""Trees in the audit sample (see the module docstring); fixed before the sample was drawn."""

SEED = 16
"""The sample's random seed: the session's number, fixed before the sample was drawn."""

PARTS = ("trunk", "left", "right")
"""The parts judged in each tree."""

VERDICTS = ("on", "off", "hidden")
"""The verdicts allowed for each part (see the module docstring)."""

SAMPLE_CSV = Path("docs/results/label_audit/sample.csv")
"""Where the sample, and then the verdicts, are kept."""


def allocate(counts: dict[int, int], n: int) -> dict[int, int]:
    """Trees to draw from each row: in proportion to its trees, at least one each, summing to ``n``.

    Each row gets the whole part of its share, at least 1; then one tree at a
    time goes to the row still owed the most (its share minus what it has;
    ties to the lower row), so a row lifted to 1 gets no more until the
    others have caught up. If the lifts overshoot ``n``, trees are taken back
    from the rows given most beyond their share.

    Args:
        counts: Trees per row.
        n: Trees to draw in all.

    Returns:
        Trees to draw per row.

    Raises:
        ValueError: If ``n`` is fewer than the rows or more than the trees.

    Example:
        >>> allocate({1: 30, 2: 10, 3: 2}, 8)
        {1: 5, 2: 2, 3: 1}
    """
    total = sum(counts.values())
    if n < len(counts) or n > total:
        raise ValueError(f"cannot draw {n} trees from {len(counts)} rows of {total}")
    share = {r: n * c / total for r, c in counts.items()}
    out = {r: min(counts[r], max(1, math.floor(s))) for r, s in share.items()}
    while sum(out.values()) < n:
        r = max((r for r in out if out[r] < counts[r]), key=lambda r: (share[r] - out[r], -r))
        out[r] += 1
    while sum(out.values()) > n:
        r = max((r for r in out if out[r] > 1), key=lambda r: (out[r] - share[r], -r))
        out[r] -= 1
    return out


def draw_sample(trees_by_row: dict[int, Sequence[str]], n: int = SAMPLE_SIZE, seed: int = SEED) -> list[str]:
    """``n`` trees drawn at random within each row (:func:`allocate`), with ``seed``; sorted by tree id.

    Args:
        trees_by_row: The tree ids of each row.
        n: Trees to draw.
        seed: Random seed.

    Returns:
        The drawn tree ids, sorted.
    """
    rng = np.random.default_rng(seed)
    take = allocate({r: len(t) for r, t in trees_by_row.items()}, n)
    picked = []
    for r in sorted(trees_by_row):
        trees = sorted(trees_by_row[r])
        picked += [trees[i] for i in sorted(rng.choice(len(trees), take[r], replace=False))]
    return sorted(picked)


def development_trees(table: str | Path = "data/shape/shape_measures.csv", split: str | Path = "splits.json",
                      overlay: str | Path = "annotations/mask_overlay.json") -> dict[int, list[str]]:
    """The development trees whose hand mask is lined up (not skipped), by row: the network's training labels.

    Args:
        table: The winter shape table, whose trees are the development trees.
        split: The fixed split (its test rows).
        overlay: The hand masks' line-ups (skipped trees are left out).

    Returns:
        The tree ids per row.

    Raises:
        ValueError: If the table holds a test-row tree.
    """
    from treeskel.summer_gt import load_overlays

    test_rows = set(json.loads(Path(split).read_text())["test_rows"])
    overlays = load_overlays(overlay)
    by_row: dict[int, list[str]] = {}
    with open(table, newline="") as f:
        for r in csv.DictReader(f):
            row, tree = int(r["row"]), r["tree"]
            if row in test_rows:
                raise ValueError(f"{table} holds test-row tree {tree}; the audit uses development trees only")
            if tree in overlays and not overlays[tree].skip:
                by_row.setdefault(row, []).append(tree)
    return by_row


def write_template(path: Path, trees: Sequence[str]) -> None:
    """The sample as a table to fill in: ``tree, row, trunk, left, right, notes`` (verdicts empty; the row is
    the tree id's R number).

    Args:
        path: The CSV file to write.
        trees: The sample's tree ids.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", *PARTS, "notes"])
        for t in trees:
            w.writerow([t, int(t[1:3]), "", "", "", ""])


def read_verdicts(path: Path) -> list[dict[str, str]]:
    """The filled-in table; every verdict must be one of :data:`VERDICTS` or empty (not judged yet).

    Args:
        path: The verdict table (``docs/results/label_audit/sample.csv``).

    Returns:
        The rows as dicts, verdicts as given.

    Raises:
        ValueError: A verdict that is not allowed (the row and value are named).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel may add a byte-order mark
        rows = list(csv.DictReader(f))
    for r in rows:
        for p in PARTS:
            v = (r.get(p) or "").strip().lower()
            if v and v not in VERDICTS:
                raise ValueError(f"{r['tree']}: {p} is {r[p]!r}; use one of {', '.join(VERDICTS)}")
            r[p] = v
    return rows


def write_verdicts(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Save the table with its verdicts (the columns of :func:`write_template`), through a temporary file, so
    that a failed write never leaves half a table.

    Args:
        path: The verdict table (``docs/results/label_audit/sample.csv``).
        rows: The table's rows.
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", *PARTS, "notes"])
        for r in rows:
            w.writerow([r["tree"], r["row"], *(r.get(p, "") for p in PARTS), r.get("notes", "")])
    tmp.replace(path)


KEYS = {"o": "on", "f": "off", "h": "hidden"}
"""The judging keys (:class:`AuditJudge`): o for on, f for off, h for hidden."""

JUDGE_HELP = ("For the part asked: o = on the wood   f = off (beside the visible wood by about its width or more)   "
              "h = hidden (too little wood visible)\n"
              "u or backspace = undo (also back into the previous tree)    n / p = next / previous tree    "
              "q = quit   (every key is saved at once; run again to carry on)")


class AuditJudge:
    """Steps through the sample, asking for the trunk, then the left and the right leader of each tree.

    Every key goes through :meth:`key`, so the logic is tested without a
    window. The table is saved after every verdict, and a new start opens
    the first tree with a part still unjudged.

    Args:
        path: The sample's table (``sample.csv``).
    """

    def __init__(self, path: Path):
        """Read the table and go to the first part not yet judged."""
        self.path = Path(path)
        self.rows = read_verdicts(self.path)
        self.message = ""
        self.done = False
        self.index, self.part = self._first_open()

    def _first_open(self) -> tuple[int, int]:
        """The first tree and part without a verdict (past the end if all are judged)."""
        for i, r in enumerate(self.rows):
            for j, p in enumerate(PARTS):
                if not r[p]:
                    return i, j
        return len(self.rows), 0

    @property
    def tree(self) -> str | None:
        """The tree being judged, or None when every tree is judged."""
        return self.rows[self.index]["tree"] if self.index < len(self.rows) else None

    def judged(self) -> int:
        """Trees with all three parts judged.

        Returns:
            The number of trees judged.
        """
        return sum(all(r[p] for p in PARTS) for r in self.rows)

    def prompt(self) -> str:
        """The window's title: the tree, the verdicts so far, and the part asked.

        Returns:
            The title, or a closing message once every tree is judged.
        """
        if self.tree is None:
            return f"All {len(self.rows)} trees judged. Press q to save and quit."
        r = self.rows[self.index]
        given = "   ".join(f"{p}: {r[p] or '-'}" for p in PARTS)
        names = {"trunk": "the TRUNK", "left": "the LEFT leader (tip further left)",
                 "right": "the RIGHT leader (tip further right)"}
        ask = (f"Judge {names[PARTS[self.part]]}" if self.part < len(PARTS)
               else "This tree is done: n for the next, u to change it")
        return (f"{self.tree}  ({self.index + 1}/{len(self.rows)}, {self.judged()} judged)    {given}\n{ask}"
                + (f"    {self.message}" if self.message else ""))

    def save(self) -> bool:
        """Write the table; False (with a message) if it cannot be written, e.g. open in Excel.

        Returns:
            True if the table was written.
        """
        try:
            write_verdicts(self.path, self.rows)
        except OSError as e:
            self.message = f"COULD NOT SAVE ({e.strerror or e}): close {self.path.name} elsewhere and press a key again"
            return False
        self.message = ""
        return True

    def key(self, key: str) -> None:
        """Handle one key press (see :data:`JUDGE_HELP`). Every change is saved at once, so a save that failed
        (the table open in another program) is tried again at the next key.

        Args:
            key: The key's name as matplotlib reports it.
        """
        k = (key or "").lower()
        if k == "q":
            self.done = self.save()
            return
        if k in ("n", "p"):
            self.index = min(max(self.index + (1 if k == "n" else -1), 0), max(len(self.rows) - 1, 0))
            r = self.rows[self.index]
            self.part = next((j for j, p in enumerate(PARTS) if not r[p]), len(PARTS))
            return
        if k in ("u", "backspace"):
            if self.part == 0 and self.index > 0:           # back into the previous tree
                self.index -= 1
                self.part = len(PARTS)
            if self.part > 0:
                self.part -= 1
                self.rows[self.index][PARTS[self.part]] = ""
                self.save()
            return
        if k in KEYS and self.tree is not None and self.part < len(PARTS):
            self.rows[self.index][PARTS[self.part]] = KEYS[k]
            self.part += 1
            self.save()
            if self.part == len(PARTS):                     # the tree is complete: on to the next open part
                self.index, self.part = self._first_open()
        elif self.message:
            self.save()                                     # any other key retries a failed save


def run_judge(judge: AuditJudge, image, show: bool = True):
    """The judging window: each tree's panel (:func:`draw_panel`), the prompt above it, keys to
    :meth:`AuditJudge.key`.

    Args:
        judge: The judge.
        image: ``image(tree) -> (H, W, 3) uint8`` panel of a tree.
        show: Block in ``plt.show()`` (False only for tests).

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.forward", "keymap.home", "keymap.pan",
              "keymap.zoom", "keymap.fullscreen", "keymap.xscale", "keymap.yscale", "keymap.grid"):
        plt.rcParams[k] = []                                # free the judging keys (o, f, h, n, p, q, u)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.86)
    ax.set_axis_off()
    fig.text(0.5, 0.995, JUDGE_HELP, ha="center", va="top", fontsize=8)
    title = fig.text(0.5, 0.93, "", ha="center", va="top", fontsize=12, fontweight="bold")
    shown = {"tree": None, "picture": None}

    def update() -> None:
        """Show the current tree's panel (if it changed) and the prompt."""
        t = judge.tree
        if t != shown["tree"]:
            shown["tree"] = t
            pic = image(t) if t is not None else np.full((10, 10, 3), 255, np.uint8)
            if shown["picture"] is None:
                shown["picture"] = ax.imshow(pic)
            else:
                shown["picture"].set_data(pic)
                shown["picture"].set_extent((-0.5, pic.shape[1] - 0.5, pic.shape[0] - 0.5, -0.5))
        title.set_text(judge.prompt())
        fig.canvas.draw_idle()

    def on_key(ev) -> None:
        """Pass a key to the judge, then redraw (or close after q)."""
        judge.key(ev.key)
        if judge.done:
            plt.close(fig)
            return
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._judge_on_key = on_key                              # for tests: the handler without a display
    update()
    if show:  # pragma: no cover - needs a display
        plt.show()
    return fig


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson's 95% interval for a share of ``k`` in ``n`` (NaN, NaN for n = 0).

    Args:
        k: Count of the outcome.
        n: Count of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        ``(low, high)`` shares, within [0, 1].

    Example:
        >>> [round(v, 3) for v in wilson(4, 40)]
        [0.04, 0.231]
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(rows: Sequence[dict[str, str]]) -> dict:
    """Per part: verdict counts, and the share ``off`` among the parts judged ``on`` or ``off`` (Wilson 95%).

    Args:
        rows: The verdict table's rows.

    Returns:
        ``{"trees", "judged_trees", "parts": {part: {"on", "off", "hidden", "share_off", "interval"}},
        "leaders": {...} (left and right together), "trees_any_off", "trees_any_off_share", "off_trees"}``.
    """
    done = [r for r in rows if all(r[p] for p in PARTS)]
    out: dict = {"trees": len(rows), "judged_trees": len(done), "parts": {}}

    def tally(values):
        """Counts of each verdict, and the share off among those judged on or off, with its interval."""
        c = {v: sum(1 for x in values if x == v) for v in VERDICTS}
        n = c["on"] + c["off"]
        c["share_off"] = c["off"] / n if n else float("nan")
        c["interval"] = list(wilson(c["off"], n))
        return c

    for p in PARTS:
        out["parts"][p] = tally([r[p] for r in done])
    out["leaders"] = tally([r[p] for r in done for p in ("left", "right")])
    off = [r["tree"] for r in done if any(r[p] == "off" for p in PARTS)]
    out["trees_any_off"] = len(off)
    out["trees_any_off_share"] = len(off) / len(done) if done else float("nan")
    out["trees_any_off_interval"] = list(wilson(len(off), len(done)))
    out["off_trees"] = off
    return out


def draw_panel(rgb: np.ndarray, mask: np.ndarray, label: str) -> Image.Image:
    """The photo's centre 480x480 beside the same with the mask's outline (magenta, 2 px), titled ``label``.

    Args:
        rgb: ``(480, 640, 3)`` summer photo.
        mask: ``(256, 256)`` bool hand mask in its frame (lined up).
        label: The title (tree id and position in the sample).

    Returns:
        The panel as a PIL image.
    """
    from PIL import ImageDraw
    from scipy import ndimage as ndi

    from treeskel.io import CROP_COLS

    crop = np.ascontiguousarray(rgb[:, CROP_COLS[0]:CROP_COLS[1]])
    side = crop.shape[0]
    big = np.asarray(Image.fromarray(np.asarray(mask, np.uint8) * 255).resize((side, side), Image.NEAREST)) > 0
    edge = big & ~ndi.binary_erosion(big, iterations=2)
    outlined = crop.copy()
    outlined[edge] = (255, 0, 255)
    head = 28
    panel = Image.new("RGB", (2 * side + 8, side + head), (255, 255, 255))
    panel.paste(Image.fromarray(crop), (0, head))
    panel.paste(Image.fromarray(outlined), (side + 8, head))
    ImageDraw.Draw(panel).text((6, 7), label, fill=(0, 0, 0))
    return panel


def main(argv: Sequence[str] | None = None) -> int:
    """``sample``: draw and save the sample; ``draw``: its images; ``summary``: the verdicts' summary.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if the sample already exists (``sample``), trees were not found in
        the data, or a file could not be written.
    """
    ap = argparse.ArgumentParser(prog="python -m treeskel.label_audit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="draw the sample and save it as the table to fill in")
    s.add_argument("--table", default="data/shape/shape_measures.csv")
    s.add_argument("--split", default="splits.json")
    s.add_argument("--mask-overlay", default="annotations/mask_overlay.json")
    d = sub.add_parser("draw", help="the sample's images (needs the photos)")
    d.add_argument("data_dirs", nargs="+")
    d.add_argument("--mask-overlay", default="annotations/mask_overlay.json")
    d.add_argument("--out", default="outputs/label_audit")
    j = sub.add_parser("judge", help="judge the sample in a window, key by key (needs the photos)")
    j.add_argument("data_dirs", nargs="+")
    j.add_argument("--mask-overlay", default="annotations/mask_overlay.json")
    m = sub.add_parser("summary", help="summarise the filled-in verdicts")
    m.add_argument("--report", default="docs/results/label_audit/report.json")
    for p in (s, d, j, m):
        p.add_argument("--sample", default=str(SAMPLE_CSV), help="the sample's table")
    args = ap.parse_args(argv)
    path = Path(args.sample)

    if args.command == "sample":
        if path.exists():
            print(f"{path} exists: the sample is drawn once. Delete it only to start the audit again.")
            return 1
        by_row = development_trees(args.table, args.split, args.mask_overlay)
        trees = draw_sample(by_row)
        write_template(path, trees)
        take = allocate({r: len(t) for r, t in by_row.items()}, SAMPLE_SIZE)
        print(f"{len(trees)} of {sum(map(len, by_row.values()))} lined-up development trees, from "
              f"{len(by_row)} rows ({', '.join(f'row {r}: {take[r]}' for r in sorted(take))}), seed {SEED}")
        print(f"Saved {path}")
        return 0

    if args.command == "draw":
        from treeskel.io import index_dataset, load_tree
        from treeskel.segment import hand_mask
        from treeskel.summer_gt import load_overlays

        rows = read_verdicts(path)
        records = {t.tree_id: t for dd in args.data_dirs for t in index_dataset(dd).trees}
        overlays = load_overlays(args.mask_overlay)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        failed = []
        for i, r in enumerate(rows, 1):
            t = r["tree"]
            if t not in records:
                failed.append(f"{t}: not in {', '.join(args.data_dirs)}")
                continue
            hm = hand_mask(records[t], overlays)
            if hm is None:
                failed.append(f"{t}: no lined-up hand mask")
                continue
            label = f"{t}  ({i}/{len(rows)})   left: photo   right: lined-up hand mask's outline"
            panel = draw_panel(load_tree(records[t], "summer")["rgb"], hm, label)
            target = out / f"{i:02d}_{t}.jpg"
            try:
                panel.save(target, quality=90)
            except OSError as e:                            # e.g. open in a viewer on Windows
                failed.append(f"{target}: {e.strerror or e}")
            print(f"  {i}/{len(rows)} {t}", flush=True)
        print(f"Saved the images in {out}/. Fill in {path}: trunk, left and right each on, off or hidden.")
        for f in failed:
            print(f"Could not draw {f}")
        return 1 if failed else 0

    if args.command == "judge":  # pragma: no cover - needs a display
        from treeskel.io import index_dataset, load_tree
        from treeskel.segment import hand_mask
        from treeskel.summer_gt import load_overlays

        judge = AuditJudge(path)
        records = {t.tree_id: t for dd in args.data_dirs for t in index_dataset(dd).trees}
        overlays = load_overlays(args.mask_overlay)
        missing = [r["tree"] for r in judge.rows if r["tree"] not in records]
        if missing:
            print(f"Not in {', '.join(args.data_dirs)}: {', '.join(missing)}")
            return 1
        order = [r["tree"] for r in judge.rows]

        def image(t):
            """The tree's panel: the photo beside the lined-up hand mask's outline."""
            label = f"{t}  ({order.index(t) + 1}/{len(order)})   left: photo   right: lined-up hand mask's outline"
            return np.asarray(draw_panel(load_tree(records[t], "summer")["rgb"], hand_mask(records[t], overlays), label))

        print(f"{judge.judged()} of {len(judge.rows)} trees judged so far. The window opens on the first open part.")
        run_judge(judge, image)
        judge.save()
        print(f"{judge.judged()} of {len(judge.rows)} trees judged; saved {path}. "
              "When all are done: python -m treeskel.label_audit summary")
        return 0

    rows = read_verdicts(path)
    s = summarise(rows)
    print(f"{s['judged_trees']} of {s['trees']} trees judged")
    for name, c in list(s["parts"].items()) + [("both leaders", s["leaders"])]:
        lo, hi = c["interval"]
        print(f"  {name:12s} on {c['on']:3d}  off {c['off']:3d}  hidden {c['hidden']:3d}   off {c['share_off']:.0%} "
              f"of those judged [95%: {lo:.0%}, {hi:.0%}]")
    lo, hi = s["trees_any_off_interval"]
    print(f"Trees with any part off: {s['trees_any_off']} ({s['trees_any_off_share']:.0%} [{lo:.0%}, {hi:.0%}])"
          + (f": {', '.join(s['off_trees'])}" if s["off_trees"] else ""))
    report = Path(args.report)
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(s, indent=2))
        print(f"Saved {report}")
    except OSError as e:
        print(f"Could not write {report}: {e.strerror or e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
