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
python -m treeskel.label_audit draw data/raw     # its images, outputs/label_audit/ (the student's PC)
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
    """``n`` trees drawn at random within each row (:func:`allocate`), with ``seed``; sorted by tree id."""
    rng = np.random.default_rng(seed)
    take = allocate({r: len(t) for r, t in trees_by_row.items()}, n)
    picked = []
    for r in sorted(trees_by_row):
        trees = sorted(trees_by_row[r])
        picked += [trees[i] for i in sorted(rng.choice(len(trees), take[r], replace=False))]
    return sorted(picked)


def development_trees(table: str | Path = "data/shape/shape_measures.csv", split: str | Path = "splits.json",
                      overlay: str | Path = "annotations/mask_overlay.json") -> dict[int, list[str]]:
    """The development trees whose hand mask is lined up (not skipped), by row: the network's training labels."""
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
    the tree id's R number)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "row", *PARTS, "notes"])
        for t in trees:
            w.writerow([t, int(t[1:3]), "", "", "", ""])


def read_verdicts(path: Path) -> list[dict[str, str]]:
    """The filled-in table; every verdict must be one of :data:`VERDICTS` or empty (not judged yet).

    Raises:
        ValueError: A verdict that is not allowed (the row and value are named).
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for p in PARTS:
            v = (r.get(p) or "").strip().lower()
            if v and v not in VERDICTS:
                raise ValueError(f"{r['tree']}: {p} is {r[p]!r}; use one of {', '.join(VERDICTS)}")
            r[p] = v
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson's 95% interval for a share of ``k`` in ``n`` (NaN, NaN for n = 0).

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
    """``sample``: draw and save the sample; ``draw``: its images; ``summary``: the verdicts' summary."""
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
    m = sub.add_parser("summary", help="summarise the filled-in verdicts")
    m.add_argument("--report", default="docs/results/label_audit/report.json")
    for p in (s, d, m):
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
