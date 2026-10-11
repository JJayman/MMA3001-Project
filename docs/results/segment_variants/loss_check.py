"""How much more the centre-line loss charges than the Dice loss for a short leader or a gap (session 16).

Before any training run: on each development hand mask in the repository (the 21 local trees), the left
leader is cut on purpose, either from halfway along its centre line to the tip (a leader that stops short)
or for 8 px of centre line 60 px from the fork (a gap). The cut mask is scored against the whole one with the
soft Dice loss and the soft clDice loss (treeskel.segment.soft_cldice_loss). Uses no winter data and no
network. Needs PyTorch.

    python docs/results/segment_variants/loss_check.py > docs/results/segment_variants/loss_check_output.txt
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from treeskel.guided_gt import arc_length
from treeskel.io import index_dataset, load_mask
from treeskel.segment import soft_cldice_loss
from treeskel.shape import mask_branches

ROOT = Path(__file__).resolve().parents[3]


def cut(mask: np.ndarray, br: dict, part: np.ndarray) -> np.ndarray:
    """The mask without the pixels nearest to ``part`` of the centre lines (trunk and leaders)."""
    lines = np.concatenate([br["trunk"], br["left"], br["right"]])
    seeds = np.zeros_like(mask)
    seeds[tuple(lines.T)] = True
    _, (iy, ix) = ndi.distance_transform_edt(~seeds, return_indices=True)
    drop = np.zeros_like(mask)
    drop[tuple(np.asarray(part).T)] = True
    return mask & ~drop[iy, ix]


def losses(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(soft Dice loss, soft clDice loss) of two binary masks, smoothed by 1 as in training."""
    p, t = torch.from_numpy(pred.astype(np.float32))[None], torch.from_numpy(truth.astype(np.float32))[None]
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return float(dice), float(soft_cldice_loss(p, t)[0])


def main() -> None:
    test_rows = set(json.loads((ROOT / "splits.json").read_text())["test_rows"])
    recs = {t.tree_id: t for d in ("sample", "tuning") for t in index_dataset(ROOT / "data" / d).trees}
    short_ratio, gap_ratio = [], []
    print(f"{'tree':8s} {'short: Dice':>12s} {'clDice':>7s} {'ratio':>6s}   {'gap: Dice':>10s} {'clDice':>7s} {'ratio':>6s}")
    for tree, rec in sorted(recs.items()):
        if rec.row in test_rows or not rec.has_mask:
            continue
        m = ndi.binary_fill_holes(load_mask(rec.paths[("summer", "mask")]))
        br = mask_branches(m)
        s = arc_length(br["left"])
        a = losses(cut(m, br, br["left"][s >= s[-1] / 2]), m)
        b = losses(cut(m, br, br["left"][(s >= 60) & (s < 68)]), m)
        short_ratio.append(a[1] / a[0])
        gap_ratio.append(b[1] / b[0])
        print(f"{tree:8s} {a[0]:12.4f} {a[1]:7.4f} {a[1] / a[0]:6.2f}   {b[0]:10.4f} {b[1]:7.4f} {b[1] / b[0]:6.2f}")
    sr, gr = np.array(short_ratio), np.array(gap_ratio)
    print(f"\n{len(sr)} development hand masks. clDice / Dice, median [range]:")
    print(f"  left leader stopping halfway: {np.median(sr):.2f} [{sr.min():.2f}, {sr.max():.2f}]")
    print(f"  8 px gap in the left leader:  {np.median(gr):.2f} [{gr.min():.2f}, {gr.max():.2f}]")
    w = 0.5
    print(f"With the loss's weight {w}: a short leader costs {(1 - w) + w * np.median(sr):.2f} times what it costs "
          f"in the Dice term alone; a gap {(1 - w) + w * np.median(gr):.2f} times.")


if __name__ == "__main__":
    main()
