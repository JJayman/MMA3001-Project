"""The label audit's guide picture: what "on" and "off" look like (session 16).

Top: R01N18 (a development tree in data/tuning) with its hand mask as the student lined it up; its
outline follows the trunk and both leaders where the wood shows. Bottom: the same, with the left leader's
part of the mask moved 18 mask px (about 34 photo px, two branch widths) sideways on purpose. That is
what "off" means. It is made up for the guide, not a real mask, and R01N18 is not in the audit's sample.

    python docs/results/label_audit/example.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from treeskel.guided_gt import arc_length
from treeskel.io import index_dataset, load_tree
from treeskel.label_audit import draw_panel
from treeskel.segment import hand_mask
from treeskel.shape import mask_branches
from treeskel.summer_gt import load_overlays

ROOT = Path(__file__).resolve().parents[3]
TREE = "R01N18"
SHIFT = 18                                                 # mask px to the right


def moved_leader(mask: np.ndarray, side: str = "left", shift: int = SHIFT, keep: float = 30.0) -> np.ndarray:
    """``mask`` with one leader's pixels (those nearest its centre line beyond ``keep`` px from the fork) moved
    ``shift`` px to the right."""
    m = ndi.binary_fill_holes(mask)
    br = mask_branches(m)
    paths = [np.asarray(br[p]) for p in ("trunk", "left", "right")]
    far = np.concatenate([np.zeros(len(paths[0]), bool),
                          arc_length(paths[1]) > keep if side == "left" else np.zeros(len(paths[1]), bool),
                          arc_length(paths[2]) > keep if side == "right" else np.zeros(len(paths[2]), bool)])
    pts = np.concatenate(paths)
    seeds = np.zeros(m.shape, bool)
    index = np.full(m.shape, -1)
    seeds[pts[:, 0], pts[:, 1]] = True
    index[pts[:, 0], pts[:, 1]] = np.arange(len(pts))
    _, (iy, ix) = ndi.distance_transform_edt(~seeds, return_indices=True)
    move = m & far[index[iy, ix]]
    out = m & ~move
    ys, xs = np.nonzero(move)
    out[ys, np.clip(xs + shift, 0, m.shape[1] - 1)] = True
    return out


def main() -> None:
    recs = {t.tree_id: t for d in ("sample", "tuning") for t in index_dataset(ROOT / "data" / d).trees}
    rec = recs[TREE]
    hand = hand_mask(rec, load_overlays(ROOT / "annotations/mask_overlay.json"))
    rgb = load_tree(rec, "summer")["rgb"]
    on = draw_panel(rgb, hand, f"ON  ({TREE} as lined up): where the wood shows, the outline follows it - "
                               "trunk on, left on, right on")
    off = draw_panel(rgb, moved_leader(hand), f"OFF  (made up for this guide: {TREE}'s left leader moved about two "
                                              "branch widths sideways) - the left leader would be off")
    sheet = Image.new("RGB", (on.width, on.height + off.height + 10), (255, 255, 255))
    sheet.paste(on, (0, 0))
    sheet.paste(off, (0, on.height + 10))
    path = ROOT / "docs/results/label_audit/on_vs_off_example.jpg"
    sheet.save(path, quality=88)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
