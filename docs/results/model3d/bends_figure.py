"""The figure of the bent summer models: each example tree from the front and from the side.

From the front a bent model looks exactly like the flat one (every point stays on its own ray); from the side
the trunk and leaders leave the tree's plane where the summer depth sees the wood. The two examples are
``R01N13`` (chosen by a rule: among the development trees in the repository whose U-Net trunk is drawn as
seen, the one whose summer-against-winter trunk angle error is nearest the median of all 214 such trees) and
``R15N05`` (the example since session 13, chosen before bends existed: little of its trunk is seen). Both use
the 5-fold run's masks. Run from the repository root:

```
python docs/results/model3d/bends_figure.py      # -> docs/figures/model3d_bends.png
```
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from treeskel.bend3d import behind_plane, bent_points, fit_bends, mask_skeleton, posterior_sd, prepare  # noqa: E402
from treeskel.depth3d import METRES_PER_UNIT, Camera  # noqa: E402
from treeskel.io import index_dataset, load_tree  # noqa: E402
from treeskel.model3d import SAVED_MASKS, bend_params, to_view  # noqa: E402
from treeskel.segment import load_masks  # noqa: E402
from treeskel.shape_model import AXIS, INK, SECONDARY, SURFACE, _style  # noqa: E402
from treeskel.tree3d import PARTS  # noqa: E402

SEEN, UNSEEN = "#94480c", "#8e8c85"   # validated pair: normal-vision ΔE 18.9, both ≥ 3:1 on the surface
TREES = (("R01N13", "data/tuning"), ("R15N05", "data/sample"))

params = bend_params(ROOT / "docs/results/bend3d/report.json")
masks = load_masks(ROOT / SAVED_MASKS)
fig, axes = plt.subplots(len(TREES), 2, figsize=(8.6, 4.3 * len(TREES)), gridspec_kw={"width_ratios": [1.5, 1]})
fig.patch.set_facecolor(SURFACE)
for (tree, folder), (front, side) in zip(TREES, axes):
    rec = {t.tree_id: t for t in index_dataset(ROOT / folder).trees}[tree]
    summer = load_tree(rec, "summer")
    sk = mask_skeleton(masks[tree])
    prep = prepare(sk["paths"], summer["depth"], summer["valid"], sk["radii"], Camera(), ridge=True,
                   scale=METRES_PER_UNIT)
    fit = fit_bends(prep, params)
    unsure = posterior_sd(fit) > params.uncertain * params.spread
    bent = bent_points(prep, fit["h"])
    fork = bent[0]
    seen_share = float(np.mean(~unsure[prep.nodes["trunk"]]))
    read_share = float(np.mean(prep.anchor[prep.nodes["trunk"]]))
    behind = behind_plane(prep, bent) * 100                    # cm behind the tree's plane: the flat model is 0
    side.axvline(0, color=AXIS, lw=1.2, zorder=1)
    for p in PARTS:
        k = prep.nodes[p]
        b = to_view(bent[k], fork)
        u = unsure[k]
        for ax, x, y in ((front, b[:, 0], b[:, 1]), (side, behind[k], b[:, 1])):
            # Seen and unseen runs drawn separately, each run joined to the next so the line is unbroken.
            edges = np.flatnonzero(np.diff(u.astype(int))) + 1
            for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(u)]):
                sl = slice(max(lo - 1, 0), hi)
                ax.plot(x[sl], y[sl], color=UNSEEN if u[lo] else SEEN, lw=2.0, ls=(0, (3, 2)) if u[lo] else "-",
                        solid_capstyle="round", zorder=2)
    front.set_aspect("equal")
    front.set_xlabel("m, across the photo", fontsize=8, color=SECONDARY)
    front.set_ylabel("m, up from the fork", fontsize=8, color=SECONDARY)
    front.set_title(f"{tree}, summer: from the front (as the photo)", fontsize=9, color=INK, loc="left")
    side.set_xlabel("cm behind the tree's plane (camera to the left; stretched)", fontsize=8, color=SECONDARY)
    side.set_title(f"from the side: trunk read {read_share:.0%}, drawn as seen {seen_share:.0%}", fontsize=9,
                   color=INK, loc="left")
    side.set_ylim(front.get_ylim())
    lim = max(10.0, float(np.max(np.abs(behind))) * 1.15)
    side.set_xlim(-lim, lim)
    for ax in (front, side):
        _style(ax)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(AXIS)
handles = [Line2D([], [], color=SEEN, lw=2, label="depth seen (summer ridge readings)"),
           Line2D([], [], color=UNSEEN, lw=2, ls=(0, (3, 2)), label="not seen: kept near the plane, uncertain"),
           Line2D([], [], color=AXIS, lw=1.2, label="the tree's plane (the flat model, --flat)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=SECONDARY)
fig.suptitle("Summer 3D models with bends (model3d build; 5-fold U-Net masks)", x=0.01, ha="left", fontsize=10,
             color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = ROOT / "docs/figures/model3d_bends.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"Saved {out.relative_to(ROOT)}")
