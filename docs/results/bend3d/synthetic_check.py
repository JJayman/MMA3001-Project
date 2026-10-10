"""Known answers for treeskel.bend3d: synthetic trees rendered as noisy 8-bit depth (10 noise seeds each).

For each case: the V opening of the bent leaders minus the plane's (and the true difference), and the errors
of the trunk's angle out of the V's plane and of the left leader's bow against the true 3D curve. Uses the
synthetic tree of ``tests/test_bend3d.py``. Run from the repository root:

```
python docs/results/bend3d/synthetic_check.py    # -> docs/results/bend3d/synthetic_check_output.txt
```
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

from test_bend3d import PARAMS, bent_tree, true_measures  # noqa: E402

from treeskel.bend3d import bend_measures, fit_bends  # noqa: E402

CASES = (("flat; lean 9, turn -5; noise 0.7 units", {}),
         ("flat; noise 1.2 units", {"noise": 1.2}),
         ("flat; lean 15", {"lean": 15.0}),
         ("trunk 20 deg out, leaders flat", {"trunk_tilt": 20.0}),
         ("trunk 20 deg out, left bow 5 units", {"trunk_tilt": 20.0, "bow": 5.0}),
         ("trunk -15 deg out, left bow -4 units", {"trunk_tilt": -15.0, "bow": -4.0}))

lines = [f"Prior tau {PARAMS.spread * 100:g} cm, L {PARAMS.length * 100:g} cm; 10 seeds each; mean ± SD.",
         f"{'case':40s} {'V bent - plane':>16s} {'(true)':>7s} {'trunk-out error':>16s} {'bow error (cm)':>15s} "
         f"{'true trunk out':>15s} {'true bow':>9s}"]
for label, kw in CASES:
    vc, tv, te, be, tt, tb = [], [], [], [], [], []
    for seed in range(10):
        paths, rad, depth, valid, tz = bent_tree(seed=seed, **kw)
        prep, truth = true_measures(paths, rad, depth, valid, tz)
        h = fit_bends(prep, PARAMS)["h"]
        got, flat = bend_measures(prep, h, params=PARAMS), bend_measures(prep, 0 * h, params=PARAMS)
        vc.append(got["v_opening"] - flat["v_opening"])
        tv.append(truth["v_opening"] - flat["v_opening"])
        te.append(got["trunk_out"] - truth["trunk_out"])
        be.append(got["left_bow"] - truth["left_bow"])
        tt.append(truth["trunk_out"])
        tb.append(truth["left_bow"])
    lines.append(f"{label:40s} {np.mean(vc):+7.2f} ± {np.std(vc):4.2f}   {np.mean(tv):+6.2f} {np.mean(te):+8.2f} ± "
                 f"{np.std(te):4.2f}   {np.mean(be):+6.2f} ± {np.std(be):4.2f}   {np.mean(tt):+13.1f}   {np.mean(tb):+7.2f}")
print("\n".join(lines))
(Path(__file__).parent / "synthetic_check_output.txt").write_text("\n".join(lines) + "\n")
