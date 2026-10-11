"""The ensemble's test-row run, like for like (session 16): every network scored on the same 92 trees.

The run's own report scores each network with ``evaluate``, which leaves out a tree whose mask has a fork
but no V opening (a stub). Seed 2's network left stubs on R03N10 and R16N12, so its printed score
(+0.88) is over 90 trees, without its two failures. Its comparison with the ensemble dropped the same two
trees, where the ensemble's own error is largest. Here every network's stub counts as the development mean
shape, as a mask with no fork always has, and as session 16's development comparisons count stubs
(``treeskel.seg_compare.run_table``). Everything comes from the saved files: no network is trained, and
nothing here can change the run.

    python docs/results/final_test_ensemble/like_for_like.py > docs/results/final_test_ensemble/like_for_like_output.txt
"""

import csv
import json
from pathlib import Path

import numpy as np

from treeskel.final_test import bootstrap_skill
from treeskel.seg_compare import paired

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    rep = json.loads((ROOT / "docs/results/final_test_ensemble/report.json").read_text())
    mean_v = rep["mean_shape"]["v_opening"]

    def read(path: Path) -> dict[str, tuple[float, float | None]]:
        """``{tree: (winter, model)}`` of a predictions.csv's V opening (model None if blank)."""
        out = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["measure"] == "v_opening" and r["winter"]:
                    out[r["tree"]] = (float(r["winter"]), float(r["model"]) if r["model"] else None)
        return out

    ens = read(ROOT / "docs/results/final_test_ensemble/predictions.csv")
    ft = read(ROOT / "docs/results/final_test/predictions.csv")
    trees = sorted(ens)
    winter = np.array([ens[t][0] for t in trees])
    preds = {"final test's network (session 12)": [ft[t][1] for t in trees],
             "ensemble (3 networks averaged)": [ens[t][1] for t in trees]}
    for m in rep["ensemble"]["members"]:
        preds[f"network seed {m['seed']} alone"] = [m["v_opening"].get(t) for t in trees]
    base = np.full(len(trees), mean_v)
    print(f"{len(trees)} test trees; a stub (fork, no V opening) or no fork counts as the development mean shape "
          f"({mean_v:.2f} deg)\n")
    print(f"  {'':36s} {'stubs':>5s} {'MAE':>6s} {'skill':>6s}  95% interval")
    filled = {}
    for name, p in preds.items():
        stubs = sum(v is None for v in p)
        v = np.array([mean_v if x is None else x for x in p], float)
        filled[name] = v
        mse, mse_b = np.mean((v - winter) ** 2), np.mean((base - winter) ** 2)
        lo, hi = bootstrap_skill(winter, v, base)
        print(f"  {name:36s} {stubs:5d} {np.mean(np.abs(v - winter)):6.2f} {1 - mse / mse_b:+6.2f}  [{lo:+.2f}, {hi:+.2f}]")
    singles = [k for k in filled if "alone" in k]
    err = {k: dict(zip(trees, np.abs(v - winter))) for k, v in filled.items()}
    avg = {t: float(np.mean([err[k][t] for k in singles])) for t in trees}
    c = paired(err["ensemble (3 networks averaged)"], avg)
    print(f"\nEnsemble against its networks' average, tree by tree: {c['mean_b']:.2f} -> {c['mean_a']:.2f}, "
          f"lower on {c['a_lower']:.0%} of {c['n']}, higher on {c['a_higher']:.0%}, p {c['p']:.3f}")
    c = paired(err["ensemble (3 networks averaged)"], err["final test's network (session 12)"])
    print(f"Ensemble against the final test's network: {c['mean_b']:.2f} -> {c['mean_a']:.2f}, lower on "
          f"{c['a_lower']:.0%}, higher on {c['a_higher']:.0%}, p {c['p']:.3f}")
    four = [filled[k] for k in filled if "ensemble" not in k]
    maes = [float(np.mean(np.abs(v - winter))) for v in four]
    print(f"\nFour single networks of the named recipe on the test rows: MAE {min(maes):.2f}-{max(maes):.2f}")
    print("\nThe trees where the networks disagree most (V opening, deg):")
    spread = np.std([filled[k] for k in singles], axis=0)
    for i in np.argsort(-spread)[:4]:
        t = trees[i]
        print(f"  {t}: winter {winter[i]:5.1f}; final test {filled[list(filled)[0]][i]:5.1f}; ensemble "
              f"{filled['ensemble (3 networks averaged)'][i]:5.1f}; networks "
              + ", ".join(f"{filled[k][i]:5.1f}{' (stub)' if preds[k][i] is None else ''}" for k in singles))


if __name__ == "__main__":
    main()
