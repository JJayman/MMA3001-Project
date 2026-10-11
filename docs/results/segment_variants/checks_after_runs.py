"""Checks made after session 16's runs: is the rule's test sound across trainings? (session 16)

The rule (rule.md) compares trainings tree by tree with a Wilcoxon test, as the project's earlier rules
did. Its pre-declared null check compares samples of the same recipe. With the second ensemble's runs
(seeds 3-5) there are seven trainings of the named recipe, enough to apply the rule's exact test to two
triples of the same recipe. This script does that, then re-examines each verdict at the level of whole
trainings, and draws the figure.

Everything comes from the committed results (docs/results/segment_lined_up*_joined/ and
docs/results/segment_variants/trees.csv, whose hand-mask scores were computed on the student's PC). No
winter data chooses anything here; the test rows are not read.

    python docs/results/segment_variants/checks_after_runs.py > docs/results/segment_variants/checks_after_runs_output.txt
"""

import csv
import itertools
from pathlib import Path

import numpy as np

from treeskel.seg_compare import column, disagreement_check, joined, paired, run_table, stability, verdict

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "docs/results"
NAMED = "segment_lined_up"
BASE = ["segment_lined_up_rerun"] + [f"segment_lined_up_seed{i}" for i in range(1, 6)]
SINGLES = [NAMED] + BASE                                   # seven trainings of the named recipe
CL = ["segment_lined_up_cldice", "segment_lined_up_cldice_seed1", "segment_lined_up_cldice_seed2"]
ENS = {"segment_lined_up_ensemble": BASE[:3], "segment_lined_up_ensemble_b": BASE[3:],
       "segment_lined_up_cldice_ensemble": CL}


def short(name: str) -> str:
    """A run's name without the common prefix."""
    return name.replace("segment_lined_up", "") or "named"


def permutation(values: dict[str, float], group: list[str], others: list[str]) -> float:
    """One-sided run-level permutation p: how often a random ``len(group)`` of all the runs has a mean as low
    as ``group``'s (every way of choosing them)."""
    runs = group + others
    v = np.array([values[n] for n in runs])
    k = len(group)
    obs = v[:k].mean() - v[k:].mean()
    diffs = []
    for idx in itertools.combinations(range(len(runs)), k):
        m = np.zeros(len(runs), bool)
        m[list(idx)] = True
        diffs.append(v[m].mean() - v[~m].mean())
    return float(np.mean(np.array(diffs) <= obs + 1e-12))


def main() -> None:
    names = SINGLES + CL + list(ENS)
    T = {n: run_table(R / joined(n)) for n in names}
    mae = {n: float(np.mean([r["error"] for r in T[n].values()])) for n in names}
    incomplete = {n: int(sum(r["incomplete"] for r in T[n].values())) for n in names}
    tsens: dict[str, list[float]] = {}
    with open(R / "segment_variants/trees.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["tsens"]:
                tsens.setdefault(r["run"], []).append(float(r["tsens"]))
    cover = {n: float(np.mean(v)) for n, v in tsens.items()}

    print("Each training's joined masks (449 development trees; V MAE with stubs and no-fork masks as the mean shape)")
    print(f"  {'run':18s} {'V MAE':>6s} {'incomplete':>10s} {'coverage':>8s}")
    for n in names:
        print(f"  {short(n):18s} {mae[n]:6.3f} {incomplete[n]:10d} {cover.get(n, float('nan')):8.3f}")
    s = np.array([mae[n] for n in SINGLES])
    print(f"  named recipe, 7 trainings: V MAE {s.mean():.3f} ± {s.std(ddof=1):.3f} (SD), range {s.min():.3f}-{s.max():.3f}")

    print("\n1. The rule's test on samples of ONE recipe (it should call them different about 1 time in 20)")
    ps = [paired(column(T[a], "error"), column(T[b], "error"))["p"] for a, b in itertools.combinations(SINGLES, 2)]
    print(f"  single training against single training: {sum(p < 0.05 for p in ps)} of {len(ps)} pairs at p < 0.05")
    hits = 0
    splits = [(a, tuple(n for n in BASE if n not in a)) for a in itertools.combinations(BASE, 3)]
    splits = [(a, b) for a, b in splits if a < b]
    for a, b in splits:
        e = verdict([T[n] for n in a], [T[n] for n in b])["error"]
        hits += e["p"] < 0.05
        mark = "   <- the rule's baseline triple" if set(a) == set(BASE[:3]) or set(b) == set(BASE[:3]) else ""
        print(f"  {'+'.join(map(short, a)):22s} vs {'+'.join(map(short, b)):22s} {e['mean_a']:.3f} vs {e['mean_b']:.3f}  "
              f"p {e['p']:.3f}{mark}")
    print(f"  triple against triple, as the rule compares: {hits} of {len(splits)} splits at p < 0.05")

    print("\n2. The centre-line loss, again")
    for label, group in (("the rule's triple (re-run, seeds 1-2)", BASE[:3]), ("the other triple (seeds 3-5)", BASE[3:]),
                         ("all seven trainings", SINGLES)):
        v = verdict([T[n] for n in CL], [T[n] for n in group])
        e, g = v["error"], v["guards"]["incomplete"]
        print(f"  against {label:40s} V MAE {e['mean_b']:.3f} -> {e['mean_a']:.3f}, p {e['p']:.3f};  "
              f"incomplete {g['mean_b']:.3f} -> {g['mean_a']:.3f} per tree, p {g['p']:.3f}")
    print("  At the level of whole trainings (3 centre-line against the 7 named-recipe trainings, one-sided "
          "permutation over all 120 ways to pick 3 of the 10):")
    for label, values in (("V MAE", mae), ("incomplete masks", incomplete), ("coverage (lower is worse)", cover)):
        if label.startswith("coverage"):
            p = permutation({n: -values[n] for n in values}, CL, SINGLES)
        else:
            p = permutation(values, CL, SINGLES)
        a = np.mean([values[n] for n in CL])
        b = np.mean([values[n] for n in SINGLES])
        print(f"    {label:26s} {b:7.3f} -> {a:7.3f}   p {p:.3f}")

    print("\n3. Ensembling, against each ensemble's own members (each tree's error averaged over them)")
    for ens, members in ENS.items():
        v = verdict([T[ens]], [T[n] for n in members])
        e, g = v["error"], v["guards"]["incomplete"]
        print(f"  {short(ens):18s} V MAE {e['mean_b']:.3f} -> {e['mean_a']:.3f}, lower on {e['a_lower']:.0%}, p {e['p']:.4f};  "
              f"incomplete {g['mean_b']:.3f} -> {g['mean_a']:.3f}, p {g['p']:.2f}")

    print("\n4. Stability between trainings (the student's reason for the ensemble)")
    single = stability([T[n] for n in SINGLES])
    pair = stability([T["segment_lined_up_ensemble"], T["segment_lined_up_ensemble_b"]])
    for label, st in (("two single trainings (all 21 pairs)", single), ("the two independent ensembles", pair)):
        print(f"  {label:38s} a median {st['median_difference']:.2f} deg apart, {st['over_5']:.1%} of trees more "
              f"than 5 deg, incomplete status differs on {st['status_flips']:.1%}")
    # On the trees where a training's mask is incomplete: session 14's unstable trees.
    def apart(tables):
        out = []
        for a, b in itertools.combinations(tables, 2):
            for t in set(a) & set(b):
                if a[t]["incomplete"] or b[t]["incomplete"]:
                    out.append(abs(a[t]["pred"] - b[t]["pred"]))
        return np.array(out)
    d1, d2 = apart([T[n] for n in SINGLES]), apart([T["segment_lined_up_ensemble"], T["segment_lined_up_ensemble_b"]])
    print(f"  where either mask is incomplete: single trainings a median {np.median(d1):.2f} deg apart "
          f"({np.mean(d1 > 5):.0%} > 5 deg, {len(d1)} tree pairs); ensembles {np.median(d2):.2f} deg "
          f"({np.mean(d2 > 5):.0%}, {len(d2)} trees)")

    print("\n5. The members' spread as a warning (pre-declared: useful if Spearman > 0 with p < 0.05)")
    for ens, members in ENS.items():
        d = disagreement_check([T[n] for n in members], T[ens])
        print(f"  {short(ens):18s} Spearman {d['spearman']:+.2f} (p {d['p']:.3f}); MAE {d['mae_flagged']:.2f} deg on the "
              f"10% most spread, {d['mae_rest']:.2f} on the rest; {d['worst_caught']:.0%} of the 10% largest errors flagged")

    figure(mae, cover, ROOT / "docs/figures/segment_variants.png")
    print("\nSaved docs/figures/segment_variants.png")


def figure(mae: dict[str, float], cover: dict[str, float], path: Path) -> None:
    """Each training and ensemble as a dot, by recipe: the V opening's MAE, and the hand masks' centre line covered."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"                    # categorical slots 1 and 2 (validated, all pairs)
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    rows = [("Named recipe:\none training", SINGLES, blue, "o"),
            ("Centre-line loss:\none training", CL, orange, "o"),
            ("Named recipe:\nensemble of 3", ["segment_lined_up_ensemble", "segment_lined_up_ensemble_b"], blue, "D"),
            ("Centre-line loss:\nensemble of 3", ["segment_lined_up_cldice_ensemble"], orange, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor("white")
    for ax, values, title, unit in ((axes[0], mae, "V opening's mean absolute error against winter", "degrees, lower is better"),
                                     (axes[1], cover, "Hand masks' centre line covered (out of fold)", "share, higher is better")):
        for y, (label, runs, colour, marker) in enumerate(rows):
            xs = [values[n] for n in runs]
            # Three levels in turn along x, so neighbouring (or equal) values never share a level.
            offs = [(-0.2, 0.0, 0.2)[k % 3] for k in range(len(xs))] if len(xs) > 1 else [0.0]
            order = np.argsort(xs, kind="stable")
            for o, i in zip(offs, order):
                ax.scatter(xs[i], -y + o, s=70, color=colour, marker=marker, edgecolor="white", linewidth=1.5, zorder=3)
            if runs == SINGLES:
                i = runs.index(NAMED)
                ax.scatter(values[NAMED], -y + offs[list(order).index(i)], s=190, facecolor="none", edgecolor=ink,
                           linewidth=1.2, zorder=4)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel(unit, fontsize=9, color=muted)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
    axes[0].set_yticks([-y for y in range(len(rows))], [r[0] for r in rows], fontsize=9, color=ink)
    fig.suptitle("Trainings of one recipe vary more than the centre-line loss moves the V opening; "
                 "it does raise coverage", fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "449 development trees, each predicted out of fold; masks joined. Dots: one training "
             "(circled: the named run); diamonds: probabilities of three trainings averaged.", fontsize=8, color=muted)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.83, bottom=0.2)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
