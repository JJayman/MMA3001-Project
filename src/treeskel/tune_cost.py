"""Tune the guided-tracing cost map against the annotator's correction clicks.

When a trace strays, the annotator adds a correction click (waypoint) on the
middle of the branch where the trace should have gone. A good cost map
should therefore trace, **from the four main clicks alone** (root, fork and
the two tips), paths that pass close to every waypoint. This module measures
exactly that:

* **Error:** for each waypoint, the distance (px) from it to the path traced
  through only the main clicks, for the same part (trunk, left or right
  leader). Reported as mean, median and the fraction further than
  ``--tolerance`` px (waypoints that would still be needed).

The waypoints are human judgements made on the image, so no cost-map setting
is favoured by the reference. An earlier version compared traces with
reference *traces* drawn using the default settings. That favoured the
default wherever no correction was made, and it hid the improvement: the
default still ranked first there, although this measure shows it about 3.6
times worse than the tuned setting (11.6 px vs 3.2 px mean).

Settings are sampled at random from :data:`GRID` (random search), always
including the default. With few trees the choice is checked by
**leave-one-out cross-validation**: for each tree, the setting that is best
on the *other* trees is scored on its waypoints. Pooling those scores
estimates the error on unseen trees.

Usage, from the command line:

```
python -m treeskel.tune_cost data/tuning --clicks annotations/winter_clicks.json \\
    --samples 250 --out outputs/tune_cost
```
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from treeskel.guided_gt import Clicks, GuidedParams, combine_cues, cost_cues, load_clicks, trace

PARTS = ("trunk", "left", "right")

GRID = {
    "depth_weight": [0.05, 0.1, 0.2, 0.35, 0.5],
    "tube_weight": [0.0, 0.05, 0.1, 0.3, 0.5],
    "colour_weight": [0.0, 0.1, 0.2, 0.4],
    "cost_power": [0.5, 1.0, 2.0, 3.0],
    "centre_weight": [0.0, 0.5, 1.0],
    "centre_source": ["depth", "wood"],
    "sigmas": [(2, 4, 6, 8, 10), (4, 6, 8, 10, 12, 14)],
}
"""Values sampled in the wide random search (``--grid wide``)."""

FOCUSED_GRID = {
    "depth_weight": [0.1, 0.2, 0.35],
    "tube_weight": [0.0, 0.05, 0.1],
    "colour_weight": [0.0, 0.1, 0.2],
    "cost_power": [0.5, 1.0],
    "centre_weight": [1.0],
    "centre_source": ["depth"],
    "sigmas": [(2, 4, 6, 8, 10)],
}
"""Region searched exhaustively (``--grid focused``): where the best settings
of two wide random searches all lay (full centring from depth, low cost
exponent, low to moderate weights). Because this region was chosen after
seeing all trees' results, the leave-one-out estimate for it is slightly
optimistic."""


def base_clicks(c: Clicks) -> Clicks:
    """Only the four main clicks: root, fork and the two tips.

    Args:
        c: A tree's clicks, with any correction clicks.

    Returns:
        New :class:`Clicks` without them.
    """
    return Clicks(trunk=[c.trunk[0], c.trunk[-1]], left=[c.left[-1]], right=[c.right[-1]])


def part_paths(cost: np.ndarray, c: Clicks) -> dict[str, np.ndarray]:
    """Trace the trunk and both leaders for the given clicks.

    Args:
        cost: The tree's cost map.
        c: The clicks to trace through.

    Returns:
        ``{"trunk", "left", "right"}`` ``(N, 2)`` ``(row, col)`` paths.
    """
    return {
        "trunk": trace(cost, c.trunk),
        "left": trace(cost, [c.fork, *c.left]),
        "right": trace(cost, [c.fork, *c.right]),
    }


def waypoints(c: Clicks, part: str) -> list[tuple[int, int]]:
    """The correction clicks for one part (excluding root, fork and tips).

    Args:
        c: A tree's clicks.
        part: ``"trunk"``, ``"left"`` or ``"right"``.

    Returns:
        The ``(row, col)`` correction clicks.
    """
    return list(c.trunk[1:-1]) if part == "trunk" else list(getattr(c, part)[:-1])


def waypoint_distances(path: np.ndarray, points: Sequence[tuple[int, int]]) -> list[float]:
    """Distance (px) from each point to the nearest pixel of ``path``.

    Args:
        path: ``(N, 2)`` ``(row, col)`` path.
        points: ``(row, col)`` points.

    Returns:
        One distance per point.
    """
    return [float(np.hypot(*(path - np.asarray(q)).T).min()) for q in points]


def grid_sample(n: int, seed: int = 0, grid: dict | None = None) -> list[GuidedParams]:
    """The default settings plus up to ``n`` distinct random points of ``grid``.

    With ``n`` at least the grid size, every point is included (exhaustive).

    Args:
        n: Random settings wanted.
        seed: Random seed.
        grid: Values per setting; :data:`GRID` by default.

    Returns:
        The settings, the defaults first.
    """
    grid = GRID if grid is None else grid
    combos = list(itertools.product(*grid.values()))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    out = [GuidedParams()]
    for i in sorted(pick):
        p = GuidedParams(**dict(zip(grid, combos[i])))
        if p not in out:
            out.append(p)
    return out


def evaluate(trees: dict, params: GuidedParams) -> dict[str, list[float]]:
    """Waypoint distances for one setting.

    Args:
        trees: ``{tree_id: {"cues": {sigmas: cues}, "clicks": Clicks}}``.
        params: Setting to test.

    Returns:
        ``{tree_id: [distance of each waypoint to the main-clicks trace]}``.
    """
    out = {}
    for tid, t in trees.items():
        c = t["clicks"]
        cost, _ = combine_cues(t["cues"][params.sigmas], params)
        cand = part_paths(cost, base_clicks(c))
        out[tid] = [d for part in PARTS for d in waypoint_distances(cand[part], waypoints(c, part))]
    return out


def summarise(dists: Sequence[float], tolerance: float) -> dict[str, float]:
    """Mean, median and fraction beyond ``tolerance`` of waypoint distances.

    Args:
        dists: Waypoint distances (px).
        tolerance: Distance (px) above which a correction click is still needed.

    Returns:
        ``{"mean_px", "median_px", "still_needed", "n"}`` (zeros if empty).
    """
    a = np.asarray(dists, float)
    if a.size == 0:
        return {"mean_px": 0.0, "median_px": 0.0, "still_needed": 0.0, "n": 0}
    return {"mean_px": float(a.mean()), "median_px": float(np.median(a)),
            "still_needed": float((a > tolerance).mean()), "n": int(a.size)}


def pooled(res: dict[str, list[float]], exclude: str | None = None) -> list[float]:
    """All waypoint distances, optionally leaving one tree out.

    Args:
        res: Waypoint distances per tree id.
        exclude: A tree id to leave out.

    Returns:
        The distances.
    """
    return [d for t, ds in res.items() if t != exclude for d in ds]


def leave_one_out(results: list[dict[str, list[float]]]) -> tuple[list[tuple[str, int, list[float]]], list[float]]:
    """Leave-one-out estimate of the tuned error on unseen trees.

    For each tree with waypoints, the setting with the lowest mean waypoint
    distance on the other trees is chosen, and the held-out tree's distances
    under that setting are kept.

    Args:
        results: One :func:`evaluate` result per setting (same trees).

    Returns:
        ``([(tree_id, chosen_setting_index, held_out_distances), ...],
        all held-out distances pooled)``.
    """
    ids = [t for t in results[0] if results[0][t]]
    rows, all_d = [], []
    for tid in ids:
        k = int(np.argmin([np.mean(pooled(r, exclude=tid)) for r in results]))
        rows.append((tid, k, results[k][tid]))
        all_d += results[k][tid]
    return rows, all_d


def main(argv: Sequence[str] | None = None) -> int:
    """Run the random search and leave-one-out check; print and save results.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if no tree in the data folder has correction clicks.
    """
    from treeskel.io import index_dataset, load_tree

    ap = argparse.ArgumentParser(prog="python -m treeskel.tune_cost", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--samples", type=int, default=250, help="random settings to try (plus the default)")
    ap.add_argument("--tolerance", type=float, default=8.0,
                    help="distance (px) beyond which a waypoint would still be needed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", choices=["wide", "focused"], default="wide",
                    help="wide: random sample of GRID; focused: every point of FOCUSED_GRID")
    ap.add_argument("--out", default="outputs/tune_cost")
    args = ap.parse_args(argv)

    clicks = load_clicks(args.clicks)
    records = {t.tree_id: t for t in index_dataset(args.data_dir).trees}
    ids = sorted(k for k, c in clicks.items() if k in records and c.complete and not c.skip)
    if not any(waypoints(clicks[t], p) for t in ids for p in PARTS):
        print("No correction clicks found for trees in the data folder; nothing to tune against.")
        return 1

    # The cost map's expensive cues are computed once per tree (and per set of tube-filter scales); each
    # setting then only re-weights them (guided_gt.combine_cues).
    grid = GRID if args.grid == "wide" else FOCUSED_GRID
    sigma_sets = set(grid["sigmas"]) | {GuidedParams().sigmas}
    t0 = time.time()
    trees = {}
    for tid in ids:
        d = load_tree(records[tid], "winter")
        trees[tid] = {"clicks": clicks[tid],
                      "cues": {s: cost_cues(d["rgb"], d["depth"], d["valid"], replace(GuidedParams(), sigmas=s))
                               for s in sigma_sets}}
    print(f"Prepared {len(ids)} trees in {time.time() - t0:.0f} s")

    # Every setting is scored by how far each correction click lies from the path traced through only the
    # four main clicks (the first setting is the default).
    n = args.samples if args.grid == "wide" else 10**9
    settings = grid_sample(n, args.seed, grid)
    t0 = time.time()
    results = []
    for i, p in enumerate(settings):
        results.append(evaluate(trees, p))
        if i == 0:
            print(f"One setting takes {time.time() - t0:.1f} s; trying {len(settings)}")
    # The simplest reference: straight lines between the main clicks (a uniform cost).
    flat = {t: [d for part in PARTS for d in waypoint_distances(
        part_paths(np.ones((480, 640)), base_clicks(clicks[t]))[part], waypoints(clicks[t], part))] for t in ids}

    summ = [summarise(pooled(r), args.tolerance) for r in results]
    order = sorted(range(len(settings)), key=lambda i: summ[i]["mean_px"])
    best = order[0]
    # Leave-one-tree-out: the best setting on the other trees, scored on the tree left out.
    loo_rows, loo_d = leave_one_out(results)
    loo = summarise(loo_d, args.tolerance)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "search.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*grid, "mean_px", "median_px", "still_needed"])
        for i in order:
            w.writerow([*(getattr(settings[i], k) for k in grid),
                        *(round(summ[i][k], 3) for k in ("mean_px", "median_px", "still_needed"))])

    def line(name: str, s: dict) -> str:
        """One setting's distances as a line of the report."""
        return (f"{name:34s} mean {s['mean_px']:5.1f} px, median {s['median_px']:5.1f} px, "
                f"{s['still_needed']:4.0%} of waypoints still needed")

    print(f"\n{summ[0]['n']} correction clicks on {len([t for t in ids if results[0][t]])} trees "
          f"(tolerance {args.tolerance:g} px):")
    print(line("Straight lines between clicks", summarise(pooled(flat), args.tolerance)))
    print(line("Default settings", summ[0]))
    print(line(f"Best of {len(settings)} settings", summ[best]))
    print(line("Leave-one-out (unseen trees)", loo))
    chosen = {k for _, k, _ in loo_rows}
    print(f"Leave-one-out chose {len(chosen)} distinct setting(s) across folds")
    print("Best settings:", {k: getattr(settings[best], k) for k in grid})
    print("Top 5:")
    for i in order[:5]:
        print("  ", {k: getattr(settings[i], k) for k in grid}, line("", summ[i]).strip())

    report = {
        "trees": ids, "tolerance_px": args.tolerance, "grid": args.grid,
        "settings_tried": len(settings),
        "straight_lines": summarise(pooled(flat), args.tolerance),
        "default": {"params": asdict(settings[0]), **summ[0]},
        "best": {"params": asdict(settings[best]), **summ[best]},
        "leave_one_out": {**loo, "distinct_settings_chosen": len(chosen),
                          "per_tree": [{"tree": t, "chosen": {k2: getattr(settings[k], k2) for k2 in grid},
                                        "mean_px": float(np.mean(d))} for t, k, d in loo_rows]},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=list))
    print(f"Saved {out / 'search.csv'} and {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
