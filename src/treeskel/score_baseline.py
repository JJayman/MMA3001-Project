"""Score the automatic winter baseline against the click-guided ground truth.

The automatic method (:mod:`treeskel.winter_gt`, session 4) was dropped as the
ground truth because the student's look at its overlays found about 45 wrong
trees in rows 1, 7 and 9, many of which had passed its quality check. It was
kept as a baseline, to be scored once the guided ground truth
(:mod:`treeskel.guided_gt`, all 541 trees clicked) existed. Both work on the
same winter photo, so no alignment is needed: the baseline's trunk and
leaders are compared directly with the traces through the student's clicks.

Development rows only (the test rows of ``splits.json`` are never loaded).
Nothing is chosen here: the levels below were fixed before the run and only
describe the result.

**Paths** (:func:`path_scores`). Each traced point is drawn as a disc of the
branch's radius (the guided mask is the union of these discs):

- *found*: the share of the guided trunk and leaders that the baseline passes
  through, a point counting if some baseline point lies within that point's
  drawn radius (a recall; also per part);
- *on the branch*: the share of the baseline's points that lie inside the
  guided mask (a precision; also per part);
- their harmonic mean, the tree's **F1**, and the same two shares at fixed
  distances of 5, 10 and 20 px from the other's centre line;
- the distance between the two forks, roots and leader tips;
- the masks' overlap (IoU, Dice) as a secondary pixel score: the baseline
  draws its widths from the depth silhouette, which reaches about 2 px past
  each edge (session 8).

The baseline's trunk, left and right are scored against the union of the
guided parts, so a fork placed higher or lower moves points between parts
without counting as a miss; the fork's own distance is reported instead. A
tree for which the baseline finds no fork has F1 0. A tree is called
**traced** at F1 ≥ 0.9 (and, as a looser level, 0.8).

**The quality check** (:attr:`treeskel.winter_gt.WinterGT.ok`): how many
passed trees are not traced, and how many flagged trees are.

**Shape** (:func:`shape_scores`): the baseline's paths are measured exactly as
the guided traces are (:func:`treeskel.shape.shape_measures`), and scored with
the same skill over the mean shape as every model
(:func:`treeskel.shape_model.evaluate`), against the winter measures and the
mean shapes saved with the named U-Net's run
(``docs/results/segment_lined_up_joined/predictions.csv``). A tree with no
fork counts as the mean shape, as for the U-Net, and a measure its paths
cannot give (a leader too short for its line) leaves the tree out of that
measure, as for the U-Net; because the baseline's leaders stop short far more
often, a second row counts every such gap as the mean shape too, and the
trees that passed its quality check are scored alone. Beside it: the hand masks,
the straight lines through the clicks alone, and the U-Net, tree by tree.
The baseline sees the bare tree in winter, the U-Net the summer photo through
the leaves.

**The student's verdicts** (:data:`SESSION4_WRONG`): the trees listed in
session 4 as wrong by eye, against the other trees of their rows: an
independent check that the scores see what a person sees. (Judged on the
session-4 baseline, before the depth registration of session 9.)

**Run time** per tree: the baseline's extraction, and the guided tracing
(cost map, tracing and widths; the clicks took about 15 s per tree,
session 5).

Usage, from the command line:

```
python -m treeskel.score_baseline data/raw          # -> docs/results/winter_baseline/
python -m treeskel.score_baseline --from-saved docs/results/winter_baseline   # re-score, no images
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree

from treeskel.guided_gt import Clicks, guided_gt, load_clicks
from treeskel.shape import MEASURES, gt_shape, shape_measures
from treeskel.winter_gt import WinterGT, extract_winter_gt, render_main_branches, report_failed, save_image

PARTS = ("trunk", "left", "right")
"""The parts of a tree, as named in :class:`treeskel.winter_gt.WinterGT`."""

TOLERANCES = (5.0, 10.0, 20.0)
"""Fixed distances (px) at which the paths are also compared, beside the branch's own radius."""

TRACED = (0.9, 0.8)
"""F1 levels at which a tree counts as traced (fixed before the run; descriptive only)."""

SESSION4_WRONG = {
    **dict.fromkeys(("R01N14", "R01N22", "R01N29", "R01N33", "R01N34", "R07N01", "R07N02", "R07N03",
                     "R07N05", "R07N06", "R07N10", "R07N11"), "stops part-way"),
    **dict.fromkeys(("R01N25", "R01N31", "R01N41", "R01N43", "R07N04", "R07N08", "R09N03"),
                    "jumps to another branch or tree"),
    **dict.fromkeys(("R01N05", "R01N08", "R01N13", "R01N18", "R01N32", "R07N17", "R07N25"), "a post"),
    **dict.fromkeys(("R07N29", "R09N18", "R09N32"), "straight unbranched tree"),
    **dict.fromkeys(("R01N01", "R01N06", "R01N35", "R07N09", "R07N26", "R09N12", "R09N16"), "wrong fork"),
    **dict.fromkeys(("R01N09", "R09N04"), "three main branches"),
}
"""The trees the student judged wrong by eye in rows 1, 7 and 9 (session 4's table of examples), by
failure type. R07N04 was listed as stopping part-way and jumping, R01N14 as stopping part-way and having
three main branches; each is kept under its first type. The table lists examples (38 trees) of about 45."""

SESSION4_ROWS = (1, 7, 9)
"""The rows the student inspected in session 4."""

PREDICTIONS = Path("docs/results/segment_lined_up_joined/predictions.csv")
"""The named U-Net's development predictions, with the winter measures, the mean shapes and the hand
masks' measures of every development tree."""

TABLE = Path("data/shape/shape_measures.csv")
"""The committed shape measures (:mod:`treeskel.shape`), for the lines through the clicks alone."""


# --------------------------------------------------------------------------
# Scores of one tree
# --------------------------------------------------------------------------

def nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of ``a`` to the nearest point of ``b`` (``inf`` if ``b`` is empty).

    Args:
        a: ``(N, 2)`` points.
        b: ``(M, 2)`` points.

    Returns:
        ``(N,)`` distances.

    Example:
        >>> nearest(np.array([[0, 0], [0, 5]]), np.array([[0, 3]])).tolist()
        [3.0, 2.0]
    """
    a = np.asarray(a, float).reshape(-1, 2)
    b = np.asarray(b, float).reshape(-1, 2)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    if len(a) == 0:
        return np.zeros(0)
    return cKDTree(b).query(a)[0]


def _share(hit: np.ndarray) -> float:
    """The mean of a bool array, NaN if it is empty."""
    return float(np.mean(hit)) if len(hit) else float("nan")


def _f1(precision: float, recall: float) -> float:
    """The harmonic mean of the two (0 if either is 0 or undefined: nothing found, or nothing drawn)."""
    if not (np.isfinite(precision) and np.isfinite(recall)) or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _distance(a, b) -> float:
    """Distance between two ``(row, col)`` points, NaN if either is missing."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def path_scores(baseline: WinterGT, guided: WinterGT, guided_mask: np.ndarray | None = None) -> dict[str, float]:
    """How well the baseline's trunk and leaders match the guided traces (see the module docstring).

    Args:
        baseline: :func:`treeskel.winter_gt.extract_winter_gt`'s result.
        guided: :func:`treeskel.guided_gt.guided_gt`'s result (with ``radii``).
        guided_mask: The guided mask (:func:`treeskel.winter_gt.render_main_branches`); drawn if None.

    Returns:
        ``found``, ``on_branch``, ``f1`` (and ``found_<part>``, ``on_branch_<part>``), the shares at
        each of :data:`TOLERANCES` (``found_5px`` ...), the median distance each way
        (``to_guided_px``: from the baseline's points to the guided centre lines; ``to_baseline_px``),
        and the distances between the forks, roots and tips (``fork_px`` ...); NaN where undefined.
    """
    if guided_mask is None:
        guided_mask = render_main_branches(guided)
    g = {p: np.asarray(getattr(guided, p)).reshape(-1, 2) for p in PARTS}
    b = {p: np.asarray(getattr(baseline, p)).reshape(-1, 2).astype(int) for p in PARTS}
    g_all = np.concatenate([g[p] for p in PARTS])
    b_all = np.concatenate([b[p] for p in PARTS])
    out: dict[str, float] = {}

    # Found: a guided point counts if the baseline passes through its drawn disc.
    d_g = {p: nearest(g[p], b_all) for p in PARTS}
    for p in PARTS:
        out[f"found_{p}"] = _share(d_g[p] <= guided.radii[p])
    d_all = np.concatenate([d_g[p] for p in PARTS])
    out["found"] = _share(d_all <= np.concatenate([guided.radii[p] for p in PARTS]))

    # On the branch: a baseline point counts if it lies inside the guided mask.
    for p in PARTS:
        out[f"on_branch_{p}"] = _share(guided_mask[b[p][:, 0], b[p][:, 1]])
    out["on_branch"] = _share(guided_mask[b_all[:, 0], b_all[:, 1]])
    out["f1"] = _f1(out["on_branch"], out["found"])

    d_b = nearest(b_all, g_all)
    for t in TOLERANCES:
        out[f"found_{t:g}px"] = _share(d_all <= t)
        out[f"on_branch_{t:g}px"] = _share(d_b <= t)
    out["to_guided_px"] = float(np.median(d_b)) if len(d_b) else float("nan")
    out["to_baseline_px"] = float(np.median(d_all)) if len(b_all) else float("nan")

    out["fork_px"] = _distance(baseline.fork, guided.fork)
    out["root_px"] = _distance(baseline.root, guided.root)
    for side in ("left", "right"):
        out[f"{side}_tip_px"] = _distance(b[side][-1] if len(b[side]) else None, g[side][-1])
    return out


def baseline_shape(baseline: WinterGT) -> dict[str, float]:
    """The shape measures of the baseline's paths (NaN throughout if it found no fork).

    The trunk radius at the fork, which :func:`treeskel.shape.shape_measures` only records (its unit is
    fixed in pixels), is the depth silhouette's half-width there.

    Args:
        baseline: The automatic method's result for the tree.

    Returns:
        ``{measure: degrees}`` for every name in :data:`treeskel.shape.MEASURES`.
    """
    if baseline.fork is None or not all(len(getattr(baseline, p)) for p in PARTS):
        return dict.fromkeys(MEASURES, float("nan"))
    radius = float(ndi.distance_transform_edt(baseline.tree_mask)[baseline.fork])
    return shape_measures(baseline.trunk, baseline.left, baseline.right, radius)


def issue_type(issue: str) -> str:
    """A quality-check issue without its numbers, so issues can be counted by type.

    Args:
        issue: One issue from the baseline's quality check.

    Returns:
        The issue's type.

    Example:
        >>> issue_type("left leader rises 52 px (< 139)")
        'left leader rises too little'
    """
    return re.sub(r" rises .*", " rises too little", issue)


def score_tree(baseline: WinterGT, guided: WinterGT) -> dict:
    """Every score of one tree: the quality check, :func:`path_scores`, the masks' overlap, and both
    methods' shape measures (``baseline_<measure>``, ``guided_<measure>``).

    Args:
        baseline: The automatic method's result.
        guided: The guided ground truth.

    Returns:
        The scores, by name.
    """
    from treeskel.segment import mask_scores

    guided_mask = render_main_branches(guided)
    row = {"ok": bool(baseline.ok), "no_fork": baseline.fork is None,
           "issues": "; ".join(baseline.issues), "posts_skipped": int(baseline.posts_skipped)}
    row.update(path_scores(baseline, guided, guided_mask))
    row.update({f"mask_{k}": v for k, v in mask_scores(render_main_branches(baseline), guided_mask).items()})
    row.update({f"baseline_{k}": v for k, v in baseline_shape(baseline).items() if k in MEASURES})
    row.update({f"guided_{k}": v for k, v in gt_shape(guided).items() if k in MEASURES})
    return row


def run_tree(rec, clicks: Clicks) -> tuple[dict, WinterGT, WinterGT, np.ndarray]:
    """Load one tree's winter images (depth registered), run both methods, time them and score.

    Args:
        rec: The tree's record.
        clicks: The tree's winter clicks.

    Returns:
        ``(row, baseline, guided, rgb)``; ``row`` holds :func:`score_tree`'s scores, the tree's id and row,
        and the seconds each method took (``baseline_s``, ``guided_s``).
    """
    from treeskel.io import load_tree

    d = load_tree(rec, "winter")
    t0 = time.perf_counter()
    baseline = extract_winter_gt(d["depth"], d["valid"])
    t1 = time.perf_counter()
    guided = guided_gt(d["rgb"], d["depth"], d["valid"], clicks)
    t2 = time.perf_counter()
    row = {"tree": rec.tree_id, "row": rec.row, **score_tree(baseline, guided),
           "baseline_s": t1 - t0, "guided_s": t2 - t1}
    return row, baseline, guided, d["rgb"]


def _job(job: tuple) -> dict:
    """:func:`run_tree` for a worker process (top-level so it can be pickled); returns the row only."""
    return run_tree(*job)[0]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

PATH_COLUMNS = ("f1", "found", "on_branch", "found_trunk", "found_left", "found_right", "on_branch_trunk",
                "on_branch_left", "on_branch_right", *(f"{k}_{t:g}px" for t in TOLERANCES
                                                       for k in ("found", "on_branch")),
                "to_guided_px", "to_baseline_px", "fork_px", "root_px", "left_tip_px", "right_tip_px",
                "mask_iou", "mask_dice", "mask_precision", "mask_recall")
"""The per-tree path and mask scores summarised by :func:`summarise`."""


def _stats(values: Sequence[float]) -> dict:
    """Median, quartiles, mean and count of the finite values (None where there are none)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(len(v)), "median": float(med), "q1": float(q1), "q3": float(q3), "mean": float(v.mean())}


def _group(rows: Sequence[dict]) -> dict:
    """:func:`_stats` of every path column, and the share of trees traced at each level."""
    out = {k: _stats([r[k] for r in rows]) for k in PATH_COLUMNS}
    out["n_trees"] = len(rows)
    out["traced"] = {f"{lv:g}": float(np.mean([r["f1"] >= lv for r in rows])) if rows else None for lv in TRACED}
    return out


def quality_check(rows: Sequence[dict]) -> dict:
    """How well the baseline's own quality check separates traced from untraced trees.

    Args:
        rows: One row per tree, from :func:`run_tree`.

    Returns:
        Per level of :data:`TRACED`: the passed trees not traced and the flagged trees traced (counts and
        shares), and a Mann-Whitney test of F1 between passed and flagged trees; plus the issues by type.
    """
    passed = [r for r in rows if r["ok"]]
    flagged = [r for r in rows if not r["ok"]]
    out = {"passed": len(passed), "flagged": len(flagged), "no_fork": int(sum(r["no_fork"] for r in rows))}
    for lv in TRACED:
        bad = sum(r["f1"] < lv for r in passed)
        good = sum(r["f1"] >= lv for r in flagged)
        out[f"at_{lv:g}"] = {"passed_not_traced": bad, "passed_not_traced_share": bad / len(passed) if passed else None,
                             "flagged_traced": good, "flagged_traced_share": good / len(flagged) if flagged else None}
    if passed and flagged:
        out["f1_mannwhitney_p"] = float(stats.mannwhitneyu([r["f1"] for r in passed],
                                                           [r["f1"] for r in flagged]).pvalue)
    types: dict[str, int] = {}
    for r in flagged:
        for issue in filter(None, r["issues"].split("; ")):
            types[issue_type(issue)] = types.get(issue_type(issue), 0) + 1
    out["issues"] = dict(sorted(types.items(), key=lambda kv: -kv[1]))
    return out


def session4_check(rows: Sequence[dict]) -> dict:
    """The trees the student judged wrong in session 4 against the other trees of the same rows.

    Args:
        rows: One row per tree, from :func:`run_tree`.

    Returns:
        ``{"listed": ..., "others": ...}`` with each group's count, median F1 and fork distance and its
        share traced at :data:`TRACED`'s first level, a Mann-Whitney test of F1 between them, and the
        same per failure type.
    """
    inspected = [r for r in rows if r["row"] in SESSION4_ROWS]
    listed = [r for r in inspected if r["tree"] in SESSION4_WRONG]
    others = [r for r in inspected if r["tree"] not in SESSION4_WRONG]
    lv = TRACED[0]

    def describe(group: Sequence[dict]) -> dict:
        """Count, median F1, median fork distance, share traced and number passing the check for a group."""
        return {"n": len(group), "f1_median": _stats([r["f1"] for r in group])["median"],
                "fork_px_median": _stats([r["fork_px"] for r in group])["median"],
                "traced": float(np.mean([r["f1"] >= lv for r in group])) if group else None,
                "passed_qc": int(sum(r["ok"] for r in group))}

    out = {"listed": describe(listed), "others": describe(others), "level": lv,
           "by_type": {t: describe([r for r in listed if SESSION4_WRONG[r["tree"]] == t])
                       for t in dict.fromkeys(SESSION4_WRONG.values())}}
    if listed and others:
        out["f1_mannwhitney_p"] = float(stats.mannwhitneyu([r["f1"] for r in listed],
                                                           [r["f1"] for r in others]).pvalue)
    return out


def read_predictions(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """``{tree: {measure: {"fold", "winter", "mean shape", "hand masks", "segmentation"}}}`` from a
    ``predictions.csv`` of :mod:`treeskel.segment` (missing values as NaN).

    Args:
        path: A ``predictions.csv``.

    Returns:
        The values per tree and measure.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            vals = {k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
                    for k in ("fold", "winter", "mean shape", "hand masks", "segmentation")}
            out.setdefault(r["tree"], {})[r["measure"]] = vals
    return out


def shape_scores(rows: Sequence[dict], predictions: str | Path = PREDICTIONS, table: str | Path = TABLE) -> dict:
    """The baseline's shape measures scored like every model, beside the other sources.

    Args:
        rows: Per-tree rows of :func:`score_tree` (``baseline_<measure>``, ``guided_<measure>``, ``ok``).
        predictions: The named U-Net's ``predictions.csv`` (truth, mean shape, hand masks, U-Net, folds).
        table: ``shape_measures.csv`` (the lines through the clicks alone).

    Returns:
        ``{"scores": {measure: {source: evaluate(...)}}, "against_unet": {measure: {...}},
        "guided_check": {...}}``: the skills over the mean shape (a tree whose baseline has no fork counts
        as the mean shape, as for the U-Net); the baseline against the U-Net tree by tree (the share of
        trees the baseline predicts more closely, and a Wilcoxon signed-rank test); and how far the
        guided traces measured here are from the committed winter measures (they should agree).
    """
    from treeskel.segment import compare_with, fill_failures
    from treeskel.shape_model import evaluate, load_table

    pred = read_predictions(predictions)
    clicks = {r["tree"]: r for r in load_table(table)}
    by_tree = {r["tree"]: r for r in rows}
    trees = [t for t in sorted(pred) if t in by_tree]

    def column(source: str) -> np.ndarray:
        """``(trees, measures)`` array of one source's predictions."""
        return np.array([[pred[t][k][source] for k in MEASURES] for t in trees])

    truth, mean, fold = column("winter"), column("mean shape"), column("fold")[:, 0].astype(int)
    base = np.array([[float(by_tree[t][f"baseline_{k}"]) for k in MEASURES] for t in trees])
    base = np.round(base, 2)                                  # as the saved predictions are
    filled = fill_failures(base, mean)
    passed = np.array([bool(by_tree[t]["ok"]) for t in trees])
    sources = {
        "mean shape": mean,
        "winter baseline": filled,
        # A leader too short for its line leaves single measures NaN; the U-Net's rule drops those trees.
        "baseline, gaps as mean": np.where(np.isfinite(base), base, mean),
        "hand masks": column("hand masks"),
        "U-Net (named)": column("segmentation"),
        "clicks only": np.array([[clicks[t].get(f"clicks_{k}", np.nan) if t in clicks else np.nan
                                  for k in MEASURES] for t in trees]),
    }
    scores = {}
    for j, k in enumerate(MEASURES):
        scores[k] = {name: evaluate(truth[:, j], p[:, j], mean[:, j], fold) for name, p in sources.items()}
        scores[k]["winter baseline, passed QC"] = evaluate(truth[passed, j], base[passed, j], mean[passed, j],
                                                           fold[passed])

    def paired(p: np.ndarray) -> dict:
        """``{(tree, measure): (winter, predicted)}`` where both are known."""
        return {(t, k): (truth[i, j], p[i, j]) for i, t in enumerate(trees) for j, k in enumerate(MEASURES)
                if np.isfinite(truth[i, j]) and np.isfinite(p[i, j])}

    against = {k: {"n": v["n"], "baseline_closer": v["closer"], "wilcoxon_p": v["wilcoxon_p"],
                   "mae_baseline": v["mae_seg"], "mae_unet": v["mae_other"]}
               for k, v in compare_with(paired(filled), paired(sources["U-Net (named)"])).items()}
    guided = np.array([[float(by_tree[t].get(f"guided_{k}", np.nan)) for k in MEASURES] for t in trees])
    diff = np.abs(guided - truth)
    check = {"n_trees": len(trees), "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else None,
             "trees_over_0.01": int(np.sum(np.nanmax(np.where(np.isfinite(diff), diff, 0), axis=1) > 0.01))}
    return {"scores": scores, "against_unet": against, "guided_check": check, "n_trees": len(trees),
            "baseline_no_fork": int((~np.isfinite(base).any(axis=1)).sum())}


def summarise(rows: Sequence[dict], predictions: str | Path | None = PREDICTIONS,
              table: str | Path = TABLE) -> dict:
    """The whole report: paths and masks (all trees, passed and flagged), the quality check, the
    session-4 verdicts, run times, and (if ``predictions`` exists) the shape scores.

    Args:
        rows: One row per tree, from :func:`run_tree`.
        predictions: The named U-Net's ``predictions.csv``, for the shape scores; None to leave them out.
        table: The winter shape table (the lines through the clicks).

    Returns:
        The report (``report.json``).
    """
    report = {
        "n_trees": len(rows),
        "levels": list(TRACED),
        "tolerances_px": list(TOLERANCES),
        "paths": {"all": _group(rows), "passed": _group([r for r in rows if r["ok"]]),
                  "flagged": _group([r for r in rows if not r["ok"]])},
        "quality_check": quality_check(rows),
        "session4": session4_check(rows),
        "seconds": {"baseline": _stats([r["baseline_s"] for r in rows]),
                    "guided": _stats([r["guided_s"] for r in rows])},
    }
    if predictions is not None and Path(predictions).exists():
        report["shape"] = shape_scores(rows, predictions, table)
    return report


def _fmt(x, spec: str = ".2f") -> str:
    """``x`` formatted, or ``-`` if missing."""
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def report_lines(report: dict) -> list[str]:
    """The report as text, for the console and ``output.txt``.

    Args:
        report: From :func:`summarise`.

    Returns:
        The lines of text.
    """
    p, qc, s4 = report["paths"], report["quality_check"], report["session4"]
    lv0, lv1 = (f"{v:g}" for v in TRACED)
    out = [f"Winter baseline against the guided ground truth: {report['n_trees']} development trees.",
           f"Quality check: {qc['passed']} passed, {qc['flagged']} flagged ({qc['no_fork']} with no fork)."]
    for issue, n in qc["issues"].items():
        out.append(f"  {n:4d}  {issue}")
    out += ["", "Paths (median [quartiles]; traced = F1 at least the level):",
            f"  {'trees':14s} {'n':>4s} {'F1':>17s} {'found':>17s} {'on branch':>17s} "
            f"{'fork px':>8s} {'traced ' + lv0:>11s} {lv1:>5s}"]
    for name in ("all", "passed", "flagged"):
        g = p[name]

        def mq(k):
            """A score's median and quartiles as text, or ``-`` for an empty group."""
            v = g[k]
            return f"{_fmt(v['median'])} [{_fmt(v['q1'])}, {_fmt(v['q3'])}]" if v["n"] else "-"
        out.append(f"  {name:14s} {g['n_trees']:4d} {mq('f1'):>17s} {mq('found'):>17s} {mq('on_branch'):>17s} "
                   f"{_fmt(g['fork_px']['median'], '.1f'):>8s} {_fmt(g['traced'][lv0], '.0%'):>11s} "
                   f"{_fmt(g['traced'][lv1], '.0%'):>5s}")
    a = p["all"]
    out.append("  by part, all trees (median found / on branch): " + ", ".join(
        f"{k} {_fmt(a['found_' + k]['median'])} / {_fmt(a['on_branch_' + k]['median'])}" for k in PARTS))
    out.append("  at fixed distances, all trees (mean found / on branch): " + ", ".join(
        f"{t:g} px {_fmt(a[f'found_{t:g}px']['mean'])} / {_fmt(a[f'on_branch_{t:g}px']['mean'])}"
        for t in TOLERANCES))
    out.append(f"  masks (median): IoU {_fmt(a['mask_iou']['median'])}, Dice {_fmt(a['mask_dice']['median'])}, "
               f"precision {_fmt(a['mask_precision']['median'])}, recall {_fmt(a['mask_recall']['median'])}")
    out += ["", "The quality check as a test of the result:"]
    for lv in (lv0, lv1):
        q = qc[f"at_{lv}"]
        out.append(f"  at F1 {lv}: {q['passed_not_traced']} passed trees not traced "
                   f"({_fmt(q['passed_not_traced_share'], '.0%')} of passed); {q['flagged_traced']} flagged "
                   f"trees traced ({_fmt(q['flagged_traced_share'], '.0%')} of flagged)")
    out.append(f"  F1 passed against flagged: Mann-Whitney p {_fmt(qc.get('f1_mannwhitney_p'), '.1e')}")
    out += ["", f"The student's session-4 verdicts (rows {', '.join(map(str, SESSION4_ROWS))}; "
                f"traced at F1 {s4['level']:g}):"]
    for name, g in [("judged wrong", s4["listed"]), ("the others", s4["others"]),
                    *((f"  {t}", g) for t, g in s4["by_type"].items())]:
        out.append(f"  {name:34s} n {g['n']:3d}, F1 median {_fmt(g['f1_median'])}, fork {_fmt(g['fork_px_median'], '.1f')} px,"
                   f" traced {_fmt(g['traced'], '.0%')}, passed QC {g['passed_qc']}")
    out.append(f"  F1 judged wrong against the others: Mann-Whitney p {_fmt(s4.get('f1_mannwhitney_p'), '.1e')}")
    sec = report["seconds"]
    out += ["", f"Seconds per tree (median): baseline {_fmt(sec['baseline']['median'])}, guided tracing "
                f"{_fmt(sec['guided']['median'])} (plus about 15 s of clicking, session 5)."]
    if "shape" in report:
        sh = report["shape"]
        out += ["", f"Shape: skill over the mean shape (MAE in degrees), {sh['n_trees']} trees; the baseline's "
                    f"{sh['baseline_no_fork']} trees with no fork count as the mean shape:"]
        names = ("winter baseline", "baseline, gaps as mean", "winter baseline, passed QC", "clicks only",
                 "hand masks", "U-Net (named)")
        out.append(f"  {'measure':12s} " + " ".join(f"{n:>22s}" for n in names))
        for k in MEASURES:
            cells = []
            for n in names:
                sc = sh["scores"][k][n]
                cells.append(f"{_fmt(sc.get('skill'), '+.2f')} ({_fmt(sc.get('mae'), '.1f')}) n {sc['n']}"
                             if sc.get("n", 0) >= 3 else "-")
            out.append(f"  {k:12s} " + " ".join(f"{c:>22s}" for c in cells))
        out.append("  baseline against the U-Net, tree by tree (share of trees the baseline is closer, p):")
        out.append("  " + ", ".join(f"{k} {_fmt(v['baseline_closer'], '.0%')} (p {_fmt(v['wilcoxon_p'], '.1e')})"
                                    for k, v in sh["against_unet"].items()))
        gc = sh["guided_check"]
        out.append(f"  check: guided traces measured here against the committed winter measures, max |diff| "
                   f"{_fmt(gc['max_abs_diff'], '.3f')} deg ({gc['trees_over_0.01']} trees over 0.01)")
    return out


# --------------------------------------------------------------------------
# Files and the figure
# --------------------------------------------------------------------------

def save_rows(path: Path, rows: Sequence[dict]) -> None:
    """Write the per-tree rows (floats to 4 decimals).

    Args:
        path: The CSV file.
        rows: One row per tree, from :func:`run_tree`.
    """
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def load_rows(path: Path) -> list[dict]:
    """Read :func:`save_rows`'s file back (numbers as floats, ``ok``/``no_fork`` as bools).

    Args:
        path: A CSV file from :func:`save_rows`.

    Returns:
        The rows.
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k in ("tree", "issues"):
                    out[k] = v
                elif k in ("ok", "no_fork"):
                    out[k] = v == "True"
                elif k in ("row", "posts_skipped"):
                    out[k] = int(v)
                else:
                    out[k] = float(v) if v not in ("", "nan") else float("nan")
            rows.append(out)
    return rows


def example_trees(rows: Sequence[dict], k: int = 4) -> list[tuple[str, str]]:
    """Trees for the figure, as ``(tree, group)``: the passed trees with the lowest F1, passed trees at
    evenly spaced quantiles of F1 (worst to best), and flagged trees at the two ends.

    Args:
        rows: One row per tree, from :func:`run_tree`.
        k: Trees per group.

    Returns:
        ``(tree id, group)`` pairs.
    """
    passed = sorted((r["f1"], r["tree"]) for r in rows if r["ok"])
    flagged = sorted((r["f1"], r["tree"]) for r in rows if not r["ok"])
    picks = [(t, "passed QC, lowest F1") for _, t in passed[:k]]
    rest = passed[k:]
    if rest:
        idx = np.unique(np.round(np.linspace(len(rest) // 4, len(rest) - 1, k)).astype(int))
        picks += [(rest[i][1], "passed QC") for i in idx]
    if flagged:
        ends = flagged[: k // 2] + flagged[-(k - k // 2):] if len(flagged) > k else flagged
        picks += [(t, "flagged") for _, t in dict.fromkeys(ends)]
    return picks


def draw_tree(rgb: np.ndarray, baseline: WinterGT, guided: WinterGT, label: str, scale: float = 0.5):
    """One tree on its winter photo: the guided mask tinted cyan with its forks circled, and the
    baseline's trunk (white), left (red) and right (yellow) leaders and fork (PIL image, resized).

    Args:
        rgb: The ``(480, 640, 3)`` uint8 winter photo.
        baseline: The automatic method's result.
        guided: The guided ground truth.
        label: Text written on the tile.
        scale: Size of the tile relative to the photo.

    Returns:
        The tile.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = (rgb * 0.6).astype(np.uint8)
    m = render_main_branches(guided)
    img[m] = (0.45 * img[m] + 0.55 * np.array((0, 220, 255))).astype(np.uint8)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    for name, colour in (("trunk", "white"), ("left", (255, 60, 60)), ("right", (255, 230, 0))):
        p = getattr(baseline, name)
        if len(p) > 1:
            dr.line([(int(x), int(y)) for y, x in p], fill=colour, width=5)
    for fork, colour in ((guided.fork, (0, 220, 255)), (baseline.fork, "white")):
        if fork is not None:
            y, x = fork
            dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=colour, width=4)
    pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.BILINEAR)
    font = ImageFont.load_default(size=15)
    ImageDraw.Draw(pil).multiline_text((6, 4), label, fill=(255, 255, 0), font=font, stroke_width=2,
                                       stroke_fill=(0, 0, 0))
    return pil


def figure(records: dict, clicks: dict[str, Clicks], rows: Sequence[dict], columns: int = 4):
    """A contact sheet of :func:`example_trees`, re-running both methods on those trees.

    Args:
        records: The tree records by id.
        clicks: The winter clicks by id.
        rows: One row per tree, from :func:`run_tree`.
        columns: Tiles per row.

    Returns:
        The contact sheet (PIL image), or None if no example tree is found.
    """
    from PIL import Image

    by_tree = {r["tree"]: r for r in rows}
    tiles = []
    for tree, group in example_trees(rows, columns):
        r = by_tree[tree]
        _, baseline, guided, rgb = run_tree(records[tree], clicks[tree])
        label = (f"{tree}  {group}\nF1 {r['f1']:.2f}  fork {_fmt(r['fork_px'], '.0f')} px  "
                 f"V {_fmt(r['baseline_v_opening'], '.0f')} vs {_fmt(r['guided_v_opening'], '.0f')} deg")
        tiles.append(draw_tree(rgb, baseline, guided, label))
    if not tiles:
        return None
    w, h = tiles[0].size
    n_rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (w * columns, h * n_rows))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % columns) * w, (i // columns) * h))
    return sheet


def main(argv: Sequence[str] | None = None) -> int:
    """Score the baseline on the development rows; print and save the results.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if no annotated development tree was found or an output file could
        not be written.
    """
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.io import index_dataset

    ap = argparse.ArgumentParser(prog="python -m treeskel.score_baseline", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s)")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--split", default="splits.json", help="its test rows are never loaded")
    ap.add_argument("--out", help="results folder (default docs/results/winter_baseline; "
                                  "outputs/score_baseline_partial with --rows or --trees)")
    ap.add_argument("--figure", default="docs/figures/winter_baseline_examples.jpg",
                    help="contact sheet of example trees ('' for none)")
    ap.add_argument("--predictions", default=str(PREDICTIONS), help="the named U-Net's predictions.csv")
    ap.add_argument("--table", default=str(TABLE), help="shape_measures.csv (the lines through the clicks)")
    ap.add_argument("--rows", type=int, nargs="+", help="only these (development) rows")
    ap.add_argument("--trees", nargs="+", help="only these trees")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--from-saved", metavar="DIR", help="re-score a saved run's trees.csv (no images needed)")
    args = ap.parse_args(argv)

    failed: list[str] = []
    if args.from_saved:                                     # only the summary again, from the saved rows
        out = Path(args.out or args.from_saved)
        rows = load_rows(Path(args.from_saved) / "trees.csv")
    else:
        if not args.data_dirs:
            ap.error("give the dataset folder(s), or --from-saved")
        # A run on chosen rows or trees goes to outputs/, so it never overwrites the committed full run.
        partial = bool(args.rows or args.trees)
        out = Path(args.out or ("outputs/score_baseline_partial" if partial else "docs/results/winter_baseline"))
        test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])
        clicks = load_clicks(args.clicks)
        records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
        ids = sorted(t for t, c in clicks.items()
                     if t in records and c.complete and not c.skip and records[t].row not in test_rows
                     and (not args.rows or records[t].row in args.rows) and (not args.trees or t in args.trees))
        if not ids:
            print("No annotated development-row trees found; nothing to score.")
            return 1
        held_out = sum(r.row in test_rows for r in records.values())
        jobs = [(records[t], clicks[t]) for t in ids]
        print(f"Scoring the baseline on {len(ids)} development-row trees ({held_out} test-row trees left out)...",
              flush=True)
        t0, step, rows = time.time(), max(1, min(25, len(jobs) // 10)), []
        # Each tree in a worker process: the baseline and the guided trace on the same winter photo, compared.
        with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
            for i, r in enumerate(ex.map(_job, jobs, chunksize=2), 1):
                rows.append(r)
                if i % step == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)} trees, {el:.0f} s elapsed, about {el / i * (len(jobs) - i):.0f} s left",
                          flush=True)
        out.mkdir(parents=True, exist_ok=True)
        save_rows(out / "trees.csv", rows)
        # The report is made from the saved (rounded) rows, so --from-saved reproduces it exactly.
        rows = load_rows(out / "trees.csv")
        if args.figure:
            sheet = figure(records, clicks, rows)
            if sheet is not None:
                Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
                save_image(sheet, Path(args.figure), failed, quality=88)

    # The summary: path scores, the quality check, the session-4 verdicts, and shape against the U-Net.
    report = summarise(rows, args.predictions, args.table)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = report_lines(report)
    print("\n".join(lines))
    try:
        (out / "output.txt").write_text("\n".join(lines) + "\n")
    except OSError as e:
        failed.append(f"{out / 'output.txt'}: {e.strerror or e}")
    print(f"\nSaved {out / 'trees.csv'}, report.json and output.txt"
          + (f", and {args.figure}" if args.figure and not args.from_saved else ""))
    report_failed(failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
