"""Compare U-Net variants on the development rows: the centre-line loss and ensembles of three trainings.

Session 16, the student's plan: keep the named U-Net and its results as the
baseline and run one targeted comparison of two changes aimed at its known
weakness, predicted leaders that stop short under the leaves (31 of 444
trees, a third of the V opening's squared error) and differ between
trainings (session 14):

- **the centre-line loss** (``segment --cldice``, :func:`treeskel.segment.soft_cldice_loss`):
  the network is rewarded for the branches' centre lines, not only their pixels;
- **an ensemble**: three trainings of the same recipe with different seeds,
  their out-of-fold probabilities averaged, then thresholded and joined
  exactly as one network's (:func:`ensemble_run`).

**Why three samples of each recipe.** Training on the GPU cannot be repeated
exactly, and two trainings of the same recipe give different masks, most of
all on short-leader trees (session 14: a median 1.3° apart there, 27% more
than 5°). One new run against the named run would mix the change with that
noise. So each recipe is trained with three seeds (the baseline's first
sample is session 12's re-run, which the unchanged code reproduces bit for
bit), each tree's error is averaged over its recipe's three samples, and the
recipes are compared on those averages. The ensembles reuse the same six
trainings, so the plan costs five new trainings (about 20 minutes each on the
student's GPU).

**The rule** (:func:`decide`, fixed in ``docs/results/segment_variants/rule.md``
before any of the new runs): a change is adopted if, on the V opening against
the winter traces, it lowers the mean absolute error tree by tree with a
Wilcoxon signed-rank p below 0.05 (the project's naming rule since session 12),
**and** it does not significantly increase either the side branches or the
short or missing leaders per tree (the student's concern that better
connectivity could come with wrong connections, and that averaging could blur
uncertain leaders). The two changes are decided separately; the named
candidate combines whichever pass.

**Structure scores** (:func:`structure`), from the masks alone (no winter):

- *fork found*, and each leader's centre-line length from the fork; a leader
  is *short* if it is shorter than the stretch the angles are fitted over
  (16 units, 157.7 mask px; session 14's definition, reproduced: 36 trees and
  37 leaders among the named run's 447 masks with a fork);
- *incomplete*: no fork, or a short leader;
- *side branches*: stretches of the centre line off the trunk and leaders that
  reach more than a trunk radius (:data:`SIDE_REACH`) beyond their outline.
  On the 21 local development hand masks none reaches past 2.2 px; on the
  named run's masks 95% stay within 2.8 px, and the 10 beyond 9.65 px are side
  branches, a leader the leader-finder passed over, or hooks joined at the
  frame's top;
- *stray pieces*: pieces left unjoined with a centre line of 10 px or more;
- *extra area*: the share of the mask outside the trunk and leaders.

Against the lined-up hand masks (when the dataset is given):
:func:`centre_line_scores` (clDice, topology precision and sensitivity: the
share of the hand mask's centre line the prediction covers) and the pixel Dice.

**Diagnostics, deciding nothing** (all pre-declared in the rule):

- the *null check*: the four samples of the named recipe against each other,
  tree by tree, showing how often the rule's test calls two samples of one
  recipe different;
- *stability*: how far each recipe's samples are from each other, tree by tree;
- *disagreement as a warning*: whether the ensemble members' spread on the V
  opening ranks the trees the ensemble gets wrong (Spearman); validated on
  winter, which is allowed for checking a flag but chooses nothing.

The test rows are never read here.

Usage, from the command line:

```
python -m treeskel.seg_compare ensemble docs/results/segment_lined_up_rerun docs/results/segment_lined_up_seed1 \\
    docs/results/segment_lined_up_seed2 --out docs/results/segment_lined_up_ensemble   # then <out>_joined
python -m treeskel.seg_compare compare data/raw     # every variant present: report, rule's verdict, trees.csv
python -m treeskel.seg_compare compare              # the same without the dataset (no hand-mask scores)
```
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from treeskel.segment import (
    JOIN_GAP,
    MASK_COLUMNS,
    _measure_all,
    _pixel_columns,
    _read_run,
    _rescored_run,
    joined_run,
    load_masks,
    load_probabilities,
    mask_scores,
    prune_to_branches,
    save_probabilities,
)
from treeskel.shape import MASK_ZOOM, MEASURES, ShapeParams, mask_branches, unit_px

NAMED = "segment_lined_up"
"""The named run (session 12): the U-Net on the lined-up masks, scored on its joined masks."""

BASELINE = ("segment_lined_up_rerun", "segment_lined_up_seed1", "segment_lined_up_seed2")
"""Three samples of the named recipe with probabilities saved: session 12's re-run (seed 0) and seeds 1, 2."""

CLDICE = ("segment_lined_up_cldice", "segment_lined_up_cldice_seed1", "segment_lined_up_cldice_seed2")
"""Three samples of the recipe with the centre-line loss, seeds 0, 1, 2."""

ENSEMBLES = {"segment_lined_up_ensemble": BASELINE, "segment_lined_up_cldice_ensemble": CLDICE}
"""Each ensemble and its members."""

SECOND_ENSEMBLE = ("segment_lined_up_ensemble_b", ("segment_lined_up_seed3", "segment_lined_up_seed4",
                                                   "segment_lined_up_seed5"))
"""Optional: a second, independent ensemble of the named recipe (seeds 3-5), the only way to see how much an
ensemble itself changes between trainings. Reported if its runs exist; part of no decision."""

REPLICATES = (NAMED,) + BASELINE
"""The four samples of the named recipe, for the null check."""

ALPHA = 0.05
"""Significance level of the rule's tests (the project's naming rule since session 12)."""

HEADLINE = "v_opening"
"""The rule's measure: the V opening, the headline since session 10 (it needs no reference direction)."""

SIDE_REACH = JOIN_GAP / 2
"""How far (mask px) a stretch of centre line must reach beyond the trunk's and leaders' outline to count as a
side branch: a trunk radius (half of :data:`treeskel.segment.JOIN_GAP`, from the hand masks' median trunk
radius). Outline noise stays within 2.2 px on the local development hand masks and 2.8 px on 95% of the named
run's masks. Fixed before any of session 16's runs."""

MIN_PIECE = ShapeParams().min_points
"""Fewest centre-line pixels for an unjoined piece to count as a stray piece (the fewest a line fit takes)."""


def joined(name: str) -> str:
    """The folder of a run's joined masks: ``<name>_joined`` (the masks every rule scores).

    Args:
        name: A run's folder name.

    Returns:
        The joined run's folder name.
    """
    return name if name.endswith("_joined") else f"{name}_joined"


# --------------------------------------------------------------------------
# The masks' structure
# --------------------------------------------------------------------------

def leader_reach(params: ShapeParams = ShapeParams()) -> float:
    """The stretch the angles are fitted over, in mask px: a leader shorter than this is *short*.

    Args:
        params: The shape measures' settings (their fitted stretch).

    Returns:
        The reach in mask px.
    """
    return params.reach * unit_px(0.0, params, MASK_ZOOM)


def structure(mask: np.ndarray, params: ShapeParams = ShapeParams()) -> dict[str, float]:
    """What a mask holds besides its angles: its fork, short leaders, side branches and stray pieces.

    Args:
        mask: ``(256, 256)`` bool mask (joined, as scored).
        params: The shape measures' stretch (:func:`leader_reach`).

    Returns:
        ``{"fork", "left_length", "right_length", "short_leaders", "incomplete", "side_branches",
        "side_reach", "stray_pieces", "extra_fraction"}``: ``fork`` 1 or 0; the leaders' centre-line lengths
        from the fork (mask px); ``short_leaders`` 0-2; ``incomplete`` 1 if there is no fork or a short
        leader; ``side_branches`` the stretches of centre line reaching more than :data:`SIDE_REACH` beyond
        the trunk's and leaders' outline, and ``side_reach`` the farthest reach; ``stray_pieces`` the
        unjoined pieces with at least :data:`MIN_PIECE` centre-line px; ``extra_fraction`` the share of the
        mask outside the trunk and leaders. With no fork, everything but ``fork`` and ``incomplete`` is NaN.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    from treeskel.guided_gt import arc_length

    m = ndi.binary_fill_holes(np.asarray(mask, bool))
    nan = float("nan")
    out = {"fork": 0.0, "left_length": nan, "right_length": nan, "short_leaders": nan, "incomplete": 1.0,
           "side_branches": nan, "side_reach": nan, "stray_pieces": nan, "extra_fraction": nan}
    br = mask_branches(m) if m.any() else None
    if br is None:
        return out
    reach = leader_reach(params)
    lengths = [float(arc_length(br[s])[-1]) if len(br[s]) > 1 else 0.0 for s in ("left", "right")]
    short = sum(v < reach for v in lengths)
    # The trunk and leaders as drawn: each path point's own half-width around the paths. What lies outside is
    # extra: side branches, stray pieces, and the outline's small bumps.
    main = prune_to_branches(m, br)
    lab, n = ndi.label(m, np.ones((3, 3)))
    tree = lab[tuple(br["trunk"][-1])]
    beyond = ndi.distance_transform_edt(~main)              # distance outside the trunk's and leaders' outline
    sk = skeletonize(m)
    spurs, k = ndi.label(sk & (lab == tree) & ~main, np.ones((3, 3)))
    reaches = np.asarray(ndi.maximum(beyond, spurs, range(1, k + 1)), float).reshape(-1) if k else np.zeros(0)
    stray = sum(int((sk & (lab == j)).sum() >= MIN_PIECE) for j in range(1, n + 1) if j != tree)
    out.update(fork=1.0, left_length=lengths[0], right_length=lengths[1], short_leaders=float(short),
               incomplete=float(short > 0), side_branches=float((reaches > SIDE_REACH).sum()),
               side_reach=float(reaches.max()) if len(reaches) else 0.0, stray_pieces=float(stray),
               extra_fraction=float((m & ~main).sum() / m.sum()))
    return out


def centre_line_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """clDice of a mask against a hand mask, with its two halves (:func:`treeskel.segment.cl_dice`).

    Args:
        pred: Predicted 256x256 boolean mask.
        truth: Hand 256x256 boolean mask.

    Returns:
        ``{"cldice", "tprec", "tsens"}``: ``tprec`` the share of the prediction's centre line inside the hand
        mask (low with side branches or blobs), ``tsens`` the share of the hand mask's centre line inside the
        prediction (low with short or missing leaders: the prediction's coverage). NaN if either is empty.
    """
    from skimage.morphology import skeletonize

    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    sp, st = skeletonize(p), skeletonize(t)
    if not sp.any() or not st.any():
        return dict.fromkeys(("cldice", "tprec", "tsens"), float("nan"))
    prec, sens = float((sp & t).sum() / sp.sum()), float((st & p).sum() / st.sum())
    return {"cldice": 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0, "tprec": prec, "tsens": sens}


# --------------------------------------------------------------------------
# Ensembles
# --------------------------------------------------------------------------

def ensemble_run(members: Sequence[Path], out: Path | None = None, hands: dict[str, np.ndarray] | None = None,
                 threshold: float = 0.5, figures: Path | None = None,
                 progress: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """An ensemble of saved runs: the mean of their probabilities, thresholded, measured and saved as a run.

    Each member's probabilities (``probabilities.npz``) are out of fold: every
    tree was predicted by a network that never saw its row. The members must
    predict the same trees with the same folds, so the mean is out of fold
    too. The probabilities are stored to 1/255, so a pixel within 1/510 of
    the threshold can fall either way. The result is a run like any other
    (``report.json``, ``predictions.csv``, ``mask_scores.csv``,
    ``predicted_masks.npz``, and the mean ``probabilities.npz``); join it with
    :func:`treeskel.segment.joined_run`, as the command line does.

    Args:
        members: The runs' folders (two or more).
        out: The ensemble's folder (default: ``<first member>_ensemble``).
        hands: ``{tree: lined-up hand mask}`` for the pixel scores, or None.
        threshold: As the members' (:attr:`treeskel.segment.SegParams.threshold`).
        figures: Where its skill figure goes (None: none).
        progress: Where the summary goes.

    Returns:
        ``(out, paths that could not be written)``.

    Raises:
        ValueError: Fewer than two members, or members with different trees or folds.
    """
    members = [Path(m) for m in members]
    if len(members) < 2:
        raise ValueError("an ensemble needs two or more runs")
    runs = [_read_run(m) for m in members]
    probs = [load_probabilities(m / "probabilities.npz") for m in members]
    first = runs[0][1]
    trees = [t for t in first if t in probs[0]]
    for m, (_, info), p in zip(members, runs, probs):
        if set(info) != set(first) or set(p) != set(probs[0]):
            raise ValueError(f"{m} predicts other trees than {members[0]}")
        moved = [t for t in trees if info[t]["fold"] != first[t]["fold"]]
        if moved:
            raise ValueError(f"{m} puts {len(moved)} trees in other folds than {members[0]} (e.g. {moved[0]})")
    mean = {t: np.mean([p[t] for p in probs], axis=0).astype(np.float32) for t in trees}
    masks = {t: mean[t] > threshold for t in trees}
    seg = _measure_all(masks, trees)
    progress(f"Ensemble of {len(members)} runs ({', '.join(m.name for m in members)}): {len(trees)} trees, "
             f"mean probability above {threshold:g}")
    if hands:
        summary, note, pixel = _pixel_columns(members[0], trees, masks, hands)
    else:
        summary, note = {}, "no hand masks given: pixel scores not computed"
        pixel = {t: [""] * len(MASK_COLUMNS) for t in trees}
    header = ["tree", "row", "fold", "fork_found", *MASK_COLUMNS]
    rows = [[t, first[t]["row"], first[t]["fold"], int(np.isfinite(seg[i]).any()), *pixel[t]]
            for i, t in enumerate(trees)]
    reports = [r for r, _ in runs]
    extra = {"masks": summary, "masks_note": note, "epochs": [r.get("epochs") for r in reports],
             "minutes": round(sum(float(r.get("minutes") or 0) for r in reports), 1),
             "ensemble": {"members": [str(m) for m in members], "threshold": threshold,
                          "how": "mean of the members' out-of-fold probabilities, then the threshold"}}
    out = Path(out) if out else members[0].parent / f"{members[0].name}_ensemble"
    failed = _rescored_run(members[0], reports[0], first, trees, masks, seg, out, extra, header, rows, figures,
                           None, tuple((m.name, "segmentation") for m in members), progress)
    try:
        save_probabilities(out / "probabilities.npz", trees, np.stack([mean[t] for t in trees]))
    except OSError as e:
        failed.append(f"{out / 'probabilities.npz'}: {e.strerror or e}")
    return out, failed


# --------------------------------------------------------------------------
# Tables and tests
# --------------------------------------------------------------------------

def run_table(folder: Path, hands: dict[str, np.ndarray] | None = None, measure: str = HEADLINE
              ) -> dict[str, dict[str, float]]:
    """Per tree of a saved (joined) run: the prediction and absolute error on ``measure`` and :func:`structure`.

    The prediction is the run's ``segmentation`` column, where a mask with no
    fork already counts as its fold's mean shape (as every skill in the
    project). A mask with a fork but no value for ``measure`` (a leader too
    short to fit a line: session 14's stubs) counts as the mean shape too
    (``filled`` 1), where the runs' own reports leave it out; so every run is
    scored on the same trees, and a variant cannot drop its worst trees by
    making stubs. With ``hands``, also :func:`centre_line_scores` and the
    pixel Dice against the hand masks.

    Args:
        folder: A saved run's folder (its ``predictions.csv`` and ``predicted_masks.npz``).
        hands: Hand masks per tree id, for the centre-line and pixel scores; None to leave them out.
        measure: The measure scored.

    Returns:
        ``{tree: {"pred", "winter", "error", "filled", <structure>, ...}}``; trees whose winter value is
        unknown are left out.
    """
    folder = Path(folder)
    masks = load_masks(folder / "predicted_masks.npz")
    table: dict[str, dict[str, float]] = {}
    with open(folder / "predictions.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["measure"] == measure and r["winter"] and r["tree"] in masks:
                filled = not r["segmentation"]
                if filled and not r["mean shape"]:
                    continue
                w, p = float(r["winter"]), float(r["mean shape"] if filled else r["segmentation"])
                table[r["tree"]] = {"pred": p, "winter": w, "error": abs(p - w), "filled": float(filled)}
    for t, row in table.items():
        row.update(structure(masks[t]))
        if hands is not None and t in hands:
            row.update(centre_line_scores(masks[t], hands[t]))
            row["dice"] = mask_scores(masks[t], hands[t])["dice"]
    return table


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """``a`` against ``b`` tree by tree, on the trees both have a finite value for.

    Args:
        a: Values per tree.
        b: Values per tree, to compare with ``a``.

    Returns:
        ``{"n", "mean_a", "mean_b", "a_lower", "a_higher", "p"}``: the means, the shares of trees where ``a``
        is lower or higher, and the Wilcoxon signed-rank p (zero differences dropped; 1 if all are zero).
    """
    from scipy import stats

    keys = [t for t in a if t in b and np.isfinite(a[t]) and np.isfinite(b[t])]
    if len(keys) < 3:
        return {"n": len(keys)}
    x, y = np.array([a[t] for t in keys]), np.array([b[t] for t in keys])
    d = x - y
    return {"n": len(keys), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "a_lower": float(np.mean(d < 0)),
            "a_higher": float(np.mean(d > 0)), "p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def column(table: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    """``{tree: value}`` of one column of :func:`run_table`.

    Args:
        table: A :func:`run_table`.
        key: The column.

    Returns:
        The column's values per tree.
    """
    return {t: float(r.get(key, float("nan"))) for t, r in table.items()}


def tree_mean(tables: Sequence[dict[str, dict[str, float]]], key: str) -> dict[str, float]:
    """``{tree: mean of key over the tables}`` for the trees in every table (NaN values skipped; NaN if all are).

    Args:
        tables: Several :func:`run_table` tables.
        key: The column averaged.

    Returns:
        The means per tree.
    """
    trees = set.intersection(*(set(t) for t in tables))
    out = {}
    for t in sorted(trees):
        v = np.array([float(tab[t].get(key, float("nan"))) for tab in tables])
        out[t] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


GUARDS = ("side_branches", "incomplete")
"""The structure scores a change must not significantly increase (side branches; no fork or a short leader)."""


def verdict(candidate: Sequence[dict], comparator: Sequence[dict], alpha: float = ALPHA) -> dict:
    """The rule for one change: its runs' tables against the comparator's, each tree's values averaged over runs.

    Passed if the candidate's mean absolute error on the V opening is lower
    with Wilcoxon p < ``alpha``, and neither guard (:data:`GUARDS`) is higher
    with p < ``alpha``.

    Args:
        candidate: The change's runs (:func:`run_table` each).
        comparator: The named recipe's runs.
        alpha: The test's significance level.

    Returns:
        ``{"error": paired(...), "guards": {name: paired(...)}, "better", "guards_hold", "passed"}``.
    """
    err = paired(tree_mean(candidate, "error"), tree_mean(comparator, "error"))
    better = err.get("n", 0) >= 3 and err["mean_a"] < err["mean_b"] and err["p"] < alpha
    guards, hold = {}, True
    for g in GUARDS:
        c = paired(tree_mean(candidate, g), tree_mean(comparator, g))
        guards[g] = c
        if c.get("n", 0) >= 3 and c["mean_a"] > c["mean_b"] and c["p"] < alpha:
            hold = False
    return {"error": err, "guards": guards, "better": bool(better), "guards_hold": hold,
            "passed": bool(better and hold)}


def decide(tables: dict[str, dict], alpha: float = ALPHA) -> dict:
    """The rule (``rule.md``): each change against its comparator, then the named candidate.

    - **The centre-line loss:** its three runs against the named recipe's three
      (:data:`CLDICE` against :data:`BASELINE`), each tree's values averaged
      over the three.
    - **Ensembling:** the named recipe's ensemble against its own three members,
      each tree's values averaged over the members (the error one training
      makes on that tree, on average).
    - **The candidate:** the recipe whose loss passed (the centre-line loss if
      it passed, else the named recipe), ensembled if ensembling passed;
      the named run if neither passed.

    Args:
        tables: ``{run name: run_table(<run>_joined)}``; every run of the rule must be there.
        alpha: The test's significance level.

    Returns:
        ``{"complete": bool, "missing": [...], "cldice": verdict, "ensemble": verdict, "candidate": {"loss",
        "ensemble", "development_run", "description"}}``.
    """
    ens = "segment_lined_up_ensemble"
    need = list(BASELINE) + list(CLDICE) + [ens]
    missing = [n for n in need if n not in tables]
    out: dict = {"complete": not missing, "missing": missing}
    if missing:
        return out
    out["cldice"] = verdict([tables[n] for n in CLDICE], [tables[n] for n in BASELINE], alpha)
    out["ensemble"] = verdict([tables[ens]], [tables[n] for n in BASELINE], alpha)
    loss, ensemble = out["cldice"]["passed"], out["ensemble"]["passed"]
    # A recipe is named, not a run: any later training (the final test's, model3d's) is a new sample of it.
    out["candidate"] = {
        "loss": "centre-line" if loss else "named", "ensemble": bool(ensemble),
        "development_run": {(False, False): joined(NAMED), (True, False): "the mean of the 3 centre-line runs",
                            (False, True): joined(ens),
                            (True, True): joined("segment_lined_up_cldice_ensemble")}[(loss, ensemble)],
        "description": {(False, False): "the named U-Net, unchanged",
                        (True, False): "one U-Net trained with the centre-line loss",
                        (False, True): "three U-Nets of the named recipe, probabilities averaged",
                        (True, True): "three U-Nets trained with the centre-line loss, probabilities averaged"
                        }[(loss, ensemble)]}
    return out


def stability(tables: Sequence[dict[str, dict]]) -> dict:
    """How far samples of one recipe are from each other, over every pair of them, tree by tree.

    Args:
        tables: One :func:`run_table` per run of the recipe.

    Returns:
        ``{"pairs", "median_difference", "over_5", "status_flips"}``: the median of the absolute
        differences in the V opening, the share of tree pairs more than 5° apart, and the share whose
        ``incomplete`` status differs.
    """
    diffs, flips = [], []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            a, b = tables[i], tables[j]
            for t in set(a) & set(b):
                diffs.append(abs(a[t]["pred"] - b[t]["pred"]))
                flips.append(a[t]["incomplete"] != b[t]["incomplete"])
    if not diffs:
        return {"pairs": 0}
    d = np.array(diffs)
    return {"pairs": len(tables) * (len(tables) - 1) // 2, "median_difference": float(np.median(d)),
            "over_5": float(np.mean(d > 5)), "status_flips": float(np.mean(flips))}


def null_check(tables: dict[str, dict], names: Sequence[str] = REPLICATES) -> list[dict]:
    """Every pair of the named recipe's samples, tree by tree on the V opening's error: how often the rule's
    test would call two samples of one recipe different.

    Args:
        tables: ``{run name: run_table}``.
        names: The named recipe's runs to pair up.

    Returns:
        One dict per pair: the two runs' names (``a``, ``b``) and :func:`paired`'s comparison of their errors.
    """
    have = [n for n in names if n in tables]
    out = []
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            c = paired(column(tables[have[i]], "error"), column(tables[have[j]], "error"))
            out.append({"a": have[i], "b": have[j], **c})
    return out


def disagreement_check(members: Sequence[dict], ensemble: dict, top: float = 0.1) -> dict:
    """Does the members' spread on the V opening flag the trees the ensemble gets wrong?

    Args:
        members: The ensemble's members (:func:`run_table` each).
        ensemble: The ensemble's :func:`run_table`.
        top: The share of trees flagged.

    Returns:
        ``{"n", "spearman", "p", "mae_flagged", "mae_rest", "worst_caught"}``: the rank correlation of each
        tree's spread (SD over the members) with the ensemble's absolute error; the ensemble's error on the
        ``top`` share of trees with the largest spread and on the rest; and the share of its ``top`` largest
        errors that fall among those flagged trees (``top`` by chance).
    """
    from scipy import stats

    trees = sorted(set(ensemble).intersection(*(set(m) for m in members)))
    if len(trees) < 10:
        return {"n": len(trees)}
    spread = np.array([np.std([m[t]["pred"] for m in members]) for t in trees])
    err = np.array([ensemble[t]["error"] for t in trees])
    rho, p = stats.spearmanr(spread, err)
    k = max(1, int(round(top * len(trees))))
    flagged = np.zeros(len(trees), bool)
    flagged[np.argsort(-spread, kind="stable")[:k]] = True
    worst = np.zeros(len(trees), bool)
    worst[np.argsort(-err, kind="stable")[:k]] = True
    return {"n": len(trees), "spearman": float(rho), "p": float(p), "mae_flagged": float(err[flagged].mean()),
            "mae_rest": float(err[~flagged].mean()), "worst_caught": float((flagged & worst).sum() / k)}


STRUCTURE_SUMMARY = ("fork", "incomplete", "short_leaders", "side_branches", "stray_pieces", "extra_fraction",
                     "cldice", "tprec", "tsens", "dice")
"""The structure columns summarised per run (the last four need the hand masks)."""


def summarise_run(table: dict[str, dict], report: dict | None = None) -> dict:
    """A run's headline numbers: the V opening's MAE and skill, and its masks' structure.

    Args:
        table: A :func:`run_table`.
        report: The run's ``report.json``, for its skill; None to leave the skill out.

    Returns:
        ``{"n", "v_opening_mae", "v_opening_skill", <structure means and counts>, "no_fork", "stubs",
        "incomplete_trees"}``.
    """
    def finite(key):
        """The run's finite values of one column, over its trees."""
        return np.array([r[key] for r in table.values() if key in r and np.isfinite(r[key])], float)

    out: dict = {"n": len(table), "v_opening_mae": float(np.mean([r["error"] for r in table.values()]))}
    if report:
        out["v_opening_skill"] = report["scores"][HEADLINE]["segmentation"].get("skill")
    for key in STRUCTURE_SUMMARY:
        v = finite(key)
        if not len(v):
            continue
        out[key] = float(v.mean())
        if key in ("side_branches", "stray_pieces", "short_leaders"):
            out[f"trees_with_{key}"] = int((v > 0).sum())
    out["no_fork"] = int(sum(1 for r in table.values() if not r["fork"]))
    out["stubs"] = int(sum(1 for r in table.values() if r["fork"] and r["filled"]))
    out["incomplete_trees"] = int(sum(1 for r in table.values() if r["incomplete"]))
    return out


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

def present_runs(results: Path) -> list[str]:
    """The variants whose joined masks are saved in ``results``, in the rule's order.

    Args:
        results: The results folder (``docs/results``).

    Returns:
        The run names.
    """
    names = list(REPLICATES) + list(CLDICE) + list(ENSEMBLES) + [SECOND_ENSEMBLE[0], *SECOND_ENSEMBLE[1]]
    seen: list[str] = []
    for n in names:
        if n not in seen and (Path(results) / joined(n) / "predicted_masks.npz").exists():
            seen.append(n)
    return seen


def compare(results: Path, hands: dict[str, np.ndarray] | None = None,
            progress: Callable[[str], None] = print) -> tuple[dict, dict[str, dict]]:
    """Score every variant present, apply the rule, and run the diagnostics.

    Args:
        results: The results folder (``docs/results``).
        hands: ``{tree: lined-up hand mask}``, or None to skip the hand-mask scores.
        progress: Where the printed report goes.

    Returns:
        ``(report, {run name: run_table})``.
    """
    results = Path(results)
    names = present_runs(results)
    tables, reports = {}, {}
    for n in names:
        progress(f"Scoring {joined(n)}...")
        tables[n] = run_table(results / joined(n), hands)
        reports[n] = json.loads((results / joined(n) / "report.json").read_text())
    report: dict = {"runs": {n: summarise_run(tables[n], reports[n]) for n in names}, "hand_masks": hands is not None,
                    "side_reach": SIDE_REACH, "leader_reach": leader_reach(), "alpha": ALPHA}
    report["rule"] = decide(tables)
    # Descriptive comparisons the rule does not use: each candidate against the named run, single seeds pairwise.
    versus = {}
    for n in names:
        if n != NAMED and NAMED in tables:
            versus[f"{n} vs {NAMED}"] = paired(column(tables[n], "error"), column(tables[NAMED], "error"))
    for a, b in zip(CLDICE, BASELINE):
        if a in tables and b in tables:
            versus[f"{a} vs {b}"] = paired(column(tables[a], "error"), column(tables[b], "error"))
    if all(e in tables for e in ENSEMBLES):
        a, b = list(ENSEMBLES)
        versus[f"{b} vs {a}"] = paired(column(tables[b], "error"), column(tables[a], "error"))
    if "segment_lined_up_cldice_ensemble" in tables and all(c in tables for c in CLDICE):
        versus["segment_lined_up_cldice_ensemble vs its members"] = paired(
            column(tables["segment_lined_up_cldice_ensemble"], "error"), tree_mean([tables[c] for c in CLDICE], "error"))
    report["versus"] = versus
    report["null_check"] = null_check(tables)
    report["stability"] = {}
    for label, group in (("named recipe", BASELINE), ("centre-line loss", CLDICE)):
        if all(g in tables for g in group):
            report["stability"][label] = stability([tables[g] for g in group])
    second, members_b = SECOND_ENSEMBLE
    if all(e in tables for e in (list(ENSEMBLES)[0], second)):
        report["stability"]["two ensembles"] = stability([tables[list(ENSEMBLES)[0]], tables[second]])
    report["disagreement"] = {e: disagreement_check([tables[m] for m in ms], tables[e])
                              for e, ms in ENSEMBLES.items() if e in tables and all(m in tables for m in ms)}
    print_report(report, progress)
    return report, tables


def _p(v: float) -> str:
    """A p-value for printing: three decimals, or powers of ten below 0.001."""
    return f"{v:.1e}" if v < 0.001 else f"{v:.3f}"


def print_report(report: dict, progress: Callable[[str], None] = print) -> None:
    """Print :func:`compare`'s report: each run, the rule's verdict, then the diagnostics.

    Args:
        report: From :func:`compare`.
        progress: Prints a line.
    """
    runs = report["runs"]
    progress("\nEach run, its joined masks (development trees, out of fold):")
    progress(f"  {'run':40s} {'V MAE':>6s} {'skill':>6s} {'no fork':>7s} {'incompl.':>8s} {'side br.':>8s} "
             f"{'stray':>5s} {'extra':>6s}" + (f" {'clDice':>6s} {'tsens':>6s} {'tprec':>6s}"
                                                if report["hand_masks"] else ""))
    for n, s in runs.items():
        skill = s.get("v_opening_skill")
        line = (f"  {n:40s} {s['v_opening_mae']:6.3f} {float('nan') if skill is None else skill:+6.3f} "
                f"{s['no_fork']:7d} {s['incomplete_trees']:8d} {s.get('trees_with_side_branches', 0):8d} "
                f"{s.get('trees_with_stray_pieces', 0):5d} {s.get('extra_fraction', float('nan')):6.3f}")
        if report["hand_masks"]:
            line += f" {s.get('cldice', float('nan')):6.3f} {s.get('tsens', float('nan')):6.3f} " \
                    f"{s.get('tprec', float('nan')):6.3f}"
        progress(line)
    progress("  (V MAE over every tree with a winter value, a mask with no fork or no V opening counting as the mean "
             "shape;\n   skill as in the run's own report; incompl.: trees with no fork or a short leader; side br., "
             "stray: trees with any)")
    rule = report["rule"]
    progress("\nThe rule (rule.md):")
    if not rule["complete"]:
        progress(f"  not decided yet: missing {', '.join(rule['missing'])}")
    else:
        for key, title in (("cldice", "Centre-line loss (3 runs) against the named recipe (3 runs)"),
                           ("ensemble", "Ensemble of the named recipe against its 3 members")):
            v = rule[key]
            e = v["error"]
            progress(f"  {title}:")
            progress(f"    V opening MAE {e['mean_b']:.3f} -> {e['mean_a']:.3f}, lower on {e['a_lower']:.0%} of "
                     f"{e['n']} trees, p {_p(e['p'])}: {'better' if v['better'] else 'not better'}")
            for g, c in v["guards"].items():
                if c.get("n", 0) >= 3:
                    progress(f"    {g:14s} {c['mean_b']:.3f} -> {c['mean_a']:.3f} per tree, p {_p(c['p'])}")
            progress(f"    {'PASSED' if v['passed'] else 'NOT PASSED'}"
                     + ("" if v["guards_hold"] else " (a guard failed)"))
        c = rule["candidate"]
        progress(f"  Named candidate: {c['description']} (development numbers: {c['development_run']})")
    if report["versus"]:
        progress("\nTree by tree on the V opening's error (descriptive):")
        for k, c in report["versus"].items():
            if c.get("n", 0) >= 3:
                progress(f"  {k:66s} {c['mean_b']:.3f} -> {c['mean_a']:.3f}  lower {c['a_lower']:.0%}  "
                         f"p {_p(c['p'])}")
    if report["null_check"]:
        hits = sum(1 for c in report["null_check"] if c.get("p", 1) < report["alpha"])
        progress(f"\nNull check: samples of the named recipe against each other, {hits} of "
                 f"{len(report['null_check'])} pairs at p < {report['alpha']}:")
        for c in report["null_check"]:
            if c.get("n", 0) >= 3:
                progress(f"  {c['a']:28s} vs {c['b']:28s} {c['mean_a']:.3f} / {c['mean_b']:.3f}  p {_p(c['p'])}")
    for label, s in report["stability"].items():
        if s.get("pairs"):
            progress(f"\nBetween samples ({label}, {s['pairs']} pairs): V opening a median {s['median_difference']:.2f}° "
                     f"apart, {s['over_5']:.1%} of trees more than 5°, incomplete status differs on "
                     f"{s['status_flips']:.1%}")
    for e, d in report["disagreement"].items():
        if d.get("n", 0) >= 10:
            progress(f"\nMembers' spread as a warning ({e}): Spearman {d['spearman']:+.2f} (p {_p(d['p'])}) with the "
                     f"error; MAE {d['mae_flagged']:.2f}° on the 10% most spread, {d['mae_rest']:.2f}° on the rest; "
                     f"{d['worst_caught']:.0%} of the 10% largest errors flagged (10% by chance)")


def write_trees(path: Path, tables: dict[str, dict]) -> None:
    """Every run's per-tree values, one row per tree and run.

    Args:
        path: The CSV file.
        tables: ``{run name: run_table}``.
    """
    keys = ["pred", "winter", "error", "filled", "fork", "left_length", "right_length", "short_leaders", "incomplete",
            "side_branches", "side_reach", "stray_pieces", "extra_fraction", "cldice", "tprec", "tsens", "dice"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "tree", *keys])
        for n, table in tables.items():
            for t, r in table.items():
                w.writerow([n, t] + ["" if k not in r or not np.isfinite(r[k]) else f"{r[k]:.4g}" for k in keys])


def load_hands(data_dirs: Sequence[str], overlay: str | None, trees: Sequence[str] | None = None,
               test_rows: set[int] | None = None) -> dict[str, np.ndarray]:
    """The lined-up hand masks of the development trees (``overlay``: ``annotations/mask_overlay.json``).

    Args:
        data_dirs: Dataset folders.
        overlay: The hand masks' line-ups; None for the masks as drawn.
        trees: Only these tree ids (all if None).
        test_rows: Rows left out.

    Returns:
        256x256 boolean hand masks per tree id (trees skipped in the line-up are left out).
    """
    from treeskel.io import index_dataset
    from treeskel.segment import hand_mask
    from treeskel.summer_gt import load_overlays

    overlays = load_overlays(overlay) if overlay else None
    hands = {}
    for d in data_dirs:
        for rec in index_dataset(d).trees:
            if (test_rows and rec.row in test_rows) or (trees is not None and rec.tree_id not in trees):
                continue
            hm = hand_mask(rec, overlays)
            if hm is not None:
                hands[rec.tree_id] = hm
    return hands


def main(argv: Sequence[str] | None = None) -> int:
    """``ensemble``: build an ensemble from saved runs (and join it); ``compare``: score every variant and apply
    the rule.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status: 0 on success, 1 if the ensemble's folder already holds a run or the ensemble cannot
        be built, no variant's masks are saved, or an output file could not be written.
    """
    ap = argparse.ArgumentParser(prog="python -m treeskel.seg_compare", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("ensemble", help="average saved runs' probabilities into an ensemble, then join its masks")
    e.add_argument("members", nargs="+", help="the runs' folders (each with probabilities.npz)")
    e.add_argument("--out", required=True, help="the ensemble's folder (its joined masks go to <out>_joined)")
    e.add_argument("--data", nargs="*", default=[], help="dataset folder(s), for pixel scores against the hand masks")
    c = sub.add_parser("compare", help="score every variant present and apply the rule")
    c.add_argument("data_dirs", nargs="*", help="dataset folder(s), for the scores against the hand masks (optional)")
    c.add_argument("--out", default="docs/results/segment_variants", help="where the report goes")
    for p in (e, c):
        p.add_argument("--results", default="docs/results", help="the results folder")
        p.add_argument("--mask-overlay", default="annotations/mask_overlay.json", help="the hand masks' line-up")
        p.add_argument("--split", default="splits.json")
        p.add_argument("--figures", default="docs/figures")
    args = ap.parse_args(argv)
    test_rows = set(json.loads(Path(args.split).read_text())["test_rows"])

    if args.command == "ensemble":
        out = Path(args.out)
        if (out / "report.json").exists():
            print(f"{out} already holds a run; remove it or give another --out.")
            return 1
        hands = load_hands(args.data, args.mask_overlay, test_rows=test_rows) if args.data else None
        try:
            out, failed = ensemble_run(args.members, out, hands, figures=Path(args.figures))
        except (ValueError, FileNotFoundError) as err:
            print(f"Cannot build the ensemble: {err}")
            return 1
        print(f"\n--- The ensemble's masks with their pieces joined (gaps up to {JOIN_GAP:g} px) ---")
        failed += joined_run(out, figures=Path(args.figures))[1]
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0

    hands = None
    if args.data_dirs:
        hands = load_hands(args.data_dirs, args.mask_overlay, test_rows=test_rows)
        print(f"{len(hands)} lined-up development hand masks loaded")
    if not present_runs(Path(args.results)):
        print(f"No variant's joined masks in {args.results}.")
        return 1
    lines: list[str] = []

    def say(text: str) -> None:
        """Print a line of the report and keep it for ``output.txt``."""
        print(text, flush=True)
        lines.append(text)

    report, tables = compare(Path(args.results), hands, say)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn in (("report.json", lambda p: p.write_text(json.dumps(report, indent=2))),
                     ("trees.csv", lambda p: write_trees(p, tables)),
                     ("output.txt", lambda p: p.write_text("\n".join(lines) + "\n"))):
        try:
            fn(out / name)
        except OSError as err:
            failed.append(f"{out / name}: {err.strerror or err}")
    print(f"\nSaved {out}/ (report.json, trees.csv, output.txt)")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
