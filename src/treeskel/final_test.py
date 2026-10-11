"""The one final evaluation on the test rows: the named U-Net's masks, measured against the winter traces.

Until now nothing has touched the test rows (3, 12 and 16 of ``splits.json``):
every choice was made on the 449 development trees, by the 5 row-grouped
folds. This module runs once, at the end (session 11's third item):

1. **The model is named first.** ``model.json`` in the output folder names one
   saved development run (e.g. ``docs/results/segment_lined_up_joined``) and
   is committed before this module runs. Its recipe is taken as it stands:
   the network's parameters, the lined-up masks it trained on, whether its
   masks are grown (:func:`treeskel.segment.grow_mask`, with its lower
   threshold) and joined (:func:`treeskel.segment.join_pieces`, with its gap), whether its short
   leaders are continued and the masks pruned (:func:`treeskel.segment.continue_leaders`), whether
   short leaders' lines are completed (:func:`treeskel.segment.complete_measures`, with the model the
   run fitted on all its development trees), and
   the number of epochs: the median of the numbers its 5 folds chose (each on
   rows held out inside the training rows, never on a test row).
2. **One network is trained on all development trees** that the run's folds
   trained on, from the same start (ImageNet weights, the run's seed), and
   predicts the test trees' summer photos. Its masks are measured exactly as
   in development (:func:`treeskel.segment.mask_shape`).
3. **The truth is the winter traces of the test trees,** traced from the
   student's clicks (:mod:`treeskel.guided_gt`) and measured by
   :func:`treeskel.shape.measure_tree`, as the development trees were; they
   decide nothing, they only score.
4. **Scores:** per shape measure, the skill over the mean shape of the
   development trees (the model's training rows), with a 95% bootstrap
   interval over trees and the skill in each test row; the hand masks of the
   test rows (a person reading the same photos) are scored the same way, as
   the reference. A mask with no fork counts as the mean shape, as in
   development. The development folds' skill of the same run is reported
   beside it. Pixel scores against the test rows' hand masks are secondary:
   those masks are not lined up (they sit tens of pixels off the branch, as
   the development ones did), so Dice is given as drawn and after the best
   shift (an upper estimate).
5. **A secondary result, declared before the test** (``model.json``'s ``secondary.complete_from``): the
   same test masks with short leaders' lines completed and drawn on
   (:func:`treeskel.segment.complete_measures`, :func:`treeskel.segment.draw_completion`, with the model a
   development run fitted on all its trees). Reported beside the named model, not instead of it.

6. **An ensemble, reported once afterwards (session 16):** a ``model.json``
   with ``ensemble_seeds`` trains one network per seed (the named recipe and
   epochs, each from its own seed), averages their probabilities on the test
   photos, and scores the average exactly as above. Each member's own masks
   are scored too, and the ensemble is compared tree by tree with its
   members and with an earlier evaluation (``compare_with``, e.g. the final
   test). Its development skill comes from ``development_run``. It goes in
   its own folder, as a secondary line: the test rows were already used.

It refuses to run if the output folder already holds a report: one final
evaluation, not one per try. Outputs (``docs/results/final_test/`` by
default): ``report.json``, ``predictions.csv``, ``mask_scores.csv``,
``predicted_masks.npz``, ``test_shape_measures.csv``,
``test_winter_traces.npz``, ``output`` printed as it runs; figures
``final_test_skill.png`` and ``final_test_examples.jpg``.

Usage (the student's PC, after ``model.json`` is committed):

```
python -m treeskel.final_test data/raw
python -m treeskel.final_test data/raw --out docs/results/final_test_ensemble    # session 16's ensemble, once
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from treeskel.segment import (
    MASK_COLUMNS,
    SegParams,
    UNet,
    complete_measures,
    continue_leaders,
    draw_completion,
    draw_example,
    example_trees,
    fill_failures,
    grow_mask,
    hand_mask,
    join_pieces,
    load_frames,
    mask_scores,
    mask_shape,
    predict_masks,
    prob_frame,
    save_masks,
    train_seg,
    whole_photo,
)
from treeskel.shape import MEASURES
from treeskel.shape_cnn import _n
from treeskel.shape_model import HEADLINE, TRUNK_RELATIVE, VERTICAL, evaluate, load_table

SEED_OFFSET = 1000
"""Added to the run's seed for the final training, so that it is none of the folds' runs (which used the seed
plus ``100 * fold`` and ``+ 50``)."""


def named_recipe(model_file: Path) -> dict:
    """The named run's recipe (see the module docstring).

    Args:
        model_file: ``model.json``: ``{"run": "<results folder>", ...}``.

    Returns:
        ``{"named": model.json's content, "run": folder, "params": SegParams, "mask_overlay": path or None,
        "members": ensemble seeds or None, "development_run": folder, "compare_with": folder or None,
        "join_gap": gap or None, "grow_low": lower threshold or None, "continue_cone": cone or None,
        "complete_coef": the completion model or None,
        "epochs": int, "fold_epochs": [...],
        "development": the development run's report (the named run's unless ``development_run``)}``.

    Raises:
        ValueError: No epochs recorded, or ``ensemble_seeds`` with fewer than two different seeds.
    """
    named = json.loads(Path(model_file).read_text())
    run = Path(named["run"])
    report = json.loads((run / "report.json").read_text())
    known = {f.name for f in fields(SegParams)}
    p = {k: v for k, v in report["params"].items() if k in known}
    if isinstance(p.get("size"), list):
        p["size"] = tuple(p["size"])
    fold_epochs = [int(e["epochs"]) for e in report["epochs"]]
    if not fold_epochs:
        raise ValueError(f"{run}/report.json records no epochs chosen by its folds")
    seeds = named.get("ensemble_seeds")
    if seeds is not None:
        seeds = [int(v) for v in seeds]
        if len(seeds) < 2 or len(set(seeds)) < len(seeds):
            raise ValueError(f"ensemble_seeds needs two or more different seeds, got {seeds}")
    dev_run = Path(named.get("development_run") or run)
    development = report if dev_run == run else json.loads((dev_run / "report.json").read_text())
    return {"named": named, "run": str(run), "params": SegParams(**p), "mask_overlay": report.get("mask_overlay"),
            "members": seeds, "development_run": str(dev_run), "compare_with": named.get("compare_with"),
            "join_gap": (report.get("join") or {}).get("gap"), "grow_low": (report.get("grow") or {}).get("low"),
            "continue_cone": (report.get("continue") or {}).get("cone"),
            "complete_coef": (report.get("complete") or {}).get("coef_all"),
            "secondary_complete_coef": _secondary_coef(named),
            "epochs": int(round(statistics.median(fold_epochs))),
            "fold_epochs": fold_epochs, "development": development}


def _secondary_coef(named: dict) -> list[float] | None:
    """The completion model of the run ``model.json``'s ``secondary.complete_from`` names (fitted on all its
    development trees), for the declared secondary result; None if there is none."""
    src = (named.get("secondary") or {}).get("complete_from")
    if not src:
        return None
    return json.loads((Path(src) / "report.json").read_text())["complete"]["coef_all"]


def recipe_mask(prob: np.ndarray, recipe: dict, params: SegParams) -> tuple[np.ndarray, dict, bool]:
    """One test photo's probabilities to the recipe's mask and measures, as in development.

    Thresholded, then grown, joined and continued if the recipe does so, and
    measured (with short leaders' lines completed if the recipe does so).

    Returns:
        ``(mask, measures, whether a piece was joined)``.
    """
    p = prob_frame(prob, whole_photo(params))
    mask = p > params.threshold
    if recipe["grow_low"]:                                  # on the probabilities as development saved them
        mask = grow_mask(np.round(p * 255) / 255, recipe["grow_low"], mask, params.threshold)
    joined = False
    if recipe["join_gap"]:
        mask, k = join_pieces(mask, recipe["join_gap"])
        joined = k > 0
    if recipe["continue_cone"]:
        mask, _ = continue_leaders(mask, recipe["continue_cone"])
    meas = complete_measures(mask, recipe["complete_coef"])[0] if recipe["complete_coef"] else mask_shape(mask)[0]
    return mask, meas, joined


def bootstrap_skill(truth: np.ndarray, pred: np.ndarray, base: np.ndarray, n: int = 2000,
                    seed: int = 0) -> tuple[float, float]:
    """95% interval of the skill over the baseline (``1 − MSE / MSE(base)``), resampling trees.

    Trees of one row share their light and background, so this interval is
    narrower than one over rows would be; the skill in each row is reported too.
    """
    ok = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(base)
    t, p, b = truth[ok], pred[ok], base[ok]
    if len(t) < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(t), (n, len(t)))
    e, eb = (p - t)[idx] ** 2, (b - t)[idx] ** 2
    good = eb.mean(axis=1) > 0
    skill = 1 - e.mean(axis=1)[good] / eb.mean(axis=1)[good]
    lo, hi = np.percentile(skill, [2.5, 97.5])
    return float(lo), float(hi)


def best_shift_dice(pred: np.ndarray, truth: np.ndarray, across: int = 40, down: int = 20, step: int = 2) -> float:
    """The largest Dice of ``pred`` moved onto ``truth`` by up to ``across`` / ``down`` mask px (an upper
    estimate: the best of many shifts), as ``docs/results/segment/diagnostics.py``."""
    return max(mask_scores(np.roll(np.roll(pred, dy, 0), dx, 1), truth)["dice"]
               for dy in range(-down, down + 1, step) for dx in range(-across, across + 1, step))


def git_head() -> str | None:
    """The code's commit, if this is a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def plot_final(report: dict, path: Path, failed: list[str],
               title: str = "The final test: the named U-Net's masks against the test rows' winter traces",
               name: str = "The named U-Net on the test rows") -> None:
    """Skill over the development mean shape per measure on the test rows: the named model and the hand masks
    with their 95% bootstrap intervals, and the model's development-fold skill as an open marker. ``title`` and
    ``name`` (the model's legend entry) change for an ensemble's evaluation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.shape_model import AXIS, GRID, INK, LABELS, MUTED, SECONDARY, SURFACE, _save, _style

    groups = (("No reference", HEADLINE), ("From the trunk", TRUNK_RELATIVE), ("From the vertical", VERTICAL))
    rows_y, labels, pos = {}, [], 0.0
    for _, members in groups:
        for k in members:
            rows_y[k] = -pos
            labels.append(LABELS[k])
            pos += 1.0
        pos += 0.6
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    series = (("hand masks", "Hand masks of the test rows (a person's reading)", "#2a78d6", "o", 0.16),
              ("model", name, "#e34948", "h", 0.0),
              ("model, leaders completed", "The same, short leaders predicted on (secondary)", "#1baf7a", "d", -0.16))
    for key, name, colour, marker, off in series:
        for k, y0 in rows_y.items():
            sc = report["scores"][k].get(key, {})
            if sc.get("n", 0) < 3:
                continue
            lo, hi = sc.get("ci95", (None, None))
            if lo is not None and np.isfinite(lo):
                ax.plot([lo, hi], [y0 + off] * 2, color=colour, lw=0.9, alpha=0.7, zorder=2, solid_capstyle="round")
            ax.plot(sc["skill"], y0 + off, marker=marker, ms=5.5, color=colour, mec=SURFACE, mew=0.8, ls="none",
                    zorder=3, label=name if k == HEADLINE[0] else None)
    for k, y0 in rows_y.items():
        dev = report["scores"][k].get("development")
        if dev is not None:
            ax.plot(dev, y0, marker="h", ms=8.5, mfc="none", mec="#e34948", mew=1.0, ls="none", zorder=3,
                    label="The same run in development (5 folds)" if k == HEADLINE[0] else None)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(list(rows_y.values()), labels, fontsize=8, color=SECONDARY)
    ax.set_xlim(-0.25, 1.0)
    ax.set_ylim(-pos + 0.2, 1.0)                           # room above the first group's label
    ax.grid(axis="x", color=GRID, lw=0.5)
    gy = 0.0
    for group, members in groups:
        ax.text(-0.25, gy + 0.55, group, fontsize=7.5, color=MUTED, ha="left", va="bottom", style="italic")
        gy -= len(members) + 0.6
    ax.set_xlabel("Skill over the development mean shape, 1 − MSE / MSE(mean)   (lines: 95% interval over trees)",
                  fontsize=8, color=SECONDARY)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_title(f"{report['n_trees']} trees of test rows {', '.join(map(str, report['test_rows']))}; "
                 f"{report['no_fork']} with no fork count as the mean shape", loc="left", fontsize=8, color=SECONDARY)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=SECONDARY)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path, failed)
    plt.close(fig)


def secondary_from_saved(out: Path, figures: Path, progress: Callable[[str], None] = print) -> list[str]:
    """The declared secondary result, from a finished final test's saved files only.

    The same calculation as the one run makes when ``model.json`` declares a
    secondary result (:func:`treeskel.segment.complete_measures` and
    :func:`treeskel.segment.draw_completion` on the saved, joined test masks,
    with the completion model fitted on the development trees), for a run
    that was made before the declaration reached the code (session 12). No
    network is trained, no photo or winter trace is read again, and the
    headline ``report.json`` is left as it is. Writes ``secondary_report.json``,
    ``secondary_output.txt`` and ``predicted_masks_completed.npz``, and redraws
    the skill figure with the secondary series.

    Returns:
        Paths that could not be written.
    """
    from treeskel.segment import compare_with, load_masks, read_predictions

    out = Path(out)
    report = json.loads((out / "report.json").read_text())
    named = json.loads((out / "model.json").read_text())
    coef = _secondary_coef(named)
    if coef is None:
        raise ValueError(f"{out / 'model.json'} declares no secondary result")
    masks = load_masks(out / "predicted_masks.npz")
    truth = {r["tree"]: r for r in load_table(out / "test_shape_measures.csv")}
    test = [t for t in masks if t in truth]
    test_rows = report["test_rows"]
    y = np.array([[truth[t][f"winter_{k}"] for k in MEASURES] for t in test], float)
    base = np.broadcast_to(np.array([report["mean_shape"][k] for k in MEASURES], float), y.shape)
    row_index = np.array([test_rows.index(int(truth[t]["row"])) for t in test])
    seg2 = np.full(y.shape, np.nan)
    drawn, added = {}, {}
    for i, t in enumerate(test):
        seg2[i] = [complete_measures(masks[t], coef)[0][k] for k in MEASURES]
        drawn[t], added[t] = draw_completion(masks[t], coef)
    seg2 = fill_failures(seg2, np.array(base))
    sec = {"what": "the named model's test masks with short leaders' lines completed (declared before the test; "
                   "computed from the saved masks)", "complete_coef": coef,
           "trees_completed": int(sum(a.any() for a in added.values())), "scores": {}, "against_named": {}}
    for j, k in enumerate(MEASURES):
        sc = evaluate(y[:, j], seg2[:, j], base[:, j], row_index)
        if sc.get("n", 0) >= 3:
            sc["ci95"] = bootstrap_skill(y[:, j], seg2[:, j], base[:, j])
        sec["scores"][k] = sc
    # Against the named model's saved predictions, with their own winter values and rounding (2 decimals), so
    # that the trees the completion leaves alone tie exactly.
    named_pred = read_predictions(out / "predictions.csv", "model")
    mine = {(t, k): (named_pred[(t, k)][0], round(float(seg2[i, j]), 2)) for i, t in enumerate(test)
            for j, k in enumerate(MEASURES) if (t, k) in named_pred and np.isfinite(seg2[i, j])}
    sec["against_named"] = compare_with(mine, named_pred)
    lines = [f"Secondary result: {sec['what']}", f"  short leaders completed in {sec['trees_completed']} of {len(test)} "
             f"test trees (a = {coef[0]:+.2f}, c = {coef[1]:+.3f})",
             "  measure      skill  [95% interval]   (the named model)   tree by tree against it: MAE, closer, p"]
    for k in MEASURES:
        sc, c = sec["scores"][k], sec["against_named"].get(k)
        if sc.get("n", 0) < 3:
            continue
        head = report["scores"][k]["model"]["skill"]
        vs = (f"{c['mae_other']:5.2f} -> {c['mae_seg']:5.2f}  {c['closer']:4.0%}  p {c['wilcoxon_p']:.2f}" if c else "")
        lines.append(f"  {k:12s} {sc['skill']:+.2f}  [{sc['ci95'][0]:+.2f}, {sc['ci95'][1]:+.2f}]   ({head:+.2f})"
                     f"            {vs}")
    for line in lines:
        progress(line)
    failed: list[str] = []

    def write(path: Path, fn) -> None:
        """Write one output file; a file that cannot be written is listed, not fatal."""
        try:
            fn(path)
        except OSError as e:                                # e.g. open in a viewer on Windows
            failed.append(f"{path}: {e.strerror or e}")

    write(out / "secondary_report.json", lambda p: p.write_text(json.dumps(sec, indent=2)))
    write(out / "secondary_output.txt", lambda p: p.write_text("\n".join(lines) + "\n"))
    write(out / "predicted_masks_completed.npz", lambda p: save_masks(p, test, np.stack([drawn[t] for t in test])))
    both = json.loads(json.dumps(report))
    for k in MEASURES:
        both["scores"][k]["model, leaders completed"] = sec["scores"][k]
    Path(figures).mkdir(parents=True, exist_ok=True)
    plot_final(both, Path(figures) / "final_test_skill.png", failed)
    return failed


def ensemble_scores(member_probs: Sequence[np.ndarray], seeds: Sequence[int], test: Sequence[str], recipe: dict,
                    params: SegParams, y: np.ndarray, base: np.ndarray, row_index: np.ndarray,
                    ensemble_filled: np.ndarray, compare_with: Path | None) -> dict:
    """An ensemble's members, each scored on its own, and the ensemble against them and an earlier evaluation.

    Args:
        member_probs: Each network's ``(n, H, W)`` test probabilities.
        seeds: Their seeds.
        test: The test trees, in order.
        recipe: :func:`named_recipe`.
        params: The recipe's parameters.
        y: ``(n, measures)`` winter truth; ``base`` the development mean shape; ``row_index`` each tree's row.
        ensemble_filled: The ensemble's measures, no-fork masks as the mean shape.
        compare_with: An earlier evaluation's folder (its ``predictions.csv`` column ``model``), or None.

    Returns:
        ``{"seeds", "members": [{"seed", "no_fork", "scores", "v_opening": {tree: degrees}}], "vs_members",
        "compare_with", "vs_compare_with"}``:
        the comparisons are on the V opening's absolute error, tree by tree
        (:func:`treeskel.seg_compare.paired`; ``a`` is the ensemble).
    """
    from treeskel.seg_compare import paired
    from treeskel.segment import read_predictions

    v = MEASURES.index("v_opening")
    members, member_err = [], []
    for seed, probs in zip(seeds, member_probs):
        seg = np.full(y.shape, np.nan)
        for i in range(len(test)):
            meas = recipe_mask(probs[i], recipe, params)[1]
            seg[i] = [meas[k] for k in MEASURES]
        filled = fill_failures(seg, base)
        members.append({"seed": int(seed), "no_fork": int((~np.isfinite(seg).any(axis=1)).sum()),
                        "scores": {k: evaluate(y[:, j], filled[:, j], base[:, j], row_index)
                                   for j, k in enumerate(MEASURES)},
                        "v_opening": {t: (None if not np.isfinite(filled[i, v]) else round(float(filled[i, v]), 2))
                                      for i, t in enumerate(test)}})
        member_err.append(np.abs(filled[:, v] - y[:, v]))
    mine = {t: float(abs(ensemble_filled[i, v] - y[i, v])) for i, t in enumerate(test)}
    average = {t: float(np.mean([e[i] for e in member_err])) for i, t in enumerate(test)}
    out = {"seeds": [int(s) for s in seeds], "members": members, "vs_members": paired(mine, average),
           "compare_with": str(compare_with) if compare_with else None, "vs_compare_with": None}
    if compare_with is not None and (Path(compare_with) / "predictions.csv").exists():
        earlier = read_predictions(Path(compare_with) / "predictions.csv", "model")
        # The earlier evaluation saved 2 decimals; the ensemble is rounded alike, so equal predictions tie.
        theirs = {t: abs(earlier[(t, "v_opening")][1] - earlier[(t, "v_opening")][0])
                  for t in test if (t, "v_opening") in earlier}
        ours = {t: abs(round(float(ensemble_filled[i, v]), 2) - earlier[(t, "v_opening")][0])
                for i, t in enumerate(test) if (t, "v_opening") in earlier}
        out["vs_compare_with"] = paired(ours, theirs)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Train the named model on all development trees and score it once on the test rows."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.guided_gt import load_clicks
    from treeskel.io import MASK_SHAPE, index_dataset
    from treeskel.shape import ShapeParams, measure_tree, save_traces

    ap = argparse.ArgumentParser(prog="python -m treeskel.final_test", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="*", help="dataset folder(s) (not needed with --secondary-from-saved)")
    ap.add_argument("--out", default="docs/results/final_test", help="results folder; its model.json names the model")
    ap.add_argument("--model", help="the naming file (default: <out>/model.json)")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--table", default="data/shape/shape_measures.csv", help="development winter measures")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="test trees traced in parallel")
    ap.add_argument("--examples", type=int, default=6, help="test trees in the example figure")
    ap.add_argument("--secondary-from-saved", action="store_true",
                    help="only compute the declared secondary result from a finished test's saved files")
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)          # tests only
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.secondary_from_saved:
        if not (out / "report.json").exists():
            print(f"No {out / 'report.json'}: run the final test first.")
            return 1
        failed = secondary_from_saved(out, Path(args.figures))
        for f in failed:
            print(f"Could not write {f}")
        return 1 if failed else 0
    if not args.data_dirs:
        ap.error("give the dataset folder(s)")
    model_file = Path(args.model) if args.model else out / "model.json"
    if not model_file.exists():
        print(f"No {model_file}: name the model first (commit it), then run this once.")
        return 1
    if (out / "report.json").exists():
        print(f"{out / 'report.json'} exists: the final evaluation has been run. It is run once; not again.")
        return 1
    recipe = named_recipe(model_file)
    params = recipe["params"]
    log_lines: list[str] = []

    def say(msg: str = "") -> None:
        """Print a line and keep it for the log file."""
        print(msg, flush=True)
        log_lines.append(msg)

    say(f"Named model: {recipe['run']} ({recipe['named'].get('why', 'no reason recorded')})")
    say(f"  recipe: size {params.size}, encoder {params.encoder}, depth {params.depth}, batch {params.batch}, "
        f"seed {params.seed}; masks {'lined up (' + recipe['mask_overlay'] + ')' if recipe['mask_overlay'] else 'as drawn'}; "
        f"{'grown into pixels above ' + format(recipe['grow_low'], 'g') + ', ' if recipe['grow_low'] else ''}"
        f"{'pieces joined within ' + format(recipe['join_gap'], 'g') + ' px' if recipe['join_gap'] else 'no joining'}; "
        f"{'short leaders continued within ' + format(recipe['continue_cone'], 'g') + ' degrees and pruned; ' if recipe['continue_cone'] else ''}"
        f"{'short leader lines completed; ' if recipe['complete_coef'] else ''}"
        f"{recipe['epochs']} epochs (median of the folds' {recipe['fold_epochs']})")
    if recipe["members"]:
        say(f"  an ensemble: {len(recipe['members'])} networks, seeds {recipe['members']}, probabilities averaged; "
            f"development skill from {recipe['development_run']}")

    split = json.loads(Path(args.split).read_text())
    test_rows = sorted(int(r) for r in split["test_rows"])
    table = [r for r in load_table(args.table) if r["row"] not in test_rows]
    by_id = {r["tree"]: r for r in table}
    records = {t.tree_id: t for d in args.data_dirs for t in index_dataset(d).trees}
    dev = sorted(t for t in by_id if t in records)
    clicks = load_clicks(args.clicks)
    test = sorted(t for t, rec in records.items() if rec.row in test_rows and t in clicks
                  and clicks[t].complete and not clicks[t].skip)
    if len(dev) < 3 or not test:
        say(f"{len(dev)} development and {len(test)} test trees found; nothing to do.")
        return 1
    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError as e:
        say(f"PyTorch and torchvision are needed: {e}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    # 1. The truth: the test trees' winter traces and hand masks, measured as the development trees were.
    say(f"\nMeasuring the winter traces and hand masks of {_n(len(test), 'test tree')} (rows {test_rows})...")
    t0 = time.time()
    shape_params = ShapeParams()
    jobs = [(records[t], clicks[t].normalised(), shape_params, False) for t in test]
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(jobs)))) as ex:
        for row, _, trace in ex.map(measure_tree, jobs, chunksize=2):
            rows.append(row)
            traces.append(trace)
    say(f"  done in {time.time() - t0:.0f} s")
    truth = {r["tree"]: r for r in rows}

    def write(path: Path, fn) -> None:
        """Write one output file; a file that cannot be written is listed, not fatal."""
        try:
            fn(path)
        except OSError as e:                                # e.g. open in a viewer on Windows
            failed.append(f"{path}: {e.strerror or e}")

    def shape_table(path):
        """Write the test trees' shape measures (rounded to 3 decimals)."""
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    write(out / "test_shape_measures.csv", shape_table)
    write(out / "test_winter_traces.npz", lambda p: save_traces(p, test, traces))

    # 2. The model: trained once on all development trees, then the test photos.
    overlays = None
    if recipe["mask_overlay"]:
        from treeskel.summer_gt import load_overlays

        overlays = load_overlays(recipe["mask_overlay"])
    say(f"\nLoading {len(dev)} development trees' summer photos and hand masks...")
    images, masks, has = load_frames(records, dev, params.size, params.depth, lambda m: None, overlays)
    say(f"  {int(has.sum())} have a hand mask to train on")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.ascontiguousarray(images[has].transpose(0, 3, 1, 2))).to(device)
    m = torch.from_numpy(np.ascontiguousarray(masks[has])).to(device)
    channels = images.shape[3]
    del images, masks
    test_images, _, _ = load_frames(records, test, params.size, params.depth, lambda m: None, None)
    test_x = torch.from_numpy(np.ascontiguousarray(test_images.transpose(0, 3, 1, 2))).to(device)
    # One network, or one per seed of an ensemble: each from its seed's first weights (as in development) and
    # trained with that seed plus SEED_OFFSET. With one seed this is exactly the final test's training.
    seeds = recipe["members"] or [params.seed]
    member_probs = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)                             # the decoder's first weights, as in development
        model = UNet(params.encoder, not args.random_weights, channels=channels).to(device)
        what = f" (network {i + 1} of {len(seeds)}, seed {seed})" if len(seeds) > 1 else ""
        say(f"Training on {_n(int(has.sum()), 'tree')} for {_n(recipe['epochs'], 'epoch')} on {device}{what}...")
        try:
            train_seg(model, x, m, recipe["epochs"], params, seed + SEED_OFFSET,
                      on_epoch=lambda r: (say(f"  epoch {r['epoch']:3d}: train {r['train_loss']:.3f}")
                                          if r["epoch"] % 5 == 0 or r["epoch"] == 1 else None))
        except (OSError, RuntimeError) as e:               # e.g. out of GPU memory
            say(f"Training stopped: {type(e).__name__}: {e}\nNothing was scored; the evaluation can be run again "
                "(close other programs using the GPU).")
            return 1
        member_probs.append(predict_masks(model, test_x))
        del model
    minutes = round((time.time() - t0) / 60, 1)
    del x, m
    say(f"  trained in {minutes} min")
    prob = member_probs[0] if len(member_probs) == 1 else np.mean(member_probs, axis=0)

    # 3. Measure the predicted masks and score them against the winter traces.
    n = len(test)
    pred = np.zeros((n,) + MASK_SHAPE, bool)
    seg = np.full((n, len(MEASURES)), np.nan)
    second = recipe["secondary_complete_coef"]              # declared before the test: short leaders predicted on
    seg2 = np.full((n, len(MEASURES)), np.nan)
    drawn = np.zeros((n,) + MASK_SHAPE, bool)
    added = np.zeros((n,) + MASK_SHAPE, bool)
    pixel = {}
    joined = 0
    for i, t in enumerate(test):
        pred[i], meas, was_joined = recipe_mask(prob[i], recipe, params)
        joined += was_joined
        seg[i] = [meas[k] for k in MEASURES]
        if second is not None:
            seg2[i] = [complete_measures(pred[i], second)[0][k] for k in MEASURES]
            drawn[i], added[i] = draw_completion(pred[i], second)
        hm = hand_mask(records[t])                          # as drawn: the test masks are not lined up
        if hm is not None:
            pixel[t] = {**mask_scores(pred[i], hm), "dice_best_shift": best_shift_dice(pred[i], hm)}
    y = np.array([[truth[t][f"winter_{k}"] for k in MEASURES] for t in test], float)
    hand = np.array([[truth[t][f"hand_{k}"] for k in MEASURES] for t in test], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.array([[by_id[t][f"winter_{k}"] for k in MEASURES] for t in dev], float), axis=0)
    base = np.broadcast_to(mean, y.shape)
    no_fork = int((~np.isfinite(seg).any(axis=1)).sum())
    seg_filled = fill_failures(seg, np.array(base))
    seg2_filled = fill_failures(seg2, np.array(base))
    row_index = np.array([test_rows.index(records[t].row) for t in test])
    dev_scores = recipe["development"]["scores"]
    ensemble = None
    if len(seeds) > 1:
        ensemble = ensemble_scores(member_probs, seeds, test, recipe, params, y, np.array(base), row_index,
                                   seg_filled, Path(recipe["compare_with"]) if recipe["compare_with"] else None)
    report = {"named": recipe["named"], "run": recipe["run"], "params": asdict(params),
              "mask_overlay": recipe["mask_overlay"], "join_gap": recipe["join_gap"], "grow_low": recipe["grow_low"],
              "continue_cone": recipe["continue_cone"], "complete_coef": recipe["complete_coef"],
              "epochs": recipe["epochs"],
              "fold_epochs": recipe["fold_epochs"],
              "seed": params.seed + SEED_OFFSET if len(seeds) == 1 else [v + SEED_OFFSET for v in seeds],
              "ensemble": ensemble, "development_run": recipe["development_run"], "device": device,
              "weights": "random" if args.random_weights else "IMAGENET1K_V1", "minutes": minutes,
              "code": git_head(), "test_rows": test_rows, "n_trees": n, "n_dev": int(has.sum()),
              "no_fork": no_fork, "masks_joined": joined, "mean_shape": dict(zip(MEASURES, map(float, mean))),
              "secondary": ({"what": "the same masks with short leaders' lines completed (declared before the test)",
                             "complete_coef": second, "trees_completed": int(added.any(axis=(1, 2)).sum())}
                            if second is not None else None),
              "scores": {}, "masks": {}}
    for j, k in enumerate(MEASURES):
        entry = {"mean shape": evaluate(y[:, j], base[:, j], base[:, j], row_index),
                 "hand masks": evaluate(y[:, j], hand[:, j], base[:, j], row_index),
                 "model": evaluate(y[:, j], seg_filled[:, j], base[:, j], row_index),
                 "model, forks found": evaluate(y[:, j], seg[:, j], base[:, j], row_index),
                 "development": dev_scores.get(k, {}).get("segmentation", {}).get("skill")}
        if second is not None:
            entry["model, leaders completed"] = evaluate(y[:, j], seg2_filled[:, j], base[:, j], row_index)
        entry["mean shape"]["r"] = None
        for key, p in (("hand masks", hand[:, j]), ("model", seg_filled[:, j]),
                       ("model, leaders completed", seg2_filled[:, j])):
            if key not in entry:
                continue
            if entry[key].get("n", 0) >= 3:
                entry[key]["ci95"] = bootstrap_skill(y[:, j], p, base[:, j])
        report["scores"][k] = entry
    for c in MASK_COLUMNS + ("dice_best_shift",):
        v = np.array([s[c] for s in pixel.values()], float)
        v = v[np.isfinite(v)]
        report["masks"][c] = {"median": float(np.median(v)) if len(v) else None, "n": int(len(v))}

    say(f"\nTest rows {test_rows}: {n} trees; {no_fork} predicted masks with no fork (the mean shape)"
        + (f"; {joined} masks joined" if recipe["join_gap"] else ""))
    say("Skill over the development mean shape [95% bootstrap interval over trees] (per test row; the same run's "
        "development skill):")
    for title, names in (("Headline (no reference direction)", HEADLINE), ("From the trunk's axis", TRUNK_RELATIVE),
                         ("From the photo's vertical", VERTICAL)):
        say(f"\n{title}:")
        for k in names:
            for key in ("hand masks", "model", "model, leaders completed"):
                if key not in report["scores"][k]:
                    continue
                sc = report["scores"][k][key]
                if sc.get("n", 0) < 3:
                    continue
                lo, hi = sc.get("ci95", (float("nan"), float("nan")))
                rowsk = " ".join("   -" if v is None else f"{v:+.2f}" for v in sc.get("fold_skill", []))
                dev_k = report["scores"][k]["development"] if key == "model" else None
                say(f"  {k:12s} {key:24s} n {sc['n']:3d}  MAE {sc['mae']:5.2f}  skill {sc['skill']:+.2f} "
                    f"[{lo:+.2f}, {hi:+.2f}]  r {sc['r']:.2f}  better than the mean {sc['improved']:4.0%}  "
                    f"rows {rowsk}" + (f"  (development {dev_k:+.2f})" if dev_k is not None else ""))
    if ensemble is not None:
        say(f"\nThe ensemble's {len(seeds)} networks, each on its own (V opening):")
        for mem in ensemble["members"]:
            sc = mem["scores"]["v_opening"]
            if sc.get("n", 0) >= 3:
                say(f"  seed {mem['seed']}: MAE {sc['mae']:5.2f}  skill {sc['skill']:+.2f}  "
                    f"({mem['no_fork']} with no fork)")
        c = ensemble["vs_members"]
        if c.get("n", 0) >= 3:
            say(f"Ensemble against its networks' average, tree by tree: MAE {c['mean_b']:.2f} -> {c['mean_a']:.2f}, "
                f"lower on {c['a_lower']:.0%} of {c['n']} trees, Wilcoxon p {c['p']:.3f}")
        c = ensemble.get("vs_compare_with")
        if c and c.get("n", 0) >= 3:
            say(f"Ensemble against {ensemble['compare_with']}, tree by tree: MAE {c['mean_b']:.2f} -> "
                f"{c['mean_a']:.2f}, lower on {c['a_lower']:.0%}, Wilcoxon p {c['p']:.3f}")
    say("\nPixels against the test rows' hand masks as drawn (not lined up; secondary), median:")
    for c, s in report["masks"].items():
        if s["n"]:
            say(f"  {c:16s} {s['median']:.3f}")

    def predictions(path):
        """Write every test tree's measures: winter, the mean shape, the hand masks, the model (also completed)."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "measure", "winter", "mean shape", "hand masks", "model",
                        "model, leaders completed"])
            for i, t in enumerate(test):
                for j, k in enumerate(MEASURES):
                    w.writerow([t, records[t].row, k] + ["" if not np.isfinite(v) else f"{v:.2f}"
                                                         for v in (y[i, j], base[i, j], hand[i, j], seg_filled[i, j],
                                                                   seg2_filled[i, j])])

    def mask_table(path):
        """Write each test mask's pixel scores against its hand mask as drawn, and the Dice after the best shift."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "fork_found", *MASK_COLUMNS, "dice_best_shift"])
            for i, t in enumerate(test):
                s = pixel.get(t, {})
                w.writerow([t, records[t].row, int(np.isfinite(seg[i]).any())]
                           + ["" if c not in s or not np.isfinite(s[c]) else f"{s[c]:.4f}"
                              for c in MASK_COLUMNS + ("dice_best_shift",)])

    write(out / "predictions.csv", predictions)
    write(out / "mask_scores.csv", mask_table)
    write(out / "predicted_masks.npz", lambda p: save_masks(p, test, pred))
    if second is not None:
        write(out / "predicted_masks_completed.npz", lambda p: save_masks(p, test, drawn))
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    stem = "final_test" if ensemble is None else out.name          # the final test's names stay as they were
    if ensemble is None:
        plot_final(report, figures / f"{stem}_skill.png", failed)
    else:
        plot_final(report, figures / f"{stem}_skill.png", failed,
                   title=f"An ensemble of {len(seeds)} U-Nets on the test rows (secondary: the rows were used once "
                         "before)", name=f"{len(seeds)} U-Nets of the named recipe, probabilities averaged")
    if args.examples > 0:
        from PIL import Image

        v = MEASURES.index("v_opening")
        err = {t: -abs(seg_filled[i, v] - y[i, v]) for i, t in enumerate(test)
               if np.isfinite(seg_filled[i, v]) and np.isfinite(y[i, v])}
        tiles = []
        for t in example_trees(err, args.examples):          # worst to best V opening
            i = test.index(t)
            tr = dict(zip(("trunk", "left", "right", "fork_radius"), (traces[i][p] for p in
                                                                      ("trunk", "left", "right", "fork_radius"))))
            label = f"{t}  V {seg_filled[i, v]:.0f} deg, winter {y[i, v]:.0f}"
            tiles.append(draw_example(records[t], pred[i], tr, label,
                                      continuation=added[i] if second is not None else None))
        if tiles:
            sheet = Image.new("RGB", (max(im.width for im in tiles), sum(im.height for im in tiles)))
            top = 0
            for im in tiles:
                sheet.paste(im, (0, top))
                top += im.height
            write(figures / f"{stem}_examples.jpg", lambda p: sheet.save(p, quality=85))
    write(out / "output.txt", lambda p: p.write_text("\n".join(log_lines) + "\n"))
    write(out / "report.json", lambda p: p.write_text(json.dumps(report, indent=2)))   # last: marks it done
    say(f"\nSaved {out}/ and the figures {stem}_skill.png, {stem}_examples.jpg in {figures}/")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
