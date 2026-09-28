"""The one final evaluation on the test rows: the named U-Net's masks, measured against the winter traces.

Until now nothing has touched the test rows (3, 12 and 16 of ``splits.json``):
every choice was made on the 449 development trees, by the 5 row-grouped
folds. This module runs once, at the end (session 11's third item):

1. **The model is named first.** ``model.json`` in the output folder names one
   saved development run (e.g. ``docs/results/segment_lined_up_joined``) and
   is committed before this module runs. Its recipe is taken as it stands:
   the network's parameters, the lined-up masks it trained on, whether its
   masks are joined (:func:`treeskel.segment.join_pieces`, with its gap), and
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

It refuses to run if the output folder already holds a report: one final
evaluation, not one per try. Outputs (``docs/results/final_test/`` by
default): ``report.json``, ``predictions.csv``, ``mask_scores.csv``,
``predicted_masks.npz``, ``test_shape_measures.csv``,
``test_winter_traces.npz``, ``output`` printed as it runs; figures
``final_test_skill.png`` and ``final_test_examples.jpg``.

Usage (the student's PC, after ``model.json`` is committed)::

    python -m treeskel.final_test data/raw
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
    draw_example,
    example_trees,
    fill_failures,
    hand_mask,
    join_pieces,
    load_frames,
    mask_scores,
    mask_shape,
    predict_masks,
    save_masks,
    to_mask_size,
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
        "join_gap": gap or None, "epochs": int, "fold_epochs": [...], "development": the run's report}``.
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
    return {"named": named, "run": str(run), "params": SegParams(**p), "mask_overlay": report.get("mask_overlay"),
            "join_gap": (report.get("join") or {}).get("gap"), "epochs": int(round(statistics.median(fold_epochs))),
            "fold_epochs": fold_epochs, "development": report}


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


def plot_final(report: dict, path: Path, failed: list[str]) -> None:
    """Skill over the development mean shape per measure on the test rows: the named model and the hand masks
    with their 95% bootstrap intervals, and the model's development-fold skill as an open marker."""
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
    series = (("hand masks", "Hand masks of the test rows (a person's reading)", "#2a78d6", "o", 0.12),
              ("model", "The named U-Net on the test rows", "#e34948", "h", -0.12))
    for key, name, colour, marker, off in series:
        for k, y0 in rows_y.items():
            sc = report["scores"][k][key]
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
            ax.plot(dev, y0 - 0.12, marker="h", ms=5.5, mfc="none", mec="#e34948", mew=1.0, ls="none", zorder=3,
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
    fig.suptitle("The final test: the named U-Net's masks against the test rows' winter traces", x=0.02, ha="left",
                 fontsize=10, color=INK, fontweight="bold")
    ax.set_title(f"{report['n_trees']} trees of test rows {', '.join(map(str, report['test_rows']))}; "
                 f"{report['no_fork']} with no fork count as the mean shape", loc="left", fontsize=8, color=SECONDARY)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=SECONDARY)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, path, failed)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Train the named model on all development trees and score it once on the test rows."""
    from concurrent.futures import ProcessPoolExecutor

    from treeskel.guided_gt import load_clicks
    from treeskel.io import MASK_SHAPE, index_dataset
    from treeskel.shape import ShapeParams, measure_tree, save_traces

    ap = argparse.ArgumentParser(prog="python -m treeskel.final_test", description=__doc__.splitlines()[0])
    ap.add_argument("data_dirs", nargs="+", help="dataset folder(s)")
    ap.add_argument("--out", default="docs/results/final_test", help="results folder; its model.json names the model")
    ap.add_argument("--model", help="the naming file (default: <out>/model.json)")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--split", default="splits.json")
    ap.add_argument("--table", default="data/shape/shape_measures.csv", help="development winter measures")
    ap.add_argument("--clicks", default="annotations/winter_clicks.json")
    ap.add_argument("--device", help="cuda or cpu (default: the GPU if there is one)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="test trees traced in parallel")
    ap.add_argument("--examples", type=int, default=6, help="test trees in the example figure")
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)          # tests only
    args = ap.parse_args(argv)

    out = Path(args.out)
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
        print(msg, flush=True)
        log_lines.append(msg)

    say(f"Named model: {recipe['run']} ({recipe['named'].get('why', 'no reason recorded')})")
    say(f"  recipe: size {params.size}, encoder {params.encoder}, depth {params.depth}, batch {params.batch}, "
        f"seed {params.seed}; masks {'lined up (' + recipe['mask_overlay'] + ')' if recipe['mask_overlay'] else 'as drawn'}; "
        f"{'pieces joined within ' + format(recipe['join_gap'], 'g') + ' px' if recipe['join_gap'] else 'no joining'}; "
        f"{recipe['epochs']} epochs (median of the folds' {recipe['fold_epochs']})")

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
        try:
            fn(path)
        except OSError as e:                                # e.g. open in a viewer on Windows
            failed.append(f"{path}: {e.strerror or e}")

    def shape_table(path):
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
    torch.manual_seed(params.seed)                          # the decoder's first weights, as in development
    model = UNet(params.encoder, not args.random_weights, channels=images.shape[3]).to(device)
    x = torch.from_numpy(np.ascontiguousarray(images[has].transpose(0, 3, 1, 2))).to(device)
    m = torch.from_numpy(np.ascontiguousarray(masks[has])).to(device)
    del images, masks
    say(f"Training on {_n(int(has.sum()), 'tree')} for {_n(recipe['epochs'], 'epoch')} on {device}...")
    t0 = time.time()
    try:
        train_seg(model, x, m, recipe["epochs"], params, params.seed + SEED_OFFSET,
                  on_epoch=lambda r: (say(f"  epoch {r['epoch']:3d}: train {r['train_loss']:.3f}")
                                      if r["epoch"] % 5 == 0 or r["epoch"] == 1 else None))
    except (OSError, RuntimeError) as e:                   # e.g. out of GPU memory
        say(f"Training stopped: {type(e).__name__}: {e}\nNothing was scored; the evaluation can be run again "
            "(close other programs using the GPU).")
        return 1
    minutes = round((time.time() - t0) / 60, 1)
    del x, m
    say(f"  trained in {minutes} min")
    test_images, _, _ = load_frames(records, test, params.size, params.depth, lambda m: None, None)
    prob = predict_masks(model, torch.from_numpy(np.ascontiguousarray(test_images.transpose(0, 3, 1, 2))).to(device))

    # 3. Measure the predicted masks and score them against the winter traces.
    n = len(test)
    pred = np.zeros((n,) + MASK_SHAPE, bool)
    seg = np.full((n, len(MEASURES)), np.nan)
    pixel = {}
    joined = 0
    for i, t in enumerate(test):
        pred[i] = to_mask_size(prob[i], params.threshold, whole_photo(params))
        if recipe["join_gap"]:
            pred[i], k = join_pieces(pred[i], recipe["join_gap"])
            joined += k > 0
        meas, _ = mask_shape(pred[i])
        seg[i] = [meas[k] for k in MEASURES]
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
    row_index = np.array([test_rows.index(records[t].row) for t in test])
    dev_scores = recipe["development"]["scores"]
    report = {"named": recipe["named"], "run": recipe["run"], "params": asdict(params),
              "mask_overlay": recipe["mask_overlay"], "join_gap": recipe["join_gap"], "epochs": recipe["epochs"],
              "fold_epochs": recipe["fold_epochs"], "seed": params.seed + SEED_OFFSET, "device": device,
              "weights": "random" if args.random_weights else "IMAGENET1K_V1", "minutes": minutes,
              "code": git_head(), "test_rows": test_rows, "n_trees": n, "n_dev": int(has.sum()),
              "no_fork": no_fork, "masks_joined": joined, "mean_shape": dict(zip(MEASURES, map(float, mean))),
              "scores": {}, "masks": {}}
    for j, k in enumerate(MEASURES):
        entry = {"mean shape": evaluate(y[:, j], base[:, j], base[:, j], row_index),
                 "hand masks": evaluate(y[:, j], hand[:, j], base[:, j], row_index),
                 "model": evaluate(y[:, j], seg_filled[:, j], base[:, j], row_index),
                 "model, forks found": evaluate(y[:, j], seg[:, j], base[:, j], row_index),
                 "development": dev_scores.get(k, {}).get("segmentation", {}).get("skill")}
        entry["mean shape"]["r"] = None
        for key, p in (("hand masks", hand[:, j]), ("model", seg_filled[:, j])):
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
            for key in ("hand masks", "model"):
                sc = report["scores"][k][key]
                if sc.get("n", 0) < 3:
                    continue
                lo, hi = sc.get("ci95", (float("nan"), float("nan")))
                rowsk = " ".join("   -" if v is None else f"{v:+.2f}" for v in sc.get("fold_skill", []))
                dev_k = report["scores"][k]["development"] if key == "model" else None
                say(f"  {k:12s} {key:10s} n {sc['n']:3d}  MAE {sc['mae']:5.2f}  skill {sc['skill']:+.2f} "
                    f"[{lo:+.2f}, {hi:+.2f}]  r {sc['r']:.2f}  better than the mean {sc['improved']:4.0%}  "
                    f"rows {rowsk}" + (f"  (development {dev_k:+.2f})" if dev_k is not None else ""))
    say("\nPixels against the test rows' hand masks as drawn (not lined up; secondary), median:")
    for c, s in report["masks"].items():
        if s["n"]:
            say(f"  {c:16s} {s['median']:.3f}")

    def predictions(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tree", "row", "measure", "winter", "mean shape", "hand masks", "model"])
            for i, t in enumerate(test):
                for j, k in enumerate(MEASURES):
                    w.writerow([t, records[t].row, k] + ["" if not np.isfinite(v) else f"{v:.2f}"
                                                         for v in (y[i, j], base[i, j], hand[i, j], seg_filled[i, j])])

    def mask_table(path):
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
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plot_final(report, figures / "final_test_skill.png", failed)
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
            tiles.append(draw_example(records[t], pred[i], tr, label))
        if tiles:
            sheet = Image.new("RGB", (max(im.width for im in tiles), sum(im.height for im in tiles)))
            top = 0
            for im in tiles:
                sheet.paste(im, (0, top))
                top += im.height
            write(figures / "final_test_examples.jpg", lambda p: sheet.save(p, quality=85))
    write(out / "output.txt", lambda p: p.write_text("\n".join(log_lines) + "\n"))
    write(out / "report.json", lambda p: p.write_text(json.dumps(report, indent=2)))   # last: marks it done
    say(f"\nSaved {out}/ and the figures final_test_skill.png, final_test_examples.jpg in {figures}/")
    for f in failed:
        print(f"Could not write {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
