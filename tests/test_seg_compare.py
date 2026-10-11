"""Tests for treeskel.seg_compare: structure scores, ensembles, the rule and its diagnostics (no PyTorch needed)."""

import csv
import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import line

from treeskel.seg_compare import (
    BASELINE,
    CLDICE,
    NAMED,
    SIDE_REACH,
    centre_line_scores,
    decide,
    disagreement_check,
    ensemble_run,
    leader_reach,
    main,
    null_check,
    paired,
    run_table,
    stability,
    structure,
    tree_mean,
    verdict,
)
from treeskel.segment import load_masks, load_probabilities, save_masks, save_probabilities

ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "docs/results/segment_lined_up_rerun"


def thick(mask_shape, a, b, half_width):
    """Pixels within ``half_width`` of the segment from ``a`` to ``b``."""
    path = np.zeros(mask_shape, bool)
    path[line(*a, *b)] = True
    return ndi.distance_transform_edt(~path) <= half_width


def v_tree(left_top=(5, 30), side_branch=False, blob=False):
    """A 256x256 V: trunk rows 180-255 (half-width 9), leaders to ``left_top`` and (5, 226) (half-width 4)."""
    m = np.zeros((256, 256), bool)
    m[180:, 119:138] = True
    m |= thick(m.shape, (180, 128), left_top, 4)
    m |= thick(m.shape, (180, 128), (5, 226), 4)
    if side_branch:
        m |= thick(m.shape, (100, 173), (80, 215), 2)        # off the right leader, about 30 px out
    if blob:
        m |= thick(m.shape, (40, 120), (70, 125), 3)          # a separate piece in the V
    return m


# --- structure ------------------------------------------------------------------------------

def test_a_complete_v_has_no_extra_structure():
    s = structure(v_tree())
    assert s["fork"] == 1 and s["short_leaders"] == 0 and s["incomplete"] == 0
    assert s["side_branches"] == 0 and s["stray_pieces"] == 0 and s["side_reach"] < SIDE_REACH
    assert min(s["left_length"], s["right_length"]) > leader_reach()


def test_a_short_leader_a_side_branch_and_a_stray_piece_are_counted():
    short = structure(v_tree(left_top=(120, 90)))             # the left leader stops after about 70 px
    assert short["short_leaders"] == 1 and short["incomplete"] == 1
    side = structure(v_tree(side_branch=True))
    assert side["side_branches"] == 1 and side["side_reach"] > 20 and side["incomplete"] == 0
    stray = structure(v_tree(blob=True))
    assert stray["stray_pieces"] == 1 and stray["side_branches"] == 0


def test_a_mask_with_no_fork_is_incomplete_and_nothing_else():
    s = structure(np.zeros((256, 256), bool))
    assert s["fork"] == 0 and s["incomplete"] == 1 and np.isnan(s["side_branches"])
    bar = np.zeros((256, 256), bool)
    bar[50:250, 120:136] = True
    assert structure(bar)["fork"] == 0


def test_leader_reach_is_the_fitted_stretch_in_mask_px():
    assert leader_reach() == pytest.approx(157.7, abs=0.1)


def test_session_14s_short_leader_count_is_reproduced():
    """Session 14: 37 short predicted leaders in 36 of the named run's 447 masks with a fork."""
    masks = load_masks(ROOT / "docs/results/segment_lined_up_joined/predicted_masks.npz")
    rows = [structure(m) for m in masks.values()]
    forks = [r for r in rows if r["fork"]]
    assert len(forks) == 447
    assert sum(r["short_leaders"] > 0 for r in forks) == 36 and sum(r["short_leaders"] for r in forks) == 37


def test_hand_masks_have_no_side_branches():
    from treeskel.io import index_dataset, load_mask

    test_rows = set(json.loads((ROOT / "splits.json").read_text())["test_rows"])
    recs = [t for d in ("sample", "tuning") for t in index_dataset(ROOT / "data" / d).trees
            if t.row not in test_rows and t.has_mask]
    assert len(recs) >= 20
    for rec in recs:
        s = structure(load_mask(rec.paths[("summer", "mask")]))
        assert s["fork"] == 1 and s["side_branches"] == 0 and s["side_reach"] <= 3, rec.tree_id


def test_centre_line_scores():
    full = v_tree()
    same = centre_line_scores(full, full)
    assert same == {"cldice": 1.0, "tprec": 1.0, "tsens": 1.0}
    short = centre_line_scores(v_tree(left_top=(120, 90)), full)
    assert short["tprec"] == pytest.approx(1.0) and short["tsens"] < 0.85     # coverage falls, precision stays
    side = centre_line_scores(v_tree(side_branch=True), full)
    assert side["tprec"] < 0.97 and side["tsens"] == pytest.approx(1.0)
    assert all(np.isnan(v) for v in centre_line_scores(np.zeros_like(full), full).values())


# --- saved runs -------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def rerun_files(trees: int = 12):
    """The re-run's first trees: their masks and probabilities, read once (the full files are large)."""
    with np.load(RERUN / "probabilities.npz") as z:
        ids = [str(t) for t in z["ids"]]
        keep = sorted(ids)[:trees]
        probs = {t: z["prob"][ids.index(t)].astype(np.float32) / 255 for t in keep}
    masks = load_masks(RERUN / "predicted_masks.npz")
    return keep, {t: masks[t] for t in keep}, probs


def small_run(dst: Path, trees: int = 12, flip_fold: str | None = None, scale: float = 1.0) -> Path:
    """A copy of the re-run reduced to its first trees (probabilities times ``scale``)."""
    dst.mkdir(parents=True)
    shutil.copy(RERUN / "report.json", dst / "report.json")
    with open(RERUN / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keep, masks, probs = rerun_files()
    keep = keep[:trees]
    with open(dst / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            if r["tree"] in keep:
                w.writerow({**r, "fold": "9"} if r["tree"] == flip_fold else r)
    save_masks(dst / "predicted_masks.npz", keep, np.stack([masks[t] for t in keep]))
    save_probabilities(dst / "probabilities.npz", keep, np.stack([np.clip(probs[t] * scale, 0, 1) for t in keep]))
    with open(RERUN / "mask_scores.csv", newline="") as f:
        ms = [r for r in csv.DictReader(f) if r["tree"] in keep]
    with open(dst / "mask_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ms[0]))
        w.writeheader()
        w.writerows(ms)
    return dst


def test_saved_masks_layout_is_what_small_run_writes(tmp_path):
    run = small_run(tmp_path / "r")
    assert len(load_masks(run / "predicted_masks.npz")) == 12


def test_an_ensemble_of_copies_is_the_run_itself(tmp_path):
    runs = [small_run(tmp_path / f"m{i}") for i in range(3)]
    out, failed = ensemble_run(runs, tmp_path / "ens", progress=lambda m: None)
    assert not failed
    ens, own = load_masks(out / "predicted_masks.npz"), load_masks(runs[0] / "predicted_masks.npz")
    probs = load_probabilities(runs[0] / "probabilities.npz")
    for t in own:
        differ = ens[t] != own[t]
        assert np.all(np.abs(probs[t][differ] - 0.5) <= 1 / 510 + 1e-6)          # only at the threshold (storage)
    report = json.loads((out / "report.json").read_text())
    assert report["ensemble"]["threshold"] == 0.5 and len(report["ensemble"]["members"]) == 3
    assert report["n_trees"] == 12 and (out / "probabilities.npz").exists()


def test_an_ensemble_averages_its_members(tmp_path):
    a = small_run(tmp_path / "a", scale=1.0)
    b = small_run(tmp_path / "b", scale=0.0)                   # a member that sees nothing
    out, _ = ensemble_run([a, b], tmp_path / "ens", progress=lambda m: None)
    pa = load_probabilities(a / "probabilities.npz")
    mean = load_probabilities(out / "probabilities.npz")
    masks = load_masks(out / "predicted_masks.npz")
    for t in pa:
        assert np.allclose(mean[t], pa[t] / 2, atol=1 / 255)
        assert masks[t].sum() == (pa[t] > 1.0).sum() == 0       # half of at most 1 is never above 0.5


def test_members_must_share_trees_and_folds(tmp_path):
    a, b = small_run(tmp_path / "a"), small_run(tmp_path / "b", trees=10)
    with pytest.raises(ValueError, match="other trees"):
        ensemble_run([a, b], tmp_path / "e1", progress=lambda m: None)
    first = sorted(load_masks(a / "predicted_masks.npz"))[0]
    c = small_run(tmp_path / "c", flip_fold=first)
    with pytest.raises(ValueError, match="other folds"):
        ensemble_run([a, c], tmp_path / "e2", progress=lambda m: None)
    with pytest.raises(ValueError):
        ensemble_run([a], tmp_path / "e3", progress=lambda m: None)


def test_run_table_scores_a_stub_as_the_mean_shape(tmp_path):
    run = small_run(tmp_path / "r")
    rows = list(csv.DictReader(open(run / "predictions.csv", newline="")))
    stub = rows[0]["tree"]
    for r in rows:
        if r["tree"] == stub and r["measure"] == "v_opening":
            r["segmentation"] = ""                                 # a leader too short for a line
    with open(run / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = run_table(run)
    row = next(r for r in rows if r["tree"] == stub and r["measure"] == "v_opening")
    assert len(table) == 12 and table[stub]["filled"] == 1
    assert table[stub]["error"] == pytest.approx(abs(float(row["mean shape"]) - float(row["winter"])))


# --- the rule ------------------------------------------------------------------------------------

def table(errors, side=None, incomplete=None, preds=None):
    """A run table from per-tree errors (trees t0, t1, ...)."""
    n = len(errors)
    side = np.zeros(n) if side is None else side
    incomplete = np.zeros(n) if incomplete is None else incomplete
    preds = np.asarray(errors, float) + 50 if preds is None else preds
    return {f"t{i}": {"error": float(errors[i]), "pred": float(preds[i]), "side_branches": float(side[i]),
                      "incomplete": float(incomplete[i]), "fork": 1.0, "filled": 0.0} for i in range(n)}


def test_paired_and_tree_means():
    a, b = {"x": 1.0, "y": 2.0, "z": 3.0, "w": np.nan}, {"x": 2.0, "y": 3.0, "z": 3.5, "w": 1.0}
    c = paired(a, b)
    assert c["n"] == 3 and c["mean_a"] == pytest.approx(2.0) and c["a_lower"] == 1.0
    m = tree_mean([table([1, 2, 3]), table([3, 2, 1])], "error")
    assert m == {"t0": 2.0, "t1": 2.0, "t2": 2.0}


def test_verdict_passes_a_clearly_better_change_and_stops_one_that_adds_side_branches():
    rng = np.random.default_rng(0)
    base = [table(rng.gamma(2, 2, 200)) for _ in range(3)]
    better = [table(np.array([r["error"] for r in b.values()]) * 0.7) for b in base]
    v = verdict(better, base)
    assert v["better"] and v["guards_hold"] and v["passed"]
    branchy = [table(np.array([r["error"] for r in b.values()]) * 0.7, side=(rng.random(200) < 0.3).astype(float))
               for b in base]
    v = verdict(branchy, base)
    assert v["better"] and not v["guards_hold"] and not v["passed"]
    same = verdict(base, base)
    assert not same["better"] and same["error"]["p"] == 1.0


def test_decide_needs_every_run_and_names_a_recipe():
    rng = np.random.default_rng(1)
    errs = rng.gamma(2, 2, 150)
    tables = {n: table(errs + rng.normal(0, 0.05, 150)) for n in BASELINE}
    assert not decide(tables)["complete"]
    tables.update({n: table(errs * 0.6) for n in CLDICE})
    tables["segment_lined_up_ensemble"] = table(errs + rng.normal(0, 0.05, 150))
    d = decide(tables)
    assert d["complete"] and d["cldice"]["passed"] and not d["ensemble"]["passed"]
    assert d["candidate"]["loss"] == "centre-line" and d["candidate"]["ensemble"] is False


def test_stability_null_check_and_disagreement():
    a = table([1, 2, 3, 4], preds=np.array([50.0, 50, 50, 50]), incomplete=np.array([0, 0, 1, 0]))
    b = table([1, 2, 3, 4], preds=np.array([50.0, 56, 50, 51]), incomplete=np.array([0, 1, 1, 0]))
    s = stability([a, b])
    assert s["pairs"] == 1 and s["median_difference"] == pytest.approx(0.5) and s["over_5"] == 0.25
    assert s["status_flips"] == 0.25
    checks = null_check({NAMED: a, BASELINE[0]: b})
    assert len(checks) == 1 and checks[0]["p"] == 1.0                  # identical errors
    rng = np.random.default_rng(2)
    spread = rng.gamma(1, 2, 300)
    members = [table(np.zeros(300), preds=50 + s * spread) for s in (-1.0, 0.0, 1.0)]
    ens = table(spread + rng.normal(0, 0.3, 300))                       # the error follows the spread
    d = disagreement_check(members, ens)
    assert d["spearman"] > 0.8 and d["p"] < 1e-10 and d["mae_flagged"] > d["mae_rest"] and d["worst_caught"] > 0.5


# --- the command line --------------------------------------------------------------------------------

def test_cli_builds_an_ensemble_and_compares(tmp_path, capsys):
    results = tmp_path / "results"
    members = [small_run(results / n) for n in BASELINE]
    split = str(ROOT / "splits.json")
    assert main(["ensemble", *map(str, members), "--out", str(results / "segment_lined_up_ensemble"),
                 "--figures", str(tmp_path / "fig"), "--split", split]) == 0
    assert (results / "segment_lined_up_ensemble_joined" / "predicted_masks.npz").exists()
    assert main(["ensemble", *map(str, members), "--out", str(results / "segment_lined_up_ensemble"),
                 "--split", split]) == 1                                      # never over a saved run
    from treeskel.segment import joined_run
    for m in members:
        joined_run(m, progress=lambda m: None)
    capsys.readouterr()
    assert main(["compare", "--results", str(results), "--out", str(tmp_path / "cmp"), "--split", split]) == 0
    text = capsys.readouterr().out
    assert "not decided yet: missing segment_lined_up_cldice" in text and "Null check" in text
    report = json.loads((tmp_path / "cmp" / "report.json").read_text())
    assert set(report["runs"]) == set(BASELINE) | {"segment_lined_up_ensemble"}
    assert report["stability"]["named recipe"]["median_difference"] == 0.0     # three copies of one run
    assert (tmp_path / "cmp" / "trees.csv").exists() and (tmp_path / "cmp" / "output.txt").exists()
    assert main(["compare", "--results", str(tmp_path / "empty"), "--out", str(tmp_path / "c2"),
                 "--split", split]) == 1
