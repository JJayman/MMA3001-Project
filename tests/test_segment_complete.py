"""Tests for predicting a short leader's line over the whole fitted stretch from complete leaders' shapes
(treeskel.segment.completion_samples, fit_completion, complete_measures, completed_run). No PyTorch."""

import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from treeskel.segment import (
    COMPLETE_FRACTIONS,
    complete_measures,
    completed_run,
    completion_samples,
    fit_completion,
    load_masks,
    mask_shape,
)

from test_segment_join import fake_run, v_mask


def curved_v(start=20.0, turn=-0.06, cut=None):
    """A V whose leaders leave the fork ``start`` degrees out from the vertical and turn ``turn`` degrees
    per px outwards (negative: they curve upright); ``cut`` keeps only that many px of the right leader."""
    pts = [(r, 128.0) for r in np.arange(200.0, 256.0, 0.5)]            # the trunk, 9 px wide below
    for sign in (-1, 1):
        r, c, a = 200.0, 128.0, start
        for s in np.arange(0, 190, 0.5):
            if sign > 0 and cut is not None and s > cut:
                break
            r, c, a = r - 0.5 * np.cos(np.radians(a)), c + sign * 0.5 * np.sin(np.radians(a)), a + 0.5 * turn
            pts.append((r, c))
    seeds = np.zeros((256, 256), bool)
    p = np.round(np.array(pts)).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < 256) & (p[:, 1] >= 0) & (p[:, 1] < 256)
    seeds[p[ok, 0], p[ok, 1]] = True
    m = ndi.distance_transform_edt(~seeds) < 3.5
    m[200:, 124:133] = True
    return m


def family():
    return {f"T{i}{j}": curved_v(s, k) for i, s in enumerate((15.0, 20.0, 25.0))
            for j, k in enumerate((-0.10, -0.06, -0.03, 0.0, 0.03))}


def test_the_fitted_model_moves_curving_leaders_the_right_way():
    samples = completion_samples(family())
    assert samples.shape[1] == 4 and len(samples) > 100
    assert set(np.round(1 - samples[:, 0], 2)) <= set(COMPLETE_FRACTIONS)
    a, c = fit_completion(samples)
    assert c > 0                                               # a leader keeps turning the way it turns
    uncurved = samples[np.abs(samples[:, 2]) < 0.5]
    assert np.abs(uncurved[:, 3] - uncurved[:, 1]).max() < 1.5     # straight leaders: cut and whole agree


def test_a_cut_curving_leader_is_completed_towards_the_whole():
    coef = fit_completion(completion_samples(family()))
    whole, _ = mask_shape(curved_v(18.0, -0.08))
    short = curved_v(18.0, -0.08, cut=95)                      # 60% of the 158 px stretch
    as_cut, _ = mask_shape(short)
    completed, n = complete_measures(short, coef)
    assert n == 1
    assert abs(completed["v_opening"] - whole["v_opening"]) < 0.5 * abs(as_cut["v_opening"] - whole["v_opening"])
    assert completed["left_bend"] == as_cut["left_bend"] and completed["left_tilt"] == as_cut["left_tilt"]
    assert completed["right_angle"] == pytest.approx(completed["right_tilt"] - completed["trunk_tilt"], abs=1e-9)


def test_whole_leaders_and_no_fork_are_left_alone():
    coef = (-3.0, 0.7)
    m = curved_v()
    assert complete_measures(m, coef) == (mask_shape(m)[0], 0)
    empty, n = complete_measures(np.zeros((256, 256), bool), coef)
    assert n == 0 and np.isnan(empty["v_opening"])


def test_a_saved_joined_run_is_scored_again_with_its_leader_lines_completed(tmp_path):
    trees = {**family(), "S1": curved_v(18.0, -0.08, cut=95), "S2": curved_v(22.0, -0.03, cut=120)}
    src = tmp_path / "results" / "segment_lined_up_joined"
    fake_run(src, trees)
    rep0 = json.loads((src / "report.json").read_text())
    (src / "report.json").write_text(json.dumps({**rep0, "join": {"gap": 19.3}}))
    out, failed = completed_run(src, figures=tmp_path / "figures", progress=lambda m: None)
    assert failed == [] and out.name == "segment_lined_up_completed"
    report = json.loads((out / "report.json").read_text())
    comp = report["complete"]
    assert comp["trees_completed"] == 2 and comp["leaders_completed"] == 2
    assert len(comp["coef_all"]) == 2 and set(comp["coef_by_fold"]) == {"0", "1", "2"}
    assert report["join"] == {"gap": 19.3} and "segment_lined_up_joined" in report["against_runs"]
    saved = load_masks(out / "predicted_masks.npz")
    assert all(np.array_equal(saved[t], trees[t]) for t in trees)     # the masks are not changed
    assert (tmp_path / "figures" / "segment_lined_up_completed_skill.png").exists()
