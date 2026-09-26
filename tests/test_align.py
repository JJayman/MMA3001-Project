"""Tests for treeskel.align (winter-to-summer alignment)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from treeskel import align
from treeskel.align import (
    MODELS,
    FitLimits,
    Matches,
    apply_transform,
    check_fit,
    describe,
    fit_quality,
    fit_transform,
    hull_fraction,
    inlier_places,
    keeps_orientation,
    linear_part,
    load_matches,
    make_matcher,
    ransac_fit,
    residuals,
    save_matches,
    sift_match,
    warp_mask,
)


def similarity(scale=1.0, deg=0.0, tx=0.0, ty=0.0):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1.0]])


TRUE = {
    "translation": similarity(tx=12, ty=-7),
    "similarity": similarity(1.08, 4.0, 25, -18),
    "affine": np.array([[1.05, 0.04, 20], [-0.03, 0.97, -11], [0, 0, 1.0]]),
    "homography": np.array([[1.02, 0.03, 15], [-0.02, 0.99, -9], [2e-5, -3e-5, 1.0]]),
}


def random_points(n, seed=0):
    return np.random.default_rng(seed).uniform([0, 0], [640, 480], size=(n, 2))


@pytest.mark.parametrize("model", MODELS)
def test_fit_transform_recovers_each_model_exactly(model):
    src = random_points(30)
    dst = apply_transform(TRUE[model], src)
    m = fit_transform(src, dst, model)
    assert m is not None
    grid = random_points(50, seed=1)
    assert np.abs(apply_transform(m, grid) - apply_transform(TRUE[model], grid)).max() < 1e-6


@pytest.mark.parametrize("model", MODELS)
def test_minimal_sample_is_fitted_exactly(model):
    src = random_points(align.MIN_SAMPLES[model], seed=2)
    dst = apply_transform(TRUE[model], src)
    m = fit_transform(src, dst, model)
    assert residuals(m, src, dst).max() < 1e-6


def test_similarity_closed_form_matches_numerical_least_squares():
    """Umeyama's closed form must give the same answer as a general optimiser."""
    rng = np.random.default_rng(3)
    src = random_points(40, seed=3)
    dst = apply_transform(TRUE["similarity"], src) + rng.normal(0, 3, (40, 2))
    m = fit_transform(src, dst, "similarity")

    def res(p):
        return (apply_transform(similarity(*p), src) - dst).ravel()

    opt = least_squares(res, x0=[1.0, 0.0, 0.0, 0.0], xtol=1e-12, ftol=1e-12).x
    grid = random_points(20, seed=4)
    assert np.abs(apply_transform(m, grid) - apply_transform(similarity(*opt), grid)).max() < 1e-4
    # and it beats the true transform on the noisy data it was fitted to
    assert (residuals(m, src, dst) ** 2).sum() <= (residuals(TRUE["similarity"], src, dst) ** 2).sum()


def test_fit_transform_degenerate_inputs():
    one = np.array([[5.0, 5.0]])
    assert fit_transform(one, one, "similarity") is None             # too few points
    same = np.array([[5.0, 5.0], [5.0, 5.0]])
    assert fit_transform(same, same + 1, "similarity") is None       # no spread
    line = np.array([[0.0, 0], [10, 10], [20, 20], [30, 30]])
    assert fit_transform(line, line + 2, "affine") is None           # collinear
    assert fit_transform(line, line + 2, "homography") is None
    with pytest.raises(ValueError):
        fit_transform(line, line, "rigid")


@pytest.mark.parametrize("model", MODELS)
def test_ransac_finds_the_transform_among_outliers(model):
    rng = np.random.default_rng(5)
    src_in = random_points(200, seed=6)
    dst_in = apply_transform(TRUE[model], src_in) + rng.normal(0, 0.5, (200, 2))
    src_out, dst_out = random_points(300, seed=7), random_points(300, seed=8)
    src, dst = np.r_[src_in, src_out], np.r_[dst_in, dst_out]
    fit = ransac_fit(src, dst, model, threshold=3.0)
    assert fit.inliers[:200].mean() > 0.97
    assert fit.inliers[200:].mean() < 0.02
    grid = random_points(50, seed=9)
    assert np.abs(apply_transform(fit.matrix, grid) - apply_transform(TRUE[model], grid)).max() < 1.0
    assert fit.rms < 1.5


def test_ransac_is_repeatable_and_handles_too_few_matches():
    src = random_points(60, seed=10)
    dst = np.r_[apply_transform(TRUE["similarity"], src[:30]), random_points(30, seed=11)]
    a, b = ransac_fit(src, dst, seed=0), ransac_fit(src, dst, seed=0)
    assert np.array_equal(a.inliers, b.inliers) and np.allclose(a.matrix, b.matrix)
    few = ransac_fit(src[:3], dst[:3], "homography")
    assert few.matrix is None and not few.inliers.any()


def test_keeps_orientation_rejects_homographies_that_flip_the_image():
    pts = random_points(20, seed=20)
    assert keeps_orientation(TRUE["homography"], pts) and keeps_orientation(TRUE["affine"], pts)
    flip = np.array([[1.0, 0, 0], [0, 1, 0], [-1 / 300, 0, 1]])   # w = 1 - x/300 < 0 right of x = 300
    assert not keeps_orientation(flip, pts)


def test_describe_reads_back_scale_rotation_and_shift():
    m = similarity(1.1, 3.0, 20, -10)
    d = describe(m, (320, 240))
    assert d["scale"] == pytest.approx(1.1)
    assert d["rotation"] == pytest.approx(3.0)
    assert d["anisotropy"] == pytest.approx(1.0)
    moved = apply_transform(m, [(320, 240)])[0]
    assert (d["dx"], d["dy"]) == pytest.approx((moved[0] - 320, moved[1] - 240))


def test_linear_part_of_a_homography_matches_finite_differences():
    h, at, e = TRUE["homography"], np.array([300.0, 200.0]), 1e-4
    num = np.stack([(apply_transform(h, [at + e * np.eye(2)[i]]) - apply_transform(h, [at - e * np.eye(2)[i]]))[0]
                    / (2 * e) for i in range(2)], 1)
    assert np.allclose(linear_part(h, at), num, atol=1e-6)


def test_warp_mask_shifts_and_reports_coverage():
    mask = np.zeros((480, 640), bool)
    mask[100:150, 200:220] = True
    warped, covered = warp_mask(mask, similarity(tx=10, ty=-5))
    ys, xs = np.nonzero(warped)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (95, 144, 210, 229)
    assert covered[:475, 10:].all() and not covered[:, :10].any()    # left strip maps outside winter
    assert not covered[-5:].any()                                    # so does the bottom strip
    same, all_in = warp_mask(mask, np.eye(3))
    assert np.array_equal(same, mask) and all_in.all()


def test_matches_round_trip(tmp_path):
    rng = np.random.default_rng(12)
    ms = {"R01N01": Matches(rng.uniform(0, 600, (7, 2)), rng.uniform(0, 600, (7, 2)), rng.uniform(0, 1, 7)),
          "R02N03": Matches(np.zeros((0, 2)), np.zeros((0, 2)))}
    path = tmp_path / "m" / "lightglue.npz"
    save_matches(path, ms, {"matcher": "lightglue"})
    back, meta = load_matches(path)
    assert meta == {"matcher": "lightglue"} and sorted(back) == ["R01N01", "R02N03"]
    assert np.allclose(back["R01N01"].winter, ms["R01N01"].winter)
    assert np.allclose(back["R01N01"].score, ms["R01N01"].score)
    assert len(back["R02N03"]) == 0
    assert load_matches(tmp_path / "missing.npz") == ({}, {})
    with pytest.raises(ValueError):
        Matches(np.zeros((3, 2)), np.zeros((2, 2)))


def test_hull_fraction():
    square = np.array([[0, 0], [64, 0], [64, 48], [0, 48], [30, 20]])
    assert hull_fraction(square, (480, 640)) == pytest.approx(0.01)
    assert hull_fraction(np.array([[0, 0], [1, 1], [2, 2]]), (480, 640)) == 0.0


def test_check_fit_flags_each_problem():
    good = {"inliers": 80, "hull": 0.2, "scale": 1.05, "rotation": 2.0}
    assert check_fit(good) == []
    assert check_fit({**good, "inliers": 5}) == ["only 5 inliers"]
    assert len(check_fit({**good, "hull": 0.001, "scale": 1.6, "rotation": -15})) == 3
    assert check_fit({"inliers": 0, "hull": 0.0}) == ["no transform could be fitted"]
    assert check_fit({**good, "scale": 1.6}, FitLimits(scale_range=(0.5, 2.0))) == []


def test_inlier_places():
    valid = np.ones((480, 640), bool)
    valid[:100] = False                                    # "background" rows without depth
    near = np.zeros((480, 640), bool)
    near[:, 300:340] = True                                # a vertical "trunk"
    pts = np.array([[310.0, 50.0], [320.2, 300.4], [100.0, 300.0], [639.6, 479.7]])
    assert inlier_places(pts, valid, near) == {"no_depth": 0.25, "on_tree": 0.5}
    assert np.isnan(inlier_places(pts, valid, None)["on_tree"])
    assert np.isnan(inlier_places(np.zeros((0, 2)), valid, near)["no_depth"])


def test_fit_quality_describes_the_fit_at_the_anchor():
    src = random_points(40, seed=13)
    fit = ransac_fit(src, apply_transform(TRUE["similarity"], src))
    q = fit_quality(fit, src, (480, 640), anchor=(100, 400))
    assert q["inliers"] == 40 and q["inlier_ratio"] == 1.0 and q["hull"] > 0.5
    moved = apply_transform(TRUE["similarity"], [(100, 400)])[0]
    assert (q["dx"], q["dy"]) == pytest.approx((moved[0] - 100, moved[1] - 400), abs=1e-6)


def textured_image(h=240, w=320, seed=14):
    """Smooth random texture with distinct blobs, so SIFT finds keypoints."""
    rng = np.random.default_rng(seed)
    img = ndi.gaussian_filter(rng.uniform(0, 1, (h, w)), 3)
    img = (img - img.min()) / (img.max() - img.min())
    return (np.stack([img] * 3, -1) * 255).astype(np.uint8)


def warp_image(rgb, matrix):
    """Image as seen after the transform: out(x, y) = in(M⁻¹(x, y))."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    src = apply_transform(np.linalg.inv(matrix), np.c_[xx.ravel(), yy.ravel()])
    return np.stack([ndi.map_coordinates(rgb[..., c].astype(float), [src[:, 1], src[:, 0]], order=1,
                                         mode="reflect").reshape(h, w) for c in range(3)], -1).astype(np.uint8)


def test_sift_pipeline_recovers_a_known_similarity():
    """End to end without learned weights: matcher coordinates must be (x, y)."""
    winter = textured_image()
    truth = similarity(1.05, 3.0, 8, -6)
    summer = warp_image(winter, truth)
    m = sift_match(winter, summer)
    assert len(m) > 30
    fit = ransac_fit(m.winter, m.summer, "similarity", threshold=2.0)
    grid = np.random.default_rng(15).uniform([40, 40], [280, 200], (30, 2))
    assert np.abs(apply_transform(fit.matrix, grid) - apply_transform(truth, grid)).max() < 1.5


def test_make_matcher_names():
    assert make_matcher("sift") is sift_match
    with pytest.raises(ValueError):
        make_matcher("orb")


def test_learned_matchers_run_with_untrained_networks():
    """Plumbing only: random weights, no download. Checks shapes and coordinate ranges."""
    torch = pytest.importorskip("torch")
    KF = pytest.importorskip("kornia.feature")
    torch.manual_seed(0)
    w, s = textured_image(64, 96, seed=16), textured_image(64, 96, seed=17)
    lg = align.disk_lightglue_match(KF.DISK().eval(), KF.LightGlue(None, input_dim=128).eval(), w, s,
                                    max_keypoints=64)
    lo = align.loftr_match(KF.LoFTR(pretrained=None).eval(), w, s, max_matches=50)
    for m in (lg, lo):
        assert m.winter.shape == m.summer.shape == (len(m), 2) and m.score.shape == (len(m),)
        if len(m):
            assert m.winter[:, 0].max() < 96 and m.winter[:, 1].max() < 64
    assert len(lo) <= 50


CLICKS = Path(__file__).resolve().parents[1] / "annotations" / "winter_clicks.json"


def fake_matcher(shift=(12.0, -8.0)):
    """Matcher that returns exact matches for a pure shift, plus some wrong ones."""
    def match(w, s):
        src = random_points(120, seed=18)
        dst = src + shift
        dst[100:] = random_points(20, seed=19)
        return Matches(src, dst, np.ones(120))
    return match


def test_cli_matches_fits_and_resumes(sample_dir, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(align, "make_matcher", lambda *a, **k: (calls.append(a), fake_matcher())[1])
    mpath, out = tmp_path / "matches.npz", tmp_path / "out"
    args = [str(sample_dir), "--trees", "R01N01", "--matches", str(mpath), "--out", str(out),
            "--winter-gt", "", "--overlays", "--clicks", str(CLICKS)]
    assert align.main(args) == 0
    assert mpath.exists() and len(calls) == 1
    with open(out / "lightglue" / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == list(MODELS)
    sim = next(r for r in rows if r["model"] == "similarity")
    assert int(sim["inliers"]) == 100 and float(sim["scale"]) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= float(sim["on_tree"]) <= 1 and 0 <= float(sim["no_depth"]) <= 1
    assert float(sim["dx"]) == pytest.approx(12.0, abs=1e-6)
    m = np.array(json.loads(sim["matrix"]))
    assert m.shape == (3, 3)
    assert (out / "lightglue" / "sheet_similarity_01.jpg").exists()
    assert (out / "lightglue" / "R01N01_similarity.jpg").exists()
    # A second run reuses the saved matches: the matcher is not loaded again.
    assert align.main(args + ["--no-match"]) == 0
    assert align.main(args) == 0 and len(calls) == 1
    assert "similarity" in capsys.readouterr().out


def test_cli_reports_a_tree_that_cannot_be_matched_and_carries_on(sample_dir, tmp_path, monkeypatch, capsys):
    good = fake_matcher()

    def flaky(w, s):
        if flaky.calls == 0:
            flaky.calls += 1
            raise RuntimeError("CUDA out of memory")
        return good(w, s)

    flaky.calls = 0
    monkeypatch.setattr(align, "make_matcher", lambda *a, **k: flaky)
    mpath = tmp_path / "matches.npz"
    args = [str(sample_dir), "--trees", "R01N01", "R01N02", "--matches", str(mpath),
            "--out", str(tmp_path / "out"), "--winter-gt", str(tmp_path), "--clicks", str(CLICKS)]
    assert align.main(args) == 0
    text = capsys.readouterr().out
    assert "could not match R01N01: RuntimeError: CUDA out of memory" in text
    saved, _ = load_matches(mpath)
    assert sorted(saved) == ["R01N02"]


def test_cli_explains_a_matcher_that_cannot_load(sample_dir, tmp_path, monkeypatch, capsys):
    """E.g. a damaged PyTorch install on Windows: a clear message instead of a traceback."""
    def broken(*a, **k):
        raise OSError('[WinError 193] %1 is not a valid Win32 application. Error loading "torch_cuda.dll"')

    monkeypatch.setattr(align, "make_matcher", broken)
    mpath = tmp_path / "matches.npz"
    args = [str(sample_dir), "--trees", "R01N01", "--matches", str(mpath), "--out", str(tmp_path / "out")]
    assert align.main(args) == 1
    text = capsys.readouterr().out
    assert "WinError 193" in text and "If PyTorch does not load" in text
    assert not mpath.exists()


def test_image_tensor_accepts_read_only_images_without_warning():
    torch = pytest.importorskip("torch")
    import warnings

    rgb = textured_image(32, 48)
    rgb.setflags(write=False)                              # as loaded by PIL
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = align._image_tensor(rgb, "cpu")
    assert t.shape == (1, 3, 32, 48) and float(t.max()) <= 1.0
    assert torch.equal(t[0, 0], torch.from_numpy(rgb[..., 0].astype(np.float32) / 255))


def test_cli_prints_progress_while_fitting(sample_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(align, "make_matcher", lambda *a, **k: fake_matcher())
    args = [str(sample_dir), "--trees", "R01N01", "R01N02", "--matches", str(tmp_path / "m.npz"),
            "--out", str(tmp_path / "out"), "--winter-gt", str(tmp_path), "--clicks", str(CLICKS)]
    assert align.main(args) == 0
    text = capsys.readouterr().out
    assert "Fitting 4 models to 2 trees" in text
    assert "fitted 1/2 trees" in text and "fitted 2/2 trees" in text


def test_cli_stops_when_the_matcher_keeps_failing(tmp_path, monkeypatch, capsys, make_tree):
    """A broken setup (e.g. out of GPU memory) fails every tree: stop early, keep what was matched."""
    for i in range(1, 9):
        make_tree(f"R05N{i:02d}")
    calls = []

    def failing(w, s):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("CUDA out of memory")
        return fake_matcher()(w, s)

    monkeypatch.setattr(align, "make_matcher", lambda *a, **k: failing)
    mpath = tmp_path / "m.npz"
    args = [str(tmp_path), "--matches", str(mpath), "--out", str(tmp_path / "out"), "--winter-gt", ""]
    assert align.main(args) == 1
    text = capsys.readouterr().out
    assert len(calls) == 1 + align.MAX_FAILURES_IN_A_ROW          # stopped, not all 8 trees
    assert text.count("could not match") == align.MAX_FAILURES_IN_A_ROW and "Stopped" in text
    saved, _ = load_matches(mpath)
    assert sorted(saved) == ["R05N01"]                             # the tree matched before is kept
