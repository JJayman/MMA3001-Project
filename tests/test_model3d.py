"""Tests for treeskel.model3d: the tube geometry, the files, and the pipeline on a sample tree."""

import base64
import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from treeskel.model3d import (
    FINAL_TEST_MASKS,
    SAVED_MASKS,
    SLAB_M,
    ball,
    frames,
    main,
    merge,
    read_ply,
    saved_source,
    smooth_line,
    to_view,
    trained_trees,
    tree_model,
    tube,
    viewer_data,
    write_html,
    write_ply_mesh,
    write_ply_points,
)

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "docs/results/segment_lined_up_joined/predicted_masks.npz"


def test_smooth_line_keeps_ends_and_spacing():
    t = np.linspace(0, np.pi / 2, 50)
    arc = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1) * 0.5      # a quarter circle, 0.79 m
    noisy = arc + np.random.default_rng(0).normal(0, 0.003, arc.shape)
    p, s = smooth_line(noisy, step=0.01, smooth=0.03)
    assert np.allclose(p[0], noisy[0]) and np.allclose(p[-1], noisy[-1])
    assert np.allclose(np.diff(s), np.diff(s)[0])
    assert abs(np.linalg.norm(p[1:-1, :2], axis=1) - 0.5).max() < 0.006    # back on the circle, within noise
    rough = lambda q: np.abs(np.diff(q, 2, axis=0)).mean()                 # noqa: E731
    assert rough(p) < rough(smooth_line(noisy, step=0.01, smooth=0.0)[0])


def test_frames_are_orthonormal_and_do_not_twist():
    t = np.linspace(0, 1, 100)
    line = np.stack([0.2 * np.sin(3 * t), t, 0.1 * t ** 2], 1)
    tan, n, b = frames(line)
    for a, c in ((tan, n), (tan, b), (n, b)):
        assert np.allclose((a * c).sum(1), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(n, axis=1), 1) and np.allclose(np.linalg.norm(b, axis=1), 1)
    assert np.min((n[1:] * n[:-1]).sum(1)) > 0.99                          # no sudden turn of the frame


def _edges(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


def test_tube_is_a_closed_surface_at_the_radius():
    z = np.linspace(0, 1, 30)
    line = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    m = tube(line, np.full(30, 0.05), sides=12)
    ring = m["vertices"][:-2]
    assert np.allclose(np.hypot(ring[:, 0], ring[:, 1]), 0.05)
    assert m["faces"].min() == 0 and m["faces"].max() == len(m["vertices"]) - 1
    assert np.all(_edges(m["faces"]) == 2)                                 # every edge shared by two faces
    # Normals point outwards on the side.
    assert np.all((ring[:, :2] * m["normals"][:-2, :2]).sum(1) > 0)


def test_ball_and_merge():
    b = ball(np.array([1.0, 2.0, 3.0]), 0.1)
    assert np.allclose(np.linalg.norm(b["vertices"] - [1, 2, 3], axis=1), 0.1)
    t = tube(np.array([[0.0, 0, 0], [0, 1, 0]]), np.array([0.1, 0.1]), sides=6)
    m = merge([t, b])
    assert len(m["vertices"]) == len(t["vertices"]) + len(b["vertices"])
    assert m["faces"][len(t["faces"]):].min() == len(t["vertices"])


def test_to_view_turns_the_camera_frame():
    p = to_view(np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 1.0, 1.0]))
    assert p.tolist() == [[0.0, -1.0, -2.0]]                               # y up, z towards the camera


def test_ply_files_round_trip(tmp_path):
    m = tube(np.array([[0.0, 0, 0], [0, 0.5, 0], [0.1, 1, 0]]), np.array([0.05, 0.04, 0.03]), sides=8)
    write_ply_mesh(tmp_path / "b.ply", m)
    back = read_ply(tmp_path / "b.ply")
    assert np.allclose(np.stack([back["vertex"][k] for k in "xyz"], 1), m["vertices"], atol=1e-6)
    assert np.array_equal(back["face"], m["faces"])
    pts = np.random.default_rng(1).normal(size=(50, 3))
    cols = np.random.default_rng(2).integers(0, 256, (50, 3)).astype(np.uint8)
    write_ply_points(tmp_path / "p.ply", pts, cols)
    back = read_ply(tmp_path / "p.ply")
    assert back["face"] is None and np.array_equal(back["vertex"]["green"], cols[:, 1])
    assert (tmp_path / "p.ply").read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")


@pytest.fixture(scope="module")
def sample_model():
    from treeskel.io import index_dataset, load_tree
    from treeskel.segment import load_masks

    rec = {t.tree_id: t for t in index_dataset(ROOT / "data/sample").trees}["R15N05"]
    s = load_tree(rec, "summer")
    return s, tree_model(s["rgb"], s["depth"], s["valid"], load_masks(SAVED)["R15N05"])


def test_the_model_matches_the_3d_run(sample_model):
    """The same mask, depth and camera give the numbers the 449-tree run saved for this tree."""
    _, model = sample_model
    with open(ROOT / "docs/results/tree3d/trees.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["tree"] == "R15N05" and r["source"] == "U-Net"
                   and r["setting"] == "main")
    for k in ("v_opening_3d", "v_opening_2d", "lean_camera", "turn"):
        assert model["measures"][k] == pytest.approx(float(row[k]), abs=1e-3)


def test_the_model_is_in_metres_around_the_fork(sample_model):
    s, model = sample_model
    v = model["branches"]["vertices"]
    assert np.abs(v).max() < 3.0                                           # a tree, in metres
    assert np.linalg.norm(v, axis=1).min() < 0.05                          # the fork's ball is at the origin
    assert v[:, 1].max() > 0.5 and v[:, 1].min() < -0.1                    # leaders up, trunk down
    leaves = model["leaves"]
    assert len(leaves["points"]) == len(leaves["colours"]) > 10000
    n = model["plane"]["normal"] * np.array([1, -1, -1])                   # the plane's normal, as viewed
    assert np.abs(leaves["points"] @ n).max() < SLAB_M + 0.01              # leaves within the slab of the plane
    assert 0.08 < model["measures"]["trunk_diameter"] < 0.2


def test_no_fork_is_an_error(sample_model):
    s, _ = sample_model
    with pytest.raises(ValueError, match="no fork"):
        tree_model(s["rgb"], s["depth"], s["valid"], np.zeros((256, 256), bool))


def test_the_viewer_page_holds_the_model(sample_model, tmp_path):
    _, model = sample_model
    write_html(tmp_path / "t.html", model, "Tree <R15N05>", "test")
    page = (tmp_path / "t.html").read_text(encoding="utf-8")
    assert "<title>Tree &lt;R15N05&gt;</title>" in page and "__DATA__" not in page
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", page, re.S).group(1).replace("<\\/", "</"))
    pos = np.frombuffer(base64.b64decode(data["mesh"]["positions"]), "<f4").reshape(-1, 3)
    idx = np.frombuffer(base64.b64decode(data["mesh"]["indices"]), "<u4")
    assert np.allclose(pos, model["branches"]["vertices"], atol=1e-6) and idx.max() < len(pos)
    assert data["points"]["count"] == len(model["leaves"]["points"])
    assert dict((m[0], m[1]) for m in viewer_data(model, "t", "s")["measures"])["V opening in the tree's plane"] \
        == f"{model['measures']['v_opening_3d']:.1f}"
    assert "http" not in page.split("<script>")[1]                         # nothing loaded from the internet


def test_build_from_saved_masks(sample_dir, tmp_path, capsys):
    assert main(["build", "--tree", "R15N05", "R01N02", "--data", str(sample_dir), "--mask", "saved",
                 "--saved", str(SAVED), "--out", str(tmp_path)]) == 0
    for t in ("R15N05", "R01N02"):
        for suffix in (".html", "_branches.ply", "_leaves.ply"):
            assert (tmp_path / f"{t}{suffix}").exists()
    assert "V opening" in capsys.readouterr().out
    # A test-row tree has no saved development mask; the message points to the final test's masks.
    assert main(["build", "--tree", "R12N02", "--data", str(sample_dir), "--mask", "saved", "--saved", str(SAVED),
                 "--out", str(tmp_path)]) == 1
    assert "final_test/predicted_masks.npz" in capsys.readouterr().out


def test_build_from_the_final_tests_masks(sample_dir, tmp_path, monkeypatch):
    """A test-row tree from the final test's own masks gives the final test's V opening, captioned as such."""
    monkeypatch.chdir(ROOT)
    assert main(["build", "--tree", "R12N02", "--data", str(sample_dir), "--mask", "saved",
                 "--saved", str(FINAL_TEST_MASKS), "--out", str(tmp_path)]) == 0
    page = (tmp_path / "R12N02.html").read_text(encoding="utf-8")
    assert "final test's network" in page and "5-fold" not in page
    with open(ROOT / "docs/results/final_test/predictions.csv", newline="") as f:
        final = next(float(r["model"]) for r in csv.DictReader(f) if r["tree"] == "R12N02" and r["measure"] == "v_opening")
    shown = re.search(r'V opening in the photo", "([0-9.]+)"', page)
    assert shown is not None and float(shown.group(1)) == pytest.approx(final, abs=0.05)


def test_saved_source_names_the_masks(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert "5-fold" in saved_source(SAVED_MASKS) and "5-fold" in saved_source(ROOT / SAVED_MASKS)
    assert "final test" in saved_source(FINAL_TEST_MASKS)
    assert saved_source("elsewhere/masks.npz") == "Branches from the saved masks in elsewhere/masks.npz"


def test_trained_trees_recorded_or_found_as_train_chose_them(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert trained_trees({"trees": ["R01N01", "R15N05"]}) == {"R01N01", "R15N05"}
    # A network saved before train recorded its trees: every lined-up development tree.
    found = trained_trees({})
    assert len(found) == 449 and "R15N05" in found and "R01N04" in found
    assert not {t for t in found if int(t[1:3]) in (3, 12, 16)}          # never a test row
    import treeskel.summer_gt as sg
    from types import SimpleNamespace
    monkeypatch.setattr(sg, "load_overlays", lambda path: {"R05N01": SimpleNamespace(skip=False),
                                                           "R05N02": SimpleNamespace(skip=True),
                                                           "R12N01": SimpleNamespace(skip=False)})
    assert trained_trees({}) == {"R05N01"}                               # skipped and test-row trees left out


def test_build_needs_the_weights(sample_dir, tmp_path, capsys):
    assert main(["build", "--tree", "R15N05", "--data", str(sample_dir), "--weights", str(tmp_path / "none.pt"),
                 "--out", str(tmp_path)]) == 1
    assert "Train it once" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["build", "--rgb", "x.png", "--out", str(tmp_path)])            # no depth


def test_train_then_build_from_a_photo(sample_dir, tmp_path, monkeypatch):
    """The whole path with a network trained for one epoch from random weights (the recipe's plumbing)."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    monkeypatch.chdir(ROOT)
    weights = tmp_path / "net.pt"
    assert main(["train", str(sample_dir), "--weights", str(weights), "--random-weights", "--epochs", "1",
                 "--device", "cpu"]) == 0
    import torch
    saved = torch.load(weights, map_location="cpu", weights_only=False)
    assert saved["epochs"] == 1 and saved["n_trees"] == 4 and saved["join_gap"] == pytest.approx(19.3)
    assert saved["trees"] == ["R01N01", "R01N02", "R15N05", "R20N06"]   # the sample's lined-up development trees
    rgb = next(sample_dir.glob("R15N05_Summer_RGB-*.png"))
    depth = next(sample_dir.glob("R15N05_Summer_D-*.png"))
    from treeskel.model3d import load_network, predict_mask
    from treeskel.io import load_rgb
    model, device, meta = load_network(weights, "cpu")
    mask = predict_mask(model, device, load_rgb(rgb), meta["params"], meta["join_gap"])
    assert mask.shape == (256, 256) and mask.dtype == bool
    # An untrained network rarely draws a tree with a fork, so the build may stop there; it must say why.
    code = main(["build", "--rgb", str(rgb), "--depth", str(depth), "--name", "photo", "--weights", str(weights),
                 "--device", "cpu", "--out", str(tmp_path)])
    assert code in (0, 1)
    assert code == 1 or (tmp_path / "photo.html").exists()


@pytest.fixture
def capture(capsys):
    """What a call printed (the build may stop at a mask without a fork; the note comes first)."""
    def run(args):
        main(args)
        return capsys.readouterr().out
    return run


def test_build_says_when_the_tree_was_in_the_training_set(sample_dir, tmp_path, monkeypatch, capture):
    """The note appears for a tree the network was trained on, not for a test-row tree; also for a network
    saved before train recorded its trees."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    monkeypatch.chdir(ROOT)
    weights = tmp_path / "net.pt"
    assert main(["train", str(sample_dir), "--weights", str(weights), "--random-weights", "--epochs", "1",
                 "--device", "cpu"]) == 0
    note = "was in the network's training set"
    common = ["--data", str(sample_dir), "--device", "cpu", "--out", str(tmp_path)]
    assert note in capture(["build", "--tree", "R15N05", "--weights", str(weights), *common])
    assert note not in capture(["build", "--tree", "R12N02", "--weights", str(weights), *common])
    import torch
    old = torch.load(weights, map_location="cpu", weights_only=False)
    del old["trees"]
    torch.save(old, tmp_path / "old.pt")
    assert note in capture(["build", "--tree", "R15N05", "--weights", str(tmp_path / "old.pt"), *common])
    assert note not in capture(["build", "--tree", "R12N02", "--weights", str(tmp_path / "old.pt"), *common])


@pytest.fixture(scope="module")
def bent_model(sample_model):
    from treeskel.bend3d import BendParams
    from treeskel.segment import load_masks

    s, _ = sample_model
    params = BendParams(spread=0.02, length=0.2)
    return params, tree_model(s["rgb"], s["depth"], s["valid"], load_masks(SAVED)["R15N05"], bends=params)


def test_bends_move_points_along_their_rays_and_mark_the_unseen(sample_model, bent_model):
    from treeskel.model3d import BARK, UNSURE

    _, flat = sample_model
    params, bent = bent_model
    b = bent["measures"]["bends"]
    assert np.isfinite(b["trunk_out"]) and b["trunk_anchors"] > b["left_anchors"]   # summer sees the trunk best
    # The plane's own measures are unchanged; only the geometry and the extra measures differ.
    for k in ("v_opening_3d", "v_opening_2d", "lean_camera"):
        assert bent["measures"][k] == flat["measures"][k]
    v = bent["branches"]["vertices"]
    n0 = len(flat["branches"]["vertices"])                                 # bent lines are a little longer: more rings
    assert n0 <= len(v) < 1.05 * n0
    assert np.linalg.norm(v, axis=1).min() < 0.05                          # the fork's ball is still at the origin
    colours = np.round(bent["branches"]["colours"], 3)
    pale = np.all(colours == np.round(UNSURE, 3), axis=1)
    assert pale.mean() > 0.5 and np.all(pale | np.all(colours == np.round(BARK, 3), axis=1))   # leaders unseen
    assert not np.any(np.all(np.round(flat["branches"]["colours"], 3) == np.round(UNSURE, 3), axis=1))


def test_bent_viewer_and_ply_carry_the_colours(bent_model, tmp_path):
    from treeskel.model3d import UNSURE

    _, bent = bent_model
    write_html(tmp_path / "t.html", bent, "t", "s")
    page = (tmp_path / "t.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", page, re.S).group(1).replace("<\\/", "</"))
    cols = np.frombuffer(base64.b64decode(data["mesh"]["colours"]), "u1").reshape(-1, 3)
    assert len(cols) == len(bent["branches"]["vertices"]) and data["bends"] is True
    names = [m[0] for m in data["measures"]]
    assert "Trunk out of the V's plane (+: root behind)" in names
    write_ply_mesh(tmp_path / "b.ply", bent["branches"])
    back = read_ply(tmp_path / "b.ply")
    assert np.any(back["vertex"]["red"] == round(UNSURE[0] * 255))


def test_build_bends_by_default_and_flat_on_request(sample_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    base = ["build", "--tree", "R15N05", "--data", str(sample_dir), "--mask", "saved", "--saved", str(SAVED)]
    assert main(base + ["--out", str(tmp_path / "default")]) == 0
    out = capsys.readouterr().out
    assert "Bends out of the plane: prior tau 2 L 20 (chosen by the 449-tree run)" in out and "drawn pale" in out
    assert main(base + ["--out", str(tmp_path / "bends"), "--bends"]) == 0
    assert "Bends out of the plane" in capsys.readouterr().out
    assert main(base + ["--out", str(tmp_path / "flat"), "--flat"]) == 0
    out = capsys.readouterr().out
    assert "Bends out of the plane" not in out and "bent:" not in out
    with pytest.raises(SystemExit):
        main(base + ["--bends", "--flat"])
