"""Tests for treeskel.label_audit: the representative sample, the verdicts and their summary, the images."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from treeskel.label_audit import (
    SAMPLE_CSV,
    SAMPLE_SIZE,
    allocate,
    development_trees,
    draw_panel,
    draw_sample,
    main,
    read_verdicts,
    summarise,
    wilson,
    write_template,
)

ROOT = Path(__file__).resolve().parents[1]


def test_allocation_is_proportional_with_one_per_row_and_sums_to_n():
    counts = {1: 45, 2: 40, 5: 30, 19: 11, 20: 6}
    take = allocate(counts, 20)
    assert sum(take.values()) == 20 and all(v >= 1 for v in take.values())
    assert take[1] >= take[2] >= take[5] >= take[19] >= take[20]
    assert allocate({1: 3, 2: 3, 3: 300}, 3) == {1: 1, 2: 1, 3: 1}       # a row raised to 1 does not overshoot
    with pytest.raises(ValueError):
        allocate({1: 5, 2: 5}, 1)                                          # fewer trees than rows


def test_the_sample_is_fixed_by_its_seed_and_stratified():
    rows = {r: [f"R{r:02d}N{i:02d}" for i in range(1, 31)] for r in (1, 2, 5)}
    a, b = draw_sample(rows, 9, seed=3), draw_sample(rows, 9, seed=3)
    assert a == b and len(set(a)) == 9
    assert {t[:3] for t in a} == {"R01", "R02", "R05"} and all(sum(t.startswith(p) for t in a) == 3
                                                                for p in ("R01", "R02", "R05"))
    assert draw_sample(rows, 9, seed=4) != a


def test_the_committed_sample_is_the_one_the_seed_draws():
    by_row = development_trees(ROOT / "data/shape/shape_measures.csv", ROOT / "splits.json",
                               ROOT / "annotations/mask_overlay.json")
    assert sum(map(len, by_row.values())) == 449 and len(by_row) == 16
    test_rows = set(json.loads((ROOT / "splits.json").read_text())["test_rows"])
    assert not test_rows & set(by_row)
    with open(ROOT / SAMPLE_CSV, newline="") as f:
        committed = [r["tree"] for r in csv.DictReader(f)]
    assert committed == draw_sample(by_row) and len(committed) == SAMPLE_SIZE


def test_test_row_trees_stop_the_sample(tmp_path):
    bad = tmp_path / "table.csv"
    bad.write_text("tree,row\nR03N01,3\n")
    with pytest.raises(ValueError, match="test-row"):
        development_trees(bad, ROOT / "splits.json", ROOT / "annotations/mask_overlay.json")


def test_verdicts_are_checked_and_summarised(tmp_path):
    path = tmp_path / "s.csv"
    write_template(path, ["R01N01", "R02N02", "R05N03", "R06N04"])
    rows = list(csv.DictReader(open(path, newline="")))
    assert [r["row"] for r in rows] == ["1", "2", "5", "6"]
    fill = {"R01N01": ("on", "on", "off"), "R02N02": ("on", "hidden", "on"), "R05N03": ("off", "on", "On ")}
    for r in rows:
        r["trunk"], r["left"], r["right"] = fill.get(r["tree"], ("", "", ""))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarise(read_verdicts(path))
    assert s["trees"] == 4 and s["judged_trees"] == 3                       # R06N04 not judged yet
    assert s["parts"]["trunk"]["off"] == 1 and s["parts"]["trunk"]["share_off"] == pytest.approx(1 / 3)
    assert s["leaders"]["on"] == 4 and s["leaders"]["off"] == 1 and s["leaders"]["hidden"] == 1
    assert s["trees_any_off"] == 2 and s["off_trees"] == ["R01N01", "R05N03"]
    rows[0]["left"] = "maybe"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="maybe"):
        read_verdicts(path)


def test_wilson_interval():
    lo, hi = wilson(0, 40)
    assert lo == 0 and 0.08 < hi < 0.09
    lo, hi = wilson(20, 40)
    assert lo == pytest.approx(1 - hi) and 0.34 < lo < 0.36
    assert all(np.isnan(v) for v in wilson(0, 0))


def test_a_panel_shows_the_photo_and_the_masks_outline():
    rgb = np.full((480, 640, 3), 90, np.uint8)
    mask = np.zeros((256, 256), bool)
    mask[100:200, 120:136] = True
    panel = np.asarray(draw_panel(rgb, mask, "R01N01"))
    assert panel.shape == (480 + 28, 2 * 480 + 8, 3)
    left, right = panel[28:, :480], panel[28:, 488:]
    magenta = np.all(right == (255, 0, 255), axis=-1)
    assert not np.all(left == (255, 0, 255), axis=-1).any() and magenta.any()
    ys, xs = np.nonzero(magenta)                                            # the outline of the mask, scaled
    assert ys.min() == pytest.approx(100 * 480 / 256, abs=2) and xs.max() == pytest.approx(136 * 480 / 256, abs=2)


def test_cli_sample_once_draw_and_summary(tmp_path, capsys):
    sample = tmp_path / "sample.csv"
    args = ["--table", str(ROOT / "data/shape/shape_measures.csv"), "--split", str(ROOT / "splits.json"),
            "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"), "--sample", str(sample)]
    assert main(["sample", *args]) == 0 and "40 of 449" in capsys.readouterr().out
    assert main(["sample", *args]) == 1                                      # drawn once
    local = tmp_path / "local.csv"
    write_template(local, ["R01N01", "R15N05"])                              # development trees in the repository
    out = tmp_path / "img"
    assert main(["draw", str(ROOT / "data/sample"), "--mask-overlay", str(ROOT / "annotations/mask_overlay.json"),
                 "--sample", str(local), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["01_R01N01.jpg", "02_R15N05.jpg"]
    assert main(["summary", "--sample", str(local), "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["judged_trees"] == 0


# --- judging key by key ------------------------------------------------------------------------------

def judge_table(tmp_path, trees=("R01N01", "R02N02", "R05N03")):
    path = tmp_path / "sample.csv"
    write_template(path, list(trees))
    return path


def test_the_judge_asks_trunk_left_right_and_saves_every_verdict(tmp_path):
    from treeskel.label_audit import AuditJudge

    path = judge_table(tmp_path)
    j = AuditJudge(path)
    assert j.tree == "R01N01" and "TRUNK" in j.prompt()
    for k in "ofh":
        j.key(k)
    assert j.tree == "R02N02" and j.part == 0                              # on to the next tree
    saved = read_verdicts(path)
    assert [saved[0][p] for p in ("trunk", "left", "right")] == ["on", "off", "hidden"]
    j.key("o")
    assert read_verdicts(path)[1]["trunk"] == "on"                          # saved at once, not only per tree
    j.key("x")                                                              # an unknown key changes nothing
    assert j.part == 1
    again = AuditJudge(path)                                                # a new start resumes where it stopped
    assert again.tree == "R02N02" and again.part == 1 and again.judged() == 1


def test_undo_reaches_back_into_the_previous_tree(tmp_path):
    from treeskel.label_audit import AuditJudge

    j = AuditJudge(judge_table(tmp_path))
    for k in "ooo":
        j.key(k)
    assert j.tree == "R02N02"
    j.key("backspace")
    assert j.tree == "R01N01" and j.part == 2 and j.rows[0]["right"] == ""
    assert read_verdicts(j.path)[0]["right"] == ""
    j.key("f")
    assert j.rows[0]["right"] == "off" and j.tree == "R02N02"
    j.key("p")
    assert j.tree == "R01N01" and "done" in j.prompt()                      # a complete tree is shown, not re-asked
    j.key("o")
    assert j.rows[0]["trunk"] == "on" and j.tree == "R01N01"               # judging keys do nothing there
    j.key("u")
    j.key("u")
    j.key("u")
    assert j.part == 0 and all(j.rows[0][p] == "" for p in ("trunk", "left", "right"))


def test_when_all_are_judged_q_quits(tmp_path):
    from treeskel.label_audit import AuditJudge

    j = AuditJudge(judge_table(tmp_path, ("R01N01",)))
    for k in "ohh":
        j.key(k)
    assert j.tree is None and "All 1 trees judged" in j.prompt()
    j.key("q")
    assert j.done and summarise(read_verdicts(j.path))["judged_trees"] == 1


def test_a_failed_save_is_reported_and_tried_again(tmp_path, monkeypatch):
    import treeskel.label_audit as la

    j = la.AuditJudge(judge_table(tmp_path))
    real = la.write_verdicts

    def locked(path, rows):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(la, "write_verdicts", locked)
    j.key("o")
    assert "COULD NOT SAVE" in j.prompt() and read_verdicts(j.path)[0]["trunk"] == ""
    j.key("q")
    assert not j.done                                                       # not quit with unsaved verdicts
    monkeypatch.setattr(la, "write_verdicts", real)
    j.key("space")                                                          # any key tries again
    assert j.message == "" and read_verdicts(j.path)[0]["trunk"] == "on"


def test_a_table_saved_by_excel_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes("﻿tree,row,trunk,left,right,notes\r\nR01N01,1,on,OFF,hidden,\r\n".encode("utf-8"))
    rows = read_verdicts(path)
    assert rows[0]["tree"] == "R01N01" and [rows[0][p] for p in ("trunk", "left", "right")] == ["on", "off", "hidden"]


def test_the_judging_window_passes_keys_to_the_judge(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from treeskel.label_audit import AuditJudge, run_judge

    j = AuditJudge(judge_table(tmp_path, ("R01N01", "R02N02")))
    asked = []

    def image(t):
        asked.append(t)
        return np.zeros((20, 30, 3), np.uint8)

    fig = run_judge(j, image, show=False)
    assert asked == ["R01N01"]

    class Key:
        def __init__(self, key):
            self.key = key

    for k in "ofo":
        fig._judge_on_key(Key(k))
    assert asked == ["R01N01", "R02N02"] and read_verdicts(j.path)[0]["left"] == "off"
    fig._judge_on_key(Key("q"))
    assert j.done
    plt.close("all")
