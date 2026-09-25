"""Tests for treeskel.winter_gt, using synthetic depth images with a known answer."""

import numpy as np
import pytest
from skimage.draw import line

from treeskel.winter_gt import (
    GTParams,
    extract_winter_gt,
    foreground_mask,
    main,
    render_main_branches,
)

FORK = (330, 320)
LEFT_TIP = (20, 180)
RIGHT_TIP = (20, 460)


def thick_line(img, p0, p1, width, value):
    """Draw a line of the given width into ``img``."""
    for off in range(-(width // 2), width // 2 + 1):
        rr, cc = line(p0[0], p0[1] + off, p1[0], p1[1] + off)
        img[rr, cc] = value


def synthetic_tree(gap=False, short_left=False, post=False):
    """Depth image of a V tree: ground below row 300, empty background above.

    Returns ``(depth, valid)``. The tree (value 150) is nearer than the ground
    (value 100). Optional: a gap in the right leader, a short left leader, or
    a post at the left edge (a distractor).
    """
    d = np.zeros((480, 640), np.uint8)
    d[300:] = 100                                     # ground
    thick_line(d, (470, 320), FORK, 13, 150)          # trunk
    thick_line(d, FORK, (280, 290) if short_left else LEFT_TIP, 9, 150)
    thick_line(d, FORK, RIGHT_TIP, 9, 150)
    if gap:
        d[150:172, 380:460] = 0                       # depth drop-out on right leader
    if post:
        d[:, 20:34] = 150
    return d, d > 0


def test_foreground_separates_tree_from_ground():
    d, v = synthetic_tree()
    fg = foreground_mask(d, v)
    assert fg[400, 320]            # trunk, in a ground row
    assert not fg[400, 100]        # ground
    # Left leader crosses row 100 at x = 320 - 140 * 230 / 310 ≈ 216.
    assert fg[100, 210:223].any()
    assert not fg[100, 300:340].any()   # empty background between the leaders


def test_finds_fork_and_both_leaders():
    gt = extract_winter_gt(*synthetic_tree())
    assert gt.ok, gt.issues
    assert abs(gt.fork[0] - FORK[0]) <= 15 and abs(gt.fork[1] - FORK[1]) <= 15
    assert gt.left[-1][0] < 40 and gt.left[-1][1] < 220      # reaches top-left tip
    assert gt.right[-1][0] < 40 and gt.right[-1][1] > 420    # reaches top-right tip
    assert gt.root[0] > 440
    # Paths join up: trunk ends where both leaders start.
    assert tuple(gt.trunk[-1]) == tuple(gt.left[0]) == tuple(gt.right[0]) == gt.fork


def test_links_across_depth_gap():
    gt = extract_winter_gt(*synthetic_tree(gap=True))
    assert gt.ok, gt.issues
    assert gt.right[-1][0] < 40


def test_gap_not_linked_when_linking_is_disabled():
    gt = extract_winter_gt(*synthetic_tree(gap=True), GTParams(link_gap=0))
    assert gt.right[-1][0] > 150      # stops below the gap


def test_ignores_post_at_edge():
    gt = extract_winter_gt(*synthetic_tree(post=True))
    assert gt.ok, gt.issues
    assert abs(gt.root[1] - 320) < 20


def test_short_leader_is_flagged():
    gt = extract_winter_gt(*synthetic_tree(short_left=True))
    assert not gt.ok
    assert any("left leader" in s for s in gt.issues)


def test_empty_image_is_flagged():
    d = np.zeros((480, 640), np.uint8)
    gt = extract_winter_gt(d, d > 0)
    assert not gt.ok and gt.issues == ["no tree region found"]
    assert not render_main_branches(gt).any()


def test_render_covers_branches_only():
    gt = extract_winter_gt(*synthetic_tree())
    m = render_main_branches(gt)
    assert m[400, 320] and m[FORK]
    assert not m[400, 100] and not m[100, 600]
    # Drawn width follows the tree's thickness (trunk 13 px wide).
    assert 9 <= m[400].sum() <= 17


def test_summary_is_plain_data():
    s = extract_winter_gt(*synthetic_tree()).summary()
    assert s["ok"] is True and s["issues"] == ""
    assert s["left_rise"] > 250 and s["right_rise"] > 250


def test_cli_on_sample(sample_dir, tmp_path, capsys):
    fig = tmp_path / "overview.png"
    assert main([str(sample_dir), "--out", str(tmp_path), "--figure", str(fig)]) == 0
    out = capsys.readouterr().out
    assert "Processed 8 trees" in out
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "params.json").exists()
    assert len(list(tmp_path.glob("*_winter_gt.png"))) == 8
    assert fig.exists()
