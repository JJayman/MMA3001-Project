# Are the leaders in the middle of the leaves?

Session 12, the student's idea: leaves and apples are around branches, so they might locate a leader the
U-Net cannot see. `leaf_check.py` (the student's PC, 448 development trees with a lined-up hand mask)
samples how green the summer photo is (2G − R − B > 20) at sideways distances from the hand-mask leaders,
over the stretch the shape measures use; positive offsets are outwards, away from the other leader.
Output: `leaf_check_output.txt`. It was a description, not a test of a method: nothing was chosen by it.

## What it shows

- **A dip on the leader itself** (0.21 at 0, against 0.29–0.31 at ±10 px): its visible bark is not green.
- **The leaves fill the V, not the leader's surroundings.** Inside the V (negative offsets) the leaf
  fraction rises to its highest, 0.39, 50–60 px from the leader and stays near 0.35 out to 160 px; outside
  (positive offsets) it falls steadily from 0.34 at 20–30 px to 0.24 at 160 px. A leader runs along the
  outer side of the densest foliage, not through its middle.
- **Too gently to place a leader:** leaves are only 1.18 times as common 20–40 px from a leader as
  120–160 px away, and a tree's leafiest offset lies within 40 px of its leader in 29% of trees, where a
  flat profile would give about 20%.
- The hand masks are a person's reading of the same photos, so where the leader was hidden the person may
  have followed the leaves; if anything, this makes the profile look more leader-centred than it is.

## What it means

The leaves say which side of a leader the V is on, but not where the leader is to within the few pixels
the angles need. With the three completions tried on the development trees (growing into faint pixels,
continuing to pieces on the leader's line, completing the line from the leaders' shapes; none helped,
`../segment_lined_up_rerun/README.md`), this supports the conclusion that where leaves hide a leader,
the summer photo carries little information about exactly where it runs.
