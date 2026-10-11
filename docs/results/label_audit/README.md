# Label audit: are the lined-up hand masks locally off? (session 16)

**Status: sample drawn; verdicts not yet given.**

## Why

In session 11, each of the 449 development hand masks was moved **as a whole** onto its summer photo
(`treeskel.annotate_mask`). That raised the U-Net's V-opening skill from +0.06 to +0.72: label position
was the largest gain in the project.

A whole-mask shift cannot fix a mask whose trunk sits on the wood while a leader is drawn beside it. This
audit measures how common that is in the labels the network trains on. The plan and what follows from the
result are in [`../segment_variants/rule.md`](../segment_variants/rule.md).

## The sample (`sample.csv`)

- **Size:** 40 of the 449 lined-up development trees.
- **How drawn:** at random within each orchard row, in proportion to the row's trees, with at least one per
  row. All 16 development rows are covered.
- **Fixed in advance:** seed 16. The sample was drawn by `python -m treeskel.label_audit sample` from the
  committed files only, and committed before any of its trees was looked at.
- **Precision:** with 40 trees, a 95% interval is about ±10–15 points. That is enough to tell a rare
  problem from a common one.
- **No test rows:** they are never read.

## How to judge (the student, by eye)

1. On the PC: `python -m treeskel.label_audit draw data/raw`. This writes one image per tree to
   `outputs/label_audit/`: the photo's centre 480x480 (the masks' frame) beside the same image with the
   lined-up hand mask's outline in magenta. No prediction is shown.
2. In `sample.csv`, give each part one verdict:
   - **Parts:** `trunk`, `left` and `right`. The left leader is the one whose tip is further left.
   - **`on`:** where the wood is visible, the outline follows it to within about half the branch's width.
   - **`off`:** along a stretch at least a trunk's width long, the mask lies beside the visible wood by
     about the branch's width or more.
   - **`hidden`:** too little of that part's wood is visible to judge.
   - **`notes`:** anything worth keeping, for example "left leader drawn on a side branch".
3. Run `python -m treeskel.label_audit summary`. It writes `report.json`: for each part, the share `off`
   among those judged, with Wilson 95% intervals, and the trees with any part off.

## Results

Not yet.
