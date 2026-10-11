# Label audit: are the lined-up hand masks locally off? (session 16)

**Status: done (12 October 2026). 2 of 75 judged leaders are off (3%), so label position is a minor
limitation.**
- **First pass:** it marked all 120 parts `on`, but the student was not sure what was being asked, so it is
  not used. It is kept in [`sample_first_pass.csv`](sample_first_pass.csv).
- **The result:** the student's second pass, after the guide below, in [`sample.csv`](sample.csv).

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

**The verdicts.** You are checking whether the hand-drawn mask, as lined up, sits on the branches you can
see in the photo. [`on_vs_off_example.jpg`](on_vs_off_example.jpg) shows both cases (made by
[`example.py`](example.py); the "off" one is made up for the guide). For each tree, judge three parts: the
trunk, the left leader (its tip is further left) and the right leader. Each gets one of:

- **on:** where the wood is visible, the magenta outline follows it to within about half the branch's width;
- **off:** along a stretch at least a trunk's width long, the outline lies beside the visible wood by about
  the branch's width or more;
- **hidden:** too little of that part's wood is visible to judge.

**In a window (the easy way):** `python -m treeskel.label_audit judge data/raw`.
- **What it shows:** each tree's photo beside the same photo with the lined-up hand mask's outline. No
  prediction is shown.
- **The keys:** **`o`** on, **`f`** off, **`h`** hidden, asked for the trunk, the left leader, then the right
  leader. **`u`** undoes the last answer, and **`n`** / **`p`** move between trees.
- **Saving:** every answer is saved to `sample.csv` at once. **`q`** quits, and running the command again
  carries on where you stopped.

**By hand instead:**
1. `python -m treeskel.label_audit draw data/raw` writes the same images to `outputs/label_audit/`.
2. Type `on`, `off` or `hidden` into the `trunk`, `left` and `right` columns of `sample.csv`. Optional notes
   go in `notes`, for example "left leader drawn on a side branch".

**Then:** `python -m treeskel.label_audit summary` writes `report.json`. It gives the share `off` for each
part among those judged, with Wilson 95% intervals, and the trees with any part off.

## Results

The student's second pass, summarised by `python -m treeskel.label_audit summary` ([`report.json`](report.json)):

| Part (40 trees) | On | Off | Hidden | Off among those judged [95%] |
|---|---|---|---|---|
| Trunk | 40 | 0 | 0 | 0% [0%, 9%] |
| Left leader | 35 | 2 | 3 | 5% [1%, 18%] |
| Right leader | 38 | 0 | 2 | 0% [0%, 9%] |
| **Both leaders** | **73** | **2** | **5** | **3% [1%, 9%]** |

- **The two masks off:** R02N27 and R05N29, both on the left leader. The trees with any part off are 5%
  [1%, 17%] of the sample.
- **The reading fixed before the audit** (`../segment_variants/rule.md`): under 10% of judged leaders off
  means label position is reported as a minor, measured limitation. It is: 3%, and the interval's upper end
  (9%) is also under 10%. The local correction tool proposed for "over 25%" is not needed.
- **For context only** (two trees cannot show more): the named U-Net's V opening is 10.3° off winter on
  R02N27, worse than on 94% of the development trees, and 4.5° off on R05N29 (median 2.9°).
- **How the second pass was made:** the student revised the first pass's sheet rather than starting from a
  blank one, so every part began as `on`. If that anchoring made anyone slower to call a part `off`, the
  true share could be somewhat higher. It cannot be told from this pass.
- **Hidden parts are leaders only** (5 of 80). Every trunk could be judged. A lined-up mask is therefore
  checked mostly where the wood is visible, and what it draws under the leaves remains a person's guess, as
  session 2 noted.
