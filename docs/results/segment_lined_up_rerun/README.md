# Completing the broken leaders: the re-run, growing, continuing and shape completion

Session 12, after the model for the final test had been named (`../final_test/naming_rule.md`). The
student saw many predicted leaders stop short or break, and proposed completing them before the final
test. Two ways were tried, each fixed before it was scored and judged by the same rule: adopted only if
it beats the joined masks of the same network tree by tree on the V opening (lower MAE and Wilcoxon
p < 0.05). **Neither was adopted;** the named model stays the 256 joined run.

## Why it looked worth it (the named run, development trees)

- 33 of 444 trees have a leader shorter than the stretch the angles are fitted over (158 mask px); no
  hand mask does. The median shortfall is 56 mask px.
- Those 7% of trees carry **47% of the V opening's squared error** (bends: 20–25%). Were they predicted
  as well as the other trees, the V-opening skill would rise from +0.75 to about +0.86 (a ceiling, not a
  forecast).
- For 13 of 37 short leaders a leftover piece lies roughly in line with the tip (within 30°, 20–100 px
  away); the others simply fade out.

## The re-run (student's PC, 18.3 min)

`python -m treeskel.segment data/raw --mask-overlay annotations/mask_overlay.json --out
docs/results/segment_lined_up_rerun`: the named recipe trained again, its probability maps saved
(`probabilities.npz`). The GPU is not deterministic, so it is a second sample of the same recipe:

| Skill over the mean shape | V opening | Left / right bend | Left / right from trunk | Trunk from vertical | No fork |
|---|---|---|---|---|---|
| Original, as predicted | +0.72 | +0.57 / +0.47 | +0.61 / +0.67 | +0.62 | 14 |
| Re-run, as predicted | +0.72 | +0.47 / +0.53 | +0.60 / +0.68 | +0.61 | 12 |
| Original, joined | +0.75 | +0.60 / +0.53 | +0.64 / +0.72 | +0.64 | 2 |
| Re-run, joined | +0.75 | +0.61 / +0.57 | +0.62 / +0.72 | +0.65 | 2 |

The V opening repeats exactly; single measures move by up to 0.10 between two runs of the same recipe
(the left bend as predicted). Tree by tree the two runs do not differ on any measure (p ≥ 0.11 as
predicted). Differences between variants smaller than this are not worth reading into.

## Growing into less sure pixels (`segment_lined_up_rerun_grown`)

- **Method:** each mask kept its pixels above 0.5 and gained every pixel above a lower threshold
  connected to it (hysteresis thresholding), then was joined. The threshold was chosen among 0.5 to 0.05
  by the mean clDice (centre-line Dice) against the lined-up hand masks, never by winter:

  | Lower threshold | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 | **0.1** | 0.05 |
  |---|---|---|---|---|---|---|---|---|
  | Mean clDice | 0.8657 | 0.8745 | 0.8785 | 0.8800 | 0.8814 | 0.8832 | **0.8837** | 0.8834 |

- **What it did:** at 0.1 every mask grew (604,615 pixels, about a quarter more), and the student saw side
  branches appear. Masks with no fork 2 → 1.
- **Against the re-run's joined masks:** V opening MAE 4.105° → 4.141°, closer on only 42% of trees
  (p = 0.003, worse more often); right bend 5.4° → 5.8° (p = 1e-6). **Not adopted.**
- **Why:** clDice rewards covering the hand masks' centre lines, and growth does cover more of them, but
  the extra branches give the leader finder more paths to choose from, and it sometimes follows one.

## Continuing short leaders and pruning (`segment_lined_up_rerun_continued`)

- **Method (the student's idea):** a leader shorter than the fitted stretch is joined to a piece with a
  pixel within the missing part of the stretch and within 25° of the leader's direction at its tip (the
  hand masks' leaders turn by at most 24.8° over the stretch in 95% of cases); then every mask is cut down
  to its trunk and two leaders (`prune_to_branches`).
- **What it did:** only **9** masks had such a piece (of 46 short leaders); pruning removed 83,583
  pixels and moved some measure by more than 0.5° in 344 masks.
- **Against the re-run's joined masks:** V opening MAE 4.105° → 4.129°, 45% closer, p = 0.21. **Not
  adopted.** The angles from the trunk got worse (left 7.3° → 7.7°, right 7.0° → 7.3°, p ≈ 1e-8).
- **Why:**
  - Continuing alone moved the V opening closer to winter on 5 of the 9 trees and further on 4 (R06N28
    51.8° → 61.6° against winter's 60.7°; R06N12 51.1° → 39.7° against 53.8°).
  - Pruning did not mainly remove side branches (the measuring already ignores branches off the chosen
    paths); it trimmed the trunk's uneven bulges where the local half-width misses them, which moves the
    trunk's centre line near the fork, and with it the angles measured from the trunk.

## Predicting short leaders' lines from the leaders' shapes (`segment_lined_up_completed`)

- **Method (the student's next idea):** where a leader stops though the leaves go on, predict where it goes
  from the shapes of the leaders the model draws in full. A short leader's line over the whole stretch is
  its visible line moved by `a·g + c·g·bend` outwards (`g` the missing fraction of the stretch, `bend` the
  visible part's), fitted by least squares on complete leaders cut short on purpose, per fold on the other
  folds' trees (`treeskel.segment.complete_measures`, `--complete`). Masks unchanged.
- **Why it looked promising (no winter used):** cutting complete predicted leaders back to the short ones'
  lengths moves the V opening by 2.7° on average, about the whole excess error of trees with a short leader
  (6.8° against 3.7° on the re-run); on the cut complete leaders the model reduces that from 2.72° to 1.95°.
  Fitted on the named run: lines turn 2.8° more upright over a missing stretch and keep 0.69 of the visible
  part's turning.
- **Applied to the named run itself:** 31 trees changed. V opening MAE 4.100° → 4.102°, p = 0.78; on the 31
  trees, 8.33° → 8.36° (12 closer, 15 farther). **Not adopted.** The re-run replicate agrees (39 trees,
  6.84° → 7.16°; p = 0.65). The right leader's angle from the trunk got worse on the changed trees
  (7.70° → 9.11°; 1 closer, 13 farther).
- **Why:** on these trees the V opening as predicted is not biased (mean +1.1°, median +0.2° from winter),
  but far off either way (MAE 8.3°). The cutting test assumed the visible part of a short leader is right
  and only the rest is missing; on real short leaders the visible part itself is unreliable, drawn under
  the same leaves that stopped it.

## What this means

Most short leaders are not broken but missing: the network sees no leader there, neither above 0.5 nor
as faint probability that stays on the leader, nor as a piece further on; and where it stops, what it
did draw of the leader is unreliable too, so predicting the rest from its shape does not help either. Leaves hide them, and no
post-processing of the mask brings back what the photo does not show. The trees with short leaders
remain the largest source of error, and a limitation to report. The 256 joined run stays named
(`../final_test/model.json`).

Files: in each folder `report.json`, `predictions.csv`, `mask_scores.csv` and `predicted_masks.npz`
(`probabilities.npz` in the re-run's); `../segment_lined_up_rerun_continued/output.txt` is the continuing
run's console output (scored here, from the saved masks); figures `docs/figures/segment_lined_up_rerun*`.
