# Cost-map tuning results

Produced by `python -m treeskel.tune_cost data/tuning --grid wide --samples 250`
and `... --grid focused`, on the 17 annotated tuning trees (development rows
only). The reference is the annotator's 64 correction clicks: the error is the
distance from each correction click to the path traced through only the four
main clicks (root, fork, two tips).

| Method | Mean (px) | Median (px) | Correction clicks > 8 px from trace |
|---|---|---|---|
| Straight lines between clicks | 16.1 | 11.0 | 59% |
| Original settings | 11.6 | 9.5 | 62% |
| Best of wide random search (251 settings) | 3.8 | 2.9 | 12% |
| Best of focused grid (55 settings) | **3.2** | **2.8** | **6%** |
| Leave-one-out, focused grid (unseen trees) | 4.8 | 3.0 | 11% |

- The chosen setting (now the default in `GuidedParams`) is depth 0.2, tube 0.1,
  colour 0, cost power 1, centre weight 1 (centredness from depth).
- Several settings tie at 3.2 px. What matters is the depth:tube ratio of
  2:1 and a fairly flat cost, which suggests the optimum is robust.
- The focused region was chosen after seeing the wide search on all trees, so
  its leave-one-out estimate is slightly optimistic.
- An earlier metric, comparing with reference *traces* drawn using the
  original settings, ranked the original settings first. That reference
  equals the original trace wherever no correction was made, so it was
  biased and was replaced.
- `docs/figures/guided_tuning_before_after.jpg`: four trees traced from the
  main clicks only, original vs tuned settings. Magenta squares are the
  annotator's correction clicks.
