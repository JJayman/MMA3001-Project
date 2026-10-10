# Bends out of the plane: preview on the 21 trees in the repository

**Not a result.** `python -m treeskel.bend3d data/sample data/tuning --out docs/results/bend3d_preview
--figure docs/results/bend3d_preview/bend3d.png --profiles R01N05 R01N13 R07N29 R15N05 R09N34 R01N18
--profiles-figure docs/figures/bend3d_profiles.png`, run in the AI assistant's environment on the
development trees it has: the 4 development trees of `data/sample` and the 17 of `data/tuning`, which were
picked in session 4 as problem trees. Claims from these trees failed on all 449 in sessions 6, 8 and 9.
The method, the checks and the rules for the 449-tree run are in [`../bend3d/README.md`](../bend3d/README.md),
where the summary of this preview is too.

It is kept apart from `docs/results/bend3d/` because `treeskel.model3d build --bends` reads the chosen prior
from there.

Files: `output.txt` (the printed summary), `report.json`, `cv.csv` (each tree's left-out errors under each
prior), `trees.csv` (each tree's measures: winter under six camera settings, winter with ridge readings only,
the hand mask and the U-Net's mask; the plane and the bends), `bend3d.png` (the summary figure).
