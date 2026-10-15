# Figures written by each training run

`python -m treeskel.segment` and `shape_cnn` write these for every run: the training curves (`_training`),
the skill per measure (`_skill`), example masks (`_examples`, `_masks`). The run's numbers are in
`docs/results/<run>/`, and the README that explains each run is listed in
[`docs/results/README.md`](../../results/README.md). They were moved here in session 17 because no README
refers to them; the figures the READMEs use are one folder up.

New runs still write to `docs/figures/` (`--figures` sets the folder).
