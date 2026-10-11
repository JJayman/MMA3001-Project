# The mask model: the centre-line loss and an ensemble (session 16)

**Status: planned; no run yet.**

- **The plan and its rule:** [`rule.md`](rule.md), fixed before any run.
- **The check done before any run:** [`loss_check.py`](loss_check.py), with its output in
  [`loss_check_output.txt`](loss_check_output.txt). It measures how much more the centre-line loss charges
  than Dice for a leader that stops short (1.63×) and for a gap (1.21×).

After the runs, `python -m treeskel.seg_compare compare data/raw` writes three files here:

- `report.json`: every run's numbers, the rule's verdict and the diagnostics;
- `trees.csv`: per tree and run, the V opening's error and the mask's structure;
- `output.txt`: the printed report.

The results will be written up below.

## Results

Not yet.
