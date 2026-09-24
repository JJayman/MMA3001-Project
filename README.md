# Seeing Branches Through Leaves: Apple-Tree Skeleton Estimation from RGB-D

MMA3001 Numerical Methods and Machine Learning, individual project.

**Goal:** estimate the main branch structure (trunk and scaffold limbs) of apple
trees from a *summer* RGB-D image, where leaves hide much of the wood. The
*winter* RGB-D image of the same tree, when the branches are bare, is used to
build the ground truth. The intended use is perception for orchard robots
(pruning, harvesting, path planning), which must avoid or target branches they
cannot directly see.

> Work in progress. See the sections below for what currently exists.

## Repository structure

```
data/sample/     8-tree sample of the dataset (full dataset not committed)
src/treeskel/    Python package
  io.py          filename parsing, dataset indexing, image loading
  splits.py      row-grouped hold-out and k-fold cross-validation splits
tests/           pytest test suite
docs/progress/   session-by-session progress notes (incl. AI use)
splits.json      fixed test rows and cross-validation folds (do not change)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
pytest                                     # run the tests
python -m treeskel.splits data/sample      # dataset summary and splits
python -m treeskel.splits data/raw         # same for the full dataset
pdoc -o docs/_build src/treeskel           # build HTML documentation
```

## Data

Each tree (e.g. `R01U01` = row 1, tree 1) has a summer and a winter RGB image
and depth image, 640x480 PNG. Depth is 8-bit relative depth (brighter = closer,
0 = no reading), not metric. See [`data/sample/README.md`](data/sample/README.md).

The full dataset (544 trees in 23 rows) comes from the MMA3001 teaching team
and is not redistributed here. Put it in `data/raw/`, which is git-ignored.

## Validation design

Whole orchard rows are held out as a final test set (about 15% of trees), and
5-fold cross-validation grouped by row is used on the remaining rows for model
selection. The split is fixed in `splits.json` (test rows 12, 18, 21) and was
committed before any model was trained. Row 4 is excluded because all of its
winter depth images are identical, leaving 513 trees in 22 rows. Grouping by row stops neighbouring trees, which share background,
lighting and capture session, from appearing in both training and test data.


## Acknowledgement of AI Assistance in Project Development

This project incorporates Artificial Intelligence (AI) tools as assistive technology throughout its development lifecycle. Our methodology emphasizes a "human-in-the-loop" approach, ensuring that human oversight, critical evaluation, and final decision-making remain central to every stage.

AI tools may have contributed to the following aspects:

*   **Code Generation and Refinement:** AI has aided in generating boilerplate code, suggesting improvements, and refactoring code segments. All AI-generated code undergoes thorough human review, testing, and comprehension before integration.
*   **Documentation and Explanation:** AI has assisted in drafting documentation, clarifying complex algorithms, and summarizing research. Human verification is paramount to ensure accuracy, clarity, and completeness.
*   **Test Case Generation:** AI has been utilized for suggesting potential test cases and identifying edge cases. Human developers rigorously validate the relevance and correctness of all generated tests.
*   **Conceptual Brainstorming:** AI has supported brainstorming sessions, exploring diverse approaches, and providing background information on relevant subjects. All conceptual outputs are critically evaluated by human team members.

Our commitment includes:

*   **Transparency:** Clearly documenting the specific applications and extent of AI usage.
*   **Responsibility:** Upholding ethical and responsible AI practices, with a focus on fairness, accountability, and privacy.
*   **Quality Assurance:** Maintaining rigorous standards for code quality and functionality through comprehensive human-led testing and review processes, irrespective of AI involvement.

Ultimately, the human developers bear full responsibility for the integrity, correctness, and functionality of this project.
