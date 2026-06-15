# Project: IEEE INDISCON 2026 Track 6 Manuscript Revision

## Architecture
- The project involves updating a LaTeX paper manuscript (`paper/main.tex` and `paper/references.bib`) to address reviewer feedback and align all quantitative numbers with the evaluation logs (`evaluation/results_video_evaluation.json`, `evaluation/results_ablation.json`, and `evaluation/results_phase7_benchmarks.json`).
- There is also a Python script `evaluation/verify_numbers.py` which computes the ground truth numbers to check consistency.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Manuscript Revision | Reframe narrative, reformulate CEI/safety metrics, update bibliography, remove manual annotation study, and align all quantitative results in text and tables with ground truth JSON files. | None | IN_PROGRESS |

## Code Layout
- `paper/main.tex`: LaTeX paper manuscript
- `paper/references.bib`: BibTeX references file
- `evaluation/results_video_evaluation.json`: Ground truth evaluation metrics for video clips
- `evaluation/results_ablation.json`: Ground truth ablation study metrics
- `evaluation/results_phase7_benchmarks.json`: Ground truth benchmark execution metrics
- `evaluation/verify_numbers.py`: Script to output ground truth values for verification
