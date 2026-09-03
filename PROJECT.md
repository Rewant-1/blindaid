# Project: Adaptive Frame Processing (AFP) — IEEE INDISCON 2026

## Status: ACCEPTED (Track 6: Signal Processing, Computing & Data Science)
**Conference:** 7th IEEE India Council International Subsections Conference (INDISCON 2026)  
**Paper ID:** 2632  
**Title:** *Adaptive Frame Processing: Latency-Bounded Temporal Scheduling for Multi-Modal Edge Inference*

---

## Architecture & Code Layout

- **`paper/`**: Camera-ready LaTeX paper manuscript (`main.tex`) and BibTeX references database (`references.bib`).
- **`blindaid/`**: Full production assistive vision system implementing AFP (`core/adaptive_processor.py`), Guardian mode, ONNX depth estimation, YOLOv8 detection, and 3D spatial fusion.
- **`clips/`**: 24 real-world evaluation video sequences (7,683 frames, 799.0s total).
- **`evaluation/`**: Reproducibility and benchmarking suite:
  - `results_video_evaluation.json`: Ground truth per-clip results for Table IV.
  - `results_ablation.json`: Ground truth ablation results for Table III across 10 strategies.
  - `results_phase7_benchmarks.json`: Ground truth latency (Table I) and sensitivity (Table V) metrics.
  - `verify_numbers.py`: Automated verification script for all paper metrics.
  - `compute_stats.py`: TOST statistical equivalence test runner ($p < 0.001$, $\delta=\pm 5$\,pp).
  - `analyze_results.py`: Quick summary of per-clip performance.
  - `evaluate_video.py`, `run_ablation.py`, `run_benchmarks.py`: Full evaluation execution scripts.
  - `test_adaptive.py`, `test_scene_intelligence.py`: Unit and regression test suites.
- **`resources/models/`**: ONNX models (`midas_small.onnx`, `yolov8n.onnx`, `yolov8n.pt`).
- **`scripts/`**: Utility scripts (e.g. `download_onnx_models.py`).

---

## Milestones Log

| # | Name | Scope | Status |
|---|------|-------|--------|
| 1 | Manuscript Reframing | Reframe narrative to latency-bounded edge inference, incorporate TOST equivalence testing, update related work with across-frame video sampling, and remove old manual annotation study. | COMPLETED |
| 2 | Peer Review Fixes | Correct latency bound equation ($L = s_\text{min}/f = 200$\,ms), replace Wilcoxon null results with TOST equivalence ($p < 0.001$), add $\tau_p$ relative depth caveat and sensitivity framing. | COMPLETED |
| 3 | IEEE Compliance | Verify double-blind anonymity and enforce 6-page formatting constraints. | COMPLETED |
| 4 | Peer Review Acceptance | Paper formally accepted at IEEE INDISCON 2026 with strong positive reviews. | COMPLETED |
| 5 | Workspace Cleanup | Purged obsolete agent logs, old manual annotation artifacts, and unreferenced dumps; structured repo for audience presentation and full reproducibility. | COMPLETED |
