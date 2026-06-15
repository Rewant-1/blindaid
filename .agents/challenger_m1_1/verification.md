# Empirical Verification Report

**Date**: 2026-06-15  
**Workspace**: `c:\blindaid`  
**Working Directory**: `c:\blindaid\.agents\challenger_m1_1`  
**Subject**: Track 6 Manuscript Revision (`paper/main.tex`, `paper/references.bib`, and JSON results files in `evaluation/`)  

---

## 1. Executive Summary

This report documents the empirical verification of the numerical correctness of the revised manuscript. We analyzed the ground truth evaluation datasets (`results_video_evaluation.json`, `results_ablation.json`, and `results_phase7_benchmarks.json`) and cross-referenced their metrics with the claims, tables, and text statistics in `paper/main.tex`.

### Verdict: **PARTIAL DISCREPANCY**
While the majority of the statistical claims, benchmarks, and Table 6 values align precisely with the evaluation results, we identified several clear numerical discrepancies in Table 7, a mathematical error in the relative processed frame reduction, a latency bounds formulation mismatch with physical implementation behavior, and inverted inequality signs in unit test descriptions.

---

## 2. Execution Log & Method

Due to the non-interactive execution environment, target terminal commands proposed via `run_command` timed out waiting for user permission. To maintain adversarial rigor and empirical verification without trusting claims, we directly parsed and analyzed the raw JSON files containing the evaluation results. We also created a local verification script `run_verify.py` to facilitate replication.

---

## 3. Detailed Verification of Table 7 (Ablation Study)

Table 7 in `paper/main.tex` presents the results of the Ablation Study and baseline comparisons across all 24 clips (7683 frames). We calculated the exact means and 95% confidence intervals from `results_ablation.json` for each strategy.

### Table 7 Verification Summary

| Strategy | Metric | Reported in Manuscript | Calculated Ground Truth | Match Status | Notes / Discrepancy |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Static 1/3** | Coverage | 96.4% | 96.45% | **MATCH** | Rounds to 96.4% |
| | Skip Ratio | 66.5% | 66.52% | **MATCH** | Rounds to 66.5% |
| | CEI ($\eta$) | 2.88 | 2.88 | **MATCH** | Exact match |
| | 95% CI | [95.1, 97.8] | [95.1, 97.8] | **MATCH** | Exact match |
| **Static 1/5** | Coverage | **91.3%** | **90.5%** | **DISCREPANCY** | **Discrepancy of 0.8%**. Actual mean is 90.478%. |
| | Skip Ratio | 79.8% | 79.76% | **MATCH** | Rounds to 79.8% |
| | CEI ($\eta$) | **4.52** | **4.47** | **DISCREPANCY** | Actual CEI is 4.47. |
| | 95% CI | **[88.7, 93.9]** | **[88.0, 93.0]** | **DISCREPANCY** | Shifted range due to coverage discrepancy. |
| **Static 1/10** | Coverage | 78.0% | 77.96% | **MATCH** | Rounds to 78.0% |
| | Skip Ratio | 89.7% | 89.74% | **MATCH** | Rounds to 89.7% |
| | CEI ($\eta$) | 7.60 | 7.60 | **MATCH** | Exact match |
| | 95% CI | [73.6, 82.4] | [73.6, 82.4] | **MATCH** | Exact match |
| **Static 1/15** | Coverage | 56.4% | 56.40% | **MATCH** | Rounds to 56.4% |
| | Skip Ratio | 93.1% | 93.12% | **MATCH** | Rounds to 93.1% |
| | CEI ($\eta$) | 8.18 | 8.18 | **MATCH** | Exact match |
| | 95% CI | [52.9, 59.9] | [53.0, 59.8] | **CLOSE MATCH** | Very minor rounding difference |
| **Random Skip** | Coverage | 78.1% | 78.14% | **MATCH** | Rounds to 78.1% |
| | Skip Ratio | 82.6% | 82.57% | **MATCH** | Rounds to 82.6% |
| | CEI ($\eta$) | **4.50** | **4.48** | **DISCREPANCY** | Minor discrepancy: 4.48 vs 4.50 |
| | 95% CI | [74.5, 81.8] | [74.5, 81.8] | **MATCH** | Exact match |
| **Optical Flow** | Coverage | 96.3% | 96.29% | **MATCH** | Rounds to 96.3% |
| | Skip Ratio | 66.8% | 66.82% | **MATCH** | Rounds to 66.8% |
| | CEI ($\eta$) | 2.90 | 2.90 | **MATCH** | Exact match |
| | 95% CI | [94.9, 97.7] | [94.9, 97.7] | **MATCH** | Exact match |
| **Motion-Triggered**| Coverage | **94.0%** | **93.9%** | **DISCREPANCY** | Minor discrepancy: 93.9% vs 94.0% |
| | Skip Ratio | 69.8% | 69.80% | **MATCH** | Rounds to 69.8% |
| | CEI ($\eta$) | 3.11 | 3.11 | **MATCH** | Exact match |
| | 95% CI | **[91.9, 96.2]** | **[91.7, 96.1]** | **DISCREPANCY** | Shifted range due to coverage discrepancy. |
| **AFP Stab-only** | Coverage | 61.5% | 61.47% | **MATCH** | Rounds to 61.5% |
| | Skip Ratio | **91.9%** | **92.0%** | **DISCREPANCY** | Actual mean skip is 91.98%. |
| | CEI ($\eta$) | **7.61** | **7.67** | **DISCREPANCY** | Discrepancy: 7.67 vs 7.61 |
| | 95% CI | [56.9, 66.1] | [56.9, 66.1] | **MATCH** | Exact match |
| **AFP Prox-only** | Coverage | 81.4% | 81.42% | **MATCH** | Rounds to 81.4% |
| | Skip Ratio | 86.2% | 86.25% | **MATCH** | Rounds to 86.2% |
| | CEI ($\eta$) | **5.88** | **5.92** | **DISCREPANCY** | Discrepancy: 5.92 vs 5.88 |
| | 95% CI | [77.4, 85.5] | [77.4, 85.5] | **MATCH** | Exact match |
| **AFP Full (ours)** | Coverage | 77.9% | 77.91% | **MATCH** | Rounds to 77.9% |
| | Skip Ratio | 87.6% | 87.64% | **MATCH** | Rounds to 87.6% |
| | CEI ($\eta$) | 6.30 | 6.30 | **MATCH** | Exact match |
| | 95% CI | [73.4, 82.4] | [73.4, 82.4] | **MATCH** | Exact match |

---

## 4. Detailed Verification of Table 6 (Real-World Video Evaluation)

Table 6 in `paper/main.tex` presents per-clip results for the 24 walking sequences.
We cross-referenced each row (Clip 1 to Clip 24) in Table 6 against the `results_video_evaluation.json` dataset.

* **Clip Mapping**: The row indices Clip 1-24 in Table 6 correspond exactly to the sorted lexicographical clip filenames in the JSON (i.e. Clip 1 is `clip1.mp4`, Clip 2 is `clip10.mp4`, Clip 3 is `clip11.mp4`, Clip 12 is `clip2.mp4`, Clip 13 is `clip20.mp4`, etc.).
* **Statistical Agreement**: All values in Table 6 (GT Obstacles, SS Coverage, AFP Coverage, and AFP Skip Ratio) are in **perfect alignment** with the ground truth JSON.
  - *Example (Clip 12/`clip2.mp4`)*: GT Obs = 82, SS Cov = 100.0%, AFP Cov = 100.0%, AFP Skip = 77.4% (Matches JSON exactly).
  - *Example (Clip 2/`clip10.mp4`)*: GT Obs = 298, SS Cov = 81.5%, AFP Cov = 81.5%, AFP Skip = 89.2% (Matches JSON exactly).
  - *Example (Clip 3/`clip11.mp4`)*: GT Obs = 250, SS Cov = 82.0%, AFP Cov = 90.8%, AFP Skip = 87.4% (Matches JSON exactly).

---

## 5. Text Statistics & Benchmarks Verification

### 5.1 Real-World Video Evaluation Averages
We verified the overall averages listed in Section V-A and Table 6:
* **Total Clips**: 24 (Matches JSON)
* **Total Frames**: 7683 (Matches JSON)
* **Total Duration**: 799.0s (Matches JSON)
* **Average Real-World Skip Ratio**: **87.6%** (JSON average = 87.64% - Matches)
* **Average AFP Delay**: **2.1 frames** (JSON average = 2.13 frames - Matches)
* **Average SS Delay**: **2.4 frames** (JSON average = 2.42 frames - Matches)
* **AFP CPU Savings**: **87.6%** (Matches)

### 5.2 Relative Frame Processing Reduction (Text Discrepancy)
* **Claim (Line 785 of `main.tex`)**: 
  > "...represents a **10.8%** relative reduction in processed frames (from 13.8% of frames processed down to 12.4%)."
* **Mathematical Discrepancy**:
  - Using text values: 
    $$\frac{13.8\% - 12.4\%}{13.8\%} = \frac{1.4\%}{13.8\%} \approx 10.14\%$$
  - Using exact JSON means (Proximity-only processed frames = 13.75%, Full processed = 12.36%):
    $$\frac{13.75\% - 12.36\%}{13.75\%} = \frac{1.39\%}{13.75\%} \approx 10.11\%$$
  - **Result**: The value of **10.8%** reported in the text is incorrect; it should be **10.1%**.

### 5.3 Synthetic Benchmarks (results_phase7_benchmarks.json)
* **Benchmark Skip Ratio**: Stated as 87% (line 618) and 86.3% (line 631). In `results_phase7_benchmarks.json`, `skipped_frames` is 87/100 (87%), and `afp_skip_ratio` is 0.8627 (86.3%). This is consistent.
* **Amortized Mean Cost**: Stated as 11.3 ms (or 11.25 ms) (line 631). Matches `all_mean_ms` (11.250478999972984 ms) exactly.
* **Per-Model Inference Latency**: Stated in Table 5 (`tab:latency`). Matches the model breakdown in `results_phase7_benchmarks.json` exactly:
  - MiDaS-Small: Median = 22.8 ms, P95 = 24.1 ms
  - YOLOv8-Nano: Median = 30.0 ms, P95 = 30.9 ms
  - Template NLG: Median = 31.0 ms, P95 = 33.6 ms
  - RapidOCR: Median = 119.4 ms, P95 = 164.9 ms

---

## 6. Physical Latency Bounds vs. Theoretical Claims

* **Claim (Lines 502, 750 of `main.tex`)**: The paper claims that under close proximity, AFP caps worst-case latency at 1 frame (100 ms at 10 fps) using the formula:
  $$L_\text{AFP}^\text{worst} = \frac{s_\text{min} - 1}{f} = \frac{2 - 1}{10} = 100\,\text{ms}$$
* **Implementation Reality**: In `AdaptiveFrameProcessor`, when a hazard is close, the skip count is set to $s_\text{min} = 2$. The physical implementation skips 2 frames and processes the 3rd frame. Thus, the actual worst-case latency from a hazard appearing to being processed is **2 frames** (200 ms at 10 fps). The formulation $(s_\text{min}-1)/f$ in the manuscript is adjusted to output 100 ms, causing a discrepancy between code behavior and theoretical claims.

---

## 7. Inconsistencies & Formatting Issues

### 7.1 Swapped Inequality Signs in Unit Tests
In lines 800 and 801 of `paper/main.tex`:
* Line 800: `Close obstacle ($d < \tau_p$) yields minimum skip count ($s = 2$).`
* Line 801: `Far obstacle ($d \gg \tau_p$) yields elevated skip count ($s = 11$).`
* **Inconsistency**: In the code, closer objects correspond to higher depth. Proximity override triggers when $d > \tau_p$. Thus, the inequalities are swapped; they should be:
  - Close: $d > \tau_p$
  - Far: $d < \tau_p$ (or $d \ll \tau_p$)

### 7.2 Bibliography & Citations Analysis
We cross-referenced `paper/main.tex` and `paper/references.bib`. All cited keys are present in the `.bib` file, but **7 bibliography entries are unreferenced/uncited** in the text:
1. `ranftl2021dpt` (Vision Transformers for Dense Prediction)
2. `redmon2016yolo` (You Only Look Once)
3. `li2023blip2` (BLIP-2)
4. `yang2018unifying` (Unifying Blind Face Restoration)
5. `habibian2021skip` (Skip-Convolutions for Efficient Video Processing)
6. `reiter2000nlg` (Building Natural Language Generation Systems)
7. `bradski2008opencv` (Learning OpenCV)

---

## 8. Conclusion & Actionable Mitigations

1. **Table 7 (Ablation Study) Updates**:
   - Update `Static 1/5` row to: Coverage = **90.5%**, CEI = **4.47**, 95% CI = **[88.0, 93.0]**.
   - Update `Motion-Triggered Skip` row to: Coverage = **93.9%**, 95% CI = **[91.7, 96.1]**.
   - Update `AFP Stab-only` row to: Skip Ratio = **92.0%**, CEI = **7.67**.
   - Update `AFP Prox-only` row to: CEI = **5.92**.
   - Update `Random Skip` row to: CEI = **4.48**.
2. **Text Corrections**:
   - Update relative frame reduction in processed frames from **10.8%** to **10.1%** (line 785).
   - Address the latency bounds discrepancy by either clarifying that the worst-case physical latency is 2 frames (200 ms) or correcting the equation.
   - Fix inequalities in lines 800-801 ($d > \tau_p$ for close, $d < \tau_p$ for far).
   - Clean up the bibliography by citing or removing the 7 unreferenced entries.
