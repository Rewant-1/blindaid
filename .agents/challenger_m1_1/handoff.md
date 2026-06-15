# Handoff Report — challenger_m1_1

## 1. Observation
1. **Manuscript Text & Tables**:
   - `paper/main.tex` at line 770:  
     `Static 1/5       & 91.3\% & 79.8\% & 4.52 & [88.7, 93.9] \\`
   - `paper/main.tex` at line 776:  
     `Motion-Triggered Skip & 94.0\% & 69.8\% & 3.11 & [91.9, 96.2] \\`
   - `paper/main.tex` at line 778:  
     `AFP Stab-only    & 61.5\% & 91.9\% & 7.61 & [56.9, 66.1] \\`
   - `paper/main.tex` at line 779:  
     `AFP Prox-only    & 81.4\% & 86.2\% & 5.88 & [77.4, 85.5] \\`
   - `paper/main.tex` at line 785:  
     `Even though the stability signal causes a small absolute increase of 1.4\% in the skip ratio, it represents a \textbf{10.8\% relative reduction in processed frames} (from 13.8\% of frames processed down to 12.4\%).`
   - `paper/main.tex` at line 502, 750:  
     `L_\text{AFP}^\text{worst} = \frac{s_\text{min} - 1}{f}` (formula)  
     `Crucially, under close proximity, AFP caps the worst-case response delay at 1 frame (100\,ms at 10\,fps) based on the $L_\text{AFP}^\text{worst}$ bound...`
   - `paper/main.tex` at line 800, 801:  
     `\item Close obstacle ($d < \tau_p$) yields minimum skip count ($s = 2$).`  
     `\item Far obstacle ($d \gg \tau_p$) yields elevated skip count ($s = 11$).`
2. **Ground Truth Data Files**:
   - `evaluation/results_ablation.json` contains the evaluation results of 24 clips for all strategies.
   - `evaluation/results_video_evaluation.json` contains real-world evaluation results.
   - `evaluation/results_phase7_benchmarks.json` contains synthetic benchmark latency and sensitivity metrics.
3. **Execution Commands**:
   - Running `python evaluation/verify_numbers.py` (proposing using `run_command` in `c:\blindaid`) timed out with:  
     `Permission prompt for action 'command' on target 'python evaluation/verify_numbers.py' timed out waiting for user response.`

---

## 2. Logic Chain
1. **Static 1/5 Metric Discrepancies**:
   - Sum of the 24 clip coverages for `"static_5"` in `results_ablation.json` yields `21.7148`.
   - Divided by 24: $\mu = 90.478\%$.
   - Sample standard deviation $s = 6.35\%$.
   - Margin of error for 95% Confidence Interval ($n=24$): $1.96 \times 6.35\% / \sqrt{24} = 2.54\%$.
   - CI limits: $90.5\% - 2.54\% = 87.96\%$, $90.5\% + 2.54\% = 93.04\%$. Range is `[88.0, 93.0]`.
   - CEI: $\mu_\text{cov} / (1 - \mu_\text{skip}) = 0.90478 / (1 - 0.7976) = 4.47$.
   - **Conclusion**: The reported values of Coverage = 91.3%, CEI = 4.52, and CI = `[88.7, 93.9]` in Table 7 are incorrect and contain a discrepancy of ~0.8%.
2. **AFP Stability-only Skip & CEI Discrepancies**:
   - Mean skip ratio for `"afp_stability_only"` in `results_ablation.json` is $91.98\%$.
   - Rounds to 92.0% (reported as 91.9%).
   - Mean coverage is $61.47\%$.
   - Actual CEI: $0.6147 / (1 - 0.9198) = 7.67$ (reported as 7.61).
3. **AFP Proximity-only CEI Discrepancy**:
   - Mean coverage for `"afp_proximity_only"` in `results_ablation.json` is $81.42\%$ and skip ratio is $86.25\%$.
   - Actual CEI: $0.8142 / (1 - 0.8625) = 5.92$ (reported as 5.88).
4. **Motion-Triggered Skip Coverage & CI Discrepancies**:
   - Mean coverage for `"motion_skip"` in `results_ablation.json` is $93.896\%$, which rounds to 93.9% (reported as 94.0%).
   - Range limits for 95% CI yield `[91.7, 96.1]` (reported as `[91.9, 96.2]`).
5. **Relative Processed Frame Reduction (Text Error)**:
   - Stated as 10.8% reduction from 13.8% down to 12.4%.
   - Mathematical formula: $(13.8\% - 12.4\%) / 13.8\% = 10.14\%$.
   - Exact JSON: $(13.75\% - 12.36\%) / 13.75\% = 10.11\%$.
   - **Conclusion**: Stating 10.8% is mathematically incorrect; it should be 10.1%.
6. **Physical Latency vs. Theoretical Mismatch**:
   - The paper claims under close proximity the worst-case latency is 1 frame (100 ms).
   - In code (`AdaptiveFrameProcessor` and `should_process`), when proximity hazard overrides skip count, $s_\text{min} = 2$. If the processor skips 2 frames, it processes on the 3rd frame.
   - **Conclusion**: The worst-case latency from a hazard appearing to being processed is physically 2 frames (200 ms at 10 fps). The formulation in Eq (502) $(s_\text{min}-1)/f$ outputs 100 ms, creating a discrepancy.
7. **Swapped Unit Test Inequalities**:
   - High depth means close proximity. Therefore close is $d > \tau_p$ and far is $d < \tau_p$.
   - The paper states $d < \tau_p$ for close and $d \gg \tau_p$ for far.
   - **Conclusion**: The inequalities are inverted in lines 800 and 801.

---

## 3. Caveats
- No shell command could be run successfully due to non-interactive environment timeout. Averages and confidence intervals were computed manually by parsing the raw JSON files.
- Wilcoxon signed-rank test values were not verified as they require non-parametric statistical code executions.

---

## 4. Conclusion
We confirmed that Table 6, per-model latencies, and total frames/times are numerically correct and perfectly align with the JSON data. However, there are multiple clear numerical discrepancies in Table 7 (especially Static 1/5 coverage, CEI, and CI), a mathematical error in the relative processed frame reduction, a latency bounds formulation mismatch with code execution behavior, and inverted inequality signs in unit test descriptions. 7 bibliography entries are unreferenced.

---

## 5. Verification Method
1. Inspect `results_ablation.json` and run the created `run_verify.py` script (when target execution is enabled) or run any script that loads `results_ablation.json` and computes:
   - `mean` and `stdev` for all strategies' coverage and skip ratio.
   - `CEI = mean(coverage) / (1 - mean(skip_ratio))`.
   - `margin = 1.96 * stdev(coverage) / sqrt(len(coverage))`.
2. Inspect `paper/main.tex` at lines 513, 750, 770, 776, 778, 779, 785, 800-801, and cross-reference them with the calculated metrics.
3. Inspect `paper/references.bib` and match each key with the occurrences of `\cite{...}` in `paper/main.tex` to confirm the 7 uncited references.
