# Handoff Report — explorer_m1_3

## 1. Observation
- In `paper/main.tex` at line 779, the baseline `Static 1/5` is listed with coverage `90.5\%`, CEI `4.48`, and CI `[87.9, 93.1]`.
- In `paper/main.tex` at line 781, `Static 1/15` is listed with CEI `8.16`.
- In `paper/main.tex` at line 785, `Motion-Triggered Skip` is listed with coverage `93.9\%`, skip ratio `69.6\%`, and CEI `3.09`.
- In `paper/main.tex` at line 787, `AFP Stab-only` is listed with skip ratio `92.0\%` and CEI `7.67`.
- In `paper/main.tex` at line 788, `AFP Prox-only` is listed with CEI `5.92`.
- In `evaluation/results_ablation.json`, the raw data for 24 clips reveals the following true averages for each strategy:
  - `static_5` (Static 1/5): Mean coverage is `91.31\%`, mean skip ratio is `79.82\%`, and CEI is `4.52`. The calculated 95% confidence interval is `[88.7, 93.9]`.
  - `static_15` (Static 1/15): Mean coverage is `56.40\%`, mean skip ratio is `93.11\%`, and CEI is `8.18`. The calculated 95% confidence interval is `[52.9, 59.9]`.
  - `motion_skip` (Motion-Triggered Skip): Mean coverage is `94.02\%`, mean skip ratio is `69.75\%`, and CEI is `3.11`. The calculated 95% confidence interval is `[91.9, 96.2]`.
  - `afp_stability_only` (AFP Stab-only): Mean coverage is `61.47\%`, mean skip ratio is `91.93\%`, and CEI is `7.61`. The calculated 95% confidence interval is `[56.9, 66.1]`.
  - `afp_proximity_only` (AFP Prox-only): Mean coverage is `81.43\%`, mean skip ratio is `86.15\%`, and CEI is `5.88`. The calculated 95% confidence interval is `[77.4, 85.5]`.
- In `paper/main.tex` at line 597, under the full pipeline performance over 100 frames, it lists `AFP skip ratio: 81\%`.
- In `evaluation/results_phase7_benchmarks.json` under `full_pipeline`, the actual values are:
  - `processed_frames`: 13
  - `skipped_frames`: 87
  - `afp_skip_ratio`: `0.862745` (~86.3%)
  - `all_mean_ms` (amortized cost): `11.25` ms (~11.3 ms).
- In `paper/main.tex` at lines 73 and 242, the fusion overhead is listed as `3.2\%`. However, at line 595, it is listed as `3.0\%` of total (`2.6\,ms` of `87.4\,ms` processed frame latency is $2.6 / 87.4 \approx 2.97\%$ which rounds to `3.0\%`).
- In `paper/references.bib`, we observed that:
  - `wu2019adaframe` (AdaFrame) is defined on lines 162-168.
  - `ghodrati2021frameexit` (FrameExit) is defined on lines 170-176.
  - The other 6 required papers (SCSampler, OCSampler, SMART, AR-Net, LiteEval, AdaFuse) are missing.

## 2. Logic Chain
1. By reading `paper/main.tex` and verifying against `evaluation/results_ablation.json` (which contains the raw per-clip results), we identified several mismatches in Table 7 (`tab:ablation`). Summing the clip coverages/skip ratios for each strategy and dividing by 24 yielded the true averages, confirming discrepancies in `Static 1/5`, `Static 1/15`, `Motion-Triggered`, `AFP Stab-only`, and `AFP Prox-only`.
2. Comparing the synthetic benchmark timings in Section V-B against `evaluation/results_phase7_benchmarks.json` revealed that the full pipeline benchmark processed 13 frames and skipped 87 frames (an 87% or 86.3% skip ratio, resulting in an amortized mean latency of 11.25 ms). The manuscript text incorrectly cited the standalone 81% skip ratio for the full pipeline.
3. Checking the relative fusion overhead (2.6 ms relative to 87.4 ms processed frame latency) shows that it is mathematically 2.97% ($\approx$3.0%), making the 3.2% figure in the Introduction and Section III internally inconsistent.
4. Parsing `paper/references.bib` confirmed that AdaFrame and FrameExit are present, while SCSampler, OCSampler, SMART, AR-Net, LiteEval, and AdaFuse must be added.

## 3. Caveats
- Since command execution timed out during our run, all calculations were performed by parsing the JSON files and conducting the math manually. The 95% confidence intervals were estimated using the standard deviation of coverages across the 24 clips in `results_ablation.json`.
- We assumed that the manual human annotation study in Table 5 (`tab:manual_annotations`) does not need numerical adjustment as its ground truth matches the strict evaluation parameters, but this study was not validated against a separate automated JSON since no such output exists in the raw benchmarking folders.

## 4. Conclusion
The manuscript has several minor numerical and textual inconsistencies in Table 7, the full pipeline performance description, and the fusion overhead percentage. To resolve R3, Table 7 and the corresponding text should be updated using the precise ground truth values from the JSON files. To resolve R4, the six missing bibliography entries should be appended to `paper/references.bib`.

## 5. Verification Method
- **Numerical Verification**: Run the verification script `c:\blindaid\evaluation\verify_numbers.py` using Python. The output prints the exact averages and confidence intervals for Table 6 and Table 7, which should match our corrected LaTeX table rows.
- **Bibliography Verification**: Ensure that LaTeX compilation is successful after appending the six drafted BibTeX entries and citing them in `paper/main.tex`.
