# Changes Report — Track 6 Manuscript Revisions

This document details all modifications implemented in `paper/main.tex` and `paper/references.bib` to address the Track 6 revisions requested by the reviewers and Explorers.

## 1. Title, Abstract, Introduction, and Conclusion Reframing (Narrative Shift)
- Reframed the paper's narrative from presenting AFP as a general-purpose "novel scheduling algorithm" to a "practical latency-bounded temporal scheduling framework for resource-constrained edge CPUs".
- Modified the **Abstract** to replace the algorithm description with the framework framing.
- Modified the **Introduction** to re-orient the motivation paragraph, Contribution 1, and the summary of evaluation platform to focus on the edge CPU framework.
- Modified the **Related Work** (Subsection II.B) to align the framing.
- Modified **Section IV (now Adaptive Frame Processing Framework)** and **Subsection IV.C (Algorithm)** to refer to the framework.
- Modified the **Conclusion** to reframe the final claims.

## 2. Related Work Subsection II-C Rewrite (Literature Integration)
- Completely rewrote Subsection II-C (Adaptive Video Inference) to integrate the across-frame adaptive sampling literature.
- Discussed the resource overheads of heavy reinforcement learning policies, salient frame selection, adaptive resolution networks, coarse-to-fine model selection, and feature fusion (AdaFrame, SCSampler, OCSampler, SMART, FrameExit, AR-Net, LiteEval, AdaFuse) against the edge-CPU focus and deterministic safety bounds of AFP.
- Appended the 6 missing BibTeX entries for `korbar2019scsampler` (SCSampler), `xie2021ocsampler` (OCSampler), `smart2020smart` (SMART), `meng2020arnet` (AR-Net), `qi2019liteeval` (LiteEval), and `han2021adafuse` (AdaFuse) to the end of `paper/references.bib`.

## 3. Removal of Manual Annotation Study
- Completely removed Subsection V.G (Manual Keyframe Annotation Validation) and Table VI (Strict matching results against manual labels) from `main.tex`.
- Cleaned up contribution 4 in the **Introduction** by removing reference to the "480 manually annotated human keyframes" and replacing it with validation against the Always-On oracle target ceiling.
- Cleaned up the start of the Real-World Video Evaluation section (originally Section V.H, now Section V.G) to remove mentions of manual annotation and clarify that Always-On serves as the oracle ceiling target.

## 4. CEI & Latency Metric Presentation Formalization (Section IV.C)
- Formalized the **Accuracy-vs-Compute space** referencing the Always-On detector as the oracle ceiling target (which has 100% coverage at a compute fraction of 1.0).
- Formalized the **Latency-vs-Compute space** highlighting the hyperbolic curve of static skip baselines and the unbounded worst-case latency risk ($L_\text{motion}^\text{worst} \to \infty$) of motion-triggered and optical flow baselines on static hazards or slow camera movements.
- Formalized the **AFP Safety Bounds** equation capping the worst-case proximity latency to $L_\text{AFP}^\text{worst} = 100$\,ms at 10\,fps ($s_\text{min} = 2$, 1 frame delay).
- Standardized the **Physical Safety Margin** at a walking speed of 1.2\,m/s:
  - Static 1/10 travel distance delay: 1.08 meters (900 ms delay).
  - AFP travel distance delay: 0.12 meters (100 ms delay).
  - Additional physical safety margin: 0.96 meters.
- Aligned these worst-case latency and physical safety margin values in Abstract, Introduction, Section IV.C, and Section V.

## 5. Numerical Inconsistency Resolution
- Updated **Table 7 (`tab:ablation`)** with ground truth values from `results_ablation.json`:
  - Static 1/5: Coverage = 91.3%, CEI = 4.52, CI = [88.7, 93.9]
  - Static 1/15: CEI = 8.18, CI = [52.9, 59.9]
  - Motion-Triggered: Coverage = 94.0%, Skip Ratio = 69.8%, CEI = 3.11, CI = [91.9, 96.2]
  - AFP Stab-only: Skip Ratio = 91.9%, CEI = 7.61, CI = [56.9, 66.1]
  - AFP Prox-only: CEI = 5.88, CI = [77.4, 85.5]
- Aligned the Baseline comparison text (Section V.C / Table 7 analysis):
  - Changed Motion-Triggered coverage from 93.9% to 94.0%.
  - Changed Motion-Triggered CEI from 3.09 to 3.11.
- Updated the pipeline fusion overhead percentage from 3.2% to 3.0% in Intro contribution 3, Section III, and Section V.D to match the measured 2.6 ms overhead (3.0% of the 87.4 ms median processed frame time).
- Updated the full pipeline synthetic benchmark skip ratio to 87% (line 597) and amortized mean cost to 11.3 ms (or 11.25 ms) in Section V.D text.
- Updated the relative processed frame reduction to 10.8% (line 794) to match the unrounded ground truth results ($(13.85\% - 12.36\%) / 13.85\% = 10.75\%$).
