# Handoff Report: CEI and Latency Metric Presentation (R2)

## 1. Observation
We examined the manuscript file `paper/main.tex` and located all sections discussing the Compute Efficiency Index (CEI), Always-On target, and safety/latency bounds:
- **Abstract (lines 48-50)**: Mentions: `"reducing the worst-case response latency by 700\,ms compared to static skip baselines, yielding a 2.1-frame average delay."`
- **Introduction (lines 65-68)**: Mentions: `"capping the worst-case response latency at a bounded 200\,ms for guaranteed safety."`
- **Introduction (line 72)**: Mentions: `"We demonstrate that AFP achieves a CEI of 6.30, which is over 2$\times$ more efficient than traditional motion-triggered and optical flow scheduling baselines."`
- **Section IV.C (lines 470-493)**:
  - Equation 1 (CEI): `\eta = \frac{\text{Coverage}}{1 - R_\text{skip}}`
  - Equation 2 (Static Skip Latency): `L_\text{static}^\text{worst} = \frac{N - 1}{f}`
  - Equation 4 (AFP Latency): `L_\text{AFP}^\text{worst} = \frac{s_\text{min} - 1}{f}`
  - Proximity Latency Claims: `"For $s_\text{min}=2$ and $f=10$\,fps, $L_\text{AFP}^\text{worst} = 100$\,ms, reducing the travel latency distance to just $0.12$\,meters, which provides an additional $0.96$-meter physical safety margin."`
- **Section V.A (lines 756-761)**: Mentions: `"Crucially, under close proximity, AFP caps the worst-case delay at 2 frames (200\,ms at 10\,fps) compared to Static Skip's worst-case of 9 frames (900\,ms), providing an additional 700\,ms (or $\sim$0.84\,m) safety margin during navigation."`
- **Section V.B (lines 796-799)**: Mentions: `"guaranteed worst-case proximity delay of 2 frames ($\sim$200\,ms)."`

We also examined the ground truth evaluation script `evaluation/verify_numbers.py` and JSON files (`evaluation/results_video_evaluation.json`, `evaluation/results_ablation.json`, and `evaluation/results_phase7_benchmarks.json`). They establish:
- Average coverage: `77.9%` (AFP Full) vs. `78.0%` (Static 1/10).
- Average skip ratio: `87.6%` (AFP Full) vs. `89.7%` (Static 1/10).
- CEI values: `6.30` (AFP Full) vs. `3.09` (Motion-Triggered) vs. `2.90` (Optical Flow) vs. `7.60` (Static 1/10).

---

## 2. Logic Chain
1. **Discrepancy Identification**: The manuscript contains an internal contradiction in the worst-case latency representation:
   - Section IV.C defines the safety bound as $L_\text{AFP}^\text{worst} = 100$\,ms (based on $(s_\text{min}-1)/f$ with $s_\text{min}=2$ and $f=10$\,fps) and asserts a $0.96$\,m physical safety margin.
   - Abstract, Introduction, Section V.A, and Section V.B assert $200$\,ms worst-case latency and a $700$\,ms ($0.84$\,m) safety margin.
2. **Standardization of $L_\text{AFP}^\text{worst} = 100$\,ms**: Based on the objective to frame the safety argument around the worst-case latency bound of $100$\,ms at 10\,fps, all occurrences of the latency and safety margin throughout the paper must be updated to $100$\,ms and $0.96$\,m (800\,ms reduction compared to Static 1/10's 900\,ms).
3. **Always-On Oracle Targeting**: Since "Always-On" represents the case where no frames are skipped, it serves as the oracle ceiling for accuracy (100% coverage by definition, since all evaluation is relative to Always-On detections). Positioning AFP's goal as approximating this oracle ceiling helps ground the trade-off.
4. **Accuracy-vs-Compute Space Formulation**: By adding the Always-On oracle ceiling ($\text{Coverage}=1.0$, $\text{Compute}=1.0$), we frame the CEI ($\eta = \text{Coverage}/\text{Compute}$) as the slope from the origin to the operating point in the Accuracy-vs-Compute plane.
5. **Latency-vs-Compute Space & Motion Baselines**:
   - Static skip baselines exhibit a hyperbolic trade-off: decreasing compute fraction $P = 1/N$ increases worst-case latency via $L_\text{static}^\text{worst} = \frac{1/P - 1}{f}$.
   - Motion and optical flow baselines only trigger when frame difference exceeds a threshold. For static obstacles or when the user stops moving, these baselines fail to trigger, resulting in an unbounded worst-case latency ($L_\text{motion}^\text{worst} \to \infty$).
   - AFP resolves this by using monocular depth to override skipping when proximity is high, capping the worst-case proximity latency to $100$\,ms.

---

## 3. Caveats
- No code modification was executed, in accordance with the read-only constraint.
- The updates assume a standard walking speed of $v=1.2$\,m/s, which is consistent with existing text in the manuscript.
- We assume that the project team wants the manuscript to be mathematically consistent, and therefore we updated all occurrences of the proximity delay/latency and physical margin to align with the $L_\text{AFP}^\text{worst} = 100$\,ms (1 frame delay) and 0.96\,m margin.

---

## 4. Conclusion
We have formulated the precise LaTeX updates needed to satisfy requirement R2. By replacing the inconsistent 200\,ms/0.84\,m values with 100\,ms/0.96\,m, supplementing Section IV.C with formal Accuracy-vs-Compute (with Always-On oracle ceiling) and Latency-vs-Compute formulations, and detailing the unbounded latency risk of motion-triggered baselines, the manuscript becomes mathematically rigorous, self-consistent, and robust against reviewer critique.

---

## 5. Verification Method
To verify these proposed changes:
1. Open `paper/main.tex` and check that the target content paragraphs match the line ranges specified in `analysis.md`.
2. Inspect the replacement LaTeX text block in Section IV.C (`\subsection{Compute Efficiency and Safety Latency Bounds}`) to verify that it correctly incorporates:
   - Always-On oracle target ceiling
   - Accuracy-vs-Compute (CEI) formulation
   - Latency-vs-Compute formulation (with hyperbolic static-skip latency)
   - Motion baseline unbounded latency ($L_\text{motion}^\text{worst} \to \infty$)
   - Bounded AFP safety latency ($L_\text{AFP}^\text{worst} = 100$\,ms)
   - Physical safety margin (0.96\,m vs. 1.08\,m vs. $\infty$)
3. Ensure that the abstract (line 49), introduction (lines 67, 72), and results (lines 759, 798) are all updated consistently to use 100\,ms worst-case latency (1 frame delay) and 0.96\,m safety margin.
