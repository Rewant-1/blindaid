## 2026-06-15T18:25:58Z

<USER_REQUEST>
**Identity**: You are worker_m1_1, a versatile worker agent.
**Working Directory**: c:\blindaid\.agents\worker_m1_1
**Caller Agent ID**: 2e8f89ae-3798-4309-8720-162eaeff7710

**Objective**: Implement all the Track 6 manuscript revisions in `c:\blindaid\paper\main.tex` and `c:\blindaid\paper\references.bib` based on the recommendations from the three Explorers.
Specifically:
1. Reframe the paper's narrative and novelty claims in `main.tex`:
   - Title, Abstract, Introduction, and Conclusion must position AFP as a "practical latency-bounded temporal scheduling framework for resource-constrained edge CPUs" instead of a "novel scheduling algorithm".
   - Drop claims of a general-purpose "novel temporal scheduling algorithm".
2. In the Related Work section, rewrite Subsection II-C to integrate across-frame adaptive sampling literature (AdaFrame, SCSampler, OCSampler, SMART, FrameExit, AR-Net, LiteEval, AdaFuse) and discuss their resource overheads vs. AFP's edge CPU focus.
3. Remove the manual annotation study completely:
   - Section V.G and Table VI must be completely removed from `main.tex`.
   - Clean up any references to Section V.G / Table VI in the Introduction contribution 4 and Section V.H (which becomes the new Section V.G).
4. Supplement the CEI / Latency Metric Presentation in Section IV.C:
   - Formalize Accuracy-vs-Compute (with Always-On oracle target ceiling) and Latency-vs-Compute spaces.
   - Describe Always-On as the oracle target ceiling.
   - Explain the physical safety margin based on walking speed.
   - Detail the unbounded worst-case latency risk of motion-triggered and optical flow baselines ($L_{motion}^{worst} \to \infty$).
   - Standardize worst-case proximity latency to $L_{AFP}^{worst} = 100$ ms at 10 fps (1 frame delay) and physical safety margin to 0.96 m (reducing delay travel distance from 1.08 m under Static 1/10 to 0.12 m under AFP). Align these values in Abstract, Introduction, and Section V.
5. Resolve numerical inconsistencies in Table 7 (`tab:ablation`) and text of `main.tex`:
   - Update Table 7 with ground truth values from `evaluation/results_ablation.json` (Static 1/5 coverage = 91.3%, CEI = 4.52, CI = [88.7, 93.9]; Static 1/15 CEI = 8.18, CI = [52.9, 59.9]; Motion-Triggered coverage = 94.0%, skip ratio = 69.8%, CEI = 3.11, CI = [91.9, 96.2]; AFP Stab-only skip ratio = 91.9%, CEI = 7.61, CI = [56.9, 66.1]; AFP Prox-only CEI = 5.88, CI = [77.4, 85.5]).
   - Update fusion overhead percentage to 3.0% in Intro and Section III.
   - Update full pipeline synthetic benchmark skip ratio to 87% (or 86.3%) and amortized mean cost to 11.3 ms (or 11.25 ms).
   - Update relative processed frame reduction to 10.8%.
6. Update `references.bib` by appending the six missing BibTeX entries (SCSampler, OCSampler, SMART, AR-Net, LiteEval, AdaFuse) and citing them in Section II-C.

**Scope boundaries**:
- You must ONLY modify `paper/main.tex` and `paper/references.bib`. DO NOT modify any other files (no scripts, no JSON files).

**Input Information**:
- Manuscript: `c:\blindaid\paper\main.tex`
- Bibliography: `c:\blindaid\paper\references.bib`
- Explorer 1 Report: `c:\blindaid\.agents\explorer_m1_1\analysis.md`
- Explorer 2 Report: `c:\blindaid\.agents\explorer_m1_2\analysis.md`
- Explorer 3 Report: `c:\blindaid\.agents\explorer_m1_3\analysis.md`
- Ground Truth files under `c:\blindaid\evaluation/`

**MANDATORY INTEGRITY WARNING**:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

**Verification Requirements**:
- Execute `python evaluation/verify_numbers.py` (run it using `run_command` in `c:\blindaid`) and check that the printed numbers match your updated text.
- Include the command output in your handoff report.
- Verify that there are no LaTeX syntax errors in main.tex.

**Output Requirements**:
- Write a detailed report `changes.md` and a `handoff.md` in `c:\blindaid\.agents\worker_m1_1\`.
- Update `c:\blindaid\.agents\worker_m1_1\progress.md` with your progress and timestamps.
- Send a completion message to the parent (ID: 2e8f89ae-3798-4309-8720-162eaeff7710) when completed.
</USER_REQUEST>
