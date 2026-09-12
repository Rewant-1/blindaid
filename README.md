<div align="center">

# Adaptive Frame Processing (AFP)
### Latency-Bounded Temporal Scheduling for Multi-Modal Edge Inference

[![IEEE INDISCON 2026](https://img.shields.io/badge/IEEE_INDISCON_2026-Accepted-success?style=for-the-badge&logo=ieee&logoColor=white)](https://indiscon2026.org/)
[![Track](https://img.shields.io/badge/Track_6-Signal_Processing%2C_Computing_%26_Data_Science-blue?style=for-the-badge)](#)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Optimized-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

*Official research codebase and reproduction suite for the IEEE INDISCON 2026 paper.*

</div>

---

## 📌 Abstract

Real-time multi-modal video inference on edge CPUs must execute depth estimation and object detection under tight compute budgets. Static frame subsampling reduces average load but imposes fixed worst-case response delays regardless of scene content. **Adaptive Frame Processing (AFP)** is a temporal scheduling framework that dynamically adjusts per-frame processing rates via a one-frame depth feedback loop, concentrating compute on proximity-critical frames while skipping redundant content. 

AFP resolves the fundamental circular dependency between scheduling decisions and depth computation without requiring auxiliary policy networks or training overhead. On 24 real-world video sequences (7,683 frames) with CPU-only ONNX inference, AFP achieves an **87.6% frame skip ratio** with detection coverage statistically equivalent to static 1/10 subsampling (TOST equivalence, $p < 0.001$, $\delta = \pm 5$\,pp), while bounding worst-case proximity response latency at **200 ms** (compared to 900 ms for static and unbounded for motion-triggered baselines). The amortized per-frame cost is **10.8 ms** on a consumer-grade CPU.

---

## 🚀 Key Highlights & Contributions

- **Circular Dependency Resolution**: Resolves the "chicken-and-egg" dilemma—scheduling depends on depth, but computing depth is the operation being scheduled—by using a $t-1$ depth map feedback loop ($D_{t-1}$) coupled with a 64-bin histogram correlation stability metric ($\sigma_t$).
- **Formal Latency Bounds**: Worst-case response delay under close proximity is bounded to $L_\text{AFP}^\text{worst} = s_\text{min}/f = 200$\,ms at 10 fps (4.5× reduction vs. static 1/10 at 900 ms; travel displacement reduced from 1.08 m to 0.24 m at pedestrian speed 1.2 m/s).
- **Rigorous Statistical Validation**: Two One-Sided Tests (TOST) equivalence testing confirms detection coverage parity with static 1/10 subsampling within $\delta = \pm 5$\,pp ($p < 0.001$, mean difference $-0.07$\,pp, 95% CI: $[-2.21, 2.07]$\,pp, Cohen's $d = -0.014$).
- **Zero-Shot & CPU-Only**: Training-free heuristic with negligible scheduling overhead (2.6 ms), executing entirely on commodity CPUs via ONNX Runtime without requiring GPUs or dedicated accelerators.

---

## 📊 Evaluation Summary

### 1. Scheduling Strategy Comparison (Table III in Paper)
Evaluated across all 24 real-world sequences (7,683 frames, 799.0s). Coverage is measured against the Always-On detector (oracle ceiling) within a $\pm$5-frame ($\pm$0.5s) window.

| Strategy | Detection Coverage (%) | Skip Ratio (%) | Compute Efficiency Index (CEI) | Worst-Case Latency ($L^\text{worst}$) |
|:---|:---:|:---:|:---:|:---:|
| **Always-On (Oracle)** | 100.0 | 0.0 | 1.00 | 0 ms |
| **Static 1/3** | 96.4 | 66.5 | 2.88 | 200 ms |
| **Static 1/5** | 90.5 | 79.8 | 4.48 | 400 ms |
| **Static 1/10** | 78.0 | 89.7 | 7.60 | 900 ms |
| **Static 1/15** | 56.4 | 93.1 | 8.16 | 1400 ms |
| **Random Skip** | 78.1 | 82.6 | 4.50 | Unbounded |
| **Optical Flow** | 96.3 | 66.8 | 2.90 | Unbounded |
| **Motion-Triggered** | 93.9 | 69.6 | 3.09 | Unbounded ($\infty$) |
| **AFP Stability-only** | 61.5 | 92.0 | 7.67 | Unbounded |
| **AFP Proximity-only**| 81.4 | 86.2 | 5.92 | 200 ms |
| **AFP Full (Proposed)**| **77.9** | **87.6** | **6.30** | **200 ms** |

### 2. Worst-Case Response Delay & Kinematics (Table II in Paper)
Measured at capture frame rate $f = 10$\,fps and typical pedestrian speed of 1.2 m/s:

$$\begin{aligned}
L_\text{static}^\text{worst} &= \frac{N - 1}{f} = \frac{10 - 1}{10} = 900\,\text{ms} \quad \implies \text{Travel} = 1.08\,\text{m} \\
L_\text{AFP}^\text{worst} &= \frac{s_\text{min}}{f} = \frac{2}{10} = 200\,\text{ms} \quad \implies \text{Travel} = 0.24\,\text{m}
\end{aligned}$$

---

## 🛠️ Repository Architecture

```text
blindaid/
├── paper/                          # Research paper manuscript (accepted at INDISCON 2026)
│   ├── main.tex                    # IEEE conference LaTeX manuscript
│   └── references.bib              # BibTeX citation database
├── blindaid/                       # Production BlindAid system package
│   ├── app.py                      # Application bootstrap
│   ├── controller.py               # Main mode switching and camera controller
│   ├── core/                       # Core perception and scheduling modules
│   │   ├── adaptive_processor.py   # AFP core scheduling engine (Algorithm 1)
│   │   ├── depth_onnx.py           # MiDaS-Small ONNX depth analyzer
│   │   ├── detector_onnx.py        # YOLOv8-Nano ONNX detector
│   │   ├── depth_fusion.py         # 3D spatial fusion (BBox + depth pooling)
│   │   ├── scene_state.py          # Temporal IoU tracking & alert deduplication
│   │   └── template_caption.py     # Lightweight rule-based scene captioning
│   └── modes/                      # Assistive operational modes
│       ├── guardian/               # Obstacle navigation mode (uses AFP)
│       ├── ocr/                    # Text reading mode
│       └── people/                 # Facial recognition mode
├── clips/                          # Evaluation dataset (24 real-world video sequences)
│   ├── clip1.mp4 ... clip24.mp4    # 7,683 frames, 799.0 seconds total duration
├── evaluation/                     # Reproducibility suite and evaluation logs
│   ├── results_video_evaluation.json  # Table IV ground truth per-clip results
│   ├── results_ablation.json          # Table III ground truth ablation results
│   ├── results_phase7_benchmarks.json # Tables I & V benchmark & latency results
│   ├── verify_numbers.py           # 1-command verification of all numbers in paper
│   ├── compute_stats.py            # TOST statistical equivalence analysis
│   ├── analyze_results.py          # Quick summary of video evaluation
│   ├── evaluate_video.py           # Video evaluation runner (Always-On vs SS vs AFP)
│   ├── run_ablation.py             # Ablation experiment runner
│   ├── run_benchmarks.py           # System benchmarking runner
│   ├── test_adaptive.py            # Unit tests for AFP algorithm and bounds
│   └── test_scene_intelligence.py  # Unit tests for depth-YOLO spatial fusion
├── resources/                      # Model weights and assets
│   ├── models/                     # midas_small.onnx, yolov8n.onnx, yolov8n.pt
│   └── known_faces/                # Face database for people mode
├── scripts/
│   └── download_onnx_models.py     # Automated model download script
├── pyproject.toml                  # Package configuration
└── requirements.txt                # Python dependencies
```

---

## ⚡ Quickstart & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Rewant-1/blindaid.git
cd blindaid

# Install dependencies
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

### 2. Download Model Weights
If ONNX models are not already in `resources/models/`, download them automatically:
```bash
python scripts/download_onnx_models.py
```

### 3. Verify Paper Numbers (One Command)
Verify all numbers cited across the paper (Tables I, II, III, IV, and V) against the recorded evaluation logs:
```bash
python evaluation/verify_numbers.py
```

### 4. Run Statistical Equivalence (TOST)
Execute the Two One-Sided Tests (TOST) procedure, effect size calculation (Cohen's $d$), and 95% confidence interval:
```bash
python evaluation/compute_stats.py
```

### 5. Run Unit & Component Tests
```bash
# Test AFP skip clamping, safety overrides, and state accumulation
python evaluation/test_adaptive.py

# Test spatial depth-detection fusion and speech deduplication
python evaluation/test_scene_intelligence.py
```

### 6. Reproduce Experiments from Raw Video Clips
Run the evaluation suite over all 24 video clips in `clips/`:
```bash
# Evaluate Always-On, Static-Skip 1/10, and AFP Full on video sequences
python evaluation/evaluate_video.py --dir clips/

# Run the 10-strategy ablation study
python evaluation/run_ablation.py

# Run latency and sensitivity benchmarks
python evaluation/run_benchmarks.py
```

### 7. Run the Live BlindAid Application
Connect a webcam and launch the live interactive application:
```bash
python -m blindaid
```
- **`1`**: Guardian Mode (Walking / Obstacle Avoidance with AFP)
- **`2`**: Reading Mode (OCR)
- **`3`**: People Mode (Facial Recognition)
- **`5`**: Scene Captioning
- **`q`**: Quit

---

## 📖 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@inproceedings{afp2026indiscon,
  title     = {Adaptive Frame Processing: Latency-Bounded Temporal Scheduling for Multi-Modal Edge Inference},
  author    = {Bhriguvanshi, Rewant and Sinha, Shubhika and Singh, Satyam},
  booktitle = {Proceedings of the 7th IEEE India Council International Subsections Conference (INDISCON)},
  year      = {2026},
  organization = {IEEE}
}
```
