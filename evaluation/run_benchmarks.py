"""Phase 7: Full-system benchmarking suite.

Experiments:
1. AFP vs Always-On vs Static-Skip baselines
2. Per-model latency breakdown
3. Full pipeline latency (depth + YOLO + AFP + template caption)
4. AFP parameter sensitivity (varying thresholds)

All measurements are REAL. No hardcoded stats.

Usage:
    python evaluation/run_benchmarks.py
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def percentile(data: list, p: float) -> float:
    idx = min(int(len(data) * p / 100), len(data) - 1)
    return sorted(data)[idx]


def generate_test_sequence(n: int = 100) -> list:
    """Generate a test sequence that simulates real-world scenarios.

    Creates frames with:
    - Gradual scene changes (smooth transitions)
    - Abrupt scene changes (sudden obstacles)
    - Stable periods (static scenes)
    """
    rng = np.random.RandomState(42)
    frames = []
    depths = []

    for i in range(n):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Background gradient that changes slowly
        phase = i / n
        bg_val = int(100 + 80 * np.sin(phase * 4 * np.pi))
        for row in range(480):
            frame[row, :, :] = min(255, bg_val + int(80 * row / 480))

        # Add random rectangles (simulating objects)
        n_objects = rng.randint(0, 5)
        for _ in range(n_objects):
            x1 = rng.randint(0, 500)
            y1 = rng.randint(0, 350)
            w = rng.randint(30, 150)
            h = rng.randint(30, 150)
            color = tuple(int(c) for c in rng.randint(40, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, -1)

        frames.append(frame)

        # Generate corresponding depth maps
        depth = np.full((480, 640), 0.3, dtype=np.float32)
        # Simulate obstacle approaching then retreating
        proximity = 0.2 + 0.6 * abs(np.sin(phase * 2 * np.pi))
        # Sometimes sudden close obstacle
        if i % 20 == 15:
            proximity = 0.95
        h_d, w_d = 480, 640
        depth[h_d // 3: 2 * h_d // 3, w_d // 4: 3 * w_d // 4] = proximity
        depths.append(depth)

    return frames, depths


# ============================================================
# Experiment 1: AFP vs Always-On vs Static-Skip
# ============================================================

def experiment_afp_comparison(frames: list, depths: list):
    """Compare AFP against baselines on the same frame sequence."""
    from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
    from blindaid.core.depth_onnx import DepthAnalyzerONNX

    depth_model = DepthAnalyzerONNX()
    # Warmup
    depth_model.compute_depth(frames[0])

    results = {}

    # --- Strategy 1: Always-On (process every frame) ---
    latencies_always = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        depth_model.compute_depth(frame)
        t1 = time.perf_counter()
        latencies_always.append((t1 - t0) * 1000)

    results["always_on"] = {
        "name": "Always-On",
        "frames_processed": len(frames),
        "frames_total": len(frames),
        "skip_ratio": 0.0,
        "median_ms": statistics.median(latencies_always),
        "total_cpu_ms": sum(latencies_always),
    }

    # --- Strategy 2: Static Skip (process every 10th frame) ---
    skip_interval = 10
    latencies_static = []
    frames_processed_static = 0
    for i, frame in enumerate(frames):
        if i % skip_interval == 0:
            t0 = time.perf_counter()
            depth_model.compute_depth(frame)
            t1 = time.perf_counter()
            latencies_static.append((t1 - t0) * 1000)
            frames_processed_static += 1

    results["static_skip"] = {
        "name": f"Static Skip (1/{skip_interval})",
        "frames_processed": frames_processed_static,
        "frames_total": len(frames),
        "skip_ratio": 1 - frames_processed_static / len(frames),
        "median_ms": statistics.median(latencies_static) if latencies_static else 0,
        "total_cpu_ms": sum(latencies_static),
    }

    # --- Strategy 3: AFP (adaptive) ---
    afp = AdaptiveFrameProcessor()
    latencies_afp = []
    for i, frame in enumerate(frames):
        should = afp.should_process("guardian", frame, depths[i] if i > 0 else None)
        if should:
            t0 = time.perf_counter()
            depth_result = depth_model.compute_depth(frame)
            t1 = time.perf_counter()
            latencies_afp.append((t1 - t0) * 1000)
            afp.update_depth(depth_result)

    afp_metrics = afp.get_metrics("guardian")

    results["afp"] = {
        "name": "Adaptive Frame Processing",
        "frames_processed": afp_metrics["frames_processed"],
        "frames_total": afp_metrics["frames_total"],
        "skip_ratio": afp_metrics["skip_ratio"],
        "median_ms": statistics.median(latencies_afp) if latencies_afp else 0,
        "total_cpu_ms": sum(latencies_afp),
        "avg_proximity": afp_metrics["avg_proximity"],
        "avg_stability": afp_metrics["avg_stability"],
    }

    return results


# ============================================================
# Experiment 2: Per-model latency breakdown
# ============================================================

def experiment_model_breakdown(frames: list):
    """Measure individual model latencies."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX
    from blindaid.core.ocr_onnx import OCREngineONNX
    from blindaid.core.template_caption import TemplateCaption

    N = min(30, len(frames))
    test_frames = frames[:N]

    models = {}

    # Depth
    depth = DepthAnalyzerONNX()
    depth.compute_depth(test_frames[0])  # warmup
    latencies = []
    for f in test_frames:
        t0 = time.perf_counter()
        depth.compute_depth(f)
        latencies.append((time.perf_counter() - t0) * 1000)
    models["depth_midas_small"] = {
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
    }

    # Detection
    det = ObjectDetectorONNX()
    det.detect(test_frames[0])
    latencies = []
    for f in test_frames:
        t0 = time.perf_counter()
        det.detect(f)
        latencies.append((time.perf_counter() - t0) * 1000)
    models["yolov8n"] = {
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
    }

    # Template caption (includes detection)
    tc = TemplateCaption()
    tc.generate_caption(test_frames[0])
    latencies = []
    for f in test_frames:
        t0 = time.perf_counter()
        tc.generate_caption(f)
        latencies.append((time.perf_counter() - t0) * 1000)
    models["template_caption"] = {
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
    }

    # OCR
    ocr = OCREngineONNX()
    ocr.read_text(test_frames[0])
    latencies = []
    for f in test_frames:
        t0 = time.perf_counter()
        ocr.read_text(f)
        latencies.append((time.perf_counter() - t0) * 1000)
    models["rapidocr"] = {
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
    }

    return models


# ============================================================
# Experiment 3: Full pipeline latency
# ============================================================

def experiment_full_pipeline(frames: list, depths: list):
    """Measure end-to-end guardian mode pipeline latency."""
    from blindaid.modes.guardian.guardian_mode import GuardianMode

    gm = GuardianMode()
    # Warmup — force model loading
    gm.process_frame(frames[0])
    gm.process_frame(frames[1])

    latencies = []

    for frame in frames:
        t0 = time.perf_counter()
        display, info, speech = gm.process_frame(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    # Split into processed (heavy) vs skipped (near-zero)
    sorted_lat = sorted(latencies)
    # Frames taking > 5ms are "processed", < 5ms are "skipped" (cached return)
    threshold = 5.0
    processed_latencies = [l for l in latencies if l > threshold]
    skipped_latencies = [l for l in latencies if l <= threshold]

    afp_metrics = gm.afp.get_metrics("guardian")

    return {
        "total_frames": len(frames),
        "all_median_ms": statistics.median(latencies),
        "all_mean_ms": statistics.mean(latencies),
        "all_p95_ms": percentile(latencies, 95),
        "processed_frames": len(processed_latencies),
        "processed_median_ms": statistics.median(processed_latencies) if processed_latencies else 0,
        "skipped_frames": len(skipped_latencies),
        "skipped_median_ms": statistics.median(skipped_latencies) if skipped_latencies else 0,
        "afp_skip_ratio": afp_metrics["skip_ratio"],
    }


# ============================================================
# Experiment 4: AFP parameter sensitivity
# ============================================================

def experiment_afp_sensitivity(frames: list, depths: list):
    """Sweep AFP parameters and measure impact."""
    from blindaid.core.adaptive_processor import AdaptiveFrameProcessor, MODE_CONFIG

    original_config = MODE_CONFIG["guardian"].copy()
    sweep_results = []

    for proximity_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for max_skip in [5, 10, 15, 20]:
            # Override config temporarily
            MODE_CONFIG["guardian"]["proximity_threshold"] = proximity_thresh
            MODE_CONFIG["guardian"]["max_skip"] = max_skip

            afp = AdaptiveFrameProcessor()
            for i, frame in enumerate(frames):
                afp.should_process("guardian", frame, depths[i] if i > 0 else None)

            metrics = afp.get_metrics("guardian")
            sweep_results.append({
                "proximity_threshold": proximity_thresh,
                "max_skip": max_skip,
                "skip_ratio": round(metrics["skip_ratio"], 3),
                "frames_processed": metrics["frames_processed"],
            })

    # Restore original config
    MODE_CONFIG["guardian"].update(original_config)

    return sweep_results


# ============================================================
# Main
# ============================================================

def main():
    N_FRAMES = 100

    print("=" * 70)
    print("  PHASE 7: Full Benchmarking Suite")
    print("=" * 70)
    print(f"  Generating {N_FRAMES} test frames with depth...")
    print()

    frames, depths = generate_test_sequence(N_FRAMES)
    all_results = {}

    # --- Experiment 1 ---
    print("-" * 70)
    print("  Experiment 1: AFP vs Baselines")
    print("-" * 70)

    comparison = experiment_afp_comparison(frames, depths)
    all_results["afp_comparison"] = comparison

    for strategy, data in comparison.items():
        print(f"  {data['name']}:")
        print(f"    Processed: {data['frames_processed']}/{data['frames_total']}")
        print(f"    Skip ratio: {data['skip_ratio']:.1%}")
        print(f"    Total CPU: {data['total_cpu_ms']:.0f} ms")
        print()

    # --- Experiment 2 ---
    print("-" * 70)
    print("  Experiment 2: Per-Model Latency Breakdown")
    print("-" * 70)

    breakdown = experiment_model_breakdown(frames)
    all_results["model_breakdown"] = breakdown

    print(f"  {'Model':<25} {'Median (ms)':<15} {'P95 (ms)':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    for model, data in breakdown.items():
        print(f"  {model:<25} {data['median_ms']:<15.1f} {data['p95_ms']:<15.1f}")
    print()

    # --- Experiment 3 ---
    print("-" * 70)
    print("  Experiment 3: Full Pipeline (Guardian Mode)")
    print("-" * 70)

    pipeline = experiment_full_pipeline(frames, depths)
    all_results["full_pipeline"] = pipeline

    print(f"  Overall median:        {pipeline['all_median_ms']:.1f} ms")
    print(f"  Processed frame:       {pipeline['processed_median_ms']:.1f} ms ({pipeline['processed_frames']} frames)")
    print(f"  Skipped frame:         {pipeline['skipped_median_ms']:.3f} ms ({pipeline['skipped_frames']} frames)")
    effective_fps = 1000 / pipeline["all_median_ms"] if pipeline["all_median_ms"] > 0 else 0
    print(f"  Effective throughput:   {effective_fps:.0f} FPS")
    print()

    # --- Experiment 4 ---
    print("-" * 70)
    print("  Experiment 4: AFP Parameter Sensitivity")
    print("-" * 70)

    sensitivity = experiment_afp_sensitivity(frames, depths)
    all_results["afp_sensitivity"] = sensitivity

    print(f"  {'Prox Thresh':<15} {'Max Skip':<12} {'Skip Ratio':<12} {'Processed':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    for entry in sensitivity:
        print(f"  {entry['proximity_threshold']:<15.1f} {entry['max_skip']:<12d} {entry['skip_ratio']:<12.3f} {entry['frames_processed']:<12d}")
    print()

    # Save results
    results_file = PROJECT_ROOT / "evaluation" / "results_phase7_benchmarks.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Also save human-readable summary
    summary_file = PROJECT_ROOT / "evaluation" / "results_phase7_summary.txt"
    with open(summary_file, "w") as f:
        f.write("PHASE 7 FULL BENCHMARK SUMMARY\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames: {N_FRAMES}\n\n")

        f.write("=== AFP vs Baselines ===\n")
        for strategy, data in comparison.items():
            f.write(f"  {data['name']}: {data['skip_ratio']:.1%} skip, {data['total_cpu_ms']:.0f}ms total CPU\n")

        f.write("\n=== Model Latencies ===\n")
        for model, data in breakdown.items():
            f.write(f"  {model}: {data['median_ms']:.1f}ms median\n")

        f.write(f"\n=== Full Pipeline ===\n")
        f.write(f"  Median: {pipeline['all_median_ms']:.1f}ms per frame\n")
        f.write(f"  Throughput: {effective_fps:.0f} FPS\n")

    print(f"  JSON results: {results_file}")
    print(f"  Summary: {summary_file}")
    print()
    print("=" * 70)
    print("  ALL BENCHMARKS COMPLETE — ALL NUMBERS ARE REAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
