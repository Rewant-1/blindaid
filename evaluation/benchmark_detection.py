"""Phase 2 Audit: Benchmark ONNX YOLOv8-Nano object detection.

Measures REAL numbers:
- Inference latency (median, P95, P99)
- Memory footprint
- Detection count and classes on synthetic frames

Usage:
    python evaluation/benchmark_detection.py
"""
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


def generate_test_frames(n: int = 50) -> list:
    """Structured test frames with gradient backgrounds and shapes.
    For LATENCY only — not accuracy measurement.
    """
    rng = np.random.RandomState(42)
    frames = []

    for _ in range(n):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Gradient background
        for row in range(480):
            frame[row, :, :] = int(180 * row / 480) + 40

        # Random colored rectangles
        for _ in range(rng.randint(3, 8)):
            x1 = rng.randint(0, 500)
            y1 = rng.randint(0, 350)
            w = rng.randint(30, 150)
            h = rng.randint(30, 150)
            color = tuple(int(c) for c in rng.randint(0, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, -1)

        frames.append(frame)

    return frames


def percentile(data: list, p: float) -> float:
    idx = min(int(len(data) * p / 100), len(data) - 1)
    return sorted(data)[idx]


def benchmark_onnx_detector(frames: list, n_warmup: int = 5) -> dict:
    """Benchmark ONNX YOLOv8-Nano detection."""
    from blindaid.core.detector_onnx import ObjectDetectorONNX

    mem_before = get_memory_mb()

    detector = ObjectDetectorONNX()

    # Warmup
    load_start = time.perf_counter()
    for i in range(min(n_warmup, len(frames))):
        detector.detect(frames[i])
    load_time = time.perf_counter() - load_start

    mem_after = get_memory_mb()

    # Benchmark
    latencies = []
    total_detections = 0
    classes_seen = set()

    for frame in frames:
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        total_detections += len(detections)
        for d in detections:
            classes_seen.add(d.class_name)

    return {
        "model": "YOLOv8n ONNX",
        "load_time_s": load_time,
        "memory_mb": mem_after - mem_before,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "n_frames": len(frames),
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / len(frames),
        "unique_classes": sorted(classes_seen),
    }


def main():
    N_FRAMES = 50

    print("=" * 70)
    print("  PHASE 2 AUDIT: Object Detection — YOLOv8-Nano ONNX")
    print("=" * 70)
    print()
    print(f"  Test frames: {N_FRAMES} synthetic structured images (640x480)")
    print(f"  Purpose: LATENCY measurement (accuracy requires real footage)")
    print()

    frames = generate_test_frames(N_FRAMES)

    print("-" * 70)
    print("  Benchmarking: YOLOv8-Nano ONNX")
    print("-" * 70)

    results = benchmark_onnx_detector(frames)

    print(f"  Model:           {results['model']}")
    print(f"  Load time:       {results['load_time_s']:.2f} s (includes warmup)")
    print(f"  Memory delta:    {results['memory_mb']:.1f} MB")
    print(f"  Median latency:  {results['median_ms']:.1f} ms")
    print(f"  Mean latency:    {results['mean_ms']:.1f} ms  (±{results['stdev_ms']:.1f})")
    print(f"  P95 latency:     {results['p95_ms']:.1f} ms")
    print(f"  P99 latency:     {results['p99_ms']:.1f} ms")
    print(f"  Min / Max:       {results['min_ms']:.1f} / {results['max_ms']:.1f} ms")
    print(f"  Detections:      {results['total_detections']} total ({results['avg_detections_per_frame']:.1f}/frame)")
    print(f"  Classes seen:    {results['unique_classes']}")
    print()

    # Save results
    results_file = PROJECT_ROOT / "evaluation" / "results_phase2_detection.txt"
    with open(results_file, "w") as f:
        f.write("PHASE 2 DETECTION BENCHMARK RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames: {N_FRAMES}\n\n")
        f.write(f"YOLOv8n ONNX:\n")
        f.write(f"  Median: {results['median_ms']:.1f} ms\n")
        f.write(f"  Mean:   {results['mean_ms']:.1f} ms (±{results['stdev_ms']:.1f})\n")
        f.write(f"  P95:    {results['p95_ms']:.1f} ms\n")
        f.write(f"  P99:    {results['p99_ms']:.1f} ms\n")
        f.write(f"  Memory: {results['memory_mb']:.0f} MB\n")
        f.write(f"  Avg detections/frame: {results['avg_detections_per_frame']:.1f}\n\n")
    print(f"  Raw results saved to: {results_file}")


if __name__ == "__main__":
    main()
