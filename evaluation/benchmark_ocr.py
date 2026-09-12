"""Phase 3 Audit: Benchmark ONNX OCR (RapidOCR).

Measures REAL numbers on synthetic text images.

Usage:
    python evaluation/benchmark_ocr.py
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


def generate_text_frames(n: int = 30) -> list:
    """Generate frames with rendered text for OCR testing.

    These have REAL text drawn on them so OCR actually has something to read.
    This tests both latency AND basic OCR functionality.
    """
    rng = np.random.RandomState(42)
    frames = []

    sample_texts = [
        "Hello World", "Exit Sign", "Room 204", "Caution Wet Floor",
        "Open 9am to 5pm", "No Smoking", "Fire Exit", "Restroom",
        "Push to Open", "Do Not Enter", "Elevator", "Stairs",
        "Welcome", "Information Desk", "Emergency Phone",
        "Keep Right", "Slow Down", "Stop", "One Way",
        "Parking Level 3", "Gate A12", "Platform 2",
        "Coffee Shop", "Pharmacy", "Library", "Bus Stop",
        "Speed Limit 30", "No Parking", "Taxi Stand",
        "Hospital Entrance",
    ]

    for i in range(n):
        # White/light background
        bg_val = rng.randint(180, 255)
        frame = np.full((480, 640, 3), bg_val, dtype=np.uint8)

        # Draw 1-3 text strings at random positions
        n_texts = rng.randint(1, 4)
        for j in range(n_texts):
            text = sample_texts[(i * 3 + j) % len(sample_texts)]
            x = rng.randint(20, 400)
            y = rng.randint(40, 440)
            font_scale = rng.uniform(0.8, 2.0)
            thickness = rng.randint(1, 3)

            # Dark text on light background
            color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

        frames.append(frame)

    return frames


def percentile(data: list, p: float) -> float:
    idx = min(int(len(data) * p / 100), len(data) - 1)
    return sorted(data)[idx]


def benchmark_onnx_ocr(frames: list, n_warmup: int = 3) -> dict:
    """Benchmark RapidOCR ONNX."""
    from blindaid.core.ocr_onnx import OCREngineONNX

    mem_before = get_memory_mb()

    ocr = OCREngineONNX()

    # Warmup
    load_start = time.perf_counter()
    for i in range(min(n_warmup, len(frames))):
        ocr.read_text(frames[i])
    load_time = time.perf_counter() - load_start

    mem_after = get_memory_mb()

    # Benchmark
    latencies = []
    total_texts = 0
    sample_outputs = []

    for frame in frames:
        t0 = time.perf_counter()
        results = ocr.read_text(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        total_texts += len(results)

        if len(sample_outputs) < 5 and results:
            sample_outputs.append([r.text for r in results])

    return {
        "model": "RapidOCR ONNX (PP-OCR models)",
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
        "total_texts_detected": total_texts,
        "avg_texts_per_frame": total_texts / len(frames),
        "sample_outputs": sample_outputs,
    }


def main():
    N_FRAMES = 30  # Fewer frames since OCR is slower

    print("=" * 70)
    print("  PHASE 3 AUDIT: OCR — RapidOCR ONNX")
    print("=" * 70)
    print()
    print(f"  Test frames: {N_FRAMES} synthetic text images (640x480)")
    print(f"  Purpose: LATENCY + basic functionality verification")
    print()

    frames = generate_text_frames(N_FRAMES)

    print("-" * 70)
    print("  Benchmarking: RapidOCR ONNX")
    print("-" * 70)

    results = benchmark_onnx_ocr(frames)

    print(f"  Model:           {results['model']}")
    print(f"  Load time:       {results['load_time_s']:.2f} s (includes warmup)")
    print(f"  Memory delta:    {results['memory_mb']:.1f} MB")
    print(f"  Median latency:  {results['median_ms']:.1f} ms")
    print(f"  Mean latency:    {results['mean_ms']:.1f} ms  (±{results['stdev_ms']:.1f})")
    print(f"  P95 latency:     {results['p95_ms']:.1f} ms")
    print(f"  P99 latency:     {results['p99_ms']:.1f} ms")
    print(f"  Min / Max:       {results['min_ms']:.1f} / {results['max_ms']:.1f} ms")
    print(f"  Texts detected:  {results['total_texts_detected']} total ({results['avg_texts_per_frame']:.1f}/frame)")
    print()

    if results["sample_outputs"]:
        print("  Sample OCR outputs:")
        for i, texts in enumerate(results["sample_outputs"][:3]):
            print(f"    Frame {i+1}: {texts}")
    print()

    # Save results
    results_file = PROJECT_ROOT / "evaluation" / "results_phase3_ocr.txt"
    with open(results_file, "w") as f:
        f.write("PHASE 3 OCR BENCHMARK RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames: {N_FRAMES}\n\n")
        f.write(f"RapidOCR ONNX:\n")
        f.write(f"  Median: {results['median_ms']:.1f} ms\n")
        f.write(f"  Mean:   {results['mean_ms']:.1f} ms (±{results['stdev_ms']:.1f})\n")
        f.write(f"  P95:    {results['p95_ms']:.1f} ms\n")
        f.write(f"  P99:    {results['p99_ms']:.1f} ms\n")
        f.write(f"  Memory: {results['memory_mb']:.0f} MB\n")
        f.write(f"  Avg texts/frame: {results['avg_texts_per_frame']:.1f}\n\n")
        if results["sample_outputs"]:
            f.write("Sample outputs:\n")
            for i, texts in enumerate(results["sample_outputs"]):
                f.write(f"  Frame {i+1}: {texts}\n")
    print(f"  Raw results saved to: {results_file}")


if __name__ == "__main__":
    main()
