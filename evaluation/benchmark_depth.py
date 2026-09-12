"""Phase 1 Audit: Benchmark PyTorch vs ONNX depth estimation.

Measures REAL numbers:
- Inference latency (median, P95, P99) via time.perf_counter()
- Memory footprint via psutil
- Output comparison (structural similarity)

Usage:
    python evaluation/benchmark_depth.py

NOTE: Uses synthetic structured frames for LATENCY measurement only.
      Accuracy evaluation requires real-world footage (Phase 7).
"""
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_memory_mb() -> float:
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def generate_test_frames(n: int = 50) -> list:
    """Generate structured test frames (640x480 BGR).

    These have gradients and shapes to give depth models something
    to work with. NOT for accuracy claims — only for latency timing.
    """
    rng = np.random.RandomState(42)  # Reproducible
    frames = []

    for _ in range(n):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Gradient background (simulates floor/wall depth gradient)
        for row in range(480):
            val = int(255 * row / 480)
            frame[row, :, :] = val

        # Random rectangles (simulates objects at various depths)
        for _ in range(rng.randint(2, 6)):
            x1 = rng.randint(0, 400)
            y1 = rng.randint(0, 300)
            x2 = x1 + rng.randint(40, 200)
            y2 = y1 + rng.randint(40, 200)
            color = tuple(int(c) for c in rng.randint(50, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        frames.append(frame)

    return frames


def percentile(data: list, p: float) -> float:
    """Compute percentile from sorted data."""
    idx = int(len(data) * p / 100)
    idx = min(idx, len(data) - 1)
    return sorted(data)[idx]


def benchmark_onnx(frames: list, n_warmup: int = 5) -> dict:
    """Benchmark ONNX MiDaS-Small."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX

    mem_before = get_memory_mb()

    analyzer = DepthAnalyzerONNX()

    # Warmup (includes model loading)
    load_start = time.perf_counter()
    for i in range(min(n_warmup, len(frames))):
        analyzer.compute_depth(frames[i])
    load_time = time.perf_counter() - load_start

    mem_after_load = get_memory_mb()

    # Benchmark
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        depth = analyzer.compute_depth(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    # Get a sample output for comparison
    sample_depth = analyzer.compute_depth(frames[0])

    return {
        "model": "MiDaS-Small ONNX (24MB)",
        "load_time_s": load_time,
        "memory_mb": mem_after_load - mem_before,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "n_frames": len(frames),
        "sample_output": sample_depth,
    }


def benchmark_pytorch(frames: list, n_warmup: int = 5) -> dict:
    """Benchmark PyTorch DPT-Hybrid-MiDaS."""
    from blindaid.core.depth import DepthAnalyzer

    mem_before = get_memory_mb()

    analyzer = DepthAnalyzer()

    # Warmup (includes model loading + download if first time)
    load_start = time.perf_counter()
    for i in range(min(n_warmup, len(frames))):
        analyzer.compute_depth(frames[i])
    load_time = time.perf_counter() - load_start

    mem_after_load = get_memory_mb()

    # Benchmark
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        depth = analyzer.compute_depth(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    sample_depth = analyzer.compute_depth(frames[0])

    return {
        "model": "DPT-Hybrid-MiDaS PyTorch (470MB)",
        "load_time_s": load_time,
        "memory_mb": mem_after_load - mem_before,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "n_frames": len(frames),
        "sample_output": sample_depth,
    }


def compare_outputs(onnx_depth: np.ndarray, pytorch_depth: np.ndarray) -> dict:
    """Compare ONNX and PyTorch depth maps to verify quality."""
    # Absolute difference
    abs_diff = np.abs(onnx_depth.astype(np.float64) - pytorch_depth.astype(np.float64))

    # Correlation (how similar are the relative depth orderings)
    flat_onnx = onnx_depth.flatten()
    flat_pytorch = pytorch_depth.flatten()
    correlation = np.corrcoef(flat_onnx, flat_pytorch)[0, 1]

    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "correlation": float(correlation),
        "onnx_range": f"[{float(onnx_depth.min()):.4f}, {float(onnx_depth.max()):.4f}]",
        "pytorch_range": f"[{float(pytorch_depth.min()):.4f}, {float(pytorch_depth.max()):.4f}]",
    }


def print_results(label: str, results: dict):
    """Pretty-print benchmark results."""
    print(f"  Model:           {results['model']}")
    print(f"  Load time:       {results['load_time_s']:.2f} s (includes warmup)")
    print(f"  Memory delta:    {results['memory_mb']:.1f} MB")
    print(f"  Median latency:  {results['median_ms']:.1f} ms")
    print(f"  Mean latency:    {results['mean_ms']:.1f} ms  (±{results['stdev_ms']:.1f})")
    print(f"  P95 latency:     {results['p95_ms']:.1f} ms")
    print(f"  P99 latency:     {results['p99_ms']:.1f} ms")
    print(f"  Min / Max:       {results['min_ms']:.1f} / {results['max_ms']:.1f} ms")
    print(f"  Frames tested:   {results['n_frames']}")


def main():
    N_FRAMES = 50

    print("=" * 70)
    print("  PHASE 1 AUDIT: Depth Estimation — ONNX vs PyTorch")
    print("=" * 70)
    print()
    print(f"  Test frames: {N_FRAMES} synthetic structured images (640x480)")
    print(f"  Purpose: LATENCY measurement only (not accuracy)")
    print()

    frames = generate_test_frames(N_FRAMES)

    # --- ONNX ---
    print("-" * 70)
    print("  ONNX: MiDaS-Small")
    print("-" * 70)
    onnx_results = benchmark_onnx(frames)
    print_results("ONNX", onnx_results)
    print()

    # --- PyTorch ---
    pytorch_results = None
    print("-" * 70)
    print("  PyTorch: DPT-Hybrid-MiDaS")
    print("-" * 70)
    try:
        pytorch_results = benchmark_pytorch(frames)
        print_results("PyTorch", pytorch_results)
    except Exception as e:
        print(f"  [SKIPPED] PyTorch benchmark failed: {e}")
        print(f"  This is expected if torch/transformers aren't installed.")
        print(f"  The research branch uses ONNX only — PyTorch comparison is optional.")
    print()

    # --- Output comparison (only if both ran) ---
    if pytorch_results is not None:
        print("-" * 70)
        print("  Output Comparison (same input frame)")
        print("-" * 70)
        comparison = compare_outputs(
            onnx_results["sample_output"],
            pytorch_results["sample_output"],
        )
        print(f"  Mean abs diff:   {comparison['mean_abs_diff']:.4f}")
        print(f"  Max abs diff:    {comparison['max_abs_diff']:.4f}")
        print(f"  Correlation:     {comparison['correlation']:.4f}")
        print(f"  ONNX range:      {comparison['onnx_range']}")
        print(f"  PyTorch range:   {comparison['pytorch_range']}")
        print()

    # --- Summary ---
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if pytorch_results is not None:
        speedup = pytorch_results["median_ms"] / onnx_results["median_ms"]
        mem_reduction = pytorch_results["memory_mb"] / max(onnx_results["memory_mb"], 0.1)
        print(f"  Latency speedup:     {speedup:.2f}x  (ONNX {onnx_results['median_ms']:.1f}ms vs PyTorch {pytorch_results['median_ms']:.1f}ms)")
        print(f"  Memory reduction:    {mem_reduction:.1f}x  (ONNX {onnx_results['memory_mb']:.0f}MB vs PyTorch {pytorch_results['memory_mb']:.0f}MB)")
    else:
        print(f"  ONNX MiDaS-Small:")
        print(f"    Median latency: {onnx_results['median_ms']:.1f} ms")
        print(f"    P95 latency:    {onnx_results['p95_ms']:.1f} ms")
        print(f"    Memory:         {onnx_results['memory_mb']:.0f} MB")
        print(f"  PyTorch: [not available — install torch + transformers for comparison]")

    print()
    print("  NOTE: These are REAL measurements from this machine.")
    print()

    # Save raw results for the paper
    results_file = PROJECT_ROOT / "evaluation" / "results_phase1_depth.txt"
    with open(results_file, "w") as f:
        f.write("PHASE 1 DEPTH BENCHMARK RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames: {N_FRAMES}\n\n")
        f.write(f"ONNX MiDaS-Small:\n")
        f.write(f"  Median: {onnx_results['median_ms']:.1f} ms\n")
        f.write(f"  Mean:   {onnx_results['mean_ms']:.1f} ms (±{onnx_results['stdev_ms']:.1f})\n")
        f.write(f"  P95:    {onnx_results['p95_ms']:.1f} ms\n")
        f.write(f"  P99:    {onnx_results['p99_ms']:.1f} ms\n")
        f.write(f"  Memory: {onnx_results['memory_mb']:.0f} MB\n\n")
        if pytorch_results:
            f.write(f"PyTorch DPT-Hybrid:\n")
            f.write(f"  Median: {pytorch_results['median_ms']:.1f} ms\n")
            f.write(f"  P95:    {pytorch_results['p95_ms']:.1f} ms\n")
            f.write(f"  Memory: {pytorch_results['memory_mb']:.0f} MB\n\n")
            speedup = pytorch_results["median_ms"] / onnx_results["median_ms"]
            f.write(f"Speedup: {speedup:.2f}x\n")
        else:
            f.write(f"PyTorch: [not tested — torch not installed]\n\n")
    print(f"  Raw results saved to: {results_file}")


if __name__ == "__main__":
    main()
