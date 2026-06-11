"""Video evaluation pipeline for BlindAid research paper.

Processes recorded video clips and compares AFP vs baselines.
NO manual annotation required — uses Always-On as ground truth.

Methodology:
- Ground truth = Always-On (processes every frame, catches everything)
- AFP = experimental condition (skips frames adaptively)
- Static Skip = baseline comparison
- Metric: "detection coverage" = % of Always-On detections that AFP also catches
- Metric: "response delay" = how many frames late is AFP vs Always-On

Usage:
    # Process a single video:
    python evaluation/evaluate_video.py --video path/to/clip.mp4

    # Process all videos in a directory:
    python evaluation/evaluate_video.py --dir path/to/clips/

    # Extract frames from videos first (for inspection):
    python evaluation/evaluate_video.py --extract path/to/clip.mp4
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    if not data:
        return 0.0
    idx = min(int(len(data) * p / 100), len(data) - 1)
    return sorted(data)[idx]


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_idx: int
    was_processed: bool
    detections: list = field(default_factory=list)
    depth_proximity: float = 0.0
    latency_ms: float = 0.0


@dataclass
class VideoResult:
    """Complete evaluation result for one video."""
    video_path: str
    total_frames: int
    fps: float
    duration_s: float
    always_on: dict = field(default_factory=dict)
    static_skip: dict = field(default_factory=dict)
    afp: dict = field(default_factory=dict)


def extract_frames(video_path: str, output_dir: str, sample_fps: float = 2.0):
    """Extract frames from video at given FPS for inspection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(video_fps / sample_fps))

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {total_frames} total ({video_fps:.1f} FPS)")
    print(f"Saved to: {output_dir}")


def run_always_on(frames: list) -> dict:
    """Process every frame — this is our ground truth."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()

    # Warmup
    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    results = []
    latencies = []
    all_detections = []  # (frame_idx, class, confidence, region)

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        depth_map = depth.compute_depth(frame)
        detections = detector.detect(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        # Record detections with spatial info
        h, w = frame.shape[:2]
        for det in detections:
            cx = (det.x1 + det.x2) / 2
            region = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
            all_detections.append({
                "frame": i,
                "class": det.class_name,
                "confidence": det.confidence,
                "region": region,
            })

        # Proximity from depth
        center = depth_map[h // 3: 2 * h // 3, w // 4: 3 * w // 4]
        proximity = float(np.max(center))

        results.append(FrameResult(
            frame_idx=i,
            was_processed=True,
            detections=[(d.class_name, d.confidence) for d in detections],
            depth_proximity=proximity,
            latency_ms=elapsed,
        ))

    # Frames with obstacles
    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                            "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet"}:
            obstacle_frames.add(det["frame"])

    return {
        "frames_processed": len(frames),
        "total_detections": len(all_detections),
        "unique_obstacle_frames": len(obstacle_frames),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "total_cpu_ms": sum(latencies),
        "detections_per_frame": len(all_detections) / len(frames) if frames else 0,
        "all_detections": all_detections,
        "frame_results": results,
    }


def run_static_skip(frames: list, skip_interval: int = 10) -> dict:
    """Process every Nth frame."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    results = []
    latencies = []
    all_detections = []
    last_detections = []

    for i, frame in enumerate(frames):
        if i % skip_interval == 0:
            t0 = time.perf_counter()
            depth_map = depth.compute_depth(frame)
            dets = detector.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            last_detections = [(d.class_name, d.confidence) for d in dets]

            h, w = frame.shape[:2]
            for det in dets:
                cx = (det.x1 + det.x2) / 2
                region = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
                all_detections.append({
                    "frame": i,
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "region": region,
                })

            results.append(FrameResult(frame_idx=i, was_processed=True,
                                        detections=last_detections, latency_ms=elapsed))
        else:
            results.append(FrameResult(frame_idx=i, was_processed=False,
                                        detections=last_detections))

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                            "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet"}:
            obstacle_frames.add(det["frame"])

    return {
        "frames_processed": len(latencies),
        "skip_ratio": 1.0 - len(latencies) / len(frames),
        "total_detections": len(all_detections),
        "unique_obstacle_frames": len(obstacle_frames),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "total_cpu_ms": sum(latencies),
        "all_detections": all_detections,
    }


def run_afp(frames: list) -> dict:
    """Process frames using AFP."""
    from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    afp = AdaptiveFrameProcessor()

    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    results = []
    latencies = []
    all_detections = []
    last_detections = []
    last_depth = None

    for i, frame in enumerate(frames):
        should = afp.should_process("guardian", frame, last_depth)

        if should:
            t0 = time.perf_counter()
            depth_map = depth.compute_depth(frame)
            dets = detector.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            afp.update_depth(depth_map)
            last_depth = depth_map
            last_detections = [(d.class_name, d.confidence) for d in dets]

            h, w = frame.shape[:2]
            for det in dets:
                cx = (det.x1 + det.x2) / 2
                region = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
                all_detections.append({
                    "frame": i,
                    "class": det.class_name,
                    "confidence": det.confidence,
                    "region": region,
                })

            results.append(FrameResult(frame_idx=i, was_processed=True,
                                        detections=last_detections, latency_ms=elapsed))
        else:
            results.append(FrameResult(frame_idx=i, was_processed=False,
                                        detections=last_detections))

    afp_metrics = afp.get_metrics("guardian")

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                            "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet"}:
            obstacle_frames.add(det["frame"])

    return {
        "frames_processed": afp_metrics["frames_processed"],
        "skip_ratio": afp_metrics["skip_ratio"],
        "total_detections": len(all_detections),
        "unique_obstacle_frames": len(obstacle_frames),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "total_cpu_ms": sum(latencies),
        "avg_proximity": afp_metrics["avg_proximity"],
        "avg_stability": afp_metrics["avg_stability"],
        "cpu_savings_pct": afp_metrics["cpu_savings_pct"],
        "all_detections": all_detections,
    }


def compute_detection_coverage(ground_truth: dict, experimental: dict, window: int = 5) -> dict:
    """Compute detection coverage: what % of GT obstacle events does experimental catch?

    An obstacle event is 'covered' if the experimental strategy detected the same
    class within ±window frames of when Always-On detected it.

    This is the key metric: AFP should catch most obstacles, just maybe a few frames late.
    """
    gt_obstacles = ground_truth.get("obstacle_frame_indices", [])
    gt_detections = ground_truth.get("all_detections", [])
    exp_detections = experimental.get("all_detections", [])

    if not gt_obstacles:
        return {"coverage": 1.0, "missed_frames": 0, "total_gt_frames": 0,
                "avg_delay": 0.0, "note": "No obstacles in ground truth"}

    # Build frame -> classes maps
    gt_frame_classes = {}
    for det in gt_detections:
        f = det["frame"]
        if f not in gt_frame_classes:
            gt_frame_classes[f] = set()
        gt_frame_classes[f].add(det["class"])

    exp_frame_classes = {}
    for det in exp_detections:
        f = det["frame"]
        if f not in exp_frame_classes:
            exp_frame_classes[f] = set()
        exp_frame_classes[f].add(det["class"])

    covered = 0
    delays = []

    for gt_frame in gt_obstacles:
        gt_classes = gt_frame_classes.get(gt_frame, set())
        found = False

        # Check if experimental detected same classes within window
        for offset in range(0, window + 1):
            for check_frame in [gt_frame + offset, gt_frame - offset]:
                exp_classes = exp_frame_classes.get(check_frame, set())
                if gt_classes & exp_classes:  # Any overlap
                    found = True
                    delays.append(abs(offset))
                    break
            if found:
                break

        if found:
            covered += 1

    coverage = covered / len(gt_obstacles) if gt_obstacles else 1.0
    avg_delay = statistics.mean(delays) if delays else 0.0

    return {
        "coverage": coverage,
        "covered_frames": covered,
        "missed_frames": len(gt_obstacles) - covered,
        "total_gt_frames": len(gt_obstacles),
        "avg_delay_frames": avg_delay,
        "max_delay_frames": max(delays) if delays else 0,
    }


def evaluate_video(video_path: str, sample_every_n: int = 3) -> VideoResult:
    """Run full evaluation on a video file.

    Args:
        video_path: Path to video file
        sample_every_n: Process every Nth video frame (to manage total computation)
    """
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {os.path.basename(video_path)}")
    print(f"{'=' * 70}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / video_fps if video_fps > 0 else 0

    print(f"  Video: {video_fps:.1f} FPS, {total_video_frames} frames, {duration:.1f}s")
    print(f"  Sampling every {sample_every_n} frames = ~{video_fps / sample_every_n:.0f} eval FPS")

    # Extract frames
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            # Resize to 640x480 for consistent evaluation
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)
        frame_idx += 1
    cap.release()

    print(f"  Extracted {len(frames)} frames for evaluation")

    if len(frames) < 10:
        print("  WARNING: Too few frames for meaningful evaluation")

    # Run all three strategies
    print(f"\n  Running Always-On (ground truth)...")
    t0 = time.perf_counter()
    always_on = run_always_on(frames)
    print(f"    Done in {time.perf_counter() - t0:.1f}s | "
          f"{always_on['total_detections']} detections, "
          f"{always_on['unique_obstacle_frames']} obstacle frames")

    print(f"  Running Static Skip (1/10)...")
    t0 = time.perf_counter()
    static = run_static_skip(frames, skip_interval=10)
    print(f"    Done in {time.perf_counter() - t0:.1f}s | "
          f"{static['total_detections']} detections")

    print(f"  Running AFP...")
    t0 = time.perf_counter()
    afp = run_afp(frames)
    print(f"    Done in {time.perf_counter() - t0:.1f}s | "
          f"{afp['total_detections']} detections")

    # Compute coverage
    print(f"\n  Computing detection coverage...")
    afp_coverage = compute_detection_coverage(always_on, afp)
    static_coverage = compute_detection_coverage(always_on, static)

    # Print results
    print(f"\n  {'-' * 60}")
    print(f"  RESULTS: {os.path.basename(video_path)}")
    print(f"  {'-' * 60}")

    print(f"\n  {'Strategy':<25} {'Processed':<12} {'Skip%':<10} {'CPU (ms)':<12} {'Coverage':<10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    print(f"  {'Always-On (GT)':<25} {always_on['frames_processed']:<12} {'0%':<10} {always_on['total_cpu_ms']:<12.0f} {'100%':<10}")
    print(f"  {'Static Skip (1/10)':<25} {static['frames_processed']:<12} {static['skip_ratio']:<10.0%} {static['total_cpu_ms']:<12.0f} {static_coverage['coverage']:<10.0%}")
    print(f"  {'AFP (ours)':<25} {afp['frames_processed']:<12} {afp['skip_ratio']:<10.0%} {afp['total_cpu_ms']:<12.0f} {afp_coverage['coverage']:<10.0%}")

    if afp_coverage['total_gt_frames'] > 0:
        print(f"\n  AFP Detection Coverage Detail:")
        print(f"    Covered: {afp_coverage['covered_frames']}/{afp_coverage['total_gt_frames']} obstacle events")
        print(f"    Missed:  {afp_coverage['missed_frames']}")
        print(f"    Avg delay: {afp_coverage['avg_delay_frames']:.1f} frames")

    result = VideoResult(
        video_path=video_path,
        total_frames=len(frames),
        fps=video_fps,
        duration_s=duration,
    )
    result.always_on = {k: v for k, v in always_on.items() if k != "frame_results"}
    result.always_on.pop("all_detections", None)
    result.static_skip = {k: v for k, v in static.items() if k != "all_detections"}
    result.afp = {k: v for k, v in afp.items() if k != "all_detections"}
    result.afp["coverage"] = afp_coverage
    result.static_skip["coverage"] = static_coverage

    return result


def evaluate_directory(dir_path: str):
    """Evaluate all video files in a directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = sorted([
        f for f in Path(dir_path).iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not videos:
        print(f"No video files found in {dir_path}")
        print(f"Supported formats: {video_extensions}")
        return

    print(f"Found {len(videos)} videos in {dir_path}")

    all_results = []
    for video in videos:
        result = evaluate_video(str(video))
        if result:
            all_results.append(result)

    # Save aggregate results
    if all_results:
        output_file = PROJECT_ROOT / "evaluation" / "results_video_evaluation.json"
        serializable = []
        for r in all_results:
            serializable.append({
                "video": os.path.basename(r.video_path),
                "total_frames": r.total_frames,
                "fps": r.fps,
                "duration_s": r.duration_s,
                "always_on": r.always_on,
                "static_skip": r.static_skip,
                "afp": r.afp,
            })

        with open(output_file, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print(f"  AGGREGATE RESULTS ({len(all_results)} videos)")
        print(f"{'=' * 70}")

        total_frames = sum(r.total_frames for r in all_results)
        total_always_cpu = sum(r.always_on.get("total_cpu_ms", 0) for r in all_results)
        total_afp_cpu = sum(r.afp.get("total_cpu_ms", 0) for r in all_results)
        total_static_cpu = sum(r.static_skip.get("total_cpu_ms", 0) for r in all_results)

        coverages = [r.afp.get("coverage", {}).get("coverage", 0) for r in all_results
                     if r.afp.get("coverage", {}).get("total_gt_frames", 0) > 0]
        avg_coverage = statistics.mean(coverages) if coverages else 0

        print(f"  Total frames evaluated: {total_frames}")
        print(f"  CPU savings (AFP vs Always-On): {(1 - total_afp_cpu / total_always_cpu) * 100:.1f}%")
        print(f"  Avg AFP detection coverage: {avg_coverage:.1%}")
        print(f"  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="BlindAid Video Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a single video clip:
    python evaluation/evaluate_video.py --video clips/indoor_hallway.mp4

    # Evaluate all clips in a directory:
    python evaluation/evaluate_video.py --dir clips/

    # Just extract frames for inspection:
    python evaluation/evaluate_video.py --extract clips/indoor_hallway.mp4
        """,
    )

    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--dir", type=str, help="Path to directory of video files")
    parser.add_argument("--extract", type=str, help="Extract frames from video (for inspection)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Process every Nth frame from video (default: 3)")

    args = parser.parse_args()

    if args.extract:
        name = Path(args.extract).stem
        output_dir = str(PROJECT_ROOT / "evaluation" / "extracted_frames" / name)
        extract_frames(args.extract, output_dir)

    elif args.video:
        result = evaluate_video(args.video, sample_every_n=args.sample)
        if result:
            output_file = PROJECT_ROOT / "evaluation" / f"results_{Path(args.video).stem}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "video": os.path.basename(result.video_path),
                    "total_frames": result.total_frames,
                    "always_on": result.always_on,
                    "static_skip": result.static_skip,
                    "afp": result.afp,
                }, f, indent=2, default=str)
            print(f"\n  Results saved to: {output_file}")

    elif args.dir:
        evaluate_directory(args.dir)

    else:
        parser.print_help()
        print("\n  TIP: Record 8-10 clips (30-45s each) with your phone, then run:")
        print("       python evaluation/evaluate_video.py --dir path/to/clips/")


if __name__ == "__main__":
    main()
