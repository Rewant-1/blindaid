"""Ablation study for the BlindAid paper.

Runs AFP variants and multiple static-skip baselines on the recorded clips:
1. AFP Full (proximity + stability) — the full system
2. AFP Proximity-only — stability signal disabled
3. AFP Stability-only — proximity signal disabled  
4. Static Skip 1/3
5. Static Skip 1/5
6. Static Skip 1/10
7. Static Skip 1/15

Uses the same Always-On ground truth from results_video_evaluation.json
to compute detection coverage for each variant.
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


def load_frames(video_path: str, sample_every_n: int = 3) -> list:
    """Load and sample frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every_n == 0:
            frames.append(cv2.resize(frame, (640, 480)))
        idx += 1
    cap.release()
    return frames


def run_always_on_get_gt(frames: list) -> dict:
    """Run Always-On to get ground truth obstacle frames."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX
    from blindaid.core.depth_fusion import fuse_detections

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    all_detections = []
    latencies = []
    obstacle_classes = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                        "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "obstacle"}

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        depth_map = depth.compute_depth(frame)
        dets = detector.detect(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        h, w = frame.shape[:2]
        fused = fuse_detections(dets, depth_map, (h, w), enable_fallback=True)
        for fd in fused:
            all_detections.append({
                "frame": i, "class": fd.class_name,
                "confidence": fd.confidence, "region": fd.region,
            })

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in obstacle_classes:
            obstacle_frames.add(det["frame"])

    return {
        "obstacle_frame_indices": sorted(obstacle_frames),
        "all_detections": all_detections,
        "total_cpu_ms": sum(latencies),
    }


def run_static_skip(frames: list, skip_interval: int) -> dict:
    """Run static skip baseline."""
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX
    from blindaid.core.depth_fusion import fuse_detections

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    latencies = []
    all_detections = []
    obstacle_classes = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                        "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "obstacle"}

    for i, frame in enumerate(frames):
        if i % skip_interval == 0:
            t0 = time.perf_counter()
            depth_map = depth.compute_depth(frame)
            dets = detector.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            h, w = frame.shape[:2]
            fused = fuse_detections(dets, depth_map, (h, w), enable_fallback=True)
            for fd in fused:
                all_detections.append({
                    "frame": i, "class": fd.class_name,
                    "confidence": fd.confidence, "region": fd.region,
                })

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in obstacle_classes:
            obstacle_frames.add(det["frame"])

    return {
        "skip_interval": skip_interval,
        "frames_processed": len(latencies),
        "skip_ratio": 1.0 - len(latencies) / len(frames),
        "total_cpu_ms": sum(latencies),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "all_detections": all_detections,
    }


def run_random_skip(frames: list, skip_prob: float = 0.85, seed: int = 42) -> dict:
    """Run random skip baseline.

    For each frame, randomly decides whether to process it.
    Always processes frame 0. Uses a fixed random seed for reproducibility.

    Args:
        frames: list of video frames
        skip_prob: probability of skipping each frame (default 0.85 to match AFP's ~85%)
        seed: random seed for reproducibility
    """
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX
    from blindaid.core.depth_fusion import fuse_detections

    rng = np.random.RandomState(seed)

    depth = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    depth.compute_depth(frames[0])
    detector.detect(frames[0])

    latencies = []
    all_detections = []
    obstacle_classes = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                        "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "obstacle"}

    for i, frame in enumerate(frames):
        # Always process frame 0; otherwise process with probability (1 - skip_prob)
        if i == 0 or rng.random() < (1.0 - skip_prob):
            t0 = time.perf_counter()
            depth_map = depth.compute_depth(frame)
            dets = detector.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            h, w = frame.shape[:2]
            fused = fuse_detections(dets, depth_map, (h, w), enable_fallback=True)
            for fd in fused:
                all_detections.append({
                    "frame": i, "class": fd.class_name,
                    "confidence": fd.confidence, "region": fd.region,
                })

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in obstacle_classes:
            obstacle_frames.add(det["frame"])

    return {
        "skip_prob": skip_prob,
        "frames_processed": len(latencies),
        "skip_ratio": 1.0 - len(latencies) / len(frames) if frames else 0,
        "total_cpu_ms": sum(latencies),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "all_detections": all_detections,
    }


def run_afp_variant(frames: list, variant: str = "full") -> dict:
    """Run AFP with different signal combinations.
    
    variant: 'full', 'proximity_only', 'stability_only'
    """
    from blindaid.core.adaptive_processor import AdaptiveFrameProcessor
    from blindaid.core.depth_onnx import DepthAnalyzerONNX
    from blindaid.core.detector_onnx import ObjectDetectorONNX
    from blindaid.core.depth_fusion import fuse_detections

    depth_analyzer = DepthAnalyzerONNX()
    detector = ObjectDetectorONNX()
    afp = AdaptiveFrameProcessor()

    depth_analyzer.compute_depth(frames[0])
    detector.detect(frames[0])

    latencies = []
    all_detections = []
    last_depth = None
    obstacle_classes = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
                        "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "obstacle"}

    for i, frame in enumerate(frames):
        if variant == "full":
            should = afp.should_process("guardian", frame, last_depth)
        elif variant == "proximity_only":
            # Override: ignore stability, only use proximity
            should = _should_process_proximity_only(afp, i, frame, last_depth)
        elif variant == "stability_only":
            # Override: ignore proximity, only use stability
            should = _should_process_stability_only(afp, i, frame)
        else:
            should = afp.should_process("guardian", frame, last_depth)

        if should:
            t0 = time.perf_counter()
            depth_map = depth_analyzer.compute_depth(frame)
            dets = detector.detect(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            afp.update_depth(depth_map)
            last_depth = depth_map

            h, w = frame.shape[:2]
            fused = fuse_detections(dets, depth_map, (h, w), enable_fallback=True)
            for fd in fused:
                all_detections.append({
                    "frame": i, "class": fd.class_name,
                    "confidence": fd.confidence, "region": fd.region,
                })

    obstacle_frames = set()
    for det in all_detections:
        if det["class"] in obstacle_classes:
            obstacle_frames.add(det["frame"])

    return {
        "variant": variant,
        "frames_processed": len(latencies),
        "skip_ratio": 1.0 - len(latencies) / len(frames) if frames else 0,
        "total_cpu_ms": sum(latencies),
        "obstacle_frame_indices": sorted(obstacle_frames),
        "all_detections": all_detections,
    }


def _should_process_proximity_only(afp, frame_idx, frame, depth_map):
    """AFP using only proximity signal (stability fixed at 0.5)."""
    from blindaid.core.adaptive_processor import MODE_CONFIG

    config = MODE_CONFIG["guardian"]
    min_skip = config["min_skip"]
    max_skip = config["max_skip"]
    threshold = config.get("proximity_threshold", 0.7)

    metrics = afp._get_metrics("prox_only")
    metrics.frames_total += 1

    if "prox_only" not in afp._frame_counters:
        afp._frame_counters["prox_only"] = 0
        afp._current_skips["prox_only"] = 0

    afp._frame_counters["prox_only"] += 1

    if afp._frame_counters["prox_only"] > afp._current_skips["prox_only"]:
        proximity = afp.compute_obstacle_proximity(depth_map)
        # Compute skip ONLY from proximity
        if proximity > threshold:
            skip = min_skip
        else:
            safety_factor = 1.0 - proximity
            skip = int(min_skip + (max_skip - min_skip) * safety_factor)

        afp._current_skips["prox_only"] = skip
        afp._frame_counters["prox_only"] = 0
        metrics.frames_processed += 1
        metrics.proximity_values.append(proximity)
        metrics.skip_values.append(skip)
        return True
    return False


def _should_process_stability_only(afp, frame_idx, frame):
    """AFP using only stability signal (proximity fixed at 0.5)."""
    from blindaid.core.adaptive_processor import MODE_CONFIG

    config = MODE_CONFIG["guardian"]
    min_skip = config["min_skip"]
    max_skip = config["max_skip"]

    metrics = afp._get_metrics("stab_only")
    metrics.frames_total += 1

    if "stab_only" not in afp._frame_counters:
        afp._frame_counters["stab_only"] = 0
        afp._current_skips["stab_only"] = 0

    afp._frame_counters["stab_only"] += 1

    if afp._frame_counters["stab_only"] > afp._current_skips["stab_only"]:
        stability = afp.compute_scene_stability(frame)
        # Compute skip ONLY from stability
        skip = int(min_skip + (max_skip - min_skip) * stability)

        afp._current_skips["stab_only"] = skip
        afp._frame_counters["stab_only"] = 0
        metrics.frames_processed += 1
        metrics.stability_values.append(stability)
        metrics.skip_values.append(skip)
        return True
    return False


def compute_coverage(gt: dict, exp: dict, window: int = 5) -> dict:
    """Compute detection coverage of experimental vs ground truth."""
    gt_obstacles = gt.get("obstacle_frame_indices", [])
    gt_detections = gt.get("all_detections", [])
    exp_detections = exp.get("all_detections", [])

    if not gt_obstacles:
        return {"coverage": 1.0, "total_gt_frames": 0}

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
    for gt_frame in gt_obstacles:
        gt_classes = gt_frame_classes.get(gt_frame, set())
        found = False
        for offset in range(0, window + 1):
            for check in [gt_frame + offset, gt_frame - offset]:
                if gt_classes & exp_frame_classes.get(check, set()):
                    found = True
                    break
            if found:
                break
        if found:
            covered += 1

    return {
        "coverage": covered / len(gt_obstacles),
        "covered": covered,
        "total_gt_frames": len(gt_obstacles),
    }


def main():
    clips_dir = PROJECT_ROOT / "clips"
    videos = sorted(clips_dir.glob("*.mp4"))

    if not videos:
        print("No videos found in clips/")
        return

    print(f"Found {len(videos)} clips")
    print("=" * 80)

    all_results = []

    for video_path in videos:
        print(f"\n{'='*70}")
        print(f"  Processing: {video_path.name}")
        print(f"{'='*70}")

        frames = load_frames(str(video_path))
        print(f"  Loaded {len(frames)} frames")

        # Ground truth
        print("  Running Always-On (ground truth)...")
        gt = run_always_on_get_gt(frames)
        n_obs = len(gt["obstacle_frame_indices"])
        print(f"    {n_obs} obstacle frames detected")

        clip_result = {
            "clip": video_path.name,
            "total_frames": len(frames),
            "gt_obstacle_frames": n_obs,
            "gt_cpu_ms": gt["total_cpu_ms"],
        }

        # Static skip baselines
        for interval in [3, 5, 10, 15]:
            print(f"  Running Static Skip 1/{interval}...")
            ss = run_static_skip(frames, interval)
            cov = compute_coverage(gt, ss)
            clip_result[f"static_{interval}"] = {
                "skip_ratio": ss["skip_ratio"],
                "coverage": cov["coverage"],
                "cpu_ms": ss["total_cpu_ms"],
                "frames_processed": ss["frames_processed"],
            }
            print(f"    Coverage: {cov['coverage']:.1%}, Skip: {ss['skip_ratio']:.1%}")

        # Random skip baseline
        print(f"  Running Random Skip (p=0.85)...")
        rs = run_random_skip(frames, skip_prob=0.85, seed=42)
        cov = compute_coverage(gt, rs)
        clip_result["random_skip"] = {
            "skip_ratio": rs["skip_ratio"],
            "coverage": cov["coverage"],
            "cpu_ms": rs["total_cpu_ms"],
            "frames_processed": rs["frames_processed"],
        }
        print(f"    Coverage: {cov['coverage']:.1%}, Skip: {rs['skip_ratio']:.1%}")

        # AFP variants
        for variant in ["full", "proximity_only", "stability_only"]:
            print(f"  Running AFP ({variant})...")
            afp = run_afp_variant(frames, variant)
            cov = compute_coverage(gt, afp)
            clip_result[f"afp_{variant}"] = {
                "skip_ratio": afp["skip_ratio"],
                "coverage": cov["coverage"],
                "cpu_ms": afp["total_cpu_ms"],
                "frames_processed": afp["frames_processed"],
            }
            print(f"    Coverage: {cov['coverage']:.1%}, Skip: {afp['skip_ratio']:.1%}")

        all_results.append(clip_result)

    # Save results
    output_file = PROJECT_ROOT / "evaluation" / "results_ablation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print(f"\n{'='*90}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*90}")

    strategies = [
        ("static_3", "Static 1/3"),
        ("static_5", "Static 1/5"),
        ("static_10", "Static 1/10"),
        ("static_15", "Static 1/15"),
        ("random_skip", "Random Skip"),
        ("afp_proximity_only", "AFP Prox-only"),
        ("afp_stability_only", "AFP Stab-only"),
        ("afp_full", "AFP Full"),
    ]

    print(f"\n{'Strategy':<20} {'Avg Coverage':>14} {'Avg Skip':>12} {'Avg CPU(ms)':>14}")
    print("-" * 62)

    for key, label in strategies:
        coverages = [r[key]["coverage"] for r in all_results if key in r]
        skips = [r[key]["skip_ratio"] for r in all_results if key in r]
        cpus = [r[key]["cpu_ms"] for r in all_results if key in r]

        if coverages:
            print(f"{label:<20} {statistics.mean(coverages):>13.1%} {statistics.mean(skips):>11.1%} {statistics.mean(cpus):>13.0f}")

    # Confidence intervals (std dev)
    print(f"\n{'Strategy':<20} {'Coverage Mean':>14} {'Coverage Std':>14} {'95% CI':>20}")
    print("-" * 70)

    for key, label in strategies:
        coverages = [r[key]["coverage"] for r in all_results if key in r]
        if len(coverages) >= 2:
            mean_c = statistics.mean(coverages)
            std_c = statistics.stdev(coverages)
            ci = 1.96 * std_c / (len(coverages) ** 0.5)
            print(f"{label:<20} {mean_c:>13.1%} {std_c:>13.3f} [{mean_c - ci:.1%}, {mean_c + ci:.1%}]")
        elif coverages:
            print(f"{label:<20} {coverages[0]:>13.1%} {'N/A':>13} {'N/A':>20}")


if __name__ == "__main__":
    main()
