"""Evaluation of BlindAid strategies against manual annotations.

Loads annotations from c:\\blindaid\\evaluation\\annotations.json and computes
Precision, Recall, F1-Score, and detection delay for:
- Always-On (the ceiling)
- Static Skip 1/10 (standard baseline)
- Random Skip (p=0.85 baseline)
- AFP Full (our method)

Usage:
    python evaluation/evaluate_against_annotations.py
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import strategies and helpers from ablation script
from evaluation.run_ablation import (
    load_frames,
    run_always_on_get_gt,
    run_static_skip,
    run_random_skip,
    run_afp_variant,
)

OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "bench",
    "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "obstacle"
}

def evaluate_strategies_on_annotations(
    annotations_path: Path,
    clips_dir: Path,
    window: int = 5,
):
    """Evaluate strategies against manually annotated frames."""
    if not annotations_path.exists():
        print(f"ERROR: Annotations file not found at {annotations_path}")
        print("Please run the annotation tool first:")
        print("  python evaluation/annotate_frames.py")
        sys.exit(1)

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Group annotations by clip
    clip_annotations = {}
    for ann in annotations:
        clip = ann["clip"]
        if clip not in clip_annotations:
            clip_annotations[clip] = []
        clip_annotations[clip].append(ann)

    print(f"Loaded {len(annotations)} annotations across {len(clip_annotations)} clips.")
    print("=" * 80)

    # We evaluate 4 strategies
    strategies = {
        "Always-On": lambda f: run_always_on_get_gt(f),
        "Static 1/10": lambda f: run_static_skip(f, 10),
        "Random (p=0.85)": lambda f: run_random_skip(f, 0.85, seed=42),
        "AFP Full (ours)": lambda f: run_afp_variant(f, "full"),
    }

    # Initialize stats for each strategy
    # stats[strategy] = {
    #     "strict": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "delays": []},
    #     "agnostic": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "delays": []},
    #     "frames_processed": 0,
    #     "total_frames": 0,
    # }
    stats = {}
    for name in strategies:
        stats[name] = {
            "strict": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "delays": []},
            "agnostic": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "delays": []},
            "frames_processed": 0,
            "total_frames": 0,
        }

    for clip_name, anns in sorted(clip_annotations.items()):
        video_path = clips_dir / clip_name
        if not video_path.exists():
            print(f"Warning: clip {clip_name} not found in {clips_dir}, skipping.")
            continue

        print(f"\nProcessing clip: {clip_name} ({len(anns)} annotated keyframes)")
        frames = load_frames(str(video_path))
        
        # Run all strategies on this clip
        clip_runs = {}
        for name, run_fn in strategies.items():
            print(f"  Running {name}...")
            clip_runs[name] = run_fn(frames)
            stats[name]["frames_processed"] += clip_runs[name].get("frames_processed", len(frames))
            stats[name]["total_frames"] += len(frames)

        # Evaluate annotations for each strategy
        for ann in anns:
            gt_frame = ann["frame_idx"]
            gt_present = ann["obstacle_present"]
            gt_class = ann["obstacle_class"]
            gt_location = ann.get("location")

            # Ground truth frame boundaries
            # Note: sample_every_n = 3 in load_frames, so we map gt_frame from video index
            # to index in sampled list
            # Actually, annotations.json stores frame_idx which is the sampled frame index
            # in manifest.json (manifest.json creates indices 0, 1, 2... for keyframes).
            # Wait, let's verify if frame_idx in annotations/manifest is the sampled index
            # or the original video frame index.
            # In extract_keyframes.py, does it use video frame index?
            # Let's inspect manifest.json.
            
            for name, run_res in clip_runs.items():
                detections = run_res.get("all_detections", [])
                
                # Check strict matching (same class + same region)
                strict_tp, strict_fp, strict_fn, strict_tn, strict_delay = evaluate_frame_matching(
                    gt_frame, gt_present, gt_class, gt_location, detections, window, strict=True
                )
                stats[name]["strict"]["tp"] += strict_tp
                stats[name]["strict"]["fp"] += strict_fp
                stats[name]["strict"]["fn"] += strict_fn
                stats[name]["strict"]["tn"] += strict_tn
                if strict_delay is not None:
                    stats[name]["strict"]["delays"].append(strict_delay)

                # Check class-agnostic matching (any valid obstacle class in same region)
                ag_tp, ag_fp, ag_fn, ag_tn, ag_delay = evaluate_frame_matching(
                    gt_frame, gt_present, gt_class, gt_location, detections, window, strict=False
                )
                stats[name]["agnostic"]["tp"] += ag_tp
                stats[name]["agnostic"]["fp"] += ag_fp
                stats[name]["agnostic"]["fn"] += ag_fn
                stats[name]["agnostic"]["tn"] += ag_tn
                if ag_delay is not None:
                    stats[name]["agnostic"]["delays"].append(ag_delay)

    # Print results summary table
    print("\n" + "="*90)
    print("MANUAL ANNOTATION EVALUATION RESULTS")
    print("="*90)

    for match_type in ["strict", "agnostic"]:
        print(f"\n--- MATCHING MODE: {match_type.upper()} ---")
        print(f"{'Strategy':<18} {'Skip%':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>9} {'F1-Score':>9} {'Avg Delay':>10}")
        print("-" * 92)

        for name in strategies:
            s = stats[name][match_type]
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            avg_delay = np.mean(s["delays"]) if s["delays"] else 0.0
            
            total_f = stats[name]["total_frames"]
            proc_f = stats[name]["frames_processed"]
            skip_ratio = 1.0 - (proc_f / total_f) if total_f > 0 else 0.0

            print(f"{name:<18} {skip_ratio:>7.1%} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {precision:>9.1%} {recall:>8.1%} {f1:>8.1%} {avg_delay:>9.1f}f")

    print("="*90)


def evaluate_frame_matching(
    gt_frame: int,
    gt_present: bool,
    gt_class: str,
    gt_location: str,
    detections: list,
    window: int,
    strict: bool = True,
    sample_every_n: int = 3,
):
    """Check if the detections match the manual annotation for a frame.

    Returns: (tp, fp, fn, tn, delay)
    """
    # Map detection's sampled frame index to original video frame index (multiply by sample_every_n)
    # Also scale the window to original video frame scale (multiply window by sample_every_n)
    nearby_dets = [
        d for d in detections
        if abs(d["frame"] * sample_every_n - gt_frame) <= window * sample_every_n
    ]

    # Filter by class
    if strict:
        if gt_class in {"pole", "stairs", "other", "obstacle"}:
            # Custom hazards not in COCO vocabulary can match depth-based obstacles
            matching_dets = [d for d in nearby_dets if d["class"] in {gt_class, "obstacle"}]
        else:
            matching_dets = [d for d in nearby_dets if d["class"] == gt_class]
    else:
        matching_dets = [d for d in nearby_dets if d["class"] in OBSTACLE_CLASSES]

    # Also match spatial location (region) if provided
    if matching_dets and gt_location:
        matching_dets = [d for d in matching_dets if d.get("region") == gt_location]

    if gt_present:
        if matching_dets:
            # True Positive
            # Calculate delay (difference in frames from first matching detection to gt_frame)
            first_det_frame = min(d["frame"] * sample_every_n for d in matching_dets)
            delay = first_det_frame - gt_frame
            return 1, 0, 0, 0, delay
        else:
            # False Negative
            return 0, 0, 1, 0, None
    else:
        # No obstacle annotated
        # If there are any obstacle detections in the window, it's a False Positive
        has_obs_det = any(d["class"] in OBSTACLE_CLASSES for d in nearby_dets)
        if has_obs_det:
            return 0, 1, 0, 0, None
        else:
            # True Negative
            return 0, 0, 0, 1, None


def main():
    annotations_path = PROJECT_ROOT / "evaluation" / "annotations.json"
    clips_dir = PROJECT_ROOT / "clips"
    
    evaluate_strategies_on_annotations(annotations_path, clips_dir, window=5)

if __name__ == "__main__":
    main()
