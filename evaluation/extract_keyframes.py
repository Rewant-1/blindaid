"""Extract key frames from BlindAid video clips for annotation.

Reads each clip from c:\\blindaid\\clips\\*.mp4, samples every 3rd frame
(matching the evaluation pipeline's sample_every_n=3), resizes to 640x480,
then selects ~20 uniformly-spaced key frames per clip.

Output:
    c:\\blindaid\\evaluation\\annotation_frames\\<clip_name>\\frame_NNNN.jpg
    c:\\blindaid\\evaluation\\annotation_frames\\manifest.json

Usage:
    python evaluation/extract_keyframes.py
    python evaluation/extract_keyframes.py --clips-dir clips --keyframes-per-clip 20
    python evaluation/extract_keyframes.py --sample-every 3 --width 640 --height 480
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# Project layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS_DIR = PROJECT_ROOT / "clips"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "annotation_frames"


def discover_clips(clips_dir: Path) -> list[Path]:
    """Find all .mp4 clips sorted naturally (clip1, clip2, ..., clip16)."""
    clips = sorted(
        clips_dir.glob("*.mp4"),
        key=lambda p: _natural_sort_key(p.stem),
    )
    return clips


def _natural_sort_key(text: str):
    """Sort 'clip2' before 'clip10'."""
    import re
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def extract_sampled_frames(
    video_path: Path,
    sample_every_n: int = 3,
    target_size: tuple[int, int] = (640, 480),
) -> tuple[list[np.ndarray], list[int], float, int]:
    """Read a video and return sampled + resized frames.

    Returns:
        frames: list of BGR images (640x480)
        indices: original video frame indices that were sampled
        fps: video FPS
        total_frames: total frame count in the original video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n == 0:
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
            indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    return frames, indices, fps, total_frames


def select_uniform_keyframes(
    frames: list[np.ndarray],
    indices: list[int],
    num_keyframes: int = 20,
) -> list[tuple[np.ndarray, int]]:
    """Select approximately `num_keyframes` evenly-spaced frames.

    Uses uniform sampling across the sampled frame list so that
    key frames span the entire clip duration.

    Returns:
        List of (frame_image, original_video_frame_index) tuples.
    """
    n = len(frames)
    if n == 0:
        return []

    # Clamp to available frames
    k = min(num_keyframes, n)

    # Compute evenly-spaced positions (linspace gives exact endpoints)
    positions = np.linspace(0, n - 1, k, dtype=int)
    # Remove duplicates while preserving order
    seen = set()
    unique_positions = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique_positions.append(p)

    return [(frames[i], indices[i]) for i in unique_positions]


def process_clip(
    clip_path: Path,
    output_dir: Path,
    sample_every_n: int,
    target_size: tuple[int, int],
    num_keyframes: int,
) -> list[dict]:
    """Process a single clip: extract sampled frames, select keyframes, save.

    Returns list of manifest entries for this clip.
    """
    clip_name = clip_path.stem  # e.g. "clip1"
    clip_output = output_dir / clip_name
    clip_output.mkdir(parents=True, exist_ok=True)

    # Extract all sampled frames
    frames, indices, fps, total_video_frames = extract_sampled_frames(
        clip_path, sample_every_n, target_size
    )

    duration = total_video_frames / fps if fps > 0 else 0.0

    print(f"  {clip_path.name}: {total_video_frames} total frames, "
          f"{fps:.1f} FPS, {duration:.1f}s, "
          f"{len(frames)} sampled (every {sample_every_n})")

    # Select uniform keyframes
    keyframes = select_uniform_keyframes(frames, indices, num_keyframes)

    print(f"    -> Selected {len(keyframes)} key frames (uniform sampling)")

    # Save and build manifest entries
    manifest_entries = []
    for kf_frame, orig_idx in keyframes:
        filename = f"frame_{orig_idx:04d}.jpg"
        filepath = clip_output / filename

        # Save with reasonable JPEG quality
        cv2.imwrite(str(filepath), kf_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Relative path for manifest (portable across machines)
        rel_path = f"annotation_frames/{clip_name}/{filename}"
        manifest_entries.append({
            "clip": clip_path.name,           # e.g. "clip1.mp4"
            "frame_idx": orig_idx,            # original video frame index
            "image_path": rel_path,           # relative to evaluation/
        })

    return manifest_entries


def main():
    parser = argparse.ArgumentParser(
        description="Extract key frames from BlindAid clips for annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clips-dir", type=str, default=str(DEFAULT_CLIPS_DIR),
        help=f"Directory containing .mp4 clips (default: {DEFAULT_CLIPS_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for frames (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-every", type=int, default=3,
        help="Sample every Nth frame, matching evaluation pipeline (default: 3)",
    )
    parser.add_argument(
        "--keyframes-per-clip", type=int, default=20,
        help="Number of key frames to select per clip (default: 20)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Target frame width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Target frame height (default: 480)",
    )
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    target_size = (args.width, args.height)

    # Discover clips
    clips = discover_clips(clips_dir)
    if not clips:
        print(f"ERROR: No .mp4 files found in {clips_dir}")
        sys.exit(1)

    print(f"BlindAid Key Frame Extractor")
    print(f"{'=' * 60}")
    print(f"  Clips directory : {clips_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Clips found     : {len(clips)}")
    print(f"  Sample every    : {args.sample_every} frames")
    print(f"  Keyframes/clip  : {args.keyframes_per_clip}")
    print(f"  Target size     : {target_size[0]}x{target_size[1]}")
    print(f"{'=' * 60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all clips
    all_manifest = []
    t_start = time.perf_counter()

    for clip_path in clips:
        entries = process_clip(
            clip_path, output_dir,
            sample_every_n=args.sample_every,
            target_size=target_size,
            num_keyframes=args.keyframes_per_clip,
        )
        all_manifest.extend(entries)

    elapsed = time.perf_counter() - t_start

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2)

    # Summary statistics
    clips_processed = len(clips)
    total_keyframes = len(all_manifest)
    frames_per_clip = {}
    for entry in all_manifest:
        clip = entry["clip"]
        frames_per_clip[clip] = frames_per_clip.get(clip, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Clips processed  : {clips_processed}")
    print(f"  Total key frames : {total_keyframes}")
    print(f"  Elapsed time     : {elapsed:.1f}s")
    print(f"  Manifest saved   : {manifest_path}")
    print()
    print(f"  Frames per clip:")
    for clip_name in sorted(frames_per_clip, key=_natural_sort_key):
        print(f"    {clip_name:15s} : {frames_per_clip[clip_name]} frames")
    print()
    print(f"  Next step: annotate with")
    print(f"    python evaluation/annotate_frames.py")


if __name__ == "__main__":
    main()
