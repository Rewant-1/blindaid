"""OpenCV-based annotation GUI for BlindAid evaluation frames.

Loads frames from the manifest.json produced by extract_keyframes.py and
provides a keyboard-driven interface for labelling obstacles.

Keyboard shortcuts:
    o           Toggle obstacle present (yes / no)
    1-6         Set obstacle class (1=person 2=chair 3=car 4=pole 5=stairs 6=other)
    d           Cycle distance band (very_close -> close -> nearby -> moderate -> far)
    l           Cycle location (left -> center -> right)
    n / RIGHT   Next frame
    p / LEFT    Previous frame
    s           Save annotations to disk
    q           Quit and save

Annotations are saved as JSON at c:\\blindaid\\evaluation\\annotations.json.
If the file already exists it is loaded on startup so you can resume.

Usage:
    python evaluation/annotate_frames.py
    python evaluation/annotate_frames.py --manifest evaluation/annotation_frames/manifest.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "annotation_frames" / "manifest.json"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "evaluation" / "annotations.json"

OBSTACLE_CLASSES = {
    ord("1"): "person",
    ord("2"): "chair",
    ord("3"): "car",
    ord("4"): "pole",
    ord("5"): "stairs",
    ord("6"): "other",
}

DISTANCE_BANDS = ["very_close", "close", "nearby", "moderate", "far"]
LOCATIONS = ["left", "center", "right"]

# Overlay appearance
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 0.7
FONT_SCALE_SMALL = 0.5
FONT_THICKNESS = 1
BG_ALPHA = 0.6  # transparency for overlay background
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_BG = (30, 30, 30)
COLOR_HEADER_BG = (50, 40, 30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _natural_sort_key(text: str):
    import re
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load the frame manifest JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(annotations_path: Path) -> list[dict]:
    """Load existing annotations for resume support."""
    if annotations_path.exists():
        with open(annotations_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_annotations(annotations: list[dict], annotations_path: Path) -> None:
    """Save annotations to JSON."""
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)


def make_annotation_key(clip: str, frame_idx: int) -> str:
    """Create a unique key for a frame annotation."""
    return f"{clip}:{frame_idx}"


def default_annotation(entry: dict) -> dict:
    """Create default (empty) annotation for a manifest entry."""
    return {
        "clip": entry["clip"],
        "frame_idx": entry["frame_idx"],
        "image_path": entry["image_path"],
        "obstacle_present": False,
        "obstacle_class": None,
        "distance_band": None,
        "location": None,
    }


def merge_annotations(
    manifest: list[dict],
    existing: list[dict],
) -> list[dict]:
    """Merge existing annotations with manifest, preserving prior work.

    Every manifest entry gets an annotation record.  If a matching record
    exists in `existing`, its labels are used; otherwise defaults are applied.
    """
    existing_map: dict[str, dict] = {}
    for ann in existing:
        key = make_annotation_key(ann["clip"], ann["frame_idx"])
        existing_map[key] = ann

    merged = []
    for entry in manifest:
        key = make_annotation_key(entry["clip"], entry["frame_idx"])
        if key in existing_map:
            # Use existing but ensure image_path is up to date
            ann = existing_map[key].copy()
            ann["image_path"] = entry["image_path"]
            merged.append(ann)
        else:
            merged.append(default_annotation(entry))
    return merged


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    annotation: dict,
    index: int,
    total: int,
) -> np.ndarray:
    """Draw the annotation overlay on a copy of the frame.

    Layout:
        Top section:    frame counter + clip name + annotation values
        Bottom section: keyboard shortcut legend
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # ── Top overlay ──────────────────────────────────────────────────
    header_h = 130
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), COLOR_HEADER_BG, -1)
    cv2.addWeighted(overlay, BG_ALPHA, display, 1 - BG_ALPHA, 0, display)

    y = 22
    line_h = 22

    # Frame counter and clip name
    counter_text = f"Frame {index + 1}/{total}"
    clip_text = f"Clip: {annotation['clip']}"
    frame_idx_text = f"Video frame index: {annotation['frame_idx']}"
    cv2.putText(display, counter_text, (10, y), FONT, FONT_SCALE_LARGE,
                COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(display, clip_text, (250, y), FONT, FONT_SCALE_LARGE,
                COLOR_YELLOW, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(display, frame_idx_text, (450, y), FONT, FONT_SCALE_SMALL,
                COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)
    y += line_h + 4

    # Obstacle present
    obs = annotation.get("obstacle_present", False)
    obs_text = "YES" if obs else "NO"
    obs_color = COLOR_GREEN if obs else COLOR_RED
    cv2.putText(display, f"Obstacle: {obs_text}", (10, y), FONT,
                FONT_SCALE_LARGE, obs_color, FONT_THICKNESS + 1, cv2.LINE_AA)
    y += line_h

    # Class
    cls = annotation.get("obstacle_class") or "—"
    cv2.putText(display, f"Class: {cls}", (10, y), FONT,
                FONT_SCALE_SMALL, COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)

    # Distance
    dist = annotation.get("distance_band") or "—"
    cv2.putText(display, f"Distance: {dist}", (200, y), FONT,
                FONT_SCALE_SMALL, COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)

    # Location
    loc = annotation.get("location") or "—"
    cv2.putText(display, f"Location: {loc}", (420, y), FONT,
                FONT_SCALE_SMALL, COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)
    y += line_h

    # Annotation completeness indicator
    complete = _is_complete(annotation)
    status = "COMPLETE" if complete else "INCOMPLETE"
    status_color = COLOR_GREEN if complete else COLOR_YELLOW
    cv2.putText(display, status, (10, y), FONT,
                FONT_SCALE_SMALL, status_color, FONT_THICKNESS, cv2.LINE_AA)

    # ── Bottom legend ────────────────────────────────────────────────
    legend_lines = [
        "[O] obstacle  [1-6] class  [D] distance  [L] location",
        "[N/RIGHT] next  [P/LEFT] prev  [S] save  [Q] quit+save",
    ]
    legend_h = 20 + len(legend_lines) * 20
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - legend_h), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, BG_ALPHA, display, 1 - BG_ALPHA, 0, display)

    for i, line in enumerate(legend_lines):
        ly = h - legend_h + 18 + i * 20
        cv2.putText(display, line, (10, ly), FONT,
                    FONT_SCALE_SMALL, COLOR_WHITE, FONT_THICKNESS, cv2.LINE_AA)

    return display


def _is_complete(annotation: dict) -> bool:
    """An annotation is complete when obstacle status is set,
    and if obstacle_present, then class/distance/location are filled."""
    if not annotation.get("obstacle_present"):
        return True  # "no obstacle" is a valid complete annotation
    return all([
        annotation.get("obstacle_class"),
        annotation.get("distance_band"),
        annotation.get("location"),
    ])


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def run_annotation_gui(
    manifest: list[dict],
    annotations: list[dict],
    base_dir: Path,
    annotations_path: Path,
) -> None:
    """Run the OpenCV annotation GUI."""
    if not annotations:
        print("ERROR: No frames to annotate.")
        return

    total = len(annotations)
    current = 0

    # Find first unannotated frame to resume there
    for i, ann in enumerate(annotations):
        if not _is_complete(ann):
            current = i
            break

    window_name = "BlindAid Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    def show_frame():
        ann = annotations[current]
        img_path = base_dir / ann["image_path"]
        if not img_path.exists():
            # Show a placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"IMAGE NOT FOUND", (120, 240), FONT,
                        1.0, COLOR_RED, 2, cv2.LINE_AA)
            cv2.putText(frame, str(img_path), (20, 280), FONT,
                        0.4, COLOR_WHITE, 1, cv2.LINE_AA)
        else:
            frame = cv2.imread(str(img_path))

        display = draw_overlay(frame, ann, current, total)
        cv2.imshow(window_name, display)

    show_frame()

    unsaved_changes = False

    while True:
        # On Windows cv2.waitKey returns full 32-bit; mask to 8-bit
        key = cv2.waitKey(0) & 0xFF
        ann = annotations[current]

        if key == ord("q"):
            # Quit and save
            save_annotations(annotations, annotations_path)
            print(f"\nAnnotations saved to {annotations_path}")
            break

        elif key == ord("s"):
            save_annotations(annotations, annotations_path)
            unsaved_changes = False
            print(f"  [saved] {annotations_path}")

        elif key == ord("o"):
            ann["obstacle_present"] = not ann["obstacle_present"]
            if not ann["obstacle_present"]:
                # Clear dependent fields when toggling off
                ann["obstacle_class"] = None
                ann["distance_band"] = None
                ann["location"] = None
            unsaved_changes = True

        elif key in OBSTACLE_CLASSES:
            ann["obstacle_class"] = OBSTACLE_CLASSES[key]
            ann["obstacle_present"] = True  # implicitly mark present
            unsaved_changes = True

        elif key == ord("d"):
            # Cycle distance band
            cur = ann.get("distance_band")
            if cur in DISTANCE_BANDS:
                idx = (DISTANCE_BANDS.index(cur) + 1) % len(DISTANCE_BANDS)
            else:
                idx = 0
            ann["distance_band"] = DISTANCE_BANDS[idx]
            unsaved_changes = True

        elif key == ord("l"):
            # Cycle location
            cur = ann.get("location")
            if cur in LOCATIONS:
                idx = (LOCATIONS.index(cur) + 1) % len(LOCATIONS)
            else:
                idx = 0
            ann["location"] = LOCATIONS[idx]
            unsaved_changes = True

        elif key == ord("n") or key == 83:
            # Next frame (83 = RIGHT arrow on Windows via cv2.waitKey)
            if current < total - 1:
                current += 1

        elif key == ord("p") or key == 81:
            # Previous frame (81 = LEFT arrow on Windows via cv2.waitKey)
            if current > 0:
                current -= 1

        # Also handle arrow keys via special key codes
        # On some Windows builds cv2.waitKeyEx returns different codes
        # We already handle the common ones above

        show_frame()

    cv2.destroyAllWindows()

    # Print summary
    annotated = sum(1 for a in annotations if _is_complete(a))
    with_obstacle = sum(1 for a in annotations if a.get("obstacle_present"))
    print(f"\n  Annotation Summary:")
    print(f"    Total frames    : {total}")
    print(f"    Fully annotated : {annotated}/{total}")
    print(f"    With obstacles  : {with_obstacle}")
    print(f"    Without obstacles: {total - with_obstacle}")

    # Class distribution
    class_counts: dict[str, int] = {}
    for a in annotations:
        cls = a.get("obstacle_class")
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    if class_counts:
        print(f"    Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"      {cls:10s} : {count}")


def main():
    parser = argparse.ArgumentParser(
        description="BlindAid frame annotation tool (OpenCV GUI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
First run extract_keyframes.py to create the frames and manifest.

Keyboard shortcuts:
    O           Toggle obstacle present (yes/no)
    1-6         Set obstacle class
    D           Cycle distance band
    L           Cycle location
    N / RIGHT   Next frame
    P / LEFT    Previous frame
    S           Save annotations
    Q           Quit and save
""",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help=f"Path to manifest.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--annotations", type=str, default=str(DEFAULT_ANNOTATIONS),
        help=f"Path to annotations JSON (default: {DEFAULT_ANNOTATIONS})",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    annotations_path = Path(args.annotations)

    # The base directory for resolving relative image paths is evaluation/
    base_dir = PROJECT_ROOT / "evaluation"

    # Load manifest
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print(f"  Run extract_keyframes.py first to create the manifest.")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    print(f"Loaded manifest with {len(manifest)} frames from {manifest_path}")

    # Load or initialize annotations
    existing = load_annotations(annotations_path)
    if existing:
        print(f"Resuming: loaded {len(existing)} existing annotations from {annotations_path}")
    else:
        print("Starting fresh annotations.")

    annotations = merge_annotations(manifest, existing)
    already_done = sum(1 for a in annotations if _is_complete(a))
    print(f"  {already_done}/{len(annotations)} frames already complete")
    print()
    print("Opening annotation window...")
    print("  (If the window doesn't appear, check your taskbar)")
    print()

    run_annotation_gui(manifest, annotations, base_dir, annotations_path)


if __name__ == "__main__":
    main()
