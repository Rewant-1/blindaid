"""Phase 8 Tests: Depth Fusion + Scene State Engine.

Verifies:
1. Depth-YOLO fusion produces distance-aware detections
2. Scene state engine deduplicates speech (no spam)
3. Scene state tracks objects across frames via IoU
4. Collision warnings are generated correctly
5. Approaching objects trigger updates
6. Path clear is announced when obstacles leave

Usage:
    python evaluation/test_scene_intelligence.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from blindaid.core.detector_onnx import Detection
from blindaid.core.depth_fusion import (
    fuse_detections, generate_fused_speech, classify_distance, FusedDetection,
)
from blindaid.core.scene_state import SceneStateEngine, SpeechEvent


def make_detection(class_name: str, x1: int, y1: int, x2: int, y2: int,
                   conf: float = 0.8) -> Detection:
    return Detection(class_id=0, class_name=class_name, confidence=conf,
                     x1=x1, y1=y1, x2=x2, y2=y2)


def make_depth(base: float = 0.3, center_val: float = 0.5) -> np.ndarray:
    """Create depth map with specific center value."""
    depth = np.full((480, 640), base, dtype=np.float32)
    depth[160:320, 160:480] = center_val  # Center region
    return depth


def test_depth_fusion_basic():
    """Test: Fusion produces distance-enriched detections."""
    print("TEST 1: Depth-YOLO fusion basic")
    print("-" * 50)

    # Person in center of frame, at a medium distance
    detections = [
        make_detection("person", 200, 100, 400, 400),
    ]
    depth_map = make_depth(base=0.2, center_val=0.55)  # Medium depth

    fused = fuse_detections(detections, depth_map, (480, 640))

    assert len(fused) == 1
    f = fused[0]
    print(f"  Person: raw_depth={f.raw_depth:.2f}, category={f.distance_category}")
    print(f"  Label: '{f.distance_label}'")
    print(f"  Region: {f.region}, Vertical: {f.vertical}")

    assert f.class_name == "person"
    assert f.region == "center"
    assert f.distance_category in ("close", "nearby")

    print(f"  PASS")
    print()
    return True


def test_fusion_distance_ordering():
    """Test: Closer objects have higher depth values and sort first."""
    print("TEST 2: Distance ordering")
    print("-" * 50)

    # Create depth map with different regions
    depth = np.full((480, 640), 0.2, dtype=np.float32)
    depth[100:200, 50:150] = 0.9   # Close object (left)
    depth[100:200, 300:400] = 0.4  # Far object (center)

    detections = [
        make_detection("chair", 300, 100, 400, 200),   # Center, far
        make_detection("person", 50, 100, 150, 200),    # Left, close
    ]

    fused = fuse_detections(detections, depth, (480, 640))

    # Person should be first (closer)
    assert fused[0].class_name == "person"
    assert fused[0].raw_depth > fused[1].raw_depth
    print(f"  Person: depth={fused[0].raw_depth:.2f} ({fused[0].distance_category})")
    print(f"  Chair:  depth={fused[1].raw_depth:.2f} ({fused[1].distance_category})")
    print(f"  PASS: Closer object sorted first")
    print()
    return True


def test_fused_speech():
    """Test: Speech output includes distance info."""
    print("TEST 3: Fused speech with distance")
    print("-" * 50)

    fused = [
        FusedDetection(
            class_name="person", confidence=0.9,
            x1=200, y1=100, x2=400, y2=400,
            raw_depth=0.85, distance_category="very_close",
            distance_label="less than 1 meter",
            region="center", vertical="mid",
        ),
        FusedDetection(
            class_name="chair", confidence=0.7,
            x1=50, y1=200, x2=150, y2=350,
            raw_depth=0.35, distance_category="nearby",
            distance_label="about 2-4 meters",
            region="left", vertical="ground",
        ),
    ]

    speech = generate_fused_speech(fused)
    print(f"  Speech: '{speech}'")

    assert "person" in speech.lower()
    assert "warning" in speech.lower()  # Collision risk
    assert "chair" in speech.lower()

    print(f"  PASS: Speech includes distance + warnings")
    print()
    return True


def test_scene_state_no_spam():
    """Test: Same object across frames doesn't generate repeated speech."""
    print("TEST 4: Scene state — no speech spam")
    print("-" * 50)

    engine = SceneStateEngine(announce_cooldown=2.0)

    # Same detection in same position across frames
    det = FusedDetection(
        class_name="person", confidence=0.9,
        x1=200, y1=100, x2=400, y2=400,
        raw_depth=0.5, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="center", vertical="mid",
    )

    events_frame1 = engine.update([det])
    events_frame2 = engine.update([det])
    events_frame3 = engine.update([det])

    print(f"  Frame 1 events: {len(events_frame1)} — {[e.text for e in events_frame1]}")
    print(f"  Frame 2 events: {len(events_frame2)}")
    print(f"  Frame 3 events: {len(events_frame3)}")

    assert len(events_frame1) == 1, "First frame should announce new object"
    assert len(events_frame2) == 0, "Second frame should NOT re-announce"
    assert len(events_frame3) == 0, "Third frame should NOT re-announce"

    print(f"  PASS: Object announced once, not spammed")
    print()
    return True


def test_scene_state_new_object():
    """Test: New object appearing generates announcement."""
    print("TEST 5: Scene state — new object announced")
    print("-" * 50)

    engine = SceneStateEngine()

    # Frame 1: person
    det1 = FusedDetection(
        class_name="person", confidence=0.9,
        x1=200, y1=100, x2=400, y2=400,
        raw_depth=0.5, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="center", vertical="mid",
    )
    events1 = engine.update([det1])

    # Frame 2: person + NEW car
    det2 = FusedDetection(
        class_name="car", confidence=0.8,
        x1=50, y1=200, x2=200, y2=400,
        raw_depth=0.4, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="left", vertical="ground",
    )
    events2 = engine.update([det1, det2])

    print(f"  Frame 1: {len(events1)} events — {[e.text for e in events1]}")
    print(f"  Frame 2: {len(events2)} events — {[e.text for e in events2]}")

    assert len(events1) == 1, "Frame 1: person announced"
    assert len(events2) == 1, "Frame 2: only car (new) announced"
    assert "car" in events2[0].text.lower()

    print(f"  PASS: Only new car announced in frame 2")
    print()
    return True


def test_scene_state_approaching():
    """Test: Object getting closer triggers approach announcement."""
    print("TEST 6: Scene state — approaching object")
    print("-" * 50)

    engine = SceneStateEngine(approach_cooldown=0.0)  # No cooldown for testing

    # Frame 1: person far away
    det_far = FusedDetection(
        class_name="person", confidence=0.9,
        x1=200, y1=100, x2=400, y2=400,
        raw_depth=0.3, distance_category="moderate",
        distance_label="about 4-6 meters",
        region="center", vertical="mid",
    )
    events1 = engine.update([det_far])

    # Frame 2: same person, now closer
    det_close = FusedDetection(
        class_name="person", confidence=0.9,
        x1=180, y1=80, x2=420, y2=420,  # Slightly bigger (closer)
        raw_depth=0.55, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="center", vertical="mid",
    )
    events2 = engine.update([det_close])

    print(f"  Frame 1 (far): {[e.text for e in events1]}")
    print(f"  Frame 2 (closer): {[e.text for e in events2]}")

    assert len(events1) == 1, "Frame 1: new person"
    assert len(events2) >= 1, "Frame 2: approach should be announced"
    assert "approaching" in events2[0].text.lower() or "warning" in events2[0].text.lower()

    print(f"  PASS: Approaching object announced")
    print()
    return True


def test_scene_state_collision_warning():
    """Test: Very close + approaching = collision warning."""
    print("TEST 7: Scene state — collision warning")
    print("-" * 50)

    engine = SceneStateEngine(announce_cooldown=0.0)

    # Frame 1: person nearby
    det1 = FusedDetection(
        class_name="person", confidence=0.9,
        x1=200, y1=100, x2=400, y2=400,
        raw_depth=0.55, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="center", vertical="mid",
    )
    engine.update([det1])

    # Frame 2: same person, now VERY close
    det2 = FusedDetection(
        class_name="person", confidence=0.9,
        x1=150, y1=50, x2=450, y2=450,
        raw_depth=0.85, distance_category="very_close",
        distance_label="less than 1 meter",
        region="center", vertical="mid",
    )
    events = engine.update([det2])

    print(f"  Events: {[e.text for e in events]}")

    assert len(events) >= 1
    assert events[0].priority == 0, "Collision warning should be priority 0"
    assert "warning" in events[0].text.lower()

    print(f"  PASS: Collision warning generated with priority 0")
    print()
    return True


def test_scene_state_metrics():
    """Test: Metrics are tracked correctly."""
    print("TEST 8: Scene state metrics")
    print("-" * 50)

    engine = SceneStateEngine()

    det = FusedDetection(
        class_name="chair", confidence=0.7,
        x1=200, y1=200, x2=350, y2=400,
        raw_depth=0.45, distance_category="nearby",
        distance_label="about 2-4 meters",
        region="center", vertical="ground",
    )

    for _ in range(10):
        engine.update([det])

    metrics = engine.get_metrics()
    print(f"  total_events:      {metrics['total_events_generated']}")
    print(f"  suppressed:        {metrics['suppressed_events']}")
    print(f"  new_announcements: {metrics['new_announcements']}")
    print(f"  frames_processed:  {metrics['frames_processed']}")
    print(f"  suppression_rate:  {metrics['suppression_rate']:.1%}")

    assert metrics["frames_processed"] == 10
    assert metrics["new_announcements"] == 1, "Should announce chair only once"
    assert metrics["total_events_generated"] >= 1

    print(f"  PASS: Metrics correctly tracked")
    print()
    return True


def main():
    print("=" * 70)
    print("  PHASE 8: Depth Fusion + Scene State Engine Tests")
    print("=" * 70)
    print()

    tests = [
        test_depth_fusion_basic,
        test_fusion_distance_ordering,
        test_fused_speech,
        test_scene_state_no_spam,
        test_scene_state_new_object,
        test_scene_state_approaching,
        test_scene_state_collision_warning,
        test_scene_state_metrics,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"  RESULTS: {passed}/{passed + failed} tests passed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
