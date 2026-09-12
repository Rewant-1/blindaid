"""Phase 4 Audit: Test Adaptive Frame Processor behavior.

Verifies that AFP:
1. Processes more frequently when obstacles are close
2. Skips more when scene is stable
3. Handles mode-specific strategies correctly
4. Metrics are accumulated (not hardcoded)
5. Previous-frame depth is used correctly

Usage:
    python evaluation/test_adaptive.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from blindaid.core.adaptive_processor import AdaptiveFrameProcessor, MODE_CONFIG


def make_frame(seed: int = 0, brightness: int = 128) -> np.ndarray:
    """Create a simple test frame."""
    rng = np.random.RandomState(seed)
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    # Add some variation
    noise = rng.randint(-20, 20, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def make_depth(proximity: float) -> np.ndarray:
    """Create a synthetic depth map with given proximity level.

    proximity: 0.0 = nothing close, 1.0 = obstacle touching camera
    The center region will have this proximity value.
    """
    depth = np.full((480, 640), 0.2, dtype=np.float32)  # Background far away

    # Set center region to desired proximity
    h, w = 480, 640
    depth[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = proximity

    return depth


def test_proximity_affects_skip():
    """Test: closer obstacle → smaller skip (more frequent processing)."""
    print("TEST 1: Proximity affects skip rate")
    print("-" * 50)

    afp = AdaptiveFrameProcessor()
    frame = make_frame(0)

    # Far obstacle
    depth_far = make_depth(0.3)
    skip_far = afp.compute_skip("guardian", frame, depth_far)

    # Close obstacle
    afp_close = AdaptiveFrameProcessor()
    depth_close = make_depth(0.9)
    skip_close = afp_close.compute_skip("guardian", frame, depth_close)

    print(f"  Obstacle far (proximity=0.3):  skip={skip_far}")
    print(f"  Obstacle close (proximity=0.9): skip={skip_close}")

    assert skip_close < skip_far, (
        f"FAIL: Close obstacle skip ({skip_close}) should be < far ({skip_far})"
    )
    assert skip_close == MODE_CONFIG["guardian"]["min_skip"], (
        f"FAIL: Close obstacle should use min_skip={MODE_CONFIG['guardian']['min_skip']}"
    )

    print(f"  PASS: Close obstacle gets min_skip={skip_close}, far gets skip={skip_far}")
    print()
    return True


def test_stability_affects_reading():
    """Test: stable text → higher skip (less frequent OCR)."""
    print("TEST 2: Stability affects reading mode skip")
    print("-" * 50)

    # Same frame twice = high stability
    afp_stable = AdaptiveFrameProcessor()
    frame = make_frame(0, brightness=200)
    afp_stable.compute_skip("reading", frame)  # First call establishes baseline
    skip_stable = afp_stable.compute_skip("reading", frame)  # Same frame = stable

    # Different frames = low stability
    afp_changing = AdaptiveFrameProcessor()
    frame1 = make_frame(0, brightness=50)
    frame2 = make_frame(99, brightness=200)  # Very different frame
    afp_changing.compute_skip("reading", frame1)
    skip_changing = afp_changing.compute_skip("reading", frame2)

    print(f"  Stable scene:   skip={skip_stable}")
    print(f"  Changing scene:  skip={skip_changing}")

    assert skip_stable >= skip_changing, (
        f"FAIL: Stable skip ({skip_stable}) should be >= changing ({skip_changing})"
    )

    print(f"  PASS: Stable scene skips more ({skip_stable}) than changing ({skip_changing})")
    print()
    return True


def test_should_process_cycle():
    """Test: should_process correctly cycles through skip periods."""
    print("TEST 3: should_process skip/process cycle")
    print("-" * 50)

    afp = AdaptiveFrameProcessor()
    frame = make_frame(0)
    depth = make_depth(0.5)

    # Feed the depth as previous
    afp.update_depth(depth)

    process_pattern = []
    for i in range(30):
        should = afp.should_process("guardian", frame, depth)
        process_pattern.append(should)

    processed = sum(process_pattern)
    skipped = len(process_pattern) - processed

    print(f"  Over 30 frames: {processed} processed, {skipped} skipped")
    print(f"  Pattern (P=process, .=skip): {''.join('P' if p else '.' for p in process_pattern)}")

    # Should process at least some frames
    assert processed > 0, "FAIL: Must process at least one frame"
    # Should skip at least some frames
    assert skipped > 0, "FAIL: Should skip some frames for efficiency"
    # First frame should always be processed
    assert process_pattern[0] is True, "FAIL: First frame must be processed"

    print(f"  PASS: Correctly alternates between process and skip")
    print()
    return True


def test_metrics_are_real():
    """Test: metrics are accumulated from real computations, not hardcoded."""
    print("TEST 4: Metrics are computed, not hardcoded")
    print("-" * 50)

    afp = AdaptiveFrameProcessor()

    # Run through some frames
    for i in range(50):
        frame = make_frame(i % 5)
        depth = make_depth(0.3 + 0.01 * i)
        afp.should_process("guardian", frame, depth)

    metrics = afp.get_metrics("guardian")

    print(f"  frames_total:     {metrics['frames_total']}")
    print(f"  frames_processed: {metrics['frames_processed']}")
    print(f"  skip_ratio:       {metrics['skip_ratio']:.3f}")
    print(f"  cpu_savings_pct:  {metrics['cpu_savings_pct']:.1f}%")
    print(f"  avg_proximity:    {metrics['avg_proximity']:.3f}")
    print(f"  avg_stability:    {metrics['avg_stability']:.3f}")

    assert metrics["frames_total"] == 50, "FAIL: Should count all frames"
    assert 0 < metrics["frames_processed"] < 50, "FAIL: Should process some but not all"
    assert 0 < metrics["skip_ratio"] < 1, "FAIL: Skip ratio should be between 0 and 1"
    assert metrics["avg_proximity"] > 0, "FAIL: Proximity should be measured"
    assert metrics["avg_stability"] > 0, "FAIL: Stability should be measured"

    print(f"  PASS: All metrics computed from real measurements")
    print()
    return True


def test_previous_depth_used():
    """Test: proximity uses previous frame's depth, not requiring current."""
    print("TEST 5: Previous-frame depth avoids circular dependency")
    print("-" * 50)

    afp = AdaptiveFrameProcessor()
    frame = make_frame(0)

    # No previous depth → default moderate proximity
    skip_no_depth = afp.compute_skip("guardian", frame)
    print(f"  No depth data:      skip={skip_no_depth}")

    # Store a close obstacle depth
    close_depth = make_depth(0.9)
    afp.update_depth(close_depth)

    # Now compute skip — should use the stored depth
    skip_with_depth = afp.compute_skip("guardian", frame)
    print(f"  After close depth:  skip={skip_with_depth}")

    assert skip_with_depth <= skip_no_depth, (
        f"FAIL: After seeing close obstacle, skip should decrease"
    )

    print(f"  PASS: Previous depth correctly influences current skip decision")
    print()
    return True


def test_mode_boundaries():
    """Test: skip values stay within configured bounds."""
    print("TEST 6: Skip values stay within mode bounds")
    print("-" * 50)

    afp = AdaptiveFrameProcessor()

    for mode, config in MODE_CONFIG.items():
        min_skip = config["min_skip"]
        max_skip = config["max_skip"]

        for i in range(20):
            frame = make_frame(i)
            depth = make_depth(np.random.uniform(0, 1))
            skip = afp.compute_skip(mode, frame, depth)

            assert min_skip <= skip <= max_skip, (
                f"FAIL: {mode} skip={skip} outside [{min_skip}, {max_skip}]"
            )

        print(f"  {mode}: all skips within [{min_skip}, {max_skip}]  PASS")

    print()
    return True


def main():
    print("=" * 70)
    print("  PHASE 4 AUDIT: Adaptive Frame Processor")
    print("=" * 70)
    print()

    tests = [
        test_proximity_affects_skip,
        test_stability_affects_reading,
        test_should_process_cycle,
        test_metrics_are_real,
        test_previous_depth_used,
        test_mode_boundaries,
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
            print(f"  ERROR: {e}")
            failed += 1

    print("=" * 70)
    print(f"  RESULTS: {passed}/{passed + failed} tests passed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
