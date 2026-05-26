"""Scene State Engine — intelligent event-based speech output.

Prevents: "Person. Person. Person. Person."
Produces: "New person ahead, about 2 meters."
Then only updates when something meaningful changes.

Uses simple IoU tracking between consecutive processed frames.
No ByteTrack dependency — keeps it explainable for the paper.

Phase 8 of the research paper implementation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from blindaid.core.depth_fusion import FusedDetection, DISTANCE_SPEECH

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object across frames."""
    track_id: int
    class_name: str
    last_bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    last_distance_category: str
    last_raw_depth: float
    last_region: str

    # Timing
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_announced: float = 0.0

    # Counters
    frames_seen: int = 0
    announce_count: int = 0

    # State change tracking
    prev_distance_category: str = ""
    is_approaching: bool = False


@dataclass
class SpeechEvent:
    """A speech event to be spoken."""
    text: str
    priority: int  # 0 = highest (collision), 1 = high (new object), 2 = info
    timestamp: float = 0.0


class SceneStateEngine:
    """Maintains scene understanding across frames.

    Core logic:
    1. Match current detections to tracked objects via IoU
    2. Identify NEW objects (not matched to any track)
    3. Identify APPROACHING objects (distance category changed toward closer)
    4. Identify DEPARTED objects (not seen for N frames)
    5. Generate speech ONLY for meaningful events

    Speech events (prioritized):
    - COLLISION WARNING: Object very close and approaching (priority 0)
    - NEW OBJECT: First time seeing this object (priority 1)
    - APPROACHING: Object moving closer (distance category change) (priority 1)
    - PATH CLEAR: All obstacles gone (priority 2)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        announce_cooldown: float = 3.0,
        departure_frames: int = 5,
        approach_cooldown: float = 4.0,
    ):
        self._tracks: dict[int, TrackedObject] = {}
        self._next_id: int = 0
        self._iou_threshold = iou_threshold
        self._announce_cooldown = announce_cooldown
        self._departure_frames = departure_frames
        self._approach_cooldown = approach_cooldown
        self._frame_count = 0
        self._last_had_obstacles = False

        # Metrics for paper
        self._total_events = 0
        self._suppressed_events = 0
        self._collision_warnings = 0
        self._new_announcements = 0
        self._approach_announcements = 0

    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _match_detections(
        self,
        detections: list[FusedDetection],
    ) -> tuple[dict[int, FusedDetection], list[FusedDetection]]:
        """Match current detections to existing tracks using IoU.

        Returns:
            matched: {track_id: detection}
            unmatched: [detections not matched to any track]
        """
        if not self._tracks or not detections:
            return {}, list(detections)

        matched: dict[int, FusedDetection] = {}
        used_detections: set[int] = set()

        # Greedy matching: for each track, find best matching detection
        for track_id, track in self._tracks.items():
            best_iou = 0.0
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                if det.class_name != track.class_name:
                    continue

                iou = self._compute_iou(
                    track.last_bbox,
                    (det.x1, det.y1, det.x2, det.y2),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self._iou_threshold and best_det_idx >= 0:
                matched[track_id] = detections[best_det_idx]
                used_detections.add(best_det_idx)

        unmatched = [
            det for i, det in enumerate(detections) if i not in used_detections
        ]

        return matched, unmatched

    def _is_approaching(self, old_category: str, new_category: str) -> bool:
        """Check if object is getting closer."""
        order = ["far", "moderate", "nearby", "close", "very_close"]
        try:
            old_idx = order.index(old_category)
            new_idx = order.index(new_category)
            return new_idx > old_idx
        except ValueError:
            return False

    def update(self, fused_detections: list[FusedDetection]) -> list[SpeechEvent]:
        """Update scene state with new detections and generate speech events.

        This is the core method. Call it every PROCESSED frame (not skipped frames).

        Args:
            fused_detections: Current frame's fused detections (depth + YOLO)

        Returns:
            List of speech events to speak (may be empty if nothing changed)
        """
        self._frame_count += 1
        now = time.time()
        events: list[SpeechEvent] = []

        # Step 1: Match detections to tracks
        matched, unmatched = self._match_detections(fused_detections)

        # Step 2: Update matched tracks
        for track_id, det in matched.items():
            track = self._tracks[track_id]
            old_category = track.last_distance_category

            track.last_bbox = (det.x1, det.y1, det.x2, det.y2)
            track.prev_distance_category = old_category
            track.last_distance_category = det.distance_category
            track.last_raw_depth = det.raw_depth
            track.last_region = det.region
            track.last_seen = now
            track.frames_seen += 1

            # Check if approaching
            approaching = self._is_approaching(old_category, det.distance_category)
            track.is_approaching = approaching

            # COLLISION WARNING: very close and approaching
            if det.is_collision_risk and approaching:
                if now - track.last_announced > self._announce_cooldown:
                    dist_speech = DISTANCE_SPEECH.get(det.distance_category, "")
                    if det.region == "center":
                        text = f"Warning! {det.class_name} ahead, {dist_speech}"
                    else:
                        text = f"Warning! {det.class_name} on the {det.region}, {dist_speech}"

                    events.append(SpeechEvent(text=text, priority=0, timestamp=now))
                    track.last_announced = now
                    track.announce_count += 1
                    self._collision_warnings += 1
                else:
                    self._suppressed_events += 1

            # APPROACH ANNOUNCEMENT: distance category changed (getting closer)
            elif approaching and not det.is_collision_risk:
                if now - track.last_announced > self._approach_cooldown:
                    dist_speech = DISTANCE_SPEECH.get(det.distance_category, "")
                    text = f"{det.class_name} approaching, now {dist_speech}"
                    events.append(SpeechEvent(text=text, priority=1, timestamp=now))
                    track.last_announced = now
                    track.announce_count += 1
                    self._approach_announcements += 1
                else:
                    self._suppressed_events += 1

        # Step 3: Create new tracks for unmatched detections
        for det in unmatched:
            track_id = self._next_id
            self._next_id += 1

            self._tracks[track_id] = TrackedObject(
                track_id=track_id,
                class_name=det.class_name,
                last_bbox=(det.x1, det.y1, det.x2, det.y2),
                last_distance_category=det.distance_category,
                last_raw_depth=det.raw_depth,
                last_region=det.region,
                first_seen=now,
                last_seen=now,
                frames_seen=1,
            )

            # NEW OBJECT announcement
            dist_speech = DISTANCE_SPEECH.get(det.distance_category, "")
            if det.region == "center":
                text = f"{det.class_name} ahead, {dist_speech}"
            else:
                text = f"{det.class_name} on the {det.region}, {dist_speech}"

            # Collision risk on first sight = immediate warning
            priority = 0 if det.is_collision_risk else 1
            if det.is_collision_risk:
                text = f"Warning! {text}"

            events.append(SpeechEvent(text=text, priority=priority, timestamp=now))
            self._tracks[track_id].last_announced = now
            self._tracks[track_id].announce_count = 1
            self._new_announcements += 1

        # Step 4: Remove departed tracks
        departed_ids = []
        for track_id, track in self._tracks.items():
            frames_since_seen = self._frame_count - track.frames_seen
            if now - track.last_seen > 2.0:  # Not seen for 2 seconds
                departed_ids.append(track_id)

        for track_id in departed_ids:
            del self._tracks[track_id]

        # Step 5: Path clear announcement
        has_obstacles = any(
            t.last_distance_category in ("very_close", "close", "nearby")
            for t in self._tracks.values()
        )
        if self._last_had_obstacles and not has_obstacles and not fused_detections:
            events.append(SpeechEvent(text="Path clear.", priority=2, timestamp=now))

        self._last_had_obstacles = has_obstacles

        # Track total events
        self._total_events += len(events)

        # Sort by priority (lower = more important)
        events.sort(key=lambda e: e.priority)

        return events

    def get_active_tracks(self) -> list[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self._tracks.values())

    def get_metrics(self) -> dict:
        """Get scene state metrics for paper reporting."""
        return {
            "total_events_generated": self._total_events,
            "suppressed_events": self._suppressed_events,
            "collision_warnings": self._collision_warnings,
            "new_announcements": self._new_announcements,
            "approach_announcements": self._approach_announcements,
            "active_tracks": len(self._tracks),
            "frames_processed": self._frame_count,
            "suppression_rate": (
                self._suppressed_events / (self._total_events + self._suppressed_events)
                if (self._total_events + self._suppressed_events) > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all state."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_count = 0
        self._total_events = 0
        self._suppressed_events = 0
        self._collision_warnings = 0
        self._new_announcements = 0
        self._approach_announcements = 0
        self._last_had_obstacles = False


__all__ = ["SceneStateEngine", "TrackedObject", "SpeechEvent"]
