import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np


# --------- Utility functions ---------

def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def euclid_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def hand_center(lm) -> Tuple[float, float]:
    w = lm[0]
    return (w.x, w.y)


def is_finger_extended_3d(landmarks, finger_indices: List[int], threshold: float = 0.8) -> bool:
    points = [
        np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])
        for i in finger_indices
    ]
    v1 = _safe_unit(points[1] - points[0])
    v2 = _safe_unit(points[2] - points[1])
    v3 = _safe_unit(points[3] - points[2])
    return float(np.dot(v1, v2)) > threshold and float(np.dot(v2, v3)) > threshold


# --------- Tracking and smoothing ---------

@dataclass
class HandTrack:
    id: int
    center: Tuple[float, float]
    last_update: float


class SimpleHandTracker:
    """Assigns stable IDs by nearest-neighbor association across frames."""

    def __init__(self, max_lost_time: float = 1.0, match_threshold: float = 0.2):
        self._next_id = 1
        self._tracks: Dict[int, HandTrack] = {}
        self._max_lost_time = max_lost_time
        self._match_threshold = match_threshold

    def update(self, centers: List[Tuple[float, float]], now: float) -> Dict[int, Tuple[float, float]]:
        # Mark all tracks as unmatched initially
        unmatched_tracks: Set[int] = set(self._tracks.keys())
        assignments: Dict[int, Tuple[float, float]] = {}

        # Greedy nearest-neighbor matching
        for c in centers:
            best_track = None
            best_dist = 999.0
            for tid in list(unmatched_tracks):
                d = euclid_2d(self._tracks[tid].center, c)
                if d < best_dist:
                    best_dist = d
                    best_track = tid
            if best_track is not None and best_dist < self._match_threshold:
                # Assign
                self._tracks[best_track].center = c
                self._tracks[best_track].last_update = now
                assignments[best_track] = c
                unmatched_tracks.discard(best_track)
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = HandTrack(id=tid, center=c, last_update=now)
                assignments[tid] = c

        # Cull stale tracks
        for tid in list(self._tracks.keys()):
            if now - self._tracks[tid].last_update > self._max_lost_time:
                del self._tracks[tid]

        return assignments


class LandmarkSmoother:
    """Exponential moving average per hand ID for landmark smoothing."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.prev: Dict[int, List[Tuple[float, float, float]]] = {}

    def smooth(self, hand_id: int, lm) -> List[Tuple[float, float, float]]:
        pts = [(p.x, p.y, p.z) for p in lm]
        if hand_id not in self.prev:
            self.prev[hand_id] = pts
            return pts
        sm = []
        for (x, y, z), (px, py, pz) in zip(pts, self.prev[hand_id]):
            sx = self.alpha * x + (1 - self.alpha) * px
            sy = self.alpha * y + (1 - self.alpha) * py
            sz = self.alpha * z + (1 - self.alpha) * pz
            sm.append((sx, sy, sz))
        self.prev[hand_id] = sm
        return sm


# --------- Gesture and Motion base classes ---------

class Gesture:
    name: str

    def __init__(self, name: str, sound_path: Optional[str] = None, volume: float = 1.0):
        self.name = name
        # Optional per-gesture sound metadata (played by the app)
        self.sound_path: Optional[str] = sound_path
        # Clamp volume to [0.0, 1.0]
        self.volume: float = max(0.0, min(1.0, volume))

    def detect(self, lm_list: List[Tuple[float, float, float]]) -> bool:
        raise NotImplementedError


class Motion:
    name: str

    def __init__(self, name: str, sound_path: Optional[str] = None, volume: float = 1.0):
        self.name = name
        # Optional per-motion sound metadata (played by the app)
        self.sound_path: Optional[str] = sound_path
        # Clamp volume to [0.0, 1.0]
        self.volume: float = max(0.0, min(1.0, volume))

    def update(self, hand_id: int, lm_list: List[Tuple[float, float, float]], now: float, gesture_hits: Set[str]) -> bool:
        """Return True when motion is triggered for this hand in this frame."""
        raise NotImplementedError


# --------- Example gestures and motions ---------

FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _is_ext(lm_list, name: str) -> bool:
    # Convert back to a simple structure expected by is_finger_extended_3d
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    landmarks = [P(*p) for p in lm_list]
    return is_finger_extended_3d(landmarks, FINGER_INDICES[name])


def _scale(lm_list) -> float:
    a = lm_list[5]
    b = lm_list[17]
    return max(euclid_2d((a[0], a[1]), (b[0], b[1])), 1e-3)


class Symbol6Gesture(Gesture):
    """Thumb and index forming circle, other 3 fingers extended."""
    def __init__(self, pinch_threshold: float = 0.60, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("six", sound_path=sound_path, volume=volume)
        self.pinch_threshold = pinch_threshold

    def detect(self, lm_list) -> bool:
        # Thumb-index circle (like OK sign)
        thumb_tip = (lm_list[4][0], lm_list[4][1])
        index_tip = (lm_list[8][0], lm_list[8][1])
        pinch = euclid_2d(thumb_tip, index_tip) / _scale(lm_list)
        
        # Other 3 fingers extended
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")
        
        return pinch < self.pinch_threshold and middle_ext and ring_ext and pinky_ext


class Symbol7Gesture(Gesture):
    def __init__(self, down_cos: float = 0.6, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("seven", sound_path=sound_path, volume=volume)
        self.down_cos = down_cos

    def detect(self, lm_list) -> bool:
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")

        mcp = np.array([lm_list[5][0], lm_list[5][1]])
        tip = np.array([lm_list[8][0], lm_list[8][1]])
        v = _safe_unit(tip - mcp)
        down = np.array([0.0, 1.0])
        downward_enough = float(np.dot(v, down)) > self.down_cos

        return thumb_ext and index_ext and downward_enough and (not middle_ext) and (not ring_ext) and (not pinky_ext)


class GunPoseGesture(Gesture):
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("gun", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        # Gun: thumb and index extended, middle/ring/pinky curled
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")
        
        # More specific: thumb+index extended, middle+ring+pinky NOT extended
        return thumb_ext and index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext)


class OpenPalmGesture(Gesture):
    """All five fingers extended (open hand)."""
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("palm", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        return (
            _is_ext(lm_list, "thumb")
            and _is_ext(lm_list, "index")
            and _is_ext(lm_list, "middle")
            and _is_ext(lm_list, "ring")
            and _is_ext(lm_list, "pinky")
        )


class FistGesture(Gesture):
    """All fingers curled (fist)."""
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("fist", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        return (
            not _is_ext(lm_list, "thumb")
            and not _is_ext(lm_list, "index")
            and not _is_ext(lm_list, "middle")
            and not _is_ext(lm_list, "ring")
            and not _is_ext(lm_list, "pinky")
        )


class PeaceSignGesture(Gesture):
    """Index and middle extended, others curled."""
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("peace", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        return (
            _is_ext(lm_list, "index")
            and _is_ext(lm_list, "middle")
            and not _is_ext(lm_list, "ring")
            and not _is_ext(lm_list, "pinky")
        )


class ThumbsUpGesture(Gesture):
    """Only thumb extended, others curled."""
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("thumbs_up", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        return (
            _is_ext(lm_list, "thumb")
            and not _is_ext(lm_list, "index")
            and not _is_ext(lm_list, "middle")
            and not _is_ext(lm_list, "ring")
            and not _is_ext(lm_list, "pinky")
        )


class OKSignGesture(Gesture):
    """Thumb and index tip touching (circle), other fingers extended."""
    def __init__(self, pinch_threshold: float = 0.2, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("ok", sound_path=sound_path, volume=volume)
        self.pinch_threshold = pinch_threshold

    def detect(self, lm_list) -> bool:
        thumb_tip = (lm_list[4][0], lm_list[4][1])
        index_tip = (lm_list[8][0], lm_list[8][1])
        pinch = euclid_2d(thumb_tip, index_tip) / _scale(lm_list)
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")
        return pinch < self.pinch_threshold and middle_ext and ring_ext and pinky_ext
    

class MiddleFingerGesture(Gesture):
    """Only middle finger extended."""
    def __init__(self, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("middle_finger", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        print(not _is_ext(lm_list, "index")
            and _is_ext(lm_list, "middle")
            and not _is_ext(lm_list, "ring")
            and not _is_ext(lm_list, "pinky")
        )
        return (
            not _is_ext(lm_list, "index")
            and _is_ext(lm_list, "middle")
            and not _is_ext(lm_list, "ring")
            and not _is_ext(lm_list, "pinky")
        )


class RecoilMotion(Motion):
    def __init__(self, movement_threshold: float = 0.05, gate_gesture: str = "gun", cooldown_s: float = 0.4, sound_path: Optional[str] = None, volume: float = 1.0):
        super().__init__("recoil", sound_path=sound_path, volume=volume)
        self.movement_threshold = movement_threshold
        self.gate_gesture = gate_gesture
        self.cooldown_s = cooldown_s
        self.state: Dict[int, Dict[str, Optional[Tuple[float, float]]]] = {}
        self.last_trigger: Dict[int, float] = {}

    def update(self, hand_id: int, lm_list, now: float, gesture_hits: Set[str]) -> bool:
        wrist = (lm_list[0][0], lm_list[0][1])
        st = self.state.setdefault(hand_id, {"prev": None, "ready": False})

        if self.gate_gesture in gesture_hits:
            if not st["ready"]:
                st["ready"] = True
                st["prev"] = wrist
                return False
            else:
                prev = st["prev"]
                movement = euclid_2d(wrist, prev) if prev is not None else 0.0
                st["prev"] = wrist
                if movement > self.movement_threshold:
                    last = self.last_trigger.get(hand_id, 0.0)
                    if now - last > self.cooldown_s:
                        self.last_trigger[hand_id] = now
                        return True
        else:
            st["ready"] = False
            st["prev"] = None
        return False


# --------- Proximity rule and engine ---------

@dataclass
class ProximityRule:
    a: str
    b: str
    threshold: float = 0.18
    active_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    cooldown_s: float = 0.4
    last_trigger: float = 0.0
    # Optional per-rule sound metadata
    sound_path: Optional[str] = None
    volume: float = 1.0

    def __post_init__(self):
        # Clamp volume
        self.volume = max(0.0, min(1.0, self.volume))

    def check(self, detections: Dict[int, Set[str]], centers: Dict[int, Tuple[float, float]], now: float) -> Tuple[bool, Optional[Tuple[int, int]]]:
        a_ids = [hid for hid, g in detections.items() if self.a in g]
        b_ids = [hid for hid, g in detections.items() if self.b in g]
        best = None
        best_d = 999.0
        for i in a_ids:
            for j in b_ids:
                # If matching the same gesture for both sides, ensure two distinct hands
                if self.a == self.b and i == j:
                    continue
                d = euclid_2d(centers[i], centers[j])
                if d < best_d:
                    best_d = d
                    # Store pairs in canonical order to avoid (i,j)/(j,i) duplicates
                    best = (min(i, j), max(i, j))
        if best is None or best_d >= self.threshold:
            # Reset active pairs that are no longer close
            self.active_pairs.clear()
            return False, None
        # Debounce by active pair and cooldown
        if best not in self.active_pairs and (now - self.last_trigger) > self.cooldown_s:
            self.active_pairs.add(best)
            self.last_trigger = now
            return True, best
        return False, best


@dataclass
class GestureTriggerRule:
    g: str
    # Fire once per hand while the gesture remains active (edge-triggered)
    active_hands: Set[int] = field(default_factory=set)

    def check(self, detections: Dict[int, Set[str]], now: float) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        current_active: Set[int] = set()
        for hid, gestures in detections.items():
            if self.g in gestures:
                current_active.add(hid)
                if hid not in self.active_hands:
                    # Rising edge: just became active
                    events.append({"type": self.g, "hand_id": hid})
        # Update latch state: keep only those still active
        self.active_hands = current_active
        return events


class GestureEngine:
    def __init__(self, smoother_alpha: float = 0.5):
        self.tracker = SimpleHandTracker()
        self.smoother = LandmarkSmoother(alpha=smoother_alpha)
        self.gestures: List[Gesture] = []
        self.motions: List[Motion] = []
        self.rules: List[ProximityRule] = []
        self.gesture_rules: List[GestureTriggerRule] = []

    def register_gesture(self, gesture: Gesture):
        self.gestures.append(gesture)

    def register_motion(self, motion: Motion):
        self.motions.append(motion)

    def register_rule(self, rule: ProximityRule):
        self.rules.append(rule)

    def register_gesture_rule(self, rule: GestureTriggerRule):
        self.gesture_rules.append(rule)

    def process(self, multi_hand_landmarks) -> Dict[str, List]:
        now = time.time()
        results: Dict[str, List] = {"events": [], "overlays": []}

        if not multi_hand_landmarks:
            return results

        # Get centers and assign stable IDs
        raw_centers = [hand_center(h.landmark) for h in multi_hand_landmarks]
        centers_by_id = self.tracker.update(raw_centers, now)

        # Build map from frame index to stable id by nearest center
        idx_to_id: Dict[int, int] = {}
        remaining_ids = set(centers_by_id.keys())
        for idx, h in enumerate(multi_hand_landmarks):
            c = hand_center(h.landmark)
            # nearest
            best_id = None
            best_d = 999.0
            for hid in remaining_ids:
                d = euclid_2d(centers_by_id[hid], c)
                if d < best_d:
                    best_d = d
                    best_id = hid
            if best_id is not None:
                idx_to_id[idx] = best_id
                remaining_ids.discard(best_id)

        # Smooth landmarks and run gesture detectors
        detections: Dict[int, Set[str]] = {}
        smoothed: Dict[int, List[Tuple[float, float, float]]] = {}
        for idx, hand in enumerate(multi_hand_landmarks):
            hid = idx_to_id.get(idx)
            if hid is None:
                continue
            sm = self.smoother.smooth(hid, hand.landmark)
            smoothed[hid] = sm
            hits: Set[str] = set()
            for g in self.gestures:
                if g.detect(sm):
                    hits.add(g.name)
            detections[hid] = hits

        # Motions
        for hid, sm in smoothed.items():
            hits = detections.get(hid, set())
            for m in self.motions:
                if m.update(hid, sm, now, hits):
                    results["events"].append({"type": m.name, "hand_id": hid})

        # Proximity rules
        for rule in self.rules:
            fired, pair = rule.check(detections, centers_by_id, now)
            if fired and pair is not None:
                results["events"].append({"type": f"proximity:{rule.a}+{rule.b}", "pair": pair})

        # Single-gesture trigger rules
        for gr in self.gesture_rules:
            evts = gr.check(detections, now)
            if evts:
                results["events"].extend(evts)

        # For overlays/debugging
        results["overlays"].append({
            "detections": detections,
            "centers": centers_by_id,
        })

        return results
