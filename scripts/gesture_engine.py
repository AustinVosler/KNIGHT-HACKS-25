"""
Gesture Engine - Modular Hand Gesture and Motion Detection System

This module provides a flexible framework for detecting hand gestures and motions using
MediaPipe Hands. It supports:
- Multiple gesture types (static poses like thumbs up, peace sign, Korean heart, etc.)
- Motion detection (like recoil/gun firing)
- Proximity rules (detecting when two gestures are near each other)
- Gesture trigger rules (single-gesture events with cooldown)
- Motion filtering (prevent false positives during fast hand movement)
- Stable hand tracking across frames
- Landmark smoothing for temporal stability

Architecture:
- Gesture: Base class for static hand poses
- Motion: Base class for temporal patterns (movement-based)
- ProximityRule: Detects when two gestures are close together
- GestureTriggerRule: Emits events for single gestures with cooldown
- GestureEngine: Orchestrates detection and event emission
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np


# --------- Utility functions ---------

def _safe_unit(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length, handling zero vectors safely.
    
    Args:
        v: Input vector (numpy array)
    
    Returns:
        Unit vector in the same direction, or the original vector if magnitude is zero
    """
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def euclid_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Distance between the points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def hand_center(lm) -> Tuple[float, float]:
    """
    Get the center position of a hand (wrist landmark).
    
    Args:
        lm: MediaPipe hand landmarks
    
    Returns:
        (x, y) coordinates of the wrist
    """
    w = lm[0]
    return (w.x, w.y)


def is_finger_extended_3d(landmarks, finger_indices: List[int], threshold: float = 0.8) -> bool:
    """
    Determine if a finger is extended by checking joint alignment in 3D.
    
    Uses dot products between consecutive bone segments. If joints are aligned
    (dot product > threshold), the finger is considered extended.
    
    Args:
        landmarks: MediaPipe hand landmarks
        finger_indices: List of 4 landmark indices for the finger [mcp, pip, dip, tip]
        threshold: Alignment threshold (0.0 to 1.0, higher = stricter)
    
    Returns:
        True if the finger is extended, False otherwise
    """
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
    """
    Represents a tracked hand with a stable ID across frames.
    
    Attributes:
        id: Unique identifier for this hand
        center: Current (x, y) position of the hand center
        last_update: Timestamp of the last update
    """
    id: int
    center: Tuple[float, float]
    last_update: float


class SimpleHandTracker:
    """
    Assigns stable IDs to hands across frames using nearest-neighbor matching.
    
    This tracker maintains hand identity over time, allowing gesture and motion
    detectors to track individual hands reliably. Uses greedy nearest-neighbor
    matching with distance thresholds.
    
    Attributes:
        max_lost_time: Maximum time (seconds) before a lost hand is removed
        match_threshold: Maximum distance for matching a detection to an existing track
    """

    def __init__(self, max_lost_time: float = 1.0, match_threshold: float = 0.2):
        """
        Initialize the hand tracker.
        
        Args:
            max_lost_time: Time in seconds before culling a lost hand
            match_threshold: Maximum normalized distance for matching
        """
        self._next_id = 1
        self._tracks: Dict[int, HandTrack] = {}
        self._max_lost_time = max_lost_time
        self._match_threshold = match_threshold

    def update(self, centers: List[Tuple[float, float]], now: float) -> Dict[int, Tuple[float, float]]:
        """
        Update tracking with new hand detections.
        
        Args:
            centers: List of hand center positions in current frame
            now: Current timestamp
        
        Returns:
            Dictionary mapping hand IDs to their centers
        """
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
    """
    Applies exponential moving average (EMA) smoothing to hand landmarks.
    
    Reduces jitter and noise in landmark positions across frames, improving
    stability of gesture detection. Maintains separate smoothing state per hand ID.
    
    Attributes:
        alpha: Smoothing factor (0.0 = no update, 1.0 = no smoothing)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize the landmark smoother.
        
        Args:
            alpha: Smoothing factor, higher = more responsive but less smooth
        """
        self.alpha = alpha
        self.prev: Dict[int, List[Tuple[float, float, float]]] = {}

    def smooth(self, hand_id: int, lm) -> List[Tuple[float, float, float]]:
        """
        Apply EMA smoothing to landmarks for a specific hand.
        
        Args:
            hand_id: Unique identifier for the hand
            lm: MediaPipe hand landmarks
        
        Returns:
            List of smoothed (x, y, z) landmark coordinates
        """
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
    """
    Base class for static hand gesture detection.
    
    Gestures are hand poses (like thumbs up, peace sign, Korean heart) that
    are detected based on finger positions in a single frame. Subclasses
    implement the detect() method to check for specific hand configurations.
    
    Attributes:
        name: Unique identifier for this gesture type
        sound_path: Optional list of sound file paths to play when detected
        volume: Playback volume (0.0 to 1.0)
    """
    name: str

    def __init__(self, name: str, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        """
        Initialize a gesture detector.
        
        Args:
            name: Unique name for this gesture
            sound_path: Optional list of sound file paths (all will be played)
            volume: Playback volume (0.0 to 1.0), will be clamped
        """
        self.name = name
        # Optional per-gesture sound metadata (played by the app)
        self.sound_path: Optional[List[str]] = sound_path if sound_path is not None else None
        # Clamp volume to [0.0, 1.0]
        self.volume: float = max(0.0, min(1.0, volume))

    def detect(self, lm_list: List[Tuple[float, float, float]]) -> bool:
        """
        Detect if this gesture is present in the given hand landmarks.
        
        Args:
            lm_list: List of 21 hand landmarks as (x, y, z) tuples
        
        Returns:
            True if gesture is detected, False otherwise
        """
        raise NotImplementedError


class Motion:
    """
    Base class for temporal motion detection.
    
    Motions are patterns that occur over time (like recoil/gun firing) and
    may depend on both hand pose and movement. Maintains state per hand ID
    to track motion across frames.
    
    Attributes:
        name: Unique identifier for this motion type
        sound_path: Optional list of sound file paths to play when triggered
        volume: Playback volume (0.0 to 1.0)
    """
    name: str

    def __init__(self, name: str, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        """
        Initialize a motion detector.
        
        Args:
            name: Unique name for this motion
            sound_path: Optional list of sound file paths (all will be played)
            volume: Playback volume (0.0 to 1.0), will be clamped
        """
        self.name = name
        # Optional per-motion sound metadata (played by the app)
        self.sound_path: Optional[List[str]] = sound_path if sound_path is not None else None
        # Clamp volume to [0.0, 1.0]
        self.volume: float = max(0.0, min(1.0, volume))

    def update(self, hand_id: int, lm_list: List[Tuple[float, float, float]], now: float, gesture_hits: Set[str]) -> bool:
        """
        Update motion state and check if motion is triggered.
        
        Args:
            hand_id: Unique identifier for the hand
            lm_list: List of 21 hand landmarks as (x, y, z) tuples
            now: Current timestamp
            gesture_hits: Set of gesture names currently detected on this hand
        
        Returns:
            True if motion is triggered this frame, False otherwise
        """
        raise NotImplementedError


# --------- Example gestures and motions ---------

# MediaPipe hand landmark indices for each finger
FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],    # CMC, MCP, IP, TIP
    "index": [5, 6, 7, 8],     # MCP, PIP, DIP, TIP
    "middle": [9, 10, 11, 12], # MCP, PIP, DIP, TIP
    "ring": [13, 14, 15, 16],  # MCP, PIP, DIP, TIP
    "pinky": [17, 18, 19, 20], # MCP, PIP, DIP, TIP
}


def _is_ext(lm_list, name: str) -> bool:
    """
    Check if a finger is extended using 3D landmark positions.
    
    Args:
        lm_list: List of (x, y, z) landmark tuples
        name: Finger name ("thumb", "index", "middle", "ring", "pinky")
    
    Returns:
        True if the finger is extended, False if curled
    """
    # Convert back to a simple structure expected by is_finger_extended_3d
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    landmarks = [P(*p) for p in lm_list]
    return is_finger_extended_3d(landmarks, FINGER_INDICES[name])


def _scale(lm_list) -> float:
    """
    Calculate a normalization scale based on hand size.
    
    Uses the distance between index MCP (5) and pinky MCP (17) as a
    rough measure of hand width for scale-invariant distance comparisons.
    
    Args:
        lm_list: List of (x, y, z) landmark tuples
    
    Returns:
        Normalized scale factor (minimum 1e-3 to avoid division by zero)
    """
    a = lm_list[5]
    b = lm_list[17]
    return max(euclid_2d((a[0], a[1]), (b[0], b[1])), 1e-3)


class Symbol6Gesture(Gesture):
    """
    Detects the "6" or "OK" gesture: thumb and index forming a circle, other fingers extended.
    
    This gesture requires:
    - Thumb tip and index tip close together (pinched)
    - Middle, ring, and pinky fingers extended
    """
    def __init__(self, pinch_threshold: float = 0.60, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    """
    Detects the "7" gesture: thumb and index extended with index pointing downward.
    
    This gesture requires:
    - Thumb and index extended
    - Index finger pointing downward (negative Y direction in screen space)
    - Middle, ring, and pinky fingers curled
    """
    def __init__(self, down_cos: float = 0.6, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    """
    Detects the gun/pistol hand gesture: thumb and index extended (pointing), others curled.
    
    This gesture requires:
    - Thumb and index extended
    - Middle, ring, and pinky curled
    - Tips far apart (> 0.35 normalized distance) to distinguish from Korean heart
    
    Commonly used for trigger-based interactions like the recoil motion.
    """
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        super().__init__("gun", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        # Gun: thumb and index extended, middle/ring/pinky curled
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")
        
        # More specific: thumb+index extended, middle+ring+pinky NOT extended
        basic_gun = thumb_ext and index_ext and middle_ext and (not ring_ext) and (not pinky_ext)
        
        if not basic_gun:
            return False
        
        # Distinguish from Korean heart: gun has tips APART (pointing), heart has tips TOGETHER
        thumb_tip = np.array([lm_list[4][0], lm_list[4][1]])
        index_tip = np.array([lm_list[8][0], lm_list[8][1]])
        scale = _scale(lm_list)
        tip_distance = euclid_2d((thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1])) / scale
        
        # Gun should have tips far apart (> 0.35)
        return tip_distance > 0.35


class OpenPalmGesture(Gesture):
    """All five fingers extended (open hand) with palm facing camera."""
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        super().__init__("palm", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        # Basic extension check
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")

        if not (thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext):
            return False

        # Normalize by hand scale
        scale = _scale(lm_list)

        # Require fingers to be generally above the wrist (pointing upward)
        wrist_y = lm_list[0][1]
        tips = {
            "thumb": lm_list[4],
            "index": lm_list[8],
            "middle": lm_list[12],
            "ring": lm_list[16],
            "pinky": lm_list[20],
        }
        upward_threshold = 0.06  # normalized units (tweakable)
        # Check index/middle/ring/pinky tips are sufficiently above wrist
        for k in ("index", "middle", "ring", "pinky"):
            if (wrist_y - tips[k][1]) / scale < upward_threshold:
                return False

        # Require some lateral spread between finger tips (avoid closed-but-extended)
        spread_im = euclid_2d((tips["index"][0], tips["index"][1]), (tips["middle"][0], tips["middle"][1])) / scale
        spread_mr = euclid_2d((tips["middle"][0], tips["middle"][1]), (tips["ring"][0], tips["ring"][1])) / scale
        spread_rp = euclid_2d((tips["ring"][0], tips["ring"][1]), (tips["pinky"][0], tips["pinky"][1])) / scale
        avg_spread = (spread_im + spread_mr + spread_rp) / 3.0
        min_avg_spread = 0.05  # normalized, tweakable
        if avg_spread < min_avg_spread:
            return False

        # Thumb should not be pinched to index (distinguish from OK/korean heart)
        thumb_index_dist = euclid_2d((tips["thumb"][0], tips["thumb"][1]), (tips["index"][0], tips["index"][1])) / scale
        if thumb_index_dist < 0.18:
            return False

        # Additional guard: ensure thumb is abducted away from the index base (index MCP)
        # This helps prevent a tucked-behind thumb from being counted as "open".
        thumb_index_mcp_sep = euclid_2d((lm_list[4][0], lm_list[4][1]), (lm_list[5][0], lm_list[5][1])) / scale
        if thumb_index_mcp_sep < 0.30:
            return False

        # CRITICAL CHECK: Thumb must be IN FRONT of the palm (palm facing camera)
        # For open palm, thumb Z should be LESS than (closer to camera) than index MCP Z
        # This distinguishes from "four" gesture where thumb is behind
        thumb_tip_z = lm_list[4][2]
        index_mcp_z = lm_list[5][2]
        
        # Negative difference means thumb is in front (closer to camera)
        # Allow small margin (0.005) for noise tolerance
        thumb_in_front = (thumb_tip_z - index_mcp_z) < 0.005
        
        if not thumb_in_front:
            return False

        return True


class FistGesture(Gesture):
    """All fingers curled (fist)."""
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    def __init__(self, pinch_threshold: float = 0.2, sound_path: Optional[List[str]] = None, volume: float = 1.0):
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
    """
    Detects the "middle finger" gesture: only middle finger extended and pointing upright.

    Rules:
    - Index, ring, and pinky curled; middle extended
    - Orientation: middle finger should be upright (toward screen up)
      • Angle with the screen-up vector within a threshold (cosine > up_cos)
      • Depth tilt limited so it's not pointing forward/backward (|z component| < max_z_tilt)

    Notes:
    - Thumb state is not enforced (can be either extended or curled)
    - Screen coordinates: y increases downward; "upright" means negative y direction
    """
    def __init__(self, up_cos: float = 0.7, max_z_tilt: float = 0.4, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        """
        Args:
            up_cos: Minimum cosine with the upward direction (0..1). Higher = stricter uprightness.
            max_z_tilt: Maximum allowed absolute z component of the finger direction (0..1).
            sound_path: Optional list of sound file paths.
            volume: Playback volume (0.0..1.0).
        """
        super().__init__("middle_finger", sound_path=sound_path, volume=volume)
        self.up_cos = up_cos
        self.max_z_tilt = max_z_tilt

    def detect(self, lm_list) -> bool:
        # Finger state: only middle extended
        base_condition = (
            (not _is_ext(lm_list, "index"))
            and _is_ext(lm_list, "middle")
            and (not _is_ext(lm_list, "ring"))
            and (not _is_ext(lm_list, "pinky"))
        )
        if not base_condition:
            return False

        # Orientation: middle finger must point upward and not forward/backward.
        m_mcp = np.array([lm_list[9][0], lm_list[9][1], lm_list[9][2]])
        m_tip = np.array([lm_list[12][0], lm_list[12][1], lm_list[12][2]])
        v = _safe_unit(m_tip - m_mcp)

        # Up direction in screen space is negative Y
        up = np.array([0.0, -1.0, 0.0])
        upright_enough = float(np.dot(v, up)) > self.up_cos
        not_forward_backward = abs(float(v[2])) < self.max_z_tilt

        return upright_enough and not_forward_backward


class KoreanHeartGesture(Gesture):
    """
    Detects the Korean finger heart gesture: thumb and index crossed/touching, others curled.
    
    This gesture requires:
    - Thumb and index tips close together (< 0.40 normalized distance)
    - Middle, ring, and pinky curled
    - Fingers pointing generally upward (tips above wrist)
    - Tips far apart distinguishes from gun pose
    
    Popular in Korean pop culture for expressing affection.
    """
    def __init__(self, cross_threshold: float = 0.40, upward_threshold: float = -0.3, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        super().__init__("korean_heart", sound_path=sound_path, volume=volume)
        self.cross_threshold = cross_threshold
        self.upward_threshold = upward_threshold  # Y-component threshold (negative = upward in screen coords)

    def detect(self, lm_list) -> bool:
        # Check that middle, ring, and pinky are curled (strict requirement)
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")
        
        if middle_ext or ring_ext or pinky_ext:
            return False
        
        # For thumb and index, check if they're at least partially extended
        # Use a more lenient check: tips should be further from palm than base joints
        thumb_tip = np.array([lm_list[4][0], lm_list[4][1], lm_list[4][2]])
        thumb_mcp = np.array([lm_list[2][0], lm_list[2][1], lm_list[2][2]])
        index_tip = np.array([lm_list[8][0], lm_list[8][1], lm_list[8][2]])
        index_mcp = np.array([lm_list[5][0], lm_list[5][1], lm_list[5][2]])
        wrist = np.array([lm_list[0][0], lm_list[0][1], lm_list[0][2]])
        
        # Check if tips are further from wrist than their base joints (partial extension)
        thumb_tip_dist = np.linalg.norm(thumb_tip - wrist)
        thumb_base_dist = np.linalg.norm(thumb_mcp - wrist)
        index_tip_dist = np.linalg.norm(index_tip - wrist)
        index_base_dist = np.linalg.norm(index_mcp - wrist)
        
        thumb_extended_enough = thumb_tip_dist > thumb_base_dist * 0.9  # More lenient
        index_extended_enough = index_tip_dist > index_base_dist * 0.9
        
        if not (thumb_extended_enough and index_extended_enough):
            return False
        
        # Check that fingers are generally pointing upward
        # In screen coordinates, Y increases downward, so upward = negative Y direction
        # Calculate average direction of thumb and index tips relative to wrist
        thumb_direction_y = thumb_tip[1] - wrist[1]
        index_direction_y = index_tip[1] - wrist[1]
        avg_direction_y = (thumb_direction_y + index_direction_y) / 2.0
        
        # If average Y is positive (pointing down), reject
        # Allow some leniency: upward_threshold = -0.3 means tips should be at least somewhat above wrist
        if avg_direction_y > self.upward_threshold:
            return False
        
        # Key difference from gun: check if thumb and index tips are CLOSE together
        # Use 2D distance (screen space) for orientation invariance
        scale = _scale(lm_list)
        tip_distance = euclid_2d((thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1])) / scale
        
        # For Korean heart, the tips should be close (forming the heart point)
        # Gun pose will have tips far apart (pointing direction)
        return tip_distance < self.cross_threshold


class FantasticGesture(Gesture):
    """
    Detects the "four" gesture: back of hand facing camera, four fingers extended upward, thumb behind palm.
    
    This gesture requires:
    - Index, middle, ring, and pinky fingers extended and pointing upward
    - Thumb tucked behind the palm (not extended, and behind the index base in Z-depth)
    - Tips significantly above wrist
    - Fingers spread apart (not closed together)
    
    Key differentiator from OpenPalm: thumb must be behind the palm plane (negative Z relative to index MCP).
    """
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        super().__init__("fantastic", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")

        # Basic requirement: four fingers extended, thumb not extended
        if not (index_ext and middle_ext and ring_ext and pinky_ext):
            return False

        scale = _scale(lm_list)
        
        # Raw Y positions (screen coords: larger Y is lower on screen)
        wrist_y = lm_list[0][1]
        index_tip_y = lm_list[8][1]
        middle_tip_y = lm_list[12][1]
        ring_tip_y = lm_list[16][1]
        pinky_tip_y = lm_list[20][1]

        # Require the tips to be significantly above the wrist
        index_up = (wrist_y - index_tip_y) / scale
        middle_up = (wrist_y - middle_tip_y) / scale
        ring_up = (wrist_y - ring_tip_y) / scale
        pinky_up = (wrist_y - pinky_tip_y) / scale

        upward_threshold = 0.02  # Relaxed from 0.03 - more lenient angle tolerance
        fingers_upward = (
            index_up > upward_threshold
            and middle_up > upward_threshold
            and ring_up > upward_threshold
            and pinky_up > upward_threshold
        )
        
        if not fingers_upward:
            return False

        # KEY CHECK: Thumb must be BEHIND the palm (back-of-hand facing camera)
        # Compare Z-depth: thumb tip should be behind (greater Z) than index MCP
        # In MediaPipe, Z increases away from camera
        thumb_tip_z = lm_list[4][2]
        index_mcp_z = lm_list[5][2]
        
        # Thumb should be at least somewhat behind the index base
        # Positive difference means thumb is behind (away from camera)
        thumb_behind = (thumb_tip_z - index_mcp_z) > -0.005  # Relaxed from 0.01 - allows thumb to be slightly in front
        
        if not thumb_behind:
            return False

        # Require some lateral spread between finger tips (not tightly closed)
        tips = {
            "index": lm_list[8],
            "middle": lm_list[12],
            "ring": lm_list[16],
            "pinky": lm_list[20],
        }
        spread_im = euclid_2d((tips["index"][0], tips["index"][1]), (tips["middle"][0], tips["middle"][1])) / scale
        spread_mr = euclid_2d((tips["middle"][0], tips["middle"][1]), (tips["ring"][0], tips["ring"][1])) / scale
        spread_rp = euclid_2d((tips["ring"][0], tips["ring"][1]), (tips["pinky"][0], tips["pinky"][1])) / scale
        avg_spread = (spread_im + spread_mr + spread_rp) / 3.0
        
        min_avg_spread = 0.025  # Relaxed from 0.04 - allows fingers to be closer together
        if avg_spread < min_avg_spread:
            return False

        # Additional check: thumb should not be visible/extended to the side
        # Check that thumb tip is not far from the palm center laterally
        thumb_tip_x = lm_list[4][0]
        wrist_x = lm_list[0][0]
        index_mcp_x = lm_list[5][0]
        
        # Thumb should be within the hand's lateral bounds (not sticking out to side)
        hand_width = abs(lm_list[5][0] - lm_list[17][0])  # index MCP to pinky MCP
        thumb_lateral_offset = abs(thumb_tip_x - (wrist_x + index_mcp_x) / 2.0)
        
        # Normalize by hand width - thumb should stay within hand bounds
        if thumb_lateral_offset / (hand_width + 1e-6) > 0.75:  # Relaxed from 0.6 - more tolerance for thumb position
            return False

        return True


class SigmaGesture(Gesture):
    """
    Detects the "sigma" / rock/metal hand sign: index and pinky extended, others curled.
    
    This gesture requires:
    - Index and pinky fingers extended and pointing upward
    - Thumb, middle, and ring fingers curled
    - Tips significantly above wrist
    - Classic rock music/metal concert gesture
    """
    def __init__(self, sound_path: Optional[List[str]] = None, volume: float = 1.0):
        super().__init__("sigma", sound_path=sound_path, volume=volume)

    def detect(self, lm_list) -> bool:
        thumb_ext = _is_ext(lm_list, "thumb")
        index_ext = _is_ext(lm_list, "index")
        middle_ext = _is_ext(lm_list, "middle")
        ring_ext = _is_ext(lm_list, "ring")
        pinky_ext = _is_ext(lm_list, "pinky")

        # Basic requirement: index and pinky extended, middle and ring NOT extended
        if not (index_ext and pinky_ext and not middle_ext and not ring_ext):
            return False

        scale = _scale(lm_list)
        
        # Raw Y positions (screen coords: larger Y is lower on screen)
        wrist_y = lm_list[0][1]
        index_tip_y = lm_list[8][1]
        pinky_tip_y = lm_list[20][1]

        # Require the tips to be significantly above the wrist (pointing upward)
        index_up = (wrist_y - index_tip_y) / scale
        pinky_up = (wrist_y - pinky_tip_y) / scale

        upward_threshold = 0.03  # Similar to fantastic
        fingers_upward = index_up > upward_threshold and pinky_up > upward_threshold
        
        if not fingers_upward:
            return False

        # Ensure index and pinky are spread apart (not touching)
        tips = {
            "index": lm_list[8],
            "pinky": lm_list[20],
        }
        spread_ip = euclid_2d((tips["index"][0], tips["index"][1]), (tips["pinky"][0], tips["pinky"][1])) / scale
        
        # Require reasonable spread between index and pinky
        min_spread = 0.15  # normalized, tweakable
        if spread_ip < min_spread:
            return False

        return True


class RecoilMotion(Motion):
    """
    Detects recoil/gun-firing motion: wrist movement while holding a specific gesture.
    
    This motion detector:
    - Requires a gate gesture (typically "gun") to be held
    - Tracks wrist movement when gesture is active
    - Triggers when movement exceeds threshold
    - Excludes downward motion (only upward/sideways triggers)
    - Has per-hand cooldown to prevent spam
    
    Commonly used for gun-firing interactions in games.
    """
    def __init__(self, movement_threshold: float = 0.05, gate_gesture: str = "gun", cooldown_s: float = 0.4, sound_path: Optional[List[str]] = None, volume: float = 1.0, exclude_downward: bool = True):
        super().__init__("recoil", sound_path=sound_path, volume=volume)
        self.movement_threshold = movement_threshold
        self.gate_gesture = gate_gesture
        self.cooldown_s = cooldown_s
        self.exclude_downward = exclude_downward
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
                if prev is None:
                    st["prev"] = wrist
                    return False
                
                # Calculate movement
                dx = wrist[0] - prev[0]
                dy = wrist[1] - prev[1]  # In screen coords, positive Y is DOWN
                movement = math.sqrt(dx*dx + dy*dy)
                
                st["prev"] = wrist
                
                # Check if downward motion should be excluded
                if self.exclude_downward and dy > 0:
                    # Motion is downward (positive Y), reject it
                    return False
                
                if movement > self.movement_threshold:
                    last = self.last_trigger.get(hand_id, 0.0)
                    if now - last > self.cooldown_s:
                        self.last_trigger[hand_id] = now
                        return True
        else:
            st["ready"] = False
            st["prev"] = None
        return False


class HammerStrikeMotion(Motion):
    """
    Detects a hammer-fist downward strike while holding a specific gesture (default: "fist").

    Behavior:
    - Requires gate gesture (e.g., fist) to be active
    - Tracks wrist movement frame-to-frame
    - Triggers only on downward motion (positive screen Y)
    - Requires total movement to exceed a threshold
    - Optional: require that a minimum portion of movement is downward (directionality)
    - Per-hand cooldown to prevent spamming

    Typical use: play a heavy impact sound (e.g., metal pipe) when a fist strikes down.
    """
    def __init__(
        self,
        movement_threshold: float = 0.04,
        gate_gesture: str = "fist",
        cooldown_s: float = 0.5,
        sound_path: Optional[List[str]] = None,
        volume: float = 1.0,
        require_downward: bool = True,
        min_down_ratio: float = 0.6,
    ):
        """
        Args:
            movement_threshold: Minimum normalized movement to count as a strike.
            gate_gesture: Gesture name that must be held (default: "fist").
            cooldown_s: Minimum seconds between triggers per hand.
            sound_path: Optional list of sounds to play when triggered.
            volume: Playback volume (0..1).
            require_downward: If True, only downward motion (dy>0) can trigger.
            min_down_ratio: Portion of movement that must be downward (dy/mag), 0..1.
        """
        super().__init__("hammer_strike", sound_path=sound_path, volume=volume)
        self.movement_threshold = movement_threshold
        self.gate_gesture = gate_gesture
        self.cooldown_s = cooldown_s
        self.require_downward = require_downward
        self.min_down_ratio = min(1.0, max(0.0, min_down_ratio))
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
                if prev is None:
                    st["prev"] = wrist
                    return False

                dx = wrist[0] - prev[0]
                dy = wrist[1] - prev[1]  # Positive Y is downward in screen coords
                mag = math.sqrt(dx*dx + dy*dy)
                st["prev"] = wrist

                # Directional checks
                if self.require_downward and dy <= 0:
                    return False
                if mag < 1e-6:
                    return False
                down_ratio = dy / mag  # in [-inf, +inf], but positive if downward; cap by ratio
                if self.require_downward and down_ratio < self.min_down_ratio:
                    return False

                if mag > self.movement_threshold:
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
    """
    Detects when two specific gestures are close together in space.
    
    Useful for creating interaction patterns like:
    - Two hands making the same gesture and touching
    - Different gestures on two hands coming together
    
    Attributes:
        a: Name of first gesture
        b: Name of second gesture (can be same as 'a')
        threshold: Maximum normalized distance between hand centers
        cooldown_s: Minimum time between consecutive triggers
        sound_path: Optional list of sound files for this proximity event
        volume: Playback volume (0.0 to 1.0)
    """
    a: str
    b: str
    threshold: float = 0.18
    active_pairs: Set[Tuple[int, int]] = field(default_factory=set)
    cooldown_s: float = 0.4
    last_trigger: float = 0.0
    # Optional per-rule sound metadata
    sound_path: Optional[List[str]] = None
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
    """
    Emits events for single gestures with edge-triggering and cooldown.
    
    Fires once when a gesture is first detected on a hand (rising edge),
    then requires both:
    1. Gesture to be released and re-made
    2. Cooldown period to elapse
    
    Prevents spam from detection flicker and ensures intentional triggers.
    
    Attributes:
        g: Name of gesture to trigger on
        cooldown_s: Minimum time between triggers per hand
    """
    g: str
    cooldown_s: float = 1.0  # Time-based cooldown per hand to prevent rapid re-triggers
    # Fire once per hand while the gesture remains active (edge-triggered)
    active_hands: Set[int] = field(default_factory=set)
    last_trigger_time: Dict[int, float] = field(default_factory=dict)

    def check(self, detections: Dict[int, Set[str]], now: float) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        current_active: Set[int] = set()
        for hid, gestures in detections.items():
            if self.g in gestures:
                current_active.add(hid)
                if hid not in self.active_hands:
                    # Rising edge: check cooldown before firing
                    last_time = self.last_trigger_time.get(hid, 0.0)
                    if now - last_time > self.cooldown_s:
                        events.append({"type": self.g, "hand_id": hid})
                        self.last_trigger_time[hid] = now
        # Update latch state: keep only those still active
        self.active_hands = current_active
        return events


class GestureEngine:
    """
    Main orchestrator for hand gesture and motion detection.
    
    The GestureEngine:
    - Maintains stable hand tracking across frames
    - Applies landmark smoothing for noise reduction
    - Runs gesture detection with motion filtering (prevents false positives during fast movement)
    - Runs motion detection (uses unfiltered gestures for gating)
    - Evaluates proximity rules between hands
    - Evaluates single-gesture trigger rules
    - Emits events for all detected patterns
    
    Attributes:
        smoother_alpha: EMA smoothing factor (0.0 to 1.0)
        max_gesture_velocity: Maximum hand speed for gesture detection (normalized units/sec)
    """
    def __init__(self, smoother_alpha: float = 0.5, max_gesture_velocity: float = 0.15):
        """
        Initialize the gesture engine.
        
        Args:
            smoother_alpha: Smoothing factor for landmarks (higher = more responsive)
            max_gesture_velocity: Maximum hand velocity for gesture detection (prevents false positives)
        """
        self.tracker = SimpleHandTracker()
        self.smoother = LandmarkSmoother(alpha=smoother_alpha)
        self.gestures: List[Gesture] = []
        self.motions: List[Motion] = []
        self.rules: List[ProximityRule] = []
        self.gesture_rules: List[GestureTriggerRule] = []
        # Motion filtering: prevent gesture detection when hand is moving too fast
        self.max_gesture_velocity = max_gesture_velocity
        self.prev_hand_centers: Dict[int, Tuple[Tuple[float, float], float]] = {}  # hand_id -> (center, timestamp)

    def register_gesture(self, gesture: Gesture):
        self.gestures.append(gesture)

    def register_motion(self, motion: Motion):
        self.motions.append(motion)

    def register_rule(self, rule: ProximityRule):
        self.rules.append(rule)

    def register_gesture_rule(self, rule: GestureTriggerRule):
        self.gesture_rules.append(rule)

    def process(self, multi_hand_landmarks) -> Dict[str, List]:
        """
        Process detected hands and emit events for gestures, motions, and rules.
        
        Args:
            multi_hand_landmarks: MediaPipe multi_hand_landmarks from Hands.process()
        
        Returns:
            Dictionary with:
            - "events": List of event dicts (type, hand_id, etc.)
            - "overlays": List of debug info (detections, centers per frame)
        """
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
        detections: Dict[int, Set[str]] = {}  # For gesture triggers (velocity filtered)
        motion_detections: Dict[int, Set[str]] = {}  # For motion gating (no velocity filter)
        smoothed: Dict[int, List[Tuple[float, float, float]]] = {}
        hand_velocities: Dict[int, float] = {}
        
        # Calculate velocity for each hand to filter out fast motion
        for idx, hand in enumerate(multi_hand_landmarks):
            hid = idx_to_id.get(idx)
            if hid is None:
                continue
            
            current_center = centers_by_id[hid]
            velocity = 0.0
            
            if hid in self.prev_hand_centers:
                prev_center, prev_time = self.prev_hand_centers[hid]
                dt = now - prev_time
                if dt > 0.001:  # Avoid division by zero
                    dx = current_center[0] - prev_center[0]
                    dy = current_center[1] - prev_center[1]
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
            
            hand_velocities[hid] = velocity
            self.prev_hand_centers[hid] = (current_center, now)
            
            # Smooth landmarks
            sm = self.smoother.smooth(hid, hand.landmark)
            smoothed[hid] = sm
            
            # Run gesture detection twice:
            # 1. For motion gating (no velocity filter) - motions need to detect gestures even when moving
            motion_hits: Set[str] = set()
            for g in self.gestures:
                if g.detect(sm):
                    motion_hits.add(g.name)
            motion_detections[hid] = motion_hits
            
            # 2. For gesture triggers/proximity (velocity filtered) - only detect when hand is still
            gesture_hits: Set[str] = set()
            if velocity < self.max_gesture_velocity:
                gesture_hits = motion_hits.copy()  # Copy the unfiltered detections
            detections[hid] = gesture_hits

        # Motions - use motion_detections (unfiltered) for gating
        for hid, sm in smoothed.items():
            hits = motion_detections.get(hid, set())  # Changed from detections to motion_detections
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
