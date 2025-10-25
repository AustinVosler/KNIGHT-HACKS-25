"""
Modular Gesture Recognition Demo

This script demonstrates the gesture engine's capabilities by:
- Detecting multiple hand gestures (6, 7, gun, Korean heart, etc.)
- Triggering sounds based on gestures, motions, and proximity events
- Running a real-time webcam feed with visual overlays

Features:
- Per-gesture sound attachment with volume control
- Motion filtering to prevent false positives during fast hand movement
- Proximity detection (e.g., two specific gestures coming together)
- Edge-triggered gesture events with cooldown
- Modular architecture: easily add new gestures, motions, and rules

Usage:
    python combo_modular.py

Press 'q' to quit.

Customization:
- Add new gestures by registering them with the engine
- Attach sounds by passing sound_path and volume parameters
- Adjust thresholds and cooldowns for fine-tuning
- Register new proximity rules or gesture trigger rules
"""

import cv2
import mediapipe as mp
import pygame
import os
import time
from typing import Optional

from gesture_engine import (
    GestureEngine,
    Symbol6Gesture,
    Symbol7Gesture,
    GunPoseGesture,
    OpenPalmGesture,
    FistGesture,
    PeaceSignGesture,
    ThumbsUpGesture,
    MiddleFingerGesture,
    KoreanHeartGesture,
    RecoilMotion,
    ProximityRule,
    GestureTriggerRule,
)


# Init audio
pygame.mixer.init()

# Cache for sounds to avoid reloading
# Key: sound file path -> pygame Sound object
sound_cache = {}


def _load_sound(path: str, volume: float, cache: dict) -> Optional[pygame.mixer.Sound]:
    """
    Load a sound file and set its volume, using cache to avoid reloading.
    
    Args:
        path: File path to the sound file
        volume: Playback volume (0.0 to 1.0)
        cache: Dictionary cache mapping paths to Sound objects
    
    Returns:
        pygame.mixer.Sound object if successful, None otherwise
    """
    if not path:
        return None
    try:
        if path not in cache:
            if not os.path.exists(path):
                print(f"Warning: Sound file not found: {path}")
                return None
            snd = pygame.mixer.Sound(path)
            cache[path] = snd
        # Always update volume in case it changed
        cache[path].set_volume(max(0.0, min(1.0, volume)))
        return cache[path]
    except Exception as ex:
        print(f"Failed to load sound {path}: {ex}")
        return None


def _play_sound(path: str, volume: float, cache: dict) -> bool:
    """
    Play a sound with given path and volume.
    
    Args:
        path: File path to the sound file
        volume: Playback volume (0.0 to 1.0)
        cache: Dictionary cache for Sound objects
    
    Returns:
        True if sound was played successfully, False otherwise
    """
    snd = _load_sound(path, volume, cache)
    if snd:
        try:
            snd.play()
            return True
        except Exception as ex:
            print(f"Failed to play sound {path}: {ex}")
    return False


def _get_gesture_by_name(engine: GestureEngine, name: str):
    """
    Find a registered gesture by name.
    
    Args:
        engine: GestureEngine instance
        name: Gesture name to search for
    
    Returns:
        Gesture object if found, None otherwise
    """
    for g in engine.gestures:
        if g.name == name:
            return g
    return None


def _get_motion_by_name(engine: GestureEngine, name: str):
    """
    Find a registered motion by name.
    
    Args:
        engine: GestureEngine instance
        name: Motion name to search for
    
    Returns:
        Motion object if found, None otherwise
    """
    for m in engine.motions:
        if m.name == name:
            return m
    return None


def _get_rule_by_event_type(engine: GestureEngine, event_type: str):
    """
    Extract proximity rule for a given event type.
    
    Parses event strings like "proximity:six+seven" to find the matching
    ProximityRule that triggered the event.
    
    Args:
        engine: GestureEngine instance
        event_type: Event type string (format: "proximity:a+b")
    
    Returns:
        ProximityRule object if found, None otherwise
    """
    if not event_type.startswith("proximity:"):
        return None
    # Parse 'proximity:a+b' -> find rule matching a and b
    _, pair = event_type.split(":", 1)
    parts = pair.split("+")
    if len(parts) != 2:
        return None
    a, b = parts
    for rule in engine.rules:
        # Check both orderings since events might be emitted either way
        if (rule.a == a and rule.b == b) or (rule.a == b and rule.b == a):
            return rule
    return None


def main():
    """
    Main entry point for the gesture recognition demo.
    
    Sets up:
    - MediaPipe Hands for hand detection
    - GestureEngine with registered gestures, motions, and rules
    - OpenCV webcam capture and display
    - Sound playback system with per-event routing
    
    Runs a real-time loop processing frames, detecting gestures, and
    playing sounds based on events.
    """
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=99,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # Engine and detectors
    # max_gesture_velocity: prevent gesture detection when hand is moving too fast (normalized units/second)
    # Lower = stricter (only detect on very still hands), Higher = more lenient
    engine = GestureEngine(smoother_alpha=0.5, max_gesture_velocity=0.15)
    
    # Register all gestures
    engine.register_gesture(Symbol6Gesture())
    engine.register_gesture(Symbol7Gesture(down_cos=0.6))
    
    # Korean heart MUST be registered before gun to get priority (both use thumb+index)
    engine.register_gesture(KoreanHeartGesture(sound_path="./sounds/kpop.mp3", volume=0.8))
    engine.register_gesture(GunPoseGesture())
    
    engine.register_gesture(OpenPalmGesture())
    engine.register_gesture(FistGesture())
    engine.register_gesture(PeaceSignGesture())
    engine.register_gesture(ThumbsUpGesture())
    
    # Attach sound directly to the gesture class with a per-gesture volume
    engine.register_gesture(MiddleFingerGesture(sound_path="./sounds/fahh.mp3", volume=1.0))
    
    # Register motions with sounds
    engine.register_motion(RecoilMotion(
        movement_threshold=0.05, 
        gate_gesture="gun", 
        cooldown_s=0.6,
        sound_path="./sounds/bang.mp3",
        volume=0.6
    ))
    
    # Register proximity rules with sounds
    engine.register_rule(ProximityRule(
        a="six", 
        b="seven", 
        threshold=0.50, 
        cooldown_s=0.6,
        sound_path="./sounds/67.mp3",
        volume=0.1
    ))
    engine.register_rule(ProximityRule(
        a="thumbs_up", 
        b="thumbs_up", 
        threshold=0.30, 
        cooldown_s=0.6,
        sound_path="./sounds/yippee.mp3",
        volume=0.7
    ))
    
    # Register single-gesture trigger rules (emit events directly for gestures)
    engine.register_gesture_rule(GestureTriggerRule(g="middle_finger", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="korean_heart", cooldown_s=1.5))  # Longer cooldown for heart
    # Uncomment to add more proximity rules:
    # engine.register_rule(ProximityRule(a="palm", b="fist", threshold=0.15, cooldown_s=0.6))
    # engine.register_rule(ProximityRule(a="peace", b="thumbs_up", threshold=0.15, cooldown_s=0.6))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        events = []
        if result.multi_hand_landmarks:
            # Draw hands and run engine
            for h in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

            out = engine.process(result.multi_hand_landmarks)
            events = out["events"]
            
            # Debug: print events as they happen
            if events:
                for e in events:
                    print(f"Event detected: {e}")

            # Optional: simple overlay of detected gestures per hand
            overlays = out.get("overlays", [])
            for ov in overlays:
                detections = ov.get("detections", {})
                centers = ov.get("centers", {})
                for hid, center in centers.items():
                    label = "+".join(sorted(list(detections.get(hid, set()))))
                    if label:
                        x = int(center[0] * frame.shape[1])
                        y = int(center[1] * frame.shape[0])
                        cv2.putText(frame, f"{hid}:{label}", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # React to events
        for e in events:
            etype = e.get("type", "")
            if not etype:
                continue

            print(etype)

            # Route to the appropriate sound source based on event type
            played = False
            
            # 1. Gesture events (simple name, no colon)
            if ":" not in etype and etype not in ["recoil"]:
                g = _get_gesture_by_name(engine, etype)
                if g and g.sound_path:
                    played = _play_sound(g.sound_path, g.volume, sound_cache)
                    if played:
                        print(f"Playing gesture sound: {g.name}")
            
            # 2. Motion events (e.g., "recoil")
            elif etype == "recoil":
                m = _get_motion_by_name(engine, etype)
                if m and m.sound_path:
                    played = _play_sound(m.sound_path, m.volume, sound_cache)
                    if played:
                        print(f"Playing motion sound: {m.name}")
            
            # 3. Proximity events (format: "proximity:a+b")
            elif etype.startswith("proximity:"):
                rule = _get_rule_by_event_type(engine, etype)
                if rule and rule.sound_path:
                    played = _play_sound(rule.sound_path, rule.volume, sound_cache)
                    if played:
                        print(f"Playing proximity sound: {etype}")

            # Visual feedback on screen
            if played:
                cv2.putText(frame, etype.upper(), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Modular Combo: Engine Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
