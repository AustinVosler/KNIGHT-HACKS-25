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
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip # JW
from typing import Optional, List

from .gesture_engine import (
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
    HammerStrikeMotion,
    ProximityRule,
    GestureTriggerRule,
)


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


def _play_sound(paths: Optional[List[str]], volume: float, cache: dict) -> bool:
    """
    Play one or more sounds with given volume.
    
    Args:
        paths: List of file paths to sound files (all will be played)
        volume: Playback volume (0.0 to 1.0)
        cache: Dictionary cache for Sound objects
    
    Returns:
        True if at least one sound was played successfully, False otherwise
    """
    if not paths:
        return False
    
    played_any = False
    for path in paths:
        snd = _load_sound(path, volume, cache)
        if snd:
            try:
                snd.play()
                played_any = True
            except Exception as ex:
                print(f"Failed to play sound {path}: {ex}")
    
    return played_any


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
    
    # Init audio
    pygame.mixer.init()
    
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
    engine.register_gesture(KoreanHeartGesture(sound_path=["./sounds/kpop.mp3"], volume=0.8))
    engine.register_gesture(GunPoseGesture())
    
    engine.register_gesture(OpenPalmGesture(sound_path=["./sounds/hi.mp3"], volume=3))
    engine.register_gesture(FistGesture())
    engine.register_gesture(PeaceSignGesture())
    engine.register_gesture(ThumbsUpGesture())
    
    # Attach sounds directly to the gesture class with a per-gesture volume
    # Middle finger plays BOTH fahh.mp3 AND thunder.mp3
    engine.register_gesture(MiddleFingerGesture(sound_path=["./sounds/fahh.mp3", "./sounds/thunder.mp3"], volume=1.0))
    
    # Register motions with sounds
    # Movement threshold: lower = more sensitive (easier to trigger), higher = less sensitive
    # Typical quick hand jerk is around 0.03-0.08 in normalized coords
    engine.register_motion(RecoilMotion(
        movement_threshold=0.03,  # Made more sensitive (was 0.05)
        gate_gesture="gun", 
        cooldown_s=0.4,  # Reduced cooldown to allow faster repeated shots
        sound_path=["./sounds/bang.mp3"],
        volume=0.6
    ))

    # Hammer fist downward strike -> metal pipe impact
    engine.register_motion(HammerStrikeMotion(
        movement_threshold=0.01,   # Slightly stricter than recoil to avoid noise
        gate_gesture="fist",
        cooldown_s=0.6,
        sound_path=["./sounds/metal_pipe.mp3"],
        volume=0.6,
        require_downward=True,
        min_down_ratio=0.6,
    ))
    
    # Register proximity rules with sounds
    engine.register_rule(ProximityRule(
        a="six", 
        b="seven", 
        threshold=0.50, 
        cooldown_s=0.6,
        sound_path=["./sounds/67.mp3"],
        volume=0.1
    ))
    engine.register_rule(ProximityRule(
        a="thumbs_up", 
        b="thumbs_up", 
        threshold=0.30, 
        cooldown_s=0.6,
        sound_path=["./sounds/yippee.mp3"],
        volume=0.7
    ))
    
    # Register single-gesture trigger rules (emit events directly for gestures)
    engine.register_gesture_rule(GestureTriggerRule(g="middle_finger", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="korean_heart", cooldown_s=1.5))  # Longer cooldown for heart
    engine.register_gesture_rule(GestureTriggerRule(g="palm", cooldown_s=1.0))
    # Uncomment to add more proximity rules:
    # engine.register_rule(ProximityRule(a="palm", b="fist", threshold=0.15, cooldown_s=0.6))
    # engine.register_rule(ProximityRule(a="peace", b="thumbs_up", threshold=0.15, cooldown_s=0.6))

    # LIVE WEBCAM -JW
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("1.webm") 
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
                        # Highlight gun gesture in green for easier debugging
                        color = (0, 255, 0) if "gun" in detections.get(hid, set()) else (0, 255, 255)
                        cv2.putText(frame, f"{hid}:{label}", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # React to events
        for e in events:
            etype = e.get("type", "")
            if not etype:
                continue

            print(etype)

            # Route to the appropriate sound source based on event type
            played = False
            
            # First, check for motion events by name
            m = _get_motion_by_name(engine, etype)
            if m is not None:
                if m.sound_path:
                    played = _play_sound(m.sound_path, m.volume, sound_cache)
                    if played:
                        print(f"Playing motion sound: {m.name}")
            
            # Gesture events (simple name, no colon, and not a motion)
            elif ":" not in etype:
                g = _get_gesture_by_name(engine, etype)
                if g and g.sound_path:
                    played = _play_sound(g.sound_path, g.volume, sound_cache)
                    if played:
                        print(f"Playing gesture sound: {g.name}")
            
            # Proximity events (format: "proximity:a+b")
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

def video_upload(filename) -> str:
    """
    Process a prerecorded video to detect hand events and overlay sound effects
    at the exact timestamps where events occur.

    Output video visuals remain identical to the original (no drawing/flip);
    only audio is augmented by mixing in the detected gesture/motion/proximity
    sounds at their detection times. The original video's audio (if present)
    is preserved under the overlays.
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
    # Use lists for sound_path for consistency with engine types
    engine.register_gesture(KoreanHeartGesture(sound_path=["./sounds/kpop.mp3"], volume=1))
    engine.register_gesture(GunPoseGesture())
    
    engine.register_gesture(OpenPalmGesture(sound_path=["./sounds/hi.mp3"], volume=1))
    engine.register_gesture(FistGesture())
    engine.register_gesture(PeaceSignGesture())
    engine.register_gesture(ThumbsUpGesture())
    
    # Attach sound directly to the gesture class with a per-gesture volume
    engine.register_gesture(MiddleFingerGesture(sound_path=["./sounds/fahh.mp3"], volume=1.0))
    
    # Register motions with sounds
    engine.register_motion(RecoilMotion(
        movement_threshold=0.05, 
        gate_gesture="gun", 
        cooldown_s=0.6,
        sound_path=["./sounds/bang.mp3"],
        volume=0.6
    ))
    # Hammer fist downward strike -> metal pipe impact
    engine.register_motion(HammerStrikeMotion(
        movement_threshold=0.01,   # Slightly stricter than recoil to avoid noise
        gate_gesture="fist",
        cooldown_s=0.6,
        sound_path=["./sounds/metal_pipe.mp3"],
        volume=0.6,
        require_downward=True,
        min_down_ratio=0.6,
    ))
    
    # Register proximity rules with sounds
    engine.register_rule(ProximityRule(
        a="six", 
        b="seven", 
        threshold=0.50, 
        cooldown_s=0.6,
        sound_path=["./sounds/67.mp3"],
        volume=0.1
    ))
    engine.register_rule(ProximityRule(
        a="thumbs_up", 
        b="thumbs_up", 
        threshold=0.30, 
        cooldown_s=0.6,
        sound_path=["./sounds/yippee.mp3"],
        volume=0.7
    ))
    
    # Register single-gesture trigger rules (emit events directly for gestures)
    engine.register_gesture_rule(GestureTriggerRule(g="middle_finger", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="korean_heart", cooldown_s=1.5))  # Longer cooldown for heart
    engine.register_gesture_rule(GestureTriggerRule(g="palm", cooldown_s=1.0))
    # Uncomment to add more proximity rules:
    # engine.register_rule(ProximityRule(a="palm", b="fist", threshold=0.15, cooldown_s=0.6))
    # engine.register_rule(ProximityRule(a="peace", b="thumbs_up", threshold=0.15, cooldown_s=0.6))

    # -------------------- VIDEO INPUT --------------------
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Failed to open video file")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: Could not detect FPS, defaulting to 30")
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # STORE SOUND EVENTS WITH TIMESTAMPS
    sound_events = []  # List of tuples (time_in_seconds, sound_path, volume)
    frame_idx = 0
    print(f"Scanning for hand events to place audio overlays... (Total frames: {total_frames}, FPS: {fps})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Prefer precise timestamp from the file if available; fallback to frame index
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_s = (pos_ms / 1000.0) if pos_ms and pos_ms > 0 else (frame_idx / float(fps))
        
        # IMPORTANT: Do NOT flip or draw on frames; visuals must remain unchanged
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        events = []
        if result.multi_hand_landmarks:
            # Run engine (no drawing)
            out = engine.process(result.multi_hand_landmarks)
            events = out["events"]
            
            # Debug: print events as they happen
            if events:
                for e in events:
                    print(f"Event detected at {time_s:.2f}s: {e}")

        # -------------------- EVENT REACTION --------------------
        for e in events:
            etype = e.get("type", "")
            if not etype:
                continue

            # Instead of playing sound live, record it for later mixing
            # 1. Gesture events (simple name, no colon)
            if ":" not in etype and etype not in ["recoil", "hammer_strike"]:
                g = _get_gesture_by_name(engine, etype)
                if g and g.sound_path:
                    # Normalize to list of paths
                    paths = g.sound_path if isinstance(g.sound_path, list) else [g.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, g.volume))
            
            # 2. Motion events (e.g., "recoil", "hammer_strike")
            elif etype in ["recoil", "hammer_strike"]:
                m = _get_motion_by_name(engine, etype)
                if m and m.sound_path:
                    paths = m.sound_path if isinstance(m.sound_path, list) else [m.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, m.volume))
            
            # 3. Proximity events (format: "proximity:a+b")
            elif etype.startswith("proximity:"):
                rule = _get_rule_by_event_type(engine, etype)
                if rule and rule.sound_path:
                    paths = rule.sound_path if isinstance(rule.sound_path, list) else [rule.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, rule.volume))

        frame_idx += 1

    cap.release()
    print(f"Processed {frame_idx} frames. Found {len(sound_events)} sound events.")

    # -------------------- POST-PROCESS AUDIO --------------------
    print("Combining detected sounds with the original video...")

    # Load original video (for its visuals and existing audio track)
    base_clip = VideoFileClip(filename)
    video_duration = base_clip.duration

    # Create audio clips for all sound events
    audio_clips = []
    for t, path, vol in sound_events:
        try:
            snd = AudioFileClip(path).volumex(vol).set_start(t)
            audio_clips.append(snd)
            print(f"Adding sound at {t:.2f}s: {os.path.basename(path)} (vol={vol})")
        except Exception as e:
            print(f"Could not load sound {path}: {e}")

    # Combine all audio (original + gesture sounds)
    if base_clip.audio:
        final_audio = CompositeAudioClip([base_clip.audio] + audio_clips)
    else:
        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
        else:
            final_audio = None

    # Merge final audio into the ORIGINAL video visuals (no visual changes)
    if final_audio:
        # Set duration to match video to prevent extra frames
        final_audio = final_audio.set_duration(video_duration)
        final_clip = base_clip.set_audio(final_audio)
    else:
        final_clip = base_clip

    # Save final video with embedded sound effects next to original
    base, _ = os.path.splitext(os.path.basename(filename))
    out_dir = os.path.dirname(filename) or "."
    output_path = os.path.join(out_dir, f"{base}_sfx.mp4")
    
    # Use preset and threads for better performance and quality
    # Key fix: Use r=fps to force constant frame rate without dropping frames
    final_clip.write_videofile(
    # output_path = filename.replace(".webm", "_sfx.mp4")
    # base_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=fps,  # Match original FPS
        preset='veryslow',  # Fastest preset to avoid frame drops
        threads=4,  # Use multiple threads for encoding
        bitrate="8000k",  # Higher bitrate for better quality
        audio_bitrate="192k",  # Better audio quality
        verbose=False,  # Reduce console spam
        write_logfile=False  # Disable log file
    )
    
    # Close clips to release resources (Windows file locks)
    final_clip.close()
    base_clip.close()
    for ac in audio_clips:
        try:
            ac.close()
        except Exception:
            pass

    print(f"Done. Wrote: {output_path}")
    return output_path


# if __name__ == "__main__":
#     main()
    #video_upload("./videos/4.webm")