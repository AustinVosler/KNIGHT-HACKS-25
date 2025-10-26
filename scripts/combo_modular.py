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
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip 
from typing import Optional, List

from .gesture_engine import (
    FantasticGesture,
    SigmaGesture,
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


def create_engine() -> GestureEngine:
    """
    Create and configure a GestureEngine with all gestures, motions, rules,
    and optional GIF overlay metadata (single-string gif_path, gif_scale_h).
    This configuration is shared by both live (main) and video_upload processing.
    """
    
    engine = GestureEngine(smoother_alpha=0.5, max_gesture_velocity=0.15)

    
    engine.register_gesture(Symbol6Gesture())
    engine.register_gesture(Symbol7Gesture(down_cos=0.6))

    
    engine.register_gesture(KoreanHeartGesture(sound_path=["./sounds/kpop.mp3"], volume=1))
    engine.register_gesture(GunPoseGesture())

    engine.register_gesture(FantasticGesture(sound_path=["./sounds/fantastic.mp3"], volume=1.0))
    engine.register_gesture(SigmaGesture(sound_path=["./sounds/sigma.mp3"], volume=1.0))
    
    engine.register_gesture(OpenPalmGesture(sound_path=["./sounds/hi.mp3"], volume=1))
    engine.register_gesture(FistGesture())
    engine.register_gesture(PeaceSignGesture())
    engine.register_gesture(ThumbsUpGesture())
    
    engine.register_gesture(MiddleFingerGesture(sound_path=["./sounds/fahh.mp3", "./sounds/thunder.mp3"], volume=0.6))

    
    recoil_motion = RecoilMotion(
        movement_threshold=0.05,
        gate_gesture="gun",
        cooldown_s=0.6,
        sound_path=["./sounds/bang.mp3"],
        volume=0.6,
    )
    
    recoil_motion.gif_path = "./gifs/gun_recoil.gif"
    recoil_motion.gif_scale_h = 0.20
    engine.register_motion(recoil_motion)

    hammer_motion = HammerStrikeMotion(
        movement_threshold=0.01,
        gate_gesture="fist",
        cooldown_s=0.6,
        sound_path=["./sounds/metal_pipe.mp3"],
        volume=0.3,  
        require_downward=True,
        min_down_ratio=0.6,
    )
    hammer_motion.gif_path = "./gifs/metal_pipe.gif"
    hammer_motion.gif_scale_h = 0.20
    engine.register_motion(hammer_motion)

    
    engine.register_rule(ProximityRule(
        a="six",
        b="seven",
        threshold=0.50,
        cooldown_s=0.6,
        sound_path=["./sounds/67.mp3"],
        volume=0.05,  
    ))
    engine.register_rule(ProximityRule(
        a="thumbs_up",
        b="thumbs_up",
        threshold=0.30,
        cooldown_s=0.6,
        sound_path=["./sounds/yippee.mp3"],
        volume=0.7,
    ))
    
    flashbang_rule = ProximityRule(
        a="palm",
        b="palm",
        threshold=0.14,      
        cooldown_s=1.2,      
        sound_path=["./sounds/flashbang.mp3"],
        volume=1.0,
    )
    
    flashbang_rule.gif_path = "./gifs/flashbang.gif"
    flashbang_rule.gif_scale_h = 0.30  
    engine.register_rule(flashbang_rule)

    
    engine.register_gesture_rule(GestureTriggerRule(g="middle_finger", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="korean_heart", cooldown_s=1.5))
    engine.register_gesture_rule(GestureTriggerRule(g="palm", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="fantastic", cooldown_s=1.0))
    engine.register_gesture_rule(GestureTriggerRule(g="sigma", cooldown_s=1.0))

    return engine


def _get_motion_by_name(engine: GestureEngine, name: str):
    for m in engine.motions:
        if m.name == name:
            return m
    return None


def _get_rule_by_event_type(engine: GestureEngine, event_type: str):
    if not event_type.startswith("proximity:"):
        return None
    _, pair = event_type.split(":", 1)
    parts = pair.split("+")
    if len(parts) != 2:
        return None
    a, b = parts
    for rule in engine.rules:
        
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
    
    
    pygame.mixer.init()
    
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=99,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    
    engine = create_engine()
    
    
    

    
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
            
            for h in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

            out = engine.process(result.multi_hand_landmarks)
            events = out["events"]
            
            
            if events:
                for e in events:
                    print(f"Event detected: {e}")

            
            overlays = out.get("overlays", [])
            for ov in overlays:
                detections = ov.get("detections", {})
                centers = ov.get("centers", {})
                for hid, center in centers.items():
                    label = "+".join(sorted(list(detections.get(hid, set()))))
                    if label:
                        x = int(center[0] * frame.shape[1])
                        y = int(center[1] * frame.shape[0])
                        
                        color = (0, 255, 0) if "gun" in detections.get(hid, set()) else (0, 255, 255)
                        cv2.putText(frame, f"{hid}:{label}", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        
        for e in events:
            etype = e.get("type", "")
            if not etype:
                continue

            print(etype)

            
            played = False
            
            
            m = _get_motion_by_name(engine, etype)
            if m is not None:
                if m.sound_path:
                    played = _play_sound(m.sound_path, m.volume, sound_cache)
                    if played:
                        print(f"Playing motion sound: {m.name}")
            
            
            elif ":" not in etype:
                g = _get_gesture_by_name(engine, etype)
                if g and g.sound_path:
                    played = _play_sound(g.sound_path, g.volume, sound_cache)
                    if played:
                        print(f"Playing gesture sound: {g.name}")
            
            
            elif etype.startswith("proximity:"):
                rule = _get_rule_by_event_type(engine, etype)
                if rule and rule.sound_path:
                    played = _play_sound(rule.sound_path, rule.volume, sound_cache)
                    if played:
                        print(f"Playing proximity sound: {etype}")

            
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
    
    
    print(filename)
    import os
    print(os.path.exists(filename))
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=99,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    
    engine = create_engine()
    
    
    

    
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Failed to open video file")
        return

    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: Could not detect FPS, defaulting to 30")
        fps = 30.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

    
    sound_events = []  
    
    visual_events = []  
    frame_idx = 0
    print(f"Scanning for hand events to place audio overlays... (Total frames: {total_frames}, FPS: {fps})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_s = (pos_ms / 1000.0) if pos_ms and pos_ms > 0 else (frame_idx / float(fps))
        
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        events = []
        centers = {}
        if result.multi_hand_landmarks:
            
            out = engine.process(result.multi_hand_landmarks)
            events = out["events"]
            overlays = out.get("overlays", [])
            if overlays:
                centers = overlays[-1].get("centers", {})  
            
            
            if events:
                for e in events:
                    print(f"Event detected at {time_s:.2f}s: {e}")

        
        for e in events:
            etype = e.get("type", "")
            if not etype:
                continue

            
            
            if ":" not in etype and etype not in ["recoil", "hammer_strike"]:
                g = _get_gesture_by_name(engine, etype)
                if g and g.sound_path:
                    
                    paths = g.sound_path if isinstance(g.sound_path, list) else [g.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, g.volume))
                
                if g is not None and getattr(g, "gif_path", None):
                    hid = e.get("hand_id")
                    if hid is not None and hid in centers:
                        cx = int(centers[hid][0] * width)
                        cy = int(centers[hid][1] * height)
                        visual_events.append({
                            "t": time_s,
                            "etype": etype,
                            "x": cx,
                            "y": cy,
                            "gif_path": getattr(g, "gif_path"),
                            "scale_h": float(getattr(g, "gif_scale_h", 0.20)),
                        })
            
            
            elif etype in ["recoil", "hammer_strike"]:
                m = _get_motion_by_name(engine, etype)
                if m and m.sound_path:
                    paths = m.sound_path if isinstance(m.sound_path, list) else [m.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, m.volume))
                
                if m is not None and getattr(m, "gif_path", None):
                    hid = e.get("hand_id")
                    if hid is not None and hid in centers:
                        cx = int(centers[hid][0] * width)
                        cy = int(centers[hid][1] * height)
                        visual_events.append({
                            "t": time_s,
                            "etype": etype,
                            "x": cx,
                            "y": cy,
                            "gif_path": getattr(m, "gif_path"),
                            "scale_h": float(getattr(m, "gif_scale_h", 0.20)),
                        })
            
            
            elif etype.startswith("proximity:"):
                rule = _get_rule_by_event_type(engine, etype)
                if rule and rule.sound_path:
                    paths = rule.sound_path if isinstance(rule.sound_path, list) else [rule.sound_path]
                    for p in paths:
                        sound_events.append((time_s, p, rule.volume))
                
                if rule is not None and getattr(rule, "gif_path", None):
                    
                    pair = e.get("pair")
                    if pair is not None and len(pair) == 2:
                        hid1, hid2 = pair
                        if hid1 in centers and hid2 in centers:
                            
                            cx = int((centers[hid1][0] + centers[hid2][0]) / 2.0 * width)
                            cy = int((centers[hid1][1] + centers[hid2][1]) / 2.0 * height)
                            visual_events.append({
                                "t": time_s,
                                "etype": etype,
                                "x": cx,
                                "y": cy,
                                "gif_path": getattr(rule, "gif_path"),
                                "scale_h": float(getattr(rule, "gif_scale_h", 0.20)),
                            })

        frame_idx += 1

    cap.release()
    print(f"Processed {frame_idx} frames. Found {len(sound_events)} sound events, {len(visual_events)} visual events.")

    
    print("Combining detected sounds and GIF overlays with the original video...")

    
    base_clip = VideoFileClip(filename)
    video_duration = base_clip.duration

    
    audio_clips = []
    for t, path, vol in sound_events:
        try:
            snd = AudioFileClip(path).volumex(vol).set_start(t)
            audio_clips.append(snd)
            print(f"Adding sound at {t:.2f}s: {os.path.basename(path)} (vol={vol})")
        except Exception as e:
            print(f"Could not load sound {path}: {e}")

    
    overlay_clips = []
    for ev in visual_events:
        gif_path = ev["gif_path"]
        try:
            if not os.path.exists(gif_path):
                print(f"GIF not found, skipping: {gif_path}")
                continue
            gif_clip = VideoFileClip(gif_path, has_mask=True)
            target_h = max(1, int(height * float(ev.get("scale_h", 0.20))))
            gif_clip = gif_clip.resize(height=target_h)
            
            pos_x = int(ev["x"] - gif_clip.w / 2)
            pos_y = int(ev["y"] - gif_clip.h / 2)
            gif_clip = gif_clip.set_start(ev["t"]).set_position((pos_x, pos_y))
            overlay_clips.append(gif_clip)
            print(f"Adding GIF at {ev['t']:.2f}s: {os.path.basename(gif_path)} @ ({pos_x},{pos_y}) h={target_h}")
        except Exception as e:
            print(f"Could not load GIF {gif_path}: {e}")

    
    final_video = CompositeVideoClip([base_clip] + overlay_clips, size=(width, height))

    
    if base_clip.audio and audio_clips:
        final_audio = CompositeAudioClip([base_clip.audio] + audio_clips).set_duration(video_duration)
    elif audio_clips:
        final_audio = CompositeAudioClip(audio_clips).set_duration(video_duration)
    else:
        final_audio = base_clip.audio  

    if final_audio is not None:
        final_video = final_video.set_audio(final_audio)

    
    base, _ = os.path.splitext(os.path.basename(filename))
    out_dir = os.path.dirname(filename) or "."
    output_path = os.path.join(out_dir, f"{base}_sfx.mp4")
    
    
    
    final_video.write_videofile(
    
    
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=fps,  
        preset='veryslow',  
        threads=4,  
        bitrate="8000k",  
        audio_bitrate="192k",  
        verbose=False,  
        write_logfile=False  
    )
    
    
    
    final_video.close()
    base_clip.close()
    for ac in audio_clips:
        try:
            ac.close()
        except Exception:
            pass
    for oc in overlay_clips:
        try:
            oc.close()
        except Exception:
            pass

    print(f"Done. Wrote: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
    