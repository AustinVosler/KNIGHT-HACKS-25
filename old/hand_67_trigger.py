import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import time


# Initialize Pygame mixer and load sound
pygame.mixer.init()
sound_path = "./sounds/bang.mp3"
sound = pygame.mixer.Sound(sound_path)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils


# Finger indices in MediaPipe (landmark indices)
FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def _safe_unit(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def is_finger_extended_3d(landmarks, finger_indices, threshold=0.8):
    """
    Returns True if the finger joints are mostly co-linear (extended), using cosine similarity.
    landmarks: iterable of MediaPipe landmarks
    finger_indices: [mcp, pip, dip, tip] for the finger
    threshold: dot-product threshold between adjacent segment unit vectors
    """
    points = [
        np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])
        for i in finger_indices
    ]
    v1 = _safe_unit(points[1] - points[0])
    v2 = _safe_unit(points[2] - points[1])
    v3 = _safe_unit(points[3] - points[2])
    return np.dot(v1, v2) > threshold and np.dot(v2, v3) > threshold


def euclid_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def hand_width_scale(lm):
    """Approximate hand scale using distance between index MCP (5) and pinky MCP (17)."""
    a = (lm[5].x, lm[5].y)
    b = (lm[17].x, lm[17].y)
    w = euclid_2d(a, b)
    # Avoid zero scale
    return max(w, 1e-3)


def hand_center(lm):
    """Use wrist as center for simplicity."""
    w = lm[0]
    return (w.x, w.y)


def is_symbol_6(lm):
    """
    6 (per request): index and thumb touching (pinch), other fingers extended.
    - Thumb tip ~ Index tip proximity
    - Middle, Ring, Pinky extended
    """
    thumb_tip = (lm[4].x, lm[4].y)
    index_tip = (lm[8].x, lm[8].y)
    scale = hand_width_scale(lm)
    pinch = euclid_2d(thumb_tip, index_tip) / scale

    middle_ext = is_finger_extended_3d(lm, FINGER_INDICES["middle"]) 
    ring_ext = is_finger_extended_3d(lm, FINGER_INDICES["ring"]) 
    pinky_ext = is_finger_extended_3d(lm, FINGER_INDICES["pinky"]) 

    # Heuristic thresholds; tweak pinch if needed
    pinch_threshold = 0.25
    return pinch < pinch_threshold and middle_ext and ring_ext and pinky_ext


def is_symbol_7(lm):
    """
    7 (per request): thumb and index only extended, with index pointing mostly downward.
    - Thumb extended
    - Index extended and oriented downward (tip below mcp directionally)
    - Middle, Ring, Pinky NOT extended
    """
    thumb_ext = is_finger_extended_3d(lm, FINGER_INDICES["thumb"]) 
    index_ext = is_finger_extended_3d(lm, FINGER_INDICES["index"]) 
    middle_ext = is_finger_extended_3d(lm, FINGER_INDICES["middle"]) 
    ring_ext = is_finger_extended_3d(lm, FINGER_INDICES["ring"]) 
    pinky_ext = is_finger_extended_3d(lm, FINGER_INDICES["pinky"]) 

    # Direction of index from MCP(5) to TIP(8) should be mostly downward in image space
    mcp = np.array([lm[5].x, lm[5].y])
    tip = np.array([lm[8].x, lm[8].y])
    v = _safe_unit(tip - mcp)
    down = np.array([0.0, 1.0])
    downward_enough = float(np.dot(v, down)) > 0.6  # cos(angle) > 0.6 => < ~53 degrees from vertical

    return (
        thumb_ext
        and index_ext
        and downward_enough
        and (not middle_ext)
        and (not ring_ext)
        and (not pinky_ext)
    )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # Proximity threshold between a 6-hand and a 7-hand (normalized image coords)
    proximity_threshold = 0.50

    # Avoid multiple triggers while poses remain together
    pair_active = False
    last_trigger_time = 0.0
    cooldown_s = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Mirror like a selfie camera
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        six_hands = []   # list of (center_xy, index_in_frame)
        seven_hands = [] # list of (center_xy, index_in_frame)

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                lm = hand_landmarks.landmark
                center = hand_center(lm)

                # Classify
                symbol6 = is_symbol_6(lm)
                symbol7 = is_symbol_7(lm)

                if symbol6:
                    six_hands.append((center, i))
                if symbol7:
                    seven_hands.append((center, i))

                # Draw landmarks and labels
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = ""
                if symbol6:
                    label = "6"
                elif symbol7:
                    label = "7"
                if label:
                    cx, cy = int(center[0] * frame.shape[1]), int(center[1] * frame.shape[0])
                    cv2.putText(
                        frame,
                        label,
                        (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

        # Find the closest 6-7 pair, if any
        closest_pair = None
        closest_dist = 10.0
        for c6, i6 in six_hands:
            for c7, i7 in seven_hands:
                d = euclid_2d(c6, c7)
                if d < closest_dist:
                    closest_dist = d
                    closest_pair = (c6, c7)

        now = time.time()
        condition_met = closest_pair is not None and closest_dist < proximity_threshold

        if condition_met and not pair_active and (now - last_trigger_time) > cooldown_s:
            # Trigger once when entering the valid state
            try:
                sound.play()
                print("Trigger sound played!")
            except Exception as e:
                print(f"Failed to play sound: {e}")
            pair_active = True
            last_trigger_time = now

            # Visual indicator of trigger
            if closest_pair is not None:
                (c6, c7) = closest_pair
                x1, y1 = int(c6[0] * frame.shape[1]), int(c6[1] * frame.shape[0])
                x2, y2 = int(c7[0] * frame.shape[1]), int(c7[1] * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(
                    frame,
                    "TRIGGER",
                    (min(x1, x2), max(30, min(y1, y2) - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

        if not condition_met:
            # Reset when the valid pair no longer exists or separates
            pair_active = False

        # Draw proximity hint if a pair exists
        if closest_pair is not None:
            (c6, c7) = closest_pair
            x1, y1 = int(c6[0] * frame.shape[1]), int(c6[1] * frame.shape[0])
            x2, y2 = int(c7[0] * frame.shape[1]), int(c7[1] * frame.shape[0])
            color = (0, 255, 0) if closest_dist < proximity_threshold else (255, 255, 0)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"d={closest_dist:.2f}",
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("6+7 Hand Trigger", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
