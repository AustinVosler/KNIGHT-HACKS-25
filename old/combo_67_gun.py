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


# Finger indices in MediaPipe
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
    return max(w, 1e-3)


def hand_center(lm):
    w = lm[0]
    return (w.x, w.y)


def is_symbol_6(lm):
    """
    6: index and thumb touching (pinch), with middle, ring, pinky extended.
    """
    thumb_tip = (lm[4].x, lm[4].y)
    index_tip = (lm[8].x, lm[8].y)
    scale = hand_width_scale(lm)
    pinch = euclid_2d(thumb_tip, index_tip) / scale

    middle_ext = is_finger_extended_3d(lm, FINGER_INDICES["middle"]) 
    ring_ext = is_finger_extended_3d(lm, FINGER_INDICES["ring"]) 
    pinky_ext = is_finger_extended_3d(lm, FINGER_INDICES["pinky"]) 

    pinch_threshold = 0.25
    return pinch < pinch_threshold and middle_ext and ring_ext and pinky_ext


def is_symbol_7(lm):
    """
    7: thumb and index only extended, with the index pointing mostly downward.
    """
    thumb_ext = is_finger_extended_3d(lm, FINGER_INDICES["thumb"]) 
    index_ext = is_finger_extended_3d(lm, FINGER_INDICES["index"]) 
    middle_ext = is_finger_extended_3d(lm, FINGER_INDICES["middle"]) 
    ring_ext = is_finger_extended_3d(lm, FINGER_INDICES["ring"]) 
    pinky_ext = is_finger_extended_3d(lm, FINGER_INDICES["pinky"]) 

    # Index direction mostly downward in image space
    mcp = np.array([lm[5].x, lm[5].y])
    tip = np.array([lm[8].x, lm[8].y])
    v = _safe_unit(tip - mcp)
    down = np.array([0.0, 1.0])
    downward_enough = float(np.dot(v, down)) > 0.6

    return (
        thumb_ext
        and index_ext
        and downward_enough
        and (not middle_ext)
        and (not ring_ext)
        and (not pinky_ext)
    )


def get_hand_shape(hand_landmarks):
    lm = hand_landmarks.landmark
    return {
        "thumb": is_finger_extended_3d(lm, FINGER_INDICES["thumb"]),
        "index": is_finger_extended_3d(lm, FINGER_INDICES["index"]),
        "middle": is_finger_extended_3d(lm, FINGER_INDICES["middle"]),
        "ring": is_finger_extended_3d(lm, FINGER_INDICES["ring"]),
        "pinky": is_finger_extended_3d(lm, FINGER_INDICES["pinky"]),
    }


def is_gun_shape(shape):
    # Inspired by original: thumb + index extended, ring not extended
    return shape["thumb"] and shape["index"] and (not shape["ring"])


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # 6+7 proximity settings
    proximity_threshold = 0.18
    pair_active = False
    last_global_trigger_time = 0.0
    global_cooldown_s = 0.6

    # Gun motion settings
    motion_threshold = 0.05  # distance in normalized coords
    hand_states = {}  # key: per-frame hand index, value: {prev_wrist, gun_ready, motion_triggered}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        six_hands = []
        seven_hands = []

        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                lm = hand_landmarks.landmark
                center = hand_center(lm)

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 6/7 classification
                symbol6 = is_symbol_6(lm)
                symbol7 = is_symbol_7(lm)
                if symbol6:
                    six_hands.append((center, i))
                if symbol7:
                    seven_hands.append((center, i))

                label = "6" if symbol6 else ("7" if symbol7 else "")
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

                # Gun pose + motion
                wrist = lm[0]
                wrist_pos = (wrist.x, wrist.y)
                if i not in hand_states:
                    hand_states[i] = {
                        "prev_wrist": None,
                        "gun_ready": False,
                        "motion_triggered": False,
                    }
                state = hand_states[i]

                shape = get_hand_shape(hand_landmarks)
                if is_gun_shape(shape):
                    if not state["gun_ready"]:
                        state["gun_ready"] = True
                        state["motion_triggered"] = False
                        state["prev_wrist"] = wrist_pos
                    else:
                        movement = euclid_2d(wrist_pos, state["prev_wrist"]) if state["prev_wrist"] else 0.0
                        now = time.time()
                        if (
                            movement > motion_threshold
                            and not state["motion_triggered"]
                            and (now - last_global_trigger_time) > global_cooldown_s
                        ):
                            try:
                                sound.play()
                            except Exception as e:
                                print(f"Failed to play sound: {e}")
                            state["motion_triggered"] = True
                            last_global_trigger_time = now
                        state["prev_wrist"] = wrist_pos
                else:
                    state["gun_ready"] = False
                    state["motion_triggered"] = False
                    state["prev_wrist"] = None

        # 6+7 proximity trigger (after processing all hands)
        closest_pair = None
        closest_dist = 10.0
        for c6, i6 in six_hands:
            for c7, i7 in seven_hands:
                d = euclid_2d(c6, c7)
                if d < closest_dist:
                    closest_dist = d
                    closest_pair = (c6, c7)

        condition_met = closest_pair is not None and closest_dist < proximity_threshold
        now = time.time()
        if condition_met and not pair_active and (now - last_global_trigger_time) > global_cooldown_s:
            try:
                sound.play()
            except Exception as e:
                print(f"Failed to play sound: {e}")
            pair_active = True
            last_global_trigger_time = now

            # Visual pop
            (c6, c7) = closest_pair
            x1, y1 = int(c6[0] * frame.shape[1]), int(c6[1] * frame.shape[0])
            x2, y2 = int(c7[0] * frame.shape[1]), int(c7[1] * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(
                frame,
                "TRIGGER 6+7",
                (min(x1, x2), max(30, min(y1, y2) - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        if not condition_met:
            pair_active = False

        # Draw proximity helper
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

        cv2.imshow("Combo: 6+7 & Gun Motion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
