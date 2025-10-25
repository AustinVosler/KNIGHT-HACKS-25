import cv2
import mediapipe as mp
import pygame
import math
import numpy as np

# Initialize Pygame mixer
pygame.mixer.init()
sound_path = "./sounds/bang.mp3"
sound = pygame.mixer.Sound(sound_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4)  # max 4 hands
mp_draw = mp.solutions.drawing_utils

# Finger indices in MediaPipe
FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

def is_finger_extended_3d(landmarks, finger_indices, threshold=0.8):
    points = [np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) for i in finger_indices]
    v1 = points[1] - points[0]
    v2 = points[2] - points[1]
    v3 = points[3] - points[2]
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    return np.dot(v1, v2) > threshold and np.dot(v2, v3) > threshold

def is_gun_shape(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb = is_finger_extended_3d(landmarks, FINGER_INDICES["thumb"])
    index = is_finger_extended_3d(landmarks, FINGER_INDICES["index"])
    middle = is_finger_extended_3d(landmarks, FINGER_INDICES["middle"])
    ring = not is_finger_extended_3d(landmarks, FINGER_INDICES["ring"])
    pinky = not is_finger_extended_3d(landmarks, FINGER_INDICES["pinky"])
    return thumb and index and middle and ring

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Store per-hand state using MediaPipe hand index
hand_states = {}  # key: hand index, value: dict with wrist, gun_ready, motion_triggered

cap = cv2.VideoCapture(0)
motion_threshold = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[0]
            wrist_pos = (wrist.x, wrist.y)
            
            if i not in hand_states:
                hand_states[i] = {"prev_wrist": None, "gun_ready": False, "motion_triggered": False}
            
            state = hand_states[i]
            
            if is_gun_shape(hand_landmarks):
                if not state["gun_ready"]:
                    # First frame gun pose detected for this hand
                    state["gun_ready"] = True
                    state["motion_triggered"] = False
                    state["prev_wrist"] = wrist_pos
                else:
                    # Check for recoil motion
                    movement = distance(wrist_pos, state["prev_wrist"])
                    if movement > motion_threshold and not state["motion_triggered"]:
                        print(f"Bang! Hand {i}")
                        sound.play()
                        state["motion_triggered"] = True
                    state["prev_wrist"] = wrist_pos
            else:
                # Reset if gun pose lost
                state["gun_ready"] = False
                state["motion_triggered"] = False
                state["prev_wrist"] = None
    
    cv2.imshow("Gun Motion Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
