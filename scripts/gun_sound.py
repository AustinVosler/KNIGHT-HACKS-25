import cv2
import mediapipe as mp
import pygame

# Initialize Pygame mixer
pygame.mixer.init()
sound_path = "sounds/bang.mp3"  
sound = pygame.mixer.Sound(sound_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Helper function to determine if finger is extended
def is_finger_extended(landmarks, finger_indices):
    # Tip vs PIP comparison (y-coordinate for vertical hand, adjust if needed)
    tip_y = landmarks[finger_indices[3]].y
    pip_y = landmarks[finger_indices[1]].y
    return tip_y < pip_y  # True if finger is extended

# Finger indices in MediaPipe
FINGER_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

# Detect "gun-shaped" hand
def is_gun_shape(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb = is_finger_extended(landmarks, FINGER_INDICES["thumb"])
    index = is_finger_extended(landmarks, FINGER_INDICES["index"])
    middle = is_finger_extended(landmarks, FINGER_INDICES["middle"])
    ring = is_finger_extended(landmarks, FINGER_INDICES["ring"])
    pinky = is_finger_extended(landmarks, FINGER_INDICES["pinky"])
    # Gun shape: thumb + index + middle extended, ring + pinky curled
    print(thumb, index, middle, ring, pinky)
    return thumb and index and middle and not ring 

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip frame for natural view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    # Draw hand landmarks and check shape
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_gun_shape(hand_landmarks):
                print("Gun-shaped hand detected!")
                sound.play()
    
    cv2.imshow("Hand Detection", frame)
    
    key = cv2.waitKey(0) & 0xFF  # wait indefinitely for key press
    if key == ord('q'):
        break
    elif key == ord('z'):
        continue  # capture next frame

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
