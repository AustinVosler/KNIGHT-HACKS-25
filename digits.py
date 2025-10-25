import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Optional: define connections manually
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=9999, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural webcam view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, c = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

                # Draw lines
                for start, end in connections:
                    cv2.line(frame, landmarks[start], landmarks[end], (0, 255, 0), 2)

                # Draw dots on knuckles
                for x, y in landmarks:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Hand Skeleton", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
