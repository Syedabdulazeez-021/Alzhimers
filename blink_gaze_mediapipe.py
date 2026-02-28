import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot

# -------------------------------
# Initialize MediaPipe FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Blink detection parameters
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 2

blink_times = []
counter = 0

cap = cv2.VideoCapture(0)

# -------------------------------
# EAR Calculation
# -------------------------------
def calculate_EAR(eye_points):
    vertical1 = hypot(eye_points[1][0] - eye_points[5][0],
                      eye_points[1][1] - eye_points[5][1])
    vertical2 = hypot(eye_points[2][0] - eye_points[4][0],
                      eye_points[2][1] - eye_points[4][1])
    horizontal = hypot(eye_points[0][0] - eye_points[3][0],
                       eye_points[0][1] - eye_points[3][1])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    blink_rate = 0
    gaze = "CENTER"

    if results.multi_face_landmarks:
        mesh_points = np.array(
            [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
             for p in results.multi_face_landmarks[0].landmark]
        )

        # ---------------- Blink Detection ----------------
        left_eye = mesh_points[LEFT_EYE]
        right_eye = mesh_points[RIGHT_EYE]

        leftEAR = calculate_EAR(left_eye)
        rightEAR = calculate_EAR(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EAR_THRESHOLD:
            counter += 1
        else:
            if counter >= CONSEC_FRAMES:
                blink_times.append(time.time())
            counter = 0

        # Sliding 60-second window
        current_time = time.time()
        blink_times = [t for t in blink_times if current_time - t <= 60]
        blink_rate = len(blink_times)

        # ---------------- Improved Gaze Detection ----------------
        left_iris = mesh_points[LEFT_IRIS]
        right_iris = mesh_points[RIGHT_IRIS]

        left_center = np.mean(left_iris, axis=0)
        right_center = np.mean(right_iris, axis=0)

        # Eye corners
        left_corner_left = mesh_points[33]
        left_corner_right = mesh_points[133]

        right_corner_left = mesh_points[362]
        right_corner_right = mesh_points[263]

        # Normalize iris position inside each eye
        left_ratio = (left_center[0] - left_corner_left[0]) / (
            left_corner_right[0] - left_corner_left[0]
        )

        right_ratio = (right_center[0] - right_corner_left[0]) / (
            right_corner_right[0] - right_corner_left[0]
        )

        avg_ratio = (left_ratio + right_ratio) / 2

        if avg_ratio < 0.40:
            gaze = "LEFT"
        elif avg_ratio > 0.60:
            gaze = "RIGHT"
        else:
            gaze = "CENTER"

        # Draw iris centers
        cv2.circle(frame, tuple(left_center.astype(int)), 3, (0,255,0), -1)
        cv2.circle(frame, tuple(right_center.astype(int)), 3, (0,255,0), -1)

    # ---------------- Display Info ----------------
    cv2.putText(frame, f"Blinks/min: {blink_rate}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Gaze: {gaze}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Blink & Gaze Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
