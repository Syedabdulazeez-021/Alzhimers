import cv2
import mediapipe as mp
import numpy as np
import time
import random
from math import hypot

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# -------------------------------
# Experiment Settings
# -------------------------------
stimulus_positions = ["LEFT", "CENTER", "RIGHT"]
stimulus_duration = 3
total_trials = 10

trial_count = 0
correct_count = 0
reaction_times = []

current_stimulus = random.choice(stimulus_positions)
stimulus_time = time.time()
reaction_logged = False

cap = cv2.VideoCapture(0)

def calculate_EAR(eye_points):
    vertical1 = hypot(eye_points[1][0] - eye_points[5][0],
                      eye_points[1][1] - eye_points[5][1])
    vertical2 = hypot(eye_points[2][0] - eye_points[4][0],
                      eye_points[2][1] - eye_points[4][1])
    horizontal = hypot(eye_points[0][0] - eye_points[3][0],
                       eye_points[0][1] - eye_points[3][1])
    return (vertical1 + vertical2) / (2.0 * horizontal)

print("Experiment Started")
print("Look at the red dot as quickly as possible.")

while trial_count < total_trials:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze = "CENTER"

    # Change stimulus after duration
    if time.time() - stimulus_time > stimulus_duration:
        trial_count += 1
        current_stimulus = random.choice(stimulus_positions)
        stimulus_time = time.time()
        reaction_logged = False

    # Draw stimulus
    if current_stimulus == "LEFT":
        cv2.circle(frame, (int(w*0.2), int(h*0.5)), 30, (0,0,255), -1)
    elif current_stimulus == "CENTER":
        cv2.circle(frame, (int(w*0.5), int(h*0.5)), 30, (0,0,255), -1)
    elif current_stimulus == "RIGHT":
        cv2.circle(frame, (int(w*0.8), int(h*0.5)), 30, (0,0,255), -1)

    if results.multi_face_landmarks:
        mesh_points = np.array(
            [(int(p.x * w), int(p.y * h))
             for p in results.multi_face_landmarks[0].landmark]
        )

        left_iris = mesh_points[LEFT_IRIS]
        right_iris = mesh_points[RIGHT_IRIS]

        left_center = np.mean(left_iris, axis=0)
        right_center = np.mean(right_iris, axis=0)

        left_ratio = (left_center[0] - mesh_points[33][0]) / (
            mesh_points[133][0] - mesh_points[33][0]
        )

        right_ratio = (right_center[0] - mesh_points[362][0]) / (
            mesh_points[263][0] - mesh_points[362][0]
        )

        avg_ratio = (left_ratio + right_ratio) / 2

        if avg_ratio < 0.40:
            gaze = "LEFT"
        elif avg_ratio > 0.60:
            gaze = "RIGHT"
        else:
            gaze = "CENTER"

        if not reaction_logged and gaze == current_stimulus:
            reaction_time = time.time() - stimulus_time
            reaction_times.append(reaction_time)
            correct_count += 1
            print(f"Trial {trial_count}: Correct in {reaction_time:.2f} sec")
            reaction_logged = True

    cv2.putText(frame, f"Trial: {trial_count}/{total_trials}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Gaze: {gaze}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gaze Stimulus Experiment", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# Final Results
# -------------------------------
print("\nExperiment Finished")
print(f"Accuracy: {correct_count}/{total_trials}")

if reaction_times:
    print(f"Average Reaction Time: {sum(reaction_times)/len(reaction_times):.2f} sec")
