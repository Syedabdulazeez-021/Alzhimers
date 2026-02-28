import cv2
import numpy as np
import time
import csv
import tkinter as tk
from PIL import Image, ImageTk

# -------------------------------
# Load Haar Cascades
# -------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# -------------------------------
# CSV Setup
# -------------------------------
csv_file = "blink_gaze_data.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time(s)", "BlinkRate", "Gaze"])

# -------------------------------
# Global Variables
# -------------------------------
cap = cv2.VideoCapture(0)

start_time = time.time()
last_log_time = 0

# Blink logic variables
WARMUP_TIME = 5          # seconds
closed_frames = 0
MIN_CLOSED_FRAMES = 4    # increase to 5 if still noisy
blink_times = []         # timestamps of blinks

# -------------------------------
# Gaze Detection Function
# -------------------------------
def get_gaze(eyes, face_width):
    if len(eyes) == 0:
        return "UNKNOWN"

    x_positions = [x + w / 2 for (x, y, w, h) in eyes]
    avg_x = sum(x_positions) / len(x_positions)

    if avg_x < face_width / 3:
        return "LEFT"
    elif avg_x > 2 * face_width / 3:
        return "RIGHT"
    else:
        return "CENTER"

# -------------------------------
# Main Update Function
# -------------------------------
def update_frame():
    global closed_frames, last_log_time, blink_times

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    elapsed = time.time() - start_time
    gaze = "UNKNOWN"

    # ---------------- Warm-up phase ----------------
    if elapsed < WARMUP_TIME:
        cv2.putText(frame, "Calibrating... Please wait",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            # -------- Correct Blink Detection --------
            if len(eyes) >= 2:
                if closed_frames >= MIN_CLOSED_FRAMES:
                    blink_times.append(time.time())
                closed_frames = 0
            else:
                closed_frames += 1

            # -------- Gaze Detection --------
            gaze = get_gaze(eyes, w)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # ---------------- Blink Rate (Sliding Window) ----------------
    current_time = time.time()
    blink_times[:] = [t for t in blink_times if current_time - t <= 60]
    blink_rate = len(blink_times)

    # ---------------- Display Info ----------------
    cv2.putText(frame, f"Blinks/min: {blink_rate}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # ---------------- Log to CSV every 5 seconds ----------------
    if elapsed > WARMUP_TIME and time.time() - last_log_time > 5:
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([int(elapsed), blink_rate, gaze])
        last_log_time = time.time()

    # ---------------- Show in GUI ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("Blink Rate & Gaze Detection")

video_label = tk.Label(root)
video_label.pack()

info_label = tk.Label(
    root,
    text="Calibrates for 5 seconds • Press ESC in window to exit",
    font=("Arial", 12)
)
info_label.pack()

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
