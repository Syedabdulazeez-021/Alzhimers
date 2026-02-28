import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

blink_count = 0
eye_closed = False
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if len(eyes) >= 2:
            if not eye_closed:
                blink_count += 1
                eye_closed = True
        else:
            eye_closed = False

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    elapsed = time.time() - start_time
    blink_rate = int((blink_count / elapsed) * 60) if elapsed > 0 else 0

    cv2.putText(frame, f"Blinks/min: {blink_rate}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Blink Rate Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
