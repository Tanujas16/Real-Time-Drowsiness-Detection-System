from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os

# ---------------- SOUND ALARM FUNCTION ----------------
def sound_alarm(path):
    playsound.playsound(path)

# ---------------- EAR & LIP FUNCTIONS ----------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# ---------------- ARGUMENTS ----------------
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav", help="path alarm .WAV file")
args = vars(ap.parse_args())

# ---------------- CONSTANTS ----------------
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30     # must be closed for ~1 second
YAWN_THRESH = 22
YAWN_CONSEC_FRAMES = 15       # mouth open for ~0.5s
ALARM_COOLDOWN = 3            # seconds between alarms

# ---------------- GLOBAL VARIABLES ----------------
alarm_status = False
alarm_status2 = False
COUNTER = 0
YAWN_COUNTER = 0
ear_history = []
MAX_HISTORY = 5
last_alarm_time = 0

# ---------------- LOAD DETECTOR ----------------
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ---------------- START VIDEO STREAM ----------------
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Smooth EAR to avoid flickering
        ear_history.append(ear)
        if len(ear_history) > MAX_HISTORY:
            ear_history.pop(0)
        smoothed_ear = np.mean(ear_history)

        # Draw facial features
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [shape[48:60]], -1, (0, 255, 0), 1)

        # ---------------- DROWSINESS CHECK ----------------
        if smoothed_ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            COUNTER = 0

        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            if not alarm_status and (time.time() - last_alarm_time > ALARM_COOLDOWN):
                alarm_status = True
                last_alarm_time = time.time()
                Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()

            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_status = False

        # ---------------- YAWN CHECK ----------------
        if distance > YAWN_THRESH:
            YAWN_COUNTER += 1
        else:
            YAWN_COUNTER = 0

        if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
            if not alarm_status2 and (time.time() - last_alarm_time > ALARM_COOLDOWN):
                alarm_status2 = True
                last_alarm_time = time.time()
                Thread(target=sound_alarm, args=(args["alarm"],), daemon=True).start()

            cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_status2 = False

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, "EAR: {:.2f}".format(smoothed_ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
vs.stop()
