
import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25
COUNTER = 0
BLINKS = 0
TOTAL = 0
DROWSY = 0

ear_history = []

cap = cv2.VideoCapture("face.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    TOTAL += 1
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE"
    color = (0,255,255)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]

            right_eye = [(int(face_landmarks.landmark[i].x * w),
                          int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_history.append(ear)

            status = "ALERT"
            color = (0,255,0)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                DROWSY += 1
                status = "DROWSY"
                color = (0,0,255)

                if COUNTER == 10:
                    BLINKS += 1
                    os.system("aplay alarm.wav &")
            else:
                COUNTER = 0

            # Draw eye points
            for (x,y) in left_eye + right_eye:
                cv2.circle(frame, (x,y), 2, (255,0,0), -1)

            # EAR text
            cv2.putText(frame, f"EAR: {ear:.2f}", (450,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    # Accuracy
    accuracy = 100 - ((DROWSY/max(TOTAL,1))*100)

    # UI
    cv2.putText(frame, f"STATUS: {status}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)

    cv2.putText(frame, f"BLINKS: {BLINKS}", (20,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

    cv2.putText(frame, f"ACCURACY: {accuracy:.1f}%", (20,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

    # Graph
    if len(ear_history) > 50:
        graph = ear_history[-50:]
        for i in range(len(graph)-1):
            cv2.line(frame,
                     (i*10, 480-int(graph[i]*200)),
                     ((i+1)*10, 480-int(graph[i+1]*200)),
                     (255,0,0),2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
