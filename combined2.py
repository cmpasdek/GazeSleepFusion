import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- Window Name ---
WINDOW_NAME = 'Drowsiness and Gaze Detection' # Define a consistent window name

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Drowsiness Configuration ---
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60
EYE_CLOSED_CONSEC_FRAMES = 15
YAWN_CONSEC_FRAMES = 25

EYE_CLOSED_COUNTER = 0
YAWN_COUNTER = 0
ALARM_ON = False

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 61, 291]

# --- Gaze Configuration ---
calibrated = False
calibration_offset = 0
calibration_samples = []
CALIBRATION_SAMPLES_NEEDED = 30
LEFT_THRESHOLD = -0.15
RIGHT_THRESHOLD = 0.15

# --- Helper Functions (Keep all helper functions as before) ---
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(landmarks, eye_indices):
    try:
        p2_p6 = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        p3_p5 = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        p1_p4 = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        if p1_p4 == 0: return 0.0
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear
    except IndexError: return 0.5

def calculate_mar(landmarks, mouth_indices):
    try:
        p_top_p_bottom = euclidean_distance(landmarks[mouth_indices[0]], landmarks[mouth_indices[1]])
        return p_top_p_bottom
    except IndexError: return 0.0

def get_gaze_ratio(face_landmarks_object):
    if len(face_landmarks_object.landmark) > 4:
        nose_tip = face_landmarks_object.landmark[4]
        gaze_ratio = (nose_tip.x - 0.5) * 2
        return gaze_ratio
    else: return 0.0

def determine_direction(gaze_ratio, offset=0):
    adjusted_ratio = gaze_ratio - offset
    if adjusted_ratio < LEFT_THRESHOLD: return "LEFT"
    elif adjusted_ratio > RIGHT_THRESHOLD: return "RIGHT"
    else: return "STRAIGHT"

# --- Camera Initialization ---
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[INFO] Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    ear, mar, gaze_ratio = 0.5, 0.0, 0.0
    direction = "N/A"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_list = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks.landmark]

        if len(landmarks_list) >= 468:
            left_ear = calculate_ear(landmarks_list, LEFT_EYE_INDICES)
            right_ear = calculate_ear(landmarks_list, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks_list, MOUTH_INDICES)
            gaze_ratio = get_gaze_ratio(face_landmarks)

        if not calibrated:
            cv2.putText(frame, "CALIBRATION: Look straight", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {len(calibration_samples)}/{CALIBRATION_SAMPLES_NEEDED}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(calibration_samples) < CALIBRATION_SAMPLES_NEEDED:
                if len(landmarks_list) >= 468:
                    calibration_samples.append(gaze_ratio)
            else:
                calibration_offset = np.mean(calibration_samples)
                calibrated = True
                cv2.putText(frame, "Calibration Complete!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Changed y-pos slightly
                # *** Use the SAME window name here ***
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(2000)
                # No need for another imshow here, the main one at the end handles it
                continue # Skip the rest of the loop for this "complete" frame display

        else: # --- Post-Calibration Logic ---
            direction = determine_direction(gaze_ratio, calibration_offset)
            drowsy_event = False

            if ear < EAR_THRESHOLD:
                EYE_CLOSED_COUNTER += 1
                if EYE_CLOSED_COUNTER >= EYE_CLOSED_CONSEC_FRAMES: drowsy_event = True
            else: EYE_CLOSED_COUNTER = 0

            if mar > MAR_THRESHOLD:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= YAWN_CONSEC_FRAMES: drowsy_event = True
            else: YAWN_COUNTER = 0

            if drowsy_event and not ALARM_ON:
                ALARM_ON = True
                print("ALARM: Drowsiness Detected!")
                # !!! ADD ALARM MECHANISM HERE !!!
            elif not drowsy_event and ALARM_ON:
                ALARM_ON = False

            # --- Display Info (Post-Calibration) ---
            y_offset = 30
            if ALARM_ON:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            cv2.putText(frame, f"Gaze: {direction}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Removed optional gaze ratio/offset display for clarity

    # Display the resulting frame using the consistent window name
    # This handles display during calibration AND after calibration
    cv2.imshow(WINDOW_NAME, frame)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()