import cv2
import mediapipe as mp
import math
import time

# --- Configuration ---
EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold (tune this value)
MAR_THRESHOLD = 0.60  # Mouth Aspect Ratio threshold for yawn (tune this value)
EYE_CLOSED_CONSEC_FRAMES = 15 # Number of consecutive frames eyes must be below threshold
YAWN_CONSEC_FRAMES = 25       # Number of consecutive frames mouth must be open for yawn

# --- State Counters and Flags ---
EYE_CLOSED_COUNTER = 0
YAWN_COUNTER = 0
ALARM_ON = False

# --- Helper Function: Calculate Euclidean Distance ---
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# --- Helper Function: Calculate Eye Aspect Ratio (EAR) ---
# Uses specific MediaPipe Face Mesh landmark indices for eyes
# Left eye landmarks: p1, p2, p3, p4, p5, p6 = 362, 385, 387, 263, 373, 380
# Right eye landmarks: p1, p2, p3, p4, p5, p6 = 33, 160, 158, 133, 153, 144
def calculate_ear(landmarks, eye_indices):
    try:
        # Vertical distances
        p2_p6 = euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        p3_p5 = euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        # Horizontal distance
        p1_p4 = euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        if p1_p4 == 0: # Avoid division by zero
            return 0.0

        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear
    except IndexError:
        # Handle cases where landmarks might not be fully detected
        # print("Warning: Could not calculate EAR due to missing landmarks.")
        return 0.5 # Return a neutral value if calculation fails

# --- Helper Function: Calculate Mouth Aspect Ratio (MAR) ---
# Uses specific MediaPipe Face Mesh landmark indices for mouth
# Vertical points: top_lip_center=13, bottom_lip_center=14
# Horizontal points: left_corner=61, right_corner=291 (or 78, 308 outer)
def calculate_mar(landmarks, mouth_indices):
    try:
        # Vertical distance
        p_top_p_bottom = euclidean_distance(landmarks[mouth_indices[0]], landmarks[mouth_indices[1]])
        # Horizontal distance (optional, can use a fixed denominator or just vertical distance)
        # p_left_p_right = euclidean_distance(landmarks[mouth_indices[2]], landmarks[mouth_indices[3]])
        # mar = p_top_p_bottom / p_left_p_right if p_left_p_right != 0 else 0

        # Simpler MAR: just the vertical distance (relative openness)
        mar = p_top_p_bottom
        return mar
    except IndexError:
        # print("Warning: Could not calculate MAR due to missing landmarks.")
        return 0.0 # Return neutral value if calculation fails


# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Face Mesh - refine_landmarks=True is crucial for detailed eye landmarks
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Use this for better eye/lip detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define landmark indices based on MediaPipe Face Mesh documentation
# https://developers.google.com/mediapipe/solutions/vision/face_mesh#face_mesh_landmark_model
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # P1, P2, P3, P4, P5, P6
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144] # P1, P2, P3, P4, P5, P6
MOUTH_INDICES = [13, 14, 61, 291] # TopLipCenter, BottomLipCenter, LeftCorner, RightCorner (for MAR calculation)

# --- Camera Initialization ---
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0) # Use 0 for default camera
time.sleep(1.0)

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[INFO] Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True # Make writeable again if needed later

    # Get frame dimensions
    img_h, img_w, _ = frame.shape

    # Default EAR and MAR
    ear = 0.5 # Assume eyes are open initially
    mar = 0.0 # Assume mouth is closed initially

    if results.multi_face_landmarks:
        # Usually only one face if max_num_faces=1
        face_landmarks = results.multi_face_landmarks[0]

        # --- Get landmarks as a list of (x, y) tuples ---
        # Important: Landmarks are normalized (0.0 - 1.0). Convert to pixel coordinates.
        landmarks_list = []
        for lm in face_landmarks.landmark:
            landmarks_list.append((int(lm.x * img_w), int(lm.y * img_h)))

        # --- Calculate EAR and MAR ---
        if len(landmarks_list) >= 468: # Ensure all landmarks are present
            left_ear = calculate_ear(landmarks_list, LEFT_EYE_INDICES)
            right_ear = calculate_ear(landmarks_list, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            mar = calculate_mar(landmarks_list, MOUTH_INDICES) # Using vertical distance for simplicity

        # --- Draw Face Mesh (Optional) ---
        # mp_drawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        # Draw refined eye/lip landmarks (optional)
        # mp_drawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS, # Or FACEMESH_IRISES
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        # --- Drowsiness Detection Logic ---
        drowsy_event = False # Flag if a drowsiness event is detected in this frame

        # Check for eye closure
        if ear < EAR_THRESHOLD:
            EYE_CLOSED_COUNTER += 1
            if EYE_CLOSED_COUNTER >= EYE_CLOSED_CONSEC_FRAMES:
                drowsy_event = True
                # Keep alarm state consistent if already on
        else:
            EYE_CLOSED_COUNTER = 0

        # Check for yawning
        # Note: A simple MAR threshold might be too sensitive (triggers on talking).
        # Consider adding duration or combining with eye state.
        if mar > MAR_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                 drowsy_event = True
                 # Keep alarm state consistent if already on
        else:
            YAWN_COUNTER = 0

        # --- Trigger Alarm ---
        if drowsy_event and not ALARM_ON:
            ALARM_ON = True
            print("ALARM: Drowsiness Detected!")
            # !!! ADD YOUR ALARM MECHANISM HERE (e.g., play sound) !!!

        elif not drowsy_event and ALARM_ON:
             ALARM_ON = False # Turn off alarm if no drowsiness detected

        # Display Alarm Status and Metrics
        if ALARM_ON:
             cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (img_w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (img_w - 150, 60), # Display MAR value
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('MediaPipe Face Mesh Drowsiness Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()