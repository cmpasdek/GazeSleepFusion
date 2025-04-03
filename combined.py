import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# Calibration variables
calibrated = False
calibration_offset = 0
calibration_samples = []
CALIBRATION_SAMPLES_NEEDED = 30

# Gaze direction thresholds
LEFT_THRESHOLD = -0.1
RIGHT_THRESHOLD = 0.1

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Function to process image for drowsiness
def process_image(frame):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.6

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all face locations using the default HOG model
    face_locations = face_recognition.face_locations(rgb_frame)

    eye_flag = mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth = np.array(landmarks['bottom_lip'])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_AR_THRESH:
            eye_flag = True

        if mar > MOUTH_AR_THRESH:
            mouth_flag = True

    return eye_flag, mouth_flag

# Function to get gaze direction
def get_gaze_direction(face_landmarks, frame_width, frame_height):
    # Left eye landmarks
    left_eye_landmarks = [
        face_landmarks.landmark[33],  
        face_landmarks.landmark[133],
        face_landmarks.landmark[159],
        face_landmarks.landmark[145]
    ]
    
    # Right eye landmarks
    right_eye_landmarks = [
        face_landmarks.landmark[362],
        face_landmarks.landmark[263],
        face_landmarks.landmark[386],
        face_landmarks.landmark[374]
    ]
    
    # Calculate eye center points
    left_eye_center = np.mean([(lm.x * frame_width, lm.y * frame_height) for lm in left_eye_landmarks], axis=0)
    right_eye_center = np.mean([(lm.x * frame_width, lm.y * frame_height) for lm in right_eye_landmarks], axis=0)
    
    # Get nose tip for head direction
    nose_tip = face_landmarks.landmark[4]
    nose_tip_point = (nose_tip.x * frame_width, nose_tip.y * frame_height)
    
    # Calculate horizontal gaze direction (simple approach)
    gaze_ratio = (nose_tip.x - 0.5) * 2  # Normalized to [-1, 1] range
    
    return gaze_ratio

def determine_direction(gaze_ratio, offset=0):
    adjusted_ratio = gaze_ratio - offset
    
    if adjusted_ratio < LEFT_THRESHOLD:
        return "LEFT"
    elif adjusted_ratio > RIGHT_THRESHOLD:
        return "RIGHT"
    else:
        return "STRAIGHT"

# Camera setup
cap = cv2.VideoCapture(0)
count = score = 0

# Calibration loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
    # Process drowsiness
    image_resized = cv2.resize(frame, (640, 360))  # Lower resolution for processing
    eye_flag, mouth_flag = process_image(image_resized)
    
    # Update score
    if eye_flag or mouth_flag:
        score += 1
    else:
        score -= 1
        if score < 0:
            score = 0
    
    # Ensure the score does not exceed 10
    if score > 10:
        score = 10

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = 10
    text_y = frame.shape[0] - 10
    text = f"Score: {score}"
    cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if score >= 5:
        text_x = frame.shape[1] - 130
        text_y = 40
        text = "Drowsy"
        cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Convert to RGB for MediaPipe face mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Gaze direction
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            gaze_ratio = get_gaze_direction(face_landmarks, frame_width, frame_height)
            
            # Calibration mode
            if not calibrated:
                cv2.putText(frame, "CALIBRATION: Look straight", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Samples collected: {len(calibration_samples)}/{CALIBRATION_SAMPLES_NEEDED}", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(calibration_samples) < CALIBRATION_SAMPLES_NEEDED:
                    calibration_samples.append(gaze_ratio)
                else:
                    calibration_offset = np.mean(calibration_samples)
                    calibrated = True
                    cv2.putText(frame, "CALIBRATION COMPLETE!", (50, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.waitKey(2000)  # Show message for 2 seconds
            else:
                # Determine direction with calibration offset
                direction = determine_direction(gaze_ratio, calibration_offset)
                
                # Display gaze direction
                cv2.putText(frame, f"Direction: {direction}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze ratio: {gaze_ratio:.2f}", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Offset: {calibration_offset:.2f}", (50, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Drowsiness and Gaze Direction', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
