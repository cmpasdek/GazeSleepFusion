import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

# Camera setup
cap = cv2.VideoCapture(0)

# Calibration variables
calibrated = False
calibration_offset = 0
calibration_samples = []
CALIBRATION_SAMPLES_NEEDED = 30

# Gaze direction thresholds (will be adjusted during calibration)
LEFT_THRESHOLD = -0.1
RIGHT_THRESHOLD = 0.1

def get_gaze_direction(face_landmarks, frame_width, frame_height):
    # Get the relevant landmarks for gaze estimation
    # Left eye
    left_eye_landmarks = [
        face_landmarks.landmark[33],  # Left eye left corner
        face_landmarks.landmark[133], # Left eye right corner
        face_landmarks.landmark[159], # Left eye bottom
        face_landmarks.landmark[145], # Left eye top
    ]
    
    # Right eye
    right_eye_landmarks = [
        face_landmarks.landmark[362], # Right eye left corner
        face_landmarks.landmark[263], # Right eye right corner
        face_landmarks.landmark[386], # Right eye bottom
        face_landmarks.landmark[374], # Right eye top
    ]
    
    # Calculate eye center points
    left_eye_center = np.mean([(lm.x * frame_width, lm.y * frame_height) for lm in left_eye_landmarks], axis=0)
    right_eye_center = np.mean([(lm.x * frame_width, lm.y * frame_height) for lm in right_eye_landmarks], axis=0)
    
    # Get nose tip (for head direction)
    nose_tip = face_landmarks.landmark[4]
    nose_tip_point = (nose_tip.x * frame_width, nose_tip.y * frame_height)
    
    # Calculate horizontal gaze direction (simple approach)
    # The ratio between eye centers and nose tip gives an indication of gaze direction
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

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get gaze direction
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
                
                # Display results
                cv2.putText(frame, f"Direction: {direction}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze ratio: {gaze_ratio:.2f}", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Offset: {calibration_offset:.2f}", (50, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Gaze Direction Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()