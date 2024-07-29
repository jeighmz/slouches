import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to determine if the user is slouching
def is_slouching(landmarks):
    # Get the landmarks for shoulders and neck
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Calculate the average shoulder height
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    # Calculate the average hip height
    avg_hip_y = (left_hip.y + right_hip.y) / 2

    # Calculate the shoulder-to-hip vertical distance
    shoulder_hip_distance = avg_shoulder_y - avg_hip_y

    # Calculate the shoulder-to-nose vertical distance
    shoulder_nose_distance = avg_shoulder_y - nose.y

    # Determine slouching based on the ratio of shoulder-to-hip and shoulder-to-nose distances
    slouching_threshold = 0.4  # Adjust threshold as needed
    if shoulder_nose_distance / shoulder_hip_distance > slouching_threshold:
        return True
    return False

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw the pose annotation on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if the user is slouching
        if is_slouching(results.pose_landmarks.landmark):
            cv2.putText(image, 'Slouching', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Good Posture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Posture Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()