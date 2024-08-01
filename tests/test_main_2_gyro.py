import cv2
import mediapipe as mp
import numpy as np
import serial
import serial.tools.list_ports

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to determine if the user is slouching
def is_slouching(landmarks, gyro_data):
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

    # Adjust distances based on gyroscope data (e.g., pitch angle)
    pitch_angle = gyro_data.get('pitch', 0)
    adjusted_shoulder_hip_distance = shoulder_hip_distance * np.cos(np.radians(pitch_angle))
    adjusted_shoulder_nose_distance = shoulder_nose_distance * np.cos(np.radians(pitch_angle))

    # Determine slouching based on the ratio of adjusted shoulder-to-hip and adjusted shoulder-to-nose distances
    slouching_threshold = 0.4  # Adjust threshold as needed
    if adjusted_shoulder_nose_distance / adjusted_shoulder_hip_distance > slouching_threshold:
        return True
    return False

# Function to read gyroscope data from serial port
def read_gyro_data(serial_port):
    try:
        line = serial_port.readline().decode('utf-8').strip()
        data = line.split(',')
        gyro_data = {
            'pitch': float(data[0]),
            'roll': float(data[1]),
            'yaw': float(data[2])
        }
        return gyro_data
    except Exception as e:
        print(f"Error reading gyro data: {e}")
        return {'pitch': 0, 'roll': 0, 'yaw': 0}

# Check available COM ports
ports = list(serial.tools.list_ports.comports())
print("Available ports:")
for port in ports:
    print(port)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize serial connection to gyroscope
try:
    # Replace 'COM3' with the correct port identified from the printed list
    serial_port = serial.Serial('COM3', 9600, timeout=1)  
except serial.serialutil.SerialException as e:
    print(f"Error opening serial port: {e}")
    cap.release()
    cv2.destroyAllWindows()
    exit()

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

        # Read gyroscope data
        gyro_data = read_gyro_data(serial_port)

        # Check if the user is slouching
        if is_slouching(results.pose_landmarks.landmark, gyro_data):
            cv2.putText(image, 'Slouching', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Good Posture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Posture Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
serial_port.close()


# this isn't going to work beacuse there isn't a gyroscope connected to the computer