import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from collections import deque

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize a deque to store emotion results over a short time window
emotion_window = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face mesh annotation on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Convert the landmarks to a bounding box
            height, width, _ = image.shape
            x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * width)
            x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * width)
            y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * height)
            y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * height)

            face_roi = image[y_min:y_max, x_min:x_max]
            if face_roi.size > 0:
                # Preprocess the face ROI
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = np.expand_dims(face_roi, axis=0)

                # Detect emotion in the face ROI using DeepFace
                try:
                    emotion_analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = emotion_analysis['dominant_emotion']
                    score = emotion_analysis['emotion'][emotion]
                except Exception as e:
                    emotion, score = 'Neutral', 1.0  # Default emotion when no dominant emotion is detected

                # Store the emotion result in the deque
                emotion_window.append((emotion, score))

                # Aggregate emotion results over the time window
                if len(emotion_window) == emotion_window.maxlen:
                    aggregated_emotion = max(set([e[0] for e in emotion_window]), key=[e[0] for e in emotion_window].count)
                    aggregated_score = np.mean([e[1] for e in emotion_window if e[0] == aggregated_emotion])
                else:
                    aggregated_emotion, aggregated_score = emotion, score

                # Display emotion text near the face bounding box
                cv2.putText(image, f'{aggregated_emotion} ({aggregated_score:.2f})', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Face Mesh and Emotion Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()