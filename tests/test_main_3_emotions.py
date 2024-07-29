import cv2
import mediapipe as mp
from fer import FER

# Initialize Mediapipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize FER (Facial Expression Recognition)
emotion_detector = FER()

# Initialize video capture
cap = cv2.VideoCapture(0)

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

            # Emotion detection
            # Convert the landmarks to a bounding box
            height, width, _ = image.shape
            x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * width)
            x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * width)
            y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * height)
            y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * height)

            face_roi = image[y_min:y_max, x_min:x_max]
            if face_roi.size > 0:
                # Detect emotion in the face ROI
                emotion, score = emotion_detector.top_emotion(face_roi)
                if emotion is None:
                    emotion, score = 'Neutral', 1.0  # Default emotion when no dominant emotion is detected
                # Display emotion text near the face bounding box
                cv2.putText(image, f'{emotion} ({score:.2f})', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Face Mesh and Emotion Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()