from fer import FER
import cv2

smile_path = "docs/smile_review.jpeg"
joe_path = "docs/240207-joe-biden-ac-906p-c9ad2b.jpeg"

img = cv2.imread(joe_path)
detector = FER(mtcnn=True)
emotions = detector.detect_emotions(img)

# Assuming only one face is detected for simplicity
if emotions:
    # Get the bounding box coordinates
    bounding_box = emotions[0]["box"]
    x_min, y_min, width, height = bounding_box
    x_max = x_min + width
    y_max = y_min + height

    # Get the top emotion
    emotion, score = detector.top_emotion(img)
    print("Top Emotion:", emotion)

    # Calculate the middle left position of the bounding box
    text_x = x_min
    text_y = y_min + (y_max - y_min) // 2

    # Display the image with the emotion text in the middle left of the face bounding box
    cv2.putText(img, f'{emotion} ({score:.2f})', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the box around the face
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Smile Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected.")