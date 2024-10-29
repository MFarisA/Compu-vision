import cv2
import dlib
import numpy as np
import joblib

# Load the pre-trained SVC model
model_filename = 'gender_classifier_model.pkl'
loaded_model = joblib.load(model_filename)
print("Model loaded successfully for future predictions.")

# Paths to your models
predictor_path = 'dat/shape_predictor_68_face_landmarks.dat'  # Update this path
face_recognition_model_path = 'dat/dlib_face_recognition_resnet_model_v1.dat'  # Update this path

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize the face recognition model
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Function to extract features
def extract_features(img):
    detections = detector(img)

    if len(detections) == 0:
        return None, None  # No face detected, return None for both

    shape = predictor(img, detections[0])
    face_descriptor = face_recognition_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor), detections[0]  # Return both descriptor and detection

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract features from the frame
    features, face_rect = extract_features(img_rgb)

    if features is not None and face_rect is not None:
        # Reshape features for the model
        features_reshaped = features.reshape(1, -1)

        # Predict gender
        prediction = loaded_model.predict(features_reshaped)
        gender = "Female" if prediction[0] == 0 else "Male"  # Assuming 0: Male, 1: Female

        # Draw a rectangle around the face
        x1, y1, x2, y2 = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        # Draw the label on the frame
        cv2.putText(frame, gender, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the predicted gender
    cv2.imshow("Gender Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
