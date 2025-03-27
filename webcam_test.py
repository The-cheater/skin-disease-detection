import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("wound_classifier_model.h5")

# Define class labels
class_labels = ["Stab wound", "Laceration", "Ingrown nails", "Cut", "Burns", "Bruises", "Abrasions"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect phone vs computer use
    height, width, _ = frame.shape
    aspect_ratio = height / width
    device_type = "Phone" if aspect_ratio > 1.3 else "Computer"

    # Preprocess the image
    img = cv2.resize(frame, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    # Predict wound type
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    label = class_labels[class_index]

    # Display results
    text = f"Wound Type: {label} | Device: {device_type}"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Wound Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
