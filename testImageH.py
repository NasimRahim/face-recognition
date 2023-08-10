import cv2
import numpy as np
import joblib
import os

# Load the haarcascade classifier for face detection
cascade_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\fyp\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the pre-trained LBPH model
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.read("lbph_modelHaar.yml")

# Load the trained SVM classifier (recognizer)
recognizer_path = "svm_modelHaar.pkl"
recognizer = joblib.load(recognizer_path)

# Create a dictionary to map label numbers to names
label_names = {
    1: "nasim",
    2: "khai",
    3: "hariz",
    16: "mirun",
    20: "tuan arif"
}

# Create a function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

# Create a function to recognize faces in an image
def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(image)
    predictions = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (100, 100))
        label, _ = lbph_model.predict(face_region)
        predictions.append(label)
    return predictions

# Get the paths of all image files in the "test" folder
test_folder = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\test"
image_paths = [os.path.join(test_folder, filename) for filename in os.listdir(test_folder)]

# Loop over the image paths and perform face recognition
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform face recognition on the image
    predictions = recognize_faces(image)
    
    # Draw bounding boxes and labels on the image
    faces = detect_faces(image)
    for (x, y, w, h), label in zip(faces, predictions):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_name = label_names.get(label, "Unknown")
        cv2.putText(image, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting image
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
