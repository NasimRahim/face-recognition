import cv2
import numpy as np
import joblib
import pickle

# Load the haarcascade classifier for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the pre-trained LBPH model
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.read("lbph_modelFYP1.yml")
# Load the trained SVM classifier (recognizer)
with open('modelALL\svm_modelFYPHaar.pkl', 'rb') as f:
    svm_model = pickle.load(f)

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
    faces = detect_faces(image)
    predictions = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (100, 100))
        label, _ = lbph_model.predict(face_region)
        predictions.append(label)
    return predictions

# Open the video file
video_path = 'C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetVideo\\arif1.mp4'
video_capture = cv2.VideoCapture(video_path)

while True:
    # Read the video frame
    ret, frame = video_capture.read()
    
    # Check if the video has ended
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face recognition on the frame
    predictions = recognize_faces(frame)
    
    # Draw bounding boxes and labels on the frame
    faces = detect_faces(frame)
    for (x, y, w, h), label in zip(faces, predictions):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_name = label_names.get(label, "Unknown")
        cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()