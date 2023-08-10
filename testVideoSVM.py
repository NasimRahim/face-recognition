import cv2
import numpy as np
import pickle
import joblib

# Load the SVM model from the saved file
with open('svm_modelFYPHaar.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load the haarcascade classifier for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Start the video capture using OpenCV
cap = cv2.VideoCapture('C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetVideo\\nasim.mp4')  # Specify the path to your MP4 file

# Check if the video capture is successful
if not cap.isOpened():
    print("Failed to open video capture.")
    exit()

label_names = {
    1: "Nasim",
    2: "Khai",
    3: "Hariz",
    16: "mirun",
    11: "akmal",
    20: "tuan arif"
    # add more names and labels as needed
}

# Start a loop to continuously read frames from the video and detect faces
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        print("Failed to read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the haarcascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Iterate over each detected face and predict its label using the SVM model
    for (x, y, w, h) in faces:
        # Preprocess the face image
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)).flatten()

        # Predict the label of the face using the SVM model
        label = svm_model.predict([face])

        # Get the name corresponding to the label number from the dictionary
        name = label_names.get(label[0], "Unknown")

        # Draw a rectangle around the detected face with the predicted label as text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy any remaining windows
cap.release()
cv2.destroyAllWindows()
