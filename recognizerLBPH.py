import os
import cv2
import csv
from datetime import datetime

# Load the LBPH model
model_path = "lbph_modelFYP23.yml"
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

# Load the cascade classifier
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Set up the video capture object using the webcam (index 0)
cap = cv2.VideoCapture(0)

# Define a dictionary that maps label numbers to names
label_name_dict = {
    1: "nasim Hensem",
    2: "khai",
    3: "hariz",
    16: "mirun",
    20: "Tuan Arif"
}

# Set to keep track of recorded names
recorded_names = set()

# Function to write the attendance data to the CSV file
def write_attendance_to_csv(label_name, timestamp):
    with open('attendance.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([label_name, timestamp])

# Loop to capture frames from the webcam and perform face recognition
while True:
    try:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray)

        # Predict the label and confidence score for each detected face
        for (x, y, w, h) in faces:
            # Extract the face region of interest (ROI) from the grayscale frame
            roi_gray = gray[y:y + h, x:x + w]

            # Predict the label and confidence score for the ROI
            label, confidence = model.predict(roi_gray)

            if label in label_name_dict:
                # Get the name corresponding to the predicted label from the label_name_dict
                name = label_name_dict[label]
            else:
                name = "UNKNOWN!!!"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the predicted name and confidence score as text
            text = f"{name}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add the detected name to the attendance list if it's not already there
            if name != "UNKNOWN!!!" and name not in recorded_names:
                # Get the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Write the label and timestamp to the CSV file
                write_attendance_to_csv(name, timestamp)
                # Add the name to the recorded_names set to avoid duplicates
                recorded_names.add(name)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit the loop and release the video capture object when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
