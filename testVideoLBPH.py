import cv2

# Set the path to the LBPH model
model_path = "lbph_modelFYP1.yml"

# Load the LBPH model 
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

# Set the path to the cascade classifier file
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# Set the path to the video file
video_path = 'C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetVideo\\dhea.mp4'

# Create a video capture object
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Failed to open video.")
    exit()

# Define a dictionary that maps label numbers to names
label_name_dict = {
    1: "nasim",
    2: "khai",
    3: "hariz",
    16: "mirun",
    20: "tuan arif"

    # Add more labels and names as needed
}

# Loop through frames from the video capture object
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI) from the grayscale frame
        roi_gray = gray[y:y + h, x:x + w]

        # Predict the label and confidence score for the ROI
        label, confidence = model.predict(roi_gray)

        if label in label_name_dict:
            # Get the name corresponding to the predicted label from the label_name_dict
            name = label_name_dict[label]
        else:
            name = "UNKNOWN"

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add the predicted name and confidence score as text
        text = f"{name}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
