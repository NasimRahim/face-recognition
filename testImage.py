import cv2
import os

# Load the haarcascade classifier for face detection

cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the LBPH face recognizer from the yml file
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelALL\lbph_modelFYPHaar.yml")

# Set variable to call the path from the folder untuk dataset
dataset_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\test"

# Define a dictionary that maps label numbers to names
label_name_dict = {
    1: "nasim",
    2: "khai",
    3: "hariz",
    16: "mirun", 
    20: "arif", 
}


# Create a function to detect and recognize faces from images
def detect_and_recognize_faces(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return None
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            # Resize the face region to 100x100 pixels
            face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            # Recognize the face using the LBPH recognizer
            label, confidence = recognizer.predict(face)
            #
            if label in label_name_dict:
            # Get the name corresponding to the predicted label from the label_name_dict
                name = label_name_dict[label]
            else:
                name = "UNKOWN!!!"
            # Draw a rectangle around the face region
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add the label and confidence score as text on the image
            text = f"{name}: {confidence:.2f}"
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image
    except Exception as e:
        print(f"Error loading or processing {image_path}: {e}")
        return None

# Create the testing folder if it does not exist
if not os.path.exists("testing"):
    os.mkdir("testing")

# Loop through all images in the dataset folder and detect and recognize faces
for filename in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, filename)
    result = detect_and_recognize_faces(image_path)
    if result is not None:
        # Save the output image in the testing folder with the same filename as the input image
        output_path = os.path.join("testing", filename)
        cv2.imwrite(output_path, result)