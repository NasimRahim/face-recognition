#evaluated using cv
import cv2
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def train():
    # Set variable to call the path from the folder untuk dataset
    dataset_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetBase"

    # Load the haarcascade classifier for face detection
    cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Define LBPH parameters
    radius = 1
    neighbors = 8
    grid_x = 8
    grid_y = 8
    threshold = 100

    # Create an LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y, threshold=threshold)

    # Define a function to read the images from the dataset folder
    def get_images_and_labels(dataset_path):
        image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        images = []
        labels = []
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                if len(faces) == 0:
                    os.remove(image_path)
                for (x, y, w, h) in faces:
                    images.append(cv2.resize(gray[y:y+h, x:x+w], (100, 100)))
                    labels.append(int(os.path.split(image_path)[-1].split("_")[0]))
            except Exception as e:
                print(f"Error loading or processing {image_path}: {e}")
        return images, labels

    # Load images and label from dataset folder
    images, labels = get_images_and_labels(dataset_path)
    print("Number of images:", len(images))
    print("Number of labels:", len(set(labels)))

    if len(images) == 0:
        print("No face images found in the dataset folder!")
        exit()

    # Perform manual cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(images):
        X_train, X_test = np.array(images)[train_index], np.array(images)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # Train the recognizer using the training data
        recognizer.train(X_train, np.array(y_train))

        # Make predictions on the testing data
        y_pred = []
        for image in X_test:
            label, confidence = recognizer.predict(image)
            y_pred.append(label)

        # Calculate accuracy score
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    # Print cross-validation scores
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))
    print("Standard Deviation:", np.std(scores))

    # Train the recognizer on the entire dataset
    recognizer.train(images, np.array(labels))

    # Save the recognizer to a file
    recognizer.save("lbph_modelFYP23.yml")