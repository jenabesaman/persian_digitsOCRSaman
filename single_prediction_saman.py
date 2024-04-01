import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

class FeatureExtraction:
    def choose(self, function, resizeImage, Labels):
        if function == "HOG":
            return self.hog(resizeImage)
        elif function == "SVD":
            return self.svd(resizeImage, Labels)
        elif function == "PCA":
            return self.pca(resizeImage, Labels)
        else:
            return self.hog(resizeImage)

    def hog(self, resizeImage):
        fd, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)
        im_bw = list(hog_image.flatten())
        return im_bw

def load_images_from_folder(folder_path, size=50):
    data = []
    labels = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            image_path = os.path.join(path, name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resize_img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            _, thresholded_img = cv2.threshold(resize_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            features = FeatureExtraction().choose("HOG", thresholded_img, labels)
            data.append(features)
            labels.append(path[14:])  # Assuming the label is extracted from the folder structure
    return data, labels

def train_knn(x_train, y_train, k=5):
    classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    classifier.fit(x_train, y_train)
    return classifier

def main():
    # Load training data from the default folder (e.g., "persian_digit")
    x_train, y_train = load_images_from_folder("persian_digit")

    # Set test folder path (e.g., "test_image")
    test_folder_path = "test_image"
    x_test, y_test = load_images_from_folder(test_folder_path)

    # Train KNN classifier
    k_value = 5  # Choose an appropriate value for k
    knn_classifier = train_knn(x_train, y_train, k=k_value)

    # Predict using the trained classifier
    y_pred = knn_classifier.predict(x_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
