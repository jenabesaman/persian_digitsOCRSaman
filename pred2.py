import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
from skimage.feature import hog
import numpy as np
def hog1(resizeImage):
    fd,hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True, channel_axis=None)
    im_bw = list(hog_image.flatten())
    return im_bw

def ExtractFeatureAndBuildDataset(path="persian_digit2",size=50):
    DATA = []
    Labels = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            imagePath = os.path.join(path, name)
            # Read image as gray scale
            img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
            # Resize image for get better feature
            resizeImg = cv2.resize(img,(size,size), interpolation=cv2.INTER_AREA)
            # Thredshold for 128 is 0 and 255 is
            _, IMG = cv2.threshold(resizeImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # Append extract featured DATA
            # DATA.append(FeatureExtraction().choose(extract_feature,IMG,Labels))
            DATA.append(hog1(IMG))
            # Label of each image
            Labels.append(path[14:])

    return DATA,Labels

def knn( x_train, x_test, y_train, y_test,path="test_image", is_parzen=False):
    x_pred,_=ExtractFeatureAndBuildDataset(path=path)
    error = []
    best_k = dict()

    # Calculating error for K values between 1 and 20
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
        best_k[i] = np.mean(pred_i != y_test)

    best_k = sorted(best_k.items(), key=lambda k: k[1])[0][0]
    if is_parzen:
        classifier = KNeighborsClassifier(n_neighbors=best_k, algorithm='ball_tree', n_jobs=-1)
    else:
        classifier = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    # print(y_pred)

    print("##########################################################")
    print("###########  Accuracy: ", accuracy_score(y_test, y_pred), "   ############")
    print("##########################################################")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20), error, color='red', linestyle='dashdot', marker='o',
             markerfacecolor='green', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    y_m_pred=classifier.predict(x_pred)
    print(y_m_pred)

ExtractFeatureAndBuildDataset()
DATA, Labels = ExtractFeatureAndBuildDataset()
# Set train and test data
X_train, X_test, y_train, y_test = train_test_split(DATA, Labels, test_size=0.10)
knn(X_train, X_test, y_train, y_test)

