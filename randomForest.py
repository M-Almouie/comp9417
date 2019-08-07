import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2, os, glob
import argparse
import imutils
import cv2

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imutils import paths
from sklearn.metrics import log_loss

# Snippets used from the following tutorial:
# https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

imageSize = 64
categories = ['positive', 'negative']

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

X = []
y = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	if "negative" in imagePath:
		label="negative"
	else:
		label="positive"

	image = cv2.imread(imagePath)
	v = image_to_feature_vector(image)

	X.append(v)
	y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

nes = [10, 20, 50, 100, 200, 500, 1000]
accuracies = []
costs = []
for i in range(len(nes)):
	clf = RandomForestClassifier(n_estimators=nes[i])
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	acc = accuracy_score(y_test,preds)
	p = clf.predict_proba(X_test)
	L = log_loss(y_test, p)
	accuracies.append(acc*100)
	costs.append(L*100)
	#print("Accuracy:{:.2f}%".format(acc*100))
	#print("Loss: {:.2f}%".format(L*100))

plt.subplot(2,1,1)
plt.plot(nes, accuracies, 'b', marker="X")
plt.title('Random Forest Accuracies and Costs (Pixel Intensity Focus)')
plt.ylabel('Accuracy (%)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(nes, costs, 'r', marker="X")
plt.ylabel('Cost (%)')
plt.xlabel('No. of Trees')
plt.grid()
plt.show()
