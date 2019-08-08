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

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	return hist.flatten()

imageSize = 64
categories = ['positive', 'negative']

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

X = []
y = []
hists = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	if "negative" in imagePath:
		label="negative"
	else:
		label="positive"

	image = cv2.imread(imagePath)
	v = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	X.append(v)
	y.append(label)
	hists.append(hist)

X = np.array(X)
hists = np.array(hists)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
hists_train,hists_test, yh_train, yh_test = train_test_split(X, y, test_size=0.25, random_state=42)

nes = [10, 20, 50, 100, 200, 500, 1000]
paccuracies = []
pcosts = []
haccuracies = []
hcosts = []

for i in range(len(nes)):
	clf = RandomForestClassifier(n_estimators=nes[i])
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	acc = accuracy_score(y_test,preds)
	p = clf.predict_proba(X_test)
	L = log_loss(y_test, p)
	paccuracies.append(acc*100)
	pcosts.append(L*100)
	
	clf = RandomForestClassifier(n_estimators=nes[i])
	clf.fit(hists_train, yh_train)
	preds = clf.predict(hists_test)
	acc = accuracy_score(yh_test,preds)
	p = clf.predict_proba(hists_test)
	L = log_loss(yh_test, p)
	haccuracies.append(acc*100)
	hcosts.append(L*100)

plt.subplot(2,1,1)
plt.plot(nes, paccuracies, 'b', marker="X")
plt.title('Random Forest Accuracy and Cost (Pixel Intensity Focus)')
plt.ylabel('Accuracy (%)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(nes, pcosts, 'r', marker="X")
plt.ylabel('Cost (%)')
plt.xlabel('No. of Trees')
plt.grid()
plt.show()

plt.subplot(2,1,1)
plt.plot(nes, haccuracies, 'b', marker="X")
plt.title('Random Forest Accuracy and Cost (Histogram Focus)')
plt.ylabel('Accuracy (%)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(nes, hcosts, 'r', marker="X")
plt.ylabel('Cost (%)')
plt.xlabel('No. of Neighbours')
plt.grid()
plt.show()