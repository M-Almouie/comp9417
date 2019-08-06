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
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
p = clf.predict_proba(X_test)
L = log_loss(y_test, p)
print("Accuracy:", accuracy_score(y_test,preds)*100)
print("Loss:", L*100)