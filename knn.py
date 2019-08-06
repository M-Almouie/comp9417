from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from imutils import paths
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import argparse
import imutils
import cv2
import os

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

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
#note, 10/11 neighbours most accurate
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

rawImages = []
features = []
labels = []
negativeCount = 0
positiveCount = 0

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	if "negative" in imagePath:
		negativeCount += 1
		label="negative"
	else:
		positiveCount += 1
		label="positive"

	image = cv2.imread(imagePath)
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
neighbours = [10,20,50,100]
paccuracies = []
pcosts = []
haccuracies = []
hcosts = []

for i in range(len(neighbours)):
	#if i == 0:
	#	nm = 1
	#else:
	model = KNeighborsClassifier(n_neighbors=neighbours[i], n_jobs=args["jobs"])
	model.fit(trainRI, trainRL)
	acc = model.score(testRI, testRL)
	p = model.predict_proba(testRI)
	L = log_loss(testRL, p)
	paccuracies.append(acc*100)
	pcosts.append(L*100)
	#print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
	#print("Raw pixel cost is: {:.2f}%".format(L))

	model = KNeighborsClassifier(n_neighbors=neighbours[i], n_jobs=args["jobs"])
	model.fit(trainFeat, trainLabels)
	acc = model.score(testFeat, testLabels)
	p = model.predict_proba(testFeat)
	L = log_loss(testLabels, p)
	haccuracies.append(acc*100)
	hcosts.append(L*100)
	#print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
	#print("Histogram cost is: {:.2f}%".format(L))

plt.subplot(2,1,1)
plt.plot(neighbours, paccuracies, 'b', marker="X")
plt.title('KNN Accuracy and Cost (Pixel Intensity Focus)')
plt.ylabel('Accuracy (%)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(neighbours, pcosts, 'r', marker="X")
plt.ylabel('Cost (%)')
plt.xlabel('No. of Neighbours')
plt.grid()
plt.show()

plt.subplot(2,1,1)
plt.plot(neighbours, haccuracies, 'b', marker="X")
plt.title('KNN Accuracy and Cost (Histogram Focus)')
plt.ylabel('Accuracy (%)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(neighbours, hcosts, 'r', marker="X")
plt.ylabel('Cost (%)')
plt.xlabel('No. of Neighbours')
plt.grid()
plt.show()