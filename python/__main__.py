from componentExtractor import componentExtractor
from classifier import classifier
from sklearn import svm
from sklearn.neighbors import  KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn import  decomposition
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

__author__ = 'user'

img = cv.imread("..//Samples//1//3.jpg")
cv.namedWindow("Sample",cv.WINDOW_NORMAL)
cv.imshow("Sample",img)



cl = classifier()

trainingData,labels, classes = cl.getTrainingData()

X = trainingData
y = classes
C=1.0
clf = svm.SVC(kernel='poly',C=C)
knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
knn.fit(X,y)
clf.fit(X,y)

## Segmentation
ce = componentExtractor(img)
components = ce.extractComponents # THIS IS A LIST


for i,component in enumerate(components):

    #cv.namedWindow(str(i), cv.WINDOW_NORMAL)
    #cv.imshow(str(i), component)


    orb = cv.ORB()
    kp = orb.detect(component,None)

    ## Normalize the Data
    if len(kp) > 15:


        kp = kp[:15]
        kp, des = orb.compute(component, kp)
        d=des.flatten()
        res = knn.predict(d)
        print "comp " + str(i) +":" + labels[res[0]]

        im2 = cv.drawKeypoints(component ,kp,color=(0,255,0), flags=0)
        cv.namedWindow(labels[res[0]] + " comp " + str(i), cv.WINDOW_NORMAL)
        cv.imshow(labels[res[0]] + " comp " + str(i), im2)

cv.waitKey()
