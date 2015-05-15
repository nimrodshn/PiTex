from classifier import classifier
import numpy as np
import cv2 as cv

__author__ = 'user'

img = cv.imread("..//Samples//1//3.jpg")
cv.namedWindow("Sample",cv.WINDOW_NORMAL)
cv.imshow("Sample",img)

cl = classifier(img)
cl.classifieSample()

cv.waitKey()
