from classifier import classifier
from componentExtractor import componentExtractor
import numpy as np
import cv2 as cv
import Tkinter

__author__ = 'user'

img = cv.imread("..//Samples//1//2.jpg")
cv.namedWindow("Sample",cv.WINDOW_NORMAL)
cv.imshow("Sample",img)


cl = classifier(img)
cl.classifieSample()

cv.waitKey()
