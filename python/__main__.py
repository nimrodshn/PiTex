from componentExtractor import componentExtractor
from classifier import classifier
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

__author__ = 'user'

img = cv.imread("..//Samples//1//1.jpg")
cv.namedWindow("Sample",cv.WINDOW_NORMAL)
cv.imshow("Sample",img)

## Segmentation
ce = componentExtractor(img)
components = ce.extractComponents # THIS IS A LIST

for i,component in enumerate(components):
    comp = np.asanyarray(component,dtype='uint8')

    ##### DEBUG ####

    cv.namedWindow("component"+str(i),cv.WINDOW_NORMAL)
    cv.imshow("component"+str(i), comp)
    

cv.waitKey()
