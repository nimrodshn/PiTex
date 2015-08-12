__author__ = 'Nimrod Shneor'

import cv2
import numpy as np
import random

# TODO:
# 1. explore more elaborate ideas of segmentation.

class componentExtractor:

    def __init__(self, inputImage):
        '''
        :param inputImage:
        '''
        self._image = inputImage



    def extractComponents(self):
        '''
        Main image processing and segmentation class, uses a simple image processing technique followed by otsu threshold to gather connected components.
        returns: a list of connected-components, these are the obto be fed to the positive-negative decomposition classifier.
        '''

        components = []

        # Check if image is grayscale.
        try:
            imgray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        except:
            imgray = self._image

        maxIntensity = 255.0 # depends on dtype of image data

        # Parameters for manipulating image data
        phi = 1
        theta = 1

        imgray = (maxIntensity/phi)*(imgray/(maxIntensity/theta))**0.5

        imgray = np.array(imgray,dtype=np.uint8)

        ret, thresh = cv2.threshold(imgray,0,255,cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)
        for i,contour in enumerate(contours):
            if (cv2.contourArea(contour)>150):

                rect = cv2.boundingRect(contours[i])
                component = cv2.cv.GetSubRect(cv2.cv.fromarray(self._image),rect)

                c = np.asanyarray(component)

                components.append((c,rect))
                cv2.drawContours(im, contours, i, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        
        ############### Debug ##################

        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", im)

        return components

