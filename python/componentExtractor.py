import cv2
import random
import numpy as np

class componentExtractor:


    def __init__(self, inputImage):
        self._image = inputImage


    @property
    def extractComponents(self):

        """
        The main segmentation function
        returns: a list of components to be analyzed as "suspected Forams".
        """

        components = []
        imgray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i,contour in enumerate(contours):
            if (cv2.contourArea(contour)>200):
                rect = cv2.boundingRect(contours[i])
                component = cv2.cv.GetSubRect(cv2.cv.fromarray(self._image),rect)
                components.append(component)
                #cv2.drawContours(im, contours, i, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        ############### Debug ##################
        '''
        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", drawing)
        '''
        return components
