import cv2
import numpy as np
import random
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

        ret, thresh = cv2.threshold(imgray, 90, 255, 0) # Better Threshold value?
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)
        for i,contour in enumerate(contours):
            if (cv2.contourArea(contour)>200):
                rect = cv2.boundingRect(contours[i])
                component = cv2.cv.GetSubRect(cv2.cv.fromarray(self._image),rect)
                c = np.asanyarray(component,dtype='uint8')
                components.append(c)
                cv2.drawContours(im, contours, i, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        ############### Debug ##################

        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", im)

        return components

