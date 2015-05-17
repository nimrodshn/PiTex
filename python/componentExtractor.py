import cv2
import numpy as np
import random
class componentExtractor:

    def __init__(self, inputImage):
        '''
        :param inputImage:
        :return: Constructor for the Main image processing and segmentation class.
        '''
        self._image = inputImage


    @property
    def extractComponents(self):

        """
        Main image processing and segmentation function
        returns: a list of components to be analyzed as "suspected Forams".
        """

        components = []
        imgray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

        # What method should we use?
        ret, thresh = cv2.threshold(imgray ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret, thresh = cv2.threshold(imgray,100,255,0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.cvtColor(imgray,cv2.COLOR_GRAY2BGR)
        for i,contour in enumerate(contours):
            if (cv2.contourArea(contour)>200):

                rect = cv2.boundingRect(contours[i])
                component = cv2.cv.GetSubRect(cv2.cv.fromarray(self._image),rect)

                c = np.asanyarray(component,dtype='uint8')

                maxIntensity = 255.0

                # Parameters for manipulating image data
                phi = 1
                theta = 1

                # Decrease intensity such that
                # dark pixels become much darker,
                # bright pixels become slightly dark
                enhanced_contrast = (maxIntensity/phi)*(c/(maxIntensity/theta))**0.5
                contrast = np.array(enhanced_contrast,dtype=np.uint8)

                components.append(contrast)
                cv2.drawContours(im, contours, i, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

        ############### Debug ##################

        cv2.namedWindow("contours", cv2.WINDOW_NORMAL)
        cv2.imshow("contours", im)

        return components

