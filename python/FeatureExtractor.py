__author__ = 'Nimrod Shneor'

import cv2 as cv
import numpy as np
import mahotas as mh
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt


# TODO: 

class FeatureExtractor:
    def __init__(self,input):
        self.im = input

    def computeFeatureVector(self):
        '''
        Computes the feature vector using different featurs.
        :return: a list representing the feature vector to be called by DatasetOrginizer to build your dataset.
        '''

        dense = self.computeDenseSIFTfeatures()

        feature_vector = dense

        return feature_vector

    ########### FEATURES #################

    def computeDenseSIFTfeatures(self):
        sift = cv.SIFT()
        dense = cv.FeatureDetector_create("Dense")
        gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        kp=dense.detect(gray)
        kp,des=sift.compute(gray,kp)
        feature_vector = np.asarray(des)
        #print feature_vector
        return feature_vector