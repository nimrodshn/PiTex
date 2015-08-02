__author__ = 'Nimrod Shneor'
import numpy as np
import os
import cv2 as cv
from featureExtractor import featureExtractor

# TODO: 
# 1. Add method to add data to training set.

class datasetOrginizer:

    def __init__(self):
       self._dataSets=[]
       defaultSet = np.load('binData/Default.npz')
       self._dataSets.append(defaultSet)

    def createTrainingFromDataset(self, dataset_name, labels_list, path_list):
        '''
        Creates a new training set to work on from given dataset in location: creating Feature Vector, Normalize etc..
        :param name: the name of the data set
        :param location: location where the data was collected
        :return:
        '''

        base_path = "binData/"

        labels = []
        trainingData = []
        classes = []
        cl = 0

        ### Building the feature matrix.
        for i, path in enumerate(path_list):

            labels.append(labels_list[i])
            print labels_list[i]


            for item in os.listdir(path):

                p = path + "/" + item
                print p # DEBUG
                im = cv.imread(p)

                fe = featureExtractor(im)

                feature_vector = fe.computeFeatureVector()

                trainingData.append(feature_vector)
                classes.append(cl)

            cl = cl + 1

        ### Normalization of features to unit range [0,1].
        B = np.asmatrix(trainingData)
        num_columns = np.shape(B)[1]
        num_rows = np.shape(B)[0]
        for j in range(num_columns):
            print B[:,j]
            ## computing min & max entrys in each feature category (column) in the feature matrix.
            max_col = np.max(B[:,j])
            min_col = np.min(B[:,j])    
            for i in range(num_rows):
                B[i,j] = (B[i,j] - min_col) / (max_col - min_col)

        ### DEBUG
        print np.shape(trainingData)
        print np.shape(classes)

        ### SAVING THE DATASETS
        np.savez(os.path.join(base_path, dataset_name), B, labels, classes)
        return B, classes, labels


