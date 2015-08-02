__author__ = 'Nimrod Shneor'
import numpy as np
import os
import cv2 as cv
from featureExtractor import featureExtractor

# TODO: add method: add data to training set

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
        num_columns = np.shape(trainingData)[1]
        num_rows = np.shape(trainingData)[0]
        print num_columns
        for i in range(num_columns):
            print range(num_columns)
            print i
            print trainingData[:,i]
            ### computing min & max entrys in each feature (column) in the feature matrix.
            max_col = np.max(trainingData[:,i])
            min_col = np.min(trainingData[:,i])
            for j in range(num_rows):
                trainingData[j,i] = (trainingData[j,i] - min_col)/ (max_col - min_col)


        ### DEBUG
        print np.shape(trainingData)
        print trainingData
        print np.shape(classes)
        print classes

        ### SAVING THE DATASETS
        np.savez(os.path.join(base_path, dataset_name), trainingData, labels, classes)
        return trainingData, classes, labels


