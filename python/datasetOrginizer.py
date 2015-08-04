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
        Creates a new training set to work on from given path list and labels.
        Notice path_list and path_labels are intended to be lists of the same length. see tests in __main__ for examples.
        :param dataset_name: the name of the data set
        :param path_list: a list of pathes frome which the images are collected.
        :param labels_list: a list of labels to use for the images collected from corresponding path. (i.e. first label correspond to first path in the path list.)
        '''

        base_path = "binData/"

        labels = []
        trainingData = []
        classes = []
        min_max_features = []
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
            max_feature = np.max(B[:,j])
            min_feature = np.min(B[:,j])
            min_max_features.append((max_feature,min_feature)) # Keep max & min entrys of feature map for normalization purposes.    
            
            for i in range(num_rows):
                B[i,j] = (B[i,j] - min_feature) / (max_feature - min_feature) 

        ### DEBUG 
        print np.shape(trainingData)
        print np.shape(classes)

        ### SAVING THE DATASETS TO NPZ FORMAT
        np.savez(os.path.join(base_path, dataset_name), B, labels, classes,  min_max_features)


