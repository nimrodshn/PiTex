__author__ = 'Nimrod Shneor'
import numpy as np
import os
import cv2 as cv
import random
from componentExtractor import componentExtractor
from featureExtractor import featureExtractor
from sklearn import cluster

# TODO: 
# 1. Add method to add data to training set.

class datasetOrginizer:

    def __init__(self):
       self._dataSets=[]
       defaultSet = np.load('binData/Default.npz')
       self._dataSets.append(defaultSet)

    def splitData(self,data_path,training_path,test_path):
        '''
        split data in path into training-set and test-set.
        param data_path: the path where that data collected is found.
        paran training_path: the path where that training set is saved.
        param test_set: the path where that test set is saved.
        '''
        numofdata = len(os.listdir(data_path))
        print numofdata
        number_of_split = 20
        #pick #number_of_split test images at random
        test_num = random.sample(range(1, numofdata), number_of_split) 
        A = np.zeros(numofdata)        
        for k in range(20):
            A[test_num[k]] = 1 
        # create training set and test set
        training_set = []
        test_set = []
        holdout_set = []
        for i, item in enumerate(os.listdir(data_path)):
            p = data_path + "/" + item
            print p # DEBUG
            if A[i] == 1:
                holdout_set.append(data_path + "/" + item)  
            else:
                training_set.append(data_path + "/" + item)

        for i, path in enumerate(holdout_set):
            im = cv.imread(path)
            cv.imwrite("../data/holdout/" + str(i)+ ".jpg", im)
            #ce = componentExtractor(im)
            #components = ce.extractComponents() 
            #for k, component in enumerate(components):
            #   cv.imwrite(training_path + "/" + str(i) + str(k)+ ".jpg", component[0])

        for i, path in enumerate(training_set):
            im = cv.imread(path)
            cv.imwrite("../data/training/" + str(i)+ ".jpg", im)
            #ce = componentExtractor(im)
            #components = ce.extractComponents() 
            #for k, component in enumerate(components):
            #   cv.imwrite(training_path + "/" + str(i) + str(k)+ ".jpg", component[0])

        for i, path in enumerate(test_set):
            im = cv.imread(path)
            cv.imwrite("../data/test/" + str(i)+ ".jpg", im)
            #ce = componentExtractor(im)
            #components = ce.extractComponents()      
            #for k, component in enumerate(components):
            #   cv.imwrite(test_path + "/" + str(i) + str(k) + ".jpg", component[0])

    def createRegressionTrainingFromDataset(self, dataset_name, labels_list=None, path):
        '''
        Creates a new training set for REGRESSION PROBLEM to work on from given path list and labels.
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

        ### Building the feature matrix.
        for item in os.listdir(path):     
            p = path + "/" + item
            print p # DEBUG
            im = cv.imread(p)
            fe = featureExtractor(im)
            feature_vector = fe.computeFeatureVector()
            #trainingData.append(feature_vector)
            trainingData.vstack(feature_vector) # for k means

        ### DEBUG 
        print np.shape(trainingData)

        ### SAVING THE DATASETS TO NPZ FORMAT
        np.savez(os.path.join(base_path, dataset_name), trainingData, labels_list, classes,  min_max_features)

    def KmeansTrainingDataset(self,KmeansData, path):
        npzfile = np.load(KmeansData)
        KmeansData = npzfile['arr_0']
        Kmeanslabels = npzfile['arr_1']
        Kmeansclasses = npzfile['arr_2']

        num_of_clusters = 10
        k_means = cluster.KMeans(n_clusters=num_of_clusters)
        k_means.fit(KmeansData)

        base_path = "binData/"

        labels = []
        trainingData = []
        classes = []
        min_max_features = []

        ### Building the feature matrix.
        for item in os.listdir(path):

            p = path + "/" + item
            print p # DEBUG
            im = cv.imread(p)
            fe = featureExtractor(im)
            feature_vector = np.zeros(num_of_clusters)
            raw_vector = fe.computeFeatureVector()
            Km_vector = k_means.predict(raw_vector) 
            for i in range(len(Km_vector)):
                feature_vector[Km_vector[i]] = feature_vector[Km_vector[i]] + 1 
                
            trainingData.append(feature_vector)

        ### DEBUG 
        print np.shape(trainingData)

        ### SAVING THE DATASETS TO NPZ FORMAT
        np.savez(os.path.join(base_path, dataset_name), trainingData, labels_list, classes,  min_max_features)

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


