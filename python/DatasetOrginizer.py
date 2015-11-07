__author__ = 'Nimrod Shneor'
import numpy as np
import os
import cv2 as cv
import random
from ComponentExtractor import ComponentExtractor
from FeatureExtractor import FeatureExtractor
from sklearn import cluster
from sklearn.externals import joblib

# TODO: 
# 1. Add method to add data to training set.

class DatasetOrginizer:

    def __init__(self):
       self._dataSets=[]
       
    def splitData(self,data_path, training_path, test_path, number_of_forams):
        '''
        splits data in path into training-set and test-set(data - images of plates/trays NOT blobs). 
        param data_path: the path where that data collected is found.
        paran training_path: the path where that training set is saved.
        param test_set: the path where that test set is saved.
        '''
        #numofdata = len(os.listdir(data_path))
        numofdata = len(os.listdir(training_path))
        print numofdata
        number_of_split = 20
        #pick number_of_split test images at random
        test_num = random.sample(range(1, numofdata), number_of_split) 
        A = np.zeros(numofdata)        
        for k in range(78): 
            A[test_num[k]] = 1 
        # create training set and test set
        training_set = []
        test_set = []
        holdout_set = []
        for i, item in enumerate(os.listdir(training_path)):
            p = data_path + "/" + item
            print p # DEBUG
            if A[i] == 1:
                #test_set.append(data_path + "/" + item)  
                holdout_set.append(training_path + "/" + item)
            else:
                training_set.append(data_path + "/" + item)

        for i, path in enumerate(holdout_set):
            im = cv.imread(path)
            cv.imwrite("../data/holdout/" + str(i)+ ".jpg", im)
            os.remove("../data/training/" + str(i)+ ".jpg")
        
    def extractComponentsFromData(self, path_list, annotations):
        '''
        This method extracts the components from each image in the data.
        If the annotation is inside a bounding rectangle then the component is labeld 'positive' else it is labeld 'negative'.  
        '''

        for i, path in enumerate(path_list):
            im = cv.imread(path)
            ce = ComponentExtractor(im)
            components = ce.extractComponents() 
            print "bounding rectangels in image " + str(i)
            if len(annotations[i]) > 0:
                for k, component in enumerate(components):         
                    bounding_rect = component[1] # see ComponentExtractor
                    for annotation in annotations[i]:                
                        if ((bounding_rect[0] <= annotation['x'] <= bounding_rect[0] + bounding_rect[2]) and (bounding_rect[1] <= annotation['y'] <= bounding_rect[1] + bounding_rect[3])): 
                            cv.imwrite("../data/training_classification/positive/" + str(i) + str(k)+ ".jpg", component[0])
                        else:
                            cv.imwrite("../data/training_classification/negative/" + str(i) + str(k)+ ".jpg", component[0])

            else:
                for k, component in enumerate(components):         
                    cv.imwrite("../data/training_classification/negative/" + str(i) + str(k)+ ".jpg", component[0])

    def splitDataForHoldout(self, positive_path, negative_path):
        '''
        Splits the data for holdout.
        '''
        num_of_positive = len(os.listdir(positive_path))
        num_of_negative = len(os.listdir(negative_path))
        number_of_split = 10
        #pick number_of_split test images at random
        positive_num = random.sample(range(1, num_of_positive), number_of_split) 
        negative_num = random.sample(range(1, num_of_negative), number_of_split) 
        
        B = np.zeros(num_of_negative)
        A = np.zeros(num_of_positive)        
        for k in range(number_of_split):
            A[positive_num[k]] = 1 
        for j in range(number_of_split):
            B[negative_num[j]] = 1

        holdout_set = []
        for i, item in enumerate(os.listdir(positive_path)):
            p = positive_path + "/" + item
            print p # DEBUG
            if A[i] == 1:  
                holdout_set.append(positive_path + "/" + item)
        
        for k, item in enumerate(os.listdir(negative_path)):
            p = negative_path + "/" + item
            print p # DEBUG
            if B[k] == 1:  
                holdout_set.append(negative_path + "/" + item)
        
        print len(holdout_set)   
        for j, path in enumerate(holdout_set):
            im = cv.imread(path)
            cv.imwrite("../data/holdout_classification/" + str(j) + ".jpg", im)
            os.remove(path)

    def createKmeansTrainingDataset(self,kmeans_data, dataset_name, kmeans_name, path_list, labels_list, num_of_clusters):
        '''
        Create Training for Kmeans With regression.
        :param: KmeansData: the training matrix obtained using createClassificationTrainingFromDataset method on the HOLDOUT set.
        :param: kmeansName: the name of the Kmeans classifier to be saved and pickeled.
        :param: dataset_name: the name of the NEW dataset created using the Kmeans classifier on the training. i.e. clustering the feature vector.
        :param: path_list: list of paths where the training set is at.
        :param: labels_list: the list of labels for the samples in the training set.
        :param: num_of_clusters: the number of clusters for the Kmeans classifier.        
        '''
        npzfile = np.load(kmeans_data)
        KmeansData = npzfile['arr_0']
        Kmeanslabels = npzfile['arr_1']
        Kmeansclasses = npzfile['arr_2']

        k_means = cluster.KMeans(n_clusters=num_of_clusters)
        k_means.fit(kmeans_data)

        base_path = "binData/"

        labels = labels_list
        trainingData = []
        classes = []
        cl=0

        ### Building the feature matrix.
        for i, path in enumerate(path_list):
            
            print labels_list[i]

            for item in os.listdir(path):
                p = path + "/" + item
                print p # DEBUG
                im = cv.imread(p)
                fe = FeatureExtractor(im)
                feature_vector = np.zeros(num_of_clusters)
                raw_vector = fe.computeFeatureVector()
                Km_vector = k_means.predict(raw_vector) 
                for j in range(len(Km_vector)):
                    feature_vector[Km_vector[j]] = feature_vector[Km_vector[j]] + 1 
                trainingData.append(feature_vector)
                classes.append(cl)
            
            # Here we multiply the number of POSITIVE samples in the training set so that the 'unbalanced' problem of "Foram vs. Not-Foram"
            # 'becomes balanced'.
            if i == 0:
                print "working on positive samples"
                print "Original training size: (should be 68 by 10)"
                print np.shape(trainingData)
                print np.shape(classes)

                for k in range(9):
                    trainingData = np.vstack((trainingData, trainingData))
                    classes = np.hstack((classes,classes))
                
                print "After Multipling Positive Samples by 8"
                print np.shape(trainingData)
                print np.shape(classes)
                
                trainingData = trainingData.tolist()
                classes = classes.tolist()
            
            cl = cl + 1
            
        ### DEBUG 
        print "final shape: (should be 54,000~ by 10):"
        print np.shape(trainingData)

        ### SAVING THE DATASETS TO NPZ FORMAT
        joblib.dump(k_means, os.path.join(base_path, kmeans_name), compress=9)
        np.savez(os.path.join(base_path, dataset_name), trainingData, labels_list, classes)

    def createClassificationTrainingFromDataset(self, dataset_name, labels_list, path_list):
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
        cl = 0

        ### Building the feature matrix.
        for i, path in enumerate(path_list):

            labels.append(labels_list[i])
            print labels_list[i]

            for item in os.listdir(path):
                p = path + "/" + item
                print p # DEBUG
                im = cv.imread(p)
                fe = FeatureExtractor(im)
                feature_vector = fe.computeFeatureVector()
                if len(trainingData) == 0:
                    trainingData = feature_vector
                else:
                    np.vstack((trainingData, feature_vector))           
                classes.append(cl)
            
            print "vstack Kmeans Classifier: "
            print np.shape(trainingData)

            classes = np.array(classes)
            cl = cl + 1

        ### DEBUG 
        print np.shape(trainingData)
        print np.shape(classes)

        ### SAVING THE DATASETS TO NPZ FORMAT
        np.savez(os.path.join(base_path, dataset_name), trainingData, labels, classes)


