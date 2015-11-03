
__author__ = 'Nimrod Shneor'

import cv2 as cv
from Tkinter import *
from GUI import ForamGUI
from Classifier import Classifier
import csv
from DatasetOrginizer import DatasetOrginizer
from sklearn.datasets import load_digits
from FeatureExtractor import FeatureExtractor
from ComponentExtractor import ComponentExtractor
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import lasagne
import theano.tensor as T
import numpy as np
import random
import json
import os
from pprint import pprint


def main():
    root = Tk()
    ForamGUI(root)
    root.attributes('-zoomed', True)
    root.mainloop()

    cv.waitKey()

########### TESTS ###########

def datasetOrginizerTrainKmeans():
    ds = DatasetOrginizer()
    
    data_path = "../Samples/Palmahim1"
    holdout_path = "../data/holdout"
    training_path = "../data/training"
    test_path = "../data/test"
    #ds.splitData(data_path,training_path,test_path)

    with open('../data/trainingAnnotations.json') as data_file:    
        data = json.load(data_file)
    
    training_list = []
    labels_list = []
    for item in data:
        training_list.append(item['filename'])
        labels_list.append(len(item['annotations']))

    print "number of labels/forams:"
    print np.sum(labels_list)

    #ds.createRegressionTrainingFromDataset(dataset_name="kmeansPalmahim1",path=holdout_path)

    #ds.KmeansTrainingDataset(KmeansData="binData/kmeansPalmahim1.npz",dataset_name="trainingPalmahim1",labels_list=labels_list,training_path_list=training_list)

def resizeImages():
    path_list = ["../data/training_classification/negative", "../data/training_classification/positive", "../data/holdout_classification"] 
    for path in path_list:
        for item in os.listdir(path):
            p = path + "/" + item
            img = cv.imread(p)
            res = cv.resize(img,(200,200), interpolation = cv.INTER_CUBIC)
            cv.imwrite(p,res) 

def splitDataForClassificationTest():
    ds = DatasetOrginizer()
    with open('../data/trainingAnnotations.json') as data_file:    
        data = json.load(data_file)
    
    training_list = []
    annotations = []
    for item in data:
        training_list.append(item['filename'])
        annotations.append(item['annotations'])
        
    path_list = ["../data/training_classification/positive", "../data/training_classification/negative"]
    # After splitting the data to positive and negative we then extract the 'positive' and 'negative' components using th following
    ds.extractComponentsFromData(training_list,annotations)
    # We then as pick randomly 10 'positive' and 10 'negative' samples for holdout. 
    negative_path = "../data/training_classification/negative"
    positive_path = '../data/training_classification/positive'
    ds.splitDataForHoldout(negative_path,positive_path)

def datasetOrginizerTrainClassification():
    ds = DatasetOrginizer()
    class_list = ["positive","negative"]
    kmeans_path = ['../data/holdout_classification']
    path_list = ["../data/training_classification/positive", "../data/training_classification/negative"]
    # First create the k-means classifier for the classification using the "holdout set" (kmeans path)
    ds.createClassificationTrainingFromDataset(dataset_name="kmeansClassificationPalmahim100",labels_list=class_list, path_list=kmeans_path)
    # Second create the training matrix using the kmeans classifier created above used on the data given by the path list.
    ds.createKmeansTrainingDataset(KmeansData="binData/kmeansClassificationPalmahim100.npz",kmeansName='KmeansBlobsPalmahim100.pkl',dataset_name="classificationTrainingPalmahim100",labels_list=class_list, path_list=path_list, num_of_clusters=100)

def segmentationTest():
    im = cv.imread("..//Samples//slides//PL29II Nov 4-5 0134.tif")    
    cv.namedWindow("..//Samples//slides//PL29II Nov 4-5 0134.tif",cv.WINDOW_NORMAL)
    cv.imshow("..//Samples//slides//PL29II Nov 4-5 0134.tif",im)
    ce = ComponentExtractor(im)
    components = ce.extractComponents()
    for i, component in enumerate(components):
        cv.namedWindow(str(i),cv.WINDOW_NORMAL)
        cv.imshow(str(i),component[0])
    cv.waitKey()

def validateClassifier():
    cl = Classifier(Dataset="binData/classificationTrainingPalmahim100.npz",regression=False)
    path_list = ["../data/training_classification/positive", "../data/training_classification/negative"]
    kmeans = 'KmeansBlobsPalmahim100.pkl'
    #cl.classificationValidation(path_list, kmeans,kernel='linear',gamma=None,C=1)            
    Cs = [0.001,0.002,0.003,0.004]
    gammas = [0.1]
    kernels = ["rbf","linear"]
    for kernel in kernels:
        if kernel == 'linear':
            gamma = None
            for C in Cs:
                cl.classificationValidation(path_list, kmeans,kernel=kernel,gamma=gamma,C=C)        
        else:
            for gamma in gammas:
                for C in Cs:
                    cl.classificationValidation(path_list, kmeans,kernel=kernel,gamma=gamma,C=C)
    
    cv.waitKey()

def crossValidateTest():
    cl = Classifier(Dataset="binData/classificationTrainingPalmahim1.npz",regression=False)
    #cl.plotHistogram()
    #cl.regressionCrossValidation(svr=True)
    cl.classificationCrossValidation()

if __name__ == '__main__':
    #main()

    #CNNTest()
    #FeatureExtractorTest()
    #resizeImagesTest()
    #splitDataForClassificationTest()
    #datasetOrginizerTrainClassification()
    #crossValidateTest()
    #segmentationTest()
    #datasetOrginizerTrainKmeans()
    validateClassifier()
    