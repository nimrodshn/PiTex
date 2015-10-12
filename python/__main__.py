
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
from pprint import pprint


def main():
    root = Tk()
    ForamGUI(root)
    root.attributes('-zoomed', True)
    root.mainloop()

    cv.waitKey()

########### TESTS ###########

def DatasetOrginizerTrainKmeans():
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

    ds.createRegressionTrainingFromDataset(dataset_name="kmeansPalmahim1",path=holdout_path)

    ds.KmeansTrainingDataset(KmeansData="binData/kmeansPalmahim1.npz",dataset_name="trainingPalmahim1",labels_list=labels_list,training_path_list=training_list)

def DatasetOrginizerTrainClassification():
    ds = DatasetOrginizer()
    with open('../data/trainingAnnotations.json') as data_file:    
        data = json.load(data_file)
    
    training_list = []
    annotations = []
    for item in data:
        training_list.append(item['filename'])
        annotations.append(item['annotations'])
        
    path_list = ["../data/training1/negative","../data/training1/positive"]
    class_list = ["negative","positive"]

    training_path = "../data/training"
    test_path = "../data/test"
    ds.splitDataForClassification(training_list, annotations)
    # ds.createTrainingFromDataset("classificationPalmahim1",class_list, path_list)

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

def ClassifierTest():
    img = cv.imread("..//Samples//slides//A0004.tif")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",img)

    cl = Classifier(Dataset="binData/test.npz")
    #cl.posNegDecompose()
    cl.plotPCA()
    cv.waitKey()

def validateClassifier():
    cl = Classifier(Dataset="binData/trainingPalmahim1.npz",regression=True)
    #cl.validation()

    with open('../data/trainingAnnotations.json') as data_file:    
        data = json.load(data_file)
    
    true_val = []
    training_list = []
    for item in data:
        training_list.append(item['filename'])
        true_val.append(len(item['annotations']))    

    cl.regressionValidation(training_list, true_val)
    cv.waitKey()

def crossValidateTest():
    cl = Classifier(Dataset="binData/trainingPalmahim1.npz",regression=True)
    #cl.plotPCA()
    #cl.crossValidateGridSearch()
    cl.regressionCrossValidation(svr=True)

if __name__ == '__main__':
    #main()

    #CNNTest()
    #FeatureExtractorTest()
    #DensityTest()
    #datasetOrgenizerRegressionTest()
    print "Something"
    #DatasetOrginizerTrainClassification()
    #ClassifierTest()
    #crossValidateTest()
    #segmentationTest()
    #DatasetOrginizerTrainKmeans()
    #validateClassifier()
    