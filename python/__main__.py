
__author__ = 'Nimrod Shneor'

import cv2 as cv
from Tkinter import *
from GUI import ForamGUI
from classifier import classifier
import csv
from datasetOrginizer import datasetOrginizer
from sklearn.datasets import load_digits
from featureExtractor import featureExtractor
from componentExtractor import componentExtractor
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

def datasetOrginizerTrainKmeans():
    ds = datasetOrginizer()
    
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

def datasetOrginizerClassificationTest():
    ds = datasetOrginizer()
    # path_list = ["../data/training1/negative","../data/training1/positive"]
    # class_list = ["negative","positive"]
    # ds.createTrainingFromDataset("test",class_list,path_list)

    data_path = "../Samples/Palmahim1"
    training_path = "../data/training"
    test_path = "../data/test"
    ds.splitData(data_path,training_path,test_path)

def segmentationTest():
    im = cv.imread("..//Samples//slides//PL29II Nov 4-5 0134.tif")    
    cv.namedWindow("..//Samples//slides//PL29II Nov 4-5 0134.tif",cv.WINDOW_NORMAL)
    cv.imshow("..//Samples//slides//PL29II Nov 4-5 0134.tif",im)
    ce = componentExtractor(im)
    components = ce.extractComponents()
    for i, component in enumerate(components):
        cv.namedWindow(str(i),cv.WINDOW_NORMAL)
        cv.imshow(str(i),component[0])
    cv.waitKey()

def classifierTest():
    img = cv.imread("..//Samples//slides//A0004.tif")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",img)

    cl = classifier(Dataset="binData/test.npz")
    #cl.posNegDecompose()
    cl.plotPCA()
    cv.waitKey()

def validateClassifier():
    cl = classifier(Dataset="binData/trainingPalmahim1.npz",regression=True)
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
    cl = classifier(Dataset="binData/trainingPalmahim1.npz",regression=True)
    #cl.plotPCA()
    #cl.crossValidateGridSearch()
    cl.regressionCrossValidation(svr=True)

if __name__ == '__main__':
    #main()

    #CNNTest()
    #featureExtractorTest()
    #DensityTest()
    #datasetOrgenizerRegressionTest()
    #datasetOrginizerClassificationTest()
    #classifierTest()
    #crossValidateTest()
    #segmentationTest()
    datasetOrginizerTrainKmeans()
    validateClassifier()
    