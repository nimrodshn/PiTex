
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

def main():
    root = Tk()
    ForamGUI(root)
    root.attributes('-zoomed', True)
    root.mainloop()

    cv.waitKey()

########### TESTS ###########

def featureExtractorTest():
    path1 = "../data/training/Default/miliolids/miliolid1.jpg"
    path2 = "../data/training/Default/Ammonia beccarii/A. beccarii5.jpg"

    im1 = cv.imread(path1)
    im2 = cv.imread(path2)
    
    plt.show()

    cv.waitKey()

def datasetOrginizerTrainKmeans():

    ds = datasetOrginizer()
    path1 = '../data/hodlout'
    path2 = '../data/training'
    ds.createRegressionTrainingFromDataset(dataset_name="kmeansPalmahim1",path=path1)
    ds.KmeansTrainingDataset(Dataset="binData/kmeansPalmahim1.npz",path2)

def datasetOrgenizerRegressionTest():
    
    ds = datasetOrginizer()
    
    path = '../data/training'
    
    labels = [0 , 6, 0, 1, 1, 0, 1, 1, 0, 2,
            0, 4, 1, 4, 1, 0, 4, 0, 0, 2,
            1, 1, 2, 1, 2, 1, 6, 3, 0, 0,
            2, 1, 0, 1, 1, 6, 0, 4, 1, 8,
            1, 2, 0, 0, 1, 0, 0, 4, 0, 3,
            2, 2, 2, 0, 2, 4, 0, 6, 1, 1,
            2, 3, 6, 0, 1, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 0, 0, 2, 5, 9, 1,
            1, 0, 0, 1, 1, 1, 1, 1, 5, 0,
            2, 1, 1, 2, 0, 2, 0, 0, 2, 2,
            0, 1, 4, 4, 1, 2, 1, 2, 1, 5,
            0, 2, 0, 0, 5, 0, 1, 4, 2, 3,
            3, 0, 5, 5, 1, 1, 1, 1, 1, 2,
            2, 1, 6, 0, 1, 3, 7, 2, 0, 0,
            0, 1, 0, 4, 0, 2, 4, 1, 3, 0,
            0, 2, 2 ,5, 1, 1, 3, 0, 5, 1,
            2, 0, 1, 2, 4, 1, 0, 1 ,1, 0,
            0, 0, 1 ,1, 0, 0, 2, 5, 1, 1,
            3, 0, 2]

    print np.avg(labels)

    print len(labels)
    
    ds.createRegressionTrainingFromDataset("test",labels,path)


def datasetOrginizerClassificationTest():
    ds = datasetOrginizer()
    path_list = ["../data/training1/negative","../data/training1/positive"]
    class_list = ["negative","positive"]
    ds.createTrainingFromDataset("test",class_list,path_list)

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
    cl = classifier(Dataset="binData/test.npz",regression=True)
    #cl.validation()

    true_val = [0, 2, 4, 1, 1, 0, 2, 0, 1, 0,
                4, 4, 1, 0, 4, 5, 1, 0, 1, 0,
                0, 2, 2, 1, 0, 0, 0, 3, 3, 1,
                3, 1, 7, 3, 2, 3, 0, 0, 2, 2,
                1, 7, 1, 1, 0, 4, 4, 0, 1, 0,
                5, 0, 1, 3, 1, 4, 0, 5, 2, 4,
                4, 1, 1, 5, 1, 1, 1, 0, 2, 1,
                1, 0, 1, 1, 3, 1, 2, 1, 2, 3,
                1, 1, 4, 0, 3, 0, 3, 0, 2, 0,
                1, 3, 2, 2, 1, 1, 4, 0, 3, 0 ]


    cl.regressionValidation(true_val)
    cv.waitKey()

def crossValidateTest():
    cl = classifier(Dataset="binData/test.npz",regression=True)
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
    #validateClassifier()
    #segmentationTest()
    datasetOrginizerTrainKmeans()