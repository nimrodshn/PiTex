
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

def DensityTest():
    img = cv.imread("..//Samples//slides//A0004.tif")
    cv.namedWindow("..//Samples//slides//A0004.tif",cv.WINDOW_NORMAL)
    cv.imshow("..//Samples//slides//A0004.tif",img)

    # fe = featureExtractor(img)
    # filters = fe.buildGaborfilters()
    # res = fe.processGabor(img,filters)
    # gabor_vector = fe.computeMeanAmplitude(res)

    # hist = cv.calcHist(res, [0], None, [256], [0, 256])

    # plt.hist(img.ravel(),256,[0,256]); plt.show()

    # plt.show()

    fe = featureExtractor(img)
    desc = fe.computeDenseSIFTfeatures()
    trainingData = []
    trainingData.append
    
    trainingData.append(desc)
    print np.shape(trainingData)
    cv.waitKey()

def featureExtractorTest():
    path1 = "../data/training/Default/miliolids/miliolid1.jpg"
    path2 = "../data/training/Default/Ammonia beccarii/A. beccarii5.jpg"

    im1 = cv.imread(path1)
    im2 = cv.imread(path2)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i,im in [(1,im1), (3,im2)]:        
        plt.subplot(2,2,i)
        plt.imshow(im,cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Original Image')

        kernel = np.ones((9,9),np.float32)/40
        dst = cv.filter2D(im,-1,kernel)
        
        #dst = cv.GaussianBlur(im,(7,7),0)

        edges = cv.Canny(dst,40,40)
        plt.subplot(2,2,i+1)
        plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image')
        plt.xticks([])
        plt.yticks([])
        
    plt.show()

    cv.waitKey()

def datasetOrgenizerRegressionTest():
    
    ds = datasetOrginizer()
    
    path = '../data/training2'
    
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

    print len(labels)
    
    ds.createRegressionTrainingFromDataset("test",labels,path)


def datasetOrginizerClassificationTest():

    ds = datasetOrginizer()
    
    path_list = ["../data/training1/negative","../data/training1/positive"]
    class_list = ["negative","positive"]
    ds.createTrainingFromDataset("test",class_list,path_list)

    data_path = "../Samples/slides"
    training_path = "../data/training1"
    test_path = "../data/test1"
    ds.splitData(data_path,training_path,test_path)


def CNNTest():
    data = load_digits()
    l_in = lasagne.layers.InputLayer((100,50))
    l_hidden = lasagne.layers.DenseLayer(l_in,num_units=200)
    l_out = lasagne.layers.DenseLayer(l_hidden,num_units=10,nonlinearity=T.nnet.softmax)

    # pl.gray()
    # pl.matshow(data.images[1])
    # pl.show()

def csvTest():
    with open('eggs.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Spam'] * 5 + ['Baked Beans'])
        writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

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
    cl.regressionValidation()
    cv.waitKey()

def crossValidateTest():
    cl = classifier(Dataset="binData/test.npz",regression=True)
    #cl.plotPCA()
    #cl.crossValidateGridSearch()
    cl.regressionCrossValidation()

if __name__ == '__main__':
    #main()

    #CNNTest()
    #featureExtractorTest()
    #DensityTest()
    #datasetOrgenizerRegressionTest()
    #datasetOrginizerClassificationTest()
    #classifierTest()
    crossValidateTest()
    #validateClassifier()
    #segmentationTest()