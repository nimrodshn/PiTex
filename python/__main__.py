
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
import matplotlib.pyplot as pl
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

def featureSelectionTest():
    clf = classifier()
    clf = classifier()
    clf.feature_selection(Dataset="binData/test4.npz")
    #feature_vector = fe.computeFeatureVector()
    #print feature_vector

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

def datasetOrgenizerTest():
    
    ds = datasetOrginizer()
    path_list = ["../data/training2/negative","../data/training2/positive"]
    class_list = ["negative","positive"]
    ds.createTrainingFromDataset("test4",class_list,path_list)

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
    ce = componentExtractor(im)
    components = ce.extractComponents()
    cv.waitKey()

def classifierTest():
    img = cv.imread("..//Samples//slides//A0004.tif")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",img)

    cl = classifier(inputImage=img,Dataset="binData/test4.npz")
    cl.posNegDecompose()
    #cl.plotPCA()
    cv.waitKey()

def validateClassifier():
    test_num =  random.sample(range(1, 203), 100)
    cl = classifier(Dataset="binData/test4.npz")
    cl.validation(test_num)
    cv.waitKey()

def crossValidateTest():
    cl = classifier(Dataset="binData/test4.npz")
    cl.plotPCA()
    cl.crossValidateGridSearch()

if __name__ == '__main__':
    #main()

    #CNNTest()
    #featureExtractorTest()
    #featureSelectionTest()
    #datasetOrgenizerTest()
    classifierTest()
    #crossValidateTest()
    #validateClassifier()
    #segmentationTest()