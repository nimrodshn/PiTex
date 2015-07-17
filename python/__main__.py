
__author__ = 'Nimrod Shneor'

import cv2 as cv
from Tkinter import *
from GUI import ForamGUI
from classifier import classifier
import csv
from dataSetOrginizer import datasetOrginizer
from sklearn.datasets import load_digits
from featureExtractor import featureExtractor
from componentExtractor import componentExtractor
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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

def featureExtractorTest():
    features = {"morphotype":[],"gabor":[],"haralick":[]}
    path = "../data/training/Default/orbiculus/orbiculus (1).jpg"

    im = cv.imread(path)

    cv.namedWindow(path,cv.WINDOW_NORMAL)
    cv.imshow(path,im)

    fe = featureExtractor(im)

    feature_vector = fe.computeFeatureVector()

    print feature_vector

def DatasetOrgenizerTest():
    ds = datasetOrginizer()
    path_list = ["../data/training2/negative","../data/training2/positive"]
    class_list = ["negative","positive"]
    training, classes , labels = ds.createTrainingFromDataset("test4",class_list,path_list)

    X = training
    y = classes


def CNNTest():
    data = load_digits()
    l_in = lasagne.layers.InputLayer((100,50))
    l_hidden = lasagne.layers.DenseLayer(l_in,num_units=200)
    l_out = lasagne.layers.DenseLayer(l_hidden,num_units=10,nonlinearity=T.nnet.softmax)
    #pl.gray()
    #pl.matshow(data.images[1])
    #pl.show()


def csvTest():
    with open('eggs.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Spam'] * 5 + ['Baked Beans'])
        writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

def segmentationTest():
    im = cv.imread("..//Samples//4//PL29II Nov 4-5 0040.tif")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",im)


    npzfile = np.load('binData/test2.npz')
    trainingData = npzfile['arr_0']
    labels = npzfile['arr_1']
    classes = npzfile['arr_2']

    X = trainingData
    y = classes

    ce = componentExtractor(im)
    components = ce.extractComponents()

    featureMatrix = []
    for i, component in enumerate(components):
        fe = featureExtractor(component[0])
        #size = fe.computeSize()

        feature_vector = fe.computeFeatureVector()
        featureMatrix.append(feature_vector)

        svm = SVC(kernel="rbf",C=1,gamma=0.01)
        svm.fit(X,y)

        res = svm.predict(feature_vector)
        print res

        if res[0] == 1:
            x,y,w,h = component[1]
            cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)


    print np.shape(featureMatrix)

    pca = PCA(n_components=2)
    X_r = pca.fit(featureMatrix).transform(featureMatrix)

    plt.scatter(X_r[:, 0], X_r[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title('PCA')

    plt.show()

    cv.waitKey()

def classifierTest():
    img = cv.imread("..//Samples//4//PL29II Nov 4-5 0059.tif")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",img)

    cl = classifier(img)
    cl.posNegDecompose(Dataset="binData/test4.npz")

    cv.waitKey()

def validateClassifier():
    img = cv.imread("..//Samples//4//PL29II Nov 4-5 0059.tif")
    test_num =  random.sample(range(1, 203), 100)
    cl = classifier(img)
    cl.validation(test_num,"binData/test4.npz")
    cv.waitKey()

if __name__ == '__main__':
    #main()

    #CNNTest()
    #featureExtractorTest()
    #DatasetOrgenizerTest()
    #classifierTest()
    validateClassifier()
    #segmentationTest()