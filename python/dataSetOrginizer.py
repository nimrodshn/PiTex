__author__ = 'nimrod shneor'
import numpy as np
import os
import cv2 as cv

class datasetOrginizer:
    def __init__(self):
       self._dataSets=[]
       defaultSet = np.load('binData/Default.npz')
       self._dataSets.append(defaultSet)



    def createTrainingFromDataset(self, dataset_name, labels_list, path_list):
        '''
        Creates a new training set to work on from given dataset in location: creating Feature Vector, Normalize etc..
        :param name: the name of the data set
        :param location: location where the data was collected
        :return:
        '''
        base_path = "binData/"

        labels = []
        trainingData = []
        classes = []
        cl = 0

        for i, path in enumerate(path_list):

            labels.append(labels_list[i])
            print labels_list[i]


            for item in os.listdir(path):

                p = path + "/" + item
                print p # DEBUG

                im = cv.imread(p)

                orb = cv.ORB()
                kp = orb.detect(im,None)

                ## Normalize the Feature Vector, taking only Data with 15 KeyPoints 
                ## This needs work! not just picking "fifteen points" at random
                if len(kp) > 15:
                    kp = kp[:15]
                    kp, des = orb.compute(im, kp)

                    ####### Transformations on the Array #######
                    d=des.flatten()
                    trainingData.append(d)
                    classes.append(cl)

            cl = cl + 1

        np.savez(os.path.join(base_path, dataset_name), trainingData, labels, classes)



    def addImageToTrainingSet(self, path, cl, Trainingset='Default.npz'):
        '''
        :param InputImage: Image to be added to training set.
        :param cl: the class number of the Foram if exist.
        :return:Void
        '''

        npzfile = np.load(Trainingset) # Loading Dataset
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        for item in os.listdir(path):

            p = path + "/" + item

            img = cv.imread(p)
            orb = cv.ORB()
            kp = orb.detect(img,None)

            ## Normalize the Data, taking only Data with 15 KeyPoints
            if len(kp) > 15:

                kp = kp[:15]
                kp, des = orb.compute(img, kp)

                ####### Transformations on the Array #######
                d=des.flatten()
                trainingData.append(d)
                classes.append(cl)

        np.savez(Trainingset,trainingData,labels, classes)
