__author__ = 'nimrodshn'
import numpy as np
import os
import cv2 as cv

class dataSetOrginizer:
    def __init__(self):
       self._dataSets=[]

       defaultSet = np.load('Data.npz')
       self._dataSets.append(defaultSet)


    def InputDataset(self,name):
        """
        Inputs The Images for Training set to be created.
        :param name:
        :return:
        """


    def createTrainingFromDataset(self, name, location=''):
        '''
        Creates a new training set to work on from given dataset in location: creating Feature Vector, Normalize etc..
        :param name: the name of the data set
        :param location: location where the data was collected
        :return:
        '''

        address = "..//data//training//" + name
        labels = []
        trainingData = []
        classes = []
        cl = 0
        for items in os.listdir(address):

            name = address + "//" + items
            labels.append(items)
            #print items

            for it in os.listdir(name):

                path = name + "//" + it
                #print path # DEBUG

                img = cv.imread(path)

                orb = cv.ORB()
                kp = orb.detect(img,None)

                ## Normalize the Data, taking only Data with 50 KeyPoints
                if len(kp) > 15:
                    kp = kp[:15]
                    kp, des = orb.compute(img, kp)

                    '''
                    ## DEBUG ##
                    im2 = cv.drawKeypoints(img ,kp,color=(0,255,0), flags=0)
                    cv.namedWindow(path, cv.WINDOW_NORMAL)
                    cv.imshow(path, im2)
                    '''

                ####### Transformations on the Array #######
                    d=des.flatten()
                    trainingData.append(d)
                    classes.append(cl)

            cl = cl + 1

        np.savez(name,trainingData,labels, classes)


    def addImageToTrainingSet(self, InputImage, cl,Trainingset='Data.npz'):
        '''

        :param InputImage: Image to be added to training set.
        :param cl: the class number of the Foram if exist.
        :return:Void
        '''

        img = InputImage
        orb = cv.ORB()
        kp = orb.detect(img,None)


        ## Normalize the Data, taking only Data with 15 KeyPoints
        if len(kp) > 15:
            npzfile = np.load(Trainingset) # Loading Dataset
            trainingData = npzfile['arr_0']
            labels = npzfile['arr_1']
            classes = npzfile['arr_2']


            kp = kp[:15]
            kp, des = orb.compute(img, kp)

            '''
            ## DEBUG ##
            im2 = cv.drawKeypoints(img ,kp,color=(0,255,0), flags=0)
            cv.namedWindow(path, cv.WINDOW_NORMAL)
            cv.imshow(path, im2)
            '''

            ####### Transformations on the Array #######
            d=des.flatten()
            trainingData.append(d)
            classes.append(cl)

            np.savez(Trainingset,trainingData,labels, classes)
