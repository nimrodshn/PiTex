__author__ = 'user'
import os
import cv2 as cv
from componentExtractor import componentExtractor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


class classifier:
    def __init__(self, inputImage):
        self._image = inputImage

    def getTrainingData(self):
        '''
        Training Model
        :return: Void
        '''

        address = "..//data//training"
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

            cl = cl+1
        np.savez('Data',trainingData,labels, classes)

    def classifieSample(self):
        npzfile = np.load('Data.npz')
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        X = trainingData
        y = classes

        C=1.0
        clf = svm.SVC(kernel='poly',C=C)
        clf.fit(X,y)

        knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
        knn.fit(X,y)


        ## Segmentation
        ce = componentExtractor(self._image)
        components = ce.extractComponents # THIS IS A LIST


        for i,component in enumerate(components):

            #cv.namedWindow(str(i), cv.WINDOW_NORMAL)
            #cv.imshow(str(i), component)


            orb = cv.ORB()
            kp = orb.detect(component,None)

            ## Normalize the Data
            if len(kp) > 15:


                kp = kp[:15]
                kp, des = orb.compute(component, kp)
                d=des.flatten()
                res = knn.predict(d)
                print "comp " + str(i) +":" + labels[res[0]]

                im2 = cv.drawKeypoints(component ,kp,color=(0,255,0), flags=0)
                cv.namedWindow(labels[res[0]] + " comp " + str(i), cv.WINDOW_NORMAL)
                cv.imshow(labels[res[0]] + " comp " + str(i), im2)








