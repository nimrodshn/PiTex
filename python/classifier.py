__author__ = 'user'
import os
import cv2 as cv

class classifier:
    def __init__(self):
        self._svm = cv.SVM()

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

        return  trainingData,labels, classes







