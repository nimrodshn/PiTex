__author__ = 'user'
import os
import cv2 as cv
import sklearn as sk
import numpy as np

class classifier:

    def __init__(self):
        self.__svm = cv.SVM()

    def getTrainingData(self):

        address = "..//data//training"
        labels = []
        trainingData = []
        label = 0;
        for items in os.listdir(address):
            ## extracts labels
            label = label+1.0
            name = address + "//" + items

            for it in os.listdir(name):

                path = name + "//" + it
                print path

                img = cv.imread(path)
                orb = cv.ORB()
                kp = orb.detect(img,None)
                kp, des = orb.compute(img, kp)
                #img2 = cv.drawKeypoints(img,kp,color=(0,255,0), flags=0)

                ####### Transformations on the Array #######
                d = np.array(des, dtype = np.float32)
                q = d.flatten()
                if (len(q) == 16000):
                    trainingData.append(q)
                    labels.append(label)

                training_arr = np.array(trainingData, dtype=np.float32)
                labels_arr = np.array(labels, dtype=np.float32)

                ####### Training the SVM #######
                svm_params = dict( kernel_type = cv.SVM_LINEAR,
                    svm_type = cv.SVM_C_SVC,
                    C=2.67, gamma=3 )

                self.__svm.train(training_arr, labels_arr, params=svm_params)
                self.__svm.save('svm_data.dat')


                ######DEBUG######
                '''
                cv.namedWindow(path,cv.WINDOW_NORMAL)
                cv.imshow(path,img2)
                '''
        return trainingData, labels

    def predict(self, desc):
        print self.__svm.predict(desc)



