__author__ = 'Nimrod Shneor'
import cv2 as cv
from componentExtractor import componentExtractor
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class classifier:

    def __init__(self, inputImage):
        self._image = inputImage

    def classifieSample(self, Dataset='binData/Default.npz'):
        '''
        :param Dataset: The dataset used to create the model.
         Main Classifier Function.
        :return:
        '''

        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        X = trainingData
        y = classes

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
