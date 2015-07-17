__author__ = 'Nimrod Shneor'
import cv2 as cv
from componentExtractor import componentExtractor
import numpy as np
from sklearn.svm import SVC
import csv
from featureExtractor import featureExtractor
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics

class classifier:

    def __init__(self, inputImage):
        self._image = inputImage
        self.X = None
        self.y = None




    def plotPCA(self,Dataset):
        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        X = trainingData
        y = classes

        print np.shape(X)

        pca = PCA(n_components=2)
        X_r = pca.fit(self.X).transform(self.X)

        for c, i, target_name in zip("rgb", [0, 1], ["negative","positive"]):
             plt.scatter(X_r[self.y == i, 0], X_r[self.y == i, 1], c=c, label=target_name)
        plt.legend()
        plt.title('PCA')
        plt.xticks([])
        plt.yticks([])
        plt.show()



    def validation(self,val_images, Dataset):
        '''
        Main Validation function to validate model on
        :param val_images:
        :param Dataset:
        :return:
        '''

        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        X = trainingData
        y = classes

        clf = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=2,
                  gamma=0.0, kernel='poly', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
        clf.fit(X,y)


        print np.shape(X)

        #print val_images

        test_set = val_images
        for num in test_set:
            print num
            im = cv.imread("..//Samples//Validation_Set//" + str(num) + ".jpg")
            fe = featureExtractor(im)
            feature_vector = fe.computeFeatureVector()
            res = clf.predict(feature_vector)
            print res

            if res[0] == 1:
                cv.namedWindow("positive" + str(num),cv.WINDOW_NORMAL)
                cv.imshow("positive" + str(num),im)
            else:
                cv.namedWindow("negative" + str(num),cv.WINDOW_NORMAL)
                cv.imshow("negative" + str(num),im)



    def posNegDecompose(self, Dataset):
        '''
        :param Dataset: The dataset used to create the model.
         Main Classifier Function.
        :return:
        '''

        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        self.X = trainingData
        self.y = classes

        print np.shape(self.X)

        clf = GaussianNB()
        clf.fit(self.X,self.y)

        ## Segmentation
        ce = componentExtractor(self._image)
        components = ce.extractComponents() # THIS IS A LIST

        # clf = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0, degree=2,
        #           gamma=0.0, kernel='poly', max_iter=-1, probability=False, random_state=None,
        #           shrinking=True, tol=0.001, verbose=False)
        # clf.fit(self.X,self.y)

        
        for i, component in enumerate(components):

            fe = featureExtractor(component[0])
            feature_vector = fe.computeFeatureVector()

            #cv.namedWindow("result"+str(i),cv.WINDOW_NORMAL)
            #cv.imshow("result"+str(i), component[0])

            #print feature_vector

            res = clf.predict(feature_vector)
            print res
            
            if res[0] == 1:
                x,y,w,h = component[1]
                cv.rectangle(self._image,(x,y),(x+w,y+h),(0,255,0),2)

        
        cv.namedWindow("positive",cv.WINDOW_NORMAL)
        cv.imshow("positive",self._image)


        cv.waitKey()



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

        self.X = trainingData
        self.y = classes

        #knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
        #knn.fit(self.X,self.y)


        ## Segmentation
        ce = componentExtractor(self._image)
        components = ce.extractComponents # THIS IS A LIST


        for i,component in enumerate(components):

            fe = featureExtractor(component[i])
            morphoType = fe.computeMorphtypeNumber(components[i])
            filters = fe.buildGaborfilters()
            results = fe.processGabor(component,filters)
            feature_vector = fe.computeMeanAmplitude(results)
            feature_vector.append(morphoType)


