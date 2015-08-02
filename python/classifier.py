__author__ = 'Nimrod Shneor'
import cv2 as cv
from componentExtractor import componentExtractor
import numpy as np
from sklearn.svm import SVC
import csv
from featureExtractor import featureExtractor
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import metrics

class classifier:

    def __init__(self, inputImage = None):
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



    def validation(self, val_images, Dataset):
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

        clf = clf = SVC(C=1, cache_size=200, class_weight={1: 10}, coef0=0.0, degree=2,
                   gamma=0.0, kernel='poly', max_iter=-1, probability=False, random_state=None,
                   shrinking=True, tol=0.001, verbose=False)
        clf.fit(X,y)


        print np.shape(X)

        fig, axes = plt.subplots(nrows=10, ncols=10)

        test_set = val_images
        for i,num in enumerate(test_set):
            print num
            im = cv.imread("..//Samples//Validation_Set//" + str(num) + ".jpg")
            fe = featureExtractor(im)
            feature_vector = fe.computeFeatureVector()
            res = clf.predict(feature_vector)
            print res

            plt.subplot(10,10,i)
            plt.imshow(im)
            plt.xticks([])
            plt.yticks([])

            if res[0] == 1:
                plt.title('positive')
                # cv.namedWindow("positive" + str(num),cv.WINDOW_NORMAL)
                # cv.imshow("positive" + str(num),im)
            else:
                plt.title('negative')
                # cv.namedWindow("negative" + str(num),cv.WINDOW_NORMAL)
                # cv.imshow("negative" + str(num),im)
        
        fig.subplots_adjust(hspace=.5)
        
        plt.show()    


    def posNegDecompose(self, Dataset):
        '''
        :param Dataset: The dataset used to create the model.
         This function returns list of 'objects of interest': e.g. suspected forams.
        :return:
        '''

        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        self.X = trainingData
        self.y = classes

        print np.shape(self.X)

        ### Segmentation
        ce = componentExtractor(self._image)
        components = ce.extractComponents() # THIS IS A LIST

        ### Model Building 
        clf = SVC(C=1, cache_size=200, class_weight={1: 10}, coef0=0.0, degree=2,
                  gamma=0.0, kernel='poly', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)
        clf.fit(self.X,self.y)

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
         Main Classifier Function. This function classifies 'Potential Forams' to their different species after posNegDecompose has be used to distinguish between positive and negative components.
        :return:
        '''

        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']

        self.X = trainingData
        self.y = classes

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


