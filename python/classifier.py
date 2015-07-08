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


    def posNegDecompose(self, Dataset='binData/test4.npz'):
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
        

        '''
        pca = PCA(n_components=2)
        X_r = pca.fit(self.X).transform(self.X)

        for c, i, target_name in zip("rgb", [0, 1], ["negative","positive"]):
            plt.scatter(X_r[self.y == i, 0], X_r[self.y == i, 1], c=c, label=target_name)
        plt.legend()
        plt.title('PCA')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        '''
        '''
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(self.X,
            self.y, test_size=0.20)

        parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.001, 0.0001]}
        # search for the best classifier within the search space and return it
        clf = grid_search.GridSearchCV(SVC(), parameters).fit(x_train, y_train)
        svm = clf.best_estimator_

        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
            svm.predict(x_test)))
        '''
        
        #rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        #rf.fit(self.X,self.y)

        #bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=300)
        #bdt.fit(self.X,self.y)

        clf = GaussianNB()
        clf.fit(self.X,self.y)

        ## Segmentation
        ce = componentExtractor(self._image)
        components = ce.extractComponents() # THIS IS A LIST

        #clf = SVC(kernel="linear",C=1,gamma=0.01)
        #clf.fit(self.X,self.y)

        
        for i, component in enumerate(components):

            fe = featureExtractor(component[0])
            feature_vector = fe.computeFeatureVector()

            #cv.namedWindow("result"+str(i),cv.WINDOW_NORMAL)
            #cv.imshow("result"+str(i), component[0])

            print feature_vector

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


