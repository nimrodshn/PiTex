__author__ = 'Nimrod Shneor'

import os
import random
import cv2 as cv
from componentExtractor import componentExtractor
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import linear_model
from featureExtractor import featureExtractor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn import preprocessing
from sklearn import metrics
from sklearn.externals import joblib


# TODO:
# 1. Add cross validation to tune SVM parameters in posNeg Decomposition method.
# 2. Add more images to validation set from different samples
# 3. Start Exploring Neural-Network as a classifier for posNeg Decomposition / Specie Classification.

class classifier:

    def __init__(self, Dataset, regression, inputImage = None):
        '''
        :param :
        a. Dataset: Path to a given training dataset in the format of npz. (see datasetOrginizer Class.)
        b. Input Image to classify.
        c. regression: regression model or classifier model.
        '''
        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']
        
        self._image = inputImage
        self.max_min_features = npzfile['arr_3']
        
        print "training matrix size: " 
        print np.shape(trainingData)
        print "min_max feature array size: "
        print np.shape(self.max_min_features)
        
        if (regression == False):
            self.X, self.features_list = self.featureSelection(trainingData,classes)
        else:
            self.X = trainingData


        self.y = labels
        
    def featureSelection(self,X,y):
        '''
         Feature selection recursive feature elimination with Linear SVM.
        :param:
         a. X the training matrix.
         b. y the labels column corresponding the X.
        :return: 
            a. The mask of top 10% features using.
            b. The transformed training matrix
        '''

        print np.shape(X)

        selector = SelectPercentile(chi2, percentile=10)
        X_new = selector.fit_transform(X, y)
        
        return X_new, selector.get_support()

    def plotPCA(self):
        print np.shape(self.X)

        pca = PCA(n_components=2)
        X_r = pca.fit(self.X).transform(self.X)

        for c, i, target_name in zip("rgb", [0, 1], ["negative","positive"]):
             plt.scatter(X_r[self.y == i, 0], X_r[self.y == i, 1], c=c, label=target_name)
        plt.legend()
        plt.title('PCA')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def regressionValidation(self,test_list , true_val_Vector):
        '''
        Main Regression Validation function to validate model on
        :param true_val_Vector: true values of test set.
        '''

        test_path = '../data/training'
        clf = SVR(C=1.0 ,gamma=1.5848931924611136, epsilon=1, kernel='rbf')

        clf.fit(self.X,self.y)

        results_vector = []

        k_means = joblib.load('KmeandPalmahim1.pkl')

        [m,num_of_clusters] = np.shape(self.X)

        for i, item in enumerate(test_list):
    
            im = cv.imread(item)
            fe = featureExtractor(im)
            feature_vector = np.zeros(num_of_clusters)
            raw_vector = fe.computeFeatureVector()
            Km_vector = k_means.predict(raw_vector) 
            for i in range(len(Km_vector)):
                feature_vector[Km_vector[i]] = feature_vector[Km_vector[i]] + 1 

            res = clf.predict(feature_vector)
            
            print "number of forams in image " + item + " is: " + str(res)

            results_vector.append(res)

        avg_classifier_res = np.zeros(len(results_vector))
        for num in avg_classifier_res: 
            num = np.average(true_val_Vector) 

        print "results avg: " + str(np.average(results_vector))
        print "true val avg: " + str(np.average(true_val_Vector))


        print "'spit avg classifier' mean squared error: " + str(metrics.mean_squared_error(true_val_Vector,avg_classifier_res)) 
        print "'spit avg classifier' mean absolut error: " + str(metrics.mean_absolute_error(true_val_Vector,avg_classifier_res)) 
        
        print "mean squared error: " + str(metrics.mean_squared_error(true_val_Vector,results_vector))
        print "mean absolut error: " + str(metrics.mean_absolute_error(true_val_Vector,results_vector))  
            
    def classificationValidation(self):
        '''
        Main Validation function to validate model on
        :param val_images: the numbers of the images. to be picked randomly.
        '''

        test_path = '../data/test1'
        numofdata = len(os.listdir(test_path))
        #pick 100 test images at random
        test_num = random.sample(range(1, numofdata), 100) 
        A = np.zeros(numofdata)        
        for k in range(100):
            A[test_num[k]] = 1 

        clf = SVC(C=0.1, gamma=0.01, kernel='rbf') 
        clf.fit(self.X,self.y)

        fig, axes = plt.subplots(nrows=10, ncols=10)

        counter = 0;

        for i, item in enumerate(os.listdir(test_path)):
            #print num
            if A[i] == 1:

                im = cv.imread(test_path + "/" + item)
                fe = featureExtractor(im)
                feature_vector = fe.computeFeatureVector()
                new_feature_vector = []

                for k, num in enumerate(feature_vector):
                     if (self.features_list[k] == True):
                         # Normalize feature vector
                        max_feature = self.max_min_features[k][0]
                        min_feature = self.max_min_features[k][1]
                        new_feature_vector.append((feature_vector[k] - min_feature) / (max_feature - min_feature))
                   
                res = clf.predict(new_feature_vector)
                print res

                plt.subplot(10,10,counter)
                plt.imshow(im)
                plt.xticks([])
                plt.yticks([])

                if res[0] == 1:
                    #plt.title('positive')
                     cv.namedWindow("positive" + str(counter),cv.WINDOW_NORMAL)
                     cv.imshow("positive" + str(counter),im)
                else:
                    #plt.title('negative')
                     cv.namedWindow("negative" + str(counter),cv.WINDOW_NORMAL)
                     cv.imshow("negative" + str(counter),im)
                
                counter = counter+1
            
        fig.subplots_adjust(hspace=.5)
        
        plt.show()


    def classificationCrossValidation(self):
        '''
        This function is used to cross validate the model using Grid Search Method.
        :return:
        '''        
        #Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
          self.X, self.y, test_size=0.5, random_state=0)

        # the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma':[0.1,0.2,0.4,0.8,1,2,4,6,8,10] ,
                            'C':[0.1 ,1, 10 ,100, 1000]}]
        
                            # [1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-6,1e-8,1e-10]
                            # [1e-5,1e-4,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]

                
        print("# Tuning hyper-parameters")
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='accuracy')
        
        clf.fit(X_train, y_train)

        # look at the results
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    def regressionCrossValidation(self ,svr=True):
        if (svr==True):
            tuned_parameters={"C": [1,2,4,6,10],
                                "gamma":np.logspace(0.2,0.4),
                                "epsilon":[1,2,4,6,10]}
            clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1),tuned_parameters, cv=5,
                               scoring='mean_squared_error')
        else:
            clf = GridSearchCV(linear_model.Ridge(alpha=1.0), cv=5,
                  param_grid={"alpha": [1e7,1e6,1e5,1e4,1e3,1e2,1e1,1e0, 0.1, 1e-2, 1e-3]},
                              scoring='mean_squared_error')
                    
        clf.fit(self.X, self.y)
        print("# Tuning hyper-parameters")
        print()
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print ("Best score is: " + str(np.sqrt(np.abs(clf.best_score_))))
        print()
        meanoftraining = np.mean(self.y)
        print ("Mean of labels is :" + str(meanoftraining))
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()


