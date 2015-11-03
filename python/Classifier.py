__author__ = 'Nimrod Shneor'

import os
import random
import cv2 as cv
from ComponentExtractor import ComponentExtractor
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import linear_model
from FeatureExtractor import FeatureExtractor
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
from sklearn.metrics import confusion_matrix


# TODO:
# 1. Add cross validation to tune SVM parameters in posNeg Decomposition method.
# 2. Add more images to validation set from different samples
# 3. Start Exploring Neural-Network as a Classifier for posNeg Decomposition / Specie Classification.

class Classifier:

    def __init__(self, Dataset, regression):
        '''
        :param :
        a. Dataset: Path to a given training dataset in the format of npz. (see DatasetOrginizer Class.)
        b. Input Image to classify.
        c. regression: regression model or Classifier model.
        '''
        npzfile = np.load(Dataset)
        trainingData = npzfile['arr_0']
        labels = npzfile['arr_1']
        classes = npzfile['arr_2']
                
        # Debug
        print "training matrix size: " 
        print np.shape(trainingData)
        print "classes size "
        print np.shape(classes)

        # Scaling data using zscore.
        x_np = np.asarray(trainingData)
        X_scaled = (x_np - x_np.mean()) / x_np.std()
        
        self.X = X_scaled 
        self.y = classes

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
   
    def plotHistogram(self):
        '''
        Plotting histogram function.
        '''
        print self.X
        print self.y

        x_np = np.asarray(self.X)
        z_scores_np = (x_np - x_np.mean()) / x_np.std()
        X_scaled = z_scores_np

        print X_scaled

        for i, num in enumerate(self.y):
            if num == 1:
                counter = i
                break

        X_positive = X_scaled[:counter][:]
        X_negative = X_scaled[counter+1:][:]

        print np.shape(X_positive)
        print np.shape(X_negative)

    def plotPCA(self):
        '''
        Plot PCA function.
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

    def classificationValidation(self,test_list, KmeansName, kernel, C, gamma):
        '''
        Main classification Validation function to validate model on
        :param true_val_Vector: true values of test set.
        :param test_list: list of paths where test images are held.
        :param KmeansName: Load the Kmeans classifier for Binerization Task.
        '''
        if gamma == None:
            clf = SVC(C=C,kernel=kernel)
        else: 
            clf = SVC(C=C,gamma=gamma,kernel=kernel)

        print "kernel: " + kernel
        print "gamma: " + str(gamma)
        print "C: " + str(C)

        clf.fit(self.X,self.y)

        results_vector = []
        y_true = []
        cl=0

        k_means = joblib.load(KmeansName)

        [m,num_of_clusters] = np.shape(self.X)

        for path in test_list:
            for item in os.listdir(path): 
                p = path + "/" + item
                im = cv.imread(p)
                fe = FeatureExtractor(im)
                feature_vector = np.zeros(num_of_clusters)
                raw_vector = fe.computeFeatureVector()
                Km_vector = k_means.predict(raw_vector) 
                for k in range(len(Km_vector)):
                    feature_vector[Km_vector[k]] = feature_vector[Km_vector[k]] + 1 

                res = clf.predict(feature_vector)
                
                # Debugging                    
                if res[0] == 1:
                    print p + " is not a foram!"
                if res[0] == 0:
                    print p + " is a foram!"

                y_true.append(cl)
                results_vector.append(res[0])
            cl = cl + 1

        print "confusion_matrix"
        print confusion_matrix(y_true,results_vector)
 
    def classificationCrossValidation(self):
        '''
        This function is used to cross validate the model using Grid Search Method.
        :return:
        '''        
        #Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
          self.X, self.y, test_size=0.5, random_state=0)

        # the parameters by cross-validation
        tuned_parameters = [
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
                       ] 
                            # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}                        
                            # [1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-6,1e-8,1e-10]
                            # [1e-5,1e-4,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]

                
        print("# Tuning hyper-parameters")
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='f1')
        
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



