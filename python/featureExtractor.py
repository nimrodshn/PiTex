__author__ = 'Nimrod Shneor'

import cv2 as cv
import numpy as np
import mahotas as mh
from skimage.feature import local_binary_pattern

# TODO: 
# 1. Add feature learning.
# 2. Add feature normalization.
# 3. Add features: mean intensinty, standard dev of intensint, "room structure" feature(!!)

class featureExtractor:
    def __init__(self,input):
        self.im = input

    def computeFeatureVector(self):
        '''
        Computes the feature vector using different featurs.
        :return: a list representing the feature vector to be called by datasetOrginizer to build your dataset.
        '''

        # morphotype = self.computeMorphtypeNumber()

        # gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)

        # hist = cv.calcHist(gray, [0], None, [8], [0, 256])

        # hist = hist.flatten()

        # hist = hist.tolist()

        haralick = mh.features.haralick(self.im, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False).flatten()
        
        haralick = haralick.tolist()

        #size = self.computeSize()

        #shape = self.computeHuShape()

        #solidity = self.computeSolidity()

        #corners = self.computeGoodFeaturesToTrack()

        # filters = self.buildGaborfilters()
        # res = self.processGabor(self.im,filters)
        # gabor_vector = self.computeMeanAmplitude(res)

        #hog = self.computeHOG()

        lbp = self.computeLBP().flatten() 

        lbp = lbp.tolist()

        feature_vector =  lbp + haralick

        return feature_vector


    ########### FEATURES #################

    def computeBrightness(self):
        gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        return np.mean(gray)


    def computeLBP(self):
        gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        res = mh.features.lbp(gray, radius=2, points=8, ignore_zeros=False)
        return res

    def computeZernikeMoments():
        return mh.features.zernike_moments(self.im, radius=20, degree=8)

    def computeHOG(self):
        hog = cv.HOGDescriptor()
        h = hog.compute(self.im)
        return h

    def computeGoodFeaturesToTrack(self):
        gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,4,0.01,10)
        corners = np.asanyarray(corners,dtype=np.float32)
        return corners.flatten()

    def computeSolidity(self):
        component = self.getMainComponent()
        area = cv.contourArea(component)
        hull = cv.convexHull(component)
        hull_area = cv.contourArea(hull)
        solidity = hull_area/float(area)
        return solidity

    def computeSize(self):
        shape = self.im.shape
        return float(shape[0])*float(shape[1])

    def computeHuShape(self):
        imgray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 110, 255, cv.THRESH_BINARY)
        feature = cv.HuMoments(cv.moments(thresh)).flatten()
        return feature.tolist()

    def getMainComponent(self):
        #Normalizing Image:
        imgray = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)

        # What method should we use?
        ret, thresh = cv.threshold(imgray ,0,255,cv.THRESH_OTSU)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        max_contour_idx = 0
        for i,contour in enumerate(contours):
            if cv.contourArea(contours[i]) > cv.contourArea(contours[max_contour_idx]): # Find max contour
                max_contour_idx = i

        #rect = cv.boundingRect(contours[max_contour_idx])
        #subrect = cv.cv.GetSubRect(cv.cv.fromarray(self.im),rect)
        #component = np.asanyarray(subrect)

        return contours[max_contour_idx]

    def computeMorphtypeNumber(self):
        # Compute MorphoType:
        shape = self.im.shape
        #print shape, shape[0],shape[1], float(shape[0])/float(shape[1])

        morphoType = float(shape[0])/float(shape[1])
        return morphoType

    def buildGaborfilters(self):
        filters = []
        ksize = 30
        for theta in np.arange(0, np.pi, np.pi / 8):
            params = {'ksize':(ksize, ksize), 'sigma':10, 'theta':theta, 'lambd':15.0,
                      'gamma':0.00, 'psi':0, 'ktype':cv.CV_32F}
            kern = cv.getGaborKernel(**params)
            kern /= 1.5*kern.sum()
            filters.append((kern,params))
        return filters


    def processGabor(self,img, filters):
        results = []
        for kern,params in filters:
            gray = cv.cvtColor(self.im,cv.COLOR_BGR2GRAY)
            fimg = cv.filter2D(gray, cv.CV_8UC1, kern)
            results.append(fimg)
        return results


    def computeMeanAmplitude(self,results):
        feature_vector = []
        for i, result in enumerate(results):
            #cv.namedWindow("result"+str(i),cv.WINDOW_NORMAL)
            #cv.imshow("result"+str(i), results[i])

            ## Computing Mean Amplitud
            temp = np.array(result)
            temp = np.abs(temp)
            sum = np.sum(temp)
            #print sum
            feature_vector.append(sum)

        print feature_vector
        #cv.waitKey()
        return feature_vector

