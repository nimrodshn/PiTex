from classifier import classifier
import cv2 as cv
from Tkinter import *
from GUI import ForamGUI


__author__ = 'user'

img = cv.imread("..//Samples//1//2.jpg")
cv.namedWindow("Sample",cv.WINDOW_NORMAL)
cv.imshow("Sample",img)

root = Tk()
ex = ForamGUI(root)
root.geometry("1280x720")
root.mainloop()

cl = classifier(img)
cl.classifieSample()


cv.waitKey()
