
__author__ = 'Nimrod Shneor'
import cv2 as cv
from Tkinter import *
from GUI import ForamGUI
from classifier import classifier
import csv

def main():

    '''
    img = cv.imread("..//Samples//1//2.jpg")
    cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    cv.imshow("Sample",img)
    cl = classifier(img)
    cl.classifieSample()
    '''

    with open('eggs.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Spam'] * 5 + ['Baked Beans'])
        writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


    root = Tk()
    ForamGUI(root)
    root.attributes('-zoomed', True)
    root.mainloop()


    cv.waitKey()


if __name__ == '__main__':
    main()