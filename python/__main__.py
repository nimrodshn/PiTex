
__author__ = 'Nimrod Shneor'
import cv2 as cv
from Tkinter import *
from GUI import ForamGUI


def main():


    #img = cv.imread("..//Samples//1//2.jpg")
    #cv.namedWindow("Sample",cv.WINDOW_NORMAL)
    #cv.imshow("Sample",img)

    root = Tk()
    ForamGUI(root)
    root.attributes('-zoomed', True)
    root.mainloop()

    cv.waitKey()


if __name__ == '__main__':
    main()