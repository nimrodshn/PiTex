__author__ = 'nimrodshn'
from Tkinter import *
import Tkconstants
import tkFileDialog
import cv2
import numpy as np


class ForamGUI(Frame):
    def __init__(self,parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()


    def initUI(self):
        self.parent.title("Pitex - Classifing Forams since 2015")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
        fileMenu = Menu(menubar)
        self.fn=''
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)

    def onOpen(self):

        ftypes = [('Image Files', '*.tif *.jpg *.png')]
        dlg = tkFileDialog.Open(self, filetypes = ftypes)
        filename = dlg.show()
        self.fn=filename
        #im = cv2.imread(str(self.fn))
        #cv2.namedWindow("LoadedImage")
        #cv2.imshow("LoadedImage",im)


    def onError(self):
        print("Error", "Could not open file")

