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
        '''
        Initialize UI
        :return:
        '''

        self.parent.title("Pitex")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)
        projectMenu = Menu(menubar)


        menubar.add_cascade(label="Project", menu=projectMenu)
        projectMenu.add_command(label="New Project")
        projectMenu.add_command(label="open Existing")

        menubar.add_cascade(label="Dataset Manager", menu=fileMenu)
        fileMenu.add_command(label="New Dataset", command=self.createNewDataset)
        fileMenu.add_command(label="Open Existing", command=self.onOpen)



    def createNewDataset(self):
        class_dir = []

        dir = tkFileDialog.askdirectory(title='Select your pictures folder')

        class_dir.append(dir)




    def onOpen(self):
        dlg = tkFileDialog.askopenfilename(parent=self.parent ,initialdir='/home/',title='Select Dataset', filetypes=[('nps', '.npz')])
        filename = dlg
        print filename
        #im = cv2.imread(str(self.fn))
        #cv2.namedWindow("LoadedImage")
        #cv2.imshow("LoadedImage",im)


    def onError(self):
        print("Error", "Could not open file")

