__author__ = 'nimrodshn'
from Tkinter import *
import tkFileDialog
import cv2
import numpy as np


class ForamGUI(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()


    def initUI(self):
        '''
        Initialize Main UI for PiTex
        :return:
        '''

        self.parent.title("PiTex 0.1 BETA")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        datasetMenu = Menu(menubar)
        projectMenu = Menu(menubar)
        helpMenu = Menu(menubar)

        menubar.add_cascade(label="Project", menu=projectMenu)
        projectMenu.add_command(label="New Project")
        projectMenu.add_command(label="Open Existing")
        projectMenu.add_command(label="Save Project")
        projectMenu.add_command(label="Recent Project")
        projectMenu.add_command(label="Exit")

        menubar.add_cascade(label="Dataset Manager", menu=datasetMenu)
        datasetMenu.add_command(label="New Dataset", command=self.DatasetManager)
        datasetMenu.add_command(label="Open Existing", command=self.OpenExistingDataset)

        menubar.add_cascade(label="Help", menu=helpMenu)
        helpMenu.add_command(label="Contents")
        helpMenu.add_command(label="About")


    def DatasetManager(self):
        class_dir = {}

        window = Toplevel(self)
        window.title('DatasetManager')
        window.geometry("400x400")

        ## LABELS ##

        header = Label(window, text="Dataset Manager")
        label1 = Label(window, text='Enter class Name')

        ## ENTRYS ##

        class_name = StringVar()
        name_entry = Entry(window, textvariable=class_name,text='enter class name')

        dir_string = StringVar()
        dir_entry = Entry(window, textvariable=dir_string, text='dir path')

        ## LIST ##

        scrollbar = Scrollbar(window)

        mylist = Listbox(window, yscrollcommand = scrollbar.set )

        scrollbar.config( command = mylist.yview )

        ## BUTTONS ##

        importButton = Button(window, text='import folder',command=self.onOpen)

        commitClassButton = Button(window, text='commit class')

        finishButton = Button(window, text='Finish')

        ## LAYOUT ##
        header.grid(column=5, row=0, columnspan=3,pady=10)
        label1.grid(column=3,row=4,columnspan=2)
        dir_entry.grid(column=5, row=3, columnspan=3)
        name_entry.grid(column=5, row=4)
        mylist.grid(column=5,row=5)
        importButton.grid(column=4, row=3)
        commitClassButton.grid(column=9, row=4)
        finishButton.grid(column=5, row=8)


    def onOpen(self):
        dir = tkFileDialog.askdirectory(title='Select your pictures folder')
        print dir
        return dir


    def OpenExistingDataset(self):
        dlg = tkFileDialog.askopenfilename(parent=self.parent, initialdir='/home/', title='Select Dataset',
                                           filetypes=[('nps', '.npz')])
        filename = dlg
        print filename


    def onError(self):
        print("Error", "Could not open file")

