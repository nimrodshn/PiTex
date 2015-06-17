__author__ = 'nimrodshn'
import Tkinter as tk
import tkMessageBox
import tkFileDialog
from dataSetOrginizer import datasetOrginizer



## DATASET MANAGER GUI ##

class DatasetManagerGUI:

    def __init__(self,parent):
        self.parent = parent

    def initUI(self):
        self.class_to_be_added = {}
        self.class_path_list = []
        self.class_name_list = []
        self.class_name_list = []

        self.window = tk.Toplevel(self.parent)
        self.window.title('Dataset Manager')
        self.window.geometry("450x400")
        self.window.lift(self.parent)

        ## LABELS ##

        header = tk.Label(self.window, text="Dataset Manager")
        label2 = tk.Label(self.window, text='Enter class Name')
        label1 = tk.Label(self.window, text='Enter dataset name')

        ## ENTRYS ##

        dataset_name = tk.StringVar()
        self.dataset_name_entry = tk.Entry(self.window, textvariable=dataset_name)

        class_name = tk.StringVar()
        self.name_entry = tk.Entry(self.window, textvariable=class_name)

        dir_string = tk.StringVar()
        self.dir_entry = tk.Entry(self.window, textvariable=dir_string)

        ## LIST ##

        listframe = tk.Frame(self.window, relief=tk.GROOVE)

        scrollbar = tk.Scrollbar(listframe,orient="vertical")

        self.mylist = tk.Listbox(listframe, yscrollcommand = scrollbar.set )
        self.mylist.pack(side=tk.RIGHT)

        scrollbar.pack(side=tk.LEFT,fill=tk.Y)
        scrollbar.config(command=self.mylist.yview)

        ## BUTTONS ##

        importButton = tk.Button(self.window, text='import folder',command=self.onOpenDir)

        commitClassButton = tk.Button(self.window, text='commit class',command=self.onCommit)

        finishButton = tk.Button(self.window, text='Finish',command=self.onFinish)

        ## GRID-LAYOUT ##

        header.grid(column=5, row=0, columnspan=3,pady=10)
        label1.grid(column=3,row=5,columnspan=2)
        label2.grid(column=3,row=7,columnspan=2)
        self.dataset_name_entry.grid(column=5, row=5, columnspan=3)
        self.dir_entry.grid(column=5, row=6, columnspan=3)
        self.name_entry.grid(column=5, row=7)
        listframe.grid(column=5,row=8)
        importButton.grid(column=4, row=6)
        commitClassButton.grid(column=9, row=6)
        finishButton.grid(column=5, row=10)

    ### DATASET MANAGER FUNCTIONS ###

    def onOpenDir(self):
        dir = tkFileDialog.askdirectory(title='Select your pictures folder')
        self.dir_entry.insert(0,dir)


    def onCommit(self):
        className = str(self.name_entry.get())
        dir = self.dir_entry.get()
        if not className:
            tkMessageBox.showinfo("Error", "Please Enter Class Name")
        if not dir:
            tkMessageBox.showinfo("Error", "Please Choose Image Directory")

        else:
            self.class_path_list.append(str(dir))
            self.class_name_list.append(str(className))
            self.mylist.insert(0,className)

            self.name_entry.delete(0,tk.END)
            self.dir_entry.delete(0,tk.END)


    def onFinish(self):
        dataset_name = self.dataset_name_entry.get()
        if not dataset_name:
                tkMessageBox.showinfo("Error", "Please Enter dataset name")
        else:
            do = datasetOrginizer()
            do.createTrainingFromDataset(dataset_name,self.class_name_list,self.class_path_list)
            self.parent.mydatasetlist.insert(0,dataset_name)
            self.window.destroy()


    def OpenExistingDataset(self):
        dlg = tkFileDialog.askopenfilename(parent=self.parent, initialdir='/home/', title='Select Dataset',
                                           filetypes=[('nps', '.npz')])
        filename = dlg
        print filename


    def onError(self):
        tkMessageBox.showinfo("Error", "Please Choose Image Directory")

