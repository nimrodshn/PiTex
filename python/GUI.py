__author__ = 'nimrodshn'
import Tkinter as tk
import tkMessageBox
import tkFileDialog
from PIL import Image, ImageTk
import cv2
import time
from collections import deque
from dataSetOrginizer import datasetOrginizer


class ForamGUI(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()


    def initUI(self):
        '''
        Initialize Main UI for PiTex
        :return:
        '''

        self.parent.title("PiTex 0.1 BETA")

        ## Menu Bar ##
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        datasetMenu = tk.Menu(menubar)
        projectMenu = tk.Menu(menubar)
        helpMenu = tk.Menu(menubar)

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


        ## CONTROL PANEL ##
        controlFrame = tk.Frame(self.parent,relief=tk.GROOVE, width = 100)

        self.stopphoto = tk.PhotoImage(file="../icons/stop.gif")
        self.pausephoto=tk.PhotoImage(file="../icons/pause.gif")
        self.playphoto=tk.PhotoImage(file='../icons/play.gif')
        self.recordphoto=tk.PhotoImage(file="../icons/record.gif")

        recordbtn = tk.Button(controlFrame,image=self.recordphoto,width=60,height=60)
        playbtn = tk.Button(controlFrame,image=self.playphoto,width=60,height=60)
        pausebtn = tk.Button(controlFrame,image=self.pausephoto,width=60,height=60)
        stopbtn = tk.Button(controlFrame,image=self.stopphoto,width=60,height=60)

        recordbtn.pack(side=tk.LEFT)
        playbtn.pack(side=tk.LEFT)
        pausebtn.pack(side=tk.LEFT)
        stopbtn.pack(side=tk.RIGHT)

        ## Current Working DatasetList ##

        self.datasetlistframe = tk.Frame(self.parent,relief=tk.GROOVE)

        datasetlbl = tk.Label(self.datasetlistframe, text='Current Working Datasets')
        datasetlbl.pack(side=tk.TOP)

        scrollbar = tk.Scrollbar(self.datasetlistframe,orient="vertical")

        self.mydatsetlist = tk.Listbox(self.datasetlistframe, yscrollcommand = scrollbar.set, width=30, height=20 )
        self.mydatsetlist.pack(side=tk.RIGHT)

        scrollbar.pack(side=tk.LEFT,fill=tk.Y)
        scrollbar.config(command=self.mydatsetlist.yview)

        self.mydatsetlist.insert(0,'Default Dataset')

        ## Main Sample View ##
        self.mainViewFrame = tk.LabelFrame(self.parent,width=800,height=600,text='Input Sample')
        self.image_label = tk.Label(master=self.mainViewFrame)
        self.image_label.pack()

        cam = cv2.VideoCapture(0)

        quit_button = tk.Button(master=self.mainViewFrame, text='Quit',command=lambda: self.quit_(self.parent))
        quit_button.pack()

        # setup the update callback
        self.parent.after(0, func=lambda: self.update_all(self.parent, self.image_label, cam))

        ## LAYOUT Main GUI ##
        self.mainViewFrame.grid(column=0,row=1, rowspan=10, padx=20 )
        controlFrame.grid(column=20,row=1)
        self.datasetlistframe.grid(column=20,row=2, columnspan=10)


    def quit_(self,root):
        root.destroy()

    def update_image(self,root,image_label, cam):
        (readsuccessful, f) = cam.read()
        im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        a = Image.fromarray(im)
        b = ImageTk.PhotoImage(image=a)
        image_label.configure(image=b)
        image_label._image_cache = b  # avoid garbage collection
        root.update()

    def update_all(self,root, image_label, cam):
        self.update_image(root,image_label, cam)
        root.after(20, func=lambda: self.update_all(root, image_label, cam))


    ## DATASET MANAGER GUI ##

    def DatasetManager(self):
        self.class_to_be_added = {}
        self.class_path_list = []
        self.class_name_list = []
        self.class_name_list = []

        self.window = tk.Toplevel(self.parent)
        self.window.title('DatasetManager')
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

    ### DATASET FUNCTIONS ###

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
            self.mydatsetlist.insert(0,dataset_name)
            self.window.destroy()


    def OpenExistingDataset(self):
        dlg = tkFileDialog.askopenfilename(parent=self.parent, initialdir='/home/', title='Select Dataset',
                                           filetypes=[('nps', '.npz')])
        filename = dlg
        print filename


    def onError(self):
        tkMessageBox.showinfo("Error", "Please Choose Image Directory")

