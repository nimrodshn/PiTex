__author__ = 'nimrodshn'
from Tkinter import *
import tkMessageBox
import tkFileDialog
import PIL
import cv2
from dataSetOrginizer import datasetOrginizer


class vid():
    def __init__(self,cam,root,canvas):
        self.cam = cam
        self.root = root
        self.canvas = canvas

    def update_video(self,cam,root,canvas):
        (readsuccessful,f) = cam.read()
        gray_im = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        a = Image.fromarray(gray_im)
        b = PhotoImage(image=a)
        canvas.create_image(0,0,image=b,anchor=NW)
        root.update()
        root.after(33,self.supdate_video(self,cam,root,canvas))


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

        ## Menu Bar ##
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


        ## CONTROL PANEL ##
        controlFrame = Frame(self.parent,relief=GROOVE, width = 100)

        self.stopphoto = PhotoImage(file="../icons/stop.gif")
        self.pausephoto=PhotoImage(file="../icons/pause.gif")
        self.playphoto=PhotoImage(file='../icons/play.gif')
        self.recordphoto=PhotoImage(file="../icons/record.gif")

        recordbtn = Button(controlFrame,image=self.recordphoto,width=60,height=60)
        playbtn = Button(controlFrame,image=self.playphoto,width=60,height=60)
        pausebtn = Button(controlFrame,image=self.pausephoto,width=60,height=60)
        stopbtn = Button(controlFrame,image=self.stopphoto,width=60,height=60)

        recordbtn.pack(side=LEFT)
        playbtn.pack(side=LEFT)
        pausebtn.pack(side=LEFT)
        stopbtn.pack(side=RIGHT)

        ## Workspace DatasetList ##

        self.datasetlistframe = Frame(self.parent,relief=GROOVE)

        datasetlbl = Label(self.datasetlistframe, text='Current Working Datasets')
        datasetlbl.pack(side=TOP)

        scrollbar = Scrollbar(self.datasetlistframe,orient="vertical")

        self.mydatsetlist = Listbox(self.datasetlistframe, yscrollcommand = scrollbar.set, width=30, height=20 )
        self.mydatsetlist.pack(side=RIGHT)

        scrollbar.pack(side=LEFT,fill=Y)
        scrollbar.config(command=self.mydatsetlist.yview)

        self.mydatsetlist.insert(0,'Default Dataset')

        ## Main Sample View ##
        self.mainViewFrame = LabelFrame(self.parent,width=800,height=600,text='Input Sample')
        w = Canvas(self.mainViewFrame, width=800,height=600)


        ## LAYOUT ##
        self.mainViewFrame.grid(column=0,row=1, rowspan=10, padx=20 )
        w.grid(column=0,row=1)

        controlFrame.grid(column=20,row=1)
        self.datasetlistframe.grid(column=20,row=2, columnspan=10)




    ## DATASET MANAGER GUI ##

    def DatasetManager(self):
        self.class_to_be_added = {}
        self.class_path_list = []
        self.class_name_list = []
        self.class_name_list = []

        self.window = Toplevel(self.parent)
        self.window.title('DatasetManager')
        self.window.geometry("450x400")
        self.window.lift(self.parent)

        ## LABELS ##

        header = Label(self.window, text="Dataset Manager")
        label2 = Label(self.window, text='Enter class Name')
        label1 = Label(self.window, text='Enter dataset name')

        ## ENTRYS ##

        dataset_name = StringVar()
        self.dataset_name_entry = Entry(self.window, textvariable=dataset_name)

        class_name = StringVar()
        self.name_entry = Entry(self.window, textvariable=class_name)

        dir_string = StringVar()
        self.dir_entry = Entry(self.window, textvariable=dir_string)

        ## LIST ##

        listframe = Frame(self.window, relief=GROOVE)

        scrollbar = Scrollbar(listframe,orient="vertical")

        self.mylist = Listbox(listframe, yscrollcommand = scrollbar.set )
        self.mylist.pack(side=RIGHT)

        scrollbar.pack(side=LEFT,fill=Y)
        scrollbar.config(command=self.mylist.yview)

        ## BUTTONS ##

        importButton = Button(self.window, text='import folder',command=self.onOpenDir)

        commitClassButton = Button(self.window, text='commit class',command=self.onCommit)

        finishButton = Button(self.window, text='Finish',command=self.onFinish)

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

            self.name_entry.delete(0,END)
            self.dir_entry.delete(0,END)


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

