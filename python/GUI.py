__author__ = 'nimrodshn'
from Tkinter import *
import tkMessageBox
import tkFileDialog
import PIL.Image
import cv2
from dataSetOrginizer import datasetOrginizer

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

        ## Main Sample View ##
        self.mainViewFrame = Canvas(self,width=500,height=500)


        ## DatasetList ##
        self.datasetlistframe = Frame(self,relief=GROOVE)

        datasetlbl = Label(self.datasetlistframe, text='Dataset List')
        datasetlbl.pack(side=TOP)

        scrollbar = Scrollbar(self.datasetlistframe,orient="vertical")

        self.mydatsetlist = Listbox(self.datasetlistframe, yscrollcommand = scrollbar.set )
        self.mydatsetlist.pack(side=RIGHT)

        scrollbar.pack(side=LEFT,fill=Y)
        scrollbar.config(command=self.mydatsetlist.yview)

        ## LAYOUT ##
        self.datasetlistframe.grid(column=1,row=0)
        self.mainViewFrame.grid(column=2,row=0, rowspan=10)

    def update_image(self,image_label, cv_capture):
        cv_image = cv_capture.read()[1]
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        tk_image = PhotoImage(image=pil_image)
        image_label.configure(image=tk_image)
        image_label._image_cache = tk_image  # avoid garbage collection

    def update_all(self, image_label, cv_capture):
        if self.parent.quit_flag:
            self.parent.destroy()  # this avoids the update event being in limbo
        else:
            self.update_image(image_label, cv_capture)
            self.parent.after(10, func=lambda: self.update_all(self, image_label, cv_capture))

    def DatasetManager(self):
        self.class_to_be_added = {}
        self.class_path_list = []
        self.class_name_list = []
        self.class_name_list = []

        window = Toplevel(self)
        window.title('DatasetManager')
        window.geometry("450x400")
        window.lift()

        ## LABELS ##

        header = Label(window, text="Dataset Manager")
        label2 = Label(window, text='Enter class Name')
        label1 = Label(window, text='Enter dataset name')

        ## ENTRYS ##
        dataset_name = StringVar()
        self.dataset_name_entry = Entry(window, textvariable=dataset_name)

        class_name = StringVar()
        self.name_entry = Entry(window, textvariable=class_name)

        dir_string = StringVar()
        self.dir_entry = Entry(window, textvariable=dir_string)

        ## LIST ##

        listframe = Frame(window, relief=GROOVE)

        scrollbar = Scrollbar(listframe,orient="vertical")

        self.mylist = Listbox(listframe, yscrollcommand = scrollbar.set )
        self.mylist.pack(side=RIGHT)

        scrollbar.pack(side=LEFT,fill=Y)
        scrollbar.config(command=self.mylist.yview)

        ## BUTTONS ##

        importButton = Button(window, text='import folder',command=self.onOpenDir)

        commitClassButton = Button(window, text='commit class',command=self.onCommit)

        finishButton = Button(window, text='Finish',command=self.onFinish)

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
            '''
            d = [m.start() for m in re.finditer("/",dir)]
            c=1
            for i in d:
                dir = dir[:i+c] + '/'+ dir[i+c:]
                c=c+1
            '''
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


    def OpenExistingDataset(self):
        dlg = tkFileDialog.askopenfilename(parent=self.parent, initialdir='/home/', title='Select Dataset',
                                           filetypes=[('nps', '.npz')])
        filename = dlg
        print filename


    def onError(self):
        tkMessageBox.showinfo("Error", "Please Choose Image Directory")

