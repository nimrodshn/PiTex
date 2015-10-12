__author__ = 'nimrodshn'
import Tkinter as tk
import tkMessageBox
from PIL import Image, ImageTk
import cv2
from Classifier import Classifier
from DatasetManagerGUI import DatasetManagerGUI

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

        self.parent.title("PiTex 0.1 Alpha")

        ## Menu Bar ##
        
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)
        self.current_frame = None
        self.class_name_list = None
        self.class_path_list = None
        self.current_frame = None
        self.class_to_be_added = None
        self.mainViewFrame = None
        self.pausephoto = None
        self.playphoto = None
        self.recordphoto = None
        self.stopphoto = None

        self.dsManagerGUI = DatasetManagerGUI(self)

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
        datasetMenu.add_command(label="New Dataset", command=self.dsManagerGUI.initUI)
        datasetMenu.add_command(label="Open Existing", command=self.dsManagerGUI.OpenExistingDataset)

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
        playbtn = tk.Button(controlFrame,image=self.playphoto,width=60,height=60,command=self.captureFrameforAnalysis)
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

        self.mydatasetlist = tk.Listbox(self.datasetlistframe, yscrollcommand = scrollbar.set, width=30, height=20 )
        self.mydatasetlist.pack(side=tk.RIGHT)

        scrollbar.pack(side=tk.LEFT,fill=tk.Y)
        scrollbar.config(command=self.mydatasetlist.yview)

        self.mydatasetlist.insert(0,'Default')

        ## Main Sample View ##
        self.mainViewFrame = tk.LabelFrame(self.parent,width=800,height=600,text='Input Sample')
        self.image_label = tk.Label(master=self.mainViewFrame)
        self.image_label.pack()

        self.cam = cv2.VideoCapture(0) # Camera

        quit_button = tk.Button(master=self.mainViewFrame, text='Quit',command=lambda: self.quit_(self.parent))
        quit_button.pack()

        # setup the update callback
        self.parent.after(0, func=lambda: self.update_all(self.parent, self.image_label, self.cam))

        ## Main GUI LAYOUT ##
        self.mainViewFrame.grid(column=0,row=1, rowspan=10, padx=20 )
        controlFrame.grid(column=20,row=1)
        self.datasetlistframe.grid(column=20,row=2, columnspan=10)

    ### Control Panel Functions ###

    def quit_(self,root):
        self.cam.release()
        root.destroy()

    def update_image(self,root,image_label, cam):
        (readsuccessful, f) = self.cam.read()
        if readsuccessful:
            self.current_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            a = Image.fromarray(self.current_frame)
            b = ImageTk.PhotoImage(image=a)
            image_label.configure(image=b)
            image_label._image_cache = b  # avoid garbage collection
            root.update()
        else: tkMessageBox("Camera Not Connected")

    def update_all(self,root, image_label, cam):
        self.update_image(root,image_label, cam)
        root.after(20, func=lambda: self.update_all(root, image_label, cam))

    def captureFrameforAnalysis(self):
        try:
            self.mydatasetlist.get(self.mydatasetlist.curselection())
            datasetName = self.mydatasetlist.get(self.mydatasetlist.curselection())
            dataset_path = "binData/"+datasetName+".npz"
            print dataset_path

            img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            cv2.namedWindow("CurrentFrame",cv2.WINDOW_NORMAL)
            cv2.imshow("CurrentFrame",img)

            cl = Classifier(img)
            cl.classifieSample(dataset_path)
        except:
            tkMessageBox.showerror("Error","Please pick a Dataset")






