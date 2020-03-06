from __future__ import division
from tkinter import *
from PIL import Image,ImageTk
import PIL.Image
import cv2
import numpy as np
try:
    from Tkinter import *
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import *
    from tkinter import filedialog





default_size            =   (500,500)
MAIN_COLORS             =   [(0,0,255),(0,255,0),(255,0,0),(100,100,100),(50,150,100),(10,20,50)]



class LabelTool():
    def put_regions(self):
        alpha = 0.5
        overlay = self.im_np.copy()
        self.im_np_2 = self.im_np.copy()
        for index,arreglo in enumerate(self.regions):
            arreglo = [ (int(A[0])/self.x_multiplier,int(A[1])/self.y_multiplier) for A in arreglo]
            pts = np.array(arreglo, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, MAIN_COLORS[index], thickness=3)
        cv2.addWeighted(overlay, alpha, self.im_np_2, 1 - alpha, 0, self.im_np_2)
    def chg_image(self):
        self.put_regions()
        if self.im_np_2 is None:
            self.im =PIL.Image.fromarray(self.im_np)
        else:
            self.im = PIL.Image.fromarray(self.im_np_2)
        if self.im.mode == "1": # bitmap image
            self.tkimg = PIL.ImageTk.BitmapImage(self.im, foreground="white")
        else:              # photo image
            self.tkimg = PIL.ImageTk.PhotoImage(self.im)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
    def OnMouseDown(self, event):
        if len(self.regions)==0:
            print("Por favor iniciar una nueva region")
            return
        self.regions[-1].append((event.x*self.x_multiplier, event.y*self.y_multiplier))
        #self.regions[-1].append((event.x, event.y ))
        self.update_list_regions()
        self.chg_image()
    def update_list_regions(self):
        if len(self.regions)>0:
            self.listbox.delete(0,END)
            for index,region in enumerate(self.regions):
                #if index==(len(self.regions)-1):
                self.listbox.insert(END, self.regions_names[index]+" -> "+''.join(['(%d, %d)/' %(x,y) for x,y in region]))
                #else:
                #    self.listbox.insert(END, ''.join(['(%d, %d)/' % (x, y) for x, y in region]))
    def open(self):
        filepath = filedialog.askopenfilename()
        self.filepath_to_save = '/'.join(filepath.split('/')[:-1]) + '/'
        self.save_video_name = filepath.split("/")[-1].split(".")[0]+'.npy'
        self.save_label.config(text=self.save_video_name)
        if filepath != "":
            ret,frame=cv2.VideoCapture(filepath).read()
            self.real_size = frame.shape[:2]
            self.x_multiplier = self.real_size[1]/default_size[1]
            self.y_multiplier = self.real_size[0]/default_size[0]
            frame =cv2.resize(frame,default_size)
            self.im_np = frame
            self.im_np_2 =None
        self.chg_image()
    def start_new_region(self):
        names = self.entry.get()
        if names == "":
            print("please insert name for region")
            return
        self.entry.delete(0,END)
        self.regions_names.append(names)
        self.regions.append([])
        self.update_list_regions()
        self.chg_image()
    def del_last_point(self):
        if len(self.regions[-1]) > 0:
            del self.regions[-1][-1]
        else:
            print("La región actual no tiene más puntos")
        self.update_list_regions()
        self.chg_image()
    def del_region(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.listbox.delete(idx)
        del self.regions[idx]
        del self.regions_names[idx]
        self.update_list_regions()
        self.chg_image()
    def save_regions(self):
        np.save(self.filepath_to_save+self.save_video_name,np.array([self.regions,self.regions_names]))
        cv2.imwrite(self.filepath_to_save+self.save_video_name.replace('.npy','_withregions.jpg'),self.im_np_2)


    def __init__(self, master):
        self.parent                 = master
        self.parent.title("Regions Creator")
        self.frame                  = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        self.regions = []
        self.regions_names =[]
        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.button_open_video = Button(self.frame, text="Open Video", command=self.open)
        self.button_open_video.grid(row=0, column=1, sticky=W)

        self.upper_frame = Frame(self.frame)
        self.upper_frame.grid(row=0, column=2, columnspan=2,sticky=W)
        self.regions_control_panel = Frame(self.upper_frame)
        self.regions_control_panel.grid(row=0, column=2, columnspan=2,sticky=W)
        self.button_start_region = Button(self.regions_control_panel, text="start region", command=self.start_new_region)
        self.button_start_region.grid(row=0, column=0, sticky=W)
        self.entry = Entry(self.regions_control_panel)
        self.entry.grid(row=0, column=1, sticky=W)
        self.button_del_last_region = Button(self.regions_control_panel, text="delete selected region", command=self.del_region)
        self.button_del_last_region.grid(row=0, column=2, sticky=W)

        self.button_del_last_point = Button(self.regions_control_panel, text="delete last point", command=self.del_last_point)
        self.button_del_last_point.grid(row=1, column=0, sticky=W)

        self.save_control_panel = Frame(self.upper_frame)
        self.save_control_panel.grid(row=2, column=2, columnspan=3, sticky=W)
        self.button_save_regions = Button(self.save_control_panel, text="save regions", command=self.save_regions)
        self.button_save_regions.grid(row=0, column=0, sticky=W)
        self.save_label =Label(self.save_control_panel, text = 'Abrir un video')
        self.save_label.grid(row=0, column=1, sticky=W)

        self.listbox = Listbox(self.frame, width = 60, height = 12)
        self.listbox.grid(row = 3, column = 2, sticky = N)

        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)
        self.mainPanel.bind("<Button-1>", self.OnMouseDown)


        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 6, column = 1, columnspan = 2, sticky = W+E)
if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
