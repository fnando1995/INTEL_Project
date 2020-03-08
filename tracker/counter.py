import numpy as np
import cv2
from tracker.counter_logic import cl_try_to_count,cl_set_counter_data
from tracker.utils import read_yml

class Counter():
    def __init__(self,yml_filepath):
        datadict            =   read_yml(yml_filepath)
        self.objects_to_detect,\
        self.counter_data,\
        self.all_rel, \
        self.relations_per_object, \
        self.max_rel_long    =   self.set_counter_data(datadict)

    def set_counter_data(self,datadict):
        return cl_set_counter_data(datadict)

    def count(self,trk):
        return cl_try_to_count(trk,self.relations_per_object,self.counter_data)



    def saveDataInfo(self,filepath):
        data        =   self.get_data_as_string()
        file=open(filepath,"w")
        file.writelines(data)
        file.close()

    def get_data_as_string(self):
        string=""
        for object in self.objects_to_detect:                 #ya estan ordenados sorted
            for relation in self.relations_per_object[object]:#ya estan ordenados sorted
                string+=str(self.counter_data[object][relation])+","
        return string[:-1] #eliminando la ultima coma

    # def set_data_from_list(self,values):
    #     i=0
    #     for object in self.objects_to_detect:
    #         for relation in self.relations_per_object[object]:
    #             self.counter_data[object][relation] = int(values[i])
    #             i+=1

    #Para visualización en la ventana si se levanta con SHOW_CAMERA = TRUE

    def generateLineForDebug(self,obj):
        if obj is not None:
            line = str(obj).ljust(self.max_rel_long+2," ")
            for relation in self.all_rel:
                if relation in self.relations_per_object[obj]:
                    line+= str(self.counter_data[obj][relation]).rjust(self.max_rel_long+2," ")
                else:
                    line+= "---".rjust(self.max_rel_long+2," ")
            return line
        else:
            line = "obj/rel".ljust(self.max_rel_long+2," ")
            for relation in self.all_rel:
                line += relation[-3:].rjust(self.max_rel_long+2," ")
            return line

    def PutDataInFrame(self,frame):
        extra = 18 + (18*len(self.objects_to_detect))
        h,w,c = frame.shape
        eframe = np.ones((h+extra,w,c),np.uint8)
        eframe[extra:,:]=frame[:,:]
        cv2.putText(eframe, self.generateLineForDebug(None) ,   (0, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        for i,obj in enumerate(self.objects_to_detect):
            cv2.putText(eframe, self.generateLineForDebug(obj), (0, 15*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        return eframe
