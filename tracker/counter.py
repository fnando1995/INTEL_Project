import numpy as np
import cv2
from modules.constantes.directories import FIGURE_TO_USE_FILE,FILE_DATA_PATH_TODAY
from modules.counters.counting_logic import cl_try_to_count,cl_set_counter_data


class Counter():
    def __init__(self):
        """
        datadict => Diccionario propio del archivo np donde se definen por el usuario
                    las distintas relaciones y parametros de las relaciones que seran
                    de interes
        objects_to_detect =>    listado ordenado  del "nombre" de los objetos de interes
        counter_data =>         diccionario que mantiene la jerarquia
                                    {'object_name':{'relation_full_name_1':int
                                                    ,'relation_full_name_2':int},
                                    ...
                                    }
                                esta jerarquia se usa para poder contar cada uno de las
                                relaciones de cada objeto.
        all_rell =>             Listado ordenado del "relation_full_name" de la relaciones
                                de todos los objetos
        relations_per_object => Diccionario que mantiene la jerarquia
                                    {'object_name':[relation_full_name,...]}
                                El listado interno (value) de cada objeto esta ordenado.
        max_rel_long   =>       Esta variable es int y mantiene el valor de la longuitud
                                del texto del nombre de la relacion mas larga:
                                        max([len(r) for r in all_rell])
        """
        datadict                                                =   np.load(FIGURE_TO_USE_FILE,allow_pickle=True)[2]
        self.objects_to_detect,\
        self.counter_data,\
        self.all_rel, \
        self.relations_per_object, \
        self.max_rel_long    =   self.set_counter_data(datadict)

    def set_counter_data(self,datadict):
        """
        Esta funcion retorna todos los atributos necesarios en la inicializacion de un contador.
        Estos atributos son generados en otro archivo (funcion) para abstraer logica de creacion.
        """
        return cl_set_counter_data(datadict)

    def count(self,trk):
        """
        Esta funcion realiza la logica de conteo para un tracker. Este tracker es ingresado
        como argumento. La idea de funcionamiento es que el tracker viene con una "path" que
        simboliza las regiones por donde ha pasado en su vida (mientras existio). Una vez que
        se "pierde" primero para por esta funcion para saber si su path ha sido objeto de uno
        de los conteos señalados en el archivo de figura, donde se encuentra que relaciones y
        parametros se desea contabilizar.
        """
        return cl_try_to_count(trk,self.relations_per_object,self.counter_data)



    def saveDataInfo(self,fecha,inicio_hora,fin_hora):
        """
        Esta funcion realiza una actualizacion en el archivo "FILE_DATA_PATH_TODAY", mismo que
        posee las lineas que guardan los conteos de todos los objetos.
        En caso de que la hora de inicio y fin no concuerden con la ultima linea, significa que
        es una nueva linea de un nuevo horario, por lo que se debe crear una nueva linea.
        NOTA: Esto no es optimo, pero sirve como almacenamiento de datos
        :param fecha:       as datetime.date()
        :param inicio_hora: as datetime.time()
        :param fin_hora:    as datetime.time()
        :return:            Nothing
        """
        data        =   self.get_data_as_string()
        file        =   open(FILE_DATA_PATH_TODAY)
        lines       =   file.readlines()
        file.close()
        A_F, A_IH, A_FH = str(fecha), str(inicio_hora), str(fin_hora)
        if len(lines)==0:
            lines.append(A_F + "," + A_IH + "," + A_FH + "," + data + "\n")
        F,IH,FH     =   lines[-1].split(",")[:3]
        if F==A_F and IH == A_IH and FH == A_FH:
            "Se esta guardando la actualizacion de un ciclo"
            lines[-1]  =  A_F + "," + A_IH + "," + A_FH + "," + data + "\n"
        else:
            lines.append(A_F + "," + A_IH + "," + A_FH + "," + data + "\n")
        file=open(FILE_DATA_PATH_TODAY,"w")
        file.writelines(lines)
        file.close()

    def get_data_as_string(self):
        string=""
        for object in self.objects_to_detect:                 #ya estan ordenados sorted
            for relation in self.relations_per_object[object]:#ya estan ordenados sorted
                string+=str(self.counter_data[object][relation])+","
        return string[:-1] #eliminando la ultima coma

    def set_data_from_list(self,values):
        i=0
        for object in self.objects_to_detect:
            for relation in self.relations_per_object[object]:
                self.counter_data[object][relation] = int(values[i])
                i+=1

    #Para visualización en la ventana si se levanta con SHOW_CAMERA = TRUE
    def generateLineForDebug(self,obj):
        """
        Esta funcion genera una linea de texto serializada de tal manera que pueda ser bien
        visualizada en la ventana que se levanta si SHOW_CAMERA = TRUE.
        Aqui se contempla el caso de la cabecera, usando obj = None.
        """
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
                line += relation.rjust(self.max_rel_long+2," ")
            return line

    def PutDataInFrame(self,frame):
        """
        Esta funcion se encarga de generar y anadir una franja en la parte superior del frame el cual
        muestra los actuales valores de conteo de las relaciones con sus parametros.
        """
        extra = 18 + (18*len(self.objects_to_detect))
        h,w,c = frame.shape
        eframe = np.ones((h+extra,w,c),np.uint8)
        eframe[extra:,:]=frame[:,:]
        cv2.putText(eframe, self.generateLineForDebug(None) ,   (0, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        for i,obj in enumerate(self.objects_to_detect):
            cv2.putText(eframe, self.generateLineForDebug(obj), (0, 15*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
        return eframe
