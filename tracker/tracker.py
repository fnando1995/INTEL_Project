from modules.trackers.sort import Sort
from modules.regions_controllers.region_controller import RegionsController
from modules.monitors.monitor import Monitor
import cv2
import numpy as np

def put_tracked_in_frame(tracked,frame):
    # b=False
    for trk in tracked:
        bbox = trk.get_state()[0]
        startX, startY,endX, endY = np.array(bbox[:4]).astype(int)
        # if trk.get_path()[-1] is None:
        #     cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #     b=True
        # else:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
    return frame # ,b

class Tracker(object):
    def __init__(self,figure_filepath,monitor_datadict,CLASSES_TO_BE_DETECTED):
        self.algorithm          =   Sort()
        self.tracked_detections =   []
        self.regions_controller =   RegionsController(figure_filepath)
        self.monitor            =   Monitor(monitor_datadict,CLASSES_TO_BE_DETECTED)


    def track_dets(self,dets_list,data_for_time,filepath=None,show=False,sampling=1):
        """
        data for time is (init_time of video, end_time of video, video__FPS,MINUTES_OF_VIDEO)
        :param dets_list:
        :param data_for_time:
        :param video:
        :param sampling:
        :return:
        """

        if show:
            video = cv2.VideoCapture(filepath)

        for index,dets in enumerate(dets_list):
            self.tracked_detections,to_check_trks = self.algorithm.update(dets,self.tracked_detections,self.regions_controller)
            self.check_trks_paths(to_check_trks,data_for_time,sampling,filepath)

            if show:
                for s in range(sampling):
                    _, f = video.read()
                # f,b=
                cv2.imshow("tracked", cv2.resize(put_tracked_in_frame(self.tracked_detections, f), (500, 500)))
                # if b :
                #     cv2.waitKey(0)
                # else:
                cv2.waitKey(1)


        if show: cv2.destroyAllWindows()


    def check_trks_paths(self,list_to_check_trks,data_for_time,sampling,filepath):
        """
        Esta funcion se encargará de verificar los paths de los objetos
        trackeados. En caso de que alguno presente la generacion de un evento en su recorrido
        se realizara un insert a la tabla de la base de dato respectiva.
        :param list_to_check_trks:
        :return:
        """
        for trk in list_to_check_trks:
            # Mandar a ver si ocurrio un evento
            self.monitor.verify(trk,data_for_time,sampling,filepath)
