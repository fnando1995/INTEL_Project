from tracker.sort import Sort
from tracker.region_controller import RegionsController
from tracker.counter import Counter


import cv2
import numpy as np

class Tracker(object):
    def __init__(self,figure_filepath):
        self.algorithm          =   Sort()
        self.tracked_detections =   []
        self.regions_controller =   RegionsController(figure_filepath)
        self.counter            =   Counter(figure_filepath.replace(".npy",".yml"))


    def track_dets(self,dets):
        self.tracked_detections,to_check_trks = self.algorithm.update(dets,self.tracked_detections,self.regions_controller)
        self.check_trks_paths(to_check_trks)
        return self.tracked_detections


    def put_tracking_in_frame_with_regions_with_couting(self,frame):
        def put_tracked_in_frame(frame,tracked):
            for trk in tracked:
                bbox = trk.get_state()[0]
                id = str(trk.get_id())
                startX, startY, endX, endY = np.array(bbox[:4]).astype(int)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)
                cv2.putText(frame, id, (startX, startY), 0, 1, (0, 255, 255), 3)
                centroide = (int(startX + ((endX - startX) / 2)), int(startY + ((endY - startY) / 2)))
                cv2.circle(frame, centroide, 6, (0,255,255), thickness=-1)
            return frame
        def put_overlay_regions(frame,reg_controller):
            alpha=0.5
            overlay = frame.copy()
            regiones, nombres_regiones = reg_controller.getRegionsControllerData()
            for i, region in enumerate(regiones):
                region = np.array(region.regionPolygon.getPointsAsXY(), np.int32)
                region = region.reshape((-1, 1, 2))
                cv2.polylines(overlay, [region], True, (0, 255, 255), thickness=3)
                region_b = region[0].copy()
                esquina = list(region_b[0])
                esquina[1] = int(esquina[1]) + 20
                esquina[0] = int(esquina[0])
                cv2.putText(frame, str(nombres_regiones[i]), tuple(esquina), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255],2)  # ROJO
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            return frame
        return self.counter.PutDataInFrame(put_overlay_regions(put_tracked_in_frame(frame,self.tracked_detections),self.regions_controller))

    def check_trks_paths(self,list_to_check_trks):
        for trk in list_to_check_trks:
            self.counter.count(trk)

    def save_data(self,filepath):
        self.counter.saveDataInfo(filepath)