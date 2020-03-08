from tracker.sort import Sort
from tracker.region_controller import RegionsController
# from tracker.counter import Counter


import cv2
import numpy as np

class Tracker(object):
    def __init__(self,figure_filepath):
        self.algorithm          =   Sort()
        self.tracked_detections =   []
        self.regions_controller =   RegionsController(figure_filepath)
        # self.counter            =   Counter(monitor_datadict,CLASSES_TO_BE_DETECTED)


    def track_dets(self,dets):
        self.tracked_detections,to_check_trks = self.algorithm.update(dets,self.tracked_detections,self.regions_controller)
        # self.check_trks_paths(to_check_trks)
        return self.tracked_detections



    #
    # def check_trks_paths(self,list_to_check_trks):
    #     for trk in list_to_check_trks:
    #         self.counter.verify(trk)
