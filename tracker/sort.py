from __future__ import print_function
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
# because of scipy 1.2.1 was updated to 1.3.0 - try lapsolver
import warnings

warnings.filterwarnings('ignore')
import modules.regions_controllers.polygon as polygon
from modules.trackers.kalman_filters import KalmanBoxTracker


def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    Relacion porcentual del Area de intercepcion sobre
    el area de union de los dos boxes. valores entre [0,1]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Hungarian algorithm
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0

    def update(self, dets, tracked_detections, region_controller):
        if dets is None:
            return None
        erased_trackers = []
        dets = np.array(dets)
        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(tracked_detections), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = tracked_detections[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            erased_trackers.append(tracked_detections.pop(t))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(tracked_detections):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d][0][:4])
                actual_region = region_controller.whereIsPoint(polygon.Point(trk.get_point_of_interest()))
                if not trk.get_last_region() == actual_region:
                    trk.add_regions_name_to_path(actual_region)
                    trk.Frame_count_path.append(self.frame_count)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i][:])
            trk.Region_path.append(region_controller.whereIsPoint(polygon.Point(trk.get_point_of_interest())))
            trk.Frame_count_path.append(self.frame_count)
            tracked_detections.append(trk)

        # Evaluate actual trackers. Eliminate trackers that dont fit the parameter time_since_update

        i = len(tracked_detections)
        for trk in reversed(tracked_detections):
            i -= 1
            if (trk.time_since_update > self.max_age):
                erased_trackers.append(tracked_detections.pop(i))

        return tracked_detections, erased_trackers
