from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])  # convert_bbox_to_z(bbox)


        self.time_since_update = 0
        self.history            = []
        self.hits               = 0
        self.hit_streak         = 0
        self.age                = 0
        self.objConfidence      = bbox[4]
        self.objclass           = bbox[5]
        self.Region_path        = []
        self.Frame_count_path   = []
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1


    """
    Updates the state vector with observed bbox.
    """

    def update(self, bbox):

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    """
    Advances the state vector and returns the predicted bounding box estimate.
    """

    def predict(self):
        """
        Realiza la prediccion de la siguiente posicion del filtro.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        retorna la caja de deteccion estimada.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_point_of_interest(self):
        """
        Retorna el centroide de la deteccion, la cual se define
        como el punto de interes.
        """
        return [round(float(i[0]), 2) for i in self.kf.x[:2]]

    def get_last_region(self):
        return self.Region_path[-1]

    def get_path(self):
        return self.Region_path

    def get_frame_count(self):
        return self.Frame_count_path

    def get_id(self):
        return self.id

    def set_path(self,path):
        self.Region_path = path

    def set_frame_count(self,frame_count):
        self.Frame_count_path = frame_count

    def add_regions_name_to_path(self,region_name):
        self.Region_path.append(region_name)

    def eliminate_None_from_path_and_frame_count(self):
        path = []
        frames = []
        for (p,f) in zip(self.get_path(),self.get_frame_count()):
            if not p is None:
                path.append(p)
                frames.append(f)
        self.set_path(path)
        self.set_frame_count(frames)


    def print(self):
        print("###########################")
        print("state:",             self.get_state())
        print("path:",              self.get_path())
        print("fcount:",            self.get_path())
        print("hits:",              self.hits)
        print("hit_streak:",        self.hit_streak)
        print("age:",               self.age)
        print("time_since_update:", self.time_since_update)
        print("###########################")