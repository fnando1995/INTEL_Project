from tracker.polygon import *

class Region(object):
    def __init__(self,label,pts):
        self.label              =   label
        self.regionPolygon      =   Polygon(pts)
    def getRegionPolygon(self):
        return self.regionPolygon
    def getRegionLabel(self):
        return self.label