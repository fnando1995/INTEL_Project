from tracker.region import *
import numpy as np

class RegionsController(object):

    def __init__(self,figure_filepath):
        NP_Regions = list(np.load(figure_filepath, allow_pickle=True))
        self.Regions_names = list(NP_Regions[1])
        self.Regions = [Region(str(self.Regions_names[i]), r) for i, r in enumerate(NP_Regions[0])]

    def getRegionsControllerData(self):
        return self.Regions,self.Regions_names

    def getRegions(self):
        return self.Regions

    def whereIsPoint(self,point):
        for r in self.Regions:
            if PointInPolygon(r.getRegionPolygon(),point):
                return r.getRegionLabel()
        return None