def PointInPolygon(polygon, point):
    punto = Point([-999999999, point.y])
    testline_left = Segment(punto, point)
    testline_right = Segment(point, punto)
    count_left = 0
    count_right = 0
    for e in polygon.GetEdges():
        if EdgesIntersect(testline_left, e):
            count_left += 1
        if EdgesIntersect(testline_right, e):
            count_right += 1
    if count_left % 2 == 0 and count_right % 2 == 0:
        return False
    else:
        return True


def EdgesIntersect(e1, e2):
    a = e1.p1
    b = e1.p2
    c = e2.p1
    d = e2.p2

    cmp = Point([c.x - a.x, c.y - a.y])
    r = Point([b.x - a.x, b.y - a.y])
    s = Point([d.x - c.x, d.y - c.y])

    cmpxr = cmp.x * r.y - cmp.y * r.x
    cmpxs = cmp.x * s.y - cmp.y * s.x
    rxs = r.x * s.y - r.y * s.x

    if cmpxr == 0:
        return (c.x - a.x < 0) != (c.x - b.x < 0)
    if rxs == 0:
        return False

    rxsr = 1 / rxs
    t = cmpxs * rxsr
    u = cmpxr * rxsr

    return t >= 0 and t <= 1 and u >= 0 and u <= 1


class Point:
    x = None
    y = None

    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]

    def __str__(self):
        return '[X:' + str(self.x) + ' Y:' + str(self.y) + ']'

    def getXY(self):
        return self.x, self.y


class Segment:
    p1 = None
    p2 = None

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class Polygon:
    points = None

    def __init__(self, p):
        if p is not None:
            self.points = [Point(i) for i in p]
        else:
            self.points = []

    def AddPoint(self, p):
        self.points.append(p)

    def getPointsAsXY(self):
        if self.points is None:
            return None
        else:
            return [p.getXY() for p in self.points]

    def GetEdges(self):
        edges = []
        for i in range(len(self.points)):
            if i == len(self.points) - 1:
                i2 = 0
            else:
                i2 = i + 1
            edges.append(Segment(self.points[i], self.points[i2]))
        return edges

    def __str__(self):
        string = ''
        for i in self.points:
            string += str(i)
        return string