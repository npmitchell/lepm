import numpy as np
import random

'''
Classes and definitions for geometric computations
NPM 2016, some adapted from https://github.com/isterin/geo-utils
and stack exchange discussions
'''

##########################################
## Geometry Classes
##########################################
class Point(object):
    """Point class represents a point on a cartesian plane"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def to_list(self):
        return [self.x, self.y]
    
    def to_array(self):
        return np.array([self.x, self.y])
    
    def __hash__(self):
        """docstring for __hash__"""
        return self.x.hash() + self.y.hash()
    
    def __eq__(self, other):
        """docstring for __eq__"""
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        """docstring for __ne__"""
        return not self.__eq__(other)
    
    def __str__(self):
        return  "%s (x=%f, y=%f)" % (self.__class__.__name__, self.x, self.y)
    
class BoundingBox(object):
    def __init__(self, *points):
        """docstring for __init__"""
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        for p in points:
            if p.x < xmin: xmin = p.x
            if p.y < ymin: ymin = p.y
            if p.x > xmax: xmax = p.x
            if p.y > ymax: ymax = p.y
        self.interval_x = Interval(xmin, xmax)
        self.interval_y = Interval(ymin, ymax)
 
    def random_point(self):
        x = self.interval_x.random_point()
        y = self.interval_y.random_point()
        return Point(x,y)

class Polygon:
    '''Polygon class represents a set of points on a cartesian plane which form a polygon
    '''
    def __init__(self, points):
        #super(Polygon, self).__init__()
        self.points = list()
        for p in points:
            if isinstance(p, Point):
                self.points.append(p)
            elif isinstance(p, np.ndarray):
                #print 'p[0]= ', p[0]
                self.points.append(Point(p[0],p[1]))    
            elif isinstance(p, dict):
                self.points.append( 
                        Point(x=p.get('x', p['lon']), y=p.get('y', p['lat'])) )
            elif isinstance(p, (list, tuple)):
                self.points.append(Point(p[0],p[1]))
            else:
                raise TypeError("Points must be provided in a form of a Point class, dictionary, list, " \
                                "or tuple instances. You've provided %s instance." % p.__class__)
        if len(self.points) < 3 or self.points[0] != self.points[len(self.points)-1]:
            raise ValueError("The points you provided do not define a polygon.  " \
                             "The first and last points must be the same.")
    
    def contains(self, point):
        seg_counter = SegmentCounter(point)
        for i in range(1, len(self.points)):
            line = Line(*self.points[i-1:i+1])
            if seg_counter.process_segment(line):
                return True
        return seg_counter.crossings % 2 == 1
 
    def random_point(self):
        bb = BoundingBox(*self.points)
        while True:
            print("GENERATING RANDOM POINT...")
            p = bb.random_point()
            if self.contains(p):
                return p
            
class Line(object):
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
    
    def is_on_point(self, point):
        p, p1, p2 = point, self.point1, self.point2
        if p.x == p2.x and p.y == p2.y:
            return True
        
        if p1.y == p.y and p2.y == p.y:
            minx = p1.x
            maxx = p2.x
            if minx > maxx:
                minx = p2.x
                maxx = p1.x
            if p.x >= minx and p.x <= maxx:
                return True

        return False
    
class Interval(object):
    """docstring for Interval"""
    def __init__(self, x1, x2):
        self.min = min(x1, x2)
        self.max = max(x1, x2)
        
    def random_point(self):
        return random.uniform(self.min, self.max)
    
class SegmentCounter(object):
    '''For testing points in polygons, pass a point to get segment counter
    '''
    def __init__(self, point):
        self.point = point
        self.crossings = 0
 
    def process_segment(self, line):
        p, p1, p2 = self.point, line.point1, line.point2
        # print 'p = ', p
        # print 'p1 = ', p1
        # print 'p2 = ', p2
        if p1.x < p.x and p2.x < p.x:
            return False
 
        if (p.x == p2.x and p.y == p2.y):
            return True
 
        if p1.y == p.y and p2.y == p.y:
            minx = p1.x
            maxx = p2.x
            if minx > maxx:
                minx = p2.x
                maxx = p1.x
            if p.x >= minx and p.x <= maxx:
                return True
            return False
 
 
        if ((p1.y > p.y) and (p2.y <= p.y)) \
                or ((p2.y > p.y) and (p1.y <= p.y)):
            x1 = p1.x - p.x
            y1 = p1.y - p.y
            x2 = p2.x - p.x
            y2 = p2.y - p.y
 
            det = np.linalg.det([[x1, y1], [x2, y2]])
            if det == 0.0:
                return True
            if y2 < y1:
                det = -det
 
            if det > 0.0:
                self.crossings += 1
      
