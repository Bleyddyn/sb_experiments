"""
This is based on Open AI's rendering code, replacing pyglet with pygame.
For an image based LunarLander I needed to be able to draw to an off-screen surface,
which pyglet doesn't support.
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym.utils import reraise
from gym import error
import pygame

import math
import numpy as np

""" Methods called from LunarLander
        self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H, visible=False)
        self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
        return self.viewer.get_array(), reward, done, {"state":np.array(state, dtype=np.float32)}
            self.viewer.window.activate()
            self.viewer.draw_polygon(p, color=(0,0,0))
            self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
            self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
            self.viewer.draw_polygon(path, color=obj.color1)
            self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

Offscreen rendering with pygame: https://stackoverflow.com/questions/21441217/how-to-draw-to-an-off-screen-display-in-pygame
Numpy array from pygame: https://stackoverflow.com/questions/34673424/how-to-get-numpy-array-of-rgb-colors-from-pygame-surface
    imgdata = pygame.surfarray.array3d(img)
    imgdata.swapaxes(0,1)
"""

RAD2DEG = 57.29577951308232

pygame.init()

class Viewer(object):
    def __init__(self, width, height, scale, display=None, visible=True):
        self.width = width
        self.height = height
        self.scale = scale
        self.window = None
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []

        self.image = pygame.Surface( (width, height) )
        self.image.fill( pygame.Color(255,255,255) )

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def draw(self):
        ret = self.image.fill( pygame.Color(255,255,255) )
        for geom in self.geoms:
            geom.render(self.image)
        for geom in self.onetime_geoms:
            geom.render(self.image)
        arr = self.get_array()
        self.onetime_geoms = []
        return arr

    def render(self, return_rgb_array=False):
        if not self.window:
            self.window = pygame.display.set_mode([self.width,self.height])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.window.fill( (60,60,60) )
        self.window.blit( pygame.transform.flip(self.image,False,True), (0,0) )
        pygame.display.flip()
        pygame.display.update()
        arr = None
        if return_rgb_array:
            arr = self.get_array()
        return arr if return_rgb_array else self.isopen

    # Convenience
    def draw_circle(self, center, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(center=center, radius=radius, res=res, filled=filled, scale=self.scale)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled, scale=self.scale)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v, scale=self.scale)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end, scale=self.scale)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        imgdata = pygame.surfarray.array3d(self.image)
        #imgdata = imgdata.swapaxes(0,1)
        imgdata = np.transpose(imgdata, (1,0,2) )
        imgdata = np.flip(imgdata, axis=0 )
        return imgdata

    def __del__(self):
        self.close()

def make_circle(center, radius=10, res=30, filled=True, scale=1):
    return Circle(radius, center, filled, scale)

def make_polygon(v, filled=True, scale=1):
    if filled: return FilledPolygon(v, scale=scale)
    else: return PolyLine(v, True, scale=scale)

def make_polyline(v, scale=1):
    return PolyLine(v, False, scale=scale)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=(0, 0, 0, 255)
        self._linewidth = 1
        self.attrs = [self._color]
        self.filled = False
    def render(self, image):
        self.render1(image)
    def render1(self, image):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b):
        self._color = np.clip( [int(255.0*r), int(255.0*g), int(255.0*b), 255], 0, 255 )
    def set_linewidth(self, w):
        self._linewidth = w
    def linewidth(self):
        if self.filled:
            return 0
        return self._linewidth

class Point(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.center = v
        self.radius = 1.0
        self.filled = True
    def render1(self,image):
        pygame.draw.circle(image, self._color, self.center, self.radius, self.linewidth())

class Circle(Geom):
    def __init__(self, r, c, filled=True, scale=1):
        Geom.__init__(self)
        #print( "c/c/r/w: {} {} {} {}".format( self._color, c, r, self.linewidth() ) )
        self.radius = max(1,round(r*scale))
        self.center = (round(c[0]*scale), round(c[1]*scale) )
        self.filled = filled
        self._linewidth = 1
        self._color = (255,255,255,255)
    def render1(self, image):
        try:
            pygame.draw.circle(image, self._color, self.center, self.radius, self.linewidth())
        except TypeError as e:
            print( "c/c/r/w: {} {} {} {}".format( self._color, self.center, self.radius, self.linewidth() ) )
            raise e
    def set_linewidth(self,w):
        if w > self.radius:
            w = self.radius
        super(Circle, self).set_linewidth(w)

class FilledPolygon(Geom):
    def __init__(self, v, scale=1):
        Geom.__init__(self)
        self.v = np.array(v) * scale
        self.filled = True
    def render1(self, image):
        pygame.draw.polygon(image, self._color, self.v, self.linewidth())

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close, scale=1):
        Geom.__init__(self)
        self.v = np.array(v) * scale
        self.close = close
        self._linewidth = 1
        self.filled = False
    def render1(self,image):
        pygame.draw.lines(image, self._color, self.close, self.v, self.linewidth())

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), scale=1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self._linewidth = 2
        self.filled = False
    def render1(self,image):
        pygame.draw.line(image, self_color, self.start, self.end, self.linewidth())

class Image(Geom):
    """ This class has not been tested. """
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pygame.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self,image):
        dest = (0,0), # (-self.width/2, -self.height/2, width=self.width, height=self.height)
        img = self.img
        if self.flip:
            img = pygame.transform.flip(img,False,True)
        image.blit( img, dest )

