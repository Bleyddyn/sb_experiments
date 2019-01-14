"""
2D rendering framework
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

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

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

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, width, height, scale, display=None, visible=True):
        display = get_display(display)

        self.width = width
        self.height = height
        self.scale = scale
        self.window = pyglet.window.Window(width=width, height=height, display=display, visible=visible)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        self.image = pygame.Surface( (width, height) )
        self.image.fill( (1,1,1) )

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        self.image.fill( (1,1,1) )
        #self.transform.enable()
        for geom in self.geoms:
            geom.render(self.image)
        for geom in self.onetime_geoms:
            geom.render(self.image)
        #self.transform.disable()
        arr = None
        if return_rgb_array:
            arr = self.get_array()
        self.onetime_geoms = []
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

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

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
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = display
        self.maxwidth = maxwidth
    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = pyglet.window.Window(width=width, height=height, 
                display=self.display, vsync=False, resizable=True)            
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 
            'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, 
            gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0) # draw
        self.window.flip()
    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
