import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import tensorflow as tf

import os
"""
How to use this class:
    Create a VideoWriter object with the filename and other parameters
    There are a few ways to add images to the video:
        add_img: adds a numpy array to the video, this can be a 2d array for a grayscale image or a 3d array for a color image
        add_grid: adds a 2d numpy array as a heat map to the video
        add_concat_grids: adds multiple grids, each heatmaps, as a grid with the specified number of columns

Here is an example:
fname = "test"
with VideoWriter("./" + fname +".mp4", fps=33) as vid:
    for i in range(100):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        vid.add_grid(img)

"""

class VideoWriter:
    def __init__(self, filename=None, scale=None, fps=30.0, **kw):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.scale = scale
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)
        self.frames = np.array([])

    def add_img(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)


    def add_concat_grids(self, grids, scale=None, cols=3, cmaps=None):
        """
        Adds a grid of images to the video
        grids: a list of numpy arrays
        scale: the scale of the image
        cols: the number of columns in the grid
        cmaps: a list of colormaps for each grid
            if None, all grids will be colored with the hot colormap
        """
        if cmaps is None:
            cmaps = ["hot"]*len(grids)

        rows = (len(grids)-1)//cols+1 
        h, w = grids[0].shape[:2]
        grid = np.zeros((h*rows, w*cols, 4))

        for i, (g, cmap) in enumerate(zip(grids, cmaps)): 
            norm = mpl.colors.Normalize(g.min(), g.max())
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            grid[i//cols*h:(i//cols+1)*h, i %
                 cols*w:(i % cols+1)*w] = m.to_rgba(g) 

        if scale is None:
            # 512 is the default size of the video, grids smaller than this will be upscaled
            self.scale = 512/grid.shape[1]
        else:
            self.scale = scale
        self.add_img(self.to_rgb(self.zoom(grid, self.scale)))

    # Creates a heat map image from a 2d numpy array and adds it to the video
    def add_grid(self, grid, scale=None, cmap="hot"):
        self.add_concat_grids([grid], scale=scale, cols=1, cmaps=[cmap])

    def add_img_buf(self, img_buf):
        img = tf.image.decode_png(img_buf, channels=3)
        self.add_img(img)

    def to_alpha(self, x):
        return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

    def to_rgb(self, x):
        # assume rgb premultiplied by alpha
        rgb, a = x[..., :3], self.to_alpha(x)
        return 1.0-a+rgb

    def zoom(self, img, scale=4):
        img = np.repeat(img, scale, 0)
        img = np.repeat(img, scale, 1)
        return img

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
