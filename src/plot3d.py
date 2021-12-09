"""
--------------------------------------------------------------------------------
Date: 09/12/2021

@author: Nikolaos Vardalakis
--------------------------------------------------------------------------------

Implementation Notes
--------------------------------------------------------------------------------
    | 1: For plotting the hippocampal structure in 3D I will be using a library
    | developed by Nicolas P. Rougier and can be found here:
    |       https://github.com/rougier/matplotlib-3d/
    | The following functions are wrappers, taken from his example code.
"""

import numpy as np

from mpl3d import glm
from mpl3d.camera import Camera


class Scatter:

    def __init__(self, ax, transform,
                 vertices, sizes=50, facecolors="white", edgecolors="black"):

        self.vertices = vertices
        self.sizes = sizes
        self.facecolors = facecolors
        self.edgecolors = edgecolors
        self.scatter = ax.scatter([], [])
        self.outline = ax.scatter([], [], zorder=-20,
                     linewidth=2, edgecolor="black", facecolor="black")
        self.update(transform)

    def update(self, transform):
        vertices = glm.transform(self.vertices, transform)

        I = np.argsort(-vertices[:,2])
        vertices = vertices[I]
        facecolors = self.facecolors[I]
        edgecolors = self.edgecolors[I]
        sizes = self.sizes[I]
        self.outline.set_offsets(vertices[:,:2])
        self.outline.set_sizes(sizes)

        vertices = np.repeat(vertices,2,axis=0)
        facecolors = np.repeat(facecolors,2,axis=0)
        facecolors[::2] = 0,0,0,.1
        edgecolors = np.repeat(edgecolors,2,axis=0)
        edgecolors[::2] = 0,0,0,0
        sizes = np.repeat(sizes,2,axis=0)
        sizes[::2] *= 2

        Z = vertices[:,2]
        Z = (Z-Z.min())/(Z.max()-Z.min())
        Z = Z[::2].reshape(-1,1)
        facecolors[1::2,:3] = Z + (1-Z)*facecolors[1::2,:3]
        edgecolors[1::2,:3] = Z + (1-Z)*edgecolors[1::2,:3]
        self.scatter.set_offsets(vertices[:,:2])
        self.scatter.set_sizes(sizes)
        self.scatter.set_facecolors(facecolors)
        self.scatter.set_edgecolors(edgecolors)
