#!/usr/bin/env python

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from math import sqrt, fabs

from pyhull.convex_hull import ConvexHull


class Pole:
    __slots__ = ("r", "c")
    
    def __init__(self, r, c):
        self.r = r
        self.c = c

        
class Triangle:
    __slots__ = ("v0", "v1", "check_vertices")
    
    def __init__(self, v0, check_vertex):
        self.v1 = None
        self.v0 = v0
        self.check_vertices = (check_vertex, )
        
    def add_simplex(self, v1, check_vertex):
        assert self.v1 is None
        
        self.check_vertices = (self.check_vertices[0], check_vertex)
        self.v1 = v1


def circumscribed_circle(X, Y, Z, W=None):
    if W is None:
        W = X ** 2 + Y ** 2 + Z ** 2
        
    O = np.ones(4)

    Dx = + np.linalg.det([W, Y, Z, O])
    Dy = - np.linalg.det([W, X, Z, O])
    Dz = + np.linalg.det([W, X, Y, O])

    a = np.linalg.det([X, Y, Z, O])
    c = np.linalg.det([W, X, Y, Z])

    C = np.array([Dx, Dy, Dz]) / 2 / a
    d = Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c
    
    if d < 0:
        return [0, 0, 0], -1

    r = sqrt(d) / 2 / fabs(a)
    
    return C, r


def draw_sphere(C, r, ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v)) + C[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + C[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + C[2]
    ax.plot_surface(x, y, z, alpha=0.2, rstride=4, cstride=4, color='b', linewidth=0.5)


# X, Y, Z = np.transpose(np.reshape(np.fromfile("resources/bun_zipper_res4.xyz", sep=" "), (-1, 3)))
X, Y, Z = np.transpose(np.reshape(np.fromfile("resources/dragon_vrip.xyz", sep=" "), (-1, 3)))
W = X ** 2 + Y ** 2 + Z ** 2
O = np.ones(4)

ch = ConvexHull(np.transpose([X, Y, Z, W]))
simplices = [s for s in ch.vertices if np.linalg.det([X[s], Y[s], Z[s], O]) < 0]

# for s in simplices:
#     C, r = circumscribed_circle(X[s], Y[s], Z[s], W[s])
#     for i in range(len(X)):
#         if not i in s:
#             v = [X[i], Y[i], Z[i]]
#             if np.linalg.norm(v - C) < r:
#                 print(s, i, np.linalg.norm(v - C), r)

poles = {}
triangles = {}
for s in simplices:
    C, r = circumscribed_circle(X[s], Y[s], Z[s], W[s])
    for v in s:
        if not v in poles or poles[v].r < r:
            poles[v] = Pole(r, C)
            
    for a, b, c, d in (0, 1, 2, 3), (0, 2, 3, 1), (0, 1, 3, 2), (1, 2, 3, 0):
        t_idx = tuple(sorted((s[a], s[b], s[c])))
        if t_idx in triangles:
            triangles[t_idx].add_simplex(C, s[d])
        else:
            triangles[t_idx] = Triangle(C, s[d])

Nx, Ny, Nz = np.transpose([(np.array(poles[i].c) - np.array([X[i], Y[i], Z[i]])) / poles[i].r if i in poles else [0, 0, 1] for i in range(len(X))])
    
def intersection_check(triangle, vertex, echo=False):
    if triangle.v1 is None:
        return False
    
    v = np.array([X[vertex], Y[vertex], Z[vertex]])
    vn = np.array([Nx[vertex], Ny[vertex], Nz[vertex]])
    v0, v1 = map(np.array, (triangle.v0, triangle.v1))
    d0, d1 = sorted(np.dot(vn, r / np.linalg.norm(r)) for r in (v0 - v, v1 - v))
    if echo:
        print(d0, d1)
        
    if d1 <= -0.38 or d0 >= 0.38:
        return False
    
    return True

candidate_triangles = {idx: triangle for idx, triangle in triangles.items() 
#                        if any(intersection_check(triangle, v) for v in idx + triangle.check_vertices)}
                       if all(intersection_check(triangle, v) for v in idx)}

# assert len(poles) == len(X)
    
with open("dragon.off", "w") as out:
    out.write("OFF\n")
    out.write("{} {} 0\n".format(len(X), len(candidate_triangles)))
    for v in zip(X, Y, Z):
        out.write("{} {} {}\n".format(*v))
    for f in candidate_triangles.keys():
        out.write("3 {} {} {}\n".format(*f))
    
