# pmetsurf

[![Travis CI Status](https://travis-ci.org/chunglabmit/pmetsurf.svg?branch=master)](https://travis-ci.org/chunglabmit/pmetsurf)


Parametric surfaces using splines

This is a library of routines to manipulate
3D surfaces parameterized by two variables.

Example usage:

```python
import numpy as np
from pmetsurf import ParametricSurface

#
# Make a half-sphere as a parametric surface of u and v
#
u = np.linspace(-np.pi, np.pi, 30)
v = np.linspace(-np.pi/2, np.pi/2, 30)
x = 10 * np.cos(u[:, np.newaxis]) * np.sin(v[np.newaxis, :])
y = 10 * np.sin(u[:, np.newaxis]) * np.sin(v[np.newaxis, :])
z = 10 * np.cos(v[np.newaxis, :])
surface = ParametricSurface(u, v, x, y, z)
#
# Accessing the surface
#
random_u = np.random.uniform(-np.pi, np.pi, 10)
random_v = np.random.uniform(-np.pi/2, np.pi/2, 10)
#
# The coordinates on the surface
#
coords = surface[random_u, random_v]
#
# The normal to the surface (maybe you have to reverse it)
#
normals = surface.normal(random_u, random_v)
#
# The curvature (in this case = 1/10 everywhere)
#
k1 = surface.kmax(random_u, random_v)
k2 = surface.kmin(random_u, random_v)

```