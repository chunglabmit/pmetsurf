# coding: utf8

import numpy as np
from scipy.interpolate import RectBivariateSpline


def vectordot(a, b):
    """Perform a dot product on vectors of coordinates

    For i in len(a), perform np.dot(a[i], b[i])

    :param a: An NxM matrix of N vectors of length M
    :param b: A similarly-shaped NxM matrix
    :returns: a vector of length N giving the dot product of each a.b pair
    """
    return np.sum(a * b, 1)


def memoize(function):
    last_u = {}
    last_v = {}
    last_result = {}

    def wrapper(self, u, v):
        selfid = id(self)
        uid = id(u)
        vid = id(v)
        if selfid in last_u and last_u[selfid] == uid and last_v[selfid] == vid:
            return last_result[selfid]
        last_result[selfid] = function(self, u, v)
        last_u[selfid] = uid
        last_v[selfid] = vid
        return last_result[selfid]

    return wrapper


class ParametricSurface:
    """A 3D surface in x, y, z parameterized by u and v

    Curvature, normals and other are implementations of equations in
    the following sources:
    http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node30.html
    https://en.wikipedia.org/wiki/Parametric_surface
    """

    def __init__(self, u, v, x, y, z):
        """
        Initialize with three matrices in x, y and z

        :param u: the linear space for the first parameter axis, a 1-D array
        :param v: the linear space for the second parameter axis, a 1-D array
        :param x: a matrix of dimension, (len(u), len(v)) giving the X
                  coordinate at each u, v
        :param y: a matrix of dimension, (len(u), len(v)) giving the Y
                  coordinate at each u, v
        :param z: a matrix of dimension, (len(u), len(v)) giving the Z
                  coordinate at each u, v
        """
        self.splx = RectBivariateSpline(u, v, x)
        self.sply = RectBivariateSpline(u, v, y)
        self.splz = RectBivariateSpline(u, v, z)

    def __getitem__(self, item):
        """Get the surface coordinate at u, v

        """
        u, v = item
        return self.__getitem(u, v)

    @memoize
    def __getitem(self, u, v):
        u = np.asanyarray(u)
        v = np.asanyarray(v)
        return np.column_stack(
            [_.ev(u, v) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def du(self, u, v):
        """The derivative with respect to U

        Return the first derivative of the surface with respect to the U
        parameter, evaluated at u, v.
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return np.column_stack(
            [_.ev(u, v, dx=1) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def dv(self, u, v):
        """The derivative with respect to V

        Return the first derivative of the surface with respect to the V
        parameter, evaluated at u, v.
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return np.column_stack(
            [_.ev(u, v, dy=1) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def duu(self, u, v):
        """The second derivative with respect to U

        Return the second derivative of the surface with respect to the U
        parameter, evaluated at u, v.
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return np.column_stack(
            [_.ev(u, v, dx=2) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def dvv(self, u, v):
        """The second derivative with respect to V

        Return the second derivative of the surface with respect to the V
        parameter, evaluated at u, v.
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return np.column_stack(
            [_.ev(u, v, dy=2) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def duv(self, u, v):
        """The u/v cross derivative dudv

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return np.column_stack(
            [_.ev(u, v, dx=1, dy=1) for _ in (self.splz, self.sply, self.splx)])

    @memoize
    def E(self, u, v):
        """The first parameter of the first fundamental form

        E = du∙du

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.du(u, v), self.du(u, v))

    @memoize
    def F(self, u, v):
        """The second parameter of the first fundamental form

        F = du∙dv

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.du(u, v), self.dv(u, v))

    @memoize
    def G(self, u, v):
        """The third parameter of the first fundamental form

        G = dv∙dv

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.dv(u, v), self.dv(u, v))

    @memoize
    def normal(self, u, v):
        """The vector normal to the surface

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        :returns: vectors normal to the surface at each u, v
        """
        result = np.cross(self.du(u, v), self.dv(u, v))
        result = result / np.sqrt(vectordot(result, result))[:, None]
        return result

    @memoize
    def L(self, u, v):
        """The first parameter of the second fundamental form

        L = duu ∙ normal
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.duu(u, v), self.normal(u, v))

    @memoize
    def M(self, u, v):
        """The second parameter of the second fundamental form

        M = dudv ∙ normal
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.duv(u, v), self.normal(u, v))

    @memoize
    def N(self, u, v):
        """The third parameter of the second fundamental form

        N = dvv ∙ normal
        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        """
        return vectordot(self.dvv(u, v), self.normal(u, v))

    @memoize
    def K(self, u, v):
        """The Gaussian curvature at u, v

        See https://en.wikipedia.org/wiki/Gaussian_curvature

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        :returns: the Gaussian curvature at each u, v
        """
        return (self.L(u, v) * self.N(u, v) - np.square(self.M(u, v))) / \
               (self.E(u, v) * self.G(u, v) - np.square(self.F(u, v)))

    @memoize
    def H(self, u, v):
        """The mean curvature at u, v

        See https://en.wikipedia.org/wiki/Mean_curvature

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        :returns: the mean curvature at each u, v
        """
        return (self.E(u, v) * self.N(u, v)
                - 2 * self.F(u, v) * self.M(u, v)
                + self.G(u, v) * self.L(u, v)) / \
               (2 * (self.E(u, v) * self.G(u, v) - np.square(self.F(u, v))))

    @memoize
    def kmax(self, u, v):
        """The maximum curvature at u, v

        The maximum curvature among all planes normal to the surface.

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        :returns: the maximum curvature at each u, v
        """
        return self.H(u, v) + np.sqrt(np.square(self.H(u, v)) - self.K(u, v))

    @memoize
    def kmin(self, u, v):
        """The minimum curvature at u, v

        The minimum curvature among all planes normal to the surface.

        :param u: a vector of the u at which to evaluate
        :param v: a vector of the v at which to evaluate
        :returns: the minimum curvature at each u, v
        """
        return self.H(u, v) - np.sqrt(np.square(self.H(u, v)) - self.K(u, v))


all=[ParametricSurface]
