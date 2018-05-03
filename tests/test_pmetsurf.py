import numpy as np
import unittest
from pmetsurf import ParametricSurface, ParametricCurve


class TestParametricSurface(unittest.TestCase):
    def make_dome_case(self):
        """Make a partial spherical dome

        Make a space going from - 1/sqrt(2) to 1 / sqrt(2) where
        x = 10 * u
        y = 10 * v
        z = 10 - np.sqrt(x*x + y*y)
        """
        u = np.linspace(-1/np.sqrt(2), 1 / np.sqrt(2), 100)
        v = np.linspace(-1/np.sqrt(2), 1 / np.sqrt(2), 100)
        x = u[:, np.newaxis] * 10  + v[np.newaxis, :] * 0
        y = u[:, np.newaxis] * 0 + v[np.newaxis, :] * 10
        z = np.sqrt(100 - x*x - y*y)
        return ParametricSurface(u, v, x, y, z)

    def test_coordinates(self):
        ps = self.make_dome_case()
        u = [- .5, 0, 0]
        v = [ 0, 0, .5]
        x = [ -5, 0, 0]
        y = [ 0, 0, 5]
        z = [ 10 * np.sqrt(.75), 10, 10 * np.sqrt(.75)]
        coords = ps[u, v]
        for xx, yy, zz, (zzz, yyy, xxx) in zip(x, y, z, coords):
            self.assertAlmostEqual(xx, xxx)
            self.assertAlmostEqual(yy, yyy)
            self.assertAlmostEqual(zz, zzz)

    def test_normal(self):
        ps = self.make_dome_case()
        u = [- 10 / np.sqrt(2), 0, 0]
        v = [ 0, 0, 10 / np.sqrt(2)]
        x = [ -1 / np.sqrt(2), 0, 0]
        y = [ 0, 0, 1 / np.sqrt(2)]
        z = [ 1/np.sqrt(2), 1, 1/np.sqrt(2)]
        normal = ps.normal(u, v)
        for xx, yy, zz, (zzz, yyy, xxx) in zip(x, y, z, normal):
            xx, yy, zz, xxx, yyy, zzz = [
                0 if np.abs(_) < .0001 else _
                for _ in (xx, yy, zz, xxx, yyy, zzz)]
            if np.sign(zzz) != np.sign(zz):
                xx, yy, zz = -xx, -yy, -zz
            self.assertAlmostEqual(np.abs(zzz), np.abs(zz), 4)
            self.assertAlmostEqual(np.abs(yyy), np.abs(yy), 4)
            self.assertAlmostEqual(np.abs(xxx), np.abs(xx), 4)

    def test_curvature(self):
        #
        # TODO: a test case that's a saddle point
        #
        ps = self.make_dome_case()
        u = [- 10 / np.sqrt(2), 0, 0]
        v = [ 0, 0, 10 / np.sqrt(2)]
        kmin = ps.kmin(u, v)
        kmax = ps.kmax(u, v)
        for kmin1 in kmin:
            self.assertAlmostEqual(kmin1, 1.0 / 10, 3)
        for kmax1 in kmax:
            self.assertAlmostEqual(kmax1, 1.0 / 10, 3)


class TestParametricCurve(unittest.TestCase):

    def make_test_case(self):
        t = np.linspace(-np.pi, np.pi, 100)
        x = np.sin(t) * 10
        y = np.cos(t) * 10
        return ParametricCurve(x, y, t, 25)

    def test_coordinates(self):
        p = self.make_test_case()
        t = np.random.RandomState(427).uniform(-np.pi, np.pi, 10)
        expected_x = np.sin(t) * 10
        expected_y = np.cos(t) * 10
        actual_y, actual_x = p[t].transpose()
        for ex, ey, ax, ay in zip(expected_x, expected_y, actual_x, actual_y):
            self.assertAlmostEqual(ex, ax, 1)
            self.assertAlmostEqual(ey, ay, 1)

    def test_normal(self):
        p = self.make_test_case()
        angles = [ 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        x_expected = [ 0, np.sqrt(.5), 1, np.sqrt(.5)]
        y_expected = [ 1, np.sqrt(.5), 0, -np.sqrt(.5)]
        y_actual, x_actual = p.normal(angles).transpose()
        for xa, ya, xe, ye in zip(x_actual, y_actual, x_expected, y_expected):
            self.assertAlmostEqual(xa, xe, 1)
            self.assertAlmostEqual(ya, ye, 1)

    def test_curvature(self):
        p = self.make_test_case()
        t = np.random.RandomState(427).uniform(-np.pi, np.pi, 10)
        curvature = p.curvature(t)
        for c in curvature:
            self.assertAlmostEqual(c, 1 / 10, 2)

if __name__ == '__main__':
    unittest.main()
