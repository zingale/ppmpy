import numpy as np

from ppmpy.grid import FVGrid
from ppmpy.gravity import constant_gravity


class TestGravity:

    @classmethod
    def setup_class(cls):
        """ this is run once for each class before any tests """

    @classmethod
    def teardown_class(cls):
        """ this is run once for each class after all tests """

    def setup_method(self):
        """ this is run before each test """

        self.g = FVGrid(10, ng=2)

    def teardown_method(self):
        """ this is run after each test """

    def test_constant_gravity(self):

        rho = np.ones(self.g.nq)

        g_const = -981.0

        params = {"g_const": g_const}

        grav = constant_gravity(self.g, rho, params)

        assert grav[self.g.lo].min() == g_const
        assert grav[self.g.lo].max() == g_const
