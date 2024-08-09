"""
Functions that can provide the gravitational acceleration
"""


def constant_gravity(grid, _, params):
    """simple constant gravity.  Set the value via params["g_const"]"""

    g = grid.scratch_array()
    g[grid.lo:grid.hi+1] = params["g_const"]
    return g
