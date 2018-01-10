import numpy as np


def constitutive_relation(tlat):
    """

    Parameters
    ----------
    tlat

    Returns
    -------

    """

    return constitutive_rel, e_extension, e_bending


def e_extension(xy, BL, bo, kk):
    # We convert xy to a 2D array here.
    xy = xy.reshape((-1, 2))
    bL = le.bond_length_list(xy, BL)
    bU = 0.5 * sum(kk * (bL - bo) ** 2)
    return bU


def constitutive_rel(tlat):
    """"""
    bU = e_extension(tlat.xy, self.lattice.BL, self.bo, self.kk)
    bB = e_bending(xy, normals)

