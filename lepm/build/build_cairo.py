import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.le_geometry as leg
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
import scipy
from scipy.spatial import Delaunay
import scipy.optimize as opt
import sys
import glob
import math
import cmath
import shapely.geometry as sg
import descartes


def build_cairo(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = generate_cairo_tiling(lp)
    BBox = blf.auto_polygon(shape, NH, NV, eps=0.00)
    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten

def generate_cairo_tiling(lp):
    """

    Parameters
    ----------
    lp : dict
        lattice parameters dict with keys: shape, NH, NV, aratio, delta, phi, check

    Returns
    -------

    """
    print 'creating cairo tiling...'
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['eta'] = 0.
    lp_tmp['NH'] += 5
    lp_tmp['NV'] += 5
    xyA, NLA, KLA, BLA, LVUCA, LVA, UCA, PVxydictA, PVxA, PVyA, lattice_exten = \
        generate_flattened_honeycomb_lattice(lp_tmp)

    xyleft = xyA[xyA[:, 0] < np.min(xyA[:, 0]) + 1e-4]
    farleft = np.argmin(xyleft[:, 1])
    xyA -= xyleft[farleft]

    if lp['check']:
        le.display_lattice_2D(xyA, BLA)

    xyB = np.dstack((xyA[:, 1], xyA[:, 0]))[0] + np.array([3., 1.])
    BLB = (np.abs(BLA) + len(xyA)) * (np.minimum(np.sign(BLA[:, 0]), np.sign(BLA[:, 1]))).reshape(len(BLA), 1)
    LVUC = 'none'
    LV = LVA
    UC = 'none'

    xy = np.vstack((xyA, xyB))
    BL = np.vstack((BLA, BLB))
    keep = np.logical_and(np.logical_and(xy[:, 0] > 10, xy[:, 1] > 10),
                          np.logical_and(xy[:, 1] < np.max(xyA[:, 1]), xy[:, 0] < np.max(xyB[:, 0])))
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL)

    # Find all crossing line segs, replace with a particle connected to each particle involved

    # todo: make periodic BCs work (currently aren't realistically used)
    PVxydict = copy.deepcopy(PVxydictA)
    for key in PVxydictA:
        PVxydict[(key[1], key[0])] = np.array([PVxydictA[key][1], PVxydictA[key][0]])
    PVxB = copy.deepcopy(PVyA)
    PVyB = copy.deepcopy(PVxA)
    PVx = np.vstack((PVxA, PVxB))
    PVy = np.vstack((PVyA, PVyB))

    lattice_exten = 'cairo_' + lattice_exten[19:]
    return xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten

