import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.integrate import dblquad
from matplotlib import cm
import cPickle as pickle
import os
from scipy.interpolate import griddata
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import chern_functions_gen as cf


def find_lattice_vecs(delta, phi):
    """finds the real-space lattice vecs for a honeycomb lattice.  it returns the two shortest lattice vectors.
    
    Parameters
    ----------
    delta : float
        Value of delta for lattice in radians
        
    phi : float
        Value of phi for lattice in radians
        
    Returns
    ----------
    a1 : array 1x3
        first lattice vector
        
    a2: np.array 1x3
        second lattice vector
     
    """
    a1 = np.array([np.sin(phi) + cos(1 / 2. * (np.pi - delta)), cos(phi) + np.sin(1 / 2. * (np.pi - delta)), 0])
    a2 = np.array([np.sin(phi) - cos(1 / 2. * (np.pi - delta)), cos(phi) + np.sin(1 / 2. * (np.pi - delta)), 0])
    a3 = a1 + a2

    vecs = np.array([a1, a2, a3])
    mags = sum(abs(vecs) ** 2, axis=-1) ** (1 / 2.)
    si = np.argsort(mags)
    vecs = vecs[si]

    return vecs[0], vecs[1]


def find_lattice_vecs_alpha_ns():
    """finds the lattice vecs for a square lattice"""
    a1 = np.array([0, 2, 0])
    a2 = np.array([sqrt(3), 0, 0])
    return a1, a2


def find_lattice_vecs_alpha():
    """finds the lattice vecs for an alpha lattice"""
    a1 = np.array([0, 1, 0])
    a2 = np.array([sqrt(3), 0, 0])

    return a1, a2


def find_bz_lattice_vecs(a1, a2):
    """finds the BZ lattice vecs
    
    Parameters
    ----------
    a1: array 1x3
        first lattice vector
        
    a2 : array 1x3
        second lattice vector
        
    Returns
    ----------
    b1 : array 1x3
        first reciprocal lattice vector

    b2 : array 1x3
        second reciprocal lattice vector
 
    """
    a3 = np.array([0, 0, 1])

    b1 = 2 * np.pi * np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))
    b2 = 2 * np.pi * np.cross(a3, a1) / (np.dot(a1, np.cross(a2, a3)))

    return b1, b2


def find_bz_zone(b1, b2):
    """This function finds the BZ zone for any lattice given the reicprocal lattice vectors.
    
    Parameters
    ----------
    b1 : array 1x3
        first reciprocal lattice vector
    b2 : array 1x3
        second reciprocal lattice vector
      
    Returns
    ----------
    vtx : array 
        x position of BZ zone corners
    vty : array 
        y position of BZ zone corners
    """
    b3 = np.array([b2[0] - b1[0], b2[1] - b1[1], 0])
    p1 = b1
    p2 = b2
    p3 = -b1
    p4 = -b2
    p5 = (b1 + b2)
    p6 = -(b2 + b1)
    p7 = (b1 - b2)
    p8 = (b2 - b1)
    p9 = b2 + b3
    p10 = -(b2 + b3)
    p11 = 2 * b1
    p12 = 2 * b2

    min_p1112 = min(np.array([p11[0] ** 2 + p11[1] ** 2, p12[0] ** 2 + p12[1] ** 2]))

    ps = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

    pdist = [ps[i, 0] ** 2 + ps[i, 1] ** 2 for i in range(len(ps))]
    si = np.argsort(pdist)
    ps = ps[si][:6]

    # a true value of rr indicates that some points (in the p array) should not be used in the BZ calculation.
    vtx, vty, rr = bz_based_on_ps(ps)
    if rr:
        ps2 = ps[:4]
    else:
        ps2 = ps
    vtx, vty, rr = bz_based_on_ps(ps2)

    return vtx, vty


def bz_based_on_ps(ps, ii=0):
    """This function finds the Brillouin zone based on calculations of the nearest equivalent points to the origin.
    The algorithm used finds the intersections between the perpendicular lines from these points to the origin.
    
    Parameters
    ----------
    ps : array
        nearest neighbor points which are equivalent to the origin in the BZ zone
      
    Returns
    ----------
    vtx : array 
        x position of BZ zone corners
    vty : array 
        y position of BZ zone corners
    rr : boolean
        once the BZ zone corners are found the algorithm checks their polar angle order.  If they are out of order, it
        indicates that too many points have been used in the calculation
        and that the incorrect Brillouin zone has been found.
    """
    angs = np.array([(np.arctan2(ps[i, 1], ps[i, 0]) + .01) % (2 * np.pi) for i in range(len(ps))])
    si = np.argsort(angs)
    ps = ps[si]

    # perpendicular bisectors between these and origin?
    mids = ps / 2

    # fig = plt.figure()
    # plt.scatter(ps[:,0], ps[:,1], c = 'k')
    # plt.scatter(mids[:,0], mids[:,1], c = 'r')
    # plt.axes().set_aspect(1)

    x = []
    l = []
    jj = np.where(abs(ps[:, 1]) < 0.001)[0]

    ps_fs = ps.copy()
    ps_fs[jj, 1] = 1.
    # slopes of the perpendicular bisectors
    slopes = np.array([-ps_fs[i, 0] / ps_fs[i, 1] for i in range(len(ps))])
    # vertical lines get a slope of -1.23
    slopes[jj] = -1.23

    # for plotting
    for i in range(len(slopes)):
        if slopes[i] != -1.23:
            x_vals = arange(mids[i, 0] - 5, mids[i, 0] + 5, 0.05)
            lines = slopes[i] * (x_vals - mids[i, 0]) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)
            # plt.plot(x_vals, lines)
        else:
            x_vals = zeros(200) + mids[i, 0]
            lines = arange(-5, 5, 0.05) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)
            # plt.plot(x_vals, lines)

    x = np.array(x)
    vtx = []
    vty = []

    for i in range(len(slopes)):
        if i == len(slopes) - 1:
            neigh = 0
        else:
            neigh = i + 1

        m1 = slopes[i]
        m2 = slopes[neigh]
        if m1 != -1.23 and m2 != -1.23:
            xy1 = mids[i]
            xy2 = mids[neigh]

            xv = (m1 * xy1[0] - xy1[1] - m2 * xy2[0] + xy2[1]) / (m1 - m2)
            yv = slopes[i] * (xv - mids[i, 0]) + mids[i, 1]
        else:
            if m1 == - 1.23:
                jk = neigh
                jl = i
            else:
                jk = i
                jl = neigh

            xv = x[jl, 0]

            yv = slopes[jk] * (xv - mids[jk, 0]) + mids[jk, 1]

        vtx.append(xv)
        vty.append(yv)

    # plt.scatter(vtx, vty, s = 60)
    # plt.show()
    angs = np.array([np.arctan2(vty[i], vtx[i]) % (2 * np.pi) for i in range(len(vtx))])
    si = np.argsort(angs)
    vtx = np.array(vtx)[si]
    vty = np.array(vty)[si]
    diff = [abs(si[i + 1] - si[i]) for i in range(len(si) - 1)]
    if 2 in diff:
        rr = True
    else:
        rr = False

    return vtx, vty, rr


def find_bz_zone_alpha(b1, b2):
    """finds the BZ zone for an alpha lattice given the reicprocal lattice vectors and delta

    Parameters
    ----------
        b1 : array 1x3
            first reciprocal lattice vector
        b2 : array 1x3
            second reciprocal lattice vector

    Returns
    -------
        vtx : array 
            x position of BZ zone corners
        vty : array 
            y position of BZ zone corners
    """
    p1 = b1
    p2 = b2
    p3 = -b1
    p4 = -b2
    p5 = b1 + b2
    p6 = b1 - b2
    p7 = -b1 + b2
    p8 = -b1 - b2

    ps = np.array([p1, p2, p3, p4])

    angs = np.array([np.arctan2(ps[i, 1], ps[i, 0]) % (2 * np.pi) for i in range(len(ps))])
    si = np.argsort(angs)

    ps = ps[si]

    # perpendicular bisectors between these and origin?

    mids = ps / 2

    slopes = np.array([ps[i, 1] / ps[i, 0] for i in range(len(ps))])
    x = []
    l = []

    # print slopes
    jj = np.where(ps[:, 1] == 0)[0]
    ps_fs = ps.copy()
    ps_fs[jj, 1] = 1.
    slopes = np.array([-ps_fs[i, 0] / ps_fs[i, 1] for i in range(len(ps))])

    slopes[jj] = -1.23

    for i in range(len(slopes)):
        if slopes[i] != -1.23:
            x_vals = arange(mids[i, 0] - 5, mids[i, 0] + 5, 0.05)
            lines = slopes[i] * (x_vals - mids[i, 0]) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)
        else:
            x_vals = zeros(200) + mids[i, 0]
            lines = arange(-5, 5, 0.05) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)

    x = np.array(x)
    l = np.array(l)

    x = np.array(x)
    vtx = []
    vty = []
    for i in range(len(slopes)):
        if i == len(slopes) - 1:
            neigh = 0
        else:
            neigh = i + 1

        m1 = slopes[i]
        m2 = slopes[neigh]
        if m1 != -1.23 and m2 != -1.23:

            xy1 = mids[i]
            xy2 = mids[neigh]

            xv = (m1 * xy1[0] - xy1[1] - m2 * xy2[0] + xy2[1]) / (m1 - m2)
            yv = slopes[i] * (xv - mids[i, 0]) + mids[i, 1]


        else:
            if m1 == - 1.23:
                jk = neigh
                jl = i
            else:
                jk = i
                jl = neigh

            xv = x[jl, 0]
            yv = slopes[jk] * (xv - mids[jk, 0]) + mids[jk, 1]

        vtx.append(xv)
        vty.append(yv)

    return vtx, vty
