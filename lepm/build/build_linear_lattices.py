import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
from scipy.spatial import Delaunay

'''Functions for making 1d lattices'''


def build_zigzag_lattice(lp):
    """Build a zigzag (linear) lattice, with angle theta and randomization eta"""
    if lp['periodicBC']:
        xy, NL, KL, BL, LVUC, LV, UC, LL, PV, PVxydict, BBox, lattice_exten = \
            generate_periodic_zigzag_lattice(lp['NH'], lp['theta'] * np.pi, lp['eta'])
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)
    else:
        xy, NL, KL, BL, LV, lattice_exten = generate_zigzag_lattice(lp['NH'], lp['theta'] * np.pi, lp['eta'])
        PVx = []
        PVy = []
        PVxydict = {}
        print 'LV = ', LV
        UC = 'none'
        LVUC = 'none'
        LL = (lp['NH'] + 1, 1)
        minx = np.min(xy[:, 0])
        miny = np.min(xy[:, 1])
        maxx = np.max(xy[:, 0])
        maxy = np.max(xy[:, 1])
        BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_periodic_zigzag_lattice(NH, theta, eta=0.):
    """

    Parameters
    ----------
    NH
    theta
    eta

    Returns
    -------

    """
    LV = np.array([[np.cos(theta), np.sin(theta)], [np.cos(theta), -np.sin(theta)]])

    xy = np.array([np.ceil(i * 0.5) * LV[0] + np.ceil((i - 1) * 0.5) * LV[1] for i in range(NH)])
    if abs(LV[0, 1]) < 1e-9:
        # There is only one kind of particle
        LVout = np.array([[1., 0.]])
        UC = np.array([0., 0.])
        LVUC = np.array([[np.ceil(i * 0.5), np.ceil((i - 1) * 0.5), 0] for i in range(NH)])
    else:
        # There are two kinds of particles, up and down
        LVout = np.array([[2 * np.cos(theta), 0]])
        UC = LV
        LVUC = np.array([[np.ceil(i * 0.5), np.ceil((i - 1) * 0.5), i % 2] for i in range(NH)])

    xy -= np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
    print 'xy = ', xy

    # Connectivity
    BL = np.zeros((NH, 2), dtype=int)
    for i in range(0, NH - 1):
        BL[i, 0] = i
        BL[i, 1] = i + 1
    # make periodic bond
    BL[NH - 1, 0] = - 0
    BL[NH - 1, 1] = - (NH - 1)

    print 'BL = ', BL
    # scale lattice down to size
    if eta == 0:
        xypts = xy
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]

    # Naming
    etastr = '{0:.3f}'.format(eta).replace('.', 'p')
    thetastr = '{0:.3f}'.format(theta / np.pi).replace('.', 'p')
    exten = '_periodic_line_theta' + thetastr + 'pi_eta' + etastr

    # BL = latticevec_filter(BL,xy, C, CBL)
    NL, KL = le.BL2NLandKL(BL, NP=NH, NN=2)

    # Create periodic and lattice info
    print 'xy = ', xy
    minx = np.min(xy[:, 0]) - np.cos(theta) * 0.5
    miny = np.min(xy[:, 1])
    maxx = np.max(xy[:, 0]) + np.cos(theta) * 0.5
    maxy = np.max(xy[:, 1])
    if miny == maxy:
        maxy = 0.5
        miny = -0.5
    LL = (maxx - minx, 1)
    PV = np.array([[LL[0], 0], [0, LL[1]]])
    print 'PV = ', PV
    BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
    PVxydict = le.BL2PVxydict(BL, xy, PV)
    print 'PVxydict = ', PVxydict
    lattice_exten = 'linear' + exten
    print 'lattice_exten = ', lattice_exten
    return xypts, NL, KL, BL, LVUC, LVout, UC, LL, PV, PVxydict, BBox, lattice_exten


def generate_zigzag_lattice(NH, theta, eta=0.):
    """Creates a zigzag (linear) lattice, with angle theta and randomization eta
    """
    LV = np.array([[np.cos(theta),  np.sin(theta)], [np.cos(theta), -np.sin(theta)]])

    xy = np.array([np.ceil(i*0.5)*LV[0] + np.ceil((i-1)*0.5)*LV[1] for i in range(NH)])
    xy -= np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
    print 'xy = ', xy

    # Connectivity
    BL = np.zeros((NH-1, 2), dtype=int)
    for i in range(0, NH-1):
        BL[i, 0] = i
        BL[i, 1] = i + 1

    print 'BL = ', BL

    TRI = le.BL2TRI(BL, xy)

    # scale lattice down to size
    if eta == 0:
        xypts = xy
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta*np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]

    # Naming
    etastr = '{0:.3f}'.format(eta).replace('.', 'p')
    thetastr = '{0:.3f}'.format(theta/np.pi).replace('.', 'p')
    exten = '_line_theta' + thetastr + 'pi_eta'+etastr

    # BL = latticevec_filter(BL,xy, C, CBL)
    NL, KL = le.BL2NLandKL(BL, NP=NH, NN=2)
    lattice_exten = 'linear' + exten
    print 'lattice_exten = ', lattice_exten
    return xypts, NL, KL, BL, LV, lattice_exten


def generate_circle_lattice(N):
    """Generate lattice as a circle of points connected by nearest neighbors.
    """
    theta = np.linspace(0, 2. * np.pi, N + 1)
    theta = theta[:-1]
    # The radius, given the length of a side is:
    # radius = s/(2 * sin(2 pi/ n)), where n is number of sides, s is length of each side
    # We set the length of a side to be 1 (the rest length of each bond)
    R = 1. / (2. * np.sin(np.pi / float(N)))
    # print '(2.* np.sin(2.*np.pi/float(N))) = ', (2.* np.sin(np.pi/float(N)))
    # print 'R = ', R
    xtmp = R * np.cos(theta)
    ytmp = R * np.sin(theta)
    xy = np.dstack([xtmp, ytmp])[0]
    if N == 1:
        BL = np.array([[]])
    elif N == 2:
        BL = np.array([[0, 1]])
    else:
        BL = np.array([[i, (i + 1) % N] for i in np.arange(N)])
    # print 'BL = ', BL
    # print 'NP = ', len(xtmp)
    NL, KL = le.BL2NLandKL(BL, NP=len(xtmp), NN=2)
    LV = np.array([[0, 0], [0, 0]], dtype=int)
    UC = xy
    ztmp = np.zeros(len(xy), dtype=int)
    LVUC = np.dstack((ztmp, ztmp, np.arange(len(xy), dtype=int)))[0]
    print LVUC
    lattice_exten = 'circlebonds'
    return xy, NL, KL, BL, LV, UC, LVUC, lattice_exten
