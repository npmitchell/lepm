import numpy as np
import lepm.dataio as dio
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
import copy
import scipy
from scipy.spatial import Delaunay

'''Functions for building networks from triangulated pointsets loaded from distk'''


def build_hyperuniform(lp):
    """Build a network from a triangulated hyperuniform pointset

    Parameters
    ----------
    lp

    Returns
    -------

    """
    networkdir = dio.prepdir(lp['rootdir']) + 'networks/'
    NH = lp['NH']
    NV = lp['NV']
    shape = lp['shape']
    if lp['NP_load'] == 0:
        points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N400/out_d' + str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        # make lattice larger than final cut
        keep = np.logical_and(abs(points[:, 0]) < NH * 1.5 + 8, abs(points[:, 1]) < NV * 1.5 + 8)
        xy = points[keep]
        xytri, NL, KL, BL, BM = le.delaunay_lattice_from_pts(xy, trimbound=False, target_z=lp['target_z'])
        # Crop lattice down to size
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)
        BBox = polygon
        print('Trimming lattice to be NH x NV...')
        keep = np.logical_and(abs(xy[:, 0]) < NH * 0.5, abs(xy[:, 1]) < NV * 0.5)
        print "Check that if lp['NP_load'] !=0 then len(keep) == len(xy):", len(keep) == len(xy)
        xy, NL, KL, BL = le.remove_pts(keep, xytri, BL, NN='min')
        # make string from origin values
        print 'lp[origin] = ', lp['origin']
        print 'type(lp[origin]) = ', type(lp['origin'])
        if (np.abs(lp['origin']) < 1e-7).all():
            originstr = ''
        else:
            originstr = '_originX' + '{0:0.2f}'.format(lp['origin'][0]).replace('.', 'p') + \
                        'Y' + '{0:0.2f}'.format(lp['origin'][1]).replace('.', 'p')
        periodicstr = ''
        LL = (NH, NV)
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]], dtype=float)
        PVx = []
        PVy = []
        PVxydict = {}
    else:
        # Unlike in the jammed case, we can't load a periodic bond list! We must make it.
        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])
        points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' + \
                            str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)
        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        BBox = polygon
        xy, NL, KL, BL, PVxydict = le.delaunay_rect_periodic_network_from_pts(points, LL, BBox='auto',
                                                                              check=lp['check'],
                                                                              zmethod=lp['cutz_method'],
                                                                              target_z=lp['target_z'])
        print 'Building periodicBC PVs...'
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        originstr = ''
        periodicstr = '_periodic'

        if lp['check']:
            le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, title='Checking periodic BCs',
                                  close=False, colorz=False)
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.plot(xy[:, 0], xy[:, 1], 'go')
            plt.show()

    lattice_exten = 'hyperuniform_' + shape + periodicstr + '_d' + '{0:02d}'.format(int(lp['conf'])) + \
                    '_z{0:0.3f}'.format(lp['target_z']) + originstr
    LV = 'none'
    UC = 'none'
    LVUC = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def build_hyperuniform_annulus(lp):
    """Build an annular sample of hyperuniform centroid network structure.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary. Uses lp['alph'] to determine the fraction of the system that is cut from the
        center

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten
    """
    xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_hyperuniform(lp)
    # Cut out the center and all particles more than radius away from center.
    radius = min(np.max(xy[:, 0]), np.max(xy[:, 1]))
    rad_inner = radius * lp['alph']
    dist = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    keep = np.logical_and(dist > rad_inner, dist < radius)
    # PVxydict and PV should really not be necessary here, since the network is not periodic
    if PVxydict:
        print 'PVxydict = ', PVxydict
        raise RuntimeError('Annulus function is not designed for periodic BCs -- what are you doing?')
    else:
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, check=lp['check'])  # , PVxydict=PVxydict, PV=PV)

    lattice_exten += '_alph' + sf.float2pstr(lp['alph'])
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten
