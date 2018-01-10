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

'''Functions for creating Voronoi (not centroidal dual) networks from loaded pointsets'''


def build_huvoronoi(lp):
    if lp['NP_load'] == 0:
        points = np.loadtxt(lp['rootdir'] + 'networks/hyperuniform_source/hyperuniform_N400/out_d' +
                            str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        keep = np.logical_and(abs(points[:, 0]) < (8 + lp['NH'] * 0.5), abs(points[:, 1]) < (8 + lp['NV'] * 0.5))
        xytmp = points[keep]
        if lp['check']:
            plt.plot(points[:, 0], points[:, 1], 'b.')
            plt.title('Point set before initial cutting')
            plt.show()
        polygon = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)

        xy, NL, KL, BL = le.voronoi_lattice_from_pts(xytmp, polygon=polygon, NN=3, kill_outliers=True,
                                                     check=lp['check'])
        if lp['check']:
            le.display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Cropped centroid lattice, before dilation')

        # Form output bounding box and extent measurement
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        PVxydict = {}
        PVx = []
        PVy = []
        BBox = polygon

        # make string from origin values
        print 'lp[origin] = ', lp['origin']
        print 'type(lp[origin]) = ', type(lp['origin'])
        if (np.abs(lp['origin']) < 1e-7).all():
            originstr = ''
        else:
            originstr = '_originX' + '{0:0.2f}'.format(lp['origin'][0]).replace('.', 'p') + \
                        'Y' + '{0:0.2f}'.format(lp['origin'][1]).replace('.', 'p')
        periodicstr = ''
    else:
        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])
        print 'sizestr = ', sizestr
        points = np.loadtxt(lp['rootdir'] + 'networks/hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' +
                            str(int(lp['conf'])) + '_xy.txt')
        print 'points = ', points
        points -= np.mean(points, axis=0)
        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        BBox = polygon
        xy, NL, KL, BL, PVxydict = le.voronoi_rect_periodic_from_pts(points, LL, BBox='auto', check=lp['check'])
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        originstr = ''
        periodicstr = '_periodic'

    # Rescale so that median bond length is unity
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    scale = 1. / np.median(bL)
    xy *= scale
    polygon *= scale
    BBox *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['NP_load'] != 0:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())

    lattice_exten = 'huvoronoi_' + lp['shape'] + periodicstr + '_d' + '{0:02d}'.format(int(lp['conf'])) + originstr
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten


def build_kagome_huvor(lp):
    """Construct a kagomized network from hyperuniform point set and voronoize it with Wigner-Seitz

    Parameters
    ----------
    lp : dict
        Lattice parameters dictionary, with keys:
        rootdir, conf, origin, shape, NH, NV, check, NP_load (if periodic)
    """
    if lp['NP_load'] == 0:
        points = np.loadtxt(dio.prepdir(lp['rootdir']) + 'networks/hyperuniform_source/hyperuniform_N400/out_d' +
                               str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        addpc = .05
        xytmp, trash1, trash2, trash3 = \
            blf.mask_with_polygon(lp['shape'], lp['NH'] + 7, lp['NV'] + 7, points, [], eps=addpc)
        polygon = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)

        xy, NL, KL, BL = blf.kagomevoronoi_network_from_pts(xytmp, polygon=polygon, kill_outliers=True, check=lp['check'])

        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        PVxydict, PVx, PVy = {}, [], []
        BBox = polygon
        if lp['check']:
            le.display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Cropped centroid lattice, before dilation')

        # make string from origin values
        print 'lp[origin] = ', lp['origin']
        print 'type(lp[origin]) = ', type(lp['origin'])
        if (np.abs(lp['origin']) < 1e-7).all():
            originstr = ''
        else:
            originstr = '_originX' + '{0:0.2f}'.format(lp['origin'][0]).replace('.', 'p') + \
                        'Y' + '{0:0.2f}'.format(lp['origin'][1]).replace('.', 'p')
        periodicBCstr = ''
    else:
        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        BBox = polygon

        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])
        points = np.loadtxt(dio.prepdir(lp['rootdir']) + 'networks/hyperuniform_source/hyperuniform_N' + sizestr +
                            '/out_d' + str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)
        xy, NL, KL, BL, PVxydict = blf.kagomevoronoi_periodic_network_from_pts(points, LL, BBox=BBox, check=lp['check'])
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        originstr = ''
        periodicBCstr = '_periodic'

    # Rescale so that median bond length is unity
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    scale = 1. / np.median(bL)
    xy *= scale
    polygon *= scale
    BBox *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['periodicBC']:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())

    lattice_exten = 'kagome_huvor_' + lp['shape'] + periodicBCstr + '_d{0:02d}'.format(int(lp['conf'])) + originstr
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten
