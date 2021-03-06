import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.data_handling as dh
import lepm.le_geometry as leg
import lepm.stringformat as sf

'''
'''


def build_randorg_gamma_spread_hexner(lp):
    """Construct a network, (periodic or not) (centroidal, kagomized, or triangulated) from the power-law kicked point
    set made by Daniel Hexner.

    Parameters
    ----------
    lp

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten
    """
    NV = lp['NV']
    NH = lp['NH']
    shape = lp['shape']
    check = lp['check']
    networkdir = lp['rootdir'] + 'networks/'
    lattice_type = lp['LatticeTop']
    gamma_tag = lattice_type.split('gamma')[-1][0:4]
    relax_tag = '{0:02d}'.format(int(lp['spreading_time'] * 100))

    if lp['periodicBC']:
        points = np.loadtxt(networkdir + 'random_organization_source/random_kick_gamma/gamma' + gamma_tag + \
                            '/gamma_kick_' + relax_tag + 'relax/' +
                            'L' + str(lp['NP_load']) + '/out_d' + '{0:02d}'.format(int(lp['conf']))
                            + '_xy.txt')
        points -= 0.5 * np.array([float(NV), float(NH)])

        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        bbox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])

        if 'cent' in lattice_type:
            xy, NL, KL, BL, PVxydict = le.delaunay_centroid_rect_periodic_network_from_pts(points, LL, BBox=bbox,
                                                                                           check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
            originstr = ''

        periodicBCstr = '_periodic'

    else:
        points = np.loadtxt(networkdir + 'random_organization_source/random_kick_gamma/gamma' + gamma_tag + \
                            '/gamma_kick_' + relax_tag + 'relax/' +
                            'L400_original/' + 'out_d' + '{0:02d}'.format(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)

        if 'cent' in lattice_type:
            bbox = blf.auto_polygon(shape, NH, NV, eps=0.00)
            polygon_initial = blf.auto_polygon(shape, NH * 2 + 10, NV * 2 + 10, eps=0.00)
            points, trash1, trash2, trash3 = blf.mask_with_polygon(polygon_initial, NH * 3, NV * 3, points,
                                                                   [], eps=0.00)
            xytmp, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(points, polygon='auto', check=check,
                                                                      trimbound=False)
            # WARNING: The convention used to be to use real distance for the Hexner randorg networks, commented below
            # Rescale so that median bond length is unity --> note that this is fine to do here, since
            # later we do it again but it will have almost no effect at all.
            # bL = le.bond_length_list(xytmp, BL, NL=NL, KL=KL, PVx=[], PVy=[])
            # xytmp *= 1. / np.median(bL)
            keep = dh.inds_in_polygon(xytmp, bbox)
            xy, NL, KL, BL = le.remove_pts(keep, xytmp, BL, NN='min', check=lp['check'], PVxydict=None, PV=None)
        else:
            xytmp, trash1, trash2, trash3 = blf.mask_with_polygon(shape, NH, NV, points, [], eps=0.00)
            bbox = blf.auto_polygon(shape, NH, NV, eps=0.00)

        if check:
            le.display_lattice_2D(xy, BL)

        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        PVxydict = {}
        PVx, PVy = [], []
        UC = np.array([0, 0])
        BBox = polygon  # Rescale so that median bond length is unity
        periodicBCstr = ''

    LVUC = 'none'
    LV = 'none'
    UC = 'none'
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

    lattice_exten = lattice_type + '_' + lp['shape'] + periodicBCstr + \
                    '_spreadt' + sf.float2pstr(lp['spreading_time'], ndigits=3) + \
                    '_d' + '{0:02d}'.format(int(lp['conf']))

    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def build_randorg_gamma_spread(lp):
    """Construct a network, (periodic or not) (centroidal, kagomized, or triangulated) from the power-law kicked point
    set generated by npm. Note that this requires the key 'kicksz' in lp, which is the average of the log kick size.
    For this function, lp['LatticeTop'] must be of the form 'randorg_gammakick0p20_cent'-- that is, the value of gamma
    is encoded in the LatticeTop value, and info on whether we should voronoize the network is signalled by '_cent'

    Parameters
    ----------
    lp

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten
    """
    print 'build_randorg_gamma: lp[kicksz] = ', lp['kicksz']
    # sys.exit()
    # Figure out dimensions based on whether NH, NV were specified or NP_load was specified
    # if NH and NV are the wrong size wrt NP, fix that below
    if lp['NP_load'] > 0 and lp['periodicBC'] and not lp['periodic_strip'] and lp['NP_load'] != lp['NV'] * lp['NH']:
        lp['NV'] = int(np.sqrt(lp['NP_load']))
        lp['NH'] = int(np.sqrt(lp['NP_load']))
    elif lp['periodic_strip']:
        # Sample is periodic strip
        # raise RuntimeError("Haven't coded periodic strip gammakick yet")
        lp['periodicBC'] = True

    NV = lp['NV']
    NH = lp['NH']

    # print 'lp[periodic] = ', lp['periodicBC']d
    shape = lp['shape']
    check = lp['check']
    networkdir = lp['rootdir'] + 'networks/'
    lattice_type = lp['LatticeTop']
    gamma_tag = lattice_type.split('kick')[-1][0:4]
    relax_tag = '{0:02d}'.format(int(lp['spreading_time'] * 100))

    if lp['periodic_strip']:
        # raise RuntimeError('Write code for handling periodic strips here.')
        lp['NP_load'] = NH * NV
        dataloaddir = networkdir + 'random_organization_source/random_kick_gamma_npm/gamma' + gamma_tag + '/' + \
                      'L' + str(int(lp['NP_load'])) + '_nh' + str(int(NH)) + '_nv' + str(int(NV)) + \
                      '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=5) + '/' + \
                      'd' + '{0:02d}'.format(int(lp['conf'])) + \
                      '/gamma_kick_ts01000_dt0p0010/'
        print 'dataloaddir = ', dataloaddir
        points = np.loadtxt(dataloaddir + 'gammakickspreadxy_t' +
                            '{0:08.3f}'.format(lp['spreading_time']).replace('.', 'p') + '.txt')
        # points -= 0.5 * np.array([float(NH), float(NV)])

        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NH'], lp['NV'])
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        sizestr = '{0:03d}'.format(lp['NP_load'])
        PV = np.array([[float(lp['NH']), 0.], [0., float(lp['NV'])]])

        if 'kagcent' in lattice_type:
            raise RuntimeError('Havenot yet made this kagome function below...')
            xy, NL, KL, BL, PVxydict = blf.kagomecentroid_periodicstrip_from_pts(points, LL, BBox=BBox,
                                                                                 check=lp['check'])
            print 'build_randorg_gamma: PVxydict = ', PVxydict
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        elif 'cent' in lattice_type:
            xy, NL, KL, BL, PVxydict = le.delaunay_centroid_periodicstrip_from_pts(points, LL, BBox=BBox,
                                                                                   check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        else:
            xy, NL, KL, BL, PVxydict = le.delaunay_periodicstrip_from_pts(points, PV, BBox=BBox, check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

        periodicBCstr = '_periodicstrip'
    elif lp['periodicBC']:
        lp['NP_load'] = NH * NV
        dataloaddir = networkdir + 'random_organization_source/random_kick_gamma_npm/gamma' + gamma_tag + '/' + \
                      'L' + str(int(lp['NP_load'])) + '_nh' + str(int(NH)) + '_nv' + str(int(NV)) + \
                      '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=5) + '/' + \
                      'd' + '{0:02d}'.format(int(lp['conf'])) + \
                      '/gamma_kick_ts01000_dt0p0010/'
        print 'dataloaddir = ', dataloaddir
        points = np.loadtxt(dataloaddir + 'gammakickspreadxy_t' +
                            '{0:08.3f}'.format(lp['spreading_time']).replace('.', 'p') + '.txt')
        # points -= 0.5 * np.array([float(NH), float(NV)])

        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NH'], lp['NV'])
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])

        sizestr = '{0:03d}'.format(lp['NP_load'])
        PV = np.array([[float(lp['NH']), 0.], [0., float(lp['NV'])]])

        if 'kagcent' in lattice_type:
            xy, NL, KL, BL, PVxydict = blf.kagomecentroid_periodic_network_from_pts(points, LL, BBox=BBox,
                                                                                    check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        elif 'cent' in lattice_type:
            xy, NL, KL, BL, PVxydict = le.delaunay_centroid_rect_periodic_network_from_pts(points, LL, BBox=BBox,
                                                                                           check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        else:
            xy, NL, KL, BL, PVxydict = le.delaunay_periodic_network_from_pts(points, PV, BBox=BBox, check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

        periodicBCstr = '_periodic'

    else:
        dataloaddir = networkdir + 'random_organization_source/random_kick_gamma_npm/gamma' + gamma_tag + '/' + \
                      'L' + str(int(lp['NP_load'])) + '_nh' + str(NH) + '_nv' + str(int(NV)) + \
                      '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=5) + '/' + \
                      'd' + '{0:02d}'.format(int(lp['conf'])) + \
                      '/gamma_kick_ts01000_dt' + sf.float2pstr(lp['spreading_dt'], ndigits=4) + '/'
        points = np.loadtxt(dataloaddir + 'gammakickspreadxy_t' +
                            '{0:08.3f}'.format(lp['spreading_time']).replace('.', 'p') + '.txt')

        # Check for nans
        bad = np.where(np.logical_or(np.isnan(points[:, 0]), np.isnan(points[:, 1])))[0]
        if len(bad) > 0:
            print 'WARNING: found nans in loaded pointset, removing...'
            keep = np.setdiff1d(np.arange(len(points)), bad)
            points = points[keep]

        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NH'], lp['NV'])
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        sizestr = '{0:06d}'.format(lp['NP_load'])
        # pattern the points around the original sample so that the edges are ok
        points = le.buffer_points_for_rectangular_periodicBC(points, LL)

        if 'cent' in lattice_type:
            polygon_initial = blf.auto_polygon(shape, NH * 2 + 10, NV * 2 + 10, eps=0.00)
            points, trash1, trash2, trash3 = blf.mask_with_polygon(polygon_initial, NH * 3, NV * 3, points,
                                                                   [], eps=0.00)
            xytmp, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(points, polygon='auto', check=check,
                                                                      trimbound=False)
            # Now cut down the patterned/tiled network to original size, severing the periodic bonds
            keep = dh.inds_in_polygon(xytmp, BBox)
            xy, NL, KL, BL, PVxydict = le.remove_pts(keep, xytmp, BL, NN='min', check=lp['check'],
                                                     PVxydict=None, PV=None)
            # print 'np.max(xy[:, 0]) = ', np.max(xy[:, 0])
            # sys.exit()
        else:
            xy, trash1, trash2, trash3 = blf.mask_with_polygon(BBox, NH, NV, points, [], eps=0.00)

        if check:
            le.display_lattice_2D(xy, BL)

        PVxydict = {}
        PVx = []
        PVy = []
        UC = np.array([0, 0])
        periodicBCstr = ''

    # Rescale so that median bond length is unity
    LVUC = 'none'
    LV = 'none'
    UC = 'none'
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    scale = 1. / np.median(bL)
    xy *= scale
    BBox *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['periodicBC'] or lp['periodic_strip']:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())

    lattice_exten = lattice_type + '_' + lp['shape'] + periodicBCstr + \
                    '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=3) +\
                    '_spreadt' + sf.float2pstr(lp['spreading_time'], ndigits=3) + \
                    '_d' + '{0:02d}'.format(int(lp['conf']))
    if not lp['periodicBC']:
        lattice_exten += '_NP' + sizestr

    # print 'PVxydict = ', PVxydict
    # print 'PV = ', PV
    # print 'BBox = ', BBox
    # sys.exit()

    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten
