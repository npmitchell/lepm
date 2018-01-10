import numpy as np
import matplotlib.pyplot as plt
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.data_handling as dh

#
# def build_hucentroid(lp):
#     print('Loading hyperuniform to build lattice...')
#     if lp['NP_load'] == 0 and not lp['periodicBC']:
#         points = np.loadtxt(
#             networkdir + 'hyperuniform_source/hyperuniform_N400/out_d' + str(int(lp['conf'])) + '_xy.txt')
#         points -= np.mean(points, axis=0) + lp['origin']
#         addpc = .05
#         keep = np.logical_and(abs(points[:, 0]) < (5 + NH * (1 + addpc) * 0.5),
#                               abs(points[:, 1]) < (5 + NV * (1 + addpc)) * 0.5)
#         xytmp = points[keep]
#         if check:
#             plt.plot(points[:, 0], points[:, 1], 'b.')
#             plt.title('Point set before initial cutting')
#             plt.show()
#         polygon = auto_polygon(shape, NH, NV, eps=0.00)
#         xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xytmp, polygon=polygon, trimbound=False, check=check)
#         if check:
#             le.display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Cropped centroid lattice, before dilation')
#
#         # Form output bounding box and extent measurement
#         LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
#         PVxydict = {}
#         PVx = []
#         PVy = []
#         BBox = polygon
#
#         # make string from origin values
#         print 'lp[origin] = ', lp['origin']
#         print 'type(lp[origin]) = ', type(lp['origin'])
#         if (np.abs(lp['origin']) < 1e-7).all():
#             originstr = ''
#         else:
#             originstr = '_originX' + '{0:0.2f}'.format(lp['origin'][0]).replace('.', 'p') + \
#                         'Y' + '{0:0.2f}'.format(lp['origin'][1]).replace('.', 'p')
#         periodicstr = ''
#         stripstr = ''
#     else:
#         if lp['periodic_strip']:
#             lp['periodicBC'] = True
#             sizestr = '{0:03d}'.format(lp['NP_load'])
#             print 'sizestr = ', sizestr
#             points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' +
#                                 str(int(lp['conf'])) + '_xy.txt')
#             print 'points = ', points
#             points -= np.mean(points, axis=0)
#             # Ensuring that origin param is centered (since using entire lattice)
#             lp['origin'] = np.array([0.0, 0.0])
#             if NH != lp['NP_load']:
#                 raise RuntimeError('NP_load should be equal to NH for a periodicstrip geometry!')
#             LL = (NH, NV)
#             polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
#             BBox = polygon
#             keep = dh.inds_in_polygon(points, polygon)
#             points = points[keep]
#             xy, NL, KL, BL, PVxydict = le.delaunay_centroid_periodicstrip_from_pts(points, LL, BBox='auto',
#                                                                                    check=check)
#             PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
#             originstr = ''
#             periodicstr = '_periodicstrip'
#             stripstr = '_NH{0:06d}'.format(NH) + '_NV{0:06d}'.format(NV)
#         else:
#             lp['periodicBC'] = True
#             sizestr = '{0:03d}'.format(lp['NP_load'])
#             print 'sizestr = ', sizestr
#             points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' +
#                                 str(int(lp['conf'])) + '_xy.txt')
#             print 'points = ', points
#             points -= np.mean(points, axis=0)
#             # Ensuring that origin param is centered (since using entire lattice)
#             lp['origin'] = np.array([0.0, 0.0])
#             LL = (lp['NP_load'], lp['NP_load'])
#             polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
#             BBox = polygon
#             xy, NL, KL, BL, PVxydict = le.delaunay_centroid_rect_periodic_network_from_pts(points, LL, BBox='auto',
#                                                                                            check=check)
#             PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
#             originstr = ''
#             periodicstr = '_periodic'
#             stripstr = ''
#
#     # Rescale so that median bond length is unity
#     bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
#     scale = 1. / np.median(bL)
#     xy *= scale
#     polygon *= scale
#     BBox *= scale
#     LL = (LL[0] * scale, LL[1] * scale)
#     if lp['NP_load'] != 0:
#         PVx *= scale
#         PVy *= scale
#         PVxydict.update((key, val * scale) for key, val in PVxydict.items())
#
#     lattice_exten = 'hucentroid_' + shape + periodicstr + '_d' + \
#                     '{0:02d}'.format(int(lp['conf'])) + stripstr + originstr
#     LVUC = 'none'
#     LV = 'none'
#     UC = 'none'


def build_hucentroid(lp):
    """Load the proper hyperuniform point set, given the parameters in dict lp

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten
    """
    check = lp['check']
    networkdir = lp['rootdir']+'networks/'
    print('Loading hyperuniform to build lattice...')
    if lp['NP_load'] == 0:
        points = np.loadtxt(networkdir+'hyperuniform_source/hyperuniform_N400/out_d'+str(int(lp['conf']))+'_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        addpc = .05
        # Note: below we crop a large region so that if the network has shape==circle, we dont cut off the sides
        if lp['shape'] == 'circle':
            keep = np.logical_and(abs(points[:, 0]) < (10 + lp['NH'] * (1 + addpc)),
                                  abs(points[:, 1]) < (10 + lp['NV'] * (1 + addpc)))
        else:
            # it will speed things up to crop more, so do so if the shape is not a circle
            keep = np.logical_and(abs(points[:, 0]) < (5 + lp['NH'] * (1 + addpc) * 0.5),
                                  abs(points[:, 1]) < (5 + lp['NV'] * (1 + addpc)) * 0.5)
        xytmp = points[keep]
        if check:
            plt.plot(points[:, 0], points[:, 1], 'b.')
            plt.title('Point set before initial cutting')
            plt.show()
        polygon = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)
        xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xytmp, polygon=polygon, trimbound=False, check=check)
        if check:
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
        stripstr = ''
    else:
        if lp['periodic_strip']:
            lp['periodicBC'] = True
            sizestr = '{0:03d}'.format(lp['NP_load'])
            print 'sizestr = ', sizestr
            points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' +
                                str(int(lp['conf'])) + '_xy.txt')
            print 'points = ', points
            points -= 0.5 * np.array([lp['NH'], lp['NV']])
            # Ensuring that origin param is centered (since using entire lattice)
            lp['origin'] = np.array([0.0, 0.0])
            if lp['NH'] != lp['NP_load']:
                raise RuntimeError('NP_load should be equal to NH for a periodicstrip geometry!')
            LL = (lp['NH'], lp['NV'])
            polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
            BBox = polygon
            keep = dh.inds_in_polygon(points, polygon)
            points = points[keep]
            xy, NL, KL, BL, PVxydict = le.delaunay_centroid_periodicstrip_from_pts(points, LL, BBox='auto',
                                                                                   check=check)
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
            originstr = ''
            periodicstr = '_periodicstrip'
            stripstr = '_NH{0:06d}'.format(lp['NH']) + '_NV{0:06d}'.format(lp['NV'])
        else:
            lp['periodicBC'] = True
            sizestr = '{0:03d}'.format(lp['NP_load'])
            print 'sizestr = ', sizestr
            points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' +
                                str(int(lp['conf'])) + '_xy.txt')
            print 'points = ', points
            points -= np.mean(points, axis=0)
            # Ensuring that origin param is centered (since using entire lattice)
            lp['origin'] = np.array([0.0, 0.0])
            LL = (lp['NP_load'], lp['NP_load'])
            polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
            BBox = polygon
            xy, NL, KL, BL, PVxydict = le.delaunay_centroid_rect_periodic_network_from_pts(points, LL, BBox='auto',
                                                                                           check=check)
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
            originstr = ''
            periodicstr = '_periodic'
            stripstr = ''

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

    lattice_exten = 'hucentroid_' + lp['shape'] + periodicstr + '_d' +\
                    '{0:02d}'.format(int(lp['conf'])) + stripstr + originstr
    LVUC = 'none'
    LV = 'none'
    UC = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten


def build_hucentroid_annulus(lp):
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
    xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten = build_hucentroid(lp)
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


def build_kagome_hucent(lp):
    print('Loading hyperuniform to build lattice...')
    networkdir = lp['rootdir'] + 'networks/'
    shape = lp['shape']
    NH = lp['NH']
    NV = lp['NV']
    check = lp['check']
    if lp['NP_load'] == 0:
        points = np.loadtxt(
            networkdir + 'hyperuniform_source/hyperuniform_N400/out_d' + str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        addpc = .05
        xytmp, trash1, trash2, trash3 = blf.mask_with_polygon(shape, NH + 4, NV + 4, points, [], eps=addpc)
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)

        xy, NL, KL, BL = blf.kagomecentroid_lattice_from_pts(xytmp, polygon=polygon, trimbound=False, check=check)

        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)

        if check:
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
        periodicBCstr = ''
    else:
        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        BBox = polygon

        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])
        points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' + \
                            str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)
        xy, NL, KL, BL, PVxydict = blf.kagomecentroid_periodic_network_from_pts(points, LL,
                                                                                BBox=BBox, check=lp['check'])
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

    lattice_exten = 'kagome_hucent_' + shape + periodicBCstr + '_d{0:02d}'.format(int(lp['conf'])) + originstr
    LV = 'none'
    LVUC = 'none'
    UC = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def build_kagome_hucent_annulus(lp):
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
    xy, NL, KL, BL, PVxydict, PVx, PVy, LVUC, BBox, LL, LV, UC, lattice_exten = build_kagome_hucent(lp)
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


def build_kagper_hucent(lp):
    """Build a hyperuniform centroidal lattice with some density of kagomization (kagper = kagome percolation)

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_hucentroid(lp)

    # Select some fraction of vertices (which are points) --> xypick gives Nkag of the vertices (xy)
    Nkag = round(lp['percolation_density'] * len(xy))
    ind_shuffled = np.random.permutation(np.arange(len(xy)))
    xypick = np.sort(ind_shuffled[0:Nkag])

    xy, BL = blf.decorate_kagome_elements(xy, BL, xypick, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # If the meshfn going to overwrite a previous realization?
    mfok = le.meshfn_is_used(le.build_meshfn(lp)[0])
    while mfok:
        lp['subconf'] += 1
        mfok = le.meshfn_is_used(le.build_meshfn(lp)[0])

    lattice_exten = 'kagper_hucent' + lattice_exten[10:] + \
                    '_perd' + sf.float2pstr(lp['percolation_density'], ndigits=2) + \
                    '_r' + '{0:02d}'.format(int(lp['subconf']))
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_accordionkag_hucent(lp):
    """Create an accordion-bonded kagomized amorphous network from a loaded hyperuniform point set.

    Parameters
    ----------
    lp

    Returns
    -------

    """
    print('Loading hyperuniform to build lattice...')
    networkdir = lp['rootdir'] + 'networks/'
    shape = lp['shape']
    NH = lp['NH']
    NV = lp['NV']
    check = lp['check']
    if lp['NP_load'] == 0:
        points = np.loadtxt(
            networkdir + 'hyperuniform_source/hyperuniform_N400/out_d' + str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0) + lp['origin']
        addpc = .05
        xytmp, trash1, trash2, trash3 = blf.mask_with_polygon(shape, NH + 4, NV + 4, points, [], eps=addpc)
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)

        xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xytmp, polygon=polygon, trimbound=False)  # check=check)
        #################################################################
        # nzag controlled by lp['intparam'] below
        xyacc, BLacc, LVUC, UC, xyvertices, lattice_exten_add = \
            blf.accordionize_network(xy, BL, lp, PVxydict=None, PVx=None, PVy=None, PV=None)

        print 'BL = ', BL

        # need indices of xy that correspond to xyvertices
        # note that xyvertices gives the positions of the vertices, not their indices
        inRx = np.in1d(xyacc[:, 0], xyvertices[:, 0])
        inRy = np.in1d(xyacc[:, 1], xyvertices[:, 1])
        vxind = np.where(np.logical_and(inRx, inRy))[0]
        print 'vxind = ', vxind

        # Note: beware, do not provide NL and KL to decorate_bondneighbors_elements() since NL,KL need
        # to be recalculated
        xy, BL = blf.decorate_bondneighbors_elements(xyacc, BLacc, vxind, PVxydict=None, viewmethod=False,
                                                     check=lp['check'])
        NL, KL = le.BL2NLandKL(BL, NP=len(xy))

        #################################################################
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)

        if check:
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
        periodicBCstr = ''
    else:
        # Ensuring that origin param is centered (since using entire lattice)
        lp['origin'] = np.array([0.0, 0.0])
        LL = (lp['NP_load'], lp['NP_load'])
        polygon = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        BBox = polygon

        lp['periodicBC'] = True
        sizestr = '{0:03d}'.format(lp['NP_load'])
        points = np.loadtxt(networkdir + 'hyperuniform_source/hyperuniform_N' + sizestr + '/out_d' + \
                            str(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)
        xy, NL, KL, BL, PVxydict = blf.kagomecentroid_periodic_network_from_pts(points, LL,
                                                                                BBox=BBox, check=lp['check'])
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

    lattice_exten = 'accordionkag_hucent_' + shape + periodicBCstr + '_d{0:02d}'.format(int(lp['conf'])) + originstr + \
                    lattice_exten_add
    LV = 'none'
    LVUC = 'none'
    UC = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


