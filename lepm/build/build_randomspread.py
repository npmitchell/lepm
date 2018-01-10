import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le

'''Funcitons for building random networks from point sets with randomly positioned points subject to repulsive force
for some time'''


def build_randomspreadcent(lp):
    """Build point set based on uniformly random points spread apart with repulsive force, and construct voronoized
    network from that."""
    NH = lp['NH']
    NV = lp['NV']
    LL = (NH, NV)
    if lp['periodicBC']:
        sourcedir = lp['rootdir'] + 'networks/random_organization_source/random_spread_unpacked/L' \
                    + str(lp['NP_load']) + '/'
        confdir = sourcedir + 'L' + str(lp['NP_load']) + '_d{0:02d}'.format(lp['conf']) + '/'
        xy = np.loadtxt(confdir + 'spreadingxy_t{0:06d}'.format(int(lp['spreading_time']*100)) + '.txt')
        xy -= np.array([0.5*float(lp['NP_load']), 0.5*float(lp['NP_load'])])
        xy, NL, KL, BL, PVxydict = \
            le.delaunay_centroid_rect_periodic_network_from_pts(xy, (lp['NP_load'], lp['NP_load']),
                                                                BBox='auto', check=False)

        if lp['periodic_strip']:
            # Define new Periodic Vectors list (PV) to be just horizontal vectors
            PV = np.array([[-NH, 0], [NH, 0]])

            # Remove bonds across the vertical periodic edges
            BL = blf.remove_vertical_periodicity(BL, PVxydict)
            PVxydict = le.BL2PVxydict(BL, xy, PV)
            NL, KL = le.BL2NLandKL(BL)
            if lp['check']:
                le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVxydict=PVxydict, colorz=True,
                                      title='Cropped centroid lattice preparing as strip, before cutting to NH x NV')

            # Warn user if not removing any points when making the strip
            shapedict = {lp['shape']: [NH, NV]}
            keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
            if len(np.where(keep)[0]) == len(xy):
                print 'Warning: Strip dimensions (BBox) was larger than or equal to periodic sample.'
                stripstr = ''
            else:
                # Remove points above and below the strip
                xy, NL, KL, BL = le.remove_pts(keep, xy, BL, check=lp['check'], PVxydict=PVxydict, PV=PV)
                PVxydict = le.BL2PVxydict(BL, xy, PV)
                stripstr = '_NH{0:06d}'.format(NH) + '_NV{0:06d}'.format(NV)

            print 'PVxydict = ', PVxydict
            polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
            perstr = '_periodicstrip'
        else:
            # check that all points are inside BBox
            shapedict = {lp['shape']: [NH, NV]}
            keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
            if len(np.where(keep)[0]) != len(xy):
                raise RuntimeError('Some points were spuriously outside the allowed BBox.')
            polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
            perstr = '_periodic'
            stripstr = ''

        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
    else:
        # Only certain sizes of networks are available, pick one bigger than dims
        avail_sizes = np.array([20, 50, 100])
        ind = np.where(np.logical_and(avail_sizes > NH+10, avail_sizes > NV+10))[0]
        sz = np.min(avail_sizes[ind])
        print 'sz = ', sz
        print 'lp[conf] = ', lp['conf']
        sourcedir = lp['rootdir'] + 'networks/random_organization_source/random_spread_unpacked/L' + str(int(sz)) + '/'
        confdir = sourcedir + 'L' + str(int(sz)) + '_d{0:02d}'.format(lp['conf']) + '/'
        xy = np.loadtxt(confdir + 'spreadingxy_t{0:06d}'.format(int(lp['spreading_time'] * 100)) + '.txt')
        xy -= np.array([0.5 * float(sz), 0.5 * float(sz)])
        xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xy, polygon=None, trimbound=False, check=lp['check'])

        # Crop to polygon
        shapedict = {lp['shape']: [NH, NV]}
        keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
        PVxydict = {}
        PVx = []
        PVy = []
        perstr = ''
        stripstr = ''

    if lp['check']:
        le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, PVxydict=PVxydict,
                              title='Cropped centroid lattice, before dilation')

    lattice_exten = lp['LatticeTop'] + '_' + lp['shape'] + perstr + '_r' + '{0:02d}'.format(int(lp['conf'])) + \
                    '_spreadt{0:0.3f}'.format(lp['spreading_time']).replace('.', 'p') + stripstr

    # Rescale so that median bond length is unity
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    scale = 1./np.median(bL)
    xy *= scale
    polygon *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['periodicBC']:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())
    LVUC = 'none'
    LV = 'none'
    UC = 'none'
    BBox = polygon
    PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def build_kagome_randomspread(lp):
    """Build uniformly random point set, and construct voronoized network from that.

    lp : dict
        lattice parameters dictionary
    """
    NH = lp['NH']
    NV = lp['NV']
    LL = (NH, NV)
    if lp['periodicBC']:
        sourcedir = lp['rootdir'] + 'networks/random_organization_source/random_spread_unpacked/L' \
                    + str(lp['NP_load']) + '/'
        confdir = sourcedir + 'L' + str(lp['NP_load']) + '_d{0:02d}'.format(lp['conf']) + '/'
        xy = np.loadtxt(confdir + 'spreadingxy_t{0:05d}'.format(int(lp['spreading_time']*100)) + '.txt')
        xy -= np.array([0.5*float(lp['NP_load']), 0.5*float(lp['NP_load'])])
        xy, NL, KL, BL, PVxydict = \
            blf.kagomecentroid_periodic_network_from_pts(xy,  (lp['NP_load'], lp['NP_load']),
                                                     BBox='auto', check=lp['check'])

        if lp['periodic_strip']:
            shapedict = {lp['shape']: [NH, NV]}
            keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
            if len(np.where(keep)[0]) == len(xy):
                raise RuntimeError('Strip dimensions (BBox) was larger than or equal to periodic sample.')
            xy, NL, KL, BL = le.remove_pts(keep, xy, BL, check=lp['check'])
            PV = np.array([[-NH, 0], [0, NH]])
            PVxydict = le.BL2PVxydict(BL, xy, PV)
            polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
            perstr = '_periodicstrip'
        else:
            # check that all points are inside BBox
            shapedict = {lp['shape']: [NH, NV]}
            keep = argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
            if len(np.where(keep)[0]) != len(xy):
                raise RuntimeError('Some points were spuriously outside the allowed BBox.')
            polygon = auto_polygon(lp['shape'], NH, NV, eps=0.00)
            perstr = '_periodic'
    else:
        # Only certain sizes of networks are available, pick one bigger than dims
        avail_sizes = np.array([20, 50, 100])
        ind = np.where(np.logical_and(avail_sizes > NH + 10, avail_sizes > NV + 10))[0]
        sz = avail_sizes[ind]
        sourcedir = lp['rootdir'] + 'networks/random_organization_source/random_spread_unpacked/L' + str(int(sz)) + '/'
        confdir = sourcedir + 'L' + str(lp['NP_load']) + '_d{0:02d}'.format(lp['conf']) + '/'
        xy = np.loadtxt(confdir + 'spreadingxy_t{0:05d}'.format(int(lp['spreading_time'] * 100)) + '.txt')
        xy -= np.array([0.5 * float(lp['NP_load']), 0.5 * float(lp['NP_load'])])
        xy, NL, KL, BL = blf.kagomecentroid_lattice_from_pts(xy, polygon=None, trimbound=False, check=lp['check'])

        # Crop to polygon
        shapedict = {lp['shape']: [NH, NV]}
        keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
        PVxydict = {}
        PVx = []
        PVy = []
        perstr = ''

    # If the meshfn going to overwrite a previous realization?
    mfok = le.meshfn_is_used(le.build_meshfn(lp)[0])
    print 'mfok = ', mfok
    while mfok:
        lp['conf'] += 1
        mfok = le.meshfn_is_used(le.build_meshfn(lp)[0])

    lattice_exten = lp['LatticeTop'] + '_' + lp['shape'] + perstr + '_r' + '{0:02d}'.format(int(lp['conf']))

    # Rescale so that median bond length is unity
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    scale = 1. / np.median(bL)
    xy *= scale
    polygon *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['periodicBC']:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())
    LVUC = 'none'
    LV = 'none'
    UC = 'none'
    BBox = polygon
    PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten