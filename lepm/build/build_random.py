import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le

'''Functions for building randomly-positioned networks'''


def build_randomcent(lp):
    """Build uniformly random point set, and construct voronoized network from that."""
    NH = lp['NH']
    NV = lp['NV']
    LL = (NH, NV)
    if lp['periodicBC']:
        xx = np.random.uniform(low=-NH*0.5, high=NH*0.5, size=NH * NV)
        yy = np.random.uniform(low=-NV*0.5, high=NV*0.5, size=NH * NV)
        xy = np.dstack((xx, yy))[0]
        xy, NL, KL, BL, PVxydict = \
            le.delaunay_centroid_rect_periodic_network_from_pts(xy, LL, BBox='auto', check=lp['check'])
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

        # check that all points are inside BBox
        shapedict = {lp['shape']: [NH, NV]}
        keep = argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        if len(np.where(keep)[0]) != len(xy):
            raise RuntimeError('Some points were spuriously outside the allowed BBox.')
        polygon = auto_polygon(lp['shape'], NH, NV, eps=0.00)
        perstr = '_periodic'
    else:
        xx = np.random.uniform(low=-NH*0.5-10, high=NH*0.5+10, size=(NH+20)*(NV+20))
        yy = np.random.uniform(low=-NV*0.5-10, high=NV*0.5+10, size=(NH+20)*(NV+20))
        xy = np.dstack((xx, yy))[0]
        xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xy, polygon=None, trimbound=False, check=lp['check'])

        # Crop to polygon
        shapedict = {lp['shape']: [NH, NV]}
        keep = argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        polygon = auto_polygon(lp['shape'], NH, NV, eps=0.00)
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


def build_kagome_randomcent(lp):
    """Build uniformly random point set, and construct voronoized network from that.

    lp : dict
        lattice parameters dictionary
    """
    NH = lp['NH']
    NV = lp['NV']
    LL = (NH, NV)
    if lp['periodicBC']:
        xx = np.random.uniform(low=-NH * 0.5, high=NH * 0.5, size=NH * NV)
        yy = np.random.uniform(low=-NV * 0.5, high=NV * 0.5, size=NH * NV)
        xy = np.dstack((xx, yy))[0]
        xy, NL, KL, BL, PVxydict = \
            kagomecentroid_periodic_network_from_pts(xy, LL, BBox='auto', check=lp['check'])
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

        # check that all points are inside BBox
        shapedict = {lp['shape']: [NH, NV]}
        keep = argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        if len(np.where(keep)[0]) != len(xy):
            raise RuntimeError('Some points were spuriously outside the allowed BBox.')
        polygon = auto_polygon(lp['shape'], NH, NV, eps=0.00)
        perstr = '_periodic'
    else:
        xx = np.random.uniform(low=-NH * 0.5 - 10, high=NH * 0.5 + 10, size=(NH + 20) * (NV + 20))
        yy = np.random.uniform(low=-NV * 0.5 - 10, high=NV * 0.5 + 10, size=(NH + 20) * (NV + 20))
        xy = np.dstack((xx, yy))[0]

        xy, NL, KL, BL = kagomecentroid_lattice_from_pts(xy, polygon=None, trimbound=False, check=lp['check'])

        # Crop to polygon
        shapedict = {lp['shape']: [NH, NV]}
        keep = argcrop_lattice_to_polygon(shapedict, xy, check=lp['check'])
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        polygon = auto_polygon(lp['shape'], NH, NV, eps=0.00)
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