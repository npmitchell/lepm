import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.build.build_hexagonal as bhex
import copy


def build_hex_kagframe(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = bhex.generate_honeycomb_lattice(lp)
    max_x = np.max(xy[:, 0])
    max_y = np.max(xy[:, 1])
    min_x = np.min(xy[:, 0])
    min_y = np.min(xy[:, 1])
    LL = (max_x - min_x, max_y - min_y)
    BBox = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    eps = 1e-9
    if lp['shape'] == 'square':
        lenscaleX = np.max(np.abs(BBox[:, 0])) * lp['alph'] + eps
        lenscaleY = np.max(np.abs(BBox[:, 1])) * lp['alph'] + eps
        kaginds = np.where(np.logical_or(np.abs(xy[:, 0]) > lenscaleX, np.abs(xy[:, 1]) > lenscaleY))[0]
    elif lp['shape'] == 'circle':
        # todo: handle circles
        pass
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = 'hex_kagframe' + lattice_exten[9:] +\
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_hex_kagcframe(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    hclp = copy.deepcopy(lp)
    hclp['eta'] = 0.0
    xy, tr1, tr2, BL, tr3, tr4, tr5, PVxydict, PVx, PVy, PV, lattice_exten = bhex.generate_honeycomb_lattice(hclp)
    max_x = np.max(xy[:, 0])
    max_y = np.max(xy[:, 1])
    min_x = np.min(xy[:, 0])
    min_y = np.min(xy[:, 1])
    LL = (max_x - min_x, max_y - min_y)
    BBox = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    eps = 1e-9
    lenscale = np.max(np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)) * lp['alph'] + eps
    print "lp['alph'] = ", lp['alph']
    print "lp['alph'] * np.abs(BBox[:, 0])) = ", lp['alph'] * np.abs(BBox[:, 0])
    print 'lenscale = ', lenscale
    kaginds = np.where(np.sqrt(xy[:, 0]**2 + xy[:, 1]**2) > lenscale)[0]

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # Only randomly displace the gyros in frame, and only if eta >0
    if lp['eta'] > 0.0:
        if 'eta_alph' not in lp:
            lp['eta_alph'] = lp['alph']
        eta_lenscale = np.max(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)) * lp['eta_alph'] + eps
        print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\neta_lenscale = ', eta_lenscale
        print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\neta_alph = ', lp['eta_alph']
        etainds = np.where(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2) > eta_lenscale)[0]
        displ = lp['eta'] * (np.random.rand(len(etainds), 2) - 0.5)
        xy[etainds, :] += displ
        addstr = '_eta' + sf.float2pstr(lp['eta'], ndigits=3)
        addstr += '_etaalph' + sf.float2pstr(lp['eta_alph'], ndigits=3)
    else:
        addstr = ''

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = 'hex_kagcframe' + lattice_exten[9:] +\
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=2) + addstr
    LV = 'none'
    UC = 'none'
    LVUC = 'none'
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_hex_kagperframe(lp):
    """Build a hyperuniform centroidal lattice with partially kagomized points beyond a distance
    alph*Radius/Halfwidth of sample. "per" here means "percolation", referring to the randomly added kagomized elements.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = bhex.generate_honeycomb_lattice(lp)
    max_x = np.max(xy[:, 0])
    max_y = np.max(xy[:, 1])
    min_x = np.min(xy[:, 0])
    min_y = np.min(xy[:, 1])
    LL = (max_x - min_x, max_y - min_y)
    BBox = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    if lp['shape'] == 'square':
        lenscaleX = np.max(BBox[:, 0]) * lp['alph']
        lenscaleY = np.max(BBox[:, 1]) * lp['alph']
        kaginds = np.where(np.logical_or(np.abs(xy[:, 0]) > lenscaleX, np.abs(xy[:, 1]) > lenscaleY))[0]
    elif lp['shape'] == 'circle':
        # todo: handle circles
        pass
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    # Select some fraction of vertices (which are points) --> xypick gives Nkag of the vertices (xy)
    Nkag = round(lp['percolation_density'] * len(kaginds))
    ind_shuffled = np.random.permutation(np.arange(len(kaginds)))
    xypick = np.sort(ind_shuffled[0:Nkag])

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds[xypick], viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = 'hex_kagperframe' + lattice_exten[9:] +\
                    '_perd' + sf.float2pstr(lp['percolation_density'], ndigits=2) +\
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_hucent_kagframe(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    from lepm.build import build_hucentroid
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_hucentroid.build_hucentroid(lp)

    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    if 'kagframe' in lp['LatticeTop']:
        lenscaleX = np.max(BBox[:, 0]) * lp['alph']
        lenscaleY = np.max(BBox[:, 1]) * lp['alph']
        kaginds = np.where(np.logical_or(np.abs(xy[:, 0]) > lenscaleX, np.abs(xy[:, 1]) > lenscaleY))[0]
    elif 'kagcframe' in lp['LatticeTop']:
        eps = 1e-9
        lenscale = np.max(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)) * lp['alph'] + eps
        kaginds = np.where(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2) > lenscale)[0]
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:]  +\
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_isocent_kagframe(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    from lepm.build import build_iscentroid
    # if lp['NP_load'] < 1:
    #     lp['NH'] += 5
    #     lp['NV'] += 5

    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_iscentroid.build_iscentroid(lp)
    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    if 'kagframe' in lp['LatticeTop']:
        lenscaleX = np.max(BBox[:, 0]) * lp['alph']
        lenscaleY = np.max(BBox[:, 1]) * lp['alph']
        kaginds = np.where(np.logical_or(np.abs(xy[:, 0]) > lenscaleX, np.abs(xy[:, 1]) > lenscaleY))[0]
    elif 'kagcframe' in lp['LatticeTop']:
        eps = 1e-9
        lenscale = np.max(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)) * lp['alph'] + eps
        kaginds = np.where(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2) > lenscale)[0]
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    # if trim_after:
    #     print 'trim here'

    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # name the output network
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kaghu_centframe(lp):
    """Build a hyperuniform centroidal lattice with kagomized points inside distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhex.build_hucentroid(lp)

    # Grab indices (vertices) to kagomize: select the ones farther than alph*characteristic length from center
    if lp['shape'] == 'square':
        lenscaleX = np.max(BBox[:, 0]) * lp['alph']
        lenscaleY = np.max(BBox[:, 1]) * lp['alph']
        kaginds = np.where(np.logical_and(np.abs(xy[:, 0]) < lenscaleX, np.abs(xy[:, 1]) < lenscaleY))[0]
    elif lp['shape'] == 'circle':
        # todo: handle circles
        pass
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, NL=NL, PVxydict=PVxydict, viewmethod=lp['viewmethod'],
                                          check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        # The ith row of PV is the vector taking the ith side of the polygon (connecting polygon[i] to
        # polygon[i+1 % len(polygon)]
        PV = np.array([[LL[0], 0.0], [LL[0], LL[1]], [LL[0], -LL[1]],
                       [0.0, 0.0], [0.0, LL[1]], [0.0, -LL[1]],
                       [-LL[0], 0.0], [-LL[0], LL[1]], [-LL[0], -LL[1]]])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = 'kaghu_centframe' + lattice_exten[10:] +\
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp

