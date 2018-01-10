import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.lattice_class as lattice_class
import lepm.mass_lattice_class as mass_lattice_class
import lepm.gyro_lattice_class as gyro_lattice_class
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
from os import getcwd,chdir
import socket
import argparse
import copy
import scipy
from scipy.spatial import Delaunay
import sys
import glob
import math
import cmath
import shapely.geometry as sg
import descartes

'''
Description
===========
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Example usage:
# This line makes an annulus of hexagonal lattice (conformal lattice) with a width specified by alph and perimiter in
#   bond lengths specified by N.)
python make_lattice.py -LT hexannulus -alph 0.2 -N 20
python ./build/make_lattice.py -LT hexjunction -intparam 50 -skip_polygons -DOSmovie -alph 0.7
python ./build/make_lattice.py -LT kagjunction -intparam 100 -skip_polygons -DOSmovie -alph 0.7

Differences between my code and Lisa's
*Ni --> NL
*Nk --> KL
KL : Correponds to NL matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
'''


def build_lattice(lattice):
    """Build the lattice based on the values stored in the lp dictionary attribute of supplied lattice instance.

    Returns
    -------
    lattice : lattice_class lattice instance
        The instance of the lattice class, with the following attributes populated:
        lattice.xy = xy
        lattice.NL = NL
        lattice.KL = KL
        lattice.BL = BL
        lattice.PVx = PVx
        lattice.PVy = PVy
        lattice.PVxydict = PVxydict
        lattice.PV = PV
        lattice.lp['BBox'] : #vertices x 2 float array
            BBox is the polygon used to crop the lattice or defining the extent of the lattice; could be hexagonal, for instance.
        lattice.lp['LL'] : tuple of 2 floats
            LL are the linear extent in each dimension of the lattice. For ex:
            ( np.max(xy[:,0])-np.min(xy[:,0]), np.max(xy[:,1])-np.min(xy[:,1]) )
        lattice.lp['LV'] = LV
        lattice.lp['UC'] = UC
        lattice.lp['lattice_exten'] : str
            A string specifier for the lattice built
        lattice.lp['slit_exten'] = slit_exten
    """
    # Define some shortcut variables
    lattice_type = lattice.lp['LatticeTop']
    shape = lattice.lp['shape']
    NH = lattice.lp['NH']
    NV = lattice.lp['NV']
    lp = lattice.lp
    networkdir = lattice.lp['rootdir'] + 'networks/'
    check = lattice.lp['check']
    if lattice_type == 'hexagonal':
        from build_hexagonal import build_hexagonal
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_hexagonal(lp)
    elif lattice_type == 'hexmeanfield':
        from build_hexagonal import build_hexmeanfield
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_hexmeanfield(lp)
    elif lattice_type == 'hexmeanfield3gyro':
        from build_hexagonal import build_hexmeanfield3gyro
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_hexmeanfield3gyro(lp)
    elif 'selregion' in lattice_type:
        # amorphregion_isocent or amorphregion_kagome_isocent, for ex
        # Example usage: python ./build/make_lattice.py -LT selregion_randorg_gammakick1p60_cent -spreadt 0.8 -NP 225
        #  python ./build/make_lattice.py -LT selregion_iscentroid -N 30
        #  python ./build/make_lattice.py -LT selregion_hyperuniform -N 80
        from lepm.build.build_select_region import build_select_region
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_select_region(lp)
    elif lattice_type == 'frame1dhex':
        from build_hexagonal import build_frame1dhex
        # keep only the boundary of the lattice, eliminate the bulk
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_frame1dhex(lp)
    elif lattice_type == 'square':
        import lepm.build.build_square as bs
        print '\nCreating square lattice...'
        xy, NL, KL, BL, lattice_exten = bs.generate_square_lattice(shape, NH, NV, lp['eta'], lp['theta'])
        print 'lattice_exten = ', lattice_exten
        PVx = []
        PVy = []
        PVxydict = {}
        LL = (NH + 1, NV + 1)
        LVUC = 'none'
        UC = 'none'
        LV = 'none'
        BBox = np.array([[-LL[0] * 0.5, -LL[1] * 0.5], [LL[0] * 0.5, -LL[1] * 0.5],
                         [LL[0] * 0.5, LL[1] * 0.5], [-LL[0] * 0.5, LL[1] * 0.5]])
    elif lattice_type == 'triangular':
        import lepm.build.build_triangular as bt
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = bt.build_triangular_lattice(lp)
    elif lattice_type == 'linear':
        from lepm.build.build_linear_lattices import build_zigzag_lattice
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_zigzag_lattice(lp)
        lp['NV'] = 1
    elif lattice_type == 'deformed_kagome':
        from lepm.build.build_kagome import generate_deformed_kagome
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = generate_deformed_kagome(lp)
    elif lattice_type == 'deformed_martini':
        from lepm.build.build_martini import build_deformed_martini
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_deformed_martini(lp)
    elif lattice_type == 'twisted_kagome':
        from lepm.build.build_kagome import build_twisted_kagome
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_twisted_kagome(lp)
    elif lattice_type == 'hyperuniform':
        from lepm.build.build_hyperuniform import build_hyperuniform
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_hyperuniform(lp)
    elif lattice_type == 'hyperuniform_annulus':
        from lepm.build.build_hyperuniform import build_hyperuniform_annulus
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_hyperuniform_annulus(lp)
        lp['shape'] = 'annulus'
    elif lattice_type == 'isostatic':
        from lepm.build.build_jammed import build_isostatic
        # Manually tune coordination of jammed packing (lets you tune through isostatic point)
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_isostatic(lp)
    elif lattice_type == 'jammed':
        from lepm.build.build_jammed import build_jammed
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_jammed(lp)
    elif lattice_type == 'triangularz':
        from lepm.build.build_triangular import build_triangularz
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_triangularz(lp)
    elif lattice_type == 'hucentroid':
        from lepm.build.build_hucentroid import build_hucentroid
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_hucentroid(lp)
    elif lattice_type == 'hucentroid_annulus':
        from lepm.build.build_hucentroid import build_hucentroid_annulus
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_hucentroid_annulus(lp)
        lp['shape'] = 'annulus'
    elif lattice_type == 'huvoronoi':
        from lepm.build.build_voronoized import build_huvoronoi
        xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten = build_huvoronoi(lp)
        LVUC = 'none'
        LV = 'none'
        UC = 'none'
    elif lattice_type == 'kagome_hucent':
        from lepm.build.build_hucentroid import build_kagome_hucent
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_kagome_hucent(lp)
    elif lattice_type == 'kagome_hucent_annulus':
        from lepm.build.build_hucentroid import build_kagome_hucent_annulus
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_kagome_hucent_annulus(lp)
        lp['shape'] = 'annulus'
    elif lattice_type == 'kagome_huvor':
        from lepm.build.build_voronoized import build_kagome_huvor
        print('Loading hyperuniform to build lattice...')
        xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten = build_kagome_huvor(lp)
        LV, LVUC, UC = 'none', 'none', 'none'
    elif lattice_type == 'iscentroid':
        from lepm.build.build_iscentroid import build_iscentroid
        print('Loading isostatic to build lattice...')
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_iscentroid(lp)
    elif lattice_type == 'kagome_isocent':
        from lepm.build.build_iscentroid import build_kagome_isocent
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_kagome_isocent(lp)
    elif lattice_type == 'overcoordinated1':
        from lepm.build.build_overcoordinated import generate_overcoordinated1_lattice
        xy, NL, KL, BL, SBL, LV, UC, LVUC, lattice_exten = generate_overcoordinated1_lattice(NH, NV)
        PVxydict = {}; PVx = []; PVy = []
        if lp['check']:
            le.display_lattice_2D(xy, BL)
    elif lattice_type == 'circlebonds':
        from lepm.build.build_linear_lattices import generate_circle_lattice
        xy, NL, KL, BL, LV, UC, LVUC, lattice_exten = generate_circle_lattice(NH)
        PVxydict = {}
        PVx = []
        PVy = []
        lp['NV'] = 1
        minx = np.min(xy[:, 0])
        maxx = np.max(xy[:, 0])
        miny = np.min(xy[:, 1])
        maxy = np.max(xy[:, 1])
        BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
        LL = (maxx - minx, maxy - miny)
    elif lattice_type == 'dislocatedRand':
        from lepm.build.build_dislocatedlattice import build_dislocated_lattice
        xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten = build_dislocated_lattice(lp)
    elif lattice_type == 'dislocated':
        if lp['dislocation_xy'] != 'none':
            dislocX = lp['dislocation_xy'].split('/')[0]
            dislocY = lp['dislocation_xy'].split('/')[1]
            dislocxy_exten = '_dislocxy_' + dislocX + '_' + dislocY
            pt = np.array([[float(dislocX), float(dislocY)]])
        else:
            pt = np.array([[0., 0.]])
            dislocxy_exten = ''
        if lp['Bvec'] == 'random':
            Bvecs = 'W'
        else:
            Bvecs = lp['Bvec']
        xy, NL, KL, BL, lattice_exten = generate_dislocated_hexagonal_lattice(shape, NH, NV, pt,
                                                                              Bvecs=Bvecs, check=lp['check'])
        lattice_exten += dislocxy_exten
        PVx = []; PVy = []; PVxydict = {}
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        BBox = np.array([[np.min(xy[:, 0]), np.min(xy[:, 1])], [np.max(xy[:, 0]), np.min(xy[:, 1])],
                         [np.max(xy[:, 0]), np.max(xy[:, 1])], [np.min(xy[:, 0]), np.max(xy[:, 1])]])
        LV = 'none'
        UC = 'none'
        LVUC = 'none'
    elif lattice_type == 'penroserhombTri':
        if lp['periodicBC']:
            from lepm.build.build_quasicrystal import generate_periodic_penrose_rhombic
            xy, NL, KL, BL, PVxydict, BBox, PV, lattice_exten = generate_periodic_penrose_rhombic(lp)
            LL = (PV[0, 0], PV[1, 1])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        else:
            from lepm.build.build_quasicrystal import generate_penrose_rhombic_lattice
            xy, NL, KL, BL, lattice_exten, Num_sub = generate_penrose_rhombic_lattice(shape, NH, NV, check=lp['check'])
            LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
            BBox = blf.auto_polygon(shape, NH, NV, eps=0.00)
            PVx = []
            PVy = []
            PVxydict = {}

        LVUC = 'none'
        LV = 'none'
        UC = 'none'
    elif lattice_type == 'penroserhombTricent':
        from lepm.build.build_quasicrystal import generate_penrose_rhombic_centroid_lattice
        xy, NL, KL, BL, PVxydict, PV, LL, BBox, lattice_exten = generate_penrose_rhombic_centroid_lattice(lp)
        # deepcopy PVxydict to generate new pointers
        PVxydict = copy.deepcopy(PVxydict)
        if lp['periodicBC']:
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        else:
            PVx, PVy = [], []

        # Rescale so that median bond length is unity
        bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
        scale = 1. / np.median(bL)
        xy *= scale
        PV *= scale
        print 'PV = ', PV
        BBox *= scale
        LL = (LL[0] * scale, LL[1] * scale)

        print 'after updating other things: PVxydict = ', PVxydict
        if lp['periodicBC']:
            PVx *= scale
            PVy *= scale
            PVxydict.update((key, val * scale) for key, val in PVxydict.items())
        else:
            PVx = []
            PVy = []

        LVUC = 'none'
        LV = 'none'
        UC = 'none'
    elif lattice_type == 'kagome_penroserhombTricent':
        from build.build_quasicrystal import generate_penrose_rhombic_centroid_lattice
        xy, NL, KL, BL, lattice_exten = generate_penrose_rhombic_centroid_lattice(shape, NH+5, NV+5, check=lp['check'])

        # Decorate lattice as kagome
        print('Decorating lattice as kagome...')
        xy, BL, PVxydict = blf.decorate_as_kagome(xy, BL)
        lattice_exten = 'kagome_' + lattice_exten

        xy, NL, KL, BL = blf.mask_with_polygon(shape, NH, NV, xy, BL, eps=0.00, check=False)

        # Rescale so that median bond length is unity
        bL = le.bond_length_list(xy, BL)
        xy *= 1./np.median(bL)

        minx = np.min(xy[:, 0])
        miny = np.min(xy[:, 1])
        maxx = np.max(xy[:, 0])
        maxy = np.max(xy[:, 1])
        BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        PVx = []; PVy = []; PVxydict = {}
        LVUC = 'none'
        LV = 'none'
        UC = 'none'
    elif lattice_type == 'random':
        xx = np.random.uniform(low=-NH*0.5 - 10, high=NH*0.5 + 10, size=(NH + 20) * (NV + 20))
        yy = np.random.uniform(low=-NV*0.5 - 10, high=NV*0.5 + 10, size=(NH + 20) * (NV + 20))
        xy = np.dstack((xx, yy))[0]
        Dtri = Delaunay(xy)
        TRI = Dtri.vertices
        BL = le.Tri2BL(TRI)
        NL, KL = le.BL2NLandKL(BL, NN='min')

        # Crop to polygon (instead of trimming boundaries)
        # NL, KL, BL, TRI = le.delaunay_cut_unnatural_boundary(xy, NL, KL, BL, TRI, thres=lp['trimbound_thres'])
        shapedict = {shape: [NH, NV]}
        keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=check)
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')

        lattice_exten = lattice_type + '_square_thres' + sf.float2pstr(lp['trimbound_thres'])
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)
        BBox = polygon
        LL = (np.max(BBox[:, 0]) - np.min(BBox[:, 0]), np.max(BBox[:, 1]) - np.min(BBox[:, 1]))
        PVx = []
        PVy = []
        PVxydict = {}
        LVUC = 'none'
        LV = 'none'
        UC = 'none'
    elif lattice_type == 'randomcent':
        from lepm.build.build_random import build_randomcent
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_randomcent(lp)
    elif lattice_type == 'kagome_randomcent':
        from lepm.build.build_random import build_kagome_randomcent
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_kagome_randomcent(lp)
    elif lattice_type == 'randomspreadcent':
        from lepm.build.build_randomspread import build_randomspreadcent
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_randomspreadcent(lp)
    elif lattice_type == 'kagome_randomspread':
        from lepm.build.build_randomspread import build_kagome_randomspread
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_kagome_randomspread(lp)
    elif 'randorg_gammakick' in lattice_type and 'kaghi' not in lattice_type:
        # lattice_type is like 'randorg_gamma0p20_cent' and uses Hexner pointsets
        # NH is width, NV is height, NP_load is total number of points. This is different than other conventions.
        from build_randorg_gamma import build_randorg_gamma_spread
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_randorg_gamma_spread(lp)
    elif 'randorg_gamma' in lattice_type and 'kaghi' not in lattice_type:
        # lattice_type is like 'randorg_gamma0p20_cent' and uses Hexner pointsets
        from build_randorg_gamma import build_randorg_gamma_spread_hexner
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = \
            build_randorg_gamma_spread_hexner(lp)
    elif 'random_organization' in lattice_type:
        try:
            if isinstance(lp['relax_timesteps'], str):
                relax_tag = lp['relax_timesteps']
            else:
                relax_tag = '{0:02d}'.format(lp['relax_timesteps'])
        except:
            raise RuntimeError('lattice parameters dictionary lp needs key relax_timesteps')
        
        points = np.loadtxt(networkdir+'random_organization_source/random_organization/random/' +
                            'random_kick_' + relax_tag + 'relax/out_d' + '{0:02d}'.format(int(lp['conf'])) + '_xy.txt')
        points -= np.mean(points, axis=0)
        xytmp, trash1, trash2, trash3 = blf.mask_with_polygon(shape, NH, NV, points, [], eps=0.00)
        
        xy, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xytmp, polygon='auto', check=check)
        polygon = blf.auto_polygon(shape, NH, NV, eps=0.00)
        
        if check:
            le.display_lattice_2D(xy, BL)
        
        # Rescale so that median bond length is unity
        bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
        xy *= 1./np.median(bL)
        polygon *= 1./np.median(bL)
        
        lattice_exten = lattice_type + '_relax' + relax_tag + '_' + shape + '_d' + \
                        '{0:02d}'.format(int(lp['loadlattice_number']))
        LL = (np.max(xy[:, 0])-np.min(xy[:, 0]), np.max(xy[:, 1])-np.min(xy[:, 1]))
        PVxydict = {}
        PVx = []
        PVy = []
        BBox = polygon
    elif lattice_type == 'flattenedhexagonal':
        from lepm.build.build_hexagonal import generate_flattened_honeycomb_lattice
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = \
            generate_flattened_honeycomb_lattice(shape, NH, NV, lp['aratio'], lp['delta'], lp['phi'], eta=0., rot=0.,
                                                 periodicBC=False, check=check)
    elif lattice_type == 'cairo':
        from lepm.build.build_cairo import build_cairo
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = build_cairo(lp)
    elif lattice_type == 'kagper_hucent':
        from lepm.build.build_hucentroid import build_kagper_hucent
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagper_hucent(lp)
    elif lattice_type == 'hex_kagframe':
        from lepm.build.build_kagcentframe import build_hex_kagframe
        # Use lp['alph'] to determine the beginning of the kagomized frame, surrounding hexagonal lattice
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_hex_kagframe(lp)
    elif lattice_type == 'hex_kagcframe':
        from lepm.build.build_kagcentframe import build_hex_kagcframe
        # Use lp['alph'] to determine the beginning of the kagomized frame, surrounding hexagonal lattice
        # as circlular annulus
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_hex_kagcframe(lp)
    elif lattice_type in ['hucent_kagframe', 'hucent_kagcframe']:
        from lepm.build.build_kagcentframe import build_hucent_kagframe
        # Use lp['alph'] to determine the beginning of the kagomized frame, surrounding hucentroid decoration
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_hucent_kagframe(lp)
    elif lattice_type == 'kaghu_centframe':
        from lepm.build.build_kagcentframe import build_kaghu_centframe
        # Use lp['alph'] to determine the beginning of the centroid frame, surrounding kagomized decoration
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kaghu_centframe(lp)
    elif lattice_type in ['isocent_kagframe', 'isocent_kagcframe']:
        from lepm.build.build_kagcentframe import build_isocent_kagframe
        # Use lp['alph'] to determine the beginning of the kagomized frame, surrounding hucentroid decoration
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_isocent_kagframe(lp)
    elif lattice_type == 'kagsplit_hex':
        from lepm.build.build_kagome import build_kagsplit_hex
        # Use lp['alph'] to determine what fraction (on the right) is kagomized
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagsplit_hex(lp)
    elif lattice_type == 'kagper_hex':
        from lepm.build.build_kagome import build_kagper_hex
        # Use lp['alph'] to determine what fraction of the radius/width (in the center) is kagomized
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagper_hex(lp)
    elif lattice_type == 'hex_kagperframe':
        from lepm.build.build_kagcentframe import build_hex_kagperframe
        # Use lp['alph'] to determine what fraction of the radius/width (in the center) is partially kagomized
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_hex_kagperframe(lp)
    elif lattice_type == 'kagome':
        from lepm.build.build_kagome import build_kagome
        # lets you make a square kagome
        xy, NL, KL, BL, PVx, PVy, PVxydict, PV, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagome(lp)
    elif 'uofc' in lattice_type:
        from lepm.build.build_kagcent_words import build_uofc
        # ['uofc_hucent', 'uofc_kaglow_hucent', 'uofc_kaghi_hucent', 'uofc_kaglow_isocent']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_uofc(lp)
    elif 'chicago' in lattice_type:
        from lepm.build.build_kagcent_words import build_chicago
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_chicago(lp)
    elif 'chern' in lattice_type:
        from lepm.build.build_kagcent_words import build_kagcentchern
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagcentchern(lp)
    elif 'csmf' in lattice_type:
        from lepm.build.build_kagcent_words import build_kagcentcsmf
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagcentcsmf(lp)
    elif 'thanks' in lattice_type:
        # example:
        # python ./build/make_lattice.py -LT kaghi_isocent_thanks -skip_gyroDOS -NH 300 -NV 70 -thres 5.5 -skip_polygons
        from lepm.build.build_kagcent_words import build_kagcentthanks
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagcentthanks(lp)
    elif 'curvys' in lattice_type:
        # for lattice_type in ['kaghi_isocent_curvys', 'kaglow_isocent_curvys',
        #                      'kaghi_hucent_curvys', 'kaglow_hucent_curvys', kaghi_randorg_gammakick1p60_cent_curvys',
        #                      'kaghi_randorg_gammakick0p50_cent_curvys', ... ]
        # example usage:
        # python ./build/make_lattice.py -LT kaghi_isocent_curvys -NH 100 -NV 70 -thres 5.5 -skip_polygons -skip_gyroDOS
        from lepm.build.build_kagcent_words import build_kagcentcurvys
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagcentcurvys(lp)
    elif 'hexannulus' in lattice_type:
        from lepm.build.build_conformal import build_hexannulus
        xy, NL, KL, BL, lattice_exten, lp = build_hexannulus(lp)
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        PVxydict = {}
        PVx = []
        PVy = []
        UC = np.array([0, 0])
        LV = 'none'
        LVUC = 'none'
        minx = np.min(xy[:, 0])
        miny = np.min(xy[:, 1])
        maxx = np.max(xy[:, 0])
        maxy = np.max(xy[:, 1])
        BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
    elif 'accordion' in lattice_type:
        # accordionhex accordiony accordionkagy
        # Example usage:
        # python ./build/make_lattice.py -LT accordionkag -alph 0.9 -intparam 2 -N 6 -skip_polygons -skip_gyroDOS
        # nzag controlled by lp['intparam'], and the rotation of the kagome element is controlled by lp['alph']
        if lattice_type == 'accordionhex':
            from lepm.build.build_hexagonal import build_accordionhex
            xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvx = build_accordionhex(lp)
        elif lattice_type == 'accordionkag':
            from lepm.build.build_kagome import build_accordionkag
            xy, NL, KL, BL, PVx, PVy, PVxydict, PV, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_accordionkag(lp)
        elif lattice_type == 'accordionkag_hucent':
            from lepm.build.build_hucentroid import build_accordionkag_hucent
            xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = \
                build_accordionkag_hucent(lp)
        elif lattice_type == 'accordionkag_isocent':
            from lepm.build.build_iscentroid import build_accordionkag_isocent
            xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = \
                build_accordionkag_isocent(lp)
    elif 'spindle' in lattice_type:
        if lattice_type == 'spindle':
            from lepm.build.build_spindle import build_spindle
            xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_spindle(lp)
    elif lattice_type == 'stackedrhombic':
        from lepm.build.build_rhombic import build_stacked_rhombic
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_stacked_rhombic(lp)
    elif 'junction' in lattice_type:
        # accordionhex accordiony accordionkagy
        if lattice_type == 'hexjunction':
            from lepm.build.build_hexagonal import build_hexjunction
            xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvx = build_hexjunction(lp)
        elif lattice_type == 'kagjunction':
            from lepm.build.build_kagome import build_kagjunction
            xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp = build_kagjunction(lp)

    ###########
    
    # For reference: this is how to change z
    # le.cut_bonds_z(BL,lp['target_z'])
    # ##cut N bonds for z
    # N2cut = int(round((z_start-target_z)*len(BL)/z_start))
    # allrows = range(len(BL))
    # inds = random.sample(allrows, N2cut) 
    # keep = np.setdiff1d(allrows,inds)
    # bnd_z = BL[keep]
    
    # Cut slit
    if lp['make_slit']:
        print('Cuting notch or slit...')
        L = max(xy[:, 0]) - min(xy[:, 0])
        cutL = L * lp['cutLfrac']
        x0 = -L * 0. + cutL * 0.5 - 1.  # 0.25*L+0.5
        BL, xy = blf.cut_slit(BL, xy, cutL+1e-3, x0, y0=0.2)
        slit_exten = '_cutL' + str(cutL/L) + 'L_x0_' + str(x0)

        print('Rebuilding NL,KL...')
        NP = len(xy)
        if latticetop == 'deformed_kagome':
            nn = 4        
        elif lattice_type == 'linear':
            if NP == 1:
                nn = 0
            else:
                nn = 2
        else:
            nn = 'min'
            
        NL, KL = le.BL2NLandKL(BL, NP=NP, NN=nn)
            
        # remake BL too
        BL = le.NL2BL(NL, KL)
    else:
        slit_exten = ''
        NP = len(xy)
        
    if lp['check']:
        print 'make_lattice: Showing lattice before saving...'
        # print 'PVx = ', PVx
        # print 'PVy = ', PVy
        le.display_lattice_2D(xy, BL, PVxydict=PVxydict)
    ################
    
    lattice.xy = xy
    lattice.NL = NL
    lattice.KL = KL
    lattice.BL = BL
    lattice.PVx = PVx
    lattice.PVy = PVy
    lattice.PVxydict = PVxydict
    lattice.LVUC = LVUC
    lattice.lp = lp
    lattice.lp['BBox'] = BBox
    lattice.lp['LL'] = LL
    lattice.lp['LV'] = LV
    lattice.lp['UC'] = UC
    lattice.lp['lattice_exten'] = lattice_exten
    lattice.lp['slit_exten'] = slit_exten
    lattice.lp['Nparticles'] = NP
    return lattice


if __name__ == "__main__":
    """Example usage:
    python make_lattice.py -LT randorg
    """
    # Create options from command line
    parser = argparse.ArgumentParser(description='Specify parameters for the network to build.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks directory', type=str,
                        default='check_string_for_empty')
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings ' +
                                          '(leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load', help='Specify to nonzero int to load a network of a particular size in ' +
                                                 'its entirety, without cropping. Will override NH and NV.',
                        type=int, default=0)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, deformed_' +
                                                    'kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hexagonal')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-kr_max', '--kr_max',
                        help='Upper bound for magnitude of k in calculating S(k), in units of 2*pi/L',
                        type=int, default=-1)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-viewmethod', '--viewmethod', help='Check steps of method of computation', action='store_true')
    parser.add_argument('-nice_plot', '--nice_plot', help='Output nice pdf plots of lattice', action='store_true')

    # For loading and coordination
    parser.add_argument('-LLID', '--loadlattice_number',
                        help='If LT=hyperuniform/isostatic, selects which lattice to use', type=str, default='01')
    parser.add_argument('-LLz', '--loadlattice_z',
                        help='If LT=hyperuniform/isostatic, selects what z index to use', type=str, default='001')
    parser.add_argument('-source', '--source',
                        help='Selects who made the lattice to load, if loaded from source (ulrich, hexner, etc)',
                        type=str, default='hexner')
    parser.add_argument('-cutLfrac', '--cutLfrac', help='Fraction of system to cut for make_slit option',
                        type=float, default=0.5)
    parser.add_argument('-perd', '--percolation_density', help='Fraction of vertices to decorate',
                        type=float, default=0.5)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='Position of single dislocation, if not centered at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')
    parser.add_argument('-origin', '--origin', help='Offset to apply to point set when cropping to polygon', type=str,
                        default='0.0/0.0')
    parser.add_argument('-spreadt', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.0)
    parser.add_argument('-dt', '--spreading_dt',
                        help='Time step for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.001)
    parser.add_argument('-kicksz', '--kicksz',
                        help='Average of log of kick magnitudes for loading randorg_gammakick pointsets.' +
                             'This sets the scale of the powerlaw kicking procedure',
                        type=float, default=-1.50)

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-periodic_strip', '--periodic_strip',
                        help='periodic boundary only on R/L ends', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-phi', '--phi', help='Shear angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=0.0)
    parser.add_argument('-delta', '--delta', help='Deformation angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=120./180.)
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice, in units of pi',
                        type=float, default=0.000)
    parser.add_argument('-alph', '--alph',
                        help='Twist angle for twisted_kagome (max is pi/3) in radians or ' +
                             'opening angle of the accordionized lattices or ' +
                             'percent of system decorated -- used in different contexts',
                        type=float, default=0.00)
    parser.add_argument('-x1', '--x1',
                        help='1st Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x2', '--x2',
                        help='2nd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x3', '--x3',
                        help='3rd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-zz', '--zz',
                        help='4th Deformation parameter for deformed_kagome', type=float, default=0.00)

    parser.add_argument('-aratio', '--aratio',
                        help='Ratio between bond lengths in Cairo lattice', type=float, default=np.sqrt(5.))
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-conf2', '--sub_realization_number', help='Decoration realization number',
                        type=int, default=01)
    parser.add_argument('-intparam', '--intparam',
                        help='Integer-valued parameter for building networks (ex # subdivisions in accordionization)',
                        type=int, default=1)
    parser.add_argument('-thres', '--thres', help='Threshold value for building networks (determining to decorate pt)',
                        type=float, default=1.0)
    parser.add_argument('-trimbound', '--trimbound_thres',
                        help='Threshold value for trimming boundary triangles when building networks',
                        type=float, default=4.0)

    # Physics params
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)

    # Things to calculate
    parser.add_argument('-skip_polygons', '--skip_polygons',
                        help='Skip calculation of polygons in the lattice', action='store_true')
    parser.add_argument('-ipr', '--ipr',
                        help='calculate participation ratio for the gyroscopic network', action='store_true')
    parser.add_argument('-fft_image', '--fft_image', help='Do calculation of FFT of scatterplot of the point set',
                        action='store_true')
    parser.add_argument('-pointset', '--pointset', help='Plot the lattice point set', action='store_true')
    parser.add_argument('-bondL_hist', '--bondL_histogram',
                        help='calculate and save bond length histogram', action='store_true')
    parser.add_argument('-gxy', '--gxy',
                        help='Calculate of g(x,y) 2D correlation function for the lattice', action='store_true')
    parser.add_argument('-varN', '--varN', help='Skip calculation of variance_N(R)', action='store_true')
    parser.add_argument('-structurefactor', '--structurefactor',
                        help='Calculate structure factor when making lattice', action='store_true')
    parser.add_argument('-summarize', '--summarize_structure',
                        help='Calculate and save a plot of structure of lattice', action='store_true')
    parser.add_argument('-massDOS', '--massDOS',
                        help='Compute and save the eigvals and eigvects of the mass-spring model', action='store_true')
    parser.add_argument('-skip_gyroDOS', '--skip_gyroDOS',
                        help='Skip computing and saving the eigvals and eigvects of the gyro-spring model',
                        action='store_true')
    parser.add_argument('-DOSmovie', '--DOSmovie', help='Whether to make a movie of the density of states',
                        action='store_true')
    parser.add_argument('-fancy_gr', '--fancy_gr',
                        help='Perform careful calculation of g(r) correlation function for the ENTIRE lattice',
                        action='store_true')
    
    # Plotting params
    parser.add_argument('-fftmax', '--fftmax', help='color limit maximum for log(Intensity) in FFT of point set',
                        type=float, default=14)
    parser.add_argument('-fftmin', '--fftmin', help='color limit minimum for log(Intensity) in FFT of point set',
                        type=float, default=2)
    parser.add_argument('-save_Sk', '--save_Sk', help='Save the data of S(k) with kx and ky as txt file',
                        action='store_true')
    args = parser.parse_args()

    if args.N > 0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV
    latticetop = args.LatticeTop  # 'linear' 'triangular' 'hexagonal' 'deformed_kagome'

    phi = np.pi * args.phi
    delta = np.pi * args.delta

    strain = 0.00 # initial
    # z = 4.0 #target z
    if latticetop == 'linear':
        shape = 'line'
    else:
        shape = args.shape
    
    theta = args.theta
    eta = args.eta
    if args.kr_max > 0:
        klim = args.kr_max
    else:
        klim = 'auto'

    origin = sf.string_sequence_to_numpy_array(args.origin, dtype=float)
    print 'origin = ', origin
    
    make_slit = args.make_slit

    # For now, enforce that if NP_load == True, then PBCs
    # if args.NP_load != 0:
    #     args.periodicBC = True

    if args.rootdir == 'check_string_for_empty':
        if socket.gethostname()[0:6] == 'midway':
            # We must be on the cluster
            rootdir = '/home/npmitchell/scratch-midway/'
        else:
            rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    else:
        rootdir = args.rootdir

    lp = {'LatticeTop': latticetop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'rootdir': rootdir,
          'phi': phi,
          'delta': delta,
          'theta': theta,
          'eta': eta,
          'x1': args.x1,
          'x2': args.x2,
          'x3': args.x3,
          'z': args.zz,
          'source': args.source,
          'loadlattice_number': args.loadlattice_number,
          'check': args.check,
          'viewmethod': args.viewmethod,
          'Ndefects': args.Ndefects,
          'Bvec': args.Bvec,
          'dislocation_xy': args.dislocation_xy,
          'target_z': args.target_z,
          'make_slit': args.make_slit,
          'cutz_method': args.cutz_method,
          'cutLfrac': args.cutLfrac,
          'conf': args.realization_number,
          'subconf': args.sub_realization_number,
          'periodicBC': args.periodicBC or args.periodic_strip,
          'periodic_strip': args.periodic_strip,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'eta_alph': args.eta_alph,
          'origin': origin,
          'NP_load': args.NP_load,
          'percolation_density': args.percolation_density,
          'thres': args.thres,
          'trimbound_thres': args.trimbound_thres,
          'aratio': args.aratio,
          'ABDelta': args.ABDelta,
          'spreading_time': args.spreading_time,
          'spreading_dt': args.spreading_dt,
          'kicksz': args.kicksz,
          'intparam': args.intparam,
          }

    #######################
    print 'LatticeTop = ', latticetop
    lattice = lattice_class.Lattice(lp)
    print '\nBuilding lattice...'
    lattice.build()
    print '\nSaving lattice...'
    lattice.save(skip_polygons=args.skip_polygons, check=lp['check'])

    if args.nice_plot:
        print 'Saving nice BW plot...'
        lattice.plot_BW_lat(meshfn=dio.prepdir(lattice.lp['meshfn']))

    infodir = lattice.lp['meshfn']+'/'
    if args.bondL_histogram:
        print 'Creating bond length histogram...'
        lattice.bond_length_histogram(fig=None, ax=None, outdir=infodir, check=args.check)

    glp = {'Omk': -1.0, 'Omg': -1.0}
    gyrolat = gyro_lattice_class.GyroLattice(lattice, glp)

    if not args.skip_gyroDOS:
        print '\n\nPerforming gyro DOS ...'
        gyrolat.save_eigval_eigvect()
        if args.DOSmovie:
            gyrolat.save_DOSmovie()
    if args.massDOS:
        print 'Performing mass DOS...'
        masslat = mass_lattice_class.mass_lattice(lattice.xy, lattice.NL, lattice.KL, [1., 0.], 'mass')
        masslat.save_eigval_eigvect(infodir)
        # if args.DOSmovie:
        #     masslat.save_DOSmovie()
    if args.varN and lattice.lp['LL'][0]*0.17 > 6:
        print 'Performing number variance...'
        lattice.nvariance = lattice.number_variance(outdir=infodir, check=args.check)
    if args.gxy:
        print '\nPerforming correlation functions g(xy)...'
        plt.close('all')
        lattice.gxy, lattice.gr = lattice.calc_gxy_gr(outdir=infodir, dr=0.1, eps=1e-7, check=False)
    if args.ipr:
        print '\nPerforming inverse participation ratio measurement...'
        lattice.ipr = gyrolat.save_ipr(attribute=False)
    if args.fancy_gr:
        print '\nPerforming fancy correlation function g(r)...'
        lattice.calc_fancy_gr(outdir=infodir, dr=0.1, check=False)
    if args.structurefactor:
        print '\nPerforming structure factor S(kx,ky)...'
        lattice.Skxy, lattice.Skr = lattice.structure_factor(klim=klim, Nhist=args.Nhist, outdir=infodir)
    if args.summarize_structure:
        print '\nPerforming structure summary...'
        lattice.summarize_structure()
    if args.fft_image:
        print '\nPerforming FFT of image of points...'
        lattice.pointset_fft(plot_output=True, outdir=infodir, check=args.check)
    if args.pointset:
        print '\nPerforming image of points...'
        lattice.pointset(outdir=infodir, wsfrac=0.8)

    print '\n\n\n OUTPUT NP = ', len(lattice.xy), '\n\n\n\n'
