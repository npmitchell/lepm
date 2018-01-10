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
import glob

'''Functions for creating networks build from loaded jammed packings'''


def build_isostatic(lp):
    """Build a network by manually tuning coordination of jammed packing (lets you tune through isostatic point)

    Parameters
    ----------
    lp

    Returns
    -------

    """
    # Manually tune coordination of jammed packing (lets you tune through isostatic point)
    networkdir = lp['rootdir'] + 'networks/'
    print('Loading isostatic: get jammed point set to build lattice with new bonds...')
    number = '{0:03d}'.format(int(lp['conf']))
    if lp['source'] == 'hexner':
        # Use Daniel Hexner - supplied networks
        points, BL, LLv, numberstr, sizestr, lp = load_hexner_jammed(lp, BL_load=False)
        LL = (LLv, LLv)
        print 'LL = ', LL
        sourcestr = '_hexner'
    else:
        zindex = '{0:03d}'.format(int(lp['loadlattice_z']))
        points = np.loadtxt(networkdir + 'isostatic_source/isostatic_homog_z' + zindex +
                            '_conf' + number + '_nodes.txt')
    points -= np.mean(points, axis=0)

    # DO initial cropping to speed up triangulation
    keep1 = np.logical_and(abs(points[:, 0]) < lp['NH'] * 0.5 + 8, abs(points[:, 1]) < lp['NV'] * 0.5 + 8)
    xy = points[keep1]
    if lp['periodicBC']:
        xy, NL, KL, BL, PVxydict = \
            le.delaunay_rect_periodic_network_from_pts(xy, LL, target_z=lp['target_z'],
                                                       zmethod=lp['cutz_method'], check=check)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        periodicstr = '_periodic'
    else:
        xy, NL, KL, BL, BM = le.delaunay_lattice_from_pts(xy, trimbound=False, target_z=lp['target_z'],
                                                          zmethod=lp['cutz_method'], check=check)

        # if lp['cutz_method'] == 'random':
        #     NL, KL, BL = le.cut_bonds_z_random(xy, NL, KL, BL, lp['target_z'],  bulk_determination='Endpts')
        # elif lp['cutz_method'] == 'highest':
        #     NL, KL, BL = le.cut_bonds_z_highest(xy, NL, KL, BL, lp['target_z'], check=check)
        # elif lp['cutz_method'] == 'none':
        #     pass

        print('Trimming lattice to be NH x NV...')
        keep = np.logical_and(abs(xy[:, 0]) < lp['NH'] * 0.5, abs(xy[:, 1]) < lp['NV'] * 0.5)
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min', check=lp['check'])
        PVxydict = {}
        PVx = []
        PVy = []
        periodicstr = ''

    le.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                     title='Output during isostatic build', show=True)
    plt.close('all')

    # Calculate coordination number
    z = le.compute_bulk_z(copy.deepcopy(xy), copy.deepcopy(NL), copy.deepcopy(KL), copy.deepcopy(BL))
    print 'FOUND z = ', z
    lattice_exten = 'isostatic_' + lp['shape'] + periodicstr + sourcestr + '_z' + '{0:0.03f}'.format(z) + '_conf' + \
                    numberstr + '_zmethod' + lp['cutz_method']
    BBox = np.array([[-LL[0] * 0.5, -LL[1] * 0.5], [LL[0] * 0.5, -LL[1] * 0.5],
                     [LL[0] * 0.5, LL[1] * 0.5], [-LL[0] * 0.5, LL[1] * 0.5]])
    LV = 'none'
    UC = 'none'
    LVUC = 'none'
    plt.close('all')
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_jammed(lp):
    """Use bonds from jammed packing --> already near isostatic, but can't tune above it, only below.

    Parameters
    ----------
    lp

    Returns
    -------

    """
    # Use bonds from jammed packing --> already near isostatic, but can't tune above it, only below.
    shape = lp['shape']
    nh = lp['NH']
    nv = lp['NV']
    networkdir = dio.ensure_dir(lp['rootdir']) + 'networks/'

    print('Loading isostatic file to build jammed lattice...')
    # Now load
    use_hexner = (lp['source'] == 'hexner')
    if use_hexner:
        print 'lp[periodicBC] = ', lp['periodicBC']
        points, BL, LLv, numberstr, sizestr, lp = load_hexner_jammed(lp, BL_load=True)
        sourcestr = '_hexner'
    else:
        # Use Stephan Ulrich's lattices
        if lp['periodicBC']:
            RuntimeError('Not sure if Stephan Ulrich lattices are periodic!')
        zindex = '{0:03d}'.format(int(lp['loadlattice_z']))
        number = '{0:03d}'.format(int(lp['conf']))
        points = np.loadtxt(networkdir + 'isostatic_source/isostatic_homog_z' + zindex + '_conf' +
                            number + '_nodes.txt')
        BL = np.loadtxt(networkdir + 'isostatic_source/isostatic_homog_z' + zindex + '_conf' + number + '_bonds.txt',
                        usecols=(0, 1), dtype=int)
        sourcestr = '_ulrich_homog'
        print 'BL[100] = ', BL[100]
        # Loaded BL uses indexing starting at 1, not 0
        BL -= 1
        print 'BL[100] = ', BL[100]
        numberstr = number[1:]

    if check:
        le.display_lattice_2D(points, np.abs(BL), title='points and bonds loaded, before pruning')
    xy = points - np.mean(points, axis=0)
    NL, KL = le.BL2NLandKL(BL, NN='min')

    # Remove any points with no bonds
    print 'Removing points without any bonds...'
    keep = KL.any(axis=1)
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
    print 'len(xy) = ', len(xy)

    if check:
        le.display_lattice_2D(xy, np.abs(BL), title='Before tuning z (down) and before fixing PBCs')

    if lp['cutz_method'] == 'random':
        NL, KL, BL = le.cut_bonds_z_random(xy, NL, KL, BL, lp['target_z'], bulk_determination='Endpts')
    elif lp['cutz_method'] == 'highest':
        NL, KL, BL = le.cut_bonds_z_highest(xy, NL, KL, BL, lp['target_z'], check=check)
    elif lp['cutz_method'] == 'none':
        pass

    if lp['periodicBC'] or lp['NP_load'] > 0:
        print 'Building periodicBC PVs...'
        lp['periodicBC'] = True
        LL = (LLv, LLv)
        polygon = 0.5 * np.array([[-LLv, -LLv], [LLv, -LLv], [LLv, LLv], [-LLv, LLv]])
        BBox = polygon
        PV = np.array([[LLv, 0.0], [LLv, LLv], [LLv, -LLv],
                       [0.0, 0.0], [0.0, LLv], [0.0, -LLv],
                       [-LLv, 0.0], [-LLv, LLv], [-LLv, -LLv]])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)

        if lp['check']:
            le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, title='Checking periodic BCs',
                                  close=False, colorz=False)
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.plot(xy[:, 0], xy[:, 1], 'go')
            plt.show()
    else:
        polygon = blf.auto_polygon(shape, nh, nv, eps=0.00)
        BBox = polygon
        print('Trimming lattice to be NH x NV...')
        keep = np.logical_and(abs(xy[:, 0]) < nh * 0.5, abs(xy[:, 1]) < nv * 0.5)
        print "Check that if lp['NP_load'] !=0 then len(keep) == len(xy):", len(keep) == len(xy)
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        LL = (nh, nv)
        PVx = []
        PVy = []
        PVxydict = {}

    z = le.compute_bulk_z(xy, NL, KL, BL)
    print 'FOUND z = ', z

    if lp['periodicBC']:
        periodicBCstr = '_periodicBC'
    else:
        periodicBCstr = ''

    print('Defining lattice_exten...')
    lattice_exten = 'jammed_' + shape + sourcestr + periodicBCstr + '_z' + '{0:0.03f}'.format(z) + '_conf' + \
                    numberstr + '_zmethod' + lp['cutz_method']
    LV = 'none'
    UC = 'none'
    LVUC = 'none'
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def load_hexner_jammed(lp, BL_load=True):
    """Load a jammed network supplied by Daniel Hexner.

    Parameters
    ----------
    lp : dict
    BL_load : bool
    """
    # Use Daniel Hexner's lattices
    # First check that if periodicBC, NP_load is nonzero.
    if lp['periodicBC']:
        if lp['NP_load'] == 0:
            RuntimeError('For iscentroid lattices, if periodicBC, then must specify NP_load instead of NH, NV.')

    # Load Daniel Hexner's lattices
    networkdir = lp['rootdir'] + 'networks/'
    number = '{0:04d}'.format(int(lp['conf']))
    if lp['NP_load'] == 0:
        if lp['NH'] < 15 and lp['NV'] < 15:
            Nsource = '00000512'
            Lpstr = 'Lp-4.00'
            col3 = False
        elif lp['NH'] < 81 and lp['NV'] < 81:
            Nsource = '00008192'
            Lpstr = 'Lp-2.40'
            col3 = False
        else:
            Nsource = '00128000'
            Lpstr = 'Lp0.85'
            col3 = True
    elif lp['NP_load'] in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 128000]:
        print "Since lp['NP_load'] nonzero, loading entire lattice and resetting NH, NV accordingly..."
        Nsource = '{0:08d}'.format(int(lp['NP_load']))
        if Nsource == '00128000':
            col3 = True
        else:
            col3 = False
    else:
        RuntimeError("When lp['NP_load'] != 0, #particles must match a saved network for this LatticeTopology...")

    sourcedir = networkdir + 'jammed_source_hexner_polydisperse/N' + Nsource[-6:] + '/'
    print 'Looking for file ' + sourcedir + 'XY_location_2d_N' + Nsource + '_*_r' + number + '_t4.txt'
    loadfile = glob.glob(sourcedir + 'XY_location_2d_N' + Nsource + '_*_r' + number + '_t4.txt')[0]
    if col3:
        xp, yp, LLvV = np.loadtxt(loadfile, unpack=True)
        LLv = LLvV[1]
    else:
        xp, yp = np.loadtxt(loadfile, unpack=True)
        # This will be useful for periodicBC
        with open(loadfile) as f:
            headers = [line for line in f if line.startswith('#')]
            header = headers[0]
            trash, NPload, deltazload, pressure, LLv = header.split(' ')
        LLv = float(LLv)
    print 'LLv = ', LLv
    print 'len(xp) = ', len(xp)
    print 'xp[0] = ', xp[0]
    points = np.dstack((xp, yp))[0]

    # Load BL for the network if needed
    if BL_load:
        BLfile = glob.glob(sourcedir + 'bonds_2d_N' + Nsource + '_*_r' + number + '_t4.txt')[0]
        BL = np.loadtxt(BLfile, usecols=(0, 1), dtype=int)
        # The loaded BL is periodic BCs
        keep = np.ones(len(BL), dtype=bool)

        # Check for long bonds that are periodic when loaded
        dmyi = 0
        if lp['periodicBC']: maxL = LLv
        else: maxL = lp['NH']
        for row in BL:
            # Check if points are very far apart
            diff = points[row[0]] - points[row[1]]
            if diff[0] ** 2 + diff[1] ** 2 > (maxL * 0.5) ** 2:
                keep[dmyi] = False
            dmyi += 1

        if lp['periodicBC']:
            print 'MAKING BCs PERIODIC'
            BL[np.setdiff1d(np.arange(len(BL)), np.where(keep)[0])] *= -1
        else:
            BL = BL[keep]
    else:
        BL = None

    numberstr = number[2:]
    sizestr = '{0:04d}'.format(int(Nsource))
    if lp['NP_load'] != 0:
        print 'Redefining NH, NV since we are loading entire lattice (and not cropping it)...'
        lp['NH'], lp['NV'] = LLv, LLv

    return points, BL, LLv, numberstr, sizestr, lp
