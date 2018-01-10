import numpy as np
from lepm.build.roipoly import roipoly
import lepm.lattice_class as lattice_class
import lepm.data_handling as dh
import lepm.lattice_elasticity as le
import copy
import matplotlib.pyplot as plt

'''Functions for creating networks with(interactively) user-defined shapes'''


def build_select_region(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    lpnew = copy.deepcopy(lp)
    lpnew['LatticeTop'] = lp['LatticeTop'].split('selregion_')[-1]
    lpnew['check'] = False
    print 'lpnew[LatticeTop] = ', lpnew['LatticeTop']
    lattice = lattice_class.Lattice(lpnew)
    print '\nBuilding lattice...'
    lattice.build()
    # xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = build_kagome_isocent(lp)
    xy = lattice.xy
    NL = lattice.NL
    KL = lattice.KL
    BL = lattice.BL
    PVx = lattice.PVx
    PVy = lattice.PVy
    PVxydict = lattice.PVxydict
    try:
        LVUC = lattice.lp['LVUC']
        LV = lattice.lp['LV']
        UC = lattice.lp['UC']
    except:
        LVUC = 'none'
        LV = 'none'
        UC = 'none'

    LL = lattice.lp['LL']
    old_lattice_exten = lattice.lp['lattice_exten']

    # Display lattice
    ax = le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                               title='Choose roi polygon', xlimv=None, ylimv=None, colorz=True, ptcolor=None,
                               ptsize=10,
                               close=False, colorpoly=False, viewmethod=False, labelinds=False,
                               colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=False, linewidth=0.0,
                               edgecolors=None, check=False)

    # let user draw ROI
    roi = roipoly(ax=ax, roicolor='r')

    print 'roi = ', roi
    print 'x = ', roi.allxpoints
    print 'y = ', roi.allypoints
    roi = np.dstack((roi.allxpoints, roi.allypoints))[0]
    inpoly = dh.inds_in_polygon(xy, roi)
    print 'inpoly = ', inpoly
    if lp['check']:
        plt.plot(xy[inpoly, 0], xy[inpoly, 1], 'b.')
        plt.show()

    xy, NL, KL, BL = le.remove_pts(inpoly, xy, BL, check=lp['check'])
    if lp['periodicBC']:
        PV = le.PVxydict2PV(PVxydict)
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        # If cropping the points has cut off all periodic BCs, update lp to reflect this
        if len(PVxydict) == 0:
            lp['periodicBC'] = False

    if LVUC is not None and LVUC is not 'none':
        LVUC = LVUC[inpoly]

    BBox = roi
    lattice_exten = 'selregion_' + old_lattice_exten + '_NP{0:06d}'.format(len(xy))
    xy -= np.mean(xy, axis=0)
    if lp['check']:
        ax = le.display_lattice_2D(xy, BL, NL=NL, KL=KL, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                                   title='Cropped network', xlimv=None, ylimv=None, colorz=True, ptcolor=None,
                                   ptsize=10,
                                   close=False, colorpoly=False, viewmethod=False, labelinds=False,
                                   colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=False, linewidth=0.0,
                                   edgecolors=None, check=False)

    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
