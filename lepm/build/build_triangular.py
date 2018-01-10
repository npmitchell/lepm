import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
from scipy.spatial import Delaunay

'''Functions for building triangular lattices'''


def build_triangular_lattice(lp):
    print('Build the lattice')
    if lp['periodicBC']:
        raise RuntimeError('Have not coded periodic triangular lattice yet... do that here.')
    else:
        shape_keep = lp['shape']
        shape = {shape_keep: [lp['NH'] + 0.01, lp['NV'] + 0.01]}
        xy, NL, KL, BL, LVUC, LV, UC, lattice_exten = \
            generate_triangular_lattice(shape, lp['NH'] * 3, lp['NV'] * 3, lp['eta'], lp['theta'], check=lp['check'])
        PVx = []
        PVy = []
        PVxydict = {}
        BBox = blf.auto_polygon(shape_keep, lp['NH'], lp['NV'], eps=0.00)
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))

    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def build_triangularz(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    NH = lp['NH']
    NV = lp['NV']
    shape = lp['shape']
    eta = lp['eta']
    theta = lp['theta']
    check = lp['check']
    if lp['cutz_method'] == 'random':
        xy, NL, KL, BL, LVUC, LV, UC, lattice_exten = \
            generate_triangular_lattice(shape, NH, NV, eta, theta, check=check)
        NL, KL, BL = \
            le.cut_bonds_z_random(xy, NL, KL, BL, lp['target_z'], bulk_determination='Endpts', check=check)
    elif lp['cutz_method'] == 'highest':
        xy, NL, KL, BL, LVUC, LV, UC, lattice_exten = \
            generate_triangular_lattice(shape, NH + 5, NV + 5, eta, theta, check=check)
        NL, KL, BL = le.cut_bonds_z_highest(xy, NL, KL, BL, lp['target_z'], check=check)
        # Now crop out correctly coordinated region
        shapedict = {shape: [NH, NV]}
        keep = blf.argcrop_lattice_to_polygon(shapedict, xy, check=check)
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
        LVUC = LVUC[keep]
    else:
        raise RuntimeError('Must specify cutz_method argument when Lattice topology == triangularz.')

    # Remove any points with no bonds
    keep = KL.any(axis=1)
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min')
    LVUC = LVUC[keep]

    print 'len(xy) = ', len(xy)
    z = le.compute_bulk_z(xy, NL, KL, BL)
    print 'FOUND z = ', z
    print('Defining lattice_exten...')
    lattice_exten = 'triangularz_' + shape + '_zmethod' + lp['cutz_method'] + '_z' + '{0:0.03f}'.format(z)
    PVxydict = {}
    PVx = []
    PVy = []
    BBox = blf.auto_polygon(shape, lp['NH'], lp['NV'], eps=0.00)
    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_triangular_lattice(shape, NH, NV, periodicBC=False, eta=0., theta=0., check=False):
    """Generate a triangular lattice NH wide and NV tall, with sites randomized by eta and rotated by theta.

    Parameters
    ----------
    shape : dictionary with string key
        key is overall shape of the mesh ('square' 'rectangle2x1' 'rectangle1x2' 'circle' 'hexagon'), value is radius,
        hexagon, sidelength, or array of closed points
    NH : int
        Number of pts along horizontal before boundary is cut
    NV : int
        Number of pts along vertical before boundary is cut
    periodicBC : bool
        make the network periodic
    eta : float
        randomization of the mesh, in units of lattice spacing
    theta : float
        overall rotation of the mesh lattice vectors in radians
    check : bool
        whether to view results at each step
    Returns
    ----------
    xy : array of dimension nx3
        Equilibrium positions of all the points for the lattice
    NL : array of dimension n x (max number of neighbors)
        Each row corresponds to a point.  The entries tell the indices of the neighbors.
    KL : array of dimension n x (max number of neighbors)
        Correponds to NL matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    LVUC : NP x 4 array
        For each particle, gives (lattice vector, unit cell vector) coordinate position of that particle: LV1, LV2, UC
        For instance, xy[0,:] = LV[0]*LVUC[0,0] + LV[1]*LVUC[0,1] + UC[LVUC[0,2]]
    LV : 3 x 2 float array
        Lattice vectors for the kagome lattice with input twist angle
    UC : 6 x 2 float array
        (extended) unit cell vectors
    lattice_type : string
        label, lattice type.  For making output directory
    """
    # If shape is a string, turn it into the appropriate dict
    if isinstance(shape, str):
        # Since shape is a string, give key as str and vals to form polygon mask
        print 'Since shape is a string, give key as str and vals to form polygon mask...'
        vals = [NH, NV]
        shape_string = shape
        NH *= 3
        NV *= 3
        shape = {shape_string: vals}

    # Establish triangular lattice, rotated by theta
    if abs(theta) < 1e-6:
        latticevecs = [[1, 0], [0.5, np.sqrt(3)*0.5]]
    else:
        latticevecs = [[np.cos(theta), np.sin(theta)],
                       [0.5 * np.cos(theta) - np.sqrt(3) * 0.5 * np.sin(theta),
                        np.sqrt(3) * 0.5 * np.cos(theta) + 0.5 * np.sin(theta)]]

    xypts_tmp, LVUC = blf.generate_lattice_LVUC([NH, NV], latticevecs)

    # CHECK
    if check:
        plt.plot(xypts_tmp[:, 0], xypts_tmp[:, 1], 'b.')
        for i in range(len(xypts_tmp)):
            plt.text(xypts_tmp[i, 0], xypts_tmp[i, 1], str(LVUC[i, 0]) + ', ' + str(LVUC[i, 1]))
        plt.title('Pre-cropped lattice')
        plt.show()

    if theta == 0:
        add_exten = ''
    else:
        add_exten = '_theta' + '{0:.3f}'.format(theta / np.pi).replace('.', 'p') + 'pi'
        # ROTATE BY THETA
        print 'Rotating by theta= ', theta, '...'
        xys = copy.deepcopy(xypts_tmp)
        xypts_tmp = np.array([[x*np.cos(theta) - y*np.sin(theta), y*np.cos(theta)+x*np.sin(theta)] for x, y in xys])

    # mask to rectangle
    if 'circle' in shape:
        '''add masking to shape here'''
        tmp2 = xypts_tmp
        NH, NV = shape['circle']
        # Modify below to allow ovals
        R = NH*0.5
        keep = np.logical_and(np.abs(tmp2[:, 0]) < R*1.000000001, np.abs(tmp2[:, 1]) < (2*R*1.0000001))
        xy = tmp2[keep, :]
        LVUC = LVUC[keep, :]
        shape = 'circle'
    elif 'hexagon' in shape:
        print 'cropping to: ', shape
        tmp2 = xypts_tmp
        NH, NV = shape['hexagon']
        # Modify below to allow different values of NH and NV on the horiz and vertical sides of the hexagon
        a = NH + 0.5
        polygon = np.array([[-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)],
            [a*0.5, -np.sqrt(a**2 - (0.5*a)**2)], [a, 0.],
            [a*0.5, np.sqrt(a**2 - (0.5*a)**2)], [-a*0.5, np.sqrt(a**2 - (0.5*a)**2)], [-a, 0.],
            [-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)]])
        bpath = mplpath.Path(polygon)
        keep = bpath.contains_points(tmp2)
        xy = tmp2[keep, :]
        LVUC = LVUC[keep, :]
        shape = 'hexagon'

        # Check'
        if check:
            codes = [mplpath.Path.MOVETO,
                    mplpath.Path.LINETO,
                    mplpath.Path.LINETO,
                    mplpath.Path.LINETO,
                    mplpath.Path.LINETO,
                    mplpath.Path.LINETO,
                    mplpath.Path.CLOSEPOLY,
                    ]
            path = mplpath.Path(polygon, codes)
            ax = plt.gca()
            patch = mpatches.PathPatch(path, facecolor='orange', lw=2)
            ax.add_patch(patch)
            ax.plot(polygon[:, 0], polygon[:, 1], 'bo')
            ax.plot(xy[:, 0], xy[:, 1], 'r.')
            plt.show()
    elif 'square' in shape:
        tmp2 = xypts_tmp
        NH,NV = shape['square']
        if periodicBC:
            keep = np.logical_and(np.abs(tmp2[:, 0]) < NH * .5 + 0.51,
                                  np.abs(tmp2[:, 1]) < (NV * .5 / np.sin(np.pi / 3.) - 0.1))
        else:
            keep = np.logical_and(np.abs(tmp2[:, 0]) < NH*.5 + 0.1, np.abs(tmp2[:, 1]) < (NV*.5/np.sin(np.pi/3.)+0.1))
        xy = tmp2[keep, :]
        LVUC = LVUC[keep, :]
        shape = 'square'
        pass
    elif 'polygon' in shape:
        bpath = mplpath.Path(shape['polygon'])
        keep = bpath.contains_points(xypts_tmp)
        xy = xypts_tmp[keep, :]
        LVUC = LVUC[keep, :]
        shape = 'polygon'
    else:
        raise RuntimeError('Polygon dictionary not specified in generate_triangular_lattice().')

    print('Triangulating points...\n')
    tri = Delaunay(xy)
    TRItmp = tri.vertices

    print('Computing bond list...\n')
    BL = le.Tri2BL(TRItmp)
    # bL = le.bond_length_list(xy,BL)
    thres = 1.1  # cut off everything longer than a diagonal
    print('thres = ' + str(thres))
    print('Trimming bond list...\n')
    BLtrim = le.cut_bonds(BL, xy, thres)
    print('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.')

    # print 'BLtrim =', BLtrim

    print('Recomputing TRI...\n')
    TRI = le.BL2TRI(BLtrim, xy)

    # Randomize if eta >0 specified
    if eta == 0:
        xypts = xy
        eta_exten = ''
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta*np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]
        eta_exten = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    # Naming
    exten = '_' + shape + add_exten + eta_exten

    # BL = latticevec_filter(BL,xy, C, CBL)
    NL, KL = le.BL2NLandKL(BLtrim, NN=6)
    lattice_exten = 'triangular' + exten
    print 'lattice_exten = ', lattice_exten
    LV = np.array(latticevecs)
    UC = np.array([0., 0.])
    return xypts, NL, KL, BLtrim, LVUC, LV, UC, lattice_exten


