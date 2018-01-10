import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
from scipy.spatial import Delaunay

"""Auxiliary functions for building square lattices"""


def generate_square_lattice(shape, NH, NV, eta=0., theta=0., check=False):
    """Create a square lattice NH wide and NV tall, with sites randomized by eta and rotated by theta.

    Parameters
    ----------
    shape : string
        overall shape of the mesh: 'circle' 'hexagon' 'square'
    NH : int
        Number of pts along horizontal before boundary is cut
    NV : int
        Number of pts along vertical before boundary is cut
    eta : float
        randomization or jitter in the positions of the particles
    theta : float
        orientation of the lattice vectors (rotation) in units of radians
    check : bool
        Whether to view intermediate results

    Results
    ----------
    """
    # Establish square lattice, rotated by theta

    if abs(theta) < 1e-6:
        latticevecs = [[1., 0.], [0., 1.]]
        # todo: SHOULD MODIFY TO INCLUDE LVUC
        # xypts_tmp, LVUC = generate_lattice_LVUC([NH,NV], latticevecs)
        xypts_tmp = blf.generate_lattice([NH, NV], latticevecs)
    else:
        latticevecs = [[np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        xypts_tmp, LVUC = blf.generate_lattice([2 * NH, 2 * NV], latticevecs)
    if NH < 3 and NV < 3:
        if NH == NV == 2:
            xypts_tmp = np.array([[0., 0.], np.array(latticevecs[0]),
                                 np.array(latticevecs[1]),
                                 np.array(latticevecs[0]) + np.array(latticevecs[1])], dtype=float)
            xypts_tmp -= np.mean(xypts_tmp)
        elif NH == 1 and NV == 1:
            '''making single point'''
            xypts_tmp = np.array([[0., 0.]])

    print 'xypts_tmp =', xypts_tmp

    if eta == 0. or eta == '':
        etastr = ''
    else:
        etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    if theta == 0. or theta == '':
        thetastr = ''
    else:
        thetastr = '_theta{0:.3f}'.format(theta/np.pi).replace('.', 'p') + 'pi'

    if shape == 'square':
        tmp2 = xypts_tmp  # shorten name for clarity
        xy = tmp2  # [np.logical_and(tmp2[:,0]<(NH) , tmp2[:,1]<NV) ,:]
    elif shape == 'circle':
        print 'assuming NH == NV since cutting out a circle...'
        tmp2 = xypts_tmp  # shorten name for clarity
        xy = tmp2[tmp2[:, 0]**2 + tmp2[:, 1]**2 < NH**2, :]
    elif shape == 'hexagon':
        tmp2 = xypts_tmp  # shorten name for clarity
        # xy = tmp2[tmp2[:,0]**2+tmp2[:,1]**2 < (NH)**2 ,:]
        print 'cropping to: ', shape
        # NH, NV = shape['hexagon']
        # Modify below to allow different values of NH and NV on the horiz and vertical sides of the hexagon
        a = NH + 0.5
        polygon = np.array([[-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)],
                            [a*0.5, -np.sqrt(a**2 - (0.5*a)**2)], [a, 0.],
                            [a*0.5, np.sqrt(a**2 - (0.5*a)**2)], [-a*0.5, np.sqrt(a**2 - (0.5*a)**2)], [-a, 0.],
                            [-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)]])
        bpath = mplpath.Path(polygon)
        keep = bpath.contains_points(tmp2)
        xy = tmp2[keep]

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
            ax.plot(tmp2[:, 0], tmp2[:, 1], 'r.')
            plt.show()
    else:
        xy = xypts_tmp

    print('Triangulating points...\n')
    print 'xy =', xy

    tri = Delaunay(xy)
    TRItmp = tri.vertices

    print('Computing bond list to remove cross braces...\n')
    BL = le.Tri2BL(TRItmp)
    thres = np.sqrt(2.0) * .99  # cut off everything as long as a diagonal
    print('thres = '+str(thres))
    print('Trimming bond list...\n')
    orig_numBL = len(BL)
    BL = le.cut_bonds(BL, xy, thres)
    print('Trimmed ' + str(orig_numBL-len(BL)) +' bonds.')

    # print('Recomputing TRI...\n')
    # TRI = le.BL2TRI(BL, xy)

    # Randomize by eta
    if eta == 0:
        xypts = xy
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta*np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]

    # Naming
    exten = '_' + shape + etastr + thetastr

    # BL = latticevec_filter(BL,xy, C, CBL)
    NL, KL = le.BL2NLandKL(BL, NN=4)
    lattice_exten = 'square' + exten
    print 'lattice_exten = ', lattice_exten
    return xypts, NL, KL, BL, lattice_exten


