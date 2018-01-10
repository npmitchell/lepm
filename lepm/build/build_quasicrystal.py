import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import lepm.stringformat as sf
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
import math
import cmath

'''Functions for generating quasicrystalline networks'''


def generate_penrose_kite_dart_tiling(Num_sub, check=False):
    """This is unfinished
    """
    goldenRatio = (1.0 + np.sqrt(5.0)) / 2.0

    def subdivide_kite_dart(triangles):
        result = []
        for color, A, B, C in triangles:
            if color == 0:
                # Subdivide red (sharp isosceles) (half kite) triangle
                Q = A + (B - A) / goldenRatio
                R = B + (C - B) / goldenRatio
                result += [(1, R, Q, B), (0, Q, A, R), (0, C, A, R)]
            else:
                # Subdivide blue (fat isosceles) (half dart) triangle
                P = C + (A - C) / goldenRatio
                result += [(1, B, P, A), (0, P, C, B)]

        return result

    return None


def setup_penrose_for_periodic(lp):
    """Creating a rhombus-shaped penroserhombTri quasicrystalline lattice, with arrangement of particles such that
    a periodic tiling nearly matches a true quasicrystal lattice."""
    if lp['NH'] < 160 and lp['NV'] < 160:
        Num_sub = int(np.ceil(np.log2(lp['NH']) + 2))
    else:
        Num_sub = int(np.ceil(np.log2(lp['NH']) + 3))
    xy, NL, KL, BL, TRI, lattice_exten = generate_penrose_rhombic_tiling(Num_sub, check=False)

    # Find a lower left sixfold particle

    sixfold = np.where(np.sum(KL, axis=1) > 5)[0]
    # print 'sixfold = ', sixfold
    # print 'sum = ', xy[sixfold, 0] + xy[sixfold, 1]
    lowerleft = np.argmin(xy[sixfold, 0] + xy[sixfold, 1])
    indx = sixfold[lowerleft]
    ll = (np.abs(xy[:, 0] - xy[indx, 0]) ** 2 + np.abs(xy[:, 1] - xy[indx, 1]) ** 2).argmin()

    # Find a matching sixfold particle along vertical line from this location closest to NV away
    vertical = np.where(np.abs(xy[:, 0] - xy[ll, 0]) < 1e-5)[0]
    ul = vertical[np.argmin(np.abs(xy[vertical, 1] - xy[ll, 1] - lp['NV']))]

    # Find a matching sixfold particle along sloped horizontal line from this location closest to NH away
    rotll = xy - xy[ll, :]
    ang = np.pi * 0.5 - 2 * np.pi * 0.2
    xyrot = np.dstack((rotll[:, 0] * np.cos(ang) + rotll[:, 1] * np.sin(ang),
                       -rotll[:, 0] * np.sin(ang) + rotll[:, 1] * np.cos(ang)))[0]
    horiz = np.where(np.abs(xyrot[:, 1]) < 1e-1)[0]
    # print 'horiz = ', horiz
    # plt.scatter(xy[:, 0], xy[:, 1])
    # plt.scatter(xy[horiz, 0], xy[horiz, 1], c='r')
    # plt.scatter(xy[ll, 0], xy[ll, 1], c='g')
    # plt.show()
    # plt.plot(xyrot[horiz, 0], np.abs(xyrot[horiz, 0] - lp['NH']), 'b.-')
    # plt.show()
    lr = horiz[np.argmin(np.abs(xyrot[horiz, 0] - lp['NH']))]

    # Is there a matching particle along vertical line from this location closest to NV away from lr?
    vertical = np.where(np.abs(xy[:, 0] - xy[lr, 0]) < 1e-2)[0]
    ur = vertical[np.argmin(np.abs(xy[vertical, 1] - xy[lr, 1] - lp['NV']))]

    if lp['check']:
        print 'idx = ', ll
        print 'ul = ', ul
        print 'lr = ', lr
        print 'ur = ', ur
        le.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], colormap='BlueBlackRed', colorz=True,
                         title='Output during periodic identification',
                         show=False)  # PVx=PVx, PVy=PVy, PVxydict=PVxydict,
        plt.plot([xy[ll, 0], xy[ul, 0], xy[lr, 0], xy[ur, 0]], [xy[ll, 1], xy[ul, 1], xy[lr, 1], xy[ur, 1]], 'ro')
        plt.show()

    vertd = xy[ul, 1] - xy[ll, 1]
    horzd = xy[lr, 0] - xy[ll, 0]
    horvd = xy[lr, 1] - xy[ll, 1]
    midpt = np.mean(np.array([xy[ll], xy[ul], xy[ul] + np.array([horzd, horvd]), xy[lr]]), axis=0)
    xy -= midpt
    PV = np.array([[horzd, horvd], [0., vertd]])
    polygon = 0.5 * np.array([[-PV[0, 0], (-PV[1, 1] - PV[0, 1])],
                              [PV[0, 0], (-PV[1, 1] + PV[0, 1])],
                              [PV[0, 0], (PV[1, 1] + PV[0, 1])],
                              [-PV[0, 0], (PV[1, 1] - PV[0, 1])]])
    if lp['check']:
        le.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], colormap='BlueBlackRed', colorz=True,
                         title='point set before cropping', show=False)
        plt.plot(polygon[:, 0], polygon[:, 1], 'r.-')
        plt.show()
        plt.clf()

    BBox = copy.deepcopy(polygon)
    eps = 1e-4
    polygon += np.array([-eps, -eps])
    bpath = mplpath.Path(polygon)
    keep = bpath.contains_points(xy)
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min', check=lp['check'])

    return xy, NL, KL, BL, PV, BBox, polygon, lattice_exten


def generate_periodic_penrose_rhombic(lp):
    """Generates periodic penrose rhombus tiling as network with rhombic periodic BCs"""
    xy, NL, KL, BL, PV, BBox, polygon, lattice_exten = setup_penrose_for_periodic(lp)
    # if lp['check']:
    #     le.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], colormap='BlueBlackRed', colorz=True,
    #                      title='Cropped point set before periodicity', show=True)

    xy, NL, KL, BL, PVxydict = le.delaunay_periodic_network_from_pts(xy, PV, BBox=polygon, check=lp['check'],
                                                   target_z=-1, max_bond_length=-1, zmethod='random', minimum_bonds=-1)

    lattice_exten = lattice_exten.split('_div')[0] + '_periodic_' + lp['shape'] + '_div' + lattice_exten.split('_div')[1]

    return xy, NL, KL, BL, PVxydict, BBox, PV, lattice_exten


def generate_penrose_rhombic_tiling(Num_sub, check=False):
    """Generates penrose rhomus tiling (P3) with given number of subdivisions (scaling method)
    http://preshing.com/20110831/penrose-tiling-explained/
    """
    # ------ Configuration --------
    NUM_SUBDIVISIONS = Num_sub
    # -----------------------------

    goldenRatio = (1.0 + math.sqrt(5.0)) / 2.0

    def subdivide(triangles):
        result = []
        for color, A, B, C in triangles:
            if color == 0:
                # Subdivide red triangle
                P = A + (B - A) / goldenRatio
                result += [(0, C, P, B), (1, P, C, A)]
            else:
                # Subdivide blue triangle
                Q = B + (A - B) / goldenRatio
                R = B + (C - B) / goldenRatio
                result += [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)]
        return result

        # Create wheel of red triangles around the origin
        triangles = []
        for i in xrange(10):
            B = cmath.rect(1, (2*i - 1) * math.pi / 10)
            C = cmath.rect(1, (2*i + 1) * math.pi / 10)
            if i % 2 == 0:
                B, C = C, B # Make sure to mirror every second triangle
            triangles.append((0, B, 0j, C))

    def plot_quasicrystal(triangles):
        ax = plt.gca()
        for color, A, B, C in triangles:
            if color == 0:
                codes = [mplpath.Path.MOVETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.CLOSEPOLY,
                        ]
                polygon = np.array([[A.real, A.imag], [B.real, B.imag], [C.real, C.imag], [A.real, A.imag]])
                path = mplpath.Path(polygon, codes)
                patch = mpatches.PathPatch(path, facecolor='orange', lw=2)
                ax.add_patch(patch)

            if color == 1:
                codes = [mplpath.Path.MOVETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.LINETO,
                        mplpath.Path.CLOSEPOLY,
                        ]
                polygon = np.array([[A.real, A.imag], [B.real, B.imag], [C.real, C.imag], [A.real, A.imag]])
                path = mplpath.Path(polygon, codes)
                patch = mpatches.PathPatch(path, facecolor='blue', lw=2)
                ax.add_patch(patch)
        return ax

    # Create wheel of red triangles around the origin
    triangles = []
    for i in xrange(10):
        B = cmath.rect(1, (2*i - 1) * math.pi / 10)
        C = cmath.rect(1, (2*i + 1) * math.pi / 10)
        if i % 2 == 0:
            B, C = C, B  # Make sure to mirror every second triangle
        triangles.append((0, 0j, B, C))

    # Perform subdivisions
    for i in xrange(NUM_SUBDIVISIONS):
        triangles = subdivide(triangles)

    # Prepare cairo surface
    # surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, IMAGE_SIZE[0], IMAGE_SIZE[1])
    # cr = cairo.Context(surface)
    # cr.translate(IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0)
    # wheelRadius = 1.2 * math.sqrt((IMAGE_SIZE[0] / 2.0) ** 2 + (IMAGE_SIZE[1] / 2.0) ** 2)
    # cr.scale(wheelRadius, wheelRadius)

    # Draw red triangles
    if check:
        plot_quasicrystal(triangles)
        plt.show()

    # Scale points
    tri = np.array(triangles)
    mindist = np.min(abs(tri[:, 1] - tri[:, 0]))
    print 'mindist = ', mindist
    scale = 1./mindist
    tri = tri[:, 1:4] * scale
    if check:
        plot_quasicrystal(triangles)

    # Convert points to numbered points
    # Create dict of locations to indices
    indexd = {}
    xy = np.zeros((len(tri) * 3, 2))
    TRI = np.zeros_like(tri, dtype=int)
    rowIND = 0
    dmyi = 0
    offs = float(np.ceil(np.max(tri.real).ravel())+1)
    for AA, BB, CC in tri:
        # reformat A,B,C
        A = ('{0:0.2f}'.format(AA.real + offs), '{0:0.2f}'.format(AA.imag + offs))
        B = ('{0:0.2f}'.format(BB.real + offs), '{0:0.2f}'.format(BB.imag + offs))
        C = ('{0:0.2f}'.format(CC.real + offs), '{0:0.2f}'.format(CC.imag + offs))
        # print '\n\n\n'
        # print 'A = ', A
        if A not in indexd:
            indexd[A] = dmyi
            xy[dmyi] = [AA.real, AA.imag]
            TRI[rowIND, 0] = dmyi
            dmyi += 1
            # print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[A]
            TRI[rowIND, 0] = index

        # print 'indexd = ', indexd
        # print '\nB = ', B
        if B not in indexd:
            indexd[B] = dmyi
            xy[dmyi] = [BB.real, BB.imag]
            TRI[rowIND, 1] = dmyi
            dmyi += 1
            # print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[B]
            TRI[rowIND, 1] = index

        # print 'indexd = ', indexd
        # print '\nC = ', C
        if C not in indexd:
            indexd[C] = dmyi
            xy[dmyi] = [CC.real, CC.imag]
            TRI[rowIND, 2] = dmyi
            dmyi += 1
            # print 'xy[0:dmyi,:] = ', xy[0:dmyi,:]
        else:
            index = indexd[C]
            TRI[rowIND, 2] = index
        rowIND += 1

    xy = xy[0:dmyi]
    print 'xy = ', xy

    if check:
        # plot_quasicrystal(triangles)
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'ro-')
        plt.show()

    BL = le.TRI2BL(TRI)
    NL, KL = le.BL2NLandKL(BL, NP='auto', NN='min')
    print 'TRI = ', TRI
    print 'BL = ', BL
    if check:
        le.display_lattice_2D(xy, BL, close=False)
        for ii in range(len(xy)):
            plt.text(xy[ii, 0], xy[ii, 1], str(ii))
        plt.show()

    lattice_exten = 'penroserhombTri_div_' + str(Num_sub)
    return xy, NL, KL, BL, TRI, lattice_exten


def generate_penrose_rhombic_centroid_lattice(lp):
    """Make lattice from centroid decoration of rhombic penrose tiling

    Parameters
    ----------
    shape : str
    NH : int
    NV : int
    check : bool

    Returns
    ----------
    xy, NL, KL, BL, TRI, lattice_exten
    """
    shape = lp['shape']
    NH = lp['NH']
    NV = lp['NV']
    check = lp['check']
    if lp['periodicBC']:
        xy, NL, KL, BL, PV, BBox, polygon, lattice_exten = setup_penrose_for_periodic(lp)
        LL = (PV[0, 0], PV[1, 1])
        xy, NL, KL, BL, PVxydict = le.delaunay_centroid_periodic_network_from_pts(xy, PV, BBox=polygon,
                                                                                  shear=0.01, check=lp['check'])
        lattice_exten = lattice_exten.split('_div')[0] + '_periodic_' + lp['shape'] + '_div' +\
                        lattice_exten.split('_div')[1]

    else:
        xy, NL, KL, BL, lattice_exten, Num_sub = generate_penrose_rhombic_lattice(shape, NH, NV, check=check)
        minx = np.min(xy[:, 0])
        miny = np.min(xy[:, 1])
        maxx = np.max(xy[:, 0])
        maxy = np.max(xy[:, 1])
        BBox = np.array([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        TRI = le.BL2TRI(BL, xy)
        if check:
            plt.triplot(xy[:, 0], xy[:, 1], TRI)
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            plt.show()
        xy, NL, KL, BL, BM = le.centroid_lattice_from_TRI(xy, TRI, check=lp['check'])
        PVxydict = {}
        PV = []

    lattice_exten = 'penroserhombTricent' + lattice_exten[15:]
    return xy, NL, KL, BL, PVxydict, PV, LL, BBox, lattice_exten


def generate_penrose_rhombic_lattice(shape, NH, NV, check=False):
    """Make lattice from rhombic penrose tiling

    Parameters
    ----------
    shape : str
    NH : int
    NV : int
    check : bool

    Returns
    ----------
    xy, NL, KL, BL, TRI, lattice_exten
    """
    if NH < 160 and NV < 160:
        Num_sub = int(np.ceil(np.log2(NH) + 3))
    else:
        Num_sub = int(np.ceil(np.log2(NH) + 4))
    xy, NL, KL, BL, TRI, lattice_exten = generate_penrose_rhombic_tiling(Num_sub, check=check)
    xy, NL, KL, BL = blf.mask_with_polygon(shape, NH, NV, xy, BL, check=check)
    lattice_exten = lattice_exten.split('_div')[0] + '_'+shape+'_div'+lattice_exten.split('_div')[1]
    return xy, NL, KL, BL, lattice_exten, Num_sub

