import numpy as np
import lepm.line_segments as lineseg
import lepm.data_handling as dh

'''
Description
===========
Functions for finding the Brillouin zone from lattice points/vectors or reciprocal lattice vectors
'''


def generate_bzpath(vtcs, npts=50):
    """

    Parameters
    ----------
    vtcs:
    npts:

    Returns
    -------
    kpts : (approx. npts) x 2 float array
        the path through the hexagonal or square BZ that traces out a characteristic
        path in the BZ (like K-Gamma-M-K for hexagonal, or R-M1-Gamma-M2 for square)
    """
    # build path from vtcs of BZ
    if len(vtcs) == 6:
        # build K-Gamma-M-K
        gammapoint = np.array([0., 0.])
        # for K point, get index of vtcs where both elements are positive
        posxy = np.where(np.logical_and(vtcs[:, 0] > 0, vtcs[:, 1] > 0))[0]
        if len(posxy) > 1:
            print 'vtcs = ', vtcs
            raise RuntimeError('There are more than one vertices of the BZ that have positive x and y components.'
                               'Handle this case here')
        else:
            kpoint = vtcs[posxy]
        # Find M point as the midpoint of the top linesegment of the BZ.
        # First get where BZ vertices have third largest value for y, to get pts with larger y value than that
        y3max = np.sort(vtcs[:, 1])[-3]
        topy = np.where(vtcs[:, 1] > y3max)[0]
        if len(topy) == 2:
            mpoint = np.mean(vtcs[topy, :], axis=0)
        else:
            print 'vtcs = ', vtcs
            print 'y3max = ', y3max
            print 'topy = ', topy
            raise RuntimeError('There are not simply two BZ vertex points with largest y values. Handle here.')

        # Build path as 4x4 array of 4 linesegments
        path = np.vstack((kpoint, gammapoint, mpoint, kpoint))
    elif len(vtcs) == 4:
        # Build R-M1-Gamma-M2
        raise RuntimeError('Have not coded for square BZ path, do so here.')

    # get total length, in order to estimate how many points to place along each segment of the path
    nvtcs = np.shape(path)[0]
    totlen = 0.
    for kk in range(len(path)):
        x0 = path[kk, 0]
        x1 = path[(kk + 1) % nvtcs, 0]
        y0 = path[kk, 1]
        y1 = path[(kk + 1) % nvtcs, 1]
        length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        totlen += length

    # Now make points along each segment
    symmetryinds = [0]
    for kk in range(len(path)):
        x0 = path[kk, 0]
        x1 = path[(kk + 1) % nvtcs, 0]
        y0 = path[kk, 1]
        y1 = path[(kk + 1) % nvtcs, 1]
        length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        if kk == 0:
            kpts_x = np.linspace(x0, x1, int(npts * length / totlen) + 1)
            kpts_y = np.linspace(y0, y1, int(npts * length / totlen) + 1)
        else:
            add_x = np.linspace(x0, x1, int(npts * length / totlen) + 1)
            add_y = np.linspace(y0, y1, int(npts * length / totlen) + 1)
            # add_x = np.arange(x0, x1, step)
            # add_y = np.arange(y0, y1, step)
            kpts_x = np.hstack((kpts_x, add_x))
            kpts_y = np.hstack((kpts_y, add_y))

        symmetryinds.append(len(kpts_x) - 1)

    kpts = np.dstack((kpts_x, kpts_y))[0]
    return kpts, symmetryinds


def bzpath_from_kxy(kxy, vtcs, eps=None):
    """Get indices of kxy that are closest to path from K-Gamma-M-K for hexagonal BZ, or R-M1-Gamma-M2 for square BZ

    Parameters
    ----------
    kxy : N x 2 float array

    vtcs : # vertices of BZ x 2 float array
        the corners of the BZ, used to pick out K and M points in path if hexagonal, or
    eps : float or None
        The maximum distance of a point in the BZ from the path to consider the point to be along the path.
        If None, uses average distance between nearest-neighbor kxy points.

    Returns
    -------
    kpath : M x 2 float array
        the wavevectors that are near
    kpath_inds : M x 1 int array
        the indices of kxy that return kpath, the wavevectors that are nearest to the path tracing out a characteristic
        path in the BZ (like K-Gamma-M-K for hexagonal, or R-M1-Gamma-M2 for square)
    """
    # build path from vtcs of BZ
    path = generate_bzpath(vtcs, npts=2)
    path_linesegs = np.array([np.hstack((path[kk, :], path[kk+1 % len(path), :])) for kk in xrange(len(path))])

    # if eps is none, set it to be the distance between adjacent kxy points
    if eps is None:
        dists = dh.dist_pts(kxy, kxy)
        mindists = np.array([np.min(dists[kk, dists[kk] > 0.]) for kk in xrange(np.shape(dists)[0])])
        eps = np.mean(mindists)

    kpts = []
    for kk in range(np.shape(path_linesegs)[0]):
        tmp = lineseg.pts_are_near_lineseg(kxy, path_linesegs[kk, 0:2], path_linesegs[kk, 2:4], eps)
        kpts.append(tmp)
        print 'tmp = ', tmp

    raise RuntimeError('have not quite finished this function')
    return kpath, kpath_inds, path_linesegs


def reciprocal_lattice_vecs(a1, a2):
    """finds the BZ lattice vecs
    Formerly called find_bz_lattice_vecs()

    Parameters
    ----------
    a1: array 1x3
        first lattice vector
    a2 : array 1x3
        second lattice vector

    Returns
    ----------
    b1 : array 1x3
        first reciprocal lattice vector
    b2 : array 1x3
        second reciprocal lattice vector
    """
    a3 = np.array([0, 0, 1])

    b1 = 2 * np.pi * np.cross(a2, a3) / (np.dot(a1, np.cross(a2, a3)))
    b2 = 2 * np.pi * np.cross(a3, a1) / (np.dot(a1, np.cross(a2, a3)))

    # Check the result
    # print 'b1 = ', b1
    # print 'b2 = ', b2
    # import matplotlib.pyplot as plt
    # plt.plot([0, a1[0]], [0, a1[1]], '.-')
    # plt.plot([0, a2[0]], [0, a2[1]], '.-')
    # plt.plot([0, b1[0]], [0, b1[1]], '.-')
    # plt.plot([0, b2[0]], [0, b2[1]], '.-')
    # plt.axis('scaled')
    # print 'area of realspace unitcell = ', np.cross(a1, a2)[2]
    # print 'area of first bz = ', np.cross(b1, b2)[2]
    # print '2pi^2/area_unitcell = ', (np.pi * 2.) ** 2 / np.cross(a1, a2)[2]
    # plt.show()
    # sys.exit()

    return b1, b2


def bz_vertices(a1, a2):
    """formerly called 'find_bz_zone(b1, b2)', but now takes two (linearly independent) lattice vectors as
    arguments (a1, a2).
    This function finds the BZ zone for any lattice given the reicprocal lattice vectors.

    Parameters
    ----------
    a1 : 1x2 or 1x3 float array
        the first lattice vector. The third component, if it exists, should be zero
    a2 : 1x2 or 1x3 float array
        the second lattice vector. The third component, if it exists, should be zero

    Returns
    ----------
    vtx : array
        x position of BZ zone corners
    vty : array
        y position of BZ zone corners
    """
    if len(a1) == 2:
        a1 = np.hstack((a1, np.array([0.])))
    elif a1[2] != 0.:
        raise RuntimeError('The third (z) component of each lattice vector should either be missing or be zero')
    if len(a2) == 2:
        a2 = np.hstack((a2, np.array([0.])))
    elif a2[2] != 0.:
        raise RuntimeError('The third (z) component of each lattice vector should either be missing or be zero')

    b1, b2 = reciprocal_lattice_vecs(a1, a2)
    b3 = np.array([b2[0] - b1[0], b2[1] - b1[1], 0])
    p1 = b1
    p2 = b2
    p3 = -b1
    p4 = -b2
    p5 = (b1 + b2)
    p6 = -(b2 + b1)
    p7 = (b1 - b2)
    p8 = (b2 - b1)
    p9 = b2 + b3
    p10 = -(b2 + b3)
    # p11 = 2 * b1
    # p12 = 2 * b2

    # min_p1112 = min(np.array([p11[0] ** 2 + p11[1] ** 2, p12[0] ** 2 + p12[1] ** 2]))
    ps = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    pdist = [ps[i, 0] ** 2 + ps[i, 1] ** 2 for i in range(len(ps))]
    si = np.argsort(pdist)
    ps = ps[si][:6]

    # Check the results going into bz_based_on_ps
    # import matplotlib.pyplot as plt
    # plt.plot([0, a1[0]], [0, a1[1]], '.-')
    # plt.plot([0, a2[0]], [0, a2[1]], '.-')
    # plt.plot([0, b1[0]], [0, b1[1]], '.-')
    # plt.plot([0, b2[0]], [0, b2[1]], '.-')
    # print 'np.cross(b1, b2)[2] = ', np.cross(b1, b2)[2]

    # a true value of rr indicates that some points (in the p array) should not be used in the BZ calculation.
    vtx, vty, rr = bz_based_on_ps(ps)
    if rr:
        ps2 = ps[:4]
        vtx, vty, rr = bz_based_on_ps(ps2)
    # else:
    #     ps2 = ps

    return vtx, vty


def bz_based_on_ps(ps):
    """This function finds the Brillouin zone based on calculations of the nearest equivalent points to the origin.
    The algorithm used finds the intersections between the perpendicular lines from these points to the origin.

    Parameters
    ----------
    ps : array
        nearest neighbor points which are equivalent to the origin in the BZ zone

    Returns
    ----------
    vtx : array
        x position of BZ zone corners
    vty : array
        y position of BZ zone corners
    rr : boolean
        once the BZ zone corners are found the algorithm checks their polar angle order.  If they are out of order, it
        indicates that too many points have been used in the calculation
        and that the incorrect Brillouin zone has been found.
    """
    angs = np.array([(np.arctan2(ps[i, 1], ps[i, 0]) + .01) % (2 * np.pi) for i in range(len(ps))])
    si = np.argsort(angs)
    ps = ps[si]

    # perpendicular bisectors between these and origin?
    mids = ps / 2

    # fig = plt.figure()
    # plt.scatter(ps[:,0], ps[:,1], c = 'k')
    # plt.scatter(mids[:,0], mids[:,1], c = 'r')
    # plt.axes().set_aspect(1)

    x = []
    l = []
    jj = np.where(abs(ps[:, 1]) < 0.001)[0]

    ps_fs = ps.copy()
    ps_fs[jj, 1] = 1.
    # slopes of the perpendicular bisectors
    slopes = np.array([-ps_fs[i, 0] / ps_fs[i, 1] for i in range(len(ps))])
    # vertical lines get a slope of -1.23
    slopes[jj] = -1.23

    # for plotting
    for i in range(len(slopes)):
        if slopes[i] != -1.23:
            x_vals = np.arange(mids[i, 0] - 5, mids[i, 0] + 5, 0.05)
            lines = slopes[i] * (x_vals - mids[i, 0]) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)
            # plt.plot(x_vals, lines)
        else:
            x_vals = np.zeros(200) + mids[i, 0]
            lines = np.arange(-5, 5, 0.05) + mids[i, 1]
            l.append(lines)
            x.append(x_vals)
            # plt.plot(x_vals, lines)

    x = np.array(x)
    vtx = []
    vty = []

    for i in range(len(slopes)):
        if i == len(slopes) - 1:
            neigh = 0
        else:
            neigh = i + 1

        m1 = slopes[i]
        m2 = slopes[neigh]
        if m1 != -1.23 and m2 != -1.23:
            xy1 = mids[i]
            xy2 = mids[neigh]

            xv = (m1 * xy1[0] - xy1[1] - m2 * xy2[0] + xy2[1]) / (m1 - m2)
            yv = slopes[i] * (xv - mids[i, 0]) + mids[i, 1]
        else:
            if m1 == - 1.23:
                jk = neigh
                jl = i
            else:
                jk = i
                jl = neigh

            xv = x[jl, 0]

            yv = slopes[jk] * (xv - mids[jk, 0]) + mids[jk, 1]

        vtx.append(xv)
        vty.append(yv)

    # plt.scatter(vtx, vty, s = 60)
    # plt.show()
    angs = np.array([np.arctan2(vty[i], vtx[i]) % (2. * np.pi) for i in range(len(vtx))])
    si = np.argsort(angs)
    vtx = np.array(vtx)[si]
    vty = np.array(vty)[si]
    diff = [abs(si[i + 1] - si[i]) for i in range(len(si) - 1)]
    if 2 in diff:
        rr = True
    else:
        rr = False

    # Check the result
    # import matplotlib.pyplot as plt
    # plt.plot(vtx, vty, 'k.-')
    # plt.axis('scaled')
    # import lepm.data_handling as dh
    # print 'dh.polygon_area() = ', dh.polygon_area(np.dstack((vtx, vty))[0])
    # plt.show()
    # sys.exit()

    return vtx, vty, rr
