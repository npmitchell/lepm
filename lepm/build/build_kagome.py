import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
import copy
import scipy
from scipy.spatial import Delaunay
from lepm.build import build_hexagonal
import lepm.build.build_kagome_auxiliary as kagaux
import lepm.data_handling as dh
import lepm.math_functions as mf

''''''


def build_twisted_kagome(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    if lp['NH'] == 1 and lp['NV'] == 1:
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = \
            kagaux.build_twisted_kagome_unitcell(lp)
    else:
        xy, NL, KL, BL, LVUC, LV, UC, lattice_exten = \
            generate_twisted_kagome(lp['shape'], lp['NH'], lp['NV'], lp['alph'])
        polygon = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)
        PVxydict = {}
        PVx = []
        PVy = []
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        BBox = polygon
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_twisted_kagome(shape, NH, NV, alph):
    """creates twisted kagome lattice as in Sun&Lubensky2012

    Parameters
    ----------
    shape : string
        overall shape of the mesh ('square' 'circle') --> haven't built in functionality yet
    NH : int
        Number of pts along horizontal before boundary is cut
    NV : int
        Number of pts along vertical before boundary is cut
    alph : float
        twist angle

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
    print('Setting up unit cell...')
    # Bravais primitive unit vecs
    scale = 2. * np.abs(np.cos(alph))
    aa = scale * np.array([[np.cos(2 * np.pi * float(p) / 3.), np.sin(2 * np.pi * float(p) / 3.)] for p in range(3)])
    a1, a2, a3 = aa[0], aa[1], aa[2]
    # NOTE that a3 = -a1 -a2
    #      ^
    #       \ a2
    #        \
    #          -------> a1
    #        /
    #       /
    #      v  a3 = -a1-a2

    # #Check
    # plt.clf()
    # plt.plot([0,a1[0]],[0,a1[1]],'b-')
    # plt.plot([0,a2[0]],[0,a2[1]],'r-')
    # plt.plot([0,a3[0]],[0,a3[1]],'g-')
    # plt.show()

    # make composite unit cell (multiple unit cells in size)
    psi = [n * np.pi * (1. / 3.) + (-1) ** n * alph for n in range(6)]
    b = np.array([[np.cos(psi[i]), np.sin(psi[i])] for i in range(6)])

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    C = np.array([[0., 0.],
                  b[0],
                  b[0] + b[1],
                  b[0] + b[1] + b[2],
                  b[0] + b[1] + b[2] + b[3],
                  b[0] + b[1] + b[2] + b[3] + b[4]])
    CU = np.array([0, 1, 2, 3, 4, 5])

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    tmp1 = np.ones_like(CU)
    tmp0 = np.zeros_like(CU)
    inds = np.arange(len(C))
    for i in np.arange(NV):
        for j in np.arange(NH):
            if i == 0:
                if j == 0:
                    # initialize
                    rr = C
                    LVUC = np.dstack((tmp0, tmp0, CU))[0]

                    # check
                    # tmp = C + i*(a2+a1) + j*a1
                    # plt.plot(tmp[:,0],tmp[:,1])
                    # plt.pause(.1)
                else:
                    # bottom row (include all but last point in translation)
                    rr = np.vstack((rr, C[0:5, :] + j * a1))
                    LVUCadd = np.dstack((j * tmp1[0:5], tmp0[0:5], CU[0:5]))[0]
                    # print 'LVUCadd = ', LVUCadd
                    LVUC = np.vstack((LVUC, LVUCadd))
                    # print 'LVUC = ', LVUC

                    # check
                    # tmp = C[inds[0:5],:] + i*(a2+a1) + j*a1
                    # plt.plot(tmp[:,0],tmp[:,1])
                    # plt.pause(.1)
            else:
                if j == 0:
                    # first cell of row, include all but pts 0,1
                    rr = np.vstack((rr, C[inds[2:], :] + i * (a2 + a1) + j * a1))
                    LVUCadd = np.dstack(((i + j) * tmp1[2:], i * tmp1[2:], CU[2:]))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))
                    # print 'LVUC = ', LVUC

                    # check
                    # tmp = C[inds[2:],:] + i*(a2+a1) + j*a1
                    # plt.plot(tmp[:,0],tmp[:,1])
                    # plt.pause(.1)
                elif j < NH - 1:
                    # not last cell of row, include all but pts 0,1,5
                    rr = np.vstack((rr, C[inds[2:5], :] + i * (a2 + a1) + j * a1))
                    LVUCadd = np.dstack(((i + j) * tmp1[2:5], i * tmp1[2:5], CU[2:5]))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))

                    # check
                    # tmp = C[inds[2:5],:] + i*(a2+a1) + j*a1
                    # plt.plot(tmp[:,0],tmp[:,1])
                    # plt.pause(.1)
                else:
                    # keep the last bottom right corner (pt 1)
                    rr = np.vstack((rr, C[inds[1:5], :] + i * (a2 + a1) + j * a1))
                    LVUCadd = np.dstack(((i + j) * tmp1[1:5], i * tmp1[1:5], CU[1:5]))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))
                    # print 'LVUC = ', LVUC

                    # check
                    # tmp = C[inds[1:5],:] + i*(a2+a1) + j*a1
                    # plt.plot(tmp[:,0],tmp[:,1])
                    # plt.pause(.1)

    # check for repeated points
    print('Checking for repeated points...')
    print 'len(R) =', len(rr)
    rcheck = dh.unique_rows(rr)
    print 'len(Rcheck) =', len(rcheck)
    if len(rr) - len(rcheck) != 0:
        sizes = np.random.rand(len(rr)) * 100
        colors = np.random.rand(len(rr))
        plt.scatter(rr[:, 0], rr[:, 1], s=sizes[::-1], edgecolor='k', facecolor='none')
        plt.show()
        # raise RuntimeError('Repeated points!')
        print 'REPEATED POINTS EXIST!! WARNING!!\n --> If alph = pi/3, then this will occur, but should be ok.'
        full_deform = True
    else:
        print 'No repeated points.\n'
        full_deform = False

    # xy = R
    # plt.plot(xy[:,0],xy[:,1],'b.')
    # for i in range(NV):
    #     for j in range(NH):
    #         plt.plot(C[:,0]+j*a1[0]+i*a2[0],C[:,1]+j*a1[1]+i*a2[1],'r-')
    # plt.show()

    xy = rr - np.array([np.mean(rr[:, 0]), np.mean(rr[:, 1])])
    print 'len(xy) = ', len(xy)
    print 'len(LVUC) = ', len(LVUC)
    if len(rr) - len(LVUC) != 0:
        sizes = np.arange(len(xy))
        raise RuntimeError('LVUC and R lengths do not match!')

    if full_deform:
        xytrash, NL, KL, BL, LVUCtrash, LVtrash, UCtrash, lattice_extentrash = generate_twisted_kagome(shape,
                                                                                                       NH, NV, 0.000)
    else:
        # Triangulate
        print('Triangulating...')
        Dtri = Delaunay(xy)
        btri = Dtri.vertices
        # translate btri --> bond list
        BL = le.Tri2BL(btri)

        # remove bonds on the sides and through the hexagons
        print('Removing extraneous bonds from triangulation...')
        # calc vecs from C bonds
        CBL = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])

        BL = blf.latticevec_filter(BL, xy, C, CBL)
        NL, KL = le.BL2NLandKL(BL, NN=4)

    lattice_exten = 'twisted_kagome_' + shape + '_alph_' + '{0:.4f}'.format(alph)
    print 'LVUC = ', LVUC

    LV = a
    UC = C
    print '\n\n\n\n\n\n\n\n xyLVoffset = ', np.array([np.mean(rr[1:, 0]), np.mean(rr[1:, 1])])
    return xy, NL, KL, BL, LVUC, LV, UC, lattice_exten


def generate_deformed_kagome(lp):
    """creates distorted kagome lattice as in Kane&Lubensky2014

    Parameters
    ----------
    lp : dict
        includes:
        shape : string
            overall shape of the mesh ('square' 'circle') --> haven't built in functionality yet
        NH : int
            Number of pts along horizontal before boundary is cut
        NV : int
            Number of pts along vertical before boundary is cut
        x1 : float
            symmetrical representation of deformation (sp = xp*(a[p-1]+a[p+1])+yp*a[p])
        x2 : float
            symmetrical representation of deformation (sp = xp*(a[p-1]+a[p+1])+yp*a[p])
        x3 : float
            symmetrical representation of deformation (sp = xp*(a[p-1]+a[p+1])+yp*a[p])
        z : float
            z= y1+y2+y3, symmetrical representation of deformation (sp = xp*(a[p-1]+a[p+1])+yp*a[p])

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
    LVUCV : NP x 4 array
        For each particle, gives (lattice vector, unit cell vector) coordinate position of that particle: LV1, LV2, UCV1, UCV2
    lattice_type : string
        label, lattice type.  For making output directory
    """
    check = lp['check']
    shape, NH, NV, x1, x2, x3, zz = lp['shape'], lp['NH'], lp['NV'], lp['x1'], lp['x2'], lp['x3'], lp['z']
    theta = lp['theta'] * np.pi
    if lp['NH'] == 1 and lp['NV'] == 1:
        xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = \
            kagaux.build_deformed_kagome_unitcell(lp)
    else:
        print('Setting up unit cell for building lattice...')
        # Bravais primitive unit vecs
        aa = np.array([[np.cos(2 * np.pi * float(p) / 3.), np.sin(2 * np.pi * float(p) / 3.)] for p in np.arange(3.)])
        a1, a2, a3 = aa[0], aa[1], aa[2]
        LV = np.vstack((a1, a1 + a2))

        # make unit cell
        xx = np.array([x1, x2, x3])
        print 'xx = ', xx
        print 'zz = ', zz
        yy = np.array([zz / 3. + xx[np.mod(i - 1, 3)] - xx[np.mod(i + 1, 3)] for i in [0, 1, 2]])
        print 'xx, yy = ', xx, yy
        ss = np.array([xx[p] * (aa[np.mod(p - 1, 3)] - aa[np.mod(p + 1, 3)]) + yy[p] * aa[p] for p in range(3)])
        s1 = ss[0]
        s2 = ss[1]
        s3 = -s1 - s2
        d1 = a1 * 0.5 + s2
        d2 = a2 * 0.5 - s1
        d3 = a3 * 0.5
        dd = np.array([d1, d2, d3])

        # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
        cc = np.array([d1 + a2,
                       d3 + a1 + a2,
                       d2 + a1,
                       d1,
                       d3 + a1,
                       d2 - a2,
                       d1 + a3,
                       d3,
                       d2 + a3,
                       d1 + a2 + a3,
                       d3 + a2,
                       d2])
        CU = np.arange(len(cc))
        #
        #         0
        #
        # 10  11     1    2
        #
        #    9        3
        #
        #  8   7     5    4
        #
        #         6
        #

        # Check it
        # print 'dd = ', dd
        # print 'ss = ', ss
        # plt.plot(cc[:, 0], cc[:, 1], 'r-')
        # plt.show()
        # sys.exit()

        if lp['periodic_strip']:
            # Build periodic connections on the right and left
            periodicstr = '_periodic_strip'
            periodicstrip = True
        elif lp['periodicBC']:
            # Build periodic connections on the bottom and top as well as right and left
            periodicstr = '_periodic'
            periodicstrip = False

        print('Translating by Bravais lattice vectors...')
        inds = np.arange(len(cc))
        eps = 1e-2
        # If shape is 'wedge', cut the sample into a triangle
        if lp['shape'] == 'wedge':
            tmp1 = np.ones_like(CU)
            tmp0 = np.zeros_like(CU)
            for i in np.arange(NV):
                # i is the index of the row (vertical index)
                for j in np.arange(max(0, i - 1), NH):
                    # j is the index of the col (horiz index)
                    if i == 0:
                        if j == i:
                            # initialize
                            inds = [0, 1, 2, 3, 4, 5, 7, 9, 11]
                            rr = cc[inds]
                            LVUC = np.dstack((tmp0, tmp0, CU))[0]
                        else:
                            # bottom row
                            rr = np.vstack((rr, cc[0:6, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack((j * tmp1[inds], tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                    elif i == NV - 1:
                        if j == i - 1:
                            # first cell of row, include only particle 5
                            inds = [5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        elif j == i:
                            # first cell of top row
                            inds = [1, 2, 3, 4, 5, 7, 9, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # only points 0 through 5 included, for top row
                            inds = [1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                    else:
                        if j == i - 1:
                            # first cell of row, include only particle 5
                            inds = [5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        elif j == i:
                            # first full cell of row
                            inds = [0, 1, 2, 3, 4, 5, 7, 9, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # only points 0 through 5 included
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((2 * i + j) * tmp1[inds], i * tmp0[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                    # check:
                    # plt.plot(rr[:, 0], rr[:, 1], 'b.')
                    # plt.title('i = ' + str(i) + ' j = ' + str(j))
                    # plt.show()
        elif np.abs(theta) < eps:
            thetastr = ''
            # todo: make LVUC for square sample
            tmp1 = np.ones_like(CU)
            tmp0 = np.zeros_like(CU)
            for i in np.arange(NV):
                for j in np.arange(NH):
                    if i == 0:
                        if j == 0 and periodicstrip:
                            # first cell of row, include all but pt 6
                            inds = [0, 7, 8, 9, 10, 11]
                            rr = cc[inds, :]
                            LVUC = np.dstack((tmp0[inds], tmp0[inds], CU[inds]))[0]
                        elif j == 0:
                            # initialize
                            rr = cc
                            LVUC = np.dstack((tmp0, tmp0, CU))[0]
                        elif j == NH - 1 and periodicstrip:
                            # last cell of row, to be connected to the left
                            inds = [0, 6, 7, 8, 9, 10, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # bottom row (include point 6 in translation)
                            inds = [0, 1, 2, 3, 4, 5, 6]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                    elif i == NV - 1:
                        # Top row: exclude particle 0 for smooth boundary
                        if j == NH - 1 and periodicstrip:
                            # last cell of top row that connects periodically to the left edge
                            inds = [7, 8, 9, 10, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        elif j == 0:
                            # first cell of top row
                            inds = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # bulk of top row
                            inds = [1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                    else:
                        if j == NH - 1 and periodicstrip:
                            # last cell of row that connects periodically to the left edge
                            inds = [0, 7, 8, 9, 10, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        elif j == 0:
                            # first cell of row, include all but pt 6
                            inds = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # bulk of row that is neither top nor bottom: only points 0 through 5 included
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * (2 * a2 + a1) + j * a1))
                            LVUCadd = np.dstack(((i + j) * tmp1[inds], 2 * i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
        elif abs(theta - np.pi * 0.5) < eps:
            lp['theta'] = 0.5
            theta = lp['theta'] * np.pi
            thetastr = '_theta0p500'
            # This network will have a bottom edge along the LV[1] direction
            # Build this rotated network, then rotate it
            tmp1 = np.ones_like(CU)
            tmp0 = np.zeros_like(CU)
            for i in np.arange(NV):
                for j in np.arange(NH):
                    if i == 0:
                        if j == 0 and periodicstrip:
                            # initialize
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = cc[inds]
                            LVUC = np.dstack((tmp0[inds], tmp0[inds], CU[inds]))[0]

                            if check:
                                plt.plot(rr[:, 0], rr[:, 1], '.')
                                for tmp in range(len(rr)):
                                    plt.text(rr[tmp, 0] + 0.1, rr[tmp, 1], str(tmp))
                                plt.axis('scaled')
                                plt.pause(0.2)
                        else:
                            raise RuntimeError('have not coded this')
                    else:
                        if j == NH - 1 and periodicstrip:
                            # first cell of row, include all but pt 6
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * a1 + j * (2 * a2 - a1)))
                            LVUCadd = np.dstack(((-i + j) * tmp1[inds], (i + j) * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            # check it
                            if check:
                                plt.plot(rr[:, 0], rr[:, 1], '.')
                                for tmp in range(len(rr)):
                                    plt.text(rr[tmp, 0] + 0.1, rr[tmp, 1], str(tmp))
                                plt.axis('scaled')
                                plt.pause(0.2)
                        else:
                            raise RuntimeError('have not coded this')
            rr = mf.rotate_vectors_2D(rr, -theta)

        elif abs(np.arctan2(a2[1] - a1[1], a2[0] - a1[0]) - theta - np.pi * 0.5) < eps:
            lp['theta'] = 1./3.
            theta = lp['theta'] * np.pi
            thetastr = '_theta0p333'
            # This network will have a bottom edge along the LV[1] direction
            # Build this rotated network, then rotate it
            tmp1 = np.ones_like(CU)
            tmp0 = np.zeros_like(CU)
            for i in np.arange(NV):
                for j in np.arange(NH):
                    if i == 0:
                        if j == 0 and periodicstrip:
                            # initialize
                            inds = [0, 1, 2, 3, 11]
                            rr = cc[inds]
                            LVUC = np.dstack((tmp0[inds], tmp0[inds], CU[inds]))[0]

                            if check:
                                plt.plot(rr[:, 0], rr[:, 1], '.')
                                for tmp in range(len(rr)):
                                    plt.text(rr[tmp, 0] + 0.1, rr[tmp, 1], str(tmp))
                                plt.axis('scaled')
                                plt.pause(0.2)
                        else:
                            raise RuntimeError('have not coded this')
                    else:
                        if j == NH - 1 and periodicstrip:
                            # first cell of row, include all but pt 6
                            inds = [0, 1, 2, 3, 4, 11]
                            rr = np.vstack((rr, cc[inds, :] + i * (a2 - a1) + j * (a1 + a2)))
                            LVUCadd = np.dstack(((-i + j) * tmp1[inds], (i + j) * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            # check it
                            if check:
                                plt.plot(rr[:, 0], rr[:, 1], '.')
                                for tmp in range(len(rr)):
                                    plt.text(rr[tmp, 0] + 0.1, rr[tmp, 1], str(tmp))
                                plt.axis('scaled')
                                plt.pause(0.2)
                        else:
                            raise RuntimeError('have not coded this')
            rr = mf.rotate_vectors_2D(rr, -theta)

            # check it
            if check:
                print(rr)
                plt.plot(rr[:, 0], rr[:, 1], '.-')
                plt.axis('scaled')
                plt.show()
        else:
            print 'theta = ', theta
            print 'np.arctan2(a2-a1)/pi - 0.5 = ', np.arctan2(a2[1] - a1[1], a2[0] - a1[0]) / np.pi - 0.5
            print 'np.arctan2(a2)/pi - 0.5 = ', np.arctan2(a2[1], a2[0]) / np.pi - 0.5
            raise RuntimeError('No matching clause for this theta exists')

        # check it
        # print(rr)
        # plot(rr[:,0],rr[:,1],'-')
        # sys.exit()

        # check for repeated points
        print('Checking for repeated points...')
        print 'len(rr) =', len(rr)
        rrcheck = dh.unique_rows(rr)
        print 'len(rrcheck) =', len(rrcheck)
        if len(rr) - len(rrcheck) != 0:
            sizes = np.arange(len(rr))
            plt.scatter(rr[:, 0], rr[:, 1], s=sizes)
            plt.title('Repeated points!')
            plt.show()
            raise RuntimeError('Repeated points!')
        else:
            print 'No repeated points.\n'
        xy = rr - np.array([np.mean(rr[1:, 0]), np.mean(rr[1:, 1])])
        # Triangulate
        print('Triangulating...')
        Dtri = Delaunay(xy)
        btri = Dtri.vertices
        # translate btri --> bond list
        BL = le.Tri2BL(btri)

        # remove bonds on the sides and through the hexagons
        print('rremoving extraneous bonds from triangulation...')
        # calc vecs from C bonds
        CBL = np.array([[0, 1], [1, 11], [0, 11], [1, 2], [2, 3], [1, 3], [3, 4], [4, 5], [3, 5],
                        [5, 6], [6, 7], [5, 7], [7, 8], [8, 9], [7, 9], [9, 10], [10, 11], [9, 11]])
        if abs(theta) < eps:
            BL = blf.latticevec_filter(BL, xy, cc, CBL)
        else:
            BL = blf.latticevec_filter(BL, xy, mf.rotate_vectors_2D(cc, -theta), CBL)

        NL, KL = le.BL2NLandKL(BL, NN=4)
        if lp['periodicBC']:
            PVx = np.zeros_like(NL, dtype=float)
            PVy = np.zeros_like(NL, dtype=float)
            PVxydict = {}

            # First check if theta is nonzero
            if np.abs(theta) < eps:
                LV = np.vstack((a1, a1 + a2))
                # build here
                for ind in range(len(xy)):
                    if NH == 1:
                        print 'ind = ', ind
                        print 'LVUC[ind] = ', LVUC[ind]
                        # ind is on right edge if LVUC[ind] == [i + NH - 1, 2*i, 0 or 6]
                        on_rightedge = LVUC[ind, 2] in [0, 6, 7, 11] and \
                                       LVUC[ind, 0] == NH - 1 + int(LVUC[ind, 1] * 0.5)
                        if LVUC[ind, 2] == 6 and LVUC[ind, 1] == 0 and not periodicstrip:
                            msg = 'This is particle 6 on bottom (in the lower appendage), ' \
                                  'to be connected with particle UC=7 ' + \
                                  'on the sample top if this is not a periodic strip'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0] - int(NV - 1), 2 * (NV - 1), 7]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1)) + 2 * (NV + 1) * LV[1]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                        elif LVUC[ind, 2] in [6, 7] and on_rightedge:
                            # connect rightmost 6 with leftmost 8
                            msg = 'This is particle 6 or 7 on the right, ' \
                                  'to be connected with particle UC=8 on the sample left'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0], LVUC[ind, 1], 8]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                        elif LVUC[ind, 2] == 11 and on_rightedge:
                            # connect rightmost 6 with leftmost 10
                            msg = 'This is particle 11 on the right, ' \
                                  'to be connected with particle UC=10 on the sample left'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0], LVUC[ind, 1], 10]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                        elif LVUC[ind, 2] == 0 and on_rightedge:
                            # connect rightmost 6 with leftmost 10
                            msg = 'This is particle 0 on the right, ' \
                                  'to be connected with particle UC=10 and UC=8 on the sample left'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0], LVUC[ind, 1], 10]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                            # Also connect to particle UC=8 two rows higher
                            want = [LVUC[ind, 0] + 1, LVUC[ind, 1] + 2, 8]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                    else:
                        raise RuntimeError('have not coded this case')
            elif abs(theta - np.pi * 0.5) < eps:
                # Theta is nonzero, build periodic bonds here
                LV = np.vstack((a1, a1 + a2))
                # build here
                for ind in range(len(xy)):
                    if NH == 1:
                        print 'ind = ', ind
                        print 'LVUC[ind] = ', LVUC[ind]
                        # ind is on right edge if LVUC[ind] == [i + NH - 1, 2*i, 0 or 6]
                        on_rightedge = LVUC[ind, 2] in [0] and LVUC[ind, 0] == -LVUC[ind, 1]
                        if LVUC[ind, 2] == 0 and on_rightedge:
                            msg = 'This is particle 0 on bottom (in the lower appendage), ' \
                                  'to be connected with particle UC=5 ' + \
                                  'on the sample top if this is not a periodic strip'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0], LVUC[ind, 1], 5]
                            # Perioidic virtual displacement vector of ind
                            PV = -np.array([2 * LV[1, 1], 0])
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])

                            # If not on top row, connect to next lower row
                            if LVUC[ind, 1] > 0:
                                msg = 'This is particle 0 on bottom (in the lower appendage), ' \
                                      'to be connected with particle UC=5 ' + \
                                      'on the sample top if this is not a periodic strip'
                                # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                                want = [LVUC[ind, 0] + 1, LVUC[ind, 1] - 1, 4]
                                # Perioidic virtual displacement vector of ind
                                PV = -np.array([2 * LV[1, 1], 0])
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=lp['check'])

            elif abs(np.arctan2(a2[1] - a1[1], a2[0] - a1[0]) - theta - np.pi * 0.5) < eps:
                # Theta is nonzero, build periodic bonds here
                LV = np.vstack((a1, a1 + a2))
                # build here
                for ind in range(len(xy)):
                    if NH == 1:
                        print 'ind = ', ind
                        print 'LVUC[ind] = ', LVUC[ind]
                        # ind is on right edge if LVUC[ind] == [i + NH - 1, 2*i, 0 or 6]
                        on_rightedge = LVUC[ind, 2] in [0, 2] and LVUC[ind, 0] == -LVUC[ind, 1]
                        if LVUC[ind, 2] == 3 and LVUC[ind, 1] == 0 and not periodicstrip:
                            msg = 'This is particle 3 on bottom (in the lower appendage), ' \
                                  'to be connected with particle UC=?? ' + \
                                  'on the sample top if this is not a periodic strip'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0] - int(NV - 1), 2 * (NV - 1), 7]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1)) + 2 * (NV + 1) * LV[1]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                        elif LVUC[ind, 2] == 0 and on_rightedge:
                            if LVUC[ind, 1] < NV - 1:
                                # connect rightmost 0 with leftmost 4, a2 - a1 above (ie next row higher), so long
                                # as not on top row.
                                msg = 'This is particle 0 on the right, ' \
                                      'to be connected with particle UC=4 on the sample left'
                                # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                                want = [LVUC[ind, 0] - 1, LVUC[ind, 1] + 1, 4]
                                # Perioidic virtual displacement vector of ind
                                PV = -LV[0] * NH
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                            # Also connect to 11 across periodic boundary
                            msg = 'This is particle 0 on the right, ' \
                                  'to be connected with particle UC=11 on the sample left'
                            want = [LVUC[ind, 0], LVUC[ind, 1], 11]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                        elif LVUC[ind, 2] == 2 and on_rightedge:
                            if LVUC[ind, 1] > 0:
                                # connect rightmost 6 with leftmost 4 at same row, as long as not on bottom row
                                msg = 'This is particle 11 on the right, ' \
                                      'to be connected with particle UC=10 on the sample left'
                                # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                                want = [LVUC[ind, 0], LVUC[ind, 1], 4]
                                # Perioidic virtual displacement vector of ind
                                PV = -LV[0] * NH
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=lp['check'])

                            # Also connect to 3 across periodic boundary
                            msg = 'This is particle 0 on the right, ' \
                                  'to be connected with particle UC=3 on the sample left'
                            want = [LVUC[ind, 0], LVUC[ind, 1], 3]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xy, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=lp['check'])
                    else:
                        raise RuntimeError('have not coded this case (NH > 1) and theta != 0')
        else:
            periodicstr = ''
            PVxydict = {}
            PVx = None
            PVy = None
            PV = None

        lattice_exten = 'deformed_kagome_' + shape + periodicstr + thetastr + \
                        '_x1_' + '{0:.4f}'.format(x1) \
                        + '_x2_' + '{0:.4f}'.format(x2) + \
                        '_x3_' + '{0:.4f}'.format(x3) + '_z_' + '{0:.4f}'.format(zz)
        UC = cc
        LV = np.vstack((a1, a1 + a2))
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        BBox = np.array([[np.min(xy[:, 0]), np.min(xy[:, 1])],
                         [np.min(xy[:, 0]), np.max(xy[:, 1])],
                         [np.max(xy[:, 0]), np.max(xy[:, 1])],
                         [np.max(xy[:, 0]), np.min(xy[:, 1])]])
        xy -= np.mean(xy, axis=0)
    return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten


def add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL, PVx, PVy, PVxydict, msg='', check=False):
    """

    Parameters
    ----------
    ind
    want
    xypts
    LVUC
    PV
    NL
    KL
    PVx
    PVy
    PVxydict
    msg
    check

    Returns
    -------

    """
    if check:
        print 'LVUC[ind] = ', LVUC[ind]
        print msg
        print 'Looking for particle with LVUC=', want
        plt.plot(xypts[:, 0], xypts[:, 1], 'b.')
        for dmyi in range(len(xypts)):
            plt.text(xypts[dmyi, 0] + 0.1, xypts[dmyi, 1] + 0.1, str(LVUC[dmyi]))
        plt.plot(xypts[ind, 0], xypts[ind, 1], 'ro')
        plt.title('Looking for particle with LVUC=' + str(want))
        plt.axis('scaled')
        plt.show()

    newn = np.where((LVUC[:, 0] == want[0]) * (LVUC[:, 1] == want[1]) *
                    (LVUC[:, 2] == want[2]))[0][0]

    # get first zero in KL
    # print 'KL = ', KL
    # print 'KL[ind] = ', KL[ind]
    try:
        firstzero = np.where(KL[ind, :] == 0)[0][0]
    except IndexError:
        print 'ind = ', ind
        print 'KL[ind] = ', KL[ind]
        raise RuntimeError('KL row seems to be full. See output above for the row.')
    NL[ind, firstzero] = newn
    KL[ind, firstzero] = -1

    fznewn = np.where(KL[newn, :] == 0)[0][0]
    KL[newn, fznewn] = -1
    NL[newn, fznewn] = ind

    BL = np.vstack((BL, np.array([-ind, -newn])))

    # Enter element into PVx and PVy arrays
    PVx[ind, firstzero] = -PV[0]
    PVy[ind, firstzero] = -PV[1]
    PVx[newn, fznewn] = PV[0]
    PVy[newn, fznewn] = PV[1]

    # check if this bond is already in PVxydict
    if ind < newn:
        if (ind, newn) in PVxydict:
            PVxydict[(ind, newn)] = np.vstack((PVxydict[(ind, newn)], -PV))
        else:
            PVxydict[(ind, newn)] = -PV
    else:
        if (newn, ind) in PVxydict:
            PVxydict[(newn, ind)] = np.vstack((PVxydict[(newn, ind)], PV))
        else:
            PVxydict[(newn, ind)] = PV

    return NL, KL, BL, PVx, PVy, PVxydict


def build_kagper_hex(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = \
        build_hexagonal.generate_honeycomb_lattice(lp)

    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    if lp['shape'] == 'square':
        polygon = blf.auto_polygon(lp['shape'], LL[0], LL[1], eps=0.00)
    else:
        polygon = blf.auto_polygon(lp['shape'], NH, NV, eps=0.00)
    BBox = polygon
    print 'BBox = ', BBox

    # Select some fraction of vertices (which are points) --> xypick gives Nkag of the vertices (xy)
    Nkag = round(lp['percolation_density'] * len(xy))
    ind_shuffled = np.random.permutation(np.arange(len(xy)))
    xypick = np.sort(ind_shuffled[0:Nkag])

    # todo: make sure periodic bcs are well handled here
    xy, BL = blf.decorate_kagome_elements(xy, BL, xypick, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)

    lattice_exten = 'kagper_hex_' + lattice_exten[10:] + \
                    '_perd' + sf.float2pstr(lp['percolation_density'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kagome(lp):
    """
    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------

    """
    if lp['NV'] == 1 and lp['NH'] == 1:
        if lp['periodicBC']:
            xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = kagaux.kagome_unitcell(lp)
        else:
            # lattice is not periodic, return a full kagome star
            raise RuntimeError("have not written yet")
    else:
        lp_big = copy.deepcopy(lp)
        lp_big['check'] = False
        # lp_big['periodicBC'] = False
        xypts, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = \
            build_hexagonal.generate_honeycomb_lattice(lp_big)
        xy, BL, PVxydict = blf.decorate_as_kagome(xypts, BL, PVxydict=PVxydict, check=lp['check'])
        xy -= np.mean(xy, axis=0)
        NL, KL = le.BL2NLandKL(BL)
        if lp['periodicBC']:
            LL = (PV[0, 0], PV[1, 1])
            BBox = np.array([[-LL[0] * 0.5, -LL[1] * 0.5], [LL[0] * 0.5, -LL[1] * 0.5],
                             [LL[0] * 0.5, LL[1] * 0.5], [-LL[0] * 0.5, LL[1] * 0.5]])

            # xy, NL, KL, BL, PVxydict = le.buffered_pts_to_periodic_network(xy, BL, LL, BBox='auto', check=lp['check'])
            PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)
            addstr = '_periodic'

        # Rescale so that median bond length is unity
        bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
        scale = 1. / np.median(bL)
        xy *= scale
        BBox *= scale
        LL = (LL[0] * scale, LL[1] * scale)
        if lp['periodicBC']:
            PV *= scale
            PVx *= scale
            PVy *= scale
            PVxydict.update((key, val * scale) for key, val in PVxydict.items())

        else:
            LL = (lp['NH'] * 2, lp['NV'] * np.sqrt(3))
            BBox = np.array([[-LL[0] * 0.5, -LL[1] * 0.5], [LL[0] * 0.5, -LL[1] * 0.5],
                             [LL[0] * 0.5, LL[1] * 0.5], [-LL[0] * 0.5, LL[1] * 0.5]])
            xy, NL, KL, BL = blf.mask_with_polygon(BBox, lp['NH'], lp['NV'], xy, BL, eps=0.00, check=lp['check'])
            addstr = ''

        # Create lattice_exten
        print 'lattice_exten = ', lattice_exten
        lattice_exten = lp['LatticeTop'] + '_' + lp['shape'] + addstr + lattice_exten[16:]

        # Create LV and UC
        # Bravais primitive unit vecs
        scale = 2. * np.abs(np.cos(0.0))
        LV = scale * np.array([[np.cos(2 * np.pi * float(p) / 3.), np.sin(2 * np.pi * float(p) / 3.)]
                               for p in np.arange(3.)])
        # NOTE that a3 = -a1 -a2
        #      ^
        #       \ a2
        #        \
        #          -------> a1
        #        /
        #       /
        #      v  a3 = -a1-a2

        # make composite unit cell (multiple unit cells in size) --> regular kagome
        psi = [n * np.pi * (1. / 3.) for n in range(6)]
        b = np.array([[np.cos(psi[i]), np.sin(psi[i])] for i in range(6)])

        # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
        UC = np.array([[0., 0.],
                       b[0],
                       b[0] + b[1],
                       b[0] + b[1] + b[2],
                       b[0] + b[1] + b[2] + b[3],
                       b[0] + b[1] + b[2] + b[3] + b[4]])
    return xy, NL, KL, BL, PVx, PVy, PVxydict, PV, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kagsplit_hex(lp):
    """Build a hyperuniform centroidal lattice with kagomized points beyond distance alph*Radius/Halfwidth of sample

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    """
    xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = build_hexagonal.generate_honeycomb_lattice(lp)
    # ['shape'], lp['NH'], lp['NV'], lp['delta'], lp['phi'], eta=lp['eta'],
    #                               rot=lp['theta'], periodicBC=lp['periodicBC'], check=lp['check'])

    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    if lp['shape'] == 'square':
        polygon = blf.auto_polygon(lp['shape'], LL[0], LL[1], eps=0.00)
    else:
        polygon = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)
    BBox = polygon
    print 'BBox = ', BBox

    # Grab indices (vertices) to kagomize: select the ones to the right of alph*characteristic length from center
    if lp['shape'] == 'square':
        lenscaleX = LL[0] * lp['alph'] + np.min(BBox[:, 0])
        kaginds = np.where(xy[:, 0] > lenscaleX)[0]
    elif lp['shape'] == 'circle':
        # todo: handle circles
        pass
    elif lp['hexagon'] == 'hexagon':
        # todo: handle hexagons
        pass

    # todo: make sure periodic bcs are well handled here
    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for kagper_hucent network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = 'kagsplit_hex' + lattice_exten[9:] + '_alph' + sf.float2pstr(lp['alph'], ndigits=2)
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def deformed_kagome_domainwall(NH1, NH2, NV, xz1, xz2):
    """Create deformed kagome lattice with domain wall between 2 types"""
    # create each domain
    xy1, NL1, KL1, BL1, lattype = generate_deformed_kagome(NH1, NV, xz1[0], xz1[1], xz1[2], xz1[3])
    xy2, NL2, KL2, BL2, lattype = generate_deformed_kagome(NH2, NV, xz1[0], xz1[1], xz2[2], xz2[3])
    # string domains together
    # equate rightmost points of xy1 with leftmost points of domain xy2
    # modify NL2 and BL2 to reflect identification, add len(xy1)-len(idpts)
    xy1
    raise RuntimeError('not finished with this yet')
    return xy, NL, KL, BL, lattype


def build_accordionkag(lp):
    """Build hexagonal-like lattice, replacing each bond with a zigzag set of bonds, then replace sites that would be
    vertices of a hexagonal lattice with kagomized elements
    example:
    python ./build/make_lattice.py -LT accordionkag -N 1 -alph 0.4 -intparam 2 -periodic -skip_polygons -skip_gyroDOS

    Parameters
    ----------
    lp

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    from lepm.build.build_hexagonal import build_accordionhex
    # nzag controlled by lp['intparam'] below
    check = copy.deepcopy(lp['check'])
    lp['check'] = False
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvertices = build_accordionhex(lp)

    # need indices of xy that correspond to xyvertices
    # note that xyvertices gives the positions of the vertices, not their indices
    inRx = np.in1d(xy[:, 0], xyvertices[:, 0])
    inRy = np.in1d(xy[:, 1], xyvertices[:, 1])
    vxind = np.where(np.logical_and(inRx, inRy))[0]
    if (BL < 0).any() > 0:
        PV = le.PVxydict2PV(PVxydict)
        xy, BL, NL, KL, PVx, PVy, PVxydict = blf.decorate_bondneighbors_elements_periodic(
                xy, BL, vxind, NL, KL, PVxydict, PV, viewmethod=False, check=check)
    else:
        PV = None
        xy, BL = blf.decorate_bondneighbors_elements(xy, BL, vxind, NL=NL, KL=KL, PVxydict=PVxydict, viewmethod=False,
                                                     check=lp['check'])
        NL, KL = le.BL2NLandKL(BL, NP=len(xy))

    lattice_exten = 'accordionkag' + lattice_exten[12:]
    return xy, NL, KL, BL, PVx, PVy, PVxydict, PV, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kagjunction(lp):
    """Build hexagonal-like lattice, replacing each bond with a zigzag set of bonds, then replace sites that would be
    vertices of a hexagonal lattice with kagomized elements

    Parameters
    ----------
    lp

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    from lepm.build.build_hexagonal import build_hexjunction
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvertices = build_hexjunction(lp)

    # need indices of xy that correspond to xyvertices
    # note that xyvertices gives the positions of the vertices, not their indices
    try:
        inRx = np.in1d(xy[:, 0], xyvertices[:, 0])
        inRy = np.in1d(xy[:, 1], xyvertices[:, 1])
    except:
        inRx = np.in1d(xy[:, 0], [xyvertices[0]])
        inRy = np.in1d(xy[:, 1], [xyvertices[1]])
    vxind = np.where(np.logical_and(inRx, inRy))[0]

    xy, BL = blf.decorate_bondneighbors_elements(xy, BL, vxind, NL=NL, KL=KL, PVxydict=PVxydict, viewmethod=False,
                                                 check=lp['check'])
    NL, KL = le.BL2NLandKL(BL, NP=len(xy))
    lattice_exten = 'kag' + lattice_exten[3:]
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
