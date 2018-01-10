import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import copy
from scipy.spatial import Delaunay
import lepm.data_handling as dh
import lepm.plotting.network_visualization as netvis
import lepm.math_functions as mf

'''
Description
===========
Functions for making spindle lattices
'''


def build_spindle(lp):
    """Create a spindle lattice using lattice parameters lp.
    Note that either 'delta' (a float, opening angle in radians) or 'delta_lattice' (a string, angle in units of pi)
    must be in lp.
    Example usage:
    python run_series.py -pro ./build/make_lattice -opts LT/spindle/-N/1/-skip_polygons/-skip_gyroDOS/-periodic \
        -var alph 0.0:0.02:0.33

    Parameters
    ----------
    lp : dict

    Returns
    -------

    """
    if 'delta' not in lp:
        # if delta not in lp, then delta_lattice must be in lp
        if lp['delta_lattice'] in ['0.667', '0p667']:
            lp['delta'] = np.pi * 2. / 3.
        else:
            if isinstance(lp['delta_lattice'], str):
                delta_lattice = lp['delta_lattice'].replace('p', '.')
            else:
                delta_lattice = lp['delta_lattice']
            lp['delta'] = np.pi * float(delta_lattice)
    elif 'delta_lattice' not in lp:
        # if delta_lattice not in lp, then delta must be in lp (a float in units of radians)
        lp['delta_lattice'] = sf.float2pstr(lp['delta'] / np.pi, ndigits=3)

    if 'phi' not in lp:
        lp['phi'] = np.pi * float(lp['phi_lattice'])

    print 'checking that theta = ', lp['theta']
    print 'checking that periodicBC = ', lp['periodicBC']
    if lp['NH'] == 1 or lp['NV'] == 1:
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_spindle_strip(lp)
    else:
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_spindle_lattice(lp)

    if PV is None:
        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        if lp['shape'] == 'square':
            BBox = blf.auto_polygon(lp['shape'], LL[0], LL[1], eps=0.00)
        else:
            BBox = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)
    else:
        if lp['shape'] == 'square':
            print 'PV = ', PV
            LL = (np.max(PV[:, 0]), np.max(PV[:, 1]))
            BBox = 0.5 * np.array([[-LL[0], -LL[1]], [-LL[0], LL[1]], [LL[0], LL[1]], [LL[0], -LL[1]]])
        else:
            raise RuntimeError('Make the hexagonal BBox here.')
            LL = ()
            BBox = np.array([])

    return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_spindle_lattice(lp):
    """Generates hexagonal lattice (points, connectivity, name).
    Example usage:
    python ./build/make_lattice.py -LT spindle

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary, with keys:
        shape : string or dict with keys 'description':string and 'polygon':polygon array
            Global shape of the mesh, in form 'square', 'hexagon', etc or as a dictionary with keys
            shape['description'] = the string to name the custom polygon, and
            shape['polygon'] = 2d numpy array
        NH : int
            Number of pts along horizontal. If shape='hexagon', this is the width (in cells) of the bottom side (a)
        NV : int
            Number of pts along vertical, or 2x the number of rows of lattice
        delta : float
            Deformation angle for the lattice in degrees (for undeformed hexagonal lattice, this is 0.66666*np.pi)
        phi : float
            Shear angle for the lattice in radians, must be less than pi/2 (for undeformed hexagonal lattice, this is 0.000)
        eta : float
            randomization of the lattice (a scaling of random jitter in units of lattice spacing)
        rot : float
            angle in units of pi to rotate the lattice vectors and unit cell
        periodicBC : bool
            Wether to apply periodic boundaries to the network
        check : bool
            Wehter to plot output at intermediate steps

    Returns
    ----------
    xy : matrix of dimension nx2
        Equilibrium lattice positions
    NL : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    lattice_exten : string
        description of the lattice, complete with parameters for properly saving the lattice files
    LV : 3 x 2 float array
        Lattice vectors for the kagome lattice with input twist angle
    UC : 6 x 2 float array
        (extended) unit cell vectors
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    PVx : NP x NN float array
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        If NL and KL are remade, PVx will not be ordered properly: use dict instead
    PVy : NP x NN float array
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        If NL and KL are remade, PVy will not be ordered properly: use dict instead
    LVUC : NP x 3 array
        Lattice vectors and (extended) unit cell vectors defining relative position of each point, as integer multiples
        of LV[0] and LV[1], and UC[LVUC[i,2]]
        For instance, xy[0,:] = LV[0]*LVUC[0,0] + LV[1]*LVUC[0,1] + UC[LVUC[0,2]]

    """
    shape = lp['shape']
    NH = lp['NH']
    NV = lp['NV']
    delta = lp['delta']
    alpha = lp['alph'] * np.pi
    # todo: handle phi_lattice and eta_lattice case
    phi = lp['phi']
    eta = lp['eta']
    # todo: handle theta_lattice case
    rot = lp['theta']
    rot *= np.pi
    periodicBC = lp['periodicBC']
    check = lp['check']

    # If we have chosen periodic_strip, set periodicBC to True
    if 'periodic_strip' in lp:
        if lp['periodic_strip']:
            lp['periodicBC'] = True
            periodicBC = True
            periodic_strip = True
        else:
            periodic_strip = False
    else:
        periodic_strip = False

    print '\n delta = ', delta, '\n'
    theta = 0.5 * (np.pi - delta)
    print '\n theta = ', theta, '\n'

    # make equilateral triangle
    equil = np.array([[-0.5, -np.sqrt(3.) / 6.], [0., np.sqrt(3.) / 3.], [0.5, -np.sqrt(3.) / 6.]])
    # rotate the equilateral triangle
    equil = mf.rotate_vectors_2D(equil, alpha)
    # make second equilateral triangle above, placed so that connecting bond has unit length
    equil2 = np.array([[0., -np.sqrt(3.) / 3.], [-0.5, np.sqrt(3.) / 6.], [0.5, np.sqrt(3.) / 6.]])
    equil2 = mf.rotate_vectors_2D(equil2, alpha)

    # if check:
    # for row in equil2:
    #     plt.plot([0, row[0]], [0, row[1]], 'r.-')
    # plt.axis('scaled')
    # plt.show()

    # elevate enough that connecting bond has unit length
    # want bond connecting equil[1] to equil2[0] to be tiled, have unit length
    # Solving for the distance between the two points,
    #  (-sa / sqrt(3), ca / sqrt(3)) and (sa / sqrt(3), -ca/sqrt(3) + d), we find
    # d = 2/sqrt(3) ca +/- sqrt(4/3 c^2a - 1/3), where ca = cos(alpha) and sa = sin(alpha).
    dd = 2. / np.sqrt(3.) * np.cos(alpha) + np.sqrt(4./3. * np.cos(alpha)**2 - 1./3.)
    equil2 += np.array([0., dd])
    cc = np.vstack((equil, equil2))
    CU = np.array([0, 1, 2, 3, 4, 5])

    # if check:
    #     plt.plot(cc[:, 0], cc[:, 1], 'b.-')
    #     plt.axis('scaled')
    #     print 'dist between tris = ', np.sqrt((equil[1, 1] - equil2[0, 1])**2 + (equil[1, 0] - equil2[0, 0])**2)
    #     plt.show()

    # Lattice vectors are determined by enforcing connections between triangles to be unit length.
    # This means that the distance between the centroids of equilateral triangles is dd
    nv = dd * np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    # Check lattice vectors
    # plt.plot(np.hstack((nv[:,0].ravel(),0.)), np.hstack((nv[:,1].ravel(),0.)),'bo')
    # plt.show()
    # plt.clf()

    # theta is angle between nv[0] and the x axis.
    # phi determines shear
    #
    #     <         >             4 _____ 5
    # nv[1] \     /    nv[0]        \   /
    #         \ /                    \ /
    #          |                      | 3
    #          |                      | 1
    #          |  nv[2]              / \
    #          v                    /___\
    #                              0     2
    #
    # A way to rotate the latticevecs below:
    latticevecs = dd * np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
    LV = latticevecs
    # generate_lattice([NH,NV],lattice_vectors)

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    tmp1 = np.ones_like(CU)
    tmp0 = np.zeros_like(CU)
    inds = np.arange(len(cc))
    if shape == 'square' or shape == 'circle':
        for i in np.arange(NV):
            for j in np.arange(NH):
                # print np.mod(i,2)-1
                if i == 0:
                    if j == 0:
                        # initialize
                        rr = cc
                        LVUC = np.dstack((tmp0, tmp0, CU))[0]
                    else:
                        # bottom row --> translate by lattice_vectors[1]
                        # Check if it is the last cell in the bottom row
                        if NH - 1 == j:
                            if periodicBC:
                                inds = [0, 1, 2, 3, 4, 5]
                                # Since periodic, only include particle 2 on the rightmost cell of even rows
                                rr = np.vstack((rr, cc[inds, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds],
                                                     i * tmp1[inds], CU[inds]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))
                            else:
                                inds = [0, 1, 2, 3, 4, 5]
                                rr = np.vstack((rr, cc[inds, :] + j * LV[0]))
                                LVUCadd = np.dstack((j * tmp1[inds], tmp0[inds], CU[inds]))[0]
                                # print 'LVUCadd = ', LVUCadd
                                LVUC = np.vstack((LVUC, LVUCadd))

                                # Add additional sites to keep bottom boundary less jagged
                                inds = [3, 4, 5]
                                rr = np.vstack((rr, cc[inds, :] + j * LV[0] - LV[1]))
                                LVUCadd = np.dstack((j * tmp1[inds], -tmp1[inds], CU[inds]))[0]
                                # print 'LVUCadd = ', LVUCadd
                                LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # this is the bottom row, bot not first column
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + j * LV[0]))
                            LVUCadd = np.dstack((j * tmp1[inds], tmp0[inds], CU[inds]))[0]
                            # print 'LVUCadd = ', LVUCadd
                            LVUC = np.vstack((LVUC, LVUCadd))
                            if not periodicBC:
                                # Add additional sites to keep bottom boundary less jagged
                                inds = [3, 4, 5]
                                rr = np.vstack((rr, cc[inds, :] + j * LV[0] - LV[1]))
                                LVUCadd = np.dstack((j * tmp1[inds], -tmp1[inds], CU[inds]))[0]
                                # print 'LVUCadd = ', LVUCadd
                                LVUC = np.vstack((LVUC, LVUCadd))

                                if check:
                                    # inspect the current progress
                                    plt.plot(rr[:, 0], rr[:, 1], 'b.')
                                    kktmp = 0
                                    for pt in rr:
                                        plt.text(pt[0] + 0.2, pt[1], str(kktmp))
                                        kktmp += 1
                                    plt.plot([0, LV[0, 0]], [0, LV[0, 1]], 'r-')
                                    plt.plot([0, LV[1, 0]], [0, LV[1, 1]], 'g-')
                                    plt.axis('scaled')
                                    plt.pause(0.3)
                                    plt.clf()
                else:
                    if j == 0:
                        # first cell of row, include all but 0,5,4 if odd, all but 5,4 if even
                        # note the term (np.mod(i,2)-1)*LV[0], which makes sure the
                        # left wall of the mesh does not drift to the right (to the
                        # direction of LV[0]):
                        #
                        #    <-----
                        #         ^
                        #        /
                        #       /
                        #      ^
                        #     /
                        #    /
                        #
                        inds = [0, 1, 2, 3, 4, 5]
                        rr = np.vstack((rr, cc[inds, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                        LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds], i * tmp1[inds], CU[inds]))[0]
                        LVUC = np.vstack((LVUC, LVUCadd))

                        # If this is the top row and the number of rows is odd, so add an additional piece to
                        # smooth the top boundary from being jagged
                        if i == NV - 1 and not periodicBC and j < NH - 1:
                            # Add additional sites to keep top boundary less jagged
                            inds = [0, 1, 2]
                            rr = np.vstack((rr, cc[inds, :] + j * LV[0] + (-np.floor(i * 0.5)) * LV[0] +
                                            (i + 1) * LV[1]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds],
                                                 i * tmp1[inds], CU[inds]))[0]
                            # print 'LVUCadd = ', LVUCadd
                            LVUC = np.vstack((LVUC, LVUCadd))

                            if check:
                                # inspect the current progress
                                plt.plot(rr[:, 0], rr[:, 1], 'b.')
                                kktmp = 0
                                for pt in rr:
                                    plt.text(pt[0] + 0.2, pt[1], str(kktmp))
                                    kktmp += 1
                                plt.plot([0, LV[0, 0]], [0, LV[0, 1]], 'r-')
                                plt.plot([0, LV[1, 0]], [0, LV[1, 1]], 'g-')
                                plt.axis('scaled')
                                plt.pause(0.4)
                    # elif j == NH - 2 and i == NV - 1 and NV % 2 == 0:
                    #     # this the top right corner of an even row, even column sample
                    #     # Because even number of rows, this cannot be a periodic sample in y, and not a periodic strip
                    #     # since we are not using generate_spindle_strip()
                    #     inds = [0, 1, 2, 3, 4, 5]
                    #     rr = np.vstack((rr, cc[inds, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                    #     LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds],
                    #                          i * tmp1[inds], CU[inds]))[0]
                    #     LVUC = np.vstack((LVUC, LVUCadd))

                    elif NH - j > np.mod(i, 2):
                        # The above clause ensures that the edges don't bulge out for even rows, I think?
                        # The cell is not the first cell of the row
                        # Check if it is the last cell of the row
                        if NH - 1 == j:
                            # An even row and not the last cell.
                            # If periodic, truncate particles 3 and 4 from the UC on the last (rightmost) cell
                            if periodicBC:
                                inds = [0, 1, 2, 3, 4, 5]
                                rr = np.vstack((rr, cc[inds, :] + i * LV[1] +
                                                (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds], i * tmp1[inds],
                                                     CU[inds]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))
                            else:
                                inds = [0, 1, 2, 3, 4, 5]
                                rr = np.vstack((rr, cc[inds, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds],
                                                     i * tmp1[inds], CU[inds]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))

                        else:
                            inds = [0, 1, 2, 3, 4, 5]
                            rr = np.vstack((rr, cc[inds, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds], i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            if i == NV - 1 and not periodicBC and j < NH - 1:
                                if not (j == NH -1 and NH % 2 == 0):
                                    # this is the top row and the number of rows is odd, so add an additional piece to
                                    # smooth the top boundary from being jagged
                                    # Add additional sites to keep top boundary less jagged
                                    # inds = [0, 1, 2]
                                    inds = [0, 1, 2]
                                    rr = np.vstack((rr, cc[inds, :] + (i + 1) * LV[1] +
                                                    (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                    LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[inds],
                                                         (i + 1) * tmp1[inds], CU[inds]))[0]
                                    # print 'LVUCadd = ', LVUCadd
                                    LVUC = np.vstack((LVUC, LVUCadd))

                                if check:
                                    # inspect the current progress
                                    plt.plot(rr[:, 0], rr[:, 1], 'b.')
                                    kktmp = 0
                                    for pt in rr:
                                        plt.text(pt[0] + 0.2, pt[1], str(kktmp))
                                        kktmp += 1
                                    plt.plot([0, LV[0, 0]], [0, LV[0, 1]], 'r-')
                                    plt.plot([0, LV[1, 0]], [0, LV[1, 1]], 'g-')
                                    plt.axis('scaled')
                                    plt.pause(1)

    elif shape == 'hexagon':
        # scheme is to order stagger rows like this
        #   | | |
        #  | | | |
        # | | | | |
        #  | | | |
        #   | | |
        #
        # If NH != NV, then it could be like (NH = 2, NV = 3)
        #   | |
        #  | | |
        # | | | |
        #  | | |
        #   | |
        #
        for i in np.arange(2 * NV - 1):
            # print 'row = ', i
            # add one cell every row, as ascending from bottom of hexagon, until midpoint, then decrease
            ncols2add = -np.abs(i - (NV - 1)) + (NV - 1)
            # print 'ncols2add = ', ncols2add
            # move the first (leftmost) cell in each row left by a lattice vector, until midpoint, then move it right
            cells2move = min(i, NV - 1)  # -np.abs(i-(NV-1)) + (NV-1)
            for j in np.arange(NH + ncols2add):
                # print 'column = ', j
                # print np.mod(i,2)-1
                if i == 0:
                    if j == 0:
                        # initialize
                        rr = cc
                        LVUC = np.dstack((tmp0, tmp0, CU))[0]
                    else:
                        # bottom row --> translate by lattice_vectors[1]
                        rr = np.vstack((rr, cc[2:6, :] + j * LV[0]))
                        LVUCadd = np.dstack((j * tmp1[2:6], tmp0[2:6], CU[2:6]))[0]
                        # print 'LVUCadd = ', LVUCadd
                        LVUC = np.vstack((LVUC, LVUCadd))
                else:
                    if j == 0:
                        # First cell of row, include all but 0,5,4 if odd, all but 5,4 if even
                        # note the term -(cells2move)*LV[0], which makes sure the
                        # left wall of the mesh drifts to the left.
                        #
                        #    <-----
                        #         ^
                        #        /
                        #       /
                        #      ^
                        #     /
                        #    /
                        #
                        if i > NV - 1:
                            rr = np.vstack((rr, cc[inds, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[inds], i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            rr = np.vstack((rr, cc[inds, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[inds], i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                            # Check
                            # plt.plot(R[:, 0], R[:, 1], 'b.')
                            # plt.pause(1)
                    else:
                        if NH + ncols2add - 1 == j:
                            if i > NV - 1:
                                lastIND = 4
                            else:
                                lastIND = 5
                            # last cell in the row --> include 2,3,4,5
                            rr = np.vstack((rr, cc[2:lastIND, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = \
                            np.dstack(((-cells2move + j) * tmp1[2:lastIND], i * tmp1[2:lastIND], CU[2:lastIND]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                            # Check
                            # plt.plot(R[:,0], R[:,1],'b.')
                            # plt.pause(1)
                        else:
                            # only points 2 and 3 included
                            rr = np.vstack((rr, cc[inds, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[inds], i * tmp1[inds], CU[inds]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            # Check
                            # plt.plot(rr[:, 0], rr[:, 1], 'b.')
                            # plt.pause(1)

    elif shape == 'hexagonalT':
        pass

    # Get rid of repeated points
    print(
    'Eliminating repeated points... \n(should not be necessary if creation was done correctly)\n and centering...')

    # Could sort particles by position here, but that would not work for delta = pi lattice!
    # R = le.unique_rows(R)
    # R, order, uinds = le.args_unique_rows_threshold(R,1e-2)
    # LVUCsort = LVUC[order]
    # LVUC = LVUCsort[uinds]
    # xy = R
    xy = rr
    xy -= np.array([np.mean(rr[1:, 0]), np.mean(rr[1:, 1])])

    # Check that we have removed duplicates
    # keep = np.ones_like(xy[:,0])
    # for ii in range(len(xy)):
    #     #check each row to see if it matches with any other
    #     if keep[ii] ==1:
    #         mask = np.ones_like(keep, dtype=bool)
    #         mask[ii] = 0
    #         todo = np.arange(len(xy))[mask]
    #         print 'todo =', todo
    #         for jj in todo:
    #             separation = np.abs(xy[jj,:]-xy[ii,:])
    #             print 'separation = ', separation
    #             if (separation< 1e-6).all():
    #                 keep[jj] = 0
    #
    # print '\nkeep = ', keep

    print 'len(xy) = ', len(xy)

    # if shape == 'circle':
    #     cutout = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0., 2*np.pi, 500)])
    #     # Cut out to shape
    #     bpath = mplpath.Path(cutout)
    #     inside = bpath.contains_points(xy)
    #     xy = xy[inside, :]
    #     LVUC = LVUC[inside, :]

    # If shape is not a string--> use dictionary to cut out polygon
    # hexagon
    #       ____
    #     /      \
    #    /        \
    #    \        /   NV*height tall, diagonals = NV*height
    #     \      /
    #       ----
    # width = LV[0][0]*NH
    # height = LV[1][1]*NV
    # a = width*0.5
    # cutout = np.array([[0,0],[a,0], [a*1.5,a*np.sin(np.pi/3.)],
    #    [a,2*a*np.sin(np.pi/3.)],[0.,2*a*np.sin(np.pi/3.)],[-a*np.cos(np.pi/3.),a*np.sin(np.pi/3.)]])
    # cutout -= np.mean(cutout)

    #        /\
    #     /      \
    #    |        |
    #    |        |   NV*height tall, diagonals = NV*height
    #     \      /
    #        \/
    # width = LV[0][0]*NH
    # height = LV[1][1]*NV
    # a = height*0.5
    # Note that hexagonT gives this shape
    # cutout = np.array([[0,0],[a*np.cos(np.pi/6.),a*np.sin(np.pi/6.)], [a*np.cos(np.pi/6.), a*(1+np.sin(np.pi/6.))],
    #    [0.,a*(1+2*np.sin(np.pi/6.))],[-a*np.cos(np.pi/6.),a*(1+np.sin(np.pi/6.))],[-a*np.cos(np.pi/6.),a*np.sin(np.pi/6.)]])

    if isinstance(shape, dict):
        print '\n\n\nshape is dict: cutting out shape...\n\n\n'
        # Cut out to shape
        bpath = mplpath.Path(shape['polygon'])
        inside = bpath.contains_points(xy)
        xy = xy[inside, :]
        shape = shape['description']

    # Triangulate:
    # First check if any vectors in the unit cell are (anti)parallel.
    # If not, triangulate and proceed.
    # If so, use the connectivity of an undeformed hexagonal lattice.
    print('Triangulating...')
    print 'nv = ', nv
    # this previously said: degenerate = nv[0] == -nv[1], but I don't think that makes sense....
    degenerate = nv[0] == -nv[1]
    # print 'degenerate = ', degenerate
    # plt.plot(nv[:,0], nv[:, 1])
    # plt.show()
    # sys.exit()
    if degenerate.all() or (nv[0:2, 1] < 1e-2).all():
        print 'Unit cell arrangement has degenerate pattern, so basing connectivity on ideal lattice...'
        print 'check = ', check
        # trash1,NL,KL,BL,trash2, trash3, trash4, trash5, trash6, trash7, trash8
        lp_new = copy.deepcopy(lp)
        lp_new['delta_lattice'] = 0.667
        lp_new['delta'] = 2. / 3. * np.pi
        xy_trash, NL_trash, KL_trash, BL, LVUC_trash, LV_trash, UC_trash, PVxydict_trash, PVx_trash, PVy_trash, \
        PV_trash, lattice_exten_trash = generate_spindle_lattice(lp_new)
        # shape,NH, NV, 0.666*np.pi, phi, check=check)

        NL, KL = le.BL2NLandKL(BL, NN=3)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BL = ', BL
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.title('showng all sites for degenerate case')
            plt.show()
            le.display_lattice_2D(xy_trash, BL, close=False)
            for ii in range(len(xy)):
                plt.text(xy_trash[ii, 0], xy_trash[ii, 1], str(ii))
            plt.title('showng all sites for nondegenerate case to use for connectivity')
            plt.show()
            le.display_lattice_2D(xy, BL, close=False)
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
            # sys.exit()
    else:
        Dtri = Delaunay(xy)
        print 'Unit cell arrangement is non-degenerate, trimming connectivity...'
        print 'check = ', check
        btri = Dtri.vertices
        # translate btri --> bond list
        BL = le.Tri2BL(btri)

        # Remove bonds on the sides and through the hexagons.
        # To do this for arbitrary theta and phi, we need to
        # create KL for an undeformed hexagonal lattice.
        print('Removing extraneous bonds from triangulation...')
        # calc vecs from cc bonds
        # form expanded version of cc
        ccexp = np.vstack((cc, cc[2] + LV[1] - LV[0], cc[0] + LV[1]))
        #
        #  sp          sp
        #
        #
        #        sp
        #
        CBL = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 4], [4, 5], [5, 3],
                        [4, 6], [5, 7]])
        BL = blf.latticevec_filter(BL, xy, ccexp, CBL)
        NL, KL = le.BL2NLandKL(BL, NN=3)

        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
            le.display_lattice_2D(xy, BL)

    if shape == 'circle':
        # remove points outside circle --> note division by four for (NH/2)**2
        keep = np.where(xy[:, 0] ** 2 + xy[:, 1] ** 2 < NH ** 2 * 0.25 + 1e-7)[0]
        print 'keep = ', keep
        xy, NL, KL, BL = le.remove_pts(keep, xy, BL=BL, NN='min', check=check)

    # NOTE: xy is non-randomized, non rotated positions.

    ###############################
    # Randomize lattice by eta
    ###############################
    if eta == 0.:
        xypts = xy
        etastr = ''
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]
        # Naming
        etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    ###############################
    # ROTATE BY THETA if theta !=0
    ###############################
    if theta == 0:
        pass
    else:
        # ROTATE BY THETA
        print 'Rotating by theta= ', theta, '...'
        xys = copy.deepcopy(xypts)
        xypts = np.array([[x * np.cos(rot) - y * np.sin(rot), y * np.cos(rot) + x * np.sin(rot)] for x, y in xys])
        # print 'max x = ', max(xypts_tmp[:,0])
        # print 'max y = ', max(xypts_tmp[:,1])

    if rot != 0.:
        rotstr = '_theta' + '{0:.3f}'.format(rot / np.pi).replace('.', 'p') + 'pi'
    else:
        rotstr = ''

    if periodic_strip:
        periodicstr = '_periodicstrip'
    elif periodicBC:
        periodicstr = '_periodicBC'
    else:
        periodicstr = ''

    lattice_exten = 'spindle_' + shape + periodicstr + \
                    '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=4) + \
                    etastr + rotstr

    # CHECK LVUC
    LVUC = np.array(LVUC, dtype=int)
    # sizes = np.arange(len(xy))+5
    # # Check
    # colorvals = np.linspace(0.1,1,len(xy))
    # plt.scatter(xy[:,0],xy[:,1], s=sizes+5, c=colorvals, cmap='afmhot')
    # xyLVtmp = np.array([LVUC[ii,0]*LV[0] + LVUC[ii,1]*LV[1] + C[LVUC[ii,2]]  for ii in range(len(xy))])
    # plt.colorbar()
    # plt.figure()
    # plt.scatter( xyLVtmp[:,0]- np.mean(xyLVtmp,axis=0)[0],\
    #             xyLVtmp[:,1] - np.mean(xyLVtmp,axis=0)[0]-0.1, s=sizes, c=colorvals, cmap='afmhot' )
    # plt.colorbar()
    # plt.show()

    ###############################
    # Make periodic BCs
    ###############################
    # Note that if using periodic_strip boundary conditions, this should make periodicBC == True.
    if periodicBC:
        # The ijth element of PVx is the xcomponent of the vector taking NL[i,j] to its image as seen by particle i.
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)
        PVxydict = {}

        if shape == 'hexagon':
            # translate lattice by NH*LV[0] and NV*LV vectors
            # If NH != NV, then it could be like (NH = 2, NV = 3), sketched below
            #   | |
            #  | | |
            # | | | |
            #  | | |
            #   | |
            #                            /\
            #     C2 /\              C2 / _  \
            # C1   /    \  C3    C1   /delta  \  C3
            #     |      |           /    |   /
            # C0  |      |       C0 /     | /  phi
            #      \    /  C4       \    /  C4
            #        \/               \/
            #        C5               C5
            boundary = le.extract_boundary(xy, NL, KL, BL)
            # For each boundary particle, give it extra neighbors
            for ind in boundary:
                raise RuntimeError('Have not coded for spindle hexagonal periodic case')
                if LVUC[ind, 1] < NV:
                    print 'ind = ', ind
                    add = False
                    # Is particle on bottom edge?
                    if LVUC[ind, 2] == 5 and LVUC[ind, 1] == 0:
                        # print 'particle 5'
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        want = [LVUC[ind, 0] - (NV - 1), (NV - 1) * 2, 2]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * (-NV) + NV * 2 * LV[1]
                        add = True
                    elif LVUC[ind, 2] == 0 and LVUC[ind, 1] < NV:
                        # print 'particle 0'
                        # grab new neighbor: NH-1*LV[0], (NV-1) for LV[1], opposing UC
                        # print '0:new neighbor = ', np.where(LVUC == [(NH-1-LVUC[ind,1]), (NV-1+LVUC[ind,1]),  3])
                        want = [(NH - 1 - LVUC[ind, 1]), (NV - 1 + LVUC[ind, 1]), 3]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * NH + NV * LV[1]
                        add = True
                    elif LVUC[ind, 2] == 4 and LVUC[ind, 1] < NV:
                        if LVUC[ind, 1] > 0 or LVUC[ind, 0] == NH - 1:
                            # print 'particle 4'
                            # grab new neighbor: NH-1*LV[0], (NV-1)*2 for LV[1], opposing UC
                            want = [-NV + 1, NV - 1 + LVUC[ind, 1], 1]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * (-NH - NV) + NV * LV[1]
                            add = True

                    if add:
                        newn = np.where((LVUC[:, 0] == want[0]) * (LVUC[:, 1] == want[1]) *
                                        (LVUC[:, 2] == want[2]))[0][0]
                        # get first zero in KL
                        firstzero = np.where(KL[ind, :] == 0)[0][0]
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

                        PVxydict[(ind, newn)] = -PV
                        # PVxydict[(newn, ind)] = PV

                        # le.display_lattice_2D(xy,BL,title='Checking particle IDs',PVxydict=PVxydict,close=False)
                        # for i in range(len(xy)):
                        #    plt.text(xy[i,0]+0.05,xy[i,1],str(i))
                        # plt.pause(0.1)
                        # le.display_lattice_2D(xy, BL, title='Checking particle IDs',NL=NL, KL=KL, PVx=PVx, PVy=PVy,
                        #                       close=False)
                        # plt.pause(0.001)

                        # Reorder BL and PVx and PVy so that
        elif shape == 'square':
            raise RuntimeError('Have not quite finished the square periodic spindle case... review it')
            if NV % 2 == 1:
                raise RuntimeError('NV must be even for periodic bcs')
            # translate lattice by NH*LV[0] and NV*LV vectors
            # If NH != NV, then it could be like (NH = 4, NV = 3), sketched below
            #
            # | | | |
            #  | | | |
            # | | | |
            #
            #                            /\
            #     C2 /\              C2 / _  \
            # C1   /    \  C3    C1   /delta  \  C3
            #     |      |           /    |   /
            # C0  |      |       C0 /     | /  phi
            #      \    /  C4       \    /  C4
            #        \/               \/
            #        C5               C5
            boundary = le.extract_boundary(xy, NL, KL, BL)
            # For each boundary particle, give it extra neighbors
            for ind in boundary:
                if LVUC[ind, 1] < NV:
                    print 'ind = ', ind
                    add = False
                    # Is particle on bottom edge? if so, to be connected with particle UC=2 on top edge
                    if LVUC[ind, 1] == 0 and not periodic_strip and LVUC[ind, 2] == 0:
                        msg = 'This is particle 0 on bottom, to be connected with particle UC=5 on the sample top'
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 5]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                        add = True
                    elif LVUC[ind, 1] == 0 and not periodic_strip and LVUC[ind, 2] == 2:
                        msg = 'This is particle 2 on bottom, to be connected with particle UC=4 on the sample top'
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        if LVUC[ind, 0] > 0 and LVUC[ind, 0] < NH - 1:
                            # Connect to the top row, somewhere in the middle (not the corners)
                            want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 4]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                        elif LVUC[ind, 0] == 0:
                            # This is the bottom left corner -- connect to top right corner
                            want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 4]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                        elif LVUC[ind, 0] == NH - 1:
                            # This is the bottom right corner -- connect to top left corner
                            want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 4]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                        add = True
                    elif LVUC[ind, 2] == 4 and LVUC[ind, 0] == -int(LVUC[ind, 1] * 0.5):
                        msg = 'This is particle 4 on the left boundary'
                        want = []
                        PV = LV[0] * (-NV)
                        add = True
                    if add:
                        if check:
                            print 'LVUC[ind] = ', LVUC[ind]
                            print msg
                            print 'Looking for particle with LVUC=', want
                            plt.plot(xypts[:, 0], xypts[:, 1], 'b.')
                            for dmyi in range(len(xypts)):
                                plt.text(xypts[dmyi, 0], xypts[dmyi, 1] - 0.1, str(LVUC[dmyi]))
                            plt.show()

                        newn = np.where((LVUC[:, 0] == want[0]) * (LVUC[:, 1] == want[1]) *
                                        (LVUC[:, 2] == want[2]))[0][0]

                        # get first zero in KL
                        firstzero = np.where(KL[ind, :] == 0)[0][0]
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

                        PVxydict[(ind, newn)] = -PV
                        # PVxydict[(newn, ind)] = PV

            PV = np.vstack((LV[0] * NH, LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]))
    else:
        # If not periodic, these are all zeros and can be discarded.
        PVx = []
        PVy = []
        PVxydict = {}
        PV = None

    UC = cc
    if check:
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BL = ', BL
        print 'PVx = ', PVx
        print 'PVy = ', PVy
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_spindle_lattice()', show=True)
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_spindle_lattice(), particles numbered', show=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.3, xy[i, 1], str(i))
        plt.show()

    return xypts, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten


def generate_spindle_strip(lp):
    """Generates hexagonal strip that is only one cell wide in at least one of the dimensions.
    Note that this also handles creating the spindle unitcell.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary, with keys:
        shape : string
            Global shape of the mesh, in form 'square', 'hexagon', etc or as a dictionary with keys
            shape['description'] = the string to name the custom polygon, and
            shape['polygon'] = 2d numpy array
            However, since this is a strip, it really only makes sense to use 'square'
        NH : int
            Number of pts along horizontal. If shape='hexagon', this is the width (in cells) of the bottom side (a)
        NV : int
            Number of pts along vertical, or 2x the number of rows of lattice
        delta : float
            Deformation angle for the lattice in degrees (for undeformed hexagonal lattice, this is 0.66666*np.pi)
        phi : float
            Shear angle for the lattice in radians, must be less than pi/2 (for undeformed hexagonal lattice, this is 0.000)
        eta : float
            randomization of the lattice (a scaling of random jitter in units of lattice spacing)
        rot : float
            angle in units of pi to rotate the lattice vectors and unit cell
        periodicBC : bool
            Wether to apply periodic boundaries to the network
        check : bool
            Wehter to plot output at intermediate steps

    Returns
    ----------
    xy : matrix of dimension nx2
        Equilibrium lattice positions
    NL : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    lattice_exten : string
        description of the lattice, complete with parameters for properly saving the lattice files
    LV : 3 x 2 float array
        Lattice vectors for the kagome lattice with input twist angle
    UC : 6 x 2 float array
        (extended) unit cell vectors
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    PVx : NP x NN float array
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        If NL and KL are remade, PVx will not be ordered properly: use dict instead
    PVy : NP x NN float array
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        If NL and KL are remade, PVy will not be ordered properly: use dict instead
    LVUC : NP x 3 array
        Lattice vectors and (extended) unit cell vectors defining relative position of each point, as integer multiples
        of LV[0] and LV[1], and UC[LVUC[i,2]]
        For instance, xy[0,:] = LV[0] * LVUC[0,0] + LV[1] * LVUC[0,1] + UC[LVUC[0,2]]

    """
    alpha = lp['alph'] * np.pi
    shape = lp['shape']
    NH = int(lp['NH'])
    NV = int(lp['NV'])
    delta = lp['delta']
    # todo: handle phi_lattice and eta_lattice case
    phi = lp['phi']
    eta = lp['eta']
    # todo: handle theta_lattice case
    rot = lp['theta']
    rot *= np.pi
    periodicBC = lp['periodicBC']
    check = lp['check']

    # If we have chosen periodic_strip, set periodicBC to True
    if 'periodic_strip' in lp:
        if lp['periodic_strip']:
            lp['periodicBC'] = True
            periodicBC = True
            periodic_strip = True
        else:
            periodic_strip = False
    else:
        periodic_strip = False

    print '\n delta = ', delta, '\n'
    theta = 0.5 * (np.pi - delta)
    print '\n theta = ', theta, '\n'

    # make equilateral triangle
    equil = np.array([[-0.5, -np.sqrt(3.) / 6.], [0., np.sqrt(3.) / 3.], [0.5, -np.sqrt(3.) / 6.]])
    # rotate the equilateral triangle
    equil = mf.rotate_vectors_2D(equil, alpha)
    # make second equilateral triangle above, placed so that connecting bond has unit length
    equil2 = np.array([[0., -np.sqrt(3.) / 3.], [-0.5, np.sqrt(3.) / 6.], [0.5, np.sqrt(3.) / 6.]])
    equil2 = mf.rotate_vectors_2D(equil2, alpha)

    # if check:
    # for row in equil2:
    #     plt.plot([0, row[0]], [0, row[1]], 'r.-')
    # plt.axis('scaled')
    # plt.show()

    # elevate enough that connecting bond has unit length
    # want bond connecting equil[1] to equil2[0] to be tiled, have unit length
    # Solving for the distance between the two points,
    #  (-sa / sqrt(3), ca / sqrt(3)) and (sa / sqrt(3), -ca/sqrt(3) + d), we find
    # d = 2/sqrt(3) ca +/- sqrt(4/3 c^2a - 1/3), where ca = cos(alpha) and sa = sin(alpha).
    dd = 2. / np.sqrt(3.) * np.cos(alpha) + np.sqrt(4. / 3. * np.cos(alpha) ** 2 - 1. / 3.)
    equil2 += np.array([0., dd])
    cc = np.vstack((equil, equil2))
    CU = np.array([0, 1, 2, 3, 4, 5])

    # if check:
    #     plt.plot(cc[:, 0], cc[:, 1], 'b.-')
    #     plt.axis('scaled')
    #     print 'dist between tris = ', np.sqrt((equil[1, 1] - equil2[0, 1])**2 + (equil[1, 0] - equil2[0, 0])**2)
    #     plt.show()

    # Lattice vectors are determined by enforcing connections between triangles to be unit length.
    # This means that the distance between the centroids of equilateral triangles is dd
    nv = dd * np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    # Check lattice vectors
    # plt.plot(np.hstack((nv[:,0].ravel(),0.)), np.hstack((nv[:,1].ravel(),0.)),'bo')
    # plt.show()
    # plt.clf()

    # theta is angle between nv[0] and the x axis.
    # phi determines shear
    #
    #     <         >             4 _____ 5
    # nv[1] \     /    nv[0]        \   /
    #         \ /                    \ /
    #          |                      | 3
    #          |                      | 1
    #          |  nv[2]              / \
    #          v                    /___\
    #                              0     2
    #
    # A way to rotate the latticevecs below:
    latticevecs = dd * np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
    LV = latticevecs

    # theta is angle between nv[0] and the x axis.
    # phi determines shear
    #                                                     /\
    #     <         >             C2 /\              C2 / _  \
    # nv[1] \     /    nv[0]  C1   /    \  C3    C1   /delta  \  C3
    #         \ /                 |      |           /    |   /
    #          |              C0  |      |       C0 /     | /  phi
    #          |                   \    /  C4       \    /  C4
    #          |  nv[2]              \/               \/
    #          v                     C5               C5

    # If the dimensions are 1 x 1, handle that case separately here
    if NH == 1 and NV == 1:
        xy = cc
        xy -= np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
        cc = CU
        BL = np.array([[0, 1], [1, 2], [0, 2], [1, 3], [3, 4], [4, 5], [3, 5]])
        NL = np.array([[1, 2, 0], [0, 2, 3],
                       [0, 1, 0], [1, 4, 5],
                       [3, 5, 0], [3, 4, 0]])
        KL = np.array([[1, 1, 0], [1, 1, 1],
                       [1, 1, 0], [1, 1, 1],
                       [1, 1, 0], [1, 1, 0]])
        # Note that BL, NL, and KL will be overwritten if periodic
        LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]])
    else:
        # translate by Bravais latt vecs
        print('Translating by Bravais lattice vectors...')
        tmp1 = np.ones_like(CU)
        tmp0 = np.zeros_like(CU)
        inds = np.arange(len(cc))
        if shape == 'square':
            if NH == NV:
                # We start with just one single unit cell here
                rr = cc
            elif NV == 1:
                # build a strip connected at each end
                for jj in np.arange(NH):
                    if jj == 0:
                        # initialize
                        rr = cc
                        LVUC = np.dstack((int(tmp0), int(tmp0), int(CU)))[0]
                    else:
                        # bottom row --> translate by lattice_vectors[1]
                        # Check if it is the last cell in the bottom row
                        if jj == NH - 1:
                            # This is the last cell in this row, and this is the only row
                            if periodicBC:
                                inds = [2, 5]
                                # Since periodic add the
                                rr = np.vstack((rr, cc[inds, :] + jj * LV[0]))
                                LVUCadd = np.dstack((-jj * tmp1[inds], 0 * tmp1[inds], CU[inds]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))
                            else:
                                # Since this is not periodic, add the complete hexagon rather than
                                # truncating for periodic attachment
                                rr = np.vstack((rr, cc[2:6, :] + jj * LV[0]))
                                LVUCadd = np.dstack((jj * tmp1[2:6], tmp0[2:6], CU[2:6]))[0]
                                # print 'LVUCadd = ', LVUCadd
                                LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            # this is not the last row, so add four more particles for every NV
                            rr = np.vstack((rr, cc[2:6, :] + jj * LV[0]))
                            LVUCadd = np.dstack((jj * tmp1[2:6], tmp0[2:6], CU[2:6]))[0]
                            # print 'LVUCadd = ', LVUCadd
                            LVUC = np.vstack((LVUC, LVUCadd))
            elif NH == 1:
                # build a vertical strip connected on left and right
                # First make a compound supercell
                cc = np.vstack((cc, cc + LV[1]))
                CU = np.hstack((CU, CU))
                for ii in np.arange(NV):
                    if ii == 0:
                        # initialize the first particles
                        inds = np.arange(12)
                        rr = cc[inds]
                        LVUC = np.dstack((np.hstack((tmp0, tmp0)),
                                          np.hstack((tmp0, tmp1)), CU[inds]))[0]
                        # Add bottom triangle to the strip:
                        inds = np.arange(3, 6)
                        rr = np.vstack((rr, cc[inds, :] + LV[0] - LV[1]))
                        LVUCadd = np.dstack((tmp1[inds], -tmp1[inds], CU[inds]))[0]
                        LVUC = np.vstack((LVUC, LVUCadd))
                    else:
                        # add more particles displaced by LV[1] and -LV[0] from previous
                        # If this is the top strip, truncate to not have dangling triangle
                        t0 = np.arange(6)
                        if ii == NV - 1:
                            inds = np.arange(9)
                            t1 = np.arange(3)
                        else:
                            inds = np.arange(12)
                            t1 = t0
                        rr = np.vstack((rr, cc[inds, :] + ii * (-LV[0] + 2 * LV[1])))
                        LVUCadd = np.dstack((np.hstack((-ii * tmp1[t0], -ii * tmp1[t1])),
                                             np.hstack((2 * ii * tmp1[t0], (2 * ii + 1) * tmp1[t1])),
                                             CU[inds]))[0]
                        LVUC = np.vstack((LVUC, LVUCadd))
            else:
                raise RuntimeError('Computing strip but neither NV and NH are equal to 1.')
        else:
            raise RuntimeError('Shape is not square but one of the dimensions (NH or NV) is only one cell wide.')

        # make sure LVUC is integer
        LVUC = np.array(LVUC, dtype=int)

        # Center the network (strip)
        xy = rr
        xy -= np.array([np.mean(rr[:, 0]), np.mean(rr[:, 1])])

        if check:
            plt.clf()
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.axis('scaled')
            plt.show()

        # Triangulate:
        # First check if any vectors in the unit cell are (anti)parallel.
        # If not, triangulate and proceed.
        # If so, use the connectivity of an undeformed hexagonal lattice.
        print('Triangulating...')
        print 'nv = ', nv
        degenerate = nv[0] == -nv[1]
        if degenerate.all() or (nv[0:2, 1] < 1e-3).all():
            print 'build_hexagonal: Unit cell arrangement has degenerate pattern, so basing connectivity ' \
                  'on ideal lattice...'
            print 'check = ', check
            # trash1,NL,KL,BL,trash2, trash3, trash4, trash5, trash6, trash7, trash8
            lp_new = copy.deepcopy(lp)
            lp_new['periodicBC'] = False
            lp_new['periodic_strip'] = False
            lp_new['delta_lattice'] = 0.667
            lp_new['delta'] = 2. / 3. * np.pi
            xy_trash, NL_trash, KL_trash, BL, LVUC_trash, LV_trash, UC_trash, PVxydict_trash, PVx_trash, PVy_trash, \
                PV_trash, lattice_exten_trash = generate_spindle_strip(lp_new)

            NL, KL = le.BL2NLandKL(BL, NN=3)
            print 'NL = ', NL
            print 'KL = ', KL
            print 'BL = ', BL
            if check and len(PVxydict_trash) < 1:
                plt.plot(xy[:, 0], xy[:, 1], 'b.')
                for ii in range(len(xy)):
                    plt.text(xy[ii, 0], xy[ii, 1], str(ii))
                plt.show()
                le.display_lattice_2D(xy_trash, BL, NL=NL, KL=KL, close=False)
                for ii in range(len(xy)):
                    plt.text(xy_trash[ii, 0], xy_trash[ii, 1], str(ii))
                plt.show()
                le.display_lattice_2D(xy, BL, NL=NL, KL=KL, close=False)
                for ii in range(len(xy)):
                    plt.text(xy[ii, 0], xy[ii, 1], str(ii))
                plt.show()
                # sys.exit()
        else:
            Dtri = Delaunay(xy)
            print 'Unit cell arrangement is non-degenerate, trimming connectivity...'
            print 'check = ', check
            btri = Dtri.vertices
            # translate btri --> bond list
            BL = le.Tri2BL(btri)

            # Remove bonds on the sides and through the hexagons.
            # To do this for arbitrary theta and phi, we need to
            # create KL for an undeformed hexagonal lattice.
            print('Removing extraneous bonds from triangulation...')
            # calc vecs from cc bonds

            # There are at least 9 particles in any network generated from this function when NV or NH > 1
            CBL = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [7, 8]])
            if NV == 1:
                # This implies that after the first nine particles, there is a lower appendage of indices 3-5 bottom
                # right of the first triangle with indices 0-2.
                # We add the bond between 2 and lower-right appendage 4 here (which would be index 8 + 2 = 10,
                # that is, 9-1 + 5-3 = 10.)
                CBL = np.vstack((CBL, np.array([2, 10])))
            else:
                # The lower right appendage's indices come AFTER the first 12 particles, so connect 2 with 11+2 = 13
                CBL = np.vstack((CBL, np.array([2, 13])))
            if NH > 1:
                # Since the strip is thick, we need to include a bond between UC#0 and UC#5, ie downward to the left
                # from 0 in the undeformed spindle.
                if NV == 1:
                    # NV == 1 implies that after the first nine particles, there is a lower appendage
                    # of indices 3-5 bottom right of the first triangle with indices 0-2, then there is the next column
                    CBL = np.vstack((CBL, np.array([11, 12])))
                else:
                    # NV > 1 implies that after the first twelve particles, there is a lower appendage
                    # of indices 3-5 bottom right of the first triangle with indices 0-2, then there is the next column
                    CBL = np.vstack((CBL, np.array([14, 15])))

            BL = blf.latticevec_filter(BL, xy, xy[0:16], CBL)
            NL, KL = le.BL2NLandKL(BL, NN=3)

            if check:
                plt.plot(xy[:, 0], xy[:, 1], 'b.')
                for ii in range(len(xy)):
                    plt.text(xy[ii, 0], xy[ii, 1], str(ii))
                plt.show()
                le.display_lattice_2D(xy, BL)

    # NOTE: xy are at this point non-randomized, non rotated positions.
    ###############################
    # Randomize lattice by eta
    ###############################
    if eta == 0.:
        xypts = xy
        etastr = ''
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]
        # Naming
        etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    if rot != 0.:
        rotstr = '_theta' + '{0:.3f}'.format(rot / np.pi).replace('.', 'p') + 'pi'
    else:
        rotstr = ''

    if periodic_strip:
        periodicstr = '_periodicstrip'
    elif periodicBC:
        periodicstr = '_periodicBC'
    else:
        periodicstr = ''

    lattice_exten = 'spindle_' + shape + periodicstr + \
                    '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                    '_alph' + sf.float2pstr(lp['alph'], ndigits=4) + \
                    etastr + rotstr

    ###############################
    # Make periodic BCs
    ###############################
    # Note that if using periodic_strip boundary conditions, this should make periodicBC == True.
    if periodicBC:
        # The ijth element of PVx is the x component of the vector taking NL[i,j] to its image as seen by particle i.
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)
        PVxydict = {}

        if shape == 'square':
            # translate lattice by NH*LV[0] and NV*LV vectors
            # If NH != NV, then it could be like (NH = 4, NV = 3), sketched below
            #
            # | | | |
            #  | | | |
            # | | | |
            #
            #                            /\
            #     C2 /\              C2 / _  \
            # C1   /    \  C3    C1   /delta  \  C3
            #     |      |           /    |   /
            # C0  |      |       C0 /     | /  phi
            #      \    /  C4       \    /  C4
            #        \/               \/
            #        C5               C5
            # Here EVERY particle is on the boundary since either NH == 1 or NV == 1 or both.
            # For each boundary particle, give it extra neighbors
            if NH == 1 and NV == 1:
                # Connect each particle to the other one three times
                # recall LV[0] takes C0 to C4, LV[1] takes C0 to C2
                NL = np.array([[1, 2, 5], [0, 2, 3],
                               [0, 1, 4], [1, 4, 5],
                               [2, 3, 5], [0, 3, 4]])
                KL = np.array([[1, 1, -1], [1, 1, 1],
                               [1, 1, -1], [1, 1, 1],
                               [-1, 1, 1], [-1, 1, 1]])
                BL = np.array([[0, 1], [1, 2], [0, 2], [0, -5], [-2, -4], [1, 3], [3, 4], [4, 5], [3, 5]])

                # Enter element into PVx and PVy arrays
                PVx[0, 2] = -LV[1][0]
                PVy[0, 2] = -LV[1][1]
                PVx[2, 2] = (LV[0] - LV[1])[0]
                PVy[2, 2] = (LV[0] - LV[1])[1]
                PVx[4, 0] = (-LV[0] + LV[1])[0]
                PVy[4, 0] = (-LV[0] + LV[1])[1]
                PVx[5, 0] = LV[1][0]
                PVy[5, 0] = LV[1][1]

                PVxydict = {(0, 5): -LV[1], (2, 4): (LV[0] - LV[1])}
            else:
                # Consider each particle (which are all on the boundary)
                for ind in range(len(xy)):
                    if NH == 1:
                        print 'ind = ', ind
                        # Is particle on bottom edge? if so, to be connected with particle UC=7 on top edge
                        if LVUC[ind, 2] == 3 and LVUC[ind, 1] == -1 and LVUC[ind, 0] == 1 and not periodic_strip:
                            msg = 'This is particle 3 on bottom (in the lower appendage), ' \
                                  'to be connected with particle UC=7 ' + \
                                  'on the sample top if this is not a periodic strip'
                            # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                            want = [LVUC[ind, 0] - int(NV - 1), 2 * (NV - 1), 7]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * int(-(NV + 1)) + 2 * (NV + 1) * LV[1]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=check)
                        elif LVUC[ind, 2] == 0 and LVUC[ind, 1] % 2 == 0:
                            msg = 'This is particle 0 on bottom left corner (more left than bottom),'
                            msg += ' to be connected with particle UC=5.'
                            msg += ' This particle participates in periodic strip since on the left.'
                            # check if bottom left
                            want = [LVUC[ind, 0] + 1, LVUC[ind, 1] - 1, 5]
                            print 'LVUC[ind] =', LVUC[ind]
                            print 'want = ', want
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=check)
                        elif LVUC[ind, 2] == 4 and LVUC[ind, 1] % 2 < 1e-3:
                            '''This particle participates in periodic_strip bcs since connected to 8'''
                            # Particle is on left side (top left of a cell)
                            msg = 'This is particle 4, left side of the sample, top left of a unit cell, ' \
                                  'connect to 8, which is unitcell[2] + LV[1]'
                            # grab new neighbor: same LV, different UC
                            want = [LVUC[ind, 0], LVUC[ind, 1] + 1, 2]
                            print 'LVUC[ind] =', LVUC[ind]
                            print 'want = ', want
                            # Perioidic virtual displacement vector of ind with respect to particle 2
                            PV = LV[0]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=check)

                    elif NV == 1:
                        print 'ind = ', ind
                        raise RuntimeError('Have not coded this case yet')
                        # Is particle on bottom edge? if so, to be connected with particle UC=2 on top edge
                        if LVUC[ind, 2] == 5 and not periodic_strip:
                            msg = 'This is particle 5 on bottom, to be connected with particle UC=2 on the sample top'
                            # grab new neighbor: same LV[0] and LV[1], opposing UC
                            want = [LVUC[ind, 0], LVUC[ind, 1], 2]
                            # Perioidic virtual displacement vector of ind
                            PV = -LV[0] + 2 * LV[1]
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL, PVx, PVy,
                                                                PVxydict, msg=msg, check=check)
                        elif LVUC[ind, 2] == 0:
                            msg = 'This is particle 0 on bottom left corner (more left than bottom),'
                            msg += ' to be connected with particle UC=5.'
                            msg += ' This particle participates in periodic strip since on the left.'
                            # print 'particle 0'
                            # check if bottom left
                            want = [NH - 1, LVUC[ind, 1], 5]
                            # Perioidic virtual displacement vector of ind wrt particle 5
                            PV = LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL, PVx, PVy,
                                                                PVxydict, msg=msg, check=check)
                        elif LVUC[ind, 2] == 1:
                            '''This particle participates in periodic_strip bcs, to be connected to particle 2'''
                            # Particle is on left side (top left of cell)
                            msg = 'This is particle 1, left side of the sample, top left of a cell.'
                            # grab new neighbor: opposing UC
                            want = [NH - 1, LVUC[ind, 1], 2]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * NH
                            NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                    PVx, PVy, PVxydict, msg=msg, check=check)

            PV = np.vstack((LV[0] * NH, LV[0] * (-int(NV * 0.5)) + NV * LV[1]))
        else:
            raise RuntimeError('Shape is not square but NV or NH is equal to 1.')
    else:
        # If not periodic, these are all zeros and can be discarded.
        PVx = []
        PVy = []
        PVxydict = {}
        PV = None

    UC = cc
    if check:
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BL = ', BL
        print 'PVx = ', PVx
        print 'PVy = ', PVy
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_spindle_lattice()', show=True)
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_spindle_lattice(), particles numbered', show=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.3, xy[i, 1], str(i))
        plt.show()

    return xypts, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten


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
            plt.text(xypts[dmyi, 0], xypts[dmyi, 1] - 0.1, str(LVUC[dmyi]))
        plt.show()

    newn = np.where((LVUC[:, 0] == want[0]) * (LVUC[:, 1] == want[1]) *
                    (LVUC[:, 2] == want[2]))[0][0]

    # get first zero in KL
    # print 'KL = ', KL
    # print 'KL[ind] = ', KL[ind]
    firstzero = np.where(KL[ind, :] == 0)[0][0]
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


