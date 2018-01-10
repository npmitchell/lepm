import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.le_geometry as leg
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
import scipy
from scipy.spatial import Delaunay
import lepm.data_handling as dh
import lepm.plotting.network_visualization as netvis

'''Functions for making hexagonal, deformed hexagonal, '''


def build_hexagonal(lp):
    """Create a hexagonal lattice using lattice parameters lp.
    Note that either 'delta' (a float, opening angle in radians) or 'delta_lattice' (a string, angle in units of pi)
    must be in lp

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten
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
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_honeycomb_strip(lp)
    else:
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_honeycomb_lattice(lp)

    # LL = (NH*2.*le.polygon_apothem(1.0,6), NV*le.polygon_circumradius(1.0,6) )
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


def build_frame1dhex(lp):
    """Generate the boundary of a hexagonal lattice, with no bulk.

    Parameters
    ----------
    lp

    Returns
    -------
    xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten
    """
    xy, NL, KL, min_NN, LVUC, LV, UC, lattice_exten = generate_honeycomb_lattice(lp)
    BL = le.NL2BL(NL, KL)
    boundary = le.extract_boundary(xy, NL, KL, BL)
    le.display_lattice_2D(xy, BL)

    print 'boundary = ', boundary
    plt.scatter(xy[boundary, 0], xy[boundary, 1])
    plt.show()
    xy, NL, KL, BL = le.remove_pts(boundary, xy, BL, NN=3)
    le.display_lattice_2D(xy, BL)
    lattice_exten = lp['LatticeTop'] + lattice_exten[9:-1]
    print 'lattice_exten = ', lattice_exten
    PVx = []
    PVy = []
    PVxydict = {}
    raise RuntimeError('Finish selected the remaining part of LV, UC, and LVUC here')
    LV = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_honeycomb_lattice(lp):
    """Generates hexagonal lattice (points, connectivity, name).

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

    nv = np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    # if abs(rot) < 1e-6:
    #     n = np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    # else:
    #     n_tmp = np.array([[np.cos(theta), np.sin(theta)],
    #         [-np.cos(theta), np.sin(theta)],
    #         [-np.sin(phi), -np.cos(phi)]])
    #     n = np.array([[ n_tmp[0,0]*np.cos(rot)+n_tmp[0,1]*np.sin(rot), n_tmp[0,1]*np.cos(rot)-n_tmp[0,0]*np.sin(rot)],
    #         [ n_tmp[1,0]*np.cos(rot)+n_tmp[1,1]*np.sin(rot), n_tmp[1,1]*np.cos(rot)-n_tmp[1,0]*np.sin(rot) ],
    #         [ n_tmp[2,0]*np.cos(rot)+n_tmp[2,1]*np.sin(rot), n_tmp[2,1]*np.cos(rot)-n_tmp[2,0]*np.sin(rot) ]])

    # Check lattice vectors
    # plt.plot(np.hstack((n[:,0].ravel(),0.)), np.hstack((n[:,1].ravel(),0.)),'bo')
    # plt.show()
    # plt.clf()

    aa = np.array([0, 0])
    cc = np.array([aa, aa - nv[2], aa - nv[2] + nv[0], aa - nv[2] + nv[0] - nv[1], aa - nv[1] + nv[0], aa - nv[1]])
    CU = np.array([0, 1, 2, 3, 4, 5])

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

    # A way to rotate the latticevecs below:
    latticevecs = np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
    # if abs(theta) < 1e-6:
    #     latticevecs = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    # else:
    #     lvt = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    #     latticevecs = np.array([[ lvt[0,0]*np.cos(rot)+lvt[0,1]*np.sin(rot),
    #                               lvt[0,1]*np.cos(rot)-lvt[0,0]*np.sin(rot)],
    #                               [ lvt[1,0]*np.cos(rot)+lvt[1,1]*np.sin(rot),
    #                               lvt[1,1]*np.cos(rot)-lvt[1,0]*np.sin(rot) ]])

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
                                prii = [2, 5]
                                # Since periodic, only include particle 2 on the rightmost cell of even rows
                                rr = np.vstack((rr, cc[prii, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[prii],
                                                     i * tmp1[prii], CU[prii]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))
                            else:
                                rr = np.vstack((rr, cc[2:6, :] + j * LV[0]))
                                LVUCadd = np.dstack((j * tmp1[2:6], tmp0[2:6], CU[2:6]))[0]
                                # print 'LVUCadd = ', LVUCadd
                                LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            rr = np.vstack((rr, cc[2:6, :] + j * LV[0]))
                            LVUCadd = np.dstack((j * tmp1[2:6], tmp0[2:6], CU[2:6]))[0]
                            # print 'LVUCadd = ', LVUCadd
                            LVUC = np.vstack((LVUC, LVUCadd))
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
                        # Is it an even row?
                        if np.mod(i, 2) == 0:
                            rr = np.vstack((rr, cc[0:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[0:4], i * tmp1[0:4], CU[0:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            rr = np.vstack((rr, cc[1:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[1:4], i * tmp1[1:4], CU[1:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                    elif NH - j > np.mod(i, 2):
                        # The cell is not the first cell of the row
                        # Check if it is the last cell of the row
                        if NH - 1 == j:
                            # An even row and not the last cell.
                            # If periodic, truncate particles 3 and 4 from the UC on the last (rightmost) cell
                            if periodicBC:
                                # Since periodic, only include particle 2 on the rightmost cell of even rows
                                rr = np.vstack((rr, cc[2, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[2], i * tmp1[2], CU[2]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))
                            else:
                                # points 2, 3, 4 included
                                rr = np.vstack((rr, cc[2:5, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                                LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[2:5], i * tmp1[2:5], CU[2:5]))[0]
                                LVUC = np.vstack((LVUC, LVUCadd))

                        else:
                            # only points 2 and 3 included
                            rr = np.vstack((rr, cc[2:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[2:4], i * tmp1[2:4], CU[2:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

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
                            rr = np.vstack((rr, cc[1:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[1:4], i * tmp1[1:4], CU[1:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            rr = np.vstack((rr, cc[0:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[0:4], i * tmp1[0:4], CU[0:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                            # Check
                            # plt.plot(R[:,0], R[:,1],'b.')
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
                            rr = np.vstack((rr, cc[2:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[2:4], i * tmp1[2:4], CU[2:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            # Check
                            # plt.plot(R[:,0], R[:,1],'b.')
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
        PV_trash, lattice_exten_trash = generate_honeycomb_lattice(lp_new)
        # shape,NH, NV, 0.666*np.pi, phi, check=check)

        NL, KL = le.BL2NLandKL(BL, NN=3)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BL = ', BL
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
            le.display_lattice_2D(xy_trash, BL, close=False)
            for ii in range(len(xy)):
                plt.text(xy_trash[ii, 0], xy_trash[ii, 1], str(ii))
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

        CBL = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        BL = blf.latticevec_filter(BL, xy, cc, CBL)
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

    lattice_exten = 'hexagonal_' + shape + periodicstr + \
                    '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                    etastr + rotstr

    print 'delta/np.pi = ', delta / np.pi
    print 'phi/np.pi = ', phi / np.pi
    print 'etastr = ', etastr

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
                    if LVUC[ind, 2] == 5 and not periodic_strip:
                        msg = 'This is particle 5 on bottom, to be connected with particle UC=2 on the sample top'
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 2]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                        add = True
                    elif LVUC[ind, 2] == 0:
                        msg = 'This is particle 0 on bottom left corner (more left than bottom),'
                        msg += ' to be connected with particle UC=5 or 3.'
                        msg += ' This particle participates in periodic strip since on the left.'
                        # print 'particle 0'
                        # check if bottom left
                        if LVUC[ind, 1] == 0:
                            # bottom row has special case
                            want = [int(NH - 1 - int(LVUC[ind, 1] * 0.5)), int(LVUC[ind, 1]), 5]
                        else:
                            want = [int(NH - 1 - int(LVUC[ind, 1] * 0.5)), int(LVUC[ind, 1] - 1), 3]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * NH
                        add = True
                    elif LVUC[ind, 2] == 1:
                        '''This particle participates in periodic_strip bcs'''
                        # select only those particle #1s which are on even #'ed rows (protruding out left)
                        if np.mod(LVUC[ind, 1], 2) == 0:
                            # Particle is on left side (top left of cell)
                            msg = 'This is particle 1, left side of the sample, top left of a cell.'
                            # grab new neighbor: opposing UC
                            want = [NH - 1 - int(LVUC[ind, 1] * 0.5), LVUC[ind, 1], 2]
                            # Perioidic virtual displacement vector of ind
                            PV = LV[0] * NH
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
                             title='Output from generate_honeycomb_lattice()', show=True)
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_honeycomb_lattice(), particles numbered', show=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.3, xy[i, 1], str(i))
        plt.show()

    return xypts, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten


def generate_honeycomb_strip(lp):
    """Generates hexagonal strip that is only one cell wide in at least one of the dimensions.
    Note that this also handles creating the honeycomb unitcell.

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

    nv = np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    aa = np.array([0, 0])

    cc = np.array([aa, aa - nv[2], aa - nv[2] + nv[0], aa - nv[2] + nv[0] - nv[1], aa - nv[1] + nv[0], aa - nv[1]])
    CU = np.array([0, 1, 2, 3, 4, 5])

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

    # A way to rotate the latticevecs below:
    latticevecs = np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
    # if abs(theta) < 1e-6:
    #     latticevecs = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    # else:
    #     lvt = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    #     latticevecs = np.array([[ lvt[0,0]*np.cos(rot)+lvt[0,1]*np.sin(rot),
    #                               lvt[0,1]*np.cos(rot)-lvt[0,0]*np.sin(rot)],
    #                               [ lvt[1,0]*np.cos(rot)+lvt[1,1]*np.sin(rot),
    #                               lvt[1,1]*np.cos(rot)-lvt[1,0]*np.sin(rot) ]])

    # LV[0] takes C0 to C4, LV[1] takes C0 to C2
    LV = latticevecs

    # If the dimensions are 1 x 1, handle that case separately here
    if NH == 1 and NV == 1:
        xy = cc[0:2]
        xy -= np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
        cc = CU[0:2]
        BL = np.array([0, 1])
        NL = np.array([[1, 0, 0], [0, 0, 0]])
        KL = np.array([[1, 0, 0], [1, 0, 0]])
    else:
        # translate by Bravais latt vecs
        print('Translating by Bravais lattice vectors...')
        tmp1 = np.ones_like(CU)
        tmp0 = np.zeros_like(CU)
        inds = np.arange(len(cc))
        if shape == 'square':
            if NH == NV:
                # We make just one single unit cell here
                rr = cc[0:2]
            elif NV == 1:
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
                                prii = [2, 5]
                                # Since periodic add the
                                rr = np.vstack((rr, cc[prii, :] + jj * LV[0]))
                                LVUCadd = np.dstack((-jj * tmp1[prii], 0 * tmp1[prii], CU[prii]))[0]
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
                for ii in np.arange(NV):
                    if ii == 0:
                        # initialize the first particles
                        inds = [5, 0, 1, 2]
                        rr = cc[inds]
                        LVUC = np.dstack((tmp0[inds], tmp0[inds], CU[inds]))[0]
                    else:
                        # add three more particles
                        prii = [5, 0, 1, 2]
                        rr = np.vstack((rr, cc[prii, :] + ii * (-LV[0] + 2 * LV[1])))
                        LVUCadd = np.dstack((ii * tmp1[prii], ii * tmp1[prii], CU[prii]))[0]
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

        plt.clf()
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
        plt.axis('scaled')
        plt.savefig('/Users/npmitchell/Desktop/test.png')

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
                PV_trash, lattice_exten_trash = generate_honeycomb_strip(lp_new)

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

            CBL = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
            BL = blf.latticevec_filter(BL, xy, cc, CBL)
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

    lattice_exten = 'hexagonal_' + shape + periodicstr + \
                    '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                    etastr + rotstr

    print 'delta/np.pi = ', delta / np.pi
    print 'phi/np.pi = ', phi / np.pi
    print 'etastr = ', etastr

    # CHECK LVUC
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
                NL[0, 0:3] = 1
                KL[0, 1:3] = -1
                NL[1, 0:3] = 0
                KL[1, 1:3] = -1
                BL = np.array([[0, 1], [0, -1], [0, -1]])
                # Enter element into PVx and PVy arrays
                PVx[0, 0] = 0.
                PVy[0, 0] = 0.
                PVx[0, 1] = (LV[0] - LV[1])[0]
                PVy[0, 1] = (LV[0] - LV[1])[1]
                PVx[0, 2] = -LV[1][0]
                PVy[0, 2] = -LV[1][1]
                PVx[1, 0] = 0.
                PVy[1, 0] = 0.
                PVx[1, 1] = LV[1][0]
                PVy[1, 1] = LV[1][1]
                PVx[1, 2] = (-LV[0] + LV[1])[0]
                PVy[1, 2] = (-LV[0] + LV[1])[1]
                PVxydict = {(0, 1): np.vstack((LV[0] - LV[1], -LV[1]))}
                LVUC = np.array([[0, 0, 0], [0, 0, 1]])
            else:
                # Consider each particle (which are all on the boundary)
                for ind in range(len(xy)):
                    if NH == 1:
                        if LVUC[ind, 1] < NV:
                            print 'ind = ', ind
                            # Is particle on bottom edge? if so, to be connected with particle UC=2 on top edge
                            if LVUC[ind, 2] == 5 and LVUC[ind, 1] == 0 and not periodic_strip:
                                msg = 'This is particle 5 on bottom, to be connected with particle UC=2 ' + \
                                      'on the sample top if this is not a periodic strip'
                                # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                                want = [LVUC[ind, 0] - int((NV - 1) * 0.5), NV - 1, 2]
                                # Perioidic virtual displacement vector of ind
                                PV = LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=check)
                            elif LVUC[ind, 2] == 0:
                                msg = 'This is particle 0 on bottom left corner (more left than bottom),'
                                msg += ' to be connected with particle UC=5 (we do not worry about particle' \
                                       '3 because there is no particle 3 for NH==1).'
                                msg += ' This particle participates in periodic strip since on the left.'
                                # check if bottom left
                                want = [LVUC[ind, 0], LVUC[ind, 1], 5]
                                # Perioidic virtual displacement vector of ind
                                PV = LV[0]
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=check)
                            elif LVUC[ind, 2] == 1:
                                '''This particle participates in periodic_strip bcs since connected to 2'''
                                # Particle is on left side (top left of a cell)
                                msg = 'This is particle 1, left side of the sample, top left of a cell, connect to 2'
                                # grab new neighbor: same LV, different UC
                                want = [LVUC[ind, 0], LVUC[ind, 1], 2]
                                # Perioidic virtual displacement vector of ind with respect to particle 2
                                PV = LV[0]
                                NL, KL, BL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, BL,
                                                                        PVx, PVy, PVxydict, msg=msg, check=check)

                            # This would double count the 5-0 pair!
                            # Also add periodic bond to particle 0
                            # if LVUC[ind, 2] == 5:
                            #     # Connect to particle 0 that is already on the upper left of this particle,
                            #     # but connect to its image LV[0] to the right. The particle 0 has same LV
                            #     # but different UC
                            #     want = [LVUC[ind, 0], LVUC[ind, 1], 0]
                            #     # Periodic virtual displacement vector of ind wrt particle to grab
                            #     PV = -LV[0]
                            #     NL, KL, PVx, PVy, PVxydict = add_PV(ind, want, xypts, LVUC, PV, NL, KL, PVx, PVy,
                            #                                         PVxydict, msg=msg, check=check)
                            # elif LVUC[ind, 2] == 2:
                            #     # This would double count the 1-2 pair!
                            #     '''This particle participates in periodic_strip bcs since connected to 1'''
                            #     # Particle is on left side (top left of cell)
                            #     msg = 'This is particle 1, left side of the sample, top left of a cell.'
                            #     # grab new neighbor: same LV, different UC
                            #     want = [LVUC[ind, 0], LVUC[ind, 1], 2]
                            #     # Perioidic virtual displacement vector of ind
                            #     PV = -LV[0]
                            #     add = True
                    elif NV == 1:
                        print 'ind = ', ind
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

            PV = np.vstack((LV[0] * NH, LV[0] * int(-(NV + 1) * 0.5) + (NV + 1) * LV[1]))
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
                             title='Output from generate_honeycomb_lattice()', show=True)
        netvis.movie_plot_2D(xy, BL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict, colormap='BlueBlackRed',
                             title='Output from generate_honeycomb_lattice(), particles numbered', show=False)
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
    print 'KL = ', KL
    print 'KL[ind] = ', KL[ind]
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


def generate_flattened_honeycomb_lattice(lp):
    """Generates hexagonal lattice with two different bond lengths on the sides and top of each hexagon.
    The ratio between the long and the short one is given by aratio, and the long one is length unity.
    (returns points, connectivity, name). This is the first step in making the Cairo lattice.

    Parameters
    ----------
    lp : dict
        lattice parameters with key, value pairs:
        shape : string or dict with keys 'description':string and 'polygon':polygon array
            Global shape of the mesh, in form 'square', 'hexagon', etc or as a dictionary with keys
            shape['description'] = the string to name the custom polygon, and
            shape['polygon'] = 2d numpy array
        NH : int
            Number of pts along horizontal. If shape='hexagon', this is the width (in cells) of the bottom side (a)
        NV : int
            Number of pts along vertical, or 2x the number of rows of lattice
        aratio : float
            ratio between long and short lines, with long one of length unity.
        delta : float
            Deformation angle for the lattice in degrees (for undeformed hexagonal lattice, this is 0.66666*np.pi)
        phi : float
            Shear angle for the lattice in radians, must be less than pi/2 (for undeformed hexagonal lattice, this is 0.000)
        eta : float
            randomization of the lattice (a scaling of random jitter in units of lattice spacing)
        rot : float
            angle in units of pi to rotate the lattice vectors and unit cell
        periodicBC : bool
            Whether to apply periodic boundaries to the network
        check : bool
            Whether to plot output at intermediate steps

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
    aratio = np.sqrt(5.0)
    lp['aratio'] = aratio
    lp['delta'] = np.pi * 2. / 3.
    delta = lp['delta']
    lp['phi'] = 0.000
    phi = lp['phi']
    eta = lp['eta']
    rot = lp['theta']
    periodicBC = lp['periodicBC']
    check = lp['check']
    rot *= np.pi

    print '\n delta = ', delta, '\n'
    theta = 0.5 * (np.pi - delta)
    # theta = np.arcsin(np.sin(theta_prime) / 3.)
    print '\n theta - pi/6= ', theta - np.pi / 6., '\n'

    # n are the internal unit cell vectors (see comments below)
    # n = np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)],
    #               [-np.sin(phi)/aratio, -np.cos(phi)/aratio]])
    n = np.array([[4., 2.], [-4., 2.], [0., -2.]])

    # Check lattice vectors
    if check:
        plt.plot(np.hstack((n[:, 0].ravel(), 0.)), np.hstack((n[:, 1].ravel(), 0.)), 'bo')
        plt.show()
        plt.clf()

    aa = np.array([0, 0])

    cc = np.array([aa, aa - n[2], aa - n[2] + n[0], aa - n[2] + n[0] - n[1], aa - n[1] + n[0], aa - n[1]])
    CU = np.array([0, 1, 2, 3, 4, 5])

    if check:
        plt.plot(cc[:, 0], cc[:, 1], 'b.-')
        print 'cc = ', cc
        plt.show()
    # theta is angle between n[0] and the x axis.
    # phi determines shear
    #                                                     /\
    #     <         >             C2 /\              C2 / _  \
    # n[1]  \     /    n[0]   C1   /    \  C3    C1   /delta  \  C3
    #         \ /                 |      |           /    |   /
    #          |              C0  |      |       C0 /     | /  phi
    #          |                   \    /  C4       \    /  C4
    #          |  n[2]               \/               \/
    #          v                     C5               C5

    # A way to rotate the latticevecs below:
    # latticevecs = np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi) / aratio,
    #                         np.sin(theta) + np.cos(phi) / aratio]])
    latticevecs = np.array([[8., 0.], [4, 4.]])
    # if abs(theta) < 1e-6:
    #     latticevecs = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    # else:
    #     lvt = np.array([[2*np.cos(theta),0],[np.cos(theta)+np.sin(phi),np.sin(theta)+np.cos(phi)]])
    #     latticevecs = np.array([[lvt[0,0]*np.cos(rot)+lvt[0,1]*np.sin(rot),lvt[0,1]*np.cos(rot)-lvt[0,0]*np.sin(rot)],
    #         [ lvt[1,0]*np.cos(rot)+lvt[1,1]*np.sin(rot), lvt[1,1]*np.cos(rot)-lvt[1,0]*np.sin(rot) ]])

    LV = latticevecs
    # generate_lattice([NH,NV],lattice_vectors)

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    tmp1 = np.ones_like(CU)
    tmp0 = np.zeros_like(CU)
    inds = np.arange(len(cc))
    if shape == 'square':
        for i in np.arange(NV):
            for j in np.arange(NH):
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
                        if np.mod(i, 2) == 0:
                            rr = np.vstack((rr, cc[0:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[0:4], i * tmp1[0:4], CU[0:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                        else:
                            rr = np.vstack((rr, cc[1:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[1:4], i * tmp1[1:4], CU[1:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                    elif NH - j > np.mod(i, 2):
                        #
                        if NH - 1 == j:
                            # points 2, 3, 4 included
                            rr = np.vstack((rr, cc[2:5, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[2:5], i * tmp1[2:5], CU[2:5]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                        else:
                            # only points 2 and 3 included
                            rr = np.vstack((rr, cc[2:4, :] + i * LV[1] + (-np.floor(i * 0.5)) * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack((((-np.floor(i * 0.5)) + j) * tmp1[2:4], i * tmp1[2:4], CU[2:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

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
                        #  print 'LVUCadd = ', LVUCadd
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
                            rr = np.vstack((rr, cc[1:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[1:4], i * tmp1[1:4], CU[1:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                        else:
                            rr = np.vstack((rr, cc[0:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[0:4], i * tmp1[0:4], CU[0:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))

                            # Check
                            # plt.plot(R[:,0], R[:,1],'b.')
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
                            rr = np.vstack((rr, cc[2:4, :] + i * LV[1] - cells2move * LV[0] + j * LV[0]))
                            LVUCadd = np.dstack(((-cells2move + j) * tmp1[2:4], i * tmp1[2:4], CU[2:4]))[0]
                            LVUC = np.vstack((LVUC, LVUCadd))
                            # Check
                            # plt.plot(R[:,0], R[:,1],'b.')
                            # plt.pause(1)

    elif shape == 'hexagonalT':
        pass

    # Get rid of repeated points
    print('Eliminating repeated points... \n(should not be necessary if creation was done correctly)\n centering...')

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

    if shape == 'circle':
        cutout = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0., 2 * np.pi, 500)])
        # Cut out to shape
        bpath = mplpath.Path(cutout)
        inside = bpath.contains_points(xy)
        xy = xy[inside, :]
        LVUC = LVUC[inside, :]
        shape = shape['description']

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
    #     [a,2*a*np.sin(np.pi/3.)],[0.,2*a*np.sin(np.pi/3.)],[-a*np.cos(np.pi/3.),a*np.sin(np.pi/3.)]])
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
    degenerate = n[0] == -n[1]
    if degenerate.all():
        print 'Unit cell arrangement has degenerate pattern, so basing connectivity on ideal lattice...'
        print 'check = ', check
        # trash1,NL,KL,BL,trash2, trash3, trash4, trash5, trash6, trash7, trash8
        xy_trash, NL_trash, KL_trash, BL, LVUC_trash, LV_trash, UC_trash, PVxydict_trash, PVx_trash, PVy_trash, \
        PV_trash, lattice_exten_trash = generate_honeycomb_lattice(lp)
        NL, KL = le.BL2NLandKL(BL, NN=3)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BL = ', BL
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
            le.display_lattice_2D(xy_trash, BL, close=False)
            for ii in range(len(xy)):
                plt.text(xy_trash[ii, 0], xy_trash[ii, 1], str(ii))
            plt.show()
            le.display_lattice_2D(xy, BL, close=False)
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
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
        if check:
            le.display_lattice_2D(xy, BL)

        print('Removing extraneous bonds from triangulation...')
        # calc vecs from cc bonds

        CBL = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        BL = blf.latticevec_filter(BL, xy, cc, CBL)
        NL, KL = le.BL2NLandKL(BL, NN=3)

        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
            le.display_lattice_2D(xy, BL)

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
        etastr = '_eta{0:.3f}'.format(eta)

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

    lattice_exten = 'flattenedhexagonal_' + shape + \
                    '_delta' + '{0:.3f}'.format(delta / np.pi).replace('.', 'p') + \
                    '_phi' + '{0:.3f}'.format(phi / np.pi).replace('.', 'p') + \
                    etastr + rotstr + periodicstr

    print 'delta/np.pi = ', delta / np.pi
    print 'phi/np.pi = ', phi / np.pi
    print 'etastr = ', etastr

    # CHECK LVUC
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
    if periodicBC:
        # The ijth element of PVx is the xcomponent of the vector taking NL[i,j] to its image as seen by particle i.
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)
        PVxydict = {}

        if shape == 'hexagon':
            # translate lattice by NH*LV[0] and NV*LV vectors
            # If NH != NV, then it could be like (NH = 2, NV = 3)
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
                if LVUC[ind, 1] < NV:
                    print 'ind = ', ind
                    add = False
                    # Is particle on bottom edge?
                    if LVUC[ind, 2] == 5 and LVUC[ind, 1] == 0:
                        # particle 5
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        want = [LVUC[ind, 0] - (NV - 1), (NV - 1) * 2, 2]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * (-NV) + NV * 2 * LV[1]
                        add = True
                    elif LVUC[ind, 2] == 0 and LVUC[ind, 1] < NV:
                        # particle 0'
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

                        # le.display_lattice_2D(xy, BL, title='Checking particle IDs', PVxydict=PVxydict, close=False)
                        # for i in range(len(xy)):
                        #   plt.text(xy[i,0]+0.05,xy[i,1],str(i))
                        # plt.pause(0.1)
                        # le.display_lattice_2D(xy,BL,title='Checking particle IDs',
                        #                       NL=NL, KL=KL, PVx=PVx, PVy=PVy,close=False)
                        # plt.pause(0.001)

                        # Reorder BL and PVx and PVy so that
    else:
        # If not periodic, these are all zeros and can be discarded.
        PVx = []
        PVy = []
        PVxydict = {}

    UC = cc
    print 'NL = ', NL
    print 'KL = ', KL
    print 'BL = ', BL
    return xypts, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten


def build_accordionhex(lp):
    """Build hexagonal-like lattice, replacing each bond with a zigzag set of bonds

    Parameters
    ----------
    lp : dict
        The lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    check0 = copy.deepcopy(lp['check'])
    lp['check'] = False
    xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten = build_hexagonal(lp)

    lp['check'] = check0
    if not (BL < 0).any():
        xy, BL, LVUC, UC, xyvertices, lattice_exten_add = \
            blf.accordionize_network(xy, BL, lp)
        NL, KL = le.BL2NLandKL(BL)
    else:
        xy, BL, NL, KL, PVxydict, PVx, PVy, LVUC, UC, xyvertices, lattice_exten_add = \
            blf.accordionize_network_periodic(xy, BL, NL, KL, lp, PVxydict=PVxydict, PVx=PVx, PVy=PVy)

    if lp['eta_alph'] > 0:
        lattice_exten = 'accordionhexeta' + sf.float2pstr(lp['eta_alph']) + lattice_exten_add
    else:
        lattice_exten = 'accordionhex' + lattice_exten_add
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvertices


def build_hexjunction(lp):
    """Create a Y junction of gyros, with accordionized bonds going in and out

    Parameters
    ----------
    lp

    Returns
    -------

    """
    angle = (np.pi - lp['delta']) * 0.5
    phi = lp['phi']
    xy = np.array([[-np.sin(phi), -np.cos(phi)], [0., 0.],
                   [np.cos(angle), np.sin(angle)], [-np.cos(angle), np.sin(angle)]])

    lp['NH'] = 1
    lp['NV'] = 1

    # Handle periodic case separately
    if not lp['periodicBC']:
        BL = np.array([[0, 1], [1, 2], [1, 3]])

        # accordionize the lattice
        nnew = lp['intparam']
        xyvertices = np.array([0., 0.])
        ptind = len(xy)
        blnew = np.zeros((len(BL) * (nnew + 2), 2), dtype=int)
        xynew = np.zeros((len(xy) + len(BL) * nnew, 2), dtype=float)
        xynew[0:len(xy)] = xy

        dists = dh.dist_pts(xy, xy, dim=-1, square_norm=False)
        distx = dh.dist_pts(xy, xy, dim=0, square_norm=False)
        disty = dh.dist_pts(xy, xy, dim=1, square_norm=False)
        # ii is the index of the row of blnew that we are creating
        ii = 0
        dmyi = 0
        sgn = np.sign(1. - lp['alph']) * np.ones(len(BL))
        for row in BL:
            print 'row = ', row
            # Add the points displaced by some amount from the original bond
            # ll is path length from old site to new site
            ll = dists[row[0], row[1]]
            # ss is the path length between intermediate particles projected along the original bond
            ss = ll / float(nnew)
            hh = (ss * 0.5) / abs(np.tan(lp['alph'] * 0.5 * np.pi)) * sgn[dmyi]
            # define vector taking row[0] to row[1]
            vec = np.array([distx[row[0], row[1]], disty[row[0], row[1]]])
            vechat = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # get unit vector normal to vec --> just swap x,y and negate new x component to go clockwise
            normal = np.array([vec[1], -vec[0]]) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # add first point along vec
            xynew[ptind] = xy[row[0]] + (ss * 0.5) * vechat + hh * normal

            # Create the new lineseg from the original point to the first added point
            blnew[ii, :] = np.array([row[0], ptind])
            ii += 1
            # connect the middle linesegs between added pts
            for jj in range(nnew - 1):
                # add bonds
                blnew[ii, :] = np.array([ptind, ptind + 1])
                ptind += 1

                # add points
                xynew[ptind] = xy[row[0]] + vechat * (ss * (jj + 1.5)) + hh * normal * (-1) ** (jj + 1)
                ii += 1

            # connect the last lineseg
            blnew[ii, :] = np.array([ptind, row[1]])
            ii += 1
            ptind += 1

        blnew = blnew[0:ii]
        xynew = xynew[0:ptind]
        if lp['check']:
            plt.plot(xynew[:, 0], xynew[:, 1], 'r.')
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xynew)):
                plt.text(xynew[ii, 0] - 0.2, xynew[ii, 1] + 0.2, str(ii))
            plt.show()
            netvis.movie_plot_2D(xynew, blnew, show=True)
        periodicstr = ''
        PVx = []
        PVy = []
        PVxydict = {}
        LV = 'none'
    else:
        periodicstr = '_periodic'
        xyvertices = np.array([0., 0.])
        raise RuntimeError("Code up periodic case here")

    LVUC = None
    LL = (2. * np.cos(angle), np.cos(phi) + np.sin(angle))
    UC = None
    BL = blnew
    xy = xynew
    NL, KL = le.BL2NLandKL(BL)
    BBox = np.array([[-np.cos(angle) - np.sin(phi), -np.cos(phi)], [-np.cos(angle), np.sin(angle)],
                     [np.cos(angle), np.sin(angle)], [np.cos(angle) - np.sin(phi), -np.cos(phi)]])

    # Rescale BBox, xy, LL, and PVxydict by average bond length
    bL = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
    if (bL != bL[0]).all():
        print 'bL = ', bL
        netvis.movie_plot_2D(xy, BL, show=True)
        print 'xy = ', xy
        raise RuntimeError('All bonds should be of same length at this point (eta randomization follows after)')

    scale = 1. / np.median(bL)
    xy *= scale
    BBox *= scale
    LL = (LL[0] * scale, LL[1] * scale)
    if lp['periodicBC']:
        PVx *= scale
        PVy *= scale
        PVxydict.update((key, val * scale) for key, val in PVxydict.items())

    # randomize positions
    if lp['eta'] > 0:
        xy += lp['eta'] * np.random.rand(np.shape(xy))

    lattice_exten = 'hexjunction' + periodicstr + \
                    '_delta' + sf.float2pstr(lp['delta'] / np.pi, ndigits=3) + \
                    '_phi' + sf.float2pstr(lp['phi'] / np.pi, ndigits=3)

    if lp['eta'] > 0:
        lattice_exten += '_eta{0:.3f}'.format(lp['eta']).replace('.', 'p')

    lattice_exten += '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp, xyvertices


def build_hexmeanfield3gyro(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    lp['LatticeTop'] = 'hexmeanfield'
    lp['NH'] = 1
    lp['NV'] = 3
    shape = 'square'
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
    if lp['phi'] > 0.:
        raise RuntimeError('Have not coded for phi>0')

    alph = 0.5 * (np.pi - lp['delta'])
    beta = np.pi - alph
    gamma = np.pi + alph
    delta = 2. * np.pi - alph
    #  6 o
    #    |
    #    o 3 o 2
    #   / \ /
    #  o   o 0
    # 7    |
    #      o 1
    #     / \
    #    o   o
    #     4   5
    xy = np.array([[0., 0.5],
                   [0., -0.5],
                   [np.cos(alph), 0.5 + np.sin(alph)],
                   [np.cos(beta), 0.5 + np.sin(beta)],
                   [np.cos(gamma), -0.5 + np.sin(gamma)],
                   [np.cos(delta), -0.5 + np.sin(delta)],
                   [np.cos(beta), 1.5 + np.sin(beta)],
                   [np.cos(beta) + np.cos(gamma), 0.5 + np.sin(beta) + np.sin(gamma)]])

    if lp['periodicBC']:
        raise RuntimeError('Have not coded periodic case -- should really use non meanfield network that is 1x1,' +
                           'which is identical to what you are looking for!')
    else:
        NL = np.array([[1, 2, 3],
                       [0, 4, 5],
                       [0, 0, 0],
                       [0, 6, 7],
                       [1, 0, 0],
                       [1, 0, 0],
                       [3, 0, 0],
                       [3, 0, 0]])
        KL = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])

        BL = le.NL2BL(NL, KL)

        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        if lp['shape'] == 'square':
            BBox = blf.auto_polygon(lp['shape'], LL[0], LL[1], eps=0.00)
        else:
            BBox = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)

        # If not periodic, these are all zeros and can be discarded.
        PVx = []
        PVy = []
        PVxydict = {}
        PV = None
        LVUC = None
        LV = None
        UC = np.array([0, 1])
        periodicstr = ''
        phi = lp['phi']
        etastr = ''
        rotstr = ''
        lattice_exten = 'hexagonal_' + shape + periodicstr + \
                        '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                        '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                        etastr + rotstr
        return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten


def build_hexmeanfield(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    shape = 'square'
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
    if lp['phi'] > 0.:
        raise RuntimeError('Have not coded for phi>0')

    alph = 0.5 * (np.pi - lp['delta'])
    beta = np.pi - alph
    gamma = np.pi + alph
    delta = 2. * np.pi - alph

    xy = np.array([[0., 0.5],
                   [0., -0.5],
                   [np.cos(alph), 0.5 + np.sin(alph)],
                   [np.cos(beta), 0.5 + np.sin(beta)],
                   [np.cos(gamma), -0.5 + np.sin(gamma)],
                   [np.cos(delta), -0.5 + np.sin(delta)]])
    if lp['periodicBC']:
        raise RuntimeError(
            'Have not coded periodic case -- should really use non meanfield network that is 1x1,' +
            'which is identical to what you are looking for!')
    else:
        NL = np.array([[1, 2, 3],
                       [0, 4, 5],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])
        KL = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])

        BL = le.NL2BL(NL, KL)

        LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
        if lp['shape'] == 'square':
            BBox = blf.auto_polygon(lp['shape'], LL[0], LL[1], eps=0.00)
        else:
            BBox = blf.auto_polygon(lp['shape'], lp['NH'], lp['NV'], eps=0.00)

        # If not periodic, these are all zeros and can be discarded.
        PVx = []
        PVy = []
        PVxydict = {}
        PV = None
        LVUC = None
        LV = None
        UC = np.array([0, 1])
        periodicstr = ''
        phi = lp['phi']
        etastr = ''
        rotstr = ''
        lattice_exten = 'hexagonal_' + shape + periodicstr + \
                        '_delta' + sf.float2pstr(delta / np.pi, ndigits=3) + \
                        '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
                        etastr + rotstr
        return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten



# Functions from Lisa
# def make_periodic_hex(R, Ni, Nk, side_pairs=[1, 2, 3]):
#     """Construct a periodic hexagonal lattice"""
#     NP = len(R)
#     num_Nei = np.sum(Nk, axis = 1)
#     edge_R = R[num_Nei == 2]
#     edge_index = np.array([i for i in range(NP) if num_Nei[i]==2])
#
#     edge_length = len(edge_R)
#
#     pA = np.array([np.arccos(edge_R[i,0]/np.sqrt(edge_R[i,0]**2+edge_R[i,1]**2)) for i in range(edge_length)])
#
#     for i in range(edge_length):
#         if edge_R[i,1]<0:
#             pA[i]=2 * pi - pA[i]
#
#     si= np.argsort(pA)
#     pA = pA[si]
#     edge_index = edge_index[si]
#
#     val = 0.02
#
#     edge1 = edge_index[np.array([i for i in range(edge_length) if pA[i]<np.pi/3-val])]
#     edge2 = edge_index[np.array([i for i in range(edge_length) if pA[i]>np.pi/3-val and pA[i]<2*np.pi/3 + val])]
#     edge3 = edge_index[np.array([i for i in range(edge_length) if pA[i]>2*np.pi/3 + val  and pA[i]<3*np.pi/3 ])]
#     edge4 = edge_index[np.array([i for i in range(edge_length) if pA[i]>3*np.pi/3 and pA[i]<4*np.pi/3-val ])]
#     edge5 = edge_index[np.array([i for i in range(edge_length) if pA[i]>4*np.pi/3 -val and pA[i]<5*np.pi/3 + val ])]
#     edge6 = edge_index[np.array([i for i in range(edge_length) if pA[i]>5*np.pi/3 + val  and pA[i]<6*np.pi/3 ])]
#
#     #reverse the order for 4,5,6
#     edge4=edge4[::-1]
#     edge5=edge5[::-1]
#     edge6=edge6[::-1]
#
#     if 1 in side_pairs:
#         Ni, Nk = match_gyros(edge1, edge4, Ni, Nk)
#     if 2 in side_pairs :
#         Ni, Nk = match_gyros(edge2, edge5, Ni, Nk)
#     if 3 in side_pairs:
#         Ni, Nk = match_gyros(edge3, edge6, Ni, Nk)
#
#     return Ni, Nk
#
#
# def match_gyros(edge1, edge4, Ni, Nk):
#     """Match the edge particles with particles on opposite side of system"""
#     for i in range(len(edge1)):
#         i1 = edge1[i]
#         i2 = edge4[i]
#
#         ne1 = np.where(Nk[i1]==0)[0]
#         ne2 = np.where(Nk[i2]==0)[0]
#
#         Nk[i1,ne1] = -1
#         Nk[i2,ne2] = -1
#
#         Ni[i1,ne1] = i2
#         Ni[i2,ne2] = i1
#
#     return Ni, Nk
