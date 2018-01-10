import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.le_geometry as leg
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import copy
import scipy
from scipy.spatial import Delaunay
import scipy.optimize as opt
import sys
import glob
import math
import cmath
import shapely.geometry as sg
import descartes


def build_dislocated_lattice(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    pt = lp['Ndefects']
    xy, NL, KL, BL, lattice_exten = generate_dislocated_hexagonal_lattice(lp['shape'], lp['NH'], lp['NV'], pt,
                                                                          Bvecs=lp['Bvec'], check=lp['check'])
    PVx = []
    PVy = []
    PVxydict = {}
    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    BBox = np.array([[np.min(xy[:, 0]), np.min(xy[:, 1])], [np.max(xy[:, 0]), np.min(xy[:, 1])],
                     [np.max(xy[:, 0]), np.max(xy[:, 1])], [np.min(xy[:, 0]), np.max(xy[:, 1])]])
    LV = 'none'
    UC = 'none'
    LVUC = 'none'
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LL, BBox, lattice_exten


def generate_dislocated_hexagonal_lattice(shape, NH, NV, points, Bvecs=[], tol=1e-3, relax_each_step=False,
                                          relax_twice=True, check=False):
    """
    Generate a hexagonal lattice with dislocations at specified locations and with given Burgers vectors

    Parameters
    ----------
    shape : string or dict with keys 'description':string and 'polygon':polygon array
        Global shape of the mesh, in form 'square', 'hexagon', etc or as a dictionary with keys
        shape['description'] = the string to name the custom polygon, and
        shape['polygon'] = 2d numpy array
    NH : int
        Number of pts along horizontal. If shape='hexagon', this is the width (in cells) of the bottom side (a)
    NV : int
        Number of pts along vertical, or 2x the number of rows of lattice
    pt : #dislocations x 2 float array or int
        array of 2D points, locations to place dislocations
    Bvecs : #dislocations x 1 string array or empty list
        orientations of Burgers vectors, as 'W' 'SW' 'SE' 'E' 'NE' 'NW'
        If empty list, orientations are randomized

    Returns
    ----------
    xy : NP x 2 array
        Equilibrium lattice positions of honeycomb-like lattice
    NL : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of honeycomb-like lattice
    lattice_exten : string
        description of the lattice, complete with parameters for properly saving the lattice files
    """
    from lepm.build import build_triangular
    if isinstance(shape, str):
        shape_string = shape
        vals = [NH, NV]
        if shape in ['hexagon', 'square']:
            shape = {shape_string: vals}

    # since shape is dict, make lattice larger than necessary for cropping purposes
    xy, NL, KL, BL, LVUC, LV, UC, trash = \
        build_triangular.generate_triangular_lattice(shape, NH * 3, NV * 3, eta=0., theta=0., check=check)

    if check:
        le.display_lattice_2D(xy, BL, title='triangular latt')

    tri = Delaunay(xy)
    TRI = tri.vertices
    BL = le.TRI2BL(TRI)
    thres = 1.05  # cut off everything longer than normal
    print('thres = ' + str(thres))
    print('Trimming bond list...\n')
    BL = le.cut_bonds(BL, xy, thres)

    # Need to multiply bond length by factor to make centroids 1 unit apart.
    restlen = 1. / np.sqrt((1. / 3. * np.cos(np.pi / 3.) - 2. / 3.) ** 2 + (1. / 3. * np.sin(np.pi / 3.)) ** 2)
    xy *= restlen

    NP0 = len(xy)

    # If pt is an integer, supply random points in the bulk
    print 'points before check = ', points
    if isinstance(points, int):
        print 'Points is integer: ascribe random points in bounding polygon as defect sites...'
        pt = []
        boundary = le.extract_boundary(xy, NL, KL, BL)
        boundarypoly = np.array(boundary.tolist() + [boundary[0]])
        xybp = xy[boundarypoly].tolist()

        polygon = leg.Polygon(xybp)
        for ii in range(points):
            # generate random points
            point = polygon.random_point()
            pt += [point.to_list()]

        points = np.array(pt)
        lattexten_is_Rand = True
    else:
        lattexten_is_Rand = False

    # Prepare Bvecs (burgers vectors)
    if Bvecs == [] or Bvecs in ['W', 'West']:
        BvecList = ['W' for ii in range(len(points))]
        Bvecstr = 'W'
        # Order points in increasing y-values for W Bvecs
        sortIND = np.argsort(points[:, 1])
        points = points[sortIND]
    elif Bvecs in ['random', 'Random', 'rand']:
        BvecList = [0] * len(points)
        rands = np.random.rand(len(points))
        for ii in range(len(points)):
            rii = rands[ii]
            if rii < 1. / 6.:
                BvecList[ii] = 'W'
            elif rii < 1. / 3.:
                BvecList[ii] = 'SW'
            elif rii < 1. / 2.:
                BvecList[ii] = 'SE'
            elif rii < 4. / 6.:
                BvecList[ii] = 'E'
            elif rii < 5. / 6.:
                BvecList[ii] = 'NE'
            else:
                BvecList[ii] = 'NW'

        Bvecstr = 'Random'

    print 'points = ', points
    relax_twice_tmp = False
    tol_tmp = 0.1
    for ii in range(len(points)):
        print 'Adding dislocation #', ii
        # If this isn't the final point, don't bother relaxing twice or precisely
        if ii == len(points) - 1:
            relax_twice_tmp = relax_twice
            tol_tmp = tol
            relax_each_step = True
        pt = points[ii]
        Bvec = BvecList[ii]
        print '\n----------\nAdding dislocation at pt = ', pt
        if Bvec in ['W', 'West']:
            xy, LVUC, BL = insert_dislocation_triangular_westBurgersvector(pt, xy, LVUC, BL, bo=restlen, check=check,
                                                                           relax_lattice=relax_each_step,
                                                                           relax_twice=relax_twice_tmp, tol=tol_tmp)

    xy0 = xy
    # Centroids to get particle positions
    # print 'BL= ', BL
    TRI = le.BL2TRI(BL, xy)
    # print 'TRI = ', TRI
    if check:
        le.display_lattice_2D(xy0, BL, title='checking BL2TRI', close=False)
        for i in range(len(xy0)):
            plt.text(xy0[i, 0] + 0.01, xy0[i, 1] + 0.01, str(i))
        plt.show()

    xy, NL, KL, BL, BM = le.centroid_lattice_from_TRI(xy0, TRI)

    ####################################
    # CHECK
    if check:
        # print 'xy0 = ', xy0
        le.display_lattice_2D(xy, BL, NL=NL, title='Centroid lattice', close=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.01, xy[i, 1] + 0.01, str(i))
        for i in range(len(xy0)):
            plt.text(xy0[i, 0] + 0.01, xy0[i, 1] + 0.01, str(i))
        plt.triplot(xy0[:, 0], xy0[:, 1], TRI, 'go-')
        plt.show()
    ####################################

    # Cannot relax lattice here because there are too many zero modes and guest modes in the system
    # --> the result would be a mess

    exten = '_' + shape_string + '_Ndefects' + str(len(points)) + '_Bvec' + Bvecstr
    if lattexten_is_Rand:
        lattice_exten = 'dislocatedRand' + exten
    else:
        lattice_exten = 'dislocated' + exten

    return xy, NL, KL, BL, lattice_exten


def insert_dislocation_triangular_westBurgersvector(pt, xy, LVUC, BL, bo=1., check=False, relax_lattice=True,
                                                    relax_twice=False, tol=1e-3):
    """Insert a dislocation with 'right'-pointing Burger's vector near pt, in a triangular lattice.

    Parameters
    ----------
    pt : 1 x 2 float list or array
        point near which to insert dislocation
    xy : NP0 x 2 float array
        points of the lattice before dislocation insertion
    LVUC :
    bo : float or string 'centroid'
        rest bond length, if auto, then prepares for centroids to have rest length of 1. --> np.sqrt( (1./3.* np.cos(np.pi/3.)-2./3.)**2 + (1./3.* np.sin(np.pi/3.))**2 )
    tol : float
        tolerance for minimization step

    Returns
    ----------
    """
    NP0 = len(xy)
    # Add two half rows into triangular lattice
    # First find two nearest points (but near in x only)
    x = xy[:, 0];
    y = xy[:, 1]
    tree = scipy.spatial.KDTree(zip(x, y))
    numfind = 6
    print 'pt = ', pt
    found = tree.query(pt, k=numfind)[1]
    print 'found = ', found

    # if not (LVUC[:,1] > LVUC[found,1] ).any():
    #     # this point is at the top, get a different one

    gotequal = False

    # First try to grab leftpoint as the particle nearest pt
    closest = tree.query(pt, k=1)[1]
    inds = np.setdiff1d(found, closest)
    print 'closest= ', closest
    print 'inds = ', inds
    getequal = np.where(np.logical_and(LVUC[inds, 1] == LVUC[closest, 1],
                                       np.abs(LVUC[inds, 0] - LVUC[closest, 0]) == 1))[0]
    # print 'getequal = ', getequal
    if len(getequal) > 0:
        equalIND = getequal[0]
        pair = [closest, inds[equalIND]]
    else:
        print '\n did not find pair for closest point, so finding pair very nearby'
        i = 0
        while not gotequal:
            # look at all the other particles (not particle found[i])
            inds_noti = np.setdiff1d(np.arange(numfind), np.array([i]))
            inds = found[inds_noti]
            # print 'i = ', i
            # print 'inds_noti = ', inds_noti
            # print 'inds = ', inds
            # print 'found[i] = ', found[i]
            #
            getequal = np.where(np.logical_and(LVUC[inds, 1] == LVUC[found[i], 1],
                                               np.abs(LVUC[inds, 0] - LVUC[found[i], 0]) == 1))[0]
            # print 'getequal = ', getequal

            if len(getequal) > 0:
                gotequal = True
                equalIND = getequal[0]
                pair = [found[i], inds[equalIND]]
            else:
                i += 1

    print 'pair = ', pair
    # Add rows by using LVUC
    # First, get LVUC parametrization of the target position
    if LVUC[pair[0], 0] < LVUC[pair[1], 0]:
        leftpt = pair[0]
        rightpt = pair[1]
    else:
        leftpt = pair[1]
        rightpt = pair[0]

    # leftpt  new  rightpt
    #       o  x  o

    # Grab LVUC coords for left inserted half row
    LEFTrow = (np.where(np.logical_and(LVUC[:, 1] > LVUC[leftpt, 1], (LVUC[:, 0] + LVUC[:, 1] ==
                                                                      LVUC[leftpt, 0] + LVUC[leftpt, 1])))[0]).tolist()
    LEFTrow.append(leftpt)
    LEFTrow = np.array(LEFTrow)
    LVUCnewLrow = LVUC[LEFTrow]
    # Adjust all points left of the inserted row.
    # These are points with LVUC[:,1] > LVUC[target,1] and ( LVUC[:,0]+LVUC[:,1] < LVUC[target,0]+LVUC[target,1] )
    LEFT = np.where(np.logical_and(LVUC[:, 1] > LVUC[leftpt, 1] - 1,
                                   (LVUC[:, 0] + LVUC[:, 1] < LVUC[rightpt, 0] + LVUC[rightpt, 1])))[0]
    LVUC[LEFT, 0] -= 1

    # Insert new point between columns in lattice
    # First get indices of points just to the right of LEFTrow
    Lr2 = np.where(np.logical_and(LVUC[:, 1] > LVUC[leftpt, 1] - 1,
                                  LVUC[:, 0] + LVUC[:, 1] == LVUC[leftpt, 0] + LVUC[leftpt, 1] + 2))[0]

    # Sort LEFTrow by increasing LV[1]
    sortIND_LEFTrow = np.argsort(LVUC[LEFTrow, 1])
    LEFTrow = LEFTrow[sortIND_LEFTrow]
    # Sort LVUCnewLrow by increasing LV[1]
    sortIND = np.argsort(LVUCnewLrow[:, 1])
    LVUCnewLrow = LVUCnewLrow[sortIND]
    # Sort Lr2 by increasing LV[1]
    sortIND_Lr2 = np.argsort(LVUC[Lr2, 1])
    Lr2 = Lr2[sortIND_Lr2]

    print 'len(xy) = ', len(xy)
    print 'len(LVUC) = ', len(LVUC)
    print 'LVUCnewLrow = ', LVUCnewLrow

    #################################
    # Check
    if check:
        plt.plot(xy[:, 0], xy[:, 1], 'bo')
        plt.scatter(xy[LEFTrow, 0], xy[LEFTrow, 1], s=300, c='r', marker='^')
        # plt.scatter(xy[RIGHTrow,0], xy[RIGHTrow,1],  s=300, c='g', marker='s')
        for i in range(len(xy[:, 0])):
            plt.text(xy[i, 0], xy[i, 1], str(LVUC[i, 0]) + ', ' + str(LVUC[i, 1]))
        plt.title('Identified left row')
        plt.show()
    #################################
    # Match Lr2 indices with LEFTrow indices, point by point:
    # LEFTrow  new  Lr2
    #       o   x   o
    LrINDs = []
    for ii in np.arange(len(LEFTrow)):
        try:
            tmp = np.where(LVUC[Lr2, 1] == LVUC[LEFTrow[ii], 1])
            LrINDs.append(tmp[0][0])
        except:
            pass
            '''Reached boundary, no match'''
    # print 'LEFTrow = ', LEFTrow
    # print 'Lr2 = ', Lr2
    # print 'Lr2[LrINDs] = ', Lr2[LrINDs]
    # extrapt_inds is a subset of Lr2, which is the column of pts to the right of the inserted one
    # This is automatically sorted
    extrapt_inds = np.setdiff1d(Lr2, Lr2[LrINDs])

    # print 'LrINDs = ', LrINDs
    # Add the left half row
    Lr2add = Lr2[LrINDs]
    xynew = np.mean([xy[LEFTrow, :], xy[Lr2add, :]], axis=0)
    # print 'xynew = ', xynew
    # print 'LVUCnewLrow = ', LVUCnewLrow
    xy = np.vstack((xy, xynew))
    LVUC = np.vstack((LVUC, LVUCnewLrow))
    # print 'xy = ', xy
    # print 'LVUC = ', LVUC
    LaddIND = np.arange(NP0, len(xy))

    # new particle with Lowest LV[1] is special
    lowest_new_pt = NP0

    #################################
    # Check
    if check:
        plt.plot(xy[:, 0], xy[:, 1], 'bo')
        plt.scatter(xy[LEFTrow, 0], xy[LEFTrow, 1], s=500, c='r', marker='^', zorder=0)
        # plt.scatter(xy[RIGHTrow,0], xy[RIGHTrow,1], c='g', marker='s')
        for i in range(len(xy[:, 0])):
            plt.text(xy[i, 0], xy[i, 1], str(LVUC[i, 0]) + ', ' + str(LVUC[i, 1]))
        plt.title('After adding one row, but not yet boundary points of row')
        plt.show()
    #################################
    # print 'len(xy) = ', len(xy)
    # print 'len(LVUC) = ', len(LVUC)
    # The added point with the largest LV[1] value is special
    # (it ends up being second highest in LV1, but call it topnew here.)
    topnew_tmp = np.where(LVUC[LaddIND, 1] == max(LVUC[LaddIND, 1]))[0][0]
    topnew = LaddIND[topnew_tmp]

    # Also need LEFTrow pt with largest LV[1] value
    topold_tmp = np.where(LVUC[LEFTrow, 1] == max(LVUC[LEFTrow, 1]))[0][0]
    topold = LEFTrow[topold_tmp]

    # Add the left half row pts on the boundary using extrapt_inds
    for pt in extrapt_inds:
        if LVUC[pt, 1] > LVUC[leftpt, 1] + 1:
            print 'Pairing extra pt ', pt
            # Get pair for Lr2 indices not already paired
            tmp = np.where(np.logical_and(LVUC[:, 1] == LVUC[pt, 1], LVUC[:, 0] == LVUC[pt, 0] + 1))[0][0]
            xynew = xy[pt, :] - (xy[tmp, :] - xy[pt, :])
            xy = np.vstack((xy, xynew))
            LVUCnew = LVUC[pt] + np.array([-1, 0, 0])
            LVUC = np.vstack((LVUC, LVUCnew))
            # Add bonds
            BL2add = np.array([[pt, len(xy) - 1], [topnew, len(xy) - 1], [topold, len(xy) - 1]])
            BL = np.vstack((BL, BL2add))

    #################################
    # Check
    if check:
        plt.plot(xy[:, 0], xy[:, 1], 'bo')
        plt.scatter(xy[LEFTrow, 0], xy[LEFTrow, 1], c='r', marker='^')
        # plt.scatter(xy[RIGHTrow,0], xy[RIGHTrow,1], c='g', marker='s')
        for i in range(len(xy[:, 0])):
            plt.text(xy[i, 0], xy[i, 1], str(LVUC[i, 0]) + ', ' + str(LVUC[i, 1]))
        plt.title('After adding one row')
        plt.show()
    #################################

    # Add the right half row
    # Rl2 = Rl2[RlINDs]
    # xynew = np.mean([ xy[RIGHTrow,:], xy[Rl2,:] ], axis=0)
    # xy = np.vstack((xy,xynew))
    # LVUC = np.vstack(( LVUC, LVUCnewRrow ))

    # CUT BONDS
    rows2cut = []
    for ind in LEFTrow:
        # Get bonds between LEFTrow and particle +LV[0] or +LV[1]
        matches = np.where(np.logical_or(np.logical_and(LVUC[Lr2, 0] == LVUC[ind, 0] + 2,
                                                        LVUC[Lr2, 1] == LVUC[ind, 1]),
                                         np.logical_and(LVUC[Lr2, 0] == LVUC[ind, 0] + 1,
                                                        LVUC[Lr2, 1] == LVUC[ind, 1] + 1)
                                         ))[0]

        matchind = Lr2[matches]

        # Kill (ind, matches) pairs in BL
        for match in matchind:
            try:
                row2cut_ii = np.where(np.logical_or(np.logical_and(BL[:, 0] == ind, BL[:, 1] == match),
                                                    np.logical_and(BL[:, 1] == ind, BL[:, 0] == match)))[0][0]
                print 'row2cut_ii = ', row2cut_ii
                rows2cut += [row2cut_ii]
            except:
                print 'no matching row to cut for index = ', match

        # Check
        if check:
            print 'matches = ', matches
            print 'matchind = ', matchind
            le.display_lattice_2D(xy, BL, title='checking bonds to cut (btwn green and red pts)',
                                  colorz=False, close=False);
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            plt.plot(xy[matchind, 0], xy[matchind, 1], 'ro')
            plt.plot(xy[ind, 0], xy[ind, 1], 'g^')
            plt.show()
    #
    # if Bvec == 'SouthWest' or Bvec == 'SW':
    #     print 'Since Bvec is SW, killing one more bond... \n'
    #     # For left point (leftpt), kill northeast bond
    #     matches = np.where( np.logical_and( LVUC[Lr2,0] == LVUC[leftpt,0] , \
    #                                            LVUC[Lr2,1] == LVUC[leftpt,1]+1) )[0]
    #
    #     matchind = Lr2[matches]
    #     row2cut_ii = [np.where(np.logical_or(np.logical_and( BL[:,0] == leftpt, BL[:,1] == match),
    #                                          np.logical_and(BL[:,1] == leftpt, BL[:,0] == match)))[0][0] for match in matchind]
    #     rows2cut += row2cut_ii
    #
    #     # Check
    #     if check:
    #         print 'matches = ', matches
    #         print 'matchind = ', matchind
    #         le.display_lattice_2D(xy,BL,colorz=False,close=False); plt.plot(xy[:,0], xy[:,1],'b.'); plt.plot(xy[matchind,0], xy[matchind,1],'ro'); plt.plot(xy[ind,0], xy[ind,1],'g^')
    #         plt.show()

    keep = np.ones(len(BL), dtype=bool)
    keep[rows2cut] = False
    BL = BL[keep, :]

    # Add new bonds
    # BL2add = np.zeros((len(LaddIND)*5-1,2), dtype = int)
    BL2add = []
    for ii in range(len(LaddIND)):
        new = LaddIND[ii]
        # Add new horizontal bonds
        # add bond  o----x    o
        BL2add.append([LEFTrow[ii], new])
        # add bond  o    x----o
        BL2add.append([Lr2add[ii], new])
        print 'added BL2add entry ', 4 * ii + 1
        # Add new vertical bonds
        # get particle below
        if new == lowest_new_pt:
            print 'Attempting to add bond for lowest point ...'
            try:
                below = np.where(np.logical_and(LVUC[:, 0] == LVUC[new, 0] + 1, LVUC[:, 1] == LVUC[new, 1] - 1))[0][0]
                BL2add.append([below, new])
            except:
                print '... Lowest point is on bottom row, no points below to bond with.'
        else:
            below = np.where(np.logical_and(LVUC[:, 0] == LVUC[new, 0], LVUC[:, 1] == LVUC[new, 1] - 1))[0][0]
            BL2add.append([below, new])

        # get particle above
        try:
            above = np.where(np.logical_and(LVUC[:, 0] == LVUC[new, 0], LVUC[:, 1] == LVUC[new, 1] + 1))[0][0]
            BL2add.append([above, new])
        except:
            print '... no particle above.'

        # check
        if check:
            le.display_lattice_2D(xy, np.vstack((BL, BL2add)), title='Unrelaxed dislocated lattice', colorz=False,
                                  close=False)
            for i in range(len(xy[:, 0])):
                plt.text(xy[i, 0], xy[i, 1] + 0.1, '(' + str(LVUC[i, 0]) + ', ' + str(LVUC[i, 1]) + ')')
            plt.show()

    ii = 0
    for new in LaddIND[0:len(LaddIND) - 1]:
        # new = LaddIND[ii]
        print 'adding cconnection between added pts...'
        BL2add.append([new, new + 1])
        ii += 1

    # Cut off parts of BL2add that weren't used
    print 'ii = ', ii
    print '4*len(LaddIND)+ii+1 = ', 4 * len(LaddIND) + ii + 1
    print 'len(LaddIND)*5-1 = ', len(LaddIND) * 5 - 1

    # if 4*len(LaddIND)+ii < (len(LaddIND)*5):
    BL2add = np.array(BL2add)

    # print 'BL2add = ', BL2add
    BL = np.vstack((BL, BL2add))
    # print 'BL in adding West Bvec= ', BL

    # Check
    if check:
        le.display_lattice_2D(xy, BL, title='Unrelaxed dislocated lattice', colorz=False, close=False)
        # for i in range(len(xy)):
        #    plt.text(xy[i,0]+0.05, xy[i,1]+0.05,  str(LVUC[i]))
        plt.show()

    # bU = potential_energy(xy,BL,bo=1.,kL=1.)
    NP = len(xy)
    kL = 1.
    if bo == 'centroid':
        bo = 1. / (np.sqrt((1. / 3. * np.cos(np.pi / 3.) - 2. / 3.) ** 2 + (1. / 3. * np.sin(np.pi / 3.)) ** 2))
    else:
        bo = bo

    def flattened_potential_energy(xy):
        # We convert xy to a 2D array here.
        xy = xy.reshape((-1, 2))
        bL = le.bond_length_list(xy, BL)
        bU = 0.5 * sum(kL * (bL - bo) ** 2)
        return bU

    if relax_lattice:
        # Relax lattice, but fix pts with low LV[1] vals
        # First get ADJACENT pts with low LV[1] values to fix
        fix_candidates = np.where(LVUC[:, 1] == np.min(LVUC[:, 1]))[0]
        # Grab particle with min abs(x) value from fix_candidates
        fix0_ind = np.argmin(xy[fix_candidates, 0])
        fix0 = fix_candidates[fix0_ind]
        print 'fix0 = ', fix0

        # look at all the other particles (not particle fix_candidates[i])
        inds_noti = np.setdiff1d(np.arange(len(fix_candidates)), np.array([fix0_ind]))
        inds = fix_candidates[inds_noti]
        # Get index of a particle next to fix0 that is one bond away
        getequal = np.where(np.logical_and(LVUC[inds, 1] == LVUC[fix0, 1],
                                           np.abs(LVUC[inds, 0] - LVUC[fix0, 0]) == 1))[0]
        print 'getequal = ', getequal
        if len(getequal) > 1:
            print '\n there is more than one...'
            getequal2 = np.argmin(xy[getequal, 0])[0]
            print 'getequal = ', getequal
            equalIND = getequal[getequal2]
            print 'equalIND = ', equalIND
        else:
            equalIND = getequal[0]

        print 'inds = ', inds
        if fix0 < inds[equalIND]:
            pair = [fix0, inds[equalIND]]
        else:
            pair = [inds[equalIND], fix0]

        Nzero_pair0 = np.arange(0, pair[0])
        Npair0_pair1 = np.arange(pair[0], pair[1])
        Npair1_end = np.arange(pair[1], NP)
        print 'pair = ', pair
        bounds = [[None, None]] * (2 * (pair[0])) + \
                 np.c_[xy[pair[0], :].ravel(), xy[pair[0], :].ravel()].tolist() + \
                 [[None, None]] * (2 * (pair[1] - pair[0] - 1)) + \
                 np.c_[xy[pair[1], :].ravel(), xy[pair[1], :].ravel()].tolist() + \
                 [[None, None]] * (2 * (NP - pair[1] - 1))

        # Old way: fix particles 0 and 1
        # bounds = np.c_[xy[:2,:].ravel(), xy[:2,:].ravel()].tolist() + \
        #                 [[None, None]] * (2*(NP-2))

        # relaxed lattice
        print 'relaxing lattice...'
        xyR = opt.minimize(flattened_potential_energy, xy.ravel(),
                           method='L-BFGS-B',
                           bounds=bounds, tol=tol).x.reshape((-1, 2))
        xy = xyR

        bL0 = bo * np.ones_like(BL[:, 0], dtype=float)
        if check:
            bs = le.bond_strain_list(xy, BL, bL0)
            le.display_lattice_2D(xyR, BL, bs=bs, title='Relaxed dislocated lattice', colorz=False, close=False)
            plt.scatter(xy[pair, 0], xy[pair, 1], s=500, c='r')
            plt.show()

        if relax_twice:
            # Kick lattice, relax again, fixing pts far from 0 and 1
            # Fix pair of particles near top of sample
            # first grab lowest particle
            fix0 = np.argmax(xy[:, 1])
            found = tree.query(xy[fix0], k=2)[1]
            print 'found =', found
            print 'tree.query(xy[fix0], k=2) = ', tree.query(xy[fix0], k=2)
            print 'found partners for second relaxing => fix0 =', fix0, ' and ', found[1]
            pair = [fix0, found[1]]

            # If for some reason found[1] == fix0, choose a different particle in found
            if fix0 == found[1]:
                print 'found[0] = ', found[0]
                pair = [fix0, found[0]]

            Nzero_pair0 = np.arange(0, pair[0])
            Npair0_pair1 = np.arange(pair[0], pair[1])
            Npair1_end = np.arange(pair[1], NP)
            print 'pair = ', pair
            bounds = [[None, None]] * (2 * (pair[0])) + \
                     np.c_[xy[pair[0], :].ravel(), xy[pair[0], :].ravel()].tolist() + \
                     [[None, None]] * (2 * (pair[1] - pair[0] - 1)) + \
                     np.c_[xy[pair[1], :].ravel(), xy[pair[1], :].ravel()].tolist() + \
                     [[None, None]] * (2 * (NP - pair[1] - 1))

            # Kick particles that are not fixed during relaxation
            kick = 0.0001 * np.random.rand(NP, 2)
            kick[fix0] = [0., 0.]
            kick[found] = [0., 0.]
            xy += kick

            print 'relaxing lattice again...'
            xyR = opt.minimize(flattened_potential_energy, xy.ravel(),
                               method='L-BFGS-B',
                               bounds=bounds, tol=tol * 0.1).x.reshape((-1, 2))
            xy = xyR

            if check:
                bs = le.bond_strain_list(xy, BL, bL0)
                le.display_lattice_2D(xyR, BL, bs=bs, title='Twice Relaxed dislocated lattice', colorz=False,
                                      close=False)
                plt.scatter(xy[NP - 2:NP, 0], xy[NP - 2:NP, 1], s=500, c='r')
                plt.show()

    return xy, LVUC, BL
