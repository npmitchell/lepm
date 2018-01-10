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


def build_deformed_martini(lp):
    """Build distorted martini lattice as in Kane&Lubensky2014--> Paulose 2015"""
    NVdk = int(round(lp['NV'] / np.sqrt(3)))
    xy, NL, KL, BL, lattice_exten, LV, UC, LVUC = generate_deformed_martini(lp['shape'], lp['NH'], NVdk,
                                                                            lp['x1'], lp['x2'], lp['x3'],
                                                                            lp['z_kagome'], check=lp['check'])
    PVxydict = {}
    PVx = []
    PVy = []
    LL = (np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))
    # polygon = auto_polygon(shape, NH, NV, eps=0.00)
    BBox = np.array([[-LL[0] * 0.5, -LL[1] * 0.5], [LL[0] * 0.5, -LL[1] * 0.5],
                     [LL[0] * 0.5, LL[1] * 0.5], [-LL[0] * 0.5, LL[1] * 0.5]])
    return xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten


def generate_deformed_martini(shape, NH, NV, x1, x2, x3, z, check=False):
    """creates distorted martini lattice as in Kane&Lubensky2014--> Paulose 2015

    Parameters
    ----------
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
    print('Setting up unit cell...')
    # Bravais primitive unit vecs
    a = np.array([[np.cos(2*np.pi*p/3.), np.sin(2*np.pi*p/3.)] for p in range(3)])
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    # make unit cell
    x = np.array([x1, x2, x3])
    y = np.array([z/3. + x[np.mod(i-1, 3)]- x[np.mod(i+1, 3)] for i in [0, 1, 2]])
    s = np.array([x[p]*(a[np.mod(p-1, 3)] - a[np.mod(p+1, 3)]) + y[p]*a[p] for p in range(3)])
    s1 = s[0]
    s2 = s[1]
    d1 = a1/2.+s2
    d2 = a2/2.-s1
    d3 = a3/2.

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    C = np.array([d1+a2,
        d3+a1+a2,
        d2+a1,
        d1,
        d3+a1,
        d2-a2,
        d1+a3,
        d3,
        d2+a3,
        d1+a2+a3,
        d3+a2,
        d2])

    LV = np.array([a1, 2.*a2+a1])
    CU = np.arange(len(C))
    tmp1 = np.ones(len(C), dtype=int)

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    inds = np.arange(len(C))
    for i in np.arange(NV):
        print 'Building row ', i, ' of ', NV
        for j in np.arange(NH):
            if i == 0:
                if j == 0:
                    # initialize
                    R = C;
                    LVUC = np.dstack(([0*tmp1, 0*tmp1, CU]))[0]
                    # Rename particle 4 as particle 7 of next over in LV0, for martini trimming
                    # LVUC[4] = np.array([1,0,7])
                else:
                    # bottom row (include point 6 in translation)
                    R = np.vstack((R, C[0:7,:] + i*(2*a2+a1) + j*a1))
                    LVUCadd = np.dstack(( (i+j)*tmp1[0:7], (2*i)*tmp1[0:7], CU[0:7] ))[0]
                    print 'LVUC = ', LVUC
                    print 'LVUCadd = ', LVUCadd
                    LVUC = np.vstack((LVUC, LVUCadd))
            else:
                if j == 0:
                    # first cell of row, include all but pt 6
                    R = np.vstack((R, C[inds!=6,:] + i*(2*a2+a1) + j*a1))
                    LVUCadd = np.dstack(( (i+j)*tmp1[inds!=6], (2*i)*tmp1[inds!=6],CU[inds!=6] ))[0]
                    # Rename particle 4 as particle 7 of next over in LV0, for martini trimming
                    # LVUCadd[4] = np.array([1,0,7])
                    LVUC = np.vstack((LVUC, LVUCadd))
                else:
                    # only points 0 through 5 included
                    R = np.vstack((R, C[0:6,:] + i*(2*a2+a1) + j*a1))
                    LVUCadd = np.dstack(( (i+j)*tmp1[0:6], (2*i)*tmp1[0:6], CU[0:6] ))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))

    if check:
        plt.plot(R[:, 0], R[:, 1], '.-')
        plt.show()

    # check for repeated points
    print('Checking for repeated points...')
    print 'len(R) =', len(R)
    Rcheck = le.unique_rows(R)
    print 'len(Rcheck) =', len(Rcheck)
    if len(R)-len(Rcheck) !=0:
        sizes = np.arange(len(xy))
        plt.scatter(xy[:,0],xy[:,1],s=sizes)
        raise RuntimeError('Repeated points!')
    else:
        print 'No repeated points.\n'
    xy = R
    xy -= np.array([np.mean(R[1:,0]),np.mean(R[1:,1])]) ;
    #Triangulate
    print('Triangulating...')
    Dtri = Delaunay(xy)
    btri = Dtri.vertices
    #translate btri --> bond list
    BL = le.Tri2BL(btri)

    #remove bonds on the sides and through the hexagons
    print('Removing extraneous bonds from triangulation...')
    #calc vecs from C bonds
    CBL = np.array([[0,1],[1,11],[0,11],[1,2],[2,3],[3,4],[4,5],[3,5],
        [5,6],[6,7],[7,8],[8,9],[7,9],[9,10],[10,11]])

    BL = latticevec_filter(BL,xy, C, CBL)

    # Now kill bonds between 1-3, 5-7, 9-11
    # Also remove bonds between 4 of left kagome and 5 of right kagome
    BLUC = np.dstack((LVUC[BL[:,0],2], LVUC[BL[:,1],2] ))[0]
    print 'len(BLUC) = ', len(BLUC)
    print len(BL)
    kill1 = le.rows_matching(BLUC, np.array([1,3] ))
    kill2 = le.rows_matching(BLUC, np.array([5,7] ))
    kill3 = le.rows_matching(BLUC, np.array([9,11]))

    # Remove bonds between 4 of left kagome and 5 of right kagome
    in45a = np.in1d(BLUC[:,0], np.array([4,5]))
    in45b = np.in1d(BLUC[:,1], np.array([4,5]))
    BLLV1 = np.dstack((LVUC[BL[:,0],0], LVUC[BL[:,1],0] ))[0]
    kill4 = np.where(np.logical_and( np.logical_and(in45a, in45b), BLLV1[:,0] != BLLV1[:,1]))[0]

    # print 'BLUC = ', BLUC
    # print 'kill1 = ', kill1
    # print 'kill2 = ', kill2
    # print 'kill3 = ', kill3
    keep = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(len(BLUC)), kill1), kill2), kill3), kill4)
    # print 'keep = ', keep

    if check:
        le.display_lattice_2D(xy, BL, close=False)
        for ii in range(len(BL)):
            plt.text( np.mean([xy[BL[ii,0],0], xy[BL[ii,1],0]]), np.mean([xy[BL[ii,0],1],xy[BL[ii,1],1]]), str(BLUC[ii]) )
        for ii in kill1:
            plt.text( np.mean([xy[BL[ii,0],0], xy[BL[ii,1],0]]), np.mean([xy[BL[ii,0],1],xy[BL[ii,1],1]]), str(BLUC[ii]), bbox=dict(facecolor='red', alpha=0.5) )
        for ii in kill2:
            plt.text( np.mean([xy[BL[ii,0],0], xy[BL[ii,1],0]]), np.mean([xy[BL[ii,0],1],xy[BL[ii,1],1]]), str(BLUC[ii]), bbox=dict(facecolor='red', alpha=0.5) )
        for ii in kill3:
            plt.text( np.mean([xy[BL[ii,0],0], xy[BL[ii,1],0]]), np.mean([xy[BL[ii,0],1],xy[BL[ii,1],1]]), str(BLUC[ii]), bbox=dict(facecolor='red', alpha=0.5) )
        for ii in kill4:
            plt.text( np.mean([xy[BL[ii,0],0], xy[BL[ii,1],0]]), np.mean([xy[BL[ii,0],1],xy[BL[ii,1],1]]), str(BLUC[ii]), bbox=dict(facecolor='red', alpha=0.5) )
        for ii in range(len(xy)):
            plt.text( xy[ii,0], xy[ii,1], str(LVUC[ii,2]) )
        plt.show()

    BL = BL[keep,:]
    NL,KL = le.BL2NLandKL(BL, NN=4)
    lattice_exten = 'deformed_martini_' + shape + '_x1_' + '{0:.4f}'.format(x1) + '_x2_' + '{0:.4f}'.format(x2) + \
                    '_x3_' + '{0:.4f}'.format(x3) + '_z_' + '{0:.4f}'.format(z)
    UC = C
    return xy,NL,KL,BL,lattice_exten, LV, UC, LVUC

