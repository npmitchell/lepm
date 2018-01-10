import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

'''Functions for building a topological overcoordinated lattice'''


def generate_overcoordinated1_lattice(NH, NV, shape='square', check=False):
    """Generates a lattice with a made-up unit cell that is overcoordinated.

    Parameters
    ----------
    NH : int
        Number of pts along horizontal before boundary is cut
    NV : int

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
    SBL : NP x 1 int array
        Sublattice bond list: 0, 1, 2 for each site of the 3-particle unit cell
    lattice_type : string
        label, lattice type.  For making output directory: for ex, 'overcoordinated1_square'

    """
    print('Setting up unit cell...')
    a0 = 2*np.array([[np.cos(n*np.pi/3+np.pi/6), np.sin(n*np.pi/3+np.pi/6), 0] for n in range(6)])

    a = np.array([a0[:, 0], a0[:, 1]]).T
    A = np.array([0, 0])
    C = np.array([np.cos(-10*np.pi/180), np.sin(-10*np.pi/180)])
    B = 1.2*np.array([np.cos(20*np.pi/180), np.sin(20*np.pi/180)])

    An = A+ a
    Bn = B+a
    Cn = C+a

    nA = np.array([B-A, B+a[3]-A, B+a[4]-A, C-A, C+a[2]-A])
    angsA = np.array([np.arctan2(nA[i, 1], nA[i, 0]) for i in range(len(nA))])

    nB = np.array([A-B, A+a[0]-B, A+a[1]-B, C-B, C+a[1]-B])
    angsB = np.array([np.arctan2(nB[i, 1], nB[i, 0]) for i in range(len(nB))])

    nC = np.array([A-C, A+a[5]-C, B-C, B+a[4]-C])
    angsC = np.array([np.arctan2(nC[i,1], nC[i,0]) for i in range(len(nC))])

    aa = a.copy()
    print 'aa = ', aa
    a1 = aa[1]
    a2 = aa[2]
    a3 = aa[3]

    # nodes at R
    C = np.array([A,#0
                  B,  # 1
                  B + a[3],  # 2
                  B + a[4],  # 3
                  C,  # 4
                  C + a[2],  # 5
                  A+a[0],  # 6
                  A+a[1],  # 7
                  C+a[1],  # 8
                  A+a[5],  # 9

        ])
    CU = np.arange(len(C), dtype='int')

    sbl = np.array([0,#0
               1,#1
               1,#2
               1,#3
               2,#4
               2,#5

               #A
               0, #6
               0, #7
               #C, #4
               2,#8

               #A
               0,#9
               #B,#1
               #B+a[4]#3

        ])

    CBL = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 6],
        [1, 7],
        [1, 4],
        [1, 8],
        [3, 4],
        [4, 9]
        ])

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    LV  = [-a2-a3, a1]
    inds = np.arange(len(C))
    tmp0 = np.zeros(len(C), dtype='int')
    tmp1 = np.ones(len(C), dtype='int')
    print 'sbl = ', sbl
    for i in np.arange(NV):
        for j in np.arange(NH):
            # bottom row
            if i == 0:
                if j == 0 :
                    Bprev = C
                    R = C
                    SBL = sbl.tolist()
                    LVUC = np.dstack((tmp0, tmp0, CU))[0]

                    # # Check
                    # colorvals = np.linspace(0,1,len(R))
                    # #plt.plot(R[:,0],R[:,1], 'k-') #,c=colorvals, cmap='spectral')
                    # le.display_lattice_2D(R,CBL,close=False)
                    # for ii in range(len(R)):
                    #     plt.text(R[ii,0]+0.05,R[ii,1],str(ii))
                    # plt.arrow(0, 0, a1[0], a1[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
                    # plt.arrow(0, 0, a2[0], a2[1], head_width=0.05, head_length=0.1, fc='b', ec='b')
                    # plt.arrow(0, 0, a3[0], a3[1], head_width=0.05, head_length=0.1, fc='g', ec='g')
                    # plt.show()
                else:
                    R = np.vstack((R, C + j*LV[0]))
                    SBL += sbl.tolist() # np.vstack((SBL, sbl))

                    LVUCadd = np.dstack((j*tmp1, tmp0, CU))[0]
                    # print 'LVUCadd = ', LVUCadd
                    LVUC = np.vstack((LVUC, LVUCadd))

                    # # Check
                    # colorvals = np.linspace(0,1,len(R))
                    # plt.scatter(R[:,0],R[:,1],c=colorvals, cmap='spectral')
                    # for ii in range(len(R)):
                    #     plt.text(R[ii,0]+0.05,R[ii,1],str(ii))
                    # plt.arrow(0, 0, a1[0], a1[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
                    # plt.arrow(0, 0, a2[0], a2[1], head_width=0.05, head_length=0.1, fc='b', ec='b')
                    # plt.arrow(0, 0, a3[0], a3[1], head_width=0.05, head_length=0.1, fc='g', ec='g')
                    # plt.pause(1)
            else:
                # vertical indices to copy over
                vinds = np.array([1,2,5,6,7,8])
                R = np.vstack((R, C[vinds] + j*LV[0] + i*LV[1]))
                SBL += sbl[vinds].tolist()
                LVUCadd = np.dstack((j*tmp1[vinds],i*tmp1[vinds],CU[vinds] ))[0]
                # print 'LVUCadd = ', LVUCadd
                LVUC = np.vstack((LVUC,LVUCadd))

                # # Check
                # sizevals = np.arange(len(R))+10
                # colorvals = np.linspace(0,1,len(R))
                # plt.scatter(R[:,0],R[:,1],s=sizevals,c=colorvals, cmap='afmhot')
                # for ii in range(len(R)):
                #     plt.text(R[ii,0]+0.05,R[ii,1],str(ii))
                # plt.arrow(0, 0, a1[0], a1[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
                # plt.arrow(0, 0, a2[0], a2[1], head_width=0.05, head_length=0.1, fc='b', ec='b')
                # plt.arrow(0, 0, a3[0], a3[1], head_width=0.05, head_length=0.1, fc='g', ec='g')
                # plt.show()
        # if i%2 ==0:
        #     Bprev = Bprev + a2
        # else:
        #     Bprev = Bprev + a3

    SBL = np.asarray(SBL)
    # get rid of repeated points
    print('Eliminating repeated points...')
    #########
    # check
    if check:
        plt.clf()
        sizevals = np.arange(len(R))+10
        colorvals = np.linspace(0,1,len(R))
        plt.scatter(R[:,0],R[:,1],s=sizevals,c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()
    #########
    print 'R = ', R
    print 'shape(R) = ', np.shape(R)
    R, si, ui = le.args_unique_rows_threshold(R, 0.01)
    print 'shape(R) = ', np.shape(R)
    #########
    # check
    if check:
        sizevals = np.arange(len(R))+50
        colorvals = np.linspace(0, 1, len(R))
        plt.scatter(R[:, 0], R[:, 1], s=sizevals, c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()
    #########
    BL2 = generate_overcoordinated1_lattice_trip(NH, NV, ui, si, C, CBL)
    SBL = SBL.flatten()
    SBL = SBL[si]
    SBL = SBL[ui]
    # get rid of first row (0,0)a
    xy = le.unique_rows_threshold(R, 0.05) - np.array([np.mean(R[1:, 0]), np.mean(R[1:, 1])])

    # xy = R[ui]
    # Triangulate
    print('Triangulating...')
    Dtri = Delaunay(xy)
    btri = Dtri.vertices

    # C = C-np.array([np.mean(R[1:,0]),np.mean(R[1:,1])])
    # fig = plt.figure()
    # plt.triplot(xy[:,0], xy[:,1], Dtri.simplices.copy())
    # plt.xlim(-10,10)
    # plt.ylim(-10, 10)
    # plt.show()

    # translate btri --> bond list
    BL = le.Tri2BL(btri)

    # remove bonds on the sides and through the hexagons
    print('Removing extraneous bonds from triangulation...')
    # calc vecs from C bonds

    BL = np.array(list(BL)+list(BL2))
    BL = le.unique_rows(BL)

    BL = blf.latticevec_filter(BL, xy, C, CBL)

    NL, KL = le.BL2NLandKL(BL,NN=10)

    UC = C
    lattice_exten = 'overcoordinated1_square'
    return xy, NL, KL, BL, SBL, LV, UC, LVUC, lattice_exten


def generate_overcoordinated1_lattice_trip(NH, NV, ui, si, Co, CBLo):
    """
    Moves the points of the weird lattice around to get bonds that a normal triangulation would miss.
    Parameters
    ----------
    NH : int
        Number of pts along horizontal before boundary is cut
    NV : int

    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    """
    print('Setting up unit cell...')
    a0 = 2*np.array([[np.cos(n*np.pi/3+np.pi/6), np.sin(n*np.pi/3+np.pi/6), 0] for n in range(6)])

    a = np.array([a0[:,0], a0[:,1]]).T
    A = np.array([0,0])
    C = np.array([np.cos(-10*np.pi/180), np.sin(-10*np.pi/180)])
    B = 0.5*np.array([np.cos(90*np.pi/180), np.sin(90*np.pi/180)])

    An = A+ a
    Bn = B+a
    Cn = C+a

    nA = np.array([B-A, B+a[3]-A, B+a[4]-A, C-A, C+a[2]-A])
    angsA = np.array([np.arctan2(nA[i,1], nA[i,0]) for i in range(len(nA))])

    nB = np.array([A-B, A+a[0]-B, A+a[1]-B, C-B, C+a[1]-B])
    angsB = np.array([np.arctan2(nB[i,1], nB[i,0]) for i in range(len(nB))])

    nC = np.array([A-C, A+a[5]-C, B-C, B+a[4]-C])
    angsC = np.array([np.arctan2(nC[i,1], nC[i,0]) for i in range(len(nC))])

    aa = a.copy()
    a1 = aa[1]
    a2 = aa[2]
    a3 = aa[3]

    # nodes at R
    C = np.array([A,  # 0
               B,     # 1
               B+a[3],  # 2
               B+a[4],  # 3
               C,     # 4
               C+a[2],  # 5

               #A
               A+a[0], #6
               A+a[1], #7
               #C, #4
               C+a[1],#8

               #A
               A+a[5],#9
               #B,#1
               #B+a[4]#3

        ])

    CBL = np.array([[0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [0, 5],
                    [1, 6],
                    [1, 7],
                    [1, 4],
                    [1, 8],
                    [3, 4],
                    [4, 9]])

    # translate by Bravais latt vecs
    print('Translating by Bravais lattice vectors...')
    print('Translating by Bravais lattice vectors...')
    LV = [-a2-a3, a1]
    inds = np.arange(len(C))
    tmp0 = np.zeros(len(C), dtype='int')
    tmp1 = np.ones(len(C), dtype='int')
    for i in np.arange(NV):
        for j in np.arange(NH):
            # bottom row
            if i == 0:
                if j == 0 :
                    R = C
                else:
                    R = np.vstack((R, C + j*LV[0]))
            else:
                # vertical indices to copy over
                vinds = np.array([1, 2, 5, 6, 7, 8])
                R = np.vstack((R, C[vinds] + j*LV[0] + i*LV[1]))

    print('Eliminating repeated points...')
    # ui = le.unique_rows_index(R)

    R = R[si]
    xy = R[ui]
    # Triangulate
    print('Triangulating...')
    Dtri = Delaunay(xy)
    btri = Dtri.vertices

    # translate btri --> bond list
    BL = le.Tri2BL(btri)
    # BL = latticevec_filter(BL,xy, Co, CBLo)

    # fig = plt.figure()
    # plt.triplot(xy[:,0], xy[:,1], Dtri.simplices.copy())
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.show()

    return BL


