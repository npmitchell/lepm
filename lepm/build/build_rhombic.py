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
Functions for making rhombic and stacked rhombic lattices
'''


def build_stacked_rhombic(lp):
    """Create a stacked rhombic lattice using lattice parameters lp.
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
    if 'phi' not in lp:
        lp['phi'] = np.pi * float(lp['phi_lattice'])

    print 'checking that theta = ', lp['theta']
    print 'checking that periodicBC = ', lp['periodicBC']
    if lp['NH'] == 1 and lp['NV'] == 1:
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_stacked_rhombic_strip(lp)
    else:
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, PV, lattice_exten = generate_stacked_rhombic(lp)

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


def generate_stacked_rhombic(lp):
    """Generates stacked rhombic lattice (points, connectivity, name).
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
        intparam : int
            how many rhombuses are stacked before repeating. If 1, then just make the sheared square lattice.
            If 2, make sheared right, then left. If 3, then make 2 sheared right, one sheared left, etc.
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
    intp = lp['intparam']
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

    # make the unit cell constituents
    if intp == 1:
        # make right leaning rhombic
        cc = np.array([[0., 0.]])
        CU = np.array([0])
    elif intp == 2:
        # make right, left stacked
        cc = np.array([[0., 0.], [np.sin(phi), np.cos(phi)]])
        CU = np.array([0, 1])
        latticevecs = np.array([[1., 0], [0., 2. * np.cos(phi)]])
    elif intp == 3:
        # make right, right, left stacked
        cc = np.array([[0., 0.], [np.sin(phi), np.cos(phi)],
                       [2. * np.sin(phi), 2. * np.cos(phi)]])
        CU = np.array([0, 1, 2])
        latticevecs = np.array([[1., 0], [np.sin(phi), 3. * np.cos(phi)]])
    else:
        raise RuntimeError('Have not coded case for intparam > 3 yet')

    # A way to rotate the latticevecs below:
    LV = latticevecs

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
                        rr = np.vstack((rr, cc + i * LV[1] + j * LV[0]))
                        LVUCadd = np.dstack((j * tmp1, i * tmp1, CU))[0]
                        LVUC = np.vstack((LVUC, LVUCadd))
                else:
                    rr = np.vstack((rr, cc + i * LV[1] + j * LV[0]))
                    LVUCadd = np.dstack((j * tmp1, i * tmp1, CU))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))
    else:
        raise RuntimeError('Currently do not support this shape')

    xy = rr
    xy -= np.array([np.mean(rr[1:, 0]), np.mean(rr[1:, 1])])

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
    ccexp = np.vstack((cc, cc + LV[0], cc + LV[1]))
    #
    #  sp          sp
    #
    #
    #        sp
    #
    if intp == 1:
        CBL = np.array([[0, 1], [0, 2]])
    elif intp == 2:
        CBL = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])
    elif intp == 3:
        CBL = np.array([[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [2, 6]])

    BL = blf.latticevec_filter(BL, xy, ccexp, CBL)
    print 'BL = ', BL
    NL, KL = le.BL2NLandKL(BL, NN=4)

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

    lattice_exten = 'stackedrhombic_' + shape + periodicstr + \
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
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
            raise RuntimeError('Have not coded for stacked rhombic hexagon periodic case')
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
            boundary = le.extract_boundary(xy, NL, KL, BL)
            # For each boundary particle, give it extra neighbors
            for ind in boundary:
                if LVUC[ind, 1] < NV:
                    print 'ind = ', ind
                    add = False
                    # Is particle on bottom edge? if so, to be connected with particle UC=2 on top edge
                    if LVUC[ind, 1] == 0:
                        msg = 'This is a particle on the left boundary, to be connected with right boundary'
                        # grab new neighbor: same LV[0], NV for LV[1], opposing UC
                        want = [NH - 1, LVUC[ind, 1], LVUC[ind, 2]]
                        # Perioidic virtual displacement vector of ind
                        PV = LV[0] * (-NH)
                        add = True
                    elif LVUC[ind, 2] == 0 and not periodic_strip:
                        msg = 'This is a particle on the bottom boundary'
                        want = [LVUC[ind, 0], NV - 1, LVUC[ind, 2]]
                        PV = LV[1] * (-NV)
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


def generate_stacked_rhombic_strip(lp):
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
    intp = lp['intparam']
    shape = lp['shape']
    NH = int(lp['NH'])
    NV = int(lp['NV'])
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

    # make the unit cell constituents
    if intp == 1:
        # make right leaning rhombic
        cc = np.array([[0., 0.]])
        CU = np.array([0])
    elif intp == 2:
        # make right, left stacked
        cc = np.array([[0., 0.], [np.sin(phi), np.cos(phi)]])
        CU = np.array([0, 1])
        latticevecs = np.array([[1., 0], [0., 2. * np.cos(phi)]])
    elif intp == 3:
        # make right, right, left stacked
        cc = np.array([[0., 0.], [np.sin(phi), np.cos(phi)],
                       [2. * np.sin(phi), 2. * np.cos(phi)]])
        CU = np.array([0, 1, 2])
        latticevecs = np.array([[1., 0], [np.sin(phi), 3. * np.cos(phi)]])
    else:
        raise RuntimeError('Have not coded case for intparam > 3 yet')

    # A way to rotate the latticevecs below:
    LV = latticevecs

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
                        rr = np.vstack((rr, cc + i * LV[1] + j * LV[0]))
                        LVUCadd = np.dstack((j * tmp1, i * tmp1, CU))[0]
                        LVUC = np.vstack((LVUC, LVUCadd))
                else:
                    rr = np.vstack((rr, cc + i * LV[1] + j * LV[0]))
                    LVUCadd = np.dstack((j * tmp1, i * tmp1, CU))[0]
                    LVUC = np.vstack((LVUC, LVUCadd))
    else:
        raise RuntimeError('Currently do not support this shape')

    # If the dimensions are 1 x 1, handle that case separately here
    if NH == 1 and NV == 1:
        xy = cc
        xy -= np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
        cc = CU
        if intp == 1:
            raise RuntimeError('How to implement stacked unitcell with N==1?')
        elif intp == 2:
            raise RuntimeError('Have not finished stacked unitcell with N==2')
            BL = np.array([[0, 1], [1, 2], [0, 2], [1, 3], [3, 4], [4, 5], [3, 5]])
            NL = np.array([[1, 1, 0, 0], [0, 2, 3],
                           [0, 1, 0], [1, 4, 5],
                           [3, 5, 0], [3, 4, 0]])
            KL = np.array([[1, 1, 0], [1, 1, 1],
                           [1, 1, 0], [1, 1, 1],
                           [1, 1, 0], [1, 1, 0]])
            # Note that BL, NL, and KL will be overwritten if periodic
            LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]])
        elif intp == 3:
            BL = np.array([[0, 1], [1, 2], [-0, -0], [-1, -1], [-2, -2]])
            NL = np.array([[1, 0, 0, 0],
                           [0, 2, 0, 0],
                           [1, 0, 0, 0]])
            KL = np.array([[1, 0, 0, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0]])
            # Note that BL, NL, and KL will be overwritten if periodic
            LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
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
        # Triangulate and filter bonds
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
        ccexp = np.vstack((cc, cc + LV[0], cc + LV[1]))
        #
        #  sp          sp
        #
        #
        #        sp
        #
        if intp == 1:
            CBL = np.array([[0, 1], [0, 2]])
        elif intp == 2:
            CBL = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])
        elif intp == 3:
            CBL = np.array([[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [2, 6]])

        BL = blf.latticevec_filter(BL, xy, ccexp, CBL)
        NL, KL = le.BL2NLandKL(BL, NN=4)

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
                    '_phi' + sf.float2pstr(phi / np.pi, ndigits=3) + \
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
                # Connect each particle to the other one -- redo NL, KL, BL
                PV = latticevecs
                pv0, pv1 = PV[0], PV[1]
                if intp == 1:
                    raise RuntimeError('How to implement stacked unitcell with N==1?')
                elif intp == 2:
                    raise RuntimeError('Have not finished stacked unitcell with N==2')
                    BL = np.array([[0, 1], [1, 2], [0, 2], [1, 3], [3, 4], [4, 5], [3, 5]])
                    NL = np.array([[1, 1, 0, 0], [0, 2, 3],
                                   [0, 1, 0], [1, 4, 5],
                                   [3, 5, 0], [3, 4, 0]])
                    KL = np.array([[1, 1, 0], [1, 1, 1],
                                   [1, 1, 0], [1, 1, 1],
                                   [1, 1, 0], [1, 1, 0]])
                    # Note that BL, NL, and KL will be overwritten if periodic
                    LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5]])
                elif intp == 3:
                    if periodic_strip:
                        BL = np.array([[0, 1], [1, 2], [-0, -0], [-1, -1], [-2, -2]])
                        NL = np.array([[1, 0, 0, 2],
                                       [0, 2, 1, 1],
                                       [1, 2, 2, 0]])
                        KL = np.array([[1, -1, -1, 0],
                                       [1, 1, -1, -1],
                                       [1, -1, -1, 0]])
                        # Note that BL, NL, and KL will be overwritten if periodic
                        LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
                    else:
                        BL = np.array([[0, 1], [1, 2], [0, -2], [-0, -0], [-1, -1], [-2, -2]])
                        NL = np.array([[1, 0, 0, 2],
                                       [0, 2, 1, 1],
                                       [1, 2, 2, 0]])
                        KL = np.array([[1, -1, -1, -1],
                                       [1, 1, -1, -1],
                                       [1, -1, -1, -1]])
                        # Note that BL, NL, and KL will be overwritten if periodic
                        LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])

                # Enter element into PVx and PVy arrays
                pv0x, pv0y = pv0[0], pv0[1]
                pv1x, pv1y = pv1[0], pv1[1]
                PVx = np.array([[0., -pv0x, pv0x, -pv1x],
                                [0., 0., -pv0x, pv0x],
                                [0., -pv0x, pv0x, pv1x]])
                PVy = np.array([[0., -pv0y, pv0y, -pv1y],
                                [0., 0., -pv0y, pv0y],
                                [0., -pv0y, pv0y, pv1y]])

                PVxydict = {(0, 0): np.array([[-pv0[0], -pv0[1]], [pv0[0], pv0[1]]]),
                            (1, 1): np.array([[-pv0[0], -pv0[1]], [pv0[0], pv0[1]]]),
                            (2, 2): np.array([[-pv0[0], -pv0[1]], [pv0[0], pv0[1]]]),
                            (0, 2): -pv1
                            }
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
        netvis.movie_plot_2D(xy, BL, KL=KL, NL=NL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict,
                             colormap='BlueBlackRed', title='Output from generate_spindle_lattice()', show=True)
        netvis.movie_plot_2D(xy, BL, KL=KL, NL=NL, bs=0.0 * BL[:, 0], PVx=PVx, PVy=PVy, PVxydict=PVxydict,
                             colormap='BlueBlackRed',
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


