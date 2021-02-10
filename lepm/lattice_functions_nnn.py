import numpy as np
import lepm.lattice_elasticity as le
import lepm.data_handling as dh
import sys


'''Functions for determining info about next-nearest-neighbors or n'th nearest neighbors
'''


def calc_NLNNN_and_KLNNN(xy, BL, NL, KL, PVx=None, PVy=None, polygons=None, ignore_tris=False):
    """Given network, return coupling array such that cc cyclic NNNs have bond strength +1, clockwise cyclic NNNs have
    bond strength -1.

    Parameters
    ----------
    xy : array of dimension nxd
        2D lattice of points (positions x,y(,z) )
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points. Negative values denote particles connected across
        periodic BCs.
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL : array of dimension #pts x (max number of neighbors)
        spring connection/constant list, where one corresponds to a true connection while 0 signifies that there is not
        a connection.
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    polygons : None or list of int lists
        The polygons comprising the network, each list contains indices of polygon vertices
    ignore_tris : bool
        Ignore NNN hoppings to sites which are also NN --> this is done by ignoring triangular polygons in the t2
        component of the dynamical matrix

    Returns
    ----------
    NLNNN : NP x maxNNN int array
        Next nearest neighbor particle indices: ith row gives next nearest neighbors of ith particle
    KLNNN : NP x maxNNN int array
        Connectivity/orientation array: ith row gives orientation (cc (+1) or clockwise (-1)) of connected next
        nearest neighbors of ith particle. ccw cyclic NNNs have bond strength +1, clockwise cyclic NNNs have
        bond strength -1.
    """
    # preallocate
    KLNNN = np.zeros((len(xy), 2 * np.shape(NL)[1]), dtype=int)
    NLNNN = np.zeros((len(xy), 2 * np.shape(NL)[1]), dtype=int)
    # For network, return polygons
    if polygons is None:
        polygons = le.extract_polygons_lattice(xy, BL, NL, KL, PVx=PVx, PVy=PVy, viewmethod=False)
    for jj in range(len(polygons)):
        pp = polygons[jj][:-1]
        # print 'poly = ', pp
        if not ignore_tris or len(pp) > 3:
            for i in range(len(pp)):
                index = np.mod(i + 2, len(pp))
                # get column in which to add to KLNNN
                firstzero = np.where(KLNNN[pp[i], :] == 0)[0][0]

                # Could exclude having a site be its own NNN
                # if pp[index] not in NLNNN[pp[i], 0:firstzero]:
                # print 'site = ', pp[i], ' NNN  = ', pp[index]
                # For each cyclic permutation, designate NNN
                KLNNN[pp[i], firstzero] = 1
                NLNNN[pp[i], firstzero] = pp[index]

                # Do counterclockwise NNN from pp[i]
                index2 = np.mod(i - 2, len(pp))
                firstzero = np.where(KLNNN[pp[i], :] == 0)[0][0]
                # if pp[index2] not in NLNNN[pp[i], 0:firstzero]:
                # print '  ->  site = ', pp[i], ' NNN = ', pp[index2]
                # For each cyclic permutation, designate NNN
                KLNNN[pp[i], firstzero] = -1
                NLNNN[pp[i], firstzero] = pp[index2]

    return NLNNN, KLNNN


def NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx=None, PVy=None, cwccw=False):
    """Compute the difference between NN and NNN bond angles of each particle, theta_{nml} = theta_nm - theta_ml,
    for the particle n, as shown below. This DOES support periodic boundary conditions.
     '''
       l o
          \
          \ theta_ml
       m  o- - - -
         /
        /  theta_nm
     n o- - - -
     '''

    Parameters
    ----------
    xy : NP x 2 float array
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x maxNN int array
        Connectivity matrix
    NLNNN : NP x maxNNN int array
        Next-nearest neighbor array; the ith row contains indices for the next nearest neighbors for the ith point.
    KLNNN : NP x maxNNN int array
        Chirality of next-nearest neighbor array, positive for ccw, negative for cw
    PVx : NP x maxNN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        x coordinates of periodic lattice vectors, in array matching NL
    PVy : NP x maxNN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        y coordinates of periodic lattice vectors, in array matching NL
    cwccw : bool
        Compute all NNN angles, for both clockwise around polygons and counterclockwise (denoted by KLNNN >0 and
        KLNNN <0)

    Returns
    -------
    NLNNNangles : NP x max #NNN float array
        difference between NN and NNN bond angles, matching NLNNN.

    """
    if PVx is not None or PVy is not None:
        if len(PVx) > 0 or len(PVy) > 0:
            return periodic_NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx, PVy, cwccw=cwccw)
        else:
            nonperiodic = True
    else:
        nonperiodic = True

    if nonperiodic:
        NLNNNangles = np.zeros(np.shape(NLNNN), dtype=float)
        for pt in range(len(NL)):
            neighbors = NL[pt, np.where(KL[pt])[0]]
            # print 'neighbors = ', neighbors
            theta1 = np.mod(np.arctan2(xy[neighbors, 1] - xy[pt, 1], xy[neighbors, 0] - xy[pt, 0]), 2. * np.pi)
            ind = 0
            for nn in neighbors:
                nnn = NL[nn, np.where(KL[nn])[0]]
                # print 'nnn =', nnn
                # print 'NLNNN[pt] = ', NLNNN[pt]
                theta2 = np.mod(np.arctan2(xy[nnn, 1] - xy[nn, 1], xy[nnn, 0] - xy[nn, 0]), 2. * np.pi)
                theta12 = theta1[ind] - theta2
                # if buildNLNNN:
                #     start = np.where(KLNNN[pt] == 0)[0][0]
                #     end = start + len(nnn)
                #     NLNNN[pt, start:end] = nnn
                #     KLNNN[pt, start:end] =
                jjnnn = 0
                # Only look at counterclockwise neighbors
                for npt in nnn:
                    col = np.where(np.logical_and(NLNNN[pt] == npt, np.abs(KLNNN[pt]) > 0))[0]
                    if KLNNN[pt, col] > 0:
                        # print 'col =', col
                        NLNNNangles[pt, col] = theta12[jjnnn] % (2. * np.pi)
                    if cwccw and KLNNN[pt, col] < 0:
                        NLNNNangles[pt, col] = theta12[jjnnn] % (2. * np.pi)
                    jjnnn += 1
                ind += 1
        # NLNNNangles[NLNNNangles > np.pi] = - (2. * np.pi - NLNNNangles[NLNNNangles > np.pi])
        return NLNNNangles


def periodic_NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx, PVy, pvxnnn=None, pvynnn=None, cwccw=False):
    """Compute he difference between NN and NNN bond angles of each particle, theta_{nml}, as shown below, for periodic
    networks. This function is used exclusively in NNN_bond_angles()
     '''
       l o
          \
          \ theta_ml
       m  o- - - -
         /
        /  theta_nm
     n o- - - -
     '''

    Parameters
    ----------
    xy : NP x 2 float array
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x maxNN int array
        Connectivity matrix
    NLNNN : NP x maxNNN int array
        Next-nearest neighbor array; the ith row contains indices for the next nearest neighbors for the ith point.
    KLNNN : NP x maxNNN int array
        Chirality of next-nearest neighbor array, positive for ccw, negative for cw
    PVx : NP x maxNN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        x coordinates of periodic lattice vectors, in array matching NL
    PVy : NP x maxNN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        y coordinates of periodic lattice vectors, in array matching NL
    cwccw : bool
        Compute all NNN angles, for both clockwise around polygons and counterclockwise (denoted by KLNNN >0 and
        KLNNN <0)

    Returns
    -------
    NLNNNangles : NP x max #NNN float array
        difference between NN and NNN bond angles, matching NLNNN
    """
    # test code
    # from lepm import lattice_class
    # import lepm.lattice_elasticity as le
    # lp = {'LatticeTop': 'hexagonal', 'shape': 'hexagon', 'NH': 2, 'NV': 2, 'phi_lattice': '0p000',
    #       'delta_lattice': '0p667', 'periodicBC': True, 'NP_load': 2,
    #       'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'}
    # lat = lattice_class.Lattice(lp=lp)
    # lat.load()
    # lat.get_NNN_info(attribute=True)
    # xy = lat.xy; NL = lat.NL; KL = lat.KL; NLNNN = lat.NLNNN; KLNNN = lat.KLNNN; PVx = lat.PVx; PVy = lat.PVy
    # NLNNNangles = le.periodic_NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx, PVy, cwccw=True)
    # print NLNNNangles

    NLNNNangles = np.zeros(np.shape(NLNNN), dtype=float)
    absKL = np.abs(KL)
    for pt in range(len(NL)):
        neighbors = NL[pt, np.where(absKL[pt])[0]]
        # print 'neighbors = ', neighbors
        theta1 = np.mod(np.arctan2(xy[neighbors, 1] + PVy[pt, np.where(absKL[pt])[0]] - xy[pt, 1],
                                   xy[neighbors, 0] + PVx[pt, np.where(absKL[pt])[0]] - xy[pt, 0]), 2. * np.pi)
        ind = 0
        for nn in neighbors:
            nnn = NL[nn, np.where(absKL[nn])[0]]
            # print 'nnn =', nnn
            # print 'NLNNN[pt] = ', NLNNN[pt]
            theta2 = np.mod(np.arctan2(xy[nnn, 1] + PVy[nn, np.where(absKL[nn])[0]] - xy[nn, 1],
                                       xy[nnn, 0] + PVx[nn, np.where(absKL[nn])[0]] - xy[nn, 0]), 2. * np.pi)
            theta12 = theta1[ind] - theta2
            # if buildNLNNN:
            #     start = np.where(KLNNN[pt] == 0)[0][0]
            #     end = start + len(nnn)
            #     NLNNN[pt, start:end] = nnn
            #     KLNNN[pt, start:end] =
            jjnnn = 0
            # Only look at counterclockwise neighbors
            for npt in nnn:
                col = np.where(np.logical_and(NLNNN[pt] == npt, np.abs(KLNNN[pt]) > 0))[0]
                if KLNNN[pt, col] > 0:
                    # print 'col =', col
                    NLNNNangles[pt, col] = theta12[jjnnn] % (2. * np.pi)
                if cwccw and KLNNN[pt, col] < 0:
                    NLNNNangles[pt, col] = theta12[jjnnn] % (2. * np.pi)
                jjnnn += 1
            ind += 1
            # NLNNNangles[NLNNNangles > np.pi] = - (2. * np.pi - NLNNNangles[NLNNNangles > np.pi])
    return NLNNNangles


def calc_nljnnn(lat):
    """Create #pts x max #NNN int array of neighbors through which particle i is a NNN with particle j.
    For the output, nljnnn, row i gives the indices of the neighbors of i, with placement matching the NNN for that
    neighbor in NLNNN. kljnnn gives whether the bond from i to j is periodic, klknnn gives whether the bond from j to k
    is periodic.

    Parameters
    ----------
    lat : lepm.lattice_class.LatticeClass instance
        lattice class instance

    Returns
    -------
    nljnnn : #pts x max(#NNN) int array
        nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
        the next nearest neighbor of i through the particle nljnnn[i, j]
    kljnnn : #pts x max(#NNN) int array
        bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
    klknnn : #pts x max(#NNN) int array
        bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
    """
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    if 'ignore_tris' not in lat.lp:
        lat.lp['ignore_tris'] = False
        print 'lattice_functions_nnn: ignore_tris not in lat.lp'
        raise RuntimeError('lattice_functions_nnn: ignore_tris not in lat.lp. Please include it')
    else:
        print 'lattice_functions_nnn: ignore_tris in lat.lp, using that to decide'

    ignore_tris = lat.lp['ignore_tris']
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=True)

    # preallocate
    nljnnn = np.zeros((len(xy), 2 * np.shape(NL)[1]), dtype=int)
    kljnnn = np.zeros((len(xy), 2 * np.shape(NL)[1]), dtype=int)
    klknnn = np.zeros((len(xy), 2 * np.shape(NL)[1]), dtype=int)

    # For network, obtain polygons
    polygons = lat.get_polygons(save_if_missing=False)
    for jj in range(len(polygons)):
        # Reverse the orientation of the polygon to match sign convention of NLNNN
        pp = polygons[jj][:-1]
        # print 'lattice_functions_nnn: pp = ', pp
        # print 'lattice_functions_nnn: ignore_tris = ', ignore_tris
        # raise RuntimeError('exiting here')
        if not ignore_tris or len(pp) > 3:
            # either polygon has more than 3 sides or we are including all polygons in nnn determination
            for i in range(len(pp)):
                ppi = pp[i]

                # Clockwise NNN particle
                # NN for particle i is ppk
                ppj = pp[np.mod(i + 1, len(pp))]
                # NNN for particle i is ppk
                ppk = pp[np.mod(i + 2, len(pp))]

                # This doesn't allow a site to be its own NNN multiple times
                # jj = np.where(NLNNN[ppi] == ppk)[0][0]
                # klj = np.where(NL[ppi] == ppj)[0][0]
                # klk = np.where(NL[ppj] == ppk)[0][0]
                # nljnnn[ppi, jj] = NL[ppi, klj]
                # kljnnn[ppi, jj] = KL[ppi, klj]
                # klknnn[ppi, jj] = KL[ppj, klk]

                # Here, we allow for a site to be its own NNN multiple times
                # print 'NLNNN = ', NLNNN
                # print 'KLNNN = ', KLNNN
                # print 'lattice_functions_nnn: ppi = ', ppi
                # print 'lattice_functions_nnn: pp = ', pp
                # print 'lattice_functions_nnn: NLNNN[ppi] = ', NLNNN[ppi]
                # print 'lattice_functions_nnn: KLNNN[ppi] = ', KLNNN[ppi]
                # print 'lattice_functions_nnn: ppk = ', ppk
                # jj are the indices of NLNNN with specified NNN
                # klj are the indices of NL[i] with the neighbor j
                # klk are the indices of NL[j] with the neighbor k
                jj = np.where(NLNNN[ppi, np.where(np.abs(KLNNN[ppi]))[0]] == ppk)[0]
                klj = np.where(NL[ppi, np.where(np.abs(KL[ppi]))[0]] == ppj)[0]
                klk = np.where(NL[ppj, np.where(np.abs(KL[ppi]))[0]] == ppk)[0]
                # NOTE: May need to adjust jj, klj, and klk to accomodate for the case where KLNNN has zeros before
                # nonzero elements here.
                # todo: make adjustment

                # print 'jj, klj, klk = ', jj, klj, klk

                # Handle the case where there is more than one route to get from site to NNN
                if len(jj) > 1:
                    if len(jj) == 6 and len(klj) == 3 and len(klk) == 3 and \
                                    lat.lp['LatticeTop'] == 'hexagonal' and lat.lp['periodicBC']:
                        # Particle ppi is a NNN of ppk through two periodic bonds (ie a 2-particle unit cell)
                        # Pattern the hoppings counterclockwise like this:
                        # Hoppings for particle #0 (note that clockwise hopping gets KLNNN[i,j] = -1)
                        #       o 2       o 1
                        #        \       /
                        #         \     /
                        #            o #1
                        #            |
                        #            |
                        #  o 3       o  #0    o 0
                        #    \    /     \    /
                        #     \  /       \  /
                        #       o         o
                        #       |         |
                        #       |         |
                        #     4 o         o 5
                        #
                        # Hoppings for particle #1 (note that clockwise hopping now gets KLNNN[i,j] = 1)
                        #
                        #        o 5     o 4
                        #        |       |
                        #        |       |
                        #        o       o
                        #       / \     / \
                        #      /   \   /   \
                        #  0  o      o #1   o 3
                        #            |
                        #            |
                        #            o  #0
                        #          /   \
                        #         /     \
                        #     1  o       o 2
                        #
                        # klj2 = np.repeat(klj, 2)
                        # nljnnn[ppi, jj] = NL[ppi, klj2]
                        klj0 = np.array([-1, 1, 1, -1, -1, -1])
                        klj1 = np.array([-1, 1, 1, -1, -1, -1])
                        klk0 = np.array([-1, -1, -1, -1, 1, 1])
                        klk1 = np.array([-1, -1, -1, -1, 1, 1])
                        nljnnn = np.dstack((np.ones(6), np.zeros(6)))[0].transpose()
                        kljnnn = np.dstack((klj0, klj1))[0].transpose()
                        klknnn = np.dstack((klk0, klk1))[0].transpose()
                    elif lat.lp['LatticeTop'] == 'hexjunction2triads':
                        # print 'latfnsnnn: jj = ', jj
                        # print 'latfnsnnn: klj = ', klj
                        # print 'latfnsnnn: klk = ', klk
                        # Particle ppi is a NNN of ppk through two periodic bonds (ie a 2-particle unit cell)
                        # Pattern the hoppings counterclockwise like this:
                        # Hoppings for particle #0 (note that clockwise hopping gets KLNNN[i,j] = -1)
                        #
                        #      |           |
                        #      |           |
                        #      0           0
                        #     1-2         1-2
                        #    /   \       /   \
                        #   /     \     /     \
                        #          5 - 4
                        #            3
                        #            |
                        #            |
                        #            0
                        #          1 - 2
                        #    \    /     \     /
                        #     \  /       \   /
                        #      5-4        5-4
                        #       3          3
                        #       |          |
                        #       |          |
                        #
                        # klj2 = np.repeat(klj, 2)
                        # nljnnn[ppi, jj] = NL[ppi, klj2]
                        if lat.lp['ignore_tris']:
                            nljnnn = np.array([[3, 1, 2, 3, 0, 0],
                                               [0, 4, 4, 2, 0, 0],
                                               [1, 5, 5, 0, 0, 0],
                                               [5, 0, 0, 4, 0, 0],
                                               [1, 5, 3, 1, 0, 0],
                                               [2, 3, 4, 2, 0, 0]])
                            kljnnn = np.array([[1,  1,  1, 1, 0, 0],
                                               [1, -1, -1, 1, 0, 0],
                                               [1, -1, -1, 1, 0, 0],
                                               [1,  1,  1, 1, 0, 0],
                                               [-1, 1,  1, -1, 0, 0],
                                               [-1, 1,  1, -1, 0, 0]])
                            klknnn = np.array([[1, -1, 1, 1, 0, 0],
                                               [1, 1, 1, -1, 0, 0],
                                               [-1, 1, 1, 1, 0, 0],
                                               [-1, 1, 1, -1, 0, 0],
                                               [1, -1, 1, 1, 0, 0],
                                               [1, 1, -1, 1, 0, 0]])

                            # nljnnn[ppi, jj] = NL[ppi, klj]
                            # kljnnn[ppi, jj] = KL[ppi, klj]
                            # klknnn[ppi, jj] = KL[ppj, klk]
    
                            # print 'latfnsnnn: nljnnn = ', nljnnn
                            # print 'latfnsnnn: kljnnn = ', kljnnn
                            # print 'latfnsnnn: klknnn = ', klknnn
                            # print 'latfnsnnn: NLNNN = ', NLNNN
                            # print 'latfnsnnn: KLNNN = ', KLNNN
                            # raise RuntimeError('exiting here')
                        else:
                            print("lat.lp['ignore_tris']", lat.lp['ignore_tris'])
                            raise RuntimeError('Have not coded for hexjunction2triads NNN with triangles included')
                    else:
                        raise RuntimeError("Have not coded for this kind of unit cell / periodic structure yet")
                        # Particle ppi is a NNN of ppk through multiple bonds (periodic)
                else:
                    nljnnn[ppi, jj] = NL[ppi, klj]
                    kljnnn[ppi, jj] = KL[ppi, klj]
                    klknnn[ppi, jj] = KL[ppj, klk]

                # Counterclockwise NNN particle
                # NN for particle i is ppk
                ppj = pp[np.mod(i - 1, len(pp))]
                # NNN for particle i is ppk
                ppk = pp[np.mod(i - 2, len(pp))]

                jj = np.where(NLNNN[ppi] == ppk)[0]
                klj = np.where(NL[ppi] == ppj)[0]
                klk = np.where(NL[ppj] == ppk)[0]

                if len(jj) > 1:
                    # Particle ppi is a NNN of ppk through multiple bonds (periodic)
                    # Already handled this case in full above
                    if len(jj) == 6 and len(klj) == 3 and len(klk) == 3 and \
                                    lat.lp['LatticeTop'] == 'hexagonal' and lat.lp['periodicBC']:
                        pass
                    elif lat.lp['LatticeTop'] == 'hexjunction2triads':
                        pass
                    else:
                        raise RuntimeError("Have not coded for this kind of unit cell / periodic structure yet")
                else:
                    nljnnn[ppi, jj] = NL[ppi, klj]
                    kljnnn[ppi, jj] = KL[ppi, klj]
                    klknnn[ppi, jj] = KL[ppj, klk]

        # print 'NLNNN = ', NLNNN[0:5]
        # print 'KLNNN = ', KLNNN[0:5]
        # print 'nljnnn = ', nljnnn[0:5]
        # print 'kljnnn = ', kljnnn[0:5]
        # print 'klknnn = ', klknnn[0:5]
        # lat.plot_BW_lat(show=False, save=False)
        # for ii in range(len(lat.xy)):
        #     plt.text(lat.xy[ii, 0] + 0.2, lat.xy[ii, 1] + 0.2, str(ii))
        # plt.show()

    # print '-------------'
    # print 'nlnnn = ', nljnnn
    # print 'kljnnn = ', kljnnn
    # print 'klknnn = ', klknnn
    # sys.exit()
    return nljnnn, kljnnn, klknnn


def calc_pvnnn(lat, attribute=False, check=False):
    """create periodic lattice vector arrays (x and y components) for next nearest neighbor couplings

    Parameters
    ----------
    lat : Lattice class
        the lattice for whom to compute the periodic vectors of the next nearest neighbors
    attribute : bool
        attribute the periodic vectors for next nearest neighbors to self

    Returns
    -------
    pvxnnn : NP x max(#NNN) float array
        ijth element of pvxnnn is the x-component of the vector taking NLNNN[i,j] to its image as seen by particle i
    pvynnn : NP x max(#NNN) float array
        ijth element of pvynnn is the y-component of the vector taking NLNNN[i,j] to its image as seen by particle i
    """
    # Get NLNNN, nljnnn, kljnnn, klknnn
    nljnnn, kljnnn, klknnn = lat.get_nljnnn(attribute=True)
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=attribute)
    PVx = lat.PVx
    PVy = lat.PVy
    NL = lat.NL
    KL = lat.KL
    pvxnnn = np.zeros_like(nljnnn, dtype=float)
    pvynnn = np.zeros_like(nljnnn, dtype=float)
    eps = 1e-9

    ################################################################################
    # Check if any rows of NLNNN have repeated elements in a row with nonzero KLNNN
    if np.any(np.diff(np.sort(NLNNN, axis=1), axis=1) == 0):
        # There are some repeats, but they might just be zeros with KLNNN[i,j]=0.
        # Check to see if there are nontrivial repeats.
        searching, kk = True, 0
        while searching:
            # See if this row contains duplicates with nonzero KLNNN[kk]
            row = NLNNN[kk][np.where(np.abs(KLNNN[kk]))[0]]
            if np.any(np.diff(np.sort(row)) == 0):
                searching = False
                trivial = False

            kk += 1

            # Stop if we've finished searching through all of NLNNN
            if kk == np.shape(NLNNN)[0]:
                searching = False
                trivial = True
    else:
        # There are no repeats in NLNNN, so we know there are no particles who are their own NNN ('trivial')
        trivial = True

    if not trivial:
        if lat.lp['LatticeTop'] == 'hexagonal' and len(lat.xy) == 2 and lat.lp['periodicBC']:
            delta = lat.lp['delta']
            phi = lat.lp['phi']
            theta = 0.5 * (np.pi - delta)
            lv = np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
            pvxnnn = np.array([[lv[0, 0],   lv[1, 0], -lv[1, 0], -lv[0, 0], -lv[1, 0],  lv[1, 0]],
                               [-lv[0, 0], -lv[1, 0],  lv[1, 0],  lv[0, 0],  lv[1, 0], -lv[1, 0]]])
            pvynnn = np.array([[lv[0, 1],   lv[1, 1],  lv[1, 1],  lv[0, 1], -lv[1, 1], -lv[1, 1]],
                               [-lv[0, 1], -lv[1, 1], -lv[1, 1],  lv[0, 1],  lv[1, 1],  lv[1, 1]]])
        elif lat.lp['LatticeTop'] == 'hexjunction2triads' and len(lat.xy) == 6 and lat.lp['periodicBC']:
            delta = lat.lp['delta']
            phi = lat.lp['phi']
            theta = 0.5 * (np.pi - delta)
            lv = np.array([[2 * np.cos(theta), 0], [np.cos(theta) + np.sin(phi), np.sin(theta) + np.cos(phi)]])
            # print 'lattice_functions_nnn.calc_pvnnn(): lv = ', lv
            # print 'lattice_functions_nnn.calc_pvnnn(): nljnnn = ', nljnnn
            # print 'lattice_functions_nnn.calc_pvnnn(): NLNNN = ', NLNNN
            # print 'lattice_functions_nnn.calc_pvnnn(): kljnnn = ', kljnnn
            # print 'lattice_functions_nnn.calc_pvnnn(): klknnn = ', klknnn
            mx, my = lv[1, 0], lv[1, 1]
            pvxnnn = np.array([[0, -mx, mx, 0, 0, 0],
                               [0, -mx, -mx, mx, 0, 0],
                               [-mx, mx, mx, 0, 0, 0],
                               [-mx, 0, 0, mx, 0, 0],
                               [mx, -mx, 0, mx, 0, 0],
                               [-mx, 0, mx, -mx, 0, 0]])
            pvynnn = np.array([[0, -my, -my, 0, 0, 0],
                               [0, -my, -my, -my, 0, 0],
                               [-my, -my, -my, 0, 0, 0],
                               [my, 0, 0, my, 0, 0],
                               [my, my, 0, my, 0, 0],
                               [my, 0, my, my, 0, 0]])
            # print 'latfnsnnn: defined pvxnnn for hexjunctions2triads unit cell:'
            # print 'pvxnnn = ', pvxnnn
            # raise RuntimeError('exiting here')
        else:
            raise RuntimeError('Have not considered this unit cell for pvnnn')
    else:
        ################################################################################
        # Prepare dictionaries to handle the cases where a particle is a NNN of another site multiple times
        # They are name to be the j'th and k'th column dictionaries, for NNN bonds i-j-k.
        jcoldict = {}
        kcoldict = {}

        if check:
            print 'NLNNN = ', NLNNN
            print 'KLNNN = ', KLNNN
            print 'nljnnn = ', nljnnn
            print 'kljnnn = ', kljnnn
            print 'klknnn = ', klknnn

        # Consider each particle
        for ii in range(len(NLNNN)):
            # check for a periodic bond from ii to the particles in NLNNN[ii]
            if (kljnnn[ii] < -0.5).any() or (klknnn[ii] < -0.5).any():
                # Consider any real connections in this row
                for jj in np.where(np.abs(KLNNN[ii]) > 0)[0]:
                    # Check if the site to nearest neighbor connection is periodic
                    if kljnnn[ii, jj] < -0.5:
                        # The bond from ii to jj is periodic, so add the periodic vector of PVx[ii, wherejj] to pvnnn's
                        # First get which column of PVx and PVy we are looking at (ie, col)
                        jjind = nljnnn[ii, jj]
                        if nljnnn[ii, jj] == 0:
                            # do a little extra work to avoid grabbing a false connection (bc NL is buffered by zeros)
                            okcols = np.where(KL[ii] < -eps)[0]
                            whichcol = np.where(NL[ii, okcols] == jjind)[0]
                            col = okcols[whichcol]
                        else:
                            col = np.where(np.logical_and(NL[ii] == jjind, KL[ii] < -eps))[0]

                        # Sometimes a site can be a NNN to another multiple times (through periodic bonds)
                        # If col has multiple entries, record and update which ones have been used
                        if len(col) > 1:
                            raise RuntimeError('Edit the algorithm below to match the CHIRALITY of the NNN hoppings. '
                                               'KLNNN tells you if cw/ccw, use that ')
                            # check for alternative connections in dictionary
                            if ii in jcoldict:
                                if jjind in jcoldict[ii]:
                                    jcoldict[ii][jjind] += 1
                                    jcoldict[ii][jjind] %= len(col)
                                else:
                                    jcoldict[ii][jjind] = 0
                            else:
                                jcoldict[ii] = {jjind: 0}

                            col = col[jcoldict[ii][nljnnn[ii, jj]]]

                        pvxnnn[ii, jj] += PVx[ii, col]
                        pvynnn[ii, jj] += PVy[ii, col]
                    # Check if the nearest neighbor to next nearest neighbor connection is periodic
                    if klknnn[ii, jj] < -0.5:
                        # The bond from jj to kk is periodic, so add the periodic vector
                        # of PVx[jjind, wherekk] to pvnnn's.
                        # First get particle index associated with the jj'th column of NNN arrays (col)
                        kk = NLNNN[ii, jj]
                        jjind = nljnnn[ii, jj]
                        if NLNNN[ii, jj] == 0:
                            # do a little extra work to avoid grabbing a false connection
                            # (since NL is buffered by zeros)
                            # okcols = np.where(np.abs(KL[jjind]) > eps)[0]
                            okcols = np.where(KL[jjind] < -eps)[0]
                            whichcol = np.where(NL[jjind, okcols] == kk)[0]
                            col = okcols[whichcol]
                        else:
                            col = np.where(np.logical_and(NL[jjind] == kk, KL[jjind] < -eps))[0]

                        # Sometimes a site can be a NNN to another multiple times (through periodic bonds)
                        # If col has multiple entries, record and update which ones have been used
                        if len(col) > 1:
                            raise RuntimeError('Edit the algorithm below to match the CHIRALITY of the NNN hoppings. '
                                               'KLNNN tells you if cw/ccw, use that ')
                            # check for alternative connections in dictionary
                            if ii in kcoldict:
                                if jjind in kcoldict[ii]:
                                    if kk in kcoldict[ii][jjind]:
                                        kcoldict[ii][jjind][kk] += 1
                                        kcoldict[ii][jjind][kk] %= len(col)
                                    else:
                                        kcoldict[ii][jjind] = {kk: 0}
                                else:
                                    kcoldict[ii][jjind] = {kk: 0}
                            else:
                                kcoldict[ii] = {}
                                kcoldict[ii][jjind] = {kk: 0}

                            # print 'kcoldict = ', kcoldict
                            # print 'kcoldict[ii][jj][kk] = ', kcoldict[ii][jjind][kk]
                            # print 'col = ', col
                            col = col[kcoldict[ii][jjind][kk]]

                        pvxnnn[ii, jj] += PVx[jjind, col]
                        pvynnn[ii, jj] += PVy[jjind, col]

    # check it
    if check:
        print 'pvxnnn = ', pvxnnn
        print 'pvynnn = ', pvynnn
        for kdmy in range(np.shape(pvxnnn)[1]):
            import matplotlib.pyplot as plt
            plt.plot([0, pvxnnn[0, kdmy]], [0, pvynnn[0, kdmy]], 'k-')
            plt.plot([0, pvxnnn[1, kdmy]], [1, 1+pvynnn[1, kdmy]], 'g-')
        plt.axis('equal')
        plt.pause(2)
        # raise RuntimeError('breaking here to debug where check==True')
        # sys.exit()

    if attribute:
        lat.pvxnnn = pvxnnn
        lat.pvynnn = pvynnn
    return pvxnnn, pvynnn


def calc_pv_intnnn(intnn, lat):
    """create periodic lattice vector arrays (x and y components) for ith-order next nearest neighbor couplings

    Parameters
    ----------

    Returns
    -------
    pvxnn : NP x max(#ithNNN) float array
        ijth element of pvxnnn is the x-component of the vector taking NLNNN[i,j] to its image as seen by particle i
    pvynn : NP x max(#ithNNN) float array
        ijth element of pvynnn is the y-component of the vector taking NLNNN[i,j] to its image as seen by particle i
    """
    if intnn == 1:
        return lat.PVx, lat.PVy
    elif intnn > 1:
        # Get NLNNN, nljnnn, kljnnn, klknnn
        pvxnnn, pvynnn = calc_pvnnn(lat, attribute=False)
        if intnn > 2:
            # we need more than just nnn connections
            nljnnn, kljnnn, klknnn = lat.get_nljnnn()
            NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN()
            PVx = lat.PVx
            PVy = lat.PVy
            NL = lat.NL
            KL = lat.KL
            for ii in range(2, intnn + 1):
                # use current
                pvxnn
                pvynn
                raise RuntimeError("Have not finished this")

        return pvxnn, pvynn
    else:
        raise RuntimeError('interaction range of zero is not currently meaningful -> no neighbors?')


def pv_intnn_step(NLnn, KLnn, nljnn, kljnn, klknn, PVx, PVy):
    """Returns pvxinn, pvyinn, the vectors taking each particle to its image as seen
    by its ith-nearest-neighbors

    Parameters
    ----------
    NLnn : the neighbor list connecting sites to nth nearest neighbors
    KLnn
    pvxnn
    pvynn

    Returns
    -------

    """
    # Prallocate the arrays which take particle i its image as seen by j
    pvxnn = np.zeros_like(nljnn, dtype=float)
    pvynn = np.zeros_like(nljnn, dtype=float)

    # Consider each particle
    for ii in range(len(NLnn)):
        # check for a periodic bond from ii to the particles in NLNNN[ii]
        if (kljnn[ii] < -0.5).any() or (klknn[ii] < -0.5).any():
            # Consider any real connections in this row
            for jj in np.where(np.abs(KLnn[ii]) > 0)[0]:
                if kljnn[ii, jj] < -0.5:
                    # The bond from ii to jj is periodic, so add the periodic vector of PVx[ii, wherejj] to pvnnn's
                    # First get which column of PVx and PVy we are looking at (ie, col)
                    if nljnn[ii, jj] == 0:
                        # do a little extra work to avoid grabbing a false connection (since NL is buffered by zeros)
                        okcols = np.where(np.abs(KL[ii]) > 0.5)[0]
                        whichcol = np.where(NL[ii, okcols] == nljnnn[ii, jj])[0]
                        col = okcols[whichcol]
                    else:
                        col = np.where(NL[ii] == nljnn[ii, jj])[0]
                    pvxnn[ii, jj] += PVx[ii, col]
                    pvynn[ii, jj] += PVy[ii, col]
                if klknnn[ii, jj] < -0.5:
                    # The bond from jj to kk is periodic, so add the periodic vector of PVx[jjind, wherekk] to pvnnn's
                    # First get particle index associated with the jj'th column of NNN arrays (col)
                    kk = NLnn[ii, jj]
                    jjind = nljnn[ii, jj]
                    if NLnn[ii, jj] == 0:
                        # do a little extra work to avoid grabbing a false connection (since NL is buffered by zeros)
                        okcols = np.where(np.abs(KL[jjind]) > 0.5)[0]
                        whichcol = np.where(NL[jjind, okcols] == kk)[0]
                        col = okcols[whichcol]
                    else:
                        col = np.where(NL[jjind] == kk)[0]
                    pvxnn[ii, jj] += PVx[jjind, col]
                    pvynn[ii, jj] += PVy[jjind, col]

    # if attribute:
    #     lat.pvxnnn = pvxnnn
    #     lat.pvynnn = pvynnn

    return pvxnn, pvynn


def calc_intnnn_info(lat, intrange, attribute=True):
    """
    Create #pts x max #NNN int arrays of neighbors through which particle i is an ii-th NNN with particle j.
    For the output, nljnnn, row i gives the indices of the neighbors of i, with placement matching the NNN for that
    neighbor in NLNNN. kljnnn gives whether the bond from i to j is periodic, klknnn gives whether the bond from j to k
    is periodic.

    Parameters
    ----------
    lat : lepm.lattice_class.LatticeClass instance
        lattice class instance
    intrange : int
        The range of interaction in units of nearest neighbor couplings. For ex, if intrange==3, computes
        NLNNN and NLNNNN as well as nljnnn and nljnnnn. The latter two are useful for determining periodic vectors
        connecting the site to its NNN and NNNN.
    attribute : bool
        attribute pvnnn, NLNNN, KLNNN to Lattice instance

    Returns
    -------
    nljnns : list of (#pts x max(#NNN) int) arrays
        each element is a nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such
        that NLNNN[i, j] is the next nearest neighbor of i through the particle nljnnn[i, j]
    kljnns : list of (#pts x max(#NNN) int) arrays
        each element is a bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes
        the bond type (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
    klknns : list of (#pts x max(#NNN) int) arrays
        each element is a bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes
        the bond type (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
    pvxnns :
    pvynns :
    """
    xy = lat.xy
    NL, KL = lat.NL, lat.KL
    PVx, PVy = lat.PVx, lat.PVy
    if PVx is None:
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)

    # start off with nnn
    # NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=attribute)
    # pvxnn, pvynn = lat.get_pvnnn(attribute=attribute)
    # NLnns, KLnns = [NL, NLNNN], [KL, KLNNN]
    # NLnn, KLnn = NLNNN, KLNNN
    # pvxnns = [PVx, pvxnn]
    # pvynns = [PVy, pvynn]
    NLnns, KLnns = [NL], [KL]
    NLnn, KLnn = NL, KL
    pvxnns, pvynns = [PVx], [PVy]
    pvxnn, pvynn = PVx, PVy

    # name the nnn bonds as current distant neighbor, to be updated to more
    # distant neighbors if intrange > 1
    # nljnns, kljnns, klknns = [nljnnn], [kljnnn], [klknnn]

    for step in range(1, intrange):
        # take next step outwards
        # get element for distant neighbors NLNNs, KLNNs
        # (previous neighbors are allowed to be more distant neighbors here)
        NLnn, KLnn, pvxnn, pvynn = calc_intnn_step(xy, NL, KL, PVx, PVy, NLnn, KLnn, pvxnn, pvynn,
                                                   check=True)
        # nljnn, kljnn, klknn = calc_nljnn_step(NLnn, KLnn, NL, KL)
        # pvxnn, pvynn = pv_intnn_step(NLnn, KLnn, nljnn, kljnn, klknn)
        # nljnns.append(nljnn)
        # kljnns.append(kljnn)
        # klknns.append(klknn)
        NLnns.append(NLnn)
        KLnns.append(KLnn)
        pvxnns.append(pvxnn)
        pvynns.append(pvynn)

    if attribute:
        for intii in range(intrange):
            if intii not in lat.intnn_info:
                lat.intnn_info[int(intrange)] = {'NLnn': NLnns[intii], 'KLnn': KLnns[intii],
                                                 'pvxnn': pvxnns[intii], 'pvynn': pvynns[intii]}

    return NLnns, KLnns, pvxnns, pvynns


def calc_intnn_step(xy, NL, KL, PVx, PVy, NLnn, KLnn, pvxnn, pvynn, check=False):
    """Return the next iteration of neighbors for a network (xy, BL, NL, KL) after the current
    iteration (NLnn, KLnn). All next-iteration neighbors are returned, without regard to exclusion.

    Parameters
    ----------
    xy
    BL
    NL
    KL
    NLnn
    KLnn
    NLexcl
    KLexcl
    PVx
    PVy

    Returns
    -------

    """
    eps = 1e-5
    # preallocate
    KLnn_new = np.zeros((len(xy), np.shape(NLnn)[1] + np.shape(NLnn)[1] * len(xy)), dtype=int)
    NLnn_new = np.zeros((len(xy), np.shape(NLnn)[1] + np.shape(NLnn)[1] * len(xy)), dtype=int)
    pvxnn_new = np.zeros((len(xy), np.shape(NLnn)[1] + np.shape(NLnn)[1] * len(xy)), dtype=float)
    pvynn_new = np.zeros((len(xy), np.shape(NLnn)[1] + np.shape(NLnn)[1] * len(xy)), dtype=float)
    #
    maxlen = 0
    for kk in range(len(xy)):
        # look at ith NN of pt, stored in NLnn
        # nn are the neighbors of kk --> we will look for neighbors of its neighbors
        bonds = np.where(np.abs(KLnn[kk]) > 0)[0]
        nn = NLnn[kk][bonds]
        pvxmaster = pvxnn[kk][bonds]
        pvymaster = pvynn[kk][bonds]

        # remove original particle from neighbor list only if PVx and PVy as seen
        # by kk are zero for this neighbor particle (which is kk itself)
        if kk in nn:
            # print 'removing original particle if it exists without nonzero pv'
            # check where nn==kk
            cutout = []
            inds = np.where(nn == kk)[0]
            # print 'inds = ', inds
            for ind in inds:
                if abs(pvxmaster[ind]) < eps and abs(pvymaster[ind]) < eps:
                    print 'pvxmaster[', ind, '] = ', pvxmaster[ind]
                    print 'pvymaster[', ind, '] = ', pvymaster[ind]
                    cutout.append(ind)

            # print 'cutout = ', cutout
            keep = np.setdiff1d(np.arange(len(nn)), np.array(cutout))
            nn = nn[keep]
            pvxmaster = pvxmaster[keep]
            pvymaster = pvymaster[keep]

        # look at the neighbors of these neighbors
        nexts, kexts, pvx_nexts, pvy_nexts = [], [], [], []
        # print 'nn = ', nn
        # cycle through the neighbors of pt kk
        for jj in range(len(nn)):
            nei = nn[jj]
            # what are the neighbors of this neighbor
            nonzero = np.where(np.abs(KL[nei]) > eps)[0]
            toadd = NL[nei][nonzero]
            # print 'latfnsnnn: toadd = ', toadd
            ktoadd = KL[nei][nonzero]
            pvx_add = PVx[nei][nonzero]
            pvy_add = PVy[nei][nonzero]
            # print 'toadd = ', toadd
            for ii in range(len(toadd)):
                # do not add particle if this is the original particle
                # (ie the original position ii cannot be ii's ithNNN)
                pvxtmp = pvx_add[ii] + pvxmaster[jj]
                pvytmp = pvy_add[ii] + pvymaster[jj]
                if not (toadd[ii] == kk and abs(pvxtmp) < eps and abs(pvytmp) < eps):
                    # avoid repeating a next-nearest neighbor for current particle
                    # ONLY IF the PVx and PVy are the same
                    nii = toadd[ii]
                    # print 'lfnsnnn:     nexts = ', nexts
                    if nii in nexts:
                        # print 'lfnsnnn:     kk=', kk, ': nii in nexts = ', nii
                        # if the pvx and pvy dont match, we'll add nii, ktoadd[ii], pvx_add[ii} + pvxmaster[jj], pvy...
                        # print 'lfnsnnn:     pvtmp = ', (pvxtmp, pvytmp)
                        # check if pvx and pvy are the same as for previous entry
                        inds = np.where(np.array(nexts) == nii)[0]
                        # print 'inds = ', inds
                        already_exists = False
                        for ind in inds:
                            matchpvx = abs(pvx_nexts[ind] - pvxtmp) < eps
                            matchpvy = abs(pvy_nexts[ind] - pvytmp) < eps
                            # print 'lfnsnnn:     current = ', (kexts[ind], pvx_nexts[ind], pvy_nexts[ind])
                            # print 'lfnsnnn:     kexts[ind] != ktoadd[ii]'
                            if matchpvx and matchpvy:
                                already_exists = True

                        if not already_exists:
                            # print 'lfnsnnn:         adding!\n-----------------------------'
                            nexts.append(nii)
                            pvx_nexts.append(pvxtmp)
                            pvy_nexts.append(pvytmp)
                            if abs(pvxtmp) < eps and abs(pvytmp) < eps:
                                # at least one component of the bond is periodic
                                kexts.append(1)
                            else:
                                # the bond is not periodic
                                kexts.append(-1)
                        # else:
                        #     print 'lfnsnnn:         NOT ADDED\n-----------------------------'

                    else:
                        nexts.append(nii)
                        pvx_nexts.append(pvxtmp)
                        pvy_nexts.append(pvytmp)
                        if abs(pvxtmp) < eps and abs(pvytmp) < eps:
                            # at least one component of the bond is periodic
                            kexts.append(1)
                        else:
                            # the bond is not periodic
                            kexts.append(-1)

        # print 'lfnsnnn: nexts = ', nexts
        # build this row
        nexts = np.array(nexts)
        NLnn_new[kk][0:len(nexts)] = nexts
        KLnn_new[kk][0:len(nexts)] = kexts
        pvxnn_new[kk][0:len(nexts)] = pvx_nexts
        pvynn_new[kk][0:len(nexts)] = pvy_nexts
        maxlen += max(len(nexts), maxlen)

    # Truncate the arrays based on maximum length
    NLnn_new = NLnn_new[:, 0:maxlen]
    KLnn_new = KLnn_new[:, 0:maxlen]
    pvxnn_new = pvxnn_new[:, 0:maxlen]
    pvynn_new = pvynn_new[:, 0:maxlen]

    # print 'This step here <----------->'
    # print 'NLnn = '
    # print NLnn_new
    # print 'KLnn = '
    # print KLnn_new
    # print 'pvxnn = '
    # print pvxnn_new
    # print 'pvynn = '
    # print pvynn_new

    return NLnn_new, KLnn_new, pvxnn_new, pvynn_new


def combine_intnnn_info(NLnns, KLnns, pvxnns, pvynns):
    """Join together the interactions with i-th NNN, where i runs over some range (0, ... k). This is for having a
    network with medium-range (or long but not infinite range) interactions

    Parameters
    ----------
    NLnns : list
    KLnns : list
    pvxnns : list
    pvynns : list

    Returns
    -------
    NLnn, KLnn, PVxnn, PVynn
    """
    eps = 1e-5
    maxwidth = 0
    for nlnntmp in NLnns:
        maxwidth += np.shape(nlnntmp)[1]
    # print 'maxwidth = ', maxwidth

    NLnnc = np.zeros((len(NLnns[0]), maxwidth), dtype=int)
    KLnnc = np.zeros((len(NLnns[0]), maxwidth), dtype=int)
    PVxc = np.zeros((len(NLnns[0]), maxwidth), dtype=float)
    PVyc = np.zeros((len(NLnns[0]), maxwidth), dtype=float)
    kk = 0
    for nl in NLnns:
        if kk == 0:
            cols = np.arange(np.shape(KLnns[kk])[1])
            NLnnc[:, cols] = nl
            KLnnc[:, cols] = KLnns[kk]
            PVxc[:, cols] = pvxnns[kk]
            PVyc[:, cols] = pvynns[kk]
        else:
            kl = KLnns[kk]
            pvx, pvy = pvxnns[kk], pvynns[kk]

            for jj in range(np.shape(nl)[0]):
                # Get where current lists are zero in first dimension (not zeroth)
                filled = np.where(np.abs(KLnnc[jj]) > eps)[0]
                first = filled[-1] + 1
                # Add ith n.neighbors to neighbor list if not already in this NLnn row
                # To do so, get only the elements that are new to this row, so
                # use mask to filter out repeats. Note that pvx,pvy, and nl all have to match
                repeats = np.where(np.in1d(nl[jj], NLnnc[jj, filled]))[0]
                # check if these repeats have the same pvx and pvy or not. If not, add them.
                matches = []
                for repeat in repeats:
                    # get where the current NLnn already has this neighbor
                    existind = np.where(NLnnc[jj, filled] == nl[jj, repeat])[0]
                    pvxind = np.where(PVxc[jj, filled] == pvx[jj, repeat])[0]
                    pvyind = np.where(PVyc[jj, filled] == pvy[jj, repeat])[0]
                    match = np.intersect1d(np.intersect1d(existind, pvxind), pvyind)
                    if len(match) > 0:
                        matches.append(repeat)

                # exclude all match indices from the addition
                keep = np.setdiff1d(np.arange(len(nl[jj])), np.array(matches))
                # print 'keep = ', keep
                # print 'NLnn[jj] = ', NLnnc[jj]
                # print 'NLnn[jj, first:first + len(keep)]= ', NLnnc[jj, first:first + len(keep)]
                NLnnc[jj, first:first + len(keep)] = nl[jj][keep]
                KLnnc[jj, first:first + len(keep)] = kl[jj][keep]
                PVxc[jj, first:first + len(keep)] = pvx[jj][keep]
                PVyc[jj, first:first + len(keep)] = pvy[jj][keep]

        kk += 1

    # Truncate the arrays in the first dimension (the width)
    totalnum = 0
    for jj in range(np.shape(nl)[0]):
        # How many connections are there in this row?
        rownum = len(np.where(np.abs(KLnnc[jj, :]) > eps)[0])
        # is this the largest number of connections so far?
        totalnum = max(totalnum, rownum)

    NLnn = NLnnc[:, 0:totalnum]
    KLnn = KLnnc[:, 0:totalnum]
    PVx = PVxc[:, 0:totalnum]
    PVy = PVyc[:, 0:totalnum]

    return NLnn, KLnn, PVx, PVy


def calc_combined_intnn_info(lat, interaction_range):
    """Get NLnn, KLnn, PVx, PVy for ii-th nearest neighbor interactions, allowing for periodic
    interaction

    Returns
    -------
    interaction_range : int
    lat : Lattice class

    """
    NLnns, KLnns, pvxnns, pvynns = calc_intnnn_info(lat, interaction_range)
    NLnn, KLnn, PVx, PVy = combine_intnnn_info(NLnns, KLnns, pvxnns, pvynns)
    return NLnn, KLnn, PVx, PVy


def infinite_range_network_info(lat):
    """Get NL, KL, PVx, and PVy for a network where all particles interact with all others.

    Parameters
    ----------
    lat : Lattice class

    Returns
    -------

    """
    nl, kl = lat.NL, lat.KL
    NL = np.array([np.setdiff1d(np.arange(len(lat.xy)), np.array([ii]))
                   for ii in range(len(lat.xy))])
    KL = np.ones_like(NL, dtype=int)
    # Build periodic vectors from existing PVx, PVy
    pvx, pvy = lat.PVx, lat.PVy
    # Right now assumes square PVs
    # Todo: fix THIS!

    # if lat.PV is not None:
    #     if abs(lat.PV[0, 1]) < 1e-9 and abs(lat.PV[1, 0]) < 1e-9:
    #         LL = (lat.PV[0, 0], lat.PV[1, 1])
    #     else:
    #         raise RuntimeError("need to generalize this!")
    #     dx_finite = le.distance_periodic(lat.xy, lat.xy, LL, dim=0)
    #     dy_finite = le.distance_periodic(lat.xy, lat.xy, LL, dim=1)
    #     dx_periodic = dh.dist_pts_periodic(lat.xy, lat.xy, lat.PV, dim=0, square_norm=False)
    #     dy_periodic = dh.dist_pts_periodic(lat.xy, lat.xy, lat.PV, dim=1, square_norm=False)
    #     # get rid of the diagonals of the distance matrix
    #     npts = len(lat.xy)
    #     dxf = np.zeros((npts, npts - 1))
    #     dyf = np.zeros((npts, npts - 1))
    #     dxp = np.zeros((npts, npts - 1))
    #     dyp = np.zeros((npts, npts - 1))
    #     for kk in range(npts):
    #         dxf[kk, :] = np.array([dx_finite[ii] for ii in np.setdiff1d(np.arange(npts), np.array([kk]))])
    #         dyf[kk, :] = np.array([dy_finite[ii] for ii in np.setdiff1d(np.arange(npts), np.array([kk]))])
    #         print '... = ', np.array([dx_periodic[ii] for ii in np.setdiff1d(np.arange(npts), np.array([kk]))])
    #         print 'dxp = ', dxp
    #         dxp[kk, :] = np.array([dx_periodic[ii] for ii in np.setdiff1d(np.arange(npts), np.array([kk]))])
    #         dyp[kk, :] = np.array([dy_periodic[ii] for ii in np.setdiff1d(np.arange(npts), np.array([kk]))])
    #
    #     # periodic vectors are where
    #     PVx = dxp - dxf
    #     PVy = dyp - dyf
    # else:
    #     PVx = np.zeros_like(NL, dtype=float)
    #     PVy = np.zeros_like(NL, dtype=float)
    #
    #     # Check it
    #     print 'PVx = ', PVx
    #     from lepm.plotting.network_visualization import movie_plot_2D
    #     BL = le.NL2BL(NL, KL)
    #     movie_plot_2D(lat.xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, show=True, close=True)
    #     raise RuntimeError('Should check that PVx and PVy are correct here')

    PVx, PVy = pvx, pvy
    return NL, KL, PVx, PVy


