import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.plotting.haldane_lattice_plotting_functions as hlatpfns
import scipy.linalg as la
import scipy
import matplotlib.pyplot as plt
import sys
import subprocess
import scipy.stats as scistat
try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob

"""Supporting functions for gyro_lattice_class.py
"""


def nnn_hopping_description(hlat):
    """Return string that describes the NNN hopping term"""
    eps = 1e-9
    if hlat.lp['t2angles']:
        outstr = r'$t_2 = \sin(2\theta_{nml})$' + '{0:0.2f}'.format(hlat.lp['t2'] / hlat.lp['t1']) + r'$t_1$'
    elif np.abs(hlat.lp['t2a']) > eps:
        outstr = r'$t_2 = ($' + '{0:0.2f}'.format(hlat.lp['t2'] / hlat.lp['t1']) + \
                 '{0:0.2f}'.format(hlat.lp['t2a'] / hlat.lp['t1']) + r'$) t_1$'
    else:
        outstr = r'$t_2 = ($' + '{0:0.2f}'.format(hlat.lp['t2'] / hlat.lp['t1']) + r'$) t_1$'

    if hlat.lp['t1'] != 1.0:
        outstr += r', $t_1 = $' + '{0:0.2f}'.format(hlat.lp['t1'])

    return outstr


def haldane_matrix(lat, t2, t1, t2a=0.0, pin=0., t2angles=False, pureimNNN=False, thetatwist=None, phitwist=None,
                   check=False, ignore_tris=False, sparse=False):
    """Calculates the matrix for finding the normal modes of the system
    Assumes that all bonds are unit length.
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body axis 3.

    Parameters
    ----------
    R : NP x 2 float array
    NL : NP x NN int array
    KL : NP x NN int array
    t2 : float
        magnitude of NNN coupling
    t1 : float or NP x NN float array
        magnitude of NN coupling (float if constant for all bonds, matrix if varies between bonds)
    t2a : float
        real component to next nearest neighbor hopping, if t2angles is False
    pin : float or float array
        The pinning for each site
    t2angles : bool
        Use geometry to determine NNN coupling strength (sin(theta_nml))
    pureimNNN : bool
        Use geometry to determine NNN coupling strength (sin(theta_nml))
    thetatwist : float
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of pi radians
    phitwist : float
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of pi radians
    check : bool
        View intermediate results
    ignore_tris : bool
        Ignore NNN hoppings to sites which are also NN --> this is done by ignoring triangular polygons in the t2
        component of the dynamical matrix

    Returns
    ----------
    KLNNN : NP x NNNN int array
        connectivity of Next Nearest Neighbors array
    NLNNN : NP x NNNN int array
        neighbor identification array
    matrix : NP x NP complex array
        haldane dynamical matrix
    """
    print 'Constructing Haldane dynamical matrix...'
    eps = 1e-6

    try:
        NP, NN = np.shape(lat.NL)
    except:
        '''There is only one particle.'''
        NP = 1

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]
    if notwist:
        if sparse:
            M1, M2 = construct_m1m2_sparse(lat, t1, t2, t2a, t2angles, eps, pureimNNN)
        else:
            M1, M2 = construct_m1m2(lat, t1, t2, t2a, t2angles, eps, pureimNNN)
    else:
        if sparse:
            M1, M2 = construct_m1m2_twist_sparse(lat, t1, t2, t2a, t2angles, eps, pureimNNN, thetatwist, phitwist)
        else:
            M1, M2 = construct_m1m2_twist(lat, t1, t2, t2a, t2angles, eps, pureimNNN, thetatwist, phitwist)

    matrix = M1 + M2
    matrix += pin * np.identity(NP)
    if check:
        print 'hlatfns: matrix = ', matrix
        le.plot_complex_matrix(matrix, show=True)

    # output NLNNN and KLNNN as well
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN()

    # le.plot_complex_matrix(matrix, show=True)
    return NLNNN, KLNNN, matrix


def construct_m1m2_sparse(lat, t1, t2, t2a, t2angles, eps, pureimNNN, ignore_tris=False):
    """Create the sparse real NN and complex NNN hopping matrices M1 and M2 from the lattice info.
    If t2angles is False (or None), then uses t2a for the real component of the NNN hoppings.

    Parameters
    ----------
    lat : Lattice class instance
    t1 : float
        nearest neighbor hopping strength
    t2 : float
        imaginary component of next-nearest neighbor hopping strength if t2angles is false, otherwise scaling factor for
        the strength of the NNN hoppings, with phase determined by geometry of the bond angle
    t2a : float
        real component of next-nearest neighbor hopping strength, if t2angles is false
    t2angles : bool
        Use the bond angles to determine the real and imaginary components of the NNN hoppings
    eps : float
        very small value, if KL[i,j] is below this value, then the bond between particles i and j is ignored
    pureimNNN : bool
        If t2angles is True, use only the imaginary component (ie 1j * sin(angle))
    ignore_tris : bool
        Ignore NNN hoppings to sites which are also NN --> this is done by ignoring triangular polygons in the t2
        component of the dynamical matrix

    Returns
    -------
    M1
    M2
    """
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    BL = lat.BL
    PVx = lat.PVx
    PVy = lat.PVy
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0
    if (BL < 0).any() and (PVx is None or PVy is None):
        raise RuntimeError('Must specify PVx and PVy in haldane_matrix() when periodic bonds exist!')

    lat.lp['ignore_tris'] = ignore_tris
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=True)

    M1 = np.zeros((NP, NP), dtype=complex)
    M2 = np.zeros((NP, NP), dtype=complex)

    NNNN = np.shape(KLNNN)[1]

    if t2angles:
        NLNNNangles = le.NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx=PVx, PVy=PVy, cwccw=True)
        dubtheta = calc_dubtheta(NLNNNangles)
        for i in range(NP):
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn]  # psi_j

            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if abs(k) > eps:
                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 * np.sin(dubtheta[i, nn])
                    if not pureimNNN:
                        M2[i, ni] += t2 * np.cos(dubtheta[i, nn])

                        # if k < -eps:
                        #     # There is a true NNN connection, so update dynamical matrix
                        #     # print 'i = ', i, ' ni= ', ni
                        #     # print 'hlatfns: negative: KLNNN[i,nn] = ', k, '  dubtheta = ', np.sin(dubtheta[i, nn])
                        #     M2[i, ni] += t2 * np.cos(dubtheta[i, nn]) + 1j * t2 * np.sin(dubtheta[i, nn])
                        # print 'M2[0,:] = ', M2[:,0]
    else:
        for i in range(NP):
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn]  # psi_j

            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if k > eps:
                    # There is a true counterclockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 + t2a
                if k < -eps:
                    # There is a true clockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += -1j * t2 + t2a

    scipy.sparse.coo_matrix((data, (row, col)), shape=(NP, NP))
    return M1, M2


def construct_m1m2(lat, t1, t2, t2a, t2angles, eps, pureimNNN, ignore_tris=False):
    """Create the real NN and complex NNN hopping matrices M1 and M2 from the lattice info.
    If t2angles is False (or None), then uses t2a for the real component of the NNN hoppings.

    Parameters
    ----------
    lat : Lattice class instance
    t1 : float
        nearest neighbor hopping strength
    t2 : float
        imaginary component of next-nearest neighbor hopping strength if t2angles is false, otherwise scaling factor for
        the strength of the NNN hoppings, with phase determined by geometry of the bond angle
    t2a : float
        real component of next-nearest neighbor hopping strength, if t2angles is false
    t2angles : bool
        Use the bond angles to determine the real and imaginary components of the NNN hoppings
    eps : float
        very small value, if KL[i,j] is below this value, then the bond between particles i and j is ignored
    pureimNNN : bool
        If t2angles is True, use only the imaginary component (ie 1j * sin(angle))
    ignore_tris : bool
        Ignore NNN hoppings to sites which are also NN --> this is done by ignoring triangular polygons in the t2
        component of the dynamical matrix

    Returns
    -------
    M1
    M2
    """
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    BL = lat.BL
    PVx = lat.PVx
    PVy = lat.PVy
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0
    if (BL < 0).any() and (PVx is None or PVy is None):
        raise RuntimeError('Must specify PVx and PVy in haldane_matrix() when periodic bonds exist!')

    lat.lp['ignore_tris'] = ignore_tris
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=True)

    M1 = np.zeros((NP, NP), dtype=complex)
    M2 = np.zeros((NP, NP), dtype=complex)

    NNNN = np.shape(KLNNN)[1]

    if t2angles:
        NLNNNangles = le.NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx=PVx, PVy=PVy, cwccw=True)
        dubtheta = calc_dubtheta(NLNNNangles)
        for i in range(NP):
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn]  # psi_j

            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if abs(k) > eps:
                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 * np.sin(dubtheta[i, nn])
                    if not pureimNNN:
                        M2[i, ni] += t2 * np.cos(dubtheta[i, nn])

                        # if k < -eps:
                        #     # There is a true NNN connection, so update dynamical matrix
                        #     # print 'i = ', i, ' ni= ', ni
                        #     # print 'hlatfns: negative: KLNNN[i,nn] = ', k, '  dubtheta = ', np.sin(dubtheta[i, nn])
                        #     M2[i, ni] += t2 * np.cos(dubtheta[i, nn]) + 1j * t2 * np.sin(dubtheta[i, nn])
                        # print 'M2[0,:] = ', M2[:,0]
    else:
        for i in range(NP):
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn]  # psi_j

            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if k > eps:
                    # There is a true counterclockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 + t2a
                if k < -eps:
                    # There is a true clockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += -1j * t2 + t2a
    return M1, M2


def construct_m1m2_twist(lat, t1, t2, t2a, t2angles, eps, pureimNNN, thetatwist, phitwist, attribute=True,
                         ignore_tris=False):
    """Create the real NN and complex NNN hopping matrices M1 and M2 from the lattice info, with at least one twisted
    boundary conditions. If periodic_strip, has one twisted bc in x dimension.

    Parameters
    ----------
    lat : Lattice class instance
    t1 : float
        nearest neighbor hopping
    t2 : float
        magnitude of NNN hopping if t2a is zero, or magnitude of imaginary component of NNN hopping if t2a is nonzero
    t2a : float
        magnitude of real component of NNN hopping if t2a is nonzero and t2angles is False or None
    t2angles : bool
        Use the bond angles to determine the real and imaginary components of the NNN hoppings
    eps : float
        very small value, if KL[i,j] is below this value, then the bond between particles i and j is ignored
    pureimNNN : bool
        If t2angles is True, use only the imaginary component (ie 1j * sin(angle))
    thetatwist : float
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of pi radians
    phitwist : float
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of pi radians
    ignore_tris : bool
        Ignore NNN hoppings to sites which are also NN --> this is done by ignoring triangular polygons in the t2
        component of the dynamical matrix

    Returns
    -------
    M1
    M2
    """
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    BL = lat.BL
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=attribute)
    PVx = lat.PVx
    PVy = lat.PVy
    PV = lat.get_PV(attribute=attribute)

    # check
    # print 'hlatfns: PV = ', PV
    # lat.plot_BW_lat(close=False)
    # tmp = np.array([np.min(lat.xy[:, 0]), np.min(lat.xy[:, 1])])
    # test = np.vstack((tmp, tmp))
    # test2 = test + PV
    # xx = np.dstack((test[:, 0], test2[:, 0]))[0]
    # yy = np.dstack((test[:, 1], test2[:, 1]))[0]
    # print 'dxx = ', xx[:, 1] - xx[:, 0]
    # print 'dyy = ', yy[:, 1] - yy[:, 0]
    # print 'PVx = ', PVx
    # print 'PVy = ', PVy
    # for ii in range(len(xx)):
    #     plt.plot(xx[ii], yy[ii], 'o-')
    # plt.show()
    # sys.exit()

    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0
    if (BL < 0).any() and (PVx is None or PVy is None):
        raise RuntimeError('Must specify PVx and PVy in haldane_matrix() when periodic bonds exist!')

    nljnnn, kljnnn, klknnn = lat.get_nljnnn(attribute=attribute)

    M1 = np.zeros((NP, NP), dtype=complex)
    M2 = np.zeros((NP, NP), dtype=complex)

    NNNN = np.shape(KLNNN)[1]
    thetatwist *= np.pi
    phitwist *= np.pi

    NLNNNangles = le.NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx=PVx, PVy=PVy, cwccw=True)
    dubtheta = calc_dubtheta(NLNNNangles)
    for i in range(NP):
        for nn in range(NN):
            ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
            k = KL[i, nn]  # true connection?

            # Split cases: k is a real connection, a periodic connection. If abs(k) < eps, not a connection.
            if k > eps:
                # hopping connection
                if isinstance(t1, float):
                    M1[i, ni] += t1
                else:
                    M1[i, ni] += t1[i, nn]
            elif k < eps:
                # periodic connection -- figure out which PV is used
                arrpv = np.array([PVx[i, nn], PVy[i, nn]])
                twistfactor = get_twist_factor(arrpv, PV, thetatwist, phitwist)

                if isinstance(t1, float):
                    M1[i, ni] += t1 * twistfactor
                else:
                    M1[i, ni] += t1[i, nn] * twistfactor

    # Do next nearest neighbor hoppings
    if t2angles:
        for i in range(NP):
            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if abs(k) > eps:
                    # get periodicity of bond ij, then also periodicity of bond jk.
                    if kljnnn[i, nn] < eps or klknnn[i, nn] < eps:
                        # There is at least one periodic bond connecting particle i to its NNN, k.
                        # Make array of periodic vectors for ij and jk
                        # the column of PVx,y taking particle j to its periodic position in relation to particle i
                        jjind = np.where(NL[i] == nljnnn[i, nn])[0]
                        # the column of PVx,y taking particle k to its periodic position in relation to particle j
                        kkind = np.where(NL[nljnnn[i, nn]] == ni)[0]
                        arrpvj = np.array([PVx[i, jjind], PVy[i, jjind]])
                        arrpvk = np.array([PVx[jjind, kkind], PVy[jjind, kkind]])

                    if kljnnn[i, nn] < eps:
                        ijtwist = get_twist_factor(arrpvj, PV, thetatwist, phitwist)
                    else:
                        ijtwist = 1.

                    if klknnn[i, nn] < eps:
                        jktwist = get_twist_factor(arrpvk, PV, thetatwist, phitwist)
                    else:
                        jktwist = 1.

                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 * np.sin(dubtheta[i, nn]) * ijtwist * jktwist
                    if not pureimNNN:
                        M2[i, ni] += t2 * np.cos(dubtheta[i, nn]) * ijtwist * jktwist
    else:
        for i in range(NP):
            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if abs(k) > eps:
                    # get periodicity of bond ij, then also periodicity of bond jk.
                    if kljnnn[i, nn] < eps or klknnn[i, nn] < eps:
                        # There is at least one periodic bond connecting particle i to its NNN, k.
                        njind = nljnnn[i, nn]
                        # Make array of periodic vectors for ij and jk
                        # the column of PVx,y taking particle j to its periodic position in relation to particle i
                        kljind = np.where(NL[i] == njind)[0][0]
                        # the column of PVx,y taking particle k to its periodic position in relation to particle j
                        kkind = np.where(NL[njind] == ni)[0][0]

                        arrpvj = np.array([PVx[i, kljind], PVy[i, kljind]])
                        arrpvk = np.array([PVx[njind, kkind], PVy[njind, kkind]])

                    if kljnnn[i, nn] < eps:
                        ijtwist = get_twist_factor(arrpvj, PV, thetatwist, phitwist)
                    else:
                        ijtwist = 1.

                    if klknnn[i, nn] < eps:
                        # print 'PVx[i, kljind] = ', PVx[i, kljind]
                        # print 'PVy[i, kljind] = ', PVy[i, kljind]
                        # print 'PVx[njind, kkind] = ', PVx[njind, kkind]
                        # print 'PVy[njind, kkind] = ', PVy[njind, kkind]
                        # print 'arrpvj = ', arrpvj
                        # print 'arrpvk = ', arrpvk
                        # print 'kkind = ', kkind
                        # print 'PV = ', PV
                        # print 'arrpvk == PV[0] : ', arrpvk == PV[0]
                        # print 'njind = ', njind
                        # print 'NL[njind] = ', NL[njind]
                        # print 'ni = ', ni
                        # print 'nljnnn[i] = ', nljnnn[i]
                        # print 'nljnnn[i, nn] = ', nljnnn[i, nn]
                        # print 'kljnnn[i] = ', kljnnn[i]
                        # print 'kljnnn[i, nn] = ', kljnnn[i, nn]
                        # print 'klknnn[i] = ', klknnn[i]
                        # print 'klknnn[i, nn] = ', klknnn[i, nn]
                        # lat.plot_BW_lat(close=False, save=False)
                        # for ii in range(len(lat.xy)):
                        #     plt.text(lat.xy[ii, 0] + 0.2, lat.xy[ii, 1] + 0.2, str(ii))
                        # plt.show()
                        jktwist = get_twist_factor(arrpvk, PV, thetatwist, phitwist)
                    else:
                        jktwist = 1.

                if k > eps:
                    # There is a true counterclockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += (1j * t2 + t2a) * ijtwist * jktwist
                if k < -eps:
                    # There is a true clockwise NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += (-1j * t2 + t2a) * ijtwist * jktwist

    return M1, M2


def get_twist_factor(arrpv, PV, thetatwist, phitwist=None):
    """

    Parameters
    ----------
    arrpv : 2 x 1 float array
        The periodic displacement taking particle j to its reflection as seen by particle i. For ex,
        np.array([PVx[i, j], PVy[i, j]]).
    PV : 2 x 1, 1 x 2, or 2 x 2 float array
        The periodic vectors of the lattice for which thetatwist and/or phitwist are positive.
        If there is only one vector (ie if PV is 2 x 1 or 1 x 2 float array), then use thetatwist only. This would be
         the case of a periodic strip, which is periodic in only one dimension. Otherwise use both if periodic in 2D.
    thetatwist : float
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of pi radians
    phitwist : float (or None if periodic_strip)
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of pi radians

    Returns
    -------
    twistfactor
    """
    # If there are two periodic lattice vectors PV (ie shape is (2,2)), then pick out which one
    if np.shape(PV) == (2, 2):
        if phitwist is None:
            raise RuntimeError('Must supply both thetatwist and phitwist for fully periodic sample')

        # Find which periodic vector takes site i to site j
        if (arrpv == PV[0]).all():
            # The first periodic vector is used -- multiply by exp[i * theta]
            twistfactor = np.exp(1j * thetatwist)
        elif (arrpv == PV[1]).all():
            # The second periodic vector is used -- multiply by exp[i * phi]
            twistfactor = np.exp(1j * phitwist)
        elif (arrpv == -PV[0]).all():
            # Crossing the first periodic vector in opposite dir -- multiply by exp[-i * theta]
            twistfactor = np.exp(-1j * thetatwist)
        elif (arrpv == -PV[1]).all():
            # Crossing the second periodic vector in opposite dir -- multiply by exp[-i * phi]
            twistfactor = np.exp(-1j * phitwist)
        elif (arrpv == PV[0] + PV[1]).all():
            # Crossing both periodic vectors -- top right
            twistfactor = np.exp(1j * (thetatwist + phitwist))
        elif (arrpv == PV[0] - PV[1]).all():
            # Crossing both periodic vectors -- bottom right
            twistfactor = np.exp(1j * (thetatwist - phitwist))
        elif (arrpv == -PV[0] + PV[1]).all():
            # Crossing both periodic vectors -- top left
            twistfactor = np.exp(1j * (-thetatwist + phitwist))
        elif (arrpv == -PV[0] - PV[1]).all():
            # Crossing both periodic vectors -- bottom left
            twistfactor = np.exp(1j * (-thetatwist - phitwist))
        else:
            raise RuntimeError('Found periodic vector with PVx,y which do not match linear combo of PV')
    elif np.shape(PV) in [(1, 2), (2,1), (2,)]:
        # There is only one peridic lattice vector (as for a periodic strip).
        # Only use phitwist
        # First get rid of the extra dimension of PV if it exists
        if np.shape(PV) in [(1, 2), (2, 1)]:
            PV = PV.ravel()

        if (arrpv == PV).all():
            # The first periodic vector is used -- multiply by exp[i * theta]
            twistfactor = np.exp(1j * thetatwist)
        elif (arrpv == -PV).all():
            # The second periodic vector is used -- multiply by exp[i * phi]
            twistfactor = np.exp(-1j * thetatwist)

    return twistfactor


def calc_dubtheta(NLNNNangles):
    """Convert Next-nearest neighbor angles to haldane phase differences from hopping.
    NLNNNangles is positive for counterclockwise NNN bond hoppings
    Positive, acute angles become positive output. Positive, obtuse angles become negative output.
    Negative, acute angles become negative output. Negative, obtuse angles become positive output.
    """
    thetaH = np.mod(2. * NLNNNangles, 2. * np.pi)
    thetaH[thetaH > np.pi] = - (2. * np.pi - thetaH[thetaH > np.pi])
    return thetaH


def normal_modes_haldane(lat, datadir=None, t2=0.01, save_ims=True, rm_images=True,
                         gapims_only=False, PVx=None, PVy=None):
    """Compute, plot, and save the normal modes of the Haldane model. If save_ims == False, then skip plotting.
    If datadir == 'none' or None, skip saving.

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : NP x NN array
        Neighbor list
    KL : NP x NN array
        spring connectivity array
    params : dict
        parameters dictionary
    dispersion : array or list
        dispersion relation of...
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    """
    if (lat.BL < 0).any() and (lat.PVx is None or lat.PVy is None):
        raise RuntimeError('Must specify PVx and PVy in save_normal_modes_haldane() when periodic bonds exist!')

    NLNNN, KLNNN, matrix = haldane_matrix(lat, t2)

    # for jjj in range(len(matrix)):
    #     row = matrix[jjj]
    #     print 'NN = ', np.where(row == 1)[0]
    #     print '+NNN = ', np.where(row == 1j*epsilon)[0]
    #     print '-NNN = ', np.where(row == -1j*epsilon)[0]

    # plt.imshow(np.imag(matrix))
    # plt.show()

    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix, sort='real')

    #####################################
    # SAVE eigenvals/vects as txt file
    #####################################
    if datadir is not None and datadir != 'none':
        output = open(datadir + 'eigval_haldane.pkl', 'wb')
        pickle.dump(eigval, output)
        output.close()

        output = open(datadir + 'eigvect_haldane.pkl', 'wb')
        pickle.dump(eigvect, output)
        output.close()

    if save_ims and datadir is not None:
        print 'plotting...'
        fig, DOS_ax, eig_ax = leplt.initialize_lattice_DOS_header_plot(eigval, lat.xy, lat.NL, lat.KL,
                                                                       sim_type='haldane', pin=-5000, preset_cbar=False)
        # fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, lat.xy, lat.NL, lat.KL,
        #                                                         sim_type='haldane')

        sim_type = 'haldane'
        dio.ensure_dir(datadir + 'DOS_haldane/')

        #####################################
        # SAVE eigenvals/vects as images if output directory is empty
        #####################################
        done_pngs = len(glob.glob(datadir + 'DOS_haldane/DOS_haldane_*.png'))
        # check if normal modes have already been done
        if not done_pngs:
            totN = len(eigval)
            if done_pngs < totN:
                # decide on which eigs to plot
                if gapims_only:
                    middle = int(round(totN * 0.25))
                    ngap = int(round(np.sqrt(totN)))
                    todo = range(middle - ngap, middle + ngap)
                else:
                    todo = range(int(round(len(eigval))))

                dmyi = 0
                for ii in todo:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        hlatpfns.construct_eigenvalue_DOS_plot_haldane(xy, fig, DOS_ax, eig_ax, eigval, eigvect,
                                                                       ii, sim_type, NL, KL, marker_num=0,
                                                                       PVx=lat.PVx, PVy=lat.PVy, color_scheme='default')
                    plt.savefig(datadir + 'DOS_haldane/DOS_haldane_' + '{0:05}'.format(dmyi) + '.png')
                    scat_fg.remove()
                    scat_fg2.remove()
                    p.remove()
                    f_mark.remove()
                    lines_12_st.remove()
                    dmyi += 1

        fig.clf()
        plt.close('all')

        ######################
        # Save DOS as movie
        ######################
        imgname = datadir + 'DOS_haldane/DOS_haldane_'
        names = datadir.split('/')[0:-1]
        # Construct movie name from datadir path string
        movname = ''
        for ii in range(len(names)):
            if ii < len(names) - 1:
                movname += names[ii] + '/'
            else:
                movname += names[ii]

        movname += '_DOS_haldane'

        subprocess.call(['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v',
                         'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

        if rm_images:
            # Delete the original images
            print 'Deleting folder ' + datadir + 'DOS/'
            subprocess.call(['rm', '-r', datadir + 'DOS/'])

    return eigvect, eigval, matrix


def ascribe_absites(lat):
    """Alternatingly define A-sites and B-sites (ABsites) for particles (typically with three nearest neighbors)

    Parameters
    ----------
    lat : Lattice class instance
        the lattice for which to find ABsites
    """
    # first just attempt to load it from lat.lp['meshfn']
    abfn = dio.prepdir(lat.lp['meshfn']) + 'absites.pkl'
    if glob.glob(abfn):
        with open(abfn, 'r') as fn:
            abdict = pickle.load(fn)

        asites = abdict['asites']
        bsites = abdict['bsites']
    else:
        thres = 1e-7
        left = np.where(lat.xy[:, 0] + thres > np.min(lat.xy[:, 0]))[0]
        botleft = left[np.argmin(lat.xy[left, 1])]
        asites = np.zeros(len(lat.xy), dtype=bool)
        bsites = np.zeros(len(lat.xy), dtype=bool)
        # done monitors which sites' neighbors have been flipped
        done = np.zeros(len(lat.xy), dtype=bool)
        asites[botleft] = True
        xy = lat.xy
        NL = lat.NL
        KL = lat.KL
        dmyi = 0
        while not done.all():
            # ascribe neighbors of a to be b's. Ascribe neighbors of b's to be a's
            todo = np.where(np.logical_and(asites, ~done))[0]
            for ii in todo:
                flip = np.setdiff1d(np.setdiff1d(NL[ii][np.where(KL[ii])[0]], np.where(asites)[0]),
                                    np.where(bsites)[0])
                # print 'flip = ', flip
                bsites[flip] = True
            done[todo] = True

            print 'todo = ', todo
            todo = np.where(np.logical_and(bsites, ~done))[0]
            for ii in todo:
                flip = np.setdiff1d(np.setdiff1d(NL[ii][np.where(KL[ii])[0]], np.where(asites)[0]),
                                    np.where(bsites)[0])
                # print 'flip = ', flip
                asites[flip] = True
            done[todo] = True

            if lat.lp['check']:
                plt.scatter(xy[:, 0], xy[:, 1], c='k')
                plt.scatter(xy[asites, 0], xy[asites, 1], c='r')
                plt.scatter(xy[bsites, 0], xy[bsites, 1], c='g')
                plt.pause(0.1)
                plt.clf()
                dmyi += 1

        # save the absites
        absites = {'asites': asites, 'bsites': bsites}
        with open(abfn, 'w') as fn:
            pickle.dump(absites, fn)

    return asites, bsites


def delta_eo(eta, omega, omegaj):
    """This is delta_eta(omega), the sharp Lorentzian used in diffusivity calculations"""
    return eta / (np.pi * (omega - omegaj)**2 + eta**2)


def lorentzian(eps, omega, omegaj):
    """This is a sharp Lorentzian used in LDOS calculations

    Parameters
    ----------
    eps : float
        A small value specifying the height and narrowness of the Lorentzian (the taller, the skinnier)
    omega : float
        The central value of symmetry for the Lorentzian
    omegaj : float or float array
        Values at which to evaluate the Lorentzian

    Returns
    -------
    a Lorentzian evaluated at the value(s) omegaj
    """
    return eps / (np.pi * ((omega - omegaj)**2 + eps**2))


def calc_ldos(eigval, eigvect, eps=5.0):
    """Compute the local density of states rho_i(omega) = sum_n delta(omega-omega_n) |<i|n>|^2
    Returns
    -------
    ldos : NP x len(eigval) float array
        row i is the mod(i, 2)th particle, column j is the jth eigenvalue
    """
    # normalize all the states
    # eigval2 = eigval[0:int(len(eigval) * 0.5)]
    # eigvect2 = eigvect[0:int(len(eigvect) * 0.5)]
    raise RuntimeError('Check that this was translated to haldane correctly')
    # METHOD 1
    eta = eps * np.median(np.abs(np.diff(eigval)))
    ldos_tmp = np.zeros((len(eigval), len(eigval)))
    ii = 0
    inds = np.arange(len(eigval))
    imeval = np.abs(eigval)

    # Check normalization of states: np.norm(a) = np.sqrt(np.sum(np.abs(a)**2)) = 1.
    # dotp = np.zeros(len(inds), dtype=complex)
    # ii = 0
    # for row in eigvect[inds]:
    #     dotp[ii] = la.norm(row)
    #     ii += 1
    # print 'dotp = ', dotp
    # plt.plot(np.imag(eigval[inds]), np.real(dotp), 'k.-')
    # plt.plot(np.imag(eigval[inds]), np.imag(dotp), 'r.-')
    # plt.show()

    for ev in imeval:
        # consider the mod(ii, 2) particle/site
        # Sum over nn, taking iith amplitude in each nnth eigenmode, weighted by delta(omega-omega_n)
        # |<ii|nn>|^2 = |nn>[ii]**2
        print 'ev = ', ev
        jj = 0
        for omega in imeval[inds]:
            # rho_i(omega) = sum_n delta(omega-omega_n) |<i|n>|^2

            # For haldane we must use the full eigenvalue vect
            # lrz = lorentzian(eta, omega, imeval)
            # plt.plot(imeval, lrz, '-')
            # ldos_tmp[ii, jj] = np.sum(lrz * (eigvect[:, ii])**2)

            lrz = lorentzian(eta, omega, imeval[inds])
            ldos_tmp[ii, jj] = np.sum(lrz * (np.abs(eigvect[inds, ii]))**2)
            jj += 1
        ii += 1
        # plt.show()
        # sys.exit()

    # METHOD 2
    # In the limit that the Lorentzian becomes a delta function, we get
    # ldos[ii, jj] = eigvect[jj, ii] = eigvect.T
    # ldos = eigvect[inds].T

    ldos = np.abs(ldos_tmp)
    return ldos


def fit_eigvect_to_exponential(xy, eigval, eigvect, locutoffd=None, hicutoffd=20., check=False):
    """Fit the excitations of a gyro lattice to exponential decay: finite system, non-PBCs

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    locutoffd : float or None
        minimum distance from max localization point to start using data to fit to exp tail
    hicutoffd : float or None
        maximum distance from max localization point to use data to fit to exp tail
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # Take absolute value of excitations
    magevec = np.abs(eigvect)

    # find COM for each eigvect
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)
        com = le.center_of_mass(xy, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle
        dist = np.sqrt((xy[:, 0] - com[0])**2 + (xy[:, 1] - com[1])**2)

        # Sort particles by distance from com
        inds = np.argsort(dist)
        # print 'dist[inds] = ', dist[inds]

        # Use cutoff if supplied
        if locutoffd is not None and hicutoffd is not None:
            inds = inds[np.logical_and(dist[inds] < hicutoffd, dist[inds] > locutoffd)]
        elif locutoffd is not None:
            inds = inds[dist[inds] > locutoffd]
        elif hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]

        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            plt.plot(dists, mags, 'b.-')
            Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(fits[ii, 4])
            Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(fits[ii, 6])
            plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()
    return fits


def fit_eigvect_to_exponential_periodic(xy, eigval, eigvect, LL, locutoffd=None, hicutoffd=20., check=False):
    """Fit the excitations of a gyro lattice to exponential decay: A exp(K * r)
    NOTE: this assumes a square sample! Todo: replace LL with PV to generalize

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    LL : tuple of floats (2 x 1 float array is allowed)
        spatial extent of the periodic system in 2d
    locutoffd : float or None
        minimum distance from max localization point to start using data to fit to exp tail
    hicutoffd : float or None
        maximum distance from max localization point to use data to fit to exp tail
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
        where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))
    """
    magevec = np.abs(eigvect)

    # find COM for each eigvect
    fits = np.zeros((len(eigval), 7), dtype=float)
    for ii in range(len(eigval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigval)
        com = le.com_periodic(xy, LL, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distance_periodic(xy, com, LL)

        # Check it
        inds = np.argsort(dist)

        # Use cutoff if supplied
        if locutoffd is not None and hicutoffd is not None:
            inds = inds[np.logical_and(dist[inds] < hicutoffd, dist[inds] > locutoffd)]
        elif locutoffd is not None:
            inds = inds[dist[inds] > locutoffd]
        elif hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]

        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            plt.plot(dists, mags, 'b.-')
            Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
            Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
            plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 2])
        plt.show()
    return fits


def fit_eigvect_to_exponential_edge(xy, boundary, eigvect, eigval=None, locutoffd=None, hicutoffd=None, check=False,
                                    interp_n=100):
    """Fit the (edge) excitations of a haldane lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : NP x 1 complex array
        eigenvalues of the system
    eigvect : NP x NP complex array
        eigenvectors of the system
    LL : float
        width of the periodic system in 2d
    locutoffd : float
        low-end cutoff distance for fitting exponential
    hicutoffd : float
        hi-end cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 5 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    if check:
        try:
            eigval[0]
        except IndexError:
            raise RuntimeError("If kwarg 'check' is True, must supply eigval to fit_eigvect_to_exponential_edge()")

    # Take absolute value of excitations for magnitudes
    magevec = np.abs(eigvect)

    # Get minimum distance for each particle, taking care to handle the interpolation of the boundary
    # Check if boundary is a tuple, in which case the sample is periodic in one dimension
    if isinstance(boundary, tuple):
        raise RuntimeError('Use fit_eigvect_edge_periodicstrip() instead')
    else:
        dist = le.distance_from_boundary(xy, boundary, interp_n=interp_n)

    # Fit the eigvals given distances of each particle from edge
    fits = np.zeros((len(eigvect), 5), dtype=float)
    for ii in range(len(eigvect)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(eigvect)

        # Use cutoff distances if provided
        inds = np.argsort(dist)
        if hicutoffd is not None and locutoffd is not None:
            inds = inds[np.logical_and(dist[inds] > locutoffd, dist[inds] < hicutoffd)]
        elif hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        elif locutoffd is not None:
            inds = inds[dist[inds] > locutoffd]

        mags = magevec[ii][inds]
        dists = dist[inds]

        # Check if exponential is possible
        # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
        # possibly_exp = possibly_exponential(dists, mags)

        # if possibly_exp:
        # Fit to an exponential y=A*exp(K*t) using linear method.
        # METHOD 1
        # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
        # A, K, cov = fit_exp_linear(binc, bin_means)
        A, K, cov = fit_exp_linear(dists, mags)
        fits[ii, 0] = A
        fits[ii, 1] = K
        fits[ii, 2] = cov[0, 0]
        fits[ii, 3] = cov[0, 1]
        fits[ii, 4] = cov[1, 1]

        # check the result
        if check:
            if np.abs(K) > 0.1:
                plt.clf()
                plt.plot(dists, mags, 'b.-')
                plt.plot(dists, A * np.exp(K*dists), 'r-')
                Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 2]))
                Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
                plt.pause(0.1)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 1])
        plt.show()

    return fits


def fit_eigvect_edge_periodicstrip(xy, boundary, eigvect, PVx, PVy, eigval=None, locutoffd=None, hicutoffd=None,
                                   check=False, interp_n=100):
    """Fit the (edge) excitations of a haldane lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    boundary : tuple of two int arrays
        Each boundary of the periodic strip as an ordered int array
    eigvect : NP x NP complex array
        eigenvectors of the system
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    eigval : NP x 1 complex array, required only if check==True
        eigenvalues of the system
    locutoffd : float
        low-end cutoff distance for fitting exponential
    hicutoffd : float
        hi-end cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 6 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K, topbottom
        topbottom is an int indicator of whether the excitation is fit to a state localized to the top (0) or bottom (1)
    """
    if check:
        try:
            eigval[0]
        except IndexError:
            raise RuntimeError("If kwarg 'check' is True, must supply eigval to fit_eigvect_to_exponential_edge()")

    # Take absolute value of excitations for magnitudes
    magevec = np.abs(eigvect)

    # Get minimum distance for each particle, taking care to handle the interpolation of the boundary
    # Supplied 'boundary' must be a tuple, as the sample is periodic in one dimension
    if PVx is None or PVy is None:
        raise RuntimeError('Must supply PVx and PVy for periodic_strip sample')
    else:
        # here boundary is a tuple of two boundaries
        dist_tup = le.distance_from_boundary_periodicstrip(xy, boundary, PVx, PVy, interp_n=interp_n)

    # Fit the eigvals given distances of each particle from edge
    fits = np.zeros((len(eigvect), 6), dtype=float)
    for ii in range(len(eigvect)):
        if ii % 100 == 1:
            print 'hlfns: eval #', str(ii), '/', len(eigvect)

        # First try to fit to the top boundary
        jj = 0
        for dist in dist_tup:
            # Use cutoff distances if provided
            inds = np.argsort(dist)
            if hicutoffd is not None and locutoffd is not None:
                inds = inds[np.logical_and(dist[inds] > locutoffd, dist[inds] < hicutoffd)]
            elif hicutoffd is not None:
                inds = inds[dist[inds] < hicutoffd]
            elif locutoffd is not None:
                inds = inds[dist[inds] > locutoffd]

            mags = magevec[ii][inds]
            dists = dist[inds]

            # Check if exponential is possible
            # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
            # possibly_exp = possibly_exponential(dists, mags)

            # if possibly_exp:
            # Fit to an exponential y=A*exp(K*t) using linear method.
            # METHOD 1
            # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
            # A, K, cov = fit_exp_linear(binc, bin_means)
            A, K, cov = fit_exp_linear(dists, mags)
            if jj == 0:
                fits[ii, 0] = A
                fits[ii, 1] = K
                fits[ii, 2] = cov[0, 0]
                fits[ii, 3] = cov[0, 1]
                fits[ii, 4] = cov[1, 1]
                fits[ii, 5] = 0
            else:
                # Check if fit is better than fitting to other boundary
                A, K, cov = fit_exp_linear(dists, mags)
                better_fit_negative = cov[1, 1] < fits[ii, 4] and K < 0.
                if better_fit_negative or K < fits[ii, 1]:
                    # This fit is better than previous, so update fits array
                    fits[ii, 0] = A
                    fits[ii, 1] = K
                    fits[ii, 2] = cov[0, 0]
                    fits[ii, 3] = cov[0, 1]
                    fits[ii, 4] = cov[1, 1]
                    fits[ii, 5] = 1

                # Check if fits better to a constant value
                variance = np.var(mags)
                # print 'fits[ii, 4] = ', fits[ii, 4]
                # print 'variance = ', variance
                if fits[ii, 1] > 0:
                    fits[ii, 0] = np.mean(mags)
                    fits[ii, 1] = 0.
                    fits[ii, 2] = variance
                    fits[ii, 3] = 0.
                    fits[ii, 4] = 0.
                    fits[ii, 5] = 2

                # check the result
                if check:
                    K = fits[ii, 1]
                    A = fits[ii, 0]
                    if abs(K) > 0.05:
                        plt.clf()
                        plt.semilogy(dists, mags, 'b.-')
                        plt.semilogy(dists, A * np.exp(K * dists), 'r-')
                        Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 2]))
                        Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                        plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
                        plt.pause(0.1)

            jj += 1

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(eigval), fits[:, 1])
        plt.show()

    return fits


# def fit_edgedecay_periodicstrip(xy, boundarytuple, eigval, eigvect, locutoffd=None, hicutoffd=None,
#                                 check=False, interp_n=100):
#     """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)
#     Note that this fits the localization in x (the periodic dimension), NOT the localization in y (the finite
#     dimension)
#
#     Parameters
#     ----------
#     xy : NP x 2 float array
#         positions of particles in 2D
#     boundarytuple : tuple of two int arrays
#         Each boundary of the periodic strip as an ordered int array
#     eigval : 2*NP x 1 complex array
#         eigenvalues of the system
#     eigvect : 2*NP x 2*NP complex array
#         eigenvectors of the system
#     locutoffd : float
#         minimum bound for distance from boundary, for fitting exponential
#     hicutoffd : float
#         maximum bound for distance from boundary, for fitting exponential
#     check : bool
#         show intermediate steps
#
#     Returns
#     -------
#     fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
#         fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
#     """
#     # convert eigvect to magnitude of eigenvectors, consider all of each (not half as in gyro case)
#     # Opted against mandating a cutoff distance which would be logical since modes are localized either to top or bottom
#     # if cutoffd is None:
#     #     cutoffd = Ly * 0.5
#
#     # Contract x and y component of each evect so that len(halfevec[ii]) = NP
#     magevec = np.abs(eigvect)
#
#     # assume amplitude largest against the top or bottom edge
#     fits = np.zeros((eigval, 7), dtype=float)
#     dist_tup = le.distance_from_boundary_periodicstrip(xy, boundarytuple, interp_n=interp_n)
#     # Sort the results
#     inds = np.argsort(dist)
#     for ii in range(len(eigval)):
#         if ii % 100 == 1:
#             print 'hlfns: eval #', str(ii), '/', len(eigval)
#
#         # first fit to bottom, then to top, keep better fit
#         # Get distance from edge for each particle
#         dist = dist_tup[1]
#         if hicutoffd is not None:
#             if locutoffd is not None:
#                 inds = inds[np.logical_and(dist[inds] < hicutoffd, dist[inds] > locutoffd)]
#             else:
#                 inds = inds[dist[inds] < hicutoffd]
#         elif locutoffd is not None:
#             inds = inds[dist[inds] > locutoffd]
#
#         mags = magevec[ii][inds]
#         dists = dist[inds]
#         Ab, Kb, covb = fit_exp_linear(dists, mags)
#
#         # fit to top
#         dist = dist_tup[0]
#         inds = np.argsort(dist)
#         if hicutoffd is not None:
#             if locutoffd is not None:
#                 inds = inds[np.logical_and(dist[inds] < hicutoffd, dist[inds] > locutoffd)]
#             else:
#                 inds = inds[dist[inds] < hicutoffd]
#         elif
#             inds = inds[dist[inds] > locutoffd]
#         mags = magevec[ii][inds]
#         dists = dist[inds]
#         At, Kt, covt = fit_exp_linear(dists, mags)
#
#         # Keep whichever has smaller K
#         if Kt < Kb:
#             fits[ii, 0:2] = np.array([0, np.min(xy[:, 1])])
#             A = At
#             K = Kt
#             cov = covt
#         else:
#             fits[ii, 0:2] = np.array([0, np.max(xy[:, 1])])
#             A = Ab
#             K = Kb
#             cov = covb
#
#         fits[ii, 2] = A
#         fits[ii, 3] = K
#         fits[ii, 4] = cov[0, 0]
#         fits[ii, 5] = cov[0, 1]
#         fits[ii, 6] = cov[1, 1]
#
#         # check the result
#         if check:
#             plt.plot(dists, mags, 'b.-')
#             Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
#             Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
#             plt.title(r'$\omega = $' + str(halfeval[ii]) + Astr + Kstr)
#             if np.imag(halfeval[ii]) < 1.1:
#                 plt.show()
#             elif 2.5 > np.imag(halfeval[ii]) > 2.1:
#                 plt.pause(1)
#             else:
#                 plt.pause(0.001)
#
#     # Inspect localization parameter K (exponent in exp(K * r)
#     if check:
#         plt.plot(np.imag(halfeval), fits[:, 2])
#         plt.show()
#
#     return fits


def possibly_exponential(xx, yy, bins=10):
    """
    """
    bin_means, bin_edges, bin_number = scistat.binned_statistic(xx, yy, bins=bins)
    # print 'bin_means = ', bin_means
    # print 'bin_edges = ', bin_edges
    # plt.plot(bin_edges[1:], np.log(bin_means), 'b.-')
    # plt.title('Possy exp = ' + str(bin_means[0] > np.mean(bin_means[1:])))
    # plt.pause(0.1)
    # plt.clf()
    return bin_means[0] > np.mean(bin_means[1:]) and bin_means[1] > np.mean(bin_means[2:])


def fit_exponential(t, y):
    """Fit data to y=A*exp(K*t) using linear method.
    """
    # Fitting to y = A * exp(K * t)
    # Linear Fit (Note that we have to provide the y-offset ("C") value!!
    A, K, cov = fit_exp_linear(t, y)
    fit_y = exponential_func(t, A, K, 0.)
    return A, K, cov, fit_y


def exponential_func(t, A, K, C):
    return A * np.exp(K * t) + C


def fit_exp_linear(t, y, C=0):
    """Fit data y(t) to y=A*exp(K*t) using a linear method by fitting y = log(A * exp(K * t)) = K * t + log(A).
    """
    K_Alog, cov = np.polyfit(t, np.log(y - C), 1, full=False, cov=True)
    K = K_Alog[0]
    A = np.exp(K_Alog[1])
    return A, K, cov


###################################
# Kolmogorov-Smirnoff test notes
###################################
def expcdf(t, beta=1.0):
    """"""
    return 1.0 - np.exp(-t/beta)


def kstest_exponential(mags, dist, bins=-1):
    """Perform Kolmogorov-Smirnoff test with data that may approximate an exponential curve.
    See http://stats.stackexchange.com/questions/110272/a-naive-question-about-the-kolmogorov-smirnov-test
    """
    if bins == -1:
        bins = int(np.max(dist))
    bin_means, bin_edges, bin_number = scistat.binned_statistic(mags, dist, bins=bins)
    print 'bin_edges = ', bin_edges
    xx = np.cumsum(bin_means)
    xx = (xx - xx.min()) / xx.ptp()
    xx = []
    jj = 0
    for be in bin_means:
        for ii in np.arange(int(100 * bin_means[jj])):
            xx.append(bin_edges[jj])
        jj += 1
    print 'xx = ', xx
    plt.hist(xx)
    plt.show()
    return scistat.kstest(xx, expcdf)  # 'expon')

# Example using fake data
# import numpy as np
# import scipy.stats as scistat
# import matplotlib.pyplot as plt
# dist = np.arange(100)/100.
# mags = np.exp(-dist)
# kstest_exponential(mags, dist, bins=12)
