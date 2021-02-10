import numpy as np
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as lsegs
import lepm.stringformat as sf
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import scipy.stats as scistat
import copy
import lepm.line_segments as linsegs
from lepm.data_handling import dist_pts
import h5py
import lepm.dataio as dio
import lepm.hdf5io as h5io
import glob
try:
    import cPickle as pickle
except ImportError:
    import pickle

"""Supporting functions for gyro_lattice_class.py
"""


def param2meshfnexten_name(param):
    """Convert a parameter (string) name into its possibly-abbreviated form that appears as part of a meshfn_exten (the
    string specifier extension to the pathname of a lattice or gyrolattice).

    Parameters
    ----------
    param : str
        A key for a GyroLattice instance's lp (lattice parameter dictionary)

    Returns
    -------
    mfestr : str
    """
    if param == 'ABDelta':
        mfestr = 'ABd'
    elif param == 'V0_pin_gauss':
        mfestr = 'Vpin'
    elif param == 'V0_pin_flat':
        mfestr = 'Vfpin'
    elif param == 'percolation_density':
        mfestr = 'perd'
    elif param == 'OmKspec':
        mfestr = param
    else:
        raise RuntimeError('Have not yet supported this parameter in string conversion -- add line for it here.')
    return mfestr


def get_basis_str(glat):
    """Return the string specifier for the basis in which dynamical matrix is cast

    Parameters
    ----------
    glat

    Returns
    -------

    """
    if 'basis' in glat.lp:
        if glat.lp['basis'] not in ['XY', None]:
            basis_str = '_' + glat.lp['basis']
        else:
            basis_str = ''
    else:
        basis_str = ''

    return basis_str


def eigvect2displacements_xy(evii):
    """Convert an eigenvector stored as x,y complex numbers into an array of xy displacements"""
    NP = int(len(evii) * 0.5)
    magx = np.array([evii[2 * i] for i in range(NP)])
    magy = np.array([evii[2 * i + 1] for i in range(NP)])
    displ = np.dstack((magx, magy))[0].real
    return displ


def dynamical_matrix_gyros(glat, basis=None):
    """Construct the dynamical matrix for the given GyroLattice.

    Parameters
    ----------
    glat : GyroLattice instance
        The GyroLattice for which to construct the dynamical matrix
    basis : 'XY' or 'psi' or None
        The basis in which to construct the dynamical matrix

    Returns
    -------
    matrix
    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in glat.lp:
        thetatwist = glat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in glat.lp:
        phitwist = glat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    if basis is None and 'basis' in glat.lp:
        basis = glat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        if glat.bL is None:
            # Rest lengths of springs == distances between particles
            if notwist:
                # not twisted, no stretch, XY basis
                matrix = calc_dynamical_matrix_gyros(glat.lattice.xy, glat.lattice.NL, glat.lattice.KL,
                                                     glat.OmK, glat.Omg, glat.lattice.PVx, glat.lattice.PVy)
            else:
                # twisted, no stretch, XY basis
                print 'PV = ', glat.lattice.PV
                print 'thetatwist = ', thetatwist
                print 'phitwist = ', phitwist
                if glat.lp['periodic_strip']:
                    # All periodic bonds are twisted
                    print 'glatfns: computing dynamical matrix for periodic strip here'
                    matrix = calc_dynamical_matrix_gyros_twist(glat.lattice.xy, glat.lattice.NL, glat.lattice.KL,
                                                               glat.OmK, glat.Omg, thetatwist, phitwist,
                                                               glat.lattice.PVx, glat.lattice.PVy, glat.lattice.PV)
                else:
                    # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                    # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                    if 'annulus' in glat.lp['LatticeTop'] or glat.lp['shape'] == 'annulus':
                        # Twist bonds in a cut of the annular network
                        twistcut = np.array([0., 0., np.max(glat.lattice.xy[:, 0]), 0.])
                        thetaKL = form_twistedKL(glat.lattice.xy, glat.lattice.BL, glat.lattice.NL, glat.lattice.KL,
                                                 twistcut)
                        phiKL = np.zeros_like(thetaKL, dtype=int)
                    else:
                        raise RuntimeError('Currently only have twistedKL set up for annular samples')

                    matrix = calc_dynamical_matrix_gyros_twist_bonds(glat.lattice.xy, glat.lattice.NL, glat.lattice.KL,
                                                                     glat.OmK, glat.Omg,
                                                                     thetaKL, phiKL,
                                                                     thetatwist, phitwist,
                                                                     glat.lattice.PVx, glat.lattice.PVy,
                                                                     glat.lattice.PV)
        else:
            # Rest lengths of springs != distances between particles
            matrix = calc_dynamical_matrix_gyros_stretched(glat.lattice.xy, glat.lattice.BL, glat.lattice.NL,
                                                           glat.lattice.KL, glat.OmK, glat.Omg, glat.bL,
                                                           PVxydict=glat.lattice.PVxydict)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            # matrix = calc_dynamical_matrix_psi(glat.lattice.xy, glat.lattice.NL, glat.lattice.KL, glat.OmK, glat.Omg,
            #                                    glat.lattice.PVx, glat.lattice.PVy)

            import lepm.gyro_lattice_kspace_functions as glatkspace_fns
            matrix = glatkspace_fns.dynamical_matrix_kspace([0., 0.], glat, eps=1e-9, basis=basis, verbose=False)

        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in glat.lp:
        if glat.lp['immobile_boundary']:
            boundary = glat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def calc_dynamical_matrix_gyros(xy, NL, KL, OmK, Omg, PVx, PVy):
    """Calculates the matrix for finding the normal modes of the system.
    Assumes linearity of force with distance from rest positions.
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body axis 3.

    Parameters
    ----------
    xy : NP x dim array
    NL : NP x NN array
    KL : NP x NN array
    OmK : float or NP x NN array
        k*l**2/I3*omega3, interaction strength
    Omg : float or NP x 1 array
        l*gn/I3*omega3, gravitational precession frequency
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    """
    try:
        NP, NN = np.shape(NL)
    except ValueError:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    # Unpack periodic boundary vectors
    if PVx is None or len(PVx) == 0 or PVy is None or len(PVy) == 0:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        omg = Omg[i]  # grav frequency for this connection

        # pinning/gravitational matrix
        M2[2 * i, 2 * i + 1] = - omg
        M2[2 * i + 1, 2 * i] = omg

        for nn in range(NN):
            # the number of the gyroscope i is connected to (particle j)
            ni = NL[i, nn]
            # true connection?
            k = KL[i, nn]
            # spring frequency for this connection
            omk = OmK[i, nn]

            if abs(k) > 1e-12:
                # There is a true connection, so update dynamical matrix
                # if len(dispersion) > 1:
                #     disp = 1. / (1. + dispersion[i])
                # else:
                #     disp = 1.
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
                diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
                alphaij = np.arctan2(diffy, diffx)
                # rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)

                # for periodic systems, KL is -1 for particles on opposing boundaries
                # However, we don't do the line below because it is handled by PVx,PVy
                # if k == -1  :
                #     alphaij = (np.pi + alphaij)%(2*np.pi)

                # What is this for?
                if k == -2:  # will only happen on first or last gyro in a line
                    if i == 0 or i == (NP - 1):
                        print i, '--> NL=-2 for this particle'
                        yy = np.where(KL[i] == 1)
                        dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
                        dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
                        al = (np.arctan2(dy, dx)) % (2 * np.pi)
                        alphaij = np.pi - al
                        if i == 1:
                            alphaij = np.pi - (0.25 * np.pi)
                        else:
                            alphaij = - (0.25 * np.pi)

                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)

                if abs(Cos) < 10E-8:
                    Cos = 0.0

                if abs(Sin) < 10E-8:
                    Sin = 0.0

                Cos2 = Cos ** 2
                Sin2 = Sin ** 2
                CosSin = Cos * Sin

                # (x components)
                M1[2 * i, 2 * i] += -omk * CosSin  # dxi - dxi
                M1[2 * i, 2 * i + 1] += -omk * Sin2  # dxi - dyi
                M1[2 * i, 2 * ni] += omk * CosSin  # dxi - dxj
                M1[2 * i, 2 * ni + 1] += omk * Sin2  # dxi - dyj

                # (y components)
                M1[2 * i + 1, 2 * i] += omk * Cos2  # dyi - dxi
                M1[2 * i + 1, 2 * i + 1] += omk * CosSin  # dyi - dyi
                M1[2 * i + 1, 2 * ni] += -omk * Cos2  # dyi - dxj
                M1[2 * i + 1, 2 * ni + 1] += -omk * CosSin  # dyi - dyj

                # Checking
                # if i==0:
                #    print '\n --- \n added M1[2*i+1, 2*i] = ',disp*k*Cos2 *((-1)**b[i]) *dir_factor
                #    print 'dir_factor = ', dir_factor
                #    print 'k = ', k
                #    print 'else =', ((-1)**b[i]) *dir_factor

    # self.pin_array.append(2*pi*1*extra_factor)
    # Assumes that b=0, c=1 so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    matrix = M1 + M2

    return matrix


def calc_dynamical_matrix_gyros_twist(xy, NL, KL, OmK, Omg, thetatwist, phitwist, PVx, PVy, PV):
    """Calculates the matrix for finding the normal modes of a gyro system with twisted boundary conditions.
    Assumes linearity of force with distance from rest positions.
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body axis 3.

    Parameters
    ----------
    xy : NP x dim array
    NL : NP x NN array
    KL : NP x NN array
    OmK : float or NP x NN array
        k*l**2/I3*omega3, interaction strength
    Omg : float or NP x 1 array
        l*gn/I3*omega3, gravitational precession frequency
    thetatwist : float
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of pi radians
    phitwist : float
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of pi radians
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    PV : 2 x1, 1 x 2, or 2 x 2 float array (or None if not periodic, but then this function is not useful)
        periodic vectors for the gyro network. If 2x2 then the network is fully periodic, otherwise the network is a
        periodic strip.
    """
    eps = 1e-12
    try:
        NP, NN = np.shape(NL)
    except ValueError:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    # Convert thetatwist and phitwist to units of radians, from units of pi radians
    if thetatwist is not None:
        thetatwist *= np.pi
    if phitwist is not None:
        phitwist *= np.pi

    m2_shape = np.shape(M2)

    # Unpack periodic boundary vectors
    # if PVxydict is not None:
    #     print '\n\n\n\n lattice_elasticity: dynamical_matrix_gyros: unpacking PVxydict...\n\n\n\n'
    #     PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
    # else:
    #     PVx = np.zeros((NP, NN), dtype=float)
    #     PVy = np.zeros((NP, NN), dtype=float)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        omg = Omg[i]  # grav frequency for this connection
        for nn in range(NN):
            # the number of the gyroscope i is connected to (particle j)
            ni = NL[i, nn]
            # true connection?
            k = KL[i, nn]
            # spring frequency for this connection
            omk = OmK[i, nn]

            if abs(k) > eps:
                # There is a true connection, so update dynamical matrix
                # if len(dispersion) > 1:
                #     disp = 1. / (1. + dispersion[i])
                # else:
                #     disp = 1.
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
                diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
                alphaij = 0.

                # Obtain twist factor if the bond is a periodic one
                # print 'glatfns: PVx[i, nn] = ', PVx[i, nn]
                # print 'glatfns: PVy[i, nn] = ', PVy[i, nn]
                # print 'glatfns: PVx = ', PVx
                if KL[i, nn] < 0.:
                    # form the array of the periodic vector from j to j's reflection as seen by i
                    arrpvj = np.array([PVx[i, nn], PVy[i, nn]])
                    ijtwist = get_twist_angle(arrpvj, PV, thetatwist, phitwist)
                else:
                    ijtwist = 0.

                # rij_mag = np.sqrt(diffx**2+diffy**2)

                if abs(k) > 0:
                    alphaij = np.arctan2(diffy, diffx)

                # for periodic systems, KL is -1 for particles on opposing boundaries
                # if k == -1  :
                #     alphaij = (np.pi + alphaij)%(2*np.pi)

                # What is this for?
                if k == -2:  # will only happen on first or last gyro in a line
                    if i == 0 or i == (NP - 1):
                        print i, '--> NL=-2 for this particle'
                        yy = np.where(KL[i] == 1)
                        dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
                        dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
                        al = (np.arctan2(dy, dx)) % (2 * np.pi)
                        alphaij = np.pi - al
                        if i == 1:
                            alphaij = np.pi - (45. * np.pi / 180.)
                        else:
                            alphaij = - (45. * np.pi / 180.)

                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)
                ca_p = np.cos(alphaij - ijtwist)
                sa_p = np.sin(alphaij - ijtwist)

                if abs(Cos) < 10E-8:
                    Cos = 0.0

                if abs(Sin) < 10E-8:
                    Sin = 0.0

                Cos2 = Cos ** 2
                Sin2 = Sin ** 2
                CosSin = Cos * Sin

                # Define influence of bond on each particle in rows for particle i
                xixi = -omk * CosSin  # dxi - dxi
                xiyi = -omk * Sin2  # dxi - dyi
                xixj = omk * Sin * ca_p  # dxi - dxj
                xiyj = omk * Sin * sa_p  # dxi - dyj

                yixi = omk * Cos2  # dyi - dxi
                yiyi = omk * CosSin  # dyi - dyi
                yixj = -omk * Cos * ca_p  # dyi - dxj
                yiyj = -omk * Cos * sa_p  # dyi - dyj

                # (x components)
                M1[2 * i, 2 * i] += xixi  # dxi - dxi
                M1[2 * i, 2 * i + 1] += xiyi  # dxi - dyi
                M1[2 * i, 2 * ni] += xixj  # dxi - dxj
                M1[2 * i, 2 * ni + 1] += xiyj  # dxi - dyj

                # (y components)
                M1[2 * i + 1, 2 * i] += yixi  # dyi - dxi
                M1[2 * i + 1, 2 * i + 1] += yiyi # dyi - dyi
                M1[2 * i + 1, 2 * ni] += yixj  # dyi - dxj
                M1[2 * i + 1, 2 * ni + 1] += yiyj  # dyi - dyj

                # Checking
                # if i==0:
                #    print '\n --- \n added M1[2*i+1, 2*i] = ',disp*k*Cos2   *((-1)**b[i]) *dir_factor
                #    print 'dir_factor = ', dir_factor
                #    print 'k = ', k
                #    print 'else =', ((-1)**b[i]) *dir_factor

                # pinning/gravitational matrix
                M2[2 * i, 2 * i + 1] = - omg
                M2[2 * i + 1, 2 * i] = omg

    # self.pin_array.append(2*pi*1*extra_factor)
    # Assumes that b=0, c=1 so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    matrix = M1 + M2

    return matrix


def calc_dynamical_matrix_gyros_twist_bonds(xy, NL, KL, OmK, Omg, thetaKL, phiKL, thetatwist, phitwist,
                                            PVx, PVy, PV):
    """Calculates the matrix for finding the normal modes of a gyro system with twisted boundary conditions.
    Arbitrary bonds are twisted, specified in thetaKL and phiKL
    Assumes linearity of force with distance from rest positions.
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body
    axis 3.

    Parameters
    ----------
    xy : NP x dim array
    NL : NP x NN array
    KL : NP x NN array
    OmK : float or NP x NN array
        k*l**2/I3*omega3, interaction strength
    Omg : float or NP x 1 array
        l*gn/I3*omega3, gravitational precession frequency
    thetaKL : NP x NN int array
        thetaKL[i, nn] is twisted wherever site i sees NL[i, nn] as twisted by thetatwist
    phiKL : NP x NN int array
        phiKL[i, nn] is twisted wherever site i sees NL[i, nn] as twisted by phitwist
    thetatwist : float
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of pi radians
    phitwist : float
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of pi radians
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    PV : 2 x1, 1 x 2, or 2 x 2 float array (or None if not periodic, but then this function is not useful)
        periodic vectors for the gyro network. If 2x2 then the network is fully periodic, otherwise the network is a
        periodic strip.
    """
    eps = 1e-12
    try:
        NP, NN = np.shape(NL)
    except ValueError:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    # Convert thetatwist and phitwist to units of radians, from units of pi radians
    if thetatwist is not None:
        thetatwist *= np.pi
    if phitwist is not None:
        phitwist *= np.pi

    # Unpack periodic boundary vectors
    if PVx is None or PVy is None:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        omg = Omg[i]  # grav frequency for this connection
        for nn in range(NN):
            # the number of the gyroscope i is connected to (particle j)
            ni = NL[i, nn]
            # true connection?
            k = KL[i, nn]
            # spring frequency for this connection
            omk = OmK[i, nn]

            if abs(k) > eps:
                # There is a true connection, so update dynamical matrix
                # if len(dispersion) > 1:
                #     disp = 1. / (1. + dispersion[i])
                # else:
                #     disp = 1.
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
                diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
                alphaij = 0.

                # Obtain twist factor if the bond is a periodic one
                if abs(thetaKL[i, nn]) > eps or abs(phiKL[i, nn]) > eps:
                    # form the array of the periodic vector from j to j's reflection as seen by i
                    if abs(thetaKL[i, nn]) and abs(phiKL[i, nn]):
                        ijtwist = thetaKL[i, nn] * thetatwist + phitwist[i, nn] * phitwist
                    elif abs(thetaKL[i, nn]):
                        ijtwist = thetaKL[i, nn] * thetatwist
                    elif abs(phiKL[i, nn]):
                        ijtwist = phitwist[i, nn] * phitwist
                    else:
                        raise RuntimeError('Something is wrong with thetaKL or phiKL')
                else:
                    ijtwist = 0.

                # rij_mag = np.sqrt(diffx**2+diffy**2)

                if abs(k) > 0:
                    alphaij = np.arctan2(diffy, diffx)

                # for periodic systems, KL is -1 for particles on opposing boundaries
                # if k == -1  :
                #     alphaij = (np.pi + alphaij)%(2*np.pi)

                # What is this for?
                if k == -2:  # will only happen on first or last gyro in a line
                    if i == 0 or i == (NP - 1):
                        print i, '--> NL=-2 for this particle'
                        yy = np.where(KL[i] == 1)
                        dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
                        dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
                        al = (np.arctan2(dy, dx)) % (2 * np.pi)
                        alphaij = np.pi - al
                        if i == 1:
                            alphaij = np.pi - (45. * np.pi / 180.)
                        else:
                            alphaij = - (45. * np.pi / 180.)

                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)
                ca_p = np.cos(alphaij - ijtwist)
                sa_p = np.sin(alphaij - ijtwist)

                if abs(Cos) < 10E-8:
                    Cos = 0.0

                if abs(Sin) < 10E-8:
                    Sin = 0.0

                # Define influence of bond on each particle in rows for particle i
                xixi = -omk * Cos * Sin  # dxi - dxi
                xiyi = -omk * Sin ** 2  # dxi - dyi
                xixj = omk * Sin * ca_p  # dxi - dxj
                xiyj = omk * Sin * sa_p  # dxi - dyj

                yixi = omk * Cos ** 2  # dyi - dxi
                yiyi = omk * Cos * Sin  # dyi - dyi
                yixj = -omk * Cos * ca_p  # dyi - dxj
                yiyj = -omk * Cos * sa_p  # dyi - dyj

                # (x components)
                M1[2 * i, 2 * i] += xixi  # dxi - dxi
                M1[2 * i, 2 * i + 1] += xiyi  # dxi - dyi
                M1[2 * i, 2 * ni] += xixj  # dxi - dxj
                M1[2 * i, 2 * ni + 1] += xiyj  # dxi - dyj

                # (y components)
                M1[2 * i + 1, 2 * i] += yixi  # dyi - dxi
                M1[2 * i + 1, 2 * i + 1] += yiyi # dyi - dyi
                M1[2 * i + 1, 2 * ni] += yixj  # dyi - dxj
                M1[2 * i + 1, 2 * ni + 1] += yiyj  # dyi - dyj

                # pinning/gravitational matrix
                M2[2 * i, 2 * i + 1] = - omg
                M2[2 * i + 1, 2 * i] = omg

    # Assumes that b=0, c=1 so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    matrix = M1 + M2

    return matrix


def form_twistedKL(xy, BL, NL, KL, twistcut):
    """
    # To determine whether bond[0] is 'to the right' of thetacut, take the crossproduct
    #              o 0
    #       aa   ^ |
    #           /  |
    #         /    |
    #     0 .______|_____________. 1 --> aa
    #              |
    #              |
    #              o 1   particle 1 sees particle 0 in the bond as rotated by positive twistangle

    Parameters
    ----------
    xy :
    BL :
    NL :
    KL :
    twistcut :

    Returns
    -------
    twistKL :
    """
    twistKL = np.zeros_like(KL, dtype=int)
    # form linesegments from xy and BL
    linesegs = linsegs.xyBL2linesegs(xy, BL)
    does_intrsct = linsegs.linesegs_intersect_linesegs(linesegs, twistcut, thres=1e-10)
    # Go through each bond that intersects and assign it to be on right or left of twistcut, defined by cross product
    for bond in BL[does_intrsct]:
        if (np.abs(bond) > 0).all():
            jj = np.where(NL[bond[0]] == bond[1])[0][0]
            # To determine whether bond[0] is 'to the right' of thetacut, take the crossproduct
            #              o 0
            #       aa   ^ |
            #           /  |
            #         /    |
            #     0 .______|_____________. 1 --> aa
            #              |
            #              |
            #              o 1   particle 1 sees particle 0 in the bond as rotated by positive twistangle
            aa = twistcut[2:4] - twistcut[0:2]
            bb = xy[bond[0]] - twistcut[0:2]
            signcross = np.sign(aa[0] * bb[1] - aa[1] * bb[0])
            twistKL[bond[0], jj] = int(-signcross)
            kk = np.where(NL[bond[1]] == bond[0])[0][0]
            twistKL[bond[1], kk] = int(signcross)
        else:
            # One of the particles indices in the bond is zero, so take more care here
            jj = np.where(np.logical_and(NL[bond[0]] == bond[1], np.abs(KL[bond[0]]) > 0))[0][0]
            # To determine whether bond[0] is 'to the right' of thetacut, take the crossproduct
            #              o 0
            #       aa   ^ |
            #           /  |
            #         /    |
            #     0 .______|_____________. 1 --> aa
            #              |
            #              |
            #              o 1   particle 1 sees particle 0 in the bond as rotated by positive twistangle
            aa = thetacut[2:4] - thetacut[0:2]
            bb = xy[bond[0]] - thetacut[0:2]
            signcross = np.sign(aa[0] * bb[1] - aa[1] * bb[0])
            twistKL[bond[0], jj] = int(-signcross)
            kk = np.where(np.logical_and(NL[bond[1]] == bond[0], np.abs(KL[bond[1]]) > 0))[0][0]
            twistKL[bond[1], kk] = int(signcross)

    return twistKL


def get_twist_factor(arrpv, PV, thetatwist, phitwist=None):
    """Obtain the multiplicative factor for the matrix element i, j if arrpv takes j to its image as seen by i.

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
        perioodic boundary twist in x direction -- additional phase picked up in PV[0], in units of radians here
        (not in units of pi)
    phitwist : float (or None if periodic_strip)
        perioodic boundary twist in y direction -- additional phase picked up in PV[1], in units of radians here
        (not in units of pi)

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
    elif np.shape(PV) in [(1, 2), (2, 1), (2,)]:
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


def get_twist_angle(arrpv, PV, thetatwist, phitwist=None):
    """Obtain the additive angle for the matrix element i, j if arrpv takes j to its image as seen
    by i.

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
    twistangle
    """
    # Check if the sample is fully periodic, or just periodic strip
    if np.shape(PV) == (2, 2):
        # If there are two copies of the same periodic vector, then this is a periodic strip
        if (PV[0] == PV[1]).all():
            PV = PV[0]
            fully_periodic = False
        else:
            fully_periodic = True
    else:
        fully_periodic = False

    # If there are two periodic lattice vectors PV (ie shape is (2,2)), then pick out which one
    if fully_periodic:
        if thetatwist is None or phitwist is None:
            raise RuntimeError('Must supply both thetatwist and phitwist for fully periodic sample')

        # Find which periodic vector takes site i to site j
        if (arrpv == PV[0]).all():
            # The first periodic vector is used -- multiply by exp[i * theta]
            twistangle = thetatwist
        elif (arrpv == PV[1]).all():
            # The second periodic vector is used -- multiply by exp[i * phi]
            twistangle = phitwist
        elif (arrpv == -PV[0]).all():
            # Crossing the first periodic vector in opposite dir -- multiply by exp[-i * theta]
            twistangle = -thetatwist
        elif (arrpv == -PV[1]).all():
            # Crossing the second periodic vector in opposite dir -- multiply by exp[-i * phi]
            twistangle = -phitwist
        elif (arrpv == PV[0] + PV[1]).all():
            # Crossing both periodic vectors -- top right
            twistangle = thetatwist + phitwist
        elif (arrpv == PV[0] - PV[1]).all():
            # Crossing both periodic vectors -- bottom right
            twistangle = thetatwist - phitwist
        elif (arrpv == -PV[0] + PV[1]).all():
            # Crossing both periodic vectors -- top left
            twistangle = -thetatwist + phitwist
        elif (arrpv == -PV[0] - PV[1]).all():
            # Crossing both periodic vectors -- bottom left
            twistangle = -thetatwist - phitwist
        else:
            print 'PV = ', PV
            print 'arrpv = ', arrpv
            raise RuntimeError('Found periodic vector with PVx,y which do not match linear combo of PV')
    elif np.shape(PV) in [(1, 2), (2, 1), (2,)]:
        # There is only one peridic lattice vector (as for a periodic strip).
        # Only use phitwist
        # First get rid of the extra dimension of PV if it exists
        if np.shape(PV) in [(1, 2), (2, 1)]:
            PV = PV.ravel()

        if (arrpv == PV).all():
            # The first periodic vector is used -- multiply by exp[i * theta]
            twistangle = thetatwist
        elif (arrpv == -PV).all():
            # The second periodic vector is used -- multiply by exp[i * phi]
            twistangle = -thetatwist

    return twistangle


def calc_dynamical_matrix_gyros_stretched(xy, BL, NL, KL, OmK, Omg, bL, PVx, PVy):
    """Calculates the matrix for finding the normal modes of the system.
    Assumes linearity of force with distance from rest positions.
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body axis 3.

    Parameters
    ----------
    xy : NP x dim array
    NL : NP x NN array
    KL : NP x NN array
    OmK : float or NP x NN array
        k*l**2/I3*omega3,
    Omg : float or NP x 1 array
        l*gn/I3*omega3, gravitational precession frequencies
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    """
    try:
        NP, NN = np.shape(NL)
    except ValueError:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    # Unpack periodic boundary vectors
    if PVx is None or PVy is None:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    # Get distpts from xy
    distpts = le.BL2BM(xy, NL, BL, KL=KL, PVx=PVx, PVy=PVy)

    # gBM is the rest bond lengths, determined from bL
    gBM = le.bL2BM(bL, BL, NL, KL)

    # Get strain of this bond from lattice's distpts and glat's BM
    displ = (distpts - gBM)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        omg = Omg[i]  # grav frequency for this connection
        for nn in range(NN):
            # the number of the gyroscope i is connected to (particle j)
            ni = NL[i, nn]
            # true connection?
            k = KL[i, nn]
            # spring frequency for this connection
            omk = OmK[i, nn]

            if abs(k) > 1e-12:
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
                diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
                alphaij = 0.

                # alphaij is the angle from particle i to particle j
                if abs(k) > 0:
                    alphaij = np.arctan2(diffy, diffx)

                # for periodic systems, KL is -1 for particles on opposing boundaries
                # if k == -1  :
                #     alphaij = (np.pi + alphaij)%(2*np.pi)

                # What is this for?
                if k == -2:  # will only happen on first or last gyro in a line
                    if i == 0 or i == (NP - 1):
                        print i, '--> NL=-2 for this particle'
                        yy = np.where(KL[i] == 1)
                        dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
                        dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
                        al = (np.arctan2(dy, dx)) % (2 * np.pi)
                        alphaij = np.pi - al
                        if i == 1:
                            alphaij = np.pi - (45. * np.pi / 180.)
                        else:
                            alphaij = - (45. * np.pi / 180.)

                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)

                if abs(Cos) < 10E-8:
                    Cos = 0.0

                if abs(Sin) < 10E-8:
                    Sin = 0.0

                Cos2 = Cos ** 2
                Sin2 = Sin ** 2
                CosSin = Cos * Sin

                if abs(k) > 1e-7:
                    # There is a true connection here, so update dynamical matrix elements
                    strain = displ[i, nn] / np.sqrt(diffx ** 2 + diffy ** 2)
                    kparai = omk
                    kparaj = omk
                    kperpi = omk * strain
                    kperpj = omk * strain

                    # The parallel and perpendicular terms on particle p for displacement of particle p
                    kxixi = kparai * Cos2 + kperpi * Sin2
                    kyixi = (kparai - kperpi) * CosSin
                    kxiyi = (kparai - kperpi) * CosSin
                    kyiyi = kperpi * Cos2 + kparai * Sin2

                    # The parallel and perpendicular terms on particle p for displacement of particle q
                    kxixj = -kparaj * Cos2 - kperpj * Sin2
                    kyixj = (-kparaj + kperpj) * CosSin
                    kxiyj = (-kparaj + kperpj) * CosSin
                    kyiyj = -kperpj * Cos2 - kparaj * Sin2

                else:
                    kxixi = 0
                    kyixi = 0
                    kxiyi = 0
                    kyiyi = 0

                    kxixj = 0
                    kyixj = 0
                    kxiyj = 0
                    kyiyj = 0

                # print 'express = ', k * kyixi
                # x components of particle i
                # -Fy --> xj CosSin + yj Sin2 - xi CosSin -yi Sin2
                #         + strain ( -xj CosSin + yj Cos2 + xi CosSin - yi Cos2 )
                # Dependence of dxi on xi
                M1[2 * i, 2 * i] += -k * kyixi  # - (kpar CosSin - kperp CosSin)
                # --> - omk * CosSin
                # limit of no strain --> - omk * CosSin
                # Dependence of dxi on yi
                M1[2 * i, 2 * i + 1] += -k * kyiyi  # - (kperp Cos2 + kpar Sin2)
                # limit of no strain --> - omk * Sin2
                # Dependence of dxi on xj
                M1[2 * i, 2 * ni] += -k * kyixj  # kpar CosSin - kperp CosSin
                # limit of no strain --> omk * CosSin
                # Dependence of dxi on yj
                M1[2 * i, 2 * ni + 1] += -k * kyiyj  # kpar Sin2 + kperp Cos2
                # limit of no strain --> omk * Sin2

                # y components of particle i
                # Fx --> -xj Cos2 - yj Cos Sin + xi Cos2 + yi CosSin
                #         + strain ( -xj Sin2 + yj CosSin + xi Sin2 - yi CosSin )
                # Dependence of dyi on xi
                M1[2 * i + 1, 2 * i] += k * kxixi  # kpar Cos2 + kperp Sin2
                # limit of no strain --> omk * Cos2
                # Dependence of dyi on yi
                M1[2 * i + 1, 2 * i + 1] += k * kxiyi  # kpar CosSin - kperp CosSin
                # limit of no strain --> omk * CosSin
                # Dependence of dyi on xj
                M1[2 * i + 1, 2 * ni] += k * kxixj  # -kpar Cos2 - kperp Sin2
                # limit of no strain --> - omk * Cos2
                # Dependence of dyi on yj
                M1[2 * i + 1, 2 * ni + 1] += k * kxiyj  # -kpar CosSin + kperp CosSin
                # limit of no strain --> - omk * CosSin

                # Check limits
                #  (x components)
                # M1[2 * i, 2 * i] += -omk * CosSin  # dxi - dxi
                # M1[2 * i, 2 * i + 1] += -omk * Sin2  # dxi - dyi
                # M1[2 * i, 2 * ni] += omk * CosSin  # dxi - dxj
                # M1[2 * i, 2 * ni + 1] += omk * Sin2  # dxi - dyj
                #
                # # (y components)
                # M1[2 * i + 1, 2 * i] += omk * Cos2  # dyi - dxi
                # M1[2 * i + 1, 2 * i + 1] += omk * CosSin  # dyi - dyi
                # M1[2 * i + 1, 2 * ni] += -omk * Cos2  # dyi - dxj
                # M1[2 * i + 1, 2 * ni + 1] += -omk * CosSin  # dyi - dyj

                # pinning/gravitational matrix
                M2[2 * i, 2 * i + 1] = - omg
                M2[2 * i + 1, 2 * i] = omg

    # self.pin_array.append(2*pi*1*extra_factor)
    # Assumes that b=0, c=1 so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    matrix = M1 + M2

    return matrix


def calc_dynamical_matrix_psi(xy, NL, KL, OmK, Omg, PVx, PVy):
    """Calculates the matrix for finding the normal modes of the system
    OmK and Omg are signed with reference to positive for b=0, c=1.
    In other words, if OmK and Omg are positive, then the gyros are hanging and spinning with dir aligned with body axis 3.

    Parameters
    ----------
    xy : NP x 2 float array
        2D positions of points (positions x,y). Row i is the x,y position of the ith particle. In other words, the index
        of each particle is given by its row, using zero-indexing (the first particle is particle 0).
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    OmK : float or NP x NN array
        k*l**2/I3*omega3, if 'auto', then computes quantity from params
    Omg : float or NP x 1 array
        l*gn/I3*omega3, if 'auto', then computes quantity from params
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i

    Returns
    ----------
    D : 2*NP x 2*NP complex array
        Dynamical matrix in psi^R and psi^L basis
    """
    RuntimeError('This seems not quite right somehow. The discrepancy is small...')
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP), dtype=complex)
    M2 = np.zeros((2 * NP, 2 * NP), dtype=complex)

    # print ' Supplied Omg = ', Omg
    # print ' glatfns: Supplied PVx = ', PVx

    print 'Constructing dynamical matrix...'
    for ii in range(NP):
        omg = Omg[ii]  # grav frequency for this particle

        # pinning/gravitational matrix
        M2[ii, ii] += omg
        M2[NP + ii, NP + ii] += -omg

        for nn in range(NN):
            # the number of the gyroscope i is connected to (particle j)
            ni = NL[ii, nn]
            # If particle nn is connected to ni, then |k| > 0 --> it is a true connection
            k = KL[ii, nn]
            # spring frequency for this connection
            omk = OmK[ii, nn]

            # if ii == 2:
            #     print 'NL = ', NL
            #     print 'KL = ', KL
            #     print 'OmK = ', OmK
            #     sys.exit()

            if abs(k) > 1e-10:
                # There is a true connection, so update dynamical matrix
                if PVx is None and PVy is None:
                    diffx = xy[ni, 0] - xy[ii, 0]
                    diffy = xy[ni, 1] - xy[ii, 1]
                else:
                    diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                    diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]

                alphaij = np.arctan2(diffy, diffx)

                ei2theta = np.exp(1j * 2. * alphaij)
                eni2thet = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                M1[ii, ii] += omk  # psi_i L
                M1[ii, ni] += -omk  # psi_j L

                # (psi_L psi_R components) top right chunk
                M1[ii, NP + ii] += omk * ei2theta  # psi_i R
                M1[ii, NP + ni] += -omk * ei2theta  # psi_j R

                # (psi_R psi_L components) bottom left chunk
                M1[NP + ii, ii] += -omk * eni2thet  # psi_i L
                M1[NP + ii, ni] += omk * eni2thet  # psi_j L

                # (psi_R psi_R components) bottom right chunk
                M1[NP + ii, NP + ii] += -omk  # psi_i R
                M1[NP + ii, NP + ni] += omk  # psi_j R

    # Assumes that b=0, c=1 (ie hanging gyroscope with angular momentum pointing down) so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    # d/dt(psiL, psiR) = - spring (-Fy,Fx)  - pin (dY,-dX)
    D = -0.5 * M1 - M2

    # divide by i (ie sqrt(-1))
    return D * (-1j)


def OmK_spec_gridlines(lat, lp, gridspacing, maxval, strongbond_val, weakbond_val, check=False):
    """Build OmK (matrix of spring frequencies) with weakened bonds from a supplied lattice.

    Parameters
    ----------
    lat : lepm.lattice_class.Lattice instance
        The lattice on which to build OmK
    gridspacing : float
        The spacing between gridlines for which to weaken bonds
    maxval : float
        The extremal value of x or y coordinate in the network, to use for finding bonds to weaken
    strongbond_val : float
        The spring frequency for the stronger, unaltered bonds
    weakbond_val : float
        The spring frequency for the altered bonds (assumed weaker)
    check : bool
        Display the pattern of strong and weak bonds as blue and red before continuing

    Returns
    -------
    OmK : N x max(#NN) float array
        bond frequencies matching the KL and NL arrays, with bonds weakened along gridlines
    """
    gridright = np.arange(gridspacing, maxval, gridspacing)
    gridleft = -gridright
    gridvals = np.hstack((gridleft, 0, gridright))
    # Draw grid
    gridlinesH = np.array([[-maxval, gridv, maxval, gridv] for gridv in gridvals])
    gridlinesV = np.array([[gridv, -maxval, gridv, maxval] for gridv in gridvals])
    gridsegs = np.vstack((gridlinesH, gridlinesV))
    print 'np.shape(gridlines) = ', np.shape(gridsegs)
    # Make the bond linesegments
    xy = lat.xy
    pvx = lat.PVx
    pvy = lat.PVy
    NL = lat.NL

    if lp['periodicBC']:
        bondsegs = np.array([[xy[b[0], 0],
                              xy[b[0], 1],
                              xy[b[1], 0] + pvx[b[0], np.where(NL[b[0]] == b[1])[0]],
                              xy[b[1], 1] + pvy[b[0], np.where(NL[b[0]] == b[1])[0]]] for b in np.abs(lat.BL)])
    else:
        bondsegs = np.array([[xy[b[0], 0],
                              xy[b[0], 1],
                              xy[b[1], 0],
                              xy[b[1], 1]] for b in np.abs(lat.BL)])  # get crossings

    # print 'gridsegs = ', gridsegs
    does_intersect = lsegs.linesegs_intersect_linesegs(bondsegs, gridsegs)

    if check:
        kk = 0
        xinds = [0, 2]
        yinds = [1, 3]
        aBL = np.abs(lat.BL)
        for bb in bondsegs:
            if does_intersect[kk]:
                plt.plot(bb[xinds], bb[yinds], 'r.-')
            else:
                plt.plot(bb[xinds], bb[yinds], 'b.-')
            kk += 1
        plt.show()

        kk = 0
        for bb in aBL:
            if does_intersect[kk]:
                plt.plot(xy[bb, 0], xy[bb, 1], 'r.-')
            else:
                plt.plot(xy[bb, 0], xy[bb, 1], 'b.-')
            kk += 1
        plt.show()

    # Build OmK as it would be constructed without the OmKspec specification
    # In a copy of lp, kill the OmK specification to avoid getting stuck in a loop
    from lepm.gyro_lattice_class import GyroLattice
    lp_copy = copy.deepcopy(lp)
    lp_copy['OmKspec'] = None
    lp_copy['Omk'] = strongbond_val
    tmp_glat = GyroLattice(lat, lp_copy)
    OmK = copy.deepcopy(tmp_glat.OmK)

    # Now weaken the gridline bonds. Note that we take the absolute value of BL to allow for periodic networks
    print 'Altering weak bonds --> ', weakbond_val
    # print 'lat.BL[does_intersect] = ', lat.BL[does_intersect]
    for bond in np.abs(lat.BL[does_intersect]):
        OmK[bond[0], np.where(lat.NL[bond[0]] == bond[1])[0]] = weakbond_val
        OmK[bond[1], np.where(lat.NL[bond[1]] == bond[0])[0]] = weakbond_val

    # check it
    if check:
        OmKv = le.KL2kL(lat.NL, OmK, lat.BL)
        test = -np.ones(len(lat.BL[:, 0]), dtype=float)
        test[does_intersect] = weakbond_val
        print 'test = ', test
        le.display_lattice_2D(lat.xy, lat.BL, NL=lat.NL, KL=lat.KL, PVxydict=lat.PVxydict,
                              PVx=lat.PVx, PVy=lat.PVy, bs=test,
                              title='', xlimv=None, ylimv=None, climv=0.1, colorz=True, ptcolor=None, ptsize=10,
                              close=True, colorpoly=False, viewmethod=False, labelinds=False,
                              colormap='seismic', bgcolor='#d9d9d9', axis_off=False, fig=None, ax=None, linewidth=0.0,
                              edgecolors=None, check=True)
        le.display_lattice_2D(lat.xy, lat.BL, NL=lat.NL, KL=lat.KL, PVxydict=lat.PVxydict,
                              PVx=lat.PVx, PVy=lat.PVy, bs=OmKv,
                              title='', xlimv=None, ylimv=None, climv=0.1, colorz=True, ptcolor=None, ptsize=10,
                              close=True, colorpoly=False, viewmethod=False, labelinds=False,
                              colormap='seismic', bgcolor='#d9d9d9', axis_off=False, fig=None, ax=None, linewidth=0.0,
                              edgecolors=None, check=True)
    return OmK


def build_OmK(lattice, lp, OmK, eps=1e-9):
    """Construct OmK from supplied OmK (may be None or 'none') and lattice parameter dictionary lp

    Parameters
    ----------
    lattice : lattice_class.Lattice() instance
    OmK : N x max(#NN) float array or None or 'none'
        bond frequencies matching the KL and NL arrays, possibly with bonds weakened along gridlines or other OmKspec
    lp : dict
        lattice parameter dictionary
    eps : float
        minimum value of abs(KL) to be considered true connection, used in OmKspec applications

    Returns
    -------
    OmK : N x max(#NN) float array
        bond frequencies matching the KL and NL arrays
    lp_Omk : float
        value for key 'Omk' to be added to lp
    lp_meshfn_exten : str
        value for key 'meshfn_exten' to be added to lp
    """
    if OmK == 'auto' or OmK is None:
        # Check if a string specifier for OmK is given. If it can be understood, create that bond strength pattern.
        # If it is an unknown string, then OmK must be supplied.
        if 'OmKspec' in lp:
            if lp['OmKspec'] not in ['', 'none', None]:
                if 'gridlines' in lp['OmKspec']:
                    # Weaken bonds along gridlines
                    # Here, OmKspec must be of the form:
                    # 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                    # 'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
                    # 'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
                    spec = lp['OmKspec'].split('gridlines')[1]
                    gridspacing = float(spec.split('strong')[0].replace('p', '.'))
                    strong = float(spec.split('strong')[1].split('weak')[0].replace('p', '.').replace('n', '-'))
                    weak = float(spec.split('weak')[1].replace('p', '.').replace('n', '-'))
                    maxval = max(np.max(np.abs(lattice.xy[:, 0])), np.max(np.abs(lattice.xy[:, 1]))) + 1
                    OmK = OmK_spec_gridlines(lattice, lp, gridspacing, maxval, strong, weak)
                elif 'PVbondfactor' in lp['OmKspec']:
                    # reduce/magnify hoppings across periodic vectors by the float given as string in PVfactor
                    OmK = lp['Omk'] * np.abs(lattice.KL.astype('float'))
                    # print 'glatfns: OmK = ', OmK
                    # print 'glatfns: lattice.KL = ', lattice.KL
                    # print lp['OmKspec']
                    pvfactor = float(lp['OmKspec'].split('PVbondfactor')[-1].replace('p', '.').replace('n', '-'))
                    # print 'glatfns: pvfactor = ', pvfactor
                    OmK[np.where(lattice.KL < -0.1)] *= pvfactor
                    # print 'glatfns: OmK = ', OmK
                    # raise RuntimeError('here')
                elif 'union' in lp['OmKspec']:
                    # Here, OmKspec must be of the form 'unionn0p100in0p2000 to make bonds connecting points a distance
                    # of less than 0.2 away from each other have strength -0.100
                    # Tune the bond strength of bonds connecting points which lie at the same location in space
                    # Now prepare the value to which to tune these bonds
                    specval = sf.str2float(lp['OmKspec'].split('union')[-1].split('in')[0])
                    specdist = sf.str2float(lp['OmKspec'].split('union')[-1].split('in')[-1])
                    # First, find points that lie at the same location in space.
                    dists = dist_pts(lattice.xy, lattice.xy, square_norm=True) + np.identity(len(lattice.xy))
                    rowcols = np.where(dists < specdist)
                    # print 'dists = ', dists
                    # print 'specdist = ', specdist
                    # print 'dists < specdist = ', dists < specdist
                    pts, nbrs = rowcols[0], rowcols[1]
                    # print 'rowcols = ', rowcols
                    # print 'OmKspec = ', lp['OmKspec']
                    # print 'specval = ', specval
                    # Place this value for bonds between points on the same site
                    OmK = lp['Omk'] * np.abs(lattice.KL.astype('float'))
                    for (pt, nbr) in zip(pts, nbrs):
                        # Find where NL[pt] has neighbor nbrs, tune that OmK to the specified value
                        omkcol = np.where(np.logical_and(lattice.NL[pt] == nbr, np.abs(lattice.KL[pt]) > eps))[0]
                        OmK[pt, omkcol] = specval
                    # print 'BL = ', lattice.BL
                    # print 'NL = ', lattice.NL
                    # print 'KL = ', lattice.KL
                    # print 'OmK = ', OmK
                    # sys.exit()
                else:
                    raise RuntimeError('OmKspec in lp cannot be translated into OmK, must supply OmK or edit to' +
                                       ' interpret given OmKspec')
                lp_meshfn_exten = '_OmKspec' + lp['OmKspec']
                if (OmK == OmK[np.nonzero(OmK)][0] * lattice.KL).all():
                    # This is just the boring case where OmK is not special (all bonds are equal).
                    lp_Omk = OmK[np.nonzero(OmK)][0]
                    if lp_Omk != -1.0:
                        lp_meshfn_exten = '_Omk' + sf.float2pstr(lp['Omk'])
                    else:
                        lp_meshfn_exten = ''
                elif (OmK != lp['Omk'] * np.abs(lattice.KL)).any():
                    lp_Omk = -5000
                else:
                    lp_Omk = lp['Omk']
                done = True
            else:
                # OmKspec is in lp, but it is None or 'none'. Try a different method to obtain OmK.
                done = False
        else:
            # OmKspec is not in lp. Try a different method to obtain OmK.
            done = False

        if not done:
            if 'Omk' in lp:
                print 'gyro_lattice_class: using Omk from lp...'
                OmK = lp['Omk'] * np.abs(lattice.KL)
                lp_Omk = lp['Omk']
                if lp_Omk != -1.0:
                    lp_meshfn_exten = '_Omk' + sf.float2pstr(lp['Omk'])
                else:
                    lp_meshfn_exten = ''
            else:
                print 'giving OmK the default value of -1s...'
                OmK = -1.0 * np.abs(lattice.KL)
                lp_Omk = -1.0
                lp_meshfn_exten = ''
    else:
        # This is the case where OmK is specified. Pass it along to output and discern what to give for lp_meshfn_exten
        # Output OmK <-- input OmK
        # Use given OmK to define lp_Omk (for lp['Omk']) and lp[meshfn_exten].
        # If Omk is a key in lp, correct it if it does not match supplied OmK
        if 'Omk' in lp:
            if (OmK == lp['Omk'] * np.abs(lattice.KL)).any():
                # This is the case that OmK = Omk * np.abs(KL). Give a nontrivial meshfn_exten if Omk != -1.0
                lp_Omk = lp['Omk']
                if lp_Omk != -1.0:
                    lp_meshfn_exten = '_Omk' + sf.float2pstr(lp['Omk'])
                else:
                    lp_meshfn_exten = ''
            else:
                # OmK given is either some constant times KL, or something complicated
                # If it is a constant, discern that constant and update lp
                kinds = np.nonzero(OmK)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in OmK. Check if all the same
                    OmKravel = OmK.ravel()
                    KLravel = lattice.KL.ravel()
                    if (OmKravel[np.where(abs(KLravel))] == OmKravel[np.where(abs(KLravel))[0]]).all():
                        print 'Updating lp[Omk] to reflect specified OmK, since OmK = constant * np.abs(KL)...'
                        lp_Omk = OmKravel[np.where(abs(KLravel))[0]]
                    else:
                        # OmK is something complicated, so tell meshfn_exten that OmK is specified.
                        lp_meshfn_exten = '_OmKspec'
                        if 'OmKspec' in lp:
                            lp_meshfn_exten = '_Omkspec' + lp['OmKspec']
                        lp_Omk = -5000

                else:
                    lp_Omk = 0.0
                    lp_meshfn_exten = '_Omk0p00'
        else:
            # Check if the values of all elements are identical
            kinds = np.nonzero(OmK)
            if len(kinds[0]) > 0:
                # There are some nonzero elements in OmK. Check if all the same
                value = OmK[kinds[0][0], kinds[1][0]]
                if (OmK[kinds] == value).all():
                    lp_Omk = value
                else:
                    lp_Omk = -5000
            else:
                lp_Omk = 0.0
                lp_meshfn_exten = '_Omk0p00'

    return OmK, lp_Omk, lp_meshfn_exten


def build_OmKspec_list(gridspacing, strongbond_val, weakbond_vals):
    """Build up OmKspec according to the format:
    'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
    'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
    'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')


    Parameters
    ----------
    gridspacing : float
    strong_bondval : float
    weak_bondvals : string, float, or numpy array
        values for the spring frequencies for the altered (weakened) springs

    Returns
    -------
    OmKspecList : list of strings
        list of string specifiers for a GyroLattice's OmK
    """
    OmKspecList = []
    if isinstance(weakbond_vals, str):
        if '/' in weakbond_vals or ':' in weakbond_vals:
            weakarr = sf.string_sequence_to_numpy_array(weakbond_vals, dtype=float)
            for weak in weakarr:
                OmKspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                                   'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') + \
                                    'weak{0:0.3f}'.format(weak).replace('.', 'p').replace('-', 'n'))
        else:
            OmKspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                               'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') + \
                               'weak{0:0.3f}'.format(float(sf.str2float(weakbond_vals, ndigits=3))).replace('.', 'p').replace('-', 'n'))
    elif isinstance(weakbond_vals, np.ndarray):
        for weak in weakarr:
            OmKspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                               'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') + \
                               'weak{0:0.3f}'.format(weak).replace('.', 'p').replace('-', 'n'))

    elif isinstance(weakbond_vals, float):
        OmKspecList.append('gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
                           'strong{0:0.3f}'.format(strongbond_val).replace('.', 'p').replace('-', 'n') + \
                           'weak{0:0.3f}'.format(weakbond_vals).replace('.', 'p').replace('-', 'n'))
    return OmKspecList


def ascribe_absites(lat):
    """Alternatingly define A-sites and B-sites (ABsites) for particles (typically with three nearest neighbors)

    Parameters
    ----------
    lat : Lattice class instance
        the lattice for which to find ABsites
    """
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

        print 'glatfns.ascribe_absites(): todo = ', todo
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

    return asites, bsites


def load_pin_spec(glat):
    """Load Omg values given Omgspec string in glat.lp['Omgspec']

    Parameters
    ----------
    glat : GyroLattice instance
        the gyro lattice for which to load the pinning configuration

    Returns
    -------
    omg_values : NP x 1 float array
        The pinning frequencies for this Omg specification in glat.lp['Omgspec']
    """
    meshfn = glat.lp['meshfn']
    with h5py.File(dio.prepdir(meshfn) + 'omg_configs.hdf5', "r") as fi:
        specname = glat.lp['Omgspec']
        inhdf5 = specname in fi.keys()
        if inhdf5:
            omg_values = fi[specname][:]
            load_from_txt = False
            print 'glatfns.load_vpin_gauss(): loaded Omgspec from hdf5, returning '
        else:
            load_from_txt = True

    if load_from_txt:
        print 'could not find Omgspec pinning config from hdf5, opening pinning configs from txt...'
        specname = 'Omg_' + glat.lp['Omgspec']
        omg_values = np.loadtxt(dio.prepdir(meshfn) + specname + '.txt')

    # print 'omg_values = ', omg_values
    # print 'glatfns: exiting'
    # sys.exit()

    return omg_values


def load_vpin_gauss(glat):
    """Load the disorder configuration specified by glat.lp['pinconf']

    Parameters
    ----------
    glat : GyroLattice instance
        the gyro lattice for which to load the pinning configuration due to disorder (to be added to glat.Omg)

    Returns
    -------
    add : len(NP) float array
        the offsets to the pinning frequencies determined by deltacorrelated disorder, with magnitude determined by
        glat.lp['V0_pin_gauss'] --- ie, gaussian of standard deviation given by glat.lp['V0_pin_gauss']
    """
    meshfn = glat.lp['meshfn']
    h5omg_file = dio.prepdir(meshfn) + 'omg_configs.hdf5'
    if glob.glob(h5omg_file):
        if abs(glat.lp['V0_pin_gauss']) > 1e-10:
            with h5py.File(h5omg_file, "r") as fi:
                dcdname = 'pinV_gauss_conf{0:04d}'.format(int(glat.lp['pinconf']))
                inhdf5 = dcdname in fi.keys()
                if inhdf5:
                    addomg = fi[dcdname][:]
                    add = glat.lp['V0_pin_gauss'] * addomg
                    load_from_txt = False
                    print 'glatfns.load_vpin_gauss(): loaded Vpin_gauss from hdf5, returning '
                else:
                    add = np.zeros_like(glat.lattice.xy[:, 0])
                    addomg = copy.deepcopy(add)
                    load_from_txt = True
        else:
            add = np.zeros_like(glat.lattice.xy[:, 0])
            addomg = copy.deepcopy(add)
            load_from_txt = False
    else:
        load_from_txt = True

    if load_from_txt:
        print 'could not find vpin pinning config from hdf5, opening pinning configs from txt...'
        dcdname = 'pinV_conf{0:04d}'.format(int(glat.lp['pinconf']))
        addomg = np.loadtxt(dio.prepdir(meshfn) + dcdname + '.txt')
        add = glat.lp['V0_pin_gauss'] * addomg

    return add, addomg


def load_omgspec(glat):
    """

    Parameters
    ----------
    glat

    Returns
    -------

    """
    if 'dab' in glat.lp['Omgspec'] and 'posneg' in glat.lp['Omgspec']:
        dabval = float(glat.lp['Omgspec'].split('dab')[-1].split('posneg')[0].replace('p', '.'))
        omg_values = glat.lp['Omg'] * np.ones_like(glat.lattice.xy[:, 0])
        omg_values[np.where(glat.lattice.xy[:, 1] > 0)] += dabval
        omg_values[np.where(glat.lattice.xy[:, 1] < 0)] -= dabval
    elif 'dab' in glat.lp['Omgspec'] and 'union' in glat.lp['Omgspec']:
        dabval = float(glat.lp['Omgspec'].split('dab')[-1].split('union')[0].replace('p', '.'))
        omg_values = glat.lp['Omg'] * np.ones_like(glat.lattice.xy[:, 0])
        omg_values[np.where(glat.lattice.xy[:, 1] > 0)] += dabval
        omg_values[np.where(glat.lattice.xy[:, 1] < 0)] -= dabval
        # Here, OmKspec must be of the form 'union0p2000 to make sites that are less than 0.2 away from each other have
        # the same pinning strength.
        # Now prepare the value to which to tune these sites
        print 'glat.lp[Omgspec].split(union)[-1] = ', glat.lp['Omgspec'].split('union')[-1]
        specdist = sf.str2float(glat.lp['Omgspec'].split('union')[-1])
        # First, find points that lie at the same location in space.
        dists = dist_pts(glat.lattice.xy, glat.lattice.xy, square_norm=True) + np.identity(len(glat.lattice.xy))
        rowcols = np.where(dists < specdist)
        pts, nbrs = rowcols[0], rowcols[1]
        # Keep track of which sites have been labeled as A or B
        done = np.zeros(len(glat.lattice.xy[:, 0]), dtype=int)
        pt = pts[0]
        nbr = nbrs[np.where(pts==pt)[0]]

        # Prepare array
        omg_values = glat.lp['Omg'] * np.ones_like(glat.lattice.xy[:, 0])
        addomg = np.zeros_like(omg_values)
        abval = dabval
        # Find where NL[pt] has neighbor nbrs, tune that OmK to the specified value
        addomg[pt] = abval
        addomg[nbr] = abval
        asites, bsites = [pt], []
        for nn in nbr:
            asites.append(nn)

        # mark as ID'd
        done[pt] = 1
        done[nbr] = 1
        if glat.lp['check']:
            print 'pts = ', pts
            print 'nbrs = ', nbrs
            print 'done = ', done
            print 'asites = ', asites
            print 'bsites = ', bsites

        while not done.all():
            abval = -abval
            new_info = False
            # next do these guys' neighbors (the neighbors of the sites we just looked at)
            if glat.lp['check']:
                print 'pt = ', pt
                print 'nbr = ', nbr
            
            for site in np.hstack((np.array([pt]), nbr)):
                # Allow a mechanism to catch the case where all neighbors have been accounted for
                for neibr in glat.lattice.NL[site, np.where(np.abs(glat.lattice.KL[pt]))[0]]:
                    if not done[neibr]:
                        addomg[neibr] = abval
                        # do nearby sites of neibr, which we call 'group'
                        group = nbrs[np.where(pts == neibr)[0]]
                        addomg[group] = abval
                        # mark these as done
                        done[neibr] = 1
                        done[group] = 1
                        if abval > 0:
                            asites.append(neibr)
                            for nn in group:
                                asites.append(nn)
                        else:
                            bsites.append(neibr)
                            for nn in group:
                                bsites.append(nn)

                        if glat.lp['check']:
                            # check it
                            sm = plt.scatter(glat.lattice.xy[:, 0], glat.lattice.xy[:, 1], c=addomg, lw=0, cmap='seismic')
                            # plt.colorbar(sm)
                            plt.pause(.1)

                        pt = neibr
                        nbr = group
                        new_info = True

            # Catch the case where all the neighbors have already been done for the current site
            if not new_info:
                print 'flipping coin to jump to new previously done site'
                coinflip = np.random.random()
                if coinflip < 0.5:
                    # pick an A site at random
                    asite_ind = np.random.choice(len(asites), 1)[0]
                    pt = asites[asite_ind]
                    nbr = nbrs[np.where(pts == pt)[0]]
                    # flib abval to be Asite abval since the next treated sites will be B sites
                    abval = dabval
                    # print 'asites = ', asites
                    # print 'glatfns: picked asite: pt = ', pt
                    # print 'glatfns: picked asite: nbr = ', nbr
                else:
                    # pick a B site at random
                    bsite_ind = np.random.choice(len(bsites), 1)[0]
                    pt = bsites[bsite_ind]
                    nbr = nbrs[np.where(pts == pt)[0]]
                    # flib abval to be Bsite abval since the next treated sites will be A sites
                    abval = - dabval
                    # print 'bsites = ', bsites
                    # print 'glatfns: picked bsite: pt = ', pt
                    # print 'glatfns: picked bsite: nbr = ', nbr

        omg_values += addomg

    # print 'inspecting coloration of AB neighbors for Omgspec'
    # sm = plt.scatter(glat.lattice.xy[:, 0], glat.lattice.xy[:, 1], c=addomg, lw=0, cmap='seismic')
    # plt.show()
    # sys.exit()

    return omg_values


def load_abdelta(glat):
    """Load the disorder configuration specified by glat.lp['pinconf']

    Parameters
    ----------
    glat : GyroLattice instance
        the gyro lattice for which to load the pinning configuration due to disorder (to be added to glat.Omg)

    Returns
    -------
    add : len(NP) float array
        the offsets to the pinning frequencies determined by deltacorrelated disorder, with magnitude determined by
        glat.lp['V0_pin_gauss'] --- ie, gaussian of standard deviation given by glat.lp['V0_pin_gauss']
    """
    meshfn = glat.lp['meshfn']
    with h5py.File(dio.prepdir(meshfn) + 'omg_configs.hdf5', "r") as fi:
        if np.abs(glat.lp['ABDelta']) > 1e-10:
            abname = 'ABDelta'
            inhdf5 = abname in fi.keys()
            if inhdf5:
                add = glat.lp['ABDelta'] * fi[abname][:]
                load_from_txt = False
            else:
                add = np.zeros_like(glat.lattice.xy[:, 0])
                load_from_txt = True
        else:
            add = np.zeros_like(glat.lattice.xy[:, 0])
            load_from_txt = False

    if load_from_txt:
        print 'glatfns: could not find abdelta pinning config from hdf5, opening pinning configs from txt...'
        abname = 'ABDelta'
        add = glat.lp['ABDelta'] * np.loadtxt(dio.prepdir(meshfn) + abname + '.txt')

    return add


def save_vpin_gauss(glat, addomg, force_hdf5=True):
    """"""
    meshfn = glat.lp['meshfn']
    dcdname = 'pinV_gauss_conf{0:04d}'.format(int(glat.lp['pinconf']))
    if force_hdf5:
        # First check that doesn't already exist
        hdf5fn = dio.prepdir(meshfn) + 'omg_configs.hdf5'
        h5io.dset_in_hdf5(dcdname, hdf5fn)
        # Save to hdf5
        h5io.save_dset_hdf5(addomg, dcdname, hdf5fn, overwrite=False)
        print 'saved ' + dcdname + ' to hdf5'
    else:
        # save to text file, as pinV_conf0023.txt
        np.savetxt(meshfn + 'pinV_gauss_conf{0:04d}'.format(int(glat.lp['pinconf'])) + '.txt', addomg)
        print 'saved ' + dcdname + '.txt to disk'


def save_abdelta(glat, addomg, force_hdf5=True):
    """"""
    meshfn = glat.lp['meshfn']
    abname = 'ABDelta'
    if force_hdf5:
        # First check that doesn't already exist
        hdf5fn = dio.prepdir(meshfn) + 'omg_configs.hdf5'
        h5io.dset_in_hdf5(dcdname, hdf5fn)
        # Save to hdf5
        h5io.save_dset_hdf5(addomg, abname, hdf5fn, overwrite=False)
        print 'saved ' + abname + ' to hdf5'
    else:
        # save to text file, as pinV_conf0023.txt
        np.savetxt(meshfn + 'ABDelta.txt', addomg)
        print 'saved ' + abname + '.txt to disk'


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
    # Only look at half the spectrum, and normalize all the states
    # eigval2 = eigval[0:int(len(eigval) * 0.5)]
    # eigvect2 = eigvect[0:int(len(eigvect) * 0.5)]

    # METHOD 1
    eta = eps * np.median(np.abs(np.diff(eigval)))
    ldos_tmp = np.zeros((len(eigval), int(0.5 * len(eigval))))
    ii = 0
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
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

            # If desired, we can use the full eigenvalue, but only half is needed.
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

    ldos = np.array([np.sqrt(ldos_tmp[2*ii] ** 2 + ldos_tmp[2*ii + 1] ** 2) for ii in range(len(inds))])

    return ldos


def calc_magevecs(eigvect, basis='XY'):
    """Compute the magnitude of the second half of all eigenvectors, by norming their x and y components in quad

    Parameters
    ----------
    eigvect : 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    basis : str ('XY', 'psi')
            string specifier for which basis the eigvect is constructed in

    Returns
    -------
    magevecs : #particles x #particles float array
        The magnitude of the upper half of eigenvectors at each site. magevecs[i, j] is the magnitude of the i+NP
        normal mode at site j.
    """
    if basis == 'XY':
        if len(np.shape(eigvect)) == 1:
            raise RuntimeError('Supplied eigvect must be 2d')

        halfevec = eigvect[int(len(eigvect) * 0.5):]

        # Contract x and y component of each evect so that len(halfevec[ii]) = NP
        magevecs = np.zeros((len(halfevec), len(halfevec)), dtype=float)
        jj = 0
        for row in halfevec:
            magevecs[jj] = np.array([np.sqrt(np.abs(row[2 * ii]) ** 2 + np.abs(row[2 * ii + 1]) ** 2)
                                    for ii in np.arange(0, int(len(eigvect) * 0.5))])
            jj += 1
    elif basis == 'psi':
        if len(np.shape(eigvect)) == 1:
            raise RuntimeError('Supplied eigvect must be 2d')

        halflen = int(len(eigvect) * 0.5)
        halfevec = eigvect[halflen:]

        # Contract x and y component of each evect so that len(halfevec[ii]) = NP
        magevecs = np.zeros((len(halfevec), len(halfevec)), dtype=float)
        jj = 0
        for row in halfevec:
            magevecs[jj] = np.array([np.sqrt(np.abs(row[ii]) ** 2 + np.abs(row[ii + halflen]) ** 2)
                                     for ii in np.arange(0, halflen)])
            jj += 1
    else:
        raise RuntimeError('Basis not recognized')
    return magevecs


def calc_magevecs_full(eigvect, basis='XY'):
    """Compute the magnitude of the ALL supplied eigenvectors, by norming their x and y components in quad

    Parameters
    ----------
    eigvect : M x 2*N complex array
        eigenvectors or some eigenvectors, in any order
        Typically eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    basis : str ('XY', 'psi')
        string specifier for which basis the eigvect is constructed in

    Returns
    -------
    magevecs : M x #particles float array
        The magnitude of all supplied eigenvectors at each site. magevecs[i, j] is the magnitude of the i+NP
        normal mode at site j.
    """
    if basis == 'XY':
        if len(np.shape(eigvect)) == 1:
            halflen = int(0.5 * len(eigvect))
            magevecs = np.array([np.sqrt(np.abs(eigvect[2 * ii]) ** 2 + np.abs(eigvect[2 * ii + 1]) ** 2)
                                 for ii in np.arange(0, halflen)])
        else:
            # Contract x and y component of each evect so that len(magevec[ii]) = NP
            halflen = int(0.5 * np.shape(eigvect)[1])
            magevecs = np.zeros((np.shape(eigvect)[0], halflen), dtype=float)
            jj = 0
            for row in eigvect:
                magevecs[jj] = np.array([np.sqrt(np.abs(row[2 * ii]) ** 2 + np.abs(row[2 * ii + 1]) ** 2)
                                         for ii in np.arange(0, halflen)])
                jj += 1
    elif basis == 'psi':
        if len(np.shape(eigvect)) == 1:
            halflen = int(0.5 * len(eigvect))
            magevecs = np.array([np.sqrt(np.abs(eigvect[ii]) ** 2 + np.abs(eigvect[ii + halflen]) ** 2)
                                 for ii in np.arange(0, halflen)])
        else:
            # Contract x and y component of each evect so that len(magevec[ii]) = NP
            halflen = int(0.5 * np.shape(eigvect)[1])
            magevecs = np.zeros((np.shape(eigvect)[0], halflen), dtype=float)
            jj = 0
            for row in eigvect:
                magevecs[jj] = np.array([np.sqrt(np.abs(row[ii]) ** 2 + np.abs(row[ii + halflen]) ** 2)
                                         for ii in np.arange(0, halflen)])
                jj += 1
    else:
        raise RuntimeError('Basis not recognized')

    return magevecs


