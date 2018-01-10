import numpy as np
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.lattice_functions_nnn as lfnsnnn
import lepm.data_handling as dh

'''Auxiliary functions for magnetic_gyro_lattice_class.py'''


def create_NL_KL_pv(mglat):
    """Based on interaction range, determine NL, KL, and PVxy. If interaction range is negative or zero, all particles
    interact. If interaction_range is 1, only NNs communicate. If interaction_range is 2, 3, etc, then NNs and NNNs,
    NNs + NNNs + NNNNs, etc interact.

    Parameters
    ----------
    mglat : MagneticGyroLattice

    Returns
    -------
    NL, KL, pv, nl_meshfn_exten
    """
    if 'interaction_range' in mglat.lp:
        if mglat.lp['interaction_range'] > 0:
            if mglat.lp['interaction_range'] == 1:
                nl_meshfn_exten = '_intrange{0:04d}'.format(int(mglat.lp['interaction_range']))
                return mglat.lattice.NL, mglat.lattice.KL, mglat.lattice.PVx, mglat.lattice.PVy, nl_meshfn_exten
            else:
                # Creates NL and KL such intrange-th nearest particles communicate
                if mglat.lp['interaction_range'] == 2 and mglat.lattice.lp['NH'] > 1 and mglat.lattice.lp['NV'] > 1:
                    # Next-nearest neighbors are interacting
                    mglat.lattice.get_nljnnn(attribute=True)
                    NLNNN, KLNNN = mglat.lattice.get_NLNNN_and_KLNNN()
                    pvxnn, pvynn = lfnsnnn.calc_pvnnn(mglat.lattice)
                    NLnns, KLnns = [mglat.lattice.NL, NLNNN], [mglat.lattice.KL, KLNNN]
                    pvxnns, pvynns = [mglat.lattice.PVx, pvxnn], [mglat.lattice.PVy, pvynn]
                    # 'wnei' means with nth neighbors
                    NLwnei, KLwnei, pvxnn, pvynn = lfnsnnn.combine_intnnn_info(NLnns, KLnns, pvxnns, pvynns)
                    nl_meshfn_exten = '_intrange{0:04d}'.format(int(mglat.lp['interaction_range']))
                else:
                    NLwnei, KLwnei, pvxnn, pvynn = \
                        lfnsnnn.calc_combined_intnn_info(mglat.lattice, mglat.lp['interaction_range'])
                    nl_meshfn_exten = '_intrange{0:04d}'.format(int(mglat.lp['interaction_range']))

                # print 'KLwnei = ', KLwnei
                # print 'NLwnei = ', NLwnei
                # print 'in magnetic_gyro_functions, exiting...'
                # sys.exit()
                return NLwnei, KLwnei, pvxnn, pvynn, nl_meshfn_exten
        else:
            # Creates NL and KL such that all particles communicate
            NL, KL, PVx, PVy = lfnsnnn.infinite_range_network_info(mglat.lattice)
            # print 'mgfns: np.shape(NL) = ', np.shape(NL)
            # print 'mgfns: np.shape(KL) = ', np.shape(KL)
            # print 'mgfns: np.shape(PVx) = ', np.shape(PVx)
            # sys.exit()

            return NL, KL, PVx, PVy, ''
    else:
        printstr = 'for now, enforcing that interaction_range should exist in lp.' + \
                   'Can comment this out to allow default of total coupling'
        raise RuntimeError(printstr)
        # Creates NL and KL such that all particles communicate
        NL = np.array([np.setdiff1d(np.arange(len(self.xy)), np.array([ii])) for ii in range(len(self.xy))])
        KL = np.ones_like(NL, dtype=int)
        return NL, KL, ''


def dynamical_matrix_magnetic_gyros(mglat, basis=None):
    """Construct the dynamical matrix for the given GyroLattice.

    Parameters
    ----------
    mglat : MagneticGyroLattice instance
        The MagneticGyroLattice for which to construct the dynamical matrix
    basis : 'XY' or 'psi' or None
        The basis in which to construct the dynamical matrix

    Returns
    -------
    matrix
    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in mglat.lp:
        thetatwist = mglat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in mglat.lp:
        phitwist = mglat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    if basis is None and 'basis' in mglat.lp:
        basis = mglat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        # Rest lengths of springs == distances between particles
        if notwist:
            # not twisted, no stretch, XY basis
            matrix = calc_matrix_magnetic(mglat)
        else:
            # twisted, no stretch, XY basis
            print 'PV = ', mglat.lattice.PV
            print 'thetatwist = ', thetatwist
            print 'phitwist = ', phitwist
            if mglat.lp['periodic_strip']:
                # All periodic bonds are twisted
                matrix = calc_dynamical_matrix_gyros_twist(mglat.lattice.xy, mglat.lattice.NL, mglat.lattice.KL,
                                                           mglat.OmK, mglat.Omg, thetatwist, phitwist,
                                                           mglat.lattice.PVx, mglat.lattice.PVy, mglat.lattice.PV)
            else:
                # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                if 'annulus' in mglat.lp['LatticeTop'] or mglat.lp['shape'] == 'annulus':
                    # Twist bonds in a cut of the annular network
                    twistcut = np.array([0., 0., np.max(mglat.lattice.xy[:, 0]), 0.])
                    thetaKL = form_twistedKL(mglat.lattice.xy, mglat.lattice.BL, mglat.lattice.NL, mglat.lattice.KL,
                                             twistcut)
                    phiKL = np.zeros_like(thetaKL, dtype=int)
                else:
                    raise RuntimeError('Currently only have twistedKL set up for annular samples')

                matrix = calc_dynamical_matrix_gyros_twist_bonds(mglat.lattice.xy, mglat.lattice.NL, mglat.lattice.KL,
                                                                 mglat.OmK, mglat.Omg,
                                                                 thetaKL, phiKL,
                                                                 thetatwist, phitwist,
                                                                 mglat.lattice.PVx, mglat.lattice.PVy,
                                                                 mglat.lattice.PV)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_matrix_magnetic_psi(mglat)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in mglat.lp:
        if mglat.lp['immobile_boundary']:
            boundary = mglat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0

    # print 'mglatfns: np.shape(mglat.xy) = ', np.shape(mglat.xy)
    # print 'mglatfns: np.shape(mglat.NL) = ', np.shape(mglat.NL)
    # print 'mglatfns: np.shape(matrix) = ', np.shape(matrix)
    # sys.exit()
    return matrix


def calc_matrix_magnetic(mlat, eps=1e-10):
    """Given an instance of MagneticGyroLattice, compute its dynamical matrix

    Parameters
    ----------
    mlat : MagneticGyroLattice instance
        the magnetic gyro lattice for which to compute the dynamical matrix
    eps : float
        Threshold absolute value to consider a real number (an element of KL) to be nonzero

    Returns
    -------

    """
    NP_total, NN = mlat.NL.shape
    NP = len(mlat.inner_indices)
    PVx = mlat.PVx
    PVy = mlat.PVy

    if PVx is not None and np.shape(PVx) != np.shape(mlat.NL):
        print 'PVx = ', PVx
        print 'NL = ', mlat.NL
        raise RuntimeError('Shape of PVx does not match NL')

    # Unpack periodic boundary vectors
    if PVx is None and PVy is None:
        PVx = np.zeros((NP_total, NN), dtype=float)
        PVy = np.zeros((NP_total, NN), dtype=float)

    M1 = np.zeros((2 * NP, 2 * NP), dtype=float)
    M2 = np.zeros((2 * NP, 2 * NP), dtype=float)

    xy_in = list(mlat.xy[mlat.inner_indices])

    xy_list = [list(mlat.xy[i]) for i in range(len(mlat.xy))]
    xy_in_list = [list(xy_in[i]) for i in range(len(xy_in))]

    # ignores the outer particles
    if 'bcs' in mlat.lp:
        if mlat.lp['bcs'] == 'free':
            bc_f = True
        else:
            bc_f = False
    else:
        bc_f = False

    km = 1
    # als is (a/l)^2 / mag^3, where a is the interparticle distance

    # print 'mgyrofns: mlat.outer_indices = ', mlat.outer_indices
    # print 'mgyrofns: NL = ', mlat.NL

    in_count = 0
    for i in range(NP_total):
        for nn in range(NN):
            if i in mlat.inner_indices:
                ni = int(mlat.NL[i, nn])  # the number of the gyroscope i is connected to
                k = abs(mlat.KL[i, nn])  # true connection?

                # This part is from Lisa
                # if k != 0:
                #     alphaij = np.arccos(diffx / rij_mag)
                #     k = 1
                # else:
                #     alphaij = 0
                #
                # if diffy < 0:
                #     alphaij = 2 * np.pi - alphaij
                #
                # if mlat.KL[i, nn] < 0:
                #     alphaij = (np.pi + alphaij) % (2 * np.pi)
                #
                #     x1 = np.cos(alphaij) + x2
                #     y1 = np.sin(alphaij) + y2

                if abs(k) > eps:
                    # print 'mgyrofns: nn = ', nn
                    # print 'mgyrofns: PVx[i, nn] = ', PVx[i, nn]
                    diffx = mlat.xy[ni, 0] - mlat.xy[i, 0] + PVx[i, nn]
                    diffy = mlat.xy[ni, 1] - mlat.xy[i, 1] + PVy[i, nn]
                    alphaij = np.arctan2(diffy, diffx)

                    x1 = mlat.xy[ni, 0] + PVx[i, nn]
                    y1 = mlat.xy[ni, 1] + PVy[i, nn]
                    x2 = mlat.xy[i, 0]
                    y2 = mlat.xy[i, 1]

                    rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)
                    # print 'rij mag', rij_mag
                    if rij_mag == 0:
                        raise RuntimeError('Two particles have no distance between them!')
                        rij_mag = 1

                    # There is a true connection here, so update dynamical matrix elements
                    denom = rij_mag ** 7

                    als = (mlat.lp['aoverl']) ** 2 / rij_mag ** 3

                    # See Nash SI eqn S6
                    kparai = (1. / 12.) * als
                    kparaj = -(1. / 6.) * als
                    kperpi = (1. / 12.) * als
                    kperpj = (1. / 12.) * als

                    # The parallel and perpendicular terms on particle p for displacement of particle p
                    kxixi = 0.25 * (-4 * (x1 - x2) ** 2 + (y1 - y2) ** 2) / denom + \
                            kparai * np.cos(alphaij) ** 2 + kperpi * np.sin(alphaij) ** 2
                    kyixi = -0.25 * 5 * (x1 - x2) * (y1 - y2) / denom + \
                            (kparai - kperpi) * np.cos(alphaij) * np.sin(alphaij)
                    kxiyi = -0.25 * 5 * (x1 - x2) * (y1 - y2) / denom + \
                            (kparai - kperpi) * np.cos(alphaij) * np.sin(alphaij)
                    kyiyi = 0.25 * (-4 * (y1 - y2) ** 2 + (x1 - x2) ** 2) / denom + \
                            kperpi * np.cos(alphaij) ** 2 + kparai * np.sin(alphaij) ** 2

                    # The parallel and perpendicular terms on particle p for displacement of particle q
                    kxixj = -0.25 * (-4 * (x1 - x2) ** 2 + (y1 - y2) ** 2) / denom - \
                            kparaj * np.cos(alphaij) ** 2 - kperpj * np.sin(alphaij) ** 2
                    kyixj = 0.25 * 5 * (x1 - x2) * (y1 - y2) / denom + \
                            (-kparaj + kperpj) * np.cos(alphaij) * np.sin(alphaij)
                    kxiyj = 0.25 * 5 * (x1 - x2) * (y1 - y2) / denom + \
                            (-kparaj + kperpj) * np.cos(alphaij) * np.sin(alphaij)
                    kyiyj = -0.25 * (-4 * (y1 - y2) ** 2 + (x1 - x2) ** 2) / denom -\
                            kperpj * np.cos(alphaij) ** 2 - kparaj * np.sin(alphaij) ** 2

                else:
                    kxixi = 0
                    kyixi = 0
                    kxiyi = 0
                    kyiyi = 0

                    kxixj = 0
                    kyixj = 0
                    kxiyj = 0
                    kyiyj = 0

                if ni in mlat.inner_indices:
                    M1[2 * in_count, 2 * in_count] += -k * kyixi  # dxi - dxi
                    M1[2 * in_count, 2 * in_count + 1] += -k * kyiyi  # dxi - dyi
                    M1[2 * in_count + 1, 2 * in_count] += k * kxixi  # dyi - dxi
                    M1[2 * in_count + 1, 2 * in_count + 1] += k * kxiyi  # dyi - dyi

                    ni_new = xy_in_list.index(xy_list[ni])

                    M1[2 * in_count, 2 * ni_new] += -k * kyixj  # dxi - dxj
                    M1[2 * in_count, 2 * ni_new + 1] += -k * kyiyj  # dxi - dyj
                    M1[2 * in_count + 1, 2 * ni_new] += k * kxixj  # dyi - dxj
                    M1[2 * in_count + 1, 2 * ni_new + 1] += k * kxiyj  # dyi - dyj
                else:
                    # The particle is interacting with the stationary, outer magnets
                    # Add the contribution of that stationary magnet to the pinning here.
                    if not bc_f:
                        M1[2 * in_count, 2 * in_count] += -1 * k * kyixi  # dxi - dxi
                        M1[2 * in_count, 2 * in_count + 1] += -1 * k * kyiyi  # dxi - dyi
                        M1[2 * in_count + 1, 2 * in_count] += 1 * k * kxixi  # dyi - dxj
                        M1[2 * in_count + 1, 2 * in_count + 1] += 1 * k * kxiyi  # dyi - dyi

        if i in mlat.inner_indices:
            # pin_in_ind =
            M2[2 * in_count, 2 * in_count + 1] = mlat.Omg[in_count]
            M2[2 * in_count + 1, 2 * in_count] = -mlat.Omg[in_count]
            in_count += 1

    matrix = float(mlat.lp['Omk']) * M1 + M2

    matrix[abs(matrix) < eps] = 0
    return matrix


def calc_matrix_magnetic_psi(mglat, eps=1e-11):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based loosely on chern_functions_gen.make_M().

    Parameters
    ----------
    kvec : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mglat : GyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    # First use mglat to create (angs, num_neis, bls, tvals, ons)
    #
    # angs : list
    #     each row represents a site in the lattice.  Each entry in the row represents the angles to that site's
    #     neighbors
    # num_nei : list or array (num_sites x num_sites)
    #     Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be
    #     num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
    # bls : len(angs) x  float array or int
    #     bondlengths, with dimensions equal to dimensions of angs.
    #     default value is an int, -1, indicating that all bond lengths are 1
    # tvals : len(angs) x 1 float array or int
    #     dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
    # ons : array (dimension = num_sites per unit cell)
    #     represents omega_g
    xy = mglat.lattice.xy
    NL, KL = mglat.NL, mglat.KL
    num_sites, NN = np.shape(NL)
    Omg, OmK = mglat.Omg, mglat.OmK
    PVx, PVy = mglat.PVx, mglat.PVy
    if PVx is None or PVy is None:
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)

    # num_sites is the total number of particles
    mm = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    # Go through each site and fill in rows i and NP + i for that site (psi_L and psi_R)
    for ii in range(num_sites):
        omg = Omg[ii]  # grav frequency for this particle

        # pinning/gravitational matrix -- note: will divide later by factor of -2
        mm[ii, ii] += 2. * omg
        mm[num_sites + ii, num_sites + ii] += -2. * omg

        for nn in range(NN):
            # the index of the gyroscope i is connected to (particle j)
            ni = NL[ii, nn]
            # true connection?
            k = KL[ii, nn]
            # spring frequency for this connection
            omk = OmK[ii, nn]

            if abs(k) > eps:
                # Compute the vector connecting site ii to site ni
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                alphaij = np.arctan2(diffy, diffx)

                rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)
                # print 'rij mag', rij_mag
                if rij_mag < eps:
                    raise RuntimeError('Distance between connected sites is very near zero (less than epsilon)!')
                    rij_mag = 1

                # get the magnitude of l, the length of the pendulum, wrt unit length
                als = rij_mag ** 2 * (mglat.lp['aoverl']) ** 2

                # These are Nash SI eqn S6, multiplied by (l^2/I\omega)
                fpara_p = - omk * (1 - (1. / 12.) * als) / rij_mag ** 5
                fpara_q = omk * (1 + (1. / 6.) * als) / rij_mag ** 5
                fperp_p = omk * 0.25 * (1 + (1. / 3.) * als) / rij_mag ** 5
                fperp_q = -omk * 0.25 * (1 + (1. / 3.) * als) / rij_mag ** 5

                omk_i_plus = fpara_p + fperp_p
                omk_i_minus = fpara_p - fperp_p
                omk_j_plus = fpara_q + fperp_q
                omk_j_minus = fpara_q - fperp_q

                # Create phase factors
                expi2t = np.exp(1j * 2. * alphaij)
                exp_negi2t = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
                mm[ii, ii] += omk_i_plus
                mm[ii, ni] += -omk_j_plus

                # (psi_L psi_R components) top right chunk
                mm[ii, ii + num_sites] += omk_i_minus * expi2t
                mm[ii, ni + num_sites] += -omk_j_minus * expi2t

                # (psi_R psi_L components) bottom left chunk
                mm[ii + num_sites, ii] += -omk_i_minus * exp_negi2t
                mm[ii + num_sites, ni] += omk_j_minus * exp_negi2t

                # (psi_R psi_R components) bottom right chunk
                mm[ii + num_sites, ii + num_sites] += -omk_i_plus
                mm[ii + num_sites, ni + num_sites] += omk_j_plus

    return -0.5 * mm * (-1j)


def build_OmK(mglat, lp, OmK):
    """Construct OmK from supplied OmK (may be None or 'none') and lattice parameter dictionary lp

    Parameters
    ----------
    mglat : MagneticGyroLattice instance
        The mgnetic gyro lattice for which to build interaction strength matrix (each element is l^2 k_m / I omega)
        Note: the distance between two particles is NOT included in OmK. Instead it is computed on the fly in dynamical
        matrix construction
    OmK : N x max(#NN) float array or None or 'none'
        bond frequencies matching the KL and NL arrays, with bonds weakened along gridlines
    lp : dict
        lattice parameter dictionary

    Returns
    -------
    OmK :
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
                    raise RuntimeError('Should OmKspec be built on lattice (NN) or magnetic (long range) connections?')
                    # Here, OmKspec must be of the form:
                    # 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                    # 'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
                    # 'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
                    spec = lp['OmKspec'].split('gridlines')[1]
                    gridspacing = float(spec.split('strong')[0].replace('p', '.'))
                    strong = float(spec.split('strong')[1].split('weak')[0].replace('p', '.').replace('n', '-'))
                    weak = float(spec.split('weak')[1].replace('p', '.').replace('n', '-'))
                    maxval = max(np.max(np.abs(mglat.lattice.xy[:, 0])), np.max(np.abs(mglat.lattice.xy[:, 1]))) + 1
                    OmK = OmK_spec_gridlines(mglat.lattice, lp, gridspacing, maxval, strong, weak)
                else:
                    raise RuntimeError('OmKspec in lp cannot be translated into OmK, must supply OmK or edit to' +
                                       ' interpret given OmKspec')
                lp_meshfn_exten = '_OmKspec' + lp['OmKspec']
                if (OmK == OmK[np.nonzero(OmK)][0] * mglat.KL).all():
                    # This is just the boring case where OmK is not special (all bonds are equal).
                    lp_Omk = OmK[np.nonzero(OmK)][0]
                    if lp_Omk != -1.0:
                        lp_meshfn_exten = '_Omk' + sf.float2pstr(lp['Omk'])
                    else:
                        lp_meshfn_exten = ''
                elif (OmK != lp['Omk'] * np.abs(mglat.KL)).any():
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
                # print 'magnetic_gyro_functions: using Omk from lp...'
                OmK = lp['Omk'] * np.abs(mglat.KL)
                lp_Omk = lp['Omk']
                if lp_Omk != -1.0:
                    lp_meshfn_exten = '_Omk' + sf.float2pstr(lp['Omk'])
                else:
                    lp_meshfn_exten = ''
            else:
                print 'magnetic_gyro_functions: giving OmK the default value of -1s...'
                OmK = -1.0 * np.abs(mglat.KL)
                lp_Omk = -1.0
                lp_meshfn_exten = ''
    else:
        # This is the case where OmK is specified. Pass it along to output and discern what to give for lp_meshfn_exten
        # Output OmK <-- input OmK
        # Use given OmK to define lp_Omk (for lp['Omk']) and lp[meshfn_exten].
        # If Omk is a key in lp, correct it if it does not match supplied OmK
        if 'Omk' in lp:
            if (OmK == lp['Omk'] * np.abs(mglat.KL)).any():
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
                    KLravel = mglat.KL.ravel()
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


def calc_boundary_inner(mglat, check=False):
    """

    Parameters
    ----------
    mglat: MagneticGyroLattice instance
    check: bool

    Returns
    -------
    boundary_inner : M x 1 int array
        the idices of mglat.xy that mark the boundary of mglat.xy_inner
    """
    # First check if there is an outer boundary to be ignored
    if len(mglat.outer_indices) == 0:
        # All particles are inner particles
        boundary_inner = mglat.lattice.get_boundary()
    else:
        # First remove the outer boundary
        print 'mgfns: removing outer boundary for tmp lattice'
        xytmp, NLtmp, KLtmp, BLtmp = le.remove_pts(mglat.inner_indices, mglat.xy, mglat.lattice.BL,
                                                   NN='min', check=check,
                                                   PVxydict=mglat.lattice.PVxydict, PV=mglat.lattice.PV)
        if mglat.lattice.PV is not None:
            PVxydict_tmp = le.BL2PVxydict(BLtmp, xytmp, mglat.lattice.PV)
            PVxtmp, PVytmp = le.PVxydict2PVxPVy(PVxydict_tmp, NLtmp, KLtmp, check=check)
        if mglat.lp['periodic_strip']:
            # Special case: if the entire strip is a boundary, then get
            boundary = le.extract_1d_boundaries(xytmp, NLtmp, KLtmp, BLtmp, PVxtmp, PVytmp,
                                                check=check)
        elif mglat.lp['periodicBC']:
            boundary = None
            raise RuntimeError('periodic boundary conditions and not periodic strip, yet outer_indices are nonempty')
        elif 'annulus' in mglat.lp['LatticeTop'] or mglat.lp['shape'] == 'annulus':
            print 'here'
            outer_boundary = le.extract_boundary(xytmp, NLtmp, KLtmp, BLtmp, check=check)
            inner_boundary = le.extract_inner_boundary(xytmp, NLtmp, KLtmp, BLtmp, check=check)
            boundary = (outer_boundary, inner_boundary)
        else:
            boundary = le.extract_boundary(xytmp, NLtmp, KLtmp, BLtmp, check=check)

        # Now determine which particles are the same as xytmp[boundary]
        boundary_inner = dh.match_points(mglat.xy, xytmp[boundary])

        if check:
            print 'mgfns: boundary_inner = ', boundary_inner
            import matplotlib.pyplot as plt
            plt.plot(mglat.xy[:, 0], mglat.xy[:, 1], 'b.')
            plt.plot(xytmp[boundary, 0], xytmp[boundary, 1], 'go')
            plt.plot(mglat.xy[boundary_inner, 0], mglat.xy[boundary_inner, 1], 'r.')
            plt.show()

        return boundary_inner