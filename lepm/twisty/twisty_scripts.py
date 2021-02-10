import numpy as np
import lepm.lattice_functions_nnn as lfnsnnn
import lepm.stringformat as sf

'''Auxiliary functions for twisty_functions'''


def create_NL_KL_pv(tlat):
    """Based on interaction range, determine NL, KL, and PVxy. If interaction range is negative or zero, all particles
    interact. If interaction_range is 1, only NNs communicate. If interaction_range is 2, 3, etc, then NNs and NNNs,
    NNs + NNNs + NNNNs, etc interact.

    Parameters
    ----------
    tlat : TwistyLattice

    Returns
    -------
    NL, KL, pv, nl_meshfn_exten
    """
    if 'interaction_range' in tlat.lp:
        if tlat.lp['interaction_range'] > 0:
            if tlat.lp['interaction_range'] == 1:
                nl_meshfn_exten = '_intrange{0:04d}'.format(tlat.lp['interaction_range'])
                return tlat.lattice.NL, tlat.lattice.KL, tlat.lattice.PVx, tlat.lattice.PVy, nl_meshfn_exten
            else:
                # Creates NL and KL such intrange-th nearest particles communicate
                if tlat.lp['interaction_range'] == 2 and tlat.lattice.lp['NH'] > 1 and tlat.lattice.lp['NV'] > 1:
                    # Next-nearest neighbors are interacting
                    tlat.lattice.get_nljnnn(attribute=True)
                    NLNNN, KLNNN = tlat.lattice.get_NLNNN_and_KLNNN()
                    pvxnn, pvynn = lfnsnnn.calc_pvnnn(tlat.lattice)
                    NLnns, KLnns = [tlat.lattice.NL, NLNNN], [tlat.lattice.KL, KLNNN]
                    pvxnns, pvynns = [tlat.lattice.PVx, pvxnn], [tlat.lattice.PVy, pvynn]
                    # 'wnei' means with nth neighbors
                    NLwnei, KLwnei, pvxnn, pvynn = lfnsnnn.combine_intnnn_info(NLnns, KLnns, pvxnns, pvynns)
                    nl_meshfn_exten = '_intrange{0:04d}'.format(tlat.lp['interaction_range'])
                else:
                    NLwnei, KLwnei, pvxnn, pvynn = \
                        lfnsnnn.calc_combined_intnn_info(tlat.lattice, tlat.lp['interaction_range'])
                    nl_meshfn_exten = '_intrange{0:04d}'.format(tlat.lp['interaction_range'])

                # print 'KLwnei = ', KLwnei
                # print 'NLwnei = ', NLwnei
                # print 'in magnetic_gyro_functions, exiting...'
                # sys.exit()
                return NLwnei, KLwnei, pvxnn, pvynn, nl_meshfn_exten
        else:
            # Creates NL and KL such that all particles communicate
            NL, KL, PVx, PVy = lfnsnnn.infinite_range_network_info(lat)

            return NL, KL, PVx, PVy, ''
    else:
        printstr = 'for now, enforcing that interaction_range should exist in lp.' + \
                   'Can comment this out to allow default of total coupling'
        raise RuntimeError(printstr)
        # Creates NL and KL such that all particles communicate
        NL = np.array([np.setdiff1d(np.arange(len(self.xy)), np.array([ii])) for ii in range(len(self.xy))])
        KL = np.ones_like(NL, dtype=int)
        return NL, KL, ''


def build_KK(tlat, lp, KK):
    """Construct stretch coefficient matrix from supplied KK (may be None or 'none'), the supplied twisty_lattice,
     and lattice parameter dictionary lp

    Parameters
    ----------
    tlat : TwistyLattice instance
        The mgnetic gyro lattice for which to build interaction strength matrix (each element is l^2 k_m / I omega)
        Note: the distance between two particles is NOT included in KK. Instead it is computed on the fly in dynamical
        matrix construction
    KK : N x max(#NN) float array or None or 'none'
        bond frequencies matching the KL and NL arrays, with bonds weakened along gridlines
    lp : dict
        lattice parameter dictionary

    Returns
    -------
    KK : N x max(#NN) float array
        constructed or modified stretch coefficient matrix
    lp_kk : float
        value for key 'Omk' to be added to lp
    lp_meshfn_exten : str
        value for key 'meshfn_exten' to be added to lp
    """
    if KK == 'auto' or KK is None:
        # Check if a string specifier for KK is given. If it can be understood, create that bond strength pattern.
        # If it is an unknown string, then KK must be supplied.
        if 'KKspec' in lp:
            if lp['KKspec'] not in ['', 'none', None]:
                if 'gridlines' in lp['KKspec']:
                    raise RuntimeError('Should KKspec be built on lattice (NN) or magnetic (long range) connections?')
                    # Here, KKspec must be of the form:
                    # 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                    # 'strong{0:0.3f}'.format(lp['KK']).replace('.', 'p').replace('-', 'n') + \
                    # 'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
                    spec = lp['KKspec'].split('gridlines')[1]
                    gridspacing = float(spec.split('strong')[0].replace('p', '.'))
                    strong = float(spec.split('strong')[1].split('weak')[0].replace('p', '.').replace('n', '-'))
                    weak = float(spec.split('weak')[1].replace('p', '.').replace('n', '-'))
                    maxval = max(np.max(np.abs(tlat.lattice.xy[:, 0])), np.max(np.abs(tlat.lattice.xy[:, 1]))) + 1
                    KK = KK_spec_gridlines(tlat.lattice, lp, gridspacing, maxval, strong, weak)
                else:
                    raise RuntimeError('KKspec in lp cannot be translated into KK, must supply KK or edit to' +
                                       ' interpret given KKspec')
                lp_meshfn_exten = '_KKspec' + lp['KKspec']
                if (KK == KK[np.nonzero(KK)][0] * tlat.KL).all():
                    # This is just the boring case where KK is not special (all bonds are equal).
                    lp_KK = KK[np.nonzero(KK)][0]
                    if lp_kk != -1.0:
                        lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                    else:
                        lp_meshfn_exten = ''
                elif (KK != lp['kk'] * np.abs(tlat.KL)).any():
                    lp_kk = -5000
                else:
                    lp_kk = lp['kk']
                done = True
            else:
                # KKspec is in lp, but it is None or 'none'. Try a different method to obtain KK.
                done = False
        else:
            # KKspec is not in lp. Try a different method to obtain KK.
            done = False

        if not done:
            if 'kk' in lp:
                print 'gyro_lattice_class: using kk from lp...'
                KK = lp['kk'] * np.abs(tlat.KL)
                lp_kk = lp['kk']
                if lp_kk != -1.0:
                    lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                else:
                    lp_meshfn_exten = ''
            else:
                print 'giving KK the default value of -1s...'
                KK = -1.0 * np.abs(tlat.KL)
                lp_kk = -1.0
                lp_meshfn_exten = ''
    else:
        # This is the case where KK is specified. Pass it along to output and discern what to give for lp_meshfn_exten
        # Output KK <-- input KK
        # Use given KK to define lp_kk (for lp['kk']) and lp[meshfn_exten].
        # If kk is a key in lp, correct it if it does not match supplied KK
        if 'kk' in lp:
            if (KK == lp['kk'] * np.abs(tlat.KL)).any():
                # This is the case that KK = kk * np.abs(KL). Give a nontrivial meshfn_exten if kk != -1.0
                lp_kk = lp['kk']
                if lp_kk != 1.0:
                    lp_meshfn_exten = '_kk' + sf.float2pstr(lp['kk'])
                else:
                    lp_meshfn_exten = ''
            else:
                # KK given is either some constant times KL, or something complicated
                # If it is a constant, discern that constant and update lp
                kinds = np.nonzero(KK)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in KK. Check if all the same
                    KKravel = KK.ravel()
                    KLravel = tlat.KL.ravel()
                    if (KKravel[np.where(abs(KLravel))] == KKravel[np.where(abs(KLravel))[0]]).all():
                        print 'Updating lp[kk] to reflect specified KK, since KK = constant * np.abs(KL)...'
                        lp_kk = KKravel[np.where(abs(KLravel))[0]]
                    else:
                        # KK is something complicated, so tell meshfn_exten that KK is specified.
                        lp_meshfn_exten = '_KKspec'
                        if 'KKspec' in lp:
                            lp_meshfn_exten = '_kkspec' + lp['KKspec']
                        lp_kk = -5000

                else:
                    lp_kk = 0.0
                    lp_meshfn_exten = '_kk0p00'
        else:
            # Check if the values of all elements are identical
            kinds = np.nonzero(KK)
            if len(kinds[0]) > 0:
                # There are some nonzero elements in KK. Check if all the same
                value = KK[kinds[0][0], kinds[1][0]]
                if (KK[kinds] == value).all():
                    lp_kk = value
                else:
                    lp_kk = -5000
            else:
                lp_kk = 0.0
                lp_meshfn_exten = '_kk0p00'

    return KK, lp_kk, lp_meshfn_exten


def dynamical_matrix_magnetic_gyros(tlat, basis=None):
    """Construct the dynamical matrix for the given TwistyLattice.

    Parameters
    ----------
    tlat : TwistyLattice instance
        The TwistyLattice for which to construct the dynamical matrix
    basis : 'XY' or 'psi' or None
        The basis in which to construct the dynamical matrix

    Returns
    -------
    matrix
    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in tlat.lp:
        thetatwist = tlat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in tlat.lp:
        phitwist = tlat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    if basis is None and 'basis' in tlat.lp:
        basis = tlat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        # Rest lengths of springs == distances between particles
        if notwist:
            # not twisted, no stretch, XY basis
            matrix = calc_matrix(tlat)
        else:
            # twisted, no stretch, XY basis
            print 'PV = ', tlat.lattice.PV
            print 'thetatwist = ', thetatwist
            print 'phitwist = ', phitwist
            if tlat.lp['periodic_strip']:
                # All periodic bonds are twisted
                matrix = calc_dynamical_matrix_gyros_twist(tlat.lattice.xy, tlat.lattice.NL, tlat.lattice.KL,
                                                           tlat.OmK, tlat.Omg, thetatwist, phitwist,
                                                           tlat.lattice.PVx, tlat.lattice.PVy, tlat.lattice.PV)
            else:
                # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                if 'annulus' in tlat.lp['LatticeTop'] or tlat.lp['shape'] == 'annulus':
                    # Twist bonds in a cut of the annular network
                    twistcut = np.array([0., 0., np.max(tlat.lattice.xy[:, 0]), 0.])
                    thetaKL = form_twistedKL(tlat.lattice.xy, tlat.lattice.BL, tlat.lattice.NL, tlat.lattice.KL,
                                             twistcut)
                    phiKL = np.zeros_like(thetaKL, dtype=int)
                else:
                    raise RuntimeError('Currently only have twistedKL set up for annular samples')

                matrix = calc_dynamical_matrix_gyros_twist_bonds(tlat.lattice.xy, tlat.lattice.NL, tlat.lattice.KL,
                                                                 tlat.OmK, tlat.Omg,
                                                                 thetaKL, phiKL,
                                                                 thetatwist, phitwist,
                                                                 tlat.lattice.PVx, tlat.lattice.PVy,
                                                                 tlat.lattice.PV)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_matrix_magnetic_psi(tlat)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in tlat.lp:
        if tlat.lp['immobile_boundary']:
            boundary = tlat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def calc_matrix(tlat, eps=1e-9):
    """

    Parameters
    ----------
    tlat :

    Returns
    -------

    """
    xy = tlat.lattice.xy
    # OmC is the N x max#NN array of twist restoring coefficients
    KK, GG, CC = tlat.KK, tlat.GG, tlat.CC
    NP_total, NN = tlat.NL.shape
    NP = len(tlat.inner_indices)
    PVx = tlat.PVx
    PVy = tlat.PVy

    # Unpack periodic boundary vectors
    if PVx is None and PVy is None:
        PVx = np.zeros((NP_total, NN), dtype=float)
        PVy = np.zeros((NP_total, NN), dtype=float)

    mm = np.zeros((3 * NP, 3 * NP), dtype=float)

    xy_in = list(tlat.xy[tlat.inner_indices])

    xy_list = [list(tlat.xy[i]) for i in range(len(tlat.xy))]
    xy_in_list = [list(xy_in[i]) for i in range(len(xy_in))]

    # ignores the outer particles
    if 'bcs' in tlat.lp:
        if tlat.lp['bcs'] == 'free':
            bc_f = True
        else:
            bc_f = False
    else:
        bc_f = False

    # print 'mgyrofns: tlat.outer_indices = ', tlat.outer_indices
    # print 'mgyrofns: NL = ', tlat.NL

    in_count = 0
    for ii in range(NP_total):
        for nn in range(NN):
            if ii in tlat.inner_indices:
                ni = int(tlat.NL[ii, nn])  # the number of the gyroscope i is connected to
                k = abs(tlat.KL[ii, nn])  # true connection?
                # stretch constant for this connection
                omk = KK[ii, nn]
                # twist constant for this connection
                omc = CC[ii, nn]
                # twist-stretch coupling constant for this connection
                omg = GG[ii, nn]

                if abs(k) > eps:
                    # There is a true connection, so update dynamical matrix
                    # if len(dispersion) > 1:
                    #     disp = 1. / (1. + dispersion[i])
                    # else:
                    #     disp = 1.
                    # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                    diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                    diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                    alphaij = np.arctan2(diffy, diffx)
                    if tlat.lp['scale_interactions']:
                        rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)
                        omk /= rij_mag
                        omg /= rij_mag
                        omc /= rij_mag

                    Cos = np.cos(alphaij)
                    Sin = np.sin(alphaij)

                    if abs(Cos) < eps:
                        Cos = 0.0

                    if abs(Sin) < eps:
                        Sin = 0.0

                    Cos2 = Cos ** 2
                    Sin2 = Sin ** 2
                    CosSin = Cos * Sin

                    if ni in tlat.inner_indices:
                        # (x components)
                        mm[2 * in_count, 2 * in_count] += omk * Cos2
                        mm[2 * in_count, 2 * in_count + 1] += omk * CosSin
                        # (y components)
                        mm[2 * in_count + 1, 2 * in_count] += omk * CosSin
                        mm[2 * in_count + 1, 2 * in_count + 1] += omk * Sin2

                        ni_new = xy_in_list.index(xy_list[ni])

                        mm[2 * in_count, 2 * ni_new] += -omk * Cos2
                        mm[2 * in_count, 2 * ni_new + 1] += -omk * CosSin
                        mm[2 * in_count + 1, 2 * ni_new] += -omk * CosSin
                        mm[2 * in_count + 1, 2 * ni_new + 1] += -omk * Sin2

                        # Add twist restoring coefficients
                        mm[2 * in_count, 2 * NP_total + 2 * in_count] = -omg * CosSin
                        mm[2 * in_count, 2 * NP_total + 2 * in_count + 1] = -omg * Cos2
                        mm[2 * in_count, 2 * NP_total + 2 * ni_new] = omg * CosSin
                        mm[2 * in_count, 2 * NP_total + 2 * ni_new + 1] = -omg * Cos2
                        mm[2 * in_count + 1, 2 * NP_total + 2 * in_count] = -omg * Sin2
                        mm[2 * in_count + 1, 2 * NP_total + 2 * in_count + 1] = omg * CosSin
                        mm[2 * in_count + 1, 2 * NP_total + 2 * ni_new] = omg * Sin2
                        mm[2 * in_count + 1, 2 * NP_total + 2 * ni_new + 1] = -omg * CosSin

                        # Fill in 2*NP_total + in_count row
                        mm[2 * NP_total + 2 * in_count, 2 * in_count] += -omg * CosSin
                        mm[2 * NP_total + 2 * in_count, 2 * in_count + 1] += omg * CosSin
                        mm[2 * NP_total + 2 * in_count, 2 * ni_new] += -omg * CosSin
                        mm[2 * NP_total + 2 * in_count, 2 * ni_new + 1] += omg * Sin2
                        mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * in_count] += omc * Sin2
                        mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * in_count + 1] += -omc * CosSin
                        mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * ni_new] += -omc * Sin2
                        mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * ni_new + 1] += -omc * CosSin
                        # row 2*(NPtotal + incount) + 1 which is for Fz_i
                        mm[2 * NP_total + 2 * in_count + 1, 2 * in_count] += omg * Cos2
                        mm[2 * NP_total + 2 * in_count + 1, 2 * in_count + 1] += omg * CosSin
                        mm[2 * NP_total + 2 * in_count + 1, 2 * ni_new] += -omg * Cos2
                        mm[2 * NP_total + 2 * in_count + 1, 2 * ni_new + 1] += omg * CosSin
                        mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + in_count] += -omc * CosSin
                        mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + in_count + 1] += omc * Cos2
                        mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + ni_new] += omc * CosSin
                        mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + ni_new + 1] += -omc * Cos2
                    else:
                        # The particle is interacting with the stationary outer particle
                        # Add the contribution of that stationary particle to the pinning of the in_count particle here.
                        # The stationary particle is not displaced, so no need to update those 'off-diagonal' ni terms
                        if not bc_f:
                            mm[2 * in_count, 2 * in_count] += omk * CosSin
                            mm[2 * in_count, 2 * in_count + 1] += -omk * Sin2
                            mm[2 * in_count, 2 * NP_total + 2 * in_count] += -omg * CosSin
                            mm[2 * in_count, 2 * NP_total + 2 * in_count + 1] += -omg * Cos2
                            mm[2 * in_count + 1, 2 * in_count] += omk * CosSin
                            mm[2 * in_count + 1, 2 * in_count + 1] += omk * Sin2
                            mm[2 * in_count + 1, 2 * in_count] += -omg * Sin2
                            mm[2 * in_count + 1, 2 * in_count + 1] += omg * CosSin
                            mm[2 * NP_total + 2 * in_count, 2 * in_count] += -omg * CosSin
                            mm[2 * NP_total + 2 * in_count, 2 * in_count + 1] += -omg * Sin2
                            mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * in_count] += omc * Sin2
                            mm[2 * NP_total + 2 * in_count, 2 * NP_total + 2 * in_count + 1] += -omc * CosSin
                            mm[2 * NP_total + 2 * in_count + 1, 2 * in_count] += omg * Cos2
                            mm[2 * NP_total + 2 * in_count + 1, 2 * in_count + 1] += -omg * CosSin
                            mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + 2 * in_count] += -omc * CosSin
                            mm[2 * NP_total + 2 * in_count + 1, 2 * NP_total + 2 * in_count + 1] += omc * Cos2

        if i in tlat.inner_indices:
            # update the indix for which the inner indices list
            in_count += 1

    matrix = mm
    matrix[abs(matrix) < eps] = 0
    return matrix
