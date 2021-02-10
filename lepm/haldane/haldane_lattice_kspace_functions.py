import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.lattice_functions_nnn as latfnnn
# import lepm.lattice_elasticity as le
import cPickle as pkl
import glob
import lepm.haldane.haldane_lattice_functions as hlatfns
import sys
import lepm.data_handling as dh

'''
Description
===========
Auxiliary functions for HaldaneLattice class / KChern class, for kspace methods
'''


def prepare_dispersion_params(hlat, kx=None, ky=None, nkxvals=50, nkyvals=20, outdir=None, name=None):
    """Prepare the filename and kx, ky for computing dispersion relation

    Returns
    """
    if not hlat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif hlat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(hlat.lattice.lp['BBox'][:, 0]) - min(hlat.lattice.lp['BBox'][:, 0])
    elif ky is None or kx is None:
        bboxx = max(hlat.lattice.lp['BBox'][:, 0]) - min(hlat.lattice.lp['BBox'][:, 0])
        bboxy = max(hlat.lattice.lp['BBox'][:, 1]) - min(hlat.lattice.lp['BBox'][:, 1])

    if kx is None:
        tmp = np.linspace(-1. / bboxx, 1. / bboxx, nkxvals - 1)
        step = np.diff(tmp)[0]
        kx = 2. * np.pi * np.linspace(-1. / bboxx, 1. / bboxx + step, nkxvals)
        # kx = np.linspace(-5., 5., 40)

    if ky is None:
        tmp = np.linspace(-1. / bboxy, 1. / bboxy, nkyvals - 1)
        step = np.diff(tmp)[0]
        ky = 2. * np.pi * np.linspace(-1. / bboxy, 1. / bboxy + step, nkyvals)
        # ky = np.linspace(-5., 5., 4)

    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(hlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'dispersion' + hlat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
        name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    name = outdir + name
    return name, kx, ky


def infinite_dispersion(hlat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                        title='haldane dispersion relation', outdir=None, name=None, ax=None, lwscale=1.):
    """Compute the real part of the eigvalues of the dynamical matrix for a grid of wavevectors kx, ky

    Parameters
    ----------
    hlat : HaldaneLattice class instance
        the haldane network whose dispersion we compute
    kx : n x 1 float array
        the x components of the wavevectors over which to diagonalize the dynamical matrix
    ky : m x 1 float array
        the y components of the wavevectors over which to diagonalize the dynamical matrix
    nkxvals : int
        If kx is unspecified, then nkxvals determines how many kvectors are sampled in x dimension.
    nkyvals : int
        If ky is unspecified and if network is not a periodic_strip, then nkyvals determines how
        many kvectors are sampled in y dimension.
    save : bool
        Save the omega vs k information in a pickle
    save_plot : bool
        Save the omega vs k info as a matplotlib figure png
    title : str
        title for the plot to save
    outdir : str or None
        The directory in which to output the image and pickle of the results if save == True

    Returns
    -------

    """
    name, kx, ky = prepare_dispersion_params(hlat, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                             outdir=outdir, name=name)
    print('checking for file: ' + name + '.pkl')
    if glob.glob(name + '.pkl'):
        saved = True
        with open(name + '.pkl', "rb") as fn:
            res = pkl.load(fn)

        omegas = res['omegas']
        kx = res['kx']
        ky = res['ky']
    else:
        # dispersion is not saved, compute it!
        saved = False

        omegas = np.zeros((len(kx), len(ky), len(hlat.lattice.xy)))
        matk = lambda k: dynamical_matrix_kspace(k, hlat, eps=1e-10)
        ii = 0
        for kxi in kx:
            print 'hlatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                # print 'hlatkspace_fns: diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.real(eigval))
                omegas[ii, jj, :] = np.real(eigval[si])
                # print 'eigvals = ', eigval
                # print 'omegas --> ', omegas[ii, jj]
                jj += 1
            ii += 1

    if save_plot or ax is not None:
        if ax is None:
            fig, ax = leplt.initialize_1panel_centered_fig()

        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=lwscale * max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k$ $[\langle \ell \rangle ^{-1}]$')
        ax.set_ylabel(r'$\omega$')
        # ylims = ax.get_ylim()
        # ax.set_ylim(0, ylims[1])
        # Save the plot
        if save_plot:
            plt.savefig(name + '.png', dpi=300)
            plt.close('all')

        # Plot in 3D
        # fig = plt.gcf()
        # ax = fig.add_subplot(projection='3d')  # 111,
        # # rows will be kx, cols wll be ky
        # kyv = np.array([[ky[i].tolist()] * len(kx) for i in range(len(ky))]).ravel()
        # kxv = np.array([[kx.tolist()] * len(ky)]).ravel()
        # print 'kyv = ', np.shape(kyv)
        # print 'kxv = ', np.shape(kxv)
        # for kk in range(len(omegas[0, 0, :])):
        #     ax.plot_trisurf(kxv, kyv, omegas[:, :, kk].ravel())
        #
        # ax.view_init(elev=0, azim=0.)
        # ax.set_title(title)
        # ax.set_xlabel(r'$k_x$ $[1/\langle l \rangle]$')
        # ax.set_ylabel(r'$k_y$ $[1/\langle l \rangle]$')
        # plt.savefig(name + '_3d.png')

    if save:
        if not saved:
            res = {'omegas': omegas, 'kx': kx, 'ky': ky}
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    return omegas, kx, ky


def infinite_dispersion_unstructured(hlat, kxy, load=False, save=True, save_plot=True,
                                     title='gyro dispersion relation', outdir=None, name=None, ax=None, lwscale=1.,
                                     verbose=False, overwrite=False, return_eigvects=False, return_matk=False):
    """ Do not assume grid structure of kxy for this function. Compute the spectrum evaluated at the points given
    by the 2d array kxy.

    Parameters
    ----------
    hlat
    kxy
    save
    save_plot
    title
    outdir
    name
    ax
    lwscale
    verbose

    Returns
    -------
    omegas : tuple (omegas, eigvects) if return_eigvects is True, otherwise len(kxy) x 2 * len(xy) float array
        The eigenvalues, and possibly the corresponding eigenvectors if kwarg return_eigvects is True, returned as
        float and complex arrays, respectively. omegas is a len(kxy) x 2 * len(xy) float array, while eigvects, if
        returned, is a len(kxy) x 2 * len(xy) x 2 * len(xy) complex array
    """
    name, kx, ky = prepare_dispersion_params(hlat, kx=kxy[:, 0], ky=kxy[:, 1], outdir=outdir, name=name)
    name += '_unstructured'

    if glob.glob(name + '.pkl') and not overwrite and load:
        print('checking for file: ' + name + '.pkl')
        saved = True
        with open(name + '.pkl', "rb") as fn:
            res = pkl.load(fn)

        omegas = res['omegas']
        kx = res['kx']
        ky = res['ky']
    else:
        # dispersion is not saved, compute it!
        saved = False

        omegas = np.zeros((len(kxy), len(hlat.lattice.xy)))
        if return_eigvects:
            eigvects = np.zeros((len(kxy), len(hlat.lattice.xy), len(hlat.lattice.xy)), dtype=complex)
        matk = lambda k: dynamical_matrix_kspace(k, hlat, eps=1e-10, gaugejump=hlat.lp['gaugejump'])
        ii = 0
        for pt in kxy:
            if ii % 50 == 1:
                print 'hlatkspace_fns: infinite_dispersion(): ii = ', ii
            matrix = matk([pt[0], pt[1]])
            # print 'haldane_lattice_kspace_functions: matk = ', matrix

            eigval, eigvect = np.linalg.eig(matrix)
            # print 'haldane_lattice_kspace_functions: eigval = ', eigval

            si = np.argsort(np.real(eigval))
            omegas[ii, :] = np.real(eigval[si])
            if return_eigvects:
                eigvect = np.array(eigvect)
                eigvect_out = eigvect.T[si]
                # print 'eigvect_out = ', eigvect_out
                # if ortho_eigvect:
                #     eigvect_out = gdh.orthonormal_eigvect(eigvect)
                # print 'eigvect_out = ', eigvect_out
                eigvects[ii] = eigvect_out

            ii += 1

    if save_plot or ax is not None:
        if ax is None:
            fig, ax = leplt.initialize_1panel_centered_fig()

        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=lwscale * max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k$ $[\langle \ell \rangle ^{-1}]$')
        ax.set_ylabel(r'$\omega$')
        ylims = ax.get_ylim()
        ax.set_ylim(0, ylims[1])
        # Save the plot
        if save_plot:
            plt.savefig(name + '.png', dpi=300)
            plt.close('all')

    if save:
        if not saved:
            res = {'omegas': omegas, 'kx': kx, 'ky': ky}
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    if return_eigvects:
        omegas = (omegas, eigvects)
    if return_matk:
        omegas = (omegas, eigvects, matk)

    return omegas, kx, ky


def vec(ang, bl=1):
    """creates a vector given a bond length and angle in radians"""
    return [bl * np.cos(ang), bl * np.sin(ang)]


def ang_fac(ang):
    """Convert angle to factor in the Hamiltonian for haldane system"""
    return np.exp(2 * 1j * ang)


def lambda_matrix_kspace(hlat, eps=1e-10):
    """Construct the dynamical matrix for the given HaldaneLattice as a function of an as-yet-unspecified wavevector kvec.

    Parameters
    ----------
    hlat : HaldaneLattice class instance
        the haldane network whose dispersion we compute
    eps : float
        resolution for discerning if value of connectivity matrix is nonzero

    Returns
    -------
    lambda function
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, hlat, eps=eps)


def dynamical_matrix_kspace(kvec, hlat, eps=1e-9, basis=None, gaugejump=False):
    """Construct the dynamical matrix for the given HaldaneLattice for the given wavevector kvec.

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    hlat : HaldaneLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------

    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in hlat.lp:
        thetatwist = hlat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in hlat.lp:
        phitwist = hlat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    if notwist:
        # not twisted
        matrix = calc_kmatrix_haldane(kvec, hlat, eps=1e-8, gaugejump=gaugejump)
        # Using psi basis for now since it is the only one that works.
        # matrix = calc_kmatrix_psi(kvec, hlat, eps=eps)
        # outname = '/Users/npmitchell/Desktop/test/' + 'kx{0:0.2f}'.format(kvec[0]) +\
        #           'ky{0:0.2f}'.format(kvec[1])
        # leplt.plot_complex_matrix(matrix, name='dynamical_matrix', outpath=outname)
    else:
        # twisted hoppings
        print 'PV = ', hlat.lattice.PV
        print 'thetatwist = ', thetatwist
        print 'phitwist = ', phitwist
        if hlat.lp['periodic_strip']:
            # All periodic bonds are twisted
            matrix = calc_kmatrix_haldane_twist(kvec, hlat, eps=eps)
        else:
            # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
            # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
            if 'annulus' in hlat.lp['LatticeTop'] or hlat.lp['shape'] == 'annulus':
                twistcut = np.array([0., 0., np.max(hlat.lattice.xy[:, 0]), 0.])
                thetaKL = hlatfns.form_twistedKL(kvec, hlat, eps=eps)
                phiKL = np.zeros_like(thetaKL, dtype=int)
            else:
                raise RuntimeError('Currently only have twistedKL set up for annular samples')

            # Certain bonds are twisted, while the others are normal.
            matrix = calc_kmatrix_haldane_twist_bonds(kvec, hlat, thetaKL, phiKL, eps=eps)

    if 'immobile_boundary' in hlat.lp:
        if hlat.lp['immobile_boundary']:
            raise RuntimeError('Why is there immobile_boundary in the lattice parameter dictionary? '
                               'This is a haldane model...')
            boundary = hlat.lattice.get_boundary()
            for ind in boundary:
                matrix[ind, :] = 0
    return matrix


def calc_kmatrix_haldane(kvec, hlat, eps=1e-8, gaugejump=True):
    """Compute the dynamical matrix for d psi /dt = D psi. This code is intended for periodic boundary conditions.
    Note that t3 is responsible for a complex (or real) hopping associated with going from site i to site i in a
    periodic copy of the system located at each of the lattice vectors in hlat.t3lvs relative to site i.
    Example usage: python ./haldane/haldane_lattice_class.py -LT hexagonal -N 3 -periodic -dispersion

    Parameters
    ----------
    kvec : 1 x 2 float array
        The values of k (wavenumber) over which to compute the dynamical matrix
    hlat : HaldaneLattice instance
        The haldane model network to consider
    eps : float
        threshold for considering an element in KL to be a real connection
    gaugejump : bool
        If True, the gauge choice of each site is constant within the unit cell, but jumps across periodic boundaries.
        Otherwise, the gauge choice varies throughout all space, even within a unit cell. The difference is that a
        hopping will have a phase of e^(i PV) if True across a periodic boundary and 1 within the cell if gaugejump is
        True.

    Returns
    -------
    matrix : NP x NP complex array
        M1 + M2 + pin * Identity
    """
    # essential tight binding parameters
    pin = hlat.pin
    t1 = hlat.t1
    t2 = hlat.t2
    t2a = hlat.t2a
    t3 = hlat.t3
    t3lvs = hlat.t3lvs
    t3lvsigns = hlat.t3lvsigns

    # other parameters to be unpacked
    t2angles = hlat.lp['t2angles']
    pureimNNN = hlat.lp['pureimNNN']
    ignore_tris = hlat.lp['ignore_tris']
    lat = hlat.lattice
    xy = lat.xy
    NL = lat.NL
    KL = lat.KL
    BL = lat.BL
    PVx = lat.PVx
    PVy = lat.PVy
    NLNNN, KLNNN = lat.get_NLNNN_and_KLNNN(attribute=True)
    # print 'hlatkspacefns: lat.lp[ignore_tris] = ', lat.lp['ignore_tris']
    pvxnnn, pvynnn = lat.get_pvnnn()

    # print 'NLNNN = ', NLNNN
    # print 'KLNNN = ', KLNNN
    # print 'nljnnn = ', lat.nljnnn
    # print 'kljnnn = ', lat.kljnnn
    # print 'klknnn = ', lat.klknnn
    # print 'pvxnnn = ', pvxnnn
    # print 'pvynnn = ', pvynnn
    # print 'haldane_lattice_kspace_functions() exiting here'
    # sys.exit()
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0
    if (BL < 0).any() and (PVx is None or PVy is None):
        raise RuntimeError('Must specify PVx and PVy in haldane_matrix() when periodic bonds exist!')

    lat.lp['ignore_tris'] = ignore_tris

    M1 = np.zeros((NP, NP), dtype=complex)
    M2 = np.zeros((NP, NP), dtype=complex)
    M3 = np.zeros((NP, NP), dtype=complex)

    NNNN = np.shape(KLNNN)[1]

    if t2angles:
        NLNNNangles = latfnnn.NNN_bond_angles(xy, NL, KL, NLNNN, KLNNN, PVx=PVx, PVy=PVy, cwccw=True)
        dubtheta = hlatfns.calc_dubtheta(NLNNNangles)
        for i in range(NP):
            # Do nearest neighbor hoppings
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # Compute factor from fourier transform
                    if gaugejump:
                        # The gauge choice jumps across periodic boundaries
                        if abs(PVx[i, nn]) > 0 or abs(PVy[i, nn]) > 0:
                            diffx = PVx[i, nn]
                            diffy = PVy[i, nn]
                            kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                        else:
                            kfactor = 1
                    else:
                        # The gauge choice is continuous, varying within a unit cell as well as across periodic
                        # boundaries.
                        diffx = lat.xy[ni, 0] - lat.xy[i, 0] + PVx[i, nn]
                        diffy = lat.xy[ni, 1] - lat.xy[i, 1] + PVy[i, nn]
                        kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1 * kfactor  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn] * kfactor  # psi_j

            # Do next nearest neighbor hoppings
            for nn in range(NNNN):
                ni = NLNNN[i, nn]
                k = KLNNN[i, nn]

                if abs(k) > eps:
                    # Compute factor from fourier transform
                    diffx = lat.xy[ni, 0] - lat.xy[i, 0] + pvxnnn[i, nn]
                    diffy = lat.xy[ni, 1] - lat.xy[i, 1] + pvynnn[i, nn]

                    kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += 1j * t2 * np.sin(dubtheta[i, nn]) * kfactor
                    if not pureimNNN:
                        M2[i, ni] += t2 * np.cos(dubtheta[i, nn]) * kfactor

                        # if k < -eps:
                        #     # There is a true NNN connection, so update dynamical matrix
                        #     # print 'i = ', i, ' ni= ', ni
                        #     # print 'hlatfns: negative: KLNNN[i,nn] = ', k, '  dubtheta = ', np.sin(dubtheta[i, nn])
                        #     M2[i, ni] += t2 * np.cos(dubtheta[i, nn]) + 1j * t2 * np.sin(dubtheta[i, nn])
                        # print 'M2[0,:] = ', M2[:,0]
    else:
        for i in range(NP):
            # Do nearest neighbor hoppings
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # Compute factor from fourier transform
                    if gaugejump:
                        # The gauge choice jumps across periodic boundaries
                        if abs(PVx[i, nn]) > 0 or abs(PVy[i, nn]) > 0:
                            diffx = PVx[i, nn]
                            diffy = PVy[i, nn]
                            kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                        else:
                            kfactor = 1
                    else:
                        # The gauge choice is continuous, varying within a unit cell as well as across periodic
                        # boundaries.
                        diffx = lat.xy[ni, 0] - lat.xy[i, 0] + PVx[i, nn]
                        diffy = lat.xy[ni, 1] - lat.xy[i, 1] + PVy[i, nn]
                        kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))


                    # (psi components)
                    if isinstance(t1, float):
                        M1[i, ni] += t1 * kfactor  # psi_j
                    else:
                        M1[i, ni] += t1[i, nn] * kfactor  # psi_j

            # Do next nearest neighbor hoppings
            if hlat.t2 != 0.:
                for nn in range(NNNN):
                    ni = NLNNN[i, nn]
                    k = KLNNN[i, nn]

                    if k > eps:
                        # Compute factor from fourier transform
                        diffx = lat.xy[ni, 0] - lat.xy[i, 0] + pvxnnn[i, nn]
                        diffy = lat.xy[ni, 1] - lat.xy[i, 1] + pvynnn[i, nn]
                        # print 'clockwise for ', i, ': ', (diffx, diffy)

                        kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                        # There is a true NNN connection, so update dynamical matrix
                        if hlat.t2func is None:
                            # Add regular complex t2 hopping
                            M2[i, ni] += (1j * t2 + t2a) * kfactor
                        else:
                            # Use supplied function to compute t2 hopping
                            M2[i, ni] += (1j * t2 + t2a) * kfactor * hlat.t2func(np.array([diffx, diffy]), i)
                    if k < -eps:
                        # There is a true NNN connection, so update dynamical matrix
                        # Compute factor from fourier transform
                        # print 'ni = ', ni, ' i = ', i, ' nn = ', nn
                        diffx = lat.xy[ni, 0] - lat.xy[i, 0] + pvxnnn[i, nn]
                        diffy = lat.xy[ni, 1] - lat.xy[i, 1] + pvynnn[i, nn]
                        # print 'cclockwise for ', i, ': ', (diffx, diffy)

                        kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                        if hlat.t2func is None:
                            M2[i, ni] += (-1j * t2 + t2a) * kfactor
                        else:
                            # Use supplied function to compute t2 hopping
                            M2[i, ni] += (-1j * t2 + t2a) * kfactor * hlat.t2func(np.array([diffx, diffy]), i)


        if hlat.t3 != 0. and hlat.t3 is not None:
            for ii in range(NP):
                for (lv, lvsign) in zip(t3lvs, t3lvsigns[ii]):
                    kfactor = np.exp(1j * (lv[0] * kvec[0] + lv[1] * kvec[1]))
                    M3[ii, ii] += t3 * lvsign * kfactor

    matrix = M1 + M2 + M3
    # if np.abs(t3) > 0.03:
    #     print 'M3 = ', M3
    #     print 't3 = ', t3
    #     print 't3 * kfactor = ', t3 * np.exp(1j * (lv[0] * kvec[0] + lv[1] * kvec[1]))
    #     print 'kfactor = ', np.exp(1j * (lv[0] * kvec[0] + lv[1] * kvec[1]))
    #     print 'lvsign = ', lvsign
    #     raise RuntimeError('exiting here')

    # sys.exit()
    matrix += pin * np.identity(NP)
    # print 'matrix[0,0] =', matrix[0, 0]
    # if np.abs(matrix[0, 0]) > 0.8:
    #     print 'haldane_lattice_kspace_functions.py: exiting here'
    #     sys.exit()
    return matrix


def calc_kmatrix_haldane_twist(kvec, hlat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    hlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_haldane_twist_bonds(kvec, hlat, thetaKL, phiKL, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    hlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


################################################
# Functions for computing band gaps
################################################
def calc_bands(hlat, kxy=None, density=100, verbose=True):
    """Compute the band eigenvalues at an unstructured array of kxy points.

    Parameters
    ----------
    hlat : HaldaneLattice class instance
        the gyro lattice on which to compute the bands
    kxy : N x 2 float array or None
        the wavevectors (each row is a wavevector) at which to compute the spectrum. If None, will generate a random
        collection in the BZ
    density : int
        the number of evaluation points per unit area of the BZ
    verbose : bool
        print information to command line

    Returns
    -------
    omegas : #bands x len(kxy) float array
        the eigenvalues of the dynamical matrix in descending(?) order
    kx, ky :
    """
    matk = lambda_matrix_kspace(hlat, eps=1e-10)
    bzvtcs = hlat.lattice.get_bz(attribute=True)
    bzarea = dh.polygon_area(bzvtcs)

    # Make the kx, ky points to add to current results
    if kxy is None:
        kxy = dh.generate_random_xy_in_polygon(density * bzarea, bzvtcs, sorted=True)

    start_time = time.time()
    bands = []
    for ii in range(len(kxy)):
        # time for evaluating point index ii
        tpi = []
        # Display how much time is left
        if ii % 4000 == 1999 and verbose:
            end_time = time.time()
            tpi.append(abs((start_time - end_time) / (ii + 1)))
            total_time = np.mean(tpi) * len(kxy)
            printstr = 'Estimated time remaining: ' + '%0.2f s' % (total_time - (end_time - start_time))
            printstr += ', ii = ' + str(ii + 1) + '/' + str(len(kxy))
            print printstr

        matii = matk([kxy[ii, 0], kxy[ii, 1]])
        eigval, eigvect = le.eig_vals_vects(matii)
        bands.append(np.imag(eigval))

    omegas = np.array(bands)
    kx, ky = kxy[:, 0], kxy[:, 1]
    return omegas, kx, ky


def get_matrix_at_kpt(hlat, kxy, verbose=False, check=False, eps=1e-10, gaugejump=False):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if len(np.shape(kxy)) == 1:
        kxy = np.array(kxy)

    matk = dynamical_matrix_kspace(kxy, hlat, eps=eps, gaugejump=gaugejump)
    return matk


def get_band_eigs_at_kpt(hlat, kxy, verbose=False, check=False, return_matk=False):
    """

    Parameters
    ----------

    Returns
    -------
    eigs, eigvs
        Note that there is an extra dimension added here to support kxy having multiple entries
    """
    if len(np.shape(kxy)) == 1:
        kxy = np.array([kxy])

    eigseigvmatk, kxout, kyout = infinite_dispersion_unstructured(hlat, kxy, save=False, save_plot=False,
                                                                         verbose=verbose, overwrite=False,
                                                                         return_eigvects=True,
                                                                         return_matk=return_matk)
    if len(eigseigvmatk) == 3:
        return eigseigvmatk[0], eigseigvmatk[1], eigseigvmatk[2]
    else:
        return eigseigvmatk[0], eigseigvmatk[1]



def calc_band_limits(hlat, omegas=None, kxy=None, density=100, verbose=True):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_limits -LT hexagonal -N 1

    Parameters
    ----------
    hlat : HaldaneLattice class instance
    kxy : N x 2 float array or None
        the wavevectors (each row is a wavevector) at which to compute the spectrum. If None, will generate a random
        collection in the BZ
    density : int
        the number of sampled points per unit area of BZ
    verbose : bool
        output text for intermediate to command line

    Returns
    -------
    limits : len(omegas) x 2 float array
        each row is the minimum and maximum frequency of a band
    """
    if omegas is None:
        omegas, kx, ky = calc_bands(hlat, kxy=kxy, density=density, verbose=verbose)

    # Given omegas, compute the tops and bottoms of the bands
    mins = np.min(omegas, axis=0)
    maxs = np.max(omegas, axis=0)
    limits = np.dstack((mins, maxs))[0]
    return limits


def calc_band_gaps(hlat, omegas=None, kxy=None, density=100, verbose=True):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_limits -LT hexagonal -N 1

    Parameters
    ----------
    hlat
    omegas : #kpts x #bands float array
        the frequencies of the band structure evaluated at each wavenumber
    kxy
    density
    verbose

    Returns
    -------

    """
    limits = calc_band_limits(hlat, omegas=omegas, kxy=kxy, density=density, verbose=verbose)
    # Given band limits, find gaps
    tops = limits[:, 1][:-1]
    bots = limits[1:, 0]
    return np.dstack((tops, bots))[0]


def calc_band_bounds(hlat, omegas=None, kxy=None, density=100, verbose=True, ngridpts=100, axis=0):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_bounds -LT hexagonal -N 1
    # or
    polygons = hlat.calc_band_bounds()
    for polygon in polygons:
        poly = Polygon(polygon, closed=True, fill=True, lw=0.50, alpha=0.5, color='g', edgecolor='k')
        ax.add_artist(poly)


    Parameters
    ----------
    hlat : HaldaneLattice class instance
    omegas : len(kxy) x #bands float array
        eigenfrequencies of the HaldaneLattice defined over some sampling of BZ, kxy.
    kxy : N x 2 float array or None
        the wavevectors (each row is a wavevector) at which to compute the spectrum. If None, will generate a random
        collection in the BZ
    density : int
        the number of sampled points per unit area of BZ
    verbose : bool
        output text for intermediate to command line
    ngridpts : int
        the number of points along the x dimension with which to estimate the bounding polygon
    axis : int
        the axis along which we examine the spectrum (x axis or y axis corresponds to 0 or 1, respectively)

    Returns
    -------
    polygon_list : list of 2*(gridpts - 1) x 2 float arrays
        The convex bounding polygon
    """
    if omegas is None or kxy is None:
        omegas, kx, ky = calc_bands(hlat, kxy=kxy, density=density, verbose=verbose)
        kxy = np.dstack((kx, ky))[0]
        # Code for reshaping omegas, kxy from a grid
        # kxx, kyy = np.meshgrid(kx, ky)
        # kxy = np.dstack((kxx.ravel(), kyy.ravel()))[0]
        # # reshape oms into omegas
        # shap = np.shape(oms)
        # print 'shap = ', shap
        # omegas = np.zeros((shap[2], shap[0] * shap[1]))
        # for (freqs, kk) in zip(oms, range(shap[2])):
        #     omegas[kk] = freqs.ravel()

    polygon_list = []
    for kk in range(np.shape(omegas)[1]):
        omega = omegas[:, kk]
        print np.shape(kxy[:, axis])
        print 'np.dstack((kxy[:, axis], omega)) = ', np.dstack((kxy[:, axis], omega))
        polygon_list.append(dh.approx_bounding_polygon(np.dstack((kxy[:, axis], omega))[0], ngridpts=ngridpts))
    return polygon_list


def sample_dos_in_BZ(hlat, density=100, save=True, save_plot=True, overwrite=False):
    """Take a random sampling of kxy wavecectors in the Brillouin zone and evaluate & return their eigenvalues

    Parameters
    ----------
    hlat
    density

    Returns
    -------
    eigvals : 1d float array
        the eigenvalues in the BZ, sampled over evenly spaced grid of wavevectors
    """
    bzvtcs = hlat.lattice.get_bz(attribute=True)
    lim = np.max(np.abs(bzvtcs).ravel())
    kx = np.linspace(-lim, lim, int(np.sqrt(density) * 2 * lim), endpoint=True)
    kxx, kyy = np.meshgrid(kx, kx)
    kxy = np.dstack((kxx.ravel(), kyy.ravel()))[0]

    # Filter out points outside BZ
    bzvtcs = hlat.lattice.get_bz(attribute=True)
    inds = dh.inds_in_polygon(kxy, bzvtcs)
    kxy = kxy[inds]
    print 'hlatkspacefns: computing dispersion for kxy of length ' + str(len(kxy))
    omegas, kx, ky = infinite_dispersion_unstructured(hlat, kxy, save=save, save_plot=save_plot, overwrite=overwrite)

    return omegas