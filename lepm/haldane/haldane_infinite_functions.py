import numpy as np
import matplotlib.pyplot as plt
import lepm.haldane.haldane_lattice_functions as hlatfns
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
from lepm.timer import Timer
import glob
import cPickle as pkl

'''Functions for computing the infinite kspace information for haldane model systems'''


def dynamical_matrix_kspace(kvec, hlat, eps=1e-8):
    """Compute the dynamical matrix for d psi /dt = D psi. This code is intended for periodic boundary conditions
    Example usage: python ./haldane/haldane_lattice_class.py -LT hexagonal -N 3 -periodic -dispersion

    Parameters
    ----------
    kvec : 1 x 2 float array
        The values of k (wavenumber) over which to compute the dynamical matrix
    hlat : HaldaneLattice instance
        The haldane model network to consider
    eps : float
        threshold for considering an element in KL to be a real connection

    Returns
    -------
    M1
    M2
    """
    pin = hlat.pin
    t1 = hlat.t1
    t2 = hlat.t2
    t2a = hlat.t2a
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
    pvxnnn, pvynnn = lat.get_pvnnn()
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
        dubtheta = hlatfns.calc_dubtheta(NLNNNangles)
        for i in range(NP):
            # Do nearest neighbor hoppings
            for nn in range(NN):
                ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                k = KL[i, nn]  # true connection?

                if abs(k) > eps:
                    # Compute factor from fourier transform
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

                if k > eps:
                    # Compute factor from fourier transform
                    diffx = lat.xy[ni, 0] - lat.xy[i, 0] + pvxnnn[i, nn]
                    diffy = lat.xy[ni, 1] - lat.xy[i, 1] + pvynnn[i, nn]

                    kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])
                    M2[i, ni] += (1j * t2 + t2a) * kfactor
                if k < -eps:
                    # There is a true NNN connection, so update dynamical matrix
                    # print 'hlatfns: positive: KLNNN[i,nn] = ', k, '           dubtheta = ', np.sin(dubtheta[i, nn])

                    # Compute factor from fourier transform
                    print 'ni = ', ni, ' i = ', i, ' nn = ', nn
                    diffx = lat.xy[ni, 0] - lat.xy[i, 0] + pvxnnn[i, nn]
                    diffy = lat.xy[ni, 1] - lat.xy[i, 1] + pvynnn[i, nn]

                    kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                    M2[i, ni] += (-1j * t2 + t2a) * kfactor

    matrix = M1 + M2
    matrix += pin * np.identity(NP)
    return matrix


def infinite_dispersion(hlat, kx=None, ky=None, save=True, title='Dispersion relation', outdir=None):
    """Compute energy versus wavenumber for a grid of kx ky values for a haldane model network.
    If the network is a periodic strip, then ky is set to zero and we look at E(kx).

    Parameters
    ----------
    hlat : HaldaneLattice instance
        the tight-binding network over which to compute the dispersion relation
    kx : N x 1 float array
        The x component of the wavenumbers to evaluate
    ky : M x 1 float array
        The y component of the wavenumbers to evaluate
    save : bool
        Whether to save the dispersion to disk
    title : str
        title of the plot to make if save is True
    outdir : str
        path to the place to save the plot if save is True

    Returns
    -------

    """
    if not hlat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif hlat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(hlat.lattice.lp['BBox'][:, 0]) - min(hlat.lattice.lp['BBox'][:, 0])
        print 'bboxx = ', bboxx
    elif ky is None or kx is None:
        bboxx = max(hlat.lattice.lp['BBox'][:, 0]) - min(hlat.lattice.lp['BBox'][:, 0])
        bboxy = max(hlat.lattice.lp['BBox'][:, 1]) - min(hlat.lattice.lp['BBox'][:, 1])

    if kx is None:
        kx = np.linspace(- 2. * np.pi / bboxx, 2. * np.pi / bboxx, 50)
        # kx = np.linspace(-5., 5., 40)

    if ky is None:
        ky = np.linspace(- 2. * np.pi / bboxy, 2. * np.pi / bboxy, 5)
        # ky = np.linspace(-5., 5., 4)

    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(hlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    name = outdir + 'dispersion' + hlat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
    name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
    name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')
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

        # Use PVx and PVy to multiply exp(i*np.dot(k, PV[0,:])) to periodic vectors in x, similar in y
        # First convert PVx and PVy to matrices that are the same shape as hlat.matrix
        # np.exp()

        omegas = np.zeros((len(kx), len(ky), len(hlat.lattice.xy)))
        matk = lambda k: dynamical_matrix_kspace(k, hlat, eps=1e-8)
        timer = Timer()
        ii = 0
        for kxi in kx:
            print 'infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                timer.restart()
                matrix = matk([kxi, kyj])
                test = timer.get_time_ms()
                print 'test = ', test
                print('constructed matrix in: ' + timer.get_time_ms())
                print 'diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                timer.restart()
                print('diagonalized matrix in: ' + timer.get_time_ms())
                si = np.argsort(np.real(eigval))
                omegas[ii, jj, :] = eigval[si]
                # print 'omegas = ', omegas
                jj += 1
            ii += 1

    if save:
        fig, ax = leplt.initialize_1panel_centered_fig()
        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k$ $[1/\langle \ell \rangle]$')
        ax.set_ylabel(r'$\omega$')
        plt.savefig(name + '.png')
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

        if not saved:
            res = {'omegas': omegas, 'kx': kx, 'ky': ky}
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    return omegas, kx, ky


