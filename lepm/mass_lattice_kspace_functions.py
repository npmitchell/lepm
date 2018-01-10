import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import cPickle as pkl
import glob
import lepm.twisty.twisty_functions as tfns
import sys
import lepm.brillouin_zone_functions as bzf
import lepm.data_handling as dh
import lepm.plotting.colormaps as lecmaps
import lepm.lattice_elasticity as le

'''Momentum-space functions for the MassLattice class
'''


def lowest_mode(mlat, nkxy=20, save=False, save_plot=True, name=None, outdir=None, imtype='png'):
    """

    Parameters
    ----------
    mlat : MassLattice class instance
    nkxy : int
    save :bool
    save_plot : bool
    name : str or None
    outdir : str or None
    imtype : str ('png', 'pdf', 'jpg', etc)

    Returns
    -------
    omegas : n x 1 float array
        the frequencies at each evaluated point in kspace
    kxy : n x 2 float array
        the kspace points at which the frequency of modes are evaluated
    vtcs : #vertices x 2 float array
        the vertices of the brillouin zone in kspace
    """
    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(mlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'lowest_mode' + mlat.lp['meshfn_exten']
        name += '_nkxy{0:06d}'.format(np.max(np.abs(nkxy)))

    name = outdir + name
    fn = glob.glob(name + '.pkl')
    if fn:
        with open(fn[0], "rb") as fn:
            res = pkl.load(fn)

        vtcs = res['vtcs']
        kxy = res['kxy']
        omegas = res['omegas']
    else:
        pvs = mlat.lattice.PV
        a1, a2 = pvs[0], pvs[1]
        vtx, vty = bzf.bz_vertices(a1, a2)
        vtcs = np.dstack((vtx, vty))[0]
        polygon = np.dstack((vtx, vty))[0]
        xlims = (np.min(vtx), np.max(vtx))
        ylims = (np.min(vty), np.max(vty))
        xextent, yextent = xlims[1] - xlims[0], ylims[1] - ylims[0]
        step = float(max(xextent, yextent)) / float(nkxy)
        print 'extent = ', xextent
        print 'step = ', step
        pts = dh.generate_gridpts_in_polygons(xlims, ylims, [polygon], dx=step, dy=step)
        # print 'tkspacefns: pts = ', np.shape(pts)
        omegas = np.zeros(len(pts))
        matk = lambda k: dynamical_matrix_kspace(k, mlat, eps=1e-10)
        ii = 0
        for kxy in pts:
            print 'mlatkspace_fns: infinite_dispersion(): ii = ', ii
            # print 'jj = ', jj
            kx, ky = kxy[0], kxy[1]
            matrix = matk([kx, ky])
            print 'mlatkspace_fns: diagonalizing...'
            eigval, eigvect = np.linalg.eig(matrix)
            si = np.argsort(np.real(-eigval))
            omegas[ii] = np.real(np.min(np.sqrt(-eigval)[si]))
            ii += 1

        # Save results to pickle if save == True
        res = {'omegas': omegas, 'kxy': pts, 'vtcs': vtcs}
        kxy = pts
        if save:
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    if save_plot:
        fig, ax, cax = leplt.initialize_1panel_cbar_cent(wsfrac=0.5, tspace=4)
        xgrid, ygrid, ZZ = dh.interpol_meshgrid(kxy[:, 0], kxy[:, 1], omegas, int(nkxy), method='nearest')
        inrois = dh.gridpts_in_polygons(xgrid, ygrid, [vtcs])
        vmax = np.max(omegas)
        ZZ[~inrois] = 0.0
        if ax is None:
            ax = plt.gca()
        pcm = ax.pcolormesh(xgrid, ygrid, ZZ, cmap=lecmaps.colormap_from_hex('#2F5179'), vmin=0., vmax=vmax, alpha=1.0)
        print 'vmax = ', vmax
        plt.colorbar(pcm, cax=cax, label=r'$\omega_0$', orientation='horizontal', ticks=[0., vmax])
        ax.axis('off')
        ax.axis('scaled')
        plt.savefig(name + '.' + imtype)

    return omegas, kxy, vtcs


def infinite_dispersion(mlat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                        title='twisty dispersion relation', outdir=None, name=None, ax=None):
    """Compute the imaginary part of the eigvalues of the dynamical matrix for a grid of wavevectors kx, ky

    Parameters
    ----------
    mlat : MassLattice class instance
        the twisty network whose dispersion we compute
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
    if not mlat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif mlat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(mlat.lattice.lp['BBox'][:, 0]) - min(mlat.lattice.lp['BBox'][:, 0])
    elif ky is None or kx is None:
        bboxx = max(mlat.lattice.lp['BBox'][:, 0]) - min(mlat.lattice.lp['BBox'][:, 0])
        bboxy = max(mlat.lattice.lp['BBox'][:, 1]) - min(mlat.lattice.lp['BBox'][:, 1])

    if kx is None:
        if nkxvals == 0:
            kx = np.array([0.])
        else:
            tmp = np.linspace(-1. / bboxx, 1. / bboxx, nkxvals - 1)
            step = np.diff(tmp)[0]
            kx = 2. * np.pi * np.linspace(-1. / bboxx, 1. / bboxx + step, nkxvals)
        # kx = np.linspace(-5., 5., 40)

    if ky is None:
        if nkyvals == 0:
            ky = np.array([0.])
        else:
            tmp = np.linspace(-1. / bboxy, 1. / bboxy, nkyvals - 1)
            step = np.diff(tmp)[0]
            ky = 2. * np.pi * np.linspace(-1. / bboxy, 1. / bboxy + step, nkyvals)
        # ky = np.linspace(-5., 5., 4)

    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(mlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'dispersion' + mlat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
        name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    name = outdir + name
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

        omegas = np.zeros((len(kx), len(ky), len(mlat.lattice.xy) * 2))
        matk = lambda k: dynamical_matrix_kspace(k, mlat, eps=1e-10)
        ii = 0
        for kxi in kx:
            print 'mlatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                print 'mlatkspace_fns: diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.real(eigval))
                omegas[ii, jj, :] = np.real(np.sqrt(-eigval[si]))
                # print 'eigvals = ', eigval
                # print 'omegas --> ', omegas[ii, jj]
                jj += 1
            ii += 1

    if save_plot:
        if ax is None:
            fig, ax = leplt.initialize_1panel_centered_fig()
            axsupplied = False
        else:
            axsupplied = True

        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k_x$ $[\langle \ell \rangle ^{-1}]$')
        ax.set_ylabel(r'$\omega$')
        ylims = ax.get_ylim()
        ylim0 = min(ylims[0], -0.1 * ylims[1])
        ax.set_ylim(ylim0, ylims[1])
        # Save the plot
        plt.savefig(name + '.png', dpi=300)

        ax.set_ylim(max(ylim0, -0.05 * ylims[1]), 0.05 * ylims[1])
        # Save the plot
        plt.savefig(name + '_zoom.png', dpi=300)

        # Fixed zoom
        ax.set_ylim(-0.3, 0.6)
        plt.savefig(name + '_zoom2.png', dpi=300)
        plt.close('all')

        # save plot of ky if no axis supplied
        if not axsupplied:
            fig, ax = leplt.initialize_1panel_centered_fig()
            for jj in range(len(kx)):
                for kk in range(len(omegas[jj, 0, :])):
                    ax.plot(ky, omegas[jj, :, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
            ax.set_title(title)
            ax.set_xlabel(r'$k_y$ $[\langle \ell \rangle ^{-1}]$')
            ax.set_ylabel(r'$\omega$')
            ylims = ax.get_ylim()
            ylim0 = min(ylims[0], -0.1 * ylims[1])
            ax.set_ylim(ylim0, ylims[1])
            # Save the plot
            plt.savefig(name + '_ky.png', dpi=300)

            ax.set_ylim(max(ylim0, -0.05 * ylims[1]), 0.05 * ylims[1])
            # Save the plot
            plt.savefig(name + '_zoom_ky.png', dpi=300)

            # Fixed zoom
            ax.set_ylim(-0.3, 0.6)
            plt.savefig(name + '_zoom2_ky.png', dpi=300)
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


def vec(ang, bl=1):
    """creates a vector given a bond length and angle in radians"""
    return [bl * np.cos(ang), bl * np.sin(ang)]


def ang_fac(ang):
    """Convert angle to factor in the Hamiltonian for gyro+spring system"""
    return np.exp(2 * 1j * ang)


def lambda_matrix_kspace(mlat, eps=1e-10):
    """Construct the dynamical matrix for the given MassLattice as a function of an as-yet-unspecified wavevector kvec.

    Parameters
    ----------
    mlat : MassLattice class instance
        the gyro network whose dispersion we compute
    eps : float
        resolution for discerning if value of connectivity matrix is nonzero

    Returns
    -------
    lambda function
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, mlat, eps=eps)


def calc_dynamical_matrix_kspace(kvec, mlat, eps=1e-9, basis=None):
    """Construct the dynamical matrix for the given MassLattice for the given wavevector kvec.

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mlat : MassLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------

    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in mlat.lp:
        thetatwist = mlat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in mlat.lp:
        phitwist = mlat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    # grab basis from lp if it is a key
    if basis is None and 'basis' in mlat.lp:
        basis = mlat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        if mlat.bL is None:
            # Rest lengths of springs == distances between particles
            if notwist:
                # not twisted, no stretch, XY basis
                matrix = dynamical_matrix_kspace(kvec, mlat, eps=eps)
                # Using psi basis for now since it is the only one that works.
                # matrix = calc_kmatrix_psi(kvec, mlat, eps=eps)
                # outname = '/Users/npmitchell/Desktop/test/' + 'kx{0:0.2f}'.format(kvec[0]) +\
                #           'ky{0:0.2f}'.format(kvec[1])
                # leplt.plot_complex_matrix(matrix, name='dynamical_matrix', outpath=outname)
            else:
                # twisted, no stretch, XY basis
                print 'PV = ', mlat.lattice.PV
                print 'thetatwist = ', thetatwist
                print 'phitwist = ', phitwist
                if mlat.lp['periodic_strip']:
                    # All periodic bonds are twisted
                    matrix = calc_kmatrix_gyros_twist(kvec, mlat, eps=eps)
                else:
                    # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                    # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                    if 'annulus' in mlat.lp['LatticeTop'] or mlat.lp['shape'] == 'annulus':
                        twistcut = np.array([0., 0., np.max(mlat.lattice.xy[:, 0]), 0.])
                        thetaKL = tfns.form_twistedKL(kvec, mlat, eps=eps)
                        phiKL = np.zeros_like(thetaKL, dtype=int)
                    else:
                        raise RuntimeError('Currently only have twistedKL set up for annular samples')

                    # Certain bonds are twisted, while the others are normal.
                    matrix = calc_kmatrix_gyros_twist_bonds(kvec, mlat, thetaKL, phiKL, eps=eps)
        else:
            # Rest lengths of springs != distances between particles
            matrix = calc_kmatrix_gyros_stretched(kvec, mlat, eps=eps)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_kmatrix_psi(kvec, mlat, eps=eps)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in mlat.lp:
        if mlat.lp['immobile_boundary']:
            boundary = mlat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def dynamical_matrix_kspace(kvec, mlat, eps=1e-10):
    """Compute the dynamical matrix for d^2 u/ dt^2 = D u. This code handles periodic or open boundary conditions

    Parameters
    ----------
    kvec : 2 x 1 float array or list of 2 floats
        the periodic wavenumbers
    mlat : MassLattice instance
    eps : float
        small number, values smaller than this are ignored

    Returns
    -------
    matrix : 2N x 2N float array
        The dynamical matrix for the positions of masses in the mass-spring network
    """
    lat = mlat.lattice
    NP, NN = lat.NL.shape
    M1 = np.zeros((2 * NP, 2 * NP), dtype=complex)
    M2 = np.zeros((2 * NP, 2 * NP), dtype=complex)

    if 'kpin' in mlat.lp:
        if np.abs(mlat.lp['kpin']) > 1e-10:
            add_pinning = True
        else:
            add_pinning = False
    else:
        add_pinning = False

    # Unpack periodic boundary vectors
    if lat.PVx is not None and lat.PVy is not None:
        PVx = lat.PVx
        PVy = lat.PVy
    elif lat.PVxydict:
        PVx, PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL)
    else:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    for ii in range(NP):
        for nn in range(NN):
            ni = lat.NL[ii, nn]
            # Note that this does not simply determine if true connection.
            # Instead finds stretch constant for this connection
            omk = np.abs(mlat.kk[ii, nn])

            # Compute factor from fourier transform
            diffx = lat.xy[ni, 0] - lat.xy[ii, 0] + PVx[ii, nn]
            diffy = lat.xy[ni, 1] - lat.xy[ii, 1] + PVy[ii, nn]

            kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

            if abs(omk) > 0:
                alphaij = np.arctan2(diffy, diffx)
                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)
                Cos2 = Cos ** 2
                Sin2 = Sin ** 2
                CosSin = Cos * Sin

                # Real equations (x components)
                massi = mlat.mass[ii]
                if massi == 0 or massi < 0:
                    raise RuntimeError('Encountered zero or negative mass: mass[' + str(i) + '] = ' + str(massi))

                # Add components to dynamical matrix of energy from displacement of
                # self due to interaction with neighbor
                M1[2 * ii, 2 * ii] += omk * Cos2 / massi
                M1[2 * ii, 2 * ii + 1] += omk * CosSin / massi
                M1[2 * ii + 1, 2 * ii] += omk * CosSin / massi
                M1[2 * ii + 1, 2 * ii + 1] += omk * Sin2 / massi

                # Add components for interation with neighbor
                M1[2 * ii, 2 * ni] += -omk * Cos2 * kfactor / massi
                M1[2 * ii, 2 * ni + 1] += -omk * CosSin * kfactor / massi
                M1[2 * ii + 1, 2 * ni] += -omk * CosSin * kfactor / massi
                M1[2 * ii + 1, 2 * ni + 1] += -omk * Sin2 * kfactor / massi

        if add_pinning:
            # pinning
            M2[2 * ii, 2 * ii] = mlat.pin[ii]
            M2[2 * ii + 1, 2 * ii + 1] = mlat.pin[ii]

    matrix = - M1 - M2
    return matrix


def calc_kmatrix_gyros_twist(kvec, mlat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_twist_bonds(kvec, mlat, thetaKL, phiKL, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_stretched(kvec, mlat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_psi(kvec, mlat, eps=1e-11):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based loosely on chern_functions_gen.make_M().
    Not working yet...

    Parameters
    ----------
    kvec : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mlat : MassLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    raise RuntimeError('Have not written yet!')
    # First use mlat to create (angs, num_neis, bls, tvals, ons)
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
    xy = mlat.lattice.xy
    NL, KL = mlat.lattice.NL, mlat.lattice.KL
    num_sites, NN = np.shape(NL)
    Omg, OmK = mlat.Omg, mlat.OmK
    PVx, PVy = mlat.lattice.PVx, mlat.lattice.PVy
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
            # the index of the twisty scope i is connected to (particle j)
            ni = NL[ii, nn]
            # true connection?
            k = KL[ii, nn]
            # spring frequency for this connection
            omk = OmK[ii, nn]

            if abs(k) > eps:
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                alphaij = np.arctan2(diffy, diffx)

                # Form kfactor
                if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
                    kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                    # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                else:
                    kfactor = 1.0

                # Create phase factors
                expi2t = np.exp(1j * 2. * alphaij)
                exp_negi2t = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
                mm[ii, ii] += omk
                mm[ii, ni] += -omk * kfactor

                # (psi_L psi_R components) top right chunk
                mm[ii, ii + num_sites] += omk * expi2t
                mm[ii, ni + num_sites] += -omk * expi2t * kfactor

                # (psi_R psi_L components) bottom left chunk
                mm[ii + num_sites, ii] += -omk * exp_negi2t
                mm[ii + num_sites, ni] += omk * exp_negi2t * kfactor

                # (psi_R psi_R components) bottom right chunk
                mm[ii + num_sites, ii + num_sites] += -omk
                mm[ii + num_sites, ni + num_sites] += omk * kfactor

    return -0.5 * mm * (-1j)


def dynamical_matrix_kspace_unitcell(k, mlat, eps=1e-8):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a twisty  + spring system,
    based on chern_functions_gen.make_M().
    I think this is in psi basis...

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mlat : MassLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    raise RuntimeError('Have not written yet!')
    # First use mlat to create (angs, num_neis, bls, tvals, ons)
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
    if mlat.unit_cell is None:
        unitcell = mlat.get_unitcell()
    else:
        unitcell = mlat.unit_cell
    if unitcell is None:
        raise RuntimeError('Network has no stored unit cell')

    angs = unitcell['angs']
    num_neis = unitcell['num_nei']
    bls = unitcell['bls']
    tvals = unitcell['tvals']
    ons = unitcell['ons']

    ons = list(ons)

    # num_sites is the total number of particles
    num_sites = len(angs)
    M = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    if bls == -1:
        bls = np.ones_like(angs)
    if tvals == -1:
        tvals = np.ones_like(angs)
    if ons == 1:
        ons = np.ones(num_sites)

    for i in range(len(M)):
        index = i % num_sites
        angs_for_row = angs[index]
        bls_for_row = bls[index]
        num_neis_row = num_neis[index]
        # num_bonds = len(angs[index])

        tv = tvals[index]
        num_bonds = sum(tv)
        # print 'num bonds', num_bonds

        # indices to keep track of what elements of dynamical matrix have already been filled
        fill_count = 0
        s_fill_count = 0
        for j in range(len(M)):
            if i == j:
                # Onsite term, on the diagonal of the dynamical matrix
                # For the top left chunk of the dynamical matrix, add positive term, bottom right add negative
                if i < num_sites:
                    # note that this will be divided by two later
                    M[i, j] = num_bonds + 2 * ons[index]
                else:
                    # note that this will be divided by two later
                    M[i, j] = - num_bonds - 2 * ons[index]
            else:
                ii = j % num_sites
                num_nei = num_neis_row[ii]
                # num_nei tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice
                # would be num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.

                # Determine if we are in the first (upper left) block of the matrix
                if i < num_sites and j < num_sites:
                    for l in range(num_nei):
                        M[i, j] += - tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i >= num_sites and j >= num_sites:
                    for l in range(num_nei):
                        M[i, j] += tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif (i < num_sites) and (j >= num_sites):
                    if j == num_sites + i:
                        M[i, j] = sum([tv[u] * ang_fac(angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            M[i, j] += -tv[s_fill_count] * ang_fac(angs_for_row[s_fill_count]) * \
                                       np.exp(1j * np.dot(k, vec(angs_for_row[s_fill_count],
                                                                 bls_for_row[s_fill_count])))
                            s_fill_count += 1
                elif (i >= num_sites) and (j < num_sites):
                    if j == (num_sites + i) % num_sites:
                        M[i, j] = -sum([tv[u] * ang_fac(-angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            M[i, j] += tv[s_fill_count] * ang_fac(-angs_for_row[s_fill_count]) * \
                                       np.exp(
                                           1j * np.dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            s_fill_count += 1

    return -0.5 * M
