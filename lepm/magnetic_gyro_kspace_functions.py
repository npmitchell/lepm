import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import cPickle as pkl
import glob
import gyro_lattice_functions as glatfns
import lepm.plotting.colormaps as lecmap

'''
Functions for analyzing dispersion relations and infinite systems of magnetically interacting gyroscopes.
Interactions are by default isolated to nearest neighbors.
'''


def infinite_dispersion(mglat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                        title='Dispersion relation', outdir=None):
    """

    Parameters
    ----------
    mglat :
    kx :
    ky :
    save :
    title :
    outdir :

    Returns
    -------
    omegas, kx, ky
    """
    if not mglat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif mglat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]

    if ky is None or kx is None:
        bboxx = max(mglat.lattice.lp['BBox'][:, 0]) - min(mglat.lattice.lp['BBox'][:, 0])
        bboxy = max(mglat.lattice.lp['BBox'][:, 1]) - min(mglat.lattice.lp['BBox'][:, 1])

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
        outdir = dio.prepdir(mglat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    name = outdir + 'dispersion' + mglat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
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
        omegas = np.zeros((len(kx), len(ky), len(mglat.lattice.xy) * 2))
        matk = lambda k: dynamical_matrix_kspace(k, mglat, eps=1e-10)
        ii = 0
        for kxi in kx:
            if ii % 25 == 0:
                print 'infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.imag(eigval))
                omegas[ii, jj, :] = np.imag(eigval[si])
                # print 'omegas = ', omegas
                jj += 1
            ii += 1

    if save_plot:
        fig, ax = leplt.initialize_1panel_centered_fig()
        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
        ax.set_title(title)
        ax.set_xlabel(r'$k$ $[1/\langle l \rangle]$')
        ax.set_ylabel(r'$\omega$')
        print 'magnetic_gyro_kspace_functions: saving image to ' + name + '.png'
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


def lambda_matrix_kspace(glat, eps=1e-10):
    """Construct the dynamical matrix for the given MagneticGyroLattice as a function of an as-yet-unspecified
    wavevector kvec.

    Parameters
    ----------
    glat : GyroLattice class instance
        the gyro network whose dispersion we compute
    eps : float
        resolution for discerning if value of connectivity matrix is nonzero

    Returns
    -------
    lambda function taking a float array of length 2 as an argument
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, glat, eps=eps)


def dynamical_matrix_kspace(kvec, mglat, eps=1e-9, basis=None):
    """Construct the dynamical matrix for the given GyroLattice for the given wavevector kvec.

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mglat : MagneticGyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------

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

    # grab basis from lp if it is a key
    if basis is None and 'basis' in mglat.lp:
        basis = mglat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        if mglat.bL is None:
            # Rest lengths of springs == distances between particles
            if notwist:
                # not twisted, no stretch, XY basis
                matrix = calc_matrix_magnetic_kvec(kvec, mglat, eps=eps)
                # Using psi basis for now since it is the only one that works.
                # matrix = calc_kmatrix_psi(kvec, mglat, eps=eps)
                # outname = '/Users/npmitchell/Desktop/test/' + 'kx{0:0.2f}'.format(kvec[0]) +\
                #           'ky{0:0.2f}'.format(kvec[1])
                # leplt.plot_complex_matrix(matrix, name='dynamical_matrix', outpath=outname)
            else:
                # twisted, no stretch, XY basis
                print 'PV = ', mglat.lattice.PV
                print 'thetatwist = ', thetatwist
                print 'phitwist = ', phitwist
                if mglat.lp['periodic_strip']:
                    # All periodic bonds are twisted
                    matrix = calc_kmatrix_maggyros_twist(kvec, mglat, eps=eps)
                else:
                    # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                    # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                    if 'annulus' in mglat.lp['LatticeTop'] or mglat.lp['shape'] == 'annulus':
                        twistcut = np.array([0., 0., np.max(mglat.lattice.xy[:, 0]), 0.])
                        thetaKL = mglatfns.form_twistedKL(kvec, mglat, eps=eps)
                        phiKL = np.zeros_like(thetaKL, dtype=int)
                    else:
                        raise RuntimeError('Currently only have twistedKL set up for annular samples')

                    # Certain bonds are twisted, while the others are normal.
                    matrix = calc_kmatrix_maggyros_twist_bonds(kvec, mglat, thetaKL, phiKL, eps=eps)
        else:
            # Rest lengths of springs != distances between particles
            matrix = calc_kmatrix_maggyros_stretched(kvec, mglat, eps=eps)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_kmatrix_magnetic_psi(kvec, mglat, eps=eps)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in mglat.lp:
        if mglat.lp['immobile_boundary']:
            boundary = mglat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def calc_kmatrix_maggyros_twist(kvec, mglat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mglat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_maggyros_twist_bonds(kvec, mglat, thetaKL, phiKL, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mglat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_maggyros_stretched(kvec, mglat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    mglat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_magnetic_psi(kvec, mlat, eps=1e-11):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based loosely on chern_functions_gen.make_M().

    Parameters
    ----------
    kvec : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    mlat : GyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
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
    NL, KL = mlat.NL, mlat.KL
    num_sites, NN = np.shape(NL)
    Omg, OmK = mlat.Omg, mlat.OmK
    PVx, PVy = mlat.PVx, mlat.PVy
    if PVx is None or PVy is None:
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)

    # num_sites is the total number of particles
    mm = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    # checking
    # print 'np.shape(Omg) = ', np.shape(Omg)
    # print 'np.shape(NL) = ', np.shape(NL)
    # print 'np.shape(PVx) = ', np.shape(PVx)

    # Go through each site and fill in rows i and NP + i for that site (psi_L and psi_R)
    kk = 0
    for ii in mlat.inner_indices:
        # grav frequency for this particle (note the difference in indexing is due to inner/outer split)
        omg = Omg[kk]

        # pinning/gravitational matrix -- note: will divide later by factor of -2
        mm[ii, ii] += -2. * omg
        mm[num_sites + ii, num_sites + ii] += 2. * omg

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
                als = rij_mag ** 2 * (mlat.lp['aoverl']) ** 2

                # These are Nash SI eqn S6, multiplied by (l^2/I\omega)
                fpara_p = - omk * (1 - (1. / 12.) * als) / rij_mag ** 5
                fpara_q = omk * (1 + (1. / 6.) * als) / rij_mag ** 5
                fperp_p = omk * 0.25 * (1 + (1. / 3.) * als) / rij_mag ** 5
                fperp_q = -omk * 0.25 * (1 + (1. / 3.) * als) / rij_mag ** 5

                omk_i_plus = fpara_p + fperp_p
                omk_i_minus = fpara_p - fperp_p
                omk_j_plus = fpara_q + fperp_q
                omk_j_minus = fpara_q - fperp_q

                # Form kfactor
                if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
                    kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                else:
                    kfactor = 1.0

                # Create phase factors
                expi2t = np.exp(1j * 2. * alphaij)
                exp_negi2t = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
                mm[ii, ii] += omk_i_plus
                if ni in mlat.inner_indices:
                    mm[ii, ni] += -omk_j_plus * kfactor

                # (psi_L psi_R components) top right chunk
                mm[ii, ii + num_sites] += omk_i_minus * expi2t
                if ni in mlat.inner_indices:
                    mm[ii, ni + num_sites] += -omk_j_minus * expi2t * kfactor

                # (psi_R psi_L components) bottom left chunk
                mm[ii + num_sites, ii] += -omk_i_minus * exp_negi2t
                if ni in mlat.inner_indices:
                    mm[ii + num_sites, ni] += omk_j_minus * exp_negi2t * kfactor

                # (psi_R psi_R components) bottom right chunk
                mm[ii + num_sites, ii + num_sites] += -omk_i_plus
                if ni in mlat.inner_indices:
                    mm[ii + num_sites, ni + num_sites] += omk_j_plus * kfactor

        kk += 1

    return 0.5 * mm * (-1j)


def magnetic_dynamical_matrix_kspace_unitcell(k, mglat):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro system coupled magnetically to
    its neighbors (check: to all particles?)
    NOTE: Currently unused.

    Parameters
    ----------
    aol : float
        "a over l" -- interparticle distance (a) divided by the length of the pendula (l)

    Returns
    -------
    """
    # First use mglat to get (angs, num_neis, bls, tvals, ons, aol=0.8)
    #
    # angs : list
    #     each row represents a site in the lattice.  Each entry in the row represents the angles to that site's
    #     neighbors
    # num_nei : list or array (num_sites x num_sites)
    #     Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be
    #     num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
    # bls : len(angs) x 1 float array or int
    #     dimension equal to dimension of angs.  default value is -1 indicating that all bond lengths are 1
    # tvals : len(angs) x 1 float array or int
    #     dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
    # ons : array (dimension = num_sites pre unit cell)
    #     represents omega_g
    if mglat.unit_cell is None:
        unitcell = mglat.get_unitcell()
    else:
        unitcell = mglat.unit_cell
    if unitcell is None:
        raise RuntimeError('Network has no stored unit cell')

    angs = unitcell['angs']
    num_nei = unitcell['num_nei']
    bls = unitcell['bls']
    tvals = unitcell['tvals']
    ons = unitcell['ons']

    num_sites = len(angs)
    Ok = tvals[0][0]

    tvals = list(np.ones_like(tvals))
    if aol > 0:
        a = aol ** 2.
        Op = (1. + a / 6. - (1. / 4 + a / 12.)) * Ok
        Om = (1. + a / 6. + (1. / 4 + a / 12.)) * Ok
    else:
        Op = 1.
        Om = 1.

    M = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex')

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
        num_bonds = len(angs[index])

        tv = tvals[index]
        num_bonds = sum(tv)

        ff = 0
        tt = tv[2] * (1. + 2. * np.sin(20 * np.pi / 180.))

        fill_count = 0
        s_fill_count = 0
        for j in range(len(M)):
            if i == j:
                if i < num_sites:
                    M[i, j] = Op * num_bonds + 2 * (ons[index] - (3. / 8) * a * Ok)
                else:
                    M[i, j] = - Om * num_bonds - 2 * (ons[index] - (3. / 8) * a * Ok)
            else:
                ii = j % num_sites
                num_nei = num_neis_row[ii]
                # print 'num nei',  num_nei

                if i < num_sites and j < num_sites:
                    for l in range(num_nei):
                        M[i, j] += - Op * tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i >= num_sites and j >= num_sites:
                    for l in range(num_nei):
                        M[i, j] += Om * tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i < num_sites and j >= num_sites:
                    if j == num_sites + i:
                        M[i, j] = Om * np.sum([tv[u] * ang_fac(angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            vv = vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])
                            M[i, j] += - Om * tv[s_fill_count] * ang_fac(angs_for_row[s_fill_count]) * \
                                       np.exp(1j * np.dot(k, vv))
                            s_fill_count += 1
                elif i >= num_sites and j < num_sites:
                    if j == (num_sites + i) % num_sites:
                        M[i, j] = - Op * np.sum([tv[u] * ang_fac(-angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            vv = vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])
                            M[i, j] += Om * tv[s_fill_count] * ang_fac(-angs_for_row[s_fill_count]) * \
                                       np.exp(1j * np.dot(k, vv))
                            s_fill_count += 1

    return -0.5 * M


def compare_dispersion_to_dos(omegas, kx, ky, mlat, outdir=None):
    """Compare the projection of the dispersion onto the omega axis with the DOS of the MagneticGyroLattice

    Parameters
    ----------
    omegas
    kx
    ky
    mlat
    outdir

    Returns
    -------

    """
    # Save DOS from projection
    if outdir is None:
        outdir = dio.prepdir(mlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)
    name = outdir + 'dispersion_gyro' + mlat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
    name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
    name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    # initialize figure
    fig, ax = leplt.initialize_1panel_centered_fig()
    ax2 = ax.twinx()
    ax.hist(omegas.ravel(), bins=1000)

    # Compare the histograms of omegas to the dos and save the figure
    eigval = np.imag(mlat.get_eigval())
    print 'eigval = ', eigval
    ax2.hist(eigval[eigval > 0], bins=50, color=lecmap.green(), alpha=0.2)
    ax.set_title('DOS from dispersion')
    xlims = ax.get_xlim()
    ax.set_xlim(0, xlims[1])
    plt.savefig(name + '_dos.png', dpi=300)
