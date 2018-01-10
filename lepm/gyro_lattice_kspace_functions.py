import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import cPickle as pkl
import glob
import lepm.gyro_lattice_functions as glatfns
import sys

'''
Description
===========
Auxiliary functions for GyroLattice class, for kspace methods
'''


def prepare_dispersion_params(glat, kx=None, ky=None, nkxvals=50, nkyvals=20, outdir=None, name=None):
    """Prepare the filename and kx, ky for computing dispersion relation

    Returns
    """
    if not glat.lp['periodicBC']:
        raise RuntimeError('Cannot compute dispersion for open BC system')
    elif glat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(glat.lattice.lp['BBox'][:, 0]) - min(glat.lattice.lp['BBox'][:, 0])
    elif ky is None or kx is None:
        bboxx = max(glat.lattice.lp['BBox'][:, 0]) - min(glat.lattice.lp['BBox'][:, 0])
        bboxy = max(glat.lattice.lp['BBox'][:, 1]) - min(glat.lattice.lp['BBox'][:, 1])

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
        outdir = dio.prepdir(glat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'dispersion_gyro' + glat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
        name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    name = outdir + name
    return name, kx, ky


def infinite_dispersion(glat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                        title='gyro dispersion relation', outdir=None, name=None, ax=None, lwscale=1.):
    """Compute the imaginary part of the eigvalues of the dynamical matrix for a grid of wavevectors kx, ky

    Parameters
    ----------
    glat : GyroLattice class instance
        the gyro network whose dispersion we compute
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
    name, kx, ky = prepare_dispersion_params(glat, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
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

        omegas = np.zeros((len(kx), len(ky), len(glat.lattice.xy) * 2))
        matk = lambda k: dynamical_matrix_kspace(k, glat, eps=1e-10)
        ii = 0
        for kxi in kx:
            print 'glatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                print 'glatkspace_fns: diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.imag(eigval))
                omegas[ii, jj, :] = np.imag(eigval[si])
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
        ylims = ax.get_ylim()
        ax.set_ylim(0, ylims[1])
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


def vec(ang, bl=1):
    """creates a vector given a bond length and angle in radians"""
    return [bl * np.cos(ang), bl * np.sin(ang)]


def ang_fac(ang):
    """Convert angle to factor in the Hamiltonian for gyro+spring system"""
    return np.exp(2 * 1j * ang)


def lambda_matrix_kspace(glat, eps=1e-10):
    """Construct the dynamical matrix for the given GyroLattice as a function of an as-yet-unspecified wavevector kvec.

    Parameters
    ----------
    glat : GyroLattice class instance
        the gyro network whose dispersion we compute
    eps : float
        resolution for discerning if value of connectivity matrix is nonzero

    Returns
    -------
    lambda function
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, glat, eps=eps)


def dynamical_matrix_kspace(kvec, glat, eps=1e-9, basis=None):
    """Construct the dynamical matrix for the given GyroLattice for the given wavevector kvec.

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    glat : GyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------

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

    # grab basis from lp if it is a key
    if basis is None and 'basis' in glat.lp:
        basis = glat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        if glat.bL is None:
            # Rest lengths of springs == distances between particles
            if notwist:
                # not twisted, no stretch, XY basis
                matrix = calc_matrix_kvec(kvec, glat, eps=eps)
                # Using psi basis for now since it is the only one that works.
                # matrix = calc_kmatrix_psi(kvec, glat, eps=eps)
                # outname = '/Users/npmitchell/Desktop/test/' + 'kx{0:0.2f}'.format(kvec[0]) +\
                #           'ky{0:0.2f}'.format(kvec[1])
                # leplt.plot_complex_matrix(matrix, name='dynamical_matrix', outpath=outname)
            else:
                # twisted, no stretch, XY basis
                print 'PV = ', glat.lattice.PV
                print 'thetatwist = ', thetatwist
                print 'phitwist = ', phitwist
                if glat.lp['periodic_strip']:
                    # All periodic bonds are twisted
                    matrix = calc_kmatrix_gyros_twist(kvec, glat, eps=eps)
                else:
                    # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                    # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                    if 'annulus' in glat.lp['LatticeTop'] or glat.lp['shape'] == 'annulus':
                        twistcut = np.array([0., 0., np.max(glat.lattice.xy[:, 0]), 0.])
                        thetaKL = glatfns.form_twistedKL(kvec, glat, eps=eps)
                        phiKL = np.zeros_like(thetaKL, dtype=int)
                    else:
                        raise RuntimeError('Currently only have twistedKL set up for annular samples')

                    # Certain bonds are twisted, while the others are normal.
                    matrix = calc_kmatrix_gyros_twist_bonds(kvec, glat, thetaKL, phiKL, eps=eps)
        else:
            # Rest lengths of springs != distances between particles
            matrix = calc_kmatrix_gyros_stretched(kvec, glat, eps=eps)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_kmatrix_psi(kvec, glat, eps=eps)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in glat.lp:
        if glat.lp['immobile_boundary']:
            boundary = glat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def calc_matrix_kvec(kvec, glat, eps=1e-11):
    """Compute the dynamical matrix for d psi /dt = D psi. This code is intended for periodic boundary conditions since
    it is in kspace
    Example usage: python gyro_lattice_class.py -LT hexagonal -N 5 -periodic_strip -dispersion

    Parameters
    ----------
    kvec : 1 x 2 float array
        The values of k (wavenumber) over which to compute the dynamical matrix
    glat : GyroLattice instance
        The gyro network to consider
    eps : float
        threshold for considering an element in KL to be a real connection


    Returns
    -------

    """
    raise RuntimeError('This XY basis calculation is still broken!')
    # Extract essentials from glat
    xy = glat.lattice.xy
    NL, KL = glat.lattice.NL, glat.lattice.KL
    OmK = glat.OmK
    Omg = glat.Omg
    PVx, PVy = glat.lattice.PVx, glat.lattice.PVy

    try:
        NP, NN = np.shape(NL)
    except ValueError:
        '''There is only one particle.'''
        NP, NN = 1, 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))

    # Unpack periodic boundary vectors
    if PVx is None:
        PVx = np.zeros((NP, NN), dtype=float)
        PVy = np.zeros((NP, NN), dtype=float)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        # grav frequency for this connection
        omg = Omg[i]

        # pinning/gravitational matrix
        M2[2 * i, 2 * i + 1] = - omg
        M2[2 * i + 1, 2 * i] = omg

        for nn in range(NN):
            # the index of the gyroscope that is connected to gyro ii (ni is a neighbor)
            ni = NL[i, nn]
            # true connection?
            k = KL[i, nn]
            # spring frequency for this connection
            omk = OmK[i, nn]

            if abs(k) > eps:
                # There is a true connection, so update dynamical matrix
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                # Compute factor from fourier transform
                diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
                diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
                if np.abs(PVx[i, nn]) > eps or np.abs(PVy[i, nn]) > eps:
                    kfactor = np.exp(1j * (PVx[i, nn] * kvec[0] + PVy[i, nn] * kvec[1]))
                else:
                    kfactor = 1.0

                # Add kfactor to each bond? Wrong
                # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                # Add kfactor to all bonds based on absolute position?
                # kfactor = np.exp(1j * ((xy[ni, 0] + PVx[i, nn]) * kvec[0] + (xy[ni, 1] + PVy[i, nn]) * kvec[1]))
                alphaij = np.arctan2(diffy, diffx)

                # # What is this for?
                # if k == -2:  # will only happen on first or last gyro in a line
                #     if i == 0 or i == (NP - 1):
                #         print i, '--> NL=-2 for this particle'
                #         yy = np.where(KL[i] == 1)
                #         dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
                #         dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
                #         al = (np.arctan2(dy, dx)) % (2 * np.pi)
                #         alphaij = np.pi - al
                #         if i == 1:
                #             alphaij = np.pi - (45. * np.pi / 180.)
                #         else:
                #             alphaij = - (45. * np.pi / 180.)

                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)

                if abs(Cos) < eps:
                    Cos = 0.0

                if abs(Sin) < eps:
                    Sin = 0.0

                # Invoke kvector here
                Cos2 = Cos ** 2
                Sin2 = Sin ** 2
                CosSin = Cos * Sin

                # (x components)
                M1[2 * i, 2 * i] += -omk * CosSin  # dxi - dxi
                M1[2 * i, 2 * i + 1] += -omk * Sin2  # dxi - dyi
                M1[2 * i, 2 * ni] += omk * CosSin * kfactor  # dxi - dxj
                M1[2 * i, 2 * ni + 1] += omk * Sin2 * kfactor  # dxi - dyj

                # (y components)
                M1[2 * i + 1, 2 * i] += omk * Cos2  # dyi - dxi
                M1[2 * i + 1, 2 * i + 1] += omk * CosSin  # dyi - dyi
                M1[2 * i + 1, 2 * ni] += -omk * Cos2 * kfactor  # dyi - dxj
                M1[2 * i + 1, 2 * ni + 1] += -omk * CosSin * kfactor  # dyi - dyj

    # self.pin_array.append(2*pi*1*extra_factor)
    # Assumes that b=0, c=1 so that:
    # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    matrix = 0.5 * M1 + M2

    return matrix


def calc_kmatrix_gyros_twist(kvec, glat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    glat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_twist_bonds(kvec, glat, thetaKL, phiKL, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    glat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_stretched(kvec, glat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    glat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_psi(kvec, glat, eps=1e-11):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based loosely on chern_functions_gen.make_M().
    Not working yet...

    Parameters
    ----------
    kvec : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    glat : GyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    # First use glat to create (angs, num_neis, bls, tvals, ons)
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
    xy = glat.lattice.xy
    NL, KL = glat.lattice.NL, glat.lattice.KL
    num_sites, NN = np.shape(NL)
    Omg, OmK = glat.Omg, glat.OmK
    PVx, PVy = glat.lattice.PVx, glat.lattice.PVy
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


def dynamical_matrix_kspace_unitcell(k, glat, eps=1e-8):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based on chern_functions_gen.make_M().
    I think this is in psi basis...

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    glat : GyroLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    # First use glat to create (angs, num_neis, bls, tvals, ons)
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
    if glat.unit_cell is None:
        unitcell = glat.get_unitcell()
    else:
        unitcell = glat.unit_cell
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
