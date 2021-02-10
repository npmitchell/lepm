import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import cPickle as pkl
import glob
import lepm.gyro_lattice_functions as glatfns
import sys
import lepm.data_handling as dh
import lepm.lattice_elasticity as le
import time
import copy
import lepm.gyro_data_handling as gdh

'''
Description
===========
Auxiliary functions for GyroLattice class, for kspace methods.
Contains functions for defining BZ, computing band structure.
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
        minx, maxx = -1. / bboxx, 1. / bboxx
    elif ky is None or kx is None:
        bzvtcs = glat.lattice.get_bz(attribute=True)
        minx, maxx = np.min(bzvtcs[:, 0]), np.max(bzvtcs[:, 0])
        miny, maxy = np.min(bzvtcs[:, 1]), np.max(bzvtcs[:, 1])

    if kx is None:
        kx = np.linspace(minx, maxx, nkxvals, endpoint=True)

    if ky is None:
        ky = np.linspace(miny, maxy, nkyvals, endpoint=True)

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


def infinite_dispersion_unstructured(glat, kxy, load=False, save=True, save_plot=True,
                                     title='gyro dispersion relation', outdir=None, name=None, ax=None, lwscale=1.,
                                     verbose=False, overwrite=False, return_eigvects=False):
    """ Do not assume grid structure of kxy for this function. Compute the spectrum evaluated at the points given
    by the 2d array kxy.

    Parameters
    ----------
    glat
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
    name, kx, ky = prepare_dispersion_params(glat, kx=kxy[:, 0], ky=kxy[:, 1], outdir=outdir, name=name)
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

        omegas = np.zeros((len(kxy), len(glat.lattice.xy) * 2))
        if return_eigvects:
            eigvects = np.zeros((len(kxy), len(glat.lattice.xy) * 2, len(glat.lattice.xy) * 2), dtype=complex)
        matk = lambda k: dynamical_matrix_kspace(k, glat, eps=1e-10)
        ii = 0
        for pt in kxy:
            if ii % 50 == 1:
                print 'glatkspace_fns: infinite_dispersion(): ii = ', ii
            matrix = matk([pt[0], pt[1]])

            eigval, eigvect = np.linalg.eig(matrix)
            si = np.argsort(np.imag(eigval))
            omegas[ii, :] = np.imag(eigval[si])
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

    return omegas, kx, ky


def infinite_dispersion(glat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, save_plot=True,
                        title='gyro dispersion relation', outdir=None, name=None, ax=None, lwscale=1., verbose=False):
    """Compute the imaginary part of the eigvalues of the dynamical matrix for a grid (or unstructued set)
    of wavevectors kx, ky.
    See also calc_bands() and calc_band_gaps()

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
            if ii % 50 == 0:
                print 'glatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                # print 'glatkspace_fns: diagonalizing...'
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
            print 'saving ' + name + '.png'
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
    lambda function with input kvec
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, glat, eps=eps)


def dynamical_matrix_kspace(kvec, glat, eps=1e-9, basis=None, verbose=False):
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
        if verbose:
            print 'glatkfns: computing matk in psi basis'
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

#
# def calc_generalized_kmatrix_psi(kvec, alpha, glat, eps=1e-9):
#     """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
#     based loosely on chern_functions_gen.make_M(), except with a prescribed factor rescaling the coupling between right
#     and left moving polarizations, denoted alpha.
#
#     Parameters
#     ----------
#     kvec : 1 x 2 float array
#         The wavenumber vector at which to compute the dynamical matrix
#     alpha : float
#         Scale factor for the coupling between right and left polarizations
#     glat : GyroLattice instance
#         The network for which to compute the dynamical matrix
#     eps : float
#         Threshold below which to ignore elements of KL
#
#     Returns
#     -------
#     mat
#     """
#     # First use glat to create (angs, num_neis, bls, tvals, ons)
#     #
#     # angs : list
#     #     each row represents a site in the lattice.  Each entry in the row represents the angles to that site's
#     #     neighbors
#     # num_nei : list or array (num_sites x num_sites)
#     #     Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be
#     #     num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
#     # bls : len(angs) x  float array or int
#     #     bondlengths, with dimensions equal to dimensions of angs.
#     #     default value is an int, -1, indicating that all bond lengths are 1
#     # tvals : len(angs) x 1 float array or int
#     #     dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
#     # ons : array (dimension = num_sites per unit cell)
#     #     represents omega_g
#     xy = glat.lattice.xy
#     NL, KL = glat.lattice.NL, glat.lattice.KL
#     num_sites, NN = np.shape(NL)
#     Omg, OmK = glat.Omg, glat.OmK
#     PVx, PVy = glat.lattice.PVx, glat.lattice.PVy
#     if PVx is None or PVy is None:
#         PVx = np.zeros_like(NL, dtype=float)
#         PVy = np.zeros_like(NL, dtype=float)
#
#     # num_sites is the total number of particles
#     mm = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')
#
#     # Go through each site and fill in rows i and NP + i for that site (psi_L and psi_R)
#     matsave = None
#     for ii in range(num_sites):
#         omg = Omg[ii]  # grav frequency for this particle
#
#         # pinning/gravitational matrix -- note: will divide later by factor of -2
#         mm[ii, ii] += 2. * omg
#         mm[num_sites + ii, num_sites + ii] += -2. * omg
#
#         for nn in range(NN):
#             # the index of the gyroscope i is connected to (particle j)
#             ni = NL[ii, nn]
#             # true connection?
#             k = KL[ii, nn]
#             # spring frequency for this connection
#             omk = OmK[ii, nn]
#
#             # if ii == 2:
#             #     print 'NL = ', NL
#             #     print 'KL = ', KL
#             #     print 'OmK = ', OmK
#             #
#             # sys.exit()
#
#             if abs(k) > eps:
#                 # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
#                 diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
#                 diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
#                 alphaij = np.arctan2(diffy, diffx)
#                 # print '\n\ndiffx, diffy = ', diffx, diffy
#                 # print 'alphaij = ', alphaij
#
#                 # Form kfactor
#                 if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
#                     kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
#                     nkfactor = np.exp(-1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
#                     # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
#                 else:
#                     kfactor = 1.0
#                     nkfactor = 1.0
#
#                 # Create phase factors
#                 expi2t = np.exp(1j * 2. * alphaij)
#                 exp_negi2t = np.exp(-1j * 2. * alphaij)
#
#                 # (psi_L psi_L components)
#                 # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
#                 mm[ii, ii] += omk
#                 mm[ii, ni] += -omk * kfactor
#
#                 # (psi_L psi_R components) top right chunk
#                 mm[ii, ii + num_sites] += omk * expi2t * alpha
#                 # mm[ii, ni + num_sites] += -omk * expi2t * nkfactor * alpha
#                 mm[ii, ni + num_sites] += -omk * expi2t * kfactor * alpha
#
#                 # (psi_R psi_L components) bottom left chunk
#                 mm[ii + num_sites, ii] += -omk * exp_negi2t * alpha
#                 mm[ii + num_sites, ni] += omk * exp_negi2t * kfactor * alpha
#
#                 # (psi_R psi_R components) bottom right chunk
#                 mm[ii + num_sites, ii + num_sites] += -omk
#                 # mm[ii + num_sites, ni + num_sites] += omk * nkfactor
#                 mm[ii + num_sites, ni + num_sites] += omk * kfactor
#
#                 # if ii == 0:
#                 #     print 'kfactor = ', kfactor
#                 #     print 'nkfactor = ', nkfactor
#                 #     print 'expi2t = ', expi2t
#                 #     print 'alphaij = ', alphaij
#                 #     print '-omk * expi2t * nkfactor = ', -omk * expi2t * nkfactor
#                 #     print 'PV[ii, nn] = ', PVx[ii, nn], PVy[ii, nn]
#                 #     print 'mm[0] = ', -0.5 * mm[0] * (-1j)
#                 #     if matsave is None:
#                 #         contrib = -0.5 * mm * (-1j)
#                 #     else:
#                 #         contrib = -0.5 * mm * (-1j) - (-0.5 * matsave * (-1j))
#                 #     print 'contrib = ', contrib
#                 #     matsave = copy.deepcopy(mm)
#
#     mat = -0.5 * mm * (-1j)
#     # mat = 0.5 * mm
#     # print 'gyro_lattice_kspace_functions.py: ', mat
#     # print 'omk = ', omk
#     # print 'omg = ', omg
#     # sys.exit()
#     return mat


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
    # # Extract essentials from glat
    # xy = glat.lattice.xy
    # NL, KL = glat.lattice.NL, glat.lattice.KL
    # OmK = glat.OmK
    # Omg = glat.Omg
    # PVx, PVy = glat.lattice.PVx, glat.lattice.PVy
    #
    # try:
    #     NP, NN = np.shape(NL)
    # except ValueError:
    #     '''There is only one particle.'''
    #     NP, NN = 1, 0
    #
    # M1 = np.zeros((2 * NP, 2 * NP))
    # M2 = np.zeros((2 * NP, 2 * NP))
    #
    # # Unpack periodic boundary vectors
    # if PVx is None:
    #     PVx = np.zeros((NP, NN), dtype=float)
    #     PVy = np.zeros((NP, NN), dtype=float)
    #
    # print 'Constructing dynamical matrix...'
    # for i in range(NP):
    #     # grav frequency for this connection
    #     omg = Omg[i]
    #
    #     # pinning/gravitational matrix
    #     M2[2 * i, 2 * i + 1] = - omg
    #     M2[2 * i + 1, 2 * i] = omg
    #
    #     for nn in range(NN):
    #         # the index of the gyroscope that is connected to gyro ii (ni is a neighbor)
    #         ni = NL[i, nn]
    #         # true connection?
    #         k = KL[i, nn]
    #         # spring frequency for this connection
    #         omk = OmK[i, nn]
    #
    #         if abs(k) > eps:
    #             # There is a true connection, so update dynamical matrix
    #             # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
    #             # Compute factor from fourier transform
    #             diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
    #             diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
    #             if np.abs(PVx[i, nn]) > eps or np.abs(PVy[i, nn]) > eps:
    #                 kfactor = np.exp(1j * (PVx[i, nn] * kvec[0] + PVy[i, nn] * kvec[1]))
    #             else:
    #                 kfactor = 1.0
    #
    #             # Add kfactor to each bond? Wrong
    #             # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
    #             # Add kfactor to all bonds based on absolute position?
    #             # kfactor = np.exp(1j * ((xy[ni, 0] + PVx[i, nn]) * kvec[0] + (xy[ni, 1] + PVy[i, nn]) * kvec[1]))
    #             alphaij = np.arctan2(diffy, diffx)
    #
    #             # # What is this for?
    #             # if k == -2:  # will only happen on first or last gyro in a line
    #             #     if i == 0 or i == (NP - 1):
    #             #         print i, '--> NL=-2 for this particle'
    #             #         yy = np.where(KL[i] == 1)
    #             #         dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
    #             #         dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
    #             #         al = (np.arctan2(dy, dx)) % (2 * np.pi)
    #             #         alphaij = np.pi - al
    #             #         if i == 1:
    #             #             alphaij = np.pi - (45. * np.pi / 180.)
    #             #         else:
    #             #             alphaij = - (45. * np.pi / 180.)
    #
    #             Cos = np.cos(alphaij)
    #             Sin = np.sin(alphaij)
    #
    #             if abs(Cos) < eps:
    #                 Cos = 0.0
    #
    #             if abs(Sin) < eps:
    #                 Sin = 0.0
    #
    #             # Invoke kvector here
    #             Cos2 = Cos ** 2
    #             Sin2 = Sin ** 2
    #             CosSin = Cos * Sin
    #
    #             # (x components)
    #             M1[2 * i, 2 * i] += -omk * CosSin  # dxi - dxi
    #             M1[2 * i, 2 * i + 1] += -omk * Sin2  # dxi - dyi
    #             M1[2 * i, 2 * ni] += omk * CosSin * kfactor  # dxi - dxj
    #             M1[2 * i, 2 * ni + 1] += omk * Sin2 * kfactor  # dxi - dyj
    #
    #             # (y components)
    #             M1[2 * i + 1, 2 * i] += omk * Cos2  # dyi - dxi
    #             M1[2 * i + 1, 2 * i + 1] += omk * CosSin  # dyi - dyi
    #             M1[2 * i + 1, 2 * ni] += -omk * Cos2 * kfactor  # dyi - dxj
    #             M1[2 * i + 1, 2 * ni + 1] += -omk * CosSin * kfactor  # dyi - dyj
    #
    # # self.pin_array.append(2*pi*1*extra_factor)
    # # Assumes that b=0, c=1 so that:
    # # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    # matrix = 0.5 * M1 + M2
    #
    # return matrix


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
    matsave = None
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

            # if ii == 2:
            #     print 'NL = ', NL
            #     print 'KL = ', KL
            #     print 'OmK = ', OmK
            #
            # sys.exit()

            if abs(k) > eps:
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                alphaij = np.arctan2(diffy, diffx)
                # print '\n\ndiffx, diffy = ', diffx, diffy
                # print 'alphaij = ', alphaij

                # Form kfactor
                if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
                    kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                    nkfactor = np.exp(-1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                    # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                else:
                    kfactor = 1.0
                    nkfactor = 1.0

                # Create phase factors
                expi2t = np.exp(1j * 2. * alphaij)
                exp_negi2t = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
                mm[ii, ii] += omk
                mm[ii, ni] += -omk * kfactor

                # (psi_L psi_R components) top right chunk
                mm[ii, ii + num_sites] += omk * expi2t
                # mm[ii, ni + num_sites] += -omk * expi2t * nkfactor
                mm[ii, ni + num_sites] += -omk * expi2t * kfactor

                # (psi_R psi_L components) bottom left chunk
                mm[ii + num_sites, ii] += -omk * exp_negi2t
                mm[ii + num_sites, ni] += omk * exp_negi2t * kfactor

                # (psi_R psi_R components) bottom right chunk
                mm[ii + num_sites, ii + num_sites] += -omk
                # mm[ii + num_sites, ni + num_sites] += omk * nkfactor
                mm[ii + num_sites, ni + num_sites] += omk * kfactor

                # if ii == 0:
                #     print 'kfactor = ', kfactor
                #     print 'nkfactor = ', nkfactor
                #     print 'expi2t = ', expi2t
                #     print 'alphaij = ', alphaij
                #     print '-omk * expi2t * nkfactor = ', -omk * expi2t * nkfactor
                #     print 'PV[ii, nn] = ', PVx[ii, nn], PVy[ii, nn]
                #     print 'mm[0] = ', -0.5 * mm[0] * (-1j)
                #     if matsave is None:
                #         contrib = -0.5 * mm * (-1j)
                #     else:
                #         contrib = -0.5 * mm * (-1j) - (-0.5 * matsave * (-1j))
                #     print 'contrib = ', contrib
                #     matsave = copy.deepcopy(mm)

    mat = -0.5 * mm * (-1j)
    # mat = 0.5 * mm
    # print 'gyro_lattice_kspace_functions.py: ', mat
    # print 'omk = ', omk
    # print 'omg = ', omg
    # sys.exit()
    return mat


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


################################################
# Functions for computing band gaps
################################################
def calc_bands(glat, kxy=None, density=100, verbose=True):
    """Compute the band eigenvalues at an unstructured array of kxy points.

    Parameters
    ----------
    glat : GyroLattice class instance
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
    matk = lambda_matrix_kspace(glat, eps=1e-10)
    bzvtcs = glat.lattice.get_bz(attribute=True)
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


def get_matrix_at_kpt(glat, kxy, verbose=False, check=False, eps=1e-10):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if len(np.shape(kxy)) == 1:
        kxy = np.array(kxy)

    matk = dynamical_matrix_kspace(kxy, glat, eps=eps)
    return matk


def get_band_eigs_at_kpt(glat, kxy, verbose=False, check=False):
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

    (eigs, eigvs), kxout, kyout = infinite_dispersion_unstructured(glat, kxy, save=False, save_plot=False,
                                                                   verbose=verbose, overwrite=False,
                                                                   return_eigvects=True)
    return eigs, eigvs


def calc_band_limits(glat, omegas=None, kxy=None, density=100, verbose=True):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_limits -LT hexagonal -N 1

    Parameters
    ----------
    glat : GyroLattice class instance
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
        omegas, kx, ky = calc_bands(glat, kxy=kxy, density=density, verbose=verbose)

    # Given omegas, compute the tops and bottoms of the bands
    mins = np.min(omegas, axis=0)
    maxs = np.max(omegas, axis=0)
    limits = np.dstack((mins, maxs))[0]
    return limits


def calc_band_gaps(glat, omegas=None, kxy=None, density=100, verbose=True):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_limits -LT hexagonal -N 1

    Parameters
    ----------
    glat
    omegas : #kpts x #bands float array
        the frequencies of the band structure evaluated at each wavenumber
    kxy
    density
    verbose

    Returns
    -------

    """
    limits = calc_band_limits(glat, omegas=omegas, kxy=kxy, density=density, verbose=verbose)
    # Given band limits, find gaps
    tops = limits[:, 1][:-1]
    bots = limits[1:, 0]
    return np.dstack((tops, bots))[0]


def calc_band_bounds(glat, omegas=None, kxy=None, density=100, verbose=True, ngridpts=100, axis=0):
    """Compute the min and max of each band, which is defined over the supplied kxy or a random sampling of the BZ with
    provided density if kxy is not supplied.

    Example usage
    -------------
    python gyro_lattice_class.py -band_bounds -LT hexagonal -N 1
    # or
    polygons = glat.calc_band_bounds()
    for polygon in polygons:
        poly = Polygon(polygon, closed=True, fill=True, lw=0.50, alpha=0.5, color='g', edgecolor='k')
        ax.add_artist(poly)


    Parameters
    ----------
    glat : GyroLattice class instance
    omegas : len(kxy) x #bands float array
        eigenfrequencies of the GyroLattice defined over some sampling of BZ, kxy.
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
        omegas, kx, ky = calc_bands(glat, kxy=kxy, density=density, verbose=verbose)
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


def sample_dos_in_BZ(glat, density=100, save=True, save_plot=True, overwrite=False):
    """Take a random sampling of kxy wavecectors in the Brillouin zone and evaluate & return their eigenvalues

    Parameters
    ----------
    glat
    density

    Returns
    -------
    eigvals : 1d float array
        the eigenvalues in the BZ, sampled over evenly spaced grid of wavevectors
    """
    bzvtcs = glat.lattice.get_bz(attribute=True)
    lim = np.max(np.abs(bzvtcs).ravel())
    kx = np.linspace(-lim, lim, int(np.sqrt(density) * 2 * lim), endpoint=True)
    kxx, kyy = np.meshgrid(kx, kx)
    kxy = np.dstack((kxx.ravel(), kyy.ravel()))[0]

    # Filter out points outside BZ
    bzvtcs = glat.lattice.get_bz(attribute=True)
    inds = dh.inds_in_polygon(kxy, bzvtcs)
    kxy = kxy[inds]
    print 'glatkspacefns: computing dispersion for kxy of length ' + str(len(kxy))
    omegas, kx, ky = infinite_dispersion_unstructured(glat, kxy, save=save, save_plot=save_plot, overwrite=overwrite)

    return omegas