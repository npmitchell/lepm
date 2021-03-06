import numpy as np
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as lsegs
import lepm.stringformat as sf
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import glob
import scipy.stats as scistat
import copy
import lepm.line_segments as linsegs
try:
    import cPickle as pickle
except ImportError:
    import pickle
from lepm.gyro_lattice_functions import calc_magevecs


'''Auxiliary functions supporting gyro_lattice_class.py for measuring localization of normal modes'''


def fit_eigvect_to_exponential(xy, eigval, eigvect, cutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay: finite system, non-PBCs

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    cutoffd : float
        cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
    halfeval = eigval[int(len(eigval) * 0.5):]
    halfevec = eigvect[int(len(eigval) * 0.5):]

    # Contract x and y component of each evect so that len(halfevec[ii]) = NP
    magevec = np.zeros((len(halfevec), len(halfevec)), dtype=float)
    jj = 0
    for row in halfevec:
        magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
                                for ii in np.arange(0, int(len(eigvect) * 0.5))])
        jj += 1

    # find COM for each eigvect
    fits = np.zeros((len(halfeval), 7), dtype=float)
    for ii in range(len(halfeval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(halfeval)
        com = le.center_of_mass(xy, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle
        dist = np.sqrt((xy[:, 0] - com[0])**2 + (xy[:, 1] - com[1])**2)

        # Sort particles by distance from com
        inds = np.argsort(dist)
        # print 'dist[inds] = ', dist[inds]
        if cutoffd is not None:
            inds = inds[dist[inds] < cutoffd]
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
            plt.title(r'$\omega = $' + str(halfeval[ii]) + Astr + Kstr)
            if np.imag(halfeval[ii]) < 1.1:
                plt.show()
            elif 2.5 > np.imag(halfeval[ii]) > 2.1:
                plt.pause(1)
            else:
                plt.pause(0.001)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.imag(halfeval), fits[:, 2])
        plt.show()
    return fits


def fit_eigvect_to_exponential_periodic(xy, eigval, eigvect, LL, locutoffd=None, hicutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)

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
    cutoffd : float
        cutoff distance for fitting exponential
    check : bool
        show intermediate steps

    Returns
    -------
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
    halfeval = eigval[int(len(eigval) * 0.5):]
    halfevec = eigvect[int(len(eigval) * 0.5):]

    # Contract x and y component of each evect so that len(halfevec[ii]) = NP
    magevec = np.zeros((len(halfevec), len(halfevec)), dtype=float)
    jj = 0
    for row in halfevec:
        magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
                                for ii in np.arange(0, int(len(eigvect) * 0.5))])
        jj += 1

    # find COM for each eigvect
    fits = np.zeros((len(halfeval), 7), dtype=float)
    for ii in range(len(halfeval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(halfeval)
        com = le.com_periodic(xy, LL, magevec[ii])
        fits[ii, 0:2] = com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distance_periodic(xy, com, LL)

        # Check it
        inds = np.argsort(dist)
        if hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        if locutoffd is not None:
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
            plt.title(r'$\omega = $' + str(halfeval[ii]) + Astr + Kstr)
            if np.imag(halfeval[ii]) < 1.1:
                plt.show()
            elif 2.5 > np.imag(halfeval[ii]) > 2.1:
                plt.pause(1)
            else:
                plt.pause(0.001)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.imag(halfeval), fits[:, 2])
        plt.show()
    return fits


# def fit_eigvect_to_exponential_periodicstrip(xy, eigval, eigvect, LL, cutoffd=None, check=False):
#     """Fit the excitations of a gyro lattice to exponential decay (for detecting and measuring localized states)
#
#     Parameters
#     ----------
#     xy : NP x 2 float array
#         positions of particles in 2D
#     eigval : 2*NP x 1 complex array
#         eigenvalues of the system
#     eigvect : 2*NP x 2*NP complex array
#         eigenvectors of the system
#     LL : float
#         width of the periodic system in 2d
#     cutoffd : float
#         cutoff distance for fitting exponential
#     check : bool
#         show intermediate steps
#
#     Returns
#     -------
#     fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
#         fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
#     """
#     # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
#     halfeval = eigval[int(len(eigval) * 0.5):]
#     halfevec = eigvect[int(len(eigval) * 0.5):]
#
#     # Contract x and y component of each evect so that len(halfevec[ii]) = NP
#     magevec = np.zeros((len(halfevec), len(halfevec)), dtype=float)
#     jj = 0
#     for row in halfevec:
#         magevec[jj] = np.array([np.sqrt(np.abs(row[2*ii])**2 + np.abs(row[2*ii + 1])**2)
#                                 for ii in np.arange(0, int(len(eigvect) * 0.5))])
#         jj += 1
#
#     # find COM for each eigvect
#     fits = np.zeros((len(halfeval), 7), dtype=float)
#     for ii in range(len(halfeval)):
#         if ii % 100 == 1:
#             print 'glfns: eval #', str(ii), '/', len(halfeval)
#         com = le.com_periodicstrip(xy, LL, magevec[ii])
#         fits[ii, 0:2] = com
#
#         # Get minimum distance for each particle, taking care to get minimum across periodic BCs
#         dist = le.distance_periodicstrip(xy, com, LL)
#
#         # Check it
#         inds = np.argsort(dist)
#         if cutoffd is not None:
#             inds = inds[dist[inds] < cutoffd]
#         mags = magevec[ii][inds]
#         dists = dist[inds]
#
#         # Check if exponential is possible
#         # kstest_exponential(magevec[ii], dist, bins=int(np.max(dist)))
#         # possibly_exp = possibly_exponential(dists, mags)
#
#         # if possibly_exp:
#         # Fit to an exponential y=A*exp(K*t) using linear method.
#         # METHOD 1
#         # bin_means, bin_edges, bin_number = scistat.binned_statistic(dists, mags, bins=int(cutoffd))
#         # A, K, cov = fit_exp_linear(binc, bin_means)
#         A, K, cov = fit_exp_linear(dists, mags)
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
#     return fits


# THIS FUNCTION IS IMPORTED, NOT REPRODUCED HERE
# def calc_magevecs(eigvect):
#     """Compute the magnitude of the second half of all eigenvectors, by norming their x and y components in quad
#     NOTE: This is the same function as in gyro_lattice_functions.py, copied for imoprt convenience.
#
#     Parameters
#     ----------
#     eigvect : 2*N x 2*N complex array
#         eigenvectors of the matrix, sorted by order of imaginary components of eigvals
#         Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
#         x0, y0, x1, y1, ... xNP, yNP.
#
#     Returns
#     -------
#     magevecs : #particles x #particles float array
#         The magnitude of the upper half of eigenvectors at each site. magevecs[i, j] is the magnitude of the i+NP
#         normal mode at site j.
#     """
#     return calc_magevecs(eigvect)


def fit_eigvect_to_exponential_1dperiodic(xy, eigval, eigvect, LL, locutoffd=None, hicutoffd=None, check=False):
    """Fit the excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x))

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
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
    fits : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
    halfeval = eigval[int(len(eigval) * 0.5):]
    halfevec = eigvect[int(len(eigval) * 0.5):]

    # Contract x and y component of each evect so that len(halfevec[ii]) = NP
    magevec = calc_magevecs(eigvect)

    # find COM for each eigvect
    fits = np.zeros((len(halfeval), 7), dtype=float)
    for ii in range(len(halfeval)):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len(halfeval)

        # le.plot_real_matrix(magevec, show=True)
        # print 'ii = ', ii
        # print 'np.shape(magevec[ii]) = ', np.shape(magevec[ii])
        # print 'np.min(magevec[ii]) = ', np.min(magevec[ii])
        # print 'np.max(magevec[ii]) = ', np.max(magevec[ii])
        com = le.com_periodicstrip(xy, LL, masses=magevec[ii])
        fits[ii, 0:2] = com
        # print 'com =', com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distancex_periodicstrip(xy, com, LL)

        # Check it
        inds = np.argsort(dist)
        if hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        if locutoffd is not None:
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
        fits[ii, 2] = A
        fits[ii, 3] = K
        fits[ii, 4] = cov[0, 0]
        fits[ii, 5] = cov[0, 1]
        fits[ii, 6] = cov[1, 1]

        # check the result
        if check:
            if np.abs(K) > 0.1:
                plt.clf()
                plt.plot(dists, mags, 'b.-')
                plt.plot(dists, A * np.exp(K*dists), 'r-')
                Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 6]))
                plt.title(r'$\omega = $' + str(halfeval[ii]) + Astr + Kstr)

                plt.show()
                com = le.com_periodicstrip(xy, LL, magevec[ii], check=True)
            # if np.imag(halfeval[ii]) > 3.7:
            #     # plt.show()
            #     pass
            # else:
            #     plt.pause(0.0001)
            #     pass

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.imag(halfeval), fits[:, 3])
        plt.show()

    return fits


def fit_eigvect_to_exponential_edge(xy, boundary, eigvect, eigval=None, locutoffd=None, hicutoffd=None, check=False,
                                    interp_n=None):
    """Fit the (edge) excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    LL : float
        width of the periodic system in 2d
    locutoffd : float
        low-end cutoff distance for fitting exponential
    hicutoffd : float
        hi-end cutoff distance for fitting exponential
    check : bool
        show intermediate steps
    interp_n : int
        The number of interpolation points along each boundary linesegment

    Returns
    -------
    fits : NP x 5 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
    """
    # convert eigvect to magnitude of eigenvectors, consider only half of eigvals
    halflength = int(len(eigvect) * 0.5)
    if check:
        try:
            halfeval = eigval[halflength:]
        except IndexError:
            raise RuntimeError("If kwarg 'check' is True, must supply eigval to fit_eigvect_to_exponential_edge()")

    halfevec = eigvect[halflength:]

    # Contract x and y component of each evect so that len(halfevec[ii]) = NP
    magevec = calc_magevecs(eigvect)

    # Get minimum distance for each particle, taking care to handle the interpolation of the boundary
    dist = le.distance_from_boundary(xy, boundary, interp_n=interp_n)

    # Fit the eigvals given distances of each particle from edge
    fits = np.zeros((halflength, 5), dtype=float)
    for ii in range(halflength):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', halflength

        # Check it
        inds = np.argsort(dist)
        if hicutoffd is not None:
            inds = inds[dist[inds] < hicutoffd]
        if locutoffd is not None:
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
                plt.title(r'$\omega = $' + str(halfeval[ii]) + Astr + Kstr)
                plt.pause(0.1)

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.imag(halfeval), fits[:, 1])
        plt.show()

    return fits


def topbottom_edgelocz_dispersion(glat, kx=None, ky=None, nkxvals=50, nkyvals=20,
                                 save=True, save_plot=True, title='Dispersion relation',
                                 save_dos_compare=False, outdir=None, name=None, ax=None, check=False, checkdir=None):
    """"""

    import lepm.gyro_lattice_kspace_functions as glatkspacefns
    import lepm.plotting.gyro_lattice_plotting_functions as glatpfns
    name, kx, ky = glatkspacefns.prepare_dispersion_params(glat, kx=None, ky=None, nkxvals=nkxvals, nkyvals=nkyvals,
                                                           outdir=None, name=None)
    print('checking for file: ' + name + '.pkl')
    elocname = name.replace('dispersion', 'dispersionelocz')
    if glob.glob(name + '.pkl') and glob.glob(elocname):
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
        elocz = np.zeros((len(kx), len(ky), len(glat.lattice.xy) * 2, 6))
        matk = lambda k: glatkspacefns.dynamical_matrix_kspace(k, glat, eps=1e-10)
        xy, PVxydict = glat.lattice.xy, glat.lattice.PVxydict
        boundaries = glat.lattice.get_boundary()
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
                eigvect = eigvect[si]
                # Save the edge localization in a dict
                elocz_ij = fit_eigvect_edge_boundaries(xy, boundaries, eigvect, PVxydict, eigval=eigval, check=check)
                nrow = np.shape(elocz_ij)[0]
                elocz_full = np.zeros((2 * nrow, 6), dtype=float)
                # keep only the top/bottom designation
                elocz_full[0:nrow] = elocz_ij[::-1]
                elocz_full[nrow:2 * nrow] = elocz_ij
                elocz[ii, jj, :, :] = elocz_full
                # plt.plot(np.imag(eigval), elocz_full[:, 1], 'k-')
                # plt.plot(np.imag(eigval), elocz_full[:, 5], 'b.-')
                # plt.show()
                # print 'glatfns_localization: exiting here'
                # sys.exit()

                # if check, plot these normal modes
                if check and checkdir is not None:
                    fig, DOS_ax, ax = glatpfns.draw_edge_localization_plots(glat, elocz_full, eigval, eigvect,
                                                                            outdir=checkdir, alpha=1.0, fontsize=8)


                jj += 1
            ii += 1

    return omegas, kx, ky, elocz


def bottommiddletop_edgelocz_dispersion(glat, kx=None, ky=None, nkxvals=50, nkyvals=20,
                                        save=True, save_plot=True, title='Dispersion relation',
                                        save_dos_compare=False, outdir=None, name=None, ax=None, check=False,
                                        checkdir=None, bndy_thres=0.333333):
    """Here, there is no fitting the edge localizaiton, instead just measure if there is more excitation on the top,
    middle, or bottom third of the system and use that as a proxy for localization bias (coloring band structures)

    """

    import lepm.gyro_lattice_kspace_functions as glatkspacefns
    import lepm.plotting.gyro_lattice_plotting_functions as glatpfns

    name, kx, ky = glatkspacefns.prepare_dispersion_params(glat, kx=None, ky=None, nkxvals=nkxvals, nkyvals=nkyvals,
                                                           outdir=None, name=None)

    print('checking for file: ' + name + '.pkl')
    elocname = name.replace('dispersion', 'dispersionelocz_bottommiddletop')
    if glob.glob(name + '.pkl') and glob.glob(elocname):
        saved = True
        with open(name + '.pkl', "rb") as fn:
            res = pkl.load(fn)

        omegas = res['omegas']
        kx = res['kx']
        ky = res['ky']
        elocz_tmb = res['bottommiddletop']
    else:
        # dispersion is not saved, compute it!
        saved = False
        if 'annulus' in glat.lp['meshfn']:
            tb = locz_innerouter_dispersion(glat, bndy_thres=bndy_thres)
        else:
            omegas = np.zeros((len(kx), len(ky), len(glat.lattice.xy) * 2))
            elocz = np.zeros((len(kx), len(ky), len(glat.lattice.xy) * 2, 4))
            matk = lambda k: glatkspacefns.dynamical_matrix_kspace(k, glat, eps=1e-10)
            xy, PVxydict = glat.lattice.xy, glat.lattice.PVxydict
            boundaries = glat.lattice.get_boundary()
            top, bottom = np.max(xy[:, 1]), np.min(xy[:, 1])
            below, above = (top - bottom) * bndy_thres + bottom, top - (top - bottom) * bndy_thres
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
                    eigvect = eigvect[si]
                    # Compute the bias in energy towards top, middle, bottom
                    elocz_ij = bottommiddletop_excitation(xy, eigvect, below=below, above=above, top=top, bottom=bottom,
                                                          check=check, checkdir=checkdir)
                    nrow = np.shape(elocz_ij)[0]
                    elocz_full = np.zeros((2 * nrow, 4), dtype=float)
                    # keep only the top/bottom designation
                    elocz_full[0:nrow] = elocz_ij[::-1]
                    elocz_full[nrow:2 * nrow] = elocz_ij
                    elocz[ii, jj, :, :] = elocz_full
                    # plt.plot(np.imag(eigval), elocz_full[:, 1], 'k-')
                    # plt.plot(np.imag(eigval), elocz_full[:, 5], 'b.-')
                    # plt.show()
                    # print 'glatfns_localization: exiting here'
                    # sys.exit()
                    jj += 1
                ii += 1

    return omegas, kx, ky, elocz


def bottommiddletop_excitation(xy, eigvect, below=None, above=None, top=None, bottom=None, check=False,
                               checkdir=None):
    """Determine if excitation is weighted toward top, middle or bottom: which chunk has the most excitation:
    0-bottom, 0.5-middle, 1.0-top

    Returns
    -------
    tmb : (#eigval*0.5) x 1 float array
        0 for bottom, 0.5 for middle, 1.0 for top
    """
    if above is None:
        if top is None:
            top = np.max(xy[:, 1])
        if bottom is None:
            bottom = np.min(xy[:, 1])

        above = (top - bottom) * 2. / 3. + bottom
    if below is None:
        if top is None:
            top = np.max(xy[:, 1])
        if bottom is None:
            bottom = np.min(xy[:, 1])

        below = (top - bottom) / 3. + bottom

    magevec = calc_magevecs(eigvect)
    # etop is the energy in the top third of the system, similar for ebot and emid
    # Note that each of etop ebot and emid should have lengths of #particles
    etop = np.sum(magevec[:, xy[:, 1] > above], axis=1)
    ebot = np.sum(magevec[:, xy[:, 1] < below], axis=1)
    emid = np.sum(magevec[:, np.logical_and(xy[:, 1] > below, xy[:, 1] < above)], axis=1)
    # Prepare ebot, emid, etop, indicator (0, 0.5, 1) as columns
    tmb = np.dstack((ebot, emid, etop, 0.5 * np.ones_like(xy[:, 0])))[0]
    tops = np.logical_and(etop > emid, etop > ebot)
    bots = np.logical_and(ebot > emid, ebot > etop)
    print 'bots = ', bots
    tmb[tops, 3] = 1.0
    tmb[bots, 3] = 0.0

    if check:
        # plot excitation magnitude, excitation, and result
        import lepm.plotting.plotting as leplt
        import lepm.dataio as dio
        fig, ax, cax = leplt.initialize_2panel_1cbar_centy()
        for (mag, kk) in zip(magevec, np.arange(len(xy[:, 0]))):
            ax[0].plot(xy[:, 1], mag, 'b.')
            ylims = ax[0].get_ylim()
            ax[0].plot([below, below], ylims, 'k--')
            ax[0].plot([above, above], ylims, 'k--')
            ax[0].set_title('tmb = {0:0.1f}'.format(tmb[kk]))
            ax[1].scatter(xy[:, 0], xy[:, 1], s=mag*250)
            ax[1].set_ylim(np.min(xy[:, 1]), np.max(xy[:, 1]))
            ax[1].set_xlim(np.min(xy[:, 0]), np.max(xy[:, 0]))
            if checkdir is not None:
                plt.savefig(dio.prepdir(checkdir) + 'magevec_{0:06d}'.format(kk) + '.png')
            else:
                plt.pause(1)
            ax[0].cla()
            ax[1].cla()
    return tmb


def locz_which_boundary_com(glat, bndy_thres, magfloor_thres=0.05):
    """Discern if excitation of each mode is localized to zeroth or first boundary by using a simple center of mass.

    Parameters
    ----------
    glat : GyroLattice instance
    bndy_thres : float
        threshold from a boundary for the center of mass to be to consider the mode to be localized to that boundary
    magfloor_thres : float
        fraction of the largest magnitude excitation in the normal mode below which to suppress any excitation in
        distance measurement

    Returns
    -------

    """
    mags = glat.get_magevecs(eigvect=None)
    bndy0, bndy1 = glat.lattice.get_boundary()

    # todo: be smarter about how to determine if we should use a radial boundary or a linear one
    # todo: generalize to get distance of com from all boundary particles and use proximity to nearest one as proxy
    if 'annulus' in glat.lp['meshfn']:
        # Get radial coordinate for each site
        rr = np.sqrt(glat.lattice.xy[:, 0] ** 2 + glat.lattice.xy[:, 1] ** 2)
        # Get inner cutoff from where two boundaries are
        outsider = np.mean(np.sqrt(glat.lattice.xy[bndy0, 0] ** 2 + glat.lattice.xy[bndy0, 1] ** 2))
        insider = np.mean(np.sqrt(glat.lattice.xy[bndy1, 0] ** 2 + glat.lattice.xy[bndy1, 1] ** 2))
        thickness = outsider - insider
        print 'insider = ', insider
        print 'outsider = ', outsider
        print 'thickness = ', thickness

        # for each mode, define the color by whether the center of mass is within bndy_thres * thickness of outer
        # or inner boundary
        # Note that mags is only half as long as eigval
        lenc = len(mags)
        tb = 0.5 * np.ones(2 * lenc, dtype=float)
        for kk in np.arange(len(mags)):
            magfloor = mags[kk]
            magfloor[magfloor < magfloor_thres * np.max(magfloor)] = 0.
            # Take center of mass
            rcom = np.sum(rr * magfloor) / np.sum(magfloor)
            # rcom = np.sqrt(com[0] ** 2 + com[1] ** 2)
            if np.abs(rcom - outsider) / thickness < bndy_thres:
                # localized to outside boundary
                tb[lenc + kk] = 0.0
            elif np.abs(rcom - insider) / thickness < bndy_thres:
                # localized to inside boundary
                tb[lenc + kk] = 1.0
            else:
                tb[lenc + kk] = 0.5
    else:
        # raise RuntimeError('todo: be smarter about how to determine if we should use a radial boundary '
        #                    'or a linear one. Also, generalize to get distance of com from all boundary '
        #                    'particles and use proximity to nearest one as proxy')

        # Get radial coordinate for each site
        yy = glat.lattice.xy[:, 1]
        outsidey = np.mean(glat.lattice.xy[bndy0, 1])
        insidey = np.mean(glat.lattice.xy[bndy1, 1])
        thickness = outsidey - insidey

        # for each mode, define the color by whether the center of mass is within bndy_thres * thickness of top
        # or bottom boundary
        # Note that mags is only half as long as eigval
        lenc = len(mags)
        tb = 0.5 * np.ones(2 * lenc, dtype=float)
        for kk in np.arange(len(mags)):
            magfloor = mags[kk]
            magfloor[magfloor < magfloor_thres * np.max(magfloor)] = 0.
            # Take center of mass
            ycom = np.sum(yy * magfloor) / np.sum(magfloor)
            # rcom = np.sqrt(com[0] ** 2 + com[1] ** 2)
            if np.abs(ycom - outsidey) / thickness < bndy_thres:
                # localized to outside boundary
                tb[lenc + kk] = 0.0
            elif np.abs(ycom - insidey) / thickness < bndy_thres:
                # localized to inside boundary
                tb[lenc + kk] = 1.0
            else:
                tb[lenc + kk] = 0.5

    # Copy the filled in part of tb to the first half
    tb[0:lenc] = tb[lenc:][::-1]
    return tb


def fit_eigvect_edge_boundaries(xy, boundary, eigvect, PVxydict, eigval=None, locutoffd=None, hicutoffd=None,
                                check=False, interp_n=None):
    """Fit the (edge) excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy : NP x 2 float array
        positions of particles in 2D
    boundary : tuple of two int arrays
        Each boundary of the periodic strip as an ordered int array
    eigvect : NP x NP complex array
        eigenvectors of the system
    PVxydict : dict

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
            # consider only half of eigvals
            halfeval = eigval[int(len(eigval)*0.5):]
        except IndexError:
            raise RuntimeError("If kwarg 'check' is True, must supply eigval to fit_eigvect_to_exponential_edge()")

    # convert eigvect to magnitude of eigenvectors
    len_halfevec = len(eigvect[int(len(eigvect) * 0.5):])

    # Take absolute value of excitations for magnitudes
    magevec = calc_magevecs(eigvect)

    # Get minimum distance for each particle, taking care to handle the interpolation of the boundary
    # Supplied 'boundary' must be a tuple, as the sample is periodic in one dimension
    # here boundary is a tuple of (usually two) boundaries, as if for periodic strip or openBC annulus
    dist_tup = le.distance_from_boundaries(xy, boundary, PVxydict, interp_n=interp_n)

    # print 'dist_tup = ', dist_tup

    # Fit the eigvals given distances of each particle from edge
    fits = np.zeros((len_halfevec, 6), dtype=float)
    for ii in range(len_halfevec):
        if ii % 100 == 1:
            print 'glfns_localization: eval #', str(ii), '/', len_halfevec

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

                # Check if fits better to a constant value, which is the case if exponential growth
                # print 'fits[ii, 4] = ', fits[ii, 4]
                # print 'variance = ', variance
                if fits[ii, 1] > 0:
                    # the fit was not localized on either boundary: it was diverging.
                    # Name this an indeterminate fit by setting fit param to zero and
                    # marking the boundary index as 2.
                    fits[ii, 0] = np.mean(mags)
                    fits[ii, 1] = 0.
                    fits[ii, 2] = np.var(mags)
                    fits[ii, 3] = 0.
                    fits[ii, 4] = 0.
                    fits[ii, 5] = 2

                # check the result
                if check:
                    K = fits[ii, 1]
                    A = fits[ii, 0]
                    if abs(K) > 0.05:
                        plt.close('all')
                        plt.semilogy(dists, mags, 'b.-')
                        plt.semilogy(dists, A * np.exp(K * dists), 'r-')
                        Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 2]))
                        Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                        plt.title(r'Fit for $\omega = $' + str(eigval[ii]) + Astr + Kstr)
                        print 'mags = ', mags
                        plt.pause(0.1)

            jj += 1

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.imag(halfeval), fits[:, 1])
        plt.title('Fit versus imag(Eigval)')
        plt.show()

    return fits


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