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
try:
    import cPickle as pickle
except ImportError:
    import pickle


'''Auxiliary functions supporting gyro_lattice_class.py for measuring localization of normal modes'''


def fit_eigvect_to_exponential(xy_inner, eigval, eigvect, cutoffd=None, check=False):
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

    # find COM for each eigvect, using only the inner particles
    fits = np.zeros((len(halfeval), 7), dtype=float)
    xy = xy_inner
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


def fit_eigvect_to_exponential_periodic(xy_inner, eigval, eigvect, LL,
                                        locutoffd=None, hicutoffd=None, check=False):
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
    xy = xy_inner
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


def calc_magevecs(eigvect):
    """Compute the magnitude of the second half of all eigenvectors, by norming their x and y components in quad
    NOTE: This is the same function as in gyro_lattice_functions.py, copied for imoprt convenience.

    Parameters
    ----------
    eigvect : 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.

    Returns
    -------
    magevecs : #particles x #particles float array
        The magnitude of the upper half of eigenvectors at each site. magevecs[i, j] is the magnitude of the i+NP
        normal mode at site j.
    """
    from lepm.gyro_lattice_functions import calc_magevecs
    return calc_magevecs(eigvect)


def fit_eigvect_to_exponential_1dperiodic(xy_inner, eigval, eigvect, LL,
                                          locutoffd=None, hicutoffd=None, check=False):
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
        com = le.com_periodicstrip(xy_inner, LL, masses=magevec[ii])
        fits[ii, 0:2] = com
        # print 'com =', com

        # Get minimum distance for each particle, taking care to get minimum across periodic BCs
        dist = le.distancex_periodicstrip(xy_inner, com, LL)

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
                # check that including all in magevec makes no difference
                com = le.com_periodicstrip(xy, LL, magevec[ii], check=True)
                com = le.com_periodicstrip(xy_inner, LL, magevec[ii], check=True)

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


def fit_eigvect_to_exponential_edge(xy_inner, inner_boundary_inner, eigvect,
                                    eigval=None, locutoffd=None, hicutoffd=None, check=False, interp_n=None):
    """Fit the (edge) excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy_inner : NP x 2 float array
        positions of particles in 2D
    inner_boundary_inner : M x 1 int array
        the indices of xy_inner that mark the boundary sites
    eigvect : 2*NP x 2*NP complex array
        eigenvectors of the system
    eigval : 2*NP x 1 complex array
        eigenvalues of the system
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
    dist = le.distance_from_boundary(xy_inner, inner_boundary_inner, interp_n=interp_n)
    print 'mglfnslocz: np.shape(eigvect) = ', np.shape(eigvect)
    print 'mglfnslocz: np.shape(xy_inner) = ', np.shape(xy_inner)
    print 'mglfnslocz: np.shape(dist) = ', np.shape(dist)

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

        print 'mglatfnslocz: np.shape(magevec) = ', np.shape(magevec)
        print 'mglatfnslocz: np.shape(magevec[ii]) = ', np.shape(magevec[ii])
        print 'mglatfnslocz: np.shape(inds) = ', np.shape(inds)
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


def fit_eigvect_edge_boundaries(xy_inner, inner_boundary_inner, eigvect, PVxydict, eigval=None,
                                locutoffd=None, hicutoffd=None,
                                check=False, interp_n=None):
    """Fit the (edge) excitations of a gyro lattice to exponential decay, using
    Delta = lim_x->infty 1/x ln(psi(x)), computing x as the distance of each site to the edge of the network

    Parameters
    ----------
    xy_inner : NP x 2 float array
        positions of particles in 2D
    inner_boundary_inner : tuple of two int arrays
        Each boundary of the periodic strip as an ordered int array that indexes mglat.xy_inner, not mglat.xy
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
    raise RuntimeError('Check that boundaries index xy_inner, not xy')
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
    dist_tup = le.distance_from_boundaries(xy_inner, inner_boundary_inner, PVxydict, interp_n=interp_n)

    # print 'dist_tup = ', dist_tup

    # Fit the eigvals given distances of each particle from edge
    fits = np.zeros((len_halfevec, 6), dtype=float)
    for ii in range(len_halfevec):
        if ii % 100 == 1:
            print 'glfns: eval #', str(ii), '/', len_halfevec

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
                        plt.clf()
                        plt.semilogy(dists, mags, 'b.-')
                        plt.semilogy(dists, A * np.exp(K * dists), 'r-')
                        Astr = '  A = {0:0.5f}'.format(A) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 2]))
                        Kstr = '  K = {0:0.5f}'.format(K) + r'$\pm$' + '{0:0.5f}'.format(np.sqrt(fits[ii, 4]))
                        plt.title(r'$\omega = $' + str(eigval[ii]) + Astr + Kstr)
                        plt.pause(0.1)

            jj += 1

    # Inspect localization parameter K (exponent in exp(K * r)
    if check:
        plt.plot(np.real(halfeval), fits[:, 1])
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