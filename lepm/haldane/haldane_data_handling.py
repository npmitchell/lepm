from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import lepm.haldane.haldane_lattice_functions as hlf

'''Data handling functions for Haldane model systems'''


def phase_minimizes_difference_complex_displacments(eigvect, realxy):
    """Find the phase theta (which is like finding the time for Hamiltonian evolution) such that the complex
    displacement zz is closest to the real displacement xy

    Parameters
    ----------
    eigvect :
        current eigenvector
    realxy

    Returns
    -------
    theta : float
        The angle that minimizes the difference betwen zz and the real displacements xy
    """
    p0 = [0.]
    theta = minimize(difference_complex_realdisplacement, p0, args=(eigvect, realxy),
                     method='L-BFGS-B', options={'disp': False})
    print 'haldane_data_handling: theta = ', theta
    return theta.x


def difference_complex_realdisplacement(pp, eigvect, realxy, basis='XY'):
    """

    Parameters
    ----------
    pp : list of one float
        the parameter to optimize (the angle)
    eigvect : 2N x 1 complex array
        single eigenvector to compare with previous excitation, realxy
    realxy : len(eigvect) * 1 float array?
        real components of previous eigenvector (two numbers for each site)

    Returns
    -------
    sum_of_distances :
        sum of distances between real part of eigval * exp(i * pp[0]) and realxy
    """
    diffs = np.real(eigvect * np.exp(1j * pp[0])) - realxy
    if basis == 'XY':
        xydiff = np.sqrt(diffs[0:len(diffs):2] ** 2 + diffs[1:len(diffs):2] ** 2)
    else:
        raise RuntimeError("Have not coded for psi basis yet")
    # if pp[0] > 1e-7:
    #     print 'pp[0] = ', pp[0], ' --> sum of diffs**2 = ', np.sum(xydiff)
    #     sys.exit()
    return np.sum(xydiff)


def phase_fix_nth_site(eigvects, thetafix=0.0, nsite=0, eps=1e-9):
    """Rotate each element of eigvect in order to fix the nth site (determined by kwarg nsite) to have a phase of
    thetafix with respect to the x axis. In typical applications, nsite should be taken to be the site with a large
    amplitude, if not the largest, if one wants to match eigenvectors that are "similar" looking.

    Parameters
    ----------
    eigvects : 2N x 2N complex array or m x 2N complex array (for a subset of the spectrum)
        the eigenvectors for which to set the nsite excitation to zero phase
    nsite : int
        the site whose phase to fix

    Returns
    -------
    thetas : m x 1 float array
        the angles by which to rotate each of the m given eigvects so that nsite has phase of thetafix
    eigvects :
        the rotated eigvects so that the nsite'th site has zero phase in each eigvect
    """
    # For each eigvect, look at nth site and get the phase of that site's displacement in the psi basis
    # Note that psi = psi_L + conj(psi_R), so that psi = psi_L e^{-i omega t} + conj(psi_R) e^{i omega t}.
    halflen = int(0.5 * np.shape(eigvects)[1])
    psi = eigvects[:, nsite]
    phis = np.arctan2(np.imag(psi), np.real(psi))

    # minimize difference between target angle (thetafix) and eigvect * input rotation (ie times e^{i thetas}) for
    # each element of psiv (ie for each eigvect's nth site).
    # theta_rots is the array of angles by which to rotate each eigvect.
    # For M eigvects, theta_rots is M x 1 float array.
    theta_rots = np.zeros_like(phis)
    for (elem, kk) in zip(psi, range(len(phis))):
        x0 = [thetafix - phis[kk]]

        # Note that the minimization bounds are overly wide in order to prevent discontinuity at pi or -pi.
        success = False
        while not success:
            result = minimize_scalar(err_theta_from_rotate_psi, x0, args=(thetafix, elem),
                              bounds=(-np.pi - eps, np.pi + eps), method='bounded')

            # Also explicitly check the result
            # out = psi_basis_to_displacements(np.array([elem * np.exp(1j * result.x)]))
            # output_theta = np.arctan2(out[1], out[0]) % (np.pi * 2.)
            # check = np.abs(output_theta - (thetafix % (np.pi * 2.))) < 1e-2

            if result.success:
                theta = result.x % (np.pi * 2.)
                success = True
            else:
                success = False
                print 'result = ', result
                print 'Optimization unsuccessful.'
                print 'output_theta = ', output_theta
                print 'target theta (thetafix) = ', thetafix
                print 'initial guess: thetafix - phis[kk] = ', thetafix - phis[kk]

                tt = np.linspace(0, 2. * np.pi, 100)
                err = np.zeros_like(tt)
                for (angle, ii) in zip(tt, range(100)):
                    err[ii] = err_theta_from_rotate_psi(angle, thetafix, elem)

                plt.close('all')
                plt.plot(tt, err, '.-')
                plt.show()
                sys.exit()
                x0 = [x0[0] + np.random.rand(1)[0]]

        theta_rots[kk] = theta

        # Check output
        # out = psi_basis_to_displacements(np.array([elem * np.exp(1j * theta)]))
        # print 'output displacements = ', out
        # print 'thetafix = ', thetafix
        # print 'output_thetas = ', np.arctan2(out[1], out[0])

    # Rotate by theta_rots
    # print 'np.shape(eigvects.T) = ', np.shape(eigvects.T)
    # print 'np.shape(theta_rots) = ', np.shape(np.exp(1j * theta_rots))
    eigvects_rotated = (eigvects.T * np.exp(1j * theta_rots)).T

    return theta_rots, eigvects_rotated


def eigvect_difference_minimum(eigvects, reference_displacement, check=False):
    """Find the mode which minimizes the euclidean distance between the eigvects and a supplied excitation realxy.
    This used to be called phase_difference_minimum(), but it doesn't compute phase differences -- it uses Euclidean
    distances.

    Parameters
    ----------
    eigvects : 2N x 2N complex array
        A set of eigenvectors for the eigenmotions of the N sites.
    reference_displacement : 2N x 1 complex or float array
        the excitation of the N sites (in XY basis or psi basis, depending on argument 'basis') -->
        the displacement to match.

    Returns
    -------
    modenum : int
    """
    # Convert eigvects to xy displacements
    xx, yy = psi_to_displacements(eigvects)
    # Convert reference_displacement to x0, y0
    x0, y0 = psi_to_displacements(np.array([reference_displacement]))
    # print 'gdh: x0 = ', x0
    # print 'gdh: xx = ', xx
    # print 'gdh: --> ', ((xx - x0) ** 2)[0]
    # print 'dist[0] = ', np.sum(np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2), axis=-1)[0]
    xydiff = np.sum(np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2), axis=-1)
    # print 'gdh: xydiff = ', xydiff

    ########################################################
    # # CHECK
    # # check that rotations have been performed
    # plt.clf()
    # xy = np.dstack((np.arange(len(x0[0])), np.zeros_like(x0[0])))[0]
    # pxs, pys = x0, y0
    # print 'xy = ', xy
    # nsite = 3
    # dxs, dys = hlf.psi_basis_to_displacements(eigvects)
    # # iterate over each eigvect (dxs, dys)
    # for (dx_evi, dy_evi) in zip(dxs, dys):
    #     # iterate over each particle
    #     for (dxi, dyi, xy0) in zip(dx_evi, dy_evi, xy):
    #         plt.plot([xy0[0], xy0[0] + dxi], [xy0[1], xy0[1] + dyi], 'k-')
    #         plt.plot(xy0[0] + dxi, xy0[1] + dyi, 'ro', markerfacecolor='none')
    #         # if this is the first pass, draw dots at the points as well
    #         if (dx_evi == dxs[0]).all():
    #             plt.plot(xy0[0], xy0[1], 'k.')
    #
    # # Compare to previous ev
    # # pxs, pys = hlf.rotate_eigvect_psi(np.array([previous_ev]), -thetafix)
    # plt.plot(pxs[0] + xy[:, 0], pys[0] + xy[:, 1], 'b^', markerfacecolor='none', markersize=10)
    # # iterate over each particle in the previous eigvect
    # for (dxi, dyi, xy0) in zip(pxs[0], pys[0], xy):
    #     print 'pxs = ', pxs
    #     print 'xy0 = ', xy0
    #     plt.plot([xy0[0], xy0[0] + dxi], [xy0[1], xy0[1] + dyi], 'b--')
    #
    # plt.plot(xy[nsite, 0], xy[nsite, 1], 'kx')
    #
    # # check that modenum is the right mode that actually minimizes differences
    # modenum = np.argmin(xydiff)
    # # iterate over each particle
    # for (dxi, dyi, xy0) in zip(dxs[modenum], dys[modenum], xy):
    #     plt.plot([xy0[0], xy0[0] + dxi], [xy0[1], xy0[1] + dyi], 'g-')
    #     plt.plot(xy0[0] + dxi, xy0[1] + dyi, 'go', markerfacecolor='none')
    #
    # plt.title('Eigenvector selection to minimize differences')
    # plt.xlabel(r'$x$ displacements + offset by site index')
    # plt.ylabel(r'$y$ displacements')
    # plt.show()
    ########################################################

    modenum = np.argmin(xydiff)
    if check:
        plt.close('all')
        for (ev, kk) in zip(eigvects, np.arange(len(eigvects))):
            xy = np.real(ev)
            add = np.arange(int(len(diffs) * 0.5))
            plt.plot(xy[0:len(diffs):2] + add, xy[1:len(diffs):2], '^', markersize=12)
            plt.plot(realxy[0:len(diffs):2] + add, realxy[1:len(diffs):2], 'gs', markersize=6)

            plt.plot(xy[0] + add[0], xy[1], 'ro', markersize=3)
            plt.plot(realxy[0] + add[0], realxy[1], 'rs', markersize=3)
            plt.plot(add, 0. * add, 'k.', markersize=1)
            plt.title('diff = ' + str(xydiff[kk]))
            plt.axis('equal')
            plt.show()

        plt.close('all')
        xy = np.real(eigvects[modenum])
        plt.plot(xy[0:len(diffs):2], xy[1:len(diffs):2], 'o')
        plt.plot(realxy[0:len(diffs):2], realxy[1:len(diffs):2], 'rs')
        plt.title('smallest difference = ' + str(xydiff[modenum]))
        plt.show()

    return modenum


def psi_eigvect_to_ellipse_displacements(eigvect, npts=80):
    """

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psiL0, psiL1, ... psiLN, psiR0, psiR1, ... psiRN], where psi_i = psiL_i + np.conj(psiR_i) = dx + i dy

    Returns
    -------
    psi : N x 1 complex array
        the displacements as x + i dy of the sites associated with the psi basis imput
    """
    if len(np.shape(eigvect)) > 1:
        print 'np.shape(eigvect) = ', np.shape(eigvect)
        raise RuntimeError('this function takes only a single eigenvector as its input')

    halflen = int(0.5 * len(eigvect))
    psiL = eigvect[0:halflen]
    psiR = eigvect[halflen:]
    # form ellipse evolution
    tt = np.linspace(0, 2. * np.pi, npts, endpoint=True)
    eiwt = np.exp(1j * tt)
    eniwt = np.exp(-1j * tt)
    psiL = np.outer(psiL, eiwt)
    psiRbar = np.outer(np.conj(psiR), eniwt)
    psi = psiL + psiRbar
    # print 'hlf: psi = ', psi
    # print 'hlf: np.abs(psi) = ', np.abs(psi)**2
    dx, dy = np.real(psi), np.imag(psi)
    return dx, dy


def xy_eigvect_to_ellipse_displacements(eigvect, npts=80):
    """

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [x0, x1, ... xN, y0, y1, ... yN], where psi_i = real(x_i) + i real(y_i)

    Returns
    -------
    psi : N x 1 complex array
        the displacements as x + i dy of the sites associated with the psi basis imput
    """
    if len(np.shape(eigvect)) > 1:
        print 'np.shape(eigvect) = ', np.shape(eigvect)
        raise RuntimeError('this function takes only a single eigenvector as its input')

    halflen = int(0.5 * len(eigvect))
    xx = eigvect[0:halflen]
    yy = eigvect[halflen:]
    # form ellipse evolution
    tt = np.linspace(0, 2. * np.pi, npts, endpoint=True)
    eiwt = np.exp(1j * tt)
    xx, yy = np.outer(xx, eiwt), np.outer(yy, eiwt)
    dx, dy = np.real(xx), np.real(yy)
    return dx, dy


def psi_to_displacements(psiL, psiR):
    """Convert (psiL, psiR) from psi = psi^L e^{i omega t} + bar{psi}^R e^{-i omega t} into (dx, dy)"""
    psi = psiL + np.conj(psiR)
    return np.real(psi), np.imag(psi)


def psi_to_displacements(eigvects):
    """

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psi0, psi1, ... psiN], where psi_i = dx + i dy

    Returns
    -------
    dx, dy : N x 1 float arrays
        the displacements in realspace (in an XY basis) of the sites associated with the psi basis imput
    """
    psi = eigvects
    dx, dy = np.real(psi), np.imag(psi)
    return dx, dy


def psi_basis_to_phases(eigvects):
    """Convert 2d eigvect array into 2d array of phases of each site

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psi0, psi1, ... psiN], where psi_i = dx + i dy

    Returns
    -------
    dx, dy : N x 1 float arrays
        the displacements in realspace (in an XY basis) of the sites associated with the psi basis imput
    """
    psi = eigvects
    dx, dy = np.real(psi), np.imag(psi)
    return np.arctan2(dy, dx)


def rotate_eigvect(eigvects, thetas):
    """Rotate each eigvects[i] by thetas[i], using the rule that psi = psiL * e^{i theta} + conj(psiR) e^{-i theta}.

    Parameters
    ----------
    eigvects : M x 2N complex array
    thetas : M x 1 float array

    Returns
    -------
    psi : M x N complex array
        the rotated eigvector
    """
    eigvects = (psi.T * np.exp(1j * thetas)).T
    return eigvects


def err_theta_from_rotate_psi(phi, theta_target, psiv):
    """Rotate eigvect psiv by the angle phi in its ellipse and return the residual from the target rotation angle,
     theta_target. phi is the guess for how much to rotate each component of the eigvect, and the error is returned

    Parameters
    ----------
    phi : list of one float
        guess for how much to rotate each component of the
    theta_target : float
        the angle by which we want to rotate the eigvect site in its ellipse
    psiv : 2 x 1 complex array
        psi = dx + i dy for a single site, as given from one element of an eigvect array.

    Returns
    -------

    """
    # Checking initial guess
    # phase = psi_basis_to_phases(np.array([psiv]))[0]
    # print 'phase = ', phase
    # print 'phi = ', phi
    eiwt = np.exp(1j * phi)
    psi = psiv * eiwt
    dx, dy = np.real(psi), np.imag(psi)

    # make euclidean error so that there is no jump discontinuity
    txy = np.sqrt(dx ** 2 + dy ** 2) * np.array([np.cos(theta_target), np.sin(theta_target)])

    return (dx - txy[0]) ** 2 + (dy - txy[1]) ** 2


def plaquette_phases(eigvect, site_indices, lvfactors, kvec, latvecs, plotfn=None):
    """Compute the phases of each site in an eigenvector as a plaquette or path is traversed. The path is
    determined by site_indices, which lists the sites visited on the path (as indexed in the unit cell), along
    with the lattice vectors that take a unit cell site to that site. The kvec determines the location in the BZ at
    which the eigvect is evaluated.

    Parameters
    ----------
    glat : GyroLattice instance

    site_indices: length m int list or array
    lvfactors : length m int list or array
    kvec : len(2) float array
        the wavevector at which to plot the plaquette phases
    latvecs : 2 x 2 float array
        the 2d realspace lattice vectors

    Returns
    -------

    """
    phases = []
    for (ind, kk) in zip(site_indices, range(len(site_indices))):
        # phase picked up is product of kvec and lattice vectors
        lvf = np.sum((lvfactors[:, kk] * latvecs.T).T, axis=0)
        phase = np.dot(lvf, kvec)
        expfactor = np.exp(1j * phase)

        # Get the phases of this site
        phases.append(psi_basis_to_phases(eigvect * expfactor)[0][ind])

    # Plot it
    if plotfn is not None:
        fig, ax = leplt.initialize_1panel_centered_fig()
        ax.plot(range(len(phases)), phases, '.-')
        ax.set_ylabel(r'phase, $\phi$')
        ax.xaxis.set_ticks([])
        plt.savefig(plotfn)

    return phases


def plaquette_phase_factors(kpt, lvfactors, latvecs):
    """Compute the phase factors for sites located at specified integer lattice vectors away from current site.

    Parameters
    ----------
    kpt : len(2) float array
        the wavevector at which to plot the plaquette phases
    lvfactors : length m int list or array
    latvecs : 2 x 2 float array
        the 2d realspace lattice vectors

    Returns
    -------
    phase_factors :
    """
    displacements = np.outer(lvfactors[0, :], latvecs[0]) + np.outer(lvfactors[1, :], latvecs[1])
    phases = np.sum(displacements * kpt, axis=1)
    # print 'lvfactors = ', lvfactors
    # print 'latvecs = ', latvecs
    # print 'displacements = ', displacements
    # print 'kpt = ', kpt
    # print 'phases = ', phases
    # sys.exit()
    phase_factors = np.exp(1j * phases)

    return phase_factors


def extend_eigvect_at_kpt(eigvects, site_indices, lvfactors, kpt, latvecs):
    """Extend the eigvect to describe the motion of a supercell, rather than just the unit cell.

    Parameters
    ----------
    eigvects
    site_indices
    kpt

    Returns
    -------
    eigvects_extended
    """
    # get phase factors to multiply
    phase_factors = plaquette_phase_factors(kpt, lvfactors, latvecs)
    nphase_factors = plaquette_phase_factors(-kpt, lvfactors, latvecs)
    # print 'eigvects[modenum][ind] = ', eigvects[modenum][ind]
    # print 'phase_factors = ', phase_factors

    eigvects_extended = np.zeros((np.shape(eigvects)[0], len(site_indices)), dtype=complex)
    for modenum in range(np.shape(eigvects)[0]):
        for (ind, kk) in zip(site_indices, range(len(site_indices))):
            eigvects_extended[modenum, kk] = eigvects[modenum][ind] * phase_factors[kk]

    return eigvects_extended


def inner_product(evect1, evect2, orthonormalize=True):
    """Take the normal inner product between two eigenvectors or eigenvector matrices:
    evect1 . evect2, where the dot product is skew symmetric
    """
    halflen = int(0.5 * np.shape(evect1)[1])
    oeigv1 = orthonormal_eigvect(evect1)
    oeigv2 = orthonormal_eigvect(evect2)
    # Note that we can either negate the right half of oeigv1 or the bottom half of oeigv2. These are identical.
    # Note also that we transpose the matrix on the right
    oedoe = np.dot(np.conjugate(oeigv1), oeigv2.T)
    return oedoe