from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import lepm.gyro_lattice_functions as glf

'''Data handling functions for gyroscopic systems'''


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
    print 'gyro_data_handling: theta = ', theta
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


def phase_fix_nth_gyro(eigvects, thetafix=0.0, ngyro=0, basis='XY', eps=1e-9):
    """Rotate each element of eigvect in order to fix the nth gyro (determined by kwarg ngyro) to have a phase of
    thetafix with respect to the x axis. In typical applications, ngyro should be taken to be the gyro with a large
    amplitude, if not the largest, if one wants to match eigenvectors that are "similar" looking.

    Parameters
    ----------
    eigvects : 2N x 2N complex array or m x 2N complex array (for a subset of the spectrum)
        the eigenvectors for which to set the ngyro excitation to zero phase
    ngyro : int
        the gyro whose phase to fix
    basis : str ('XY' or 'psi')
        the basis in which the spectrum is computed. If XY, then x and y components are given in alternating fashion
        (X0 Y0 X1 Y1 ... XN YN). If 'psi', then clockwise and counterclockwise are given in block fashion
        (it may actually be vice versa, check this) (cw0 cw1 ... cwN ccw0 ccw1 ccw2... ccwN)

    Returns
    -------
    thetas : m x 1 float array
        the angles by which to rotate each of the m given eigvects so that ngyro has phase of thetafix
    eigvects :
        the rotated eigvects so that the ngyro'th gyro has zero phase in each eigvect
    """
    if basis == 'XY':
        # thetas is a 1d array of floats -- the angles by which to rotate each eigvector
        thetas = -np.arctan2(np.real(eigvects[:, 2 * ngyro + 1]), np.real(eigvects[:, 2 * ngyro])) + thetafix
        eigvects = (eigvects.T * np.exp(1j * thetas)).T
        # print 'gdh: thetas = ', thetas
        # print 'gdh: np.shape(eigvects) = ', np.shape(eigvects)
        raise RuntimeError('I think this should be ammended in the same manner as psi basis calc to account for ellipticity of the eigvects')
    else:
        # For each eigvect, look at nth gyro and get the phase of that gyro's displacement in the psi basis
        # Note that psi = psi_L + conj(psi_R), so that psi = psi_L e^{-i omega t} + conj(psi_R) e^{i omega t}.
        halflen = int(0.5 * np.shape(eigvects)[1])
        psiL = eigvects[:, ngyro]
        psiR = eigvects[:, ngyro + halflen]
        psiv = np.dstack((psiL, psiR))[0]
        phis = psi_basis_to_phases(psiv).ravel()

        # minimize difference between target angle (thetafix) and eigvect * input rotation (ie times e^{i thetas}) for
        # each element of psiv (ie for each eigvect's nth gyro).
        # theta_rots is the array of angles by which to rotate each eigvect.
        # For M eigvects, theta_rots is M x 1 float array.
        theta_rots = np.zeros_like(phis)
        for (row, kk) in zip(psiv, range(len(phis))):
            x0 = [thetafix - phis[kk]]
            # Check initial displacements
            # print 'initial phi= ', phis[kk]
            # print 'thetafix = ', thetafix
            # print 'x0 = ', x0
            # out = psi_basis_to_displacements(np.array([row]))
            # print 'initial displacements = ', out

            # Note that the minimization bounds are overly wide in order to prevent discontinuity at pi or -pi.
            success = False
            while not success:
                result = minimize_scalar(err_theta_from_rotate_psi, x0, args=(thetafix, row),
                                  bounds=(-np.pi - eps, np.pi + eps), method='bounded')

                # Also explicitly check the result
                # out = psi_basis_to_displacements(np.array([row * np.exp(1j * result.x)]))
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
                        err[ii] = err_theta_from_rotate_psi(angle, thetafix, row)

                    plt.close('all')
                    plt.plot(tt, err, '.-')
                    plt.show()
                    sys.exit()
                    x0 = [x0[0] + np.random.rand(1)[0]]

            theta_rots[kk] = theta

            # Check output
            # out = psi_basis_to_displacements(np.array([row * np.exp(1j * theta)]))
            # print 'output displacements = ', out
            # print 'thetafix = ', thetafix
            # print 'output_thetas = ', np.arctan2(out[1], out[0])

        # Rotate by theta_rots
        # print 'np.shape(eigvects.T) = ', np.shape(eigvects.T)
        # print 'np.shape(theta_rots) = ', np.shape(np.exp(1j * theta_rots))
        eigvects_rotated = (eigvects.T * np.exp(1j * theta_rots)).T

    return theta_rots, eigvects_rotated


def eigvect_difference_minimum(eigvects, reference_displacement, basis='XY', check=False):
    """Find the mode which minimizes the euclidean distance between the eigvects and a supplied excitation realxy.
    This used to be called phase_difference_minimum(), but it doesn't compute phase differences -- it uses Euclidean
    distances.

    Parameters
    ----------
    eigvects : 2N x 2N complex array
        A set of eigenvectors for the eigenmotions of the N gyroscopes.
    reference_displacement : 2N x 1 complex or float array
        the excitation of the N gyroscopes (in XY basis or psi basis, depending on argument 'basis') -->
        the displacement to match.
    basis : str ('XY' or 'psi')
        string specifier for which basis the eigenvectors (eigvects) and the excitation (realxy) are given

    Returns
    -------

    """
    import gyro_lattice_functions as glf
    # print 'gdh: (eigvects) = ', np.real(eigvects[0])
    # print 'gdh: realxy = ', realxy
    # print 'gdh: diffs = ', diffs[0]
    # print 'gdh: diffs[0::2]', diffs[:, 0:len(diffs):2]
    # print 'gdh: diffs[0::2]', diffs[:, 1:len(diffs):2]
    if basis == 'XY':
        diffs = np.real(eigvects) - np.real(reference_displacement)
        xydiff = np.sum(np.sqrt(diffs[:, 0:len(diffs):2] ** 2 + diffs[:, 1:len(diffs):2] ** 2), axis=-1)
    else:
        # Convert eigvects to xy displacements
        xx, yy = psi_basis_to_displacements(eigvects)
        # Convert reference_displacement to x0, y0
        x0, y0 = psi_basis_to_displacements(np.array([reference_displacement]))
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
        # ngyro = 3
        # dxs, dys = glf.psi_basis_to_displacements(eigvects)
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
        # # pxs, pys = glf.rotate_eigvect_psi(np.array([previous_ev]), -thetafix)
        # plt.plot(pxs[0] + xy[:, 0], pys[0] + xy[:, 1], 'b^', markerfacecolor='none', markersize=10)
        # # iterate over each particle in the previous eigvect
        # for (dxi, dyi, xy0) in zip(pxs[0], pys[0], xy):
        #     print 'pxs = ', pxs
        #     print 'xy0 = ', xy0
        #     plt.plot([xy0[0], xy0[0] + dxi], [xy0[1], xy0[1] + dyi], 'b--')
        #
        # plt.plot(xy[ngyro, 0], xy[ngyro, 1], 'kx')
        #
        # # check that modenum is the right mode that actually minimizes differences
        # modenum = np.argmin(xydiff)
        # # iterate over each particle
        # for (dxi, dyi, xy0) in zip(dxs[modenum], dys[modenum], xy):
        #     plt.plot([xy0[0], xy0[0] + dxi], [xy0[1], xy0[1] + dyi], 'g-')
        #     plt.plot(xy0[0] + dxi, xy0[1] + dyi, 'go', markerfacecolor='none')
        #
        # plt.title('Eigenvector selection to minimize differences')
        # plt.xlabel(r'$x$ displacements + offset by gyro index')
        # plt.ylabel(r'$y$ displacements')
        # plt.show()
        ########################################################

        # raise RuntimeError('Have not coded for the psi basis yet')

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
        the displacements as x + i dy of the gyroscopes associated with the psi basis imput
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
    # print 'glf: psi = ', psi
    # print 'glf: np.abs(psi) = ', np.abs(psi)**2
    dx, dy = np.real(psi), np.imag(psi)
    return dx, dy


def psi_eigvect_to_ellipse_displacements_timedomain(eigvect, eigval, timepoints):
    """Psi basis eigenvector conversion to realspace displacements for given timepoints

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psiL0, psiL1, ... psiLN, psiR0, psiR1, ... psiRN], where psi_i = psiL_i + np.conj(psiR_i) = dx + i dy

    Returns
    -------
    psi : N x 1 complex array
        the displacements as x + i dy of the gyroscopes associated with the psi basis imput
    """
    if len(np.shape(eigvect)) > 1:
        print 'np.shape(eigvect) = ', np.shape(eigvect)
        raise RuntimeError('this function takes only a single eigenvector as its input')

    halflen = int(0.5 * len(eigvect))
    psiL = eigvect[0:halflen]
    psiR = eigvect[halflen:]
    # form ellipse evolution
    eiwt = np.exp(timepoints * eigval)
    eniwt = np.exp(-timepoints * eigval)
    psiL = np.outer(psiL, eiwt)
    psiRbar = np.outer(np.conj(psiR), eniwt)
    psi = psiL + psiRbar
    # print 'glf: psi = ', psi
    # print 'glf: np.abs(psi) = ', np.abs(psi)**2
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
        the displacements as x + i dy of the gyroscopes associated with the psi basis imput
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


def psi_basis_to_displacements(eigvects):
    """

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psiL0, psiL1, ... psiLN, psiR0, psiR1, ... psiRN], where psi_i = psiL_i + np.conj(psiR_i) = dx + i dy

    Returns
    -------
    dx, dy : N x 1 float arrays
        the displacements in realspace (in an XY basis) of the gyroscopes associated with the psi basis imput
    """
    halflen = int(0.5 * np.shape(eigvects)[1])
    psiL = eigvects[:, 0:halflen]
    psiR = eigvects[:, halflen:]
    psi = psiL + np.conj(psiR)
    dx, dy = np.real(psi), np.imag(psi)
    return dx, dy


def psi_basis_to_phases(eigvects):
    """Convert 2d eigvect array into 2d array of phases of each gyroscope

    Parameters
    ----------
    eigvects : M x 2N complex array
        one or more eigenvectors (M eigenvectors) in the psi basis: in the form
        [psiL0, psiL1, ... psiLN, psiR0, psiR1, ... psiRN], where psi_i = psiL_i + np.conj(psiR_i) = dx + i dy

    Returns
    -------
    dx, dy : N x 1 float arrays
        the displacements in realspace (in an XY basis) of the gyroscopes associated with the psi basis imput
    """
    halflen = int(0.5 * np.shape(eigvects)[1])
    psiL = eigvects[:, 0:halflen]
    psiR = eigvects[:, halflen:]
    # print 'psi_basis_to_phases(): psiL = ', psiL
    psi = psiL + np.conj(psiR)
    dx, dy = np.real(psi), np.imag(psi)
    return np.arctan2(dy, dx)


def rotate_eigvect_psi(eigvects, thetas):
    """Rotate each eigvects[i] by thetas[i], using the rule that psi = psiL * e^{i theta} + conj(psiR) e^{-i theta}.

    Parameters
    ----------
    eigvects : M x 2N complex array
    thetas : M x 1 float array

    Returns
    -------
    psi : M x N complex array
    """
    halflen = int(0.5 * np.shape(eigvects)[1])
    psiL = eigvects[:, 0:halflen]
    psiR = eigvects[:, halflen:]
    # psiv_out = np.zeros_like(eigvects, dtype=complex)
    # psiv_out[:, 0:halflen] = (psiL.T * np.exp(1j * thetas)).T
    # psiv_out[:, halflen:] = (psiR.T * np.exp(-1j * thetas)).T
    psi = (psiL.T * np.exp(1j * thetas)).T + (np.conj(psiR).T * np.exp(1j * thetas)).T
    eigvects = (eigvects.T * np.exp(1j * thetas)).T
    return psi, eigvects


def eigvect_psi2xy(eigvect):
    """Convert displacement in the psi basis (psiL, psiR) to xy basis (dx, dy)

    Parameters
    ----------
    eigvect : M x 2N complex array
        the eigenvectors in the psi basis

    Returns
    -------
    xyev
    """
    halflen = int(0.5 * np.shape(eigvect)[1])
    psiL = eigvect[:, 0:halflen]
    psiR = eigvect[:, halflen:]
    xx = psiL + psiR
    yy = 1j * (psiR - psiL)
    # Form the output eigenvectors
    xyev = np.zeros_like(eigvect)
    xyev[:, 0:halflen] = xx
    xyev[:, halflen:] = yy
    return xyev


def eigvect_xy2psi(eigvect):
    """

    Parameters
    ----------
    eigvect

    Returns
    -------

    """
    halflen = int(0.5 * np.shape(eigvect)[1])
    xx = eigvect[:, 0:halflen]
    yy = eigvect[:, halflen:]
    psiL = 0.5 * (xx + 1j * yy)
    psiR = 0.5 * (xx - 1j * yy)
    # Form the output eigenvectors
    psiev = np.zeros_like(eigvect)
    psiev[:, 0:halflen] = psiL
    psiev[:, halflen:] = psiR
    return psiev


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
        psiL and psiR for a single gyroscope, as given from an eigvect array. psiv = np.array([psiL, psiR]), where psiL
        and psiR are complex floats.

    Returns
    -------

    """
    # Checking initial guess
    # phase = psi_basis_to_phases(np.array([psiv]))[0]
    # print 'phase = ', phase
    # print 'phi = ', phi
    eiwt = np.exp(1j * phi)
    psiL = psiv[0] * eiwt
    psiRbar = np.conj(psiv[1] * eiwt)
    psi = psiL + psiRbar
    dx, dy = np.real(psi), np.imag(psi)

    # make euclidean error so that there is no jump discontinuity
    txy = np.sqrt(dx ** 2 + dy ** 2) * np.array([np.cos(theta_target), np.sin(theta_target)])

    return (dx - txy[0]) ** 2 + (dy - txy[1]) ** 2


def plaquette_phases(eigvect, gyro_indices, lvfactors, kvec, latvecs, plotfn=None):
    """Compute the phases of each gyroscope in an eigenvector as a plaquette or path is traversed. The path is
    determined by gyro_indices, which lists the gyroscope sites visited on the path (as indexed in the unit cell), along
    with the lattice vectors that take a unit cell site to that site. The kvec determines the location in the BZ at
    which the eigvect is evaluated.

    Parameters
    ----------
    glat : GyroLattice instance

    gyro_indices: length m int list or array
    lvfactors : length m int list or array
    kvec : len(2) float array
        the wavevector at which to plot the plaquette phases
    latvecs : 2 x 2 float array
        the 2d realspace lattice vectors

    Returns
    -------

    """
    phases = []
    for (ind, kk) in zip(gyro_indices, range(len(gyro_indices))):
        # phase picked up is product of kvec and lattice vectors
        lvf = np.sum((lvfactors[:, kk] * latvecs.T).T, axis=0)
        phase = np.dot(lvf, kvec)
        expfactor = np.exp(1j * phase)

        # Get the phases of this gyro
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


def extend_eigvect_at_kpt(eigvects, gyro_indices, lvfactors, kpt, latvecs):
    """Extend the eigvect to describe the motion of a supercell, rather than just the unit cell.

    Parameters
    ----------
    eigvects
    gyro_indices
    kpt

    Returns
    -------
    eigvects_extended
    """
    halflen = int(0.5 * np.shape(eigvects)[1])

    # get phase factors to multiply
    phase_factors = plaquette_phase_factors(kpt, lvfactors, latvecs)
    nphase_factors = plaquette_phase_factors(-kpt, lvfactors, latvecs)
    # print 'eigvects[modenum][ind] = ', eigvects[modenum][ind]
    # print 'phase_factors = ', phase_factors

    eigvects_extended = np.zeros((np.shape(eigvects)[0], 2 * len(gyro_indices)), dtype=complex)
    for modenum in range(np.shape(eigvects)[0]):
        for (ind, kk) in zip(gyro_indices, range(len(gyro_indices))):
            eigvects_extended[modenum, kk] = eigvects[modenum][ind] * phase_factors[kk]
            eigvects_extended[modenum, kk + len(gyro_indices)] = eigvects[modenum][ind + halflen] * phase_factors[kk]

    # print 'eigvects = ', eigvects[2]
    # print 'eigvects_extended = ', eigvects_extended[2]
    # sys.exit()
    return eigvects_extended


def orthonormal_eigvect(eigvect, basis='psi', negative=False):
    """Normalize some eigenvectors of a GyroLattice instance by the rule
    <a | b > = sum_i [(a^*_L)_i (b_L)_i - (a^*_R)_i (b_R)_i]

    Parameters
    ----------
    eigvect
    negative : bool

    Returns
    -------

    """
    halflen = int(0.5 * np.shape(eigvect)[1])
    # I think this should work in any basis...
    if basis == 'psi':
        # form denominator: sqrt(psi_L^2 - psi_R^2)
        # Note that the sign doesn't matter here
        denom = np.sqrt(np.abs(np.sum(np.abs(eigvect[:, 0:halflen]) ** 2 - np.abs(eigvect[:, halflen:]) ** 2, axis=1)))

        # flip sign of eigvects whose norm is negative if negative==True
        if negative:
            negative = np.sum(np.abs(eigvect[:, 0:halflen]) ** 2 - np.abs(eigvect[:, halflen:]) ** 2, axis=1) < 0
            denom[negative] = - denom[negative]

        # print 'gdh: denom = ', denom
        oeig = (eigvect.T / denom).T
        # print 'gdh: eigvect = ', eigvect
        # print 'gdh: oeig = ', oeig
        # sys.exit()
    else:
        raise RuntimeError('Have not coded for any basis other than psi, but I think norm should be the same...')
    return oeig


def skew_inner_product(eigvect, orthonormalize=True):
    """Take the symplectic inner product between an eigenvector or eigenvector matrix and itself"""
    halflen = int(0.5 * np.shape(eigvect)[1])
    oeigv = orthonormal_eigvect(eigvect)
    co = np.conjugate(np.hstack((oeigv[:, 0:halflen], -oeigv[:, halflen:])))
    oedoe = np.dot(co, oeigv.T)
    return oedoe


def skew_inner_product2(evect1, evect2, orthonormalize=True):
    """Take the symplectic inner product between two eigenvectors or eigenvector matrices:
    evect1 . evect2, where the dot product is skew symmetric
    Assume that evect1 and evect2 are given such that each row is a state.
    """
    halflen = int(0.5 * np.shape(evect1)[1])
    oeigv1 = orthonormal_eigvect(evect1)
    oeigv2 = orthonormal_eigvect(evect2)
    # Note that we can either negate the right half of oeigv1 or the bottom half of oeigv2. These are identical.
    # Note also that we transpose the matrix on the right so that each state is a column.
    co = np.conjugate(np.hstack((oeigv1[:, 0:halflen], -oeigv1[:, halflen:])))
    oedoe = np.dot(co, oeigv2.T)
    return oedoe

def skew_inner_product2_noconj(evect1, evect2, orthonormalize=True):
    """Take the symplectic inner product between two eigenvectors or eigenvector matrices:
    evect1 . evect2, where the dot product is skew symmetric
    Assume that evect1 and evect2 are given such that each row is a state.
    """
    halflen = int(0.5 * np.shape(evect1)[1])
    oeigv1 = orthonormal_eigvect(evect1)
    oeigv2 = orthonormal_eigvect(evect2)
    # Note that we can either negate the right half of oeigv1 or the bottom half of oeigv2. These are identical.
    # Note also that we transpose the matrix on the right so that each state is a column.
    co = np.hstack((oeigv1[:, 0:halflen], -oeigv1[:, halflen:]))
    oedoe = np.dot(co, oeigv2.T)
    return oedoe


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