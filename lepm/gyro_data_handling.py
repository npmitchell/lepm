from scipy.optimize import minimize
import numpy as np

'''Data handling functions for gyroscopic systems'''


def phase_minimizes_difference_complex_displacments(eigvect, realxy):
    """Find the phase theta (which is like finding the time for Hamiltonian evolution) such that the complex
    displacement zz is closest to the real displacement xy

    Parameters
    ----------
    eigval
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


def difference_complex_realdisplacement(pp, eigvect, realxy):
    """

    Parameters
    ----------
    pp : list of one float
        the parameter to optimize (the angle)
    eigvect
    realxy

    Returns
    -------
    sum_of_distances :
        sum of distances between real part of eigval * exp(i * pp[0]) and realxy
    """
    diffs = np.real(eigvect * np.exp(1j * pp[0])) - realxy
    xydiff = np.sqrt(diffs[0:len(diffs):2] ** 2 + diffs[1:len(diffs):2] ** 2)
    return np.sum(xydiff)
