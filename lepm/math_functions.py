import numpy as np

'''
Description
===========
Auxiliary discrete math functions
'''


def round_sigfigs(x, sigfigs):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or np.issubdtype(sigfigs, np.integer)):
        raise TypeError("round_sigfigs: sigfigs must be an integer.")

    if not np.all(np.isreal( x )):
        raise TypeError("round_sigfigs: all x must be real.")

    if sigfigs <= 0:
        raise ValueError("round_sigfigs: sigfigs must be positive.")

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = np.frexp( absx )

    decimalExponents = __logBase10of2 * binaryExponents
    intParts = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - intParts)

    if type(mantissas) is float or np.issctype(np.dtype(mantissas)):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0

    elif np.issubdtype(mantissas, np.ndarray):
        fixmsk = mantissas < 1.0
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    return xsgn * np.around(mantissas, decimals=sigfigs - 1) * 10.0**intParts


def numerical_derivative_2d(func, h, kvec):
    """Calculates a 2D numerical derivative of a function taking a 1x2 float array as input

    Parameters
    ------------
    func : function
        function to take derivative of
    h : float
        spacing in BZ for derivative calculation.
    kvec : 2x1 float array
        points at kx, ky in the BZ

    Returns
    -------
    [fpx, fpy]:
        derivatives along x and y
    """
    x, y = kvec[0], kvec[1]

    ax = func([x + 2 * h, y])[0]
    bx = func([x + h, y])[0]
    cx = func([x - h, y])[0]
    dx = func([x - 2 * h, y])[0]

    ay = func([x, y + 2 * h])[0]
    by = func([x, y + h])[0]
    cy = func([x, y - h])[0]
    dy = func([x, y - 2 * h])[0]

    fpx = (-ax + 8 * bx - 8 * cx + dx) / (12 * h)
    fpy = (-ay + 8 * by - 8 * cy + dy) / (12 * h)

    return [fpx, fpy]


def tensor_polar2cartesian2D(Mrr, Mrt, Mtr, Mtt, x, y):
    """converts a Polar tensor into a Cartesian one

    Parameters
    ----------
    Mrr, Mtt, Mrt, Mtr : N x 1 arrays
        radial, azimuthal, and shear components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined

    Returns
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        the cartesian components
    """
    A = Mrr;
    B = Mrt;
    C = Mtr;
    D = Mtt;
    theta = np.arctan2(y, x);
    ct = np.cos(theta);
    st = np.sin(theta);

    Mxx = ct * (A * ct - B * st) - st * (C * ct - D * st);
    Mxy = ct * (B * ct + A * st) - st * (D * ct + C * st);
    Myx = st * (A * ct - B * st) + ct * (C * ct - D * st);
    Myy = st * (B * ct + A * st) + ct * (D * ct + C * st);
    return Mxx, Mxy, Myx, Myy


def tensor_cartesian2polar2D(Mxx, Myx, Mxy, Myy, x, y):
    """converts a Cartesian tensor into a Polar one

    Parameters
    ----------
    Mxx,Mxy,Myx,Myy : N x 1 arrays
        cartesian components of the tensor M
    x : N x 1 array
        the x positions of the points on which M is defined
    y : N x 1 array
        the y positions of the points on which M is defined

    Returns
    ----------
    Mrr, Mrt, Mtr, Mtt : N x 1 arrays
        radial, shear, and azimuthal components of the tensor M
    """
    A = Mxx;
    B = Mxy;
    C = Myx;
    D = Myy;
    theta = np.arctan2(y, x);
    ct = np.cos(theta);
    st = np.sin(theta);

    Mrr = A * ct ^ 2 + (B + C) * ct * st + D * st ^ 2;
    Mrt = B * ct ^ 2 + (-A + D) * ct * st - C * st ^ 2;
    Mtr = C * ct ^ 2 + (-A + D) * ct * st - B * st ^ 2;
    Mtt = D * ct ^ 2 - (B + C) * ct * st + A * st ^ 2;
    return Mrr, Mrt, Mtr, Mtt


def tensor_polar2cartesian_tractions(Mrr, Mtt, beta):
    """
    Given a stress tensor with locally diagonal values Mrr, Mtt, pick out tractions along x and y directions,
    where x axis is oriented an angle beta from the radial.

    Parameters
    ----------
    Mrr,Mtt : N x 1 arrays
        polar components of the tensor M
    beta : float
        angle of x axis wrt radial axis (phi =0)

    Returns
    ----------
    px : N x 1 float array
        the traction in the x direction
    py : N x 1 float array
        the traction in the y direction
    """
    py = Mrr * np.sin(beta) ** 2 + Mtt * np.cos(beta) ** 2
    px = (Mtt - Mrr) * np.sin(beta) * np.cos(beta)
    return px, py


def rotate_tensor_cartesian(Mxx, Mxy, Myy, beta):
    """Rotate a symmetric tensor by an angle beta in the xy plane.

    see http://www.creatis.insa-lyon.fr/~dsarrut/bib/Archive/others/phys/www.jwave.vt.edu/crcd/batra/lectures/esmmse5984/node38.html
    or elasticity.tex
    for more notes

    Parameters
    ----------
    Mxx,Mxy,Myy : N x 1 arrays
        cartesian components of the tensor M in xy coord sys
    beta : float
        angle of x' wrt x (counterclockwise rotation)

    Returns
    ----------
    Mxxprime, Mxyprime, Mxyprime : N x 1 float arrays
        The stress in the rotated xy' coordinates
    """
    Mxxprime = Mxx * np.cos(beta) ** 2 + Myy * np.sin(beta) ** 2 + 2. * Mxy * np.sin(beta) * np.cos(beta)
    Mxyprime = Mxy * (np.cos(beta) ** 2 - np.sin(beta) ** 2) + (Myy - Mxx) * np.sin(theta) * np.cos(theta)
    Myyprime = Mxx * np.sin(beta) ** 2 + Myy * np.cos(theta) ** 2 - 2. * Mxy * np.sin(theta) * np.cos(theta)
    return Mxxprime, Mxyprime, Myyprime


def rotate_vectors_2D(XY, theta):
    """Given a list of vectors, rotate them actively counterclockwise in the xy plane by theta.

    Parameters
    ----------
    XY : NP x 2 array
        Each row is a 2D x,y vector to rotate counterclockwise
    theta : float
        Rotation angle

    Returns
    ----------
    XYrot : NP x 2 array
        Each row is the rotated row vector of XY

    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    XYrot = np.dot(R, XY.transpose()).transpose()
    return XYrot


def rotation_matrix_3D(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    import math
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def savitzky_golay(y, window_size, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
        window_size : int
        the length of the window. Must be an odd integer number.
    polyorder : int
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative integer. The default is 0, which means to
        filter the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array x along which the filter is to be applied. Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This determines the type of extension to use for
        the padded signal to which the filter is applied. When mode is 'constant', the padding value is given by cval.
        See the Notes for more details on 'mirror', 'constant', 'wrap', and 'nearest'. When the 'interp' mode is
        selected (the default), no extension is used. Instead, a degree polyorder polynomial is fit to the last
        window_length values of the edges, and this polynomial is used to evaluate the last
        window_length // 2 output values.
    cval : scalar, optional
        Value to fill past the edges of the input if mode is 'constant'. Default is 0.0

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
    """
    from scipy.signal import savgol_filter
    return savitzky_golay(y, window_size, polyorder, deriv=deriv, delta=delta, axis=axis,
                          mode=mode, cval=cval)