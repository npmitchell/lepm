import numpy as np

'''
Description
===========
Auxiliary discrete math functions
'''


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
