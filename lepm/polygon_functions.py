import numpy as np
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt

'''
Functions for slicing polygons into three parts, calculating apothem and circumradii of polygons.
'''

##########################################
# Geometry functions
##########################################


def polygon_circumradius(s, n):
    """Return circumradius of n-sided polygon with sidelength s."""
    return s*0.5/(np.sin(np.pi/n))


def polygon_apothem(s, n):
    """Return apothem of n-sided polygon with sidelength s."""
    return s*np.cos(np.pi/n)/(2.*np.sin(np.pi/n))


def deformed_hexcell_to_hexagonal_sidelengths(delta, NH, NV):
    """
    Parameters
    ----------
    delta : float
        opening angle of top corner of hexagon (oriented with vertices aligned vertically)
    NH :
    NV :

    Returns
    ----------
    a1 : float
        bottom side length of hexagonal polygon
    a2 : float
        vertical/tilted side length of hexagonal polygon.
    """
    eta = 0.25 * (4.*np.pi - 2.*delta)
    a1 = 2. * NH * np.sin(delta*0.5)
    a2 = 2. * NV * np.sin(eta * 0.5)
    return a1, a2


def deformed_hex_polygon(a1, a2):
    """
    Make an array demarcating a squished hexagonal polygon with sides a1 (base, horizontal) and a2 (sides, vertical).
    """
    hh = np.sqrt(a2 ** 2 - (a1 * 0.5) ** 2)
    theta2 = np.arctan2(hh, a1 * 0.5)

    polygon = np.array([[a1, 0.], [a2*np.cos(theta2), a2*np.sin(theta2)],
                        [-a2*np.cos(theta2),  a2*np.sin(theta2)], [-a1, 0.],
                        [-a2*np.cos(theta2), -a2*np.sin(theta2)],
                        [a2*np.cos(theta2), -a2*np.sin(theta2)]])
    # Check
    # print 'polygon_functions.py: plotting...'
    # plt.plot(polygon[:, 0], polygon[:, 1], 'b.-')
    # plt.show()
    return polygon


def deformed_unitcellhex_polygon(delta, phi):
    """
    Make an array demarcating a squished hexagonal polygon with sides a1 (base, horizontal) and a2 (sides, vertical).
    """
    theta = 0.5 * (np.pi - delta)
    n = np.array([[np.cos(theta), np.sin(theta)], [-np.cos(theta), np.sin(theta)], [-np.sin(phi), -np.cos(phi)]])
    A = np.array([0, 0])
    polygon = np.array([A, A-n[2], A-n[2]+n[0], A-n[2]+n[0]-n[1], A-n[1]+n[0], A-n[1]])

    # Check
    # print 'polygon_functions.py: plotting...'
    # plt.plot(polygon[:, 0], polygon[:, 1], 'b.-')
    # plt.show()
    return polygon


def divide_hexagon_by_sidelengths(hexagon, eps=1e-8):
    """
    Input hexagon must be oriented counter-clockwise
    For now assuming starts with vertex on x axis --> make this more general later!
    """
    hex = hexagon
    a1 = hex[0, 0]
    a2 = np.sqrt(hex[1, 0]**2 + hex[1, 1]**2)
    # hh = np.sqrt(a2**2 - (a1*0.5)**2)
    hh = hex[1, 1]
    theta1 = np.arctan2(a1*0.5, hh)*2.
    apoth = np.sqrt(a2**2 * 1.25 - a2**2 * np.cos(theta1))
    nu = np.arccos((a1**2 + apoth**2 - a2**2 * 0.25) / (2 * a1 * apoth))

    if np.isnan(nu):
        print 'correcting nan nu to nu=0'
        nu = 0.

    # print 'lepm.polygon_functions: nu = ', nu
    # print 'lepm.polygon_functions: (2 * a1 * apoth) = ', (2 * a1 * apoth)
    # sys.exit()
    poly1 = np.array([[apoth*np.cos(nu), -apoth*np.sin(nu)], hex[0], hex[1], [0, hh], [0, 0]])
    poly2 = np.array([[-eps, hh], hex[2], [-a1, 0], [-apoth*np.cos(nu), -apoth*np.sin(nu)], [-eps, 0]])
    poly3 = np.array([[-apoth*np.cos(nu), -apoth*np.sin(nu)], hex[4], hex[5],
                      [apoth*np.cos(nu+eps), -apoth*np.sin(nu+eps)], [0., -eps]])

    # check
    # plt.clf()
    # plt.plot(np.hstack((poly1[:, 0], poly1[0, 0])), np.hstack((poly1[:, 1], poly1[0, 1])), 'r-')
    # plt.plot(np.hstack((poly2[:, 0], poly2[0, 0])), np.hstack((poly2[:, 1], poly2[0, 1])), 'g-')
    # plt.plot(np.hstack((poly3[:, 0], poly3[0, 0])), np.hstack((poly3[:, 1], poly3[0, 1])), 'b-')
    # print 'poly1 = ', poly1
    # print 'poly2 = ', poly2
    # plt.show()
    # sys.exit()

    return poly1, poly2, poly3


def slice_polygon_regions(polygon, alpha, beta, gamma, eps=1e-9, check=False):
    """
    Parameters
    ----------
    polygon : #vertices x 2 float array
        the polygon to slice into three regions along angles alpha, beta, gamma
        Must be counterclockwise points starting with points near the x axis
        Must also not double back on itself, so that polar angle coordinates are monotonically increasing near
        crossings with alpha, beta, gamma.
    alpha : float
        the smallest angle for a slice
    beta : float
        the second largest (smallest) angle for slicing
    gamma : float
        the largest angle for slicing into three regions

    Returns
    ----------
    polygon1 : #vertices x 2 float array
        first slice of polygon
    polygon2 : #vertices x 2 float array
        second slice of polygon
    polygon3 : #vertices x 2 float array
        third slice of polygon
    """
    if len(np.where(np.abs(polygon.ravel()) > eps)[0]) == 0:
        polygon1 = np.array([[0., 0.], [0., 0.], [0., 0.]])
        polygon2 = np.array([[0., 0.], [0., 0.], [0., 0.]])
        polygon3 = np.array([[0., 0.], [0., 0.], [0., 0.]])
    else:
        # Take alpha, beta, gamma wrt 2 pi
        alpha %= (2. * np.pi)
        beta %= (2. * np.pi)
        gamma %= (2. * np.pi)

        # Make a long ray in each direction alpha, beta, gamma
        norm2s = np.max(polygon[:, 0]**2 + polygon[:, 1]**2)

        # Order linesegs of polygon by the angles of their endpts
        thetas = np.mod(np.arctan2(polygon[:, 1], polygon[:, 0]), 2. * np.pi)
        reg1a = np.where(thetas > gamma)[0]
        reg1b = np.where(thetas < alpha)[0]
        reg1IND = np.hstack((reg1a, reg1b))
        reg2IND = np.where(np.logical_and(thetas > alpha, thetas < beta))[0]
        reg3IND = np.where(np.logical_and(thetas > beta, thetas < gamma))[0]

        # First do alpha intersection
        # Get intersection points for divisions between lattice regions
        # Modify the thetas of reg1 points to be wrt gamma
        theta1 = np.hstack((thetas[reg1a] - gamma, thetas[reg1b] + np.pi*2 - gamma))
        a1 = polygon[reg1IND[np.argmax(theta1)], :]
        a2 = polygon[reg2IND[np.argmin(thetas[reg2IND])], :]
        b1 = np.array([0.0, 0.0])
        b2 = norm2s * np.array([np.cos(alpha), np.sin(alpha)])
        intr1 = linsegs.intersection_lines(a1, a2, b1, b2)
        # Formerly used linesegs
        # intr1 = linsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)

        # Checking this
        # print 'polygon_functions.py: intr1 = ', intr1
        # plt.plot(polygon[:, 0], polygon[:, 1], 'k--')
        # plt.plot([a1[0], a2[0]], [a1[1], a2[1]], 'b.-')
        # plt.plot([b1[0], b2[0]], [b1[1], b2[1]], 'r.-')
        # plt.show()
        # import sys
        # sys.exit()

        if check:
            print 'intr1 = ', intr1
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr1[0], intr1[1], 'rx')
            plt.show()

        a1 = polygon[reg2IND[np.argmax(thetas[reg2IND])], :]
        a2 = polygon[reg3IND[np.argmin(thetas[reg3IND])], :]
        b2 = norm2s*np.array([np.cos(beta), np.sin(beta)])
        # intr2 = linsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)
        intr2 = linsegs.intersection_lines(a1, a2, b1, b2)

        if check:
            print 'intr2 = ', intr2
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr2[0], intr2[1], 'rx')
            plt.show()

        a1 = polygon[reg3IND[np.argmax(thetas[reg3IND])], :]
        a2 = polygon[reg1IND[np.argmin(theta1)], :]
        b2 = norm2s*np.array([np.cos(gamma), np.sin(gamma)])
        # intr3 = linsegs.intersection_linesegs(a1, a2, b1, b2, thres=1e-6)
        intr3 = linsegs.intersection_lines(a1, a2, b1, b2)

        if check:
            print 'thetas[reg1IND] = ', thetas[reg1IND]
            print 'np.mod(thetas[reg1IND],gamma) = ', np.mod(thetas[reg1IND], gamma)
            print 'intr3 = ', intr3
            plt.plot(polygon[reg3IND, 0], polygon[reg3IND, 1], 'b.')
            plt.plot(polygon[reg2IND, 0], polygon[reg2IND, 1], 'c.')
            plt.plot(polygon[reg1IND, 0], polygon[reg1IND, 1], 'k.')
            plt.plot(np.array([a1[0]]), np.array([a1[1]]), 'go')
            plt.plot(np.array([a2[0]]), np.array([a2[1]]), 'ro')
            plt.plot(intr3[0], intr3[1], 'rx')
            plt.show()

        normalp = np.sqrt(intr1[0]**2 + intr1[1]**2)
        normbet = np.sqrt(intr2[0]**2 + intr2[1]**2)
        normgam = np.sqrt(intr3[0]**2 + intr3[1]**2)

        intgamma_reg1 = normgam * np.array([np.cos(gamma+eps), np.sin(gamma+eps)])
        intalpha_reg1 = normalp * np.array([np.cos(alpha-eps), np.sin(alpha-eps)])
        intalpha_reg2 = normalp * np.array([np.cos(alpha+eps), np.sin(alpha+eps)])
        intbeta_reg2 = normbet * np.array([np.cos(beta-eps), np.sin(beta-eps)])
        intbeta_reg3 = normbet * np.array([np.cos(beta+eps), np.sin(beta+eps)])
        intgamma_reg3 = normgam * np.array([np.cos(gamma-eps), np.sin(gamma-eps)])

        polygon1 = np.vstack((np.array([0.0,0.0]), intgamma_reg1, polygon[reg1a], polygon[reg1b], intalpha_reg1))
        polygon2 = np.vstack((np.array([0.0,0.0]), intalpha_reg2, polygon[reg2IND], intbeta_reg2))
        polygon3 = np.vstack((np.array([0.0,0.0]), intbeta_reg3 , polygon[reg3IND], intgamma_reg3))

    return polygon1, polygon2, polygon3
