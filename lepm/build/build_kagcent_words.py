import numpy as np
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import lepm.stringformat as sf
import lepm.build.build_lattice_functions as blf
import lepm.build.build_hucentroid as bhucent
import lepm.build.build_iscentroid as biscent
import lepm.build.build_randorg_gamma as bgammacent
import lepm.data_handling as dh
import copy


"""Functions for making networks that are half kagomized, with a boundary that spells out a word."""


def build_kagcentcurvys(lp):
    """Build network with an 'S' spelled in kagomization. Periodic boundaries are not supported.
    example usage:
    python ./build/make_lattice.py -LT kaghi_isocent_curvys -NH 40 -NV 40 -thres 5.5 -skip_polygons -skip_gyroDOS \
           -aratio 4.0

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 5.0

    check = lp['check']
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['check'] = False
    if 'hucent' in lp['LatticeTop']:
        lp_tmp['NH'] += 10
        lp_tmp['NV'] += 10
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp_tmp)
        lp = lp_tmp
        lp['NH'] -= 10
        lp['NV'] -= 10
    elif 'isocent' in lp['LatticeTop']:
        lp_tmp['NH'] += 10
        lp_tmp['NV'] += 10
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp_tmp)
        lp = lp_tmp
        lp['NH'] -= 10
        lp['NV'] -= 10
    elif 'randorg' in lp['LatticeTop']:
        xy, NL, KL, BL, PVxydict, PVx, PVy, LL, LVUC, LV, UC, BBox, lattice_exten = \
            bgammacent.build_randorg_gamma_spread(lp_tmp)
        lp = lp_tmp

    lp['check'] = check

    # Draw curve 'S'
    # use the fact that we've scaled the centroidal network by medbL to infer what it was previously
    medbL = (lp['NV'] + 10.) / (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    # yspan = np.max(xy[:, 1]) - np.min(xy[:, 1]) - float(10.)/float(medbL)
    xspan = np.max(xy[:, 0]) - np.min(xy[:, 0]) - float(10.) / float(medbL)
    rescale = 1.4
    xstart = 0.0
    x0 = 0.1 * rescale
    tt = lp['thres'] / xspan
    wletter = 0.2 * rescale
    hletter = 0.4 * rescale
    space = 0.1 * rescale
    # Tune xend by the number of letters, here just one.
    xend = wletter + 2. * x0
    y0 = 0.5 - hletter * 0.5
    # sletter = draw_letter_s(y0, y0 + hletter, x0, x0 + wletter, slant=lp['alph'])
    sletter = draw_letter_curvys(y0, y0 + hletter, x0, x0 + wletter, aratio=lp['aratio'])

    if lp['check']:
        print 'sletter = ', sletter
        plt.plot(sletter[:, 0], sletter[:, 1], 'b.-')
        plt.title('Letter S')
        plt.show()

    postss = np.array([xend, y0 + hletter])
    sword = np.vstack((sletter, postss))

    if lp['check']:
        plt.plot(sword[:, 0], sword[:, 1], 'b.-')
        ax = plt.gca()
        ax.axis('equal')
        plt.show()

    sword[:, 0] -= (xend - xstart) * 0.5
    sword[:, 1] -= 0.5
    sword *= xspan

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(sword[:, 0], sword[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([rightx, rightx, leftx, leftx])
        boty = np.min(xy[:, 1]) - 10.
        closecy = np.array([sword[-1, 1], boty, boty, sword[0, 1]])
        curvx = np.hstack((np.array([leftx]), sword[:, 0], closecx))
        curvy = np.hstack((np.array([sword[0, 1]]), sword[:, 1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        lp['word_path'] = cpoly
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([leftx, leftx, rightx, rightx])
        topy = np.max(xy[:, 1]) + 10.
        closecy = np.array([sword[0, 1], topy, topy, sword[-1, 1]])
        curvx = np.hstack((np.array([rightx]), sword[:, 0][::-1], closecx))
        curvy = np.hstack((np.array([sword[-1, 1]]), sword[:, 1][::-1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        lp['word_path'] = cpoly
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    xy, NL, KL, BL = blf.mask_with_polygon(lp['shape'], lp['NH']/medbL, lp['NV']/medbL, xy, BL, eps=0.00,
                                           check=lp['check'])

    # prepare meshfn as lattice_exten
    aratiostr = '_aratio' + sf.float2pstr(lp['aratio'], ndigits=3)
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + \
                    sf.float2pstr(lp['thres'], ndigits=1) + aratiostr
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_uofc(lp):
    """Build network with 'UC' spelled in kagomization.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 2.0

    if 'hucent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp)
    elif 'isocent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp)

    # Draw curve UC
    ytop = 0.75
    x0u = 0.325 - 0.015
    y0u = 0.4
    yy0 = np.array([ytop, ytop])
    xx0 = np.array([0., 0.15])
    xx1 = np.linspace(.15, .47, 100)

    # ---. (0.15, 0.75)
    #    |
    #    |
    #    | (0.15, 0.4)
    #     \     /
    #       ---
    RR = x0u - 0.15
    yy1 = y0u - np.sqrt(RR**2 - (xx1 - x0u)**2)
    xx2 = np.array([0.47])
    yy2 = np.array([0.75])
    xxu = np.hstack((xx0, xx1, xx2))
    yyu = np.hstack((yy0, yy1, yy2))

    # Now form the letter C as two half circles separated in y
    # # Use ellipse: (y/a)**2 + (x/b)**2 = 1
    # lowy = 0.4 - RR
    # x0c = 0.5 + 0.15
    # y0c = 0.5
    # xx2 = np.linspace(0.5, 0.85, 200)
    # a = 0.85 - 0.55
    # b = .15
    # yy2 = -a * np.sqrt(1 - ((xx2-x0c)/b)**2) + y0c

    # Use circle for C
    RR = 0.325 - 0.15
    x0c = 0.53 + RR
    xx3 = np.linspace(.5301, .85, 100)
    yy3 = ytop - RR + np.sqrt(RR ** 2 - (xx3 - x0c) ** 2)
    xx4 = np.linspace(.5301, .875, 100)
    yy4 = y0u - np.sqrt(RR ** 2 - (xx3 - x0c) ** 2)
    xx5 = np.array([1.0])
    yy5 = np.array([yy4[-1]])
    xxc = np.hstack((xx3[::-1], xx4, xx5))
    yyc = np.hstack((yy3[::-1], yy4, yy5))

    # Check the UC curves
    # print 'yy2 = ', yy2
    # plt.plot(xxu, yyu, 'b.-')
    # print 'yyc = ', yyc
    # plt.plot(xxc, yyc, 'r.-')
    # plt.show()

    scale = np.max(xy[:, 0]) - np.min(xy[:, 0])
    xyu = np.dstack((xxu - 0.5, yyu - 0.5))[0] * scale
    xyc = np.dstack((xxc - 0.5, yyc - 0.5))[0] * scale

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    for ii in range(len(xyu) - 1):
        endpt1u = xyu[ii]
        endpt2u = xyu[ii + 1]
        close = le.pts_are_near_lineseg(xy, endpt1u, endpt2u, lp['thres'])
        todec[close] = True

    for ii in range(len(xyc) - 1):
        endpt1c = xyc[ii]
        endpt2c = xyc[ii + 1]
        close = le.pts_are_near_lineseg(xy, endpt1c, endpt2c, lp['thres'])
        todec[close] = True

    # Check linseg connection between U and C
    endpt1 = xyu[-1, :]
    endpt2 = np.array([(xx3[int(len(xx3) * 0.5)] - 0.5) * scale, xyu[-1, 1]])
    close = le.pts_are_near_lineseg(xy, endpt1, endpt2, lp['thres'])
    todec[close] = True

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(xyu[:, 0], xyu[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.plot([endpt1[0], endpt2[0]], [endpt1[1], endpt2[1]], 'c-')
        plt.plot(xyc[:, 0], xyc[:, 1], 'g-')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = xx0[0] - 1.
        closecx = np.array([xx5 + 1., leftx])
        closecy = np.array([-1.5, -1.5])
        ind3 = int(len(xx3) * 0.5)
        curvx = np.hstack((np.array([leftx]), xxu, xx3[ind3::-1], xx4, xx5 + 1., closecx))
        curvy = np.hstack((np.array([yy0[0]]), yyu, yy3[ind3::-1], yy4, yy5, closecy))
        cpoly = np.dstack(((curvx - 0.5) * scale, (curvy - 0.5) * scale))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = xx0[0] - 1.
        closecx = np.array([xx5 + 1., leftx])
        closecy = np.array([1.5, 1.5])
        ind3 = int(len(xx3) * 0.5)
        curvx = np.hstack((np.array([leftx]), xxu, xx3[ind3::] + 1e-6, xx3[::-1] - 1e-6, xx4, xx5 + 1., closecx))
        curvy = np.hstack((np.array([yy0[0]]), yyu, yy3[ind3::], yy3[::-1], yy4, yy5, closecy))
        cpoly = np.dstack(((curvx - 0.5) * scale, (curvy - 0.5) * scale))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)

        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[inpoly, 0], xy[inpoly, 1], 'bo')
            plt.title('The points that are inside the polygon above the letters (letters will be excluded next)')
            plt.show()

        letters = np.where(todec)[0]
        inpoly = np.setdiff1d(inpoly, letters)
        todec = np.zeros_like(todec, dtype=bool)
        todec[inpoly] = True

        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])
    NL, KL = le.BL2NLandKL(BL)
    if (BL < 0).any():
        # todo: make this work
        print 'Creating periodic boundary vector dictionary for decorated network...'
        PV = np.array([])
        PVxydict = le.BL2PVxydict(BL, xy, PV)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL)
        raise RuntimeError('This is not set up yet...')

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + sf.float2pstr(lp['thres'], ndigits=1)
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def draw_letter_a(y0, y1, x0, x1, tt):
    """Draw a curve spelling the letter (capital) A

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : thickness of sides of letters when turn around
    """
    slope = float(y1 - y0) / (float(x1 - x0) * 0.5)
    print 'slope = ', slope
    return np.array([[x0, y0],
                    [np.mean([x0, x1]), y1],
                    [x0 + (x1 - x0) * 0.75, y1 - slope * ((x1 - x0) * 0.25) + tt*0.5],
                    [x0 + 0.5 * (x1 - x0), y1 - slope * ((x1 - x0) * 0.25) + tt*0.5],
                    [x0 + 0.5 * (x1 - x0), y1 - slope * ((x1 - x0) * 0.25) - tt*0.5],
                    [x0 + (x1 - x0) * 0.75, y1 - slope * ((x1 - x0) * 0.25) - tt*0.5],
                    [x1, y0]])


def draw_letter_c(y0, y1, x0, x1, rr):
    """Draw a curve spelling the letter (capital) C

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    rr : radius
    """
    x0c = (x0 + x1) * 0.5
    xx3 = np.linspace(x0, x1, 100)
    yy3 = y1 - rr + np.sqrt(rr ** 2 - (xx3 - x0c) ** 2)
    xx4 = np.linspace(x0, x1, 100)
    yy4 = y0 + rr - np.sqrt(rr** 2 - (xx4 - x0c) ** 2)
    xxc = np.hstack((xx3[::-1], xx4))
    yyc = np.hstack((yy3[::-1], yy4))
    letter = np.dstack((xxc, yyc))[0]
    return letter


def draw_letter_e(y0, y1, x0, x1, tt):
    """Draw a curve spelling the letter (capital) E

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : float
        Spacing between curve segments that would overlie each other
    """
    x0e = (x0 + x1) * 0.5
    xx = [x0, x1, x1, x0,
           x0, x0 + (x1 - x0) *0.6,
           x0 + (x1 - x0) *0.6, x0,
           x0, x1]
    yy = [y1, y1, y1-tt, y1-tt,
           0.5 * (y0 + y1) + tt * 0.5, 0.5 * (y0 + y1) + tt * 0.5,
           0.5 * (y0 + y1) - tt * 0.5, 0.5 * (y0 + y1) - tt * 0.5,
           y0, y0]
    letter = np.dstack((np.array(xx), np.array(yy)))[0]
    return letter


def draw_letter_f(y0, y1, x0, x1, tt, midline_width, midline_height):
    """Draw a curve spelling the letter (capital) F from bottom up

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : float
        Spacing between curve segments that would overlie each other
    midline_width : float
        Fraction of letter width to make the middle horizontal line in the letter F
    midline_height : float
        Fraction of letter height to place the middle horizontal line in the letter F
    """
    mw = midline_width * (x1 - x0)
    mh = midline_height * (y1 - y0)
    x0c = (x0 + x1) * 0.5
    xx = [x0, x0,
          x0 + mw, x0 + mw,
          x0, x0, x1]
    yy = [y0, y0 + mh - tt*0.5,
          y0 + mh - tt * 0.5, y0 + mh + tt*0.5,
          y0 + mh + tt * 0.5, y1, y1]
    letter = np.dstack((np.array(xx), np.array(yy)))[0]
    return letter


def draw_letter_g(y0, y1, x0, x1, rr, tt, jutt):
    """Draw a curve spelling the letter (capital) G

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    rr :
    tt :
    jutt : jut height above bottom of C curve for end of G
    """
    x0c = (x0 + x1) * 0.5
    xx3 = np.linspace(x0, x1, 100)
    yy3 = y1 - rr + np.sqrt(rr ** 2 - (xx3 - x0c) ** 2)
    xx4 = np.linspace(x0, x1, 100)
    yy4 = y0 + rr - np.sqrt(rr ** 2 - (xx4 - x0c) ** 2)
    jutx = np.array([xx4[-1], x0 + 0.7*(x1-x0), x0 + 0.7*(x1-x0), x1 + tt, x1 + tt])
    juty = np.array([yy4[-1] + jutt,  yy4[-1] + jutt, yy4[-1] + jutt + tt, yy4[-1] + jutt + tt, y0])
    xxc = np.hstack((xx3[::-1], xx4, jutx))
    yyc = np.hstack((yy3[::-1], yy4, juty))
    return np.dstack((xxc, yyc))[0]


def draw_letter_h(y0, y1, x0, x1, tt, exittop=True):
    """Draw a curve spelling the letter (capital) H

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : thickness of sides of letters when turn around
    exittop : bool
        End with curve at top rather than bottom
    """
    if exittop:
        letter = np.array([[x0, y0],
                           [x0, y1], [x0 + tt, y1],
                           [x0 + tt, np.mean([y0, y1])],
                           [x1 - tt, np.mean([y0, y1])],
                           [x1 - tt, y1],
                           [x1, y1],
                           [x1, y0]])
    else:
        letter = np.array([[x0, y0],
                           [x0, y1], [x0 + tt, y1],
                           [x0 + tt, np.mean([y0, y1])],
                           [x1 - tt, np.mean([y0, y1])],
                           [x1 - tt, y0],
                           [x1, y0],
                           [x1, y1]])

    return letter


def draw_letter_i(y0, y1, x0, x1, skew):
    """Draw a curve spelling the letter (capital) I

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    skew : how far to bias centerline to right (positive x direction)
    """
    return np.array([[x0, y0],
                    [(x0 + x1) * 0.5 + skew, y0],
                    [(x0 + x1) * 0.5 + skew, y1],
                    [x1, y1]])


def draw_letter_k(y0, y1, x0, x1, tt, from_bottom=True):
    """Draw a curve spelling the letter (capital) K, starting at the bottom, exiting at the bottom

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : distance for parallel lines to avoid each other
    from_bottom : bool
        Whether to start the letter from bottom left or top left
    """
    y0c = 0.5 * (y0 + y1)
    if from_bottom:
        xx1 = [x0, x0]
        yy1 = [y1, y0]
        xx2 = [x0 + tt, x0 + tt,
               x1 - 0.5 * tt, x1 + 0.5 * tt,
               x0 + 2. * tt, x1]
        yy2 = [y0, y0c + tt,
               y1, y1,
               y0c + tt, y0]
    else:
        xx1 = [x0, x0]
        yy1 = [y0, y1]
        xx2 = [x0 + tt, x0 + tt,
               x1 - 0.5 * tt, x1 + 0.5 * tt,
               x0 + 2. * tt, x1]
        yy2 = [y1, y0c + tt,
               y1, y1,
               y0c + tt, y0]
    xx = np.hstack((np.array(xx1), np.array(xx2)))
    yy = np.hstack((np.array(yy1), np.array(yy2)))
    return np.dstack((xx, yy))[0]


def draw_letter_m(y0, y1, x0, x1, midpt_height=0.5):
    """Draw a curve spelling the letter (capital) M

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    midpt_height : float
        fraction of total height to place the valley of the M

    """
    xx = [x0, x0, x0 + (x1-x0) * 0.5, x1, x1]
    yy = [y0, y1, y0 + (y1-y0) * midpt_height, y1, y0]
    return np.dstack((np.array(xx), np.array(yy)))[0]


def draw_letter_n(y0, y1, x0, x1, tt):
    """Draw a curve spelling the letter (capital) N

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    rr : float
        radius of the circular portion of the letter R

    """
    xx = [x0, x0, x1, x1]
    yy = [y0, y1, y0, y1]
    return np.dstack((np.array(xx), np.array(yy)))[0]


def draw_letter_o(y0, y1, x0, x1, rr, tt):
    """Draw a curve spelling the letter (capital) O

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    """
    x0c = (x0 + x1) * 0.5
    xx3 = np.linspace(x0, x1, 100)
    yy3 = y1 - rr + np.sqrt(rr ** 2 - (xx3 - x0c) ** 2)
    xx4 = np.linspace(x0, x1, 100)
    yy4 = y0 + rr - np.sqrt(rr ** 2 - (xx4 - x0c) ** 2)
    midind = int(len(xx4) * 0.5)
    midind = min(midind, np.max(np.where(xx4 < (x0 + x1 - tt) * 0.5)[0]))
    back = np.arange(0, midind)[::-1]
    right_of_tt = np.where(xx4 > (x0 + x1 + tt) * 0.5)[0]
    ending = np.intersect1d(np.arange(midind, len(xx4)), right_of_tt)[::-1]
    xxc = np.hstack((np.array(xx4[midind-1]), xx4[back], xx3, xx4[ending], np.array([xx4[ending[-1]]])))
    yyc = np.hstack((np.array(yy4[midind-1]), yy4[back] + tt, yy3, yy4[ending] + tt, np.array(yy4[ending[-1]])))
    return np.dstack((xxc, yyc))[0]


def draw_letter_r(y0, y1, x0, x1, rr, tt):
    """Draw a curve spelling the letter (capital) R

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    rr : float
        radius of the circular portion of the letter R

    """
    x0c = 0.5 * (x0 + x1)
    xx1 = [x0, x0]
    yy1 = [y0, y1]
    xx2 = np.linspace(x1 - rr, x1, 50)
    yy2 = y1 - rr + np.sqrt(rr ** 2 - (xx2 - x0c) ** 2)  # , np.zeros_like(xx2)))
    xx3 = xx2[::-1]
    yy3 = y1 - rr - np.sqrt(rr ** 2 - (xx3 - x0c) ** 2)
    xx4 = [x0 + tt, x0 + tt, x0 + (x1 - x0) * 0.45, x1]
    yy4 = [y1 - 2 * rr, y1 - 2 * rr - tt, y1 - 2 * rr - tt, y0]
    xx = np.hstack((np.array(xx1), xx2, xx3, np.array(xx4)))
    yy = np.hstack((np.array(yy1), yy2, yy3, np.array(yy4)))
    return np.dstack((xx, yy))[0]


def draw_letter_s(y0, y1, x0, x1, slant=0.0):
    """Draw a curve spelling the letter (capital) S

    Parameters
    ----------
    y0 : float
        bottom y value
    y1 : float
        top y value
    x0 : float
        left x value
    x1 : float
        right x value
    rr : float
        radius of the circular portions of the letter S
           ___
        C
        ___ C
    slant : float (default=0.0)
        overall slope to the S, to make italic

    """
    rr = 0.5 * (x1 - x0)
    # 2*rr = rr [(1 + aratio) + (1 - aratio)]
    # addfrac = 1. - sizeratio_topbottom
    # rrtop = (1 - addfrac) * rr
    # rrbot = rr * (1. + addfrac)
    # define the centers of the letters
    x0c = 0.5 * (x0 + x1)
    y0c = 0.5 * (y0 + y1)
    # left bottom flat line
    xx1 = [x0, x0c]
    yy1 = [y0, y0]
    # Right half circle, on bottom
    xx2 = np.linspace(x1 - rr, x1, 150)
    print 'np.max(rr ** 2 - (xx2 - x0c) ** 2, np.zeros_like(xx2)'
    yy2 = y0c - rr - np.sqrt(np.maximum(rr ** 2 - (xx2 - (x1 - rr)) ** 2, np.zeros_like(xx2)))
    xx3 = xx2[::-1]
    yy3 = y0c - rr + np.sqrt(np.maximum(rr ** 2 - (xx3 - (x1 - rr)) ** 2, np.zeros_like(xx3)))
    # Left half circle, on top
    xx4 = np.linspace(x0 + rr - rr, x0 + rr, 150)
    print 'xx4 = ', xx4
    yy4 = y0c + rr - np.sqrt(np.maximum(rr ** 2 - (xx3 - (x1 - rr)) ** 2, np.zeros_like(xx4)))
    xx5 = xx4
    yy5 = y0c + rr + np.sqrt(np.maximum(rr ** 2 - (xx3 - (x1 - rr)) ** 2, np.zeros_like(xx5)))
    xx4 = xx4[::-1]
    yy4 = yy4[::-1]
    xx6 = np.array([x1])
    yy6 = np.array([y1])

    xx = np.hstack((np.array(xx1), xx2, xx3, xx4, xx5, xx6))
    yy = np.hstack((np.array(yy1), yy2, yy3, yy4, yy5, yy6))

    if abs(slant) > 0.:
        # slope the S forward by slope 'slant' to make more italic
        addx = yy * slant
        xx += addx
        # renormalize x
        xx *= (2 * rr) / (2 * rr + np.max(addx))

    return np.dstack((xx, yy))[0]


def draw_letter_curvys(y0, y1, x0, x1, aratio=1.0):
    """Draw a curve spelling the letter (capital) S

    Parameters
    ----------
    y0 : float
        bottom y value
    y1 : float
        top y value
    x0 : float
        left x value
    x1 : float
        right x value
    rr : float
        radius of the circular portions of the letter S
           ___
        C
        ___ C
    aratio : float (default=1.0)
        Controls difference in height/depth of the peak and trough of the S, so tunes asymmetry.
        aratio == 1.0 is symmetric

    """
    rr = 0.5 * (x1 - x0)
    # 2*rr = rr [(1 + aratio) + (1 - aratio)]
    # addfrac = 1. - sizeratio_topbottom
    # rrtop = (1 - addfrac) * rr
    # rrbot = rr * (1. + addfrac)
    # define the centers of the letters
    x0c = 0.5 * (x0 + x1)
    y0c = 0.5 * (y0 + y1)
    # left bottom flat line
    xx1 = [x0, x0c]
    yy1 = [y0, y0]

    # Curvy s portion
    yy2 = np.linspace(y0, y1, 100)
    # Here is a function controlled by aratio
    ab = aratio
    # let b == 1.0, so a = ab
    # normally, c = (-a**2 + (b-a)**2) / b**2
    cc = -ab ** 2 + (1. - ab) ** 2
    # Normalize the y coordinate
    yy = (yy2 - y0) / (y1 - y0)
    eps = 1e-9
    xx2 = np.sin(2. * np.pi * yy) / (np.sqrt(yy + eps) * np.sqrt(1. + eps - yy))
    xx2 += (cc * yy**2 - (yy - ab) ** 2 + ab**2)
    # rescale to be the right width
    xx2 *= (x1 - x0) / (np.max(xx2) - np.min(xx2))
    xx2 += x0c

    xx3 = np.array([x1])
    yy3 = np.array([y1])

    xx = np.hstack((np.array(xx1), xx2, xx3))
    yy = np.hstack((np.array(yy1), yy2, yy3))

    return np.dstack((xx, yy))[0]


def draw_letter_t(y0, y1, x0, x1, tt):
    """Draw a curve spelling the letter (capital) T from top, exiting on the top

    Parameters
    ----------
    y0 : bottom y value
    y1 : top y value
    x0 : left x value
    x1 : right x value
    tt : float
        Spacing between curve segments that would overlie each other
    """
    x0c = (x0 + x1) * 0.5
    xx = [x0, x0c - tt * 0.5,
          x0c - tt * 0.5, x0c + tt * 0.5,
          x0c + tt * 0.5, x1]
    yy = [y1, y1,
          y0, y0,
          y1, y1]
    letter = np.dstack((np.array(xx), np.array(yy)))[0]
    return letter


def build_kagcentcsmf(lp):
    """Build network with 'CSMF' spelled in kagomization. Periodic boundaries are not supported.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 2.0

    check = lp['check']
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['check'] = False
    lp_tmp['NH'] += 10
    lp_tmp['NV'] += 10
    if 'hucent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp_tmp)
    elif 'isocent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp_tmp)
    lp = lp_tmp
    lp['check'] = check
    lp['NH'] -= 10
    lp['NV'] -= 10

    # Draw curve CSMF
    # use the fact that we've scaled the centroidal network by medbL to infer what it was previously
    medbL = (lp['NV'] + 10.) / (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    yspan = np.max(xy[:, 1]) - np.min(xy[:, 1]) - float(10.)/float(medbL)
    rescale = 1.7
    xstart = 0.0
    x0 = 0.1 * rescale
    rr = 0.11 * rescale
    tt = lp['thres'] / yspan
    wletter = 0.2 * rescale
    hletter = 0.4 * rescale
    space = 0.1 * rescale
    xend = 4. * wletter + 2. * x0 + 3. * space
    y0 = 0.5 - hletter * 0.5

    # Create letter C
    cc = draw_letter_c(y0, y0 + hletter, x0, x0 + wletter, rr)
    precc = copy.deepcopy(cc[0:int(len(cc) * 0.5), :])[::-1]
    precc = np.vstack((np.array([xstart, precc[0, 1]]), precc))
    # Lower the top of the letter by tt
    cc[0:int(len(cc)*0.5), 1] -= tt
    top = np.where(cc[:, 1] > y0 + hletter * 0.5)[0]
    precc[:, 0] = np.mean(cc[top, 0]) + (wletter + tt)/wletter * (precc[:, 0] - np.mean(cc[top, 0]))
    # Double back at the end of the C
    # first just copy the bottom part of the C
    bot = np.where(cc[:, 1] < y0 + hletter * 0.5)[0]
    postcc = cc[np.where(cc[:, 1] < (y0 + hletter * 0.5))[0], :]
    print 'postcc = ', postcc
    # dilate the bottom part of the C
    postcc[:, 0] = np.mean(cc[bot, 0]) - (wletter + tt)/wletter * (postcc[:, 0] - np.mean(cc[bot, 0]))
    postcc[:, 1] -= tt

    # reverse and truncate at first instance of C hitting bottom bounding line
    postcc = postcc
    stop = np.where(postcc[:, 1] < y0)[0][0]
    postcc = postcc[0:stop, :]

    x0 += wletter + space
    ss = draw_letter_s(y0, y0 + hletter, x0, x0 + wletter)

    x0 += wletter + space + tt * 0.5
    premm = np.array([[x0 - tt, y0 + hletter], [x0 - tt, y0]])
    mm = draw_letter_m(y0, y0 + hletter, x0, x0 + wletter, midpt_height=0.6)

    x0 += wletter + space
    ff = draw_letter_f(y0, y0 + hletter, x0, x0 + wletter, tt, midline_height=0.5, midline_width=0.6)
    # mask = np.where(np.logical_and(rr[:, 0] > (x0 + wletter * 0.5), rr[:, 1] > (y0 + hletter * 0.5)))[0]
    # prerr2 = copy.deepcopy(rr[mask])[::-1]
    # rr2[0:int(len(rr) * 0.5), 1] -= tt
    # top = np.where(rr2[:, 1] > y0 + hletter * 0.5)[0]
    # prerr2[:, 0] = np.mean(rr2[top, 0]) + (wletter + tt)/wletter * (prerr2[:, 0] - np.mean(rr2[top, 0]))

    postff = np.array([xend, y0 + hletter])

    chernword = np.vstack((precc, cc, postcc, ss, premm, mm, ff, postff))
    if lp['check']:
        plt.plot(chernword[:, 0], chernword[:, 1], 'b.-')
        ax = plt.gca()
        ax.axis('equal')
        plt.show()

    chernword[:, 0] -= (xend - xstart) * 0.5
    chernword[:, 1] -= 0.5
    chernword *= yspan

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(chernword[:, 0], chernword[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([rightx, rightx, leftx, leftx])
        boty = np.min(xy[:, 1]) - 10.
        closecy = np.array([chernword[-1, 1], boty, boty, chernword[0, 1]])
        curvx = np.hstack((np.array([leftx]), chernword[:, 0], closecx))
        curvy = np.hstack((np.array([chernword[0, 1]]), chernword[:, 1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([leftx, leftx, rightx, rightx])
        topy = np.max(xy[:, 1]) + 10.
        closecy = np.array([chernword[0, 1], topy, topy, chernword[-1, 1]])
        curvx = np.hstack((np.array([rightx]), chernword[:, 0][::-1], closecx))
        curvy = np.hstack((np.array([chernword[-1, 1]]), chernword[:, 1][::-1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    xy, NL, KL, BL = mask_with_polygon(lp['shape'], lp['NH']/medbL, lp['NV']/medbL, xy, BL, eps=0.00, check=lp['check'])

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + sf.float2pstr(lp['thres'], ndigits=1)
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_chicago(lp):
    """Build network with 'CHICAGO' spelled in kagomization. Periodic boundaries are not supported.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 2.0

    check = lp['check']
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['check'] = False
    lp_tmp['NH'] += 10
    lp_tmp['NV'] += 10
    if 'hucent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp_tmp)
    elif 'isocent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp_tmp)
    lp = lp_tmp
    lp['check'] = check
    lp['NH'] -= 10
    lp['NV'] -= 10

    # Draw curve UC
    # Draw U: part0 is left vertical, part1 is bottom curve, part2 is right vertical
    # use the fact that we've scaled the centroidal network by medbL to infer what it was previously
    medbL = (lp['NV'] + 10.) / (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    yspan = np.max(xy[:, 1]) - np.min(xy[:, 1]) - float(10.)/float(medbL)
    rescale = 1.5
    xstart = 0.0
    x0 = 0.1 * rescale
    rr = 0.11 * rescale
    tt = lp['thres']/yspan
    wletter = 0.2 * rescale
    iiwidth = wletter * 0.6  # special width of letter 'I'
    hletter = 0.4 * rescale
    space = 0.1 * rescale
    xend = 6. * wletter + iiwidth + 2. * x0 + 6. * space
    y0 = 0.5 - hletter * 0.5
    cc = draw_letter_c(y0, y0 + hletter, x0, x0 + wletter, rr)
    precc = copy.deepcopy(cc[0:int(len(cc) * 0.5), :])[::-1]
    precc = np.vstack((np.array([xstart, precc[0, 1]]), precc))
    cc[0:int(len(cc)*0.5), 1] -= tt
    top = np.where(cc[:, 1] > y0 + hletter * 0.5)[0]
    precc[:, 0] = np.mean(cc[top, 0]) + (wletter + tt)/wletter * (precc[:, 0] - np.mean(cc[top, 0]))

    x0 += wletter + space
    hh = draw_letter_h(y0, y0 + hletter, x0, x0 + wletter, tt)
    prehh = np.array([[hh[0, 0] - tt, cc[-1, 1]], [hh[0, 0] - tt, hh[0, 1]]])

    x0 += wletter + space
    # if 'kaglow' in lp['LatticeTop']:
    #     skew = lp['thres'] / yspan
    # elif 'kaghi' in lp['LatticeTop']:
    #     skew = - lp['thres'] / yspan
    # else:
    skew = 0.
    ii = draw_letter_i(y0, y0 + hletter, x0, x0 + iiwidth, skew)
    # ii[:, 1] = ii[::-1, 1]

    x0 += iiwidth + space
    cc2 = draw_letter_c(y0, y0 + hletter, x0, x0 + wletter, rr)
    mask = np.where(np.logical_and(cc2[:, 0] > (x0 + wletter * 0.5), cc2[:, 1] > (y0 + hletter * 0.5)))[0]
    precc2 = copy.deepcopy(cc2[mask])[::-1]
    cc2[0:int(len(cc2) * 0.5), 1] -= tt
    top = np.where(cc2[:, 1] > y0 + hletter * 0.5)[0]
    precc2[:, 0] = np.mean(cc2[top, 0]) + (wletter + tt)/wletter * (precc2[:, 0] - np.mean(cc2[top, 0]))

    x0 += wletter + space
    aa = draw_letter_a(y0, y0 + hletter, x0, x0 + wletter, tt)
    slope = float(hletter) / (wletter * 0.5)
    dx = (cc2[-1, 1] - y0) / slope
    preaa = np.array([[x0 + dx - tt, cc2[-1, 1]], [x0 - tt, y0]])

    x0 += wletter + space
    gg = draw_letter_g(y0, y0 + hletter, x0, x0 + wletter, rr, tt, hletter * 0.1)
    mask = np.logical_or(gg[:, 0] < x0 + wletter * 0.5, gg[:, 1] > y0 + hletter * 0.5)
    pregg = copy.deepcopy(gg[mask])[::-1]
    top = np.where(gg[:, 1] > y0 + hletter * 0.5)[0]
    bot = np.where(gg[:, 1] < y0 + hletter * 0.5)[0]
    gg[:, 0] = np.mean(gg[top, 0]) + (wletter - tt)/wletter * (gg[:, 0] - np.mean(gg[top, 0]))
    gg[top, 1] -= tt
    gg[bot, 1] += tt
    gg[-1, 1] -= tt
    gg[gg[:, 0] < x0 + tt, 0] = x0 + tt

    x0 += wletter + space
    oo = draw_letter_o(y0, y0 + hletter, x0, x0 + wletter - 1e-6, wletter * 0.5, tt)
    postoo = np.array([xend, y0])

    chicago = np.vstack((precc, cc, prehh, hh, ii, precc2, cc2, preaa, aa, pregg, gg, oo, postoo))
    if lp['check']:
        plt.plot(chicago[:, 0], chicago[:, 1], 'b.-')
        ax = plt.gca()
        ax.axis('equal')
        plt.show()

    chicago[:, 0] -= (xend - xstart) * 0.5
    chicago[:, 1] -= 0.5
    chicago *= yspan

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    # for ii in range(len(chicago) - 1):
    #     endpt1 = chicago[ii]
    #     endpt2 = chicago[ii + 1]
    #     close = le.pts_are_near_lineseg(xy, endpt1, endpt2, lp['thres'])
    #     todec[close] = True

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(chicago[:, 0], chicago[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([rightx, rightx, leftx, leftx])
        boty = np.min(xy[:, 1]) - 10.
        closecy = np.array([chicago[-1, 1], boty, boty, chicago[0, 1]])
        curvx = np.hstack((np.array([leftx]), chicago[:, 0], closecx))
        curvy = np.hstack((np.array([chicago[0, 1]]), chicago[:, 1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([leftx, leftx, rightx, rightx])
        topy = np.max(xy[:, 1]) + 10.
        closecy = np.array([chicago[0, 1], topy, topy, chicago[-1, 1]])
        curvx = np.hstack((np.array([rightx]), chicago[:, 0][::-1], closecx))
        curvy = np.hstack((np.array([chicago[-1, 1]]), chicago[:, 1][::-1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    xy, NL, KL, BL = mask_with_polygon(lp['shape'], lp['NH']/medbL, lp['NV']/medbL, xy, BL, eps=0.00, check=lp['check'])

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + sf.float2pstr(lp['thres'], ndigits=1)
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kagcentchern(lp):
    """Build network with 'CHERN' spelled in kagomization. Periodic boundaries are not supported.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 2.0

    check = lp['check']
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['check'] = False
    lp_tmp['NH'] += 10
    lp_tmp['NV'] += 10
    if 'hucent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp_tmp)
    elif 'isocent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp_tmp)
    lp = lp_tmp
    lp['check'] = check
    lp['NH'] -= 10
    lp['NV'] -= 10

    # Draw curve CHERN
    # use the fact that we've scaled the centroidal network by medbL to infer what it was previously
    medbL = (lp['NV'] + 10.) / (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    yspan = np.max(xy[:, 1]) - np.min(xy[:, 1]) - float(10.)/float(medbL)
    rescale = 1.7
    xstart = 0.0
    x0 = 0.1 * rescale
    rr = 0.11 * rescale
    tt = lp['thres'] / yspan
    wletter = 0.2 * rescale
    hletter = 0.4 * rescale
    space = 0.1 * rescale
    xend = 5. * wletter + 2. * x0 + 4. * space
    y0 = 0.5 - hletter * 0.5
    cc = draw_letter_c(y0, y0 + hletter, x0, x0 + wletter, rr)
    precc = copy.deepcopy(cc[0:int(len(cc) * 0.5), :])[::-1]
    precc = np.vstack((np.array([xstart, precc[0, 1]]), precc))
    cc[0:int(len(cc)*0.5), 1] -= tt
    top = np.where(cc[:, 1] > y0 + hletter * 0.5)[0]
    precc[:, 0] = np.mean(cc[top, 0]) + (wletter + tt)/wletter * (precc[:, 0] - np.mean(cc[top, 0]))

    x0 += wletter + space
    hh = draw_letter_h(y0, y0 + hletter, x0, x0 + wletter, tt, exittop=False)
    prehh = np.array([[hh[0, 0] - tt, cc[-1, 1]], [hh[0, 0] - tt, hh[0, 1]]])

    x0 += wletter + space
    # if 'kaglow' in lp['LatticeTop']:
    #     skew = lp['thres'] / yspan
    # elif 'kaghi' in lp['LatticeTop']:
    #     skew = - lp['thres'] / yspan
    # else:
    skew = 0.
    ee = draw_letter_e(y0, y0 + hletter, x0, x0 + wletter, tt)

    x0 += wletter + space
    rr = draw_letter_r(y0, y0 + hletter, x0, x0 + wletter, rr, tt)
    # mask = np.where(np.logical_and(rr[:, 0] > (x0 + wletter * 0.5), rr[:, 1] > (y0 + hletter * 0.5)))[0]
    # prerr2 = copy.deepcopy(rr[mask])[::-1]
    # rr2[0:int(len(rr) * 0.5), 1] -= tt
    # top = np.where(rr2[:, 1] > y0 + hletter * 0.5)[0]
    # prerr2[:, 0] = np.mean(rr2[top, 0]) + (wletter + tt)/wletter * (prerr2[:, 0] - np.mean(rr2[top, 0]))

    x0 += wletter + space
    nn = draw_letter_n(y0, y0 + hletter, x0, x0 + wletter, tt)
    slope = float(hletter) / (wletter * 0.5)
    dx = (nn[-1, 1] - y0) / slope
    # prenn = np.array([[x0 + dx - tt, nn[-1, 1]], [x0 - tt, y0]])

    postnn = np.array([xend, y0 + hletter])

    chernword = np.vstack((precc, cc, prehh, hh, ee, rr, nn, postnn))
    if lp['check']:
        plt.plot(chernword[:, 0], chernword[:, 1], 'b.-')
        ax = plt.gca()
        ax.axis('equal')
        plt.show()

    chernword[:, 0] -= (xend - xstart) * 0.5
    chernword[:, 1] -= 0.5
    chernword *= yspan

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(chernword[:, 0], chernword[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([rightx, rightx, leftx, leftx])
        boty = np.min(xy[:, 1]) - 10.
        closecy = np.array([chernword[-1, 1], boty, boty, chernword[0, 1]])
        curvx = np.hstack((np.array([leftx]), chernword[:, 0], closecx))
        curvy = np.hstack((np.array([chernword[0, 1]]), chernword[:, 1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([leftx, leftx, rightx, rightx])
        topy = np.max(xy[:, 1]) + 10.
        closecy = np.array([chernword[0, 1], topy, topy, chernword[-1, 1]])
        curvx = np.hstack((np.array([rightx]), chernword[:, 0][::-1], closecx))
        curvy = np.hstack((np.array([chernword[-1, 1]]), chernword[:, 1][::-1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    xy, NL, KL, BL = mask_with_polygon(lp['shape'], lp['NH']/medbL, lp['NV']/medbL, xy, BL, eps=0.00, check=lp['check'])

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + sf.float2pstr(lp['thres'], ndigits=1)
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp


def build_kagcentthanks(lp):
    """Build network with 'CHERN' spelled in kagomization. Periodic boundaries are not supported.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary

    Returns
    -------
    xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
    """
    # The kagtype : str specifier
    # (Where to kagomize: 'kagcurves' kags just along letters, 'kaglow' kags below letters, 'kaghi' above letters)
    # is determined from lp['LatticeTop']
    if 'thres' not in lp:
        lp['thres'] = 2.0

    check = lp['check']
    lp_tmp = copy.deepcopy(lp)
    lp_tmp['check'] = False
    lp_tmp['NH'] += 10
    lp_tmp['NV'] += 10
    if 'hucent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = bhucent.build_hucentroid(lp_tmp)
    elif 'isocent' in lp['LatticeTop']:
        xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten = biscent.build_iscentroid(lp_tmp)
    lp = lp_tmp
    lp['check'] = check
    lp['NH'] -= 10
    lp['NV'] -= 10

    # Draw curve THANKS
    # use the fact that we've scaled the centroidal network by medbL to infer what it was previously
    medbL = (lp['NV'] + 10.) / (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    yspan = np.max(xy[:, 1]) - np.min(xy[:, 1]) - float(10.)/float(medbL)
    rescale = 1.7
    xstart = 0.0
    x0 = 0.1 * rescale
    tt = lp['thres'] / yspan
    wletter = 0.2 * rescale
    hletter = 0.4 * rescale
    space = 0.1 * rescale
    xend = 6. * wletter + 2. * x0 + 5. * space
    y0 = 0.5 - hletter * 0.5
    tlet = draw_letter_t(y0, y0 + hletter, x0, x0 + wletter, tt)

    x0 += wletter + space
    hh = draw_letter_h(y0, y0 + hletter, x0, x0 + wletter, tt, exittop=False)
    hh[:, 1] = hh[:, 1][::-1]

    x0 += wletter + space
    aa = draw_letter_a(y0, y0 + hletter, x0, x0 + wletter, tt)

    x0 += wletter + space
    nn = draw_letter_n(y0, y0 + hletter, x0, x0 + wletter, tt)

    x0 += wletter + space
    kk = draw_letter_k(y0, y0 + hletter, x0, x0 + wletter, tt)

    x0 += wletter + space
    ss = draw_letter_s(y0, y0 + hletter, x0, x0 + wletter)

    postss = np.array([xend, y0 + hletter])

    thanksword = np.vstack((tlet, hh, aa, nn, kk, ss, postss))

    if lp['check']:
        plt.plot(thanksword[:, 0], thanksword[:, 1], 'b.-')
        ax = plt.gca()
        ax.axis('equal')
        plt.show()

    thanksword[:, 0] -= (xend - xstart) * 0.5
    thanksword[:, 1] -= 0.5
    thanksword *= yspan

    # Find all points to decorate
    todec = np.zeros(len(xy), dtype=bool)

    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(thanksword[:, 0], thanksword[:, 1], 'r-')
        plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
        plt.show()

    if 'kaglow' in lp['LatticeTop']:
        # Add all points below lower curve:
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([rightx, rightx, leftx, leftx])
        boty = np.min(xy[:, 1]) - 10.
        closecy = np.array([thanksword[-1, 1], boty, boty, thanksword[0, 1]])
        curvx = np.hstack((np.array([leftx]), thanksword[:, 0], closecx))
        curvy = np.hstack((np.array([thanksword[0, 1]]), thanksword[:, 1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    if 'kaghi' in lp['LatticeTop']:
        # Add all points above letters
        leftx = np.min(xy[:, 0]) - 10.
        rightx = np.max(xy[:, 0]) + 10.
        closecx = np.array([leftx, leftx, rightx, rightx])
        topy = np.max(xy[:, 1]) + 10.
        closecy = np.array([thanksword[0, 1], topy, topy, thanksword[-1, 1]])
        curvx = np.hstack((np.array([rightx]), thanksword[:, 0][::-1], closecx))
        curvy = np.hstack((np.array([thanksword[-1, 1]]), thanksword[:, 1][::-1], closecy))
        cpoly = np.dstack((curvx, curvy))[0]
        inpoly = dh.inds_in_polygon(xy, cpoly)
        todec[inpoly] = True
        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[todec, 0], xy[todec, 1], 'bo')
            plt.show()

    # decorate todec
    kaginds = np.where(todec)[0]
    xy, BL = blf.decorate_kagome_elements(xy, BL, kaginds, viewmethod=lp['viewmethod'], check=lp['check'])

    xy, NL, KL, BL = blf.mask_with_polygon(lp['shape'], lp['NH']/medbL, lp['NV']/medbL, xy, BL, eps=0.00,
                                           check=lp['check'])

    # If the meshfn going to overwrite a previous realization?
    lattice_exten = lp['LatticeTop'] + lattice_exten[10:] + '_thres' + sf.float2pstr(lp['thres'], ndigits=1)
    print 'lattice_exten = ', lattice_exten
    return xy, NL, KL, BL, PVx, PVy, PVxydict, LVUC, BBox, LL, LV, UC, lattice_exten, lp
