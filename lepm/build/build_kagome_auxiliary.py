import numpy as np
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt

'''auxiliary functions for making kagome and kagome-like networks/lattices'''


def build_kagome_unitcell(lp):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    # Just draw the unit cell
    delta = lp['delta']
    # todo: handle phi_lattice and eta_lattice case
    phi = lp['phi']
    th = (np.pi - delta) * 0.5
    xy = np.array([[0., 0.], [np.cos(th) + np.sin(phi), np.sin(th) + np.cos(phi)], [2. * np.cos(th), 0.]])
    NL = np.array([[1, 2, 1, 2],
                   [0, 2, 0, 2],
                   [0, 1, 0, 1]])
    KL = np.array([[1, 1, -1, -1],
                   [1, 1, -1, -1],
                   [1, 1, -1, -1]])
    BL = np.array([[0, 1], [0, 2], [1, 2],
                   [0, -1], [0, -2], [-1, -2]])
    PV = np.array([[4 * np.cos(th), 0.],
                   [2. * np.sin(phi) + 2. * np.cos(th), 2. * np.cos(phi) + 2. * np.sin(th)]])
    pv0, pv1 = PV[0], PV[1]
    PVx = np.array([[0., 0., -pv1[0], -pv0[0]],
                    [0., 0., pv1[0], -pv0[0] + pv1[0]],
                    [0., 0., pv0[0], -pv1[0] + pv0[0]]])
    PVy = np.array([[0., 0., -pv1[1], -pv0[1]],
                    [0., 0., pv1[1], -pv0[1] + pv1[1]],
                    [0., 0., pv0[1], -pv1[1] + pv0[1]]])
    PVxydict = le.PVxy2PVxydict(PVx, PVy, NL, KL=KL)
    print 'PVxydict = ', PVxydict
    # sys.exit()
    LL = (pv0[0], pv1[1])

    # Create lattice_exten
    deltastr = '_delta{0:0.3f}'.format(delta).replace('.', 'p')
    phistr = '_phi{0:0.3f}'.format(phi).replace('.', 'p')
    lattice_exten = lp['LatticeTop'] + '_square_periodic' + deltastr + phistr

    # Create LV and UC
    # Bravais primitive unit vecs
    scale = 2. * np.abs(np.cos(0.0))
    LV = scale * np.array([[np.cos(2 * np.pi * float(p) / 3.), np.sin(2 * np.pi * float(p) / 3.)] for p in range(3)])
    # NOTE that a3 = -a1 -a2
    #      ^
    #       \ a2
    #        \
    #          -------> a1
    #        /
    #       /
    #      v  a3 = -a1-a2

    # make composite unit cell (multiple unit cells in size) --> regular kagome
    psi = [n * np.pi * (1. / 3.) for n in range(6)]
    b = np.array([[np.cos(psi[i]), np.sin(psi[i])] for i in range(6)])

    # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
    UC = np.array([[0., 0.], b[0] + b[1], b[0]])
    LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    xy -= np.mean(xy, axis=0)
    xx = PV[0, 0] + PV[1, 0]
    yy = PV[0, 1] + PV[1, 1]
    BBox = 0.5 * np.array([[-xx, -yy], [-xx, yy], [xx, yy], [xx, -yy]])
    # BBox = 'none'
    return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten


def build_deformed_kagome_unitcell(lp):
    """

    Returns
    -------

    """
    if lp['periodicBC'] and not lp['periodic_strip']:
        # Just draw the unit cell
        # Bravais lattice to be hexagonal with primitive vectors: a_{p+1} = (cos 2*pi*p/3, sin2*pi*p/3).
        # We parameterize the basis vectors as d1 = a1/2 + s2, d2= a2/2 - s1, and d3 = a3/2.
        # Defining s3 = -s1 -s2, sp describe the displacement of d_{p-1} relative to the
        # midpoint of the line along a_p that connects its neighbours at
        # d_{p+1} +/- a_{p-/+1} (with p defined mod 3), as indicated in Fig. 2a.
        # s_p are specified by 6 parameters with 2 constraints.
        # A symmetrical representation is to take sp = xp * (a_{p-1} - a_{p+1}) + y_p * a_P and
        # to use independent variables (x1, x2, x3; z) with z = y1 + y2 + y3.
        # The constrains then determine yp = z/3 + x_{p-1} - xp + 1
        # xp describes the buckling of the line of bonds along ap, so that when xp= 0 theline of bonds is straight.
        # z describes the asymmetry in the sizes ofthe two triangles.
        #
        x1, x2, x3, zz = lp['x1'], lp['x2'], lp['x3'], lp['z']

        aa = np.array([[np.cos(2 * np.pi * float(p) / 3.), np.sin(2 * np.pi * float(p) / 3.)] for p in range(3)])
        a1 = aa[0]
        a2 = aa[1]
        a3 = aa[2]

        # make unit cell
        xx = np.array([x1, x2, x3])
        yy = np.array([zz / 3. + xx[np.mod(i - 1, 3)] - xx[np.mod(i + 1, 3)] for i in [0, 1, 2]])
        ss = np.array([xx[p] * (aa[np.mod(p - 1, 3)] - aa[np.mod(p + 1, 3)]) + yy[p] * aa[p] for p in range(3)])
        s1 = ss[0]
        s2 = ss[1]
        d1 = a1 / 2. + s2
        d2 = a2 / 2. - s1
        d3 = a3 / 2.

        print 's1 = ', s1
        print 's2 = ', s2
        print 'd1 = ', d1
        print 'd2 = ', d2
        print 'd3 = ', d3

        # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
        cc = np.array([d1 + a2,
                       d3 + a1 + a2,
                       d2 + a1,
                       d1,
                       d3 + a1,
                       d2 - a2,
                       d1 + a3,
                       d3,
                       d2 + a3,
                       d1 + a2 + a3,
                       d3 + a2,
                       d2])
        #         0
        #
        # 10  11     1    2
        #
        #    9        3
        #
        #  8   7     5    4
        #
        #         6
        #
        #
        #     2      0
        #      \    /
        #       \ /
        #        1
        #      /   \
        # 2---0 --- 2---- 0
        #    /       \
        #   /         \
        #  1           1
        #
        # The unit cell is taken here to be 7-9, with periodic bonds elsewhere.
        xy = np.vstack((cc[8], cc[9], cc[7]))
        # Check it
        # plt.plot(cc[:, 0], cc[:, 1], 'r-')
        # plt.plot(xy[:, 0], xy[:, 1], 'bo')
        # plt.title('%0.2f + %0.2f + %0.2f + %0.2f' % (x1, x2, x3, zz))
        # plt.show()
        NL = np.array([[1, 2, 1, 2],
                       [0, 2, 0, 2],
                       [0, 1, 0, 1]])
        KL = np.array([[1, 1, -1, -1],
                       [1, 1, -1, -1],
                       [1, 1, -1, -1]])
        BL = np.array([[0, 1], [0, 2], [1, 2],
                       [0, -1], [0, -2], [-1, -2]])

        PV = np.array([a1, -a3])
        # PV = np.array([[4 * np.cos(th), 0.],
        #                [2. * np.sin(phi) + 2. * np.cos(th), 2. * np.cos(phi) + 2. * np.sin(th)]])
        pv0, pv1 = PV[0], PV[1]
        PVx = np.array([[0., 0., -pv1[0], -pv0[0]],
                        [0., 0., pv1[0], -pv0[0] + pv1[0]],
                        [0., 0., pv0[0], -pv1[0] + pv0[0]]])
        PVy = np.array([[0., 0., -pv1[1], -pv0[1]],
                        [0., 0., pv1[1], -pv0[1] + pv1[1]],
                        [0., 0., pv0[1], -pv1[1] + pv0[1]]])
        PVxydict = le.PVxy2PVxydict(PVx, PVy, NL, KL=KL)
        print 'PVxydict = ', PVxydict
        # sys.exit()
        LL = (pv0[0], pv1[1])

        # Create lattice_exten
        addstr = '_x1_{0:0.4f}'.format(x1).replace('.', 'p').replace('-', 'n')
        addstr += '_x2_{0:0.4f}'.format(x2).replace('.', 'p').replace('-', 'n')
        addstr += '_x3_{0:0.4f}'.format(x3).replace('.', 'p').replace('-', 'n')
        addstr += '_z_{0:0.4f}'.format(zz).replace('.', 'p').replace('-', 'n')
        lattice_exten = lp['LatticeTop'] + '_square_periodic' + addstr

        # Create LV and UC
        # Bravais primitive unit vecs
        scale = 2. * np.abs(np.cos(0.0))
        LV = scale * np.array([[np.cos(2. * np.pi * float(p) / 3.), np.sin(2. * np.pi * float(p) / 3.)]
                               for p in range(3)])
        # NOTE that a3 = -a1 -a2
        #      ^
        #       \ a2
        #        \
        #          -------> a1
        #        /
        #       /
        #      v  a3 = -a1-a2

        # make composite unit cell (multiple unit cells in size) --> regular kagome
        psi = [float(n) * np.pi * (1. / 3.) for n in range(6)]
        b = np.array([[np.cos(psi[i]), np.sin(psi[i])] for i in range(6)])

        # nodes at R (from top to bottom) -- like fig 2a of KaneLubensky2014
        UC = np.array([[0., 0.], b[0] + b[1], b[0]])
        LVUC = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        xy -= np.mean(xy, axis=0)
        xx = PV[0, 0] + PV[1, 0]
        yy = PV[0, 1] + PV[1, 1]
        BBox = 0.5 * np.array([[-xx, -yy], [-xx, yy], [xx, yy], [xx, -yy]])
        # BBox = 'none'
    else:
        # lattice is not periodic, return a full kagome star
        raise RuntimeError("have not written yet")

    return xy, NL, KL, BL, PVxydict, PVx, PVy, PV, LL, LVUC, LV, UC, BBox, lattice_exten
