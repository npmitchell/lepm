import numpy as np
import lepm.build.build_lattice_functions as blf
import lepm.lattice_elasticity as le
import lepm.le_geometry as leg
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

'''Auxiliary functions for building networks that resemble conformal transformations of a lattice'''


def build_hexannulus(lp):
    """Build a U(1) symmetric honeycomb lattice within an annulus. The inner radius is given by alpha"""
    N = lp['NH']
    shape = 'circle'
    theta = np.linspace(0, 2. * np.pi, N + 1)
    theta = theta[:-1]
    # The radius, given the length of a side is:
    # radius = s/(2 * sin(2 pi/ n)), where n is number of sides, s is length of each side
    # We set the length of a side to be 1 (the rest length of each bond)
    Rout = 1. / (2. * np.sin(np.pi / float(N)))
    if lp['alph'] < 1e-5 or lp['alph'] > 1:
        raise RuntimeError('lp param alph must be > 0 and less than 1')
    Rinner = lp['alph'] * Rout

    # Make the first row
    xtmp = Rout * np.cos(theta)
    ytmp = Rout * np.sin(theta)
    xy = np.dstack([xtmp, ytmp])[0]
    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.2*np.random.rand(1)[0], xy[ii, 1]+0.2*np.random.rand(1)[0], str(ii))
        plt.title('First row')
        plt.show()

    # rotate by delta(theta)/2 and shrink xy so that outer bond lengths are all equal to unity
    # Here we have all bond lengths of the outer particles == 1, so that the inner radius is determined by
    #
    #           o z1
    #        /  |
    #      /    |
    # z2  o     |       |z2 - z0| = |z2 - z1| --> if z2 = r2 e^{i theta/2}, and z1 = r1 e^{i theta}, and z0 = r1,
    #       \   |                                 then r2 = r1 cos[theta/2] - sqrt[s**2 - r1**2 + r1**2 cos[theta/2]**2]
    #         \ |                                   or r2 = r1 cos[theta/2] + sqrt[s**2 - r1**2 + r1**2 cos[theta/2]**2]
    #           o z0                              Choose the smaller radius (the first one).
    #                                             s is the sidelength, initially == 1
    #                                             iterate sidelength: s = R*2*sin(2pi/N)
    #                                                --> see for ex, http://www.mathopenref.com/polygonradius.html
    dt = np.diff(theta)
    if (dt - dt[0] < 1e-12).all():
        dt = dt[0]
    else:
        print 'dt = ', dt
        raise RuntimeError('Not all thetas are evenly spaced')
    RR = Rout
    ss = 1.
    Rnext = RR * np.cos(dt * 0.5) - np.sqrt(ss**2 - RR**2 + RR**2 * np.cos(dt * 0.5)**2)
    rlist = [RR]

    # Continue adding more rows
    kk = 0
    while Rnext > Rinner:
        print 'Adding Rnext = ', Rnext
        print 'with bond length connecting to last row = ', ss
        # Add to xy
        if np.mod(kk, 2) == 0:
            xtmp = Rnext * np.cos(theta + dt*0.5)
            ytmp = Rnext * np.sin(theta + dt*0.5)
        else:
            xtmp = Rnext * np.cos(theta)
            ytmp = Rnext * np.sin(theta)
        xyadd = np.dstack([xtmp, ytmp])[0]
        xy = np.vstack((xy, xyadd))
        rlist.append(Rnext)

        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.2 * np.random.rand(1)[0], xy[ii, 1] + 0.2 * np.random.rand(1)[0], str(ii))
            plt.title('next row')
            plt.show()

        RR = Rnext
        ss = RR * np.sin(2. * np.pi / N)
        print 'next sidelength = ', ss
        Rnext = RR * np.cos(dt * 0.5) - np.sqrt(ss ** 2 - RR ** 2 + RR ** 2 * np.cos(dt * 0.5) ** 2)
        kk += 1

    Rfinal = RR
    # Get NV from the number of rows laid down.
    NVtri = len(rlist)
    lp['NV'] = NVtri - 2

    print('Triangulating points...\n')
    tri = Delaunay(xy)
    TRItmp = tri.vertices

    print('Computing bond list...\n')
    BL = le.Tri2BL(TRItmp)
    # bL = le.bond_length_list(xy,BL)
    thres = 1.1  # cut off everything longer than a diagonal
    print('thres = ' + str(thres))
    print('Trimming bond list...\n')
    BLtrim = le.cut_bonds(BL, xy, thres)
    print('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.')

    xy, NL, KL, BL = le.voronoi_lattice_from_pts(xy, NN=3, kill_outliers=True)

    # Remove any bonds that cross the inner circle
    tmpt = np.linspace(0, 2. * np.pi, 2000)
    innercircle = Rinner * np.dstack((np.cos(tmpt), np.sin(tmpt)))[0]
    # cycle the inner circle by one
    ic_roll = np.dstack((np.roll(innercircle[:, 0], -1), np.roll(innercircle[:, 1], -1)))[0]
    ic = np.hstack((innercircle, ic_roll))
    bondsegs = np.hstack((xy[BL[:, 0]], xy[BL[:, 1]]))
    does_intersect = linsegs.linesegs_intersect_linesegs(bondsegs, ic, thres=1e-6)
    keep = np.ones(len(BL[:, 0]), dtype=bool)
    keep[does_intersect] = False
    BL = BL[keep]

    # Remove any points that ended up inside the inner radius (this should not be necessary)
    inpoly = le.inds_in_polygon(xy, innercircle)
    keep = np.setdiff1d(np.arange(len(xy)), inpoly)
    print 'keep = ', keep
    print 'len(xy) = ', len(xy)
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min', check=lp['check'])

    # Randomize if eta > 0 specified
    eta = lp['eta']
    if eta == 0:
        xypts = xy
        eta_exten = ''
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]
        eta_exten = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    # Naming
    exten = '_circle_delta0p667_phi0p000_alph{0:0.2f}'.format(lp['alph']).replace('.', 'p') + eta_exten

    lattice_exten = 'hexannulus' + exten
    print 'lattice_exten = ', lattice_exten

    if lp['check']:
        le.display_lattice_2D(xy, BL, NL=NL, KL=KL, title='output from build_hexannulus()', close=False)
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.2 * np.random.rand(1)[0], xy[ii, 1] + 0.2 * np.random.rand(1)[0], str(ii))
        plt.show()

    # Scale up xy
    BM = le.BL2BM(xy, NL, BL, KL=KL, PVx=None, PVy=None)
    bL = le.BM2bL(NL, BM, BL)
    scale = 1./np.median(bL)
    xy *= scale
    return xy, NL, KL, BL, lattice_exten, lp


def build_hexannulus_filled_defects(lp):
    """Build a U(1) symmetric honeycomb lattice within an annulus. The inner radius is given by alpha
    At some cutoff radius, decrease the number of particles in each new row. Also, place a particle at the center.
    Instead of lp['alph'] determining the inner radius cutoff as a fraction of the system size,
    lp['alph'] as the cutoff radius for when to decrease the number of particles in each row.
    Remove lp['Ndefects'] at each row within a radius of lp['alph'] * system_size.

    Parameters
    ----------
    lp : dict

    """
    N = lp['NH']
    shape = 'circle'
    theta = np.linspace(0, 2. * np.pi, N + 1)
    theta = theta[:-1]
    # The radius, given the length of a side is:
    # radius = s/(2 * sin(2 pi/ n)), where n is number of sides, s is length of each side
    # We set the length of a side to be 1 (the rest length of each bond)
    Rout = 1. / (2. * np.sin(np.pi / float(N)))
    if lp['alph'] < 1e-5 or lp['alph'] > 1:
        raise RuntimeError('lp param alph must be > 0 and less than 1')
    Rinner = lp['alph'] * Rout

    # Make the first row
    xtmp = Rout * np.cos(theta)
    ytmp = Rout * np.sin(theta)
    xy = np.dstack([xtmp, ytmp])[0]
    if lp['check']:
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.2*np.random.rand(1)[0], xy[ii, 1]+0.2*np.random.rand(1)[0], str(ii))
        plt.title('First row')
        plt.show()

    # rotate by delta(theta)/2 and shrink xy so that outer bond lengths are all equal to unity
    # Here we have all bond lengths of the outer particles == 1, so that the inner radius is determined by
    #
    #           o z1
    #        /  |
    #      /    |
    # z2  o     |       |z2 - z0| = |z2 - z1| --> if z2 = r2 e^{i theta/2}, and z1 = r1 e^{i theta}, and z0 = r1,
    #       \   |                                 then r2 = r1 cos[theta/2] - sqrt[s**2 - r1**2 + r1**2 cos[theta/2]**2]
    #         \ |                                   or r2 = r1 cos[theta/2] + sqrt[s**2 - r1**2 + r1**2 cos[theta/2]**2]
    #           o z0                              Choose the smaller radius (the first one).
    #                                             s is the sidelength, initially == 1
    #                                             iterate sidelength: s = R*2*sin(2pi/N)
    #                                                --> see for ex, http://www.mathopenref.com/polygonradius.html
    dt = np.diff(theta)
    if (dt - dt[0] < 1e-12).all():
        dt = dt[0]
    else:
        print 'dt = ', dt
        raise RuntimeError('Not all thetas are evenly spaced')
    RR = Rout
    ss = 1.
    Rnext = RR * np.cos(dt * 0.5) - np.sqrt(ss**2 - RR**2 + RR**2 * np.cos(dt * 0.5)**2)
    rlist = [RR]

    # Continue adding more rows
    kk = 0
    while Rnext > Rinner:
        print 'Adding Rnext = ', Rnext
        print 'with bond length connecting to last row = ', ss
        # Add to xy
        if np.mod(kk, 2) == 0:
            xtmp = Rnext * np.cos(theta + dt*0.5)
            ytmp = Rnext * np.sin(theta + dt*0.5)
        else:
            xtmp = Rnext * np.cos(theta)
            ytmp = Rnext * np.sin(theta)
        xyadd = np.dstack([xtmp, ytmp])[0]
        xy = np.vstack((xy, xyadd))
        rlist.append(Rnext)

        if lp['check']:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.2 * np.random.rand(1)[0], xy[ii, 1] + 0.2 * np.random.rand(1)[0], str(ii))
            plt.title('next row')
            plt.show()

        RR = Rnext
        ss = RR * np.sin(2. * np.pi / N)
        print 'next sidelength = ', ss
        Rnext = RR * np.cos(dt * 0.5) - np.sqrt(ss ** 2 - RR ** 2 + RR ** 2 * np.cos(dt * 0.5) ** 2)
        kk += 1

    Rfinal = RR
    # Get NV from the number of rows laid down.
    NVtri = len(rlist)
    lp['NV'] = NVtri - 2

    print('Triangulating points...\n')
    tri = Delaunay(xy)
    TRItmp = tri.vertices

    print('Computing bond list...\n')
    BL = le.Tri2BL(TRItmp)
    # bL = le.bond_length_list(xy,BL)
    thres = 1.1  # cut off everything longer than a diagonal
    print('thres = ' + str(thres))
    print('Trimming bond list...\n')
    BLtrim = le.cut_bonds(BL, xy, thres)
    print('Trimmed ' + str(len(BL) - len(BLtrim)) + ' bonds.')

    xy, NL, KL, BL = le.voronoi_lattice_from_pts(xy, NN=3, kill_outliers=True)

    # Remove any bonds that cross the inner circle
    tmpt = np.linspace(0, 2. * np.pi, 2000)
    innercircle = Rinner * np.dstack((np.cos(tmpt), np.sin(tmpt)))[0]
    # cycle the inner circle by one
    ic_roll = np.dstack((np.roll(innercircle[:, 0], -1), np.roll(innercircle[:, 1], -1)))[0]
    ic = np.hstack((innercircle, ic_roll))
    bondsegs = np.hstack((xy[BL[:, 0]], xy[BL[:, 1]]))
    does_intersect = linsegs.linesegs_intersect_linesegs(bondsegs, ic, thres=1e-6)
    keep = np.ones(len(BL[:, 0]), dtype=bool)
    keep[does_intersect] = False
    BL = BL[keep]

    # Remove any points that ended up inside the inner radius (this should not be necessary)
    inpoly = le.inds_in_polygon(xy, innercircle)
    keep = np.setdiff1d(np.arange(len(xy)), inpoly)
    print 'keep = ', keep
    print 'len(xy) = ', len(xy)
    xy, NL, KL, BL = le.remove_pts(keep, xy, BL, NN='min', check=lp['check'])

    # Randomize if eta > 0 specified
    eta = lp['eta']
    if eta == 0:
        xypts = xy
        eta_exten = ''
    else:
        print 'Randomizing lattice by eta=', eta
        jitter = eta * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        xypts = np.dstack((xy[:, 0] + jitter[:, 0], xy[:, 1] + jitter[:, 1]))[0]
        eta_exten = '_eta{0:.3f}'.format(eta).replace('.', 'p')

    # Naming
    exten = '_circle_delta0p667_phi0p000_alph{0:0.2f}'.format(lp['alph']).replace('.', 'p') + eta_exten

    lattice_exten = 'hexannulus' + exten
    print 'lattice_exten = ', lattice_exten

    if lp['check']:
        le.display_lattice_2D(xy, BL, NL=NL, KL=KL, title='output from build_hexannulus()', close=False)
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.2 * np.random.rand(1)[0], xy[ii, 1] + 0.2 * np.random.rand(1)[0], str(ii))
        plt.show()

    # Scale up xy
    BM = le.BL2BM(xy, NL, BL, KL=KL, PVx=None, PVy=None)
    bL = le.BM2bL(NL, BM, BL)
    scale = 1./np.median(bL)
    xy *= scale
    return xy, NL, KL, BL, lattice_exten, lp
