import numpy as np
import lepm.lattice_elasticity as le
import lepm.stringformat as sf
import lepm.line_segments as linsegs
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
import lepm.plotting.network_visualization as netvis
import lepm.data_handling as dh
import copy

"""
Description
===========
Low-level functions for general use in the build modules"""


def unique_rows_index(a, rs=True):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    # ui[1:] = (diff != 0).any(axis=1)
    ui[1:] = (abs(diff) > .1).any(axis=1)

    if rs:
        b = np.array(a.copy(), dtype='float')
        dist = np.array([np.sqrt(np.sum((b - b[i]) ** 2, axis=1)) for i in range(len(b))])
        t = np.array(np.where(dist < .01)).T
        for k in range(len(t)):
            if t[k, 0] < t[k, 1]:
                ui[t[k, 1]] = False

    return ui, order


def cut_slit(BL, xy, cutL, x0, y0=0):
    """Cuts bonds along a specified slit.
    [Line segments go from q to q+s, and p to p+r. Taking the equation for their crossing p+tr=q+us, solve for t (and u)
     by taking cross product of both sides with s (and with r).]

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    cutL : float
        length of slit
    x0 : float
        x position of the center of the slit
    y0 : float (default = 0 )
        y position of the slit

    Returns
    ----------
    BLout : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points (slit removed)
    xyout : array of dimension nx2
        2D lattice of points (positions x,y), with points in slit removed
    """
    # uses http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    # for the algorithm
    #
    #
    # BL is bond list, x0,y0 position of center, cutL is cut length,
    # if rxs = 0, then lines are parallel
    # Line segments go from q to q+s, and p to p+r.
    # (p,p+r) is the slit itself.
    p = np.array([[x0 - cutL / 2., y0]])
    r = np.array([[x0 + cutL / 2., y0]]) - p
    q = np.dstack((xy[BL[:, 0], 0], xy[BL[:, 0], 1]))[0]
    s = np.dstack((xy[BL[:, 1], 0], xy[BL[:, 1], 1]))[0] - q
    # Taking the equation for their crossing p+tr = q +us, solve for t and u by
    # taking cross product of both sides with s, and with r
    rxs = r[0, 0] * s[:, 1] - r[0, 1] * s[:, 0]
    # rxs!=0 and t,u between 0 and 1 means intersection
    # r x s !=0:  unless 0<=t<=1 0<=u<=1 then in bulk or collinear
    slitIND = rxs != 0
    qmpxs = (q[:, 0] - p[:, 0]) * s[:, 1] - (q[:, 1] - p[:, 1]) * s[:, 0]
    qmpxr = (q[:, 0] - p[:, 0]) * r[0, 1] - (q[:, 1] - p[:, 1]) * r[0, 0]
    t = qmpxs[slitIND] / rxs[slitIND]
    u = qmpxr[slitIND] / rxs[slitIND]
    # if 0<=t<=1 and 0<=u<=1 then make false
    slitIND[rxs != 0] = np.logical_not(np.logical_and(np.logical_and(0 <= t, t <= 1), np.logical_and(0 <= u, u <= 1)))
    # slitIND[rxs!=0] = logical_not(logical_and(logical_and(0<t,t<1),logical_and(0<u,u<1)))

    # check for collinearity --> slitIND[rxs==0] = (array_len_rxs==0):False_where_collinear
    # If (q-p)x r !=0, then parallel but non-intersecting
    parnoninters = np.logical_and(rxs == 0, qmpxr != 0)
    slitIND[parnoninters] = True
    # collinearity possible-> 'cp'
    cp = np.logical_and(rxs == 0, qmpxr == 0)
    # for collinearity_possible, expressing endpts of second seg (q,q+s)
    # in terms of first as p+t*r, and check if
    # [t0,t1] intersects [0,1] --> if so, collinear and overlapping
    qmpdr = (q[cp, 0] - p[0, 0]) * r[0, 0] - (q[cp, 1] - p[0, 1]) * r[0, 1]
    rdr = np.dot(r, r.transpose())[0][0]
    t0 = qmpdr / rdr
    t1term = np.dot(s[cp, :], r.transpose()) / rdr
    t1 = t0 + t1term.transpose()
    # if t0<t1: t0>=1 or t1<=0  --> no overlap
    # if t1<t0: t1>=1 or t0<=0  --> no overlap
    slitIND[cp] = np.logical_or(np.logical_and(t0 < t1, np.logical_or(t0 >= 1, t1 <= 0)),
                                np.logical_and(t0 > t1, np.logical_or(t1 >= 1, t0 <= 0)))
    # Now slitIND has all nonintersecting, nonoverlapping segment indices.

    # Make BLout
    BLt = BL[slitIND, :]

    # Make xyout
    # Clear xy points on the slit (but within the endpts)
    # Reorder BLout to match new coords by making map from old to new
    on_seg = linsegs.point_is_on_linesegment_2D(xy, p[0], p[0] + r[0])
    # points to remove
    pts_to_rmv = np.where(on_seg)[0]
    # BL to reorder
    BL_r = BLt.copy()
    for ind in pts_to_rmv:
        BL_r[BLt > ind] = (BL_r[BLt > ind] - 1)

    BLout = np.sort(BL_r, axis=1)
    BLout = dh.unique_rows(BLout)
    xyout = xy[~on_seg]

    return BLout, xyout


def decorate_as_kagome(xy, BL, PVxydict=None, check=False):
    """Decorate a honeycomb-like lattice to be a kagome.
    For periodic systems, if (i,j) is in PVxydict, then the new points will contain the point closer to i (midpt of
    bond ij).

    Parameters
    ----------
    xy : NP x 2 array
        Equilibrium lattice positions of honeycomb-like lattice
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of honeycomb-like lattice
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    check : bool
        Display intermediate results

    Returns
    ----------
    pts : NP x 2 array
        Equilibrium lattice positions of kagome lattice
    BLout : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of kagome lattice
    """
    # if check:
    #     print 'inspecting input network (the honeycomb-like network)'
    #     netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=True, colormap='inferno')

    if (BL < 0).any():
        if PVxydict is None:
            raise RuntimeError('make_lattice.py -> decorate_as_kagome: PVxydict not supplied but BL has periodic bonds')
        kk = 0
        pts = np.array([np.mean(xy[row], axis=0) for row in BL])
        for row in BL:
            # if (i,j) or (j,i) is in PVxydict, then the new pointset will contain the point closer to row[0]
            if (row < 0).any():
                if (abs(row[0]), abs(row[1])) in PVxydict:
                    PVxy = PVxydict[(abs(row[0]), abs(row[1]))]
                    if len(np.shape(PVxy)) > 1:
                        # There are more than one periodic bond connecting the two sites
                        for pvxy in PVxy:
                            xyrow = np.vstack((xy[np.abs(row[0])], xy[np.abs(row[1])] + pvxy))
                            pts[kk] = np.mean(xyrow, axis=0)
                    else:
                        xyrow = np.vstack((xy[np.abs(row[0])], xy[np.abs(row[1])] + PVxy))
                        pts[kk] = np.mean(xyrow, axis=0)
                elif (abs(row[1]), abs(row[0])) in PVxydict:
                    PVxy = PVxydict[(abs(row[1]), abs(row[0]))]
                    if len(np.shape(PVxy)) > 1:
                        # There are more than one periodic bond connecting the two sites
                        for pvxy in PVxy:
                            xyrow = np.vstack((xy[np.abs(row[0])], xy[np.abs(row[1])] - pvxy))
                            pts[kk] = np.mean(xyrow, axis=0)
                    else:
                        xyrow = np.vstack((xy[np.abs(row[0])], xy[np.abs(row[1])] - PVxy))
                        pts[kk] = np.mean(xyrow, axis=0)
                else:
                    print 'build_latfns: available in PVxydict:'
                    for key in PVxydict:
                        print 'key = ', key
                    print 'build_latfns: row = ', row
                    raise RuntimeError('periodic bond in BL not found in PVxydict as key')
            else:
                pts[kk] = np.mean(xy[row], axis=0)
            kk += 1

        if check:
            print 'inspecting input network (the honeycomb-like network) with new points'
            netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
            plt.plot(pts[:, 0], pts[:, 1], 'ko')
            plt.show()

        # Connect new pt to pts on other bonds connected to original node
        BLout = []
        done_inds = []
        was_periodic = np.zeros_like(BL[:, 0], dtype=bool)
        site_near_first = np.zeros_like(BL[:, 0], dtype=bool)
        dmyi = 0
        PVxyd_out = {}
        for row in np.abs(BL):
            # Creating new vertices at midpoints of bonds.
            # Note that the new vertices are simply indexed by their bond #, so the current new pt is dmyi
            # The current bomd, dmyi, will connect by real bonds to the neighboring new particles (formerly bonds) of i,
            # and will ahve periodic connections to the neighboring new particles of j.
            #
            #               |  This is a periodic bond to j
            #               |
            #               *
            #          kk   |
            #        ___*___| i
            #                \                 Case 1: kk, kk~, jj, jj~ are all real bonds
            #                 \                Case 2: jj and or jj~ are periodic bonds
            #      |           * kk~           Case 3: kk and or kk~ are periodic bonds
            #      |            \
            #      * jj          \
            #      |     jj~
            #    j |___*____
            #     /
            #    /
            #   *  This is a periodic bond going to i
            #  /
            # /
            # Get where in BL the first particle is row[0] and the second is row[1] (ie i and j)
            isin_i = np.abs(BL) == row[0]
            isin_j = np.abs(BL) == row[1]
            # From those, get all bonds with these particles in them
            isin = np.where(np.logical_or(isin_i, isin_j))[0]
            # Remove the current bond from each list
            isin = np.setdiff1d(np.unique(isin), [dmyi])
            isin_i = np.setdiff1d(np.where(isin_i)[0], [dmyi])
            isin_j = np.setdiff1d(np.where(isin_j)[0], [dmyi])
            # Check if this bond is periodic

            # if (BL[dmyi] < 0).any():
            # Determined that bondij is periodic
            # Is decoration pt on bondij on other side of system?
            # No, connect up bonds normally, with no PV
            # Yes, connect decoration point on row to image point and add to PV

            # Determine which side of the system the decoration pt on this bond lies -->
            # create periodic bonds to jj and jj~, real bonds to kk and kk~
            was_periodic[dmyi] = (BL[dmyi] < 0).any()
            site_near_first[dmyi] = True
            # decoration pt is on same side of system as row[0], form periodic bonds with isjj elements
            # we add the bonds here, it's ok if we add it again later in reverse order

            if check:
                netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
                plt.plot(pts[:, 0], pts[:, 1], 'ko')
                plt.plot(pts[dmyi, 0], pts[dmyi, 1], 'ro', markersize=20)
                for key in BLout:
                    if (np.array(key) < 0).any():
                        if tuple(np.abs(key)) in PVxyd_out:
                            pvxy = PVxyd_out[tuple(np.abs(key))]
                            pt0 = pts[abs(key[0])]
                            pt1 = pts[abs(key[1])] + pvxy
                            pt2 = pts[abs(key[0])] - pvxy
                            pt3 = pts[abs(key[1])]

                        elif tuple(np.abs(key[::-1])) in PVxyd_out:
                            pvxy = PVxyd_out[tuple(np.abs(key[::-1]))]
                            pt0 = pts[abs(key[0])] + pvxy
                            pt1 = pts[abs(key[1])]
                            pt2 = pts[abs(key[0])]
                            pt3 = pts[abs(key[1])] - pvxy

                        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'k--')
                        plt.plot([pt2[0], pt3[0]], [pt2[1], pt3[1]], 'k--')
                    else:
                        plt.plot(pts[np.abs(key), 0], pts[np.abs(key), 1], 'k-')
                for ind in range(len(xy)):
                    plt.text(xy[ind, 0] + 0.01, xy[ind, 1], str(ind), color='r')
                for ind in range(len(pts)):
                    plt.text(pts[ind, 0] + 0.01, pts[ind, 1], str(ind), color='k')

                plt.title('Inspecting highlighted bond (red dot): ' + str(row))
                print 'to connect', row, 'to isin_i = ', isin_i
                print '                   and  isin_j = ', isin_j
                plt.pause(0.001)
                # plt.show()

            BLout, PVxyd_out = kagome_decorate_add_bonds(xy, BL, PVxydict, dmyi, BLout, PVxyd_out,
                                                         isin_i, isin_j, pts, check=check)

            # else:
            #     for neighbor in isin:
            #         BLout.append([dmyi, neighbor])
            #
            #         # # Check
            #         # if check:
            #         #     plt.plot([pts[dmyi, 0], pts[neighbor, 0]],
            #                        [pts[dmyi, 1], pts[neighbor, 1]], 'r-', alpha=0.3)
            #         #     plt.pause(0.0001)

            done_inds.append(dmyi)
            dmyi += 1

        BLout = np.asarray(BLout)
        BLout = np.sort(BLout, axis=1)
        BLout = dh.unique_rows(BLout)

        PVxydictout = PVxyd_out
    else:
        # For each bond, place a pt at the center.
        pts = np.array([np.mean(xy[row], axis=0) for row in BL])

        # Connect new pt to pts on other bonds connected to original node
        BLout = []
        dmyi = 0
        if check:
            plt.clf()
            le.display_lattice_2D(xy, BL, close=False)

        for row in BL:
            isin = np.where(np.logical_or(BL == row[0], BL == row[1]))[0]
            isin = np.setdiff1d(np.unique(isin), [dmyi])
            for neighbor in isin:
                BLout.append([dmyi, neighbor])

                # Check
                # plt.plot( [pts[dmyi,0],pts[neighbor,0]], [pts[dmyi,1],pts[neighbor,1]],'r-',alpha=0.3)
                # plt.pause(0.0001)
            dmyi += 1

        BLout = np.asarray(BLout)
        BLout = np.sort(BLout, axis=1)
        BLout = dh.unique_rows(BLout)
        PVxydictout = None

    if check:
        plt.clf()
        netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
        netvis.movie_plot_2D(pts, BLout, PVxydict=PVxydictout, show=False, colormap='inferno')
        # for row in BLout:
        #     plt.plot([pts[row[0], 0], pts[row[1], 0]], [pts[row[0], 1], pts[row[1], 1]], 'r-', alpha=0.3)
        plt.show()

    return pts, BLout, PVxydictout


def kagome_decorate_add_bonds(xy, BL, PVxydict, dmyi, BLout, PVxyd_out, isin_i, isin_j, pts, check=False):
    """

    Parameters
    ----------
    xy
    BL
    PVxydict
    dmyi
    BLout
    PVxyd_out
    isin_i
    isin_j
    pts
    check

    Returns
    -------

    """
    row = np.abs(BL[dmyi])

    # check what's next
    if check:
        print 'for dmyi = ', dmyi, ' -> isin_j = ', isin_j, ' --> BL[isin_j] = ', BL[isin_j]

    for isjj in isin_j:
        if check:
            print 'dmyi=', dmyi, 'isjj = ', isjj
            print 'dmyi=', dmyi, 'BL[isjj] = ', BL[isjj]

        # get pvij
        # If key = (i,j) and val = np.array([ 5.0, 2.0]), then particle i sees particle j at xy[j]+val
        if (BL[dmyi] < 0).any():
            if (row[0], row[1]) in PVxydict:
                pvij = copy.deepcopy(PVxydict[(row[0], row[1])])
            elif (row[1], row[0]) in PVxydict:
                pvij = copy.deepcopy(-PVxydict[(row[1], row[0])])
            else:
                raise RuntimeError('Bond is periodic but there is no corresponding entry in PVxydict')
            if check:
                print 'row periodic bond: ', row, ' pvij = ', pvij
        else:
            # This is not a periodic bond
            pvij = np.array([0., 0.])
            if check:
                print 'row is not periodic bond: ', row

        # Check if the bond/new particle being connected to is also periodic
        if (BL[isjj] < 0).any():
            # The bond being connected to across a periodic bond is also periodic. Check where it is.
            if np.abs(BL[isjj])[0] == row[1]:
                # The new particle for bond isjj is near j, not at a reflection--> no change to pvij
                pass
            elif np.abs(BL[isjj])[1] == row[1]:
                print 'new particle for bond isjj is not near j'
                # The new particle for bond isjj is not near j --> add periodic vector to pvij
                # add displacement taking BL[isijj][0] to image as seen by row[1]
                if (row[1], abs(BL[isjj][0])) in PVxydict:
                    pvij += PVxydict[(row[1], abs(BL[isjj][0]))]
                    print ' pvij ->', pvij, ' row = ', row
                elif (abs(BL[isjj][0]), row[1]) in PVxydict:
                    pvij += -PVxydict[(abs(BL[isjj][0]), row[1])]
                    print ' pvij -> ', pvij, ' row = ', row
            else:
                print 'BL[isjj] = ', BL[isjj]
                print 'row = ', row
                print 'np.abs(BL[isjj])[0] == row[1] =>', np.abs(BL[isjj])[0] == row[1]
                print 'np.abs(BL[isjj])[1] == row[1] =>', np.abs(BL[isjj])[1] == row[1]
                raise RuntimeError('The bond seems to be missing the point it must have (point j)')

        # check if it hasn't been added to PVxyd_out, add the info it not already there
        if (np.abs(pvij) > 0.).any():
            if (isjj, dmyi) not in PVxyd_out and (dmyi, isjj) not in PVxyd_out:
                if isjj < dmyi:
                    PVxyd_out[(isjj, dmyi)] = -pvij
                else:
                    PVxyd_out[(dmyi, isjj)] = pvij
            elif (isjj, dmyi) in PVxyd_out:
                # check to add -pvij to this if not already part of the value
                if (PVxyd_out[(isjj, dmyi)] != -pvij).any():
                    if check:
                        print '(dmyi, isjj) = ', (dmyi, isjj)
                        print 'PVxyd_out[(isjj, dmyi)] = ', PVxyd_out[(isjj, dmyi)]
                        netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
                        plt.plot(pts[:, 0], pts[:, 1], 'ko')
                        plt.plot(pts[[dmyi, isjj], 0], pts[[dmyi, isjj], 1], 'r^')
                        plt.title('Found bond with two PVs, connecting red triangles')
                        for ind in range(len(BL)):
                            plt.text(pts[ind, 0] + 0.2, pts[ind, 1], str(ind))
                        plt.show()
                    PVxyd_out[(isjj, dmyi)] = np.vstack((PVxyd_out[(isjj, dmyi)], pvij))
            elif (dmyi, isjj) in PVxyd_out:
                # check to add pvij to this if not already part of the value
                if (PVxyd_out[(dmyi, isjj)] != pvij).any():
                    if check:
                        print '(dmyi, isjj) = ', (dmyi, isjj)
                        print 'PVxyd_out[(dmyi, isjj)] = ', PVxyd_out[(dmyi, isjj)]
                        netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
                        plt.plot(pts[:, 0], pts[:, 1], 'ko')
                        plt.plot(pts[[dmyi, isjj], 0], pts[[dmyi, isjj], 1], 'r^', markersize=20)
                        plt.title('Found bond with two PVs, connecting red triangles')
                        for ind in range(len(BL)):
                            plt.text(pts[ind, 0] + 0.2, pts[ind, 1], str(ind))
                        plt.show()
                    PVxyd_out[(dmyi, isjj)] = np.vstack((PVxyd_out[(dmyi, isjj)], pvij))

            BLout.append([-dmyi, -isjj])
        else:
            # Bond is not periodic
            BLout.append([dmyi, isjj])

    # check what's next
    if check:
        print 'for dmyi = ', dmyi, ' -> isin_i = ', isin_i, ' --> BL[isin_i] = ', BL[isin_i]

    for isii in isin_i:
        if check:
            print 'dmyi=', dmyi, 'isii = ', isii
            print 'dmyi=', dmyi, 'BL[isii] = ', BL[isii]
        # Make real bonds to kk and kk~ unless bonds kk and kk~ are periodic with particle i being
        # the second element (element 1) in those bonds.
        if (BL[isii] < 0).any():
            # The bond to connect to is periodic. Check if we're connecting to a distant point
            # (as would be the case if i is second element of BL[isii)
            if abs(BL[isii][1]) == row[0]:
                # site i is the second element, so we connect new site dmyi to the image particle of kk
                if (row[0], abs(BL[isii, 0])) in PVxydict:
                    pvij = PVxydict[(row[0], abs(BL[isii, 0]))]
                elif (abs(BL[isii, 0]), row[0]) in PVxydict:
                    pvij = -PVxydict[(abs(BL[isii, 0]), row[0])]
                else:
                    print 'BL[isii] = ', BL[isii]
                    print 'row = ', row
                    raise RuntimeError('Bond is periodic but not in PVxydict')
            else:
                pvij = np.array([0., 0.])

            if (np.abs(pvij) > 0.).any():
                # The bond is an image bond, not a real one in the bulk
                # check if it hasn't been added to PVxyd_out, add the info if not already there
                if (isii, dmyi) not in PVxyd_out and (dmyi, isii) not in PVxyd_out:
                    if isii < dmyi:
                        PVxyd_out[(isii, dmyi)] = -pvij
                    else:
                        PVxyd_out[(dmyi, isii)] = pvij
                elif (isii, dmyi) in PVxyd_out:
                    # check to add -pvij to this if not already part of the value
                    if (PVxyd_out[(isii, dmyi)] != -pvij).any():
                        if check:
                            print '(dmyi, isii) = ', (dmyi, isii)
                            print 'PVxyd_out[(dmyi, isii)] = ', PVxyd_out[(dmyi, isii)]
                            netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
                            plt.plot(pts[:, 0], pts[:, 1], 'ko')
                            plt.plot(pts[[dmyi, isii], 0], pts[[dmyi, isii], 1], 'r^', markersize=20)
                            plt.title('Found bond with two PVs, connecting red triangles')
                            for ind in range(len(BL)):
                                plt.text(pts[ind, 0] + 0.2, pts[ind, 1], str(ind))
                            plt.show()
                        PVxyd_out[(isii, dmyi)] = np.vstack((PVxyd_out[(isii, dmyi)], pvij))
                elif (dmyi, isii) in PVxyd_out:
                    # check to add pvij to this if not already part of the value
                    if (PVxyd_out[(dmyi, isii)] != pvij).any():
                        if check:
                            print '(dmyi, isii) = ', (dmyi, isii)
                            print 'PVxyd_out[(dmyi, isii)] = ', PVxyd_out[(dmyi, isii)]
                            netvis.movie_plot_2D(xy, BL, PVxydict=PVxydict, show=False, colormap='inferno')
                            plt.plot(pts[:, 0], pts[:, 1], 'ko')
                            plt.plot(pts[[dmyi, isii], 0], pts[[dmyi, isii], 1], 'r^', markersize=20)
                            plt.title('Found bond with two PVs, connecting red triangles')
                            for ind in range(len(BL)):
                                plt.text(pts[ind, 0] + 0.2, pts[ind, 1], str(ind))
                            plt.show()
                        PVxyd_out[(dmyi, isii)] = np.vstack((PVxyd_out[(dmyi, isii)], pvij))

                BLout.append([-dmyi, -isii])
            else:
                BLout.append([dmyi, isii])
        else:
            BLout.append([dmyi, isii])

    return BLout, PVxyd_out


def generate_lattice(image_shape, lattice_vectors):
    """Creates lattice of positions from arbitrary lattice vectors.

    Parameters
    ----------
    image_shape : 2 x 1 list (eg image_shape=[L,L])
        Width and height of the lattice (square)
    lattice_vectors : 2 x 1 list of 2 x 1 lists (eg [[1 ,0 ],[0.5,sqrt(3)/2 ]])
        The two lattice vectors defining the unit cell.

    Returns
    ----------
    xy : NPx2 float array
        2D lattice of points (positions x,y)
    LVUC : NP x 3 int array
        Positions in terms of lattice vectors (supplied) and unit cell vectors (which are all zero here)
        For instance, xy[0,:] = LV[0]*LVUC[0,0] + LV[1]*LVUC[0,1] + UC[LVUC[0,2]]
    """
    eps = 1e-10
    # Generate lattice that lives in
    # center_pix = np.array(image_shape) // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = image_shape[0] // dx_cell
    ny = image_shape[1] // dy_cell

    # Generate a square lattice, with too many points.
    # Here I generate a factor of 4 more points than I need, which ensures
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.

    # Old method (03-2017)
    x_sq = np.arange(-nx, nx + 1, dtype=float)
    y_sq = np.arange(-ny, ny + 1, dtype=float)
    # New method
    # x_sq = np.arange(-nx, 2*nx + 1, dtype=float)
    # y_sq = np.arange(-ny, 2*ny + 1, dtype=float)

    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    # transpose so that row is along x axis
    x_lattice = lattice_vectors[0][1] * x_sq + lattice_vectors[1][1] * y_sq
    y_lattice = lattice_vectors[0][0] * x_sq + lattice_vectors[1][0] * y_sq

    # Trim to fit in box.
    mask = ((x_lattice < image_shape[0] * 0.5 + eps)
            & (x_lattice > -image_shape[0] * 0.5))
    mask = mask & ((y_lattice < image_shape[1] * 0.5 + eps)
                   & (y_lattice > -image_shape[1] * 0.5))

    # Check
    # import matplotlib.pyplot as plt
    # print 'x_lattice = ', x_lattice
    # print 'y_lattice = ', y_lattice
    # xtmp = 0.5 * image_shape[0] * np.array([-1, -1, 1, 1, -1])
    # ytmp = 0.5 * image_shape[1] * np.array([-1, 1, 1, -1, -1])
    # plt.plot(xtmp, ytmp, 'g-')
    # plt.plot(x_lattice[mask].ravel(), y_lattice[mask].ravel(), 'ro')
    # plt.plot(x_lattice.ravel(), y_lattice.ravel(), 'b.')
    # plt.show()
    # sys.exit()

    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]

    # Added 03-2017
    x_lattice -= np.mean(x_lattice)
    y_lattice -= np.mean(y_lattice)

    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = x_lattice
    out[:, 1] = y_lattice
    # sort primarily by x, then y
    i = np.lexsort((out[:, 1], out[:, 0]))
    xy = out[i]

    # # Check the ordering
    # LV = np.array(lattice_vectors)
    # print 'LV = ', LV
    # sizes = np.arange(len(xy))+5
    # colorvals = np.linspace(0.1,1,len(xy))
    # plt.scatter(xy[:,0],xy[:,1], s=sizes+5, c=colorvals, cmap='afmhot')
    # xyLVtmp = np.array([LVUC[ii,0]*LV[0] + LVUC[ii,1]*LV[1] for ii in range(len(xy))])
    # plt.colorbar()
    # plt.figure()
    # print 'xyLVtmp = ', xyLVtmp
    # plt.scatter( xyLVtmp[:,0]- np.mean(xyLVtmp,axis=0)[0],\
    #              xyLVtmp[:,1] - np.mean(xyLVtmp,axis=0)[0]-0.1, s=sizes, c=colorvals, cmap='afmhot' )
    # plt.colorbar()
    # plt.show()
    return xy


def generate_lattice_LVUC(image_shape, lattice_vectors):
    """Creates lattice of positions from arbitrary lattice vectors.

    Parameters
    ----------
    image_shape : 2 x 1 list (eg image_shape=[L,L])
        Width and height of the lattice (square)
    lattice_vectors : 2 x 1 list of 2 x 1 lists (eg [[1 ,0 ],[0.5,sqrt(3)/2 ]])
        The two lattice vectors defining the unit cell.

    Returns
    ----------
    xy : NPx2 float array
        2D lattice of points (positions x,y)
    LVUC : NP x 3 int array
        Positions in terms of lattice vectors (supplied) and unit cell vectors (which are all zero here)
        For instance, xy[0,:] = LV[0]*LVUC[0,0] + LV[1]*LVUC[0,1] + UC[LVUC[0,2]]
    """
    # Generate lattice that lives in
    # center_pix = np.array(image_shape) // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = image_shape[0] // dx_cell
    ny = image_shape[1] // dy_cell
    # Generate a square lattice, with too many points.
    # Here I generate a factor of 4 more points than I need, which ensures
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    # transpose so that row is along x axis
    x_lattice = np.array(lattice_vectors[0][0] * x_sq + lattice_vectors[1][0] * y_sq).ravel()
    y_lattice = np.array(lattice_vectors[0][1] * x_sq + lattice_vectors[1][1] * y_sq).ravel()

    # Make identification in terms of lattice vectors (UC = 0 for all)
    # LVUC = np.dstack((x_sq+0*y_sq, y_sq+0*x_sq))
    LVUC0 = ((x_sq + 0 * y_sq).astype(int)).ravel()
    LVUC1 = ((0 * x_sq + y_sq).astype(int)).ravel()
    LVUC = np.dstack((LVUC0, LVUC1))[0]

    # print 'LVUC = ', LVUC
    # print 'np.shape(x_lattice) = ', np.shape(x_lattice)
    # plt.plot(x_lattice, y_lattice,'b.')
    # for i in range(len(x_lattice)):
    #     plt.text(x_lattice.ravel()[i],y_lattice.ravel()[i],str(LVUC[i,0])+', '+str(LVUC[i,1]))
    # plt.show()

    # Trim to fit in box.
    mask = np.where(np.logical_and(x_lattice < image_shape[0] * 0.5,
                                   (x_lattice > -image_shape[0] * 0.5)))[0]
    mask2 = np.where(np.logical_and(y_lattice < image_shape[1] * 0.5,
                                    (y_lattice > -image_shape[1] * 0.5)))[0]
    mask = np.intersect1d(mask, mask2)

    # plt.plot(x_lattice, y_lattice,'b.')
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # plt.plot(x_lattice, y_lattice,'r.')
    # plt.show()

    # Trim LVUC to fit in box
    LVUC = LVUC[mask, :]
    LVUC = np.array([[int(LVUC[ii, 0]), int(LVUC[ii, 1]), 0] for ii in range(len(LVUC))])

    # Make output compatible with original version.
    out = np.zeros((len(x_lattice), 2), dtype=float)
    out[:, 1] = y_lattice
    out[:, 0] = x_lattice
    i = np.lexsort((out[:, 1], out[:, 0]))  # sort primarily by x, then y
    xy = out[i]
    LVUC = LVUC[i]

    # print 'LVUC = ', LVUC
    # print 'mask = ', mask
    # plt.plot(xy[:,0], xy[:,1],'b.')
    # for i in range(len(x_lattice)):
    #     plt.text(xy[i,0], xy[i,1], str(LVUC[i,0])+', '+str(LVUC[i,1]))
    # plt.show()

    # # Check the ordering
    # LV = np.array(lattice_vectors)
    # print 'LV = ', LV
    # sizes = np.arange(len(xy))+5
    # colorvals = np.linspace(0.1,1,len(xy))
    # plt.scatter(xy[:,0],xy[:,1], s=sizes+5, c=colorvals, cmap='afmhot')
    # xyLVtmp = np.array([LVUC[ii,0]*LV[0] + LVUC[ii,1]*LV[1] for ii in range(len(xy))])
    # plt.colorbar()
    # plt.figure()
    # print 'xyLVtmp = ', xyLVtmp
    # plt.scatter( xyLVtmp[:,0]- np.mean(xyLVtmp,axis=0)[0],\
    #              xyLVtmp[:,1] - np.mean(xyLVtmp,axis=0)[0]-0.1, s=sizes, c=colorvals, cmap='afmhot' )
    # plt.colorbar()
    # plt.show()

    return xy, LVUC


def remove_vertical_periodicity(BL, PVxydict):
    """Given a periodic lattice, remove the periodicity in BL of the lattive vectors that are vertical.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    PV : # periodic sides in output x 2 float array
        The periodic vectors matching each side that is periodic in the output to its opposing side

    Returns
    -------

    """
    eps = 1e-6
    # Identify periodic bonds as those for which BL < 0
    perbs = np.unique(np.where(BL < 0)[0])
    print 'perbs = ', perbs
    toremove = []
    for ind in perbs:
        bond = np.abs(BL[ind])
        if (bond[0], bond[1]) in PVxydict:
            pair = (bond[0], bond[1])
        elif (bond[1], bond[0]) in PVxydict:
            pair = (bond[1], bond[1])
        else:
            raise RuntimeError('Periodic bond has been identified from BL but is not in PVxydict')
        if np.abs(PVxydict[pair][1]) > eps:
            toremove.append(ind)

    keep = np.setdiff1d(np.arange(len(BL)), np.array(toremove))
    BL = BL[keep]
    return BL


def latticevec_filter(BL, xy, C, CBL, check=False):
    """Filter out bonds from BL based on whether the bonds match vecs defined by bonds in CBL (unit cell bond list).
    Must match mag and direction.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    C : array of dimension #particles in unit cell x 2
        unit cell positions (xy)
    CBL : array of dim #bonds in unit cell x 2
        bond list for unit cell

    Returns
    ----------
    BLout : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points (extraneous bonds removed)
    """
    # make Cvecs have both orientations for each possible bond in CBL
    Cvecs_pos = np.array([C[CBL[i, 1], :] - C[CBL[i, 0], :] for i in range(len(CBL))])
    Cvecs = np.vstack((Cvecs_pos, -Cvecs_pos))

    print 'np.shape(BL) = ', np.shape(BL)
    # if bond in BL does not match Cvec, then cut it
    vecs = np.array([xy[BL[i, 1], :] - xy[BL[i, 0], :] for i in range(len(BL))])

    vecmatch = np.zeros(len(BL), dtype=bool)
    for i in range(len(BL)):
        # Unfortunately, this doesn't seem to work well with floats
        # vecmatch[i] = any((Cvecs[:]==vecs[i]).all(1))
        vecmatch[i] = any((abs(Cvecs[:] - vecs[i]) < 1e-2).all(1))

    # vecmatch = any(vecmatchM,0)
    BLout = BL[vecmatch]

    # PLOT TO CHECK:
    if check:
        # check Cvecs (lattice)
        for i in range(len(CBL)):
            plt.plot(C[CBL[i, 0], 0] + np.array([0, Cvecs[i, 0]]), C[CBL[i, 0], 1] + np.array([0, Cvecs[i, 1]]), 'r-',
                     alpha=0.5)
        for i in range(len(BL)):
            plt.plot(xy[BL[i, 0], 0] + np.array([0, vecs[i, 0]]), xy[BL[i, 0], 1] + np.array([0, vecs[i, 1]]), 'b-',
                     alpha=0.5)
        plt.title('Checking Cvecs')
        plt.show()

        # check vecs (vectors only)
        for i in range(len(vecs)):
            plt.plot(np.array([0, vecs[i, 0]]), np.array([0, vecs[i, 1]]), 'b-', alpha=0.5)
        for i in range(len(Cvecs)):
            plt.plot(np.array([0, Cvecs[i, 0]]), np.array([0, Cvecs[i, 1]]), 'r-', alpha=0.5)
        plt.title('Checking vectors only')
        plt.show()

        # check outvecs
        outvecs = np.array([xy[BLout[i, 1], :] - xy[BLout[i, 0], :] for i in range(len(BLout))])
        for i in range(len(Cvecs)):
            plt.plot(np.array([0, Cvecs[i, 0]]), np.array([0, Cvecs[i, 1]]), 'r-', alpha=0.5)
        # check outvecs (vectors only)
        for i in range(len(outvecs)):
            plt.plot(np.array([0, outvecs[i, 0]]), np.array([0, outvecs[i, 1]]), 'g-', alpha=0.5)
        plt.title('Checking outvectors only')
        plt.show()

        # check output (lattice)
        for i in range(len(BLout)):
            plt.plot(np.array([xy[BLout[i, 0], 0], xy[BLout[i, 1], 0]]),
                     np.array([xy[BLout[i, 0], 1], xy[BLout[i, 1], 1]]), 'g-', alpha=0.5)
        plt.title('Checking output BL')
        plt.show()

    return BLout


def argcrop_lattice_to_polygon(shape, xy, check=False):
    """Returns mask to keep where lattice is inside polygon

    Parameters
    ----------
    shape : dict
        key is string describing shape, val is list (or tuple?) of floats
    xy : NP x 2 float array
        points in the lattice
    check : bool
        Whether to view intermediate results

    Returns
    ----------
    keep : bool 1D array
        True for inds of xy to keep from lattice
    """
    if 'circle' in shape:
        '''add masking to shape here'''
        NH, NV = shape['circle']
        # Modify below to allow ovals
        R = NH * 0.5
        keep = np.logical_and(np.abs(xy[:, 0]) < R * 1.000000001, np.abs(xy[:, 1]) < (2 * R * 1.0000001))
    elif 'hexagon' in shape:
        print 'cropping to: ', shape
        NH, NV = shape['hexagon']
        # Modify below to allow different values of NH and NV on the horiz and vertical sides of the hexagon
        a = NH + 0.5
        polygon = np.array([[-a * 0.5, -np.sqrt(a ** 2 - (0.5 * a) ** 2)],
                            [a * 0.5, -np.sqrt(a ** 2 - (0.5 * a) ** 2)], [a, 0.],
                            [a * 0.5, np.sqrt(a ** 2 - (0.5 * a) ** 2)], [-a * 0.5, np.sqrt(a ** 2 - (0.5 * a) ** 2)],
                            [-a, 0.],
                            [-a * 0.5, -np.sqrt(a ** 2 - (0.5 * a) ** 2)]])
        bpath = mplpath.Path(polygon)
        keep = bpath.contains_points(xy)

        # Check'
        if check:
            codes = [mplpath.Path.MOVETO,
                     mplpath.Path.LINETO,
                     mplpath.Path.LINETO,
                     mplpath.Path.LINETO,
                     mplpath.Path.LINETO,
                     mplpath.Path.LINETO,
                     mplpath.Path.CLOSEPOLY,
                     ]
            path = mplpath.Path(polygon, codes)
            ax = plt.gca()
            patch = mpatches.PathPatch(path, facecolor='orange', lw=2)
            ax.add_patch(patch)
            ax.plot(polygon[:, 0], polygon[:, 1], 'bo')
            ax.plot(xy[:, 0], xy[:, 1], 'r.')
            plt.show()
    elif 'square' in shape:
        NH, NV = shape['square']
        keep = np.logical_and(np.abs(xy[:, 0]) < NH * .5, np.abs(xy[:, 1]) < (NV * 0.5))
    elif 'polygon' in shape:
        bpath = mplpath.Path(shape['polygon'])
        keep = bpath.contains_points(xy)
    else:
        raise RuntimeError('Polygon dictionary not specified in generate_triangular_lattice().')
    # print 'keep = ', keep
    if check:
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        plt.plot(xy[keep, 0], xy[keep, 1], 'r.')
        plt.show()
    return keep


def mask_with_polygon(shape, NH, NV, xy, BL, eps=0.00, check=False):
    """Crop out points from a network and their connectivity, keeping only the pts inside a polygon.
    The polygon (shape) can be supplied as a string specifier or as a numpy array.

    Parameters
    ----------
    shape : str or 2D numpy float array
        string description of the overall geometry (boundary shape) of the lattice, or polygon to mask to.
    NH : int or float
        Magnitude of horizontal length scale in the polygon
        Note: would be first val in dict shape['square'], for example
    NV : int or float
        Magnitude of vertical length scale in the polygon
        Note: would be second val in dict shape['square'], for example
    xy : array of dimension nx3
        Equilibrium positions of all the points for the lattice
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    eps : float
        Additional buffer space by which to enlarge the automatic polygon, if shape is a string
    check : bool
        Whether to plot intermediate steps to check them

    Returns
    ----------
    xy : array of dimension nx3
        Equilibrium positions of all the points for the lattice
    NL : array of dimension n x (max number of neighbors)
        Each row corresponds to a point.  The entries tell the indices of the neighbors.
    KL : array of dimension n x (max number of neighbors)
        Correponds to NL matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    """
    if isinstance(shape, str):
        if shape == 'square':
            keep = np.logical_and(np.abs(xy[:, 0]) < NH * 0.5 + eps, np.abs(xy[:, 1]) < NV * 0.5 + eps)
            # print 'keep = ', keep
            if check:
                plt.plot(xy[:, 0], xy[:, 1], 'b.')
                plt.plot(xy[keep, 0], xy[keep, 1], 'ro')
                plt.show()
        elif shape == 'circle':
            # Modify below to allow ellipses
            keep = (xy[:, 0] ** 2 + xy[:, 1] ** 2 < (NH * 0.5 + eps) ** 2)
        elif shape == 'hexagon':
            tmp2 = xy
            print 'cropping to: ', shape
            # NH, NV = shape['hexagon']
            # Modify below to allow different values of NH and NV on the horiz and vertical sides of the hexagon
            a = NH + eps
            polygon = auto_polygon(shape, NH, NV, eps)
            # np.array([[-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)], \
            #     [a*0.5,-np.sqrt(a**2 - (0.5*a)**2)],[a,0.], \
            #     [a*0.5, np.sqrt(a**2 - (0.5*a)**2)],[-a*0.5,np.sqrt(a**2 - (0.5*a)**2)],[-a,0.],\
            #     [-a*0.5, -np.sqrt(a**2 - (0.5*a)**2)]])
            bpath = mplpath.Path(polygon)
            keep = bpath.contains_points(tmp2)
            if check:
                plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
                plt.plot(xy[:, 0], xy[:, 1], 'b.')
                plt.show()
    elif isinstance(shape, np.ndarray):
        polygon = shape
        bpath = mplpath.Path(polygon)
        keep = bpath.contains_points(xy)
    else:
        RuntimeError('Parameter shape in mask_with_polygon() must be either string (description) or numpy float array!')

    xy, NL, KL, BL, PVdout = le.remove_pts(keep, xy, BL, NN='min')
    return xy, NL, KL, BL


def auto_polygon(shape, NH, NV, eps=0.00):
    """Generate polygon based on shape in default manner

    Parameters
    ----------
    NH : int or float
        horizontal length scale of the polygon
    NV : int or float
        vertical length scale of the polygon
    eps : float
        dilation of the polygon in absolute distance units

    Returns
    ----------
    polygon : #vertices x 2 array
        2D polygon array
    """
    if shape == 'square':
        a = NH * 0.5 + eps
        b = NV * 0.5 + eps
        polygon = np.array([[-a, -b], [a, -b], [a, b], [-a, b]])

    elif shape == 'hexagon':
        h = np.sqrt(NV ** 2 - (NH * 0.5) ** 2)
        theta1 = 2. * np.arcsin(NH * 0.5 / NV)
        theta2 = np.arccos(NH * 0.5 / h)
        a = NH
        b = NV
        he = h + eps
        ae = NH + eps
        be = NV + eps
        polygon = np.array([[-ae * 0.5, -he],
                            [ae * 0.5, -he], [ae, 0.],
                            [ae * 0.5, he], [-ae * 0.5, he], [-ae, 0.],
                            [-ae * 0.5, -he]])
    elif shape == 'circle':
        tt = np.arange(0, 2 * np.pi, 0.03)
        if NH == NV:
            polygon = float(NH) * np.dstack((np.cos(tt), np.sin(tt)))[0]
        else:
            radius = float(NV) * float(NH) / np.sqrt(
                float(NH) ** 2 * np.sin(tt) ** 2 + float(NV) ** 2 * np.cos(tt) ** 2)
            polygon = radius * np.dstack((np.cos(tt), np.sin(tt)))[0]

    return polygon


def decorate_kagome_elements(xy_hex, BL, xypick, NL=None, PVxydict={}, viewmethod=False, check=False):
    """Decorate a honeycomb-like lattice to be a kagome. Works for periodic systems as well.
    If periodic and check==True, NL and PVxydict must be supplied. Supplying NL does NOT speed up calculation --> it is
    only for visualization purposes.

    Parameters
    ----------
    xy_hex : NP x 2 array
        Equilibrium lattice positions of honeycomb-like lattice
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of honeycomb-like lattice
    xypick: 1D int array
        Which vertices to kagomize
    viewmethod : bool
        Show additional details about the method of decoration
    check: bool
        Whether to display intermediate results

    Returns
    ----------
    pts : NP x 2 array
        Equilibrium lattice positions of kagome lattice
    BLout : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of kagome lattice
    """
    # Note: pts are indexed just as BL, since there is one halfway point for each bond

    xy = xy_hex

    # For each bond, place a pt at the center.
    pts = np.array([np.mean(xy[row], axis=0) for row in BL])

    # For each point chosen (in xypick), find three neighboring halfway points in pts
    inds2add = []
    BL2add_pts = []
    BL2add_xy = []
    BL2add_pt = []
    for vertex in xypick:
        # There are two cases.
        # Case 1: One vertex connected by the bond is decorated
        #               n
        #             / | \    n's mark new halfway points.
        #  ------.---n--. |     .'s mark xy points
        #        |    \  \|
        #        |       \\
        #                  n
        # Case 2: Both vertices connected by the bond are decorated
        #      n          n
        #       \---    / | \    n's mark new halfway points.
        #      | \.--\-n--. |     .'s mark xy points
        #       \|/     \  \|
        #        n        \ \
        #        |          n

        # Get indices of array 'pts' around this vertex ('bond indices --> binds')
        binds = np.where(BL == vertex)[0].tolist()

        if viewmethod:
            print 'displaying method: removing this vertex'
            le.display_lattice_2D(xy, BL, colorz=False, ptcolor='k', close=False)
            plt.plot(pts[binds, 0], pts[binds, 1], 'ko')
            keepxy = np.setdiff1d(np.arange(len(xy)), xypick)
            plt.plot(xy[keepxy, 0], xy[keepxy, 1], 'ro')
            plt.plot(xy[vertex, 0], xy[vertex, 1], 'b^')
            plt.show()

        # Add to xy_out. Note that there will be repeats in Case 2.
        inds2add.extend(binds)

        # Make bonds that will be added to BL_out
        # Add three bonds connecting the three halfway points
        if len(binds) == 3:
            BL2add_pts.extend([[binds[0], binds[1]], [binds[0], binds[2]], [binds[1], binds[2]]])
        elif len(binds) == 2:
            BL2add_pts.extend([[binds[0], binds[1]]])
        # For each bond, if case 1, add bond from halwaypt to xy point. If Case 2, move on.
        for ind_pts in binds:
            bond = BL[ind_pts]
            # To check for case 2, look for the index that isn't pt in indpick
            other_vertex = np.setdiff1d(bond, np.array([vertex]))[0]
            if other_vertex not in xypick:
                # Have two lists, one indexing each array (xy, pts). Then [BL2add_xy[i],BL2add_pt[i]] makes a new bond.
                # For this added halfway pt, add a bond from the neighboring xy point
                BL2add_pt.append(ind_pts)
                # BL2add_xy indexes the xy half of the bond
                BL2add_xy.append(other_vertex)

    # Translate indices of pts to indices of added halway points
    uinds2add = np.unique(np.array(inds2add)) + len(xy)

    # Add all of pts to xy
    xy_buffed = np.vstack((xy, pts))  # pts[uinds2add]))

    # Add to BL --> easiest to add all points to xy, so that we can add all bonds as we have already indexed them to BL
    BLadd = np.dstack((np.array(BL2add_xy), np.array(BL2add_pt) + len(xy)))[0]
    if not BL2add_pts:
        print 'no bonds to add!'
        BLadd = BLadd
    else:
        BLadd = np.vstack((BLadd, np.array(BL2add_pts) + len(xy)))
    BL_buffed = np.vstack((BL, BLadd)).astype(int)

    # Building new array from old array by removing unused portion of buffed array
    # keepxy = np.sort(ind_shuffled[Nkag:len(xy)])
    keepxy = np.setdiff1d(np.arange(len(xy)), xypick)
    keep = np.hstack((keepxy, uinds2add)).astype(int)
    # print 'keep = ', keep

    if check:
        print 'BL_buffed = ', BL_buffed
        le.display_lattice_2D(xy_buffed, BL_buffed, NL=NL, PVxydict=PVxydict, colorz=False, ptcolor='k', close=False,
                              title='All added points and bonds, plus old ones')
        plt.plot(xy_buffed[keepxy, 0], xy_buffed[keepxy, 1], 'ro')
        plt.plot(xy_buffed[keep, 0], xy_buffed[keep, 1], 'bo')
        for ii in range(len(xy_buffed)):
            plt.text(xy_buffed[ii, 0] + 0.05, xy_buffed[ii, 1], str(ii))
        plt.show()

    BLbuff = np.sort(BL_buffed, axis=1)
    BLbuff = dh.unique_rows(BLbuff)

    # Can only check the intermediate steps here if the lattice is not periodic, since remove_pts doesn't take the extra
    # arguments needed for periodic system visualization
    check_rmpts = check and len(PVxydict) == 0
    xyout, trash0, trash1, BL_out, PVd_out = le.remove_pts(keep, xy_buffed, BLbuff, check=check_rmpts)

    if check:
        plt.plot(xyout[:, 0], xyout[:, 1], 'b.')
        for ii in range(len(xyout)):
            plt.text(xyout[ii, 0] + 0.05, xyout[ii, 1], str(ii))
        plt.show()

    BLout = np.sort(BL_out, axis=1)
    BLout = dh.unique_rows(BLout)

    return xyout, BLout


def decorate_bondneighbors_elements_periodic(xy_hex, BL, xypick, NL, KL, PVxydict, PV,
                                             PVx=None, PVy=None, viewmethod=False, check=False):
    """Decorate a honeycomb-like lattice, removing vertex points xypick, replacing those bonds by bonding its neighbors
    together.
    DOES NOT Workfor periodic systems YET.
    If periodic and check==True, NL and PVxydict must be supplied. Supplying NL does NOT speed up calculation --> it is
    only for visualization purposes.

    Parameters
    ----------
    xy_hex : NP x 2 array
        Equilibrium lattice positions of honeycomb-like lattice
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of honeycomb-like lattice
    xypick: 1D int array
        Which vertices to remove, replace by bonding its neighbors
    NL : NP x max#nn int array
    KL : NP x max#nn int array
    viewmethod : bool
        Show additional details about the method of decoration
    check: bool
        Whether to display intermediate results

    Returns
    ----------
    pts : NP x 2 array
        Equilibrium lattice positions of kagome lattice
    BLout : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of kagome lattice
    """
    # Note: pts are indexed just as BL, since there is one halfway point for each bond
    xy = xy_hex
    if NL is None or KL is None:
        NL, KL = le.BL2NLandKL(BL)

    # check initial input
    if check:
        import lepm.plotting.network_visualization as netvis
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)
        netvis.movie_plot_2D(xy, BL, NL=NL, KL=KL, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                             bs=np.random.rand(len(BL)) - 0.5, show=False, colorz=False, ptcolor='k')
        ax = plt.gca()
        for ii in range(len(xy)):
            ax.text(xy[ii, 0] + 0.001, xy[ii, 1] + 0.002, str(ii))
        plt.title('network input for decorate_bondneighbors_elements_periodic()')
        plt.show()

    # NLout = np.zeros((np.shape(NL)[0], np.shape(NL)[1] * 2))
    # KLout = np.zeros_like(NLout)
    # NLout[:, 0:np.shape(NL)[1]] = NL
    # KLout[:, 0:np.shape(NL)[1]] = KL

    eps = 1e-8
    # link up neighbors of each xypick if not already linked
    bladd = []
    BLbig = copy.deepcopy(BL)
    if (BL < 0).any():
        for pt in xypick:
            # get neighbors of pt
            nbrs = NL[pt, np.where(np.abs(KL[pt, :]))[0]]
            for kk in range(len(nbrs)):
                mask = np.ones_like(nbrs, dtype=bool)
                mask[kk] = False
                nbr = nbrs[kk]
                others = nbrs[mask]
                # here we must be careful to allow dual bonds (for ex, a bulk bond and
                # periodic bond both between i and j)
                if len(set(nbrs.tolist()) - set([nbr])) == 0:
                    raise RuntimeError('Have not allowed for case where particle must hook up to itself')
                print 'NL[pt] = ', NL[pt]
                print 'KL[pt] = ', KL[pt]
                for other in others:
                    if other < nbr:
                        new0, new1 = other, nbr
                    else:
                        new0, new1 = nbr, other
                    if KL[pt, kk] < -eps:
                        new_row = [-new0, -new1]
                    elif KL[pt, np.where(NL[pt, :] == other)[0]] < -eps:
                        new_row = [-new0, -new1]
                    else:
                        new_row = [new0, new1]

                    if not le.row_is_in_array(np.array(new_row), BLbig):
                        print 'adding new_row = ', new_row
                        print 'bladd = ', bladd
                        print 'PVxydict = ', PVxydict
                        bladd.append(new_row)
                        BLbig = np.vstack((BLbig, np.array(new_row)))
                        if KL[pt, kk] < -eps or KL[pt, np.where(NL[pt, :] == other)[0]] < -eps:
                            # figure out whether this new bond is periodic
                            if (pt, new1) in PVxydict:
                                pva = PVxydict[(pt, new1)]
                            elif (new1, pt) in PVxydict:
                                pva = -PVxydict[(new1, pt)]
                            else:
                                pva = np.array([0., 0.])

                            # now establish the periodic vector from pt to new0
                            if (pt, new0) in PVxydict:
                                pvb = PVxydict[(pt, new0)]
                            elif (new0, pt) in PVxydict:
                                pvb = -PVxydict[(new0, pt)]
                            else:
                                pvb = np.array([0., 0.])

                            print 'pva = ', pva
                            print 'pvb = ', pvb
                            pvadd = pva - pvb
                            # Check if pva and pvb cancel. If not, add to PVxydict
                            if (np.abs(pvadd) > eps).any():
                                # the periodic vectors do not cancel. Add this periodic vector to the dict
                                if (new0, new1) in PVxydict:
                                    PVxydict[(new0, new1)] = np.vstack((PVxydict[(new0, new1)], pvadd))
                                else:
                                    PVxydict[(new0, new1)] = pvadd
                                    print 'adding key to PVxydict: ', PVxydict
    else:
        raise RuntimeError('Network is not periodic, use decorate_bondneighbors_elements() instead.')

    ####################################################################################
    # check it
    if check:
        print 'BLout = ', BLbig
        import lepm.plotting.network_visualization as netvis
        NL, KL = le.BL2NLandKL(BLbig)
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)
        netvis.movie_plot_2D(xy, BLbig, NL=NL, KL=KL, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                             bs=np.random.rand(len(BLbig)) - 0.5, show=False, colorz=False, ptcolor='k')
        ax = plt.gca()
        for ii in range(len(xy)):
            ax.text(xy[ii, 0] + 0.001, xy[ii, 1] + 0.002, str(ii))
        plt.show()
    ####################################################################################

    # remove each particle in xypick
    keep = np.setdiff1d(np.arange(len(xy_hex)), xypick)
    print 'before: PVxydict = ', PVxydict
    xyout, NL, KL, BLout, PVxydict = le.remove_pts(keep, xy, BLbig, PVxydict=PVxydict)
    # PVxydict = le.BL2PVxydict(BLout, xyout, PV)
    print 'after: PVxydict = ', PVxydict
    PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)

    ####################################################################################
    if check:
        print 'BLout = ', BLout
        import lepm.plotting.network_visualization as netvis
        PVx, PVy = le.PVxydict2PVxPVy(PVxydict, NL, KL)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'PVxydict = ', PVxydict
        netvis.movie_plot_2D(xyout, BLout, NL=NL, KL=KL, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                             bs=np.random.rand(len(BLout)), show=False, colorz=False, ptcolor='k')
        ax = plt.gca()
        for ii in range(len(xyout)):
            ax.text(xyout[ii, 0] - 0.001, xyout[ii, 1] + 0.002, str(ii))
        plt.show()
    ####################################################################################

    return xyout, BLout, NL, KL, PVx, PVy, PVxydict


def decorate_bondneighbors_elements(xy_hex, BL, xypick, NL=None, KL=None, PVxydict={}, viewmethod=False, check=False):
    """Decorate a honeycomb-like lattice, removing vertex points xypick, replacing those bonds by bonding its neighbors
    together.
    DOES NOT Workfor periodic systems YET.
    If periodic and check==True, NL and PVxydict must be supplied. Supplying NL does NOT speed up calculation --> it is
    only for visualization purposes.

    Parameters
    ----------
    xy_hex : NP x 2 array
        Equilibrium lattice positions of honeycomb-like lattice
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of honeycomb-like lattice
    xypick: 1D int array
        Which vertices to remove, replace by bonding its neighbors
    NL : NP x max#nn int array
    KL : NP x max#nn int array
    PVxydict : dict
    viewmethod : bool
        Show additional details about the method of decoration
    check: bool
        Whether to display intermediate results

    Returns
    ----------
    pts : NP x 2 array
        Equilibrium lattice positions of kagome lattice
    BLout : #bonds x 2 int array
        Each row is a bond and contains indices of connected points, of kagome lattice
    """
    # Note: pts are indexed just as BL, since there is one halfway point for each bond
    xy = xy_hex
    if NL is None or KL is None:
        NL, KL = le.BL2NLandKL(BL)

    # link up neighbors of each xypick if not already linked
    bladd = []

    if (BL < 0).any():
        raise RuntimeError('Lattice is periodic: use decorate_bondneighbors_elements_periodic() instead.')
    else:
        for pt in xypick:
            # get neighbors of pt
            nbrs = NL[pt, np.where(KL[pt, :])[0]]
            for nbr in nbrs:
                for other in set(nbrs.tolist()) - set([nbr]):
                    if not le.row_is_in_array(np.array([nbr, other]), BL):
                        if not le.row_is_in_array(np.array([other, nbr]), BL):
                            if other < nbr:
                                bladd.append([other, nbr])
                            else:
                                bladd.append([nbr, other])

    print 'np.array(bladd) = ', np.array(bladd)
    BLbig = np.vstack((BL, np.array(bladd)))

    # remove each particle in xypick
    keep = np.setdiff1d(np.arange(len(xy_hex)), xypick)
    xyout, trash0, trash1, BLout, PVxydict = le.remove_pts(keep, xy, BLbig, PVxydict=PVxydict)

    if check:
        print 'BLout = ', BLout
        import lepm.plotting.network_visualization as netvis
        netvis.movie_plot_2D(xyout, BLout, bs=np.random.rand(len(BLout)), PVxydict=PVxydict,
                             show=False, colorz=False, ptcolor='k')
        ax = plt.gca()
        for ii in range(len(xyout)):
            ax.text(xyout[ii, 0] - 0.001, xyout[ii, 1] + 0.002, str(ii))
        plt.show()

    return xyout, BLout


def kagomecentroid_lattice_from_pts(xy, polygon=None, trimbound=True, thres=2.0, check=False):
    """Decorate a point set by kagomizing a voronoi tesselation of the points.

    Parameters
    ----------
    xy: NP x 2 float array
    polygon: #veritces x 2 float array or None
        If None, does not crop the points to a polygon
    trimbound: bool
    thres: float
    check: bool

    Returns
    -------
    xy, NL, KL, BL
    """
    xycent, NL, KL, BL = le.delaunay_centroid_lattice_from_pts(xy, polygon=None, trimbound=trimbound, thres=thres,
                                                               check=check)

    # Decorate lattice as kagome
    print('Decorating lattice as kagome...')
    xy, BL, PVxydict = decorate_as_kagome(xycent, BL)

    if polygon is None or polygon == 'auto':
        NL, KL = le.BL2NLandKL(BL, NP=len(xy), NN='min')
    else:
        pth = mplpath.Path(polygon, closed=False)
        keep = pth.contains_points(xy)
        # print 'keep = ', keep
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
            plt.show()
        xy, NL, KL, BL, PVxydict = le.remove_pts(keep, xy, BL, NN=4)

    return xy, NL, KL, BL


def kagomecentroid_periodic_network_from_pts(xy, LL, BBox='auto', check=False):
    """Convert 2D pt set to lattice (xy, NL, KL, BL, BM, PVxydict) via traingulation, handling periodic BCs.
    Performs:
    A) Triangulate an enlarged version of the point set.
    B) Find centroids
    C) Kagomize centroid lattice (so that z=4)
    D) Crops to original bounding box and connects periodic BCs
    Note: Normally BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = le.buffer_points_for_rectangular_periodicBC(xy, LL)
    xy, NL, KL, BL = kagomecentroid_lattice_from_pts(xytmp, polygon=None, trimbound=False, check=check)
    xytrim, NL, KL, BLtrim, PVxydict = le.buffered_pts_to_periodic_network(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def kagomevoronoi_network_from_pts(xy, polygon=None, kill_outliers=True, check=False):
    """Decorate a point set by kagomizing a Wigner-Seitz voronoi tesselation of the points.

    Parameters
    ----------
    xy: NP x 2 float array
    polygon: #veritces x 2 float array or None
        If None, does not crop the points to a polygon
    trimbound: bool
    thres: float
    check: bool

    Returns
    -------
    xy, NL, KL, BL
    """
    xycent, NL, KL, BL = le.voronoi_lattice_from_pts(xy, polygon=polygon, NN=3, kill_outliers=kill_outliers,
                                                     check=check)

    # Decorate lattice as kagome
    print('Decorating lattice as kagome...')
    xy, BL, PVxydict = decorate_as_kagome(xycent, BL)

    if polygon is None or polygon == 'auto':
        NL, KL = le.BL2NLandKL(BL, NP=len(xy), NN='min')
    else:
        pth = mplpath.Path(polygon, closed=False)
        keep = pth.contains_points(xy)
        # print 'keep = ', keep
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
            plt.show()
        xy, NL, KL, BL, PVxydict = le.remove_pts(keep, xy, BL, NN=4)

    return xy, NL, KL, BL


def kagomevoronoi_periodic_network_from_pts(xy, LL, BBox='auto', check=False):
    """Convert 2D pt set to lattice (xy, NL, KL, BL, BM, PVxydict) via traingulation, handling periodic BCs.
    Performs:
    A) Triangulate an enlarged version of the point set.
    B) Find centroids
    C) Kagomize centroid lattice (so that z=4)
    D) Crops to original bounding box and connects periodic BCs
    Note: Normally BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = le.buffer_points_for_rectangular_periodicBC(xy, LL)
    xy, NL, KL, BL = kagomevoronoi_network_from_pts(xytmp, polygon=None, trimbound=False, check=check)
    xytrim, NL, KL, BLtrim, PVxydict = le.buffered_pts_to_periodic_network(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def accordionize_network(xy, BL, lp):
    """Add lines or zig-zags of gyroscopes between every vertex, with specified angle. Number of gyros between vertices
    specified by lp['intparam'].

    Parameters
    ----------
    xy
    BL
    lp
    PV

    Returns
    -------

    """
    # accordionize the lattice
    nnew = lp['intparam']
    xyvertices = copy.deepcopy(xy)
    ptind = len(xy)
    blnew = np.zeros((len(BL) * (nnew + 2), 2), dtype=int)
    xynew = np.zeros((len(xy) + len(BL) * nnew, 2), dtype=float)
    xynew[0:len(xy)] = xy

    if lp['check']:
        plt.plot(xyvertices[:, 0], xyvertices[:, 1], 'b.')
        plt.title('Input to blf.accordionize_network()')
        plt.show()
        netvis.movie_plot_2D(xyvertices, BL, show=True, title='Input to blf.accordionize_network()', bondcolor='k',
                             colorz='b', axcb=None)

    # Handle periodic case separately
    if not (BL < 0).any():
        dists = dh.dist_pts(xy, xy, dim=-1, square_norm=False)
        distx = dh.dist_pts(xy, xy, dim=0, square_norm=False)
        disty = dh.dist_pts(xy, xy, dim=1, square_norm=False)
        # ii is the index of the row of blnew that we are creating
        ii = 0
        dmyi = 0
        if lp['eta_alph'] > 0:
            sgn = np.sign(np.random.rand(1) < lp['eta_alph'])
        else:
            sgn = np.sign(1. - lp['alph']) * np.ones(len(BL))
        for row in BL:
            # Add the points displaced by some amount from the original bond
            # ll is path length from old site to new site
            ll = dists[row[0], row[1]]
            # ss is the path length between intermediate particles projected along the original bond
            ss = ll / float(nnew)
            hh = (ss * 0.5) / abs(np.tan(lp['alph'] * 0.5 * np.pi)) * sgn[dmyi]
            # define vector taking row[0] to row[1]
            vec = np.array([distx[row[0], row[1]], disty[row[0], row[1]]])
            vechat = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # get unit vector normal to vec --> just swap x,y and negate new x component to go clockwise
            normal = np.array([vec[1], -vec[0]]) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # add first point along vec
            xynew[ptind] = xy[row[0]] + (ss * 0.5) * vechat + hh * normal

            # Create the new lineseg from the original point to the first added point
            blnew[ii, :] = np.array([row[0], ptind])
            ii += 1
            # connect the middle linesegs between added pts
            for jj in range(nnew - 1):
                # add bonds
                blnew[ii, :] = np.array([ptind, ptind + 1])
                ptind += 1

                # add points
                xynew[ptind] = xy[row[0]] + vechat * (ss * (jj + 1.5)) + hh * normal * (-1) ** (jj + 1)
                ii += 1

            # connect the last lineseg
            blnew[ii, :] = np.array([ptind, row[1]])
            ii += 1
            ptind += 1

        blnew = blnew[0:ii]
        xynew = xynew[0:ptind]
        if lp['check']:
            plt.plot(xynew[:, 0], xynew[:, 1], 'r.')
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            print 'blnew = ', blnew
            for ii in range(len(xynew)):
                plt.text(xynew[ii, 0] - 0.2, xynew[ii, 1] + 0.2, str(ii))
            for row in blnew:
                plt.plot([xynew[row[0], 0], xynew[row[1], 0]], [xynew[row[0], 1], xynew[row[1], 1]], 'k-')
            plt.show()
            netvis.movie_plot_2D(xynew, blnew, show=True, bondcolor='k', ptcolor='b', axcb=None)
    else:
        # network is periodic
        raise RuntimeError('Network is periodic -- use build_lattice_functions.accordionize_network_periodic()')

    LVUC = None
    UC = None
    BL = blnew
    xy = xynew
    lattice_exten_add = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
    return xy, BL, LVUC, UC, xyvertices, lattice_exten_add


# def accordionize_network_periodic(xy, BL, NL, KL, lp, PVx, PVy, PVxydict=None):
#     """Add lines or zig-zags of gyroscopes between every vertex, with specified angle, for periodic network.
#     Number of gyros between vertices is specified by lp['intparam'].
#
#     Parameters
#     ----------
#     xy
#     BL
#     lp
#     PV
#
#     Returns
#     -------
#
#     """
#     # accordionize the lattice
#     nnew = lp['intparam']
#     xyvertices = copy.deepcopy(xy)
#     ptind = len(xy)
#     blnew = np.zeros((len(BL) * (nnew + 2), 2), dtype=int)
#     xynew = np.zeros((len(xy) + len(BL) * nnew, 2), dtype=float)
#     xynew[0:len(xy)] = xy
#
#     if lp['check']:
#         plt.plot(xyvertices[:, 0], xyvertices[:, 1], 'b.')
#         plt.title('Input to blf.accordionize_network()')
#         plt.show()
#         netvis.movie_plot_2D(xyvertices, BL, PVxydict=PVxydict, show=True,
#                              title='Input to blf.accordionize_network()', bondcolor='k', colorz='b', axcb=None)
#
#     # Handle periodic case separately
#     if not (BL < 0).any():
#         raise RuntimeError('Network is not periodic -- use accordionize_network() instead')
#     else:
#         # network is periodic
#         pvdict_out, pvdcount = {}, {}
#         print 'blf: PVxydict = ', PVxydict
#         print 'blf: BL = ', BL
#         BLabs = np.abs(BL)
#         KLabs = np.abs(KL)
#         BMx, BMy = le.NL2BMxy(xy, NL, KL, PVx=PVx, PVy=PVy)
#         BM = le.NL2BM(xy, NL, KL, PVx=PVx, PVy=PVy)
#         # ii is the index of the row of blnew that we are creating
#         ii = 0
#         if lp['eta_alph'] > 0:
#             sgn = np.sign(np.random.rand(len(BL)) - lp['eta_alph'])
#         else:
#             sgn = np.sign(1. - lp['alph']) * np.ones(len(BL))
#         for dmyi in np.shape(NL)[0]:
#             jjs = NL[dmyi, np.where(KLabs[dmyi] > 0)[0]]
#             pairs = [[dmyi, jj] for jj in jjs]
#             for pair in pairs:
#                 # Only add particles to the bond if the bond
#                 if pair[0] < pair[1]:
#                     # Add the points displaced by some amount from the original bond
#                     # ll is path length from old site to new site
#                     ll = BM[row[0], row[1]]
#                     # ss is the path length between intermediate particles projected along the original bond
#                     ss = ll / float(nnew)
#                     hh = (ss * 0.5) / abs(np.tan(lp['alph'] * 0.5 * np.pi)) * sgn[dmyi]
#                 # define vector taking row[0] to row[1]
#                 vec = np.array([distx[row[0], row[1]], disty[row[0], row[1]]])
#                 vechat = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
#                 # get unit vector normal to vec --> just swap x,y and negate new x component to go clockwise
#                 normal = np.array([vec[1], -vec[0]]) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
#                 # add first point along vec
#                 xynew[ptind] = xy[row[0]] + (ss * 0.5) * vechat + hh * normal
#
#                 # Create the new lineseg from the original point to the first added point
#                 blnew[ii, :] = np.array([row[0], ptind])
#                 ii += 1
#                 # connect the middle linesegs between added pts
#                 for jj in range(nnew - 1):
#                     # add bonds
#                     blnew[ii, :] = np.array([ptind, ptind + 1])
#                     ptind += 1
#
#                     # add points
#                     xynew[ptind] = xy[row[0]] + vechat * (ss * (jj + 1.5)) + hh * normal * (-1) ** (jj + 1)
#                     ii += 1
#
#                 # connect the last lineseg
#                 if (BL[dmyi] < 0).any():
#                     print 'blf: BL[dmyi] = ', BL[dmyi]
#                     # get periodic vector from PVxydict
#                     if tuple(row) in PVxydict:
#                         pvxy = PVxydict[tuple(row)]
#                         key = tuple(row)
#                     elif (row[1], row[0]) in PVxydict:
#                         pvxy = -PVxydict[(row[1], row[0])]
#                         key = (row[1], row[0])
#                     else:
#                         print 'row = ', row
#                         print 'PVxydict = ', PVxydict
#                         raise RuntimeError('Could not find bond pair in PVxydict')
#
#                     # make sure we just have a single vector, not multiple
#                     if len(np.shape(pvxy)) > 1:
#                         # we have multiple vectors, use just the first one we haven't already used
#                         if key in pvdcount:
#                             pvdcount[key] += 1
#                         else:
#                             pvdcount[key] = 0
#
#                         print 'blf: key = ', key
#                         print 'blf: determined pvdcount = ', pvdcount[key]
#                         pvxy = pvxy[pvdcount[key]]
#                         print 'blf: determined pvxy = ', pvxy
#
#                     # take care not to overwrite an entry for the bond if it already exists
#                     if (ptind, row[1]) in pvdict_out:
#                         pvdict_out[(ptind, row[1])] = np.vstack((pvdict_out[(ptind, row[1])], pvxy))
#                         print 'pvdict = ', pvdict_out
#                         sys.exit()
#                         pvdcount[(ptind, row[1])]
#                     else:
#                         # define element of dictionary for the first time
#                         pvdict_out[(ptind, row[1])] = pvxy
#                     blnew[ii, :] = np.array([-ptind, -row[1]])
#                 else:
#                     blnew[ii, :] = np.array([ptind, row[1]])
#                 ii += 1
#                 ptind += 1
#                 dmyi += 1
#
#             blnew = blnew[0:ii]
#             xynew = xynew[0:ptind]
#             if lp['check']:
#                 plt.plot(xynew[:, 0], xynew[:, 1], 'r.')
#                 plt.plot(xy[:, 0], xy[:, 1], 'b.')
#                 print 'blnew = ', blnew
#                 for ii in range(len(xynew)):
#                     plt.text(xynew[ii, 0] - 0.2, xynew[ii, 1] + 0.2, str(ii))
#                 for row in blnew:
#                     plt.plot([xynew[row[0], 0], xynew[row[1], 0]], [xynew[row[0], 1], xynew[row[1], 1]], 'k-')
#                 plt.show()
#                 netvis.movie_plot_2D(xynew, blnew, PVxydict=pvdict_out, show=True,
#                                      bondcolor='k', ptcolor='b', axcb=None)
#
#     LVUC = None
#     UC = None
#     BL = blnew
#     xy = xynew
#     lattice_exten_add = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
#     # Convert nlnew and klnew into NL, KL
#     # first get max length of nlnew element
#     maxnnn = np.max(np.array([len(neis) for neis in nlnew]))
#     NL = np.zeros(len(xy), maxnnn, dtype=int)
#     KL = np.zeros(len(xy), maxnnn, dtype=int)
#     kk = 0
#     for neis in nlnew:
#         NL[kk, 0:len(neis)] = neis
#         KL[kk, 0:len(neis)] = klnew[kk]
#         kk += 1
#
#     print 'blf: NL = ', NL
#     print 'blf: KL = ', KL
#     plt.plot(xy[:, 0], xy[:, 1], 'b.')
#     for ind in range(len(xy)):
#         plt.text(xy[ind, 0] + 0.03 * ind, xy[ind, 1], str(ind))
#
#     for row in BL:
#         if (row < 0).any():
#             plt.plot(xy[row, 0], xy[row, 1], 'k--')
#         else:
#             plt.plot(xy[row, 0], xy[row, 1], 'k-')
#
#     plt.show()
#     sys.exit()
#     PVx, PVy = le.PVxydict2PVxPVy(pvdict_out, NL, KL)
#     return xy, BL, pvdict_out, PVx, PVy, LVUC, UC, xyvertices, lattice_exten_add


# Old version relied on having no bonds such that i connects to j through 2 periodic bonds or through
#   both 1 bulk bond and 1+ periodic bond.
def accordionize_network_periodic(xy, BL, NL, KL, lp, PVxydict=None, PVx=None, PVy=None, PV=None):
    """Add lines or zig-zags of gyroscopes between every vertex, with specified angle, for periodic network.
    Number of gyros between vertices is specified by lp['intparam'].

    Parameters
    ----------
    xy
    BL
    lp
    PV

    Returns
    -------

    """
    # accordionize the lattice
    nnew = lp['intparam']
    xyvertices = copy.deepcopy(xy)
    ptind = len(xy)
    blnew = np.zeros((len(BL) * (nnew + 2), 2), dtype=int)
    xynew = np.zeros((len(xy) + len(BL) * nnew, 2), dtype=float)
    xynew[0:len(xy)] = xy

    if lp['check']:
        plt.plot(xyvertices[:, 0], xyvertices[:, 1], 'b.')
        plt.title('Input to blf.accordionize_network()')
        plt.show()
        netvis.movie_plot_2D(xyvertices, BL, PVxydict=PVxydict, show=True,
                             title='Input to blf.accordionize_network()', bondcolor='k', colorz='b', axcb=None)

    # Handle periodic case separately
    if not (BL < 0).any():
        raise RuntimeError('Network is not periodic -- use accordionize_network() instead')
    else:
        # network is periodic
        pvdict_out, pvdcount = {}, {}
        print 'blf: PVxydict = ', PVxydict
        print 'blf: BL = ', BL
        BLabs = np.abs(BL)

        # Get distances in x and y from i to j for each bond
        # dists = dh.dist_pts_periodic(xy, xy, PV, dim=-1, square_norm=False)
        # distx = -dh.dist_pts_periodic(xy, xy, PV, dim=0, square_norm=False)
        # disty = -dh.dist_pts_periodic(xy, xy, PV, dim=1, square_norm=False)
        dists = le.bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
        BMx, BMy = le.NL2BMxy(xy, NL, KL, PVx=PVx, PVy=PVy)
        distx = le.BM2bL(NL, BMx, BL)
        disty = le.BM2bL(NL, BMy, BL)
        print 'blf: dists = ', dists
        print 'blf: distx = ', distx
        print 'blf: disty = ', disty

        # ii is the index of the row of blnew that we are creating
        ii, dmyi = 0, 0
        if lp['eta_alph'] > 0:
            sgn = np.sign(np.random.rand(1) < lp['eta_alph'])
        else:
            sgn = np.sign(1. - lp['alph']) * np.ones(len(BL))
        for row in BLabs:
            # Add the points displaced by some amount from the original bond
            # ll is path length from old site to new site
            ll = dists[dmyi]
            # ss is the path length between intermediate particles projected along the original bond
            ss = ll / float(nnew)
            hh = (ss * 0.5) / abs(np.tan(lp['alph'] * 0.5 * np.pi)) * sgn[dmyi]
            # define vector taking row[0] to row[1]
            vec = np.array([distx[dmyi], disty[dmyi]])
            vechat = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # get unit vector normal to vec --> just swap x,y and negate new x component to go clockwise
            normal = np.array([vec[1], -vec[0]]) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            # add first point along vec
            xynew[ptind] = xy[row[0]] + (ss * 0.5) * vechat + hh * normal

            # Create the new lineseg from the original point to the first added point
            blnew[ii, :] = np.array([row[0], ptind])
            ii += 1
            # connect the middle linesegs between added pts
            for jj in range(nnew - 1):
                # add bonds
                blnew[ii, :] = np.array([ptind, ptind + 1])
                ptind += 1

                # add points
                xynew[ptind] = xy[row[0]] + vechat * (ss * (jj + 1.5)) + hh * normal * (-1) ** (jj + 1)
                ii += 1

            # connect the last lineseg
            if (BL[dmyi] < 0).any():
                print 'blf: BL[dmyi] = ', BL[dmyi]
                # get periodic vector from PVxydict
                if tuple(row) in PVxydict:
                    pvxy = PVxydict[tuple(row)]
                    key = tuple(row)
                elif (row[1], row[0]) in PVxydict:
                    pvxy = -PVxydict[(row[1], row[0])]
                    key = (row[1], row[0])
                else:
                    print 'row = ', row
                    print 'PVxydict = ', PVxydict
                    raise RuntimeError('Could not find bond pair in PVxydict')

                # make sure we just have a single vector, not multiple
                if len(np.shape(pvxy)) > 1:
                    # we have multiple vectors, use just the first one we haven't already used
                    if key in pvdcount:
                        pvdcount[key] += 1
                    else:
                        pvdcount[key] = 0

                    print 'blf: key = ', key
                    print 'blf: determined pvdcount = ', pvdcount[key]
                    pvxy = pvxy[pvdcount[key]]
                    print 'blf: determined pvxy = ', pvxy

                # take care not to overwrite an entry for the bond if it already exists
                if (ptind, row[1]) in pvdict_out:
                    pvdict_out[(ptind, row[1])] = np.vstack((pvdict_out[(ptind, row[1])], pvxy))
                    print 'pvdict = ', pvdict_out
                    sys.exit()
                    pvdcount[(ptind, row[1])]
                else:
                    # define element of dictionary for the first time
                    pvdict_out[(ptind, row[1])] = pvxy
                blnew[ii, :] = np.array([-ptind, -row[1]])
            else:
                blnew[ii, :] = np.array([ptind, row[1]])
            ii += 1
            ptind += 1
            dmyi += 1

        blnew = blnew[0:ii]
        xynew = xynew[0:ptind]
        if lp['check']:
            plt.plot(xynew[:, 0], xynew[:, 1], 'r.')
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            print 'blnew = ', blnew
            for ii in range(len(xynew)):
                plt.text(xynew[ii, 0] - 0.2, xynew[ii, 1] + 0.2, str(ii))
            for row in blnew:
                plt.plot([xynew[row[0], 0], xynew[row[1], 0]], [xynew[row[0], 1], xynew[row[1], 1]], 'k-')
            plt.show()
            netvis.movie_plot_2D(xynew, blnew, PVxydict=pvdict_out, show=True, bondcolor='k', ptcolor='b', axcb=None)

    LVUC = None
    UC = None
    BL = blnew
    xy0 = copy.deepcopy(xy)
    xy = xynew
    lattice_exten_add = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
    NL, KL = le.BL2NLandKL(BL)
    PVx, PVy = le.PVxydict2PVxPVy(pvdict_out, NL, KL)

    # Check it
    if lp['check']:
        print 'blf: NL = ', NL
        print 'blf: KL = ', KL
        plt.plot(xy0[:, 0], xy0[:, 1], 'ro')
        plt.plot(xynew[:, 0], xynew[:, 1], 'b.')
        for ind in range(len(xy)):
            plt.text(xy[ind, 0] + 0.01 * ind, xy[ind, 1], str(ind))

        for row in BL:
            if (row < 0).any():
                plt.plot(xy[row, 0], xy[row, 1], 'k--')
            else:
                plt.plot(xy[row, 0], xy[row, 1], 'k-')
        plt.title('Checking pre-output: periodic bonds are shown as finite.')
        plt.show()
        netvis.movie_plot_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, show=True, colormap='jet',
                             title='Checking output of blf.accordionize_network_periodic()')
    # sys.exit()

    return xy, BL, NL, KL, pvdict_out, PVx, PVy, LVUC, UC, xyvertices, lattice_exten_add


# Old version, works except on lattices with dual bonds
# def accordionize_network_periodic(xy, BL, lp, PVxydict=None, PVx=None, PVy=None, PV=None):
#     """Add lines or zig-zags of gyroscopes between every vertex, with specified angle, for periodic network.
#     Number of gyros between vertices is specified by lp['intparam'].
#
#     Parameters
#     ----------
#     xy
#     BL
#     lp
#     PV
#
#     Returns
#     -------
#
#     """
#     # accordionize the lattice
#     nnew = lp['intparam']
#     xyvertices = copy.deepcopy(xy)
#     ptind = len(xy)
#     blnew = np.zeros((len(BL) * (nnew + 2), 2), dtype=int)
#     xynew = np.zeros((len(xy) + len(BL) * nnew, 2), dtype=float)
#     xynew[0:len(xy)] = xy
#
#     if lp['check']:
#         plt.plot(xyvertices[:, 0], xyvertices[:, 1], 'b.')
#         plt.title('Input to blf.accordionize_network()')
#         plt.show()
#         netvis.movie_plot_2D(xyvertices, BL, PVxydict=PVxydict, show=True,
#                              title='Input to blf.accordionize_network()', bondcolor='k', colorz='b', axcb=None)
#
#     # Handle periodic case separately
#     if not (BL < 0).any():
#         raise RuntimeError('Network is not periodic -- use accordionize_network() instead')
#     else:
#         # network is periodic
#         pvdict_out, pvdcount = {}, {}
#         print 'blf: PVxydict = ', PVxydict
#         print 'blf: BL = ', BL
#         dists = dh.dist_pts_periodic(xy, xy, PV, dim=-1, square_norm=False)
#         BLabs = np.abs(BL)
#         distx = -dh.dist_pts_periodic(xy, xy, PV, dim=0, square_norm=False)
#         disty = -dh.dist_pts_periodic(xy, xy, PV, dim=1, square_norm=False)
#         # ii is the index of the row of blnew that we are creating
#         ii, dmyi = 0, 0
#         if lp['eta_alph'] > 0:
#             sgn = np.sign(np.random.rand(1) < lp['eta_alph'])
#         else:
#             sgn = np.sign(1. - lp['alph']) * np.ones(len(BL))
#         for row in BLabs:
#             # Add the points displaced by some amount from the original bond
#             # ll is path length from old site to new site
#             ll = dists[row[0], row[1]]
#             # ss is the path length between intermediate particles projected along the original bond
#             ss = ll / float(nnew)
#             hh = (ss * 0.5) / abs(np.tan(lp['alph'] * 0.5 * np.pi)) * sgn[dmyi]
#             # define vector taking row[0] to row[1]
#             vec = np.array([distx[row[0], row[1]], disty[row[0], row[1]]])
#             vechat = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
#             # get unit vector normal to vec --> just swap x,y and negate new x component to go clockwise
#             normal = np.array([vec[1], -vec[0]]) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
#             # add first point along vec
#             xynew[ptind] = xy[row[0]] + (ss * 0.5) * vechat + hh * normal
#
#             # Create the new lineseg from the original point to the first added point
#             blnew[ii, :] = np.array([row[0], ptind])
#             ii += 1
#             # connect the middle linesegs between added pts
#             for jj in range(nnew - 1):
#                 # add bonds
#                 blnew[ii, :] = np.array([ptind, ptind + 1])
#                 ptind += 1
#
#                 # add points
#                 xynew[ptind] = xy[row[0]] + vechat * (ss * (jj + 1.5)) + hh * normal * (-1) ** (jj + 1)
#                 ii += 1
#
#             # connect the last lineseg
#             if (BL[dmyi] < 0).any():
#                 print 'blf: BL[dmyi] = ', BL[dmyi]
#                 # get periodic vector from PVxydict
#                 if tuple(row) in PVxydict:
#                     pvxy = PVxydict[tuple(row)]
#                     key = tuple(row)
#                 elif (row[1], row[0]) in PVxydict:
#                     pvxy = -PVxydict[(row[1], row[0])]
#                     key = (row[1], row[0])
#                 else:
#                     print 'row = ', row
#                     print 'PVxydict = ', PVxydict
#                     raise RuntimeError('Could not find bond pair in PVxydict')
#
#                 # make sure we just have a single vector, not multiple
#                 if len(np.shape(pvxy)) > 1:
#                     # we have multiple vectors, use just the first one we haven't already used
#                     if key in pvdcount:
#                         pvdcount[key] += 1
#                     else:
#                         pvdcount[key] = 0
#
#                     print 'blf: key = ', key
#                     print 'blf: determined pvdcount = ', pvdcount[key]
#                     pvxy = pvxy[pvdcount[key]]
#                     print 'blf: determined pvxy = ', pvxy
#
#                 # take care not to overwrite an entry for the bond if it already exists
#                 if (ptind, row[1]) in pvdict_out:
#                     pvdict_out[(ptind, row[1])] = np.vstack((pvdict_out[(ptind, row[1])], pvxy))
#                     print 'pvdict = ', pvdict_out
#                     sys.exit()
#                     pvdcount[(ptind, row[1])]
#                 else:
#                     # define element of dictionary for the first time
#                     pvdict_out[(ptind, row[1])] = pvxy
#                 blnew[ii, :] = np.array([-ptind, -row[1]])
#             else:
#                 blnew[ii, :] = np.array([ptind, row[1]])
#             ii += 1
#             ptind += 1
#             dmyi += 1
#
#         blnew = blnew[0:ii]
#         xynew = xynew[0:ptind]
#         if lp['check']:
#             plt.plot(xynew[:, 0], xynew[:, 1], 'r.')
#             plt.plot(xy[:, 0], xy[:, 1], 'b.')
#             print 'blnew = ', blnew
#             for ii in range(len(xynew)):
#                 plt.text(xynew[ii, 0] - 0.2, xynew[ii, 1] + 0.2, str(ii))
#             for row in blnew:
#                 plt.plot([xynew[row[0], 0], xynew[row[1], 0]], [xynew[row[0], 1], xynew[row[1], 1]], 'k-')
#             plt.show()
#             netvis.movie_plot_2D(xynew, blnew, PVxydict=pvdict_out, show=True, bondcolor='k', ptcolor='b', axcb=None)
#
#     LVUC = None
#     UC = None
#     BL = blnew
#     xy = xynew
#     lattice_exten_add = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
#     NL, KL = le.BL2NLandKL(BL)
#     print 'blf: NL = ', NL
#     print 'blf: KL = ', KL
#     plt.plot(xy[:, 0], xy[:, 1], 'b.')
#     for ind in range(len(xy)):
#         plt.text(xy[ind, 0] + 0.03 * ind, xy[ind, 1], str(ind))
#
#     for row in BL:
#         if (row < 0).any():
#             plt.plot(xy[row, 0], xy[row, 1], 'k--')
#         else:
#             plt.plot(xy[row, 0], xy[row, 1], 'k-')
#
#     plt.show()
#     sys.exit()
#     PVx, PVy = le.PVxydict2PVxPVy(pvdict_out, NL, KL)
#     return xy, BL, pvdict_out, PVx, PVy, LVUC, UC, xyvertices, lattice_exten_add


def accordionkagomize_network(xy, BL, lp, PVxydict=None, PVx=None, PVy=None, PV=None):
    """Add lines or zig-zags of gyroscopes between every vertex, with specified angle. Number of gyros between vertices
    specified by lp['intparam'].

    Parameters
    ----------
    xy
    BL
    lp
    PV

    Returns
    -------

    """
    # accordionize the lattice
    xyacc, BLacc, LVUC, UC, xyvertices, lattice_exten_add = accordionize_network(xy, BL, lp, PVxydict=PVxydict,
                                                                                 PVx=PVx, PVy=PVy, PV=PV)

    print 'BL = ', BL

    # need indices of xy that correspond to xyvertices
    # note that xyvertices gives the positions of the vertices, not their indices
    inRx = np.in1d(xyacc[:, 0], xyvertices[:, 0])
    inRy = np.in1d(xyacc[:, 1], xyvertices[:, 1])
    vxind = np.where(np.logical_and(inRx, inRy))[0]
    print 'vxind = ', vxind

    # Note: beware, do not provide NL and KL to decorate_bondneighbors_elements() since NL,KL need
    # to be recalculated
    xy, BL = decorate_bondneighbors_elements(xyacc, BLacc, vxind, PVxydict=None, viewmethod=False, check=lp['check'])

    return xy, BL, LVUC, UC, xyvertices, lattice_exten_add


if __name__ == '__main__':
    import lepm.build.build_lattice_functions as blf
    import lepm.lattice_elasticity as le
    import lepm.plotting.network_visualization as netvis

    xy = np.random.rand(20, 2)
    lp = {'intparam': 2, 'alph': 1.0, 'check': True, 'eta_alph': 0.0}
    xy, NL, KL, BL, BM = le.delaunay_lattice_from_pts(xy, trimbound=False, max_bond_length=10., thres=10., check=False)
    netvis.movie_plot_2D(xy, BL, bs=0 * (BL[:, 0]), show=True, bondcolor='k')
    xyc, NLc, KLc, BLc = le.voronoi_lattice_from_pts(xy, kill_outliers=True)
    xy, BL, out, out2, out3, out4 = blf.accordionkagomize_network(xyc, BLc, lp)
    netvis.movie_plot_2D(xy, BL, bs=0 * (BL[:, 0]), show=True, bondcolor='k')
