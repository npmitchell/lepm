import lepm.lattice_elasticity as le
import lepm.data_handling as dh
import numpy as np
import lepm.plotting.colormaps as lecmaps
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import copy
import scipy.optimize as opt

'''
Relax a lattice via overdamped 2-point interaction or by minimizing bond energy
'''


def relax_overdamped(xy, pv, bbox, timesteps, dt, modout=10, outpath=None, save_im=False):
    """Relax a system of particles by a central force, with velocity equal to Force.
    For the interaction, use F=(1-r)rhat for r<1, where r is the distance between particles and rhat is the vector
    between them.

    Parameters
    ----------
    xy
    PV
    timesteps
    dt
    modout
    outpath : str
        the full path, except for the txt extension, to put the output files

    Returns
    -------

    """
    xyo = copy.deepcopy(xy)

    # Check if the bbox is a rectangle aligned with the xy axes
    isrect = (dh.unique_count(bbox[:, 0])[:, 1] > 1).all() and (dh.unique_count(bbox[:, 1])[:, 1] > 1).all()
    if isrect:
        minx = np.min(bbox[:, 0])
        maxx = np.max(bbox[:, 0])
        miny = np.min(bbox[:, 1])
        maxy = np.max(bbox[:, 1])
        # Since the bbox is a rectangle, we can assume one element of each of the periodic vectors is zero
        # Form pvx, the distance in x of the periodic vector pointing in the x dir, and similarly for y
        pvx = np.max(np.abs(pv[:, 0]))
        pvy = np.max(np.abs(pv[:, 1]))

    # Save initial positions if outpath is not None
    ii = 0
    time = 0.
    if outpath is not None:
        print 'saving to ', outpath + '{0:06d}'.format(ii)
        header = 'Relaxing (spreading) xy pts generated by gammakick procedure: t={0:0.12e}'.format(time) +\
                 ', dt={0:0.12e}'.format(dt)
        np.savetxt(outpath + '{0:08.3f}'.format(time).replace('.', 'p') + '.txt', xy, header=header)
    if save_im:
        lecmaps.register_colormaps()
        cmap = plt.get_cmap('viridis')

    for ii in (np.arange(timesteps) + 1):
        # Relax for a time dt
        if ii % modout == 0:
            print 'relaxing pts: ii=', ii
        time += dt
        dxy = spreading_forces(xyo, pv) * dt
        xyo += dxy
        if isrect:
            # Since the bbox is a rectangle, we can just move the points outside the bounds back inside
            xyo[xyo[:, 0] < minx, 0] += pvx
            xyo[xyo[:, 1] < miny, 1] += pvy
            xyo[xyo[:, 0] > maxx, 0] -= pvx
            xyo[xyo[:, 1] > maxy, 1] -= pvy
        else:
            # The bbox is a more complicated shape, so we have to use something like matplotlib's inpoly
            bpath = mplpath.Path(bbox)
            outside = not bpath.contains_points(xy)
            raise RuntimeError('Have not included functionality for non-rectangular periodic BCs. Build that here.')

        if np.mod(ii, modout) == 0 and outpath is not None:
            header = 'Relaxing (spreading) xy pts generated by gammakick procedure: t={0:0.12e}'.format(time) + \
                     ', dt={0:0.12e}'.format(dt)
            np.savetxt(outpath + '{0:08.3f}'.format(time).replace('.', 'p') + '.txt', xyo, header=header)

    if save_im:
        # load and plot points
        time = 0.
        for ii in np.arange(timesteps + 1):
            if np.mod(ii, modout) == 0:
                xyo = np.loadtxt(outpath + '{0:08.3f}'.format(time).replace('.', 'p') + '.txt')
                plt.scatter(xyo[:, 0], xyo[:, 1], c=cmap(float(ii) / float(timesteps)), edgecolor='none')
            time += dt

        plt.savefig(outpath + 'relaxation_visualization.png')
    return xyo


def spreading_forces(xy, pv, eps=1e-12, size_thres=100, check=False):
    """Compute force F=(1-r)rhat for r<1, where r is the distance between particles and rhat is the vector between them

    Parameters
    ----------
    xy : n x 2 float array
        initial positions of the points to spread (repel)
    pv : 2 x 2 float array
        periodic lattice vectors taking points to their mirrors across each boundary
    eps : float
        small number to check if something is nonzero
    size_thres : int
        maximum number of particles for which to compute distances using numpy arrays rather than iterating over floats
    check : bool
        View intermediate results

    Returns
    -------
    forces : n x 2 float array
        The forces on each particle from repulsive interactions
    """
    if len(xy) > size_thres:
        forces = np.zeros_like(xy, dtype=float)
        for ii in range(len(xy)):
            if ii % (len(xy) * 0.2) < 1.:
                print 'examining forces on particle ii = ', ii
            # Only consider particles with indices greater than ii
            jjrange = np.arange(ii + 1, len(xy))
            # print 'jjrange= ', jjrange
            for jj in jjrange:
                # print '(ii, jj) = ', (ii, jj)
                if abs(xy[jj, 0] - xy[ii, 0]) < 1.0:
                    if abs(xy[jj, 1] - xy[ii, 1]) < 1.0:
                        xdist = xy[jj, 0] - xy[ii, 0]
                        ydist = xy[jj, 1] - xy[ii, 1]
                        dist = np.sqrt(xdist ** 2 + ydist ** 2)
                        if dist < 1.0:
                            xdhat = xdist / np.abs(dist)
                            ydhat = ydist / np.abs(dist)
                            xforce = (1. - dist) * xdhat
                            yforce = (1. - dist) * ydhat
                            forces[ii, 0] += -xforce
                            forces[ii, 1] += -yforce
                            forces[jj, 0] += xforce
                            forces[jj, 1] += yforce
    else:
        dists = dh.dist_pts_periodic(xy, xy, pv, dim=-1, square_norm=False)
        # print 'np.shape(dists) = ', np.shape(dists)
        xdists = dh.dist_pts_periodic(xy, xy, pv, dim=0)
        ydists = dh.dist_pts_periodic(xy, xy, pv, dim=1)
        xdhat = np.nan_to_num(xdists / np.abs(dists))
        ydhat = np.nan_to_num(ydists / np.abs(dists))
        xforce = (1. - dists) * xdhat
        yforce = (1. - dists) * ydhat

        # print 'np.shape(xforce) = ', np.shape(xforce)
        # find where distances are greater than eps and less than 1
        mask = np.logical_and(dists > eps, dists < 1.0).astype(float)
        # sum over each row to get the force on that particle
        forces = np.zeros_like(xy, dtype=float)
        if check:
            le.plot_real_matrix(xdists, show=True, name='xdists')
            le.plot_real_matrix(xdhat, show=True, name='xdhat')
            le.plot_real_matrix(xforce, show=True, name='xforce')
            le.plot_real_matrix(xforce * mask, show=True, name='xforce * mask')

        # Note that it actually doesn't matter which axis we sum over, since the distance matrix is
        # equal to its transpose.
        # If there is any repulsion, sum over repulsive forces in net forces
        if len(np.nonzero(mask)) > 0:
            forces[:, 0] = np.sum(xforce * mask, axis=1)
            forces[:, 1] = np.sum(yforce * mask, axis=1)

    return forces


def relax_overdamped_openbc(xy, timesteps, dt, modout=10, outpath=None, save_im=False, clamp_boundary=False):
    """Relax a system with open boundary conditions of particles by a central force, with velocity equal to Force.
    For the interaction, use F=(1-r)rhat for r<1, where r is the distance between particles and rhat is the vector
    between them.

    Parameters
    ----------
    xy
    PV
    timesteps
    dt
    modout
    outpath : str
        the full path, except for the txt extension, to put the output files
    clamp_boundary : bool
        Whether to affix the boundary particles during each relaxation step

    Returns
    -------

    """
    xyo = copy.deepcopy(xy)

    # Save initial positions if outpath is not None
    ii = 0
    time = 0.

    if outpath is not None:
        print 'saving to ', outpath + '{0:06d}'.format(ii)
        header = 'Relaxing (spreading) xy pts generated by gammakick procedure: t={0:0.12e}'.format(time) +\
                 ', dt={0:0.12e}'.format(dt)
        np.savetxt(outpath + '{0:08.3f}'.format(time).replace('.', 'p') + '.txt', xy, header=header)
    if save_im:
        lecmaps.register_colormaps()
        cmap = plt.get_cmap('viridis')

    for ii in (np.arange(timesteps) + 1):
        # Relax for a time dt
        if ii % modout == 0:
            print 'relaxing pts: ii=', ii
        time += dt
        dxy = -spreading_forces_openbc(xyo) * dt

        if clamp_boundary:
            # todo: add code for boundary clamping
            raise RuntimeError('Add code to clamp the boundary here!')

        xyo += dxy

        if np.mod(ii, modout) == 0 and outpath is not None:
            header = 'Relaxing (spreading) xy pts generated by gammakick procedure: t={0:0.12e}'.format(time) + \
                     ', dt={0:0.12e}'.format(dt)
            np.savetxt(outpath + '{0:08.3f}'.format(time).replace('.', 'p') + '.txt', xyo, header=header)
        if save_im:
            plt.scatter(xyo[:, 0], xyo[:, 1], c=cmap(float(ii)/float(timesteps)), edgecolor='none')
    return xyo


def spreading_forces_openbc(xy, eps=1e-12, check=False):
    """Compute force F=(1-r)rhat for r<1, where r is the distance between particles and rhat is the vector between them

    Parameters
    ----------
    xy
    pv

    Returns
    -------
    forces
    """
    dists = dh.dist_pts(xy, xy, dim=-1, square_norm=False)
    # print 'np.shape(dists) = ', np.shape(dists)
    xdists = dh.dist_pts(xy, xy, dim=0)
    ydists = dh.dist_pts(xy, xy, dim=1)
    xdhat = np.nan_to_num(xdists / np.abs(dists))
    ydhat = np.nan_to_num(ydists / np.abs(dists))
    xforce = (1 - dists) * xdhat
    yforce = (1 - dists) * ydhat
    # print 'np.shape(xforce) = ', np.shape(xforce)
    # find where distances are greater than eps and less than 1
    mask = np.logical_and(dists > eps, dists < 1.0).astype(float)
    # sum over each row to get the force on that particle
    forces = np.zeros_like(xy, dtype=float)
    if check:
        le.plot_real_matrix(xdists, show=True, name='xdists')
        le.plot_real_matrix(xdhat, show=True, name='xdhat')
        le.plot_real_matrix(xforce, show=True, name='xforce')
        le.plot_real_matrix(xforce * mask, show=True, name='xforce * mask')

    # Note that it actually doesn't matter which axis we sum over, since the distance matrix is equal to its transpose
    # If there is any repulsion, sum over repulsive forces in net forces
    if len(np.nonzero(mask)) > 0:
        forces[:, 0] = np.sum(xforce * mask, axis=1)
        forces[:, 1] = np.sum(yforce * mask, axis=1)

    return forces


def relax_bondenergy(xy, BL, fixedpts, tol=1e-2, kL=1., bo=1., check=False, checkim_outpath=None):
    """

    Parameters
    ----------
    xy
    BL
    fixedpts
    tol
    kL
    bo
    check

    Returns
    -------

    """
    # Relax lattice, but fix pts with low LV[1] vals
    # First get ADJACENT pts with low LV[1] values to fix
    NP = len(xy)
    if len(fixedpts) == 2:
        # ensure that indices are in ascending order
        if fixedpts[0] > fixedpts[1]:
            pair = [fixedpts[1], fixedpts[0]]
        else:
            pair = fixedpts

        Nzero_pair0 = np.arange(0, pair[0])
        Npair0_pair1 = np.arange(pair[0], pair[1])
        Npair1_end = np.arange(pair[1], NP)
        print 'pair = ', pair
        bounds = [[None, None]] * (2 * (pair[0])) + \
                 np.c_[xy[pair[0], :].ravel(), xy[pair[0], :].ravel()].tolist() + \
                 [[None, None]] * (2 * (pair[1] - pair[0] - 1)) + \
                 np.c_[xy[pair[1], :].ravel(), xy[pair[1], :].ravel()].tolist() + \
                 [[None, None]] * (2 * (NP - pair[1] - 1))
        print 'lepm.build.pointsets.relax_pointset.relax_bondenergy(): len(bounds) = ', len(bounds)
        print 'lepm.build.pointsets.relax_pointset.relax_bondenergy(): len(xy) = ', len(xy)
    else:
        raise RuntimeError('Currently cannot handle more than two fixed points -- add that code here -- should be easy')

    # Old way: fix particles 0 and 1
    # bounds = np.c_[xy[:2,:].ravel(), xy[:2,:].ravel()].tolist() + \
    #                 [[None, None]] * (2*(NP-2))

    # relaxed lattice
    print 'relaxing lattice...'

    def flattened_potential_energy(xy):
        # We convert xy to a 2D array here.
        xy = xy.reshape((-1, 2))
        bL = le.bond_length_list(xy, BL)
        # print 'bL = ', bL
        bU = 0.5 * sum(kL * (bL - bo) ** 2)
        return bU

    xyR = opt.minimize(flattened_potential_energy, xy.ravel(),
                       method='L-BFGS-B',
                       bounds=bounds, tol=tol).x.reshape((-1, 2))
    xy = xyR

    bL0 = bo * np.ones_like(BL[:, 0], dtype=float)
    if check:
        bs = le.bond_strain_list(xy, BL, bL0)
        le.display_lattice_2D(xyR, BL, bs=bs, title='Energy-relaxed network', colorz=False, close=False)
        plt.scatter(xy[pair, 0], xy[pair, 1], s=500, c='r')
        if checkim_outpath is not None:
            plt.savefig(checkim_outpath)
        plt.show()

    return xy