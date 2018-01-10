import numpy as np
import matplotlib.pyplot as plt
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import cPickle as pkl
import glob
import lepm.twisty.twisty_functions as tfns
import sys
import lepm.brillouin_zone_functions as bzf
import lepm.data_handling as dh
import lepm.plotting.colormaps as lecmaps
import lepm.twisty.plotting.plotting as tplt
import lepm.plotting.movies as lemov
import copy

'''
Description
===========
Momentum-space functions for TwistyLattice class methods
'''


def lowest_eigenmode(tlat, nkxy=50, save=True, save_plots=True, name=None, outdir=None, imtype='png', rm_images=True):
    """Plot the eigenmode with lowest energy for the network

    Parameters
    ----------
    tlat
    nkxy
    save
    save_plots
    name
    outdir
    imtype

    Returns
    -------

    """
    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(tlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'lowest_eigenmode' + tlat.lp['meshfn_exten']
        name += '_nkxy{0:06d}'.format(np.max(np.abs(nkxy)))

    outname = outdir + name
    fn = glob.glob(outname + '.pkl')
    if fn:
        with open(fn[0], "rb") as fn:
            res = pkl.load(fn)

        vtcs = res['vtcs']
        kxy = res['kxy']
        omegas = res['omegas']
        eigvects = res['eigvects']
        eigvals_all = res['eigvals_all']
        symmetryinds = res['bz_symmetrypt_inds']
    else:
        pvs = tlat.lattice.PV
        a1, a2 = pvs[0], pvs[1]
        vtx, vty = bzf.bz_vertices(a1, a2)
        vtcs = np.dstack((vtx, vty))[0]
        # generate the kspace points
        if tlat.lp['periodic_strip']:
            symmetryinds = [0]
            pts = np.linspace(-np.pi, 0.0, int(nkxy * 0.5))
            symmetryinds.append(len(pts))
            pts = np.hstack((pts, np.linspace(0., np.pi, int(nkxy * 0.5))))
            symmetryinds.append(len(pts) - 1)
            pts = np.dstack((pts, np.zeros_like(pts)))[0]
        else:
            pts, symmetryinds = bzf.generate_bzpath(vtcs, npts=nkxy)

        # # check pts
        # plt.plot(pts[:, 0], pts[:, 1], 'b.')
        # plt.axis('equal')
        # plt.show()
        # sys.exit()

        # print 'tkspacefns: pts = ', np.shape(pts)
        omegas = np.zeros(len(pts))
        eigvects = np.zeros((len(pts), len(tlat.xy_inner) * 4), dtype=complex)
        eigvals_all = np.zeros((len(pts), len(tlat.xy_inner) * 4), dtype=complex)
        matk = lambda k: dynamical_matrix_kspace(k, tlat, eps=1e-10)
        ii = 0
        for kxy in pts:
            print 'tlatkspace_fns: infinite_dispersion(): ii = ', ii
            # print 'jj = ', jj
            kx, ky = kxy[0], kxy[1]
            matrix = matk([kx, ky])
            print 'tlatkspace_fns: diagonalizing...'
            eigval, eigvect = np.linalg.eig(matrix)
            si = np.argsort(np.real(eigval))
            omegas[ii] = np.real(np.min(eigval[si]))
            eigvals_all[ii] = eigval[si]
            minii = np.min(eigval)
            eigvects[ii] = eigvect[minii]
            ii += 1

        # Save results to pickle if save == True
        res = {'omegas': omegas, 'kxy': pts, 'vtcs': vtcs, 'eigvects': eigvects, 'eigvals_all': eigvals_all,
               'bz_symmetrypt_inds': symmetryinds}
        kxy = pts
        if save:
            with open(outname + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    if save_plots:
        if tlat.lp['periodic_strip']:
            plot_lowest_eigenmode_periodicstrip(tlat, eigvects, eigvals_all, omegas, kxy, symmetryinds, vtcs, nkxy,
                                                outdir=dio.prepdir(outname), name=name, imtype=imtype,
                                                rm_images=rm_images)
        else:
            plot_lowest_eigenmode(tlat, eigvects, eigvals_all, omegas, kxy, symmetryinds, vtcs, nkxy,
                                  outdir=dio.prepdir(outname), name=name, imtype=imtype, rm_images=rm_images)

    return eigvects, omegas, kxy, symmetryinds, vtcs


def plot_lowest_eigenmode(tlat, eigvects, eigval, omegas, kxy, symmetryinds, vtcs, nkxy,
                          outdir=None, name=None, imtype='png',
                          fig=None, axes=None, cax=None, eps=1e-8, title=None, rm_images=True):
    """

    Parameters
    ----------
    eigvects
    omegas
    kxy
    vtcs
    nkxy
    outdir
    eps

    Returns
    -------

    """
    if name is None:
        # get name from meshfn
        name = tlat.lp['meshfn_exten']

    dio.ensure_dir(dio.prepdir(outdir))

    if axes is None:
        wfig = 180.
        hfig = wfig * 9./16.
        fig, axes = leplt.initialize_2panel_4o3ar_cent(Wfig=wfig, Hfig=hfig, wsfrac=0.3, wssfrac=0.5)

    # ax will be for the BZ, ax1 for the mode decomposition
    ax, ax1 = axes[0], axes[1]

    # plot the BZ path
    ylims = (min(-0.1, np.min(omegas)), min(np.max(eigval.ravel()), max(0.2, np.max(omegas))))
    ax.plot(np.arange(len(kxy)), eigval, 'k-')
    ax.plot(np.arange(len(kxy)), omegas, '-')
    ax.xaxis.set_ticks(symmetryinds)
    ax.xaxis.set_ticklabels([r'$K$', r'$\Gamma$', r'$M$', r'$K$'])
    ax.set_xlabel(r'wavevector, $k$')
    ax.set_ylabel(r'energy, $\omega$')
    ax.set_ylim(ylims)
    if title is None:
        title = r'$k=$' + '{0:0.3f}'.format(tlat.lp['kk']) + ', '
        title += r'$g=$' + '{0:0.3f}'.format(tlat.lp['gg']) + ', '
        title += r'$c=$' + '{0:0.3f}'.format(tlat.lp['cc'])
        ax.set_title(title)
    elif title is not '' and title is not 'none':
        ax.set_title(title)

    # Plot the network on ax1
    tlat.lattice.plot_BW_lat(fig=fig, ax=ax1, save=False, close=False, axis_off=False, title='')
    ax1.set_xlim(np.min(tlat.lattice.xy[:, 0]) - 1., np.max(tlat.lattice.xy[:, 0]) + 1.)
    ax1.set_ylim(np.min(tlat.lattice.xy[:, 1]) - 1., np.max(tlat.lattice.xy[:, 1]) + 1.)
    ax1.axis('off')

    # get a path from origin to K and K' points
    kk = 0
    for kpt in kxy:
        print 'kk = ', kk
        dot = ax.plot(kk, omegas[kk], '.', color=lecmaps.red())

        # Plot normal mode eigvects[kk]
        # print 'tkspace_fns: np.shape(omegas) = ', np.shape(omegas)
        # print 'tkspace_fns: np.shape(eigvects) = ', np.shape(eigvects)
        # print 'tkspace_fns: np.shape(eigvects[kk]) = ', np.shape(eigvects[kk])
        fig, [scat_fg, scat_fg2, f_mark, polygons, tiltgons], cw_ccw = \
            tplt.plot_eigvect_excitation(tlat.xy_inner, fig, None, ax1,
                                         [omegas[kk]], [eigvects[kk]], 0, marker_num=0,
                                         black_t0lines=False, mark_t0=True, title='', normalization=1., alpha=0.6,
                                         lw=1, zorder=10, cmap='isolum_rainbow', color='phase', theta=None,
                                         t0_ptsize=.02)

        textadd = ax1.text(0.5, 1.0, r'$\omega=$' + '{0:0.3f}'.format(omegas[kk]),
                           ha='center', va='center', transform=ax1.transAxes)
        # ycoord = np.min(tlat.lattice.xy[:, 1]) - 1.
        # textadd = ax1.text(0., ycoord, r'$\omega=$' + '{0:0.3f}'.format(omegas[kk]),
        #                    ha='center', va='top')  # , transform=ax1.transAxes)

        if outdir is not None:
            if '.' not in outdir:
                outname = outdir + name + '_{0:05d}'.format(kk) + '.' + imtype
            else:
                outname = copy.deepcopy(outdir)

            print '{0:04d}'.format(kk)
            print 'saving to ', outname
            plt.savefig(outname)
        else:
            plt.show()

        # remove previous from plot
        scat_fg.remove()
        scat_fg2.remove()
        polygons.remove()
        tiltgons.remove()
        if f_mark is not None:
            f_mark.remove()
        dot.pop(0).remove()
        textadd.remove()
        ax1
        kk += 1

    imgname = outdir + name + '_'
    movname = copy.deepcopy(outdir)
    if movname[-1] == '/':
        movname = movname[0:-1]

    framerate = len(kxy) * 0.15
    lemov.make_movie(imgname, movname, indexsz='05', framerate=framerate, imgdir=outdir, rm_images=rm_images,
                     save_into_subdir=True)
    return fig, axes


def plot_lowest_eigenmode_periodicstrip(tlat, eigvects, eigval, omegas, kxy, symmetryinds, vtcs, nkxy,
                                        outdir=None, name=None, imtype='png',
                                        fig=None, axes=None, cax=None, eps=1e-8, title=None, rm_images=True):
    """

    Parameters
    ----------
    eigvects
    omegas
    kxy
    vtcs
    nkxy
    outdir
    eps

    Returns
    -------

    """
    if name is None:
        # get name from meshfn
        name = tlat.lp['meshfn_exten']

    dio.ensure_dir(dio.prepdir(outdir))

    if axes is None:
        wfig = 180.
        hfig = wfig * 9./16.
        fig, axes = leplt.initialize_2panel_4o3ar_cent(Wfig=wfig, Hfig=hfig)

    ax, ax1 = axes[0], axes[1]

    # plot the BZ path
    ylims = (min(-0.1, np.min(omegas)), min(np.max(eigval.ravel()), max(0.2, np.max(omegas))))
    ax.plot(np.arange(len(kxy)), eigval, 'k-')
    ax.plot(np.arange(len(kxy)), omegas, '-')
    ax.xaxis.set_ticks(symmetryinds)
    ax.xaxis.set_ticklabels([r'$K$', r'$\Gamma$', r'$M$', r'$K$'])
    ax.set_xlabel(r'wavevector, $k$')
    ax.set_ylabel(r'energy, $\omega$')
    ax.set_ylim(ylims)
    if title is None:
        title = r'$k=$' + '{0:0.3f}'.format(tlat.lp['kk']) + ', '
        title += r'$g=$' + '{0:0.3f}'.format(tlat.lp['gg']) + ', '
        title += r'$c=$' + '{0:0.3f}'.format(tlat.lp['cc'])
        ax.set_title(title)
    elif title is not '' and title is not 'none':
        ax.set_title(title)

    # Plot the network on ax1
    tlat.lattice.plot_BW_lat(fig=fig, ax=ax1, save=False, close=False, axis_off=False, title='')
    ax1.set_xlim(np.min(tlat.lattice.xy[:, 0]) - 1., np.max(tlat.lattice.xy[:, 0]) + 1.)
    ax1.set_ylim(np.min(tlat.lattice.xy[:, 1]) - 1., np.max(tlat.lattice.xy[:, 1]) + 1.)
    ax1.axis('off')

    # get a path from origin to K and K' points
    kk = 0
    for kpt in kxy:
        print 'twistykspace_fns: kk = ', kk
        dot = ax.plot(kk, omegas[kk], '.', color=lecmaps.red())

        # Plot normal mode eigvects[kk]
        # print 'tkspace_fns: np.shape(omegas) = ', np.shape(omegas)
        # print 'tkspace_fns: np.shape(eigvects) = ', np.shape(eigvects)
        # print 'tkspace_fns: np.shape(eigvects[kk]) = ', np.shape(eigvects[kk])
        fig, [scat_fg, scat_fg2, f_mark, polygons, tiltgons], cw_ccw = \
            tplt.plot_eigvect_excitation(tlat.xy_inner, fig, None, ax1, [omegas[kk]], [eigvects[kk]], 0, marker_num=0,
                                         black_t0lines=False, mark_t0=True, title='', normalization=1., alpha=0.6,
                                         lw=1, zorder=10, cmap='isolum_rainbow', color='phase', theta=None,
                                         t0_ptsize=.02)

        title = r'$k_x=$' + '{0:0.3f}'.format(kpt[0])
        title += r', $\omega=$' + '{0:0.3f}'.format(omegas[kk])
        ax1.set_title(title)

        # ycoord = np.min(tlat.lattice.xy[:, 1]) - 1.
        # textadd = ax1.text(0., ycoord, r'$\omega=$' + '{0:0.3f}'.format(omegas[kk]),
        #                    ha='center', va='top')  # , transform=ax1.transAxes)

        if outdir is not None:
            if '.' not in outdir:
                outname = outdir + name + '_{0:05d}'.format(kk) + '.' + imtype
            else:
                outname = copy.deepcopy(outdir)

            print '{0:04d}'.format(kk)
            print 'saving to ', outname
            plt.savefig(outname)
        else:
            plt.show()

        # remove previous from plot
        scat_fg.remove()
        scat_fg2.remove()
        polygons.remove()
        tiltgons.remove()
        if f_mark is not None:
            f_mark.remove()
        dot.pop(0).remove()
        kk += 1

    imgname = outdir + name + '_'
    movname = copy.deepcopy(outdir)
    if movname[-1] == '/':
        movname = movname[0:-1]

    framerate = len(kxy) * 0.15
    lemov.make_movie(imgname, movname, indexsz='05', framerate=framerate, rm_images=rm_images, save_into_subdir=True)
    return fig, axes


def plot_lowest_mode(omegas, kxy, vtcs, nkxy=20, fig=None, ax=None, cax=None, outpath=None, eps=1e-8):
    """After obtaining the lowest mode via lowest_mode(), pass the output into this function to
    plot the results on a figure.

    Parameters
    ----------
    omegas
    kxy
    vtcs
    nkxy : int
    fig : matplotlib.pyplot Figure instance or None
        the figure instance on which to plot the lowest mode of a twisty lattice
    ax : axis instance or None
        the axis on which to plot the lowest mode in the brillouin zone
    cax : axis instance or None
        the axis for the colorbar
    outpath : str or None
    eps: float
        threshold minimum value

    Returns
    -------
    fig : matplotlib Figure instance
    ax : matplotlib axis instance
    """
    if fig is None and ax is None and cax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_cent(wsfrac=0.5, tspace=4)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    xgrid, ygrid, ZZ = dh.interpol_meshgrid(kxy[:, 0], kxy[:, 1], omegas, int(nkxy), method='nearest')
    inrois = dh.gridpts_in_polygons(xgrid, ygrid, [vtcs])
    vmax = max(np.max(omegas), eps)
    # Here use blue to white
    ZZ[~inrois] = vmax
    cmap = lecmaps.colormap_from_hex('#2F5179', reverse=True)
    # could use white to blue
    # ZZ[~inrois] = vmax
    # cmap = lecmaps.colormap_from_hex('#2F5179', reverse=False)

    if ax is None:
        ax = plt.gca()
    pcm = ax.pcolormesh(xgrid, ygrid, ZZ, cmap=cmap, vmin=0., vmax=vmax, alpha=1.0)
    # plt.show()
    # sm = leplt.empty_scalar_mappable(vmin=0., vmax=vmax, cmap=cmap)
    # plt.colorbar(sm, cax=cax, label=r'$\omega_0$', orientation='horizontal', ticks=[0., vmax])
    plt.colorbar(pcm, cax=cax, label=r'$\omega_0$', orientation='horizontal', ticks=[0., vmax])
    ax.axis('off')
    ax.axis('scaled')
    if outpath is not None:
        plt.savefig(outpath)
    return fig, ax, cax


def lowest_mode(tlat, nkxy=20, save=False, save_plot=True, name=None, outdir=None, imtype='png'):
    """

    Parameters
    ----------
    tlat : TwistyLattice class instance
        the lattice for which to plot the lowest vibrational mode
    nkxy : int
        how many points to sample in each dimension, or the minimum number in the dimension with fewer sample points
    save : bool
        whether to save the results in pkl
    save_plot : bool
        whether to save the plot of the results (value of lowest mode in BZ)
    name : str or None
        the filename to save the pkl (and image, optional)
    outdir : str or None
        the directory in which to save the image
    imtype : str ('png', 'pdf', 'jpg', etc)
        the image type for the output image

    Returns
    -------

    """
    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(tlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'lowest_mode' + tlat.lp['meshfn_exten']
        name += '_nkxy{0:06d}'.format(np.max(np.abs(nkxy)))

    name = outdir + name
    fn = glob.glob(name + '.pkl')
    if fn:
        with open(fn[0], "rb") as fn:
            res = pkl.load(fn)

        vtcs = res['vtcs']
        kxy = res['kxy']
        omegas = res['omegas']
    else:
        pvs = tlat.lattice.PV
        a1, a2 = pvs[0], pvs[1]
        vtx, vty = bzf.bz_vertices(a1, a2)
        vtcs = np.dstack((vtx, vty))[0]
        polygon = np.dstack((vtx, vty))[0]
        xlims = (np.min(vtx), np.max(vtx))
        ylims = (np.min(vty), np.max(vty))
        xextent, yextent = xlims[1] - xlims[0], ylims[1] - ylims[0]
        step = float(max(xextent, yextent)) / float(nkxy)
        print 'extent = ', xextent
        print 'step = ', step
        pts = dh.generate_gridpts_in_polygons(xlims, ylims, [polygon], dx=step, dy=step)
        # print 'tkspacefns: pts = ', np.shape(pts)
        omegas = np.zeros(len(pts))
        matk = lambda k: dynamical_matrix_kspace(k, tlat, eps=1e-10)
        ii = 0
        for kxy in pts:
            print 'tlatkspace_fns: infinite_dispersion(): ii = ', ii
            # print 'jj = ', jj
            kx, ky = kxy[0], kxy[1]
            matrix = matk([kx, ky])
            print 'tlatkspace_fns: diagonalizing...'
            eigval, eigvect = np.linalg.eig(matrix)
            si = np.argsort(np.real(eigval))
            omegas[ii] = np.real(np.min(eigval[si]))
            ii += 1

        # Save results to pickle if save == True
        res = {'omegas': omegas, 'kxy': pts, 'vtcs': vtcs}
        kxy = pts
        if save:
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    if save_plot:
        plot_lowest_mode(omegas, kxy, vtcs, nkxy, outpath=name + '.' + imtype)

    return omegas, kxy, vtcs


def infinite_dispersion(tlat, kx=None, ky=None, nkxvals=50, nkyvals=20, save=True, plot=True, save_plot=True,
                        title='twisty dispersion relation', outdir=None, name=None, ax=None, xaxis=None):
    """Compute the imaginary part of the eigvalues of the dynamical matrix for a grid of wavevectors kx, ky

    Parameters
    ----------
    tlat : TwistyLattice class instance
        the twisty network whose dispersion we compute
    kx : n x 1 float array
        the x components of the wavevectors over which to diagonalize the dynamical matrix
    ky : m x 1 float array
        the y components of the wavevectors over which to diagonalize the dynamical matrix
    nkxvals : int
        If kx is unspecified, then nkxvals determines how many kvectors are sampled in x dimension.
    nkyvals : int
        If ky is unspecified and if network is not a periodic_strip, then nkyvals determines how
        many kvectors are sampled in y dimension.
    save : bool
        Save the omega vs k information in a pickle
    plot : bool
        Plot the resulting dispersion on the supplied or initialized axis
    save_plot : bool
        Save the omega vs k info as a matplotlib figure png
    title : str
        title for the plot to save
    outdir : str or None
        The directory in which to output the image and pickle of the results if save == True

    Returns
    -------

    """
    # if not tlat.lp['periodicBC']:
    #     raise RuntimeError('Cannot compute dispersion for open BC system')
    if tlat.lp['periodic_strip']:
        print 'Evaluating infinite-system dispersion for strip: setting ky=constant=0.'
        ky = [0.]
        bboxx = max(tlat.lattice.lp['BBox'][:, 0]) - min(tlat.lattice.lp['BBox'][:, 0])
    elif ky is None or kx is None:
        bboxx = max(tlat.lattice.lp['BBox'][:, 0]) - min(tlat.lattice.lp['BBox'][:, 0])
        bboxy = max(tlat.lattice.lp['BBox'][:, 1]) - min(tlat.lattice.lp['BBox'][:, 1])

    if kx is None:
        if nkxvals == 0:
            kx = np.array([0.])
        else:
            tmp = np.linspace(-1. / bboxx, 1. / bboxx, nkxvals - 1)
            step = np.diff(tmp)[0]
            kx = 2. * np.pi * np.linspace(-1. / bboxx, 1. / bboxx + step, nkxvals)
        # kx = np.linspace(-5., 5., 40)

    if ky is None:
        if nkyvals == 0:
            ky = np.array([0.])
        else:
            tmp = np.linspace(-1. / bboxy, 1. / bboxy, nkyvals - 1)
            step = np.diff(tmp)[0]
            ky = 2. * np.pi * np.linspace(-1. / bboxy, 1. / bboxy + step, nkyvals)
        # ky = np.linspace(-5., 5., 4)

    # First check for saved dispersion
    if outdir is None:
        outdir = dio.prepdir(tlat.lp['meshfn'])
    else:
        outdir = dio.prepdir(outdir)

    if name is None:
        name = 'dispersion' + tlat.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
        name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
        name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

    name = outdir + name
    print('checking for file: ' + name + '.pkl')
    if glob.glob(name + '.pkl'):
        saved = True
        with open(name + '.pkl', "rb") as fn:
            res = pkl.load(fn)

        omegas = res['omegas']
        kx = res['kx']
        ky = res['ky']
    else:
        # dispersion is not saved, compute it!
        saved = False

        omegas = np.zeros((len(kx), len(ky), len(tlat.xy_inner) * 4))
        matk = lambda k: dynamical_matrix_kspace(k, tlat, eps=1e-10)
        ii = 0
        for kxi in kx:
            print 'tlatkspace_fns: infinite_dispersion(): ii = ', ii
            jj = 0
            for kyj in ky:
                # print 'jj = ', jj
                matrix = matk([kxi, kyj])
                print 'tlatkspace_fns: diagonalizing...'
                eigval, eigvect = np.linalg.eig(matrix)
                si = np.argsort(np.real(eigval))
                omegas[ii, jj, :] = np.real(eigval[si])
                # print 'eigvals = ', eigval
                # print 'omegas --> ', omegas[ii, jj]
                jj += 1
            ii += 1

    if plot:
        if ax is None:
            fig, ax = leplt.initialize_1panel_centered_fig()
            axsupplied = False
        else:
            axsupplied = True

        if xaxis == 'kx':
            for jj in range(len(ky)):
                for kk in range(len(omegas[0, jj, :])):
                    ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
        elif xaxis == 'ky':
            for jj in range(len(ky)):
                for kk in range(len(omegas[jj, 0, :])):
                    ax.plot(kx, omegas[jj, :, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
        else:
            raise RuntimeError("Argument 'xaxis' must be either 'kx' or 'ky'")

        if save_plot:
            ax.set_title(title)
            if xaxis == 'kx':
                ax.set_xlabel(r'wavenumber, $k_x$')
            elif xaxis == 'ky':
                ax.set_xlabel(r'wavenumber, $k_y$')
            ax.set_ylabel(r'$\omega$')
            ylims = ax.get_ylim()
            ylim0 = min(ylims[0], -0.1 * ylims[1])
            ax.set_ylim(ylim0, ylims[1])
            # Save the plot
            plt.savefig(name + '.png', dpi=300)

            ax.set_ylim(max(ylim0, -0.05 * ylims[1]), 0.05 * ylims[1])
            # Save the plot
            plt.savefig(name + '_zoom.png', dpi=300)

            # Fixed zoom
            ax.set_ylim(-0.05, 0.25)
            ax.set_xlim(-3.14, 3.14)
            plt.savefig(name + '_zoom2.png', dpi=300)
            plt.close('all')

            # save plot of ky if no axis supplied
            if not axsupplied and xaxis==None and not tlat.lp['periodic_strip']:
                fig, ax = leplt.initialize_1panel_centered_fig()
                for jj in range(len(kx)):
                    for kk in range(len(omegas[jj, 0, :])):
                        ax.plot(ky, omegas[jj, :, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
                ax.set_title(title)
                ax.set_xlabel(r'$k_y$ $[\langle \ell \rangle ^{-1}]$')
                ax.set_ylabel(r'$\omega$')
                ylims = ax.get_ylim()
                ylim0 = min(ylims[0], -0.1 * ylims[1])
                ax.set_ylim(ylim0, ylims[1])
                # Save the plot
                plt.savefig(name + '_ky.png', dpi=300)

                ax.set_ylim(max(ylim0, -0.05 * ylims[1]), 0.05 * ylims[1])
                # Save the plot
                plt.savefig(name + '_zoom_ky.png', dpi=300)

                # Fixed zoom
                ax.set_ylim(-0.05, 0.25)
                ax.set_xlim(-3.14, 3.14)
                plt.savefig(name + '_zoom2_ky.png', dpi=300)
                plt.close('all')

            # Plot in 3D
            # fig = plt.gcf()
            # ax = fig.add_subplot(projection='3d')  # 111,
            # # rows will be kx, cols wll be ky
            # kyv = np.array([[ky[i].tolist()] * len(kx) for i in range(len(ky))]).ravel()
            # kxv = np.array([[kx.tolist()] * len(ky)]).ravel()
            # print 'kyv = ', np.shape(kyv)
            # print 'kxv = ', np.shape(kxv)
            # for kk in range(len(omegas[0, 0, :])):
            #     ax.plot_trisurf(kxv, kyv, omegas[:, :, kk].ravel())
            #
            # ax.view_init(elev=0, azim=0.)
            # ax.set_title(title)
            # ax.set_xlabel(r'$k_x$ $[1/\langle l \rangle]$')
            # ax.set_ylabel(r'$k_y$ $[1/\langle l \rangle]$')
            # plt.savefig(name + '_3d.png')

    if save:
        if not saved:
            res = {'omegas': omegas, 'kx': kx, 'ky': ky}
            with open(name + '.pkl', "wb") as fn:
                pkl.dump(res, fn)

    return omegas, kx, ky


def vec(ang, bl=1):
    """creates a vector given a bond length and angle in radians"""
    return [bl * np.cos(ang), bl * np.sin(ang)]


def ang_fac(ang):
    """Convert angle to factor in the Hamiltonian for gyro+spring system"""
    return np.exp(2 * 1j * ang)


def lambda_matrix_kspace(tlat, eps=1e-10):
    """Construct the dynamical matrix for the given TwistyLattice as a function of an as-yet-unspecified wavevector kvec.

    Parameters
    ----------
    tlat : TwistyLattice class instance
        the gyro network whose dispersion we compute
    eps : float
        resolution for discerning if value of connectivity matrix is nonzero

    Returns
    -------
    lambda function
    """
    return lambda kvec: dynamical_matrix_kspace(kvec, tlat, eps=eps)


def dynamical_matrix_kspace(kvec, tlat, eps=1e-9, basis=None):
    """Construct the dynamical matrix for the given TwistyLattice for the given wavevector kvec.

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    tlat : TwistyLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------

    """
    # Determine if the network has twisted boundary conditions
    if 'theta_twist' in tlat.lp:
        thetatwist = tlat.lp['theta_twist']
    else:
        thetatwist = None
    if 'phi_twist' in tlat.lp:
        phitwist = tlat.lp['phi_twist']
    else:
        phitwist = None

    notwist = thetatwist in [None, 0.] and phitwist in [None, 0.]

    # grab basis from lp if it is a key
    if basis is None and 'basis' in tlat.lp:
        basis = tlat.lp['basis']

    if basis in [None, 'XY']:
        '''Compute the dynamical matrix using the xy realspace positions in a simple Euclidean basis'''
        if tlat.bL is None:
            # Rest lengths of springs == distances between particles
            if notwist:
                # not twisted, no stretch, XY basis
                matrix = calc_matrix_kvec(kvec, tlat, eps=eps)
                # Using psi basis for now since it is the only one that works.
                # matrix = calc_kmatrix_psi(kvec, tlat, eps=eps)
                # outname = '/Users/npmitchell/Desktop/test/' + 'kx{0:0.2f}'.format(kvec[0]) +\
                #           'ky{0:0.2f}'.format(kvec[1])
                # leplt.plot_complex_matrix(matrix, name='dynamical_matrix', outpath=outname)
            else:
                # twisted, no stretch, XY basis
                print 'PV = ', tlat.lattice.PV
                print 'thetatwist = ', thetatwist
                print 'phitwist = ', phitwist
                if tlat.lp['periodic_strip']:
                    # All periodic bonds are twisted
                    matrix = calc_kmatrix_gyros_twist(kvec, tlat, eps=eps)
                else:
                    # First create thetaKL and phiKL, such that thetaKL[i, nn] is 1 if NL[i, nn] is rotated by theta as
                    # viewed by particle i and similar for phiKL[i, nn] rotated by phi.
                    if 'annulus' in tlat.lp['LatticeTop'] or tlat.lp['shape'] == 'annulus':
                        twistcut = np.array([0., 0., np.max(tlat.lattice.xy[:, 0]), 0.])
                        thetaKL = tfns.form_twistedKL(kvec, tlat, eps=eps)
                        phiKL = np.zeros_like(thetaKL, dtype=int)
                    else:
                        raise RuntimeError('Currently only have twistedKL set up for annular samples')

                    # Certain bonds are twisted, while the others are normal.
                    matrix = calc_kmatrix_gyros_twist_bonds(kvec, tlat, thetaKL, phiKL, eps=eps)
        else:
            # Rest lengths of springs != distances between particles
            matrix = calc_kmatrix_gyros_stretched(kvec, tlat, eps=eps)
    elif basis == 'psi':
        '''Compute the dynamical matrix using the basis of clockwise and counter-clockwise oscillating modes'''
        if notwist:
            matrix = calc_kmatrix_psi(kvec, tlat, eps=eps)
        else:
            raise RuntimeError('Have not handled twisted psi-basis case yet')

    if 'immobile_boundary' in tlat.lp:
        if tlat.lp['immobile_boundary']:
            boundary = tlat.lattice.get_boundary()
            for ind in boundary:
                matrix[2 * ind, :] = 0
                matrix[2 * ind + 1, :] = 0
    return matrix


def calc_matrix_kvec(kvec, tlat, eps=1e-11):
    """Compute the dynamical matrix for d psi /dt = D psi. This code is intended for periodic boundary conditions since
    it is in kspace
    Example usage: python gyro_lattice_class.py -LT hexagonal -N 5 -periodic_strip -dispersion

    Parameters
    ----------
    kvec : 1 x 2 float array
        The values of k (wavenumber) over which to compute the dynamical matrix
    tlat : TwistyLattice instance
        The gyro network to consider
    eps : float
        threshold for considering an element in KL to be a real connection


    Returns
    -------

    """
    # Extract essentials from tlat
    xy = tlat.lattice.xy
    NL, KL = tlat.lattice.NL, tlat.lattice.KL
    KK = tlat.KK
    GG = tlat.GG
    CC = tlat.CC
    PVx, PVy = tlat.lattice.PVx, tlat.lattice.PVy

    # print 'xy = ', np.shape(xy)
    # print 'NL = ', np.shape(NL)
    # print 'KL = ', np.shape(KL)
    # print 'PVx = ', np.shape(PVx)
    # print 'PVy = ', np.shape(PVy)
    # BL = tlat.lattice.BL
    # import lepm.plotting.network_visualization as netvis
    # netvis.movie_plot_2D(xy, BL, PVx=PVx, PVy=PVy, NL=NL, KL=KL, show=True, bondcolor='k')
    # sys.exit()

    try:
        NP_total, NN = tlat.NL.shape
        NP = len(tlat.inner_indices)
    except ValueError:
        '''There is only one particle.'''
        NP_total, NP, NN = 1, 1, 0

    # Unpack periodic boundary vectors
    if PVx is None and PVy is None:
        PVx = np.zeros((NP_total, NN), dtype=float)
        PVy = np.zeros((NP_total, NN), dtype=float)
    elif PVx is None or PVy is None:
        raise RuntimeError('Either PVx or PVy is None, but the other is not.')

    print 'Constructing dynamical matrix...'
    ##################################################
    mm = np.zeros((4 * NP, 4 * NP), dtype=complex)

    xy_in = list(tlat.xy[tlat.inner_indices])
    xy_list = [list(tlat.xy[i]) for i in range(len(tlat.xy))]
    xy_in_list = [list(xy_in[i]) for i in range(len(xy_in))]

    # ignores the outer particles
    if 'bcs' in tlat.lp:
        if tlat.lp['bcs'] == 'free':
            bc_f = True
        else:
            bc_f = False
    else:
        bc_f = False

    in_count = 0
    for ii in range(NP_total):
        for nn in range(NN):
            if ii in tlat.inner_indices:
                ni = int(tlat.NL[ii, nn])  # the number of the gyroscope ii is connected to
                k = abs(tlat.KL[ii, nn])  # true connection or not
                # stretch constant for this connection
                omk = KK[ii, nn]
                # twist constant for this connection
                omc = CC[ii, nn]
                # twist-stretch coupling constant for this connection
                omg = GG[ii, nn]

                if abs(k) > eps:
                    # There is a true connection, so update dynamical matrix
                    # if len(dispersion) > 1:
                    #     disp = 1. / (1. + dispersion[i])
                    # else:
                    #     disp = 1.
                    # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                    diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                    diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                    alphaij = np.arctan2(diffy, diffx)
                    if tlat.lp['scale_interactions']:
                        rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)
                        omk /= rij_mag
                        omg /= rij_mag
                        omc /= rij_mag

                    if not tlat.lp['periodicBC']:
                        # free boundary conditions: apply kvector to all displacements in real space
                        kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                    else:
                        if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
                            kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                        else:
                            kfactor = 1.0
                    # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))

                    Cos = np.cos(alphaij)
                    Sin = np.sin(alphaij)
                    Cos2 = Cos ** 2
                    Sin2 = Sin ** 2
                    CosSin = Cos * Sin

                    # print 'twisty_kspacefns: tlat.inner_indices = ', tlat.inner_indices
                    if ni in tlat.inner_indices:
                        ni_new = xy_in_list.index(xy_list[ni])
                        iii = 2 * in_count
                        npnp = 2 * NP
                        jjj = 2 * ni_new
                        # (x components)
                        mm[iii, iii] += omk * Cos2
                        mm[iii, iii + 1] += omk * CosSin
                        mm[iii, jjj] += -omk * Cos2 * kfactor
                        mm[iii, jjj + 1] += -omk * CosSin * kfactor
                        mm[iii, npnp + iii] = -omg * CosSin
                        mm[iii, npnp + iii + 1] = omg * Cos2
                        mm[iii, npnp + jjj] = omg * CosSin * kfactor
                        mm[iii, npnp + jjj + 1] = -omg * Cos2 * kfactor

                        # (y components)
                        mm[iii + 1, iii] += omk * CosSin
                        mm[iii + 1, iii + 1] += omk * Sin2
                        mm[iii + 1, jjj] += -omk * CosSin * kfactor
                        mm[iii + 1, jjj + 1] += -omk * Sin2 * kfactor
                        # Add twist restoring coefficients
                        mm[iii + 1, npnp + iii] = -omg * Sin2
                        mm[iii + 1, npnp + iii + 1] = omg * CosSin
                        mm[iii + 1, npnp + jjj] = omg * Sin2 * kfactor
                        mm[iii + 1, npnp + jjj + 1] = -omg * CosSin * kfactor

                        # Fill in 2*NP_total + in_count row
                        mm[npnp + iii, iii] += -omg * CosSin
                        mm[npnp + iii, iii + 1] += -omg * Sin2
                        mm[npnp + iii, jjj] += omg * CosSin * kfactor
                        mm[npnp + iii, jjj + 1] += omg * Sin2 * kfactor
                        mm[npnp + iii, npnp + iii] += omc * Sin2
                        mm[npnp + iii, npnp + iii + 1] += -omc * CosSin
                        mm[npnp + iii, npnp + jjj] += -omc * Sin2 * kfactor
                        mm[npnp + iii, npnp + jjj + 1] += omc * CosSin * kfactor
                        # row 2*(NPtotal + incount) + 1 which is for Fz_i
                        mm[npnp + iii + 1, iii] += omg * Cos2
                        mm[npnp + iii + 1, iii + 1] += omg * CosSin
                        mm[npnp + iii + 1, jjj] += -omg * Cos2 * kfactor
                        mm[npnp + iii + 1, jjj + 1] += -omg * CosSin * kfactor
                        mm[npnp + iii + 1, npnp + iii] += -omc * CosSin
                        mm[npnp + iii + 1, npnp + iii + 1] += omc * Cos2
                        mm[npnp + iii + 1, npnp + jjj] += omc * CosSin * kfactor
                        mm[npnp + iii + 1, npnp + jjj + 1] += -omc * Cos2 * kfactor
                    else:
                        # The particle is interacting with the stationary outer particle
                        # Add the contribution of that stationary particle to the pinning of the in_count particle here.
                        # The stationary particle is not displaced, so no need to update those 'off-diagonal' ni terms
                        if not bc_f:
                            mm[2 * in_count, 2 * in_count] += omk * CosSin
                            mm[2 * in_count, 2 * in_count + 1] += -omk * Sin2
                            mm[2 * in_count, 2 * NP + 2 * in_count] += -omg * CosSin
                            mm[2 * in_count, 2 * NP + 2 * in_count + 1] += -omg * Cos2
                            mm[2 * in_count + 1, 2 * in_count] += omk * CosSin
                            mm[2 * in_count + 1, 2 * in_count + 1] += omk * Sin2
                            mm[2 * in_count + 1, 2 * in_count] += -omg * Sin2
                            mm[2 * in_count + 1, 2 * in_count + 1] += omg * CosSin
                            mm[2 * NP + 2 * in_count, 2 * in_count] += -omg * CosSin
                            mm[2 * NP + 2 * in_count, 2 * in_count + 1] += -omg * Sin2
                            mm[2 * NP + 2 * in_count, 2 * NP + 2 * in_count] += omc * Sin2
                            mm[2 * NP + 2 * in_count, 2 * NP + 2 * in_count + 1] += -omc * CosSin
                            mm[2 * NP + 2 * in_count + 1, 2 * in_count] += omg * Cos2
                            mm[2 * NP + 2 * in_count + 1, 2 * in_count + 1] += -omg * CosSin
                            mm[2 * NP + 2 * in_count + 1, 2 * NP + 2 * in_count] += -omc * CosSin
                            mm[2 * NP + 2 * in_count + 1, 2 * NP + 2 * in_count + 1] += omc * Cos2

        if ii in tlat.inner_indices:
            # update the indix for which the inner indices list
            in_count += 1

    # print 'mm = ', mm
    # leplt.plot_complex_matrix(mm, show=True)
    # sys.exit()
    return mm

    # ##################################################
    # for i in range(NP):
    #     raise RuntimeError('working on this here')
    #     # pinning/gravitational matrix
    #     M2[2 * i, 2 * i + 1] = - omg
    #     M2[2 * i + 1, 2 * i] = omg
    #
    #     for nn in range(NN):
    #         # the index of the gyroscope that is connected to gyro ii (ni is a neighbor)
    #         ni = NL[i, nn]
    #         # true connection?
    #         k = KL[i, nn]
    #         # spring frequency for this connection
    #         omk = OmK[i, nn]
    #
    #         if abs(k) > eps:
    #             # There is a true connection, so update dynamical matrix
    #             # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
    #             # Compute factor from fourier transform
    #             diffx = xy[ni, 0] - xy[i, 0] + PVx[i, nn]
    #             diffy = xy[ni, 1] - xy[i, 1] + PVy[i, nn]
    #             if np.abs(PVx[i, nn]) > eps or np.abs(PVy[i, nn]) > eps:
    #                 kfactor = np.exp(1j * (PVx[i, nn] * kvec[0] + PVy[i, nn] * kvec[1]))
    #             else:
    #                 kfactor = 1.0
    #
    #             # Add kfactor to each bond? Wrong
    #             # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
    #             # Add kfactor to all bonds based on absolute position?
    #             # kfactor = np.exp(1j * ((xy[ni, 0] + PVx[i, nn]) * kvec[0] + (xy[ni, 1] + PVy[i, nn]) * kvec[1]))
    #             alphaij = np.arctan2(diffy, diffx)
    #
    #             # # What is this for?
    #             # if k == -2:  # will only happen on first or last gyro in a line
    #             #     if i == 0 or i == (NP - 1):
    #             #         print i, '--> NL=-2 for this particle'
    #             #         yy = np.where(KL[i] == 1)
    #             #         dx = xy[NL[i, yy], 0] - xy[NL[i, yy], 0]
    #             #         dy = xy[NL[i, yy], 1] - xy[NL[i, yy], 1]
    #             #         al = (np.arctan2(dy, dx)) % (2 * np.pi)
    #             #         alphaij = np.pi - al
    #             #         if i == 1:
    #             #             alphaij = np.pi - (45. * np.pi / 180.)
    #             #         else:
    #             #             alphaij = - (45. * np.pi / 180.)
    #
    #             Cos = np.cos(alphaij)
    #             Sin = np.sin(alphaij)
    #
    #             if abs(Cos) < eps:
    #                 Cos = 0.0
    #
    #             if abs(Sin) < eps:
    #                 Sin = 0.0
    #
    #             # Invoke kvector here
    #             Cos2 = Cos ** 2
    #             Sin2 = Sin ** 2
    #             CosSin = Cos * Sin
    #
    #             # (x components)
    #             M1[2 * i, 2 * i] += -omk * CosSin  # dxi - dxi
    #             M1[2 * i, 2 * i + 1] += -omk * Sin2  # dxi - dyi
    #             M1[2 * i, 2 * ni] += omk * CosSin * kfactor  # dxi - dxj
    #             M1[2 * i, 2 * ni + 1] += omk * Sin2 * kfactor  # dxi - dyj
    #
    #             # (y components)
    #             M1[2 * i + 1, 2 * i] += omk * Cos2  # dyi - dxi
    #             M1[2 * i + 1, 2 * i + 1] += omk * CosSin  # dyi - dyi
    #             M1[2 * i + 1, 2 * ni] += -omk * Cos2 * kfactor  # dyi - dxj
    #             M1[2 * i + 1, 2 * ni + 1] += -omk * CosSin * kfactor  # dyi - dyj
    #
    # # self.pin_array.append(2*pi*1*extra_factor)
    # # Assumes that b=0, c=1 so that:
    # # (-1)**c adot =  spring* (-1)**(b+1) (-Fy,Fx)  - pin (dY,-dX)
    # # (dXdot, dYdot) = - spring (-Fy,Fx)  - pin (dY,-dX)
    # matrix = 0.5 * M1 + M2
    #
    # return matrix


def calc_kmatrix_gyros_twist(kvec, tlat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    tlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_twist_bonds(kvec, tlat, thetaKL, phiKL, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    tlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_gyros_stretched(kvec, tlat, eps=1e-11):
    """

    Parameters
    ----------
    kvec
    tlat
    eps

    Returns
    -------

    """
    raise RuntimeError('Have not written yet')
    return None


def calc_kmatrix_psi(kvec, tlat, eps=1e-11):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a gyro + spring system,
    based loosely on chern_functions_gen.make_M().
    Not working yet...

    Parameters
    ----------
    kvec : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    tlat : TwistyLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    raise RuntimeError('Have not written yet!')
    # First use tlat to create (angs, num_neis, bls, tvals, ons)
    #
    # angs : list
    #     each row represents a site in the lattice.  Each entry in the row represents the angles to that site's
    #     neighbors
    # num_nei : list or array (num_sites x num_sites)
    #     Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be
    #     num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
    # bls : len(angs) x  float array or int
    #     bondlengths, with dimensions equal to dimensions of angs.
    #     default value is an int, -1, indicating that all bond lengths are 1
    # tvals : len(angs) x 1 float array or int
    #     dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
    # ons : array (dimension = num_sites per unit cell)
    #     represents omega_g
    xy = tlat.lattice.xy
    NL, KL = tlat.lattice.NL, tlat.lattice.KL
    num_sites, NN = np.shape(NL)
    Omg, OmK = tlat.Omg, tlat.OmK
    PVx, PVy = tlat.lattice.PVx, tlat.lattice.PVy
    if PVx is None or PVy is None:
        PVx = np.zeros_like(NL, dtype=float)
        PVy = np.zeros_like(NL, dtype=float)

    # num_sites is the total number of particles
    mm = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    # Go through each site and fill in rows i and NP + i for that site (psi_L and psi_R)
    for ii in range(num_sites):
        omg = Omg[ii]  # grav frequency for this particle

        # pinning/gravitational matrix -- note: will divide later by factor of -2
        mm[ii, ii] += 2. * omg
        mm[num_sites + ii, num_sites + ii] += -2. * omg

        for nn in range(NN):
            # the index of the twisty scope i is connected to (particle j)
            ni = NL[ii, nn]
            # true connection?
            k = KL[ii, nn]
            # spring frequency for this connection
            omk = OmK[ii, nn]

            if abs(k) > eps:
                # We index PVx as [i,nn] since it is the same shape as NL (and corresponds to its indexing)
                diffx = xy[ni, 0] - xy[ii, 0] + PVx[ii, nn]
                diffy = xy[ni, 1] - xy[ii, 1] + PVy[ii, nn]
                alphaij = np.arctan2(diffy, diffx)

                # Form kfactor
                if np.abs(PVx[ii, nn]) > eps or np.abs(PVy[ii, nn]) > eps:
                    kfactor = np.exp(1j * (PVx[ii, nn] * kvec[0] + PVy[ii, nn] * kvec[1]))
                    # kfactor = np.exp(1j * (diffx * kvec[0] + diffy * kvec[1]))
                else:
                    kfactor = 1.0

                # Create phase factors
                expi2t = np.exp(1j * 2. * alphaij)
                exp_negi2t = np.exp(-1j * 2. * alphaij)

                # (psi_L psi_L components)
                # add top left chunk: -/+1/2 Omk, note: will divide by -2 later
                mm[ii, ii] += omk
                mm[ii, ni] += -omk * kfactor

                # (psi_L psi_R components) top right chunk
                mm[ii, ii + num_sites] += omk * expi2t
                mm[ii, ni + num_sites] += -omk * expi2t * kfactor

                # (psi_R psi_L components) bottom left chunk
                mm[ii + num_sites, ii] += -omk * exp_negi2t
                mm[ii + num_sites, ni] += omk * exp_negi2t * kfactor

                # (psi_R psi_R components) bottom right chunk
                mm[ii + num_sites, ii + num_sites] += -omk
                mm[ii + num_sites, ni + num_sites] += omk * kfactor

    return -0.5 * mm * (-1j)


def dynamical_matrix_kspace_unitcell(k, tlat, eps=1e-8):
    """Create a Hamiltonian matrix to diagonalize describing the energetics of a twisty  + spring system,
    based on chern_functions_gen.make_M().
    I think this is in psi basis...

    Parameters
    ----------
    k : 1 x 2 float array
        The wavenumber vector at which to compute the dynamical matrix
    tlat : TwistyLattice instance
        The network for which to compute the dynamical matrix
    eps : float
        Threshold below which to ignore elements of KL

    Returns
    -------
    """
    raise RuntimeError('Have not written yet!')
    # First use tlat to create (angs, num_neis, bls, tvals, ons)
    #
    # angs : list
    #     each row represents a site in the lattice.  Each entry in the row represents the angles to that site's
    #     neighbors
    # num_nei : list or array (num_sites x num_sites)
    #     Tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice would be
    #     num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.
    # bls : len(angs) x  float array or int
    #     bondlengths, with dimensions equal to dimensions of angs.
    #     default value is an int, -1, indicating that all bond lengths are 1
    # tvals : len(angs) x 1 float array or int
    #     dimension equal to number of different kinds of springs in unit cell x 1.  represents omega_k
    # ons : array (dimension = num_sites per unit cell)
    #     represents omega_g
    if tlat.unit_cell is None:
        unitcell = tlat.get_unitcell()
    else:
        unitcell = tlat.unit_cell
    if unitcell is None:
        raise RuntimeError('Network has no stored unit cell')

    angs = unitcell['angs']
    num_neis = unitcell['num_nei']
    bls = unitcell['bls']
    tvals = unitcell['tvals']
    ons = unitcell['ons']

    ons = list(ons)

    # num_sites is the total number of particles
    num_sites = len(angs)
    M = np.zeros([2 * num_sites, 2 * num_sites], dtype='complex128')

    if bls == -1:
        bls = np.ones_like(angs)
    if tvals == -1:
        tvals = np.ones_like(angs)
    if ons == 1:
        ons = np.ones(num_sites)

    for i in range(len(M)):
        index = i % num_sites
        angs_for_row = angs[index]
        bls_for_row = bls[index]
        num_neis_row = num_neis[index]
        # num_bonds = len(angs[index])

        tv = tvals[index]
        num_bonds = sum(tv)
        # print 'num bonds', num_bonds

        # indices to keep track of what elements of dynamical matrix have already been filled
        fill_count = 0
        s_fill_count = 0
        for j in range(len(M)):
            if i == j:
                # Onsite term, on the diagonal of the dynamical matrix
                # For the top left chunk of the dynamical matrix, add positive term, bottom right add negative
                if i < num_sites:
                    # note that this will be divided by two later
                    M[i, j] = num_bonds + 2 * ons[index]
                else:
                    # note that this will be divided by two later
                    M[i, j] = - num_bonds - 2 * ons[index]
            else:
                ii = j % num_sites
                num_nei = num_neis_row[ii]
                # num_nei tells how many neighbors of on each kind of sublattice.  For example a honeycomb lattice
                # would be num_nei = [[0,3], [3,0]] because each point has 3 neighbors of the other lattice type.

                # Determine if we are in the first (upper left) block of the matrix
                if i < num_sites and j < num_sites:
                    for l in range(num_nei):
                        M[i, j] += - tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif i >= num_sites and j >= num_sites:
                    for l in range(num_nei):
                        M[i, j] += tv[fill_count] * \
                                   np.exp(1j * np.dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                        fill_count += 1
                elif (i < num_sites) and (j >= num_sites):
                    if j == num_sites + i:
                        M[i, j] = sum([tv[u] * ang_fac(angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            M[i, j] += -tv[s_fill_count] * ang_fac(angs_for_row[s_fill_count]) * \
                                       np.exp(1j * np.dot(k, vec(angs_for_row[s_fill_count],
                                                                 bls_for_row[s_fill_count])))
                            s_fill_count += 1
                elif (i >= num_sites) and (j < num_sites):
                    if j == (num_sites + i) % num_sites:
                        M[i, j] = -sum([tv[u] * ang_fac(-angs_for_row[u]) for u in range(len(angs_for_row))])
                    else:
                        for l in range(num_nei):
                            M[i, j] += tv[s_fill_count] * ang_fac(-angs_for_row[s_fill_count]) * \
                                       np.exp(
                                           1j * np.dot(k, vec(angs_for_row[s_fill_count], bls_for_row[s_fill_count])))
                            s_fill_count += 1

    return -0.5 * M
