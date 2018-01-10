import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import numpy as np
import lepm.plotting.science_plot_style as sps
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl
import heapq
import lepm.line_segments as linesegs
import lepm.dataio as dio

"""Auxiliary functions for use in the gyro_lattice_class module, for plotting things.
"""


def test_lorentzians(eps, omega):
    """Plot a bunch of Lorentzians with different epsilon values to get a sense of how they look."""
    curvs = np.zeros((len(eps), len(omega)), dtype=float)
    ii = 0
    for ii in range(len(eps)):
        curvs[ii] = eps[ii] / (np.pi * (omega ** 2 + eps[ii] ** 2))
        ii += 1

    plt.plot(omega, curvs.T)
    plt.show()


def draw_ldos_plots(eigval, ldos, glat, outdir=None, FSFS=12):
    """

    Parameters
    ----------
    eigval : N x 1 float array
        eigenvalues
    ldos : NP x NP float array
        local density of states, only defined for positive omega values and xy components are combined
    outdir : str
        The dir in which to save the ldos plots. If none, uses pwd
    FSFS : int
        fontsize for labels, etc

    Returns
    -------

    """
    if outdir is None:
        print 'Outputting images in current working directory...'
        outdir = './'
    ipr = glat.get_ipr()

    vmax = min(np.mean(ldos.ravel()) + 4. * np.std(ldos.ravel()), 0.8 * np.max(ldos.ravel()))

    # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    # cb = plt.colorbar(ax1, cax = cbaxes)

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1./ipr)[-1])) / 10.
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
    fig, DOS_ax, ax = leplt.initialize_lattice_DOS_header_plot(eigval[inds], glat.lattice.xy, glat.lattice.NL,
                                                               glat.lattice.KL, sim_type='gyro',
                                                               preset_cbar=True, cbar_pos=[0.82, 0.78, 0.012, 0.15],
                                                               colorV=1./ipr[inds], colormap='viridis_r',
                                                               norm=None,
                                                               facecolor='#80D080', nbins=75, fontsize=FSFS,
                                                               vmin=0.0, vmax=ipr_vmax,
                                                               linewidth=0,
                                                               make_cbar=True, climbars=True,
                                                               xlabel='Oscillation frequency $\omega/\Omega_g$',
                                                               ylabel=r'$D(\omega)$', ylabel_pad=20,
                                                               cax_label=r'$p$',
                                                               cbar_labelpad=10, ticks=[0., ipr_vmax],
                                                               cbar_nticks=None,
                                                               cbar_tickfmt=None,
                                                               orientation='vertical', cbar_orientation='vertical',
                                                               invert_xaxis=False, yaxis_tickright=False,
                                                               yaxis_ticks=None, ylabel_right=False, ylabel_rot=0,
                                                               DOSexcite=None, DOSexcite_color='r')

    glat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # Create twin axis on top of DOS with LDOS curve
    ldos_ax = DOS_ax.twinx()
    ldos_ax.set_xlim(0., 1.0)
    # ldos_ax.set_ylabel(r'$LDOS(\omega)$', rotation=180, fontsize=FSFS)
    # ldos_ax.yaxis.label.set_color('#CD5555')
    for tl in ldos_ax.get_yticklabels():
        tl.set_color('#CD5555')

    # cycle through each gyro -- plot local dos for that site
    dmyi = 0
    ytopdos = DOS_ax.get_ylim()[1]
    sortii = np.lexsort((glat.lattice.xy[:, 0], -np.floor(glat.lattice.xy[:, 1])))
    for site in glat.lattice.xy[sortii]:
        ldosii = ldos[sortii[dmyi]]
        scat = ax.scatter([site[0]], [site[1]], s=20, c='red')
        # leplt.colored_DOS_plot(eigval[inds], DOS_ax, 'gyro', alpha=1.0, colorV=1/ipr[inds], colormap='viridis_r',
        #                        norm=None, facecolor='#ffffff', nbins=75, fontsize=FSFS, cbar_ax=None, vmin=0.0,
        #                        vmax=ipr_vmax, linewidth=0., make_cbar=False, climbars=True,
        #                        xlabel='Oscillation frequency $\omega/\Omega_g$',
        #                        ylabel=r'$D(\omega)$', ylabel_pad=None, cax_label=r'$p$',
        #                        cbar_labelpad=10, ticks=[0., 0.5], cbar_nticks=None, cbar_tickfmt=None,
        #                        orientation='vertical', cbar_orientation='vertical',
        #                        invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False,
        #                        ylabel_rot=90, DOSexcite=None, DOSexcite_color='r')

        # Plot curve for LDOS
        # line = DOS_ax.plot(np.imag(eigval[int(len(eigval)*0.5):]), ytopdos * ldosii[int(len(eigval)*0.5):], 'k-')
        line = ldos_ax.plot(np.imag(eigval[int(len(eigval) * 0.5):]), ldosii, '-', color='#CD5555', lw=2)
        ldos_ax.set_xlim(1, 4)
        ldos_ax.set_ylim(0., 1.)

        # Save this image
        print 'saving image to ', outdir + 'ldos_site{0:08d}'.format(dmyi) + '.png'
        plt.savefig(outdir + 'ldos_site{0:08d}'.format(dmyi) + '.png')

        # cleanup
        scat.remove()
        line.pop(0).remove()
        del scat
        dmyi += 1

    return fig, ax


def draw_localization_plots(glat, localization, eigval, eigvect, outdir=None, alpha=1.0, fontsize=12):
    """Draw plots of each normal mode excitation with colormap behind it denoting the fitted exponential localization

    Parameters
    ----------
    glat : GyroLattice instance
        The gyro lattice whose modes we plot
    localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
    eigval : 2*N x 1 complex array
        eigenvalues of the matrix, sorted by order of imaginary components
    eigvect : typically 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    outdir : str or None
        the output directory to use for saving localization plots. If None, uses pwd as ouput dir
    alpha : float
        The opacity of the excitation ellipses
    fontsize : int
        The fontsize to use for labels, etc

    Returns
    -------
    fig, dos_ax, ax
    """
    if outdir is None:
        print 'Outputting images in current working directory...'
        outdir = './'
    ipr = glat.get_ipr()

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1. / ipr)[-1])) / 10.
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
    fig, dos_ax, ax = leplt.initialize_eigvect_DOS_header_plot(eigval[inds], glat.lattice.xy,
                                                               sim_type='gyro',
                                                               preset_cbar=True,
                                                               colorV=1. / ipr[inds], colormap='viridis_r',
                                                               norm=None,
                                                               facecolor='#80D080', nbins=75, fontsize=fontsize,
                                                               vmin=0.0, vmax=ipr_vmax,
                                                               linewidth=0,
                                                               make_cbar=True, climbars=True,
                                                               xlabel='Oscillation frequency $\omega/\Omega_g$',
                                                               ylabel=r'$D(\omega)$', ylabel_pad=20,
                                                               cax_label=r'$p$',
                                                               cbar_labelpad=10, ticks=[0., ipr_vmax],
                                                               cbar_nticks=None,
                                                               cbar_tickfmt=None,
                                                               orientation='vertical', cbar_orientation='vertical',
                                                               invert_xaxis=False, yaxis_tickright=False,
                                                               yaxis_ticks=None, ylabel_right=False, ylabel_rot=0,
                                                               DOSexcite=None, DOSexcite_color='r')
    if glat.lp['periodic_strip']:
        x0 = 0.1
        y0 = 0.2
        w = 0.8
        h = 0.2
        ax.set_position([x0, y0, w, h])
        eax_x0 = x0
        eax_y0 = y0 + h + 0.05
        eax_w = w
        eax_h = 0.2
        exp_ax = fig.add_axes([eax_x0, eax_y0, eax_w, eax_h])

    glat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # Get the xlims and ylims for plotting the exponential decay fit
    xlims = [np.min(glat.lattice.xy[:, 0]) - 1, np.max(glat.lattice.xy[:, 0]) + 1]
    ylims = [np.min(glat.lattice.xy[:, 1]) - 1, np.max(glat.lattice.xy[:, 1]) + 1]

    # If periodic, use LL to plot localization fit assuming periodic boundaries
    if glat.lp['periodicBC']:
        if glat.lp['periodic_strip']:
            LL = np.array([glat.lp['LL'][0]])
            magevecs = glat.calc_magevecs(eigvect)
        else:
            LL = glat.lp['LL']
    else:
        LL = None

    # cycle through each gyro -- plot local dos for that site
    dmyi = 0
    # just look at localized states
    # for en in (np.where(np.abs(localization[:, 3]) > 0.09)[0] + int(len(eigval)* 0.5)):
    # Look at all states
    for en in np.arange(int(len(eigval) * 0.5), len(eigval)):
        fig, [scat_fg, pp, f_mark, lines12_st] =\
            leplt.plot_eigvect_excitation(glat.lattice.xy, fig, dos_ax, ax, eigval, eigvect, en,
                                          marker_num=0, black_t0lines=True)
        locz = localization[en - int(len(eigval)*0.5)]
        localz_handle = plot_localization_heatmap(locz, ax, LL=LL, xlims=xlims, ylims=ylims, alpha=1.0)
        title = ax.get_title()
        ax.set_title(title + r', $|\psi(r)| \approx$ $($' +
                     '{0:0.3f}'.format(locz[2]) + r'$\pm$' + '{0:0.3f}'.format(locz[4]) + r'$)$ ' +
                     r'$\exp[($' + '{0:0.3f}'.format(locz[3]) + r'$\pm$' + '{0:0.3f}'.format(locz[6]) + '$)\, r]$',
                     fontsize=fontsize)

        if glat.lp['periodic_strip']:
            magind = en - len(eigval)*0.5
            plt_handle, fit_handle = plot_localization_1d(locz, glat.lattice.xy, magevecs[magind],
                                                          exp_ax, LL, xlims=xlims)
            exp_ax.plot([locz[0], locz[0]], [0, np.max(magevecs[magind])], 'g-')
            exp_ax.set_xlabel('Position, $x$', fontsize=12)
            exp_ax.set_ylabel('Excitation, $|\psi|$', fontsize=12)

        # Save this image
        print 'saving image to ', outdir + 'localization' + glat.lp['meshfn_exten'] + '_{0:06d}'.format(dmyi) + '.png'
        plt.savefig(outdir + 'localization' + glat.lp['meshfn_exten'] + '_{0:06d}'.format(dmyi) + '.png')

        # cleanup
        localz_handle.remove()
        scat_fg.remove()
        pp.remove()
        f_mark.remove()
        lines12_st.remove()
        if glat.lp['periodic_strip']:
            exp_ax.cla()

        del localz_handle
        del scat_fg
        del pp
        del f_mark
        del lines12_st
        dmyi += 1

    return fig, dos_ax, ax


def draw_edge_localization_plots(glat, elocz, eigval, eigvect, outdir=None, alpha=1.0, fontsize=12):
    """Draw plots of each normal mode excitation with colormap behind it denoting the fitted exponential localization

    Parameters
    ----------
    glat : GyroLattice instance
        The gyro lattice whose modes we plot
    elocz : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
        edge localization fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
    eigval : 2*N x 1 complex array
        eigenvalues of the matrix, sorted by order of imaginary components
    eigvect : typically 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    outdir : str or None
        the output directory to use for saving localization plots. If None, uses pwd as ouput dir
    alpha : float
        The opacity of the excitation ellipses
    fontsize : int
        The fontsize to use for labels, etc

    Returns
    -------
    fig, dos_ax, ax
    """
    if outdir is None:
        print 'Outputting images in current working directory...'
        outdir = './'
    else:
        dio.ensure_dir(outdir)
    ipr = glat.get_ipr()

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1. / ipr)[-1])) / 10.
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
    fig, dos_ax, ax = leplt.initialize_eigvect_DOS_header_plot(eigval[inds], glat.lattice.xy,
                                                               sim_type='gyro',
                                                               preset_cbar=True,
                                                               colorV=1. / ipr[inds], colormap='viridis_r',
                                                               norm=None,
                                                               facecolor='#80D080', nbins=75, fontsize=fontsize,
                                                               vmin=0.0, vmax=ipr_vmax,
                                                               linewidth=0,
                                                               make_cbar=True, climbars=True,
                                                               xlabel='Oscillation frequency $\omega/\Omega_g$',
                                                               ylabel=r'$D(\omega)$', ylabel_pad=20,
                                                               cax_label=r'$p$',
                                                               cbar_labelpad=10, ticks=[0., ipr_vmax],
                                                               cbar_nticks=None,
                                                               cbar_tickfmt=None,
                                                               orientation='vertical', cbar_orientation='vertical',
                                                               invert_xaxis=False, yaxis_tickright=False,
                                                               yaxis_ticks=None, ylabel_right=False, ylabel_rot=0,
                                                               DOSexcite=None, DOSexcite_color='r')
    # Make axis for showing quality of fit
    x0 = 0.1
    y0 = 0.2
    w = 0.8
    h = 0.2
    ax.set_position([x0, y0, w, h])
    eax_w = 0.5
    eax_h = 0.15
    eax_x0 = x0 + (w - eax_w) * 0.5
    eax_y0 = y0 + h * 1.7
    exp_ax = fig.add_axes([eax_x0, eax_y0, eax_w, eax_h])

    glat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # If fully periodic, escape
    if glat.lp['periodicBC'] and not glat.lp['periodic_strip']:
        # There are two periodic vectos, so there can be no boundary --> exit with error
        raise RuntimeError('Cannot compute distance to boundary in a periodic sample --> there is no boundary.')

    # Compute magnitude of excitation at each site
    magevecs = glat.calc_magevecs(eigvect)
    # Also get distance of each particle from boundary
    bndry_segs = glat.lattice.get_boundary_linesegs()
    if isinstance(bndry_segs, tuple):
        xydists = []
        for bsegii in bndry_segs:
            xydists.append(linesegs.mindist_from_multiple_linesegs(glat.lattice.xy, bsegii))

        # Convert list xydists into NP x 2 array
        xydists = np.dstack(tuple(xydists))[0]
    else:
        # There is only one boundary, so use lattice boundary to connect consecutive linesegments
        xydists = linesegs.mindist_from_multiple_linesegs(glat.lattice.xy, bndry_segs)

    # Check if there are multiple boundaries
    multiple_boundaries = len(np.shape(xydists)) > 1

    # Get the xlims and ylims for plotting the exponential decay fit
    xlims_fit = [-0.1, np.max(xydists.ravel()) + 1]
    xlims = [np.min(glat.lattice.xy[:, 0]) - 1, np.max(glat.lattice.xy[:, 0]) + 1]
    ylims = [np.min(glat.lattice.xy[:, 1]) - 1, np.max(glat.lattice.xy[:, 1]) + 1]

    # defining dmyi to be -1 so that the zeroth frame (first one saved) gets done twice to fix formatting
    dmyi = -1
    # Look at all states with positive frequency
    todo = np.hstack((np.array([int(len(eigval) * 0.5)]), np.arange(int(len(eigval) * 0.5), len(eigval))))
    for en in todo:
        if en % 10 == 0:
            print 'glat_plottingfns: eigval = ', dmyi, '/', len(todo)
        # magind is the index of current eigval as seen by an array that knows only about the top half of
        # the eigenvectors.
        magind = int(en - len(eigval) * 0.5)

        fig, [scat_fg, pp, f_mark, lines12_st] =\
            leplt.plot_eigvect_excitation(glat.lattice.xy, fig, dos_ax, ax, eigval, eigvect, en,
                                          marker_num=0, black_t0lines=True)
        locz = elocz[magind]

        if multiple_boundaries:
            localz_handle = plot_edge_localization_heatmap_periodicstrip(locz, bndry_segs, ax, xlims=xlims,
                                                                         ylims=ylims, alpha=1.0)
            title = ax.get_title()
            # Draw the exponential localization fit
            plt_handle, fit_handle = plot_localization_dists(locz, xydists[:, int(locz[5] % 2)], magevecs[magind],
                                                             exp_ax, xlims=xlims_fit)
        else:
            # assuming no periodicity
            # Plot heat map for decay from min distance to any boundary segments
            localz_handle = plot_edge_localization_heatmap(locz, bndry_segs, ax, xlims=xlims, ylims=ylims, alpha=1.0)
            title = ax.get_title()
            # Draw the exponential localization fit
            # If there are multiple boundaries, pick out the one that distances are fit from
            if len(np.shape(xydists)) > 1:
                xydists_temp = xydists[:, int(locz[5] % 2)]
            else:
                xydists_temp = xydists

            plt_handle, fit_handle = plot_localization_dists(locz, xydists_temp, magevecs[en], exp_ax, xlims=xlims_fit)

        ax.set_title(title + r', $|\psi(r)| \approx$ $($' +
                     '{0:0.3f}'.format(locz[0]) + r'$\pm$' + '{0:0.3f}'.format(locz[2]) + r'$)$ ' +
                     r'$\exp[($' + '{0:0.3f}'.format(locz[1]) + r'$\pm$' + '{0:0.3f}'.format(locz[4]) + '$)\, r]$',
                     fontsize=fontsize)

        # labels for the localization fit
        exp_ax.set_xlabel('Distance from boundary, $x$', fontsize=12)
        exp_ax.set_ylabel('Excitation, $|\psi|$', fontsize=12)

        # Save this image
        if en % 10 == 0:
            print 'saving image to ', outdir + 'localization' + glat.lp['meshfn_exten'] + \
                                      '_{0:06d}'.format(max(0, dmyi)) + '.png'
        plt.savefig(outdir + 'localization' + glat.lp['meshfn_exten'] + '_{0:06d}'.format(max(0, dmyi)) + '.png')

        # cleanup
        localz_handle.remove()
        scat_fg.remove()
        pp.remove()
        f_mark.remove()
        lines12_st.remove()
        exp_ax.cla()

        del localz_handle
        del scat_fg
        del pp
        del f_mark
        del lines12_st
        dmyi += 1

    return fig, dos_ax, ax


def plot_localization_1d(ev_localization, xy, magevecs, ax, LL, xlims=None):
    """Plot the excitations as a function of x and overlay the exponential decay fit to the data, for system which is
    periodic in x, but not in y.

    """
    plt_handle = ax.plot(xy[:, 0], np.abs(magevecs), 'b.')
    if xlims is None:
        xlims = 2 * np.max(np.abs(xy[:, 0]))
        xx = np.linspace(0, xlims, 1000)
    else:
        xx = np.linspace(xlims[0], xlims[1], 1000)
    lz = ev_localization
    dist = le.distancex_periodicstrip(np.dstack((xx, 0*xx))[0], lz[0:2], LL)
    heatdata = lz[2] * np.exp(lz[3] * dist)

    fit_handle = ax.plot(xx, heatdata, 'r-')
    if xlims is not None:
        ax.set_xlim(xlims)

    return plt_handle, fit_handle


def plot_localization_dists(ev_localization, xydists, magevecs, ax, xlims=None):
    """Plot the excitations as a function of x and overlay the exponential decay fit to the data, for system which is
        periodic in x, but not in y.
    """
    plt_handle = ax.plot(xydists, np.abs(magevecs), 'b.')
    if xlims is None:
        xlims = 2 * np.max(xydists)
        xx = np.linspace(0, xlims, 1000)
    else:
        xx = np.linspace(xlims[0], xlims[1], 1000)

    lz = ev_localization
    heatdata = lz[0] * np.exp(lz[1] * xx)

    fit_handle = ax.plot(xx, heatdata, 'r-')
    if xlims is not None:
        ax.set_xlim(xlims)

    return plt_handle, fit_handle


def plot_localization_heatmap(ev_localization, ax, LL=None, xlims=None, ylims=None, cmap='Oranges', alpha=1.0):
    """Plot the fitted exponential decay of excitations as a heatmap on the supplied axis.

    Parameters
    ----------
    ev_localization : 1 x 7 float array
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K for this eigenvector
    ax : matplotlib.pyplot axis instance
        The axis on which to plot the localization heatmap
    LL : float (if periodic in 1d), or tuple of 2 floats (list of 2 floats also accepted) if periodic in 2d
        spatial extent of periodic dimensions, if supplied. Otherwise, not treated as periodic
    xlims : list or tuple of two floats
        The min and max in x for the heatmap
    ylims : list or tuple of two floats
        The min and max in y for the heatmap
    cmap : colormap spec
        The colormap to use for the localization overlay

    Returns
    -------
    localz_handle : plotting handle
        The instance of the plotted heatmap
    """
    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()
    if isinstance(xlims, tuple):
        xlims = [xlims[0], xlims[1]]
    if isinstance(ylims, tuple):
        ylims = [ylims[0], ylims[1]]
    x = np.linspace(xlims[0], xlims[1], 1000)
    y = np.linspace(ylims[0], ylims[1], 1000)
    X, Y = np.meshgrid(x, y)
    lz = ev_localization

    if LL is not None:
        if isinstance(LL, float) or len(LL) == 1:
            # Assuming periodic in 1D, since LL is supplied and is of len = 1
            XY = np.dstack((X.ravel(), Y.ravel()))[0]
            dist = le.distancex_periodicstrip(XY, lz[0:2], LL).reshape(np.shape(X))
            heatdata = lz[2] * np.exp(lz[3] * dist)
        elif len(LL) == 2:
            # Assuming periodic in 2D, since LL is supplied and is of len = 2
            XY = np.dstack((X.ravel(), Y.ravel()))[0]
            dist = le.distance_periodic(XY, lz[0:2], LL).reshape(np.shape(X))
            heatdata = lz[2] * np.exp(lz[3] * dist)
        else:
            raise RuntimeError('LL must be either float, len(1) array, or len(2) tuple or array')
    else:
        heatdata = lz[2] * np.exp(lz[3] * np.sqrt((X - lz[0]) ** 2 + (Y - lz[1]) ** 2))

    localz_handle = ax.pcolormesh(X, Y, heatdata, cmap=cmap, zorder=0, vmin=0.)
    return localz_handle


def plot_edge_localization_heatmap(ev_localization, boundary_linesegs, ax, xlims=None, ylims=None, cmap='Oranges',
                                   alpha=1.0):
    """Plot the fitted exponential decay from the defined boundary/edge of excitations as a heatmap on the supplied
    axis.

    Parameters
    ----------
    ev_localization : 1 x 5 float array
        fit details for exponential decaying from the edge:
        A, K, uncertainty_A, covariance_AK, uncertainty_K for this eigenvector
    boundary_linesegs : #vertices on boundary x 4 float array
        The points defining the (possibly closed) path boundary from which exponential confinement has been measured.
        Could be two linsegments defining the top and bottom of the sample if periodicstrip.
    ax : matplotlib.pyplot axis instance
        The axis on which to plot the localization heatmap
    xlims : list or tuple of two floats
        The min and max in x for the heatmap
    ylims : list or tuple of two floats
        The min and max in y for the heatmap
    cmap : colormap spec
        The colormap to use for the localization overlay

    Returns
    -------
    localz_handle : plotting handle
        The instance of the plotted heatmap
    """
    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()
    if isinstance(xlims, tuple):
        xlims = [xlims[0], xlims[1]]
    if isinstance(ylims, tuple):
        ylims = [ylims[0], ylims[1]]
    x = np.linspace(xlims[0], xlims[1], 1000)
    y = np.linspace(ylims[0], ylims[1], 1000)
    X, Y = np.meshgrid(x, y)
    lz = ev_localization

    # Whether or periodic in 1D, find distance of each gridpt XY from the boundary that is supplied
    XY = np.dstack((X.ravel(), Y.ravel()))[0]
    dist = linesegs.mindist_from_multiple_linesegs(XY, boundary_linesegs).reshape(np.shape(X))
    heatdata = lz[0] * np.exp(lz[1] * dist)

    localz_handle = ax.pcolormesh(X, Y, heatdata, cmap=cmap, zorder=0, vmin=0.)
    return localz_handle


def plot_edge_localization_heatmap_periodicstrip(ev_localization, bndry_linsegs, ax, xlims=None, ylims=None,
                                                 cmap='Oranges', alpha=1.0):
    """Plot the fitted exponential decay from the defined boundary/edge of excitations as a heatmap on the supplied
    axis, for periodic_strip boundary conditions

    Parameters
    ----------
    ev_localization : 1 x 6 float array
        fit details for exponential decaying from the edge:
        A, K, uncertainty_A, covariance_AK, uncertainty_K, topbottom_indicator for this eigenvector
    bndry_linsegs : tuple of (#vertices on boundary x 4) float arrays
        The points defining each closed path boundary from which exponential confinement has been measured.
        These are two contiguous sets of linsegments defining the top and bottom of the periodicstrip sample.
    ax : matplotlib.pyplot axis instance
        The axis on which to plot the localization heatmap
    xlims : list or tuple of two floats
        The min and max in x for the heatmap
    ylims : list or tuple of two floats
        The min and max in y for the heatmap
    cmap : colormap spec
        The colormap to use for the localization overlay

    Returns
    -------
    localz_handle : plotting handle
        The instance of the plotted heatmap
    """
    if xlims is None:
        xlims = ax.get_xlim()
    if ylims is None:
        ylims = ax.get_ylim()
    if isinstance(xlims, tuple):
        xlims = [xlims[0], xlims[1]]
    if isinstance(ylims, tuple):
        ylims = [ylims[0], ylims[1]]
    x = np.linspace(xlims[0], xlims[1], 1000)
    y = np.linspace(ylims[0], ylims[1], 1000)
    X, Y = np.meshgrid(x, y)
    lz = ev_localization

    # Whether or not periodic in 1D, find distance of each gridpt XY from the boundary that is supplied
    # Note that we take the (lz[5] % 2) element of bndry_linsegs, which gives us zeroth element for top, first
    # element for the bottom boundary, and the zeroth element (compare to top boundary) if excitation fit better to a
    # constant value (denoted by lz[5] = 2)
    XY = np.dstack((X.ravel(), Y.ravel()))[0]
    dist = linesegs.mindist_from_multiple_linesegs(XY, bndry_linsegs[int(lz[5] % 2)]).reshape(np.shape(X))
    heatdata = lz[0] * np.exp(lz[1] * dist)

    localz_handle = ax.pcolormesh(X, Y, heatdata, cmap=cmap, zorder=0, vmin=0.)
    return localz_handle


def plot_ill_dos(glat, dos_ax=None, cbar_ax=None, alpha=1.0, vmin=None, vmax=None, **kwargs):
    """
    Parameters
    ----------
    dos_ax : axis instance
        Axis on which to plot the Localization-colored DOS
    cbar_ax : axis instance or None
        axis to use for colorbar
    alpha : float
        opacity of the DOS added to the plot
    vmin : float
        minimum value for inverse localization length in colormap
    vmax : float
        maximum value for inverse localization length in colormap
    **kwargs: lepm.plotting.plotting.colored_DOS_plot() keyword arguments
        Excluding fontsize
    """
    # Register cmaps if necessary
    if 'viridis' not in plt.colormaps():
        lecmaps.register_colormaps()

    eigval = glat.get_eigval(attribute=False)

    # Load or compute localization
    localization = glat.get_localization(attribute=False)
    ill = np.zeros(len(eigval), dtype=float)
    print 'eigval = ', eigval
    print 'localization = ', localization
    print 'eigval = ', np.shape(eigval)
    print 'localization = ', np.shape(localization)
    ill[0:int(len(eigval) * 0.5)] = localization[:, 2][::-1]
    ill[int(len(eigval) * 0.5):len(eigval)] = localization[:, 2]
    # print 'gcollpfns: localization = ', localization
    # print 'gcollpfns: shape(localization) = ', np.shape(localization)

    # Now overlay the ipr
    if vmin is None:
        print 'setting vmin...'
        vmin = 0.0
    if dos_ax is None:
        print 'dos_ax is None, initializing...'
        fig, dos_ax, cax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro',
                                                                   alpha=alpha, colorV=ill,
                                                                   colormap='viridis',
                                                                   linewidth=0, cax_label=r'$\lambda^{-1}$',
                                                                   vmin=vmin, vmax=vmax, **kwargs)
    else:
        print 'gcollpfns: calling leplt.colored_DOS_plot...'
        dos_ax, cbar_ax, cbar, n, bins = \
            leplt.colored_DOS_plot(eigval, dos_ax, 'gyro', alpha=alpha, colorV=ill,
                                   cbar_ax=cbar_ax, colormap='viridis', linewidth=0,
                                   vmin=vmin, vmax=vmax, **kwargs)

    return dos_ax, cbar_ax


if __name__ == '__main__':
    eps = np.arange(0., 1.0, 0.05)
    omega = np.arange(-3, 3, 0.05)
    test_lorentzians(eps, omega)
