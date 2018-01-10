import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl
import heapq
import lepm.line_segments as linesegs
import lepm.dataio as dio

"""Auxiliary functions for use in the haldane_lattice_class module, for plotting things.
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


def construct_haldane_eigvect_DOS_plot(xy, fig, DOS_ax, eig_ax, eigval, eigvect, en, NL, KL, marker_num=0,
                                       color_scheme='default', sub_lattice=-1, normalization=None):
    """puts together lattice and DOS plots and draws normal mode ellipsoids on top

    Parameters
    ----------
    xy: array 2N x 3
        Equilibrium position of the gyroscopes
    fig :
        figure with lattice and DOS drawn
    DOS_ax:
        axis for the DOS plot
    eig_ax
        axis for the eigenvalue plot
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system.
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    en: int
        Number of the eigenvalue you are plotting

    Returns
    ----------
    fig :
        completed figure for normal mode

    [scat_fg, p, f_mark] :
        things to be cleared before next normal mode is drawn
        """
    s = leplt.absolute_sizer()

    plt.sca(DOS_ax)

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot
    (f_mark,) = plt.plot([ev, ev], plt.ylim(), '-r')

    NP = len(xy)

    im1 = np.imag(ev)
    re1 = np.real(ev)
    plt.sca(eig_ax)
    plt.title('Mode %d; $\Omega=( %0.6f + %0.6f i)$' % (en, re1, im1))

    # Preallocate ellipsoid plot vars
    angles_arr = np.zeros(NP)

    patch = []
    colors = np.zeros(NP + 2)

    x0s = np.zeros(NP)
    y0s = np.zeros(NP)

    mag1 = eigvect[en]
    if normalization is None:
        mag1 /= np.max(np.abs(mag1))
    else:
        mag1 *= normalization * float(len(xy))

    # Pick a series of times to draw out the ellipsoid
    time_arr = np.arange(81.0) * 2. * np.pi / float(abs(ev1) * 80)
    exp1 = np.exp(1j * ev1 * time_arr)
    cw = []
    ccw = []
    lines_1 = []
    for i in range(NP):
        x_disps = 0.5 * (exp1 * mag1[i]).real
        y_disps = 0.5 * (exp1 * mag1[i]).imag
        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        poly_points = np.array([x_vals, y_vals]).T
        polygon = Polygon(poly_points, True)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]

        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]

        # These are the black lines protruding from pivot point to current position
        lines_1.append([[xy[i, 0], x_vals[marker_num]], [xy[i, 1], y_vals[marker_num]]])

        mag = np.sqrt(x0 ** 2 + y0 ** 2)
        if mag > 0:
            anglez = np.arccos(x0 / mag)
        else:
            anglez = 0

        if y0 < 0:
            anglez = 2 * np.pi - anglez

        angles_arr[i] = anglez
        patch.append(polygon)

        if color_scheme == 'default':
            colors[i] = anglez
        else:
            if sub_lattice[i] == 0:
                colors[i] = 0
            else:
                colors[i] = np.pi
            ccw.append(i)

    colors[NP] = 0
    colors[NP + 1] = 2 * np.pi

    plt.yticks([])
    plt.xticks([])
    # this is the part that puts a dot a t=0 point
    scat_fg = eig_ax.scatter(x0s[cw], y0s[cw], s=s(.02), c='DodgerBlue')
    scat_fg2 = eig_ax.scatter(x0s[ccw], y0s[ccw], s=s(.02), c='Red', zorder=3)

    NP = len(xy)
    try:
        NN = np.shape(NL)[1]
    except IndexError:
        NN = 0

    z = np.zeros(NP)

    Rnorm = np.array([x0s, y0s, z]).T

    # Bond Stretches
    inc = 0
    stretches = np.zeros(4 * len(xy))
    for i in range(len(xy)):
        if NN > 0:
            for j, k in zip(NL[i], KL[i]):
                if i < j and abs(k) > 0:
                    n1 = float(linalg.norm(Rnorm[i] - Rnorm[j]))
                    n2 = linalg.norm(xy[i] - xy[j])
                    stretches[inc] = (n1 - n2)
                    inc += 1

    # For particles with neighbors, get list of bonds to draw by stretches
    test = list(np.zeros([inc, 1]))
    inc = 0
    xy = np.array([x0s, y0s, z]).T
    for i in range(len(xy)):
        if NN > 0:
            for j, k in zip(NL[i], KL[i]):
                if i < j and abs(k) > 0:
                    test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
                    inc += 1

    stretch = np.array(stretches[0:inc])

    # lines connect sites (bonds), while lines_12 draw the black lines from the pinning to location sites
    lines = [zip(x, y) for x, y in test]
    lines_12 = [zip(x, y) for x, y in lines_1]

    lines_st = LineCollection(lines, array=stretch, cmap='seismic', linewidth=8)
    lines_st.set_clim([-1. * 0.25, 1 * 0.25])
    lines_st.set_zorder(2)

    lines_12_st = LineCollection(lines_12, linewidth=0.8)
    lines_12_st.set_color('k')

    p = PatchCollection(patch, cmap='hsv', alpha=0.6)

    p.set_array(np.array(colors))
    p.set_clim([0, 2 * np.pi])
    p.set_zorder(1)

    # eig_ax.add_collection(lines_st)
    eig_ax.add_collection(lines_12_st)
    eig_ax.add_collection(p)
    eig_ax.set_aspect('equal')

    # erased ev/(2*pi) here npm 2016
    cw_ccw = [cw, ccw, ev]
    # print cw_ccw[1]

    return fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw


def draw_ldos_plots(eigval, ldos, hlat, outdir=None, FSFS=12):
    """

    Parameters
    ----------
    eigval : N x 1 float array
        eigenvalues
    ldos : NP x NP float array
        local density of states, only defined for positive omega values and xy components are combined

    Returns
    -------

    """
    if outdir is None:
        print 'Outputting images in current working directory...'
        outdir = './'
    ipr = hlat.get_ipr()

    vmax = min(np.mean(ldos.ravel()) + 4. * np.std(ldos.ravel()), 0.8 * np.max(ldos.ravel()))

    # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    # cb = plt.colorbar(ax1, cax = cbaxes)

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1./ipr)[-1])) / 10.
    inds = np.arange(int(0.5 * len(eigval)), len(eigval))
    fig, DOS_ax, ax = leplt.initialize_lattice_DOS_header_plot(eigval[inds], hlat.lattice.xy, hlat.lattice.NL,
                                                               hlat.lattice.KL, sim_type='haldane',
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

    hlat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # Create twin axis on top of DOS with LDOS curve
    ldos_ax = DOS_ax.twinx()
    ldos_ax.set_xlim(0., 1.0)
    # ldos_ax.set_ylabel(r'$LDOS(\omega)$', rotation=180, fontsize=FSFS)
    # ldos_ax.yaxis.label.set_color('#CD5555')
    for tl in ldos_ax.get_yticklabels():
        tl.set_color('#CD5555')

    # cycle through each site -- plot local dos for that site
    dmyi = 0
    ytopdos = DOS_ax.get_ylim()[1]
    sortii = np.lexsort((hlat.lattice.xy[:, 0], -np.floor(hlat.lattice.xy[:, 1])))
    for site in hlat.lattice.xy[sortii]:
        ldosii = ldos[sortii[dmyi]]
        scat = ax.scatter([site[0]], [site[1]], s=20, c='red')
        # leplt.colored_DOS_plot(eigval[inds], DOS_ax, 'haldane', alpha=1.0, colorV=1/ipr[inds], colormap='viridis_r',
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


def draw_localization_plots(hlat, localization, eigval, eigvect, outdir=None, alpha=1.0, fontsize=12):
    """Draw the eigenvectors with heatmaps of their averaged excitations overlaid. This is adapted from gyro plotting

    Parameters
    ----------
    localization
    eigval
    eigvect
    outdir
    alpha
    fontsize

    Returns
    -------

    """
    if outdir is None:
        print 'Outputting images in current working directory...'
        outdir = './'
    ipr = hlat.get_ipr()

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1. / ipr)[-1])) / 10.
    fig, dos_ax, ax = leplt.initialize_eigvect_DOS_header_plot(eigval, hlat.lattice.xy,
                                                               sim_type='haldane',
                                                               preset_cbar=True,
                                                               colorV=1. / ipr, colormap='viridis_r',
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

    hlat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # Get the xlims and ylims for plotting the exponential decay fit
    xlims = [np.min(hlat.lattice.xy[:, 0]) - 1, np.max(hlat.lattice.xy[:, 0]) + 1]
    ylims = [np.min(hlat.lattice.xy[:, 1]) - 1, np.max(hlat.lattice.xy[:, 1]) + 1]

    # If periodic, use LL to plot localization fit assuming periodic boundaries
    if hlat.lp['periodicBC']:
        LL = hlat.lp['LL']
    else:
        LL = None

    # cycle through each eigval -- plot local dos for that site
    dmyi = 0
    for en in np.arange(len(eigval)):
        fig, [scat_fg, pp, f_mark, lines12_st] = \
            plot_eigvect_excitation_haldane(hlat.lattice.xy, fig, dos_ax, ax, eigval, eigvect, en, marker_num=0,
                                            black_t0lines=True)
        locz = localization[dmyi]
        localz_handle = plot_localization_heatmap(locz, ax, LL=LL, xlims=xlims, ylims=ylims, alpha=1.0)
        title = ax.get_title()
        ax.set_title(title + r', $|\psi(r)| \approx$ $($' +
                     '{0:0.3f}'.format(locz[2]) + r'$\pm$' + '{0:0.3f}'.format(locz[4]) + r'$)$ ' +
                     r'$\exp[($' + '{0:0.3f}'.format(locz[3]) + r'$\pm$' + '{0:0.3f}'.format(locz[6]) + '$)\, r]$',
                     fontsize=fontsize)

        # Save this image
        print 'saving image to ', outdir + 'localization' + hlat.lp['meshfn_exten'] + '_{0:06d}'.format(dmyi) + '.png'
        plt.savefig(outdir + 'localization' + hlat.lp['meshfn_exten'] + '_{0:06d}'.format(dmyi) + '.png')

        # cleanup
        localz_handle.remove()
        scat_fg.remove()
        pp.remove()
        f_mark.remove()
        lines12_st.remove()
        del localz_handle
        del scat_fg
        del pp
        del f_mark
        del lines12_st
        dmyi += 1

    return fig, dos_ax, ax


def draw_edge_localization_plots(hlat, elocz, eigval, eigvect, outdir=None, alpha=1.0, fontsize=12):
    """Draw plots of each normal mode excitation with colormap behind it denoting the fitted exponential localization

    Parameters
    ----------
    hlat : HaldaneLattice instance
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
    ipr = hlat.get_ipr()

    # Get third largest value for ipr vmax
    # ipr_vmax = np.max(1. / ipr.sort())[3]
    ipr_vmax = float(np.floor(10 * heapq.nlargest(6, 1. / ipr)[-1])) / 10.
    fig, dos_ax, ax = leplt.initialize_eigvect_DOS_header_plot(eigval, hlat.lattice.xy,
                                                               sim_type='haldane',
                                                               preset_cbar=True,
                                                               colorV=1. / ipr, colormap='viridis_r',
                                                               norm=None,
                                                               facecolor='#80D080', nbins=75, fontsize=fontsize,
                                                               vmin=0.0, vmax=ipr_vmax,
                                                               linewidth=0,
                                                               make_cbar=True, climbars=True,
                                                               xlabel='Energy $E/t_1$',
                                                               ylabel=r'$D(E)$', ylabel_pad=20,
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

    hlat.lattice.plot_BW_lat(fig=fig, ax=ax, meshfn='none', save=False, close=False, axis_off=True, title='')

    # If periodic, use LL to plot localization fit assuming periodic boundaries
    if hlat.lp['periodicBC']:
        if hlat.lp['periodic_strip']:
            magevecs = np.abs(eigvect)
            # Also get distance of each particle from boundary
            bseg_tuple = hlat.lattice.get_boundary_linesegs()
            xydists = []
            for bsegs in bseg_tuple:
                xydists.append(linesegs.mindist_from_multiple_linesegs(hlat.lattice.xy, bsegs))

            # Convert list xydists into NP x 2 array
            xydists = np.dstack(tuple(xydists))[0]
        else:
            # There are two periodic vectos, so there can be no boundary --> exit with error
            raise RuntimeError('Cannot compute distance to boundary in a fully periodic sample ' +
                               '--> there is no boundary.')
    else:
        # Not periodic, no LL, so use lattice boundary to connect consecutive linesegments
        magevecs = hlat.calc_magevecs(eigvect)
        bndry_segs = hlat.lattice.get_boundary_linesegs()
        xydists = linesegs.mindist_from_multiple_linesegs(hlat.lattice.xy, bndry_segs)

    # Get the xlims and ylims for plotting the exponential decay fit
    xlims_fit = [-0.1, np.max(xydists.ravel()) + 1]
    xlims = [np.min(hlat.lattice.xy[:, 0]) - 1, np.max(hlat.lattice.xy[:, 0]) + 1]
    ylims = [np.min(hlat.lattice.xy[:, 1]) - 1, np.max(hlat.lattice.xy[:, 1]) + 1]

    dmyi = 0
    # Look at all states
    todo = np.hstack((np.array([0]), np.arange(len(eigval))))
    for en in todo:
        fig, [scat_fg, pp, f_mark, lines12_st] =\
            plot_eigvect_excitation_haldane(hlat.lattice.xy, fig, dos_ax, ax, eigval, eigvect, en,
                                            marker_num=0, black_t0lines=True)
        locz = elocz[en]
        if hlat.lp['periodic_strip']:
            localz_handle = plot_edge_localization_heatmap_periodicstrip(locz, bseg_tuple, ax, xlims=xlims, ylims=ylims,
                                                                         alpha=1.0)
            title = ax.get_title()

            # Draw the exponential localization fit
            plt_handle, fit_handle = plot_localization_dists(locz, xydists[:, int(locz[5] % 2)], magevecs[en], exp_ax,
                                                             xlims=xlims_fit)
        else:
            # assuming no periodicity
            localz_handle = plot_edge_localization_heatmap(locz, bndry_segs, ax, xlims=xlims, ylims=ylims, alpha=1.0)
            title = ax.get_title()

            # Draw the exponential localization fit
            plt_handle, fit_handle = plot_localization_dists(locz, xydists, magevecs[en], exp_ax, xlims=xlims_fit)

        ax.set_title(title + r', $|\psi(r)| \approx$ $($' +
                     '{0:0.3f}'.format(locz[0]) + r'$\pm$' + '{0:0.3f}'.format(locz[2]) + r'$)$ ' +
                     r'$\exp[($' + '{0:0.3f}'.format(locz[1]) + r'$\pm$' + '{0:0.3f}'.format(locz[4]) + '$)\, r]$',
                     fontsize=fontsize)

        # Add axis labels
        exp_ax.set_xlabel('Distance from boundary, $r$', fontsize=12)
        exp_ax.set_ylabel('Excitation, $|\psi|$', fontsize=12)

        # Save this image
        outname = outdir + 'localization_edge' + hlat.lp['meshfn_exten'] + '_{0:06d}'.format(en) + '.png'
        print 'saving image to ', outname
        plt.savefig(outname)

        # cleanup
        localz_handle.remove()
        scat_fg.remove()
        pp.remove()
        f_mark.remove()
        lines12_st.remove()
        if hlat.lp['periodic_strip']:
            exp_ax.cla()

        del localz_handle
        del scat_fg
        del pp
        del f_mark
        del lines12_st
        dmyi += 1

    return fig, dos_ax, ax


def plot_eigvect_excitation_haldane(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, marker_num=0,
                                    black_t0lines=False, mark_t0=True, title='auto', normalization=1., alpha=0.6,
                                    lw=1, zorder=10):
    """Draws normal mode ellipsoids on axis eig_ax.
    If black_t0lines is true, draws the black line from pinning site to positions. The difference from
    hlatpfns.construct_haldane_eigvect_DOS_plot() is doesn't draw lattice.

    Parameters
    ----------
    xy: array 2N x 3
        Equilibrium position of the gyroscopes
    fig :
        figure with lattice and DOS drawn
    dos_ax: matplotlib axis instance or None
        axis for the DOS plot. If None, ignores this input
    eig_ax : matplotlib axis instance
        axis for the eigenvalue plot
    eigval : array of dimension 2nx1
        Eigenvalues of matrix for system
    eigvect : array of dimension 2nx2n
        Eigenvectors of matrix for system.
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    en: int
        Number of the eigenvalue you are plotting
    marker_num : int in (0, 80)
        where in the phase (0 to 80) to call t=t0. This sets "now" for drawing where in the normal mode to draw
    black_t0lines : bool
        Draw black lines extending from the pinning site to the current site (where 'current' is determined by
        marker_num)

    Returns
    ----------
    fig : matplotlib figure instance
        completed figure for normal mode
    [scat_fg, pp, f_mark, lines12_st] :
        things to be cleared before next normal mode is drawn
    """
    s = leplt.absolute_sizer()

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot
    if dos_ax is not None:
        (f_mark,) = dos_ax.plot([np.real(ev), np.real(ev)], dos_ax.get_ylim(), '-r')

    NP = len(xy)

    im1 = np.real(ev)
    plt.sca(eig_ax)

    if title == 'auto':
        eig_ax.set_title('$\omega = %0.6f$' % im1)
    elif title is not None and title not in ['', 'none']:
        eig_ax.set_title(title)

    # Preallocate ellipsoid plot vars
    angles_arr = np.zeros(NP, dtype=float)

    patch = []
    colors = np.zeros(NP)
    x0s = np.zeros(NP, dtype=float)
    y0s = np.zeros(NP, dtype=float)
    mag1 = eigvect[en]

    # Pick a series of times to draw out the ellipsoid
    time_arr = np.arange(81) * 2 * np.pi / (np.abs(ev1) * 80)
    exp1 = np.exp(1j * ev1 * time_arr)

    # Normalization for the ellipsoids
    mag1 /= np.max(np.abs(mag1))
    mag1 *= normalization

    if black_t0lines:
        lines_1 = []
    else:
        lines_12_st = []

    for i in range(NP):
        x_disps = 0.5 * (exp1 * mag1[i]).real
        y_disps = 0.5 * (exp1 * mag1[i]).imag
        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        poly_points = np.array([x_vals, y_vals]).T
        polygon = Polygon(poly_points, True)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]

        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]

        if black_t0lines:
            # These are the black lines protruding from pivot point to current position
            lines_1.append([[xy[i, 0], x_vals[marker_num]], [xy[i, 1], y_vals[marker_num]]])

        mag = np.sqrt(x0 ** 2 + y0 ** 2)
        if mag > 0:
            anglez = np.arccos(x0 / mag)
        else:
            anglez = 0

        if y0 < 0:
            anglez = 2 * np.pi - anglez

        angles_arr[i] = anglez
        patch.append(polygon)
        colors[i] = anglez

    # this is the part that puts a dot a t=0 point
    if mark_t0:
        scat_fg = eig_ax.scatter(x0s, y0s, s=s(.02), c='k')
    else:
        scat_fg = []

    pp = PatchCollection(patch, cmap='hsv', lw=lw, alpha=alpha, zorder=zorder)

    pp.set_array(np.array(colors))
    pp.set_clim([0, 2 * np.pi])
    pp.set_zorder(1)

    eig_ax.add_collection(pp)

    if black_t0lines:
        lines_12 = [zip(x, y) for x, y in lines_1]
        lines_12_st = LineCollection(lines_12, linewidth=0.8)
        lines_12_st.set_color('k')
        eig_ax.add_collection(lines_12_st)

    eig_ax.set_aspect('equal')

    return fig, [scat_fg, pp, f_mark, lines_12_st]


def plot_localization_dists(fitparams, xydists, magevecs, ax, xlims=None):
    """Plot the excitations as a function of y and overlay the exponential decay fit to the data, for system which is
        periodic in x, but not in y.

    Parameters
    ----------
    fitparams : (1 x 2+) float array or list
        fit details for exponential decaying from the edge:
        A, K, other stuff optional, for this eigenvector

    Returns
    -------
    plt_handle :
        The handle for the data plotted on axis ax
    fit_handle : instance of ax.plot()
        The handle for the fit plotted on axis ax
    """
    # Note below that the variable used on x axis is the distance in the y dimension
    plt_handle = ax.plot(xydists, np.abs(magevecs), 'b.')
    if xlims is None:
        xlims = 2 * np.max(xydists)
        xx = np.linspace(0, xlims, 1000)
    else:
        xx = np.linspace(xlims[0], xlims[1], 1000)

    lz = fitparams
    heatdata = lz[0] * np.exp(lz[1] * xx)

    fit_handle = ax.plot(xx, heatdata, 'r-')
    if xlims is not None:
        ax.set_xlim(xlims)

    return plt_handle, fit_handle


def plot_localization_heatmap(ev_localization, ax, LL=None, xlims=None, ylims=None, cmap='Oranges', alpha=1.0):
    """Plot the heatmap of the exponentially localized 2D excitation, localized around a 2D coordinate stored in
    ev_localization[0:2].

    Parameters
    ----------
    ev_localization : 1 x 7 float array
        fit details for exponential decaying from the edge:
        xcenter, ycenter, A, K, uncertainty_A, covariance_AK, uncertainty_K for this eigenvector
    ax :
    LL : tuple of 2 floats (list of 2 floats also accepted)
        spatial extent of periodic dimensions, if supplied. Otherwise, not treated as periodic
    xlims : list or tuple of two floats
        The min and max in x for the heatmap
    ylims : list or tuple of two floats
        The min and max in y for the heatmap
    cmap : colormap spec
        The colormap to use for the localization overlay

    Returns
    -------
    localz_handle:
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
        # Assuming periodic, since LL is supplied
        XY = np.dstack((X.ravel(), Y.ravel()))[0]
        dist = le.distance_periodic(XY, lz[0:2], LL).reshape(np.shape(X))
        heatdata = lz[2] * np.exp(lz[3] * dist)
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
        The points defining the (typically closed) path boundary from which exponential confinement has been measured.
        If the sample is a periodicstrip, use plot_edge_localization_heatmap_periodicstrip().
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

    # Whether or periodic in 1D, find distance of each gridpt XY from the boundary that is supplied
    # Note that we take the (lz[5] % 2) element of bndry_linsegs, which gives us zeroth element for top, first
    # element for the bottom boundary, and the zeroth element (compare to top boundary) if excitation fit better to a
    # constant value (denoted by lz[5] = 2)
    XY = np.dstack((X.ravel(), Y.ravel()))[0]
    dist = linesegs.mindist_from_multiple_linesegs(XY, bndry_linsegs[int(lz[5] % 2)]).reshape(np.shape(X))
    heatdata = lz[0] * np.exp(lz[1] * dist)

    localz_handle = ax.pcolormesh(X, Y, heatdata, cmap=cmap, zorder=0, vmin=0.)
    return localz_handle


def plot_ill_dos(hlat, dos_ax=None, cbar_ax=None, alpha=1.0, vmin=None, vmax=None, **kwargs):
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

    eigval = hlat.get_eigval(attribute=False)

    # Load or compute localization
    localization = hlat.get_localization(attribute=False)
    ill = localization[:, 2]
    # print 'gcollpfns: localization = ', localization
    # print 'gcollpfns: shape(localization) = ', np.shape(localization)

    # Now overlay the ipr
    if vmin is None:
        print 'setting vmin...'
        vmin = 0.0
    if dos_ax is None:
        print 'dos_ax is None, initializing...'
        fig, dos_ax, cax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'haldane',
                                                                   alpha=alpha, colorV=ill,
                                                                   colormap='viridis',
                                                                   linewidth=0, cax_label=r'$\lambda^{-1}$',
                                                                   vmin=vmin, vmax=vmax, **kwargs)
    else:
        print 'gcollpfns: calling leplt.colored_DOS_plot...'
        dos_ax, cbar_ax, cbar, n, bins = \
            leplt.colored_DOS_plot(eigval, dos_ax, 'haldane', alpha=alpha, colorV=ill,
                                   cbar_ax=cbar_ax, colormap='viridis', linewidth=0,
                                   vmin=vmin, vmax=vmax, **kwargs)

    return dos_ax, cbar_ax


if __name__ == '__main__':
    eps = np.arange(0., 1.0, 0.05)
    omega = np.arange(-3, 3, 0.05)
    test_lorentzians(eps, omega)
