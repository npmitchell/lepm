# note cyclic import: cannot use from... syntax, here it just imports an empty module lattice_elasticity,
# but when called down below, it will find what it needs
import weakref
import lepm.plotting.colormaps as lecmaps
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import pylab as P
import numpy as np
import matplotlib.cm
import lepm.plotting.science_plot_style as sps
from mpl_toolkits.axes_grid1 import AxesGrid
import sys
import lepm.plotting.plotting as leplt

"""General functions for twisty lattice plotting"""


def excitation_DOS_plot(eigval, DOSexcite, DOS_ax=None, **kwargs):
    """Plot Gaussian excitation in kspace on top of DOS.

    Parameters
    ----------
    eigval: #modes x 1 complex or float array
        the eigenvalues of the system
    DOSexcite: tuple of floats or None
        (excitation frequency, stdev time), or else None if DOS plot is not desired.
        stdev time is conventionally excite_sigmatime
    DOS_ax: matplotlib axis instance or None
        the axis on which to plot the DOS and the Gaussian excitation spectrum
    **kwargs: DOS_plot() keyword arguments

    Returns
    -------
    DOS_ax
    """
    if DOS_ax is None:
        DOS_ax = plt.gca()

    DOS_plot(eigval, DOS_ax, **kwargs)

    # DOSexcite = (frequency, sigma_time)
    # amp(x) = exp[- acoeff * time**2]
    # amp(k) = sqrt(pi/acoeff) * exp[- pi**2 * k**2 / acoeff]
    # So 1/(2 * sigma_freq**2) = pi**2 /acoeff
    # So sqrt(acoeff/(2 * pi**2)) = sigma_freq

    sigmak = 1. / DOSexcite[1]
    xlims = DOS_ax.get_xlim()
    ktmp = np.linspace(xlims[0], xlims[1], 300)
    gaussk = 0.8 * DOS_ax.get_ylim()[1] * np.exp(-(ktmp - DOSexcite[0]) ** 2 / (2. * sigmak))
    DOS_ax.plot(ktmp, gaussk, 'r-')
    plt.sca(DOS_ax)

    return DOS_ax


def construct_eigvect_DOS_plot(xy, fig, DOS_ax, eig_ax, eigval, eigvect, en, sim_type, Ni, Nk, marker_num=0,
                               color_scheme='default', sub_lattice=-1, cmap_lines='BlueBlackRed', line_climv=None,
                               cmap_patches='isolum_rainbow', draw_strain=False, lw=3, bondval_matrix=None,
                               dotsz=.04, normalization=0.9, xycolor=lecmaps.green(), wzcolor=lecmaps.yellow()):
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
    marker_num : int
        the index of the 'timestep' (or phase of the eigvect) at which we place the t=0 line/dot for each particle
    color_scheme : str (default='default')
    sub_lattice : int
    cmap_lines : str
    line_climv : tuple of floats or None
    cmap_patches : str
    draw_strain : bool
    lw : float
    bondval_matrix : (NP x max#NN) float array or None
        a color specification for the bonds in the network
    
    Returns
    ----------
    fig :
        completed figure for normal mode
    
    [scat_fg, p, f_mark] :
        things to be cleared before next normal mode is drawn
        """
    # ppu = leplt.get_points_per_unit()
    s = leplt.absolute_sizer()

    plt.sca(DOS_ax)

    ev = eigval[en]
    ev1 = ev

    # Show where current eigenvalue is in DOS plot
    (f_mark,) = plt.plot([np.real(ev), np.real(ev)], plt.ylim(), '-r')

    NP = len(xy[:, 0])
    im1 = np.imag(ev)
    re1 = np.real(ev)
    plt.sca(eig_ax)
    plt.title('Mode %d; $\Omega=( %0.6f + %0.6f i)$' % (en, re1, im1))

    # Preallocate ellipsoid plot vars
    angles_arr = np.zeros(NP)
    tangles_arr = np.zeros(NP)

    # patch = []
    polygons, tiltgons = [], []
    colors = np.zeros(2 * NP + 2)
    # x_mag = np.zeros(NP)
    # y_mag = np.zeros(NP)

    x0s, y0s = np.zeros(NP), np.zeros(NP)
    w0s, z0s = np.zeros(NP), np.zeros(NP)

    mag1 = eigvect[en]

    # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    # x0, y0, x1, y1, ... xNP, yNP ... w0, z0, ... wNP, zNP.
    mag1x = np.array([mag1[2 * i] for i in range(NP)])
    mag1y = np.array([mag1[2 * i + 1] for i in range(NP)])
    mag1w = np.array([mag1[2 * i] for i in np.arange(NP, 2 * NP)])
    mag1z = np.array([mag1[2 * i + 1] for i in np.arange(NP, 2 * NP)])

    # Pick a series of times to draw out the ellipsoids
    if abs(ev1) > 0:
        time_arr = np.arange(21) * 2 * np.pi / (np.abs(ev1) * 20)
        exp1 = np.exp(1j * ev1 * time_arr)
    else:
        time_arr = np.arange(21) * 2 * np.pi / 20
        exp1 = np.exp(1j * time_arr)

    # Normalization for the ellipsoids
    lim_mag1 = np.max(np.array([np.sqrt(np.abs(exp1 * mag1x[i]) ** 2 + np.abs(exp1 * mag1y[i]) ** 2 +
                                np.abs(exp1 * mag1w[i]) ** 2 + np.abs(exp1 * mag1z[i]) ** 2)
                                for i in range(len(mag1x))]).flatten())
    if np.isnan(lim_mag1):
        print 'found nan for limiting magnitude, replacing lim_mag1 with 1.0'
        lim_mag1 = 1.

    mag1x *= normalization / lim_mag1
    mag1y *= normalization / lim_mag1
    mag1w *= normalization / lim_mag1
    mag1z *= normalization / lim_mag1
    # sys.exit()
    cw = []
    ccw = []
    lines_stretch = []
    lines_twist = []
    for i in range(NP):
        # Draw COM movement of each node as an ellipse
        x_disps = 0.5 * (exp1 * mag1x[i]).real
        y_disps = 0.5 * (exp1 * mag1y[i]).real
        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        # Draw movement of the orientation of each node as an ellipse
        w_disps = 0.5 * (exp1 * mag1w[i]).real
        z_disps = 0.5 * (exp1 * mag1z[i]).real
        w_vals = xy[i, 0] + w_disps
        z_vals = xy[i, 1] + z_disps

        poly_points = np.array([x_vals, y_vals]).T
        tilt_points = np.array([w_vals, z_vals]).T
        # polygon = Polygon(poly_points, True, lw=lw, ec='g')
        # tiltgon = Polygon(tilt_points, True, lw=lw, ec='r')
        # polygon = plt.plot(poly_points[:, 0], poly_points[:, 1], 'g-', lw=lw)
        # tiltgon = plt.plot(tilt_points[:, 0], tilt_points[:, 1], 'r-', lw=lw)
        npolypts = len(poly_points[:, 0])
        for ii in range(npolypts):
            lines_s = [[poly_points[ii, 0], poly_points[(ii + 1) % npolypts, 0]],
                       [poly_points[ii, 1], poly_points[(ii + 1) % npolypts, 1]]]
            lines_t = [[tilt_points[ii, 0], tilt_points[(ii + 1) % npolypts, 0]],
                       [tilt_points[ii, 1], tilt_points[(ii + 1) % npolypts, 1]]]
            lines_stretch.append(lines_s)
            lines_twist.append(lines_t)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]
        w0 = w_disps[marker_num]
        z0 = z_disps[marker_num]

        # x0s is the position (global pos, not relative) of each gyro at time = marker_num(out of 81)
        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]
        w0s[i] = w_vals[marker_num]
        z0s[i] = z_vals[marker_num]

        # Get angle for xy
        mag = np.sqrt(x0 ** 2 + y0 ** 2)
        if mag > 0:
            anglez = np.arccos(x0 / mag)
        else:
            anglez = 0
        if y0 < 0:
            anglez = 2 * np.pi - anglez

        # Get angle for wz
        tmag = np.sqrt(w0 ** 2 + z0 ** 2)
        if tmag > 0:
            tanglez = np.arccos(w0 / tmag)
        else:
            tanglez = 0
        if y0 < 0:
            tanglez = 2 * np.pi - tanglez

        angles_arr[i] = anglez
        tangles_arr[i] = tanglez
        # polygons.append(polygon)
        # tiltgons.append(tiltgon)

        # Do Fast Fourier Transform (FFT)
        # ff = abs(fft.fft(x_disps + 1j*y_disps))**2
        # ff_freq = fft.fftfreq(len(x_vals), 1)
        # mm_f = ff_freq[ff == max(ff)][0]

        if color_scheme == 'default':
            colors[2 * i] = anglez
            colors[2 * i + 1] = anglez

    # add two more colors to ensure clims go from 0 to 2pi, I think...
    colors[2 * NP] = 0
    colors[2 * NP + 1] = 2 * np.pi

    plt.yticks([])
    plt.xticks([])
    # this is the part that puts a dot a t=0 point
    scat_fg = eig_ax.scatter(x0s, y0s, s=s(dotsz), edgecolors='k', facecolors='none')
    scat_fg2 = eig_ax.scatter(w0s, z0s, s=s(dotsz), edgecolors='gray', facecolors='none')

    try:
        NN = np.shape(Ni)[1]
    except IndexError:
        NN = 0

    Rnorm = np.array([x0s, y0s]).T

    # Bond Stretches
    if draw_strain:
        inc = 0
        stretches = np.zeros(4 * len(xy))
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        n1 = float(np.linalg.norm(Rnorm[i] - Rnorm[j]))
                        n2 = np.linalg.norm(xy[i] - xy[j])
                        stretches[inc] = (n1 - n2)
                        inc += 1

        stretch = np.array(stretches[0:inc])
    else:
        # simply get length of BL (len(BL) = inc) by iterating over all bondsssa
        inc = 0
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        inc += 1

    # For particles with neighbors, get list of bonds to draw.
    # If bondval_matrix is not None, color by the elements of that matrix
    if bondval_matrix is not None or draw_strain:
        test = list(np.zeros([inc, 1]))
        bondvals = list(np.ones([inc, 1]))
        inc = 0
        xy = np.array([x0s, y0s]).T
        for i in range(len(xy)):
            if NN > 0:
                for j, k in zip(Ni[i], Nk[i]):
                    if i < j and abs(k) > 0:
                        test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
                        if bondval_matrix is not None:
                            bondvals[inc] = bondval_matrix[i, j]
                        inc += 1

            # lines connect sites (bonds), while lines_12 draw the black lines from the pinning to location sites
            lines = [zip(x, y) for x, y in test]

    # Check that we have all the cmaps
    if cmap_lines not in plt.colormaps() or cmap_patches not in plt.colormaps():
        lecmaps.register_cmaps()

    # Add lines colored by strain here
    if bondval_matrix is not None:
        lines_st = LineCollection(lines, array=bondvals, cmap=cmap_lines, linewidth=0.8)
        if line_climv is None:
            maxk = np.max(np.abs(bondvals))
            mink = np.min(np.abs(bondvals))
            if (bondvals - bondvals[0] < 1e-8).all():
                lines_st.set_clim([mink - 1., maxk + 1.])
            else:
                lines_st.set_clim([mink, maxk])

        lines_st.set_zorder(2)
        eig_ax.add_collection(lines_st)
    else:
        if draw_strain:
            lines_st = LineCollection(lines, array=stretch, cmap=cmap_lines, linewidth=0.8)
            if line_climv is None:
                maxstretch = np.max(np.abs(stretch))
                if maxstretch < 1e-8:
                    line_climv = 1.0
                else:
                    line_climv = maxstretch

            lines_st.set_clim([-line_climv, line_climv])
            lines_st.set_zorder(2)
            eig_ax.add_collection(lines_st)

    # Draw lines for movement
    lines_stretch = [zip(x, y) for x, y in lines_stretch]
    polygons = LineCollection(lines_stretch,
                              linewidth=lw,
                              linestyles='solid',
                              color=xycolor)
    polygons.set_zorder(1)
    eig_ax.add_collection(polygons)
    lines_twist = [zip(x, y) for x, y in lines_twist]
    tiltgons = LineCollection(lines_twist,
                              linewidth=lw,
                              linestyles='solid',
                              color=wzcolor)
    tiltgons.set_zorder(1)
    eig_ax.add_collection(tiltgons)

    # Draw polygons
    # polygons = PatchCollection(polygons, cmap=cmap_patches, alpha=0.6)
    # # p.set_array(np.array(colors))
    # polygons.set_clim([0, 2 * np.pi])
    # polygons.set_zorder(1)
    # eig_ax.add_collection(polygons)
    # tiltgons = PatchCollection(tiltgons, cmap=cmap_patches, alpha=0.6)
    # # p.set_array(np.array(colors))
    # tiltgons.set_clim([0, 2 * np.pi])
    # tiltgons.set_zorder(1000)
    # eig_ax.add_collection(tiltgons)

    eig_ax.set_aspect('equal')

    # erased ev/(2*pi) here npm 2016
    cw_ccw = [cw, ccw, ev]
    # print cw_ccw[1]

    # If on a virtualenv, check it here
    # if not hasattr(sys, 'real_prefix'):
    #     plt.show()
    #     eig_ax.set_facecolor('#000000')
    #     print 'leplt: construct_eigvect_DOS_plot() exiting'

    return fig, [scat_fg, scat_fg2, f_mark, polygons, tiltgons], cw_ccw


def plot_eigvect_excitation(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, marker_num=0, draw_strain=False,
                            black_t0lines=False, mark_t0=True, title='auto', normalization=1., alpha=0.6,
                            lw=1, zorder=10, cmap='isolum_rainbow', color_scheme='default', color='phase', theta=None,
                            t0_ptsize=.02, bondval_matrix=None, dotsz=.04, xycolor=lecmaps.green(), wzcolor=lecmaps.yellow()):
    """Draws normal mode ellipsoids on axis eig_ax.
    If black_t0lines is true, draws the black line from pinning site to positions

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
    color : str ('phase' or 'displacement')
        Whether to color the excitations by their instantaneous (relative) phase, or by the magnitude of the mean
        displacement
    theta : float or None
        angle by which to rotate the initial displacement plotted by black lines, dots, etc
    bondval_matrix : (NP x max#NN) float array or None
        a color specification for the bonds in the network

    Returns
    ----------
    fig : matplotlib figure instance
        completed figure for normal mode
    [scat_fg, pp, f_mark, lines12_st] :
        things to be cleared before next normal mode is drawn
        """
    # ensure that colormap is registered
    lecmaps.register_cmap(cmap)

    # ppu = get_points_per_unit()
    s = leplt.absolute_sizer()

    ev = eigval[en]

    # Show where current eigenvalue is in DOS plot
    if dos_ax is not None:
        (f_mark,) = dos_ax.plot([abs(ev), abs(ev)], dos_ax.get_ylim(), '-r')
    else:
        f_mark = None

    NP = len(xy)
    print 'twisty.plotting.plotting: NP = ', NP
    print 'twisty.plotting.plotting: np.shape(xy) = ', np.shape(xy)

    re1 = np.real(ev)
    plt.sca(eig_ax)

    if title == 'auto':
        eig_ax.set_title('$\omega = %0.6f$' % re1)
    elif title is not None and title not in ['', 'none']:
        eig_ax.set_title(title)

    # Preallocate ellipsoid plot vars
    angles_arr = np.zeros(NP)
    tangles_arr = np.zeros(NP)

    # patch = []
    polygons, tiltgons = [], []
    colors = np.zeros(2 * NP + 2)
    # x_mag = np.zeros(NP)
    # y_mag = np.zeros(NP)

    x0s, y0s = np.zeros(NP), np.zeros(NP)
    w0s, z0s = np.zeros(NP), np.zeros(NP)

    mag1 = eigvect[en]
    print ''

    # Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    # x0, y0, x1, y1, ... xNP, yNP ... w0, z0, ... wNP, zNP.
    mag1x = np.array([mag1[2 * i] for i in range(NP)])
    mag1y = np.array([mag1[2 * i + 1] for i in range(NP)])
    mag1w = np.array([mag1[2 * i] for i in np.arange(NP, 2 * NP)])
    mag1z = np.array([mag1[2 * i + 1] for i in np.arange(NP, 2 * NP)])

    # Pick a series of times to draw out the ellipsoids
    if abs(ev) > 0:
        time_arr = np.arange(21) * 2 * np.pi / (np.abs(ev) * 20)
        exp1 = np.exp(1j * ev * time_arr)
    else:
        time_arr = np.arange(21) * 2 * np.pi / 20
        exp1 = np.exp(1j * time_arr)

    # Normalization for the ellipsoids
    lim_mag1 = np.max(np.array([np.sqrt(np.abs(exp1 * mag1x[i]) ** 2 + np.abs(exp1 * mag1y[i]) ** 2 +
                                        np.abs(exp1 * mag1w[i]) ** 2 + np.abs(exp1 * mag1z[i]) ** 2)
                                for i in range(len(mag1x))]).flatten())
    if np.isnan(lim_mag1):
        print 'found nan for limiting magnitude, replacing lim_mag1 with 1.0'
        lim_mag1 = 1.

    mag1x *= normalization / lim_mag1
    mag1y *= normalization / lim_mag1
    mag1w *= normalization / lim_mag1
    mag1z *= normalization / lim_mag1
    # sys.exit()
    cw = []
    ccw = []
    lines_stretch = []
    lines_twist = []
    for i in range(NP):
        # Draw COM movement of each node as an ellipse
        x_disps = 0.5 * (exp1 * mag1x[i]).real
        y_disps = 0.5 * (exp1 * mag1y[i]).real
        x_vals = xy[i, 0] + x_disps
        y_vals = xy[i, 1] + y_disps

        # Draw movement of the orientation of each node as an ellipse
        w_disps = 0.5 * (exp1 * mag1w[i]).real
        z_disps = 0.5 * (exp1 * mag1z[i]).real
        w_vals = xy[i, 0] + w_disps
        z_vals = xy[i, 1] + z_disps

        poly_points = np.array([x_vals, y_vals]).T
        tilt_points = np.array([w_vals, z_vals]).T
        # polygon = Polygon(poly_points, True, lw=lw, ec='g')
        # tiltgon = Polygon(tilt_points, True, lw=lw, ec='r')
        # polygon = plt.plot(poly_points[:, 0], poly_points[:, 1], 'g-', lw=lw)
        # tiltgon = plt.plot(tilt_points[:, 0], tilt_points[:, 1], 'r-', lw=lw)
        npolypts = len(poly_points[:, 0])
        for ii in range(npolypts):
            lines_s = [[poly_points[ii, 0], poly_points[(ii + 1) % npolypts, 0]],
                       [poly_points[ii, 1], poly_points[(ii + 1) % npolypts, 1]]]
            lines_t = [[tilt_points[ii, 0], tilt_points[(ii + 1) % npolypts, 0]],
                       [tilt_points[ii, 1], tilt_points[(ii + 1) % npolypts, 1]]]
            lines_stretch.append(lines_s)
            lines_twist.append(lines_t)

        # x0 is the marker_num^th element of x_disps
        x0 = x_disps[marker_num]
        y0 = y_disps[marker_num]
        w0 = w_disps[marker_num]
        z0 = z_disps[marker_num]

        # x0s is the position (global pos, not relative) of each gyro at time = marker_num(out of 81)
        x0s[i] = x_vals[marker_num]
        y0s[i] = y_vals[marker_num]
        w0s[i] = w_vals[marker_num]
        z0s[i] = z_vals[marker_num]

        # Get angle for xy
        mag = np.sqrt(x0 ** 2 + y0 ** 2)
        if mag > 0:
            anglez = np.arccos(x0 / mag)
        else:
            anglez = 0
        if y0 < 0:
            anglez = 2 * np.pi - anglez

        # Get angle for wz
        tmag = np.sqrt(w0 ** 2 + z0 ** 2)
        if tmag > 0:
            tanglez = np.arccos(w0 / tmag)
        else:
            tanglez = 0
        if y0 < 0:
            tanglez = 2 * np.pi - tanglez

        angles_arr[i] = anglez
        tangles_arr[i] = tanglez
        # polygons.append(polygon)
        # tiltgons.append(tiltgon)

        # Do Fast Fourier Transform (FFT)
        # ff = abs(fft.fft(x_disps + 1j*y_disps))**2
        # ff_freq = fft.fftfreq(len(x_vals), 1)
        # mm_f = ff_freq[ff == max(ff)][0]

        if color_scheme == 'default':
            colors[2 * i] = anglez
            colors[2 * i + 1] = anglez

    # add two more colors to ensure clims go from 0 to 2pi, I think...
    colors[2 * NP] = 0
    colors[2 * NP + 1] = 2 * np.pi

    plt.yticks([])
    plt.xticks([])
    # this is the part that puts a dot a t=0 point
    scat_fg = eig_ax.scatter(x0s, y0s, s=s(dotsz), edgecolors='k', facecolors='none')
    scat_fg2 = eig_ax.scatter(w0s, z0s, s=s(dotsz), edgecolors='gray', facecolors='none')

    # try:
    #     NN = np.shape(tlat.lattice.NL)[1]
    # except IndexError:
    #     NN = 0
    #
    # Rnorm = np.array([x0s, y0s]).T
    #
    # # Bond Stretches
    # if draw_strain:
    #     inc = 0
    #     stretches = np.zeros(4 * len(xy))
    #     for i in range(len(xy)):
    #         if NN > 0:
    #             for j, k in zip(Ni[i], Nk[i]):
    #                 if i < j and abs(k) > 0:
    #                     n1 = float(np.linalg.norm(Rnorm[i] - Rnorm[j]))
    #                     n2 = np.linalg.norm(xy[i] - xy[j])
    #                     stretches[inc] = (n1 - n2)
    #                     inc += 1
    #
    #     stretch = np.array(stretches[0:inc])
    # else:
    #     # simply get length of BL (len(BL) = inc) by iterating over all bondsssa
    #     inc = 0
    #     for i in range(len(xy)):
    #         if NN > 0:
    #             for j, k in zip(Ni[i], Nk[i]):
    #                 if i < j and abs(k) > 0:
    #                     inc += 1
    #
    # # For particles with neighbors, get list of bonds to draw.
    # # If bondval_matrix is not None, color by the elements of that matrix
    # if bondval_matrix is not None or draw_strain:
    #     test = list(np.zeros([inc, 1]))
    #     bondvals = list(np.ones([inc, 1]))
    #     inc = 0
    #     xy = np.array([x0s, y0s]).T
    #     for i in range(len(xy)):
    #         if NN > 0:
    #             for j, k in zip(Ni[i], Nk[i]):
    #                 if i < j and abs(k) > 0:
    #                     test[inc] = [xy[(i, j), 0], xy[(i, j), 1]]
    #                     if bondval_matrix is not None:
    #                         bondvals[inc] = bondval_matrix[i, j]
    #                     inc += 1
    #
    #         # lines connect sites (bonds), while lines_12 draw the black lines from the pinning to location sites
    #         lines = [zip(x, y) for x, y in test]
    #
    # # Check that we have all the cmaps
    # if cmap_lines not in plt.colormaps() or cmap_patches not in plt.colormaps():
    #     lecmaps.register_cmaps()
    #
    # # Add lines colored by strain here
    # if bondval_matrix is not None:
    #     lines_st = LineCollection(lines, array=bondvals, cmap=cmap_lines, linewidth=0.8)
    #     if line_climv is None:
    #         maxk = np.max(np.abs(bondvals))
    #         mink = np.min(np.abs(bondvals))
    #         if (bondvals - bondvals[0] < 1e-8).all():
    #             lines_st.set_clim([mink - 1., maxk + 1.])
    #         else:
    #             lines_st.set_clim([mink, maxk])
    #
    #     lines_st.set_zorder(2)
    #     eig_ax.add_collection(lines_st)
    # else:
    #     if draw_strain:
    #         lines_st = LineCollection(lines, array=stretch, cmap=cmap_lines, linewidth=0.8)
    #         if line_climv is None:
    #             maxstretch = np.max(np.abs(stretch))
    #             if maxstretch < 1e-8:
    #                 line_climv = 1.0
    #             else:
    #                 line_climv = maxstretch
    #
    #         lines_st.set_clim([-line_climv, line_climv])
    #         lines_st.set_zorder(2)
    #         eig_ax.add_collection(lines_st)

    # Draw lines for movement
    lines_stretch = [zip(x, y) for x, y in lines_stretch]
    polygons = LineCollection(lines_stretch,
                              linewidth=lw,
                              linestyles='solid',
                              color=xycolor)
    polygons.set_zorder(1)
    eig_ax.add_collection(polygons)
    lines_twist = [zip(x, y) for x, y in lines_twist]
    tiltgons = LineCollection(lines_twist,
                              linewidth=lw,
                              linestyles='solid',
                              color=wzcolor)
    tiltgons.set_zorder(1)
    eig_ax.add_collection(tiltgons)

    # Draw polygons
    # polygons = PatchCollection(polygons, cmap=cmap_patches, alpha=0.6)
    # # p.set_array(np.array(colors))
    # polygons.set_clim([0, 2 * np.pi])
    # polygons.set_zorder(1)
    # eig_ax.add_collection(polygons)
    # tiltgons = PatchCollection(tiltgons, cmap=cmap_patches, alpha=0.6)
    # # p.set_array(np.array(colors))
    # tiltgons.set_clim([0, 2 * np.pi])
    # tiltgons.set_zorder(1000)
    # eig_ax.add_collection(tiltgons)

    eig_ax.set_aspect('equal')

    # erased ev/(2*pi) here npm 2016
    cw_ccw = [cw, ccw, ev]
    # print cw_ccw[1]

    return fig, [scat_fg, scat_fg2, f_mark, polygons, tiltgons], cw_ccw


def lt2description(lp):
    """
    Convert latticetopology string shorthand into a description for a title.

    Parameters
    ----------
    lp : dict or str
        a lattice parameters dictionary with key 'LatticeTop' or the string specifier for the LatticeTopology
    """
    if isinstance(lp, dict):
        lt = lp['LatticeTop']
    elif isinstance(lp, str):
        lt = lp

    if lt == 'hucentroid':
        return 'voronoized hyperuniform network'
    elif lt == 'kagome_hucent':
        return 'kagomized hyperuniform network'
    elif lt == 'kagper_hucent':
        return r'partially kagomized hyperuniform ($d=${0:0.2f}'.format(lp['percolation_density']) + ') network'
    elif lt == 'hexagonal':
        return 'honeycomb network'
    elif lt == 'iscentroid':
        return 'voronoized jammed network'
    elif lt == 'kagome_isocent':
        return 'kagomized jammed network'
    elif lt == 'penroserhombTricent':
        return 'voronoized rhombic Penrose lattice'
    elif lt == 'kagome_penroserhombTricent':
        return 'kagomized rhombic Penrose lattice'
    elif lt in ['hex_kagframe', 'hex_kagcframe']:
        return 'honeycomb lattice with kagome frame'


def param2description(param_name):
    """
    Convert parameter name string shorthand into a description for a title, label, or legend.
    """
    if param_name == 'ABDelta':
        return r'Inversion symmetry breaking, $\Delta_{AB}$'
    elif param_name == 'thetatwist':
        return r'$\theta_{\mathrm{twist}} / \pi $'
