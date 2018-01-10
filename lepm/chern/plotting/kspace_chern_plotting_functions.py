# from chern_functions_gen import *
import numpy as np
import lepm.data_handling as dh
from matplotlib.path import Path
import matplotlib.pyplot as plt
# import bz_funcs as bz
# import lattice_functions as lf
import lepm.plotting.science_plot_style as sps
import matplotlib.tri as mtri

'''Functions for plotting kspace chern number calculations (using traditional wedge product measures)'''


def make_plot(kkx, kky, bands, traces, tvals, ons, pin, num='all'):
    """One of Lisa's functions for plotting chern results with lattice structure"""
    w = 180
    h = 140
    label_params = dict(size=12, fontweight='normal')

    s = 2
    h1 = (h - 3 * s) / 2.
    w1 = 0.75 * w

    fig = sps.figure_in_mm(w, h)
    ax1 = sps.axes_in_mm(0, h - h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax2 = sps.axes_in_mm(0, h - 2 * h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax3 = sps.axes_in_mm(1 * s + w1, h - 2 * h1, w - w1 - 3 * s, 2 * h1, label=None, label_params=label_params)
    ax_lat = sps.axes_in_mm(1 * s + w1, h - 0.4 * h1, w - w1 - 3 * s, 0.4 * h1, label=None, label_params=label_params)

    c1 = '#000000'
    c2 = '#FFFFFF'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    cc1 = '#96adb4'
    cc2 = '#b16566'
    cc3 = '#665656'
    cc4 = '#9d96b4'

    pin = 1
    gor = 'random'  # grid or random, lines
    cols = [c1, c2, c3, c4]
    l_cols = [cc1, cc2, cc3, cc4]
    mM, vertex_points, dump_dir = lf.alpha_lattice(tvals, pin, ons, ax=ax_lat, col=cols, lincol=l_cols)
    ax3.axis([0, 1, 0, 1.])
    ax3.text(0.025, .75, '$k_1 = %.2f$' % tvals[0], style='italic', color=cc1, fontsize=12)
    ax3.text(0.525, .75, '$k_2 = %.2f$' % tvals[1], style='italic', color=cc2, fontsize=12)
    ax3.text(0.025, .7, '$k_3 = %.2f$' % tvals[2], style='italic', color=cc3, fontsize=12)
    ax3.text(0.525, .7, '$k_4 = %.2f$' % tvals[3], style='italic', color=cc4, fontsize=12)
    # bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}
    ax3.text(0.025, .65, '$M_1 = %.2f$' % ons[0], style='italic', color=c1, fontsize=12)
    ax3.text(0.525, .65, '$M_2 = %.2f$' % ons[1], style='italic', color='#B9B9B9', fontsize=12)
    ax3.text(0.025, .6, '$M_3 = %.2f$' % ons[2], style='italic', color=c3, fontsize=12)
    ax3.text(0.525, .6, '$M_4 = %.2f$' % ons[3], style='italic', color=c4, fontsize=12)
    # bbox={'facecolor':'red', 'alpha':0.5, 'pad':10}
    ax3.axis('off')
    ax_lat.axis('off')

    nb = len(bands[0])
    b = []
    bz_area = dh.polygon_area(vertex_points)
    band_gaps = np.zeros(nb - 1)
    ax3.text(0.025, .55, '$\mathrm{Chern\/ numbers}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .35, '$\mathrm{Band\/ boundaries}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .15, '$\mathrm{Min.\/differences}$', fontweight='bold', color='k', fontsize=12)
    for i in range(int(nb)):
        j = i

        # ax.scatter(kkx, kky, 1j*traces[:,j])
        if j >= nb / 2:
            if num == 'all':
                ax1.plot_trisurf(kx[:1000], ky[:1000], bands[:, j][:1000], cmap='cool', vmin=min(abs(bands.flatten())),
                                 vmax=max(abs(bands.flatten())), linewidth=0.0)
                # ax2.plot_trisurf(kx, ky, bands[:,j], cmap = 'cool', vmin = min(abs(bands.flatten())),
                # vmax=max(abs(bands.flatten())), linewidth = 0.0)
            elif j == num:
                ax2.plot_trisurf(kx[:1000], ky[:1000], 1j * traces[:, j][:1000], cmap='cool',
                                 vmin=min(abs(traces[:, j].flatten())), vmax=max(abs(traces[:, j].flatten())),
                                 linewidth=0.0)

            ax1.set_ylabel('ky')
            ax1.axes.get_xaxis().set_ticks([])
            ax2.set_xlabel('kx')
            ax2.axes.get_yaxis().set_ticks([])

            ax1.set_zlabel('$\Omega$')
            ax2.set_zlabel('$\Omega$')
            bv = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bz_area)
            if abs(bv) > 0.7:
                cc = 'r'
            else:
                cc = 'k'
            if j == num:
                ax3.text(0.025, .55 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % bv, color=cc, fontsize=10)
            if j >= nb / 2:
                maxtb = max(bands[:, j])
                mintb = min(bands[:, j])
                ax3.text(0.025, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % mintb, color='k', fontsize=10)
                ax3.text(0.525, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % maxtb, color='k', fontsize=10)
            if j > nb / 2:
                min_diff = min(bands[:, j] - bands[:, j - 1])
                ax3.text(0.025, .15 - (j - nb / 2) * 0.04, '$%0.2f$' % min_diff, color='k', fontsize=10)

        b.append((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bz_area)

    ax1.view_init(elev=0, azim=0.)
    ax2.view_init(elev=-0, azim=90.)

    minx = min(kx)
    maxx = max(kx)
    miny = min(ky)
    maxy = max(ky)

    # plt.show()
    # plt.savefig(save_dir+'/images/test.png')

    s1 = '%0.2f_' % tvals[0]
    s2 = '%0.2f_' % tvals[1]
    s3 = '%0.2f_' % tvals[2]
    s4 = '%0.2f_' % tvals[3]

    s5 = '%0.2f_' % ons[0]
    s6 = '%0.2f_' % ons[1]
    s7 = '%0.2f_' % ons[2]
    s8 = '%0.2f_' % ons[3]

    plt.show()


def view_plot1(kx, ky, bands, traces, tvals, ons, vertex_points, od, pnum=500):
    """

    Parameters
    ----------
    kx
    ky
    bands
    traces
    tvals
    ons
    vertex_points
    od
    pnum

    Returns
    -------

    """
    bz_area = dh.polygon_area(vertex_points)

    w = 180
    h = 140
    label_params = dict(size=12, fontweight='normal')

    s = 2
    h1 = (h - 3 * s) / 2.
    w1 = 0.75 * w

    fig = sps.figure_in_mm(w, h)

    ax1 = sps.axes_in_mm(0, h - h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax2 = sps.axes_in_mm(0, h - 2 * h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax3 = sps.axes_in_mm(1 * s + w1, h - 2 * h1, w - w1 - 3 * s, 2 * h1, label=None, label_params=label_params)
    ax_lat = sps.axes_in_mm(1 * s + w1, h - 0.4 * h1, w - w1 - 3 * s, 0.4 * h1, label=None, label_params=label_params)

    c1 = '#000000'
    c2 = '#B9B9B9'
    c3 = '#E51B1B'
    c4 = '#18CFCF'

    cc1 = '#96adb4'
    cc2 = '#b16566'
    cc3 = '#665656'
    cc4 = '#9d96b4'

    pin = 1
    cols = [c1, c2, c3, c4]
    ccols = [cc1, cc2, cc3, cc4]
    l_cols = [cc1, cc2, cc3, cc4]

    ax3.axis([0, 1, 0, 1.])
    xp = [0.025, 0.525, 0.025, .525]
    yp = [0.65, 0.65, 0.6, 0.6]

    for i in range(len(tvals)):
        ax3.text(xp[i], yp[i] + 0.1, '$k_%1d$' % (i + 1) + '= $%.2f$' % tvals[i], style='italic',
                 color=ccols[i],
                 fontsize=12)
    for i in range(len(ons)):
        ax3.text(xp[i], yp[i], '$M_%1d$' % (i + 1) + '$= %.2f$' % ons[i], style='italic',
                 color=cols[i],
                 fontsize=12
                 )
    ax3.axis('off')
    ax_lat.axis('off')

    nb = len(bands[0])
    b = []
    band_gaps = np.zeros(nb - 1)

    ax3.text(0.025, .55, '$\mathrm{Chern\/ numbers}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .35, '$\mathrm{Band\/ boundaries}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .15, '$\mathrm{Min.\/differences}$', fontweight='bold', color='k', fontsize=12)

    if len(kx) < pnum:
        pnum = len(kx)

    for i in range(int(nb)):
        j = i
        if j >= nb / 2:
            ax1.plot_trisurf(kx[:pnum], ky[:pnum], bands[:, j][:pnum], cmap='cool', vmin=min(abs(bands.flatten())),
                             vmax=max(abs(bands.flatten())), linewidth=0.0, alpha=1)
            ax2.plot_trisurf(kx[:pnum], ky[:pnum], bands[:, j][:pnum], cmap='cool', vmin=min(abs(bands.flatten())),
                             vmax=max(abs(bands.flatten())), linewidth=0.0, alpha=1)

            ax1.set_ylabel('ky')
            ax1.axes.get_xaxis().set_ticks([])
            ax2.set_xlabel('kx')
            ax2.axes.get_yaxis().set_ticks([])

            ax1.set_zlabel('$\Omega$')
            ax2.set_zlabel('$\Omega$')

            bv = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bz_area)
            if abs(bv) > 0.7:
                cc = 'r'
            else:
                cc = 'k'
            ax3.text(0.025, .55 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % bv,
                     color=cc,
                     fontsize=10)

            maxtb = max(bands[:, j])
            mintb = min(bands[:, j])
            ax3.text(0.025, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % mintb,
                     color='k',
                     fontsize=10)
            ax3.text(0.525, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % maxtb,
                     color='k',
                     fontsize=10)

            if j > nb / 2:
                min_diff = min(bands[:, j] - bands[:, j - 1])
                ax3.text(0.025, .15 - (j - nb / 2) * 0.04, '$%0.2f$' % min_diff,
                         color='k',
                         fontsize=10)

    ax1.view_init(elev=0, azim=0.)
    ax2.view_init(elev=0, azim=90.)

    minx = min(kx)
    maxx = max(kx)
    miny = min(ky)
    maxy = max(ky)

    return ax_lat, cols, l_cols, fig


def save_plot(kx, ky, bands, traces, tvals, ons, vertex_points, od):
    """

    Parameters
    ----------
    kx
    ky
    bands
    traces
    tvals
    ons
    vertex_points
    od

    Returns
    -------

    """
    bz_area = dh.polygon_area(vertex_points)

    w = 180
    h = 140
    label_params = dict(size=12, fontweight='normal')

    s = 2
    h1 = (h - 3 * s) / 2.
    w1 = 0.75 * w

    fig = sps.figure_in_mm(w, h)
    ax1 = sps.axes_in_mm(0, h - h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax2 = sps.axes_in_mm(0, h - 2 * h1, w1, h1, label=None, label_params=label_params, projection='3d')
    ax3 = sps.axes_in_mm(1 * s + w1, h - 2 * h1, w - w1 - 3 * s, 2 * h1, label=None, label_params=label_params)
    ax_lat = sps.axes_in_mm(1 * s + w1, h - 0.4 * h1, w - w1 - 3 * s, 0.4 * h1, label=None, label_params=label_params)

    c1 = '#000000'
    c2 = '#B9B9B9'
    c3 = '#E51B1B'
    c4 = '#18CFCF'
    c5 = 'violet'
    c6 = 'k'

    cc1 = '#96adb4'
    cc2 = '#b16566'
    cc3 = '#665656'
    cc4 = '#9d96b4'

    pin = 1
    cols = [c1, c2, c3, c4, c5, c6]
    ccols = [cc1, cc2, cc3, cc4, c5]
    l_cols = [cc1, cc2, cc3, cc4, c5]
    ax3.axis([0, 1, 0, 1.])
    print len(ons)

    xp = [0.025, 0.525, 0.025, .525]
    yp = [0.65, 0.65, 0.6, 0.6]
    min_l = min([len(xp), len(ons)])
    for i in range(len(tvals)):
        ax3.text(xp[i], yp[i] + 0.1, '$k_%1d$' % (i + 1) + '= $%.2f$' % tvals[i], style='italic', color=ccols[i],
                 fontsize=12)
    for i in range(min_l):
        ax3.text(xp[i], yp[i], '$M_%1d$' % (i + 1) + '$= %.2f$' % ons[i], style='italic',
                 color=cols[i], fontsize=12)
    ax3.axis('off')
    ax_lat.axis('off')

    nb = len(bands[0])
    b = []
    band_gaps = np.zeros(nb - 1)

    ax3.text(0.025, .55, '$\mathrm{Chern\/ numbers}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .35, '$\mathrm{Band\/ boundaries}$', fontweight='bold', color='k', fontsize=12)
    ax3.text(0.025, .15, '$\mathrm{Min.\/differences}$', fontweight='bold', color='k', fontsize=12)

    for i in range(int(nb)):
        j = i
        bv = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bz_area)

        if j >= nb / 2:
            ax1.plot_trisurf(kx, ky, bands[:, j], cmap='cool', vmin=min(abs(bands.flatten())),
                             vmax=max(abs(bands.flatten())), linewidth=0.0, alpha=1)
            ax2.plot_trisurf(kx, ky, bands[:, j], cmap='cool', vmin=min(abs(bands.flatten())),
                             vmax=max(abs(bands.flatten())), linewidth=0.0, alpha=1)

            v_min = min(abs(bands.flatten()))
            v_max = max(abs(bands.flatten()))
            # c_3D_plot(kx, ky, bands[:,j], vertex_points, ax1, v_min, v_max)
            # c_3D_plot(kx, ky, bands[:,j], vertex_points, ax2, v_min, v_max)

            ax1.set_ylabel('ky')
            ax1.axes.get_xaxis().set_ticks([])
            ax2.set_xlabel('kx')
            ax2.axes.get_yaxis().set_ticks([])

            ax1.set_zlabel('$\Omega$')
            ax2.set_zlabel('$\Omega$')

            bv = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bz_area)
            if abs(bv) > 0.7:
                cc = 'r'
            else:
                cc = 'k'
            ax3.text(0.025, .55 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % bv,
                     color=cc,
                     fontsize=10)

            maxtb = max(bands[:, j])
            mintb = min(bands[:, j])
            ax3.text(0.025, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % mintb,
                     color='k',
                     fontsize=10)
            ax3.text(0.525, .35 - (j - nb / 2 + 1) * 0.04, '$%0.2f$' % maxtb,
                     color='k',
                     fontsize=10)

            if j > nb / 2:
                min_diff = min(bands[:, j] - bands[:, j - 1])
                ax3.text(0.025, .15 - (j - nb / 2) * 0.04, '$%0.2f$' % min_diff,
                         color='k',
                         fontsize=10)

    ax1.view_init(elev=0, azim=0.)
    ax2.view_init(elev=0, azim=90.)

    minx = min(kx)
    maxx = max(kx)
    miny = min(ky)
    maxy = max(ky)

    return ax_lat, cols, l_cols, fig


def c_3D_plot(x, y, z, vertex_points, ax, v_min, v_max):
    """

    Parameters
    ----------
    x
    y
    z
    vertex_points
    ax
    v_min
    v_max

    Returns
    -------

    """
    # Create the Triangulation; no triangles so Delaunay triangulation created.
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    triang = mtri.Triangulation(x, y)

    min_radius = 1
    # Mask off unwanted triangles.
    xmid = x[triang.triangles]
    ymid = y[triang.triangles]

    vp = np.array([np.array([xmid[i], ymid[i]]).T for i in range(len(xmid))])
    areas = np.array([dh.polygon_area(vp[i]) for i in range(len(vp))])

    poly_path = Path(vertex_points)

    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)

    mask = np.array([not poly_path.contains_point([xmid[i], ymid[i]]) for i in range(len(xmid))])
    triang.set_mask(mask)
    triang.set_mask(mask)
    ax.plot_trisurf(triang, z, cmap='cool', vmin=v_min, vmax=v_max, linewidth=0.0, alpha=1)


def movie_honeycomb_single_frame(deltaz, phi, text=True):
    """

    Parameters
    ----------
    deltaz
    phi
    text

    Returns
    -------

    """
    a1, a2 = bz.find_lattice_vecs(deltaz * np.pi / 180., phi * np.pi / 180.)
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    R, Ni, Nk, cols, line_cols = lf.honeycomb_sheared_vis(deltaz, phi)

    fig = sps.figure_in_mm(90, 90)
    ax_lat = sps.axes_in_mm(0, 0, 90, 90)

    lf.lattice_plot(R, Ni, Nk, ax_lat, cols, line_cols)
    plt.ylim(-2, 1)
    plt.xlim(-3, 3)
    ax_lat.axis('off')

    if text:
        ax_lat.text(-2.75, 1.5, '$\delta$ = %d$^{\circ}$' % deltaz, fontweight='bold',
                    color='k',
                    fontsize=16
                    )
        ax_lat.text(-2.75, 1.3, '$\phi$ = %d$^{\circ}$' % phi, fontweight='bold',
                    color='k',
                    fontsize=16
                    )

    return fig, ax_lat


if __name__ == '__main__':
    base_fn = '1.00_1.00_1.00_1.00___0.00_0.30_1.00_1.00_'
    tot_path = '/Users/lisa/Dropbox/Research/Chern_number_calculation/phase_sheared/save_all_try2/data_dict45.0_195.0.np.pickle'
    data_dir = '/Users/lisa/Dropbox/Research/New_Chern_general/data/honeycomb_sheared/data_new_correctBZ/'

    deltaz = 120
    phi = 0

    a1, a2 = bz.find_lattice_vecs(deltaz * np.pi / 180., phi * np.pi / 180.)
    b1, b2 = bz.find_bz_lattice_vecs(a1, a2)
    R, Ni, Nk, cols, line_cols = lf.honeycomb_sheared_vis(deltaz, phi)

    vtx, vty = bz.find_bz_zone(b1, b2)
    vertex_points = np.array([vtx, vty]).T

    # print 'vertex points', vertex_points

    fn = 'honeycomb_sheared'
    fn += '_%0.2f__' % 1.
    fn += '%0.2f_' % 0.8
    fn += '%0.2f__' % 1.2
    fn += '%0.2f_' % (deltaz)
    fn += '%0.2f' % (phi)

    of = open(data_dir + '/data_dict_' + fn + '.pickle', 'rb')
    data = pickle.load(of)

    kx = np.array(data['kx'])
    ky = np.array(data['ky'])
    traces = np.array(data['traces'])
    bands = np.array(data['bands'])
    ons = np.array(data['ons'])
    tvals = np.array(data['tvals'])
    pin = 1

    # print len(kx)

    ax_lat, cc, lc, fig = view_plot1(kx, ky, bands, traces, tvals[:1], ons[:2], vertex_points, -1, pnum=len(kx))
    lf.lattice_plot(R, Ni, Nk, ax_lat, cols, line_cols)
    plt.show()
    # plt.savefig('%0.2f.png'%deltaz)
    #
    tot = 0
    # deltas = arange(120, 215, 2)
    deltas = [180]
    phis = -np.arange(0, 80, 1)

    # movie_honeycomb_single_frame(220, 70, False)
    # plt.show()
    # plt.savefig('lattice_120_60.pdf')

    # for u in range(len(deltas)):
    #
    #    deltaz = deltas[u]
    #    phi = 0
    #    
    #    fig, ax_lat = movie_honeycomb_single_frame(deltaz, phi)
    #    plt.show()
    #    plt.savefig('/Users/lisa/Dropbox/Research/New_Chern_general/movie_images/'+'%04d.png'%tot)
    #    tot = tot + 1
    #    
    # for uu in range(len(phis)):
    #
    #    deltaz = deltas[u]
    #    phi = phis[uu]
    #  
    #    fig, ax_lat = movie_honeycomb_single_frame(deltaz, phi)  
    #  
    #    plt.savefig('/Users/lisa/Dropbox/Research/New_Chern_general/movie_images/'+'%04d.png'%tot)
    #    tot = tot + 1
