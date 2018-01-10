import numpy as np
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.lattice_elasticity as le
import matplotlib.pyplot as plt
import copy

'''Functions for visualizing gliding heavy symmetric top (gHST) networks'''


def le_plot_gHST(xy, NL, KL, BM, params, t, ii, name, outdir, climv=0.1, xlimv='auto',
                 exaggerate=1.0, PlanarLimit=False):
    """Plots a gliding heavy symmetric top network

    Parameters
    ----------
    xy : NP x 5 array
        positions of pivots (x,y) and euler angles for all HSTs
    t : float
        time stamp for image
    ii : int
        index for naming
    name : string
        the name of the file (before _index.png)
    outdir : string
        The output directory for the image
    climv : float or tuple
        Color limit for coloring bonds by bond strain
    xlimv
    exaggerate : float (default 1.0 --> in which case it is ignored)
        Exaggerate the displacements of each particle from its initial position by this factor. Ignored if == 1.0
    PlanarLimit : bool

    Returns
    ----------
    """
    # make output dir
    outdir = dio.prepdir(outdir)
    dio.ensure_dir(outdir)
    # set range of window from first values
    if xlimv == 'auto':
        xlimv = params['xlimv']
    ylimv = xlimv

    # save current data as stills
    index = '{0:08d}'.format(ii)
    outname = outdir + '/' + name + '_' + index + '.png'
    BL = NL2BL(NL, KL)

    if 'prestrain' in params:
        prestrain = params['prestrain']
    else:
        prestrain = 0.

    if 'shrinkrate' in params:
        shrinkrate = params['shrinkrate']
        title = 't = ' + '%09.1f' % t + '  a = ' + '%07.5f' % (1. - shrinkrate * t - prestrain)
    elif 'prestrain' in params:
        shrinkrate = 0.0
        title = 't = ' + '%09.1f' % t + '  a = ' + '%07.5f' % (1. - prestrain)
    else:
        shrinkrate = 0.0
        title = 't = ' + '%09.1f' % t

    if exaggerate != 1.0:
        title += ' magnify=' + str(exaggerate)

    if 'Omg' in params:
        title += '\n' + r'$\Omega_g$=' + '{0:.3f}'.format(params['Omg'])
    if 'OmK' in params:
        title += r' $\Omega_k$=' + '{0:.3f}'.format(params['OmK'])
    if 'g' in params:
        gstr = str(params['g'])
        if len(gstr.split('.')[1]) > 3:
            gstr = '{0:.3f}'.format(params['g'])
        title += '\n' + r'$g$=' + gstr
    if 'k' in params:
        kstr = str(params['k'])
        if len(kstr.split('.')[1]) > 3:
            kstr = '{0:.3f}'.format(params['k'])
        title += r' $k$=' + kstr
    if 'Mm' in params:
        try:
            mstr = str(params['Mm'][0])
            if len(mstr.split('.')[1]) > 3:
                mstr = '{0:.3f}'.format(params['Mm'][0])
            title += r' $m$=' + mstr
        except:
            mstr = str(params['Mm'])
            if len(mstr.split('.')[1]) > 3:
                mstr = '{0:.3f}'.format(params['Mm'])
            title += r' $m$=' + mstr
    if 'l' in params:
        try:
            lstr = str(params['l'][0])
            if len(lstr.split('.')[1]) > 3:
                lstr = '{0:.3f}'.format(params['l'][0])
            title += r' $l$=' + lstr
        except:
            lstr = str(params['l'])
            if len(lstr.split('.')[1]) > 3:
                lstr = '{0:.3f}'.format(params['l'])
            title += r' $l$=' + lstr
    if 'w3' in params:
        try:
            if (params['w3'] - params['w3'][0] < 1e-5).all():
                wstr = str(params['w3'][0])
                if len(lstr.split('.')[1]) > 3:
                    wstr = '{0:.3f}'.format(params['w3'][0])
                title += r' $\omega_3$=' + wstr
                # Otherwise, don't print w3: it is heterogeneous
        except:
            wstr = str(params['w3'])
            if len(wstr.split('.')[1]) > 3:
                wstr = '{0:.3f}'.format(params['w3'])
            title += r' $\omega_3$=' + wstr

    if params['BCtype'] == 'excite':
        title += r' $\omega_d$=' + '{0:.5f}'.format(params['frequency'])

    # calculate strain
    # bs = bond_strain_list(xy,BL,bL0)
    # if exaggerate==1.0:
    #   movie_plot_2D_gyros(xy, BL, bs, outname, title, xlimv, ylimv, climv)
    # else:
    #   xye = xy0+(xy-xy0)*exaggerate
    #   movie_plot_2D_gyros(xye, BL, bs, outname, title, xlimv, ylimv, climv)

    # set limits
    fig = plt.gcf()
    plt.clf()

    ax = plt.gca()
    ax.set_xlim(-xlimv, xlimv)
    ax.set_ylim(-ylimv, ylimv)

    if PlanarLimit:
        gHST_plot_PL(xy, NL, KL, BM, params, factor=exaggerate, climv=climv, title=title)
    else:
        gHST_plot(xy, NL, KL, BM, params, factor=exaggerate, climv=climv, title=title)

    plt.savefig(outname)


def le_plot_gyros(xy, xy0, NL, KL, BM, params, t, ii, name, fig, ax, outdir, climv='auto', exaggerate=1.0,
                  dpi=300, fontsize=12, title='', **kwargs):
    """Plots a single gyroscopic lattice time step using timestep plot.

    Parameters
    ----------
    t : float
        time stamp for image
    ii : int
        index to name file ( ie 'name_000ii.png')
    name : string
        the name of the file (before _index.png)
    fig : matplotlib.pyplot figure handle, or 'none'
        the figure on which to plot the gyros. If 'none', uses plt.gcf() and clears figure.
    ax : matplotlib.pyplot axis handle, or 'none'
        the axis on which to plot the gyros. If 'none', uses plt.gca()
    outdir : string
        The output directory for the image
    climv : float or tuple
        Color limit for coloring bonds by bond strain, overriddes params['climv'] if climv!='auto'
        If 'climv' is not a key in params, and climv=='auto', then uses default min/max.
        If 'climv' is a key in params, and climv=='auto', then uses params['climv'].
    exaggerate : float (default 1.0 --> in which case it is ignored)
        Exaggerate the displacements of each particle from its initial position by this factor. Ignored if == 1.0
    dpi : float
        pixels per inch, resolution of saved plot
    **kwargs: Additional timestep_plot() keyword arguments
        color_particles='k', fontsize=14, linewidth=2

    Returns
    ----------
    """
    # If fig and ax are not supplied, declare them
    if fig is None or fig == 'none':
        fig = plt.gcf()
        plt.clf()
    if ax is None or ax == 'none':
        ax = plt.gca()

    # make output dir
    outdir = dio.prepdir(outdir)
    dio.ensure_dir(outdir)
    # set range of window from first values
    if 'xlimv' in params:
        xlimv = params['xlimv']
        if 'ylimv' in params:
            ylimv = params['ylimv']
        else:
            ylimv = (xlimv - max(xy0[:, 0])) + np.ceil(max(xy0[:, 1]))
    else:
        xlimv = np.ceil(max(xy0[:, 0]) * 5./ 4.)
        ylimv = (xlimv - max(xy0[:, 0])) + max(xy0[:, 1])

    # save current data as stills
    index = '{0:08d}'.format(ii)
    outname = outdir + '/' + name + '_' + index + '.png'
    BL = le.NL2BL(NL, KL)
    # print 'BL = ', BL

    suptitle = copy.deepcopy(title)

    if 'prestrain' in params:
        prestrain = params['prestrain']
    else:
        prestrain = 0.

    if 'shrinkrate' in params:
        shrinkrate = params['shrinkrate']
        title = 't = ' + '%07.0f' % t + '  a = ' + '%07.5f' % (1. - shrinkrate * t - prestrain)
    elif 'prestrain' in params:
        shrinkrate = 0.0
        title = 't = ' + '%07.0f' % t + '  a = ' + '%07.5f' % (1. - prestrain)
    else:
        shrinkrate = 0.0
        title = 't = ' + '%07.0f' % t + r' $\Omega_g^{-1}$'

    if exaggerate != 1.0:
        title += '   amplified ' + str(int(exaggerate)) + 'x  '

    if 'Omk' in params and params['Omk'] != -1.:
        title += '\n' + r'$\Omega_g$=' + '{0:.3f}'.format(params['Omg'])
    if 'Omg' in params and params['Omk'] != -1.:
        title += r'   $\Omega_k$=' + '{0:.3f}'.format(params['Omk'][0, 0])
    if 'split_spin' in params:
        title += ' NV=' + str(int(params['NV']))
        if 'split_k' in params:
            title += r'   $k_s$=' + str(params['split_k'])

    # title +='\n'

    # if params['BCtype'] == 'excite':
    #     title += r'   $\omega_d$ = ' + '{0:.3f}'.format(params['frequency'])

    if 'eta' in params:
        if params['eta'] != 0.000:
            title += r'   $\eta$=' + '{0:.3f}'.format(params['eta'])

            # calculate strain
            # bs = bond_strain_list(xy,BL,bL0)

            # if exaggerate==1.0:
            # movie_plot_2D_gyros(xy, BL, bs, outname, title, xlimv, ylimv, climv)
            # else:
            # xye = xy0+(xy-xy0)*exaggerate
            # movie_plot_2D_gyros(xye, BL, bs, outname, title, xlimv, ylimv, climv)

    if climv == 'auto':
        if 'climv' in params:
            climv = params['climv']

    [scat_fg, lines_st, p] = leplt.timestep_plot(xy, xy0, NL, KL, BM, ax=ax, factor=exaggerate, amp=climv, title=title,
                                                 fontsize=fontsize, suptitle=suptitle, **kwargs)

    # print 'color_particles = ', scat_fg
    ax.set_xlim(-xlimv, xlimv)
    ax.set_ylim(-ylimv, ylimv)

    plt.savefig(outname, dpi=dpi)

    # clear_array = [scat_fg, lines_st, p]
    # for i in range(len(clear_array)):
    #     clear_array[i].remove()

    return [scat_fg, lines_st, p]
