import lepm.kitaev.kitaev_collection_functions as kcollfns
import lepm.dataio as dio
import lepm.haldane.haldane_lattice_functions as hlatfns
import lepm.lattice_functions as latfns
import matplotlib.pyplot as plt
import numpy as np
import lepm.data_handling as dh
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov

'''Auxiliary functions for KitaevChernGyroCollection class'''


def retrieve_param_value(lp_value):
    """

    Parameters
    ----------
    lp_value

    Returns
    -------

    """
    return kcollfns.retrieve_param_value(lp_value)


def plot_cherns_vary_param(kcgcoll, rootdir=None, param_type='hlat', sz_param_nu=None,
                           reverse=False, param='percolation_density',
                           title='Chern index calculation', xlabel=None):
    """Plot the 1d curve of chern number as a function of the varying lattice parameter

    Parameters
    ----------
    kcgcoll :
    param_type :
    sz_param_nu :
    reverse :
    param :
    title :
    xlabel :

    Returns
    -------
    """
    if rootdir is None:
        rootdir = kcgcoll.cp['rootdir']

    if sz_param_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_nu = kcgcoll.collect_cherns_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'hlat':
            param_nu = kcgcoll.collect_cherns_vary_hlatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'hlat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_nu[:, 0]
    nu = param_nu[:, 1]

    # Make figure
    import lepm.plotting.plotting as leplt
    fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=4)

    if xlabel is None:
        xlabel = param.replace('_', ' ')

    # Plot the curve
    # first sort the paramV values
    if isinstance(paramV[0], float):
        si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)
    ax.plot(paramV[si], nu[si], '.-')
    ax.set_xlabel(xlabel)

    # Add title
    ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

    # Save the plot
    outdir, outbase, outd_ex = build_outdir_kchern_varyparam(param, kcgcoll, param_type, rootdir)
    dio.ensure_dir(outdir)
    fname = outdir + outbase
    fname += '_chern_' + param + '_Ncoll' + '{0:03d}'.format(len(kcgcoll.haldane_collection.haldane_lattices))
    fname += outd_ex
    print 'saving to ' + fname + '.png'
    plt.savefig(fname + '.png')
    plt.clf()


def plot_chernbands_vary_param(kcgcoll, rootdir=None, param_type='hlat', param_band_nu=None,
                               reverse=False, param='percolation_density',
                               title='Chern index calculation', xlabel=None, round_chern=False, vmin=-1, vmax=1,
                               alpha=1.0, ymax=None, absx=False, logparam=False, **kwargs):
    """Plot the band limits colored by chern number as a function of the varying lattice parameter.

    Parameters
    ----------
    kcgcoll :
    param_type :
    sz_param_nu :
    reverse :
    param :
    title :
    xlabel :
    round_chern : bool
        whether to round the value of the chern number to nearest integer for coloring
    vmin : float, int, or None
        The minimum value for the chern number coloring
    vmax : float, int, or None
        The maximum value for the chern number coloring

    Returns
    -------
    """
    if rootdir is None:
        rootdir = kcgcoll.cp['rootdir']

    # Note that param_band_nu has rows [paramval, band0min, band0max, nu0, band1min, band1max, nu1, ...]
    if param_band_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_band_nu = kcgcoll.collect_chernbands_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'hlat':
            param_band_nu = kcgcoll.collect_chernbands_vary_hlatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'hlat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_band_nu[:, 0]
    if absx:
        paramV = np.abs(paramV)

    # first sort the paramV values
    if isinstance(paramV[0], float) or isinstance(paramV[0], int):
        si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)

    nbands = (np.shape(param_band_nu)[1] - 1) / 3
    params = paramV[si]
    param_band_nu = param_band_nu[si]
    # print 'np.shape(param_band_nu) = ', np.shape(param_band_nu)
    # print '(3 * (np.arange(nbands) + 1) - 2) = ', (3 * (np.arange(nbands) + 1) - 2)
    bandmins = param_band_nu[:, (3 * (np.arange(nbands) + 1) - 2)]
    bandmaxs = param_band_nu[:, (3 * (np.arange(nbands) + 1) - 1)]
    nus = param_band_nu[:, 3 * (np.arange(nbands) + 1)]

    if vmin is None:
        vmin = - round(np.max(np.abs(nus.ravel())))
        if vmin > -0.5:
            vmin = -1
    if vmax is None:
        vmax = round(np.max(np.abs(nus.ravel())))
        if vmax > 0.5:
            vmax = 1
    if vmin == -1:
        cmap = lecmaps.ensure_cmap('rbb0')
    elif vmin == -2:
        cmap = lecmaps.ensure_cmap('gbbro')

    # Make figure
    fig, ax = leplt.initialize_1panel_centered_fig(**kwargs)

    if xlabel is None:
        paramlabel = leplt.param2description_haldane(param)
        if logparam:
            xlabel = r'$\log_{10}($' + paramlabel + r'$)$'
        else:
            xlabel = paramlabel

    # Plot the band structure colored by the Chern number
    dps = np.diff(params).tolist()
    dps.append(dps[-1])
    dps = np.array(dps)
    if ymax is None:
        find_ymax = True
    else:
        find_ymax = False
    for (pval, bmins, bmaxs, nuv, dp) in zip(params, bandmins, bandmaxs, nus, dps):
        for (bmin, bmax, nu) in zip(bmins, bmaxs, nuv):
            if logparam:
                polygon = np.array([[np.log10(pval), bmin], [np.log10(pval + dp), bmin],
                                    [np.log10(pval + dp), bmax], [np.log10(pval), bmax]])
            else:
                polygon = np.array([[pval, bmin], [pval + dp, bmin],
                                    [pval + dp, bmax], [pval, bmax]])
                if find_ymax:
                    ymax = max(max(abs(bmax), abs(bmin)), ymax)

            if round_chern:
                colorval = (round(np.real(nu)) - vmin) / (vmax - vmin)
            else:
                # Don't round to the nearest integer
                colorval = (np.real(nu) - vmin) / (vmax - vmin)

            # Add approx bounding polygon to the axis
            poly = Polygon(polygon, closed=True, fill=True, lw=0.00, alpha=alpha, color=cmap(colorval), edgecolor=None)
            ax.add_artist(poly)

    xmin, xmax = np.min(params), np.max(params) + dps[-1]
    if logparam:
        ax.set_xlim(np.log10(xmin), np.log10(xmax))
    else:
        ax.set_xlim(xmin, xmax)
    ax.set_ylim(-ymax, ymax)

    # Add title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('frequency, $\omega / t_1$')
    ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

    # Save the plot
    if param_type == 'hlat':
        outdir = rootdir + 'kspace_cherns_haldane/chern_hlatparam/' + param + '/'
        dio.ensure_dir(outdir)

        # Add meshfn name to the output filename
        outbase = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn']
        if outbase[-1] == '/':
            outbase = outbase[:-1]
        outbase = outbase.split('/')[-1]

        outd_ex = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn_exten']
        # If the parameter name is part of the meshfn_exten, replace its value with XXX in
        # the meshfnexten part of outdir.
        mfestr = hlatfns.param2meshfnexten_name(param)
        if mfestr in outd_ex:
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outd_ex.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            print 'odrest = ', odrest
            print 'od2val_rest = ', od2val_rest
            outd_ex = od_split[0] + param + 'XXX'
            outd_ex += odrest
            print 'outd_ex = ', outd_ex
        else:
            outd_ex += '_' + param + 'XXX'
    elif param_type == 'lat':
        outdir = rootdir + 'kspace_cherns_haldane/chern_lpparam/' + param + '/'
        outbase = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn']
        # Take apart outbase to parse out the parameter that is varying
        mfestr = latfns.param2meshfnexten_name(param)
        if mfestr in outbase:
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outbase.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            print 'odrest = ', odrest
            print 'od2val_rest = ', od2val_rest
            outd_ex = od_split[0] + param + 'XXX'
            outd_ex += odrest
            print 'outd_ex = ', outd_ex
        else:
            outbase += '_' + param + 'XXX'

        outd_ex = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn_exten']

    dio.ensure_dir(outdir)
    fname = outdir + outbase
    fname += '_chernbands_' + param + '_Ncoll' + '{0:03d}'.format(len(kcgcoll.haldane_collection.haldane_lattices))
    fname += outd_ex
    print 'saving to ' + fname + '.pdf'
    plt.savefig(fname + '.pdf')
    plt.clf()


def movie_chernbands_vary_param(kcgcoll, rootdir=None, param_type='hlat', param_band_nu=None,
                                reverse=False, param='percolation_density',
                                title='Chern index calculation', ax=None,
                                xlabel=None, round_chern=False, vmin=-1, vmax=1,
                                alpha=1.0, ymax=None, absx=False, framerate=10, cbar_ticks=None):
    """Plot the 2d dispersion with bands colored by chern number as a function of the varying lattice parameter

    Parameters
    ----------
    kcgcoll :
    param_type :
    sz_param_nu :
    reverse :
    param :
    title :
    xlabel :
    round_chern : bool
        whether to round the value of the chern number to nearest integer for coloring
    vmin : float, int, or None
        The minimum value for the chern number coloring
    vmax : float, int, or None
        The maximum value for the chern number coloring
    alpha : float
        the opacity of the chern-colored bands
    ymax : float
        the maximum value for the ylims of each plot
    absx : bool
        sort the paramV by abs(paramV)
    framerate : int
        the frame rate of the movie to be made

    Returns
    -------
    """
    if rootdir is None:
        rootdir = kcgcoll.cp['rootdir']

    # Note that param_band_nu has rows [paramval, band0min, band0max, nu0, band1min, band1max, nu1, ...]
    if param_band_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_band_nu = kcgcoll.collect_chernbands_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'hlat':
            param_band_nu = kcgcoll.collect_chernbands_vary_hlatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'hlat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_band_nu[:, 0]

    # first sort the paramV values
    if isinstance(paramV[0], float) or isinstance(paramV[0], int):
        if absx:
            si = np.argsort(np.abs(paramV))
        else:
            si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)

    nbands = (np.shape(param_band_nu)[1] - 1) / 3
    params = paramV[si]
    param_band_nu = param_band_nu[si]
    # print 'np.shape(param_band_nu) = ', np.shape(param_band_nu)
    # print '(3 * (np.arange(nbands) + 1) - 2) = ', (3 * (np.arange(nbands) + 1) - 2)
    bandmins = param_band_nu[:, (3 * (np.arange(nbands) + 1) - 2)]
    bandmaxs = param_band_nu[:, (3 * (np.arange(nbands) + 1) - 1)]
    nus = param_band_nu[:, 3 * (np.arange(nbands) + 1)]

    if vmin is None:
        vmin = - round(np.max(np.abs(nus.ravel())))
        if vmin > -0.5:
            vmin = -1
    if vmax is None:
        vmax = round(np.max(np.abs(nus.ravel())))
        if vmax > 0.5:
            vmax = 1

    if vmin == -1:
        cmap = lecmaps.ensure_cmap('bbr0')
    elif vmin == -2:
        cmap = lecmaps.ensure_cmap('gbbro')
    if ymax is None:
        ymax = np.max(np.abs(bandmaxs)) * 1.2

    # Make figure
    if ax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(wsfrac=0.5, tspace=4)

    if xlabel is None:
        xlabel = param.replace('_', ' ')

    # Naming for directory in which to save
    outdir, outbase, outd_ex = build_outdir_kchern_varyparam(param, kcgcoll, param_type, rootdir)

    dio.ensure_dir(outdir)
    fname = outdir + outbase
    fname += '_chernbands_' + param + '_Ncoll' + '{0:03d}'.format(len(kcgcoll.haldane_collection.haldane_lattices))
    fname += outd_ex
    dio.ensure_dir(fname + '/')

    # Plot the band structure colored by the Chern number
    jj = 0
    for key in kcgcoll.cherns:
        print 'key = ', key
        kcherns = kcgcoll.cherns[key]
        if len(kcherns) > 1:
            raise RuntimeError('Did not expect multiple kcherns in kcgcoll.cherns[key] list. Handle here.')
        else:
            kchern = kcherns[0]
        print 'kchern = ', kcherns
        # vertex_points = kchern.chern['bzvtcs']
        kkx = kchern.chern['kx']
        # kky = kchern.chern['ky']
        bands = kchern.chern['bands']
        chern = kchern.chern['chern']
        # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)

        if isinstance(cmap, str):
            cmap = lecmaps.ensure_cmap(cmap)

        xmax = 0.
        for kk in range(len(bands[0, :])):
            band = bands[:, kk]
            # If round_chern is True, round Chern number to the nearest integer
            if round_chern:
                colorval = (round(np.real(chern[kk])) - vmin) / (vmax - vmin)
            else:
                # Don't round to the nearest integer
                colorval = (np.real(chern[kk]) - vmin) / (vmax - vmin)
            # ax.plot(kkx, band, color=cmap(colorval))
            polygon = dh.approx_bounding_polygon(np.dstack((kkx, band))[0], ngridpts=np.sqrt(len(kkx)) * 0.5)
            # Add approx bounding polygon to the axis
            poly = Polygon(polygon, closed=True, fill=True, lw=0.00, alpha=alpha, color=cmap(colorval),
                           edgecolor=None)
            xmax = max(xmax, np.max(np.abs(kkx)))
            ax.add_artist(poly)
            # ax.plot(polygon[:, 0], polygon[:, 1], 'k.-')

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)

        # Add title
        ax.set_xlabel('wavenumber, $k_x$')
        ax.set_ylabel('frequency, $\omega$')
        ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

        # Do colorbar
        if cbar_ticks is None:
            cbar_ticks = [int(vmin), int(vmax)]

        sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sm, cax=cax, ticks=cbar_ticks, label=r'Chern number, $C$', alpha=alpha)

        # Save to file, but get index of this image in params
        paramval = retrieve_param_value(kchern.haldane_lattice.lp[param])
        index = np.where(paramval == params)[0][0]
        print 'index = ', index
        fname_out = fname + '/chernbands_{0:06d}'.format(index) + '.png'
        print 'saving to ' + fname_out + '.png'
        plt.savefig(fname_out)
        ax.cla()
        jj += 1

    lemov.make_movie(fname + '/chernbands_', fname, indexsz='06', framerate=framerate, rm_images=True,
                     save_into_subdir=True, imgdir=fname)

    #########################################################################
    # Now make move with ky as the independent axis
    # Plot the band structure colored by the Chern number
    #########################################################################
    dio.ensure_dir(fname + '_ky/')
    jj = 0
    for key in kcgcoll.cherns:
        print 'key = ', key
        kcherns = kcgcoll.cherns[key]
        if len(kcherns) > 1:
            raise RuntimeError('Did not expect multiple kcherns in kcgcoll.cherns[key] list. Handle here.')
        else:
            kchern = kcherns[0]
        print 'kchern = ', kcherns
        # vertex_points = kchern.chern['bzvtcs']
        kky = kchern.chern['ky']
        # kky = kchern.chern['ky']
        bands = kchern.chern['bands']
        chern = kchern.chern['chern']
        # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)

        if isinstance(cmap, str):
            cmap = lecmaps.ensure_cmap(cmap)

        xmax = 0.
        for kk in range(len(bands[0, :])):
            band = bands[:, kk]
            # If round_chern is True, round Chern number to the nearest integer
            if round_chern:
                colorval = (round(np.real(chern[kk])) - vmin) / (vmax - vmin)
            else:
                # Don't round to the nearest integer
                colorval = (np.real(chern[kk]) - vmin) / (vmax - vmin)
            # ax.plot(kkx, band, color=cmap(colorval))
            polygon = dh.approx_bounding_polygon(np.dstack((kky, band))[0], ngridpts=np.sqrt(len(kkx)) * 0.5)
            # Add approx bounding polygon to the axis
            poly = Polygon(polygon, closed=True, fill=True, lw=0.00, alpha=alpha, color=cmap(colorval),
                           edgecolor=None)
            xmax = max(xmax, np.max(np.abs(kkx)))
            ax.add_artist(poly)
            # ax.plot(polygon[:, 0], polygon[:, 1], 'k.-')

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-ymax, ymax)

        # Add title
        ax.set_xlabel('wavenumber, $k_y$')
        ax.set_ylabel('frequency, $\omega$')
        ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

        # Do colorbar
        if cbar_ticks is None:
            cbar_ticks = [int(vmin), int(vmax)]

        sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sm, cax=cax, ticks=cbar_ticks, label=r'Chern number, $C$', alpha=alpha)

        # Save to file, but get index of this image in params
        paramval = retrieve_param_value(kchern.haldane_lattice.lp[param])
        index = np.where(paramval == params)[0][0]
        print 'index = ', index
        fname_out = fname + '_ky/chernbands_{0:06d}'.format(index) + '.png'
        print 'saving to ' + fname_out + '.png'
        plt.savefig(fname_out)
        ax.cla()
        jj += 1

    lemov.make_movie(fname + '_ky/chernbands_', fname + '_ky', indexsz='06', framerate=framerate, rm_images=True,
                     save_into_subdir=True, imgdir=fname + '_ky')


def movie_berrybands_vary_param(kcgcoll, rootdir=None, param_type='hlat', param_band_nu=None,
                                reverse=False, param='percolation_density',
                                title=None, ax=None, vmin=None, vmax=None,
                                alpha=1.0, absx=False, framerate=5, cbar_ticks=None, ngrid=50):
    """Plot the 2d berry curvature over the BZ, with each frame a value of the varying lattice/haldanelattice parameter

    Parameters
    ----------
    kcgcoll :
    param_type :
    param_band_nu : float array or None
    reverse :
    param :
    title :
    xlabel :
    round_chern : bool
        whether to round the value of the chern number to nearest integer for coloring
    vmin : float, int, or None
        The minimum value for the chern number coloring
    vmax : float, int, or None
        The maximum value for the chern number coloring
    alpha : float
        the opacity of the chern-colored bands
    ymax : float
        the maximum value for the ylims of each plot
    absx : bool
        sort the paramV by abs(paramV)
    framerate : int
        the frame rate of the movie to be made

    Returns
    -------
    """
    if rootdir is None:
        rootdir = kcgcoll.cp['rootdir']

    # Note that param_band_nu has rows [paramval, band0min, band0max, nu0, band1min, band1max, nu1, ...]
    # Note that this is just to get the paramV to sort it, as well as to count the number of bands
    if param_band_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_band_nu = kcgcoll.collect_chernbands_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'hlat':
            param_band_nu = kcgcoll.collect_chernbands_vary_hlatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'hlat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_band_nu[:, 0]

    # first sort the paramV values
    if isinstance(paramV[0], float) or isinstance(paramV[0], int):
        if absx:
            si = np.argsort(np.abs(paramV))
        else:
            si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)

    nbands = (np.shape(param_band_nu)[1] - 1) / 3
    params = paramV[si]
    hlatnames = np.array(kcgcoll.hlat_names)[si]
    param_band_nu = param_band_nu[si]
    nus = param_band_nu[:, 3 * (np.arange(nbands) + 1)]
    cmap = lecmaps.ensure_cmap('rwb0')

    # Make figure
    if ax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(Wfig=180, wsfrac=0.5, tspace=10, y0frac=0.08, x0frac=0.15)

    paramlabel = leplt.param2description(param)

    outdir, outbase, outd_ex = build_outdir_kchern_varyparam(param, kcgcoll, param_type, rootdir)
    dio.ensure_dir(outdir)
    fname0 = outdir + 'berrybands/' + outbase
    fname1 = fname0 + '_berrybands_' + param + '_Ncoll' + '{0:03d}'.format(len(kcgcoll.haldane_collection.haldane_lattices))
    fname = fname1 + outd_ex
    dio.ensure_dir(fname + '/')

    # Prepare title and limits
    if title is not None:
        titlelock = True
    else:
        titlelock = False

    if vmin is not None and vmax is not None:
        vminmax_lock = True
    else:
        vminmax_lock = False

    # Get vmin, vmax from berry arrays
    vmin, vmax = 0., 0.
    if not vminmax_lock:
        for key in kcgcoll.cherns:
            berry = kcgcoll.cherns[key][0].chern['traces']
            # print 'berrykk = ', berrykk
            # sys.exit()
            vmin = min(vmin, -0.2 * np.max(np.abs(berry.ravel())))
            vmax = max(vmax, 0.2 * np.max(np.abs(np.abs(berry.ravel()))))

    # Plot the berry curvature over the BZ as heatmap
    jj = 0
    for key in kcgcoll.cherns:
        print 'key = ', key
        kcherns = kcgcoll.cherns[key]
        if len(kcherns) > 1:
            raise RuntimeError('Did not expect multiple kcherns in kcgcoll.cherns[key] list. Handle here.')
        else:
            kchern = kcherns[0]
        print 'kchern = ', kcherns
        vertex_points = kchern.chern['bzvtcs']
        kkx = kchern.chern['kx']
        kky = kchern.chern['ky']
        bands = kchern.chern['bands']
        cherns = kchern.chern['chern']
        berry = kchern.chern['traces']
        # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)

        # Get index and parameter value of this hlat
        paramval = retrieve_param_value(kchern.haldane_lattice.lp[param])
        index = np.where(paramval == params)[0][0]

        if isinstance(cmap, str):
            cmap = lecmaps.ensure_cmap(cmap)

        # Create int array indexing bands to plot, but start halfway up and only do positive eigval bands
        todo = np.arange(int(0.5 * len(bands[0, :])), len(bands[0, :]))
        for kk in todo:
            berrykk = berry[:, kk]

            # Plot berry curvature as heatmap
            xx, yy, zz = dh.interpol_meshgrid(kkx, kky, np.imag(berrykk), ngrid, method='nearest')
            ax.pcolormesh(xx, yy, zz, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

            # Also draw the BZ
            patch_handle = mask_ax_to_bz(ax, vertex_points)
            # poly = Polygon(vertex_points, closed=True, fill=True, lw=1, alpha=1.0, facecolor='none', edgecolor='k')
            # ax.add_artist(poly)

            # Add title
            ax.set_xlabel('wavenumber, $k_x$')
            ax.set_ylabel('wavenumber, $k_y$')
            if not titlelock:
                title = r'Berry curvature, $\nu = $' + \
                        '{0:0.1f}'.format(float(np.real(cherns[kk]))) + ' ' + \
                        paramlabel + '=' + '{0:0.2f}'.format(float(paramval))
            ax.text(0.5, 1.04, title, transform=ax.transAxes, ha='center', va='center')

            print 'np.max(np.abs(berrykk)) = ', np.max(np.abs(berrykk))

            # Do colorbar
            if cbar_ticks is None:
                cbar_ticks = [vmin, vmax]

            sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(sm, cax=cax, ticks=cbar_ticks, label=r'Berry curvature, $\mathcal{F}$', alpha=alpha)

            # Save to file, but get index of this image in params
            fname_out = fname + '/berrybands_band' + '{0:03d}_'.format(kk) + '{0:06d}'.format(index) + '.png'
            dio.ensure_dir(fname + '/')
            print 'saving to ' + fname_out

            # xy limits
            vpr = vertex_points.ravel()
            ax.set_xlim(np.min(vpr), np.max(vpr))
            ax.set_ylim(np.min(vpr), np.max(vpr))
            plt.savefig(fname_out)
            ax.cla()
            cax.cla()

        jj += 1

    # Make movies for each band with positive eigval
    for kk in todo:
        print 'todo = ', todo
        imgdir = fname + '/'
        imgname = fname + '/berrybands_band' + '{0:03d}_'.format(kk)
        movname = fname + '_berrybands_band' + '{0:03d}'.format(kk)
        if kk == todo[-1]:
            lemov.make_movie(imgname, movname, indexsz='06', framerate=framerate, rm_images=True, save_into_subdir=True,
                             imgdir=imgdir)
        else:
            lemov.make_movie(imgname, movname, indexsz='06', framerate=framerate)


def movie_freqbands_vary_param(kcgcoll, rootdir=None, param_type='hlat', param_band_nu=None,
                               reverse=False, param='percolation_density',
                               title=None, ax=None, vmin=None, vmax=None, cmap='gist_earth',
                               alpha=1.0, absx=False, framerate=5, cbar_ticks=None, ngrid=50):
    """Plot the 2d berry curvature over the BZ, with each frame a value of the varying lattice/haldanelattice parameter

    Parameters
    ----------
    kcgcoll :
    param_type :
    param_band_nu : float array or None
    reverse :
    param :
    title :
    xlabel :
    round_chern : bool
        whether to round the value of the chern number to nearest integer for coloring
    vmin : float, int, or None
        The minimum value for the chern number coloring
    vmax : float, int, or None
        The maximum value for the chern number coloring
    alpha : float
        the opacity of the chern-colored bands
    ymax : float
        the maximum value for the ylims of each plot
    absx : bool
        sort the paramV by abs(paramV)
    framerate : int
        the frame rate of the movie to be made

    Returns
    -------
    """
    if rootdir is None:
        rootdir = kcgcoll.cp['rootdir']

    # Note that param_band_nu has rows [paramval, band0min, band0max, nu0, band1min, band1max, nu1, ...]
    # Note that this is just to get the paramV to sort it, as well as to count the number of bands
    if param_band_nu is None:
        if param_type == 'lat' or param_type == 'lp':
            param_band_nu = kcgcoll.collect_chernbands_vary_lpparam(param=param, reverse=reverse)
        elif param_type == 'hlat':
            param_band_nu = kcgcoll.collect_chernbands_vary_hlatparam(param=param, reverse=reverse)
        else:
            raise RuntimeError("param_type argument passed is not 'hlat' or 'lat/lp'")

    # Plot it as colormap
    plt.close('all')

    paramV = param_band_nu[:, 0]

    # first sort the paramV values
    if isinstance(paramV[0], float) or isinstance(paramV[0], int):
        if absx:
            si = np.argsort(np.abs(paramV))
        else:
            si = np.argsort(paramV)
    else:
        si = np.arange(len(paramV), dtype=int)

    nbands = (np.shape(param_band_nu)[1] - 1) / 3
    params = paramV[si]
    hlatnames = np.array(kcgcoll.hlat_names)[si]
    param_band_nu = param_band_nu[si]
    nus = param_band_nu[:, 3 * (np.arange(nbands) + 1)]
    cmap = lecmaps.ensure_cmap(cmap)

    # Make figure
    if ax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(Wfig=180, wsfrac=0.5, tspace=10, y0frac=0.08, x0frac=0.15)

    paramlabel = leplt.param2description(param)

    outdir, outbase, outd_ex = build_outdir_kchern_varyparam(param, kcgcoll, param_type, rootdir)
    dio.ensure_dir(outdir)
    fname0 = outdir + 'eigvalbands/' + outbase
    fname1 = fname0 + '_eigvalbands_' + param + '_Ncoll' + '{0:03d}'.format(len(kcgcoll.haldane_collection.haldane_lattices))
    fname = fname1 + outd_ex
    dio.ensure_dir(fname + '/')

    # Prepare title and limits
    if title is not None:
        titlelock = True
    else:
        titlelock = False

    if vmin is not None and vmax is not None:
        vminmax_lock = True
    else:
        vminmax_lock = False

    # Get vmin, vmax from berry arrays
    first = True
    if not vminmax_lock:
        for key in kcgcoll.cherns:
            eigvals = kcgcoll.cherns[key][0].chern['bands']
            if first:
                vmin, vmax = np.max(eigvals.ravel()) * np.ones(len(eigvals[0])), np.zeros(len(eigvals[0]))
            for kk in range(len(eigvals[0])):
                vmin[kk] = min(vmin[kk], np.min(np.abs(eigvals[:, kk].ravel())))
                vmax[kk] = max(vmax[kk], np.max(np.abs(eigvals[:, kk].ravel())))

            first = False

    # Plot the berry curvature over the BZ as heatmap
    jj = 0
    for key in kcgcoll.cherns:
        print 'key = ', key
        kcherns = kcgcoll.cherns[key]
        if len(kcherns) > 1:
            raise RuntimeError('Did not expect multiple kcherns in kcgcoll.cherns[key] list. Handle here.')
        else:
            kchern = kcherns[0]
        print 'kchern = ', kcherns
        vertex_points = kchern.chern['bzvtcs']
        kkx = kchern.chern['kx']
        kky = kchern.chern['ky']
        bands = kchern.chern['bands']
        cherns = kchern.chern['chern']
        # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)

        # Get index and parameter value of this hlat
        paramval = retrieve_param_value(kchern.haldane_lattice.lp[param])
        index = np.where(paramval == params)[0][0]

        if isinstance(cmap, str):
            cmap = lecmaps.ensure_cmap(cmap)

        # Create int array indexing bands to plot, but start halfway up and only do positive eigval bands
        todo = np.arange(int(0.5 * len(bands[0, :])), len(bands[0, :]))
        for kk in todo:
            bandkk = bands[:, kk]

            if vminmax_lock:
                vminkk, vmaxkk = vmin, vmax
            else:
                vminkk, vmaxkk = vmin[kk], vmax[kk]

            # Plot berry curvature as heatmap
            xx, yy, zz = dh.interpol_meshgrid(kkx, kky, np.real(bandkk), ngrid, method='nearest')
            print 'np.min(zz) = ', np.min(zz)
            print 'np.max(zz) = ', np.max(zz)
            ax.pcolormesh(xx, yy, zz, cmap=cmap, vmin=vminkk, vmax=vmaxkk, alpha=alpha)

            # Also draw the BZ
            patch_handle = mask_ax_to_bz(ax, vertex_points)
            # poly = Polygon(vertex_points, closed=True, fill=True, lw=1, alpha=1.0, facecolor='none', edgecolor='k')
            # ax.add_artist(poly)

            # Add title
            ax.set_xlabel('wavenumber, $k_x$')
            ax.set_ylabel('wavenumber, $k_y$')
            if not titlelock:
                title = r'band frequencies, $\nu = $' + \
                        '{0:0.1f}'.format(float(np.real(cherns[kk]))) + ' ' + \
                        paramlabel + '=' + '{0:0.2f}'.format(float(paramval))
            ax.text(0.5, 1.04, title, transform=ax.transAxes, ha='center', va='center')

            print 'np.max(np.abs(bandkk)) = ', np.max(np.abs(bandkk))

            # Do colorbar
            cbar_ticks = [vminkk, vmaxkk]

            sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vminkk, vmax=vmaxkk)
            plt.colorbar(sm, cax=cax, ticks=cbar_ticks, label=r'frequency, $\omega$', alpha=alpha)

            # Save to file, but get index of this image in params
            fname_out = fname + '/eigvalbands_band' + '{0:03d}_'.format(kk) + '{0:06d}'.format(index) + '.png'
            dio.ensure_dir(fname + '/')
            print 'saving to ' + fname_out

            # xy limits
            vpr = vertex_points.ravel()
            ax.set_xlim(np.min(vpr), np.max(vpr))
            ax.set_ylim(np.min(vpr), np.max(vpr))
            plt.savefig(fname_out)
            ax.cla()
            cax.cla()

        jj += 1

    # Make movies for each band with positive eigval
    for kk in todo:
        print 'todo = ', todo
        imgdir = fname + '/'
        imgname = fname + '/eigvalbands_band' + '{0:03d}_'.format(kk)
        movname = fname + '_eigvalbands_band' + '{0:03d}'.format(kk)
        if kk == todo[-1]:
            lemov.make_movie(imgname, movname, indexsz='06', framerate=framerate, rm_images=True, save_into_subdir=True,
                             imgdir=imgdir)
        else:
            lemov.make_movie(imgname, movname, indexsz='06', framerate=framerate)


def build_outdir_kchern_varyparam(param, kcgcoll, param_type, rootdir):
    """Create output directory for sequence of k-space chern numbers varying a hlat param or lattice param

    Parameters
    ----------
    param : str
        the parameter being varied
    kcgcoll : KChernGyroCollection instance
        the chern collection
    param_type : str
        the string specifier of whether 'latparam' or 'hlatparam' is being varied
    rootdir : str
        the base directory in which 'kspace_cherns_haldane/chern_hlatparam/' and 'kspace_cherns_haldane/chern_latparam/' live

    Returns
    -------
    outdir : str
    outbase : str
    outd_ex : str
    """
    # Naming for directory in which to save
    if param_type == 'hlat':
        outdir = rootdir + 'kspace_cherns_haldane/chern_hlatparam/' + param + '/'
        dio.ensure_dir(outdir)

        # Add meshfn name to the output filename
        outbase = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn']
        if outbase[-1] == '/':
            outbase = outbase[:-1]
        outbase = outbase.split('/')[-1]

        outd_ex = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn_exten']
        # If the parameter name is part of the meshfn_exten, replace its value with XXX in
        # the meshfnexten part of outdir.
        mfestr = hlatfns.param2meshfnexten_name(param)
        if mfestr in outd_ex:
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outd_ex.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            print 'odrest = ', odrest
            print 'od2val_rest = ', od2val_rest
            outd_ex = od_split[0] + param + 'XXX'
            outd_ex += odrest
            print 'outd_ex = ', outd_ex
        else:
            outd_ex += '_' + param + 'XXX'
    elif param_type == 'lat':
        outdir = rootdir + 'kspace_cherns_haldane/chern_lpparam/' + param + '/'
        outbase = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn']
        if outbase[-1] == '/':
            outbase = outbase[:-1]
        outbase = outbase.split('/')[-1]

        # Take apart outbase to parse out the parameter that is varying
        mfestr = latfns.param2meshfnexten_name(param)
        if mfestr in outbase:
            'param is in meshfn_exten, splitting...'
            # split the outdir by the param string
            od_split = outbase.split(mfestr)
            # split the second part by the value of the param string and the rest
            od2val_rest = od_split[1].split('_')
            odrest = od_split[1].split(od2val_rest[0])[1]
            outbase = od_split[0] + param + 'XXX'
            outbase += odrest
            print 'outbase = ', outbase
        else:
            outbase += '_' + param + 'XXX'

        outd_ex = kcgcoll.cherns[kcgcoll.cherns.items()[0][0]][0].haldane_lattice.lp['meshfn_exten']
        outd_ex += '_density{0:06d}'.format(kcgcoll.cp['density'])

    return outdir, outbase, outd_ex


def mask_ax_to_bz(ax, vertex_points):
    """Mask the space outside the BZ to white"""
    # get corners of the plot
    xv = ax.get_xlim()
    yv = ax.get_ylim()
    corners = np.array([[xv[0], yv[0]], [xv[0], yv[1]], [xv[1], yv[1]], [xv[1], yv[0]]])
    vertex_points_augmented = np.vstack((vertex_points, vertex_points[-1]))
    # Define code for matplotlib to draw the path
    codes = np.ones(len(vertex_points) + 1, dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    all_codes = np.concatenate((codes[0:4], codes))
    vertices = np.concatenate((corners, vertex_points_augmented))
    path = mpath.Path(vertices, all_codes)
    # Add plot it
    patch = mpatches.PathPatch(path, facecolor='#FFFFFF', edgecolor='none')
    patch_handle = ax.add_patch(patch)
    return patch_handle