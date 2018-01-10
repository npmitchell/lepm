import numpy as np
import lepm.lattice_elasticity as le
import lepm.kitaev.kitaev_functions as kfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.kitaev_plotting_functions as kpfns
import lepm.stringformat as sf
import lepm.plotting.network_visualization as netvis
import lepm.haldane.haldane_chern_functions as hcfns
from PIL import Image
from os import getcwd,chdir
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors
import argparse
import plotting as leplt
import colormaps as cmaps
import subprocess
import socket
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl


"""Functions for plotting results of collecting many cherns
"""


def initialize_1p5panelcbar_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hs=None,
                                wssfrac=0.25, hssfrac=None, vspace=8, tspace=10,
                                fontsize=8, center0_frac=0.35, center2_frac=0.65):
    """Plot a chern figure with a panel below the plot (as in for showing kitaev regions)

    Parameters
    ----------
    Wfig : width of the figure in mm
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig
    ax
    """
    # Make figure
    x0 = round(Wfig * x0frac)
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    wss = Wfig * wssfrac
    if hssfrac is None:
        hssfrac = wssfrac
    hss = Wfig * hssfrac
    if hs is None:
        hs = ws
    wscbar = ws * 0.3
    hscbar = wscbar * 0.1
    if Hfig is None:
        Hfig = y0 + hs + vspace + hscbar + tspace
    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
            [Wfig * center0_frac - ws * 0.5, y0, ws, hs, ''],  # Chern vary hlatparam vs ksize
            [Wfig * center0_frac - wscbar, y0 + hs + vspace, wscbar * 2, hscbar, ''],  # cbar for chern (DOS)
            [Wfig * center2_frac - ws * 0.5, y0, wss, hss, '']  # subplot for showing kitaev region
          )]
    return fig, ax


def movie_cherns_varyloc(ccoll, title='Chern number calculation for varied positions',
                         filename='chern_varyloc', rootdir=None, exten='.png', max_boxfrac=None, max_boxsize=None,
                         xlabel=None, ylabel=None, step=0.5, fracsteps=False, framerate=3):
    """Plot the chern as a function of space for each haldane_lattice examined

    Parameters
    ----------
    ccoll : ChernCollection instance
        The collection of varyloc chern calcs to make into a movie
    title : str
        title of the movie
    filename : str
        the name of the files to save
    rootdir : str or None
        The cproot directory to use (usually self.cp['rootdir'])
    exten : str (.png, .jpg, etc)
        file type extension
    max_boxfrac : float
        Fraction of spatial extent of the sample to use as maximum bound for kitaev sum
    max_boxsize : float or None
        If None, uses max_boxfrac * spatial extent of the sample asmax_boxsize
    xlabel : str
        label for x axis
    ylabel : str
        label for y axis
    step : float (default=1.0)
        how far apart to sample kregion vertices in varyloc
    fracsteps : bool
    framerate : int
        The framerate at which to save the movie
    max_boxfrac : float
        Fraction of spatial extent of the sample to use as maximum bound for kitaev sum
    max_boxsize : float or None
        If None, uses max_boxfrac * spatial extent of the sample asmax_boxsize
    """
    rad = 1.0
    divgmap = cmaps.diverging_cmap(250, 10, l=30)

    # plot it
    for hlat_name in ccoll.cherns:
        hlat = ccoll.cherns[hlat_name][0].haldane_lattice
        if hlat.lp['shape'] == 'square':
            # get extent of the network from Bounding box
            Radius = np.abs(hlat.lp['BBox'][0, 0])
        else:
            # todo: allow different geometries
            pass

        # Initialize the figure
        h_mm = 90
        w_mm = 120
        # To get space between subplots, figure out how far away ksize region needs to be, based on first chern
        # Compare max ksize to be used with spatial extent of the lattice. If comparable, make hspace large.
        # Otherwise, use defaults
        ksize = ccoll.cherns[hlat_name][0].chern_finsize[:, 2]
        cgll = ccoll.cherns[hlat_name][0].haldane_lattice.lattice
        maxsz = max(np.max(cgll.xy[:, 0]) - np.min(cgll.xy[:, 0]),
                    np.max(cgll.xy[:, 1]) - np.min(cgll.xy[:, 1]))
        if max_boxsize is not None:
            ksize = ksize[ksize < max_boxsize]
        else:
            if max_boxfrac is not None:
                max_boxsize = max_boxfrac * maxsz
                ksize = ksize[ksize < max_boxsize]
            else:
                ksize = ksize
                max_boxsize = np.max(ksize)
        if max_boxsize > 0.9 * maxsz:
            center0_frac = 0.3
            center2_frac = 0.75
        elif max_boxsize > 0.65 * maxsz:
            center0_frac = 0.35
            center2_frac = 0.72
        elif max_boxsize > 0.55 * maxsz:
            center0_frac = 0.375
            center2_frac = 0.71
        else:
            center0_frac = 0.4
            center2_frac = 0.7

        fig, ax = initialize_1p5panelcbar_fig(Wfig=w_mm, Hfig=h_mm, wsfrac=0.4, wssfrac=0.4,
                                              center0_frac=center0_frac, center2_frac=center2_frac)

        # dimensions of video in pixels
        final_h = 720
        final_w = 960
        actual_dpi = final_h / (float(h_mm) / 25.4)

        # Add the network to the figure
        hlat = ccoll.cherns[hlat_name][0].haldane_lattice
        netvis.movie_plot_2D(hlat.lattice.xy, hlat.lattice.BL, 0 * hlat.lattice.BL[:, 0],
                             None, None, ax=ax[0], fig=fig, axcb=None,
                             xlimv='auto', ylimv='auto', climv=0.1, colorz=False, ptcolor=None, figsize='auto',
                             colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=True, axis_equal=True,
                             lw=0.2)

        # Add title
        if title is not None:
            ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                           horizontalalignment='center', verticalalignment='center')
        if xlabel is not None:
            ax[0].set_xlabel(xlabel)
        if ylabel is not None:
            ax[0].set_xlabel(ylabel)

        # Position colorbar
        sm = plt.cm.ScalarMappable(cmap=divgmap, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable.
        sm._A = []
        cbar = plt.colorbar(sm, cax=ax[1], orientation='horizontal', ticks=[-1, 0, 1])
        ax[1].set_xlabel(r'$\nu$')
        ax[1].xaxis.set_label_position("top")
        ax[2].axis('off')

        # Add patches (rectangles from cherns at each site) to the figure
        print 'Opening hlat_name = ', hlat_name
        done = False
        ind = 0
        while done is False:
            rectps = []
            colorL = []
            for chernii in ccoll.cherns[hlat_name]:
                # Grab small, medium, and large circles
                ksize = chernii.chern_finsize[:, 2]
                if max_boxsize is not None:
                    ksize = ksize[ksize < max_boxsize]
                else:
                    if max_boxfrac is not None:
                        cgll = chernii.haldane_lattice.lattice
                        maxsz = max(np.max(cgll.xy[:, 0]) - np.min(cgll.xy[:, 0]),
                                    np.max(cgll.xy[:, 1]) - np.min(cgll.xy[:, 1]))
                        max_boxsize = max_boxfrac * maxsz
                        ksize = ksize[ksize < max_boxsize]
                    else:
                        ksize = ksize
                        max_boxsize = np.max(ksize)

                # print 'ksize =  ', ksize
                # print 'max_boxsize =  ', max_boxsize

                xx = float(chernii.cp['poly_offset'].split('/')[0])
                yy = float(chernii.cp['poly_offset'].split('/')[1])
                nu = chernii.chern_finsize[:, -1]
                rad = step
                rect = plt.Rectangle((xx-rad*0.5, yy-rad*0.5), rad, rad, ec="none")
                colorL.append(nu[ind])
                rectps.append(rect)

            p = PatchCollection(rectps, cmap=divgmap, alpha=1.0, edgecolors='none')
            p.set_array(np.array(np.array(colorL)))
            p.set_clim([-1., 1.])

            # Add the patches of nu calculations for each site probed
            ax[0].add_collection(p)

            # Draw the kitaev cartoon in second axis with size ksize[ind]
            polygon1, polygon2, polygon3 = kfns.get_kitaev_polygons(ccoll.cp['shape'], ccoll.cp['regalph'],
                                                                    ccoll.cp['regbeta'], ccoll.cp['reggamma'],
                                                                    ksize[ind])
            patchlist = []
            patchlist.append(patches.Polygon(polygon1, color='r'))
            patchlist.append(patches.Polygon(polygon2, color='g'))
            patchlist.append(patches.Polygon(polygon3, color='b'))
            polypatches = PatchCollection(patchlist, cmap=cm.jet, alpha=0.4, zorder=99, linewidths=0.4)
            colors = np.linspace(0, 1, 3)[::-1]
            polypatches.set_array(np.array(colors))
            ax[2].add_collection(polypatches)
            ax[2].set_xlim(ax[0].get_xlim())
            ax[2].set_ylim(ax[0].get_ylim())

            # Save the plot
            # make index string
            indstr = '_{0:06d}'.format(ind)
            hlat_cmesh = kfns.get_cmeshfn(ccoll.cherns[hlat_name][0].haldane_lattice.lp, rootdir=rootdir)
            specstr = '_Nks' + '{0:03d}'.format(len(ksize)) + '_step' + sf.float2pstr(step) \
                      + '_maxbsz' + sf.float2pstr(max_boxsize)
            outdir = hlat_cmesh + '_' + hlat.lp['LatticeTop'] + '_varyloc_stills' + specstr + '/'
            fnout = outdir + filename + specstr + indstr + exten
            print 'saving figure: ' + fnout
            le.ensure_dir(outdir)
            fig.savefig(fnout, dpi=actual_dpi*2)

            # Save at lower res after antialiasing
            f_img = Image.open(fnout)
            f_img.resize((final_w, final_h), Image.ANTIALIAS).save(fnout)

            # clear patches
            p.remove()
            polypatches.remove()
            # del p

            # Update index
            ind += 1
            if ind == len(ksize):
                done = True

        # Turn into movie
        imgname = outdir + filename + specstr
        movname = hlat_cmesh + filename + specstr + '_mov'
        subprocess.call(['./ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '_%06d' + exten, movname + '.mov',
                         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0',
                         '-r', '30', '-pix_fmt', 'yuv420p'])


def plot_cherns_varyloc(ccoll, title='Chern number calculation for varied positions',
                        filename='chern_varyloc', exten='.pdf', rootdir=None, outdir=None,
                        max_boxfrac=None, max_boxsize=None,
                        xlabel=None, ylabel=None, step=0.5, fracsteps=False,
                        singleksz_frac=None, singleksz=-1.0, maxchern=False,
                        ax=None, cbar_ax=None, save=True, make_cbar=True, colorz=False,
                        dpi=600, colormap='divgmap_blue_red'):
    """Plot the chern as a function of space for each haldane_lattice examined. If save==True, saves figure to


    Parameters
    ----------
    ccoll : ChernCollection instance
    filename : str
        name of the file to output as the plot
    rootdir : str or None
        if specified, dictates the path where the file is stored, with the output directory as
        outdir = kfns.get_cmeshfn(ccoll.cherns[hlat_name][0].haldane_lattice.lp, rootdir=rootdir)
    max_boxfrac : float
        Fraction of spatial extent of the sample to use as maximum bound for kitaev sum
    max_boxsize : float or None
        If None, uses max_boxfrac * spatial extent of the sample asmax_boxsize
    singleksz : float
        if positive, plots the spatially-resolved chern number for a single kitaev summation region size, closest to
        the supplied value in actual size. Otherwise, draws many rectangles of different sizes for different ksizes.
        If positive, IGNORES max_boxfrac and max_boxsize arguments
    maxchern : bool
        if True, plots the EXTREMAL spatially-resolved chern number for each voxel --> ie the maximum (signed absolute)
        value it reaches. Otherwise, draws many rectangles of different sizes for different ksizes.
        If True, the function IGNORES max_boxfrac and max_boxsize arguments
    ax : axis instance or None
    cbar_ax : axis instance or None
    save : bool
    make_cbar : bool
    dpi : int
        dots per inch, if exten is '.png'
    """
    if colormap == 'divgmap_blue_red':
        divgmap = cmaps.diverging_cmap(250, 10, l=30)
    elif colormap == 'divgmap_red_blue':
        divgmap = cmaps.diverging_cmap(10, 250, l=30)

    # plot it
    for hlat_name in ccoll.cherns:
        rectps = []
        colorL = []
        print 'all hlats should have same pointer:'
        print 'ccoll[hlat_name][0].haldane_lattice = ', ccoll.cherns[hlat_name][0].haldane_lattice
        print 'ccoll[hlat_name][1].haldane_lattice = ', ccoll.cherns[hlat_name][1].haldane_lattice

        print 'Opening hlat_name = ', hlat_name

        if outdir is None:
            outdir = hcfns.get_cmeshfn(ccoll.cherns[hlat_name][0].haldane_lattice.lp, rootdir=rootdir)

        print 'when saving, will save to ' + outdir + 'filename'

        if maxchern:
            for chernii in ccoll.cherns[hlat_name]:
                # Grab small, medium, and large circles
                ksize = chernii.chern_finsize[:, 2]

                # Build XYloc_sz_nu from all cherns done on this network
                xx = float(chernii.cp['poly_offset'].split('/')[0])
                yy = float(chernii.cp['poly_offset'].split('/')[1])
                nu = chernii.chern_finsize[:, -1]
                ind = np.argmax(np.abs(nu))
                rad = step
                rect = plt.Rectangle((xx-rad*0.5, yy-rad*0.5), rad, rad, ec="none")
                colorL.append(nu[ind])
                rectps.append(rect)
        elif singleksz > 0:
            for chernii in ccoll.cherns[hlat_name]:
                # Grab small, medium, and large circles
                ksize = chernii.chern_finsize[:, 2]

                # Build XYloc_sz_nu from all cherns done on this network
                xx = float(chernii.cp['poly_offset'].split('/')[0])
                yy = float(chernii.cp['poly_offset'].split('/')[1])
                nu = chernii.chern_finsize[:, -1]
                ind = np.argmin(np.abs(ksize - singleksz))
                # print 'ksize = ', ksize
                # print 'singleksz = ', singleksz
                # print 'ind = ', ind
                rad = step
                rect = plt.Rectangle((xx-rad*0.5, yy-rad*0.5), rad, rad, ec="none")
                colorL.append(nu[ind])
                rectps.append(rect)
        elif singleksz_frac > 0:
            for chernii in ccoll.cherns[hlat_name]:
                # Grab small, medium, and large circles
                ksize_frac = chernii.chern_finsize[:, 1]

                # Build XYloc_sz_nu from all cherns done on this network
                xx = float(chernii.cp['poly_offset'].split('/')[0])
                yy = float(chernii.cp['poly_offset'].split('/')[1])
                nu = chernii.chern_finsize[:, -1]
                ind = np.argmin(np.abs(ksize_frac - singleksz_frac))
                # print 'ksize = ', ksize
                # print 'singleksz = ', singleksz
                # print 'ind = ', ind
                rad = step
                rect = plt.Rectangle((xx - rad * 0.5, yy - rad * 0.5), rad, rad, ec="none")
                colorL.append(nu[ind])
                rectps.append(rect)
        else:
            print 'stacking rectangles in list to add to plot...'
            for chernii in ccoll.cherns[hlat_name]:
                # Grab small, medium, and large circles
                # Note: I used to multiply by 0.5 here: Why did I multiply by 0.5 here? Not sure...
                # perhaps before I measured ksize by its diameter (or width) but used radius as an imput for drawing
                # a kitaev region.
                ksize = chernii.chern_finsize[:, 2]
                if max_boxsize is not None:
                    ksize = ksize[ksize < max_boxsize]
                else:
                    if max_boxfrac is not None:
                        cgll = chernii.haldane_lattice.lattice
                        maxsz = max(np.max(cgll.xy[:, 0]) - np.min(cgll.xy[:, 0]),
                                    np.max(cgll.xy[:, 1]) - np.min(cgll.xy[:, 1]))
                        max_boxsize = max_boxfrac * maxsz
                        ksize = ksize[ksize < max_boxsize]
                    else:
                        ksize = ksize
                        max_boxsize = np.max(ksize)

                # print 'ksize =  ', ksize
                # print 'max_boxsize =  ', max_boxsize

                # Build XYloc_sz_nu from all cherns done on this network
                xx = float(chernii.cp['poly_offset'].split('/')[0])
                yy = float(chernii.cp['poly_offset'].split('/')[1])
                nu = chernii.chern_finsize[:, -1]
                rectsizes = ksize / np.max(ksize) * step
                # Choose which rectangles to draw
                if len(ksize) > 30:
                    # Too many rectangles to add to the plot! Limit the number to keep file size down
                    inds2use = np.arange(0, len(ksize), int(float(len(ksize))*0.05))[::-1]
                else:
                    inds2use = np.arange(0, len(ksize), 1)[::-1]
                # print 'Adding ' + str(len(inds2use)) + ' rectangles...'

                # Make a list of the rectangles
                for ind in inds2use:
                    rad = rectsizes[ind]
                    rect = plt.Rectangle((xx-rad*0.5, yy-rad*0.5), rad, rad, ec="none")
                    colorL.append(nu[ind])
                    rectps.append(rect)

        print 'Adding patches to figure...'
        p = PatchCollection(rectps, cmap=divgmap, alpha=1.0, edgecolors='none')
        p.set_array(np.array(np.array(colorL)))
        p.set_clim([-1., 1.])

        if ax is None:
            # Make figure
            FSFS = 8
            Wfig = 90
            x0 = round(Wfig * 0.15)
            y0 = round(Wfig * 0.1)
            ws = round(Wfig * 0.4)
            hs = ws
            wsDOS = ws * 0.3
            hsDOS = hs
            wscbar = wsDOS
            hscbar = wscbar * 0.1
            vspace = 8  # vertical space btwn subplots
            hspace = 8  # horizonl space btwn subplots
            tspace = 10  # space above top figure
            Hfig = y0 + hs + vspace + hscbar + tspace
            fig = sps.figure_in_mm(Wfig, Hfig)
            label_params = dict(size=FSFS, fontweight='normal')
            ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
                  for x0, y0, width, height, part in (
                      [Wfig * 0.5 - ws * 0.5, y0, ws, hs, ''],  # Chern vary hlatparam vs ksize
                      [Wfig * 0.5 - wscbar, y0 + hs + vspace, wscbar * 2, hscbar, '']  # cbar for chern
                  )]

            # Add the patches of nu calculations for each site probed
            ax[0].add_collection(p)
            hlat = ccoll.cherns[hlat_name][0].haldane_lattice
            netvis.movie_plot_2D(hlat.lattice.xy, hlat.lattice.BL, 0*hlat.lattice.BL[:, 0],
                             None, None, ax=ax[0], fig=fig, axcb=None,
                             xlimv='auto', ylimv='auto', climv=0.1, colorz=colorz, ptcolor=None, figsize='auto',
                             colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=True, axis_equal=True,
                             lw=0.2)

            # Add title
            ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                           horizontalalignment='center', verticalalignment='center')
            if xlabel is not None:
                ax[0].set_xlabel(xlabel)
            if ylabel is not None:
                ax[0].set_xlabel(ylabel)

            # Position colorbar
            sm = plt.cm.ScalarMappable(cmap=divgmap, norm=plt.Normalize(vmin=-1, vmax=1))
            # fake up the array of the scalar mappable.
            sm._A = []
            cbar = plt.colorbar(sm, cax=ax[1], orientation='horizontal', ticks=[-1, 0, 1])
            ax[1].set_xlabel(r'$\nu$')
            ax[1].xaxis.set_label_position("top")
        else:
            # Add the patches of nu calculations for each site probed
            ax.add_collection(p)
            hlat = ccoll.cherns[hlat_name][0].haldane_lattice
            netvis.movie_plot_2D(hlat.lattice.xy, hlat.lattice.BL, 0*hlat.lattice.BL[:, 0],
                                 None, None, ax=ax, fig=plt.gcf(), axcb=None,
                                 xlimv='auto', ylimv='auto', climv=0.1, colorz=colorz, ptcolor=None, figsize='auto',
                                 colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=True, axis_equal=True,
                                 lw=0.2)

            # Add title
            if title is not None and title != 'none':
                ax.annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                            horizontalalignment='center', verticalalignment='center')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_xlabel(ylabel)

            if make_cbar:
                # Position colorbar
                sm = plt.cm.ScalarMappable(cmap=divgmap, norm=plt.Normalize(vmin=-1, vmax=1))
                # fake up the array of the scalar mappable.
                sm._A = []
                if cbar_ax is not None:
                    # figure out if cbar_ax is horizontal or vertical
                    if leplt.cbar_ax_is_vertical(cbar_ax):
                        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', ticks=[-1, 0, 1],
                                            label='Chern\n' + r'number, $\nu$')
                    else:
                        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=[-1, 0, 1],
                                            label=r'Chern number, $\nu$')
                        cbar_ax.set_xlabel(r'Chern number, $\nu$')
                        cbar_ax.xaxis.set_label_position("top")
                else:
                    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=[-1, 0, 1],
                                        label=r'Chern number, $\nu$')

        if save:
            # Save the plot
            # make outdir the hlat_cmesh
            print 'kcollpfns: saving figure:\n outdir =', outdir, ' filename = ', filename
            # outdir = kfns.get_cmeshfn(ccoll.cherns[hlat_name][0].haldane_lattice.lp, rootdir=rootdir)
            if filename == 'chern_varyloc':
                filename += '_Nks'+'{0:03d}'.format(len(ksize)) + '_step' + sf.float2pstr(step) + '_maxbsz' +\
                            sf.float2pstr(max_boxsize) + exten
            print 'saving figure: ' + outdir + filename
            if exten == '.png':
                plt.savefig(outdir + filename, dpi=dpi)
            else:
                plt.savefig(outdir + filename)
            plt.clf()

    return ax, cbar


def plot_chern_spectrum_on_axis(kcoll, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, fontsize=8, alpha=1.0,
                                ipr_vmin=None, ipr_vmax=None, cbar_labelpad=3, invert_xaxis=True,
                                ipr_cax_label='participation\nratio,' + r' $p$',
                                chern_cax_label='Chern\nnumber, ' + r'$\nu$'):
    """
    Parameters
    ----------
    kcoll : HaldaneCollection instance
        The chern collection to add to the plot
    ax : matplotlib axis instance
        the axis on which to add the spectrum
    dos_ax : matplotlib axis instance
        the axis on which to add
    alpha : float
        opacity of the spectrum and DOS added to the plot

    Returns
    -------
    ax, dos_ax
    """
    dmyi = 0
    for hlat_name in kcoll.cherns:
        # Grab a pointer to the haldane_lattice
        hlat = kcoll.cherns[hlat_name][0].haldane_lattice
        dmyi += 1
    if dmyi > 1:
        raise RuntimeError('This function takes a HaldaneCollection which is allowed only ONE haldane_lattice-- ' +
                           'ie, a chern spectrum for a single haldaneLattice instance')

    print 'Opening hlat_name = ', hlat_name
    ngrid = len(kcoll.cherns[hlat_name][0].chern_finsize)

    # Assume all ksize elements are the same size for now
    ksys_frac = kcoll.cherns[hlat_name][0].chern_finsize[:, -2]*0.5
    ksys_fracM = np.array([ksys_frac for i in range(len(kcoll.cherns[hlat_name]))])

    # Build omegacV
    omegacV = np.zeros(len(kcoll.cherns[hlat_name]))
    nuM = np.zeros((len(kcoll.cherns[hlat_name]), len(ksys_frac)))

    for ind in range(len(kcoll.cherns[hlat_name])):
        print 'ind = ', ind
        cp_ii = kcoll.cherns[hlat_name][ind].cp
        omegacV[ind] = cp_ii['omegac']
        nuM[ind, :] = kcoll.cherns[hlat_name][ind].chern_finsize[:, -1]

    omegacM = omegacV.reshape(len(nuM), 1) * np.ones_like(nuM)
    print 'omegacM = ', omegacM

    print 'kpfns: adding ipr and chern spectrum to axes...'
    print 'Adding chern spectrum to axis: ax = ', ax
    leplt.plot_pcolormesh(ksys_fracM, omegacM, nuM, ngrid, ax=ax, cax=cbar_nu_ax, method='nearest',
                          cmap=cmaps.diverging_cmap(250, 10, l=30),
                          vmin=-1.0, vmax=1.0, title=None, xlabel=None, ylabel=None,  cax_label=chern_cax_label,
                          cbar_labelpad=cbar_labelpad, cbar_orientation='horizontal', ticks=[-1, 0, 1],
                          fontsize=fontsize, alpha=alpha)

    if dos_ax is not None and dos_ax != 'none':
        print 'Adding ipr to DOS axis: dos_ax = ', dos_ax
        hlat.add_ipr_to_ax(dos_ax, ipr=None, alpha=alpha, inverse_PR=False,
                           norm=None, nbins=75, fontsize=fontsize, cbar_ax=cbar_ipr_ax,
                           vmin=ipr_vmin, vmax=ipr_vmax, linewidth=0, cax_label=ipr_cax_label,
                           make_cbar=True, climbars=True,
                           cbar_labelpad=cbar_labelpad, orientation='horizontal', cbar_orientation='horizontal',
                           invert_xaxis=invert_xaxis, ylabel=None, xlabel=None,
                           cbar_nticks=2, cbar_tickfmt='%0.1f')
        ax.set_ylim(dos_ax.get_ylim())

    return ax, dos_ax, cbar_ipr_ax, cbar_nu_ax


def plot_nugrid_chern(chern, hlat, xyvec, nugrids, repi, outdir=None, name_exten='', ax=None, fontsize=8):
    """Plot a pcolormesh of the spatially-resolved chern calcs on ax and plot the lattice on top of this.

    Parameters
    ----------
    chern : HaldaneChern instance
    hlat : haldaneLattice instance
    xyvec :
    nugrids :
    nugsum :
    repi :
    outdir : str or None
    name_exten

    """
    leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugrids[repi].T.ravel(), 100, ax=ax,
                          method='nearest', cmap='rwb0',
                          vmin=-1.0, vmax=1.0, xlabel='x', ylabel='y',  cax_label=r'$\nu$', fontsize=fontsize)
    netvis.movie_plot_2D(hlat.lattice.xy, hlat.lattice.BL, 0 * hlat.lattice.BL[:, 0],
                         None, None, ax=plt.gca(), axcb=None, bondcolor='k',
                         colorz=False, ptcolor=None, figsize='auto',
                         colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
                         lw=0.2)
    kpfns.add_kitaev_regions_to_plot(chern=chern, ksize_ind=repi, offsetxy=np.array([15, 0]))
    plt.title(r'$\nu$')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    if outdir is not None:
        plt.savefig(outdir + '_nugrid' + name_exten + '.png')


def plot_gradnusum_chern(chern, hlat, xyvec, nugsum, repi, outdir=None, name_exten='', ax=None, fontsize=8):
    """Plot the gradient of the computed chern number -- plots the sum of abs(del_i nu) where the sum is taken over
    the two directions (x and y).

    Parameters
    ----------
    chern : HaldaneChern instance
    hlat : HaldaneLattice instance
    xyvec :
    nugrids :
    nugsum :
    repi :
    outdir : str or None
    name_exten

    """
    leplt.plot_pcolormesh(xyvec[:, 0], xyvec[:, 1], nugsum[repi].T.ravel(), 100, ax=ax,
                          method='nearest', cmap='viridis',
                          vmin=None, vmax=None, xlabel='x', ylabel='y',  cax_label=r'$\sum_j |\nabla_j \nu|$',
                          fontsize=fontsize)
    netvis.movie_plot_2D(hlat.lattice.xy, hlat.lattice.BL, 0 * hlat.lattice.BL[:, 0],
                         None, None, ax=plt.gca(), axcb=None, bondcolor='k',
                         colorz=False, ptcolor=None, figsize='auto',
                         colormap='BlueBlackRed', bgcolor='#ffffff', axis_off=False, axis_equal=True,
                         lw=0.2)
    kpfns.add_kitaev_regions_to_plot(chern=chern, ksize_ind=repi, offsetxy=np.array([15, 0]))
    plt.title(r'$|\sum_j \nabla_j \nu|$')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_nugradsumabs' + name_exten + '.png')


def plot_various_gnu_psi2(hlat, gnu_psi2, eigval, top, repi, repi2, sz=1, outdir=None):
    """Plot gnu_psi2 (representation of excitations at gradients in the computed chern number, when chern computation
    is spatially resolved).

    Parameters
    ----------
    repi : int
    repi2 : int
    """
    zetastr = r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'

    copper = plt.get_cmap('copper_r')
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(gnu_psi2))
    copperMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=copper)
    jet = plt.get_cmap('spectral')
    cNorm_ev = matplotlib.colors.Normalize(vmin=1.0, vmax=max(np.imag(eigval)))
    jetMap = matplotlib.cm.ScalarMappable(norm=cNorm_ev, cmap=jet)
    ##########################################
    plt.clf()
    ind = 0
    for ll in gnu_psi2:
        plt.plot(np.abs(np.imag(eigval[top])), ll, '-', color=copperMap.to_rgba(ind))
        ind += 1
    plt.title(r'Edge localization $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    plt.savefig(outdir + '_gradsumnuabs_vs_omega.png')

    # Correlate with ipr
    if hlat.ipr is None:
        ipr = hlat.get_ipr()

    plt.clf()
    ind = 0
    for ll in gnu_psi2:
        plt.scatter(ipr[top], ll, color=copperMap.to_rgba(ind))
        ind += 1
    plt.title(r'Edge localization vs $p^{-1}$, ' + zetastr)
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$p^{-1}$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnu_vs_ipr.png')

    plt.clf()
    ind = 0
    for ll in gnu_psi2:
        plt.scatter(1./ipr[top], ll, s=20, color=copperMap.to_rgba(ind))
        ind += 1
    plt.title(r'Edge localization vs $p$, ' + zetastr)
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$p$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnu_vs_p.png')

    plt.clf()
    for ind in range(len(eigval)/2):
        plt.scatter(1./ipr[ind], gnu_psi2[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
    # fake up the array of the scalar mappable.
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $p$, ' + zetastr)
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$p$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnu_vs_p_repi.png')

    plt.clf()
    for ind in range(len(eigval)/2):
        plt.scatter(ipr[ind], gnu_psi2[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
    # fake up the array of the scalar mappable.
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    title = r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
    plt.title(title)
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$p^{-1}$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_ipr_coloromega.png')
        plt.xlim(0, 30)
        plt.savefig(outdir + '_gradsumnuabs_vs_ipr_coloromega_zoom.png')

    plt.clf()
    for ind in range(len(eigval)/2):
        plt.scatter(1./ipr[ind], gnu_psi2[repi, ind], s=sz, color=jetMap.to_rgba(np.abs(eigval[ind])))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
    # fake up the array of the scalar mappable.
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    ttl = r'Edge localization vs $p$, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
    plt.title(ttl)
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$p$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_p_coloromega.png')

    ####################################
    # REPI
    ####################################
    plt.clf()
    for ind in range(len(eigval)/2):
        plt.scatter(abs(eigval[ind]), gnu_psi2[repi, ind], s=sz,
                    color=jetMap.to_rgba(np.abs(eigval[ind])))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
    # fake up the array of the scalar mappable.
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega.png')

    ####################################
    # REPI2
    ####################################
    plt.clf()
    for ind in range(len(eigval)/2):
        plt.scatter(abs(eigval[ind]), gnu_psi2[repi2, ind], s=sz,
                    color=jetMap.to_rgba(np.abs(eigval[ind])))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=jet, norm=plt.Normalize(vmin=1.0, vmax=max(np.imag(eigval))))
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_early.png')

    ######################################################
    # AVERAGING grads
    plt.clf()
    inds = range(len(eigval)/2)
    evrm3 = le.running_mean(eigval[inds], 3)
    for rii in range(0, len(gnu_psi2)-3, int(len(gnu_psi2)/20.0)):
        l5rm3 = le.running_mean(gnu_psi2[rii, inds], 3)
        plt.plot(abs(evrm3), l5rm3, '-', color=copperMap.to_rgba(float(rii/(len(gnu_psi2)-3))))
    sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm._A = []
    plt.colorbar(sm, label=r'kitaev region size')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega - 2.25 \Omega_g$')
    plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg3.png')

    plt.clf()
    ind = 0
    inds = range(len(eigval)/2)
    evrm5 = le.running_mean(eigval[inds], 5)
    for rii in range(0, len(gnu_psi2)-5, int(len(gnu_psi2)/20.0)):
        l5rm5 = le.running_mean(gnu_psi2[rii, inds], 5)
        plt.plot(abs(evrm5), l5rm5, '-', color=copperMap.to_rgba(float(rii/(len(gnu_psi2)-5))))
        ind += 1
    sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg5.png')

    plt.clf()
    inds = range(len(eigval)/2)
    evrm10 = le.running_mean(eigval[inds], 10)
    for rii in range(0, len(gnu_psi2)-10, int(len(gnu_psi2)/20.0)):
        print 'rii = ', rii
        l5rm10 = le.running_mean(gnu_psi2[rii, inds], 10)
        plt.plot(abs(evrm10), l5rm10, '-', color=copperMap.to_rgba(float(rii/(len(gnu_psi2)-10))))
    sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg10.png')

    plt.clf()
    inds = range(len(eigval)/2)
    evrm20 = le.running_mean(eigval[inds], 20)
    for rii in range(0, len(gnu_psi2)-20, int(len(gnu_psi2)/20.0)):
        print 'rii = ', rii
        l5rm20 = le.running_mean(gnu_psi2[rii, inds], 20)
        plt.plot(abs(evrm20), l5rm20, '-', color=copperMap.to_rgba(float(rii/(len(gnu_psi2)-20))))
    sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm._A = []
    plt.colorbar(sm, label=r'$\omega$')
    plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
              r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    plt.ylabel(r'$\zeta$')
    plt.xlabel(r'$\omega$')
    if outdir is None:
        plt.show()
    else:
        plt.savefig(outdir + '_gradsumnuabs_vs_omegadiff_coloromega_avg20.png')

    # ######################################################
    # # MEDIAN grads
    # plt.clf()
    # step = 0.1
    # bins = np.arange(1.0, 4.0, step)
    # digits = np.digitize(np.abs(eigval[inds]), bins, right=False)
    # ind = 0
    # for ind in range(len(bins)-1):
    #     inds[ind] = np.where(digits == ind)[0]
    #
    # for rii in range(0, len(gnu_psi2), int(len(gnu_psi2)/20.0)):
    #     medloc5[ind] = np.median(gnu_psi2[rii, inds])
    #     plt.plot(bins[0:len(bins)-1], gnu_psi2[rii], '-', color=copperMap.to_rgba(rii/(len(local5))))
    # sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=5.0, vmax=1.0))
    # sm._A = []
    # plt.colorbar(sm, label=r'$kitaev region size (arb units)$')
    # plt.title(r'Edge localization vs $\omega-\omega_c$, ' +
    #           r'$ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$')
    # plt.ylabel(r'$\zeta$')
    # plt.xlabel(r'$\omega - 2.25 \Omega_g$')
    # plt.savefig(outdir + '_gradsumnuabs_vs_omegad_coloromega_median0p1.png')


def plot_kcoll_gnusum_chern(ev, gnupsi2, cmap='copper_r', ax=None, cbar_ax=None, make_cbar=True,
                            outdir=None, fn_exten='', fexten='.png', max_delta=None, ncurves=20, show=False, alpha=1.0,
                            title='auto', cbar_labelpad=None, cbar_ticks=None):
    """Plot sum(grad(nu))*psi^2 versus eigval for a composite collection of lattice realizations

    Parameters
    ----------
    ev : N x 1 float array
        sorted eigenvalues of all haldane_lattices in the chern collection
    gnupsi2 : Nks x N float array
        Each row corresponds to a different kitaev region size, each column corresponds to an element of the eigenvalue
        array (an energy). This is the sorted, collated results of measuring the gradients of the spatially resolved
        chern number, weighted by the amplitudes of the excitations in each eigenstate
    cmap : colormap specifier
        colormap to use to differentiate summation region sizes
    ax : axis instance or None
        axis on which to plot gradient in chern vs eigvalues
    cbar_ax : axis instance or None
        axis on which to plot the colorbar. Automatically detects if orientation is horizontal or vertical based on
        aspect ratio
    outdir : str or None
        Path in which to save plot, if not None
    fn_exten : str
        optional string extension to the filename to save the plot
    fexten : str
        file type extention (.png, .pdf, .jpg, etc)
    max_delta : float or None
        Maximum size of kitaev region as fraction of system size. If None, labels colorbar by "delta/max(delta)"
    ncurves : int
        Number of different kitaev summation region sizes to plot
    title : str
        Title of the axis
    """
    if ax is None:
        ax = plt.gca()

    copper = plt.get_cmap(cmap)
    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=len(gnupsi2) - 10)
    copperMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=copper)

    print 'size(ev_v) = ', np.shape(ev)
    print 'size(gnupsi2_v) = ', np.shape(gnupsi2)

    for rii in range(0, len(gnupsi2) - 10, int(len(gnupsi2)/ncurves)):
        ax.plot(ev, gnupsi2[rii], '-', color=copperMap.to_rgba(float(rii)), alpha=alpha)

    # Create colorbar
    if make_cbar:
        if max_delta is None:
            vmax = 1.0
        else:
            vmax = max_delta

        sm = plt.cm.ScalarMappable(cmap=copper, norm=plt.Normalize(vmin=0.0, vmax=max_delta))
        sm._A = []
        # Discern if colorbar axis is horizontal or vertical
        if cbar_ax is not None:
            pos = cbar_ax.get_position().get_points()
            if pos[0][1] - pos[0][0] > (pos[1][1] - pos[1][0]):
                orientation = 'horizontal'
            else:
                orientation = 'vertical'
        else:
            orientation = 'vertical'
        if max_delta is None:
            cbar = plt.colorbar(sm, cax=cbar_ax, label=r'Summation region size, $\delta/\delta_{\mathrm{max}}$',
                                orientation=orientation, ticks=cbar_ticks)
        else:
            cbar = plt.colorbar(sm, cax=cbar_ax, label=r'Summation region size, $\delta/L$',
                                orientation=orientation, ticks=cbar_ticks)
    else:
        cbar = None

    if title is not None and title != 'none':
        if title is 'auto':
            title = r'Edge localization, $ \zeta \equiv \frac{1}{N} \sum_i |\psi_i|^2 \sum_j |\partial_j \nu|_i$'
        ax.set_title(title)

    ax.set_ylabel(r'Gradient-weighted excitation, $\zeta$')
    ax.set_xlabel(r'Eigenmode frequency, $\omega/\Omega_g$')
    if outdir is None:
        if show:
            plt.show()
    else:
        plt.savefig(outdir + '_kcollgnusum_vs_omega' + fn_exten + fexten)
        plt.clf()

    return ax, cbar_ax, cbar
