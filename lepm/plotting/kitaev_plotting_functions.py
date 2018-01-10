import numpy as np
import lepm.lattice_elasticity as le
import lepm.kitaev.kitaev_functions as kfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmaps
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
import argparse
import plotting as leplt
import colormaps as cmaps
import subprocess
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

'''
Plot Chern number measurements made via the kitaev realspace method.
'''


def initialize_1panelcbar_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hs=None, vspace=8, hspace=8,
                              tspace=10, fontsize=8):
    """Create new figure with 1 panel and a colorbar, with defaults set for kitaev plots

    Parameters
    ----------
    Wfig : width of the figure in mm
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
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
              [Wfig * 0.5 - ws * 0.5, y0, ws, hs, ''],  # Chern vary glatparam vs ksize
              [Wfig * 0.5 - wscbar, y0 + hs + vspace, wscbar * 2, hscbar, '']  # cbar for chern
          )]
    return fig, ax


def initialize_spectrum_with_dos_plot(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hs=None,
                                      wsdosfrac=0.3, vspace=8, hspace=8, tspace=10, fontsize=8):
    """Initialize a figure with 4 axes, for plotting chern spectrum with accompanying DOS plot on the side

    Returns
    -------
    fig :
    ax : list of 4 matplotlib axis instances
        ax[0] is DOS axis, ax[1] is chern spectrum axis, ax[2] is ipr cbar axis, ax[3] is cbar for chern axis
    """
    x0 = round(Wfig * x0frac)
    y0 = round(Wfig * y0frac)
    ws = round(Wfig * wsfrac)
    hs = ws
    wsDOS = wsdosfrac * ws
    hsDOS = hs
    wscbar = wsDOS
    hscbar = wscbar * 0.1
    if Hfig is None:
        Hfig = y0 + hs + vspace + hscbar + tspace

    fig = sps.figure_in_mm(Wfig, Hfig)
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
            [x0, y0, wsDOS, hsDOS, ''],  # DOS
            [x0 + wsDOS + hspace, y0, ws, hs, ''],    # Chern omegac vs ksize
            [x0 + wsDOS * 0.5 - wscbar * 0.5, y0 + hs + vspace, wscbar, hscbar, ''],  # cbar for ipr
            [x0 + wsDOS + hspace + ws * 0.5-wscbar, y0 + hs + vspace, wscbar*2, hscbar, '']  # cbar for chern
          )]
    return fig, ax


def plot_spectrum_with_dos(glat, ksys_fracM, omegacM, nuM, ngrid, Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1,
                           wsfrac=0.4, hs=None, wsdosfrac=0.3, vspace=8, hspace=8, tspace=10, FSFS=8,
                           xlabel=None, ylabel=None, title=None):
    """Plot chern spectrum vs ksize (or other variable on x axis), with corresponding DOS plot aligned with y axis
    (increasing in frequency).

    Parameters
    ----------
    glat : gyro_lattice
    ksys_fracM : len(kitaev_collection.cherns[glat_name]) x len(ksys_frac or paramV) float array
    omegacM : len(kitaev_collection.cherns[glat_name]) x len(ksys_frac or paramV) float array
    nuM : len(kitaev_collection.cherns[glat_name]) x len(ksys_frac or paramV) float array
        The chern number results in array form
    Wfig : int or float
        width of the figure in mm
    Hfig : int or float or None
        Height of the figure in mm
    x0frac : fraction of Wfig to leave blank left of plot
    y0frac : fraction of Wfig to leave blank below plot
    wsfrac : fraction of Wfig to make width of subplot
    hs : height of subplot in mm. If none, uses ws = wsfrac * Wfig
    vspace : vertical space between subplots
    hspace : horizontal space btwn subplots
    tspace : space above top figure
    fontsize : size of text labels, title

    Returns
    -------
    fig : matplotlib.pyplot figure instance
    ax : matplotlib.pyplot axis instance
    """
    # Make figure
    fig, ax = initialize_spectrum_with_dos_plot(Wfig=Wfig, Hfig=Hfig, x0frac=x0frac, y0frac=y0frac, wsfrac=wsfrac,
                                                hs=hs, wsdosfrac=wsdosfrac, vspace=vspace, hspace=hspace,
                                                tspace=tspace, fontsize=FSFS)

    # ax[2].xaxis.set_major_locator( MaxNLocator(nbins = 3) ) #, prune = 'lower') )
    # ax[3].xaxis.set_major_locator( MaxNLocator(nbins = 3) )

    # Plot sideways DOS colored by PR: http://matplotlib.org/examples/pylab_examples/scatter_hist.html
    print 'kpfns: adding ipr to axis...'
    glat.add_ipr_to_ax(ax[0], ipr=None, alpha=1.0, inverse_PR=False,
                       norm=None, nbins=75, fontsize=FSFS, cbar_ax=ax[2],
                       vmin=None, vmax=None, linewidth=0, cax_label=r'$p$', make_cbar=True, climbars=True,
                       cbar_labelpad=3, orientation='horizontal', cbar_orientation='horizontal',
                       invert_xaxis=True, ylabel=r'$D(\omega)$', xlabel='Oscillation frequency $\omega$',
                       cbar_nticks=3, cbar_tickfmt='%0.1f')
    leplt.plot_pcolormesh(ksys_fracM, omegacM, nuM, ngrid, ax=ax[1], cax=ax[3], method='nearest',
                          cmap=cmaps.diverging_cmap(250, 10, l=30),
                          vmin=-1.0, vmax=1.0, title=title, xlabel=xlabel, ylabel=ylabel,  cax_label=r'$\nu$',
                          cbar_labelpad=3, cbar_orientation='horizontal', ticks=[-1, 0, 1], fontsize=FSFS)

    # Add title
    ax[0].annotate(title, xy=(0.5, 0.95), xycoords='figure fraction',
                   horizontalalignment='center', verticalalignment='center')
    # Match axes
    ax[0].set_ylim(ax[1].get_ylim())
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=2))  #, prune = 'upper') )
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.set_ticks_position('both')
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[2].xaxis.set_label_position("top")
    ax[3].xaxis.set_label_position("top")

    return fig, ax


def add_kitaev_regions_to_plot(chern=None, ksize_ind=0, ax=None, scale=1.0, offsetxy=np.array([0., 0.]),
                               manual_params={}):
    """Create a patch for the kitaev region and add it to current axes

    Parameters
    ----------
    chern : KitaevChern instance or None
        contains attributes chern_finsize and cp (chern parameter dictionary). If None, draws square region of size
        unity * scale with default angles
    ksize_ind : int (default=0)
        index of ksize for this chern instance to add to the axis (plot)
    ax : matplotlib.pyplot axis instance or None
        if None, uses current axis
    scale : float
        factor by which to magnify the region
    offsetxy : 2 x 1 float array
        where to place the center of the region in xy
    manual_params : dict
        if chern is None, then provides parameters for the summation region, with keys:
        shape, regalph, regbeta, reggamma, ksize, and polyT

    Returns
    -------
    ax : axis instance on which kitaev regions have been added as patches
    """
    if chern is None:
        if 'ksize' in manual_params:
            ksize = manual_params['ksize']
        else:
            ksize = 1.0
        if 'regalph' in manual_params:
            regalph = manual_params['regalph']
        else:
            regalph = np.pi * (11. / 6.)
        if 'regbeta' in manual_params:
            regbeta = manual_params['regbeta']
        else:
            regbeta = np.pi * (7. / 6.)
        if 'regbeta' in manual_params:
            reggamma = manual_params['reggamma']
        else:
            reggamma = np.pi * 0.5
        if 'shape' in manual_params:
            shape = manual_params['shape']
        else:
            shape = 'square'
        polygon1, polygon2, polygon3 = kfns.get_kitaev_polygons(shape, regalph, regbeta, reggamma, ksize)
        if 'polyT' in manual_params:
            if manual_params['polyT']:
                polygon1 = np.fliplr(polygon1)
                polygon2_tmp = np.fliplr(polygon3)
                polygon3 = np.fliplr(polygon2)
                polygon2 = polygon2_tmp

    else:
        ksize = chern.chern_finsize[:, 2] * 0.5
        polygon1, polygon2, polygon3 = kfns.get_kitaev_polygons(chern.cp['shape'], chern.cp['regalph'],
                                                                chern.cp['regbeta'], chern.cp['reggamma'],
                                                                ksize[ksize_ind])
    patchlist = [patches.Polygon(polygon1 * scale + offsetxy, color='r')]
    patchlist.append(patches.Polygon(polygon2 * scale + offsetxy, color='g'))
    patchlist.append(patches.Polygon(polygon3 * scale + offsetxy, color='b'))
    polypatches = PatchCollection(patchlist, cmap=cm.jet, alpha=0.4, zorder=99, linewidths=0.4)
    colors = np.linspace(0, 1, 3)[::-1]
    polypatches.set_array(np.array(colors))
    if ax is None:
        ax = plt.gca()
    ax.add_collection(polypatches)
    return ax


def plot_projector_locality(gyro_lattice, dists, magproj, evxymag, outdir=None, network_str='none', show=False,
                            alpha=None):
    """
    Plot the locality of the projection operator

    Parameters
    ----------
    gyro_lattice : GyroLattice instance
        The network on which to characterize the projector
    dists : dists : NP x NP float array
        Euclidean distances between points. Element i,j is the distance between particle i and j
    magproj : NP x NP float array
        Element i,j gives the magnitude of the projector connecting site i to particle j
    evxymag : 2*NP x NP float array
        Same as magproj, but with x and y components separate. So, element 2*i,j gives the magnitude of the x component
        of the projector connecting site i to the full xy of particle j and element 2*i+1,j gives the
        magnitude of the y component of the projector connecting site i to the full xy of particle j.
    outdir : str or None
        Path to save dists and magproj as pickles in outdir, also saves plots there
    network_str : str
        Description of the network for the title of the plot. If 'none', network_str = gyro_lattice.lp['LatticeTop'].
    show : bool
        Plot the result (forces a matplotlib close event)
    alpha : float or None
        opacity. If none, sets alpha = 3./len(dists)
    """
    if alpha is None:
        alpha = 3./len(dists)

    # Initialize plot
    fig, ax = leplt.initialize_1panel_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.15, wsfrac=0.4, hs=None,
                                          vspace=5, hspace=8, tspace=10, fontsize=8)

    # print 'np.shape(xymag) = ', np.shape(xymag)
    for ind in range(len(dists)):
        ax[0].scatter(dists[ind], np.log10(evxymag[:, 2 * ind].ravel()), s=1, color='r', alpha=alpha)
        ax[0].scatter(dists[ind], np.log10(evxymag[:, 2 * ind + 1].ravel()), s=1, color='b', alpha=alpha)

    if network_str is 'none':
        network_str = gyro_lattice.lp['LatticeTop']

    ax[0].set_title('Locality of projection operator $P$\nfor ' + network_str + ' network')
    ax[0].set_xlabel('Distance $|\mathbf{x}_i-\mathbf{x}_j|$')
    ax[0].set_ylabel('$|P_{ij}|$')
    if outdir is None:
        outdir = le.prepdir(gyro_lattice.lp['meshfn'].replace('networks', 'projectors'))
    le.ensure_dir(outdir)
    plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_log.png', dpi=300)

    ax[0].set_xlim(-0.5, 5)
    ax[0].set_ylim(-3.5, 0.5)
    plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_zoom_log.png', dpi=300)
    if show:
        plt.show()
    plt.clf()

    # Combine x and y
    # put projector magnitude on log-log plot
    fig, ax = leplt.initialize_1panel_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.15, wsfrac=0.4, hs=None,
                                          vspace=5, hspace=8, tspace=10, fontsize=8)
    for ind in range(len(dists)):
        ax[0].scatter(dists[ind], np.log10(magproj[ind]), s=1, color='k', alpha=alpha)
    ax[0].set_title('Locality of projection operator $P$\nfor ' + network_str + ' network')
    ax[0].set_xlabel('Distance $|\mathbf{x}_i-\mathbf{x}_j|$')
    ax[0].set_ylabel('$\log_{10} |P_{ij}|$')
    if outdir is None:
        outdir = le.prepdir(gyro_lattice.lp['meshfn'].replace('networks', 'projectors'))
    le.ensure_dir(outdir)
    plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_xy2_log.png', dpi=300)

    ax[0].set_xlim(-0.5, 5)
    ax[0].set_ylim(-3.5, 0.5)
    plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_xy2_zoom_log.png', dpi=300)
    if show:
        plt.show()
    plt.clf()

    # save dists and magproj as pkl
    with open(outdir + gyro_lattice.lp['LatticeTop'] + "_dists.pkl", "wb") as fn:
        pickle.dump(dists, fn)
    with open(outdir + gyro_lattice.lp['LatticeTop'] + "_magproj.pkl", "wb") as fn:
        pickle.dump(magproj, fn)


def plot_projector_locality_singlept(gyro_lattice, proj_ind, dists, magproj, ax=None, save=True,
                                     outdir=None, network_str='none', show=False, alpha=1.0):
    """
    Plot the locality of the projection operator wrt a point

    Parameters
    ----------
    gyro_lattice : GyroLattice instance
        The network on which to characterize the projector
    proj_ind : int
        The gyro index to analyze wrt
    dists : dists : NP x NP float array
        Euclidean distances between points. Element i,j is the distance between particle i and j
    magproj : NP x NP float array
        Element i,j gives the magnitude of the projector connecting site i to particle j
    evxymag : 2*NP x NP float array
        Same as magproj, but with x and y components separate. So, element 2*i,j gives the magnitude of the x component
        of the projector connecting site i to the full xy of particle j and element 2*i+1,j gives the
        magnitude of the y component of the projector connecting site i to the full xy of particle j.
    outdir : str or None
        Path to save dists and magproj as pickles in outdir, also saves plots there
    network_str : str
        Description of the network for the title of the plot. If 'none', network_str = gyro_lattice.lp['LatticeTop'].
    show : bool
        Plot the result (forces a matplotlib close event)
    alpha : float
        opacity
    """
    # Initialize plot
    if ax is None:
        fig, axes = leplt.initialize_1panel_fig(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.15, wsfrac=0.4, hs=None,
                                                vspace=5, hspace=8, tspace=10, fontsize=8)
        ax = axes[0]

    if network_str is 'none':
        network_str = gyro_lattice.lp['LatticeTop']

    ax.scatter(dists[proj_ind], np.log10(magproj[proj_ind]), s=1, color='k', alpha=alpha)
    ax.set_title('Locality of projection operator $P$\nfor ' + network_str + ' network')
    ax.set_xlabel('Distance $|\mathbf{x}_i-\mathbf{x}_j|$')
    ax.set_ylabel('$\log_{10} |P_{ij}|$')
    if save:
        if outdir is None:
            outdir = le.prepdir(gyro_lattice.lp['meshfn'].replace('networks', 'projectors'))
        le.ensure_dir(outdir)
        plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_xy2_log_{0:06d}'.format(proj_ind) +
                    '.png', dpi=300)

        ax.set_xlim(-0.5, 5)
        ax.set_ylim(-3, 0)
        plt.savefig(outdir + gyro_lattice.lp['LatticeTop'] + '_projector_xy2_zoom_log_{0:06d}'.format(proj_ind) +
                    '.png', dpi=300)

        # save dists and magproj as pkl
        outfn = outdir + gyro_lattice.lp['LatticeTop'] + "_dists_singlept_{0:06d}".format(proj_ind) + ".pkl"
        with open(outfn, "wb") as fn:
            pickle.dump(dists[proj_ind], fn)
        with open(outdir + gyro_lattice.lp['LatticeTop'] + "_magproj_{0:06d}".format(proj_ind) + ".pkl", "wb") as fn:
            pickle.dump(magproj[proj_ind], fn)
    if show:
        plt.show()


def plot_projector_singlept_network(glat, evxyproj, fig=None, ax=None, cax=None,
                                    cmap='viridis', save=True, sz=15, axis_off=False, fontsize=10, **kwargs):
    """Plot a network with each site colored by the magnitude of its projector value relative to a point closest to
    proj_XY.

    Parameters
    ----------
    glat : GyroLattice instance
    evxyproj : 2 x 2*NP complex array
        projector components corresponding to the particle of interest. First row corresponds to x component,
        second to y component
    fig : matplotlib.pyplot figure instance
    ax : axis instance
    cax : colorbar axis instance
    cmap : string specifier for colormap to use
    **kwargs : keyword arguments for leplt.initialize_1panel_cbar_cent()
        Wfig=90, Hfig=None, wsfrac=0.4, hsfrac=None, cbar_pos='above',
                                wcbarfrac=0.06, hcbarfrac=0.05, cbar_label='',
                                vspace=8, hspace=5, tspace=10, fontsize=8

    Returns
    -------

    """
    # make magproj, which has dims 1 x NP
    tmp = np.sqrt(np.abs(evxyproj[0]).ravel() ** 2 + np.abs(evxyproj[1]).ravel() ** 2)
    magproj = np.array([np.sqrt(tmp[2 * ind].ravel() ** 2 + tmp[2 * ind + 1].ravel() ** 2) for ind in
                        range(int(0.5 * len(tmp)))]).ravel()

    if ax is None:
        fig, ax, cbar_ax_tmp = leplt.initialize_1panel_cbar_cent(**kwargs)
    if cax is None:
        cbar_ax = cbar_ax_tmp
    else:
        cbar_ax = cax

    glat.lattice.plot_BW_lat(fig=None, ax=ax, save=False, close=False, axis_off=axis_off, title='')
    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()
    colormap = plt.get_cmap(cmap)
    colors = colormap(magproj/np.max(magproj))
    # print 'colors = ', colors
    ax.scatter(glat.lattice.xy[:, 0], glat.lattice.xy[:, 1], s=sz, c=colors, edgecolors='none', zorder=200)
    # create scalar mappable
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0., 0.5, 1.0])
    cbar_ax.set_title(r'$P_{i0}/P_{00}$', fontsize=fontsize)

    return fig, ax, cbar_ax


######################################################
######################################################
######################################################
######################################################
if __name__ == '__main__':
    '''Perform an example of using the lattice_collection class'''
    from lepm import gyro_collection

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-prpoly_overlay', '--prpoly_overlay', help='Make overlaid DOS plots', action='store_true')
    parser.add_argument('-ipr_overlay', '--ipr_overlay', help='Overlay participation ratio DOS plots',
                        action='store_true')
    parser.add_argument('-ipr_stack', '--ipr_stack', help='Stack subplots of participation-ratio-colored DOS',
                        action='store_true')

    # Geometry arguments for the lattices to load
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings ' +
                                          '(leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load',
                        help='Specify to nonzero int to load a network of a particular size in its entirety, ' +
                             'without cropping. Will override NH and NV',
                        type=int, default=20)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                   'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-kr_max', '--kr_max', help='Upper bound for magnitude of k in calculating S(k)', type=int,
                        default=30)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-nice_plot', '--nice_plot', help='Output nice pdf plots of lattice', action='store_true')
    
    # For loading and coordination
    parser.add_argument('-LLID', '--loadlattice_number', help='If LT=hyperuniform/isostatic, selects which ' +
                                                             'lattice to use', type=str, default='01')
    parser.add_argument('-LLz', '--loadlattice_z', help='If LT=hyperuniform/isostatic, selects what z index to use',
                        type=str, default='001')
    parser.add_argument('-source', '--source', help='Selects who made the lattice to load, if loaded from source ' +
                                                   '(ulrich, hexner, etc)', type=str, default='hexner')
    parser.add_argument('-cut_z', '--cut_z',
                        help='Declare whether or not to cut bonds to obtain target coordination number z',
                        type=bool, default=False)
    parser.add_argument('-cutz_method','--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy', help='Position of single dislocation, if not centered at' +
                                                              '(0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')
    
    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-phi', '--phi', help='Shear angle for hexagonal (honeycomb) lattice in radians/pi', type=float,
                        default=0.0)
    parser.add_argument('-delta', '--delta', help='Deformation angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=120./180.)
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-huno', '--hyperuniform_number', help='Hyperuniform realization number',
                        type=str, default='01')
    parser.add_argument('-skip_gr', '--skip_gr', help='Skip calculation of g(r) correlation function for the lattice',
                        action='store_true')
    parser.add_argument('-skip_gxy', '--skip_gxy', help='Skip calculation of g(x,y) 2D correlation function ' +
                                                        'for the lattice', action='store_true')
    parser.add_argument('-skip_sigN', '--skip_sigN', help='Skip calculation of variance_N(R)', action='store_true')
    parser.add_argument('-fancy_gr', '--fancy_gr', help='Perform careful calculation of g(r) correlation ' +
                                                        'function for the ENTIRE lattice', action='store_true')
    args = parser.parse_args()
    
    print 'args.delta  = ', args.delta
    if args.N > 0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV
    lattice_type = args.LatticeTop

    phi = np.pi * args.phi
    delta = np.pi * args.delta

    strain = 0.00  # initial
    # z = 4.0 #target z
    if lattice_type == 'linear':
        shape = 'line'
    else:
        shape = args.shape
    
    theta = args.theta
    eta = args.eta
    transpose_lattice = 0
    
    make_slit = args.make_slit
    # deformed kagome params
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    z = 0.0

    # Define description of the network topology
    if args.LatticeTop == 'iscentroid':
        description = 'voronoized jammed'
    elif args.LatticeTop == 'kagome_isocent':
        description = 'kagomized jammed'
    elif args.LatticeTop == 'hucentroid':
        description = 'voronoized hyperuniform'
    elif args.LatticeTop == 'kagome_hucent':
        description = 'kagomized hyperuniform'
    elif args.LatticeTop == 'kagper_hucent':
        description = 'kagome decoration percolation'

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    outdir = rootdir + 'experiments/DOS_scaling/' + args.LatticeTop + '/'
    le.ensure_dir(outdir)
    lp = {'LatticeTop': args.LatticeTop,
          'NH': NH,
          'NV': NV,
          'rootdir': rootdir,
          'periodicBC': True,
          }

    # Collate DOS for many lattices
    gc = gyro_collection.GyroCollection()
    if args.LatticeTop == 'iscentroid':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/'+args.LatticeTop+'/' +
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'kagome_isocent':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' +
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' +
            args.LatticeTop + '_square_periodic_d*'+'_NP*' + str(args.NP_load))
    elif args.LatticeTop == 'kagper_hucent':
        gc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' +
                      args.LatticeTop + '_square_d*' + '_' + '{0:06d}'.format(NH))

    title = r'$D(\omega)$ for ' + description + ' networks'

    if args.prpoly_overlay:
        gc.ensure_all_ipr_saved()
        gc.plot_prpoly_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_prpoly_overlay',title=title)

    if args.ipr_overlay:
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_ipr_overlay',
                                title=title, inverse_PR=True)
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_pr_overlay',
                                title=title, inverse_PR=False)
        gc.plot_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_hist_overlay', title=title)

    if args.ipr_stack:
        # Get ylabels from meshfn names
        ylabels = []
        ii = 0
        for meshfn in gc.meshfns:
            pstring = meshfn[meshfn.index('perd')+4:meshfn.index('perd')+8].replace('p', '.')
            ylabels.append(pstring)
            ii += 1

        gc.plot_ipr_DOS_stack(outdir=outdir, fname='N'+str(args.N)+'_eigval_ipr_stack', title=title,
                              ylabels=ylabels, inverse_PR=False, vmin=0.0, vmax=0.5)
