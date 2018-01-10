import numpy as np
import lepm.lattice_elasticity as le
import lepm.kitaev.kitaev_functions as kfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmaps
from PIL import Image
from os import getcwd, chdir
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

'''Plot Bott Index computations
'''


def plot_spectrum_with_dos(glat, omegacv, nuv, nbins=None,
                           xlabel=r'frequency, $\omega$', ylabel=r'$D(\omega)$', ylabel2=r'$B$', title=None,
                           xlim=None, vmin=None, vmax=None, alpha=1.0,
                           nucolor='#009cfb', colormap='inferno', locz=None, color_locz=True,
                           fontsize=None):
    """Plot bott spectrum in 1d histogram, with corresponding DOS plot behind the bott calculation.

    Parameters
    ----------
    glat : GyroLattice instance
        The gyro network to use for getting the DOS (should match bott info, though this fn will still work if they are
        mismatched)
    omegac : n x 1 float array
        the cutoff frequencies for the calculation
    nu : n x 1 float array
        the bott index results
    nbins : int
        number of bars in histogram of eigvals
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
    eigv = glat.get_eigval()
    if color_locz and locz is None:
        locz = glat.get_ill()

    fig, dos_ax, lev_ax, cax, cbar = \
        leplt.initialize_colored_DOS_plot_twinax(eigv, 'gyro', axis_pos=(0.15, .2, .8, .62),
                                                 alpha=alpha, colorV=locz, nbins=nbins, histrange=xlim,
                                                 colormap=colormap, cbar_nticks=6,
                                                 linewidth=0, cax_label=r'$\xi^{-1}$',
                                                 vmin=vmin, vmax=vmax,
                                                 cbar_labelpad=20, fontsize=fontsize)

    # Also plot level spacing variance
    lev_ax.tick_params('y', colors=nucolor)
    lev_ax.plot(omegacv, nuv, color=nucolor)
    # lev_ax.plot([-maxlim[nn]['xlim'], maxlim[nn]['xlim']], [0.178, 0.178], '--', color=nucolor)
    # lev_ax.set_ylim(0, 1)
    # lev_ax.set_ylabel(r'$\sigma^2_{\Delta \omega}$', va='center', rotation=0, labelpad=15, color=nucolor)
    dos_ax.set_xlabel(xlabel)
    dos_ax.set_ylabel(ylabel)
    lev_ax.set_ylabel(ylabel2)

    # Add title
    if title is not None:
        dos_ax.annotate(title, xy=(0.5, 0.95), xycoords='figure fraction',
                        horizontalalignment='center', verticalalignment='center')

    return fig, dos_ax, lev_ax, cax, cbar


def initialize_spectrum2d_with_dos_plot(Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1, wsfrac=0.4, hs=None,
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
              [x0 + wsDOS + hspace, y0, ws, hs, ''],  # Chern omegac vs ksize
              [x0 + wsDOS * 0.5 - wscbar * 0.5, y0 + hs + vspace, wscbar, hscbar, ''],  # cbar for ipr
              [x0 + wsDOS + hspace + ws * 0.5 - wscbar, y0 + hs + vspace, wscbar * 2, hscbar, '']  # cbar for chern
          )]
    return fig, ax


def plot_spectrum2d_with_dos(glat, paramM, omegacM, nuM, ngrid, Wfig=90, Hfig=None, x0frac=0.15, y0frac=0.1,
                             wsfrac=0.4, hs=None, wsdosfrac=0.3, vspace=8, hspace=8, tspace=10, FSFS=8,
                             xlabel=None, ylabel=None, title=None):
    """Plot bott spectrum in 2D colormap with other variable (paramM) on x axis, with corresponding DOS plot aligned
    with y axis (increasing in frequency).

    Parameters
    ----------
    glat : gyro_lattice
    paramM :
    omegacM :
    nuM :
    ngrid : int
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
    fig, ax = initialize_spectrum2d_with_dos_plot(Wfig=Wfig, Hfig=Hfig, x0frac=x0frac, y0frac=y0frac, wsfrac=wsfrac,
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
    leplt.plot_pcolormesh(paramM, omegacM, nuM, ngrid, ax=ax[1], cax=ax[3], method='nearest',
                          cmap=cmaps.diverging_cmap(250, 10, l=30),
                          vmin=-1.0, vmax=1.0, title=title, xlabel=xlabel, ylabel=ylabel, cax_label=r'$\nu$',
                          cbar_labelpad=3, cbar_orientation='horizontal', ticks=[-1, 0, 1], fontsize=FSFS)

    # Add title
    ax[0].annotate(title, xy=(0.5, 0.95), xycoords='figure fraction',
                   horizontalalignment='center', verticalalignment='center')
    # Match axes
    ax[0].set_ylim(ax[1].get_ylim())
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=2))  # , prune = 'upper') )
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.set_ticks_position('both')
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[2].xaxis.set_label_position("top")
    ax[3].xaxis.set_label_position("top")

    return fig, ax
