import numpy as np
import lepm.lattice_elasticity as le
import lepm.kitaev.kitaev_functions as kfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.kitaev_plotting_functions as kpfns
import lepm.plotting.network_visualization as netvis
import lepm.stringformat as sf
##################
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
import random
import socket
import glob
import lepm.dataio as dio
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl


"""Functions for plotting results of collecting many bott index instances
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
            [Wfig * center0_frac - ws * 0.5, y0, ws, hs, ''],  # Chern vary glatparam vs ksize
            [Wfig * center0_frac - wscbar, y0 + hs + vspace, wscbar * 2, hscbar, ''],  # cbar for chern (DOS)
            [Wfig * center2_frac - ws * 0.5, y0, wss, hss, '']  # subplot for showing kitaev region
          )]
    return fig, ax


def plot_bott_spectrum_on_axis(bcoll, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, fontsize=8, alpha=1.0,
                               xlabel=r'frequency, $\omega$', ylabel=r'$D(\omega)$', ylabel2=r'$B$', title=None,
                               ipr_vmin=None, ipr_vmax=None, cbar_labelpad=3, invert_xaxis=True,
                               ipr_cax_label='participation\nratio,' + r' $p$',
                               bott_cax_label='Bott\nindex, ' + r'$B$', nucolor='#009cfb'):
    """Plot Bott index vs cutoff frequency for the projector for a collection of bott index measurements

    Parameters
    ----------
    bcoll : BottCollection instance
        The bott collection to add to the plot
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
    # to be used after instantiating axes, like:
    # fig, dos_ax, lev_ax, cax, cbar = \
    #     leplt.initialize_colored_DOS_plot_twinax(eigv, 'gyro', axis_pos=(0.15, .2, .8, .62),
    #                                              alpha=alpha, colorV=locz, nbins=nbins, histrange=xlim,
    #                                              colormap=colormap, cbar_nticks=6,
    #                                              linewidth=0, cax_label=r'$\xi^{-1}$',
    #                                              vmin=vmin, vmax=vmax,
    #                                              cbar_labelpad=20, fontsize=fontsize)

    dmyi = 0
    for glat_name in bcoll.botts:
        # Grab a pointer to the gyro_lattice
        glat = bcoll.botts[glat_name][0].gyro_lattice
        dmyi += 1
    if dmyi > 1:
        raise RuntimeError('This function takes a GyroCollection which is allowed only ONE gyro_lattice-- ' +
                           'ie, a Bott spectrum for a single GyroLattice instance')

    print 'Opening glat_name = ', glat_name
    ngrid = len(bcoll.botts[glat_name][0].chern_finsize)

    # Build omegacV
    omegacv = np.zeros(len(bcoll.botts[glat_name]))
    nuv = np.zeros((len(bcoll.botts[glat_name]), 1))

    for ind in range(len(bcoll.botts[glat_name])):
        print 'ind = ', ind
        cp_ii = bcoll.botts[glat_name][ind].cp
        omegacv[ind] = cp_ii['omegac']
        nuv[ind] = bcoll.botts[glat_name][ind].bott

    print 'omegacv = ', omegacv

    print 'bcpfns: adding ipr and bott spectrum to axes...'
    print 'Adding bott spectrum to axis...'

    # Also plot level spacing variance
    ax.tick_params('y', colors=nucolor)
    ax.plot(omegacv, nuv, color=nucolor)
    # lev_ax.plot([-maxlim[nn]['xlim'], maxlim[nn]['xlim']], [0.178, 0.178], '--', color=levcolor)
    # lev_ax.set_ylim(0, 1)
    # lev_ax.set_ylabel(r'$\sigma^2_{\Delta \omega}$', va='center', rotation=0, labelpad=15, color=levcolor)
    dos_ax.set_xlabel(xlabel)
    dos_ax.set_ylabel(ylabel)
    lev_ax.set_ylabel(ylabel2)

    if dos_ax is not None and dos_ax != 'none':
        print 'Adding ipr to DOS axis: dos_ax = ', dos_ax
        glat.add_ipr_to_ax(dos_ax, ipr=None, alpha=alpha, inverse_PR=False,
                           norm=None, nbins=75, fontsize=fontsize, cbar_ax=cbar_ipr_ax,
                           vmin=ipr_vmin, vmax=ipr_vmax, linewidth=0, cax_label=ipr_cax_label,
                           make_cbar=True, climbars=True,
                           cbar_labelpad=cbar_labelpad, orientation='horizontal', cbar_orientation='horizontal',
                           invert_xaxis=invert_xaxis, ylabel=None, xlabel=None,
                           cbar_nticks=2, cbar_tickfmt='%0.1f')
        ax.set_ylim(dos_ax.get_ylim())

    return ax, dos_ax, cbar_ipr_ax, cbar_nu_ax
