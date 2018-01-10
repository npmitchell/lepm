import lepm.lattice_elasticity as le
import lepm.plotting.colormaps as lecmaps
import lepm.dataio as dio
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

"""Functions for plotting time domain simulation output"""


def le_plot_lattice(xy, xy0, NL, KL, BL, bL0, params, t, ii, name, outdir, climv=0.1,
                    exaggerate=1.0, colorz=False, ax='none', axcb='none'):
    """Plots a spring lattice with strain colored bonds, and with particles colored by coordination if arg colorz is
    True.

    Parameters
    ----------
    xy :
    xy0:
    NL :
    KL :
    BL :
    bL0 : NP x 1 float array
        rest lengths of the bonds
    params : dict
        parameters for the simulation
    t : float
        time stamp for image
    name : string
        the name of the file (before _index.png)
    outdir : string
        The output directory for the image
    climv : float or tuple
        Color limit for coloring bonds by bond strain
    numbering : 'natural' or 'adopt' (default = 'adopt')
        Use indexing '0','1','2','3',... or adopt the index of the input file.
    exaggerate : float (default 1.0 --> in which case it is ignored)
        Exaggerate the displacements of each particle from its initial position by this factor. Ignored if == 1.0
    colorz : bool
        Whether to color particles by their coordination number (z).

    Returns
    ----------
    ax : axis to clear after plotting
    cbar : colorbar to clear after plotting
    """
    # make output dir
    outdir = dio.prepdir(outdir)
    dio.ensure_dir(outdir)
    # set range of window from first values
    xlimv = np.ceil(max(max(xy0[:, 0]), max(xy0[:, 1]) ) * 5. /4.)
    ylimv = xlimv

    # save current data as stills
    index = '{0:08d}'.format(ii)
    outname = outdir+ '/' + name + '_' + index + '.png'
    BL = le.NL2BL(NL, KL)

    if 'prestrain' in params:
        prestrain = params['prestrain']
    else:
        prestrain = 0.

    if 'shrinkrate' in params:
        shrinkrate = params['shrinkrate']
        title = 't = ' + '%09.4f' % t + '  a = ' + '%07.5f' % (1. - shrinkrate * t - prestrain)
    elif 'prestrain' in params:
        shrinkrate = 0.0
        title = 't = ' + '%09.4f' % t + '  a = ' + '%07.5f' % (1. - prestrain)
    else:
        shrinkrate = 0.0
        title = 't = ' + '%09.4f' % t

    # calculate strain
    bs = le.bond_strain_list(xy, BL, bL0)

    if exaggerate == 1.0:
        [ax, axcb] = movie_plot_2D(xy, BL, bs, outname, title, xlimv=xlimv, ylimv=ylimv, climv=climv, colorz=colorz,
                                   ax=ax, axcb=axcb)
    else:
        xye = xy0 + (xy - xy0) * exaggerate
        [ax, axcb] = movie_plot_2D(xye, BL, bs, outname, title, xlimv=xlimv, ylimv=ylimv, climv=climv, colorz=colorz,
                                   ax=ax, axcb=axcb)
    return [ax, axcb]


def plot_binned_energies_time(energy_bins, hstep, ylabel='Energy', ax=None, outdir=None, fn='energy_vs_time.pdf',
                              title="Energy in each bin over time", show=False, cmap='viridis_r'):
    """Plot energy vs time. Save or show the plot if outdir is not None or if show is True.

    Parameters
    ----------
    energy_bins : #timesteps x #bins float array
        Element ij gives the energy (potential, kinetic, or both) at timestep i for the jth bin.
    hstep: float
        Time step in units of inverse precession period
    outdir : str or None
        The path for the output
    fn : str
        filename with extension to save plot (inside outdir, so not full path, just filename)
    title : str
        Plot title
    show : bool
        Show the results if outdir is None

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()

    # colors = lecmaps.husl_palette(n_colors=np.shape(energy_bins)[1], s=1, l=0.6)
    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()

    colormap = plt.get_cmap(cmap)
    colors = colormap(np.linspace(0, 1, np.shape(energy_bins)[1]))
    print 'colors = ', colors
    for kk in range(np.shape(energy_bins)[1]):
        ax.plot(hstep * np.arange(len(energy_bins)), energy_bins[:, kk], c=colors[kk])

    ax.set_xlabel('time [$\Omega_g^{-1}$]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin=0)

    if outdir is not None:
        print 'saving fig: ', outdir + fn
        plt.savefig(outdir + fn)
    elif show:
        plt.show()
    return ax


def plot_binned_energies_binsy(energy_bins, hstep, binsy, xlabel='Position [bond lengths]', ylabel='Energy',
                               ax=None, cbar_ax=None, outdir=None, fn='energy_vs_bin.pdf',
                               title="Energy in each bin over time", cmap='viridis_r', show=False):
    """Plot energy vs time, and energy versus binsy. Save or show the plot if outdir is not None or if show is True.

    Parameters
    ----------
    energy_bins : #timesteps x #bins float array
        Element ij gives the energy (potential, kinetic, or both) at timestep i for the jth bin.
    binsy : #bins or #bins-1 x 1 float array
        The distance along the wave propagation direction of the bin or the bounds of the bin if length is #bins + 1.
    ax : matplotlib axis instance or None
        Axis on which to plot. If none, grabs current axis
    outdir : str or None
        The path for the output
    show : bool
        Show the results if outdir is None

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()

    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()

    for kk in range(len(binsy)):
        ax.scatter(binsy[kk] * np.ones(len(energy_bins)), energy_bins[:, kk],
                   c=np.arange(len(energy_bins), dtype=float)/len(energy_bins),
                   cmap=cmap, vmin=0.0, vmax=1.0, edgecolors='none')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0., vmax=len(energy_bins)*hstep))
    # fake up the array of the scalar mappable.
    sm._A = []
    cbar = plt.colorbar(sm, cax=cbar_ax, label='time [$\Omega_g^{-1}$]')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin=0)

    if outdir is not None:
        print 'saving fig: ', outdir + fn
        plt.savefig(outdir + fn)
    elif show:
        plt.show()

    return ax


def DOSexcite_curve(freq, stdev_time, kvals):
    """Return Gaussian excitation in kspace.

    Parameters
    ----------
    freq : float
        The excitation frequency
    stdev_time : float
        The standard deviation in time of the gaussian excitation
    kvals : 300 x 1 float array
        An array of frequency values at which the amplitude of the excitation in k space is evaluated.
        For example, kvals = np.linspace(1.0, 4.0, 300)

    Returns
    -------
    gaussk : 300 x 1 float array
        The amplitude (in k space) of the gaussian (in time) excitation, for each k value in kout
    """
    # DOSexcite = (frequency, sigma_time)
    # amp(x) = exp[- acoeff * time**2]
    # amp(k) = sqrt(pi/acoeff) * exp[- pi**2 * k**2 / acoeff]
    # So 1/(2 * sigma_freq**2) = pi**2 /acoeff
    # So sqrt(acoeff/(2 * pi**2)) = sigma_freq

    sigmak = 1./stdev_time
    gaussk = np.exp(-(kvals - freq)**2 / (2. * sigmak))

    return gaussk
