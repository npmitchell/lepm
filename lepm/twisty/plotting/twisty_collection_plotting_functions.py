import numpy as np
import lepm.plotting.colormaps as cmaps
import lepm.plotting.plotting as leplt
import matplotlib.pyplot as plt

"""Auxiliary functions for plotting gyro_collections, such as overlaying colored DOS histograms.
"""


def plot_ill_dos_overlay(gcoll, dos_ax=None, cbar_ax=None, alpha=None, vmin=None, vmax=None, **kwargs):
    """
    Parameters
    ----------
    dos_ax : axis instance
        Axis on which to plot the Localization-colored DOS
    **kwargs: lepm.plotting.plotting.colored_DOS_plot() keyword arguments
        Excluding fontsize
    """
    gcoll.ensure_all_eigval()

    # Decide what alpha value to use if not provided (here, 1/N)
    if alpha is None:
        alpha = 1./float(len(gcoll.gyro_lattices))

    # Register cmaps if necessary
    if 'viridis' not in plt.colormaps():
        cmaps.register_colormaps()

    # Plot the overlays
    for ii in range(len(gcoll.gyro_lattices)):
        if ii % 10 == 0:
            print 'Overlaying ', ii, ' of ', len(gcoll.gyro_lattices)

        eigval = gcoll.gyro_lattices[ii].eigval

        # Load or compute localization
        print 'gyrocollection_pltfns: getting localization...'
        localization = gcoll.gyro_lattices[ii].get_localization(attribute=False)
        ill = np.zeros(len(eigval), dtype=float)
        ill[0:int(len(eigval) * 0.5)] = localization[:, 2][::-1]
        ill[int(len(eigval) * 0.5):len(eigval)] = localization[:, 2]
        print 'ill = ', ill
        # print 'gcollpfns: localization = ', localization
        # print 'gcollpfns: shape(localization) = ', np.shape(localization)

        # Now overlay the ipr
        if vmin is None:
            print 'setting vmin...'
            vmin = 0.0
        if ii == 0 and dos_ax is None:
            print 'dos_ax is None, initializing...'
            fig, dos_ax, cax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro',
                                                                       alpha=alpha, colorV=ill,
                                                                       colormap='viridis',
                                                                       linewidth=0, cax_label=r'$\lambda^{-1}$',
                                                                       vmin=vmin, vmax=vmax,
                                                                       climbars=False, **kwargs)
            print 'cbar_ax = ', cax
            if vmax is None:
                print 'setting vmax...'
                vmax = cbar.get_clim()[1]
        elif ii == 0:
            dos_ax, cbar_ax, cbar, n, bins = \
                leplt.colored_DOS_plot(eigval, dos_ax, 'gyro', alpha=alpha, colorV=ill,
                                       cbar_ax=cbar_ax, colormap='viridis', linewidth=0,
                                       make_cbar=True, vmin=vmin, vmax=vmax, **kwargs)
            if vmax is None:
                print 'setting vmax...'
                vmax = cbar.get_clim()[1]
        else:
            print 'gcollpfns: calling leplt.colored_DOS_plot...'
            leplt.colored_DOS_plot(eigval, dos_ax, 'gyro', alpha=alpha, colorV=ill, cbar_ax=cbar_ax,
                                   colormap='viridis', linewidth=0, make_cbar=False, vmin=vmin, vmax=vmax, **kwargs)

    return dos_ax, cbar_ax
