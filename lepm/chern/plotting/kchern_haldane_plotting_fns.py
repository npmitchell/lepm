import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import numpy as np
import matplotlib.pyplot as plt
import lepm.data_handling as dh
from matplotlib.patches import Polygon

'''
Description
===========
Auxiliary plotting functions for kspace haldane model network chern calculation
'''


def plot_chernbands(kchern, fig=None, ax=None, cax=None, outpath=None, eps=1e-10, cmap='bbr0',
                    vmin=-1.0, vmax=1.0, ticks=None, round_chern=True, dpi=150, alpha=1.0):
    """

    Parameters
    ----------
    kchern : KChern class instance
    fig : matplotlib figure instance or None
    ax : matplotlib axis instance or None

    Returns
    -------
    fig, ax
    """
    # vertex_points = kchern.chern['bzvtcs']
    kkx = kchern.chern['kx']
    # kky = kchern.chern['ky']
    bands = kchern.chern['bands']
    chern = kchern.chern['chern']
    # b = ((1j / (2 * np.pi)) * np.mean(traces[:, i]) * bzarea)
    if ax is None:
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(wsfrac=0.5, tspace=4)

    if isinstance(cmap, str):
        cmap = lecmaps.ensure_cmap(cmap)

    xmax, ymax = 0., 0.
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
        poly = Polygon(polygon, closed=True, fill=True, lw=0.00, alpha=alpha, color=cmap(colorval), edgecolor=None)
        xmax = max(xmax, np.max(np.abs(kkx)))
        ymax = max(ymax, np.max(np.abs(band)))
        ax.add_artist(poly)
        # ax.plot(polygon[:, 0], polygon[:, 1], 'k.-')

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)

    if cax is not None:
        if vmin is None:
            vmin = np.min(np.real(np.array(chern)))
        if vmax is None:
            vmax = np.max(np.real(np.array(chern)))
        if ticks is None:
            ticks = [int(vmin), int(vmax)]

        sm = leplt.empty_scalar_mappable(cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(sm, cax=cax, ticks=ticks, label=r'Chern number, $C$', alpha=alpha)

    if outpath is not None:
        ax.set_xlabel('wavenumber, $k_x$')
        ax.set_ylabel('frequency, $\omega$')
        plt.savefig(outpath, dpi=dpi)
        plt.close()

    return fig, ax, cax

