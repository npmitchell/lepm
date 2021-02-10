import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
from magnetic_gyro_lattice_class import MagneticGyroLattice
import lepm.plotting.plotting as leplt
import lepm.dataio as dio
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov
from lepm.gyro_data_handling import phase_minimizes_difference_complex_displacments
import glob
import subprocess
import sys
import lepm.stringformat as sf


'''Auxiliary script-like functions called by magnetic_gyro_lattice_class.py when __name__=='__main__'.
'''


def dispersion_abtransition(lp, save_plots=True, return_omegas=False, abvals=None, fullspectrum=False):
    """

    Parameters
    ----------
    lp : dictionary
        lattice parameter dictionary
    return_omegas : bool
        Return a list of the frequencies in the dispersion
    abvals : n x 1 float array or list
        The Delta_AB values for each spectrum

    Returns
    -------

    """
    # prepare the values of inversion symmetry breaking
    if abvals is None:
        # abvals = np.arange(0, 2.4, 0.1)
        abvals = np.arange(0, 1.0, 0.025)

    # Prepare magnetic gyro lattice
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lpmaster = copy.deepcopy(lp)
    # create a place to put images
    outdir = dio.prepdir(meshfn) + 'dispersion_abtransition_aol' + sf.float2pstr(lp['aoverl']) + '/'
    dio.ensure_dir(outdir)
    fs = 20

    if return_omegas:
        omegas_out = []
        kx_out = []
        ky_out = []

    # go through each ab and make the image
    ii = 0
    for ab in abvals:
        lp = copy.deepcopy(lpmaster)
        lp['ABDelta'] = ab
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        glat = MagneticGyroLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        fig, ax = leplt.initialize_portrait(ax_pos=[0.12, 0.2, 0.76, 0.6])
        omegas, kx, ky = glat.infinite_dispersion(save=False, nkxvals=50, nkyvals=50, outdir=outdir, save_plot=False)
        if return_omegas:
            omegas_out.append(omegas)
            kx_out.append(kx)
            ky_out.append(ky)

        if save_plots:
            # plot and save it
            title = r'$\Delta_{AB} =$' + '{0:0.2f}'.format(ab)
            for jj in range(len(ky)):
                for kk in range(len(omegas[0, jj, :])):
                    ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.5, 30. / (len(kx) * len(ky))))

            ax.set_title(title, fontsize=fs)
            ax.set_xlabel(r'$k$ $[\langle \ell \rangle ^{-1}]$', fontsize=fs)
            ax.set_ylabel(r'$\omega$', fontsize=fs)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fs)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs)
            if ii == 0:
                ylims = ax.get_ylim()

            if fullspectrum:
                ax.set_ylim(-ylims[1] * 1.2, ylims[1] * 1.2)
            else:
                ax.set_ylim(0., ylims[1] * 1.2)

            # Save it
            name = outdir + 'dispersion{0:04d}'.format(ii)
            plt.savefig(name + '.png', dpi=300)
            plt.close('all')

        ii += 1

    # Save a movie of the dispersions
    movname = dio.prepdir(meshfn) + 'dispersion_abtrans' + glat.lp['meshfn_exten']
    if save_plots:
        # Turn images into a movie
        imgname = outdir + 'dispersion'
        lemov.make_movie(imgname, movname, indexsz='04', framerate=4, imgdir=outdir, rm_images=True,
                         save_into_subdir=True)

    if return_omegas:
        # Save the gap bounds and return the arrays of kx, ky, and omega
        omegas, kx, ky = np.array(omegas_out), np.array(kx_out), np.array(ky_out)
        # Get gap bounds:
        botmin = []
        botmax = []
        topmin = []
        topmax = []
        for band in omegas:
            botmin.append(np.min(band[:, :, 2]))
            botmax.append(np.max(band[:, :, 2]))
            topmin.append(np.min(band[:, :, 3]))
            topmax.append(np.max(band[:, :, 3]))

        botmin = np.array(botmin)
        botmax = np.array(botmax)
        topmin = np.array(topmin)
        topmax = np.array(topmax)
        gapbounds = np.dstack((abvals, botmin, botmax, topmin, topmax))[0]
        outfn = movname + '_gapbounds.txt'
        print 'saving to ', outfn
        header = 'Delta_AB, band min frequency, band max frequency, band 2 min, band 2 max...'
        np.savetxt(outfn, gapbounds, header=header)
        plot_gapbounds(abvals, gapbounds,  movname + '_gapbounds.png', lp)
        return omegas, kx, ky
    else:
        return None


def dispersion_abtransition_gapbounds(lp, abvals=None):
    """

    Parameters
    ----------
    lp

    Returns
    -------

    """
    # prepare the values of inversion symmetry breaking
    if abvals is None:
        # abvals = np.arange(0, 2.4, 0.1)
        abvals = np.arange(0, 1.0, 0.025)

    meshfn = le.find_meshfn(lp)
    lat = lattice_class.Lattice(lp)
    lat.load(check=lp['check'])
    lp['ABDelta'] = np.max(abvals)
    mglat = MagneticGyroLattice(lat, lp)
    outname = dio.prepdir(meshfn) + 'dispersion_abtrans' + mglat.lp['meshfn_exten'] + '_gapbounds'
    gapbounds_glob = glob.glob(outname + '.txt')
    if gapbounds_glob:
        # Load the text file and save the image
        gapbounds = np.loadtxt(outname + '.txt')
        plot_gapbounds(gapbounds[:, 0], gapbounds[:, 1:], outname + '.png', lp)
    else:
        # Compute the dispersions and save the image (occurs inside dispersion_abtransition
        omegas, kx, ky = dispersion_abtransition(lp, save_plots=False, return_omegas=True)


def plot_gapbounds(abv, gapbounds, outname, lp):
    """Plot the gap closing as a function of Delta_AB for magnetic system.
    Axis parameters are fixed so that output can be made into a movie

    Parameters
    ----------
    abv : n x 1 float array
        The Delta_AB values for each spectrum
    gapbounds: n x 2(#bands) float array
        The bottom and top of each band for each value of abv
    """
    plt.close('all')
    fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=2)
    # ax.plot(abv, gapbounds[:, 0], '.-', color=lecmaps.violet())
    # ax.plot(abv, gapbounds[:, 1], '.-', color=lecmaps.green())
    nbands = int(0.5 * np.shape(gapbounds)[1])
    colors = [lecmaps.green(), lecmaps.violet(), lecmaps.orange(), lecmaps.blue(), lecmaps.red(), '#000000']
    for ii in range(nbands):
        ax.fill_between(abv, gapbounds[:, 1 + 2 * ii], gapbounds[:, 0 + 2 * ii], color=colors[ii])

    ax.text(0.5, 1.12, 'Bands for magnetic gyros (' + r'$a/\ell=$' + '{0:0.2f}'.format(lp['aoverl']) + ', '
            r'$\Omega_k/\Omega_g=$' + '{0:0.3f}'.format(lp['Omk']/lp['Omg']) + ')',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel(r'Inversion symmetry breaking $\Delta_{AB}$')
    ax.set_ylabel(r'Frequency, $\omega$')
    ax.set_ylim(0, 3.5)
    print 'saving figure: ', outname
    plt.savefig(outname)
    plt.close('all')
