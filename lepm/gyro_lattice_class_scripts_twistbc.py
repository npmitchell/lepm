import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
from lepm.gyro_lattice_class import GyroLattice
import lepm.plotting.plotting as leplt
import lepm.dataio as dio
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov
from lepm.gyro_data_handling import phase_minimizes_difference_complex_displacments
import glob
import subprocess
import sys
import lepm.plotting.science_plot_style as sps
try:
    import pylab
except ImportError:
    print 'WARNING: Could not import pylab'

'''Auxiliary script-like functions called by gyro_lattice_class.py when __name__=='__main__' for
gyro_lattice_class.py. All functions in this module are for making plots and movies for mtwisted boundary conditions.
'''


def twiststrip(lp):
    """Example usage:
   python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 14 -NV 7 -twiststrip
   python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 10 -NP 50 -twiststrip
   python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 15 -NP 50 -twiststrip
   python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 15 -NP 50 -twiststrip -conf 2
   python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 20 -NV 17 -NP 20 -twiststrip
   python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 30 -alph 0.3 -twiststrip
   python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 40 -alph 0.25 -twiststrip
   python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 50 -alph 0.3 -twiststrip -conf 3

   # to make more periodicstrips
   python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 20 -NV 17 -NP 20 -skip_polygon -skip_gyroDOS
   python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 15 -NP 50 -skip_polygon -skip_gyroDOS -conf 2
   python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 15 -NP 50 -skip_polygon -skip_gyroDOS -conf 2

   # to make annuli
   python ./build/make_lattice.py -LT hucentroid_annulus -N 50 -alph 0.3 -skip_polygons -skip_gyroDOS -conf 2
   python ./build/make_lattice.py -LT hucentroid_annulus -N 40 -alph 0.25 -skip_polygons -skip_gyroDOS
   python ./build/make_lattice.py -LT hucentroid_annulus -N 30 -alph 0.2 -skip_polygons -skip_gyroDOS

    Parameters
    ----------
    lp

    Returns
    -------

    """
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lpmaster = copy.deepcopy(lp)
    lat = lattice_class.Lattice(lp)
    lat.load()
    # print 'glattwistscripts: PVxydict = ', lat.PVxydict
    # print 'glattwistscripts: PV = ', lat.PV
    # Set vmax to be 2/L, where L is the full width of the sample in y (the strip width)
    lsize = 2. * np.max(np.abs(lat.xy[:, 1]))
    vmax = 2. / lsize
    vmax_tb = 1.
    eigvals = []
    loczs = []
    tbs = []
    thetavals = np.linspace(0, 2., 41)[::-1]

    # Note: theta_twist and phi_twist are in units of pi
    for thetatwist in thetavals:
        lpmaster['theta_twist'] = thetatwist
        glat = GyroLattice(lat, lpmaster)
        print 'loading GyroLattice...'
        glat.load()
        glat.ensure_eigval_eigvect(attribute=True, force_hdf5=True)
        eigvals.append(glat.get_eigval())
        # get the localization of these eigenvalues
        loc_half = glat.ensure_edge_localization(attribute=True, force_hdf5=True)
        locz = glat.get_edge_ill()
        topbottom = glat.get_topbottom_edgelocz()
        loczs.append(locz)
        tbs.append(topbottom)

    ###############################################################
    # Plot the spectra as function of thetatwist as scatterplot
    ###############################################################
    fig, ax = leplt.initialize_1panel_centered_fig()
    # fig, ax, cax = leplt.initialize_1panel_cbar_fig()
    lecmaps.register_colormaps()
    cmap = lecmaps.colormap_from_3hexes('greenblackviolet', hex_color0='#55A254', hex_color2='#C042C8')
    vmin = 0.
    ind = 0
    ii = 0
    # Get the spacing between x values
    dval = abs(thetavals[1] - thetavals[0])
    # Prepare to tally to find the largest frequency on the plot
    maxfreq = 0.
    minfreq = 0.
    for val in thetavals:
        ep0 = zip(val * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        ep1 = zip((val + dval) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        lines = [list(a) for a in zip(ep0, ep1)]

        maxfreq = max(maxfreq, np.max(np.imag(eigvals[ii])))
        minfreq = min(minfreq, np.min(np.imag(eigvals[ii])))

        # Define colors based on whether top or bottom
        colors = tbs[ii]
        # print 'colors = ', colors
        # gray out where neither top nor bottom
        colors[np.where(colors == 2)] = 0.5
        # gray out where locz length is above 10 bond lengths long
        colors[np.where(loczs[ii] < 0.1)] = 0.5

        # Define colors based on inverse localization length
        colors = cmap(colors)

        # print 'colors = ', colors
        lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))

        ax.add_collection(lc)
        ii += 1

    ax.set_xlim(0., 2.0)
    ax.set_ylim(minfreq - 0.1, maxfreq + 0.1)
    ax.set_ylabel('frequency, $\omega$')
    ax.set_xlabel(leplt.param2description('thetatwist'))

    # Colormap
    # sm = leplt.empty_scalar_mappable(vmin, vmax_tb, cmap)
    # cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 0.5, 1.])
    # # cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
    # cb.set_label(r'$\xi^{-1}$', labelpad=10, rotation=0, fontsize=8, va='center')

    fname = dio.prepdir(lat.lp['meshfn']) + 'gyro_thetatwistsweep_nthetas' + str(len(thetavals))
    plt.text(0.5, 1.2, r'Spectra for ' + str(lp['NH']) + 'x' + str(lp['NV']) + ' ' + lp['LatticeTop'],
             ha='center', va='center', transform=ax.transAxes)
    print 'saving figure: ' + fname + '.png'
    plt.savefig(fname + 'scatter.png', dpi=300)
    ax.set_ylim(0., maxfreq + 0.1)
    plt.savefig(fname + 'scatter_zeromin.png', dpi=300)
    ax.set_ylim(2., 2.35)
    plt.savefig(fname + 'scatter_gap.png', dpi=300)

    #############################################
    # Plot the spectra as function of thetatwist
    #############################################
    fig, ax, cax = leplt.initialize_1panel_cbar_fig()
    lecmaps.register_colormaps()
    cmap = plt.get_cmap('viridis')
    vmin = 0.
    ind = 0
    ii = 0
    # Get the spacing between x values
    dval = abs(thetavals[1] - thetavals[0])
    # Prepare to tally to find the largest frequency on the plot
    maxfreq = 0.
    minfreq = 0.
    for val in thetavals:
        ep0 = zip((val - dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        ep1 = zip((val + dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        lines = [list(a) for a in zip(ep0, ep1)]

        maxfreq = max(maxfreq, np.max(np.imag(eigvals[ii])))
        minfreq = min(minfreq, np.min(np.imag(eigvals[ii])))

        # Define colors based on inverse localization length
        colors = cmap(loczs[ii] / float(vmax))
        print 'np.shape(colors) = ', np.shape(colors)

        # print 'colors = ', colors
        lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # lc.set_array(colors)
        ax.add_collection(lc)
        ii += 1

    ax.set_xlim(0, 2.0 + dval)
    ax.set_ylim(minfreq - 0.1, maxfreq + 0.1)
    ax.set_ylabel('frequency, $\omega$')
    ax.set_xlabel(leplt.param2description('thetatwist'))

    sm = leplt.empty_scalar_mappable(vmin, vmax, cmap)
    cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 1. / lsize, 2. / lsize])
    cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
    cb.set_label(r'$\xi^{-1}$', labelpad=10, rotation=0, fontsize=8, va='center')

    fname = dio.prepdir(lat.lp['meshfn']) + 'gyro_thetatwistsweep_nthetas' + str(len(thetavals))
    plt.suptitle(r'Spectra for ' + str(lp['NH']) + 'x' + str(lp['NV']) + ' ' + lp['LatticeTop'])
    print 'saving figure: ' + fname + '.png'
    plt.savefig(fname + '.png', dpi=300)
    ax.set_ylim(0., maxfreq + 0.1)
    plt.savefig(fname + '_zeromin.png', dpi=300)
    ax.set_ylim(2., 2.5)
    plt.savefig(fname + '_gap.png', dpi=300)
    plt.close('all')


def twistmodes(lp):
    """Twist one boundary condition of a sample (perhaps periodic strip, perhaps annulus) and plot the modes as they
    adiabatically evolve. Use all modes in the gap.

    Example usage:
    # for amorphous paper
    python gyro_lattice_class.py -LT hyperuniform_annulus -shape annulus -N 60 -alph 0.2 -twistmodes -conf 3
    python gyro_lattice_class.py -LT kagome_hucent_annulus -shape annulus -N 60 -alph 0.2 -twistmodes -conf 3
    python gyro_lattice_class.py -LT hucentroid_annulus -AB 0.8 -shape annulus -N 60 -alph 0.2 -twistmodes -conf 3
    python gyro_lattice_class.py -LT hucentroid_annulus -AB 0.6 -shape annulus -N 60 -alph 0.2 -twistmodes -conf 3

    # others
    python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 5 -NV 5 -twistmodes
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 10 -NP 50 -twistmodes
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 30 -NP 50 -twistmodes
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 20 -NV 17 -NP 20 -twistmodes
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 50 -alph 0.3 -twistmodes -conf 3
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes -conf 3

    # to make more periodicstrips
    python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 20 -NV 17 -NP 20 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 30 -NP 50 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 8 -NV 5 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 5 -NV 5 -skip_polygon -skip_gyroDOS

    # to make annuli
    python ./build/make_lattice.py -LT hyperuniform_annulus -N 60 -alph 0.2 -skip_polygons -skip_gyroDOS -conf 3
    python ./build/make_lattice.py -LT hucentroid_annulus -N 50 -alph 0.3 -skip_polygons -skip_gyroDOS -conf 2
    python ./build/make_lattice.py -LT hucentroid_annulus -N 60 -alph 0.2 -skip_polygons -skip_gyroDOS -conf 3

    Returns
    -------

    """
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lpmaster = copy.deepcopy(lp)
    lat = lattice_class.Lattice(lp)
    lat.load()
    # Set vmax to be 2/L, where L is the full width of the sample in y (the strip width)
    lsize = 2. * np.max(np.abs(lat.xy[:, 1]))
    vmax = 2. / lsize
    vmax_tb = 1.
    eigvals = []
    glats = []
    loczs = []
    tbs = []
    thetavals = np.linspace(0, 2., 41)
    if lp['ABDelta'] == 0:
        if lp['LatticeTop'] == 'hexagonal':
            gapbounds = [1.9, 2.7]
        elif lp['LatticeTop'] in ['hucentroid', 'hucentroid_annulus', 'hyperuniform_annulus']:
            if len(lat.xy) > 1100:
                gapbounds = [2.07, 2.21]
            else:
                gapbounds = [2.045, 2.21]
            pltbounds = [2.0, 2.35]
        elif lp['LatticeTop'] == 'kagome_hucent_annulus':
            gapbounds = [3.47, 3.55]
            pltbounds = [3.4, 3.6]
    else:
        gapbounds = [2.14, 2.21]
        pltbounds = [2.0, 2.35]

    # Note: theta_twist and phi_twist are in units of pi
    kk = 0
    for thetatwist in thetavals:
        lp = copy.deepcopy(lpmaster)
        lp['theta_twist'] = thetatwist
        glat = GyroLattice(lat, lp)
        print 'loading GyroLattice...'
        glat.load()
        glat.ensure_eigval_eigvect(attribute=True, force_hdf5=True)
        eigvalkk = glat.get_eigval()
        eigvals.append(eigvalkk)
        # get the localization of these eigenvalues
        loc_half = glat.ensure_edge_localization(attribute=True, force_hdf5=True)
        locz = glat.get_edge_ill()
        topbottom = glat.get_topbottom_edgelocz()
        # modify so that topbottom values ==2 are mapped to 0.5 (indeterminate)
        topbottom[topbottom == 2] = 0.5
        loczs.append(locz)
        tbs.append(topbottom)
        glats.append(glat)
        if kk == 0:
            inbounds = np.logical_and(np.imag(eigvalkk) > gapbounds[0], np.imag(eigvalkk) < gapbounds[1])
        # topeigv = np.where(np.logical_and(inbounds, topbottom == 0))[0]
        # boteigv = np.where(np.logical_and(inbounds, topbottom == 1))[0]
        kk += 1

    #############################################################################
    # Determine which modes will be strung together
    # For each mode in inbounds at the start, trace out a reasonable path
    # thres is maximum distance in frequency for two modes to be identified, and
    # thres0 is the starting guess for this max
    # numnearby sets the minimum number of states to examine if no nearby states share the topbottom identification
    en_inbounds = np.where(inbounds)[0]
    thres0 = np.mean(np.imag(np.diff(glats[0].eigval[en_inbounds]))) * 0.5
    numnearby = 1
    print 'thres0 = ', thres0
    trace_en = np.zeros((len(en_inbounds), len(glats)), dtype=int)
    trace_ev = np.zeros((len(en_inbounds), len(glats)), dtype=float)
    trace_tb = np.zeros((len(en_inbounds), len(glats)), dtype=float)
    jj = 0
    for en in en_inbounds:
        for ii in range(len(glats)):
            glat = glats[ii]
            if ii == 0:
                enii = en
                ev0 = np.imag(glat.eigval[en])
                # is the mode on top or bottom?
                tb0 = tbs[ii][en]
            else:
                # Could run a loop that gradually makes the threshold larger until at least one eigval is included,
                thres = thres0
                done = False
                while done is False:
                    inrange = np.where(np.logical_and(np.imag(glat.eigval) > (ev0 - thres),
                                                      np.imag(glat.eigval) < (ev0 + thres)))[0]
                    if len(inrange) == 1:
                        enii = inrange
                        done = True
                    else:
                        # Select the eigvect indices whose eigvalues are in range
                        ev_ok = np.imag(glat.eigval[inrange])

                        # If the original eigvector was not localized to one side, redefine tb0 so that
                        # it is the top or bottom of most of the evects in this series so far, so long as
                        # there have been more than one added
                        if tb0 == 0.5 and ii > 1:
                            tb0 = 0.5 * np.round(np.mean(trace_tb[jj][0:ii]) * 2.)

                        try:
                            subset = np.where(tbs[ii][inrange] == tb0)[0]
                        except IndexError:
                            subset = []

                        # if len(subset) > 1:
                        #     # Grab the closest eigval with the same topbottom prescription
                        #     subsetid = np.argmin(np.abs(ev_ok - ev0)[subset])
                        #     enii = inrange[subsetid]
                        #     done = True
                        if len(subset) == 0 and len(inrange) < numnearby:
                            # retry with bigger thres --> increase by 10%
                            done = False
                            thres += thres0 * 0.1
                        elif len(subset) == 1:
                            # There is only one close eigval with same topbottom prescription, use it!
                            enii = inrange[subset][0]
                            done = True
                        else:
                            # There are zero or multiple close eigvals with the same topbottom prescription
                            # If there are any close modes, look at their differences with last mode
                            # and take most similar
                            if len(inrange) > 0:
                                # Find the mode that is most like the previous mode, but only look
                                # at magnitude info for speed
                                # note here enii is the previous glat's eigvect number
                                prevmode = glats[ii - 1].eigvect[enii]
                                diffs = []
                                for thismode in glat.eigvect[inrange]:
                                    # Choose difference to be difference in magnitudes times difference in energies
                                    diff = np.sum((np.abs(thismode) - np.abs(prevmode)) ** 2)
                                    diffs.append(diff)

                                # diffev_factor = max(thres0 * 0.01, np.abs(ev_ok - ev0))
                                difftb_factor = 1. + np.abs(tbs[ii][inrange] - tb0)
                                diffs = np.array(diffs) * difftb_factor
                                enii = inrange[np.argmin(np.array(diffs))]
                                done = True
                            else:
                                # retry with bigger thres --> increase by 10%
                                done = False
                                thres += thres0 * 0.1

            trace_en[jj, ii] = enii
            trace_tb[jj, ii] = tbs[ii][enii]
            ev0 = np.imag(glat.eigval[enii])
            print 'ev0 ->', ev0
            trace_ev[jj, ii] = ev0

        jj += 1

    # Check the connections
    # cmap = lecmaps.colormap_from_3hexes('greenblackviolet', hex_color0='#55A254', hex_color2='#C042C8')
    # jj = 0
    # for trace in trace_ev:
    #     plt.plot(thetavals, trace, '.-')  #, color=trace_tb[jj], vmin=0., vmax=1.)
    #     jj += 1
    #
    # plt.savefig('/Users/npmitchell/Desktop/test.png')
    # sys.exit()

    #########################################################################################
    # Plot the spectra as function of thetatwist as scatterplot along with eigvect plots
    # First draw all normal modes in the gap
    #########################################################################################
    fontsize = 20
    leplt.set_fontsizes(sizes=(12, 14, 16))
    fig = plt.figure(figsize=(16. * 7.6 / 16., 9. * 7.6 / 16.))
    fig, ax = leplt.initialize_nxmpanel_fig(1, 2, Wfig=90, x0frac=0., y0frac=0.07, hspace=30, wsfrac=0.4, fig=fig,
                                            fontsize=fontsize)
    lecmaps.register_colormaps()
    cmap = lecmaps.colormap_from_3hexes('greenblackviolet', hex_color0='#55A254', hex_color2='#C042C8')
    vmin = 0.
    vmax = 1.
    ii = 0
    # Get the spacing between x values
    dval = abs(thetavals[1] - thetavals[0])
    # Prepare to tally to find the largest frequency on the plot
    maxfreq = 0.
    minfreq = 0.
    for val in thetavals:
        ep0 = zip((val - dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        ep1 = zip((val + dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        lines = [list(a) for a in zip(ep0, ep1)]

        maxfreq = max(maxfreq, np.max(np.imag(eigvals[ii])))
        minfreq = min(minfreq, np.min(np.imag(eigvals[ii])))

        # Define colors based on whether top or bottom
        colors = tbs[ii]
        # gray out where neither top nor bottom -- this should already have been done
        colors[np.where(colors == 2)] = 0.5
        # gray out where locz length is above 10 bond lengths long
        colors[np.where(loczs[ii] < 0.1)] = 0.5

        # Define colors based on inverse localization length
        colors = cmap(colors)

        # print 'colors = ', colors
        lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))

        ax[1].add_collection(lc)
        ii += 1

    ax[1].set_xlim(0., 2.0)
    ax[1].set_ylim(pltbounds[0], pltbounds[1])
    ax[1].set_ylabel('frequency, $\omega/\Omega_g$')
    ax[1].set_xlabel(leplt.param2description('thetatwist'))
    ax[1].text(0.5, 1.1, 'adiabatic pumping', transform=ax[1].transAxes, va='center', ha='center')
    ax[0].axis('off')

    # prepare outname
    fdir = dio.prepdir(glat.lp['meshfn']) + 'gyro_thetatwistsweep_nthetas' + str(len(thetavals)) + '/'
    compiledir = fdir + 'thetatwist_compile/'
    compilefn = compiledir + 'thetatwist_'
    dio.ensure_dir(compiledir)

    # Now that the spectrum has been drawn, draw each mode on the left axis
    todo = np.where(inbounds)[0]
    jj = 0
    for kk in todo:
        subdir = fdir + 'thetatwist_eigvect{0:05d}'.format(kk)
        fname = subdir + '/thetatwist_eigvect{0:05d}_'.format(kk)
        dio.ensure_dir(fname)
        previous_ev = None

        # Use trace_en to decide which state this first one goes to
        for ii in np.hstack((np.array([0]), np.arange(len(glats)))):
            enii = trace_en[jj][ii]
            print 'plotting frame ', ii, '/', len(glats)

            first = (jj == 0 and ii == 0)
            # Draw the excitation
            # first get the angle by which to rotate the initial condition
            if previous_ev is not None:
                realxy = np.real(previous_ev)
                theta = phase_minimizes_difference_complex_displacments(glats[ii].eigvect[enii], realxy)
                print 'theta = ', theta
            else:
                theta = 0.

            fig, ax_tmp, [scat_fg, pp, f_mark, lines12_st] = \
                glats[ii].plot_eigvect_excitation(enii, eigval=None, eigvect=None, ax=ax[0], plot_lat=first,
                                                  theta=theta, normalization=3.)  # color=lecmaps.blue())
            # Mark the eigenvalue plotted
            # print 'ii = ', ii
            # print "glats[ii].lp['theta_twist'] =", glats[ii].lp['theta_twist']
            # print 'glats[ii].eigval[enii] = ', glats[ii].eigval[enii]
            redmark = ax[1].plot(glats[ii].lp['theta_twist'], np.imag(glats[ii].eigval[enii]),
                                 'o', markerfacecolor='none', markeredgecolor=lecmaps.red())

            # save this figure
            ax[0].set_title('')
            plt.savefig(fname + '{0:05d}'.format(ii) + '.png', dpi=300)
            plt.savefig(compilefn + '{0:05d}'.format(jj * len(glats) + ii) + '.png', dpi=300)

            # Store this current eigvector as 'previous_ev'
            previous_ev = glats[ii].eigvect[enii] * np.exp(1j * theta)

            # Remove excitation from figure
            for handle in [scat_fg, pp, f_mark, lines12_st]:
                if handle:
                    handle.remove()

            redmark.pop(0).remove()

        imgname = fname
        movname = subdir + '.mov'
        lemov.make_movie(imgname, movname, indexsz='05', framerate=12, imgdir=dio.prepdir(subdir), rm_images=True,
                         save_into_subdir=True)
        jj += 1

    # Make compile movie
    imgname = compilefn
    movname = fdir + 'thetatwist_sweep.mov'
    lemov.make_movie(imgname, movname, indexsz='05', framerate=12, imgdir=dio.prepdir(compiledir), rm_images=True,
                     save_into_subdir=True)

    plt.close('all')


def twistmodes_spiral(lp, nrungs=4, startfreq=2.1, springsubplot=False, thres0=None, numnearby=1, ylims=None,
                      nthetavals=41, normalization=None):
    """Twist one boundary condition of a sample (perhaps periodic strip, perhaps annulus) and plot the modes as they
    adiabatically evolve.
    Use args.twistmode_startfreq --> 'startfreq' is for initial guess for which mode moves up.

    Parameters
    ----------
    lp : dict
        the lattice parameter dictionary to use for finding and loading the GyroLattie
    nrungs : int
        How many times to wind around 2pi for the twisted bc
    startfreq : float
        guess for which mode frequency to start climbing/descending
    springsubplot : bool
        Whether to include a subplot showing the moving boundary twisted spring condition

    Example usage:
    for paper:
    python gyro_lattice_class.py -LT hyperuniform_annulus -shape annulus -N 60 -alph 0.2 -twistmodes
            # -startfreq 2.082 -conf 3 -thres0 0.0032 -nrungs 22 -springax
    python gyro_lattice_class.py -LT kagome_hucent_annulus -shape annulus -N 60 -alph 0.2 -twistmodes -startfreq 2.082 -conf 3 -thres0 0.0032 -nrungs 22 -springax
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.082 -conf 3 -thres0 0.0032 -nrungs 22 -springax
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.082 -conf 3 -thres0 0.0032 -nrungs 22 -springax
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.23 -nrungs 19 -conf 3 -thres0 0.005 -springax
    python gyro_lattice_class.py -LT randorg_gammakick0p50_cent -shape square -periodic_strip -NH 20 -NV 50 -twistmodes_spiral -spreading_time 0.3 -nrungs 5 -springax -conf 3
    python gyro_lattice_class.py -LT randorg_gammakick0p50_cent -shape square -periodic_strip -NH 15 -NV 35 -twistmodes -spreading_time 0.3
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 40 -alph 0.25 -twistmodes_spiral -startfreq 2.28 -conf 1 -AB 0.6
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.25 -conf 3 -AB 0.6


    no spring axis inset
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.082 -conf 3
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.23 -nrungs 19 -conf 3 -thres0 0.005

    not as great:
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 30 -NP 50 -twistmodes_spiral -conf 3
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 75 -alph 0.15 -twistmodes_spiral -startfreq 2.07
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -startfreq 2.07

    python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 5 -NV 5 -twistmodes_spiral -startfreq 2.07 -springax -nrungs 1
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 10 -NP 50 -twistmodes_spiral -startfreq 2.07
    python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 20 -NV 17 -NP 20 -twistmodes_spiral -startfreq 2.07
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 50 -alph 0.3 -twistmodes_spiral -conf 3 -startfreq 2.07
    python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 60 -alph 0.2 -twistmodes_spiral -conf 3 -startfreq 2.07

    # to make more periodicstrips
    python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 20 -NV 17 -NP 20 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 30 -NP 50 -skip_polygon -skip_gyroDOS -conf 3
    python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 8 -NV 5 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 5 -NV 5 -skip_polygon -skip_gyroDOS
    python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 1 -NV 5 -skip_polygon -skip_gyroDOS

    # to make annuli
    python ./build/make_lattice.py -LT hucentroid_annulus -N 50 -alph 0.3 -skip_polygons -skip_gyroDOS -conf 2
    python ./build/make_lattice.py -LT hucentroid_annulus -N 60 -alph 0.2 -skip_polygons -skip_gyroDOS -conf 3
    python ./build/make_lattice.py -LT kagome_hucent_annulus -N 30 -alph 0.2 -skip_polygons -skip_gyroDOS -conf 1
    python ./build/make_lattice.py -LT kagome_hucent_annulus -N 60 -alph 0.2 -skip_polygons -skip_gyroDOS -conf 3
    python ./build/make_lattice.py -LT hucentroid_annulus -N 75 -alph 0.15 -skip_polygons -skip_gyroDOS -conf 3

    # to make randorg_gamma0p50
    python ./build/pointsets/gammakick.py -make_pointset -gamma 0.5 -NH 15 -NV 35
    python ./build/make_lattice.py -LT randorg_gammakick0p50_cent -shape square -spreadt 0.3 -NP 1000 -NH 20 -NV 50 \
        -skip_gyroDOS -skip_polygons -conf 1 -periodic_strip
    python ./build/make_lattice.py -LT randorg_gammakick0p50_cent -shape square -spreadt 0.3 -NP 525 -NH 15 -NV 35 \
        -skip_gyroDOS -skip_polygons -conf 1 -periodic_strip


    Returns
    -------

    """
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lpmaster = copy.deepcopy(lp)
    lat = lattice_class.Lattice(lp)
    lat.load()
    print 'glatclass_scripts_twistbc: lat.PV = ', lat.PV

    # Set vmax to be 2/L, where L is the full width of the sample in y (the strip width)
    lsize = 2. * np.max(np.abs(lat.xy[:, 1]))
    vmax = 2. / lsize
    vmax_tb = 1.
    eigvals = []
    glats = []
    loczs = []
    tbs = []
    thetavals = np.linspace(0, 2., nthetavals)
    if normalization is None:
        normalization = max(1., float(len(lat.xy)) / 300.)

    # Note: theta_twist and phi_twist are in units of pi
    kk = 0
    for thetatwist in thetavals:
        lp = copy.deepcopy(lpmaster)
        lp['theta_twist'] = thetatwist
        glat = GyroLattice(lat, lp)

        print 'loading GyroLattice...'
        glat.load()
        print 'glatclass_scripts_twistbc: glat.PVx = ', glat.lattice.PVx
        print 'glatclass_scripts_twistbc: glat.PV = ', glat.lattice.PV
        print '-1-'
        print 'glat.Omg = ', glat.Omg
        print '-1-'

        glat.ensure_eigval_eigvect(attribute=True, force_hdf5=True)
        eigvalkk = glat.get_eigval()
        eigvals.append(eigvalkk)
        # get the localization of these eigenvalues
        loc_half = glat.ensure_edge_localization(attribute=True, force_hdf5=True)
        locz = glat.get_edge_ill()

        # Use the robust center of mass measurement for whether a state is weighted towards a boundary or not instead
        # of fitting to exponential decay
        # topbottom = glat.get_topbottom_edgelocz()
        topbottom = glat.get_topbottom_com()

        print 'glatclass_scripts_twistbc: glat.PVx = ', glat.lattice.PVx
        print 'glatclass_scripts_twistbc: glat.PV = ', glat.lattice.PV
        print '--'

        # modify so that topbottom values ==2 are mapped to 0.5 (indeterminate) (should no longer be necessary if
        # we use com).
        topbottom[topbottom == 2] = 0.5
        loczs.append(locz)
        tbs.append(topbottom)
        glats.append(glat)
        # topeigv = np.where(np.logical_and(inbounds, topbottom == 0))[0]
        # boteigv = np.where(np.logical_and(inbounds, topbottom == 1))[0]
        kk += 1

    #############################################################################
    # Determine which modes will be strung together. In this script option, we spiral along one (periodically)
    # continuous set of states, for example along all states confined to the top of a periodic strip.
    # thres is maximum distance in frequency for two modes to be identified, and
    # thres0 is the starting guess for this max
    # numnearby sets the minimum number of states to examine if no nearby states share the topbottom identification
    # Get nearest eigval to the supplied startfreq
    en = np.argmin(np.abs(np.imag(glats[0].eigval) - startfreq))
    enfirst = copy.deepcopy(en)
    if thres0 is None:
        thres0 = np.mean(np.imag(np.diff(glats[0].eigval[en - 7: en + 7])))

    print 'thres0 = ', thres0
    trace_en = np.zeros(len(glats) * nrungs, dtype=int)
    trace_ev = np.zeros(len(glats) * nrungs, dtype=float)
    trace_tb = np.zeros(len(glats) * nrungs, dtype=float)
    for ii in range(len(glats) * nrungs):
        glat = glats[ii % len(glats)]
        tbii = tbs[ii % len(glats)]
        if ii == 0:
            enii = en
            ev0 = np.imag(glat.eigval[en])
            # is the mode on top or bottom?
            tb0 = tbs[ii][en]
        else:
            # Run a loop that gradually makes the threshold larger until at least numnearby eigvals are within range
            thres = thres0
            done = False
            if numnearby == 0:
                # Keep enii same as previous round
                done = True

            while done is False:
                inrange = np.where(np.logical_and(np.imag(glat.eigval) > (ev0 - thres),
                                                  np.imag(glat.eigval) < (ev0 + thres)))[0]
                if len(inrange) == 1:
                    enii = inrange
                    done = True
                else:
                    # Select the eigvect indices whose eigvalues are in range
                    ev_ok = np.imag(glat.eigval[inrange])

                    # If the original eigvector was not localized to one side, redefine tb0 so that
                    # it is the top or bottom of most of the evects in this series so far, so long as
                    # there have been more than one added
                    if tb0 == 0.5 and ii > 1:
                        tb0 = 0.5 * np.round(np.mean(trace_tb[0:ii]) * 2.)

                    subset = np.where(tbii[inrange] == tb0)[0]

                    # if len(subset) > 1:
                    #     # Grab the closest eigval with the same topbottom prescription
                    #     subsetid = np.argmin(np.abs(ev_ok - ev0)[subset])
                    #     enii = inrange[subsetid]
                    #     done = True
                    if len(subset) == 0 and len(inrange) < numnearby:
                        # retry with bigger thres --> increase by 10%
                        done = False
                        thres += thres0 * 0.1
                    elif len(subset) == 1:
                        # There is only one close eigval with same topbottom prescription, use it!
                        enii = inrange[subset][0]
                        done = True
                    else:
                        # There are zero or multiple close eigvals with the same topbottom prescription
                        # If there are any close modes, look at their differences with last mode
                        # and take most similar
                        if len(inrange) > 0:
                            # Find the mode that is most like the previous mode, but only look
                            # at magnitude info for speed
                            # note here enii is the previous glat's eigvect number
                            prevmode = glats[(ii % len(glats)) - 1].eigvect[enii]
                            diffs = []
                            for thismode in glat.eigvect[inrange]:
                                # Choose difference to be difference in magnitudes times difference in energies
                                diff = np.sum((np.abs(thismode) - np.abs(prevmode)) ** 2)
                                diffs.append(diff)

                            # diffev_factor = max(thres0 * 0.01, np.abs(ev_ok - ev0))
                            difftb_factor = 1. + np.abs(tbii[inrange] - tb0)
                            diffs = np.array(diffs) * difftb_factor
                            enii = inrange[np.argmin(np.array(diffs))]
                            done = True
                        else:
                            # retry with bigger thres --> increase by 10%
                            done = False
                            thres += thres0 * 0.1

        trace_en[ii] = enii
        trace_tb[ii] = tbii[enii]
        ev0 = np.imag(glat.eigval[enii])
        print 'ev0 ->', ev0
        trace_ev[ii] = ev0

    # Check the connections
    # cmap = lecmaps.colormap_from_3hexes('greenblackviolet', hex_color0='#55A254', hex_color2='#C042C8')
    # thetas = thetavals.tolist() * nrungs
    # plt.plot(thetas, trace_ev, '.-')  #, color=trace_tb[jj], vmin=0., vmax=1.)
    # plt.savefig('/Users/npmitchell/Desktop/test.png')

    #########################################################################################
    # Plot the spectra as function of thetatwist as scatterplot along with eigvect plots
    # First draw all normal modes in the gap
    #########################################################################################
    fontsize = 20
    leplt.set_fontsizes(sizes=(12, 14, 16))
    fig = plt.figure(figsize=(16. * 7.6 / 16., 9. * 7.6 / 16.))
    if springsubplot:
        wfig = 90
        x0frac = 0.
        y0frac = 0.07
        hspace = 65
        wsfrac = 0.35
        hsfrac = 0.4
        vspace = 8
        hspace = 67
        tspace = 10

        # get supplied figure size and convert to mm
        size = fig.get_size_inches() * 25.4
        Wfig, Hfig = size[0], size[1]
        x0, y0, ws = Wfig * x0frac, Wfig * y0frac, Wfig * wsfrac
        hs = hsfrac * Wfig

        label_params = dict(size=fontsize, fontweight='normal')
        eax = sps.axes_in_mm(x0 + (ws + hspace) * 0, y0, ws, hs, label='', label_params=label_params)
        sax = sps.axes_in_mm(x0 + (ws + hspace) * 1, y0, ws / 1.33333, hs, label='', label_params=label_params)
        ws = 45
        hs = 45
        sprax = sps.axes_in_mm(wfig - ws * 0.5, 50, ws, hs, fig=fig)
    else:
        fig, ax = leplt.initialize_nxmpanel_fig(1, 2, Wfig=90, x0frac=0., y0frac=0.07, hspace=30, wsfrac=0.4, fig=fig,
                                                fontsize=fontsize)
        eax = ax[0]
        sax = ax[1]

    lecmaps.register_colormaps()
    cmap = lecmaps.colormap_from_3hexes('greenblackviolet', hex_color0='#55A254', hex_color2='#C042C8')
    vmin = 0.
    vmax = 1.
    ii = 0
    # Get the spacing between x values
    dval = abs(thetavals[1] - thetavals[0])
    # Prepare to tally to find the largest frequency on the plot
    maxfreq = 0.
    minfreq = 0.
    for val in thetavals:
        ep0 = zip((val - dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        ep1 = zip((val + dval * 0.5) * np.ones(len(eigvals[ii])), np.imag(eigvals[ii]))
        lines = [list(a) for a in zip(ep0, ep1)]

        maxfreq = max(maxfreq, np.max(np.imag(eigvals[ii])))
        minfreq = min(minfreq, np.min(np.imag(eigvals[ii])))

        # Define colors based on whether top or bottom
        colors = tbs[ii]
        # gray out where neither top nor bottom -- this should already have been done
        colors[np.where(colors == 2)] = 0.5
        # gray out where locz length is above 10 bond lengths long
        # raise RuntimeError('This isnt looking right')
        colors[np.where(loczs[ii] < 0.08)] = 0.5

        # Define colors based on inverse localization length
        colors = cmap(colors)

        # print 'colors = ', colors
        lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))

        sax.add_collection(lc)
        ii += 1

    sax.set_xlim(0., 2.0)
    if ylims is not None:
        sax.set_ylim(ylims[0], ylims[1])
    else:
        sax.set_ylim(0, maxfreq * 1.1)

    sax.set_ylabel('frequency, $\omega/\Omega_g$')
    sax.set_xlabel(leplt.param2description('thetatwist'))
    sax.text(0.5, 1.1, 'adiabatic pumping', transform=sax.transAxes, va='center', ha='center')
    eax.axis('off')

    # prepare outname
    fdir = dio.prepdir(glat.lp['meshfn']) + 'gyro_thetatwistsweep_nthetas' + str(len(thetavals)) + '/'
    if springsubplot:
        spraxstr = '_withspringax'
    else:
        spraxstr = ''
    compiledir = fdir + 'thetatwist_spiral_enstart{0:05d}'.format(enfirst) + spraxstr + '/'
    compilefn = compiledir + 'thetatwist_enstart{0:05d}'.format(enfirst) + '_'
    dio.ensure_dir(compiledir)

    # Now that the spectrum has been drawn, draw each mode on the left axis
    previous_ev = None
    ii = 0
    first = True
    todo = np.hstack((np.array([0]), np.arange(len(trace_en))[::2]))
    for kk in todo:
        glat = glats[kk % len(glats)]

        # Use trace_en to decide which state this comes from
        enii = trace_en[kk]
        print 'plotting frame ', kk, '/', len(trace_en)

        # Draw the excitation
        # first get the angle by which to rotate the initial condition
        if previous_ev is not None:
            realxy = np.real(previous_ev)
            theta = phase_minimizes_difference_complex_displacments(glat.eigvect[enii], realxy)
            print 'theta = ', theta
        else:
            theta = 0.

        fig, ax_tmp, [scat_fg, pp, f_mark, lines12_st] = \
            glat.plot_eigvect_excitation(enii, eigval=None, eigvect=None, ax=eax, plot_lat=first,
                                         theta=theta, normalization=normalization)  # color=lecmaps.blue())

        if first:
            # xlim_eax = eax.get_xlim()
            ylim_eax = eax.get_ylim()
            # eax.set_xlim(xlim_eax[0] - 1, xlim_eax[1] + 1)
            eax.set_ylim(ylim_eax[0] - 1, ylim_eax[1] + 1)
            eax.axis('scaled')

        # Mark the eigenvalue plotted
        redmark = sax.plot(glat.lp['theta_twist'], np.imag(glat.eigval[enii]),
                           'o', markerfacecolor='none', markeredgecolor=lecmaps.red())

        # Add the picture of the spring boundary condition to sprax
        if springsubplot:
            # Set line width for border and cut
            lwcut = 1.5
            # get index of mechanical spring bc image to load
            # there are 21 images in the stack from 0 to 2* np.pi
            thetav = np.linspace(0, 2., 21)
            spr_dir = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_disorder_paper/figure_drafts/twistbc_movie/' + \
                      'frames_automatorcolor/'
            spr_ims = sorted(glob.glob(spr_dir + '*.png'))
            # print 'spr_ims = ', spr_ims
            match = np.argmin(np.abs(thetav - glat.lp['theta_twist']))
            # print 'glattwistscripts: match = ', match
            # print 'glattwistscripts: spr_ims[match] = ', spr_ims[match]
            im = pylab.imread(spr_ims[match], 'png')
            sprax.imshow(im)
            sprax.axis('scaled')
            sprax.set_xlim(450, 1100)
            sprax.set_ylim(800, 120)
            for side in ['bottom', 'top', 'right', 'left']:
                sprax.spines[side].set_color(lecmaps.light_blue())
                sprax.spines[side].set_linewidth(lwcut)

            sprax.xaxis.set_ticks([])
            sprax.yaxis.set_ticks([])

            if glat.lp['shape'] == 'annulus' and first:
                # get minimum distance from center
                distmin = np.min(np.sqrt(lat.xy[:, 0] ** 2 + lat.xy[:, 1] ** 2))
                dmin = max(0, distmin * 0.5)
                dmax = np.max(np.sqrt(lat.xy[:, 0] ** 2 + lat.xy[:, 1] ** 2)) + dmin
                eax.plot([dmin, dmax], [0, 0], '--', color=lecmaps.light_blue(), linewidth=lwcut)
                eax.set_xlim(right=dmax)

        # save this figure
        eax.set_title('')
        plt.savefig(compilefn + '{0:05d}'.format(ii) + '.png', dpi=300)

        # Store this current eigvector as 'previous_ev'
        previous_ev = glat.eigvect[enii] * np.exp(1j * theta)

        # Remove excitation from figure
        for handle in [scat_fg, pp, f_mark, lines12_st]:
            if handle:
                handle.remove()

        redmark.pop(0).remove()

        if springsubplot:
            sprax.cla()

        if not first:
            ii += 1
        else:
            first = False

    # Make compile movie
    imgname = compilefn
    movname = fdir + 'thetatwist_spiral_enstart{0:05d}'.format(enfirst) + '.mov'
    lemov.make_movie(imgname, movname, indexsz='05', framerate=12, imgdir=dio.prepdir(compiledir), rm_images=False,
                     save_into_subdir=True)

    plt.close('all')


if __name__ == '__main__':
    import pdb
    spring_xy_list = False
    manual_movie = True
    manual_movie_rename = False

    if spring_xy_list:
        # Make a list of angles swept out by a spring on the twisted bc
        # define radius of the circle swept out in units of the spring length
        outdir = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_disorder_paper/figure_drafts/iDraw_figs/twistbc/'
        theta = np.linspace(0, 2. * np.pi, 21)
        rad = 1.537 / 5.185
        scfactor = 5.185 - rad
        print '1 + rad = ', 1 + rad
        # the center of the circle is
        xy = np.array([[rad * np.cos(th) + 1., rad * np.sin(th)] for th in theta])
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        plt.savefig(outdir + 'points.png')
        # scale in units of (1 + rad)
        scale = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
        scale /= (1. + rad)
        # pdb.set_trace()
        # rot is the angle of rotation of the spring
        rot = np.round((np.array([np.arctan2(pt[1], pt[0]) for pt in xy])) * 180. / np.pi + 188.)
        # displacement for the ball
        dxy = (xy - xy[0]) * scfactor
        # save as a text file
        print np.shape(scale)
        print np.shape(rot)
        print np.shape(dxy)
        ints = np.arange(1, 42, 2)
        mm = np.dstack((ints, scale, rot, dxy[:, 0], dxy[:, 1]))[0]
        np.savetxt(outdir + 'twist_springrender_params.txt', mm, header='scale, rotation angle [deg]', fmt='%0.3f',
                   delimiter=',')

    if manual_movie:
        # Fix a gyro movie
        dir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hucentroid_annulus/' + \
              'hucentroid_annulus_d03_alph0p20_000060_x_000060/gyro_thetatwistsweep_nthetas41/'
        ims = sorted(glob.glob(dir + 'manual_movie2/*.png'))
        ii = 0
        for im in ims:
            if im.split('.png')[0][-5:] != '{0:05d}'.format(ii):
                newname = im[0:-9] + '{0:05d}'.format(ii) + '.png'
                print 'moving ', im, ' to ', newname
                subprocess.call(['mv', im, newname])
            ii += 1

        imgname = im[0:-9]
        movname = dir + 'thetatwist_spiral_skim.mov'
        lemov.make_movie(imgname, movname, framerate=10)

    if manual_movie_rename:
        dir = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_disorder_paper/figure_drafts/twistbc_movie/' + \
              'frames_automatorcolor/'
        ims = sorted(glob.glob(dir + '*.png'))

        ii = 0
        for im in ims:
            basename = im.split('.png')[0].split('/')[-1]
            rootname = im.split(basename)[0]
            new_basename = 'twist_spring_{0:05d}'.format(ii)
            if basename != new_basename:
                newname = rootname + new_basename + '.png'
                print 'moving ', im, ' to ', newname
                subprocess.call(['mv', im, newname])
            ii += 1

        imgname = rootname + new_basename
        movname = dir + 'twist_spring.mov'
        lemov.make_movie(imgname, movname, framerate=9)

