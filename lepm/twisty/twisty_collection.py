import numpy as np
import lepm.lattice_elasticity as le
import lepm.lattice_collection as lattice_collection
import lepm.lattice_class as lattice_class
import lepm.dataio as dio
import lepm.twisty.twisty_lattice as twisty_lattice
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import lepm.twisty.plotting.twisty_collection_plotting_functions as tcpfns
import matplotlib.pyplot as plt
import argparse
import os
import glob
import sys
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

'''
Description
===========
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Example usage:
import lepm.lattice_collection as latcoll
lc = latcoll.lattice_collection()
lc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/iscentroid/iscentroid_square_hexner_size64_conf*_NP000064')
twisty_collection()

'''


class TwistyCollection:
    """Create a collection of twisty spring lattices. Lattices can exist in memory, on hard disk, or both.
    
    Attributes
    ----------
    self.twisty_lattices : list of instances of twisty_lattice.TwistyLattice() corresponding to each network
    self.physics : dict, containing dos_collection, chern_collection
    """
    def __init__(self):
        """Create an instance of a lattice_collection."""
        self.twisty_lattices = []
        self.meshfns = []

    def add_meshfn(self, meshfn):
        """Add one or more lattice file paths to self.meshfns.

        Parameters
        ----------
        meshfn : string or list of strings
            The paths of lattices, to add to self.meshfns
        """
        if isinstance(meshfn, list):
            for fn in meshfn:
                ind = 0
                # check that it exists or get list of matching lattices
                fnglob = sorted(glob.glob(fn))
                print 'fnglob = ', fnglob
                is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
                addfn = [ fnglob[ind] for ind in is_a_dir]
                if len(addfn) > 1:
                    for eachfn in addfn:
                        self.meshfns.append(eachfn)
                        ind += 1
                else:
                    self.meshfns.append(addfn)
                    ind += 1
                print 'Added ' + str(ind) + ' lattice filenames to lattice collection'
        elif isinstance(meshfn, str):
            fnglob = sorted(glob.glob(meshfn))
            print 'meshfn = ', meshfn
            print 'fnglob = ', fnglob
            is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
            # print 'is_a_dir = ', is_a_dir
            addfn = [fnglob[ind] for ind in is_a_dir]
            if np.size(is_a_dir) > 1:
                for eachfn in addfn:
                    self.meshfns.append(eachfn)
                print 'Added ' + str(len(addfn)) + ' lattice filenames to lattice collection'
            else:
                self.meshfns.append(addfn)
                print 'Added lattice filename to lattice collection: ', addfn
        else:
            print RuntimeError(
                'Argument to lattice_collection instance method add_meshfns() must be string or list of strings.')
        return self.meshfns

    def add_twisty_lattice(self, tlat):
        """Add twisty_lattice instance and its meshfn to current collection"""
        self.twisty_lattices.append(tlat)
        self.meshfns.append(tlat.lp['meshfn'])

    def get_meshfns(self):
        try:
            return self.meshfns
        except NameError:
            # Load meshfns from each lattice in the collection
            meshfns = []
            for tlat in lattice_collection.twisty_lattices:
                try:
                    meshfns.append(tlat.lp['meshfn'])
                except NameError:
                    meshfns.append('in_memory')

            self.meshfns = meshfns
            return self.meshfns

    def get_localizations(self, save=False, save_eigvect_eigval=False):
        """Return list of all tlat.localization arrays"""
        localizations = []
        for tlat in self.twisty_lattices:
            localizations.append(tlat.get_localization(save=save, save_eigvect_eigval=save_eigvect_eigval))

        return localizations

    def get_eigvals(self):
        """Get all eigval arrays for all tlats"""
        eigvals = []
        for tlat in self.twisty_lattices:
            eigvals.append(tlat.get_eigval())

        return eigvals

    def load_twisty_lattices(self, load_DOS=True):
        # Use ensure_all_twisty_lattices() instead
        # for meshfn in self.meshfns:
        #     lat = lattice_class.lattice()
        #     lat.load(meshfn=meshfn)
        #     ##tlat.load()
        #     !!!!!!! CONTINUE HERE !!!!!
        #     todo
        #     self.twisty_lattices.append()
        pass

    def load_eigvals(self, save_eigvals=False):
        """

        Returns
        -------

        """
        for ii in range(len(self.meshfns)):
            self.ensure_all_twisty_lattices()
            eigval = self.twisty_lattices[ii].load_eigval(attribute=True)
            if eigval is None:
                self.twisty_lattices[ii].calc_eigvals(attribute=True)
                if save_eigvals:
                    self.twisty_lattices[ii].save_eigvals()

    def load_eigvects(self):
        """Ensure that for every twisty_lattice there is an eigvect array loaded"""
        for ii in range(len(self.meshfns)):
            self.ensure_all_twisty_lattices()
            self.twisty_lattices[ii].load_eigvect(attribute=True)

    def calc_max_polygon_sides(self):
        maxpno = 0
        for ii in range(len(self.twisty_lattices)):
            if self.twisty_lattices[ii].lattice.polygons is not None:
                polygons = self.twisty_lattices[ii].lattice.polygons
            else:
                polygons = self.twisty_lattices[ii].lattice.load_polygons()

            # print 'polygons = ', polygons

            # number of polygon sides
            Pno = np.array([len(polyg) - 1 for polyg in polygons], dtype=int)
            maxpno = max(maxpno, np.max(Pno))
            print 'maxpno = ', maxpno

        return maxpno

    def ensure_all_twisty_lattices(self):
        """Ensure that all twisty lattices called for by self.meshfns are loaded.
        """
        # todo: add the ability to change tlat.lp parameters (like Omk) at function call-- specify as list of lp's or as single lp
        print 'Ensuring all twisty lattices...'
        for ii in range(len(self.meshfns)):
            meshfn = self.meshfns[ii]
            if isinstance(meshfn, list):
                meshfn = meshfn[0]
            print 'Ensuring ', meshfn
            try:
                self.twisty_lattices[ii]
                append_tlat = False
            except IndexError:
                append_tlat = True

            try:
                # Check if lp['meshfn'] of twisty_lattice matches meshfn
                self.twisty_lattices[ii].lp['meshfn']
                # print('Already have twisty_lattice defined for ii='+str(ii)+': ', test)
            except IndexError:
                lp = le.load_params(dio.prepdir(meshfn), 'lattice_params')
                lat = lattice_class.Lattice(lp=lp)
                lat.load(meshfn=meshfn)
                lp['Omk'] = -1.0
                lp['Omg'] = -1.0
                if append_tlat:
                    self.twisty_lattices.append(twisty_lattice_class.TwistyLattice(lat, lp))
                else:
                    self.twisty_lattices[ii] = twisty_lattice_class.TwistyLattice(lat, lp)

                tlat = twisty_lattice_class.TwistyLattice(lat, self.twisty_lattices[ii].lp)
                self.twisty_lattices[ii] = tlat

    def ensure_all_eigval(self, save_eigval=False):
        """Ensure eigenval is attribute for each twisty_lattice in twisty_collection"""
        self.ensure_all_twisty_lattices()

        # Make sure all eigvals are accounted for
        test = self.twisty_lattices[0].eigval is not None
        for ii in np.arange(1, len(self.twisty_lattices)):
            test = self.twisty_lattices[ii].eigval is not None and test
        print 'All twisty_lattices have eigval attribute, all set to overlay DOS...'
        if not test:
            print 'Some twisty_lattices do not already have eigval as an attribute, loading those now...'
            self.load_eigvals(save_eigvals=save_eigval)

    def ensure_all_ill_saved(self, attribute=False):
        self.ensure_all_twisty_lattices()
        for ii in range(len(self.twisty_lattices)):
            tlat = self.twisty_lattices[ii]
            if not glob.glob(tlat.lp['meshfn'] + '/localization' + tlat.lp['meshfn_exten'] + '.txt'):
                print 'saving localization ', ii, ' of ', len(self.twisty_lattices), '...'
                tlat.save_localization(attribute=attribute)
        print 'All networks have localization saved.'

    def ensure_all_ipr_saved(self):
        self.ensure_all_twisty_lattices()
        for ii in range(len(self.twisty_lattices)):
            tlat = self.twisty_lattices[ii]
            if not glob.glob(tlat.lp['meshfn'] + '/ipr' + tlat.lp['meshfn_exten'] + '.pkl'):
                print 'saving ipr ', ii, ' of ', len(self.twisty_lattices), '...'
                tlat.save_ipr(attribute=False)
        print 'All networks have ipr saved.'

    def ensure_all_polygons_saved(self):
        self.ensure_all_twisty_lattices()
        for ii in range(len(self.twisty_lattices)):
            tlat = self.twisty_lattices[ii]
            if not glob.glob(tlat.lp['meshfn'] + '/polygons.pkl'):
                print 'saving polygons ', ii, ' of ', len(self.twisty_lattices), '...'
                tlat.lattice.save_polygons(attribute=False)
        print 'All networks have prpoly saved.'

    def ensure_all_prpoly_saved(self):
        self.ensure_all_twisty_lattices()
        for ii in range(len(self.twisty_lattices)):
            tlat = self.twisty_lattices[ii]
            if not glob.glob(tlat.lp['meshfn'] + '/prpoly' + tlat.lp['meshfn_exten'] + '.pkl'):
                print 'saving prpoly ', ii, ' of ', len(self.twisty_lattices), '...'
                tlat.save_prpoly(attribute=False)
                plt.close('all')
            # check that pickle file isn't empty
            # else:
            #     filename = glob.glob(tlat.lp['meshfn'] + '/prpoly' + tlat.lp['meshfn_exten'] + '.pkl')[0]
            #     with open(filename, "rb") as fn:
            #         prpoly = pickle.load(fn)
            #     if prpoly is None:
            #         print 'loaded prpoly = ', prpoly
            #         sys.exit()
        print 'All networks have prpoly saved.'

    def plot_DOS_overlay(self, outdir=None, title=r'$D(\omega)$ for twisty networks', fname='DOS_overlay.png',
                         alpha=None, FSFS=12, close=True):
        """
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        """
        self.ensure_all_twisty_lattices()
        self.ensure_all_eigval()

        # Decide what alpha value to use if not provided (here, 1/N)
        if alpha is None:
            alpha = 1./float(len(self.twisty_lattices))

        for ii in range(len(self.twisty_lattices)):
            eigval = self.twisty_lattices[ii].eigval

            if ii == 0:
                fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'twisty', pin=-5000, alpha=alpha)
            else:
                leplt.DOS_plot(eigval, DOS_ax, 'twisty', pin=-5000, alpha=alpha)

        plt.title(title, fontsize=FSFS)
        if outdir is not None:
            plt.savefig(outdir+fname+'.png')
            pickle.dump(fig, file(outdir+fname+'.pkl', 'w'))
        if close:
            plt.close('all')

    def plot_ipr_DOS_stack(self, outdir=None, title=r'$D(\omega)$ for twisty network', fname='ipr_DOS_stack',
                           vmin=None, vmax=None, alpha=None, FSFS=12,
                           inverse_PR=True, show=False, close=True, ylabels=None, **kwargs):
        """Plot Inverse Participation Ratio of the twisty networks (with DOS) and stack them in a column

        Parameters
        ----------
        outdir : str or None
        title : str
        fname : str
        vmin : float or None
        vmax : float or None
        alpha : float or None
        FSFS : int
        inverse_PR : bool
        show : bool
        close : bool
        ylabels : None
        **kwargs: arguments passed to tlat.add_ipr_to_ax()
            --> most of lepm.plotting.plotting.colored_DOS_plot() keyword arguments
            Excluding pin, alpha, colorV, colormap, linewidth, make_cbar, vmin, vmax,
        """
        self.ensure_all_eigval()

        # Decide what alpha value to use if not provided (here, 1.)
        if alpha is None:
            alpha = 1.

        # Register cmaps if necessary
        if 'viridis' not in plt.colormaps() or 'viridis_r' not in plt.colormaps():
            cmaps.register_colormaps()

        fig, ax = plt.subplots(len(self.twisty_lattices), 1, sharex=True, sharey=True)

        n_tlats = len(self.twisty_lattices)

        # Plot the DOS
        xlabel = ''
        cax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
        for ii in range(n_tlats):
            print 'Plotting ', ii, ' of ', len(self.twisty_lattices)
            if ii == n_tlats - 1:
                xlabel = 'Oscillation frequency $\omega$'
            if ylabels is None:
                ylabel = ''
            else:
                ylabel = ylabels[ii]
            print 'inverse_PR = ', inverse_PR
            DOSaxis = self.twisty_lattices[ii].add_ipr_to_ax(ax[ii], alpha=alpha, inverse_PR=inverse_PR,
                                                           xlabel=xlabel, ylabel=ylabel, vmax=vmax, vmin=vmin,
                                                           cbar_ax=cax, **kwargs)
            ii += 1

        ax[0].set_title(title, fontsize=FSFS)
        ax[0].annotate(r'$D(\omega)$', xy=(.01, .5),
                       xycoords='figure fraction', horizontalalignment='left', verticalalignment='center',
                       fontsize=FSFS, rotation=90)
        if outdir is not None:
            print 'saving as ', outdir+fname+'.pkl'
            pl.dump(fig, file(outdir+fname+'.pkl', 'w'))
            plt.savefig(outdir+fname+'.png')
            plt.savefig(outdir+fname+'.pdf')
        if show:
            plt.show()
        if close:
            plt.close('all')

    def plot_ipr_DOS_overlay(self, outdir=None, title=r'$D(\omega)$ for twisty network', fname='ipr_DOS_overlay',
                             alpha=None, FSFS=12, inverse_PR=True, show=False, close=True, initialize=True,
                             DOS_ax=None, vmin=None, vmax=None, check=False, **kwargs):
        """Plot Inverse Participation Ratio of the twisty networks and overlay them

        Parameters
        ----------
        close : bool
            Whether to show the figure after plotting
        close : bool
            Whether to close all figure instances at the end
        initialize : bool
            If True or if DOSax is not supplied, creates a new plot for the ipr DOS image
        DOSax : axis instance
            Axis on which to plot the PR DOS, if initialize is False
        **kwargs: most of lepm.plotting.plotting.colored_DOS_plot() keyword arguments
            Excluding pin, alpha, colorV, colormap, linewidth, make_cbar, vmin, vmax
            Includes
        """
        self.ensure_all_eigval()

        # Decide what alpha value to use if not provided (here, 1/N)
        if alpha is None:
            alpha = 1./float(len(self.twisty_lattices))

        # Register cmaps if necessary
        if 'viridis' not in plt.colormaps() or 'viridis_r' not in plt.colormaps():
            cmaps.register_colormaps()

        # Plot the overlays
        for ii in range(len(self.twisty_lattices)):
            if ii % 10 == 0:
                print 'Overlaying ', ii, ' of ', len(self.twisty_lattices)

            eigval = self.twisty_lattices[ii].eigval

            # Load ipr
            if self.twisty_lattices[ii].ipr is None:
                # Attempt to load ipr directly
                try:
                    ipr = self.twisty_lattices[ii].load_ipr(attribute=False)
                except:
                    # To calc ipr, load eigvect
                    if self.twisty_lattices[ii].eigvect is None:
                        eigvect = self.twisty_lattices[ii].load_eigvect(attribute=False)
                    else:
                        eigvect = self.twisty_lattices[ii].eigvect
                    ipr = self.twisty_lattices[ii].calc_ipr(eigvect=eigvect, attribute=False)

            # Now overlay the ipr
            if vmin is None and inverse_PR:
                print 'setting vmin...'
                vmin = 1.0
            if ii == 0 and initialize and DOS_ax is None:
                if inverse_PR:
                    fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'twisty', pin=-5000,
                                                                                   alpha=alpha, colorV=ipr,
                                                                                   colormap='viridis', fontsize=FSFS,
                                                                                   linewidth=0, cax_label=r'$p^{-1}$',
                                                                                   vmin=vmin, vmax=vmax,
                                                                                   climbars=False, **kwargs)
                    print 'cbar_ax = ', cbar_ax
                    if vmax is None:
                        print 'setting vmax...'
                        vmax = cbar.get_clim()[1]
                else:
                    fig, DOS_ax, cbar_ax, cbar = \
                        leplt.initialize_colored_DOS_plot(eigval, 'twisty', pin=-5000, alpha=alpha,
                                                          colorV=1./ipr, colormap='viridis_r', fontsize=FSFS,
                                                          linewidth=0, cax_label=r'$p$', climbars=False,
                                                          vmin=vmin, vmax=vmax, **kwargs)
                    if vmin is None:
                        print 'setting vmin...'
                        vmin = cbar.get_clim()[0]
                    if vmax is None:
                        print 'setting vmax...'
                        vmax = cbar.get_clim()[1]
            elif ii == 0:
                if inverse_PR:
                    DOS_ax, cbar_ax, cbar, n, bins = \
                        leplt.colored_DOS_plot(eigval, DOS_ax, 'twisty', pin=-5000, alpha=alpha, colorV=ipr,
                                               fontsize=FSFS, colormap='viridis', linewidth=0, make_cbar=True,
                                               vmin=vmin, vmax=vmax, **kwargs)
                    if vmax is None:
                        print 'setting vmax...'
                        vmax = cbar.get_clim()[1]
                else:
                    DOS_ax, cbar_ax, cbar, n, bins = \
                        leplt.colored_DOS_plot(eigval, DOS_ax, 'twisty', pin=-5000, alpha=alpha, colorV=1./ipr,
                                               fontsize=FSFS, colormap='viridis_r', linewidth=0,
                                               make_cbar=True, vmin=vmin, vmax=vmax, **kwargs)
                    if vmin is None:
                        print 'setting vmin...'
                        vmin = cbar.get_clim()[0]
                    if vmax is None:
                        print 'setting vmax...'
                        vmax = cbar.get_clim()[1]
            else:
                if inverse_PR:
                    leplt.colored_DOS_plot(eigval, DOS_ax, 'twisty', pin=-5000, alpha=alpha, colorV=ipr, fontsize=FSFS,
                                           colormap='viridis', linewidth=0, make_cbar=False, vmin=vmin,
                                           vmax=vmax, **kwargs)
                else:
                    leplt.colored_DOS_plot(eigval, DOS_ax, 'twisty', pin=-5000, alpha=alpha, colorV=1./ipr, fontsize=FSFS,
                                           colormap='viridis_r', linewidth=0, make_cbar=False, vmin=vmin,
                                           vmax=vmax, **kwargs)

            if check:
                plt.pause(1)

        DOS_ax.set_title(title, fontsize=FSFS)
        if outdir is not None:
            print 'saving as ', outdir+fname+'.pkl'
            pl.dump(fig, file(outdir+fname+'.pkl', 'w'))
            plt.savefig(outdir+fname+'.png')
            plt.savefig(outdir+fname+'.pdf')
        if show:
            plt.show()
        if close:
            plt.close('all')

    def plot_ill_DOS_overlay(self, outdir=None,
                             title=r'$D(\omega)$ for twisty network', fname='ipr_DOS_overlay',
                             show=False, close=True, dos_ax=None, cbar_ax=None, check=False, fontsize=8, **kwargs):
        """Plot localization-colored DOS of the twisty networks and overlay them

        Parameters
        ----------
        outdir : str or None
            If not None, saves figures in this directory
        title : str or None
            If not None, gives figure this title
        fname : str
            filename to save the figure if outdir is not None
        show : bool
            Show the figure after plotting
        close : bool
            Close the figure after running function
        dos_ax : axis instance
            Axis on which to plot the DOS, if not None
        cax : axis instance
            Axis on which to plot the colorbar for the DOS, if not None
        check : bool
            Display intermediate results
        fontsize : int
            Fontsize for labels, title
        **kwargs: lepm.plotting.plotting.colored_DOS_plot() keyword arguments
            Excluding fontsize
        """
        dos_ax, cbar_ax = tcpfns.plot_ill_dos_overlay(self, dos_ax=dos_ax, cbar_ax=cbar_ax, fontsize=fontsize, **kwargs)
        if check:
            plt.pause(1)
        if title is not None:
            dos_ax.set_title(title, fontsize=fontsize)
        if outdir is not None:
            plt.savefig(outdir + fname + '.png')
            plt.savefig(outdir + fname + '.pdf')
        if show:
            plt.show()
        if close:
            plt.close('all')

        return dos_ax, cbar_ax

    def plot_prpoly_overlay(self, outdir=None, title=r'Polygonal contributions to normal mode excitations',
                            fname='prpoly_twisty_hist_overlay', fontsize=8, show=True,
                            shaded=False, vmax=None):
        """Plot Inverse Participation Ratio of the twisty network for each n-polygon

        Parameters
        ----------
        outdir : None or str
            If not None, outputs plot and pkl to this directory
        title : str
            The title of the plot
        fname : str
            the name of the file to save as png
        fontsize: float or int
            fontsize
        show : bool
            Whether to show the plot after creating it
        shaded : bool
            Plot contributions with opacity rather than a colormap signifying amplitude of contrib

        Returns
        -------
        ax: tuple of matplotlib axis instance handles
            handles for the axes of the histograms
        """
        n_tlats = max(len(self.twisty_lattices), len(self.meshfns))
        print 'n_tlats = ', n_tlats
        maxPno = self.calc_max_polygon_sides()
        print 'maxPno = ', maxPno

        n_ax = maxPno - 2
        fig, ax, cbar_ax = leplt.initialize_axis_stack(n_ax, Wfig=90, Hfig=120, make_cbar=True, hfrac=None,
                                                       wfrac=0.6, x0frac=None, y0frac=0.12, cbarspace=5, tspace=10,
                                                       vspace=2, cbar_orientation='horizontal')
        ind = 0
        for tlat in self.twisty_lattices:
            print '\nAdding the', ind, 'th twisty_lattice to prpoly plot...'
            tlat.add_prpoly_to_plot(fig, ax, cbar_ax=cbar_ax, global_alpha=1.0/float(n_tlats), shaded=shaded, vmax=vmax)
            if ind == 0 and vmax is None:
                # vmax = plt.gci().get_clim()[1]
                vmax = cbar_ax.get_xlim()[1]
                print 'vmax = ', vmax
            ind += 1
            if ind % 10 == 0:
                plt.pause(0.001)

        ii = 0
        ylims = ax[0].get_ylim()
        for axis in ax:
            axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks([int(ylims[1] * 0.1) * 10])
        if ii < len(ax) - 1:
            axis.set_xticklabels([])
        ii += 1

        # Set title and axes labels
        if title is not None:
            plt.suptitle(title, fontsize=fontsize)
        ax[len(ax) - 1].set_xlabel(r'Oscillation frequency, $\omega/\Omega_g$')
        ax[len(ax) - 1].annotate('Density of states, $D(\omega)$', xy=(.01, .5), xycoords='figure fraction',
                                 horizontalalignment='left', verticalalignment='center',
                                 fontsize=fontsize, rotation=90)

        for ii in range(len(ax)):
            ax[ii].set_xlabel('')
            ax[ii].set_ylabel(str(ii+3))

        if outdir is not None:
            pickle.dump(fig, file(outdir + fname + '.pkl', 'w'))
            print 'Saving figure to ' + outdir+fname+'.png'
            plt.savefig(outdir+fname+'.png')
            plt.savefig(outdir + fname + '.pdf')

        if show:
            print 'Displaying plot...'
            plt.show()
        else:
            return fig, ax, cbar_ax


if __name__ == '__main__':
    '''Perform an example of using the lattice_collection class'''
    import lepm.twisty.twisty_functions as twisty_functions
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for twisty simulation.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-lowest_mode', '--lowest_mode_series',
                        help='Draw lowest mode for series of lattices', action='store_true')
    parser.add_argument('-prpoly_overlay', '--prpoly_overlay', help='Make overlaid DOS plots', action='store_true')
    parser.add_argument('-ipr_overlay', '--ipr_overlay', help='Overlay participation ratio DOS plots', action='store_true')
    parser.add_argument('-ipr_stack', '--ipr_stack', help='Stack subplots of participation-ratio-colored DOS', action='store_true')
    parser.add_argument('-KKspec', '--KKspec', help='string specifier for OmK bond frequency matrix',
                        type=str, default='')
    parser.add_argument('-savepintxt', '--save_pinning_to_txt',
                        help='when creating a new array of pinning frequencies, save to hdf5 instead of txt',
                        action='store_true')
    parser.add_argument('-lpparam_reverse', '--lpparam_reverse',
                        help='When computing a quantity for multiple lattices, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-xaxis', '--xaxis',
                        help='String specifier for which variable to use as x axis in a plot', type=str,
                        default='kx')

    # Geometry arguments for the lattices to load
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings ' +
                                          '(leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load',
                        help='Specify to nonzero int to load a network of a particular size in its entirety, ' +
                             'without cropping. Will override NH and NV',
                        type=int, default=20)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-LT', '--LatticeTop',
                        help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, hyperuniform, ' +
                             'circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-ABk', '--ABDelta_k', help='Difference in stetch coefficient for AB bonds', type=float,
                        default=0.)
    parser.add_argument('-ABg', '--ABDelta_g', help='Difference in twist-stetch coupling for AB bonds', type=float,
                        default=0.)
    parser.add_argument('-ABc', '--ABDelta_c', help='Difference in twist coefficient for AB bonds', type=float,
                        default=0.)
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-kr_max', '--kr_max', help='Upper bound for magnitude of k in calculating S(k)',
                        type=int, default=30)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-nice_plot', '--nice_plot', help='Output nice pdf plots of lattice', action='store_true')
    
    # For loading and coordination
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='x1')
    parser.add_argument('-paramV', '--paramV',
                        help='Sequence of values to assign to lp[param] if vary_lpparam or vary_glatparam is True',
                        type=str, default='0.0:0.1:2.0')
    parser.add_argument('-paramVdtype', '-paramVdtype',
                        help='The data type for the numpy array formed from paramV',
                        type=str, default='float')
    parser.add_argument('-LLID', '--loadlattice_number',
                        help='If LT=hyperuniform/isostatic, selects which lattice to use', type=str, default='01')
    parser.add_argument('-LLz', '--loadlattice_z',
                        help='If LT=hyperuniform/isostatic, selects what z index to use', type=str, default='001')
    parser.add_argument('-source', '--source',
                        help='Selects who made the lattice to load, if loaded from source (ulrich, hexner, etc)',
                        type=str, default='hexner')
    parser.add_argument('-cut_z', '--cut_z',
                        help='Declare whether or not to cut bonds to obtain target coordination number z',
                        type=bool, default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)', type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-perd', '--percolation_density', help='Fraction of vertices to decorate', type=float,
                        default=0.5)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='Position of single dislocation, if not centered at (0,0), as strings sep by / (ex 1/4.4)',
                        type=str, default='none')
    parser.add_argument('-spreading_time', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.0)

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-periodic_strip', '--periodic_strip',
                        help='Enforce strip periodic boundary condition in horizontal dim', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.667')
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-x1', '--x1',
                        help='1st Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x2', '--x2',
                        help='2nd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x3', '--x3',
                        help='3rd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-zz', '--zz',
                        help='4th Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-nkx', '--nkx',
                        help='Number of kx values in dispersion', type=int, default=50)
    parser.add_argument('-nky', '--nky',
                        help='Number of ky values in dispersion', type=int, default=50)
    parser.add_argument('-huno', '--hyperuniform_number', help='Hyperuniform realization number',
                        type=str, default='01')
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-pinconf', '--pinconf',
                        help='Lattice disorder realization number', type=int, default=01)
    parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                        default=01)
    parser.add_argument('-intparam', '--intparam',
                        help='Integer-valued parameter for building networks (ex # subdivisions in accordionization)',
                        type=int, default=1)
    parser.add_argument('-thres', '--thres', help='Threshold value for building networks (determining to decorate pt)',
                        type=float, default=1.0)

    # Geometry and physics arguments
    parser.add_argument('-basis', '--basis', help='basis for computing eigvals', type=str, default='XY')
    parser.add_argument('-kk', '--kk', help='Spring frequency', type=str, default='1.0')
    parser.add_argument('-gg', '--gg', help='twist-stretch coupling', type=str, default='1.0')
    parser.add_argument('-cc', '--cc', help='Twist-twist coupling', type=str, default='1.0')
    parser.add_argument('-fix_interactions', '-fix_interactions',
                        help='Do not scale the interaction strength by the length of each bond',
                        action='store_true')
    parser.add_argument('-intrange', '--intrange',
                        help='Consider couplings only to nth nearest neighbors (if ==2, then NNNs, '
                             'for intrange=0, consider infinite range (all coupled), etc)',
                        type=int, default=1)
    parser.add_argument('-Vg', '--V0_gauss_g',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)
    parser.add_argument('-Vk', '--V0_gauss_k',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)
    parser.add_argument('-Vc', '--V0_gauss_c',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)

    args = parser.parse_args()

    if args.N > 0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV

    strain = 0.00
    # z = 4.0 # target z
    if args.LatticeTop == 'linear':
        shape = 'line'
    else:
        shape = args.shape

    theta = args.theta
    eta = args.eta
    transpose_lattice = 0

    make_slit = args.make_slit
    print 'theta = ', theta
    dcdisorder = args.V0_gauss_k > 0 or args.V0_gauss_g > 0 or args.V0_gauss_c > 0

    outdir = dio.prepdir(args.rootdir) + 'twisty_collections/' + args.LatticeTop + '/'
    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': args.rootdir,
          'phi_lattice': args.phi_lattice,
          'delta_lattice': args.delta_lattice,
          'theta': theta,
          'eta': eta,
          'x1': args.x1,
          'x2': args.x2,
          'x3': args.x3,
          'z': args.zz,
          'source': args.source,
          'loadlattice_number': args.loadlattice_number,
          'check': args.check,
          'Ndefects': args.Ndefects,
          'Bvec': args.Bvec,
          'dislocation_xy': args.dislocation_xy,
          'target_z': args.target_z,
          'make_slit': args.make_slit,
          'cutz_method': args.cutz_method,
          'cutLfrac': 0.0,
          'conf': args.realization_number,
          'subconf': args.sub_realization_number,
          'periodicBC': args.periodicBC or args.periodic_strip,
          'periodic_strip': args.periodic_strip,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'eta_alph': args.eta_alph,
          'origin': np.array([0., 0.]),
          'basis': args.basis,
          'kk': float(args.kk.replace('n', '-').replace('p', '.')),
          'gg': float(args.gg.replace('n', '-').replace('p', '.')),
          'cc': float(args.cc.replace('n', '-').replace('p', '.')),
          'V0_gauss_k': args.V0_gauss_k,
          'V0_gauss_g': args.V0_gauss_g,
          'V0_gauss_c': args.V0_gauss_c,
          'dcdisorder': args.V0_gauss_k > 0 or args.V0_gauss_g > 0 or args.V0_gauss_c > 0,
          'percolation_density': args.percolation_density,
          'ABDelta_k': args.ABDelta_k,
          'ABDelta_g': args.ABDelta_g,
          'ABDelta_c': args.ABDelta_c,
          'thres': args.thres,
          'pinconf': args.pinconf,
          'KKspec': args.KKspec,
          'spreading_time': args.spreading_time,
          'intparam': args.intparam,
          'interaction_range': args.intrange,
          'save_pinning_to_hdf5': not args.save_pinning_to_txt,
          'scale_interactions': not args.fix_interactions,
          }

    # Collate DOS for many lattices
    gc = TwistyCollection()
    if args.LatticeTop == 'iscentroid':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/'+args.LatticeTop+'/'+
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
    elif args.LatticeTop == 'randomcent':
        gc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' +
                      args.LatticeTop + '_square_r*' + '_' + '{0:06d}'.format(NH))

    if args.prpoly_overlay:
        """Example usage:

        python twisty_collection.py -prpoly_overlay -LT hucentroid -NP 20 -periodic
        """
        gc.ensure_all_polygons_saved()
        gc.ensure_all_prpoly_saved()
        gc.plot_prpoly_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_prpoly_overlay')

    if args.ipr_overlay:
        title = r'$D(\omega)$ for ' + leplt.lt2description(lp) + ' networks'
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_ipr_overlay', title=title,
                                inverse_PR=True)
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_pr_overlay', title=title,
                                inverse_PR=False)
        gc.plot_DOS_overlay(outdir=outdir, fname= 'NP'+str(args.NP_load)+'_eigval_hist_overlay', title=title)

    if args.ipr_stack:
        # Get ylabels from meshfn names
        ylabels = []
        ii = 0
        for meshfn in gc.meshfns:
            pstring = meshfn[meshfn.index('perd') + 4:meshfn.index('perd') + 8].replace('p', '.')
            ylabels.append(pstring)
            ii += 1

        gc.plot_ipr_DOS_stack(outdir=outdir, fname='N' + str(args.N) + '_eigval_ipr_stack', title=title,
                              ylabels=ylabels, inverse_PR=False, vmin=0.0, vmax=0.5)

    if args.lowest_mode_series:
        """
        python ./twisty/twisty_collection.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 \
            -x3 0.1 -gg 0. -cc 1. -fix_interactions -paramV n0.1:0.02:0.1 -paramVdtype float -lpparam x1 \
            -lpparam_reverse -nkx 50 -nky 50


        python ./twisty/twisty_collection.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 \
            -x3 0.1 -gg 0. -cc 1. -fix_interactions -paramV n1.1:0.05:1.1 -paramVdtype float -lpparam gg \
            -nkx 50 -nky 50

        # DNA parameters: k = 965, c=460, g=-90 --> k=1, c=0.476684, g=0.093264
        # DNA, vary gg
        python ./twisty/twisty_collection.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 \
            -x3 0.1 -gg 0. -cc 0.476684 -fix_interactions -paramV n0.1:0.01:0.1 -paramVdtype float -lpparam gg \
            -nkx 50 -nky 50

        # DNA, vary x1 in deformed kagome
        python ./twisty/twisty_collection.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 \
            -x3 0.1 -gg -0.10 -cc 0.48 -fix_interactions -paramV n0.1:0.02:0.1 -paramVdtype float -lpparam x1 \
            -lpparam_reverse -nkx 50 -nky 50

        python ./twisty/twisty_collection.py -LT hexagonal -N 1 -periodic -lowest_mode \
            -gg 0. -cc 1. -fix_interactions -paramV n0.1:0.02:0.1 -paramVdtype float -lpparam gg \
            -nkx 50 -nky 50

        # make the networks
        python run_series.py -pro ./build/make_lattice -opts LT/deformed_kagome/-x2/0.1/-x3/0.1/-N/1/-periodic/-skip_polygons \
            -var x2 n0.11:0.02:0.11
        """
        import copy
        import lepm.plotting.movies as lemov
        import lepm.stringformat as sf
        import lepm.data_handling as dh
        lp0 = copy.deepcopy(lp)
        gc = TwistyCollection()
        wfig = 180.
        y0frac, wsfrac, hspace = 0.07, 0.18, 11.0
        x0frac = (1. - 4.0 * wsfrac - 3.0 * hspace / wfig) * 0.5
        hcbarfrac, wcbarfrac = 0.05, 0.7
        print 'x0frac = ', x0frac
        x0cbarfrac, y0cbarfrac = x0frac + wsfrac + (wsfrac - wcbarfrac * wsfrac) * 0.5 + hspace / wfig, 0.85
        print 'x0cbarfrac =', x0cbarfrac
        fig, axes, cax = leplt.initialize_nxmpanel_cbar_fig(1, 4, Wfig=wfig, cbar_placement='above_center',
                                                            hcbarfrac=hcbarfrac, wcbarfrac=wcbarfrac,
                                                            orientation='horizontal',
                                                            vspace=8, y0frac=y0frac, x0frac=x0frac, wsfrac=wsfrac,
                                                            hspace=hspace, tspace=16.1,
                                                            x0cbarfrac=x0cbarfrac, y0cbarfrac=y0cbarfrac,)
        # plt.show()
        # sys.exit()
        ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]
        outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/twisty_collections/'
        outdir += 'lowest_modes/' + lp['LatticeTop'] + '/'

        # form the specstr for naming
        if args.lpparam != 'kk':
            specstr = '_k{0:0.1f}'.format(lp['kk'])
        else:
            specstr = '_kvary'
        if args.lpparam != 'gg':
            specstr += '_g{0:0.1f}'.format(lp['gg'])
        else:
            specstr += '_gvary'
        if args.lpparam != 'cc':
            specstr += '_c{0:0.1f}'.format(lp['cc'])
        else:
            specstr += '_cvary'
        specstr = specstr.replace('.', 'p')

        dio.ensure_dir(outdir)
        if lp['LatticeTop'] in ['deformed_kagome', 'hexagonal']:
            params = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)
            outdir0 = copy.deepcopy(outdir)
            outdir += 'series_' + args.lpparam + '_nparams{0:04d}'.format(len(params)) + specstr + '/'
            dio.ensure_dir(outdir)
            ii = 0
            # check it
            print 'params = ', params
            print 'args.lpparm = ', args.lpparam
            if args.lpparam_reverse:
                params = params[::-1]
            for param in params:
                name = 'twisty_' + lp['LatticeTop'].replace('_', '') + '{0:05d}'.format(ii)
                outpath = outdir + name
                if not glob.glob(outpath):
                    lp = copy.deepcopy(lp0)
                    lp[args.lpparam] = param
                    lat = lattice_class.Lattice(lp=lp)
                    lat.load()
                    tlat = twisty_lattice.TwistyLattice(lat, lp=lp)

                    # Plot the lattice in ax0
                    lat.plot_BW_lat(fig=fig, ax=ax0, meshfn='./', exten='.pdf', save=False, close=False, axis_off=True,
                                    title='', includeNNN=False)

                    # Plot the dispersion near omega==0
                    omegas, kx, ky = tlat.infinite_dispersion(kx=None, ky=None, nkxvals=args.nkx, nkyvals=args.nky,
                                                              save=False, save_plot=False, title='',
                                                              save_dos_compare=False, outdir=None, ax=ax2, xaxis='kx')
                    omegas, kx, ky = tlat.infinite_dispersion(kx=None, ky=None, nkxvals=args.nkx, nkyvals=args.nky,
                                                              save=False, save_plot=False, title='',
                                                              save_dos_compare=False, outdir=None, ax=ax3, xaxis='ky')
                    # Alternatively, plot dispersion manually here
                    # for jj in range(len(ky)):
                    #     for kk in range(len(omegas[0, jj, :])):
                    #         ax.plot(kx, omegas[:, jj, kk], 'k-', lw=max(0.03, 5. / (len(kx) * len(ky))))
                    # ax.set_title(title)
                    # ax.set_xlabel(r'$k_x$ $[\langle \ell \rangle ^{-1}]$')
                    # ax.set_ylabel(r'$\omega$')
                    # ylims = ax.get_ylim()
                    # ylim0 = min(ylims[0], -0.1 * ylims[1])
                    # ax.set_ylim(ylim0, ylims[1])

                    fig, ax1, cax, omegas, kxy, vtcs = tlat.plot_lowest_mode(nkxy=args.nkx, fig=fig, ax=ax1, cax=cax,
                                                                             name=name, outdir=outdir, imtype='png')

                    # Formatting
                    ax0.axis('scaled')
                    ax0.set_xlim(-1., 1.)
                    ax0.set_ylim(-1., 1.)
                    ax2.set_xlabel(r'wavenumber, $k_x$')
                    ax3.set_xlabel(r'wavenumber, $k_y$')
                    ax2.xaxis.set_ticks([-np.pi, 0., np.pi])
                    ax2.xaxis.set_ticklabels([r'$-\pi$', 0., r'$\pi$'])
                    ax2.set_ylabel(r'energy, $\omega$')
                    ax2.set_ylim(-0.1, 0.5)
                    ax3.set_ylim(-0.1, 0.5)
                    ax3.xaxis.set_ticks([-np.pi, 0., np.pi])
                    ax3.xaxis.set_ticklabels([r'$-\pi$', 0., r'$\pi$'])
                    title = r'Dispersion for $k=$' + '{0:0.2f}'.format(lp['kk']) +\
                            r', $g=$' + '{0:0.2f}'.format(lp['gg']) + \
                            r', $c=$' + '{0:0.2f}'.format(lp['cc'])
                    ax1.text(0.5, 0.99, title, ha='center', va='top', transform=fig.transFigure)
                    plt.savefig(outpath + '.png')
                    plt.savefig(outpath + '.pdf')

                # plt.show()
                # sys.exit()
                ax0.cla()
                ax1.cla()
                ax2.cla()
                ax3.cla()
                cax.cla()
                ii += 1

            movname = lp['LatticeTop'] + '_twisty_lowmode_series_' + args.lpparam + \
                      '_nparams{0:04d}'.format(len(params)) + specstr
            basename = 'twisty_' + lp['LatticeTop'].replace('_', '')
            lemov.make_movie(outdir + basename, outdir0 + movname, framerate=float(len(params)) * 0.15,
                             imgdir=outdir, rm_images=False, save_into_subdir=True)
        else:
            raise RuntimeError('Have not coded for this case')

