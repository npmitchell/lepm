import numpy as np
import lepm.lattice_elasticity as le
import lepm.lattice_collection as lattice_collection
import lepm.lattice_class as lattice_class
import lepm.dataio as dio
import lepm.gyro_lattice_class as gyro_lattice_class
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import lepm.plotting.gyro_collection_plotting_functions as gcpfns
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
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Example usage:
import lepm.lattice_collection as latcoll
lc = latcoll.lattice_collection()
lc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/iscentroid/iscentroid_square_hexner_size64_conf*_NP000064')
gyro_collection()

'''


class GyroCollection:
    """Create a collection of gyroscopic spring lattices. Lattices can exist in memory, on hard disk, or both.
    
    Attributes
    ----------
    self.gyro_lattices : list of instances of gyro_lattice_class.GyroLattice() corresponding to each network
    self.physics : dict, containing dos_collection, chern_collection
    """
    def __init__(self):
        """Create an instance of a lattice_collection."""
        self.gyro_lattices = []
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

    def add_gyro_lattice(self, glat):
        """Add gyro_lattice instance and its meshfn to current collection"""
        self.gyro_lattices.append(glat)
        self.meshfns.append(glat.lp['meshfn'])

    def get_meshfns(self):
        try:
            return self.meshfns
        except NameError:
            # Load meshfns from each lattice in the collection
            meshfns = []
            for glat in lattice_collection.gyro_lattices:
                try:
                    meshfns.append(glat.lp['meshfn'])
                except NameError:
                    meshfns.append('in_memory')

            self.meshfns = meshfns
            return self.meshfns

    def get_localizations(self, save=False, save_eigvect_eigval=False):
        """Return list of all glat.localization arrays"""
        localizations = []
        for glat in self.gyro_lattices:
            localizations.append(glat.get_localization(save=save, save_eigvect_eigval=save_eigvect_eigval))

        return localizations

    def get_eigvals(self):
        """Get all eigval arrays for all glats"""
        eigvals = []
        for glat in self.gyro_lattices:
            eigvals.append(glat.get_eigval())

        return eigvals

    def load_gyro_lattices(self, load_DOS=True):
        # Use ensure_all_gyro_lattices() instead
        # for meshfn in self.meshfns:
        #     lat = lattice_class.lattice()
        #     lat.load(meshfn=meshfn)
        #     ##glat.load()
        #     !!!!!!! CONTINUE HERE !!!!!
        #     todo
        #     self.gyro_lattices.append()
        pass

    def load_eigvals(self, save_eigvals=False):
        """

        Returns
        -------

        """
        for ii in range(len(self.meshfns)):
            self.ensure_all_gyro_lattices()
            eigval = self.gyro_lattices[ii].load_eigval(attribute=True)
            if eigval is None:
                self.gyro_lattices[ii].calc_eigvals(attribute=True)
                if save_eigvals:
                    self.gyro_lattices[ii].save_eigvals()

    def load_eigvects(self):
        """Ensure that for every gyro_lattice there is an eigvect array loaded"""
        for ii in range(len(self.meshfns)):
            self.ensure_all_gyro_lattices()
            self.gyro_lattices[ii].load_eigvect(attribute=True)

    def calc_max_polygon_sides(self):
        maxpno = 0
        for ii in range(len(self.gyro_lattices)):
            if self.gyro_lattices[ii].lattice.polygons is not None:
                polygons = self.gyro_lattices[ii].lattice.polygons
            else:
                polygons = self.gyro_lattices[ii].lattice.load_polygons()

            # print 'polygons = ', polygons

            # number of polygon sides
            Pno = np.array([len(polyg) - 1 for polyg in polygons], dtype=int)
            maxpno = max(maxpno, np.max(Pno))
            print 'maxpno = ', maxpno

        return maxpno

    def ensure_all_gyro_lattices(self):
        """Ensure that all gyro lattices called for by self.meshfns are loaded.
        """
        # todo: add the ability to change glat.lp parameters (like Omk) at function call-- specify as list of lp's or as single lp
        print 'Ensuring all gyro lattices...'
        for ii in range(len(self.meshfns)):
            meshfn = self.meshfns[ii]
            if isinstance(meshfn, list):
                meshfn = meshfn[0]
            print 'Ensuring ', meshfn
            try:
                self.gyro_lattices[ii]
                append_glat = False
            except IndexError:
                append_glat = True

            try:
                # Check if lp['meshfn'] of gyro_lattice matches meshfn
                self.gyro_lattices[ii].lp['meshfn']
                # print('Already have gyro_lattice defined for ii='+str(ii)+': ', test)
            except IndexError:
                lp = le.load_params(dio.prepdir(meshfn), 'lattice_params')
                lat = lattice_class.Lattice(lp=lp)
                lat.load(meshfn=meshfn)
                lp['Omk'] = -1.0
                lp['Omg'] = -1.0
                if append_glat:
                    self.gyro_lattices.append(gyro_lattice_class.GyroLattice(lat, lp))
                else:
                    self.gyro_lattices[ii] = gyro_lattice_class.GyroLattice(lat, lp)

                glat = gyro_lattice_class.GyroLattice(lat, self.gyro_lattices[ii].lp)
                self.gyro_lattices[ii] = glat

    def ensure_all_eigval(self, save_eigval=False):
        """Ensure eigenval is attribute for each gyro_lattice in gyro_collection"""
        self.ensure_all_gyro_lattices()

        # Make sure all eigvals are accounted for
        test = self.gyro_lattices[0].eigval is not None
        for ii in np.arange(1,len(self.gyro_lattices)):
            test = self.gyro_lattices[ii].eigval is not None and test
        print 'All gyro_lattices have eigval attribute, all set to overlay DOS...'
        if not test:
            print 'Some gyro_lattices do not already have eigval as an attribute, loading those now...'
            self.load_eigvals(save_eigvals=save_eigval)

    def ensure_all_ill_saved(self, attribute=False):
        self.ensure_all_gyro_lattices()
        for ii in range(len(self.gyro_lattices)):
            glat = self.gyro_lattices[ii]
            if not glob.glob(glat.lp['meshfn'] + '/localization' + glat.lp['meshfn_exten'] + '.txt'):
                print 'saving localization ', ii, ' of ', len(self.gyro_lattices), '...'
                glat.save_localization(attribute=attribute)
        print 'All networks have localization saved.'

    def ensure_all_ipr_saved(self):
        self.ensure_all_gyro_lattices()
        for ii in range(len(self.gyro_lattices)):
            glat = self.gyro_lattices[ii]
            if not glob.glob(glat.lp['meshfn'] + '/ipr' + glat.lp['meshfn_exten'] + '.pkl'):
                print 'saving ipr ', ii, ' of ', len(self.gyro_lattices), '...'
                glat.save_ipr(attribute=False)
        print 'All networks have ipr saved.'

    def ensure_all_polygons_saved(self):
        self.ensure_all_gyro_lattices()
        for ii in range(len(self.gyro_lattices)):
            glat = self.gyro_lattices[ii]
            if not glob.glob(glat.lp['meshfn'] + '/polygons.pkl'):
                print 'saving polygons ', ii, ' of ', len(self.gyro_lattices), '...'
                glat.lattice.save_polygons(attribute=False)
        print 'All networks have prpoly saved.'

    def ensure_all_prpoly_saved(self):
        self.ensure_all_gyro_lattices()
        for ii in range(len(self.gyro_lattices)):
            glat = self.gyro_lattices[ii]
            if not glob.glob(glat.lp['meshfn'] + '/prpoly' + glat.lp['meshfn_exten'] + '.pkl'):
                print 'saving prpoly ', ii, ' of ', len(self.gyro_lattices), '...'
                glat.save_prpoly(attribute=False)
                plt.close('all')
            # check that pickle file isn't empty
            # else:
            #     filename = glob.glob(glat.lp['meshfn'] + '/prpoly' + glat.lp['meshfn_exten'] + '.pkl')[0]
            #     with open(filename, "rb") as fn:
            #         prpoly = pickle.load(fn)
            #     if prpoly is None:
            #         print 'loaded prpoly = ', prpoly
            #         sys.exit()
        print 'All networks have prpoly saved.'

    def plot_DOS_overlay(self, outdir=None, title=r'$D(\omega)$ for gyroscopic networks', fname='DOS_overlay.png',
                         alpha=None, FSFS=12, close=True):
        """
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        """
        self.ensure_all_gyro_lattices()
        self.ensure_all_eigval()

        # Decide what alpha value to use if not provided (here, 1/N)
        if alpha is None:
            alpha = 1./float(len(self.gyro_lattices))

        for ii in range(len(self.gyro_lattices)):
            eigval = self.gyro_lattices[ii].eigval

            if ii == 0:
                fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro', pin=-5000, alpha=alpha)
            else:
                leplt.DOS_plot(eigval, DOS_ax, 'gyro', pin=-5000, alpha=alpha)

        plt.title(title, fontsize=FSFS)
        if outdir is not None:
            plt.savefig(outdir+fname+'.png')
            pickle.dump(fig, file(outdir+fname+'.pkl', 'w'))
        if close:
            plt.close('all')

    def plot_ipr_DOS_stack(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='ipr_DOS_stack',
                           vmin=None, vmax=None, alpha=None, FSFS=12,
                           inverse_PR=True, show=False, close=True, ylabels=None, **kwargs):
        """Plot Inverse Participation Ratio of the gyroscopic networks (with DOS) and stack them in a column

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
        **kwargs: arguments passed to glat.add_ipr_to_ax()
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

        fig, ax = plt.subplots(len(self.gyro_lattices), 1, sharex=True, sharey=True)

        n_glats = len(self.gyro_lattices)

        # Plot the DOS
        xlabel = ''
        cax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
        for ii in range(n_glats):
            print 'Plotting ', ii, ' of ', len(self.gyro_lattices)
            if ii == n_glats - 1:
                xlabel = 'Oscillation frequency $\omega$'
            if ylabels is None:
                ylabel = ''
            else:
                ylabel = ylabels[ii]
            print 'inverse_PR = ', inverse_PR
            DOSaxis = self.gyro_lattices[ii].add_ipr_to_ax(ax[ii], alpha=alpha, inverse_PR=inverse_PR,
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

    def plot_ipr_DOS_overlay(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='ipr_DOS_overlay',
                             alpha=None, FSFS=12, inverse_PR=True, show=False, close=True, initialize=True,
                             DOS_ax=None, vmin=None, vmax=None, check=False, **kwargs):
        """Plot Inverse Participation Ratio of the gyroscopic networks and overlay them

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
            alpha = 1./float(len(self.gyro_lattices))

        # Register cmaps if necessary
        if 'viridis' not in plt.colormaps() or 'viridis_r' not in plt.colormaps():
            cmaps.register_colormaps()

        # Plot the overlays
        for ii in range(len(self.gyro_lattices)):
            if ii % 10 == 0:
                print 'Overlaying ', ii, ' of ', len(self.gyro_lattices)

            eigval = self.gyro_lattices[ii].eigval

            # Load ipr
            if self.gyro_lattices[ii].ipr is None:
                # Attempt to load ipr directly
                try:
                    ipr = self.gyro_lattices[ii].load_ipr(attribute=False)
                except:
                    # To calc ipr, load eigvect
                    if self.gyro_lattices[ii].eigvect is None:
                        eigvect = self.gyro_lattices[ii].load_eigvect(attribute=False)
                    else:
                        eigvect = self.gyro_lattices[ii].eigvect
                    ipr = self.gyro_lattices[ii].calc_ipr(eigvect=eigvect, attribute=False)

            # Now overlay the ipr
            if vmin is None and inverse_PR:
                print 'setting vmin...'
                vmin = 1.0
            if ii == 0 and initialize and DOS_ax is None:
                if inverse_PR:
                    fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro', pin=-5000,
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
                        leplt.initialize_colored_DOS_plot(eigval, 'gyro', pin=-5000, alpha=alpha,
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
                        leplt.colored_DOS_plot(eigval, DOS_ax, 'gyro', pin=-5000, alpha=alpha, colorV=ipr,
                                               fontsize=FSFS, colormap='viridis', linewidth=0, make_cbar=True,
                                               vmin=vmin, vmax=vmax, **kwargs)
                    if vmax is None:
                        print 'setting vmax...'
                        vmax = cbar.get_clim()[1]
                else:
                    DOS_ax, cbar_ax, cbar, n, bins = \
                        leplt.colored_DOS_plot(eigval, DOS_ax, 'gyro', pin=-5000, alpha=alpha, colorV=1./ipr,
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
                    leplt.colored_DOS_plot(eigval, DOS_ax, 'gyro', pin=-5000, alpha=alpha, colorV=ipr, fontsize=FSFS,
                                           colormap='viridis', linewidth=0, make_cbar=False, vmin=vmin,
                                           vmax=vmax, **kwargs)
                else:
                    leplt.colored_DOS_plot(eigval, DOS_ax, 'gyro', pin=-5000, alpha=alpha, colorV=1./ipr, fontsize=FSFS,
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
                             title=r'$D(\omega)$ for gyroscopic network', fname='ipr_DOS_overlay',
                             show=False, close=True, dos_ax=None, cbar_ax=None, check=False, fontsize=8, **kwargs):
        """Plot localization-colored DOS of the gyroscopic networks and overlay them

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
        dos_ax, cbar_ax = gcpfns.plot_ill_dos_overlay(self, dos_ax=dos_ax, cbar_ax=cbar_ax, fontsize=fontsize, **kwargs)
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
                            fname='prpoly_gyro_hist_overlay', fontsize=8, show=True,
                            shaded=False, vmax=None):
        """Plot Inverse Participation Ratio of the gyroscopic network for each n-polygon

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
        n_glats = max(len(self.gyro_lattices), len(self.meshfns))
        print 'n_glats = ', n_glats
        maxPno = self.calc_max_polygon_sides()
        print 'maxPno = ', maxPno

        n_ax = maxPno - 2
        fig, ax, cbar_ax = leplt.initialize_axis_stack(n_ax, Wfig=90, Hfig=120, make_cbar=True, hfrac=None,
                                                       wfrac=0.6, x0frac=None, y0frac=0.12, cbarspace=5, tspace=10,
                                                       vspace=2, cbar_orientation='horizontal')
        ind = 0
        for glat in self.gyro_lattices:
            print '\nAdding the', ind, 'th gyro_lattice to prpoly plot...'
            glat.add_prpoly_to_plot(fig, ax, cbar_ax=cbar_ax, global_alpha=1.0/float(n_glats), shaded=shaded, vmax=vmax)
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
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-prpoly_overlay', '--prpoly_overlay', help='Make overlaid DOS plots', action='store_true')
    parser.add_argument('-ipr_overlay', '--ipr_overlay', help='Overlay participation ratio DOS plots', action='store_true')
    parser.add_argument('-ipr_stack', '--ipr_stack', help='Stack subplots of participation-ratio-colored DOS', action='store_true')

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
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-kr_max', '--kr_max', help='Upper bound for magnitude of k in calculating S(k)',
                        type=int, default=30)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-nice_plot', '--nice_plot', help='Output nice pdf plots of lattice', action='store_true')
    
    # For loading and coordination
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
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1 )
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='Position of single dislocation, if not centered at (0,0), as str sep by / (ex: 1/4.4)',
                        type=str, default='none')
    
    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-phi', '--phi', help='Shear angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=0.0)
    parser.add_argument('-delta', '--delta', help='Deformation angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=120./180.)
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.000)
    parser.add_argument('-huno', '--hyperuniform_number', help='Hyperuniform realization number', type=str, default='01')
    parser.add_argument('-skip_gr', '--skip_gr', help='Skip calculation of g(r) correlation function for the lattice',
                        action='store_true')
    parser.add_argument('-skip_gxy', '--skip_gxy',
                        help='Skip calculation of g(x,y) 2D correlation function for the lattice', action='store_true')
    parser.add_argument('-skip_sigN', '--skip_sigN', help='Skip calculation of variance_N(R)', action='store_true')
    parser.add_argument('-fancy_gr', '--fancy_gr',
                        help='Perform careful calculation of g(r) correlation function for the ENTIRE lattice',
                        action='store_true')
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

    strain = 0.00
    if lattice_type == 'linear':
        shape = 'line'
    else:
        shape = args.shape
    
    theta = args.theta
    eta = args.eta
    transpose_lattice=0
    
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
    elif args.LatticeTop == 'randomcent':
        description = 'voronoized, unformly random'

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    outdir = rootdir+'experiments/DOS_scaling/'+args.LatticeTop+'/'
    le.ensure_dir(outdir)
    lp = {'LatticeTop' : args.LatticeTop,
          'NH': NH,
          'NV': NV,
          'rootdir': rootdir,
          'periodicBC': True,
          }

    # Collate DOS for many lattices
    gc = GyroCollection()
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

    title = r'$D(\omega)$ for ' + description + ' networks'

    if args.prpoly_overlay:
        """Example usage:

        python gyro_collection.py -prpoly_overlay -LT hucentroid -NP 20 -periodic
        """
        gc.ensure_all_polygons_saved()
        gc.ensure_all_prpoly_saved()
        gc.plot_prpoly_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_prpoly_overlay')

    if args.ipr_overlay:
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
