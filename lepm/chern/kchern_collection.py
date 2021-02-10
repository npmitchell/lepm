import numpy as np
import lepm.chern.kchern_fns as kcfns
import lepm.chern.kchern_collection_functions as kccollfns
import lepm.haldane.haldane_lattice_class as haldane_lattice_class
import lepm.haldane.haldane_lattice_functions as hlatfns
import lepm.plotting.science_plot_style as sps
import matplotlib.pyplot as plt
import cPickle as pkl
import time
import os.path
import sys
import copy
import lepm.dataio as dio
import lepm.chern.kchern as kchern
import glob

'''Class for a collection of k-space Chern calculations for haldane_lattices.
I hacked this class together from kchern_gyro_collection.py, so it might have some bugs.'''


class KChernCollection:
    """Create a collection of kspace chern measurements for haldane networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.cherns is a list of tuples of dicts: self.cherns = [chern1, chern2,...]
    where chern1, chern2 are each a class with attributes:
            cp : dict
                chern calc parameters
            haldane_lattice : HaldaneLattice class instance
                the haldane lattice for which to compute the chern number
            chern : dict
                {'kx': kkx --> the x component of the wavenumbers where Berry curvature is evaluated
                 'ky': kky --> the y component of the wavenumbers where Berry curvature is evaluated
                 'chern': bv --> the chern numbers of each band whan all kx and ky pts taken into consideration
                 'bands': bands --> the eigenvalues of the bands at each kx and ky pt
                 'traces': traces -->
                 'bzvtcs': bzvtcs --> the corners of the Brillouin zone for the reciprocal lattice
                 }

    Attributes
    ----------
    self.cherns : dict
        keys are hlat_names (strings), values are lists of KitaevChern instances
    self.cp : dict
        chern calculation master dictionary
    self.haldane_collection : list of haldane_lattice instances
        list of instances of haldane_lattice_class.HaldaneLattice() corresponding to each haldane network
    self.meshfns : list of strings
        string paths to the location of the haldane_lattices in the haldane_collection
    self.hlat_names : list of strings
        the names of the haldane_lattices in the chern collection
    """

    def __init__(self, haldane_collection, cp=None, cpmeshfn=None):
        """

        Parameters
        ----------
        haldane_collection
        cp
        cpmeshfn

        Attributes
        ----------
        haldane_collection
        hlat_names
        meshfns
        cherns : dict
            the chern calculations, with keys being elements of hlat_names
        """
        self.haldane_collection = haldane_collection
        self.hlat_names = []
        self.meshfns = haldane_collection.meshfns
        self.cherns = {}

        if cp is None:
            cp = {}
        if 'basis' not in cp:
            cp['basis'] = 'psi'

        self.cp = cp

    def chern_is_saved(self, hlat, cp=None, verbose=False):
        """"""
        if verbose:
            print '\nChecking if chern is already saved...'
        if cp is None:
            cp = copy.deepcopy(self.cp)
        chern = kchern.KChernHaldane(hlat, cp)
        if glob.glob(chern.cp['cpmeshfn'] + 'chern.txt'):
            return True
        else:
            return False

    def add_chern(self, hlat, cp=None, attribute_evs=False, verbose=False, save_png=True):
        """Add a ChernIndex instance to the ChernHaldaneCollection, and append the hlat_name to self.hlat_names if not
        already in that list

        Parameters
        ----------
        hlat : HaldaneLattice instance
            the haldanelattice class in the collection (or to be added to collection) for which to compute the chern
        cp : dict
            the chern parameter dictionary, with keys density, rootdir,...
        attribute_evs : bool
            attribute the eigenvectors/values to self
        verbose : bool
            print lots of info to command line
        save_png : bool
            Save the resulting chern measurement as an image
        """
        if cp is None:
            cp = copy.deepcopy(self.cp)
        chern = kchern.KChern(hlat, cp)
        chern.get_chern(verbose=verbose, save_png=save_png, attribute=True)
        hlat_name = hlat.lp['meshfn'] + hlat.lp['meshfn_exten']
        if hlat_name not in self.cherns:
            self.cherns[hlat_name] = []

        self.cherns[hlat_name].append(chern)
        if hlat_name not in self.hlat_names:
            self.hlat_names.append(hlat_name)
        return chern

    def get_cherns(self, cp=None, verbose=False, reverse=False, save_png=True):
        """Retrieve cherns for each hlat in self.haldane_collection.haldane_lattices matching the supplied/attributed cp

        Parameters
        ----------
        cp : none or dict
            chern parameters dictionary
        verbose : bool
            Print more statements to commandline out
        """
        if cp is None:
            cp_orig = self.cp
        else:
            cp_orig = cp

        if reverse:
            hlatstodo = self.haldane_collection.haldane_lattices[::-1]
        else:
            hlatstodo = self.haldane_collection.haldane_lattices

        for hlat in hlatstodo:
            cp = copy.deepcopy(cp_orig)
            cp.pop('cpmeshfn', None)
            print 'Adding chern for hlat =', hlat.lp['meshfn']
            self.add_chern(hlat, cp=cp, verbose=verbose, save_png=save_png)

    def get_avg_chern(self, band_index=-1):
        """Considering all cherns in the kcoll, get the average chern value obtained

        Parameters
        ----------
        """
        print 'kchern_haldane_collection: Obtaining average chern over all cherns in the collection'
        nuoutlist = []
        hlat_name_list = []
        for hlat_name in self.cherns:
            print 'hlat_name = ', hlat_name
            vals = []
            # get val for this chern
            for chernii in self.cherns[hlat_name]:
                newnu = chernii.chern['chern'][band_index]
                vals.append(newnu)

            val = np.mean(vals)
            nuoutlist.append(val)
            hlat_name_list.append(hlat_name)

        return hlat_name_list, nuoutlist

    def calc_cherns_vary_hlatparam(self, hlatparam, paramV, reverse=False, verbose=False, save_png=True):
        """For a single Lattice instance, vary a HaldaneLattice parameter and compute the chern index for each value.
        When supplied, a single haldane_lattice instance is required for each physical lattice, but each new hlat
        (same lattice, different lp) will be appended to self.haldane_collection.haldane_lattices when lp is updated

        Parameters
        ----------
        hlatparam : str key for lp dictionary
            the string specifier for the parameter to change FOR THE SAME LATTICE, for each value in paramV
        paramV : list or 1d numpy array
            the values to assign to hlat.lp[hlatparam] for each chern calculation
        reverse : bool
            reverse the order of paramV in which to compute
        verbose : bool
            print intermediate output
        """
        print 'Looping over all networks in the haldane collection. ' + \
              'For each one, cycling through hlat lpparams in paramV..'
        print('--> hlats to do:', [key for key in self.haldane_collection.haldane_lattices])
        hlatstodo = [key for key in self.haldane_collection.haldane_lattices]
        for hlat in hlatstodo:
            print '\n\nchern_collection: Getting cherns for HaldaneLattice ', hlat, '...'
            print ' of name ', hlat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)

            proj = None
            print 'paramV = ', paramV
            if reverse:
                paramtodo = paramV[::-1]
            else:
                paramtodo = paramV

            lp_master = copy.deepcopy(hlat.lp)
            lat = copy.deepcopy(hlat.lattice)
            for val in paramtodo:
                cp = copy.deepcopy(self.cp)
                lp = copy.deepcopy(lp_master)
                lp[hlatparam] = val
                print 'hlatparam = ', hlatparam
                # print 'lp[OmKspec]=', lp['OmKspec']
                hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
                print 'kchern_haldane_collection: hlat.Omg = ', hlat.Omg

                # This should not be necessary since appended in self.add_chern()
                # self.haldane_collection.haldane_lattices.append(hlat)

                # Simply add the one chern to the collection
                self.add_chern(hlat, cp=cp, verbose=verbose, save_png=save_png)

        return self.cherns

    def calc_cherns_vary_lpparam(self, lpparam, paramV, reverse=False, verbose=False, save_png=True):
        """For a single Lattice instance, vary a lp lattice parameter and compute the chern index for each value.
        When supplied, a single haldane_lattice instance is required for each physical lattice, but each new hlat
        (same lattice, different lp) will be appended to self.haldane_collection.haldane_lattices when lp is updated

        Parameters
        ----------
        lpparam : str key for lp dictionary
            the string specifier for the parameter to change FOR THE SAME LATTICE, for each value in paramV
        paramV : list or 1d numpy array
            the values to assign to hlat.lp[hlatparam] for each chern calculation
        reverse : bool
            reverse the order of paramV in which to compute
        verbose : bool
            print intermediate output
        save_png : bool
            whether to save an image of the chern calculations over the varied parameter (? is that right?)
        """
        print 'Looping over all networks in the haldane collection. '
        print('--> hlats to do:', [key for key in self.haldane_collection.haldane_lattices])
        hlatstodo = [key for key in self.haldane_collection.haldane_lattices]
        for hlat in hlatstodo:
            print '\n\nchern_collection: Getting cherns for HaldaneLattice ', hlat, '...'
            print ' of name ', hlat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)

            proj = None
            lp_master = copy.deepcopy(hlat.lp)
            lat = copy.deepcopy(hlat.lattice)
            cp = copy.deepcopy(self.cp)
            lp = copy.deepcopy(lp_master)

            # This should not be necessary since appended in self.add_chern()
            # self.haldane_collection.haldane_lattices.append(hlat)

            # Simply add the one chern to the collection
            self.add_chern(hlat, cp=cp, verbose=verbose, save_png=save_png)

        return self.cherns

    def collect_cherns_vary_lpparam(self, param='percolation_density', reverse=False, band_index=-1):
        """Collect chern calculations for each value of the parameter which is varying between networks.
        Self.cherns must be a dictionary with populated chern calculations already

        Parameters
        ----------
        param: str
            string specifier for the parameter name being varied
        reverse : bool
            whether to collect/load/compute cherns in reverse order
        band_index : int
            which band

        Returns
        -------
        param_nu : #paramvals x #bands float array
            the parameter values (column 0) and the chern indices for each band (additional columns)
        """
        if self.cherns == {}:
            self.get_cherns(reverse=reverse)

        params = []
        nuList = []
        for hlat_name in self.cherns:
            print 'adding params from hlat_name = ', hlat_name
            first = True
            for chern in self.cherns[hlat_name]:
                if first:
                    # Grab the param associated with this haldane_lattice, which is stored as an attribute of
                    # the chern instance
                    params.append(chern.haldane_lattice.lp[param])
                    print 'kcoll: added param for varying hlat_param: ', param, ' --> ', chern.haldane_lattice.lp[param]
                    nu = np.array([np.real(chern.chern['chern'])])
                    print 'nu = ', nu
                    first = False
                else:
                    # Collate the other Chern calculations
                    params.append(chern.haldane_lattice.lp[param])
                    nu_tmp = np.real(chern.chern['chern'])
                    nu_tmp = nu_tmp.reshape(len(nu_tmp), 1)
                    nu = np.hstack((nu, nu_tmp))

            # If there were multiple, average over them all
            if len(self.cherns[hlat_name]) > 1:
                # Average over them all
                nu = np.mean(nu, axis=1)

            nuList.append(nu)

        # Build output array
        params = np.array(params)[:, np.newaxis]
        nuarr = np.array(nuList)[:, 0, :]
        print 'params = ', params
        print 'nuarr = ', nuarr
        param_nu = np.hstack((params, nuarr))
        # print 'param_nu = ', param_nu
        # print 'np.shape(param_nu) = ', np.shape(param_nu)
        return param_nu

    def collect_cherns_vary_hlatparam(self, param='ABDelta', reverse=False, band_index=-1):
        """Plot chern indices with x axis being the parameter which is varying between networks.
        If there are multiple chern calculations of a particular HaldaneLattice, average nu over them all.

        Parameters
        ----------
        param : str
        reverse : bool
            Compute the chern numbers for the HaldaneLattice instances in reverse order
        band_index : int (default=-1)
            the index of the band for which to store the chern number
        """
        if self.cherns == {}:
            print 'kchern_haldane_collection: running get_cherns()'
            self.get_cherns(reverse=reverse)

        params = []
        nuList = []
        for hlat_name in self.cherns:
            print 'kchern_haldane_collection: adding params from hlat_name = ', hlat_name
            first = True
            for chern in self.cherns[hlat_name]:
                if first:
                    # Grab the param associated with this haldane_lattice, which is stored as an attribute of
                    # the chern instance
                    paramval = kccollfns.retrieve_param_value(chern.haldane_lattice.lp[param])
                    params.append(paramval)
                    print 'kcoll: added param for varying hlat_param: ', param, ' --> ', chern.haldane_lattice.lp[param]
                    # For now assume that ksize_frac_arr (and regions, etc) are uniform across them all
                    nu = np.real(chern.chern['chern'][band_index])
                    nuList.append(nu)
                    first = False
                else:
                    # Collate the other Chern calculations
                    paramval = kccollfns.retrieve_param_value(chern.haldane_lattice.lp[param])
                    params.append(paramval)
                    nuList.append(np.real(chern.chern['chern'][band_index]))

        # Build output array
        print 'params = ', params
        print 'shape(params) = ', np.shape(params)
        print 'nuList = ', nuList
        for ii in range(len(params)):
            print 'ii = ', ii
            param = kccollfns.retrieve_param_value(params[ii])
            nu = nuList[ii]
            if ii == 0:
                print 'param = ', param
                param_nu = np.array([param, nu])
            else:
                add_array = np.array([param, nu])
                param_nu = np.vstack((param_nu, add_array))

        return param_nu

    def collect_chernbands_vary_hlatparam(self, param='ABDelta', reverse=False):
        """Collect cherns into an array with the parameter which is varying between networks.
        If there are multiple chern calculations of a particular HaldaneLattice, average nu over them all.

        Parameters
        ----------
        param : str
        reverse : bool
            Compute the chern numbers for the HaldaneLattice instances in reverse order
        """
        if self.cherns == {}:
            print 'kchern_haldane_collection: running get_cherns()'
            self.get_cherns(reverse=reverse)

        params = []
        nuList, bminList, bmaxList = [], [], []
        for hlat_name in self.cherns:
            print 'kchern_haldane_collection: adding params from hlat_name = ', hlat_name
            first = True
            for chern in self.cherns[hlat_name]:
                if first:
                    # Grab the param associated with this haldane_lattice, which is stored as an attribute of
                    # the chern instance
                    paramval = kccollfns.retrieve_param_value(chern.haldane_lattice.lp[param])
                    params.append(paramval)
                    print 'kcoll: added param for varying hlat_param: ', param, ' --> ', chern.haldane_lattice.lp[param]
                    # For now assume that ksize_frac_arr (and regions, etc) are uniform across them all
                    cherns = np.real(chern.chern['chern'])
                    bands = np.real(chern.chern['bands'])
                    bmin = np.min(bands, axis=0)
                    bmax = np.max(bands, axis=0)
                    # Append these values to the lists
                    nuList.append(cherns.tolist())
                    bminList.append(bmin)
                    bmaxList.append(bmax)
                    first = False
                else:
                    # Collate the other Chern calculations
                    paramval = kccollfns.retrieve_param_value(chern.haldane_lattice.lp[param])
                    params.append(paramval)
                    cherns = np.real(chern.chern['chern'])
                    bands = np.real(chern.chern['bands'])
                    bmin = np.min(bands, axis=0)
                    bmax = np.max(bands, axis=0)
                    # Append these values to the lists
                    nuList.append(cherns.tolist())
                    bminList.append(bmin)
                    bmaxList.append(bmax)

        # Build output array
        # print 'params = ', params
        # print 'shape(params) = ', np.shape(params)
        # print 'nuList = ', np.array(nuList)
        for ii in range(len(params)):
            print 'ii = ', ii
            param = kccollfns.retrieve_param_value(params[ii])
            nus = nuList[ii]
            bmin = bminList[ii]
            bmax = bmaxList[ii]
            if ii == 0:
                bandnu = np.array([[bn, bx, nu] for (bn, bx, nu) in zip(bmin, bmax, nus)])
                param_nu = np.hstack((np.array([param]), bandnu.ravel()))
            else:
                bandnu = np.array([[bn, bx, nu] for (bn, bx, nu) in zip(bmin, bmax, nus)])
                add_array = np.hstack((np.array([param]), bandnu.ravel()))
                param_nu = np.vstack((param_nu, add_array))

        return param_nu

    def collect_chernbands_vary_lpparam(self, param='phi', reverse=False):
        """Collect cherns into an array with the parameter which is varying between networks.
        If there are multiple chern calculations of a particular HaldaneLattice, average nu over them all.

        Parameters
        ----------
        param : str
        reverse : bool
            Compute the chern numbers for the HaldaneLattice instances in reverse order
        """
        # this is the same as collect_chernbands_vary_hlatparam()
        return self.collect_chernbands_vary_hlatparam(param=param, reverse=reverse)

    def plot_cherns_vary_param(self, param_type='hlat', sz_param_nu=None,
                               reverse=False, param='percolation_density',
                               title='Chern index calculation', xlabel=None):
        """Plot chern indices with x axis being the parameter which is varying between networks.
        If there are multiple chern calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        param_type : str ('hlat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a haldanelattice parameter (same lattice,
            different physics)
        reverse : bool
            Compute cherns for HaldaneLattice instances in self.haldane_collection in reverse order
        param : string
            string specifier for HaldaneLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))

        """
        kccollfns.plot_cherns_vary_param(self, param_type=param_type, sz_param_nu=sz_param_nu,
                                          reverse=reverse, param=param, title=title, xlabel=xlabel)

    def plot_chernbands_vary_param(self, param_type='hlat', param_band_nu=None,
                                   reverse=False, param='percolation_density',
                                   title='Chern index calculation', xlabel=None, ymax=None, absx=False):
        """Plot band structure as x axis varies, with bands colored by chern indices. The x axis is the parameter which
        is varying between networks.
        If there are multiple chern calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        param_type : str ('hlat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a haldanelattice parameter (same lattice,
            different physics)
        reverse : bool
            Compute cherns for HaldaneLattice instances in self.haldane_collection in reverse order
        param : string
            string specifier for HaldaneLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))
        ymax : float
            The upper ylimit in the plot
        absx : bool
            Take the absolute value of x
        """
        kccollfns.plot_chernbands_vary_param(self, param_type=param_type, param_band_nu=param_band_nu,
                                              reverse=reverse, param=param, title=title, xlabel=xlabel, ymax=ymax,
                                              absx=absx)

    def movie_chernbands_vary_param(self, param_type='hlat', param_band_nu=None,
                                    reverse=False, param='percolation_density',
                                    title='Chern index calculation', xlabel=None, ymax=None, absx=False):
        """Plot wavenumber as x axis varies, with bands colored by chern indices. The parameter which is varying
        between networks is varying between FRAMES.
        If there are multiple chern calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        param_type : str ('hlat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a haldanelattice parameter (same lattice,
            different physics)
        reverse : bool
            Compute cherns for HaldaneLattice instances in self.haldane_collection in reverse order
        param : string
            string specifier for HaldaneLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))
        ymax : float
            The upper ylimit in the plot
        absx : bool
            Take the absolute value of x
        """
        kccollfns.movie_chernbands_vary_param(self, param_type=param_type, param_band_nu=param_band_nu,
                                               reverse=reverse, param=param, title=title, xlabel=xlabel, ymax=ymax,
                                               absx=absx)

    def movie_berrybands_vary_param(self, param_type='hlat', param_band_nu=None,
                                    reverse=False, param='percolation_density',
                                    title=None, absx=False, ngrid=100):
        """Plot berry curvature as heatmap on the BZ. Each frame is a different value of the parameter being varied
        between networks.

        Parameters
        ----------
        param_type : str ('hlat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a haldanelattice parameter (same lattice,
            different physics)
        reverse : bool
            Compute cherns for HaldaneLattice instances in self.haldane_collection in reverse order
        param : string
            string specifier for HaldaneLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))
        ymax : float
            The upper ylimit in the plot
        absx : bool
            Take the absolute value of x
        """
        kccollfns.movie_berrybands_vary_param(self, param_type=param_type, reverse=reverse, param=param,
                                               title=title, absx=absx, ngrid=ngrid)

    def movie_freqbands_vary_param(self, param_type='hlat', param_band_nu=None,
                                   reverse=False, param='percolation_density',
                                   title=None, absx=False, ngrid=100):
        """Plot berry curvature as heatmap on the BZ. Each frame is a different value of the parameter being varied
        between networks.

        Parameters
        ----------
        param_type : str ('hlat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a haldanelattice parameter (same lattice,
            different physics)
        reverse : bool
            Compute cherns for HaldaneLattice instances in self.haldane_collection in reverse order
        param : string
            string specifier for HaldaneLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))
        ymax : float
            The upper ylimit in the plot
        absx : bool
            Take the absolute value of x
        """
        kccollfns.movie_freqbands_vary_param(self, param_type=param_type, reverse=reverse, param=param,
                                              title=title, absx=absx, ngrid=ngrid)


if __name__ == '__main__':
    '''running example for calculating a collection of kspace cherns on haldane lattices for various samples
    
    Example usage:
    python ./chern/kchern_collection.py -N 1

    '''
    import argparse
    import socket
    import lepm.lattice_class as lattice_class
    from lepm.haldane.haldane_lattice_class import HaldaneLattice
    import lepm.lattice_elasticity as le
    import lepm.brillouin_zone_functions as bzfns
    import lepm.stringformat as sf
    import copy
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create KChern class for HaldaneLattice class instance,' +
                                                 ' with options to save or compute attributes of the class.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-silicene_phases', '-silicene_phases',
                        help='Compute the chern numbers using the berry curvature for patterned hoppings',
                        action='store_true')
    parser.add_argument('-calc_chern', '-calc_chern', help='Compute the chern numbers using the berry curvature',
                        action='store_true')
    parser.add_argument('-pin2hdf5', '--pin2hdf5',
                        help='Enforce saving pinning as hdf5 rather than as txt', action='store_true')
    parser.add_argument('-density', '--density', help='The density of points to sample in the BZ', type=int, default=40)
    # examples of OmKspec:
    #   'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
    #      'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') +\
    #      'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n') --> dicing lattice
    #   'Vpin0p10' --> a specific, single random configuration of potentials
    #   'Vpin0p10theta0p00phi0p00' --> a specific, single random configuration of potentials with twisted BCs
    parser.add_argument('-twistbcs', '--twistbcs',
                        help='Examine Hall conductance as berry curvature associated with state ' +
                             '|alpha(theta_twist, phi_twist)>', action='store_true')
    parser.add_argument('-twiststrip', '--twiststrip',
                        help='Examine spectrum as function of twist angle theta_twist with states |alpha(theta_twist>',
                        action='store_true')
    parser.add_argument('-twistmodes', '--twistmodes',
                        help='Examine normal modes as function of twist angle theta_twist, following each state',
                        action='store_true')
    parser.add_argument('-twistmodes_spiral', '--twistmodes_spiral',
                        help='Examine normal modes as function of twist angle theta_twist, following ' +
                             'one state through gap from one end to the other',
                        action='store_true')
    parser.add_argument('-nrungs', '--twistmodes_nrungs',
                        help='If twistmodes_spiral, the number of times to wind around theta = (0, 2pi) through ' +
                             'the gap', type=int, default=3)
    parser.add_argument('-startfreq', '--twistmodes_startfreq',
                        help='If twistmodes_spiral, begin winding with a mode closest to this frequency, ' +
                             'winding around theta = (0, 2pi) through the gap', type=float, default=2.1)
    parser.add_argument('-thres0', '--twistmodes_thres0',
                        help='If twistmodes_spiral, Look for adjacent modes with freqs within this range of prev freq',
                        type=float, default=-1)
    parser.add_argument('-springax', '--twistmodes_springax',
                        help='If twistmodes_spiral, include a panel shoing the twisted bc rotating',
                        action='store_true')
    parser.add_argument('-edgelocalization', '--edgelocalization',
                        help='Check localization properties to the boundary of the sample', action='store_true')

    # Geometry and physics arguments
    parser.add_argument('-pureimNNN', '--pureimNNN', help='Make NNN hoppings purely imaginary', action='store_true')
    parser.add_argument('-t2angles', '--t2angles', help='Make NNN hoppings based on bond angles', action='store_true')
    parser.add_argument('-hexNNN', '--hexNNN', help='Ignore NNN hoppings in polygons other than hexagons',
                        action='store_true')
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-N', '--N',
                        help='Mesh width AND height, in number of lattice spacings (leave blank to spec separate dims)',
                        type=int, default=-1)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=6)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=6)
    parser.add_argument('-NP', '--NP_load', help='Number of particles in mesh, overwrites N, NH, and NV.',
                        type=int, default=0)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                    'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hexagonal')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-basis', '--basis', help='basis for computing eigvals', type=str, default='psi')
    parser.add_argument('-Omk', '--Omk', help='Spring frequency', type=str, default='-1.0')
    parser.add_argument('-Omg', '--Omg', help='Pinning frequency', type=str, default='-1.0')
    parser.add_argument('-bl0', '--bl0', help='rest length for all springs, if specified', type=float, default=-5000)
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-thetatwist', '-thetatwist',
                        help='Angle in units of pi radians for first twisted Boundary condition',
                        type=float, default=0.)
    parser.add_argument('-phitwist', '-phitwist',
                        help='Angle in units of pi radians for second twisted Boundary condition',
                        type=float, default=0.)
    parser.add_argument('-kr_max', '--kr_max', help='Upper bound for magnitude of k in calculating S(k)', type=int,
                        default=30)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-verbose', '--verbose', help='Print liberally to command line', action='store_true')
    parser.add_argument('-nice_plot', '--nice_plot', help='Output nice pdf plots of lattice', action='store_true')

    # For loading and coordination
    parser.add_argument('-LLID', '--loadlattice_number',
                        help='If LT=hyperuniform/isostatic, selects which lattice to use', type=str, default='01')
    parser.add_argument('-LLz', '--loadlattice_z', help='If LT=hyperuniform/isostatic, selects what z index to use',
                        type=str, default='001')
    parser.add_argument('-source', '--source',
                        help='Selects who made the lattice to load, if loaded from source (ulrich, hexner, etc)',
                        type=str, default='hexner')
    parser.add_argument('-cut_z', '--cut_z',
                        help='Declare whether or not to cut bonds to obtain target coordination number z',
                        type=bool, default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
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
    parser.add_argument('-kicksz', '--kicksz',
                        help='Average of log of kick magnitudes for loading randorg_gammakick pointsets.' +
                             'This sets the scale of the powerlaw kicking procedure',
                        type=float, default=-1.50)
    parser.add_argument('-OmKspec', '--OmKspec', help='string specifier for OmK bond frequency matrix',
                        type=str, default='')

    # Global geometric params
    parser.add_argument('-nonperiodic', '--openBC', help='Enforce open (non periodic) boundary conditions',
                        action='store_true')
    parser.add_argument('-periodic_strip', '--periodic_strip',
                        help='Enforce strip periodic boundary condition in horizontal dim', action='store_true')
    parser.add_argument('-immobile_boundary', '--immobile_boundary', help='Affix the particles on the boundary',
                        action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.667')
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph',
                        help='Twist angle for twisted_kagome (max is pi/3) in radians or ' +
                             'opening angle of the accordionized lattices or ' +
                             'percent of system decorated -- used in different contexts',
                        type=float, default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)

    parser.add_argument('-t1', '--t1', help='NN hopping strength', type=str, default='-1.0')
    parser.add_argument('-t2', '--t2', help='NNN hopping strength prefactor (imaginary part)', type=str, default='0.1')
    parser.add_argument('-t2a', '--t2a', help='NNN hopping strength real component', type=str, default='0.0')
    parser.add_argument('-theta_twist', '--theta_twist', help='Twisted phase in x for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-phi_twist', '--phi_twist', help='Twisted phase in y for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-pin', '--pin', help='Pinning energy (on-site)', type=str, default='0.0')
    parser.add_argument('-pinconf', '--pinconf',
                        help='Lattice disorder realization number (0 or greater)', type=int, default=0)
    parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                        default=01)
    parser.add_argument('-intparam', '--intparam',
                        help='Integer-valued parameter for building networks (ex # subdivisions in accordionization)',
                        type=int, default=1)
    parser.add_argument('-thres', '--thres', help='Threshold value for building networks (determining to decorate pt)',
                        type=float, default=1.0)
    parser.add_argument('-skip_gr', '--skip_gr', help='Skip calculation of g(r) correlation function for the lattice',
                        action='store_true')
    parser.add_argument('-skip_gxy', '--skip_gxy',
                        help='Skip calculation of g(x,y) 2D correlation function for the lattice', action='store_true')
    parser.add_argument('-skip_sigN', '--skip_sigN', help='Skip calculation of variance_N(R)', action='store_true')
    parser.add_argument('-fancy_gr', '--fancy_gr',
                        help='Perform careful calculation of g(r) correlation function for the ENTIRE lattice',
                        action='store_true')
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
    # deformed kagome params
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    z = 0.0

    print 'theta = ', theta
    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    if socket.gethostname()[0:6] == 'midway':
        print '\n\nWe are on Midway!\n\n\n\n'
        rootdir = '/home/npmitchell/scratch-midway/'
        cprootdir = '/home/npmitchell/scratch-midway/'
    elif socket.gethostname()[0:10] == 'nsit-dhcp-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
    elif 'Messiaen' in socket.gethostname():
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
        if not os.path.isdir(cprootdir):
            cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
    elif socket.gethostname()[0:5] == 'cvpn-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
    else:
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
          'phi_lattice': args.phi_lattice,
          'delta_lattice': args.delta_lattice,
          'theta': theta,
          'eta': eta,
          'x1': x1,
          'x2': x2,
          'x3': x3,
          'z': z,
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
          'periodicBC': not args.openBC,
          'periodic_strip': args.periodic_strip,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'eta_alph': args.eta_alph,
          'origin': np.array([0., 0.]),
          't1': float((args.t1).replace('n', '-').replace('p', '.')),
          't2': float((args.t2).replace('n', '-').replace('p', '.')),
          't2a': float((args.t2a).replace('n', '-').replace('p', '.')),
          'pin': float((args.pin).replace('n', '-').replace('p', '.')),
          'pinconf': int(args.pinconf),
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta': args.ABDelta,
          'thres': args.thres,
          'pureimNNN': args.pureimNNN,
          't2angles': args.t2angles,
          'theta_twist': args.theta_twist,
          'phi_twist': args.phi_twist,
          }

    cp = {'density': args.density,  # number of points per area in BZ for calculation.
          'rootdir': cprootdir,
          'basis': args.basis,
          }

    if args.calc_cherns_ABtrans:
        '''Compute the phase diagram of varying both opening angle delta and inversion symmetry ABDelta.
        Example usage:
        python ./chern/kchern_collection.py -LT hexagonal -N 1 -calc_ABtrans -density 4000 -paramV 0:0.05:0.5 -verbose
        python ./chern/kchern_collection.py -LT hexagonal -N 1 -calc_ABtrans -density 400 -paramV 0:0.05:0.5 -verbose
        python ./chern/kchern_collection.py -LT hexagonal -N 1 -calc_ABtrans -density 40 -paramV 0:0.05:0.5 -verbose
        '''
        # loop for values in the lattice paramters
        kcolls = []

        # define the lattice parameter values
        lpvals = np.arange(0.5, 1.3, 0.02)
        # lpvals = np.arange(0.7, 1.3, 0.05)
        # lpvals = np.hstack((np.array([2./3.]), lpvals))

        lpmaster = copy.deepcopy(lp)
        if args.lpparam_reverse:
            lpvals = lpvals[::-1]

        for lpval in lpvals:
            # Create hlat
            lp = copy.deepcopy(lpmaster)
            lp['delta_lattice'] = '{0:0.3f}'.format(lpval)
            lp['delta'] = lpval * np.pi

            meshfn = le.find_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.load()
            hlat = HaldaneLattice(lat, lp)

            # Collect HaldaneLattice instances
            gc = haldane_collection.HaldaneCollection()
            gc.add_haldane_lattice(hlat)

            print 'Creating chern collection from single-lattice haldane_collection...'
            kcoll = KChernCollection(gc, cp=cp)

            if args.paramVdtype == 'str':
                if args.hlatparam == 'OmKspec':
                    OmKspec_list = hlatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                    print 'OmKspec_list = ', OmKspec_list
                    paramV = np.array(OmKspec_list)
                else:
                    raise RuntimeError('Need to make exception for this paramV option')
            else:
                paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)

            kcoll.calc_cherns_vary_hlatparam(args.hlatparam, paramV, reverse=args.hlatparam_reverse,
                                             verbose=True, save_png=False)
            kcolls.append(kcoll)
            kcoll.plot_cherns_vary_param(param=args.hlatparam, param_type='hlat')

            # Also save movie of the band structure for each phi as ABDelta is varied
            kcoll.plot_chernbands_vary_param(param=args.hlatparam, param_type='hlat', param_band_nu=None)

            # print 'finished one sequence --> exiting now'
            # sys.exit()

        deltagrid = []
        abdgrid = []
        cherngrid = []
        for kcoll in kcolls:
            abds = []
            deltas = []
            cherns = []
            print 'kcoll = ', kcoll
            for hlatname in kcoll.cherns:
                chernii = kcoll.cherns[hlatname]
                print 'chernii = ', chernii
                if len(chernii) == 1:
                    chernii = chernii[0]
                else:
                    raise RuntimeError('More than one chern in this list!')
                # get ab value from hlat.lp
                abds.append(chernii.haldane_lattice.lp['ABDelta'])
                deltas.append(chernii.haldane_lattice.lp['delta'])
                # grab the chern number of the top band
                cherns.append(np.real(chernii.chern['chern'][-1]))

            deltas = np.array(deltas)
            abds = np.array(abds)
            cherns = np.array(cherns)
            if len(abdgrid) == 0:
                deltagrid = deltas
                abdgrid = abds
                cherngrid = cherns
            else:
                deltagrid = np.vstack((deltagrid, deltas))
                abdgrid = np.vstack((abdgrid, abds))
                cherngrid = np.vstack((cherngrid, cherns))

        plt.close('all')
        fig, ax, cax = leplt.initialize_1panel_cbar_cent()
        lecmaps.register_colormaps()
        cmap = 'bbr0'
        # deltagrid.reshape((len(abds), -1))
        # abdgrid.reshape((len(abds), -1))
        # cherngrid.reshape((len(abds), -1))
        # leplt.plot_pcolormesh_scalar(deltagrid, abdgrid, cherngrid, outpath=None, cmap=cmap, vmin=-1.0, vmax=1.0)
        leplt.plot_pcolormesh(deltagrid / np.pi, abdgrid, cherngrid, 100, ax=ax,
                              make_cbar=False, cmap=cmap, vmin=-1.0, vmax=1.0)
        # ax.scatter(deltagrid, abdgrid, c=cherngrid, cmap=cmap, vmin=-1.0, vmax=1.0)
        sm = leplt.empty_scalar_mappable(vmin=-1.0, vmax=1.0, cmap=cmap)
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_label(label=r'Chern number, $\nu$', labelpad=-35)
        cb.set_ticks([-1, 0, 1])
        ax.set_xlabel(r'Lattice deformation angle, $\delta/\pi$')
        ax.set_ylabel(r'Inversion symmetry breaking, $\Delta$')
        specstr = '_delta' + sf.float2pstr(np.min(deltagrid)) + '_' + sf.float2pstr(np.max(deltagrid))
        specstr += '_abd' + sf.float2pstr(np.min(abdgrid)) + '_' + sf.float2pstr(np.max(abdgrid))
        specstr += '_ndeltas{0:05d}'.format(len(deltas)) + '_nabd{0:05d}'.format(len(abds))
        specstr += '_density{0:07d}'.format(cp['density'])
        plt.savefig(rootdir + 'kspace_cherns_haldane/abtransition' + specstr + '.png')

    if args.calc_berry_varyphi_cantedbrick:
        '''Compute the phase diagram of varying both opening angle delta and inversion symmetry ABDelta.
        To make lattices:
        python run_series.py -pro ./build/make_lattice -opts \
            LT/hexagonal/-N/1/-periodic/-skip_haldaneDOS/-skip_polygons/-delta/1.00 -var phi n0.5:0.05:0.

        Example usage:
        python ./chern/kchern_collection.py-LT hexagonal -N 1 -calc_berry_varyphi_cantedbrick -lpparam phi \
                -density 100 -paramV n0.45:0.05:0.5 -verbose
        '''
        # Specify the canted bricklayer geometry
        lp['delta_lattice'] = '1.000'
        lp['delta'] = np.pi
        eps = 1e-9

        print 'Creating chern collection from single-lattice haldane_collection...'
        gc = haldane_collection.HaldaneCollection()

        # create array for parameter values to pass to lp
        if args.paramVdtype == 'str':
            if args.hlatparam == 'OmKspec':
                OmKspec_list = hlatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                print 'OmKspec_list = ', OmKspec_list
                paramV = np.array(OmKspec_list)
            else:
                raise RuntimeError('Need to make exception for this paramV option')
        else:
            paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)

        # reverse the lattice parameter values for tile angle phi if necessary
        lpmaster = copy.deepcopy(lp)
        if args.lpparam_reverse:
            lpvals = lpvals[::-1]

        for lpval in paramV:
            print 'lpval = ', lpval
            # Create hlat
            lp = copy.deepcopy(lpmaster)
            lp['phi_lattice'] = '{0:0.3f}'.format(lpval)
            if abs(lpval) < eps:
                lp['phi_lattice'] = '0.000'
            lp['phi'] = lpval * np.pi

            meshfn = le.find_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.load()
            hlat = HaldaneLattice(lat, lp)

            # Collect HaldaneLattice instances
            gc.add_haldane_lattice(hlat)

        kcoll = KChernCollection(gc, cp=cp)
        kcoll.calc_cherns_vary_lpparam(args.lpparam, paramV, reverse=args.hlatparam_reverse,
                                       verbose=True, save_png=False)

        # Also save movie of the band structure for each phi as ABDelta is varied
        kcoll.movie_chernbands_vary_param(param_type='lat', param_band_nu=None,
                                          reverse=False, param=args.lpparam,
                                          title='Chern index calculation', xlabel=None, ymax=None, absx=False)

        kcoll.movie_berrybands_vary_param(param_type='lat', param_band_nu=None,
                                          reverse=False, param=args.lpparam, title=None, absx=False, ngrid=500)

        # Plot heatmaps of the band energies as a function of the varying parameter
        kcoll.movie_freqbands_vary_param(param_type='lat', param_band_nu=None,
                                         reverse=False, param=args.lpparam, title=None, absx=False, ngrid=500)

    if args.calc_berry_varylatparam:
        '''Example usage:
        python ./chern/kchern_collection.py-LT hexagonal -N 1 -calc_berry_varylatparam -lpparam delta \
            -density 400 -paramV 0.6:0.05:1.2 -verbose -lpparam delta

        To make lattices:
        python run_series.py -pro ./build/make_lattice -opts LT/hexagonal/-N/1/-periodic/-skip_gyroDOS/-skip_polygons -var delta 0.6:0.05:1.2

        '''
        print 'Creating chern collection from single-lattice haldane_collection...'
        gc = haldane_collection.HaldaneCollection()

        # create array for parameter values to pass to lp
        if args.paramVdtype == 'str':
            if args.hlatparam == 'OmKspec':
                OmKspec_list = hlatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                print 'OmKspec_list = ', OmKspec_list
                paramV = np.array(OmKspec_list)
            else:
                raise RuntimeError('Need to make exception for this paramV option')
        else:
            paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)

        # reverse the lattice parameter values for tile angle phi if necessary
        lpmaster = copy.deepcopy(lp)
        if args.lpparam_reverse:
            lpvals = lpvals[::-1]

        for lpval in paramV:
            print 'lpval = ', lpval
            # Create hlat
            lp = copy.deepcopy(lpmaster)
            if args.lpparam == 'delta':
                lp['delta_lattice'] = '{0:0.3f}'.format(lpval)
                lp['delta'] = lpval * np.pi
            else:
                lp[args.lpparam] = lpval

            meshfn = le.find_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.load()
            hlat = HaldaneLattice(lat, lp)

            # Collect HaldaneLattice instances
            gc.add_haldane_lattice(hlat)

        kcoll = KChernCollection(gc, cp=cp)
        kcoll.calc_cherns_vary_lpparam(args.lpparam, paramV, reverse=args.hlatparam_reverse,
                                       verbose=True, save_png=False)

        # Also save movie of the band structure for each phi as ABDelta is varied
        # kcoll.movie_chernbands_vary_param(param_type='lat', param_band_nu=None,
        #                                   reverse=False, param=args.lpparam,
        #                                   title='Chern index calculation', xlabel=None, ymax=None, absx=False)
        #
        # kcoll.movie_berrybands_vary_param(param_type='lat', param_band_nu=None,
        #                                   reverse=False, param=args.lpparam, title=None, absx=False, ngrid=500)

        # Plot heatmaps of the band energies as a function of the varying parameter
        kcoll.movie_freqbands_vary_param(param_type='lat', param_band_nu=None,
                                         reverse=False, param=args.lpparam, title=None, absx=False, ngrid=500)

    if args.tune_junction:
        '''Increase the coupling between a triad of bonds and visualize evolution of spectrum. To tune the Gamma point
        spectrum only, see
        $ python haldane_lattice_class.py -LT hexjunctiontriad -tune_junction -N 1 \
            -OmKspec unionn0p00in0p2000 -alph 0.1 -periodic

        Example network creation
        python ./build/make_lattice.py -LT hexjunctiontriad -alph 0.1 -periodic

        Example usage:
        python ./chern/kchern_collection.py -LT hexjunctiontriad -tune_junction -N 1 -OmKspec unionn0p00in0p1000 -alph 0.1
        python ./chern/kchern_collection.py -LT hexjunction2triads -tune_junction -N 1 -OmKspec unionn0p00in0p1000 -alph 0.1 -density 100
        python ./chern/kchern_collection.py -LT hexjunction2triads -tune_junction -N 1 -OmKspec unionn0p00in0p1000 -Omgspec dab0p3union0p200 -alph 0.1 -density 20

        '''
        if lp['LatticeTop'] in ['hexjunctiontriad', 'hexjunction2triads']:
            kvals = -np.unique(np.round(np.logspace(-2, 2., 51), 2))  # [::-1]
            dist_thres = lp['OmKspec'].split('union')[-1].split('in')[-1]
            lpmaster = copy.deepcopy(lp)
            gcoll = haldane_collection.HaldaneCollection()
            lat = lattice_class.Lattice(lp)
            lat.load()

            for kval in kvals:
                lp = copy.deepcopy(lpmaster)
                lp['OmKspec'] = 'union' + sf.float2pstr(kval, ndigits=3) + 'in' + dist_thres
                # lat = lattice_class.Lattice(lp)
                hlat = HaldaneLattice(lat, lp)
                gcoll.add_haldane_lattice(hlat)

            print 'kchern_haldane_collection: creating KChernCollection...'
            kccoll = KChernCollection(gcoll, cp=cp)
            # Collect the cherns/bands
            print 'kchern_haldane_collection: collecting cherns...'
            kccoll.collect_cherns_vary_hlatparam(param='OmKspec')
            # Plot the cherns/bands with varying union bond strength
            print 'kchern_haldane_collection: plotting cherns/bands...'
            kccoll.plot_chernbands_vary_param(param_type='hlat', param='OmKspec', ymax=4.0, absx=True)
            kccoll.movie_chernbands_vary_param(param_type='hlat', param='OmKspec', ymax=4.0, absx=True)

            # eigvals = np.vstack(eigvals)
            # # print 'np.shape(eigvals)= ', np.shape(eigvals)
            # fig, ax = leplt.initialize_1panel_centered_fig()
            # ax.set_ylim(0, 4)
            # ax.set_ylabel('frequency, $\omega$')
            # ax.set_xlabel('coupling constant, $k$')
            # ax.set_title('Junction spectrum')
            # fn = 'junction_chernspectrum_OmKspec_union_maxk_' + sf.float2pstr(np.max(kval)) + '.pdf'
            # plt.savefig(dio.prepdir(lat.lp['meshfn']) + fn)
        else:
            raise RuntimeError('Have not coded for this lattice topology yet')

