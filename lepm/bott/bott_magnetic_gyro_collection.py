import numpy as np
import lepm.lattice_elasticity as le
import lepm.structure as lestructure
import lepm.plotting.colormaps as cmaps
import lepm.plotting.science_plot_style as sps
import lepm.bott.bott_magnetic_gyro_functions as mbgfns
import lepm.gyro_lattice_functions as glatfns
import lepm.bott.bott_magnetic_gyro as magnetic_bott_gyro
import lepm.bott.bott_collection_functions as bcollfns
import lepm.magnetic_gyro_functions as mglatfns
import lepm.magnetic_gyro_collection as magnetic_gyro_collection
import socket
import matplotlib.pyplot as plt
import argparse
import lepm.lattice_class as lattice_class
import lepm.magnetic_gyro_lattice_class as magnetic_gyro_lattice_class
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import lepm.plotting.bott_plotting_functions as bpfns
import lepm.plotting.bott_collection_plotting_functions as bcollpfns
import lepm.stringformat as sf
import glob
import copy
import lepm.dataio as dio
import pdb
import pdb
import lepm.bott.bott_gyro_functions as bgfns
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl
import os
import lepm.magnetic_gyro_spectral_data as magnetic_gyro_spectral_data
import sys


'''
Collections of Bott Index measurements made via the realspace method of Loring and Hastings 2010.
Note: methods with 'vary_glatparam' cycle through the same geometric lattice with different physics on that lattice,
      while methods with 'vary_lpparam' cycle through different geometries of networks.
'''


class BottMagneticGyroCollection:
    """Create a collection of bott measurements for gyroscopic spring networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.botts is a list of tuples of dicts: self.botts = [bott1, bott2,...]
    where bott is a class with attributes cp, bott_finsize array.
        cp : dict
            keys : str
                'meshfn', 'omegac', 'poly_offset', 'ksize_frac_arr', 'regalph', 'regbeta', 'reggamma', 'polyT',
                'poly_offset'
        bott : float
            contains [Nreg1, ksize_frac, ksize, ksys_size (note this is 2*NP_summed), ksys_frac, nu
            for Chern calculation]

    Attributes
    ----------
    self.mgyro_collection : list of gyro_lattice instances
        list of instances of lattice_class.lattice() corresponding to each network
    self.meshfns : list of strings
        string paths to the location of the gyro_lattices in the mgyro_collections
    self.botts : dict
        keys are glat_names (strings), values are lists of MagneticGyroBott instances
    self.glat_names : list of strings
        the names of the gyro_lattices in the bott collection
    """
    def __init__(self, mgyro_collection, cp=None):
        """Create an instance of a lattice_collection."""
        self.mgyro_collection = mgyro_collection
        self.meshfns = mgyro_collection.meshfns
        self.botts = {}
        if cp is None:
            cp = {}
        if 'omegac' not in cp:
            cp['omegac'] = np.array([2.25])
        if 'basis' not in cp:
            cp['basis'] = 'XY'

        self.cp = cp
        self.glat_names = []
        print 'instantiated MagneticGyroBottCollection with cp = ', self.cp

    def bott_is_saved(self, glat, cp=None, verbose=False, enforcehdf5=False):
        if verbose:
            print '\nChecking if bott is already saved...'
        if cp is None:
            cp = self.cp
        bott = magnetic_bott_gyro.MagneticGyroBott(glat, cp)
        if verbose:
            print 'First checking hdf5: ', bott.cp['h5fn']
            print '... with subgroup: ', bott.cp['h5subgroup']

        inh5 = bott.bott_in_hdf5()
        if inh5:
            return True
        else:
            if not enforcehdf5 and glob.glob(bott.cp['cpmeshfn'] + 'bott.txt'):
                return True
            else:
                return False

    def add_bott(self, glat, cp=None, proj=None, attribute_evs=False, verbose=False):
        """Add a MagneticGyroBott instance to the MagneticGyroBottCollection, and append the glat_name to
        self.glat_names if not already in that list"""
        if cp is None:
            cp = self.cp
        bott = magnetic_bott_gyro.MagneticGyroBott(glat, cp)
        bott.get_bott(proj=proj, attribute_evs=attribute_evs, verbose=verbose)
        glat_name = glat.lp['meshfn'] + glat.lp['meshfn_exten']
        if glat_name not in self.botts:
            self.botts[glat_name] = []

        self.botts[glat_name].append(bott)
        if glat_name not in self.glat_names:
            self.glat_names.append(glat_name)
        return bott

    def get_botts(self, cp=None, verbose=False, omegac_glat_func=None, omegac_glatparam=None, reverse=False):
        """Retrieve botts for each glat in self.mgyro_collection.gyro_lattices matching the supplied/attributed cp

        Parameters
        ----------
        cp : none or dict
            bott parameters dictionary
        verbose : bool
            Print more statements to commandline out
        omegac_glat_func : None or function
            function to find omegac based on features of the gyro_lattice (for example, a function to get the middle of
            a band gap based on the deformation angle of a lattice). Must be used in conjunction with glatparam used as
            input to determine omegac value.
        glatparam : None or any type (key for lp)
            input for function omegac_glat_func to determine omegac (which is output of omegac_glat_func(glatparam).
        """
        if cp is None:
            cp_orig = self.cp
        else:
            cp_orig = cp

        if reverse:
            glatstodo = self.mgyro_collection.mgyro_lattices[::-1]
        else:
            glatstodo = self.mgyro_collection.mgyro_lattices
        for glat in glatstodo:
            cp = copy.deepcopy(cp_orig)
            cp.pop('cpmeshfn', None)
            # print '\n\n\nbott_collection: get_botts: glat = ', glat.lp['meshfn']
            # if verbose:
            print 'Adding bott for glat =', glat.lp['meshfn']
            # The optional omegac_glat_func function can map the current glatparam to a desired cutoff freq --
            # for ex, if the middle of the gap is a function of the gyrolattice parameter that is varying
            if omegac_glat_func is not None:
                    cp['omegac'] = omegac_glat_func(glat.lp[omegac_glatparam])
                    if len(cp['omegac']) > 0:
                        for ii in range(len(cp['omegac'])):
                            cpii = copy.deepcopy(cp)
                            cpii['omegac'] = cp['omegac'][ii]
                            self.add_bott(glat, cpii, verbose=verbose)
                    else:
                        raise RuntimeError('Supplied omegac_glat_func returned empty array')
            else:
                if len(cp['omegac']) > 0:
                    for ii in range(len(cp['omegac'])):
                        cpii = copy.deepcopy(cp)
                        cpii['omegac'] = cp['omegac'][ii]
                        self.add_bott(glat, cpii, verbose=verbose)
                else:
                    self.add_bott(glat, cp=cp, verbose=verbose)

    def calc_botts_omegac(self, omegacV=None, reverse=False):
        """For each GyroLattice instance in this bott collection, compute botts over the list of omegac values (cutoff
        frequencuies for the projector)"""
        if omegacV is None:
            omegacV = self.cp['omegac']
        else:
            self.cp['omegac'] = omegacV

        if reverse:
            omegacV = omegacV[::-1]

        # Add bott calculation for each omegac value
        print 'going to get/calc/add botts for mglat = ', self.mgyro_collection.mgyro_lattices
        for glat in self.mgyro_collection.mgyro_lattices:
            print 'going to get/calc/add botts for omegac values = ', omegacV
            for omegac in omegacV:
                cp = copy.deepcopy(self.cp)
                print 'omegac = ', omegac
                cp['omegac'] = omegac
                self.add_bott(glat, cp, attribute_evs=True)

        return self.botts

    def get_maxbott_omegac(self):
        """Considering all omegac in the bcoll, get the extremal bott value obtained for ksize_frac=singleksz_frac

        Parameters
        ----------
        """
        print 'Obtaining extremal bott over all omegac values'
        nuoutlist = []
        glat_name_list = []
        for glat_name in self.botts:
            print 'glat_name = ', glat_name
            val = 0.
            # get val when ksys_frac = 0.5
            for bottii in self.botts[glat_name]:
                # Grab small, medium, and large circles
                newnu = bottii.bott
                if np.abs(newnu) > np.abs(val):
                    val = newnu

            nuoutlist.append(val)
            glat_name_list.append(glat_name)

        return glat_name_list, nuoutlist

    def get_botts_omegac(self, omegacV=None, reverse=False):
        self.calc_botts_omegac(omegacV=omegacV, reverse=reverse)
        return self.botts

    def calc_botts_vary_glatparam(self, glatparam, paramV, reverse=False, verbose=False, auto_omegac=True,
                                  save_eigv=False, force_hdf5_eigv=True):
        """For a single Lattice instance, compute botts for many MagneticGyroLattice instances, each with a varied
        MagneticGyroLattice parameter.
        When supplied, a single gyro_lattice instance is required for each physical lattice, but each new glat
        (same lattice, different lp) will be appended to self.mgyro_collection.gyro_lattices when lp is updated

        Parameters
        ----------
        glatparam : str key for lp dictionary
            the string specifier for the parameter to change FOR THE SAME LATTICE, for each value in paramV,
            for instance ABDelta
        paramV : list or 1d numpy array
            the values to assign to glat.lp[glatparam] for each bott calculation
        reverse : bool
            reverse the order of paramV in which to compute
        verbose : bool
            print intermediate output
        """
        # prepare xy locations as fraction of system size
        print 'bottmagnetic_gyrocoll: Looping over all networks in the gyro collection. ' + \
              'For each one, cycling through glat lpparams in paramV..'
        print('--> glats to do:', [key for key in self.mgyro_collection.mgyro_lattices])
        glatstodo = [key for key in self.mgyro_collection.mgyro_lattices]

        for glat in glatstodo:
            print '\n\nbott_collection: Getting botts for GyroLattice ', glat,  '...'
            print ' of name ', glat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)
            proj = None
            print 'paramV = ', paramV
            if reverse:
                paramtodo = paramV[::-1]
            else:
                paramtodo = paramV

            lp_master = copy.deepcopy(glat.lp)
            lat = copy.deepcopy(glat.lattice)
            for val in paramtodo:
                cp = copy.deepcopy(self.cp)
                lp = copy.deepcopy(lp_master)
                lp[glatparam] = val
                print 'glatparam = ', glatparam
                # print 'lp[OmKspec]=', lp['OmKspec']
                mglat = magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
                # Determine cp[omegac] from lp if not forced to be invariant
                if auto_omegac:
                    cp['omegac'] = magnetic_gyro_spectral_data.gapfreq_magnetic_glat(mglat)

                # print 'mglat.lp[meshfn]=', mglat.lp['meshfn']
                self.mgyro_collection.mgyro_lattices.append(mglat)

                # Test whether projector should be constructed --> if multiple botts are computed with same
                # projector, then its best to compute projector only once.
                if not self.bott_is_saved(mglat, cp=cp):
                    print '\nmgbcoll: bott is not saved, computing projector...'
                    print '... for cpmeshfn = ', cp['cpmeshfn']
                    proj = mbgfns.calc_small_projector(mglat, self.cp['omegac'][0], save_eigv=save_eigv,
                                                       force_hdf5_eigv=force_hdf5_eigv)

                # Simply add the one bott to the collection
                self.add_bott(mglat, cp=cp, proj=proj, verbose=verbose)

        print 'self.mgyro_collection.mgyro_lattices = ', self.mgyro_collection.mgyro_lattices
        return self.botts

    def plot_botts_omegac(self, filename='omegac_overlay', savedir=None, vmin=-1.0, vmax=1.0,
                          title='Bott Index calculation', xlabel=r'Cutoff frequency $\omega_c$'):
        """Plot a bott index overlaying DOS

        Parameters
        ----------

        """
        plt.close('all')
        for glat_name in self.botts:
            # Grab a pointer to the gyro_lattice
            glat = self.botts[glat_name][0].gyro_lattice

            print 'Opening glat_name = ', glat_name

            # Build omegacV
            omegacv = np.zeros(len(self.botts[glat_name]))
            nuv = np.zeros((len(self.botts[glat_name]), 1))
            # print 'nuM = ', nuM
            for ind in range(len(self.botts[glat_name])):
                print 'ind = ', ind
                cp_ii = self.botts[glat_name][ind].cp
                omegacv[ind] = cp_ii['omegac']
                nuv[ind] = self.botts[glat_name][ind].bott
                # print 'nuM = ', nuM

            print 'omegacv = ', omegacv

            # Save the plot
            savedir = bgfns.get_cmeshfn(self.botts[glat_name][ind].mgyro_lattice.lp,
                                        rootdir=self.botts[glat_name][ind].cp['rootdir'])
            dio.ensure_dir(savedir)

            print 'saving plot to ' + savedir + filename
            bpfns.plot_spectrum_with_dos(glat, omegacv, nuv, xlabel=xlabel, title=title)

            plt.savefig(savedir + filename + '.pdf')
            plt.savefig(savedir + filename + '.png', dpi=300)
            plt.clf()

    def plot_bott_spectrum_on_axis(self, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, **kwargs):
        return bcollpfns.plot_bott_spectrum_on_axis(self, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, **kwargs)

    def collect_botts_vary_lpparam(self, param='percolation_density', omegac_glat_func=None, omegac_glatparam=None,
                                   reverse=False):
        """Plot bott indices versus the parameter which is varying between networks.
        If there are multiple bott calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all bott calculations on a particular gyro_lattice.

        """
        if self.botts == {}:
            self.get_botts(omegac_glat_func=omegac_glat_func, omegac_glatparam=omegac_glatparam, reverse=reverse)

        params = []
        nuList = []
        for glat_name in self.botts:
            print 'adding params from glat_name = ', glat_name
            first = True
            for bott in self.botts[glat_name]:
                if first:
                    # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                    # the bott instance
                    params.append(bott.mgyro_lattice.lp[param])
                    print 'mgbcoll: added param for varying glat_param: ', param, ' --> ', bott.mgyro_lattice.lp[param]
                    nu = np.array(bott.bott)
                    print 'nu = ', nu
                    first = False
                else:
                    # Collate the other Chern calculations
                    nu_tmp = bott.bott
                    nu = np.hstack((nu, nu_tmp))

            # If there were multiple, average over them all
            if len(self.botts[glat_name]) > 1:
                # Average over them all
                nu = np.mean(nu, axis=1)

            nuList.append(nu)

        # Build output array
        print 'params = ', params
        print 'nuList = ', nuList
        print 'ksysList = ', ksysList
        for ii in range(len(params)):
            param = params[ii]
            ksys = ksysList[ii]
            nu = nuList[ii]
            if ii == 0:
                sz_param_nu = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
            else:
                add_array = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
                sz_param_nu = np.vstack((sz_param_nu, add_array))

        return sz_param_nu, max_nksize

    def collect_botts_vary_glatparam(self, param='ABDelta', omegac_glat_func=None, omegac_glatparam=None,
                                      reverse=False):
        """Plot bott indices with x axis being the parameter which is varying between networks.
        If there are multiple bott calculations of a particular MagneticGyroLattice, average nu over them all.
        The ksize_frac_array need not be the same for all bott calculations on a particular gyro_lattice. Note that
        this may overlook errors in which the frac_arrays are different now.

        """
        if self.botts == {}:
            self.get_botts(omegac_glat_func=omegac_glat_func, omegac_glatparam=omegac_glatparam, reverse=reverse)

        params = []
        nuList = []
        for glat_name in self.botts:
            print 'adding params from glat_name = ', glat_name
            for bott in self.botts[glat_name]:
                # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                # the bott instance
                print 'bott.mgyro_lattice = ', bott.mgyro_lattice
                params.append(bcollfns.retrieve_param_value(bott.mgyro_lattice.lp[param]))
                nuList.append(bott.bott)

        # Build output array
        print 'params = ', params
        print 'shape(params) = ', np.shape(params)
        print 'nuList = ', nuList
        param_nu = np.dstack((np.array(params), np.array(nuList)))[0]

        return param_nu

    def plot_botts_vary_param(self, param_type='glat', param_nu=None, omegac_glat_func=None, omegac_glatparam=None,
                              reverse=False, param='percolation_density',
                              title='Bott index calculation', xlabel=None, save_data=True):
        """Plot bott indices with x axis being the parameter which is varying between networks.
        If there are multiple bott calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        param_type : str ('glat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a MagneticGyroLattice parameter
            (same lattice, different physics)
        sz_param_nu : None or 'glat' or ('lat' or 'lp' -- last two do same thing)
            string specifier for how to vary params
        omegac_glat_func : None or python function
            function taking lp[omegac_glatparam] as input and supplying numpy array of omegac values
        omegac_glatparam : None or string
            key for glat.lp[omegac_glatparam], to be used to identify numpy array of omegac values
        reverse : bool
            Compute botts for MagneticGyroLattice instances in self.mgyro_collection in reverse order
        param : string
            string specifier for MagneticGyroLattice parameter to vary between networks
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))

        """
        if param_nu is None:
            if param_type == 'lat' or param_type == 'lp':
                param_nu = \
                    self.collect_botts_vary_lpparam(param=param, omegac_glat_func=omegac_glat_func,
                                                    omegac_glatparam=omegac_glatparam, reverse=reverse)
            elif param_type == 'glat':
                param_nu = \
                    self.collect_botts_vary_glatparam(param=param, omegac_glat_func=omegac_glat_func,
                                                      omegac_glatparam=omegac_glatparam, reverse=reverse)
            else:
                raise RuntimeError("param_type argument passed is not 'glat' or 'lat/lp'")

        # Plot it as colormap
        plt.close('all')

        paramV = param_nu[:, 0]
        nu = param_nu[:, 1]
        # order the points
        inds = np.argsort(paramV)
        paramV = paramV[inds]
        nu = nu[inds]

        # Make figure
        FSFS = 8
        Wfig = 90
        x0 = round(Wfig * 0.15)
        y0 = round(Wfig * 0.15)
        ws = round(Wfig * 0.4)
        hs = ws
        wsDOS = ws * 0.3
        hsDOS = hs
        wscbar = wsDOS
        hscbar = wscbar*0.1
        vspace = 8  # vertical space btwn subplots
        hspace = 8  # horizonl space btwn subplots
        tspace = 10  # space above top figure
        fig = sps.figure_in_mm(Wfig, y0 + hs + vspace + hscbar + tspace)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [Wfig * 0.5 - ws * 0.5, y0, ws, hs, ''],    # bott vary glatparam vs ksize
                # [Wfig * 0.5 - wscbar, y0 + hs + vspace, wscbar*2, hscbar, '']  # cbar for bott
              )]

        if xlabel is None:
            xlabel = param.replace('_', ' ')

        # Plot the curve
        ax[0].plot(paramV, nu, '.-')
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(r'Bott index, $B$')

        # Add title
        ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                       horizontalalignment='center', verticalalignment='center')
        # Match axes
        # ax[1].xaxis.set_label_position("top")

        # Save the plot
        if param_type == 'glat':
            outdir = rootdir + 'botts/bott_glatparam/' + param + '/'
            outd_ex = self.botts[self.botts.items()[0][0]][0].mgyro_lattice.lp['meshfn_exten']
            # If the parameter name is part of the meshfn_exten, replace its value with XXX in
            # the meshfnexten part of outdir.
            mfestr = glatfns.param2meshfnexten_name(param)
            print 'outd_ex = ', outd_ex
            if mfestr in outd_ex:
                'param is in meshfn_exten, splitting...'
                # split the outdir by the param string
                od_split = outd_ex.split(mfestr)
                # split the second part by the value of the param string and the rest
                od2val_rest = od_split[1].split('_')
                odrest = od_split[1].split(od2val_rest[0])[1]
                print 'odrest = ', odrest
                print 'od2val_rest = ', od2val_rest
                outd_ex = od_split[0] + param + 'XXX'
                outd_ex += odrest
                print 'outd_ex = ', outd_ex
            else:
                outd_ex += '_' + param + 'XXX'
            print 'outd_ex = ', outd_ex
            # pdb.set_trace()
        elif param_type == 'lat':
            outdir = rootdir + 'botts/bott_lpparam/' + param + '/'
            outd_ex = self.botts[self.botts.items()[0][0]][0].mgyro_lattice.lp['meshfn_exten']

        dio.ensure_dir(outdir)
        fname = outdir + self.mgyro_collection.mgyro_lattices[0].lp['LatticeTop']
        fname += '_bott_' + param + '_Ncoll' + '{0:03d}'.format(len(self.mgyro_collection.mgyro_lattices))
        fname += outd_ex
        print 'saving to ' + fname + '.png'
        plt.savefig(fname + '.png')
        plt.clf()

        # Save the data that is plotted
        if save_data:
            print 'saving data that is plottied: ' + fname + '.txt'
            data = np.dstack((paramV, nu))[0]
            np.savetxt(fname + '.txt', data)

    def collect_bottspectra_vary_glatparam(self, param='percolation_density'):
        """Plot a 2D colormap with x axis being cutoff frequency and y axis being the parameter which is varying
        between networks.
        If there are multiple bott calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all bott calculations on a particular gyro_lattice.
        EDIT THIS TO GET RID OF KSIZES
        """
        if self.botts == {}:
            self.get_botts()

        params = []
        nuList = []
        omegaList = []
        max_nomegac = 0
        for glat_name in self.botts:
            print 'adding params from glat_name = ', glat_name
            nu = []
            omega = []
            # Collate the other bott calculations
            print 'self.botts[glat_name] = ', self.botts[glat_name]
            for bott in self.botts[glat_name]:
                # Add this value of the lattice param being varied
                params.append(bott.mgyro_lattice.lp[param])
                print 'params = ', params
                # Use maximum abs() bott value for this omegac (looking over all ksizes used)
                nu_tmp = bott.bott
                nu.append(nu_tmp)
                omega.append(bott.cp['omegac'])
                print 'len(params) = ', len(params)
                print 'len(nu) = ', len(nu)
                print 'len(omega) = ', len(omega)

            max_nomegac = max(len(omega), max_nomegac)
            omegaList.append(omega)
            nuList.append(nu)

        omegaV = np.array(omegaList).ravel()
        nuV = np.array(nuList).ravel()

        # Build output array
        param_om_nu = np.dstack((np.array(params), omegaV, nuV))[0]

        return param_om_nu, max_nomegac

    def plot_bottspectra_vary_glatparam(self, param_om_nu=None, param='percolation_density',
                                        title='Bott Index calculation', xlabel=r'$\omega_c/\Omega_g$', ylabel=None):
        """Plot a 2D colormap with x axis being cutoff freq and y axis being the parameter which is varying
        between networks.
        If there are multiple bott calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        """
        if param_om_nu is None:
            param_om_nu, max_nomegac = self.collect_bottspectra_vary_glatparam(param=param)

        # Plot it as heatmap
        plt.close('all')

        if ngrid is None:
            ngrid = max(max_nomegac, len(self.botts)) + 7

        om = param_om_nu[:, 1]
        paramV = param_om_nu[:, 0]
        nu = param_om_nu[:, 2]

        # Make figure
        FSFS = 8
        Wfig = 90
        x0 = round(Wfig * 0.15)
        y0 = round(Wfig * 0.12)
        ws = round(Wfig * 0.4)
        hs = ws
        wsDOS = ws * 0.3
        hsDOS = hs
        wscbar = wsDOS
        hscbar = wscbar*0.1
        vspace = 8  # vertical space btwn subplots
        hspace = 8  # horizonl space btwn subplots
        tspace = 10  # space above top figure
        fig = sps.figure_in_mm(Wfig, y0 + hs + vspace + hscbar + tspace)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [Wfig*0.5-ws*0.5, y0 , ws, hs, ''],    # bott vary glatparam vs ksize
                [Wfig*0.5-wscbar, y0 + hs + vspace, wscbar*2, hscbar, '']  # cbar for bott
              )]

        if ylabel is None:
            ylabel = param.replace('_', ' ')
        leplt.plot_pcolormesh(om, paramV, nu, ngrid, ax=ax[0], cax=ax[1], method='nearest',
                              cmap=cmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1.0, vmax=1.0, title='', xlabel=xlabel, ylabel=ylabel,  cax_label=r'$\nu$',
                              cbar_labelpad=3, cbar_orientation='horizontal', ticks=[-1, 0, 1], fontsize=FSFS)

        # Add title
        ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                       horizontalalignment='center', verticalalignment='center')
        # Match axes
        ax[1].xaxis.set_label_position("top")

        # Save the plot
        outdir = rootdir + 'botts/bott_glatparam/' + self.botts + param + '/'
        le.ensure_dir(outdir)
        fname = outdir + 'kiataev_spectra_' + param + '_Ncoll' + \
                '{0:03d}'.format(len(self.mgyro_collection.mgyro_lattices)) + '.png'
        print 'saving to ', fname
        plt.savefig(fname)
        plt.clf()


if __name__ == '__main__':
    '''Perform an example of using the bott_collection class.

    Example usage to create bott spectrum:
    python bott_collection.py -LT kagome -N 16 -vary_omegac -omegac 1.0:0.1:4.0
    python ./bott/bott_collection.py -LT accordionkag -N 5 -shape square -vary_omegac -omegac 1.0:.1:4.0 -intparam 2 -alph 1.0
    python ./bott/bott_collection.py -LT accordionkag -N 6 -shape square -vary_omegac -omegac 1.0:.1:4.0 -intparam 2 -alph 0.85

    Example usage to vary glat parameter:
    python bott_collection.py -vary_lpparam -conf2 02 -LT kagper_hucent
    python bott_collection.py -LT kagper_hucent -perd 0.5 -N 40
    python bott_collection.py -vary_lpparam -glatparam alpha_kagframe -LT hex_kagframe -N 11 -ksize_array_frac 0.0:0.001:1.10 -visua
    python bott_collection.py -vary_glatparam -glatparam ABDelta -LT hucentroid -N 30
    '''
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create collection of bott index calculations.')
    # Script options
    parser.add_argument('-verbose', '--verbose', help='Print verbose output during computation',
                        action='store_true')
    parser.add_argument('-save_as_txt', '--save_as_txt',
                        help='Enforce saving as txt file rather than hdf5', action='store_true')
    parser.add_argument('-bott_collection', '--bott_collection',
                        help='Create collection of Bott indices for many systems of the same LT',
                        action='store_true')
    parser.add_argument('-vary_omegac', '--calc_botts_omegac',
                        help='Compute bott index using a range of omegac values for each glat in the collection' +
                             ' -- ie, create a bott spectrum',
                        action='store_true')
    parser.add_argument('-omegac_reverse', '--omegac_reverse',
                        help='When computing bott number for multiple omegac cutoffs, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-vary_lpparam', '--vary_lpparam',
                        help='Plot bott index calcs for many mgyrolattices build on DISTINCT Lattice instances ' +
                             'with a parameter varying between them',
                        action='store_true')
    parser.add_argument('-spectra_vary_glatparam', '--spectra_vary_glatparam',
                        help='Plot bott index spectra (so for range of cutoff frequencies) for many gyrolattices ' + \
                             'with a parameter varying between them vs omegac', action='store_true')
    parser.add_argument('-vary_glatparam', '--vary_glatparam',
                        help='Plot bott index calcs for many MagneticGyroLattices build on THE SAME Lattice instance ' +
                             'with a parameter varying between them',
                        action='store_true')
    parser.add_argument('-vary_lpparam_vary_glatparam', '--vary_lpparam_vary_glatparam',
                        help='Do multiple bott index calcs for many MagneticGyroLattices build on THE SAME Lattice ' +
                             'instance with a parameter varying between them, then also vary which lattice to use, ' +
                             'by varying glatparam',
                        action='store_true')
    parser.add_argument('-bott_fps', '--bott_fps', help='Framerate for movies of scanning Bott calculation',
                        type=int, default=90)

    # Variation options
    parser.add_argument('-glatparam', '--glatparam',
                        help='String specifier for which parameter is varied across a single gyro lattice in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-glatparam_reverse', '--glatparam_reverse',
                        help='When computing bott index varying a gyrolattice param, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='delta')
    parser.add_argument('-paramV', '--paramV',
                        help='Sequence of values to assign to lp[param] if vary_lpparam is True',
                        type=str, default='0.0:0.1:2.0')
    parser.add_argument('-paramVdtype', '--paramVdtype', help='Description for datatype of paramV',
                        type=str, default='float')
    parser.add_argument('-savepintxt', '--save_pinning_to_txt',
                        help='when creating a new array of pinning frequencies, save to hdf5 instead of txt',
                        action='store_true')

    # Bott parameters
    parser.add_argument('-intrange', '--intrange',
                        help='Consider magnetic couplings only to nth nearest neighbors (if ==2, then NNNs, for ex)',
                        type=int, default=1)
    parser.add_argument('-aol', '--aoverl',
                        help='interparticle distance divided by length of pendulum from pivot to center of mass',
                        type=float, default=0.60)
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector, specified as string with /s',
                        type=str, default='1.5')
    parser.add_argument('-hexbow_omegac', '--hexbow_omegac',
                        help='decide on cutoff freq as middle of the gap for a deformed honeycomb lattice',
                        action='store_true')
    parser.add_argument('-basis', '--basis', help='basis for performing bott calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-title', '--title', help='Title of the Bott calculation saved images', type=str, default='')

    # Geometry and physics arguments
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.01)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-Omk', '--Omk', help='Spring frequency', type=str, default='-1.0')
    parser.add_argument('-Omg', '--Omg', help='Pinning frequency', type=str, default='-1.0')
    parser.add_argument('-bl0', '--bl0', help='rest length for all springs, if specified', type=float, default=-5000)
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-OmKspec', '--OmKspec', help='string specifier for spring frequencies', type=str, default='')
    parser.add_argument('-dice', '--dice', help='Weaken bonds along a grid in the Lattice instance',
                        action='store_true')
    parser.add_argument('-gridspacing', '--gridspacing', help='Spacing of gridlines for dice_glat', type=float,
                        default=7.5)
    parser.add_argument('-weakbond_val', '--weakbond_val',
                        help='Spring frequency of bonds intersecting dicing lines for dice_glat', type=float,
                        default=-0.5)
    parser.add_argument('-N', '--N',
                        help='Mesh width AND height, in number of lattice spacings (leave blank to spec separate dims)',
                        type=int, default=-1)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=7)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=7)
    parser.add_argument('-NP', '--NP_load', help='Number of particles in mesh, overwrites N, NH, and NV.',
                        type=int, default=0)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                    'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hexagonal')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-theta_twist', '--theta_twist', help='Twisted phase in x for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-phi_twist', '--phi_twist', help='Twisted phase in y for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')

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

    # Global geometric params
    parser.add_argument('-nonperiodic', '--openBC', help='Enforce open (non periodic) boundary conditions',
                        action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=float, default=0.667)
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph',
                        help='Twist angle for twisted_kagome (max is pi/3) in radians or ' +
                             'opening angle of the accordionized lattices or ' +
                             'percent of system decorated -- used in different contexts',
                        type=float, default=0.00)
    parser.add_argument('-x1', '--x1',
                        help='1st Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x2', '--x2',
                        help='2nd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x3', '--x3',
                        help='3rd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-zz', '--zz',
                        help='4th Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-pinconf', '--pinconf',
                        help='Lattice disorder realization number (0 or greater)', type=int, default=0)
    parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                        default=01)
    parser.add_argument('-intparam', '--intparam',
                        help='Integer-valued parameter for building networks (ex # subdivisions in accordionization)',
                        type=int, default=1)
    parser.add_argument('-thres', '--thres', help='Threshold value for building networks (determining to decorate pt)',
                        type=float, default=1.0)
    parser.add_argument('-aratio', '--aratio',
                        help='Ratio between bond lengths in Cairo lattice', type=float, default=np.sqrt(5.))
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    args = parser.parse_args()

    if args.N > 0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV

    strain = 0.00
    # z = 4.0 #target z
    if args.LatticeTop == 'linear':
        shape = 'line'
    else:
        shape = args.shape

    theta = args.theta
    eta = args.eta
    transpose_lattice = 0

    make_slit = args.make_slit

    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    hostname = socket.gethostname()
    if hostname[0:6] == 'midway':
        print '\n\nWe are on Midway!\n\n\n\n'
        rootdir = '/home/npmitchell/scratch-midway/'
        cprootdir = '/home/npmitchell/scratch-midway/'
    elif hostname[0:10] == 'nsit-dhcp-' or hostname[0:10] == 'npmitchell':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
    elif 'Messiaen' in hostname or hostname[0:8] == 'wireless':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
        if not os.path.isdir(cprootdir):
            cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
    elif hostname[0:5] == 'cvpn-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
    else:
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

    print 'cprootdir = ', cprootdir

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': rootdir,
          'phi_lattice': args.phi_lattice,
          'delta_lattice': '{0:0.3f}'.format(args.delta_lattice),
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
          'periodicBC': not args.openBC,
          'periodic_strip': False,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'origin': np.array([0., 0.]),
          'Omk': float((args.Omk).replace('n', '-').replace('p', '.')),
          'Omg': float((args.Omg).replace('n', '-').replace('p', '.')),
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta': args.ABDelta,
          'thres': args.thres,
          'pinconf': args.pinconf,
          'OmKspec': args.OmKspec,
          'spreading_time': args.spreading_time,
          'intparam': args.intparam,
          'immobile_boundary': False,
          'aoverl': args.aoverl,
          'interaction_range': args.intrange,
          'save_pinning_to_hdf5': not args.save_pinning_to_txt,
          }

    omegac = args.omegac

    print 'args = ', args
    cp = {'omegac': sf.string_sequence_to_numpy_array(args.omegac, dtype=float),
          'basis': args.basis,
          'rootdir': cprootdir,
          'save_as_txt': args.save_as_txt,
          }

    print '\n\n cprootdir = ', cprootdir

    if args.title == '':
        title = None
    else:
        print 'title = ', eval("r'" + args.title.replace('_', ' ') + "'")
        titlelist = args.title.split('newline')
        print 'titlelist = ', titlelist
        title = ''
        for tt in titlelist:
            title += eval("r'" + tt.replace('_', ' ') + "'") + '\n'
        print 'title = ', title

        # title = eval("r'" + args.title.replace('_', ' ') + "'")

    if args.calc_botts_omegac:
        # For a single lattice, collate botts with varying omegac
        meshfn, xyffind = le.build_meshfn(lp)
        print 'meshfn = ', meshfn
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        try:
            lat.load()
        except IOError:
            lat.build()
            lat.save()
        mglat = magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lp)
        mglat.load()
        mgc = magnetic_gyro_collection.MagneticGyroCollection()
        mgc.add_gyro_lattice(mglat)
        print 'Creating bott collection from magnetic_gyro_collection...'
        mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
        print 'Getting bott calculations with varying omegac:\n omegac=', cp['omegac']
        mgbcoll.calc_botts_omegac(reverse=args.omegac_reverse)
        mgbcoll.plot_botts_omegac()

    if args.vary_glatparam:
        """Vary an lp parameter compute the bott index for each value

        python ./bott/bott_magnetic_gyro_collection.py -vary_glatparam -glatparam ABDelta -LT hexagonal -N 9 -shortrange
        python ./bott/bott_magnetic_gyro_collection.py -vary_glatparam -LT hexagonal -shape square -paramV 0.:0.025:1.0 -N 11 -shortrange -aol 0.6 -Vpin 0.01 -save_as_txt
        """
        # Collate botts for one lattice with a gyro_lattice parameter that varies between instances of that lattice
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mgc = magnetic_gyro_collection.MagneticGyroCollection()
        # Create list of mgyro_lattices for mgyrocoll, one for each glatparam value
        # First make the paramV
        if args.paramVdtype == 'str':
            # if sf.is_number_sequence(args.paramV):
            #     paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype, corrections=False)
            # else:
            if args.glatparam == 'OmKspec':
                OmKspec_list = mglatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                print 'OmKspec_list = ', OmKspec_list
                paramV = np.array(OmKspec_list)
            else:
                raise RuntimeError('Need to make exception for this paramV option')
        else:
            paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)

        # We need only supply a single mglat to the calc_botts_vary_glatparam() function
        # for param in paramV:
        lpnew = copy.deepcopy(lp)
        lpnew[args.glatparam] = paramV[0]
        mglat = magnetic_gyro_lattice_class.MagneticGyroLattice(lat, lpnew)
        mglat.load()
        mgc.add_mgyro_lattice(mglat)

        print 'Creating bott collection from single-lattice mgyro_collection...'
        mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
        mgbcoll.calc_botts_vary_glatparam(args.glatparam, paramV, reverse=False, verbose=False)
        mgbcoll.plot_botts_vary_param(param=args.glatparam, param_type='glat')
        # Check the omegac values
        # omcs = []
        # abd = []
        # for mglatname in mgbcoll.botts:
        #     bott = mgbcoll.botts[mglatname][0]
        #     print 'bott.cp[omegac] = ', bott.cp['omegac']
        #     omcs.append(bott.cp['omegac'])
        #     abd.append(bott.mgyro_lattice.lp['ABDelta'])
        #
        # plt.plot(abd, omcs, '.')
        # plt.savefig('/Users/npmitchell/Desktop/test.png')

    if args.vary_lpparam:
        """Example usage:
        # topological transition hexagonal --> bowtie
        python bott_collection.py -vary_lpparam -LT hexagonal -N 15 -lpparam delta -shape hexagon
        """
        # Collate botts for many lattices with a gyro_lattice parameter that varies between them
        mgc = magnetic_gyro_collection.GyroCollection()
        if args.lpparam == 'delta' and lp['LatticeTop'] == 'hexagonal':
            mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_' + lp['shape'] + '_delta*_phi0p000_' +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            mgc.ensure_all_gyro_lattices()
            mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
            midgaps = bcollfns.gap_midpoints_honeycomb()
            mgbcoll.plot_botts_vary_param(param=args.lpparam, omegac_glat_func=midgaps, omegac_glatparam='delta',
                                             reverse=args.glatparam_reverse)
        if args.lpparam == 'percolation_density':
            mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*perd*_' + 'r{0:02d}_'.format(lp['subconf']) +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            mgc.ensure_all_gyro_lattices()
            mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
            mgbcoll.plot_botts_vary_param()
        # To find the relevant locality size to compute bott index, decorate outside the bott region with varying
        # alphas (distances from the center that the frame is decorated).
        if args.lpparam == 'alpha_kagframe':
            # Assume that LatticeTop was specified as either hex_kagframe, hucent_kagframe, or similar
            if lp['LatticeTop'] in ['hex_kagframe', 'hex_kamgcframe', 'hucent_kagframe']:
                print 'looking for meshfns of type hex_*frames...'
                mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_square_delta' + le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) + '_alph*0_' +
                              '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))
            elif lp['LatticeTop'] in ['hex_kagperframe']:
                print 'bott_collection: looking for meshfns...'
                mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_square_delta' + le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) +
                              '_perd' + le.float2pstr(lp['percolation_density'], ndigits=2) +
                              '_alph*0_' + '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))
            mgc.ensure_all_gyro_lattices()
            mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
            mgbcoll.plot_botts_vary_param(param='alph')

    if args.spectra_vary_glatparam:
        # Collate botts for many lattices with a gyro_lattice parameter that varies between them
        mgc = magnetic_gyro_collection.GyroCollection()
        if args.glatparam == 'percolation_density':
            mgc.add_meshfn(rootdir + 'networks/'+args.LatticeTop+'/'+
                          args.LatticeTop+'_square_d*perd*0_' + 'r{0:02d}_'.format(lp['subconf']) +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            mgc.ensure_all_gyro_lattices()
            mgbcoll = BottMagneticGyroCollection(mgc, cp=cp)
            mgbcoll.plot_bottspectra_vary_glatparam()

    if args.bott_collection:
        # Collate botts for many lattices
        mgc = magnetic_gyro_collection.MagneticGyroCollection()
        if args.LatticeTop == 'iscentroid':
            mgc.add_meshfn('networks/' + args.LatticeTop+'/' +
                          args.LatticeTop + '_square_periodic_hexner_size' + str(args.NP_load) +
                          '_conf*_NP*' + str(args.NP_load))
        elif args.LatticeTop == 'kagome_isocent':
            mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop+'_square_periodic_hexner_size' + str(args.NP_load) +
                          '_conf*_NP*'+str(args.NP_load))
        elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
            mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_periodic_d*'+'_NP*' + str(args.NP_load))
        elif args.LatticeTop == 'kagper_hucent':
            mgc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*' + '_' + '{0:06d}'.format(NH))

        description = leplt.lt2description(args.LatticeTop)
        title = r'$D(\omega)$ for magnetic ' + description + ' networks'
        mgbcoll = BottMagneticGyroCollection(mgc)
        mgbcoll.plot_botts_vary_param()
