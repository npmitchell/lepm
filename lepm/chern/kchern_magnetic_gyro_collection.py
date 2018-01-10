import numpy as np
from lepm.magnetic_gyro_lattice_class import MagneticGyroLattice
import lepm.chern.kchern_gyro_collection_functions as kchern_collfns
import lepm.magnetic_gyro_functions as mglatfns
from lepm.gyro_lattice_functions import param2meshfnexten_name as glat_param2meshfnexten_name
from lepm.lattice_functions import param2meshfn_name as lat_param2meshfn_name
import lepm.plotting.science_plot_style as sps
import matplotlib.pyplot as plt
import cPickle as pkl
import time
import os.path
import sys
import copy
import lepm.dataio as dio
import lepm.chern.kchern_magnetic_gyro as kchern_mg
import glob

'''Class for a collection of k-space Chern calculations for magnetic gyro_lattices'''


class KChernMagneticGyroCollection:
    """Create a collection of kspace chern measurements for magnetic gyroscopic networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.cherns is a list of tuples of dicts: self.cherns = [chern1, chern2,...]
    where chern1, chern2 are each a KChernMagneticGyro class instance with attributes:
            cp : dict
                chern calc parameters
            mgyro_lattice : GyroLattice class instance
                the gyro lattice for which to compute the chern number
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
        keys are glat_names (strings), values are lists of KitaevChern instances
    self.cp : dict
        chern calculation master dictionary
    self.gyro_collection : list of mgyro_lattice instances
        list of instances of magnetic_gyro_lattice_class.MagneticGyroLattice() corresponding to
        each gyro+spring network
    self.meshfns : list of strings
        string paths to the location of the mgyro_lattices in the gyro_collection
    self.mglat_names : list of strings
        the names of the mgyro_lattices in the chern collection
    """
    def __init__(self, mgyro_collection, cp=None, cpmeshfn=None):
        self.mgyro_collection = mgyro_collection
        self.mglat_names = []
        self.meshfns = mgyro_collection.meshfns
        self.cherns = {}

        if cp is None:
            cp = {}
        if 'basis' not in cp:
            cp['basis'] = 'psi'

        self.cp = cp

    def chern_is_saved(self, mglat, cp=None, verbose=False):
        """"""
        if verbose:
            print '\nChecking if chern is already saved...'
        if cp is None:
            cp = self.cp
        chern = kchern_mg.KChernMagneticGyro(mglat, cp)
        if glob.glob(chern.cp['cpmeshfn'] + 'chern.txt'):
            return True
        else:
            return False

    def add_chern(self, mglat, cp=None, attribute_evs=False, verbose=False, save_png=True):
        """Add a ChernIndex instance to the ChernGyroCollection, and append the mglat_name to self.mglat_names if not
        already in that list"""
        if cp is None:
            cp = self.cp
        chern = kchern_mg.KChernMagneticGyro(mglat, cp)
        chern.get_chern(attribute=True, verbose=verbose, save_png=save_png)
        mglat_name = mglat.lp['meshfn'] + mglat.lp['meshfn_exten']
        if mglat_name not in self.cherns:
            self.cherns[mglat_name] = []

        self.cherns[mglat_name].append(chern)
        if mglat_name not in self.mglat_names:
            self.mglat_names.append(mglat_name)
        return chern

    def get_cherns(self, cp=None, verbose=False, reverse=False, save_png=True):
        """Retrieve cherns for each mglat in self.mgyro_collection.mgyro_lattices matching
        the supplied/attributed cp

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
            mglatstodo = self.mgyro_collection.mgyro_lattices[::-1]
        else:
            mglatstodo = self.mgyro_collection.mgyro_lattices

        for mglat in mglatstodo:
            cp = copy.deepcopy(cp_orig)
            cp.pop('cpmeshfn', None)
            # print '\n\n\nchern_collection: get_cherns: mglat = ', mglat.lp['meshfn']
            # if verbose:
            print 'Adding chern for mglat =', mglat.lp['meshfn']
            self.add_chern(mglat, cp=cp, verbose=verbose, save_png=save_png)

    def get_avg_chern(self, band_index=-1):
        """Considering all cherns in the kcoll, get the average chern value obtained

        Parameters
        ----------
        """
        print 'kchern_gyro_collection: Obtaining average chern over all cherns in the collection'
        nuoutlist = []
        mglat_name_list = []
        for mglat_name in self.cherns:
            print 'KChernMagneticGyroCollection: mglat_name = ', mglat_name
            vals = []
            # get val for this chern
            for chernii in self.cherns[mglat_name]:
                newnu = chernii.chern['chern'][band_index]
                vals.append(newnu)

            val = np.mean(vals)
            nuoutlist.append(val)
            mglat_name_list.append(mglat_name)

        return mglat_name_list, nuoutlist

    def calc_cherns_vary_glatparam(self, glatparam, paramV, reverse=False, verbose=False, save_png=True):
        """For a single Lattice instance, vary a GyroLattice parameter and compute the chern index for each value.
        When supplied, a single magnetic gyro_lattice instance is required for each physical lattice, but each new glat
        (same lattice, different mglat.lp) will be appended to self.mgyro_collection.mgyro_lattices when lp is updated

        Parameters
        ----------
        glatparam : str key for lp dictionary
            the string specifier for the parameter to change FOR THE SAME LATTICE, for each value in paramV
        paramV : list or 1d numpy array
            the values to assign to mglat.lp[glatparam] for each chern calculation
        reverse : bool
            reverse the order of paramV in which to compute
        verbose : bool
            print intermediate output
        """
        # prepare xy locations as fraction of system size
        print 'Looping over all networks in the gyro collection. ' + \
              'For each one, cycling through glat lpparams in paramV..'
        print('--> mglats to do:', [key for key in self.mgyro_collection.mgyro_lattices])
        mglatstodo = [key for key in self.mgyro_collection.mgyro_lattices]
        for mglat in mglatstodo:
            print '\n\nchern_collection: Getting cherns for MagneticGyroLattice ', mglat, '...'
            print ' of name ', mglat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)

            proj = None
            print 'paramV = ', paramV
            if reverse:
                paramtodo = paramV[::-1]
            else:
                paramtodo = paramV

            lp_master = copy.deepcopy(mglat.lp)
            lat = copy.deepcopy(mglat.lattice)
            for val in paramtodo:
                cp = copy.deepcopy(self.cp)
                lp = copy.deepcopy(lp_master)
                lp[glatparam] = val
                print 'glatparam = ', glatparam
                # print 'lp[OmKspec]=', lp['OmKspec']
                mglat = MagneticGyroLattice(lat, lp)
                print 'mglat.lp[meshfn]=', mglat.lp['meshfn']
                # This should not be necessary since appended in self.add_chern()
                # self.mgyro_collection.gyro_lattices.append(mglat)

                # Simply add the one chern to the collection
                self.add_chern(mglat, cp=cp, verbose=verbose, save_png=save_png)

        return self.cherns

    def collect_cherns_vary_lpparam(self, param='percolation_density', reverse=False, band_index=-1):
        """Plot chern indices versus the parameter which is varying between networks.
        If there are multiple chern calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        band_index : int

        """
        print 'kchern_mgyro_collection: wrong place?'
        sys.exit()
        if self.cherns == {}:
            self.get_cherns(reverse=reverse)

        params = []
        nuList = []
        for mglat_name in self.cherns:
            print 'adding params from mglat_name = ', mglat_name
            first = True
            for chern in self.cherns[mglat_name]:
                if first:
                    # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                    # the chern instance
                    params.append(chern.mgyro_lattice.lp[param])
                    print 'kcoll: added param for varying mglat_param: ', param, ' --> ', chern.gyro_lattice.lp[param]
                    nu = np.array([np.real(chern.chern['chern'])])
                    print 'nu = ', nu
                    first = False
                else:
                    # Collate the other Chern calculations
                    params.append(chern.mgyro_lattice.lp[param])
                    nu_tmp = np.real(chern.chern['chern'])
                    nu_tmp = nu_tmp.reshape(len(nu_tmp), 1)
                    nu = np.hstack((nu, nu_tmp))

            # If there were multiple, average over them all
            if len(self.cherns[mglat_name]) > 1:
                # Average over them all
                nu = np.mean(nu, axis=1)

            nuList.append(nu)

        # Build output array
        print 'params = ', params
        print 'nuList = ', nuList
        param_nu = np.dstack((np.array(params), np.array(nuList)))[0]
        return param_nu

    def collect_cherns_vary_glatparam(self, param='ABDelta', reverse=False, band_index=-1):
        """Plot chern indices with x axis being the parameter which is varying between networks.
        If there are multiple chern calculations of a particular GyroLattice, average nu over them all.

        Parameters
        ----------
        param : str
            the key for the mglat.lp dictionary element whose value will be varied from one
            MagneticGyroLattice instance to another
        reverse : bool
            Compute the chern numbers for the GyroLattice instances in reverse order
        band_index : int (default=-1)
            the index of the band for which to store the chern number
        """
        if self.cherns == {}:
            self.get_cherns(reverse=reverse)

        params = []
        nuList = []
        for mglat_name in self.cherns:
            print 'adding params from mglat_name = ', mglat_name
            first = True
            for chern in self.cherns[mglat_name]:
                if first:
                    # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                    # the chern instance
                    paramval = kchern_collfns.retrieve_param_value(chern.mgyro_lattice.lp[param])
                    params.append(paramval)
                    print 'kcoll: added param for varying mglat_param: ', param, ' --> ', chern.mgyro_lattice.lp[param]
                    # For now assume that ksize_frac_arr (and regions, etc) are uniform across them all
                    nu = np.real(chern.chern['chern'][band_index])
                    nuList.append(nu)
                    first = False
                else:
                    # Collate the other Chern calculations
                    paramval = kchern_collfns.retrieve_param_value(chern.mgyro_lattice.lp[param])
                    params.append(paramval)
                    nuList.append(np.real(chern.chern['chern'][band_index]))

        # Build output array
        print 'params = ', params
        print 'shape(params) = ', np.shape(params)
        print 'nuList = ', nuList
        for ii in range(len(params)):
            print 'ii = ', ii
            param = kchern_collfns.retrieve_param_value(params[ii])
            nu = nuList[ii]
            if ii == 0:
                print 'param = ', param
                param_nu = np.array([param, nu])
            else:
                add_array = np.array([param, nu])
                param_nu = np.vstack((param_nu, add_array))

        return param_nu

    def plot_cherns_vary_param(self, param_type='glat', sz_param_nu=None,
                              reverse=False, param='percolation_density',
                              title='Chern index calculation', xlabel=None):
        """Plot chern indices with x axis being the parameter which is varying between networks.
        If there are multiple chern calculations of a particular lattice, average nu over them all.

        Parameters
        ----------
        param_type : str ('glat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a gyrolattice parameter (same lattice,
            different physics)
        sz_param_nu : None or 'glat' or ('lat' or 'lp' -- last two do same thing)
            string specifier for how to vary params
        reverse : bool
            Compute cherns for GyroLattice instances in self.mgyro_collection in reverse order
        param : string
            string specifier for GyroLattice parameter to vary between networks; key for mglat.lp dict
        title : str
            title of the plot
        xlabel : str
            xlabel, if desired to be other than default for param (ie, param.replace('_', ' '))

        """
        if sz_param_nu is None:
            if param_type == 'lat' or param_type == 'lp':
                param_nu = self.collect_cherns_vary_lpparam(param=param, reverse=reverse)
            elif param_type == 'glat':
                param_nu = self.collect_cherns_vary_glatparam(param=param, reverse=reverse)
            else:
                raise RuntimeError("param_type argument passed is not 'glat' or 'lat/lp'")

        # Plot it as colormap
        plt.close('all')

        paramV = param_nu[:, 0]
        nu = param_nu[:, 1]

        # Make figure
        import lepm.plotting.plotting as leplt
        fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=4)

        if xlabel is None:
            xlabel = param.replace('_', ' ')

        # Plot the curve
        # first sort the paramV values
        if isinstance(paramV[0], float):
            si = np.argsort(paramV)
        else:
            si = np.arange(len(paramV), dtype=int)
        ax.plot(paramV[si], nu[si], '.-')
        ax.set_xlabel(xlabel)

        # Add title
        ax.text(0.5, 0.95, title, transform=fig.transFigure, ha='center', va='center')

        # Save the plot
        if param_type == 'glat':
            outdir = rootdir + 'kspace_cherns_mgyro/chern_glatparam/' + param + '/'
            dio.ensure_dir(outdir)

            # Add meshfn name to the output filename
            outbase = self.cherns[self.cherns.items()[0][0]][0].mgyro_lattice.lp['meshfn']
            if outbase[-1] == '/':
                outbase = outbase[:-1]
            outbase = outbase.split('/')[-1]

            outd_ex = self.cherns[self.cherns.items()[0][0]][0].mgyro_lattice.lp['meshfn_exten']
            # If the parameter name is part of the meshfn_exten, replace its value with XXX in
            # the meshfnexten part of outdir.
            mfestr = glat_param2meshfnexten_name(param)
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
        elif param_type == 'lat':
            outdir = rootdir + 'kspace_cherns_mgyro/chern_lpparam/' + param + '/'
            outbase = self.cherns[self.cherns.items()[0][0]][0].mgyro_lattice.lp['meshfn']
            # Take apart outbase to parse out the parameter that is varying
            mfestr = lat_param2meshfn_name(param)
            if mfestr in outbase :
                'param is in meshfn_exten, splitting...'
                # split the outdir by the param string
                od_split = outbase.split(mfestr)
                # split the second part by the value of the param string and the rest
                od2val_rest = od_split[1].split('_')
                odrest = od_split[1].split(od2val_rest[0])[1]
                print 'odrest = ', odrest
                print 'od2val_rest = ', od2val_rest
                outd_ex = od_split[0] + param + 'XXX'
                outd_ex += odrest
                print 'outd_ex = ', outd_ex
            else:
                outbase += '_' + param + 'XXX'

            outd_ex = self.cherns[self.cherns.items()[0][0]][0].mgyro_lattice.lp['meshfn_exten']

        dio.ensure_dir(outdir)

        fname = outdir + outbase
        fname += '_chern_' + param + '_Ncoll' + '{0:03d}'.format(len(self.mgyro_collection.mgyro_lattices))
        fname += outd_ex
        print 'saving to ' + fname + '.png'
        plt.savefig(fname + '.png')
        plt.clf()


if __name__ == '__main__':
    '''running example for calculating a collection of cherns for various samples'''
    import argparse
    import socket
    import lepm.stringformat as sf
    import lepm.lattice_class as lattice_class
    import lepm.lattice_elasticity as le
    import lepm.plotting.colormaps as lecmaps
    import lepm.plotting.plotting as leplt
    import lepm.magnetic_gyro_collection as mgyro_collection
    from lepm.magnetic_gyro_lattice_class import MagneticGyroLattice
    import lepm.gyro_lattice_functions as glatfns

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create GyroLattice class instance,' +
                                                 ' with options to save or compute attributes of the class.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-verbose', '--verbose', help='Print progress to command line output', action='store_true')
    parser.add_argument('-calc_ABtrans', '--calc_cherns_ABtrans',
                        help='Compute the chern numbers using the berry curvature', action='store_true')
    parser.add_argument('-density', '--density',
                        help='Density of points per unit reciprocal area in the BZ', type=int, default=400)
    parser.add_argument('-pin2hdf5', '--pin2hdf5',
                        help='Enforce saving pinning as hdf5 rather than as txt', action='store_true')
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

    # Options for script running, iterating over mglats
    parser.add_argument('-glatparam', '--glatparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-glatparam_reverse', '--glatparam_reverse',
                        help='When computing chern number varying a gyrolattice param, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-lpparam_reverse', '--lpparam_reverse',
                        help='When computing chern number varying a lattice param, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-paramV', '--paramV',
                        help='Sequence of values to assign to lp[param] if vary_lpparam or vary_glatparam is True',
                        type=str, default='0.0:0.1:1.0')
    parser.add_argument('-paramVdtype', '-paramVdtype',
                        help='The data type for the numpy array formed from paramV',
                        type=str, default='float')

    # Geometry and physics arguments
    parser.add_argument('-intrange', '--intrange',
                        help='Consider magnetic couplings only to nth nearest neighbors (if ==2, then NNNs, for ex)',
                        default=0)
    parser.add_argument('-aol', '--aoverl',
                        help='interparticle distance divided by length of pendulum from pivot to center of mass',
                        type=float, default=1.0)
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
    parser.add_argument('-Omk', '--Omk', help='Spring frequency', type=str, default='-0.7')
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
          'rootdir': rootdir,
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
          'basis': args.basis,
          'Omk': float(args.Omk.replace('n', '-').replace('p', '.')),
          'Omg': float(args.Omg.replace('n', '-').replace('p', '.')),
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
          'immobile_boundary': args.immobile_boundary,
          'bl0': args.bl0,  # this is for prestrain in the bonds (the bonds are stretched to their current positions)
          'save_pinning_to_hdf5': args.pin2hdf5,
          'kicksz': args.kicksz,
          'theta_twist': args.thetatwist,
          'phi_twist': args.phitwist,
          'aoverl': args.aoverl,
          'interaction_range': args.intrange,
          }

    cp = {'density': args.density,  # number of points per area in BZ for calculation.
          'rootdir': cprootdir,
          'basis': args.basis,
          }

    if args.calc_cherns_ABtrans:
        '''Example usage:
        python ./chern/kchern_magnetic_gyro_collection.py -LT hexagonal -N 1 -calc_ABtrans -density 400 \
                -paramV 0:0.1:1. -verbose
        python ./chern/kchern_magnetic_gyro_collection.py -LT hexagonal -N 1 -calc_ABtrans -density 40 \
                -paramV 0:0.1:1. -verbose
        '''
        # loop for values in the lattice paramters
        kcolls = []

        # define the lattice parameter values
        lpvals = np.arange(0.6, 1.31, 0.02)
        # lpvals = np.hstack((np.array([2./3.]), lpvals))

        lpmaster = copy.deepcopy(lp)
        if args.lpparam_reverse:
            lpvals = lpvals[::-1]

        for lpval in lpvals:
            # Create mglat
            lp = copy.deepcopy(lpmaster)
            lp['delta_lattice'] = '{0:0.3f}'.format(lpval)
            lp['delta'] = lpval * np.pi

            meshfn = le.find_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.load()
            mglat = MagneticGyroLattice(lat, lp)

            # Collect GyroLattice instances
            mgc = mgyro_collection.MagneticGyroCollection()
            mgc.add_mgyro_lattice(mglat)

            print 'Creating chern collection from single-lattice gyro_collection...'
            kcoll = KChernMagneticGyroCollection(mgc, cp=cp)

            if args.paramVdtype == 'str':
                if args.glatparam == 'OmKspec':
                    OmKspec_list = glatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                    print 'OmKspec_list = ', OmKspec_list
                    paramV = np.array(OmKspec_list)
                else:
                    raise RuntimeError('Need to make exception for this paramV option')
            else:
                paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)

            kcoll.calc_cherns_vary_glatparam(args.glatparam, paramV, reverse=args.glatparam_reverse,
                                             verbose=args.verbose, save_png=False)
            kcolls.append(kcoll)
            kcoll.plot_cherns_vary_param(param=args.glatparam, param_type='glat')

        deltagrid = []
        abdgrid = []
        cherngrid = []
        for kcoll in kcolls:
            abds = []
            deltas = []
            cherns = []
            print 'kcoll = ', kcoll
            for glatname in kcoll.cherns:
                chernii = kcoll.cherns[glatname]
                print 'chernii = ', chernii
                if len(chernii) == 1:
                    chernii = chernii[0]
                else:
                    raise RuntimeError('More than one chern in this list!')
                # get ab value from glat.lp
                abds.append(chernii.mgyro_lattice.lp['ABDelta'])
                deltas.append(chernii.mgyro_lattice.lp['delta'])
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
        specstr = '_aol' + sf.float2pstr(lp['aoverl']) + '_Omk' + sf.float2pstr(lp['Omk'])
        specstr += '_delta' + sf.float2pstr(np.min(deltagrid)) + '_' + sf.float2pstr(np.max(deltagrid))
        specstr += '_abd' + sf.float2pstr(np.min(abdgrid)) + '_' + sf.float2pstr(np.max(abdgrid))
        specstr += '_ndeltas{0:05d}'.format(len(deltas)) + '_nabd{0:05d}'.format(len(abds))
        specstr += '_density{0:07d}'.format(cp['density'])
        plt.savefig(rootdir + 'kspace_cherns_mgyro/abtransition' + specstr + '.png')


