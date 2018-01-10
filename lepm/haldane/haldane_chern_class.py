import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
from lepm.haldane import haldane_chern_functions
import lepm.plotting.plotting as leplt
import lepm.plotting.kitaev_plotting_functions as kpfns
from lepm import lattice_class
from lepm.haldane import haldane_lattice_class
import h5py
import matplotlib.pyplot as plt
import argparse
import glob
import socket
import time
import lepm.stringformat as sf
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl


class HaldaneChern:
    """Class for a chern number computation.

    Attributes
    ----------
    cp : dict
        'Chern parameters' --> parameter dictionary for the chern number calculation
    haldane_lattice : a haldane_lattice instance
        The haldane network for which to compute the chern number
    chern_finsize : len(ksize) x 5 float array
        contains [Nreg1, ksize_frac, ksize, ksys_size (note this is NP_summed), nu for Chern calculation]
        where
            Nreg1 : #particles in region 1
            ksize_frac :
                the kitaev width, as fraction of the maximum extent of the system in 2D: max(L_x, L_y), where
                L_x = np.max(xy[:, 0]) - np.min(xy[:, 0])
            ksize : this is the width/height/diameter of the kitaev summation region if it is circular or square
                    If hexagonal, this is one sidelength (at least for a regular hexagon -- not sure for deformed hex's)
            ksys_size : 2 x # particles in sum
            ksys_frac : 2 x # particles in sum / total # particles
            nu : the chern number for each ksize computation

    params_regs : dict
        a nested dictionary with key,value pairs given by
            keys : str
                each key is a string element of ksize: '{0:0.3f}'.format(ksize)
            values : dict
                dictionary with key,value pairs given by
                    reg1_xy : int list
                        list of self.haldane_lattice.lattice.xy indices in reg1 of kitaev sum region
                    reg2_xy : int list
                        list of self.haldane_lattice.lattice.xy indices in reg2 of kitaev sum region
                    reg3_xy : int list
                        list of self.haldane_lattice.lattice.xy indices in reg3 of kitaev sum region
                    polygon1 : #vertices x 2 float array
                        coordinates of vertices of polygon enclosing reg1
                    polygon2 : #vertices x 2 float array
                        coordinates of vertices of polygon enclosing reg2
                    polygon3 : #vertices x 2 float array
                        coordinates of vertices of polygon enclosing reg3
                    reg1 : int list
                        list of self.haldane_lattice.lattice.xy indices in reg1 of kitaev sum region
                    reg2 : int list
                        list of self.haldane_lattice.lattice.xy indices in reg2 of kitaev sum region
                    reg3 : int list
                        list of self.haldane_lattice.lattice.xy indices in reg3 of kitaev sum region
    """
    def __init__(self, haldane_lattice, cp=None, cpmeshfn=None):
        self.haldane_lattice = haldane_lattice
        self.chern_finsize = None
        self.contribs = None
        self.params_regs = {}

        if cp is None:
            cp = {}
        if 'ksize_frac_arr' not in cp:
            cp['ksize_frac_arr'] = np.arange(0.0, 1.1, 0.01)
        if 'omegac' not in cp:
            cp['omegac'] = np.array([2.25])
        if 'regalph' not in cp:
            cp['regalph'] = np.pi * (11. / 6.)
            cp['regbeta'] = np.pi * (7. / 6.)
            cp['reggamma'] = np.pi * 0.5
        if 'shape' not in cp:
            cp['shape'] = haldane_lattice.lattice.lp['shape']
        if 'polyT' not in cp:
            cp['polyT'] = False
        if 'poly_offset' not in cp:
            cp['poly_offset'] = 'none'
        if 'basis' not in cp:
            cp['basis'] = 'XY'
        if 'cpmeshfn' not in cp:
            if cpmeshfn is None:
                cpmeshfn = haldane_chern_functions.get_cpmeshfn(cp, self.haldane_lattice.lp)
            cp['cpmeshfn'] = cpmeshfn
        if 'modsave' not in cp:
            cp['modsave'] = 20
        if 'save_ims' not in cp:
            cp['save_ims'] = False
        if 'outerH' not in cp:
            # for making a kitaev annulus region
            cp['outerH'] = 0.0

        self.cp = cp

    def load_chern(self, skip_paramsregs=False, skip_contribs=False, h5fn='none'):
        """
        Parameters
        ----------
        skip_paramsregs : bool
            Skip loading parameters and regions dictionary (params_regs)
        h5fn : str
            The path for the hdf5 file containing the data, or 'none' to pull data from accessible txt file
        """
        if h5fn is 'none':
            # Load data from txt files
            fn = self.cp['cpmeshfn'] + 'chern_finsize.txt'
            self.chern_finsize = np.loadtxt(fn, delimiter=',')

            if not skip_paramsregs:
                # Load the parameters and regions for the chern calculation if it exists
                pregpkl = glob.glob(self.cp['cpmeshfn'] + 'params_regs.pkl')
                paramregdicts = glob.glob(self.cp['cpmeshfn'] + 'params_regs/params_regs*')
                if pregpkl:
                    print 'Loading params_regs from pickle...'
                    with open(pregpkl[0], "rb") as fn:
                        self.params_regs = pickle.load(fn)
                elif paramregdicts:
                    # Load each kitaev region dictionary into the nested params_regs dictionary
                    print 'Could not load params_regs from pickle, loading from subdirectory...'
                    ind = 0
                    for regp_fn in paramregdicts:
                        if ind % 200 == 0:
                            print 'Loading params_regs #', ind, ' of ', len(paramregdicts), '\n'
                        filename = regp_fn.split('/')[-1]
                        ksizekey = filename.split('ksize')[-1].split('.txt')[0]
                        self.params_regs[ksizekey] = le.load_params(self.cp['cpmeshfn'] + 'params_regs', filename)
                        ind += 1
                else:
                    print 'Could not load params_regs from pickle or subdirectory, skipping...'

            if not skip_contribs:
                # Load contributions, if it is saved and exists
                contribpkl = glob.glob(self.cp['cpmeshfn'] + 'contribs.pkl')
                contribdicts = glob.glob(self.cp['cpmeshfn'] + 'contribs/contribs*')
                if contribpkl:
                    print 'Loading contribs from pickle...'
                    with open(contribpkl[0], "rb") as fn:
                        self.contribs = pickle.load(fn)
                elif contribdicts:
                    # Load each kitaev region dictionary into the nested contribs dictionary
                    print 'Could not load contribs from pickle, loading from subdirectory...'
                    self.contribs = {}
                    ind = 0
                    for contrib_fn in contribdicts:
                        if ind % 200 == 0:
                            print 'Loading contribs #', ind, ' of ', len(contribdicts), '\n'
                        filename = contrib_fn.split('/')[-1]
                        ksizekey = filename.split('ksize')[-1].split('.txt')[0]
                        self.contribs[ksizekey] = le.load_params(self.cp['cpmeshfn'] + 'contribs', filename)
                        ind += 1
        else:
            self.load_hdf5_chern(h5fn)

    def chern_in_hdf5(self, h5fn):
        fi = h5py.File(h5fn, "r")
        # Get subdir path from h5fn
        hfnsplit = self.cp['cpmeshfn'].split('/' + self.haldane_lattice.lp['LatticeTop'] + '/')
        subgroup = hfnsplit[-1]
        print 'to check if in hdf5, looking for subgroup = ', subgroup
        print ' ... in file: ', h5fn
        if subgroup in fi:
            if 'chern_finsize' in fi[subgroup].attrs:
                fi.close()
                return True
            else:
                fi.close()
                return False
        else:
            fi.close()
            return False

    def load_hdf5_chern(self, h5fn, load_paramsregs=False):
        # todo: test this method
        # Load data from hdf5 file
        fi = h5py.File(h5fn, "r")

        # Get subdir path from h5fn
        hfnsplit = self.cp['cpmeshfn'].split('/' + self.haldane_lattice.lp['LatticeTop'] + '/')
        subgroup = hfnsplit[-1]
        self.chern_finsize = fi[subgroup].attrs['chern_finsize']
        if load_paramsregs:
            # Load each kitaev region dictionary into the nested params_regs dictionary
            subg_preg = dio.prepdir(subgroup) + 'params_regs'
            if subg_preg in fi:
                preg_groups = fi[dio.prepdir(subgroup) + 'params_regs']
                ind = 0
                for preg_fn in preg_groups:
                    if ind % 200 == 0:
                        print 'Loading params_regs #', ind, ' of ', len(paramregdicts.attrs), '\n'
                    ksizekey = preg_fn.split('ksize')[-1]
                    pregdict = {}
                    for key in fi[subg_preg + '/' + preg_fn].attrs:
                        pregdict[key] = fi[subg_preg + '/' + preg_fn].attrs[key]
                    self.params_regs[ksizekey] = pregdict
                    ind += 1
            else:
                print 'Could not load requested params_regs from hdf5...'
        fi.close()

    def get_haldane_chern(self, proj=None, save=True, skip_paramsregs=False, check=False, attribute_evs=False,
                         verbose=False, title=None):
        """
        Load or compute chern for range of ksizes on self.haldane_lattice with chern parameters self.cp

        Parameters
        ----------
        proj : None or complex #eigvals x #eigvals array
            projection operator, if pre-supplied or already computed
        save : bool
            whether to save the chern output
        skip_paramsregs : bool
            skip loading and saving the params_regs dictionary
        check : bool
            Display intermediate results
        attribute_evs : bool
            Whether to attribute eigenvectors and eigvals to the haldane_lattice object (if needed later for futher
            computation, for ex)
        verbose : bool
            Print output on command line

        Returns
        -------
        chern_finsize : len(ksize) x 5 float array
            contains [Nreg1, ksize_frac, ksize, ksys_size (note this is 2*NP_summed), nu for Chern calculation]
        params_regs : dict
            a nested dictionary with key,value pairs given by
                keys : str
                    each key is a string element of ksize: '{0:0.3f}'.format(ksize)
                values : dict
                    dictionary with key,value pairs
        """
        if self.chern_finsize is None or self.params_regs is None:
            # First try to load chern, otherwise calculate it
            # if verbose:
            print 'Searching for saved chern: ', self.cp['cpmeshfn']
            data_on_disk = glob.glob(self.cp['cpmeshfn'] + 'chern_finsize.txt')
            if data_on_disk:
                # if verbose:
                print '\nFound chern, loading: ', self.cp['cpmeshfn']
                self.load_chern(skip_paramsregs=skip_paramsregs)
            else:
                h5fn = self.haldane_lattice.lp['rootdir'] + 'haldane_chern/' + self.haldane_lattice.lp['LatticeTop'] + '.hdf5'
                hdf5_on_disk = glob.glob(h5fn)
                if hdf5_on_disk:
                    data_on_hdf5 = self.chern_in_hdf5(h5fn)
                else:
                    data_on_hdf5 = False

                if data_on_hdf5:
                    print '\nFound chern on hdf5, loading...'
                    self.load_chern(skip_paramsregs=skip_paramsregs, h5fn=h5fn)
                else:
                    # Otherwise calculate it
                    print '\nkchern: calculating chern...'
                    self.calc_haldane_chern(proj=proj, check=check, paramsregs=not skip_paramsregs,
                                            attribute_evs=attribute_evs, verbose=verbose, title=title)

                    if save:
                        print 'kchern: saving chern...'
                        self.save_chern()

        return self.chern_finsize, self.params_regs

    def calc_haldane_chern(self, proj=None, paramsregs=True, contributions=False,
                           check=False, verbose=False, attribute_evs=False, title=None):
        """
        Compute chern for range of ksizes on self.haldane_lattice with chern parameters self.cp
        """
        print 'HaldaneChern.calc_haldane_chern: calculating chern...'
        chern_finsize, params_regs, contribs = \
            haldane_chern_functions.calc_haldane_chern(self.haldane_lattice, self.cp, pp=proj,
                                                       contributions=contributions, check=check, verbose=verbose,
                                                       attribute_evs=attribute_evs, title=title)
        self.chern_finsize = chern_finsize
        if contributions:
            self.contribs = contribs
            self.params_regs = params_regs
        elif paramsregs:
            self.params_regs = params_regs

    def get_projector(self, attribute_evs=False):
        """
        Parameters
        ----------
        omegac : float
            Cutoff frequency for projector
        attribute_evs : bool
            Attribute eigvect/eigval to self.haldane_lattice during computation of projector

        """
        if isinstance(self.cp['omegac'], np.ndarray):
            if len(self.cp['omegac']) > 1:
                print 'HaldaneChern: Warning! More than one value for omegac is supplied. Using the first value...'
            omegac = self.cp['omegac'][0]
        else:
            omegac = self.cp['omegac']
        proj = haldane_chern_functions.calc_projector(self.haldane_lattice, omegac, attribute=attribute_evs)
        return proj

    def get_contribs(self):
        """If contributions are part of this chern instance, return those contributions. If not, I hesitate to instigate
        a computation to get them, so just return None.
        """
        if self.contribs is None:
            self.load_chern()

        return self.contribs

    def save_chern(self):
        # Make output directories
        print 'saving chern to: ', self.cp['cpmeshfn']
        dio.ensure_dir(self.cp['cpmeshfn'])
        # if self.params_regs or self.contribs is not None:
        #     dio.ensure_dir(self.cp['cpmeshfn'] + 'params_regs/')
        # if self.contribs is not None:
        #     dio.ensure_dir(self.cp['cpmeshfn'] + 'contribs/')

        # Save the chern calculation to cp['cpmeshfn']
        if self.chern_finsize is None:
            raise RuntimeError('Cannot save chern calculation since it has not been computed!')

        # Save chern parameters
        fn = self.cp['cpmeshfn'] + 'chern_params.txt'
        header = 'Parameters for chern calculation'
        dio.save_dict(self.cp, fn, header)

        # Save chern_finsize
        fn = self.cp['cpmeshfn'] + 'chern_finsize.txt'
        header = 'Nreg1, ksize_frac, ksize, ksys_size (note this is NP_summed), ksys_frac, nu' + \
                 'for Chern calculation: basis=' + \
                 self.cp['basis'] + ' pin=' + '{0:0.3f}'.format(self.haldane_lattice.lp['pin']) + \
                 ' V0_pin_gauss=' + '{0:0.3f}'.format(self.haldane_lattice.lp['V0_pin_gauss']) + \
                 ' t1=' + '{0:0.3f}'.format(self.haldane_lattice.lp['t1']) +\
                 ' t2=' + '{0:0.3f}'.format(self.haldane_lattice.lp['t2'])

        print 'saving chern_finsize to ', fn
        np.savetxt(fn, self.chern_finsize, delimiter=',', header=header)

        # Save chern_finsize as a plot
        plt.clf()
        plt.plot(self.chern_finsize[:, -2], self.chern_finsize[:, -1], 'o-')
        plt.title('Chern number versus system size')
        plt.xlabel('Fraction of system size in sum')
        plt.ylabel('Chern number')
        plt.savefig(self.cp['cpmeshfn'] + 'chern_finsize_ksizeSum.png')
        plt.close('all')

        # Save kitaev region parameters for each ksize
        # if self.params_regs or self.contribs is not None:
        #     for ksize in self.params_regs:
        #         header = 'Indices of kitaev regions: ksize = ' + sf.prepstr(ksize)
        #         fn = self.cp['cpmeshfn']+'params_regs/params_regs_ksize' + sf.prepstr(ksize) + '.txt'
        #         dio.save_dict(self.params_regs[ksize], fn, header)
        #
        #     print 'saved (many) params_regs. Last one was:', fn

        # Save kitaev region parameters as pkl
        with open(self.cp['cpmeshfn'] + 'params_regs.pkl', "wb") as fn:
            pickle.dump(self.params_regs, fn)

        # Save lattice parameters too for convenience (gives info about disorder, spinning speeds etc)
        print 'saving lattice_params...'
        header = 'Lattice parameters, copied from meshfn: ' + self.haldane_lattice.lp['meshfn']
        dio.save_dict(self.haldane_lattice.lp, self.cp['cpmeshfn'] + 'lattice_params.txt', header)

        if self.contribs is not None:
            # Save kitaev region parameters for each ksize
            # for ksize in self.contribs:
            #     header = 'Saving contribs for ksize = ' + sf.prepstr(ksize)
            #     fn = self.cp['cpmeshfn'] + 'contribs/contribs_ksize' + sf.prepstr(ksize) + '.txt'
            #     dio.save_dict(self.contribs[ksize], fn, header)
            with open(self.cp['cpmeshfn'] + 'contribs.pkl', "wb") as fn:
                pickle.dump(self.contribs, fn)


####################################
####################################
if __name__ == '__main__':
    '''Perform an example of using the HaldaneChern() class

    # Example usage:
    python haldane_chern_class.py -LT hucentroid -N 20 -shape square -ksize_frac_array 0.0:0.1:1.2 -calc_chern

    # Example for contributions
    python haldane_chern_class.py -save_ims -LT hex_kagcframe -shape circle -alph 1.0 -N 30 -contributions -modsave 1

    # Example of kagome chern with triangles ignored
    python ./haldane/haldane_chern_class.py -ignore_tris -LT kagome -shape square -N 10 -omegac 0.00
    '''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify arguments for creating HaldaneChern instance.')
    parser.add_argument('-calc_chern', '--calc_chern',
                        help='Calculate chern number for an array of kitaev sum sizes', action='store_true')
    parser.add_argument('-ensure_chern', '--ensure_chern',
                        help='Calculate chern number (for many kitaev sum sizes) if the result cannot be loaded',
                        action='store_true')
    parser.add_argument('-contributions', '--contributions',
                        help='Calculate contributions from each gyro to chern number for an array of kitaev sum sizes',
                        action='store_true')
    parser.add_argument('-projector', '--characterize_projector',
                        help='Characterize the projector with supplied params', action='store_true')
    parser.add_argument('-projector_single', '--characterize_projector_singlept',
                        help='Characterize the projector with supplied params wrt supplied pt proj_ind',
                        action='store_true')
    parser.add_argument('-proj_ind', '--proj_ind',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to this point',
                        type=int, default=-1)
    parser.add_argument('-proj_XY', '--proj_XY',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to the point closest to this location',
                        type=str, default='0.0/0.0')

    # chern parameters
    parser.add_argument('-ksize_frac_array', '--ksize_frac_array',
                        help='Array of fractional sizes to make the kitaev region, specified with /s', type=str,
                        default='0.0:0.01:1.10')
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector', type=float, default=0.0)
    parser.add_argument('-regalph', '--regalph', help='largest angle dividing kitaev region',
                        type=float, default=np.pi * (11. / 6.))
    parser.add_argument('-regbeta', '--regbeta', help='middle angle dividing kitaev region',
                        type=float, default=np.pi * (7. / 6.))
    parser.add_argument('-reggamma', '--reggamma', help='smallest angle dividing kitaev region',
                        type=float, default=np.pi * 0.5)
    parser.add_argument('-polyT', '--polyT', help='whether to transpose the kitaev region', action='store_true')
    parser.add_argument('-poly_offset', '--poly_offset', help='coordinates to translate the kitaev region, as string',
                        type=str, default='none')
    parser.add_argument('-basis', '--basis', help='basis for performing kitaev calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-modsave', '--modsave',
                        help='How often to output an image of the kitaev region and calculation result',
                        type=int, default=20)
    parser.add_argument('-save_ims', '--save_ims', help='Whether to save images of the calculations',
                        action='store_true')
    parser.add_argument('-title', '--title', help='Title of the chern calculation saved images', type=str, default='')

    # Geometry arguments for the lattices to load
    parser.add_argument('-pureimNNN', '--pureimNNN', help='Make NNN hoppings purely imaginary', action='store_true')
    parser.add_argument('-ignore_tris', '--ignore_tris',
                        help='Ignore NNN hoppings to sites which are also NN', action='store_true')
    parser.add_argument('-t2angles', '--t2angles', help='Make NNN hoppings based on bond angles', action='store_true')
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-t1', '--t1', help='NN hopping strength', type=str, default='-1.0')
    parser.add_argument('-t2', '--t2', help='NNN hopping strength prefactor', type=str, default='0.1')
    parser.add_argument('-t2a', '--t2a', help='NNN hopping strength prefactor', type=str, default='0.0')
    parser.add_argument('-pin', '--pin', help='Pinning energy (on-site)', type=str, default='0.0')
    parser.add_argument('-pinconf', '--pinconf', help='Pinning distribution realization number',
                        type=int, default=0)
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-N', '--N',
                        help='Mesh width AND height, in # lattice spacings (leave blank to specify separate dims)',
                        type=int, default=-1)
    parser.add_argument('-NP', '--NP_load',
                        help='Specify to nonzero int to load a network of a particular size in its entirety, ' +
                             'without cropping. Will override NH and NV',
                        type=int, default=0)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-LT', '--LatticeTop',
                        help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, hyperuniform, ' +
                             'circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')

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
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-perd', '--percolation_density', help='Fraction of vertices to decorate', type=float,
                        default=0.5)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='Position of single dislocation, if not at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=float, default=0.667)
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                        default=01)
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
    elif socket.gethostname() == 'Messiaen.local':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
        if not glob.glob(cprootdir):
            cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
            if not glob.glob(cprootdir):
                cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
                if not glob.glob(cprootdir):
                    if not glob.glob(cprootdir):
                        dio.ensure_dir(cprootdir)
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
          'delta_lattice': '{0:0.3f}'.format(args.delta_lattice),
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
          'periodicBC': args.periodicBC,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'origin': np.array([0., 0.]),
          't1': float((args.t1).replace('n', '-').replace('p', '.')),
          't2': float((args.t2).replace('n', '-').replace('p', '.')),
          't2a': float((args.t2a).replace('n', '-').replace('p', '.')),
          'pin': float((args.pin).replace('n', '-').replace('p', '.')),
          't2angles': args.t2angles,
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'pinconf': args.pinconf,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta': args.ABDelta,
          'pureimNNN': args.pureimNNN,
          'ignore_tris': args.ignore_tris,
          }

    cp = {'ksize_frac_arr': sf.string_sequence_to_numpy_array(args.ksize_frac_array, dtype=float),
          'omegac': args.omegac,
          'regalph': args.regalph,
          'regbeta': args.regbeta,
          'reggamma': args.reggamma,
          'shape': args.shape,
          'polyT': args.polyT,
          'poly_offset': args.poly_offset,
          'basis': args.basis,
          'modsave': args.modsave,
          'save_ims': args.save_ims,
          'rootdir': cprootdir
          }

    start = time.time()

    if args.calc_chern:
        # Calc chern number for specified network
        # try:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        # except RuntimeError:
        #     print '\n\n Could not find lattice --> creating it!'
        #     meshfn, trash = le.build_meshfn(lp)
        #     lp['meshfn'] = meshfn
        #     lat = lattice_class.Lattice(lp)
        #     lat.build()
        #     lat.save()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        print '\n\n\n', hlat.lattice.lp, '\n'
        chern = HaldaneChern(hlat, cp=cp)
        if args.title == '':
            title = None
        else:
            print 'title = ', eval("r'" + args.title.replace('_', ' ') + "'")
            title = eval("r'" + args.title.replace('_', ' ') + "'")
            title += r', with $t_1 = $' + sf.float2pstr(lp['t1'])
            title += r' $t_2 = $' + sf.float2pstr(lp['t2'])
            if hlat.lp['V0_pin_gauss']:
                title += r' $V \pm \sigma = $' + sf.float2pstr(lp['pin']) + r'$\pm$' + sf.float2pstr(lp['V0_pin_gauss'])
            else:
                title += r' $V = $' + str(int(lp['pin']))
        chern.calc_haldane_chern(check=args.check, title=title)
        chern.save_chern()

    if args.ensure_chern:
        # Calc chern number for specified network if it does not already exist
        # try:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        # except RuntimeError:
        #     print '\n\n Could not find lattice --> creating it!'
        #     meshfn, trash = le.build_meshfn(lp)
        #     lp['meshfn'] = meshfn
        #     lat = lattice_class.Lattice(lp)
        #     lat.build()
        #     lat.save()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        print '\n\n\n', hlat.lattice.lp, '\n'
        chern = HaldaneChern(hlat, cp=cp)
        if args.title == '':
            title = None
        else:
            print 'title = ', eval("r'" + args.title.replace('_', ' ') + "'")
            title = eval("r'" + args.title.replace('_', ' ') + "'")
            title += r', with $t_1 = $' + sf.float2pstr(lp['t1'])
            title += r' $t_2 = $' + sf.float2pstr(lp['t2'])
            if hlat.lp['V0_pin_gauss']:
                title += r' $V \pm \sigma = $' + sf.float2pstr(lp['pin']) + r'$\pm$' + sf.float2pstr(lp['V0_pin_gauss'])
            else:
                title += r' $V = $' + str(int(lp['pin']))
        chern.get_haldane_chern(check=args.check, title=title)

    if args.contributions:
        # Calc chern number for specified network
        try:
            meshfn = le.find_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.load()
        except RuntimeError:
            print '\n\n Could not find lattice --> creating it!'
            meshfn, trash = le.build_meshfn(lp)
            lp['meshfn'] = meshfn
            lat = lattice_class.Lattice(lp)
            lat.build()
            lat.save()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        print '\n\n\nhaldane_chern_class: ', hlat.lattice.lp, '\n'
        chern = HaldaneChern(hlat, cp=cp)
        if args.title == '':
            title = None
        else:
            title = eval("r'" + args.title.replace('_', ' ') + "'")
        chern.calc_haldane_chern(contributions=True, title=title)
        chern.save_chern()

    if args.characterize_projector:
        '''
        Example Usage:

        python haldane_chern_class.py -LT hexagonal -shape hexagon -periodic -N 8 -projector
        python haldane_chern_class.py -LT hucentroid -shape square -periodic -NP 20 -projector
        python haldane_chern_class.py -LT kagome_hucent -shape square -periodic -NP 20 -projector
        python haldane_chern_class.py -LT iscentroid -shape square -periodic -NP 256 -projector
        python haldane_chern_class.py -LT kagome_isocent -shape square -periodic -NP 256 -projector
        python haldane_chern_class.py -LT penroserhombTricent -shape square -N 10 -projector
        python haldane_chern_class.py -LT kagome_penroserhombTricent -shape square -N 10 -projector
        python haldane_chern_class.py -LT kagper_hucent -perd 0.25 -shape square -N 20 -projector
        python haldane_chern_class.py -LT kagper_hucent -perd 0.50 -shape square -N 20 -projector
        python haldane_chern_class.py -LT kagper_hucent -perd 0.75 -shape square -N 20 -projector
        '''
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        print '\n\n\nhaldane_chern_class: ', hlat.lattice.lp, '\n'
        chern = HaldaneChern(hlat, cp=cp)
        proj = chern.get_projector()
        print 'proj = ', proj
        dists, magproj, evxymag = haldane_chern_functions.characterize_projector(proj, chern.gyro_lattice)
        # plot and save characterization of locality
        network_str = leplt.lt2description(hlat.lp)
        kpfns.plot_projector_locality(hlat, dists, magproj, evxymag, outdir=None, network_str=network_str)

    if args.characterize_projector_singlept:
        # Given specified lattice index proj_pt, measure projector magnitudes of particles relative to this point
        '''
        Example usage:
        python haldane_chern_class.py -projector_single -proj_XY 0.0/0.0 -LT hex_kagframe -NH 11 -NV 12 -alph 0.5
        python haldane_chern_class.py -projector_single -proj_XY 0.0/0.0 -LT hex_kagcframe -N 20 -alph 1.0 -shape circle
        '''
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        [ax, axcb] = lat.plot_numbered(axis_off=False)
        plt.show()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        print '\n\n\nhaldane_chern_class: ', hlat.lattice.lp, '\n'
        chern = HaldaneChern(hlat, cp=cp)
        proj = chern.get_projector()
        dists, magproj, evxymag = haldane_chern_functions.characterize_projector(proj, chern.haldane_lattice, check=True)

        # plot and save characterization of locality
        network_str = leplt.lt2description(hlat.lp)

        # If args.proj_ind is not specified (negative), then use args.proj_XY to find nearest
        if args.proj_ind < 0:
            xyloc = le.string_sequence_to_numpy_array(args.proj_XY, dtype=float)
            print 'xyloc = ', xyloc
            proj_ind = ((lat.xy[:, 0] - xyloc[0])**2 + (lat.xy[:, 1] - xyloc[1])**2).argmin()
            print 'proj_ind = ', proj_ind
        else:
            proj_ind = args.proj_ind
        kpfns.plot_projector_locality_singlept(hlat, proj_ind, dists, magproj, outdir=None, network_str=network_str)

        # Save evxyproj as pkl
        evxyproj = np.dstack((proj[2 * proj_ind, :], proj[2 * proj_ind + 1, :]))[0].T
        outdir = dio.prepdir(hlat.lp['meshfn'].replace('networks', 'projectors'))
        with open(outdir + hlat.lp['LatticeTop'] + "_evxyproj_singlept_{0:06d}".format(proj_ind) + ".pkl", "wb") as fn:
            pickle.dump(evxyproj, fn)

        fig, ax, cbar_ax = kpfns.plot_projector_singlept_network(hlat, evxyproj, fig=None, ax=None,
                                                                 wsfrac=0.5, vspace=0)
        plt.savefig(outdir + hlat.lp['LatticeTop'] + '_projector_singlept_network.pdf')

    print 'elapsed time: %5.1f s' % (time.time() - start)
