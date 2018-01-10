import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.bott.bott_functions as bfns
from lepm import lattice_class
from lepm.haldane import haldane_lattice_class
import h5py
import argparse
import os
import glob
import socket
import sys
import time
import numpy.linalg as la
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl


'''
Class for computing the Bott index of a tight binding network.
Interaction with the hard drive is done through hdf5 instead of writing of text files.
'''


class BottIndex:
    """Class for a Bott index computation.

    Attributes
    ----------
    cp : dict
        'calculation parameters' --> parameter dictionary for the bott index calculation
    haldane_lattice : a haldane_lattice instance
        The tight binding network for which to compute the bott index
    bott : float
        The bott index for this haldane_lattice with current bott parameters
    """
    def __init__(self, haldane_lattice, cp=None, cpmeshfn=None, h5fn=None):
        self.haldane_lattice = haldane_lattice
        self.bott = None
        self.params_regs = {}

        if cp is None:
            cp = {}
        if 'omegac' not in cp:
            cp['omegac'] = np.array([0.0])
        if 'basis' not in cp:
            cp['basis'] = 'XY'
        if 'cpmeshfn' not in cp:
            if cpmeshfn is None:
                cpmeshfn = bfns.get_cpmeshfn(cp, self.haldane_lattice.lp)
            cp['cpmeshfn'] = cpmeshfn

        # Get subdir path from h5fn
        hfnsplit = cp['cpmeshfn'].split('/' + self.haldane_lattice.lp['LatticeTop'] + '/')
        subgroup = hfnsplit[-1][:-1]
        cp['h5fn'] = hfnsplit[0] + '.hdf5'
        cp['h5subgroup'] = subgroup
        self.cp = cp

    def load_bott(self, verbose=True):
        """Uses self.h5fn (str, The path for the hdf5 file containing the data) to load the bott index calculation
        """
        try:
            self.load_hdf5_bott()
        except KeyError:
            if verbose:
                print 'Could not find hdf5 bott file, searching for txt file...'
            self.load_txt_bott()
        except IOError or KeyError:
            if verbose:
                print 'Could not find bott in existing hdf5 file, searching for txt file...'
            self.load_txt_bott()

    def bott_in_hdf5(self):
        fi = h5py.File(self.h5fn[0], "r")
        subgroup = self.cp['h5subgroup']
        print 'to check if in hdf5, looking for subgroup = ', subgroup
        print ' ... in file: ', self.cp['h5fn']
        if subgroup in fi:
            if 'bott' in fi[subgroup].attrs:
                fi.close()
                return True
            else:
                fi.close()
                return False
        else:
            fi.close()
            return False

    def load_hdf5_bott(self):
        # Load data from hdf5 file
        fi = h5py.File(self.cp['h5fn'], "r")
        self.bott = fi[self.cp['h5subgroup']].attrs['bott']
        fi.close()
        return self.bott

    def load_txt_bott(self, verbose=True):
        """Load the bott index from a saved text file

        Returns
        -------
        bott : float
            the value of the bott index loaded from text file
        """
        # Load data from hdf5 file
        # content = np.loadtxt(bott.cp['cpmeshfn'] + 'bott.txt')
        with open(self.cp['cpmeshfn'] + 'bott.txt') as f:
            self.bott = map(float, f)[0]

        if verbose:
            print 'GyroBott: loaded bott calc from txt file'
        return self.bott

    def get_bott(self, proj=None, save=True, check=False, attribute_evs=False, verbose=False):
        """
        Load or compute bott index on self.haldane_lattice with bott parameters self.cp
        Note that to save the result as a txt file, set cp['save_as_txt'] = True.

        Parameters
        ----------
        proj : None or complex #eigvals x #eigvals array
            projection operator, if pre-supplied or already computed
        save : bool
            whether to save the bott output
        check : bool
            Display intermediate results
        attribute_evs : bool
            Whether to attribute eigenvectors and eigvals to the haldane_lattice object (if needed later for futher
            computation, for ex)
        verbose : bool
            Print output on command line

        Returns
        -------
        bott : float
            The Bott index for this tight binding network with cp parameters
        """
        if self.bott is None:
            # First try to load bott, otherwise calculate it
            # if verbose:
            print 'Searching for saved bott: ', self.cp['cpmeshfn']

            # Determine if the bott is to be loaded/saved as txt or hdf5
            if 'save_as_txt' in self.cp:
                save_as_txt = self.cp['save_as_txt']
            else:
                save_as_txt = False

            # Check if the data is on the disk (hdf5)
            data_on_disk = bfns.data_on_disk(self)
            if not data_on_disk:
                print 'searching for bott as text file...'
                data_on_disk = bfns.data_on_disk_txt(self)

            if data_on_disk:
                print '\nFound bott on hdf5, loading...'
                self.load_bott()
            else:
                # Otherwise calculate it
                print 'data_on_disk = ', data_on_disk
                print '\nBottIndex : calculating bott...'
                self.calc_bott(proj=proj, check=check, attribute_evs=attribute_evs, verbose=verbose)

                if save:
                    print 'BottIndex: saving bott...'
                    if save_as_txt:
                        self.save_bott_txt()
                    else:
                        self.save_bott()

        return self.bott

    def calc_bott(self, proj=None, check=False, verbose=False, attribute_evs=False):
        """
        Compute Bott index on self.haldane_lattice with calculation parameters self.cp
        """
        self.bott = bfns.calc_bott(self.haldane_lattice, self.cp, psub=proj, check=check,
                                   verbose=verbose, attribute_evs=attribute_evs)
        return self.bott

    def get_projector(self, attribute_evs=False):
        """Obtain projection operator for current instance of BottIndex class

        Parameters
        ----------
        attribute_evs : bool
            Attribute eigvect/eigval to self.haldane_lattice during computation of projector

        """
        if isinstance(self.cp['omegac'], np.ndarray):
            if len(self.cp['omegac']) > 1:
                print 'BottIndex: Warning! More than one value for omegac is supplied. Using the first value...'
            omegac = self.cp['omegac'][0]
        else:
            omegac = self.cp['omegac']
        proj = bfns.calc_small_projector(self.haldane_lattice, omegac, attribute=attribute_evs)
        return proj

    def save_bott(self):
        # Make output directories
        print 'saving bott to: ', self.cp['cpmeshfn']
        print ' in hdf5 file ', self.cp['h5fn']
        print 'bott = ', self.bott
        fi = h5py.File(self.cp['h5fn'], "w")
        subgroup = self.cp['h5subgroup']
        print 'subgroup = ', subgroup
        if subgroup in fi:
            print 'overwriting subgroup = ', subgroup

        fi.create_group(subgroup)
        # Save bott parameters
        cpgroup = fi[subgroup].create_group('cp')
        for key in self.cp:
            cpgroup.attrs[key] = self.cp[key]

        lpgroup = fi[subgroup].create_group('lp')
        for key in self.haldane_lattice.lp:
            lpgroup.attrs[key] = self.haldane_lattice.lp[key]

        # fi[subgroup].attrs['lp'] = self.lp
        fi[subgroup].attrs['bott'] = self.bott
        fi.close()

    def save_bott_txt(self):
        """Save the bott to disk as a txt file

        Returns
        -------
        None
        """
        # Make output directories
        print 'saving chern to: ', self.cp['cpmeshfn']
        dio.ensure_dir(self.cp['cpmeshfn'])

        # Save the chern calculation to cp['cpmeshfn']
        if self.bott is None:
            raise RuntimeError('Cannot save bott calculation since it has not been computed!')

        # Save bott parameters
        fn = self.cp['cpmeshfn'] + 'bott_params.txt'
        header = 'Parameters for bott calculation'
        dio.save_dict(self.cp, fn, header)

        # Save chern_finsize
        fn = self.cp['cpmeshfn'] + 'bott.txt'

        print 'saving chern_finsize to ', fn
        with open(fn, "w") as txtfile:
            txtfile.write('{0:0.18e}'.format(self.bott))

        # Save lattice parameters too for convenience (gives info about disorder, spinning speeds etc)
        print 'saving lattice_params...'
        header = 'Lattice parameters, copied from meshfn: ' + self.haldane_lattice.lp['meshfn']
        dio.save_dict(self.haldane_lattice.lp, self.cp['cpmeshfn'] + 'lattice_params.txt', header)


####################################
####################################
if __name__ == '__main__':
    '''Perform an example of using the kitaev_collection class

    # Example usage:
    python bott_class.py -LT hucentroid -N 30 -shape square -contributions -ksize_frac_array 0.0:0.1:1.2
    python bott_class.py -save_ims -LT hex_kagcframe -shape circle -alph 0.3 -calc_bott -N 30 -modsave 1
    python ./kitaev/bott_class.py -LT accordionkag -N 5 -shape square -contributions -Nks 110 -intparam 2 -alph 1.0
    python ./kitaev/bott_class.py -LT accordionkag -N 6 -shape square -calc_bott -Nks 110 -intparam 2 -alph 0.85

    # Example for contributions
    python bott_class.py -save_ims -LT hex_kagcframe -shape circle -alph 1.0 -N 30 -contributions -modsave 1
    '''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for haldane simulation.')
    parser.add_argument('-calc_bott', '--calc_bott',
                        help='Calculate Bott Index for determined hlat', action='store_true')
    parser.add_argument('-save_as_txt', '--save_as_txt',
                        help='Enforce saving bott as txt file rather than hdf5', action='store_true')
    parser.add_argument('-pin2hdf5', '--pin2hdf5',
                        help='Enforce saving pinning as hdf5 rather than as txt', action='store_true')
    parser.add_argument('-ensure_bott', '--ensure_bott',
                        help='Calculate bott index for an array of kitaev sum sizes if not already saved',
                        action='store_true')
    parser.add_argument('-verbose', '--verbose', help='Print verbose output during computation',
                        action='store_true')

    # Bott parameters
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector', type=float, default=0.0)
    parser.add_argument('-basis', '--basis', help='basis for performing kitaev calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-title', '--title', help='Title of the Bott calculation saved images', type=str, default='')

    # Geometry and physics arguments
    parser.add_argument('-pureimNNN', '--pureimNNN', help='Make NNN hoppings purely imaginary', action='store_true')
    parser.add_argument('-t2angles', '--t2angles', help='Make NNN hoppings based on bond angles', action='store_true')
    parser.add_argument('-hexNNN', '--hexNNN', help='Ignore NNN hoppings in polygons other than hexagons',
                        action='store_true')
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.1)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
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
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-t1', '--t1', help='NN hopping strength', type=str, default='-1.0')
    parser.add_argument('-t2', '--t2', help='NNN hopping strength prefactor (imaginary part)', type=str, default='0.1')
    parser.add_argument('-t2a', '--t2a', help='NNN hopping strength real component', type=str, default='0.0')
    parser.add_argument('-theta_twist', '--theta_twist', help='Twisted phase in x for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-phi_twist', '--phi_twist', help='Twisted phase in y for periodic BCs', type=float,
                        default=0.0)
    parser.add_argument('-pin', '--pin', help='Pinning energy (on-site)', type=str, default='0.0')
    parser.add_argument('-pinconf', '--pinconf', help='Pinning distribution realization number',
                        type=int, default=0)
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
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

    # Global geometric params
    parser.add_argument('-nonperiodic', '--openBC', help='Enforce open (non periodic) boundary conditions',
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
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float,
                        default=0.000)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                        default=01)
    parser.add_argument('-thres', '--thres', help='Threshold value for building networks (determining to decorate pt)',
                        type=float, default=1.0)
    parser.add_argument('-spreading_time', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)', type=float, default=0.0)
    parser.add_argument('-kicksz', '--kicksz',
                        help='Average of log of kick magnitudes for loading randorg_gammakick pointsets.' +
                             'This sets the scale of the powerlaw kicking procedure',
                        type=float, default=-1.50)
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

    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    hostname = socket.gethostname()
    print 'hostname = ', hostname
    # sys.exit()
    if hostname[0:6] == 'midway':
        print '\n\nWe are on Midway!\n\n\n\n'
        rootdir = '/home/npmitchell/scratch-midway/'
        cprootdir = '/home/npmitchell/scratch-midway/'
    elif hostname[0:10] == 'nsit-dhcp-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
    elif hostname == 'Messiaen.local' or hostname[0:8] == 'wireless':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
        if not os.path.isdir(cprootdir):
            cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
    elif hostname[0:5] == 'cvpn-':
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
          'save_pinning_to_hdf5': args.pin2hdf5,
          'spreading_time': args.spreading_time,
          'kicksz': args.kicksz,
          }

    omegac = args.omegac

    cp = {'omegac': omegac,
          'basis': args.basis,
          'rootdir': cprootdir,
          'save_as_txt': args.save_as_txt,
         }

    start = time.time()
    if args.title == '':
        title = None
    else:
        print 'title = ', eval("r'" + args.title.replace('_', ' ') + "'")
        title = eval("r'" + args.title.replace('_', ' ') + "'")

    if args.calc_bott:
        # Calc bott index for specified network
        # try:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        # print 'lat.lp[PV] = ', lat.lp['PV']
        print 'lat.lp[periodicBC] = ', lat.lp['periodicBC']
        print 'lat.PV = ', lat.PV
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
        bott = BottIndex(hlat, cp=cp)

        bott.calc_bott(check=args.check, verbose=args.verbose)
        bott.save_bott_txt()

    if args.ensure_bott:
        # Calc bott index for specified network
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
        bott = BottIndex(hlat, cp=cp)
        bott.get_bott(check=args.check, verbose=args.verbose)
