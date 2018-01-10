import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.bott.bott_gyro_functions as bgfns
import lepm.plotting.plotting as leplt
import lepm.plotting.kitaev_plotting_functions as kpfns
from lepm import lattice_class
from lepm import gyro_lattice_class
import h5py
import matplotlib.pyplot as plt
import argparse
import os
import glob
import socket
import sys
import copy
import time
import lepm.stringformat as sf
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl


'''
Class for computing the Bott index of a gyro network.
Interaction with the hard drive is done through hdf5 instead of writing of text files.
'''


class GyroBott:
    """Class for a Bott index computation.

    Attributes
    ----------
    cp : dict
        'calculation parameters' --> parameter dictionary for the bott index calculation
    gyro_lattice : a gyro_lattice instance
        The gyroscopic network for which to compute the bott index
    bott : float
        The bott index for this gyro_lattice with current bott parameters
    """
    def __init__(self, gyro_lattice, cp=None, cpmeshfn=None, h5fn=None):
        self.gyro_lattice = gyro_lattice
        self.bott = None
        self.params_regs = {}

        if cp is None:
            cp = {}
        if 'omegac' not in cp:
            cp['omegac'] = np.array([2.25])
        if 'basis' not in cp:
            cp['basis'] = 'XY'
        if 'cpmeshfn' not in cp:
            if cpmeshfn is None:
                cpmeshfn = bgfns.get_cpmeshfn(cp, self.gyro_lattice.lp)
            cp['cpmeshfn'] = cpmeshfn

        # Get subdir path from h5fn
        hfnsplit = cp['cpmeshfn'].split('/' + self.gyro_lattice.lp['LatticeTop'] + '/')
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
        Load or compute bott index on self.gyro_lattice with bott parameters self.cp
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
            Whether to attribute eigenvectors and eigvals to the gyro_lattice object (if needed later for futher
            computation, for ex)
        verbose : bool
            Print output on command line

        Returns
        -------
        bott : float
            The Bott index for this gyro network with cp parameters
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

            # Check if the data is on the disk (look for hdf5, then txt)
            data_on_disk = bgfns.data_on_disk(self)
            # print 'data_on_disk = ', data_on_disk
            # sys.exit()
            if not data_on_disk:
                print 'searching for bott as text file...'
                data_on_disk = bgfns.data_on_disk_txt(self)

            if data_on_disk:
                print '\nFound bott on hdf5 or txt file, loading...'
                self.load_bott()
            else:
                # Otherwise calculate it
                print 'data_on_disk = ', data_on_disk
                # print 'bott_gyro: could not find bott saved, exiting...'
                # sys.exit()
                print '\nBottGyro : calculating bott...'
                self.calc_bott(proj=proj, check=check, attribute_evs=attribute_evs, verbose=verbose)

                if save:
                    print 'BottGyro: saving bott...'
                    if save_as_txt:
                        self.save_bott_txt()
                        print 'BottGyro: saved bott as text'
                    else:
                        self.save_bott()
                    print 'BottGyro: saved it!'
                # sys.exit()

        return self.bott

    def calc_bott(self, proj=None, check=False, verbose=False, attribute_evs=False):
        """
        Compute Bott index on self.gyro_lattice with calculation parameters self.cp
        """
        self.bott = bgfns.calc_bott(self.gyro_lattice, self.cp, psub=proj, check=check,
                                    verbose=verbose, attribute_evs=attribute_evs)
        return self.bott

    def get_projector(self, attribute_evs=False):
        """Obtain projection operator for current instance of BottIndex class

        Parameters
        ----------
        attribute_evs : bool
            Attribute eigvect/eigval to self.gyro_lattice during computation of projector

        """
        if isinstance(self.cp['omegac'], np.ndarray):
            if len(self.cp['omegac']) > 1:
                print 'BottIndex: Warning! More than one value for omegac is supplied. Using the first value...'
            omegac = self.cp['omegac'][0]
        else:
            omegac = self.cp['omegac']
        proj = bgfns.calc_small_projector(self.gyro_lattice, omegac, attribute=attribute_evs)
        return proj

    def save_bott(self):
        # Make output directories
        print 'saving bott to: ', self.cp['cpmeshfn']
        print ' in hdf5 file ', self.cp['h5fn']
        if glob.glob(self.cp['h5fn']):
            fi = h5py.File(self.cp['h5fn'], "r+")
        else:
            fi = h5py.File(self.cp['h5fn'], "w")

        subgroup = self.cp['h5subgroup']
        print 'subgroup = ', subgroup
        if subgroup in fi:
            print 'overwriting subgroup = ', subgroup

        fi.require_group(subgroup)
        # Save bott parameters
        cpgroup = fi[subgroup].require_group('cp')
        for key in self.cp:
            cpgroup.attrs[key] = self.cp[key]

        lpgroup = fi[subgroup].require_group('lp')
        for key in self.gyro_lattice.lp:
            lpgroup.attrs[key] = self.gyro_lattice.lp[key]

        fi[subgroup].attrs['bott'] = self.bott
        fi.close()

    def save_bott_txt(self):
        """Save the bott to disk as a txt file

        Returns
        -------
        None
        """
        # Make output directories
        print 'saving bott to: ', self.cp['cpmeshfn']
        dio.ensure_dir(self.cp['cpmeshfn'])

        # Save the bott calculation to cp['cpmeshfn']
        if self.bott is None:
            raise RuntimeError('Cannot save bott calculation since it has not been computed!')

        # Save bott parameters
        fn = self.cp['cpmeshfn'] + 'bott_params.txt'
        header = 'Parameters for bott calculation'
        dio.save_dict(self.cp, fn, header)

        # Save bott_finsize
        fn = self.cp['cpmeshfn'] + 'bott.txt'

        print 'saving bott to ', fn
        with open(fn, "w") as txtfile:
            txtfile.write('{0:0.18e}'.format(self.bott))

        # Save lattice parameters too for convenience (gives info about disorder, spinning speeds etc)
        print 'saving lattice_params...'
        header = 'Lattice parameters, copied from meshfn: ' + self.gyro_lattice.lp['meshfn']
        dio.save_dict(self.gyro_lattice.lp, self.cp['cpmeshfn'] + 'lattice_params.txt', header)


####################################
####################################
if __name__ == '__main__':
    '''Perform an example of using the kitaev_collection class

    # Example usage:
    # Check bowtie lattice has bott of +1:
    python ./bott/bott_gyro.py -LT hexagonal -N 15 -save_as_txt -delta 1.275 -hexbow_omegac -calc_bott

    python bott_class.py -LT hucentroid -N 30 -shape square -contributions -ksize_frac_array 0.0:0.1:1.2
    python bott_class.py -save_ims -LT hex_kagcframe -shape circle -alph 0.3 -calc_bott -N 30 -modsave 1
    python ./kitaev/bott_class.py -LT accordionkag -N 5 -shape square -contributions -Nks 110 -intparam 2 -alph 1.0
    python ./kitaev/bott_class.py -LT accordionkag -N 6 -shape square -calc_bott -Nks 110 -intparam 2 -alph 0.85

    # Example for contributions
    python bott_class.py -save_ims -LT hex_kagcframe -shape circle -alph 1.0 -N 30 -contributions -modsave 1
    '''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-calc_bott', '--calc_bott',
                        help='Calculate Bott Index for determined glat', action='store_true')
    parser.add_argument('-save_as_txt', '--save_as_txt',
                        help='Enforce saving as txt file rather than hdf5', action='store_true')
    parser.add_argument('-pin2hdf5', '--pin2hdf5',
                        help='Enforce saving pinning as hdf5 rather than as txt', action='store_true')
    parser.add_argument('-ensure_bott', '--ensure_bott',
                        help='Calculate bott index for an array of kitaev sum sizes if not already saved',
                        action='store_true')
    parser.add_argument('-contributions', '--contributions',
                        help='Calculate contributions from each gyro to bott index for an array of kitaev sum sizes',
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

    # bott parameters
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector', type=float, default=2.25)
    parser.add_argument('-hexbow_omegac', '--hexbow_omegac',
                        help='decide on cutoff freq as middle of the gap for a deformed honeycomb lattice',
                        action='store_true')
    parser.add_argument('-basis', '--basis', help='basis for performing kitaev calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-title', '--title', help='Title of the bott calculation saved images', type=str, default='')

    # Geometry arguments for the lattices to load
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-pinconf', '--pinconf', help='Pinning distribution realization number',
                        type=int, default=0)
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
                        help='Mesh width AND height, in # lattice spacings (leave blank to specify separate dims)',
                        type=int, default=-1)
    parser.add_argument('-NP', '--NP_load',
                        help='Specify to nonzero int to load a network of a particular size in its entirety, ' +
                             'without cropping. Will override NH and NV',
                        type=int, default=20)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=07)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=07)
    parser.add_argument('-LT', '--LatticeTop',
                        help='Lattice topology: linear, hexagonal, triangular, deformed_kagome, hyperuniform, ' +
                             'circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-check', '--check', help='Check outputs during computation of lattice', action='store_true')
    parser.add_argument('-verbose', '--verbose', help='Output to command line during bott calc', action='store_true')

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
    parser.add_argument('-spreading_time', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.0)
    parser.add_argument('-kicksz', '--kicksz',
                        help='Average of log of kick magnitudes for loading randorg_gammakick pointsets.' +
                             'This sets the scale of the powerlaw kicking procedure',
                        type=float, default=-1.50)

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
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
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
    elif socket.gethostname()[0:10] in ['nsit-dhcp-', 'npmitchell']:
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
    elif socket.gethostname() == 'Messiaen.local':
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
          'periodicBC': not args.openBC,
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
          'bl0': args.bl0,
          'save_pinning_to_hdf5': args.pin2hdf5,
          'kicksz': args.kicksz,
    }

    if args.OmKspec is not '':
        lp['OmKspec'] = args.OmKspec
    elif args.dice:
        lp['OmKspec'] = 'gridlines{0:0.2f}'.format(args.gridspacing).replace('.', 'p') + \
                        'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
                        'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')

    if args.hexbow_omegac:
        import lepm.kitaev.kitaev_collection_functions as kcfns
        print 'bott_gyro.py: finding appropriate omegac for this geometry of lattice...'
        print 'delta_lattice --> ', float(args.delta_lattice)
        print 'angle = ', float(args.delta_lattice) * np.pi
        omegac = kcfns.gap_midpoints_honeycomb(float(args.delta_lattice) * np.pi)
        print 'omegac = ', omegac
    else:
        omegac = args.omegac
    # sys.exit()

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
        print 'meshfn = ', meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        # print 'lat.lp[PV] = ', lat.lp['PV']
        print 'lat.lp[periodicBC] = ', lat.lp['periodicBC']
        print 'lat.PV = ', lat.PV
        print 'lat.lp[delta_lattice] = ', lat.lp['delta_lattice']
        print 'lat.lp[delta] = ', lat.lp['delta']
        sys.exit()
        # except RuntimeError:
        #     print '\n\n Could not find lattice --> creating it!'
        #     meshfn, trash = le.build_meshfn(lp)
        #     lp['meshfn'] = meshfn
        #     lat = lattice_class.Lattice(lp)
        #     lat.build()
        #     lat.save()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        print '\n\n\n', glat.lattice.lp, '\n'
        bott = GyroBott(glat, cp=cp)

        bott.calc_bott(check=args.check, verbose=args.verbose)

        # Ensure the gyro lattice
        xy = bott.gyro_lattice.lattice.xy
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        plt.show()
        sys.exit()

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
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        print '\n\n\n', glat.lattice.lp, '\n'
        bott = GyroBott(glat, cp=cp)
        bott.get_bott(check=args.check, verbose=args.verbose)
