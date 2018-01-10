import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
from lepm import lattice_class
from lepm.haldane import haldane_lattice_class
from lepm.haldane import haldane_chern_collection
from lepm.haldane import haldane_chern_class
from lepm.haldane import haldane_chern_collection
import argparse
import matplotlib.pyplot as plt
import copy
import socket
import glob
import cPickle as pkl
import lepm.stringformat as sf

'''
Collections of collections of Chern number measurements made via the Kitaev realspace method on Haldane lattices.
Each collection of measurements is grouped by a parameter defining the lattice used, with the collection containing
different haldane_lattices defined on that lattice. These haldane_lattices will have a hlatparam (a physics parameter on
the haldane lattice) that varies from hlat to hlat.

This class has not yet been tested
'''


class HaldaneChernCollectionCollection:
    """Create a collection of collections of chern measurements for haldane spring networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.cherns is a list of tuples of dicts: self.cherns = [chern1, chern2,...]
    where chern is a class with attributes cp, chern_finsize array, params_regs dict.
       cp : dict
           keys : str
               'meshfn', 'omegac', 'poly_offset', 'ksize_frac_arr', 'regalph', 'regbeta', 'reggamma', 'polyT',
               'poly_offset'
       chern_finsize : len(ksize) x 5 float array
           contains [Nreg1, ksize_frac, ksize, ksys_size (note this is 2*NP_summed), ksys_frac, nu for Chern calculation]
       params_regs : dict
           a nested dictionary with key,value pairs given by
               keys : str
                   each key is a string element of ksize: '{0:0.3f}'.format(ksize)
               values : dict
                   dictionary with key,value pairs given by
                       reg1_xy : int list
                           list of self.haldane_lattice.lattice.xy indices in reg1 of haldane_chern sum region
                       reg2_xy : int list
                           list of self.haldane_lattice.lattice.xy indices in reg2 of haldane_chern sum region
                       reg3_xy : int list
                           list of self.haldane_lattice.lattice.xy indices in reg3 of haldane_chern sum region
                       polygon1 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg1
                       polygon2 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg2
                       polygon3 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg3
                       reg1 : int list
                           list of self.haldane_lattice.lattice.xy indices in reg1 of haldane_chern sum region
                       reg2 : int list
                           list of self.haldane_lattice.lattice.xy indices in reg2 of haldane_chern sum region
                       reg3 : int list
                           list of self.haldane_lattice.lattice.xy indices in reg3 of haldane_chern sum region

    Attributes
    ----------
    self.param_keys : list of string, float, etc parameter
       list of keys for the self.haldane_chern_collections dict
    self.haldane_chern_collections : dict
       keys are hlat_names (strings)
    """
    def __init__(self, haldane_chern_collections={}, param_keys=[]):  # lattices=[], lattice_names=[]):
        """Create an instance of a lattice_collection."""
        self.haldane_chern_collections = haldane_chern_collections
        self.param_keys = param_keys
        # self.lattice_names = lattice_names
        # self.lattices = lattices

    def add_haldane_chern_collection(self, hccoll, param_key):
        """Group a collection of measurements under the lattice used

        Parameters
        ----------
        hccoll : HaldaneChernCollection instance
            The haldane_chern collection to add
        param_key : object of any type
            the identifier (key for a dict) under which the haldane_chern collection is stored
        """
        self.haldane_chern_collections[param_key] = hccoll
        self.param_keys.append(param_key)

    def collect_maxchern_lpparam_hlatparam(self, hlatparam='ABDelta'):
        """Collect the extremal chern values (averaged if there are more than one vertex locations) while varying a
        lattice parameter (lpparam) AND a GryoLattice parameter (hlatparam)
        """
        maxcherns = []
        hlatV = []
        lpV = []
        for key in self.haldane_chern_collections:
            print 'key = ', key
            for hlat_name in self.haldane_chern_collections[key].hlat_names:
                cherns = self.haldane_chern_collections[key].cherns[hlat_name]
                # print 'cherns = ', cherns
                if len(cherns) > 1:
                    # Average their values -- this would be for a varyloc, for instance (a spatially
                    # distributed collection of cherns)
                    # NOTE: We assume that all the cherns here have the SAME hlat parameter!
                    chernmat = np.zeros((len(cherns[0].chern_finsize[:, -1]), len(cherns)))
                    for dmyi in range(len(cherns)):
                        chernmat[:, dmyi] = cherns[dmyi].chern_finsize[:, -1]

                    # Average along the rows of chernmat (chern matrix)
                    chernv = np.mean(chernmat, axis=1)
                else:
                    chernv = cherns[0].chern_finsize[:, -1]

                ind = np.argmax(np.abs(chernv))
                maxcherns.append(chernv[ind])
                lpV.append(key)
                hlatV.append(cherns[0].haldane_lattice.lp[hlatparam])

        print 'lpV = ', lpV
        print 'hlatV = ', hlatV
        print 'maxcherns = ', maxcherns

        return lpV, hlatV, maxcherns

    # def collect_maxchern_varyloc_varyomegac(self, omegacV=np.arange(1.0, 4.0, 0.1)):
    #     """Collect chern instances for varying the vertex location AND varying cutoff freqeucny of projector.
    #     omegacV is a n x 1 array of cutoff freqs, polyoff2D is a m x 2 array of x, y positions,
    #     maxcherns is n x m array
    #     of extremal chern values.
    #
    #     Parameters
    #     ----------
    #     omegacV : 1d float array
    #         cutoff frequency values for the projector
    #     """
    #     maxcherns = []
    #     for key in self.haldane_chern_collections:
    #         print 'key = ', key
    #         hlat_names = self.haldane_chern_collections[key].hlat_names
    #         if len(hlat_names) > 1:
    #             raise RuntimeError("hccollcoll.collect_maxchern_varyloc should have haldane_chern_collections" +
    #                                "based on only one single HaldaneLattice instance")
    #         hlat_name = hlat_names[0]
    #         cherns = self.haldane_chern_collections[key].cherns[hlat_name]
    #         # print 'cherns = ', cherns
    #         if len(cherns) > 1:
    #             # Average their values -- this would be for a varyloc, for instance (a spatially
    #             # distributed collection of cherns)
    #             # NOTE: We assume that all the cherns here have the SAME hlat parameter!
    #             chernmat = np.zeros((len(cherns[0].chern_finsize[:, -1]), len(cherns)))
    #             for dmyi in range(len(cherns)):
    #                 chernmat[:, dmyi] = cherns[dmyi].chern_finsize[:, -1]
    #
    #             # Average along the rows of chernmat (chern matrix)
    #             chernv = np.mean(chernmat, axis=1)
    #         else:
    #             chernv = cherns[0].chern_finsize[:, -1]
    #
    #         ind = np.argmax(np.abs(chernv))
    #         maxcherns.append(chernv[ind])
    #         lpV.append(key)
    #         hlatV.append(cherns[0].haldane_lattice.lp[hlatparam])
    #
    #     print 'lpV = ', lpV
    #     print 'hlatV = ', hlatV
    #     print 'maxcherns = ', maxcherns
    #
    #     return omegacV, polyoff2D, maxcherns

    def plot_maxchern_lpparam_hlatparam(self, lhchern=None, hlatparam='ABDelta'):
        """"""
        if lhchern is None:
            lpV, hlatV, maxcherns = self.collect_maxchern_lpparam_hlatparam(hlatparam=hlatparam)
            lhchern = np.dstack((lpV, hlatV, maxcherns))[0]

        print 'lhchern = ', lhchern
        ngrid = len(lhchern) + 7
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent()
        leplt.plot_pcolormesh(lhchern[:, 0], lhchern[:, 1], lhchern[:, 2], ngrid, ax=ax, cax=cbar_ax, method='nearest',
                              make_cbar=True, cmap=lecmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1., vmax=1., title=None, xlabel=None, ylabel=None, ylabel_right=True,
                              ylabel_rot=90, cax_label=r'Chern number, $\nu$',
                              cbar_labelpad=-30, cbar_orientation='horizontal',
                              ticks=[-1, 0, 1], fontsize=8, title_axX=None, title_axY=None, alpha=1.0)
        plt.show()


if __name__ == '__main__':
    '''Perform an example of using the haldane_chern_collection_collection class.

    Example usage to create AB site phase diagram:
    python haldane_chern_collection_collection.py -ABphase
    '''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create collection of haldane_chern chern calculations.')
    # Script options
    parser.add_argument('-vary_omegac', '--calc_cherns_omegac',
                        help='Compute chern number using a range of omegac values for each hlat in the collection' +
                             ' -- ie, create a chern spectrum',
                        action='store_true')
    parser.add_argument('-varyloc', '--calc_cherns_varyloc',
                        help='Compute chern number using a grid of haldane_chern region locations for each' +
                             'haldane lattice in the collection', action='store_true')
    parser.add_argument('-varyloc_reverse', '--varyloc_reverse',
                        help='When computing chern number using a grid of locations, reverse the order of computing' +
                             ' on the grid', action='store_true')
    parser.add_argument('-ABphase', '--ABphase',
                        help='Do multiple chern number calcs for many HaldaneLattices build on THE SAME Lattice ' +
                             'instance with an lp_param varying between them, then also vary which lattice to use, ' +
                             'by varying ABDelta',
                        action='store_true')
    parser.add_argument('-chern_varyloc_varyomegac', '--chern_varyloc_varyomegac',
                        help='Plot the extremal chern number as a function of space for different projector ' +
                             'cutoff freqs',
                        action='store_true')
    parser.add_argument('-spectrum_varyloc_varyomegac', '--spectrum_varyloc_varyomegac',
                        help='Plot the chern spectra averaged over different vertex positions',
                        action='store_true')
    parser.add_argument('-chern_varyloc_varyhlatparam', '--chern_varyloc_varyhlatparam',
                        help='Plot the extremal chern number as a function of space for different values of a ' +
                             'HaldaneLattice parameter dict (hlat.lp)',
                        action='store_true')
    parser.add_argument('-singleksz_frac', '--singleksz_frac',
                        help='Fractional size (in units of sys size) of the haldane_chern summation region to ' + \
                             'use for plotting spatially resolved chern number. Ignored if singleksz > 0 or ' + \
                             'maxchern = True.',
                        type=float, default=-1.)
    parser.add_argument('-singleksz', '--singleksz',
                        help='Size of the haldane_chern summation region to use for plotting spatially' + \
                             ' resolved chern number. Ignored if maxchern == True.',
                        type=float, default=-1.)
    parser.add_argument('-maxchern', '--maxchern',
                        help='Use extremal value of chern number when plotting spatially resolved chern number',
                        action='store_true')
    parser.add_argument('-spectrum_varyloc_varyhlatparam', '--spectrum_varyloc_varyhlatparam',
                        help='Plot the chern spectra averaged over different vertex positions',
                        action='store_true')
    # parser.add_argument('-vary_lpparam_vary_hlatparam', '--vary_lpparam_vary_hlatparam',
    #                     help='Do multiple chern number calcs for many HaldaneLattices build on THE SAME Lattice ' +
    #                          'instance with a parameter varying between them, then also vary which lattice to use, ' +
    #                          'by varying hlatparam',
    #                     action='store_true')

    # Options for script method choice
    parser.add_argument('-Nks', '--Nks',
                        help='How many haldane_chern region sizes to sample at each site if -calc_cherns_varyloc',
                        type=int, default=0)
    parser.add_argument('-step', '--step', help='Step size to take while sampling the regions of the network via chern',
                        type=float, default=1.0)
    parser.add_argument('-hlatparam', '--hlatparam',
                        help='String specifier for which parameter is varied across a single haldane ' + \
                             'lattice in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-hlatparam_reverse', '--hlatparam_reverse',
                        help='When computing chern number varying a haldanelattice param, reverse the order ' + \
                             'of computing',
                        action='store_true')
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across haldane lattices in collection',
                        type=str, default='delta')
    parser.add_argument('-paramV', '--paramV',
                        help='Sequence of values to assign to lp[param] if vary_lpparam is True',
                        type=str, default='0.0:0.1:2.0')
    parser.add_argument('-locV', '--locV',
                        help='Sequence of xyoffset values to assign to cp[poly_offset] if method involves varyloc',
                        type=str, default='n6.0:1.0:6.0')
    parser.add_argument('-chern_collection', '--chern_collection',
                        help='Whether to collect cherns by LT and analyze their cherns',
                        action='store_true')
    parser.add_argument('-annulus_chern', '--annulus_chern',
                        help='Whether to collect cherns computed with an annular summation region',
                        action='store_true')
    parser.add_argument('-proj_site_hlatparam', '--proj_site_hlatparam',
                        help='Compare the projector elements near a point proj_XY for a collection of hlats',
                        action='store_true')
    parser.add_argument('-proj_XY', '--proj_XY',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to the point closest to this location',
                        type=str, default='0.0/0.0')

    # chern parameters
    parser.add_argument('-ksize_frac_array', '--ksize_frac_array',
                        help='Array of fractional sizes to make the haldane_chern region, specified with /s', type=str,
                        default='0.0:0.01:1.10')
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector, specified as string with /s',
                        type=str, default='2.25')
    parser.add_argument('-regalph', '--regalph', help='largest angle dividing haldane_chern region',
                        type=float, default=np.pi * (11. / 6.))
    parser.add_argument('-regbeta', '--regbeta', help='middle angle dividing haldane_chern region',
                        type=float, default=np.pi * (7. / 6.))
    parser.add_argument('-reggamma', '--reggamma', help='smallest angle dividing haldane_chern region',
                        type=float, default=np.pi * 0.5)
    parser.add_argument('-polyT', '--polyT', help='whether to transpose the haldane_chern region', action='store_true')
    parser.add_argument('-poly_offset', '--poly_offset',
                        help='coordinates to translate the haldane_chern region, as string',
                        type=str, default='none')
    parser.add_argument('-basis', '--basis', help='basis for performing haldane_chern calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-modsave', '--modsave',
                        help='How often to output an image of the haldane_chern region and calculation result',
                        type=int, default=40)
    parser.add_argument('-save_ims', '--save_ims', help='Whether to save images of the calculations',
                        action='store_true')

    # Geometry arguments for the lattices to load
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings' +
                                          ' (leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load', help='Specify to nonzero int to load a network of a particular size' +
                                                 ' in its entirety, without cropping. Will override NH and NV',
                        type=int, default=0)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                    'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
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
    parser.add_argument('-LLz', '--loadlattice_z', help='If LT=hyperuniform/isostatic, selects what z index to use',
                        type=str, default='001')
    parser.add_argument('-source', '--source',
                        help='Selects who made the lattice to load, if loaded from source (ulrich, hexner, etc)',
                        type=str, default='hexner')
    parser.add_argument('-cut_z', '--cut_z', help='Declare whether or not to cut bonds to obtain target' +
                                                  'coordination number z', type=bool, default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-perd', '--percolation_density', help='Fraction of vertices to decorate', type=float,
                        default=0.5)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy', help='Position of single dislocation, if not centered' +
                                                              ' at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.667')
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-conf2', '--sub_realization_number', help='Decoration realization number', type=int, default=1)
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
    lattice_type = args.LatticeTop

    # phi = np.pi* args.phi
    # delta = np.pi* args.delta

    strain = 0.00  # initial
    # z = 4.0 #target z
    if lattice_type == 'linear':
        shape = 'line'
    else:
        shape = args.shape

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

    if socket.gethostname()[0:6] == 'midway':
        rootdir = '/home/npmitchell/scratch-midway/'
        cp_rootdir = rootdir
    elif socket.gethostname()[0:8] == 'Messiaen' or socket.gethostname()[0:5] == 'cvpn-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cp_rootdir = '/Users/npmitchell/Desktop/GPU/'
    else:
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cp_rootdir = '/Volumes/research4TB/Soft_Matter/GPU/'

    outdir = rootdir + 'experiments/DOS_scaling/' + args.LatticeTop + '/'
    dio.ensure_dir(outdir)

    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': rootdir,
          'phi_lattice': args.phi_lattice,
          'delta_lattice': args.delta_lattice,
          'theta': args.theta,
          'eta': args.eta,
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
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'viewmethod': False,
          'ABDelta': args.ABDelta,
          }

    ksize_frac_arr = sf.string_sequence_to_numpy_array(args.ksize_frac_array, dtype=float)

    print 'ksize_frac_arr = ', ksize_frac_arr
    cp = {'ksize_frac_arr': ksize_frac_arr,
          'omegac': sf.string_sequence_to_numpy_array(args.omegac, dtype=float),
          'regalph': args.regalph,
          'regbeta': args.regbeta,
          'reggamma': args.reggamma,
          'shape': args.shape,
          'polyT': args.polyT,
          'poly_offset': args.poly_offset,
          'basis': args.basis,
          'modsave': args.modsave,
          'save_ims': args.save_ims,
          'rootdir': cp_rootdir
          }

    print 'Creating chern collection from haldane_collection...'
    if args.Nks == 211:
        cp['ksize_frac_arr'] = np.arange(0.0, 1.0500001, 0.005)
        fps = 20
    elif args.Nks == 201:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.5000001, 0.0025)
        fps = 20
    elif args.Nks == 121:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.3000001, 0.0025)
        fps = 12
    elif args.Nks == 76:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.76, 0.01)
        fps = 7
    elif args.Nks == 30:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.30, 0.01)
        fps = 3
    elif args.Nks == 31:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.31, 0.01)
        fps = 3

    print '\n\n cp_rootdir = ', cp_rootdir

    if args.ABphase:
        """Vary an lp and AB parameter to make the Lisa lattice deformation phase diagram"""
        from lepm.haldane import haldane_collection
        # Collate cherns for one lattice with a haldane_lattice parameter that varies between instances of that lattice
        lp_master = copy.deepcopy(lp)
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        if args.lpparam == 'delta':
            # vary delta between lattices
            deltaV = np.arange(0.7, 1.31, 0.1)
            deltaV = np.hstack((0.667, deltaV))
            hccollcoll = HaldaneChernCollectionCollection()
            for delta in deltaV:
                lp = copy.deepcopy(lp_master)
                lp['delta_lattice'] = '{0:0.3f}'.format(delta)
                meshfn = le.find_meshfn(lp)
                lp['meshfn'] = meshfn
                print '\n\n\nlp[meshfn] = ', lp['meshfn']
                lat = lattice_class.Lattice(lp)
                lat.load()
                hc = haldane_collection.HaldaneCollection()
                lp_submaster = copy.deepcopy(lp)
                for hlatpval in paramV:
                    lpii = copy.deepcopy(lp_submaster)
                    lpii[args.hlatparam] = hlatpval
                    hlat = haldane_lattice_class.HaldaneLattice(lat, lpii)
                    hlat.load()
                    hc.add_haldane_lattice(hlat)

                print 'Creating chern collection from single-lattice haldane_collection...'
                hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)
                hccoll.get_cherns(reverse=args.hlatparam_reverse)
                hccollcoll.add_haldane_chern_collection(hccoll, delta)
                print '\n\n\n\nhccollcoll.haldane_chern_collections = ', hccollcoll.haldane_chern_collections

            hccollcoll.plot_maxchern_lpparam_hlatparam()

    if args.ABphase:
        """Vary an lp and AB parameter to make the Lisa lattice deformation phase diagram"""
        from lepm.haldane import haldane_collection
        # Collate cherns for one lattice with a haldane_lattice parameter that varies between instances of that lattice
        lp_master = copy.deepcopy(lp)
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        if args.lpparam == 'delta':
            # vary delta between lattices
            deltaV = np.arange(0.7, 1.31, 0.1)
            deltaV = np.hstack((0.667, deltaV))
            hccollcoll = HaldaneChernCollectionCollection()
            for delta in deltaV:
                lp = copy.deepcopy(lp_master)
                lp['delta_lattice'] = '{0:0.3f}'.format(delta)
                meshfn = le.find_meshfn(lp)
                lp['meshfn'] = meshfn
                print '\n\n\nlp[meshfn] = ', lp['meshfn']
                lat = lattice_class.Lattice(lp)
                lat.load()
                hc = haldane_collection.HaldaneCollection()
                lp_submaster = copy.deepcopy(lp)
                for hlatpval in paramV:
                    lpii = copy.deepcopy(lp_submaster)
                    lpii[args.hlatparam] = hlatpval
                    hlat = haldane_lattice_class.HaldaneLattice(lat, lpii)
                    hlat.load()
                    hc.add_haldane_lattice(hlat)

                print 'Creating chern collection from single-lattice haldane_collection...'
                hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)
                hccoll.get_cherns(reverse=args.hlatparam_reverse)
                hccollcoll.add_haldane_chern_collection(hccoll, delta)
                print '\n\n\n\nhccollcoll.haldane_chern_collections = ', hccollcoll.haldane_chern_collections

            hccollcoll.plot_maxchern_lpparam_hlatparam()

    if args.chern_varyloc_varyhlatparam:
        """Make images and movie of spatially-resolved chern number for a sequence of haldanelattice parameters
        (for ex, ABDelta = 0.0, 0.1, ... 1.0). Plots the DOS for each hlat as well.

        Example usage:
        python haldane_chern_collection_collection.py -LT iscentroid -N 15 -Nks 110 -singleksz_frac 0.3 -chern_varyloc_varyhlatparam -hlatparam ABDelta -paramV 0.0:0.1:1.0
        python haldane_chern_collection_collection.py -LT iscentroid -N 20 -Nks 201 -singleksz_frac 0.3 -chern_varyloc_varyhlatparam -hlatparam ABDelta -paramV 0.0:0.1:2.0
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()

        print 'For each hlat param, creating chern collection from single-lattice haldane_collection...'
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        lp_master = copy.deepcopy(lp)
        cp_master = copy.deepcopy(cp)
        hccollcoll = HaldaneChernCollectionCollection()

        outdir = dio.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        dio.ensure_dir(outdir)
        paramspec = str(args.hlatparam) + "_lenlppV" + str(len(paramV))
        pklfn = outdir + "hccollcoll_chern_varyloc_varyhlatparam_" + paramspec + ".pkl"

        if args.maxchern:
            stillname = 'maxchern_varyloc'
        elif args.singleksz > 0:
            stillname = 'chern_ksz_' + '{0:0.2f}'.format(args.chern_singlesz).replace('.', 'p') + '_varyloc'
        elif args.singleksz_frac > 0:
            stillname = 'chern_kszfrac_' + '{0:0.2f}'.format(args.singleksz_frac).replace('.', 'p') + '_varyloc'
        else:
            stillname = 'chern_varyloc'
        imgdir = outdir + stillname + '_varyhlatparam_' + paramspec + '_stills/'
        dio.ensure_dir(imgdir)

        if glob.glob(pklfn):
            print 'hccollcoll: Loading pickle file for hccollcoll since already stored: pklfn = ', pklfn
            with open(pklfn, "rb") as fn:
                hccollcoll = pkl.load(fn)
                print 'hccollcoll = ', hccollcoll
            for param_key in hccollcoll.haldane_chern_collections:
                ind = np.argmin(np.abs(param_key - paramV))
                hccoll = hccollcoll.haldane_chern_collections[param_key]
                cherns_key = hccoll.cherns.keys()
                first_chern = hccoll.cherns[cherns_key[0]][0]
                hlat = first_chern.haldane_lattice
                hlatparam = hlat.lp[args.hlatparam]
                print 'Plotting chern collection for hlatparam: ', args.hlatparam, '=', hlatparam

                fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent()

                title = r'Spatially-resolved Chern number, with $\Delta$ = ' + '{0:0.2f}'.format(hlatparam)
                filename = stillname + '_{0:04d}'.format(ind)
                hccoll.plot_cherns_varyloc(title=title, filename=filename, rootdir='auto',
                                          outdir=dio.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False,
                                          ax=ax, cbar_ax=cbar_ax,
                                          singleksz_frac=args.singleksz_frac,
                                          singleksz=args.singleksz, maxchern=args.maxchern,
                                          save=False, make_cbar=True, colorz=False, dpi=500)

                # If AB sites, color each site white or black
                if args.hlatparam == 'ABDelta':
                    white = hlat.Omg > np.mean(hlat.Omg)
                    black = hlat.Omg < np.mean(hlat.Omg)
                    print 'white = ', white
                    print 'black = ', black
                    print 'hlat.Omg = ', hlat.Omg
                    print 'np.mean(hlat.Omg) = ', np.mean(hlat.Omg)
                    ax.scatter(hlat.lattice.xy[white, 0], hlat.lattice.xy[white, 1], s=2, facecolor='w', lw=0.2,
                               zorder=999999)
                    ax.scatter(hlat.lattice.xy[black, 0], hlat.lattice.xy[black, 1], s=2, facecolor='k', lw=0.2,
                               zorder=999999)
                    if len(np.where(white)[0]) == 0:
                        ax.scatter(hlat.lattice.xy[:, 0], hlat.lattice.xy[:, 1], s=2, facecolor='gray', lw=0.2,
                                   zorder=999999)

                print 'hccollpfns: saving figure:\n outdir =', dio.prepdir(imgdir), ' filename = ', filename
                # outdir = kfns.get_cmeshfn(ccoll.cherns[hlat_name][0].haldane_lattice.lp, rootdir=rootdir)
                print 'saving figure: ' + dio.prepdir(imgdir) + filename
                plt.savefig(dio.prepdir(imgdir) + filename + '.png', dpi=600)
                plt.close('all')
        else:
            ind = 0
            for lppval in paramV:
                lp = copy.deepcopy(lp_master)
                lp[args.hlatparam] = lppval
                hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
                hlat.load()
                hc = haldane_collection.HaldaneCollection()
                hc.add_haldane_lattice(hlat)

                print 'Creating chern collection for hlatparam: ', args.hlatparam, '= ', lppval
                cp = copy.deepcopy(cp_master)
                hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)
                hccoll.get_cherns_varyloc(step=1., fracsteps=False, reverse=False, verbose=False)
                hccollcoll.add_haldane_chern_collection(hccoll, lppval)
                print 'Plotting chern collection for hlatparam: ', args.hlatparam, '=', lppval
                title = r'Spatially-resolved Chern number, with $\Delta$ = ' + '{0:0.2f}'.format(lppval)
                hccoll.plot_cherns_varyloc(title=title, filename=stillname + '_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=dio.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz_frac=args.singleksz_frac,
                                          singleksz=args.singleksz, maxchern=args.maxchern, save=True,
                                          make_cbar=True, colorz=False, dpi=500)
                ind += 1

        import lepm.plotting.movies as lemov

        movname = outdir + stillname + '_varyhlatparam_' + str(args.lpparam) + '_lenlppV' + str(len(paramV))
        lemov.make_movie(imgdir + stillname + '_', movname, indexsz='04', framerate=2, imgdir=imgdir)
        with open(pklfn, "wb") as fn:
            pkl.dump(hccollcoll, fn)

    if args.plot_varyhlatparam_avgloc:
        """Print images of the averages of the chern 2D images (hlatparam vs ksize) over many spatially-varying voxels.
        This would be useful, for ex, for plotting ABDelta phase diagrams.
        Since we are varying hlatparam here, we are NOT changing the lattice used, just the HaldaneLattice.

        Example usage:
        python haldane_chern_collection_collection.py -LT hucentroid -N 20 -Nks 110 -plot_varyhlatparam_avgloc -hlatparam ABDelta -paramV 0.0:0.1:1.0 -locV n1.0:1.0
        python haldane_chern_collection_collection.py -LT hucentroid -N 30 -Nks 201 -plot_varyhlatparam_avgloc -hlatparam ABDelta -paramV 0.0:0.1:1.0 -locV n1.0:1.0
        """
        import lepm.plotting.haldane_chern_plotting_functions as kpfns

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()

        print 'For each param offset, creating chern collection from single-lattice haldane_collection...'
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        locV = sf.string_sequence_to_numpy_array(args.locV, dtype=float)
        cp_master = copy.deepcopy(cp)
        lp_master = copy.deepcopy(lp)
        hccollcoll = HaldaneChernCollectionCollection()

        outdir = dio.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        dio.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        dio.ensure_dir(imgdir)
        ind = 0

        fig, ax = kpfns.initialize_spectrum_with_dos_plot()
        spect_ax = ax[1]
        dos_ax = ax[0]
        cbar_ipr_ax = ax[2]
        cbar_nu_ax = ax[3]
        alpha = 1. / float(len(locV) ** 2)

        fname = outdir + "hccollcoll_spectrum_varyloc_varyomegac_nlocpts" + str(len(locV)) + ".pkl"
        if glob.glob(fname):
            print 'hccollcoll: Loading pickle file for hccollcoll since already stored: fn = ', fname
            with open(fname, "rb") as fn:
                hccollcoll = pkl.load(fn)
            for param_key in hccollcoll.haldane_chern_collections:
                hccoll = hccollcoll.haldane_chern_collections[param_key]
                print 'hccoll = ', hccoll
                hccoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                  cbar_labelpad=-35)
                ind += 1
        else:
            for lppval in paramV:
                print 'Creating chern collection (varyloc) for hlatparam: ', hlatparam, ' = ', lppval
                lp = copy.deepcopy(lp_master)

                hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
                hlat.load()
                hc = haldane_collection.HaldaneCollection()
                hc.add_haldane_lattice(hlat)
                hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)

                for xx in locV:
                    for yy in locV:
                        cp = copy.deepcopy(cp_master)
                        cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                        hccoll.add_chern(hlat, cp=cp)

                hccoll.plot_cherns_vary_param(param_type='hlat', sz_param_nu=None, omegac_hlat_func=None,
                                             omegac_hlatparam=None, reverse=False, param='percolation_density',
                                             ngrid=None, title='Chern number calculation',
                                             xlabel=r'Fraction of particles in sum', ylabel=None)

                # plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                #                                       cbar_labelpad=-35)  # , **kwargs)
                hccollcoll.add_haldane_chern_collection(hccoll, '{0:0.3f}'.format(xx) + '{0:0.3f}'.format(yy))
                plt.pause(1)

            with open(fname, "wb") as fn:
                pkl.dump(hccollcoll, fn)

        plt.savefig(outdir + 'spectrum_varyloc_varyomegac_nlocpts' + str(len(locV)) + '.png', dpi=600)

    if args.chern_varyloc_varyomegac:
        """Make images and movie of spatially-resolved chern number

        Example usage:
        python haldane_chern_collection_collection.py -LT penroserhombTricent -N 30 -shape circle -Nks 201 -chern_varyloc_varyomegac -paramV 1.0:.1:4.0
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        hc = haldane_collection.HaldaneCollection()
        hc.add_haldane_lattice(hlat)
        print 'For each cutoff freq, creating chern collection from single-lattice haldane_collection...'
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        cp_master = copy.deepcopy(cp)
        hccollcoll = HaldaneChernCollectionCollection()

        outdir = dio.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        dio.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        dio.ensure_dir(imgdir)
        ind = 0

        if glob.glob(outdir + "hccollcoll_chern_varyloc_varyomegac.pkl"):
            fn = outdir + "hccollcoll_chern_varyloc_varyomegac.pkl"
            print 'hccollcoll: Loading pickle file for hccollcoll since already stored: fn = ', fn
            with open(outdir + "hccollcoll_chern_varyloc_varyomegac.pkl", "rb") as fn:
                hccollcoll = pkl.load(fn)
                print 'hccollcoll = ', hccollcoll
            for hccoll in hccollcoll.haldane_chern_collections:
                omegac = hccoll.cp['omegac'][0]
                print 'Plotting chern collection for omegac = ', omegac
                title = r'Spatially-resolved Chern number, with $\omega_c$ = ' + '{0:0.2f}'.format(omegac)
                hccoll.plot_cherns_varyloc(title=title, filename='chern_varyloc_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=dio.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz=-1, save=True, make_cbar=True, colorz=False)
                ind += 1
        else:
            for omegac in paramV:
                print 'Creating chern collection for omegac = ', omegac
                cp = copy.deepcopy(cp_master)
                cp['omegac'] = np.array([omegac])
                hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)
                hccoll.get_cherns_varyloc(step=1., fracsteps=False, reverse=False, verbose=False)
                hccollcoll.add_haldane_chern_collection(hccoll, omegac)
                title = r'Spatially-resolved Chern number, with $\omega_c$ = ' + '{0:0.2f}'.format(omegac)
                hccoll.plot_cherns_varyloc(title=title, filename='chern_varyloc_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=dio.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz=-1, save=True, make_cbar=True, colorz=False)
                ind += 1

        import lepm.plotting.movies as lemov
        movname = outdir + 'chern_varyloc_varyomegac_stills'
        lemov.make_movie(imgdir + 'chern_varyloc_', movname, indexsz='04', framerate=3, imgdir=imgdir)
        with open(outdir + "hccollcoll_chern_varyloc_varyomegac.pkl", "wb") as fn:
            pkl.dump(hccollcoll, fn)

    if args.spectrum_varyloc_varyomegac:
        """Print images of the averages of the chern spectra over many spatially-varying voxels

        Example usage:
        python haldane_chern_collection_collection.py -LT penroserhombTricent -N 30 -shape circle -Nks 201 -spectrum_varyloc_varyomegac -paramV 1.0:0.1:4.0 -locV n1.0:1.0
        """
        import lepm.plotting.haldane_chern_plotting_functions as kpfns

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
        hlat.load()
        hc = haldane_collection.HaldaneCollection()
        hc.add_haldane_lattice(hlat)
        print 'For each param offset, creating chern collection from single-lattice haldane_collection...'
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        locV = sf.string_sequence_to_numpy_array(args.locV, dtype=float)
        cp_master = copy.deepcopy(cp)
        hccollcoll = HaldaneChernCollectionCollection()

        outdir = dio.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        dio.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        dio.ensure_dir(imgdir)
        ind = 0

        fig, ax = kpfns.initialize_spectrum_with_dos_plot()
        spect_ax = ax[1]
        dos_ax = ax[0]
        cbar_ipr_ax = ax[2]
        cbar_nu_ax = ax[3]
        alpha = 1. / float(len(locV) ** 2)

        fname = outdir + "hccollcoll_spectrum_varyloc_varyomegac_nlocpts" + str(len(locV)) + ".pkl"
        if glob.glob(fname):
            print 'hccollcoll: Loading pickle file for hccollcoll since already stored: fn = ', fname
            with open(fname, "rb") as fn:
                hccollcoll = pkl.load(fn)
            for param_key in hccollcoll.haldane_chern_collections:
                hccoll = hccollcoll.haldane_chern_collections[param_key]
                print 'hccoll = ', hccoll
                hccoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                   cbar_labelpad=-35)
                ind += 1
        else:
            for xx in locV:
                for yy in locV:
                    cp = copy.deepcopy(cp_master)
                    cp['omegac'] = paramV
                    hccoll = haldane_chern_collection.HaldaneChernCollection(hc, cp=cp)
                    for omegac in paramV:
                        print 'Creating chern collection (spectrum) for xyoff, yoff = (', xx, ',', yy, ')\n'
                        cp = copy.deepcopy(cp_master)
                        cp['omegac'] = np.array([omegac])
                        cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                        hccoll.add_chern(hlat, cp=cp)

                    hccoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                       cbar_labelpad=-35)  # , **kwargs)
                    hccollcoll.add_haldane_chern_collection(hccoll, '{0:0.3f}'.format(xx) + '{0:0.3f}'.format(yy))
                    plt.pause(1)

            with open(fname, "wb") as fn:
                pkl.dump(hccollcoll, fn)

        plt.savefig(outdir + 'spectrum_varyloc_varyomegac_nlocpts' + str(len(locV)) + '.png', dpi=600)


