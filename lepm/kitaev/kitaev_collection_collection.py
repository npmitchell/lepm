import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
from lepm import lattice_class
from lepm import gyro_lattice_class
from lepm import gyro_collection
from lepm.kitaev import kitaev_chern_class
from lepm.kitaev import kitaev_collection
import argparse
import matplotlib.pyplot as plt
import copy
import socket
import glob
import cPickle as pkl

'''
Collections of collections of Chern number measurements made via the kitaev realspace method.
Each collection of measurements is grouped by a parameter defining the lattice used, with the collection containing
different gyro_lattices defined on that lattice. These gyro_lattices will have a glatparam (a physics parameter on the
gyro lattice) that varies from glat to glat.
'''


class KitaevCollectionCollection:
    """Create a collection of collections of chern measurements for gyroscopic spring networks.
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
                           list of self.gyro_lattice.lattice.xy indices in reg1 of kitaev sum region
                       reg2_xy : int list
                           list of self.gyro_lattice.lattice.xy indices in reg2 of kitaev sum region
                       reg3_xy : int list
                           list of self.gyro_lattice.lattice.xy indices in reg3 of kitaev sum region
                       polygon1 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg1
                       polygon2 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg2
                       polygon3 : #vertices x 2 float array
                           coordinates of vertices of polygon enclosing reg3
                       reg1 : int list
                           list of self.gyro_lattice.lattice.xy indices in reg1 of kitaev sum region
                       reg2 : int list
                           list of self.gyro_lattice.lattice.xy indices in reg2 of kitaev sum region
                       reg3 : int list
                           list of self.gyro_lattice.lattice.xy indices in reg3 of kitaev sum region

    Attributes
    ----------
    self.param_keys : list of string, float, etc parameter
       list of keys for the self.kitaev_collections dict
    self.kitaev_collections : dict
       keys are glat_names (strings)
    """
    def __init__(self, kitaev_collections={}, param_keys=[]):  # lattices=[], lattice_names=[]):
        """Create an instance of a lattice_collection."""
        self.kitaev_collections = kitaev_collections
        self.param_keys = param_keys
        # self.lattice_names = lattice_names
        # self.lattices = lattices

    def add_kitaev_collection(self, kcoll, param_key):
        """Group a collection of measurements under the lattice used

        Parameters
        ----------
        kcoll : KitaevCollection instance
            The kitaev collection to add
        param_key : object of any type
            the identifier (key for a dict) under which the kitaev collection is stored
        """
        self.kitaev_collections[param_key] = kcoll
        self.param_keys.append(param_key)

    def collect_maxchern_lpparam_glatparam(self, glatparam='ABDelta'):
        """Collect the extremal chern values (averaged if there are more than one vertex locations) while varying a
        lattice parameter (lpparam) AND a GryoLattice parameter (glatparam)
        """
        maxcherns = []
        glatV = []
        lpV = []
        for key in self.kitaev_collections:
            print 'key = ', key
            for glat_name in self.kitaev_collections[key].glat_names:
                cherns = self.kitaev_collections[key].cherns[glat_name]
                # print 'cherns = ', cherns
                if len(cherns) > 1:
                    # Average their values -- this would be for a varyloc, for instance (a spatially
                    # distributed collection of cherns)
                    # NOTE: We assume that all the cherns here have the SAME glat parameter!
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
                glatV.append(cherns[0].gyro_lattice.lp[glatparam])

        print 'lpV = ', lpV
        print 'glatV = ', glatV
        print 'maxcherns = ', maxcherns

        return lpV, glatV, maxcherns

    # def collect_maxchern_varyloc_varyomegac(self, omegacV=np.arange(1.0, 4.0, 0.1)):
    #     """Collect chern instances for varying the vertex location AND varying cutoff freqeucny of projector.
    #     omegacV is a n x 1 array of cutoff freqs, polyoff2D is a m x 2 array of x, y positions, maxcherns is n x m array
    #     of extremal chern values.
    #
    #     Parameters
    #     ----------
    #     omegacV : 1d float array
    #         cutoff frequency values for the projector
    #     """
    #     maxcherns = []
    #     for key in self.kitaev_collections:
    #         print 'key = ', key
    #         glat_names = self.kitaev_collections[key].glat_names
    #         if len(glat_names) > 1:
    #             raise RuntimeError("kcollcoll.collect_maxchern_varyloc should have kitaev_collections based on " +
    #                                "only one single GyroLattice instance")
    #         glat_name = glat_names[0]
    #         cherns = self.kitaev_collections[key].cherns[glat_name]
    #         # print 'cherns = ', cherns
    #         if len(cherns) > 1:
    #             # Average their values -- this would be for a varyloc, for instance (a spatially
    #             # distributed collection of cherns)
    #             # NOTE: We assume that all the cherns here have the SAME glat parameter!
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
    #         glatV.append(cherns[0].gyro_lattice.lp[glatparam])
    #
    #     print 'lpV = ', lpV
    #     print 'glatV = ', glatV
    #     print 'maxcherns = ', maxcherns
    #
    #     return omegacV, polyoff2D, maxcherns

    def plot_maxchern_lpparam_glatparam(self, lgchern=None, glatparam='ABDelta'):
        """"""
        if lgchern is None:
            lpV, glatV, maxcherns = self.collect_maxchern_lpparam_glatparam(glatparam=glatparam)
            lgchern = np.dstack((lpV, glatV, maxcherns))[0]

        print 'lgchern = ', lgchern
        ngrid = len(lgchern) + 7
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent()
        leplt.plot_pcolormesh(lgchern[:, 0], lgchern[:, 1], lgchern[:, 2], ngrid, ax=ax, cax=cbar_ax, method='nearest',
                              make_cbar=True, cmap=lecmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1., vmax=1., title=None, xlabel=None, ylabel=None, ylabel_right=True,
                              ylabel_rot=90, cax_label=r'Chern number, $\nu$',
                              cbar_labelpad=-30, cbar_orientation='horizontal',
                              ticks=[-1, 0, 1], fontsize=8, title_axX=None, title_axY=None, alpha=1.0)
        plt.show()


if __name__ == '__main__':
    '''Perform an example of using the kitaev_collection_collection class.

    Example usage to create AB site phase diagram:
    python kitaev_collection_collection.py -ABphase
    '''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create collection of kitaev chern calculations.')
    # Script options
    parser.add_argument('-vary_omegac', '--calc_cherns_omegac',
                        help='Compute chern number using a range of omegac values for each glat in the collection' +
                             ' -- ie, create a chern spectrum',
                        action='store_true')
    parser.add_argument('-varyloc', '--calc_cherns_varyloc',
                        help='Compute chern number using a grid of kitaev region locations for each gyro lattice ' +
                             'in the collection', action='store_true')
    parser.add_argument('-varyloc_reverse', '--varyloc_reverse',
                        help='When computing chern number using a grid of locations, reverse the order of computing' +
                             ' on the grid', action='store_true')
    parser.add_argument('-ABphase', '--ABphase',
                        help='Do multiple chern number calcs for many GyroLattices build on THE SAME Lattice ' +
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
    parser.add_argument('-chern_varyloc_varyglatparam', '--chern_varyloc_varyglatparam',
                        help='Plot the extremal chern number as a function of space for different values of a ' +
                             'GyroLattice parameter dict (glat.lp)',
                        action='store_true')
    parser.add_argument('-singleksz_frac', '--singleksz_frac',
                        help='Fractional size (in units of sys size) of the kitaev summation region to use for ' +\
                             'plotting spatially resolved chern number. Ignored if singleksz > 0 or maxchern = True.',
                        type=float, default=-1.)
    parser.add_argument('-singleksz', '--singleksz',
                        help='Size of the kitaev summation region to use for plotting spatially resolved chern ' + \
                             'number. Ignored if maxchern == True.',
                        type=float, default=-1.)
    parser.add_argument('-maxchern', '--maxchern',
                        help='Use extremal value of chern number when plotting spatially resolved chern number',
                        action='store_true')
    parser.add_argument('-spectrum_varyloc_varyglatparam', '--spectrum_varyloc_varyglatparam',
                        help='Plot the chern spectra averaged over different vertex positions',
                        action='store_true')
    # parser.add_argument('-vary_lpparam_vary_glatparam', '--vary_lpparam_vary_glatparam',
    #                     help='Do multiple chern number calcs for many GyroLattices build on THE SAME Lattice ' +
    #                          'instance with a parameter varying between them, then also vary which lattice to use, ' +
    #                          'by varying glatparam',
    #                     action='store_true')

    # Options for script method choice
    parser.add_argument('-Nks', '--Nks',
                        help='How many kitaev region sizes to sample at each site if -calc_cherns_varyloc',
                        type=int, default=0)
    parser.add_argument('-step', '--step', help='Step size to take while sampling the regions of the network via chern',
                        type=float, default=1.0)
    parser.add_argument('-glatparam', '--glatparam',
                        help='String specifier for which parameter is varied across a single gyro lattice in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-glatparam_reverse', '--glatparam_reverse',
                        help='When computing chern number varying a gyrolattice param, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
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
    parser.add_argument('-proj_site_glatparam', '--proj_site_glatparam',
                        help='Compare the projector elements near a point proj_XY for a collection of glats',
                        action='store_true')
    parser.add_argument('-proj_XY', '--proj_XY',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to the point closest to this location',
                        type=str, default='0.0/0.0')

    # chern parameters
    parser.add_argument('-ksize_frac_array', '--ksize_frac_array',
                        help='Array of fractional sizes to make the kitaev region, specified with /s', type=str,
                        default='0.0:0.01:1.10')
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector, specified as string with /s',
                        type=str, default='2.25')
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
    le.ensure_dir(outdir)

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
          'Omk': -1.0,
          'Omg': -1.0,
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'viewmethod': False,
          'ABDelta': args.ABDelta,
          }

    ksize_frac_arr = le.string_sequence_to_numpy_array(args.ksize_frac_array, dtype=float)

    print 'ksize_frac_arr = ', ksize_frac_arr
    cp = {'ksize_frac_arr': ksize_frac_arr,
          'omegac': le.string_sequence_to_numpy_array(args.omegac, dtype=float),
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

    print 'Creating chern collection from gyro_collection...'
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
        # Collate cherns for one lattice with a gyro_lattice parameter that varies between instances of that lattice
        lp_master = copy.deepcopy(lp)
        paramV = le.string_sequence_to_numpy_array(args.paramV, dtype=float)
        if args.lpparam == 'delta':
            # vary delta between lattices
            deltaV = np.arange(0.7, 1.31, 0.1)
            deltaV = np.hstack((0.667, deltaV))
            kcollcoll = KitaevCollectionCollection()
            for delta in deltaV:
                lp = copy.deepcopy(lp_master)
                lp['delta_lattice'] = '{0:0.3f}'.format(delta)
                meshfn = le.find_meshfn(lp)
                lp['meshfn'] = meshfn
                print '\n\n\nlp[meshfn] = ', lp['meshfn']
                lat = lattice_class.Lattice(lp)
                lat.load()
                gc = gyro_collection.GyroCollection()
                lp_submaster = copy.deepcopy(lp)
                for glatpval in paramV:
                    lpii = copy.deepcopy(lp_submaster)
                    lpii[args.glatparam] = glatpval
                    glat = gyro_lattice_class.GyroLattice(lat, lpii)
                    glat.load()
                    gc.add_gyro_lattice(glat)

                print 'Creating chern collection from single-lattice gyro_collection...'
                kcoll = kitaev_collection.KitaevCollection(gc, cp=cp)
                kcoll.get_cherns(reverse=args.glatparam_reverse)
                kcollcoll.add_kitaev_collection(kcoll, delta)
                print '\n\n\n\nkcollcoll.kitaev_collections = ', kcollcoll.kitaev_collections

            kcollcoll.plot_maxchern_lpparam_glatparam()

    if args.chern_varyloc_varyglatparam:
        """Make images and movie of spatially-resolved chern number for a sequence of gyrolattice parameters
        (for ex, ABDelta = 0.0, 0.1, ... 1.0). Plots the DOS for each glat as well.

        Example usage:
        python kitaev_collection_collection.py -LT iscentroid -N 15 -Nks 110 -singleksz_frac 0.3 -chern_varyloc_varyglatparam -glatparam ABDelta -paramV 0.0:0.1:1.0
        python kitaev_collection_collection.py -LT iscentroid -N 20 -Nks 201 -singleksz_frac 0.3 -chern_varyloc_varyglatparam -glatparam ABDelta -paramV 0.0:0.1:2.0
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()

        print 'For each glat param, creating chern collection from single-lattice gyro_collection...'
        paramV = le.string_sequence_to_numpy_array(args.paramV, dtype=float)
        lp_master = copy.deepcopy(lp)
        cp_master = copy.deepcopy(cp)
        kcollcoll = KitaevCollectionCollection()

        outdir = le.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        le.ensure_dir(outdir)
        paramspec = str(args.glatparam) + "_lenlppV" + str(len(paramV))
        pklfn = outdir + "kcollcoll_chern_varyloc_varyglatparam_" + paramspec + ".pkl"

        if args.maxchern:
            stillname = 'maxchern_varyloc'
        elif args.singleksz > 0:
            stillname = 'chern_ksz_' + '{0:0.2f}'.format(args.chern_singlesz).replace('.', 'p') + '_varyloc'
        elif args.singleksz_frac > 0:
            stillname = 'chern_kszfrac_' + '{0:0.2f}'.format(args.singleksz_frac).replace('.', 'p') + '_varyloc'
        else:
            stillname = 'chern_varyloc'
        imgdir = outdir + stillname + '_varyglatparam_' + paramspec + '_stills/'
        le.ensure_dir(imgdir)

        if glob.glob(pklfn):
            print 'kcollcoll: Loading pickle file for kcollcoll since already stored: pklfn = ', pklfn
            with open(pklfn, "rb") as fn:
                kcollcoll = pkl.load(fn)
                print 'kcollcoll = ', kcollcoll
            for param_key in kcollcoll.kitaev_collections:
                ind = np.argmin(np.abs(param_key - paramV))
                kcoll = kcollcoll.kitaev_collections[param_key]
                cherns_key = kcoll.cherns.keys()
                first_chern = kcoll.cherns[cherns_key[0]][0]
                glat = first_chern.gyro_lattice
                glatparam = glat.lp[args.glatparam]
                print 'Plotting chern collection for glatparam: ', args.glatparam, '=', glatparam

                fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent()

                title = r'Spatially-resolved Chern number, with $\Delta$ = ' + '{0:0.2f}'.format(glatparam)
                filename = stillname + '_{0:04d}'.format(ind)
                kcoll.plot_cherns_varyloc(title=title, filename=filename, rootdir='auto',
                                          outdir=le.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False,
                                          ax=ax, cbar_ax=cbar_ax,
                                          singleksz_frac=args.singleksz_frac,
                                          singleksz=args.singleksz, maxchern=args.maxchern,
                                          save=False, make_cbar=True, colorz=False, dpi=500)

                # If AB sites, color each site white or black
                if args.glatparam == 'ABDelta':
                    white = glat.Omg > np.mean(glat.Omg)
                    black = glat.Omg < np.mean(glat.Omg)
                    print 'white = ', white
                    print 'black = ', black
                    print 'glat.Omg = ', glat.Omg
                    print 'np.mean(glat.Omg) = ', np.mean(glat.Omg)
                    ax.scatter(glat.lattice.xy[white, 0], glat.lattice.xy[white, 1], s=2, facecolor='w', lw=0.2,
                               zorder=999999)
                    ax.scatter(glat.lattice.xy[black, 0], glat.lattice.xy[black, 1], s=2, facecolor='k', lw=0.2,
                               zorder=999999)
                    if len(np.where(white)[0]) == 0:
                        ax.scatter(glat.lattice.xy[:, 0], glat.lattice.xy[:, 1], s=2, facecolor='gray', lw=0.2,
                                   zorder=999999)

                print 'kcollpfns: saving figure:\n outdir =', le.prepdir(imgdir), ' filename = ', filename
                # outdir = kfns.get_cmeshfn(ccoll.cherns[glat_name][0].gyro_lattice.lp, rootdir=rootdir)
                print 'saving figure: ' + le.prepdir(imgdir) + filename
                plt.savefig(le.prepdir(imgdir) + filename + '.png', dpi=600)
                plt.close('all')
        else:
            ind = 0
            for lppval in paramV:
                lp = copy.deepcopy(lp_master)
                lp[args.glatparam] = lppval
                glat = gyro_lattice_class.GyroLattice(lat, lp)
                glat.load()
                gc = gyro_collection.GyroCollection()
                gc.add_gyro_lattice(glat)

                print 'Creating chern collection for glatparam: ', args.glatparam, '= ', lppval
                cp = copy.deepcopy(cp_master)
                kcoll = kitaev_collection.KitaevCollection(gc, cp=cp)
                kcoll.get_cherns_varyloc(step=1., fracsteps=False, reverse=False, verbose=False)
                kcollcoll.add_kitaev_collection(kcoll, lppval)
                print 'Plotting chern collection for glatparam: ', args.glatparam, '=', lppval
                title = r'Spatially-resolved Chern number, with $\Delta$ = ' + '{0:0.2f}'.format(lppval)
                kcoll.plot_cherns_varyloc(title=title, filename=stillname + '_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=le.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz_frac=args.singleksz_frac,
                                          singleksz=args.singleksz, maxchern=args.maxchern, save=True,
                                          make_cbar=True, colorz=False, dpi=500)
                ind += 1

        import lepm.plotting.movies as lemov

        movname = outdir + stillname + '_varyglatparam_' + str(args.lpparam) + '_lenlppV' + str(len(paramV))
        lemov.make_movie(imgdir + stillname + '_', movname, indexsz='04', framerate=2, imgdir=imgdir)
        with open(pklfn, "wb") as fn:
            pkl.dump(kcollcoll, fn)

    if args.plot_varyglatparam_avgloc:
        """Print images of the averages of the chern 2D images (glatparam vs ksize) over many spatially-varying voxels.
        This would be useful, for ex, for plotting ABDelta phase diagrams.
        Since we are varying glatparam here, we are NOT changing the lattice used, just the GyroLattice.

        Example usage:
        python kitaev_collection_collection.py -LT hucentroid -N 20 -Nks 110 -plot_varyglatparam_avgloc -glatparam ABDelta -paramV 0.0:0.1:1.0 -locV n1.0:1.0
        python kitaev_collection_collection.py -LT hucentroid -N 30 -Nks 201 -plot_varyglatparam_avgloc -glatparam ABDelta -paramV 0.0:0.1:1.0 -locV n1.0:1.0
        """
        import lepm.plotting.kitaev_plotting_functions as kpfns

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()

        print 'For each param offset, creating chern collection from single-lattice gyro_collection...'
        paramV = le.string_sequence_to_numpy_array(args.paramV, dtype=float)
        locV = le.string_sequence_to_numpy_array(args.locV, dtype=float)
        cp_master = copy.deepcopy(cp)
        lp_master = copy.deepcopy(lp)
        kcollcoll = KitaevCollectionCollection()

        outdir = le.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        le.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        le.ensure_dir(imgdir)
        ind = 0

        fig, ax = kpfns.initialize_spectrum_with_dos_plot()
        spect_ax = ax[1]
        dos_ax = ax[0]
        cbar_ipr_ax = ax[2]
        cbar_nu_ax = ax[3]
        alpha = 1. / float(len(locV) ** 2)

        fname = outdir + "kcollcoll_spectrum_varyloc_varyomegac_nlocpts" + str(len(locV)) + ".pkl"
        if glob.glob(fname):
            print 'kcollcoll: Loading pickle file for kcollcoll since already stored: fn = ', fname
            with open(fname, "rb") as fn:
                kcollcoll = pkl.load(fn)
            for param_key in kcollcoll.kitaev_collections:
                kcoll = kcollcoll.kitaev_collections[param_key]
                print 'kcoll = ', kcoll
                kcoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                  cbar_labelpad=-35)
                ind += 1
        else:
            for lppval in paramV:
                print 'Creating chern collection (varyloc) for glatparam: ', glatparam, ' = ', lppval
                lp = copy.deepcopy(lp_master)

                glat = gyro_lattice_class.GyroLattice(lat, lp)
                glat.load()
                gc = gyro_collection.GyroCollection()
                gc.add_gyro_lattice(glat)
                kcoll = kitaev_collection.KitaevCollection(gc, cp=cp)

                for xx in locV:
                    for yy in locV:
                        cp = copy.deepcopy(cp_master)
                        cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                        kcoll.add_chern(glat, cp=cp)

                kcoll.plot_cherns_vary_param(param_type='glat', sz_param_nu=None, omegac_glat_func=None,
                                             omegac_glatparam=None, reverse=False, param='percolation_density',
                                             ngrid=None, title='Chern number calculation',
                                             xlabel=r'Fraction of particles in sum', ylabel=None)

                # plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                #                                       cbar_labelpad=-35)  # , **kwargs)
                kcollcoll.add_kitaev_collection(kcoll, '{0:0.3f}'.format(xx) + '{0:0.3f}'.format(yy))
                plt.pause(1)

            with open(fname, "wb") as fn:
                pkl.dump(kcollcoll, fn)

        plt.savefig(outdir + 'spectrum_varyloc_varyomegac_nlocpts' + str(len(locV)) + '.png', dpi=600)

    if args.chern_varyloc_varyomegac:
        """Make images and movie of spatially-resolved chern number

        Example usage:
        python kitaev_collection_collection.py -LT penroserhombTricent -N 30 -shape circle -Nks 201 -chern_varyloc_varyomegac -paramV 1.0:.1:4.0
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()
        gc.add_gyro_lattice(glat)
        print 'For each cutoff freq, creating chern collection from single-lattice gyro_collection...'
        paramV = le.string_sequence_to_numpy_array(args.paramV, dtype=float)
        cp_master = copy.deepcopy(cp)
        kcollcoll = KitaevCollectionCollection()

        outdir = le.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        le.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        le.ensure_dir(imgdir)
        ind = 0

        if glob.glob(outdir + "kcollcoll_chern_varyloc_varyomegac.pkl"):
            fn = outdir + "kcollcoll_chern_varyloc_varyomegac.pkl"
            print 'kcollcoll: Loading pickle file for kcollcoll since already stored: fn = ', fn
            with open(outdir + "kcollcoll_chern_varyloc_varyomegac.pkl", "rb") as fn:
                kcollcoll = pkl.load(fn)
                print 'kcollcoll = ', kcollcoll
            for kcoll in kcollcoll.kitaev_collections:
                omegac = kcoll.cp['omegac'][0]
                print 'Plotting chern collection for omegac = ', omegac
                title = r'Spatially-resolved Chern number, with $\omega_c$ = ' + '{0:0.2f}'.format(omegac)
                kcoll.plot_cherns_varyloc(title=title, filename='chern_varyloc_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=le.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz=-1, save=True, make_cbar=True, colorz=False)
                ind += 1
        else:
            for omegac in paramV:
                print 'Creating chern collection for omegac = ', omegac
                cp = copy.deepcopy(cp_master)
                cp['omegac'] = np.array([omegac])
                kcoll = kitaev_collection.KitaevCollection(gc, cp=cp)
                kcoll.get_cherns_varyloc(step=1., fracsteps=False, reverse=False, verbose=False)
                kcollcoll.add_kitaev_collection(kcoll, omegac)
                title = r'Spatially-resolved Chern number, with $\omega_c$ = ' + '{0:0.2f}'.format(omegac)
                kcoll.plot_cherns_varyloc(title=title, filename='chern_varyloc_{0:04d}'.format(ind), rootdir='auto',
                                          outdir=le.prepdir(imgdir), exten='.png',
                                          max_boxfrac=None, max_boxsize=None,
                                          xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                                          singleksz=-1, save=True, make_cbar=True, colorz=False)
                ind += 1

        import lepm.plotting.movies as lemov
        movname = outdir + 'chern_varyloc_varyomegac_stills'
        lemov.make_movie(imgdir + 'chern_varyloc_', movname, indexsz='04', framerate=3, imgdir=imgdir)
        with open(outdir + "kcollcoll_chern_varyloc_varyomegac.pkl", "wb") as fn:
            pkl.dump(kcollcoll, fn)

    if args.spectrum_varyloc_varyomegac:
        """Print images of the averages of the chern spectra over many spatially-varying voxels

        Example usage:
        python kitaev_collection_collection.py -LT penroserhombTricent -N 30 -shape circle -Nks 201 -spectrum_varyloc_varyomegac -paramV 1.0:0.1:4.0 -locV n1.0:1.0
        """
        import lepm.plotting.kitaev_plotting_functions as kpfns

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()
        gc.add_gyro_lattice(glat)
        print 'For each param offset, creating chern collection from single-lattice gyro_collection...'
        paramV = le.string_sequence_to_numpy_array(args.paramV, dtype=float)
        locV = le.string_sequence_to_numpy_array(args.locV, dtype=float)
        cp_master = copy.deepcopy(cp)
        kcollcoll = KitaevCollectionCollection()

        outdir = le.prepdir(lp['meshfn']).replace('networks', 'cherncollcolls')
        le.ensure_dir(outdir)
        imgdir = outdir + 'chern_varyloc_varyomegac_stills/'
        le.ensure_dir(imgdir)
        ind = 0

        fig, ax = kpfns.initialize_spectrum_with_dos_plot()
        spect_ax = ax[1]
        dos_ax = ax[0]
        cbar_ipr_ax = ax[2]
        cbar_nu_ax = ax[3]
        alpha = 1. / float(len(locV) ** 2)

        fname = outdir + "kcollcoll_spectrum_varyloc_varyomegac_nlocpts" + str(len(locV)) + ".pkl"
        if glob.glob(fname):
            print 'kcollcoll: Loading pickle file for kcollcoll since already stored: fn = ', fname
            with open(fname, "rb") as fn:
                kcollcoll = pkl.load(fn)
            for param_key in kcollcoll.kitaev_collections:
                kcoll = kcollcoll.kitaev_collections[param_key]
                print 'kcoll = ', kcoll
                kcoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                  cbar_labelpad=-35)
                ind += 1
        else:
            for xx in locV:
                for yy in locV:
                    cp = copy.deepcopy(cp_master)
                    cp['omegac'] = paramV
                    kcoll = kitaev_collection.KitaevCollection(gc, cp=cp)
                    for omegac in paramV:
                        print 'Creating chern collection (spectrum) for xyoff, yoff = (', xx, ',', yy, ')\n'
                        cp = copy.deepcopy(cp_master)
                        cp['omegac'] = np.array([omegac])
                        cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                        kcoll.add_chern(glat, cp=cp)

                    kcoll.plot_chern_spectrum_on_axis(spect_ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, alpha=alpha,
                                                      cbar_labelpad=-35)  # , **kwargs)
                    kcollcoll.add_kitaev_collection(kcoll, '{0:0.3f}'.format(xx) + '{0:0.3f}'.format(yy))
                    plt.pause(1)

            with open(fname, "wb") as fn:
                pkl.dump(kcollcoll, fn)

        plt.savefig(outdir + 'spectrum_varyloc_varyomegac_nlocpts' + str(len(locV)) + '.png', dpi=600)


