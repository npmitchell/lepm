import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
from lepm import lattice_class
from lepm import gyro_lattice_class
from lepm import gyro_collection
from lepm.bott import bott_gyro_collection
import lepm.stringformat as sf
import lepm.dataio as dio
import argparse
import matplotlib.pyplot as plt
import copy
import socket
import glob
import cPickle as pkl

'''
Collections of collections of Bott index measurements made via the Bott realspace method.
Each collection of measurements is grouped by a parameter defining the lattice used, with the collection containing
different gyro_lattices defined on that lattice. These gyro_lattices will have a glatparam (a physics parameter on the
gyro lattice) that varies from glat to glat.
'''


class BottGyroCollectionCollection:
    """Create a collection of collections of bott measurements for gyroscopic spring networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.botts is a list of tuples of dicts: self.botts = [bott1, bott2,...]
    where bott is a class with attributes cp, bott, params_regs dict.
       cp : dict
           keys : str
               'meshfn', 'omegac', 'poly_offset', 'ksize_frac_arr', 'regalph', 'regbeta', 'reggamma', 'polyT',
               'poly_offset'
       bott : float
           contains nu for Bott calculation
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
       list of keys for the self.bott_gyro_collections dict
    self.bott_gyro_collections : dict
       keys are glat_names (strings)
    """
    def __init__(self, bott_gyro_collections={}, param_keys=[]):  # lattices=[], lattice_names=[]):
        """Create an instance of a lattice_collection."""
        self.bott_gyro_collections = bott_gyro_collections
        self.param_keys = param_keys
        # self.lattice_names = lattice_names
        # self.lattices = lattices

    def add_bott_gyro_collection(self, kcoll, param_key):
        """Group a collection of measurements under the lattice used

        Parameters
        ----------
        kcoll : KitaevCollection instance
            The kitaev collection to add
        param_key : object of any type
            the identifier (key for a dict) under which the kitaev collection is stored
        """
        self.bott_gyro_collections[param_key] = kcoll
        self.param_keys.append(param_key)

    def collect_botts_lpparam_glatparam(self, glatparam='ABDelta'):
        """Collect the extremal bott values (averaged if there are more than one vertex locations) while varying a
        lattice parameter (lpparam) AND a GryoLattice parameter (glatparam)
        """
        botts = []
        glatV = []
        lpV = []
        for key in self.bott_gyro_collections:
            print 'key = ', key
            for glat_name in self.bott_gyro_collections[key].glat_names:
                botts_tmp = self.bott_gyro_collections[key].botts[glat_name]
                # print 'botts = ', botts
                if len(botts_tmp) > 1:
                    # Average their values -- this would be for a varyloc, for instance (a spatially
                    # distributed collection of botts)
                    # NOTE: We assume that all the botts here have the SAME glat parameter!
                    print 'botts_tmp = ', botts_tmp
                    bottmat = np.zeros((len(botts_tmp)))
                    for dmyi in range(len(botts_tmp)):
                        bottmat[dmyi] = botts_tmp[dmyi].bott

                    # Average along the rows of bottmat (bott matrix)
                    print 'bott_gyrocollcoll: bottmat = ', bottmat
                    bottval = np.mean(bottmat)
                else:
                    bottval = botts_tmp[0].bott

                botts.append(bottval)
                lpV.append(key)
                glatV.append(botts_tmp[0].gyro_lattice.lp[glatparam])

        print 'lpV = ', np.shape(lpV)
        print 'glatV = ', np.shape(glatV)
        print 'botts = ', np.shape(botts)

        return lpV, glatV, botts

    # def collect_bott_varyloc_varyomegac(self, omegacV=np.arange(1.0, 4.0, 0.1)):
    #     """Collect bott instances for varying the vertex location AND varying cutoff freqeucny of projector.
    #     omegacV is a n x 1 array of cutoff freqs, polyoff2D is a m x 2 array of x, y positions, botts is n x m array
    #     of extremal bott values.
    #
    #     Parameters
    #     ----------
    #     omegacV : 1d float array
    #         cutoff frequency values for the projector
    #     """
    #     botts = []
    #     for key in self.bott_gyro_collections:
    #         print 'key = ', key
    #         glat_names = self.bott_gyro_collections[key].glat_names
    #         if len(glat_names) > 1:
    #             raise RuntimeError("kcollcoll.collect_bott_varyloc should have bott_gyro_collections based on " +
    #                                "only one single GyroLattice instance")
    #         glat_name = glat_names[0]
    #         botts = self.bott_gyro_collections[key].botts[glat_name]
    #         # print 'botts = ', botts
    #         if len(botts) > 1:
    #             # Average their values -- this would be for a varyloc, for instance (a spatially
    #             # distributed collection of botts)
    #             # NOTE: We assume that all the botts here have the SAME glat parameter!
    #             bottmat = np.zeros((len(botts[0].bott), len(botts)))
    #             for dmyi in range(len(botts)):
    #                 bottmat[:, dmyi] = botts[dmyi].bott
    #
    #             # Average along the rows of bottmat (bott matrix)
    #             bottval = np.mean(bottmat, axis=1)
    #         else:
    #             bottval = botts[0].bott
    #
    #         ind = np.argmax(np.abs(bottv))
    #         botts.append(bottval)
    #         lpV.append(key)
    #         glatV.append(botts[0].gyro_lattice.lp[glatparam])
    #
    #     print 'lpV = ', lpV
    #     print 'glatV = ', glatV
    #     print 'botts = ', botts
    #
    #     return omegacV, polyoff2D, botts

    def plot_bott_lpparam_glatparam(self, lgbott=None, glatparam='ABDelta', outname=None):
        """Plot phase diagram with lattice param varying on x axis and glatparam varying on y axis

        Parameters
        ----------
        lgbott : N x 3 float array or None
            lattice params array, glatparam array, bott indices array
        glatparam : str

        outname : str or None
        """
        if lgbott is None:
            lpV, glatV, botts = self.collect_botts_lpparam_glatparam(glatparam=glatparam)
            # lgbott is the [lattice param, glat param, botts] vector
            lgbott = np.dstack((lpV, glatV, botts))[0]

        print 'lgbott = ', lgbott
        ngrid = len(lgbott) + 7
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent()
        leplt.plot_pcolormesh(lgbott[:, 0], lgbott[:, 1], lgbott[:, 2], ngrid, ax=ax, cax=cbar_ax, method='nearest',
                              make_cbar=True, cmap=lecmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1., vmax=1., title=None, xlabel=None, ylabel=None, ylabel_right=True,
                              ylabel_rot=90, cax_label=r'Bott index, $B$',
                              cbar_labelpad=-30, cbar_orientation='horizontal',
                              ticks=[-1, 0, 1], fontsize=8, title_axX=None, title_axY=None, alpha=1.0)
        if outname is not None:
            plt.savefig(outname)

        return fig, ax


if __name__ == '__main__':
    '''Perform an example of using the bott_gyro_collection_collection class.

    Example usage to create AB site phase diagram:
    python bott_gyro_collection_collection.py -ABphase
    '''
    import os

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create collection of bott calculations.')
    # Script options
    parser.add_argument('-vary_omegac', '--calc_botts_omegac',
                        help='Compute bott index using a range of omegac values for each glat in the collection' +
                             ' -- ie, create a bott spectrum',
                        action='store_true')
    parser.add_argument('-varyloc', '--calc_botts_varyloc',
                        help='Compute bott index using a grid of kitaev region locations for each gyro lattice ' +
                             'in the collection', action='store_true')
    parser.add_argument('-varyloc_reverse', '--varyloc_reverse',
                        help='When computing bott index using a grid of locations, reverse the order of computing' +
                             ' on the grid', action='store_true')
    parser.add_argument('-ABphase', '--ABphase',
                        help='Do multiple bott index calcs for many GyroLattices build on THE SAME Lattice ' +
                             'instance with an lp_param varying between them, then also vary which lattice to use, ' +
                             'by varying ABDelta',
                        action='store_true')
    # below is an argument which could be used as a more general version of -ABphase argument.
    # parser.add_argument('-vary_lpparam_vary_glatparam', '--vary_lpparam_vary_glatparam',
    #                     help='Do multiple bott index calcs for many GyroLattices build on THE SAME Lattice ' +
    #                          'instance with a parameter varying between them, then also vary which lattice to use, ' +
    #                          'by varying glatparam',
    #                     action='store_true')

    # Options for script method choice
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
    parser.add_argument('-locV', '--locV',
                        help='Sequence of xyoffset values to assign to cp[poly_offset] if method involves varyloc',
                        type=str, default='n6.0:1.0:6.0')
    parser.add_argument('-bott_collection', '--bott_collection',
                        help='Whether to collect botts by LT and analyze their botts',
                        action='store_true')
    parser.add_argument('-proj_site_glatparam', '--proj_site_glatparam',
                        help='Compare the projector elements near a point proj_XY for a collection of glats',
                        action='store_true')
    parser.add_argument('-proj_XY', '--proj_XY',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to the point closest to this location',
                        type=str, default='0.0/0.0')

    # bott parameters
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
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=5)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=5)
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
    parser.add_argument('-nonperiodic', '--openBC', help='Enforce open (non periodic) boundary conditions',
                        action='store_true')
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

    hostname = socket.gethostname()
    print 'hostname = ', hostname
    if hostname[0:6] == 'midway':
        print '\n\nWe are on Midway!\n\n\n\n'
        rootdir = '/home/npmitchell/scratch-midway/'
        cprootdir = '/home/npmitchell/scratch-midway/'
    elif hostname[0:10] == 'nsit-dhcp-' or hostname[0:10] == 'npmitchell':
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

    outdir = rootdir + 'experiments/DOS_scaling/' + args.LatticeTop + '/'
    dio.ensure_dir(outdir)
    print 'cprootdir = ', cprootdir

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
          'periodicBC': not args.openBC,
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

    cp = {'omegac': sf.string_sequence_to_numpy_array(args.omegac, dtype=float),
          'basis': args.basis,
          'rootdir': cprootdir
          }

    if args.ABphase:
        """Vary an lp and AB parameter to make the Lisa lattice deformation phase diagram

        python ./bott/bott_gyro_collection_collection.py -ABphase -LT hexagonal -shape square -paramV 0.:0.05:0.45
        """
        # Collate botts for one lattice with a gyro_lattice parameter that varies between instances of that lattice
        lp_master = copy.deepcopy(lp)
        paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=float)
        if args.lpparam == 'delta':
            # vary delta between lattices
            deltaV = np.arange(0.7, 1.31, 0.1)
            deltaV = np.hstack((0.667, deltaV))
            bgcollcoll = BottGyroCollectionCollection()
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

                print 'Creating bott collection from single-lattice gyro_collection...'
                bgcoll = bott_gyro_collection.BottGyroCollection(gc, cp=cp)
                bgcoll.get_botts(reverse=args.glatparam_reverse)
                bgcollcoll.add_bott_gyro_collection(bgcoll, delta)
                print '\n\n\n\nbgcollcoll.bott_gyro_collections = ', bgcollcoll.bott_gyro_collections

            outname = cprootdir + 'bott_gyro/' + lp['LatticeTop'] + '_NH' + str(lp['NH']) + 'abphase.png'
            bgcollcoll.plot_bott_lpparam_glatparam(outname=outname)
