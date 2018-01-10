import numpy as np
import lepm.chern.kchern_gyro_functions as kcgfns
import lepm.chern.plotting.kchern_gyro_plotting_fns as kspacecpfns
import matplotlib.pyplot as plt
import cPickle as pkl
import time
import os.path
import sys

'''Class for k-space Chern calculations for gyro_lattices'''


class KChernGyro:
    """kspace Chern calculation class with attributes
    cp : dict
        chern calc parameters
    gyro_lattice : GyroLattice class instance
        the gyro lattice for which to compute the chern number
    chern : dict
        {'kx': kkx --> the x component of the wavenumbers where Berry curvature is evaluated
         'ky': kky --> the y component of the wavenumbers where Berry curvature is evaluated
         'chern': bv --> the chern numbers of each band whan all kx and ky pts taken into consideration
         'bands': bands --> the eigenvalues of the bands at each kx and ky pt
         'traces': traces -->
         'bzvtcs': bzvtcs --> the corners of the Brillouin zone for the reciprocal lattice
         }

    """
    def __init__(self, gyro_lattice, cp=None, cpmeshfn=None):
        self.gyro_lattice = gyro_lattice
        self.chern = None

        if cp is None:
            cp = {}
        if 'basis' not in cp:
            cp['basis'] = 'psi'
        if 'cpmeshfn' not in cp:
            if cpmeshfn is None:
                cpmeshfn = kcgfns.get_cpmeshfn(cp, self.gyro_lattice.lp)
            cp['cpmeshfn'] = cpmeshfn

        self.cp = cp

    def load_chern(self, attribute=True):
        """"""
        chern = kcgfns.load_chern(self)
        if attribute:
            self.chern = chern
        return chern

    def get_chern(self, attribute=True, verbose=True, save_png=True):
        """Load or calculate the chern for the current instance's parameters"""
        chern = kcgfns.load_chern(self)
        if chern is None:
            chern = kcgfns.calc_chern(self, verbose=verbose)
            kcgfns.save_chern(self, verbose=verbose, save_png=save_png)
        if attribute:
            self.chern = chern

        # print 'kchern_gyro : computed and saved chern or loaded it, exiting here for db'
        # sys.exit()
        return chern

    def ensure_chern(self, attribute=True):
        """Ensure that the chern calculation exists on disk"""
        chern = kcgfns.load_chern(self)
        if chern is None:
            chern = kcgfns.calc_chern(self)

        if attribute:
            self.chern = chern
        return chern

    def save_chern(self):
        """Save the chern calculation to disk. If does self has no chern, loads or calculates it first (and
        attributes to self)"""
        if self.chern is None:
            self.get_chern(self, attribute=True)

        kcgfns.save_chern(self)
        return self.chern

    def calc_chern(self, attribute=True):
        chern = kcgfns.calc_chern(self)
        if attribute:
            self.chern = chern
        return chern

    def plot_chernbands(self, fig=None, ax=None, cax=None, outpath=None, **kwargs):
        """

        Parameters
        ----------
        fig : matplotlib Figure instance or None
        ax : None or matplotlib axis instance for plotting dispersions
            axis instance in which to create the dispersion, or None
        cax : None or matplotlib axis instance, for colorbar
            axis instance in which to create the colorbar, or None
        outpath : str
        kwargs : keyword arguments for kspacecpfns.plot_chernbands()
            eps=1e-10, cmap='bbr0', vmin=-1.0, vmax=1.0, round_chern=True, dpi=150, alpha=1.0

        Returns
        -------
        """
        return kspacecpfns.plot_chernbands(self, fig=fig, ax=ax, cax=cax, **kwargs)


if __name__ == '__main__':
    '''running example for using my own chern calculator'''
    import argparse
    import socket
    import lepm.lattice_class as lattice_class
    from lepm.gyro_lattice_class import GyroLattice

    import lepm.lattice_elasticity as le
    import lepm.brillouin_zone_functions as bzfns
    import lepm.gyro_lattice_class_scripts as glatscripts
    import lepm.gyro_lattice_class_scripts_twistbc as glattwistscripts
    import lepm.gyro_lattice_class_scripts_twist2d as glattwist2dscripts
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create GyroLattice class instance,' +
                                                 ' with options to save or compute attributes of the class.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-calc_chern', '-calc_chern', help='Compute the chern numbers using the berry curvature',
                        action='store_true')
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

    # Geometry and physics arguments
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
          }

    cp = {'density': 400,  # number of points per area in BZ for calculation.
          'rootdir': cprootdir,
          'basis': args.basis,
          }

    if args.calc_chern:
        # Create glat
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        kchern = KChernGyro(glat, cp=cp)
        kchern.ensure_chern()

