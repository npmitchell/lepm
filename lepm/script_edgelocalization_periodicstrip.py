import numpy as np
from lepm.gyro_lattice_class import GyroLattice
from lepm.lattice_class.lattice_class import Lattice

# check input arguments for timestamp (name of simulation is timestamp)
parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
parser.add_argument('-save_prpoly', '--save_prpoly',
                    help='Create dict and hist of excitation participation, grouped by polygonal contributions',
                    action='store_true')
parser.add_argument('-dcdisorder', '--dcdisorder', help='Construct DOS with delta correlated disorder and view ipr',
                    action='store_true')
parser.add_argument('-save_ipr', '--save_ipr', help='Load GyroLattice and save ipr',
                    action='store_true')
parser.add_argument('-DOSmovie', '--make_DOSmovie', help='Load the gyro lattice and make DOS movie of normal modes',
                    action='store_true')
parser.add_argument('-save_lattice', '--save_lattice', help='Construct a network and save lattice and the physics',
                    action='store_true')
parser.add_argument('-load_and_resave', '--load_lattice_resave_physics',
                    help='Load a lattice, and overwrite the physics like eigvals, ipr, DOS',
                    action='store_true')
parser.add_argument('-ldos', '--load_calc_ldos', help='Compute local density of states', action='store_true')
parser.add_argument('-gap_scaling', '--gap_scaling', help='Study scaling of the numerical gap', action='store_true')
parser.add_argument('-localization', '--localization', help='Seek exponential localization', action='store_true')
parser.add_argument('-save_eig', '--save_eig', help='Save eigvect/val during get_localization', action='store_true')
parser.add_argument('-save_images', '--save_images', help='Save movie for localization', action='store_true')
parser.add_argument('-charge', '--plot_charge', help='Sum amplitudes of modes in band', action='store_true')
parser.add_argument('-omegac', '--omegac', help='Cutoff (upper) freq for summing charge', type=float, default=0.0)
parser.add_argument('-illpoly', '--plot_localized_states',
                    help='Plot all localized states and show non-hex polygons', action='store_true')
parser.add_argument('-frange', '--freq_range', help='Range of freqs to analyze in illpoly', type=str, default='0/0')

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
parser.add_argument('-Omk', '--Omk', help='Spring frequency', type=str, default='-1.0')
parser.add_argument('-Omg', '--Omg', help='Pinning frequency', type=str, default='-1.0')
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
parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                    type=str, default='0.667')
parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                    type=str, default='0.000')
parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                    default=0.00)
parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.000)
parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
parser.add_argument('-subconf', '--sub_realization_number', help='Decoration realization number', type=int,
                    default=01)
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
      'periodicBC': args.periodicBC,
      'loadlattice_z': args.loadlattice_z,
      'alph': args.alph,
      'eta_alph': args.eta_alph,
      'origin': np.array([0., 0.]),
      'Omk': float((args.Omk).replace('n', '-').replace('p', '.')),
      'Omg': float((args.Omg).replace('n', '-').replace('p', '.')),
      'V0_pin_gauss': args.V0_pin_gauss,
      'V0_spring_gauss': args.V0_spring_gauss,
      'dcdisorder': dcdisorder,
      'percolation_density': args.percolation_density,
      'ABDelta': args.ABDelta,
      'thres': args.thres,
      }

meshfn = le.find_meshfn(lp)
lp['meshfn'] = meshfn
lat = Lattice(lp)
lat.load()
glat = GyroLattice(lat, lp)
glat.load()
eigval, eigvect = glat.load_eigval_eigvect()
