import numpy as np
import lepm.lattice_class as lattice_class
import lepm.plotting.colormaps as lecmaps
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from lepm.haldane.haldane_lattice_class import HaldaneLattice
import lepm.plotting.plotting as leplt
import lepm.plotting.movies as lemov
import lepm.dataio as dio
import argparse
import copy

'''Use the HaldaneLattice class to see if squaring, cubing, etc the hamiltonian leads to diagonalization

Example usage:
python script_ari_locality_testing.py -N 1 -t1 0.0 -t2 0.1 -pin 1
python script_ari_locality_testing.py -N 3 -t1 0.0

'''

def set_cbar_lims(mat, cbar1, cbar2):
    minimag, maximag = np.min(np.imag(mat.ravel())), np.max(np.imag(mat.ravel()))
    minreal, maxreal = np.min(np.real(mat.ravel())), np.max(np.real(mat.ravel()))
    cbar1.set_ticks([np.min(np.imag(mat.ravel())), np.max(np.imag(mat.ravel()))])
    cbar1.set_ticklabels(['{0:0.1e}'.format(minimag), '{0:0.1e}'.format(maximag)])
    cbar2.set_ticks([np.min(np.real(mat.ravel())), np.max(np.real(mat.ravel()))])
    cbar2.set_ticklabels(['{0:0.1e}'.format(minreal), '{0:0.1e}'.format(maxreal)])


fs = 18
rootout = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/gyro_extension_paper/figure_drafting/ari_locality_idea/'
dio.ensure_dir(rootout)

# check input arguments for timestamp (name of simulation is timestamp)
parser = argparse.ArgumentParser(description='Specify parameters HaldaneLattice instance creation.')
parser.add_argument('-test_hlat', '--test_hlat',
                    help='Construct HaldaneLattice to test that it works/save pinning distribution',
                    action='store_true')
parser.add_argument('-plot_bands', '-plot_bands', help='plot the band structure in 3D in kspace',
                    action='store_true')
parser.add_argument('-dispersion', '--dispersion', help='Draw infinite/semi-infinite dispersion relation',
                    action='store_true')
parser.add_argument('-save_prpoly', '--save_prpoly',
                    help='Create dict and hist of excitation participation, grouped by polygonal contributions',
                    action='store_true')
parser.add_argument('-dcdisorder', '--dcdisorder', help='Construct DOS with delta correlated disorder and view ipr',
                    action='store_true')
parser.add_argument('-save_ipr', '--save_ipr', help='Load HaldaneLattice and save ipr',
                    action='store_true')
parser.add_argument('-DOSmovie', '--make_DOSmovie', help='Load the haldane lattice and make DOS movie of normal modes',
                    action='store_true')
parser.add_argument('-save_lattice', '--save_lattice', help='Construct a network and save lattice and the physics',
                    action='store_true')
parser.add_argument('-load_and_resave', '--load_lattice_resave_physics',
                    help='Load a lattice, and overwrite the physics like eigvals, ipr, DOS',
                    action='store_true')
parser.add_argument('-plot_pin', '--plot_pin',
                    help='Save a figure where sites are colored by their pinning strengths', action='store_true')
parser.add_argument('-show', '--show', help='Show results that have a show option', action='store_true')
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
parser.add_argument('-twistbcs', '--twistbcs',
                    help='Examine Hall conductance as berry curvature associated with state ' +
                         '|alpha(theta_twist, phi_twist)>', action='store_true')
parser.add_argument('-twiststrip', '--twiststrip',
                    help='Examine spectrum as function of twist angle theta_twist with states |alpha(theta_twist>',
                    action='store_true')
parser.add_argument('-edgelocalization', '--edgelocalization',
                    help='Check localization properties to the boundary of the sample', action='store_true')

# Geometry and physics arguments
parser.add_argument('-pureimNNN', '--pureimNNN', help='Make NNN hoppings purely imaginary', action='store_true')
parser.add_argument('-t2angles', '--t2angles', help='Make NNN hoppings based on bond angles', action='store_true')
parser.add_argument('-hexNNN', '--hexNNN', help='Ignore NNN hoppings in polygons other than hexagons',
                    action='store_true')
parser.add_argument('-Vpin', '--V0_pin_gauss',
                    help='St.deviation of distribution of delta-correlated pinning disorder',
                    type=float, default=0.0)
parser.add_argument('-Vspr', '--V0_spring_gauss',
                    help='St.deviation of distribution of delta-correlated bond disorder',
                    type=float, default=0.0)
parser.add_argument('-N', '--N',
                    help='Mesh width AND height, in number of lattice spacings (leave blank to spec separate dims)',
                    type=int, default=1)
parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=1)
parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=1)
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
parser.add_argument('-nonperiodic', '--not_periodicBC', help='Enforce periodic boundary conditions', action='store_true')
parser.add_argument('-periodic_strip', '--periodic_strip',
                    help='Enforce strip periodic boundary condition in horizontal dim only.' +
                         'Note that if this is true, then lp[periodicBC] is ALSO True.',
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
      'periodicBC': not args.not_periodicBC or args.periodic_strip,
      'periodic_strip': args.periodic_strip,
      'loadlattice_z': args.loadlattice_z,
      'alph': args.alph,
      'eta_alph': args.eta_alph,
      'origin': np.array([0., 0.]),
      't1': float(args.t1.replace('n', '-').replace('p', '.')),
      't2': float(args.t2.replace('n', '-').replace('p', '.')),
      't2a': float(args.t2a.replace('n', '-').replace('p', '.')),
      'pin': float(args.pin.replace('n', '-').replace('p', '.')) * 1j,
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
      }

lat = lattice_class.Lattice(lp)
try:
    lat.load()
except:
    lat.build()

hlat = HaldaneLattice(lat, lp)
mat = hlat.get_matrix()
print 'mat = ', mat

# Get name from meshfn and exten
name = lp['meshfn'].split('/')[-1] + lp['meshfn_exten']
outdir = rootout + name + '/'
dio.ensure_dir(outdir)

# Create figures
# Check NNN hoppings
[ax, axcb] = lat.plot_lat_colorbonds(np.ones_like(lat.BL[:, 0]), includeNNN=True, close=False, save=False, arrow_alpha=0.2)
pin = lp['pin']
print 'pin = ', pin
print ''
ax.set_title(r'$M=$' + '{0:0.1f}'.format(float(np.real(pin))) + '+i{0:0.1f}'.format(float(np.imag(pin))) +
             r', $t_1=$' + '{0:0.1f}'.format(lp['t1']) + r', $t_2=$' + '{0:0.1f}'.format(lp['t2']), fontsize=fs)
plt.savefig(outdir + name + '_mat00.png')

# fig, ax, cax = leplt.initialize_1panel_cbar_fig()
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat, close=False)
fig.text(0.5, 0.9, r'$D$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat, cbar1, cbar2)
plt.savefig(outdir + name + '_mat01.png')

# square the matrix
mat2 = np.dot(mat, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat2, close=False)
fig.text(0.5, 0.9, r'$D^2$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat2, cbar1, cbar2)
plt.savefig(outdir + name + '_mat02.png')

# cube the matrix
mat3 = np.dot(np.dot(mat, mat), mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat3, close=False)
fig.text(0.5, 0.9, r'$D^3$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat3, cbar1, cbar2)
plt.savefig(outdir + name + '_mat03.png')

# mult the matrix again
mat4 = np.dot(mat3, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat4, close=False)
fig.text(0.5, 0.9, r'$D^4$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat4, cbar1, cbar2)
plt.savefig(outdir + name + '_mat04.png')
# mult the matrix again
mat5 = np.dot(mat4, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat5, close=False)
fig.text(0.5, 0.9, r'$D^5$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat5, cbar1, cbar2)
plt.savefig(outdir + name + '_mat05.png')
# mult the matrix again
mat6 = np.dot(mat5, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat6, close=False)
fig.text(0.5, 0.9, r'$D^6$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat6, cbar1, cbar2)
plt.savefig(outdir + name + '_mat06.png')
# mult the matrix again
mat7 = np.dot(mat6, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat7, close=False)
fig.text(0.5, 0.9, r'$D^7$', transform=fig.transFigure, fontsize=fs)
set_cbar_lims(mat7, cbar1, cbar2)
plt.savefig(outdir + name + '_mat07.png')
# mult the matrix again
mat8 = np.dot(mat7, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat8, close=False)
set_cbar_lims(mat8, cbar1, cbar2)
fig.text(0.5, 0.9, r'$D^8$', transform=fig.transFigure, fontsize=fs)
plt.savefig(outdir + name + '_mat08.png')
# mult the matrix again
mat9 = np.dot(mat8, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat9, close=False)
set_cbar_lims(mat9, cbar1, cbar2)
fig.text(0.5, 0.9, r'$D^9$', transform=fig.transFigure, fontsize=fs)
plt.savefig(outdir + name + '_mat09.png')
# mult the matrix again
mat10 = np.dot(mat9, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat10, close=False)
set_cbar_lims(mat10, cbar1, cbar2)
fig.text(0.5, 0.9, r'$D^{10}$', transform=fig.transFigure, fontsize=fs)
plt.savefig(outdir + name + '_mat10.png')

# mult the matrix many times
mat15 = copy.deepcopy(mat10)
for ii in range(5):
    mat15 = np.dot(mat15, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat15, close=False)
set_cbar_lims(mat15, cbar1, cbar2)
fig.text(0.5, 0.9, r'$D^{15}$', transform=fig.transFigure, fontsize=fs)
plt.savefig(outdir + name + '_mat11.png')

mat16 = copy.deepcopy(mat10)
for ii in range(6):
    mat16 = np.dot(mat16, mat)
fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(mat16, close=False)
set_cbar_lims(mat16, cbar1, cbar2)
fig.text(0.5, 0.9, r'$D^{16}$', transform=fig.transFigure, fontsize=fs)
plt.savefig(outdir + name + '_mat12.png')

lemov.make_movie(outdir + name + '_mat', outdir[:-1], indexsz='02', framerate=1., imgdir=outdir, rm_images=True,
                 save_into_subdir=True)
