import numpy as np
import lepm.lattice_elasticity as le
import lepm.lattice_functions as latfns
import lepm.lattice_functions_nnn as lfnsnnn
import lepm.le_geometry as leg
import lepm.structure as lestructure
import lepm.plotting.colormaps as cmaps
import lepm.plotting.science_plot_style as sps
import lepm.stringformat as sf
import lepm.dataio as dio
import lepm.plotting.network_visualization as nvis
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import argparse
import matplotlib.path as mplpath
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy
from scipy.ndimage import imread
from scipy import fftpack
import scipy
from scipy.spatial import Delaunay
import scipy.optimize as opt
import sys
import plotting.colormaps as cmaps
import math
import cmath
import shapely.geometry as sg

'''
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Differences between my code and Lisa's
*Ni --> NL
*Nk --> KL
KL : Correponds to NL matrix.  1 corresponds to a true connection, 0 signifies there is not a connection, -1 is periodic connection
'''


class Vertex2d:
    """Create a lattice instance. Note that all physics, such as spring constants or masses of particles is reserved for
    other classes that use the Lattice instance. The Lattice instance is the geometry and topology (ie the connections)
    only, no physics.

    Attributes
    ----------
    lp : dict with keys including BBox, LL, and meshfn
        lp is a dictionary ('lattice parameters') which can contain:
            LatticeTop, shape, NH, NV,
            rootdir, meshfn, lattice_exten, phi, delta, theta, eta, x1, x2, x3, z, source, loadlattice_number,
            check, Ndefects, Bvec, dislocation_xy, target_z, make_slit, cutz_method, cutLfrac
            LVUC
            LV
            UC
            LL : tuple of 2 floats
    xy : N x 2 float array
        2D positions of points (positions x,y). Row i is the x,y position of the ith particle.
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have
        the maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x max(#neighbors) int array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    BM : array of length #pts x max(#neighbors)
        The (i,j)th element is the distance bwtn particles (length of the bond) connecting the ith particle to its jth
        neighbor (the particle with index NL[i,j]). Not an attribute
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points. Negative values denote particles connected through
        periodic bonds.
    bL : array of length #bonds
        The ith element is the length of of the ith bond in BL. Not an attribute.
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
        These give the dimensions of the network in x and y, for S(k) measurements and other periodic things.
    polygons : list of int lists
        indices of xy points defining polygons.
    NLNNN : array of length #pts x max(#next-nearest-neighbors)
        Next-nearest-neighbor array: The ith row contains indices for the next nearest neighbors for the ith point.
    KLNNN : array of length #pts x max(#next-nearest-neighbors)
        Next-nearest-neighbor connectivity/orientation array:
        The ith row states whether a next nearest neighbors is counterclockwise (1) or clockwise(-1)
    nljnnn : #pts x max(#NNN) int array
        nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
        the next nearest neighbor of i through the particle nljnnn[i, j]
    kljnnn : #pts x max(#NNN) int array
        bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
    klknnn : #pts x max(#NNN) int array
        bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
        (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    PVxij : NP x NP float array (optional, for periodic lattices)
        ijth element of PVxij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
        particle i (at xy[i])
    PVyij : NP x NP float array (optional, for periodic lattices)
        ijth element of PVyij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
        particle i (at xy[i])
    BBox : #vertices x 2 float array
        bounding polygon for the network, usually a rectangle
    PV : 2 x 2 float array
        periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
    gxy : tuple of NX x NY arrays
        two-point correlation function in the positions of particles as function of vector distance x,y
    gr :
        two-point correlation function in the positions of particles as function of scalar distance
    """

    def __init__(self, lat, model='Farhadifar', aa=1., bb=1., sigma=1.,
                 A0=1., L0=0.):
        """Create a lattice instance."""
        self.lattice = lat
        self.model = model
        self.aa = aa
        self.bb = bb
        self.sigma = sigma
        self.A0 = A0
        self.L0 = L0

    def __hash__(self):
        return hash((self.xy, self.NL, self.KL, self.BL, self.PVx, self.PVy))

    def __eq__(self, other):
        return hasattr(other, 'xy') and hasattr(other, 'NL') and hasattr(other, 'KL') and hasattr(other, 'BL') and \
               self.xy == other.xy and self.NL == other.NL and self.KL == other.KL and self.BL == other.BL and \
               self.PVx == other.PVx and self.PVy == other.PVy

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def relax(self):



if __name__ == '__main__':
    import lepm.Lattice
    '''Perform an example of using the 2d Vertex model class'''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify arguments for the current vertex model instance.')

    # Task
    parser.add_argument('-relax', '--relax',
                        help='Elastically relax the 2d vertex model', action='store_true')

    # Lattice Geometry Parameters
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings ' +
                                          '(leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-NP', '--NP_load', help='Number of particles in mesh, overwrites N, NH, and NV.',
                        type=int, default=0)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular,' +
                                                    ' deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hexagonal')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
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
                        help='Declare whether or not to cut bonds to obtain target coordination number z', type=bool,
                        default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='Position of single dislocation, if not centered at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')
    parser.add_argument('-thres', '--thres', help='Threshold distance (from letters in uofc, for ex)',
                        type=float, default=1.0)
    parser.add_argument('-spreading_time', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)', type=float, default=0.0)
    parser.add_argument('-dt', '--spreading_dt',
                        help='Time step for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.001)
    parser.add_argument('-kicksz', '--kicksz',
                        help='Average of log of kick magnitudes for loading randorg_gammakick pointsets.' +
                             'This sets the scale of the powerlaw kicking procedure',
                        type=float, default=-1.50)

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-periodic_strip', '--periodic_strip',
                        help='Enforce strip periodic boundary condition in horizontal dim', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-phi', '--phi', help='Shear angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=0.0)
    parser.add_argument('-delta', '--delta', help='Deformation angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=120. / 180.)
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float,
                        default=0.000)
    parser.add_argument('-aratio', '--aratio', help='Aspect ratio used in determining the geometry of the lattice',
                        type=float, default=1.000)
    parser.add_argument('-x1', '--x1',
                        help='1st Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x2', '--x2',
                        help='2nd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x3', '--x3',
                        help='3rd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-zz', '--zz',
                        help='4th Deformation parameter for deformed_kagome', type=float, default=0.00)

    parser.add_argument('-huno', '--hyperuniform_number', help='Hyperuniform realization number', type=str,
                        default='01')
    parser.add_argument('-conf', '--conf', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-skip_gr', '--skip_gr', help='Skip calculation of g(r) correlation function for the lattice',
                        action='store_true')
    parser.add_argument('-skip_gxy', '--skip_gxy',
                        help='Skip calculation of g(x,y) 2D correlation function for the lattice', action='store_true')
    parser.add_argument('-skip_sigN', '--skip_sigN', help='Skip calculation of variance_N(R)', action='store_true')
    args = parser.parse_args()

    if args.N > 0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV

    # initial strain
    strain = 0.0
    # z = 4.0 #target z
    if args.LatticeTop == 'linear':
        shape = 'line'
    else:
        shape = args.shape

    theta = args.theta
    eta = args.eta
    transpose_lattice = 0

    make_slit = args.make_slit
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
          'phi': args.phi * np.pi,
          'delta': np.pi * args.delta,
          'theta': theta,
          'eta': eta,
          'x1': args.x1,
          'x2': args.x2,
          'x3': args.x3,
          'z': args.zz,
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
          'conf': args.conf,
          'periodicBC': args.periodicBC or args.periodic_strip,
          'periodic_strip': args.periodic_strip,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'origin': np.array([0., 0.]),
          'thres': args.thres,
          'spreading_time': args.spreading_time,
          'spreading_dt': args.spreading_dt,
          'kicksz': args.kicksz,
          'aratio': args.aratio,
          }

    lat = lepm.Lattice(lp)
    try:
        lat.load()
    except IOError:
        lat.build()

    if args.relax:
        vm = Vertex2d(lat)
        vm.relax()
