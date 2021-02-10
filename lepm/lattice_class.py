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
import lepm.plotting.colormaps as lecmaps

'''
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Differences between my code and Lisa's
*Ni --> NL
*Nk --> KL
KL : Correponds to NL matrix.  1 corresponds to a true connection, 0 signifies there is not a connection, -1 is periodic connection
'''


class Lattice:
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

    def __init__(self, lp={}, xy=np.array([]), NL=np.array([]), KL=np.array([]), BL=np.array([]), polygons=None,
                 PVxydict=None, PVx=None, PVy=None, PVxij=None, PVyij=None, NLNNN=None, KLNNN=None, NLNNNangles=None,
                 nljnnn=None, kljnnn=None, klknnn=None, pvxnnn=None, pvynnn=None,
                 BBox=None, PV=None, LVUC=None, gxy=None, gr=None, boundary=None, bz=None):
        """Create a lattice instance."""
        self.xy = xy
        self.NL = NL
        self.KL = KL
        self.BL = BL
        lp = latfns.complete_lp(lp)
        self.lp = lp
        self.PVx = PVx
        self.PVy = PVy
        self.PVxij = PVxij
        self.PVyij = PVyij
        self.PVxydict = PVxydict

        # Non-essential attributes
        self.polygons = polygons
        self.NLNNN = NLNNN
        self.KLNNN = KLNNN
        self.NLNNNangles = NLNNNangles
        self.nljnnn = nljnnn
        self.kljnnn = kljnnn
        self.klknnn = klknnn
        self.pvxnnn = pvxnnn
        self.pvynnn = pvynnn
        self.PV = PV
        self.LVUC = LVUC
        self.gxy = gxy
        self.gr = gr
        self.boundary = boundary
        self.bz = bz
        self.intnn_info = {}

    def build(self):
        from lepm.build.make_lattice import build_lattice
        build_lattice(self)
        # print 'Lattice: lp[delta] = ', self.lp['delta']

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

    def get_meshfn(self, attribute=True):
        """Return or build the location where the lattice is or would be stored

        Parameters
        ----------
        attribute : bool
            store meshfn as self.lp['meshfn']

        Returns
        -------
        meshfn : str
        """
        if 'meshfn' in self.lp:
            return self.lp['meshfn']
        else:
            # attempt to find meshfn based on stored lattice
            meshfn, xyfn_trash = le.build_meshfn(self.lp)
            print 'lepm.lattice_class.Lattice(): output of le.build_meshfn() = ', meshfn
            if attribute:
                self.lp['meshfn'] = meshfn
            return meshfn

    def automeshfn(self):
        try:
            fn = self.lp['meshfn']
            print 'glob.glob(fn) = ', glob.glob(fn)
            if not glob.glob(fn):
                print '\n\nLatticeClass: WARNING: Replacing rootdir path with midways in lattice params dict\n\n\n'
                fn = fn.replace('/Users/npmitchell/Dropbox/Soft_Matter/GPU/', '/home/npmitchell/scratch-midway/')
                self.lp['meshfn'] = fn
        except KeyError:
            print 'lattice_functions: No meshfn specified for lattice.load(), attempting to find a match...'
            self.get_meshfn()
            fn = self.lp['meshfn']
            # raise RuntimeError('There is no meshfn in the lp (params) dictionary for this lattice instance')

        return fn

    def load(self, meshfn='auto', load_polygons=False, load_LVUC=False, load_gxy=False, check=False):
        """Load a saved lattice into the lattice instance. If meshfn is specified, loads that lattice.
        Otherwise, attempts to load lattice based on parameter self.lp['meshfn']. If that is also unavailable,
        loads from lp[rootdir]/networks/self.LatticeTop/self.lp[lattice_exten]_NH_x_NV.
        """
        latfns.load(self, meshfn=meshfn, load_polygons=load_polygons, load_LVUC=load_LVUC,
                    load_gxy=load_gxy, check=check)

    # def fix_delta(self):
    #     """Replace self.lp['delta'] with the correct value determined from self.lp['delta_lattice'].
    #     Then resave lattice_params.txt
    #     """
    #     print 'replacing lp[delta] = ', self.lp['delta']
    #     print '... with lp[delta] = ', sf.str2float(self.lp['delta_lattice']) * np.pi
    #     self.lp['delta'] = sf.str2float(self.lp['delta_lattice']) * np.pi
    #
    #     print 'resaving...'
    #     dio.save_dict(self.lp, dio.prepdir(meshfn) + 'lattice_params.txt', header, keyfmt='auto', valfmt='auto',
    #                   padding_var=7)
    #     sys.exit()

    def load_PV(self, meshfn='auto'):
        if meshfn == 'auto':
            meshfn = self.automeshfn()
        if meshfn[-1] == '/':
            meshfn = meshfn[:-1]
        name = meshfn.split('/')[-1]
        print 'Lattice.load_PV(): loading ' + meshfn + '/' + name + '_PV.txt'
        self.PV = np.loadtxt(meshfn + '/' + name + '_PV.txt', delimiter=',', dtype=float)
        # print 'Lattice: loaded PV'
        return self.PV

    def load_LVUC(self):
        # Attempt to load polygons
        if glob.glob(dio.prepdir(self.lp['meshfn']) + "LVUC.txt"):
            self.LVUC = np.loadtxt(dio.prepdir(self.lp['meshfn']) + "LVUC.txt", delimiter=',')
        else:
            raise RuntimeError('No polygons pickle stored for this lattice.')

        return self.LVUC

    def load_polygons(self):
        # Attempt to load polygons
        if glob.glob(dio.prepdir(self.lp['meshfn']) + "polygons.pkl"):
            self.polygons = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "polygons.pkl", "r"))
        else:
            print 'No polygons pickle stored for this lattice. Creating it! <-- meshfn = ', self.lp['meshfn']
            self.save_polygons(attribute=True)

        return self.polygons

    def load_NLNNN(self, attribute=True, nnnexten=None):
        """"""
        # Attempt to load NLNNN
        # first grab name extension of the NNN files
        if nnnexten is None:
            nnnexten = self.get_nnnexten()

        if glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl"):
            NLNNN = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl", "r"))
        else:
            print 'No NLNNN pickle stored for this lattice. Creating it! <-- meshfn = ', self.lp['meshfn']
            NLNNN = self.save_NLNNN(attribute=True, nnnexten=nnnexten)

        if attribute:
            self.NLNNN = NLNNN

        return self.NLNNN

    def load_KLNNN(self, attribute=True, nnnexten=None):
        """"""
        # Attempt to load KLNNN
        # first grab name extension of the NNN files
        if nnnexten is None:
            nnnexten = self.get_nnnexten()

        if glob.glob(dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl"):
            KLNNN = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl", "r"))
        else:
            print 'No KLNNN pickle stored for this lattice. Creating it! <-- meshfn = ', self.lp['meshfn']
            KLNNN = self.save_KLNNN(attribute=True, nnnexten=nnnexten)

        if attribute:
            self.KLNNN = KLNNN

        return self.KLNNN

    def load_NLNNNangles(self, attribute=True):
        """"""
        # Attempt to load NLNNN
        if glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNNangles.pkl"):
            NLNNNangles = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "NLNNNangles.pkl", "r"))
        else:
            print 'No NLNNN pickle stored for this lattice. Creating it! <-- meshfn = ', self.lp['meshfn']
            self.save_NLNNNangles(attribute=True)

        if attribute:
            self.NLNNNangles = NLNNNangles

        return self.NLNNNangles

    def load_NNN_info(self):
        """"""
        # Attempt to load NLNNN
        g0 = glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNN.pkl")
        g1 = glob.glob(dio.prepdir(self.lp['meshfn']) + "KLNNN.pkl")
        g2 = glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNNangles.pkl")
        if g0 and g1 and g2:
            self.NLNNN = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "NLNNN.pkl", "r"))
            self.KLNNN = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "KLNNN.pkl", "r"))
            self.NLNNNangles = pickle.load(open(dio.prepdir(self.lp['meshfn']) + "NLNNNangles.pkl", "r"))
        else:
            print 'NNN info pickles missing for this lattice. Creating them! <-- meshfn = ', self.lp['meshfn']
            self.save_NNN_info(attribute=True)

        return self.NLNNN, self.KLNNN, self.NLNNNangles

    def load_gxy(self):
        """"""
        fn = dio.prepdir(self.lp['meshfn']) + "gxy.txt"
        if glob.glob(fn):
            gxy = np.loadtxt(fn, delimiter=',')
        raise RuntimeError('Need to cast loaded array gxy as xygrid for saving function...')
        lestructure.save_gxy()

    def load_bz(self):
        """"""
        fn = dio.prepdir(self.lp['meshfn']) + "bz.txt"
        if glob.glob(fn):
            gxy = np.loadtxt(fn, delimiter=',')
        else:
            print 'Could not load bz, returning None'
            return None

    def save(self, skip_polygons=False, check=False):
        """Save lattice to hard disk, outputting txt files (and images) to meshfn, which is located at
        lp[rootdir]/networks/LatticeTop/lattice_exten_NH_x_NV
        """
        print('Saving... ' + self.lp['lattice_exten'])
        fbase = dio.prepdir(self.lp['rootdir']) + 'networks/' + self.lp['LatticeTop'] + '/'
        dio.ensure_dir(fbase)
        fmain = self.get_fmain()

        meshfn = fbase + fmain
        self.lp['meshfn'] = meshfn
        fextn = '.txt'
        print 'lattice_functions: Saving ', fbase + fmain + '_xy' + fextn
        np.savetxt(fbase + fmain + '_xy' + fextn, self.xy, fmt='%.18e', delimiter=',', header='x,y')
        np.savetxt(fbase + fmain + '_BL' + fextn, self.BL, fmt='%i', delimiter=',', header='BL')
        np.savetxt(fbase + fmain + '_NL' + fextn, self.NL, fmt='%i', delimiter=',', header='NL')
        np.savetxt(fbase + fmain + '_KL' + fextn, self.KL, fmt='%i', delimiter=',', header='KL')

        try:
            np.savetxt(meshfn + '/' + fmain + '_LVUC' + fextn, self.LVUC, fmt='%i', delimiter=',',
                       header='LVUC : lattice vectors and unit cell vector identification')
        except:
            print 'lattice_functions: Could not output LVUC...'
        # try:
        #   np.savetxt(meshfn+'/'+fmain+'_LV'+fextn, self.LV, fmt='%.18e', delimiter=',', header='LV : lattice vectors')
        # except:
        #     print 'lattice_functions: Could not output LV...'
        # try:
        #    np.savetxt(meshfn+'/'+fmain+'_UC'+fextn, self.UC, fmt='%.18e',
        #               delimiter=',', header='UC : unit cell vectors --> vectors to points in repeated macrocell')
        # except:
        #     print 'lattice_functions: Could not output UC...'
        # try:
        #    np.savetxt(meshfn+'/'+fmain+'_LL'+fextn, self.lp['LL'], fmt='%.18e', delimiter=',',
        #    header='LL : real-space dimensions in x,y (for use in sampling k-space, for ex.)')
        # except:
        #     print 'lattice_functions: Could not output LL...'
        # try:
        #    np.savetxt(meshfn+'/'+fmain+'_polygon'+fextn, self.polygon, fmt='%.18e', delimiter=',',
        #               header='polygon : real-space bounding polygon, for cropping lattice')
        # except:
        #     print 'lattice_functions: Could not output polygon...'

        # Save everything else as lattice_params.txt
        header = 'lattice parameters dictionary'
        # self.lp['LatticeTop'] = self.LatticeTop
        # self.lp['shape'] = self.shape
        # self.lp['NH'] = self.NH
        # self.lp['NV'] = self.NV
        # self.lp['BBox'] = self.lp['BBox']
        dio.ensure_dir(meshfn + '/')
        dio.save_dict(self.lp, meshfn + '/lattice_params.txt', header, keyfmt='auto', valfmt='auto', padding_var=7)

        if 'periodicBC' in self.lp:
            if self.lp['periodicBC']:
                # PVxydict
                header = 'PVxydict : '
                filename = meshfn + '/' + fmain + '_PVxydict' + fextn
                print 'lattice_functions: saving ', filename
                dio.save_dict(self.PVxydict, filename, header)

                header = 'PVx: ijth element of PVx are the x-components of the vector taking NL[i,j] to its image' + \
                         ' as seen by particle i'
                filename = meshfn + '/' + fmain + '_PVx' + fextn
                print 'lattice_functions: saving ', filename
                np.savetxt(filename, self.PVx, delimiter=',', header=header)

                header = 'PVy: ijth element of PVy are the y-components of the vector taking NL[i,j] to its image' + \
                         ' as seen by particle i'
                filename = meshfn + '/' + fmain + '_PVy' + fextn
                print 'lattice_functions: PVy ->', self.PVy
                print 'lattice_functions: saving ', filename
                np.savetxt(filename, self.PVy, delimiter=',', header=header)

                self.save_PV()

        # Save polygons
        if self.lp['LatticeTop'] != 'linear' and skip_polygons is False:
            self.save_polygons(check=check)

        # Check lattice
        print('Plotting and printing image...')
        fname = fbase + fmain + '.png'
        title = self.lp['lattice_exten'] + ' ' + str(self.lp['NH']) + ' x ' + str(self.lp['NV'])
        xlimv = max(max(self.lp['NH'] * 0.5 + 0.5, self.lp['NV'] * 0.5 + 0.5),
                    max((np.max(self.xy[:, 0]) + 2) * 1.05, (np.max(self.xy[:, 1]) + 2) * 1.05))
        ylimv = xlimv
        climv = 0.1

        # Register cmap
        if not 'BlueBlackRed' in plt.colormaps():
            plt.register_cmap(name='BlueBlackRed', cmap=cmaps.BlueBlackRed)

        if self.BL.size > 0:
            nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), fname, title, NL=self.NL, KL=self.KL,
                               PVx=self.PVx, PVy=self.PVy, xlimv=xlimv, ylimv=ylimv, climv=climv,
                               colormap='BlueBlackRed', bgcolor='#FFFFFF')
        else:
            '''There are no bonds'''
            nvis.movie_plot_2D(self.xy, self.BL, np.zeros((len(self.BL), 1)), fname, title,
                               PVx=self.PVx, PVy=self.PVy, xlimv=xlimv, ylimv=ylimv, climv=climv,
                               colormap='BlueBlackRed', bgcolor='#FFFFFF')

    def get_fmain(self):
        """Get the name of the network, without the path

        Returns
        -------
        fmain : str
            The name of the network to follow the meshfn, mimicks the final part of the meshfn after last '/'
        """
        meshfn = self.automeshfn()
        if meshfn[-1] == '/':
            meshfn = meshfn[:-1]
        fmain = meshfn.split('/')[-1]
        return fmain

    def save_PV(self):
        fmain = self.get_fmain()
        header = 'lattice_class: PV: periodic vectors for periodic boundary conditions'
        if 'meshfn' not in self.lp:
            self.get_meshfn(attribute=True)
        elif self.lp['meshfn'] is None:
            self.get_meshfn(attribute=True)

        self.get_PV(attribute=True)
        print 'lattice_class.Lattice: saving as txt-> self.PV = ', self.PV
        filename = dio.prepdir(self.lp['meshfn']) + fmain + '_PV.txt'
        np.savetxt(filename, self.PV, delimiter=',', header=header)
        print 'lattice_class: saved file: ', filename

    def save_polygons(self, attribute=True, check=False):
        """Obtain the polygons comprising the network and save it as pickle"""
        if self.polygons is not None:
            polygons = self.polygons
        else:
            polygons = le.extract_polygons_lattice(self.xy, self.BL, self.NL, self.KL, PVx=self.PVx, PVy=self.PVy,
                                                   viewmethod=False, check=check)
            if attribute:
                self.polygons = polygons

        print 'dumping polygons: ', dio.prepdir(self.lp['meshfn']) + "polygons.pkl"
        pickle.dump(polygons, open(dio.prepdir(self.lp['meshfn']) + "polygons.pkl", "wb"))

    def save_NLNNNangles(self, attribute=True, nnnexten=None):
        """Save the NLNNNangles as a pickle in meshfn directory"""
        if nnnexten is None:
            nnnexten = self.get_nnnexten()
        NLNNNangles = self.get_NLNNNangles(attribute=attribute, nnnexten=nnnexten)
        print 'dumping NLNNNangles: ', dio.prepdir(self.lp['meshfn']) + "NLNNNangles" + nnnexten + ".pkl"
        pickle.dump(NLNNNangles, open(dio.prepdir(self.lp['meshfn']) + "NLNNNangles" + nnnexten + ".pkl", "wb"))

    def save_NLNNN(self, attribute=True, nnnexten=None):
        """"""
        if nnnexten is None:
            nnnexten = self.get_nnnexten()
        NLNNN = self.get_NLNNN(attribute=attribute, nnnexten=nnnexten)
        print 'dumping NLNNN: ' + dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl"
        pickle.dump(NLNNN, open(dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl", "wb"))
        return NLNNN

    def save_KLNNN(self, KLNNN=None, attribute=True, nnnexten=None):
        """Save KLNNN to disk for current Lattice

        Parameters
        ----------
        attribute : bool
            ensure that KLNNN is attributed to self
        nnnexten : str or None
            string specifier for details on this particular KLNNN being saved
        """
        if nnnexten is None:
            nnnexten = self.get_nnnexten()
        KLNNN = self.get_KLNNN(attribute=attribute, nnnexten=None)
        # polygons = self.get_polygons()
        # NLNNN, KLNNN = le.calc_NLNNN_and_KLNNN(self.xy, self.BL, self.NL, self.KL, self.PVx, self.PVy,
        #                                        polygons=polygons, ignore_tris=self.lp['ignore_tris'])

        print 'dumping KLNNN: ', dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl"
        pickle.dump(KLNNN, open(dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl", "wb"))

    def save_NNN_info(self, attribute=True, nnnexten=None):
        """Dump NLNNN, KLNNN, and NLNNNangles as pickles in meshfn"""
        if nnnexten is None:
            nnnexten = self.get_nnnexten()
        NLNNN, KLNNN, NLNNNangles = self.get_NNN_info(attribute=attribute)
        print 'dumping NLNNN: ', dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl"
        pickle.dump(NLNNN, open(dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl", "wb"))
        print 'dumping KLNNN: ', dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl"
        pickle.dump(KLNNN, open(dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl", "wb"))
        print 'dumping NLNNNangles: ', dio.prepdir(self.lp['meshfn']) + "NLNNNangles" + nnnexten + ".pkl"
        pickle.dump(NLNNNangles, open(dio.prepdir(self.lp['meshfn']) + "NLNNNangles" + nnnexten + ".pkl", "wb"))

    def save_bz(self, bz=None, attribute=True):
        """

        Parameters
        ----------
        bz : n x 2 float array or None
            The corners of the brillouin zone to save or None (to compute it or grab stored bz)
        attribute : bool
            attribute the computed/loaded bz to self
        """
        bz = self.get_bz(attribute=attribute)
        fn = dio.prepdir(self.lp['meshfn']) + "bz" + nnnexten + ".txt"
        print 'dumping bz to ' + fn
        np.savetxt(fn, bz, header='Brillouin zone vertices')

    def save_gxy_gr(self, outdir=None):
        self.get_gxy_gr()
        raise RuntimeError('Lattice.save_gxy_gr() is not written!')

    def get_polygons(self, attribute=False, save_if_missing=False, check=False):
        """Obtain the polygons comprising the network
        Returns
        -------
        polygons : list of lists of ints
            list of lists of indices of each polygon
        """
        if self.polygons is not None:
            return self.polygons
        elif glob.glob(dio.prepdir(self.lp['meshfn']) + "polygons.pkl"):
            return self.load_polygons()
        else:
            polygons = self.calc_polygons(attribute=attribute)

            if save_if_missing:
                self.save_polygons()
            return polygons

    def get_NLNNN(self, attribute=False, nnnexten=None):
        if self.NLNNN is not None:
            return self.NLNNN
        else:
            if nnnexten is None:
                nnnexten = self.get_nnnexten()
            nlnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNN" + nnnexten + ".pkl")
            if nlnnnglob:
                NLNNN = self.load_NLNNN(attribute=attribute, nnnexten=nnnexten)
                return NLNNN
            else:
                polygons = self.get_polygons()
                NLNNN, KLNNN = lfnsnnn.calc_NLNNN_and_KLNNN(self.xy, self.BL, self.NL, self.KL, self.PVx, self.PVy,
                                                       polygons=polygons, ignore_tris=self.lp['ignore_tris'])
                if attribute:
                    self.NLNNN = NLNNN
            return NLNNN

    def get_KLNNN(self, attribute=False, nnnexten=None):
        if self.KLNNN is not None:
            return self.KLNNN
        else:
            if nnnexten is None:
                nnnexten = self.get_nnnexten()
            nlnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "KLNNN" + nnnexten + ".pkl")
            if nlnnnglob:
                KLNNN = self.load_KLNNN(attribute=attribute, nnnexten=nnnexten)
                return KLNNN
            else:
                polygons = self.get_polygons()
                NLNNN, KLNNN = lfnsnnn.calc_NLNNN_and_KLNNN(self.xy, self.BL, self.NL, self.KL, self.PVx, self.PVy,
                                                            polygons=polygons, ignore_tris=self.lp['ignore_tris'])
                if attribute:
                    self.KLNNN = KLNNN
            return KLNNN

    def get_nnnexten(self):
        """Add description about any odd treatment of next nearest neighbor definitions"""
        if 'ignore_tris' in self.lp:
            if self.lp['ignore_tris']:
                nnnexten = '_notriangles'
            else:
                nnnexten = ''
        else:
            self.lp['ignore_tris'] = False
            nnnexten = ''
        return nnnexten

    def get_PVxyij(self, attribute=False):
        """Return PVxij and PVyij, computing if not already done so

        Parameters
        ----------
        attribute : bool
            store PVxyij and PVyij as attributes of the current Lattice instance

        Returns
        -------
        PVxij : NP x NP float array (optional, for periodic lattices)
            ijth element of PVxij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
            particle i (at xy[i])
        PVyij : NP x NP float array (optional, for periodic lattices)
            ijth element of PVyij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
            particle i (at xy[i])
        """
        if self.PVxij is not None and self.PVyij is not None:
            return self.PVxij, self.PVyij
        else:
            return self.calc_PVxyij(attribute=attribute)

    def get_NLNNN_and_KLNNN(self, attribute=False, save=True):
        """Compute the Next-nearest neighbor bond list and connection list.

        Parameters
        ----------
        attribute : bool
            attribute NLNNN and KLNNN to self after computed
        save : bool (default=True)
            save NLNNN and KLNNN to disk if they are not already loaded in memory and attributed to self

        Returns
        -------

        """
        if self.NLNNN is not None and self.KLNNN is not None:
            return self.NLNNN, self.KLNNN
        else:
            if 'ignore_tris' not in self.lp:
                self.lp['ignore_tris'] = False

            if self.lp['ignore_tris']:
                nlnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNN_notriangles.pkl")
                klnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "KLNNN_notriangles.pkl")
                self.lp['nnn_exten'] = '_notriangles'
            else:
                nlnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "NLNNN.pkl")
                klnnnglob = glob.glob(dio.prepdir(self.lp['meshfn']) + "KLNNN.pkl")
                self.lp['nnn_exten'] = ''

            if nlnnnglob and klnnnglob:
                print 'loading NLNNN and KLNNN from disk...'
                NLNNN = self.load_NLNNN(attribute=attribute, nnnexten=self.lp['nnn_exten'])
                KLNNN = self.load_KLNNN(attribute=attribute, nnnexten=self.lp['nnn_exten'])
                return NLNNN, KLNNN
            else:
                polygons = self.get_polygons()
                NLNNN, KLNNN = lfnsnnn.calc_NLNNN_and_KLNNN(self.xy, self.BL, self.NL, self.KL, self.PVx, self.PVy,
                                                            polygons=polygons, ignore_tris=self.lp['ignore_tris'])
                if attribute:
                    self.NLNNN = NLNNN
                    self.KLNNN = KLNNN

                if save:
                    self.save_NLNNN()
                    self.save_KLNNN()
            return NLNNN, KLNNN

    def get_NLNNNangles(self, attribute=False, cwccw=False):
        """Obtain next-nearest-neighbor bond angles
        Note: by default, cwccw is always False! Keep it that way unless using this function in a standalone script.
        This ensures that NLNNNangles contains ONLY counterclockwise bond angles
        """
        if self.NLNNNangles is not None:
            return self.NLNNNangles
        else:
            NLNNN, KLNNN = self.get_NLNNN_and_KLNNN(attribute=False)
            # Make sure lattice has PVx,y if periodic
            if self.lp['periodicBC'] and (self.PVx is None or self.PVy is None):
                raise RuntimeError('Periodic lattice has None for self.PVx or self.PVy')
            NLNNNangles = lfnsnnn.NNN_bond_angles(self.xy, self.NL, self.KL, NLNNN, KLNNN, PVx=self.PVx, PVy=self.PVy)
            if attribute:
                self.NLNNNangles = NLNNNangles

            return NLNNNangles

    def get_NNN_info(self, attribute=False):
        """Obtain the NNN neighbor list, connectivity list, and NNN angles shaped like NLNNN"""
        NLNNN, KLNNN = self.get_NLNNN_and_KLNNN(attribute=attribute)
        NLNNNangles = self.get_NLNNNangles(attribute=attribute)
        return NLNNN, KLNNN, NLNNNangles

    def get_nljnnn(self, attribute=False):
        """Obtain the nearest-neighbor list corresponding to the intermediate hoppings to the next nearest neighbor.

        Returns
        -------
        nljnnn : #pts x max(#NNN) int array
            nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that
            NLNNN[i, j] is the next nearest neighbor of i through the particle nljnnn[i, j]
        kljnnn : #pts x max(#NNN) int array
            bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
            (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
        klknnn : #pts x max(#NNN) int array
            bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
            (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
        """
        if self.nljnnn is not None and self.kljnnn is not None and self.klknnn is not None:
            return self.nljnnn, self.kljnnn, self.klknnn
        else:
            return self.calc_nljnnn(attribute=attribute)

    def get_pvnnn(self, attribute=False):
        if self.pvxnnn is not None and self.pvynnn is not None:
            return self.pvxnnn, self.pvynnn
        else:
            pvxnnn, pvynnn = self.calc_pvnnn(attribute=attribute)
            return pvxnnn, pvynnn

    def get_intnn_info(self, interaction_range, attribute=False):
        if interaction_range in self.intnn_info:
            # Build lists of intnn info for each order up to interaction range
            NLnns, KLnns, pvxnns, pvynns = [], [], [], []
            for ii in range(interaction_range):
                NLnns.append(self.intnn_info[ii]['NLnn'])
                KLnns.append(self.intnn_info[ii]['KLnn'])
                pvxnns.append(self.intnn_info[ii]['pvxnns'])
                pvynns.append(self.intnn_info[ii]['pvynns'])
        else:
            NLnns, KLnns, pvxnns, pvynns = lfnsnnn.calc_intnnn_info(self, interaction_range, attribute=attribute)

        return NLnns, KLnns, pvxnns, pvynns

    def get_combined_intnn_info(self, interaction_range):
        return lfnsnnn.calc_combined_intnn_info(self, interaction_range)

    def get_intnn_info_exclusive(self, interaction_range):
        """Get ith-nneighbor info and exclude a ith nn from being an (i+1)th nn or a
        (i+j)th nn, for j>0.
        """
        return lfnsnnn.calc_intnnn_info_exclusive(self, interaction_range)

    def get_PV(self, attribute=False):
        """Compute the periodic vectors for a periodic boundary condition network, with x-dominant vector first,
        y-dominant vector second.

        Returns
        -------
        PV : 2 x 2 float array
            periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
        """
        if self.PV is not None:
            return self.PV
        else:
            PV = self.calc_PV(attribute=attribute)
            if attribute:
                self.PV = PV
            return PV

    def get_lattice_vectors(self, attribute=False):
        PV = self.get_PV(attribute=attribute)
        return PV

    def get_bz(self, attribute=False):
        """Compute the brillouin zone vertices for a periodic boundary condition network

        Returns
        -------
        bz : n x 2 float array
            brillouin zone vertices for the current network (periodic)
        """
        if self.bz is not None:
            return self.bz
        else:
            bz = self.calc_bz(attribute=attribute)
            return bz

    def get_boundary(self, attribute=False, check=False):
        if self.boundary is not None:
            boundary = self.boundary
        else:
            boundary = self.calc_boundary(attribute=attribute, check=check)

        return boundary

    def get_boundarynn(self, check=False):
        boundary = self.get_boundary()
        boundarynn = latfns.calc_nn_of_sites(self, boundary)
        boundarynn = np.setdiff1d(boundarynn, boundary)
        return boundarynn

    def get_boundary_linesegs(self, attribute=False):
        """Return a set of line segments defining each contiguous boundary of the sample

        Returns
        -------
        blinesegs : # points on voundary x 4 float array or tuple of (# points on voundary x 4) float arrays
            The positions of the particles on the edge of the boundary, in counterclockwise order if open bcs.
            If there are periodic bcs in one dimension, gives contiguous line segments for each boundary in a tuple of
            two arrays.
        """
        boundary = self.get_boundary(attribute=attribute)

        if isinstance(boundary, tuple):
            # sample has multiple boundaries. For each, create a set of contiguous linesegs
            bsegs = []
            for barray in boundary:
                bsegs.append(np.array([[self.xy[barray[ii], 0], self.xy[barray[ii], 1],
                                        self.xy[barray[(ii + 1) % len(barray)], 0],
                                        self.xy[barray[(ii + 1) % len(barray)], 1]]
                                       for ii in xrange(len(barray))]))
            # Stack the elements of the list bsegs
            blinesegs = tuple(bsegs)
        else:
            # sample has only one boundary, create a single set of contiguous linesegs
            blinesegs = np.array([[self.xy[boundary[ii], 0], self.xy[boundary[ii], 1],
                                   self.xy[boundary[(ii + 1) % len(boundary)], 0],
                                   self.xy[boundary[(ii + 1) % len(boundary)], 1]]
                                  for ii in xrange(len(boundary))])

        # Check it
        # plt.clf()
        # print 'lattice_class: blinesegs = ', blinesegs
        # for line in blinesegs:
        #     plt.plot([line[0], line[2]], [line[1], line[3]], '.-')
        #
        # plt.show()
        # sys.exit()
        return blinesegs

    def get_bL(self):
        """Compute and return the bond length list, bL"""
        bL = self.calc_bL()
        return bL

    def assign_PVs(self):
        """For each periodic bond, index the periodic vector(s) taking a neighbor to its image. This gives
        PVdict, a dict which gives where the values give how many times a periodic vector (PV[0] or PV[1])
        is applied for each periodic bond.

        Returns
        -------
        PVdict : dict
            PVdict has key,val pairs of [i,j], [amount in PV[0], amound in PV[1]]
        """
        # For each periodic bond, find the periodic vector taking a neighbor to its image
        PV = le.PVxydict2PV(self.PVxydict)
        raise RuntimeError("didn't finish writing this but should be straightforward")
        return PVdict

    def bond_length_histogram(self, **kwargs):
        """
        Keyword arguments
        -----------------
        fig : figure instance or None
        ax : axis instance or None
        outdir : str or None
        check : bool
        """
        print 'creating bond length histogram (in lattice_class.py)'
        lestructure.bond_length_histogram(self.xy, self.NL, self.KL, self.BL, PVx=self.PVx, PVy=self.PVy, **kwargs)

    def number_variance(self, **kwargs):  # outdir=None,check=False):
        """Computes number variance = (<N^2> - <N>^2) / <N> as a function of radius of circles sampling the pointset.
        """
        radV, varN, a_regr = lestructure.calc_number_variance(self.xy, self.lp['LL'], **kwargs)
        return np.dstack((radV, varN))[0], a_regr

    def calc_polygons(self, attribute=False, check=False):
        polygons = le.extract_polygons_lattice(self.xy, self.BL, self.NL, self.KL, PVx=self.PVx, PVy=self.PVy,
                                               viewmethod=False, check=check)
        if attribute:
            self.polygons = polygons
        return polygons

    def calc_PVxyij(self, attribute=False):
        """

        Parameters
        ----------
        attribute : bool
            store the computed PVxij and PVyij as attributes of current Lattice instance (self)

        Returns
        -------
        PVxij : NP x NP float array (optional, for periodic lattices)
            ijth element of PVxij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
            particle i (at xy[i])
        PVyij : NP x NP float array (optional, for periodic lattices)
            ijth element of PVyij is the x-component of the vector taking particle j (at xy[j]) to its image as seen by
            particle i (at xy[i])
        """
        PVxij, PVyij = latfns.calc_PVxyij(self)
        if attribute:
            self.PVxij = PVxij
            self.PVyij = PVyij
        return PVxij, PVyij

    def calc_nljnnn(self, attribute=False):
        """Compute neighbor and connectivity periodicity for i-NN and NN-NNN bonds

        Parameters
        ----------
        attribute : bool
            ascribe the computed nljnnn, kljnnn, and klknnn to current lattice instance (self)

        Returns
        -------
        nljnnn : #pts x max(#NNN) int array
            nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
            the next nearest neighbor of i through the particle nljnnn[i, j]
        kljnnn : #pts x max(#NNN) int array
            bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
            (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
        klknnn : #pts x max(#NNN) int array
            bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
            (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
        """
        nljnnn, kljnnn, klknnn = lfnsnnn.calc_nljnnn(self)
        if attribute:
            self.nljnnn = nljnnn
            self.kljnnn = kljnnn
            self.klknnn = klknnn
        return nljnnn, kljnnn, klknnn

    def calc_pvnnn(self, attribute=False):
        pvxnnn, pvynnn = lfnsnnn.calc_pvnnn(self, attribute=attribute)
        return pvxnnn, pvynnn

    def calc_PV(self, attribute=False):
        """Get periodic translation vectors for this lattice.

        Parameters
        ----------
        attribute

        Returns
        -------
        PV : 2 x 2 float array
            periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
        """
        PV = latfns.calc_PV(self)
        if attribute:
            self.PV = PV
        return PV

    def calc_bz(self, attribute=False):
        """Get vertices of the brillouin zone for this lattice

        Parameters
        ----------
        attribute : bool
            attribute brillouin zone to self

        Returns
        -------
        bz : n x 2 float array
            brillouin zone vertices for the current network (periodic)
        """
        bz = latfns.calc_bz(self)
        if attribute:
            self.bz = bz
        return bz

    def calc_gxy_gr(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : outdir=None, dr=0.1, eps=1e-7, maxgr=0.04, check=False
        
        Returns
        ----------
        gxy_info : xc, yc, gxy
        gr_info : rc, grV
        """
        xgrid, ygrid, gxy_grid, rcenters, grV = lestructure.calc_gxy_gr(self.xy, self.lp['BBox'], **kwargs)
        return (xgrid, ygrid, gxy_grid), np.dstack((rcenters, grV))[0]

    def calc_fancy_gr(self, **kwargs):
        return lestructure.calc_fancy_gr(self.xy, self.lp['BBox'], **kwargs)

    def calc_boundary(self, attribute=False, check=False):
        """Returns the indices of the particles living on the boundary, in counterclockwise order. The first particle
        is not repeated at the end of the boundary array.

        Returns
        -------
        boundary : # boundary vertices x 1 int array or tuple of (# boundary vertices on a boundary) x 1 int arrays if
        mutiple boundaries
           The indices of the particles that live on the boundary
        """
        if check is None:
            if 'check' in self.lp:
                check = self.lp['check']
            else:
                check = False

        if self.lp['periodic_strip']:
            print 'lattice_class.Lattice(): periodic_strip -->', self.lp['periodic_strip']
            # Special case: if the entire strip is a boundary, then get
            boundary = le.extract_1d_boundaries(self.xy, self.NL, self.KL, self.BL, self.PVx, self.PVy, check=check)
            # !!!
        elif self.lp['periodicBC']:
            boundary = None
        elif 'annulus' in self.lp['LatticeTop'] or self.lp['shape'] == 'annulus':
            print 'here'
            outer_boundary = le.extract_boundary(self.xy, self.NL, self.KL, self.BL, check=check)
            inner_boundary = le.extract_inner_boundary(self.xy, self.NL, self.KL, self.BL, check=check)
            boundary = (outer_boundary, inner_boundary)
        else:
            print 'lattice_class: here! self.xy = ', self.xy
            boundary = le.extract_boundary(self.xy, self.NL, self.KL, self.BL, check=check)

        if attribute:
            self.boundary = boundary
        return boundary

    def calc_bL(self):
        """Calculate the bond length list, bL"""
        BM = le.NL2BM(self.xy, self.NL, self.KL, PVx=self.PVx, PVy=self.PVy)
        print 'lattice_class.Lattice(): BM = ', BM
        bL = le.BM2bL(self.NL, BM, self.BL)
        return bL

    def pointset_fft(self, **kwargs):
        return lestructure.pointset_fft(self.xy, **kwargs)

    def pointset(self, **kwargs):
        return lestructure.pointset(self.xy, **kwargs)

    def structure_factor(self, **kwargs):
        """
        Returns
        --------
        Skxy : tuple of 3 float arrays
        Skr : (NX*NY) x 2 float array
        """
        kx, ky, Skmesh, kr, Skr = lestructure.calc_structure_factor(self.xy, self.lp['LL'], **kwargs)
        return (kx, ky, Skmesh), np.dstack((kr, Skr))[0]

    def plot_NNNangle_hist(self, outdir=None, title=r'$2 \theta_{nml}$, next-nearest neighbor bond angle distribution',
                           fname='figNNNangles', FSFS=12, show=True):
        """"""
        if outdir is not None and outdir != 'none':
            dio.ensure_dir(dio.prepdir(outdir))

        NLNNN, KLNNN, NLNNNangles = self.get_NNN_info()

        # print 'NLNNNangles = ', NLNNNangles
        thetaH = np.mod(2. * NLNNNangles, 2. * np.pi)
        # print 'thetaH= ', thetaH
        thetaH[thetaH > np.pi] = - (2. * np.pi - thetaH[thetaH > np.pi])
        # print 'thetaH= ', thetaH
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        gs = gridspec.GridSpec(7, 7)
        ax[0].set_position(gs[1:6, 0:2].get_position(fig))
        ax[1].set_position(gs[1:6, 3:].get_position(fig))
        nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), None, '', NL=self.NL, KL=self.KL,
                           PVx=self.PVx, PVy=self.PVy, axcb=None,
                           colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=True, ax=ax[0])
        ax[1].hist(thetaH[np.where(KLNNN > 0)].ravel() / np.pi, bins=100)
        ax[1].set_title(title, fontsize=FSFS)
        ax[1].set_ylabel('Occurence', fontsize=FSFS)
        ax[1].set_xlabel(r'$2 \theta_{nml}/\pi$', fontsize=FSFS)
        ax[1].set_xlim([-1, 1])

        if outdir is not None:
            # pickle.dump(fig, file(outdir+fname+'.pkl','w'))
            plt.savefig(outdir + fname + '.png')

        if show:
            print 'Displaying figure...'
            plt.show()
        else:
            return fig, ax

    def plot_lat_colorbonds(self, bondcolors, fig=None, ax=None, meshfn='./', exten='.pdf', save=True, close=True,
                            axis_off=True, title='auto', includeNNN=False, climv=None, **kwargs):
        """Plot (and save if desired) an image of the lattice with connectivity, with bonds colored by the values of
        the array bondcolors.
        If includeNNN is True, also plots NNN vectors in blue/red.
        If save is False, displays the lattice if show is True. Otherwise, adds lattice to axis.

        Parameters
        ----------
        bondcolors : #bonds x 1 float array
        fig : matplotlib figure instance
        ax : matplotlib axis instance
        meshfn : str or None
        exten : str
        save : bool
        close : bool
        axis_off : bool
        title : str
        includeNNN : bool
        climv : None, float, or tuple
            the color limits for the bonds
        **kwargs : keyword arguments for nvis.movie_plot_2D()

        Returns
        -------
        [ax, axcb]
        """
        # Register cmap
        if 'BlueBlackRed' not in plt.colormaps():
            plt.register_cmap(name='BlueBlackRed', cmap=cmaps.BlueBlackRed)

        # print('lattice_class: Plotting as black and white...')
        if save:
            if meshfn != './' and meshfn != 'none' and meshfn is not None:
                dio.ensure_dir(dio.prepdir(meshfn))
            else:
                meshfn = self.lp['meshfn']
            if includeNNN:
                exten = '_NNN' + exten
            fname = meshfn + '/' + meshfn.split('/')[-1] + '_BW' + exten
        else:
            fname = 'none'

        if title == 'auto':
            title = self.lp['lattice_exten'] + ' ' + str(self.lp['NH']) + ' x ' + str(self.lp['NV'])
        # xlimv = max(max(self.lp['NH'] * 0.5 + 5, self.lp['NV'] * 0.5 + 5),
        # xlimv = max((np.max(self.xy[:, 0]) * 1.05, np.max(self.xy[:, 1]) * 1.05))
        # ylimv = xlimv
        if climv is None:
            climv = 0.1
        if self.BL.size > 0:
            if includeNNN:
                self.get_NLNNN_and_KLNNN(attribute=True)
                self.get_nljnnn(attribute=True)
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, bondcolors * np.ones_like(self.BL[:, 0]),
                                                fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, NLNNN=self.NLNNN, KLNNN=self.KLNNN,
                                                PVx=self.PVx, PVy=self.PVy,
                                                nljnnn=self.nljnnn, kljnnn=self.kljnnn, klknnn=self.klknnn,
                                                climv=climv,
                                                axcb=None, colorz=False, colormap='bbr0', bgcolor='#FFFFFF',
                                                axis_off=axis_off, **kwargs)
            else:
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, bondcolors * np.ones_like(self.BL[:, 0]),
                                                fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, PVx=self.PVx, PVy=self.PVy, climv=climv,
                                                axcb=None, colorz=False, colormap='bbr0', bgcolor='#FFFFFF',
                                                axis_off=axis_off, **kwargs)
        else:
            raise RuntimeWarning('Could not save BW plot since BL is empty (no bonds)!')

        return [ax, axcb]

    def plot_lat_dash_certain_bonds(self, dashbonds, fig=None, ax=None, meshfn='./', exten='.pdf', save=True,
                                    close=True, axis_off=True, title='auto', includeNNN=False, climv=None,
                                    bondcolor=lecmaps.black(), **kwargs):
        """Plot (and save if desired) an image of the lattice with connectivity, with certain designated bonds dashed
        instead of solid. If includeNNN is True, also plots NNN vectors in blue/red.
        If save is False, displays the lattice if show is True. Otherwise, adds lattice to axis.

        Parameters
        ----------
        dashbonds : #bonds x 1 bool array
            the bonds of
        fig : matplotlib figure instance
        ax : matplotlib axis instance
        meshfn : str or None
        exten : str
        save : bool
        close : bool
        axis_off : bool
        title : str
        includeNNN : bool
        climv : None, float, or tuple
            the color limits for the bonds
        **kwargs : keyword arguments for nvis.movie_plot_2D()

        Returns
        -------
        [ax, axcb]
        """
        # Register cmap
        if 'BlueBlackRed' not in plt.colormaps():
            plt.register_cmap(name='BlueBlackRed', cmap=cmaps.BlueBlackRed)

        # print('lattice_class: Plotting as black and white...')
        if save:
            if meshfn != './' and meshfn != 'none' and meshfn is not None:
                dio.ensure_dir(dio.prepdir(meshfn))
            else:
                meshfn = self.lp['meshfn']
            if includeNNN:
                exten = '_NNN' + exten
            fname = meshfn + '/' + meshfn.split('/')[-1] + '_BW' + exten
        else:
            fname = 'none'

        if title == 'auto':
            title = self.lp['lattice_exten'] + ' ' + str(self.lp['NH']) + ' x ' + str(self.lp['NV'])
        # xlimv = max(max(self.lp['NH'] * 0.5 + 5, self.lp['NV'] * 0.5 + 5),
        # xlimv = max((np.max(self.xy[:, 0]) * 1.05, np.max(self.xy[:, 1]) * 1.05))
        # ylimv = xlimv
        if climv is None:
            climv = 0.1
        if self.BL.size > 0:
            bondcolors = np.ones_like(self.BL[:, 0])
            bondcolors[dashbonds] = 0
            cmap = lecmaps.colormap_from_hex(bondcolor)
            if includeNNN:
                self.get_NLNNN_and_KLNNN(attribute=True)
                self.get_nljnnn(attribute=True)
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, bondcolors,
                                                fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, NLNNN=self.NLNNN, KLNNN=self.KLNNN,
                                                PVx=self.PVx, PVy=self.PVy,
                                                nljnnn=self.nljnnn, kljnnn=self.kljnnn, klknnn=self.klknnn,
                                                climv=climv,
                                                axcb=None, colorz=False, colormap=cmap, bgcolor='#FFFFFF',
                                                axis_off=axis_off, **kwargs)
            else:
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, bondcolors,
                                                fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, PVx=self.PVx, PVy=self.PVy, climv=climv,
                                                axcb=None, colorz=False, colormap=cmap, bgcolor='#FFFFFF',
                                                axis_off=axis_off, **kwargs)
        else:
            raise RuntimeWarning('Could not save dashed plot since BL is empty (no bonds)!')

        return [ax, axcb]

    def plot_BW_lat(self, fig=None, ax=None, meshfn='./', exten='.pdf', save=True, close=True, axis_off=True,
                    title='auto', includeNNN=False, periodic_linestyles='dashed', periodic_bondcolor=None,
                    ptsize=10, ptcolor=None, **kwargs):
        """Plot (and save if desired) a black and white image of the lattice with connectivity.
        If includeNNN is True, also plots NNN vectors in blue/red.
        If save is False, displays the lattice if show is True. Otherwise, adds lattice to axis.

        Parameters
        ----------
        fig : matplotlib figure instance
        ax : matplotlib axis instance
        meshfn : str or None
        exten : str
        save : bool
        close : bool
        axis_off : bool
        title : str
        includeNNN : bool
        **kwargs : keyword arguments for nvis.movie_plot_2D()

        Returns
        -------
        [ax, axcb]
        """
        # Register cmap
        if 'BlueBlackRed' not in plt.colormaps():
            plt.register_cmap(name='BlueBlackRed', cmap=cmaps.BlueBlackRed)

        # print('lattice_class: Plotting as black and white...')
        if save:
            if meshfn != './' and meshfn != 'none' and meshfn is not None:
                dio.ensure_dir(dio.prepdir(meshfn))
            else:
                meshfn = self.lp['meshfn']
            if includeNNN:
                exten = '_NNN' + exten
            fname = meshfn + '/' + meshfn.split('/')[-1] + '_BW' + exten
        else:
            fname = 'none'

        if title == 'auto':
            title = self.lp['lattice_exten'] + ' ' + str(self.lp['NH']) + ' x ' + str(self.lp['NV'])
        # xlimv = max(max(self.lp['NH'] * 0.5 + 5, self.lp['NV'] * 0.5 + 5),
        # xlimv = max((np.max(self.xy[:, 0]) * 1.05, np.max(self.xy[:, 1]) * 1.05))
        # ylimv = xlimv
        climv = 0.1
        if self.BL.size > 0:
            if includeNNN:
                self.get_NLNNN_and_KLNNN(attribute=True)
                self.get_nljnnn(attribute=True)
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, NLNNN=self.NLNNN, KLNNN=self.KLNNN,
                                                PVx=self.PVx, PVy=self.PVy,
                                                nljnnn=self.nljnnn, kljnnn=self.kljnnn, klknnn=self.klknnn,
                                                climv=climv,
                                                axcb=None, colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF',
                                                axis_off=axis_off, periodic_linestyles=periodic_linestyles,
                                                periodic_bondcolor=periodic_bondcolor,
                                                ptsize=ptsize, ptcolor=ptcolor,
                                                **kwargs)
            else:
                [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), fname, title, fig=fig, ax=ax,
                                                NL=self.NL, KL=self.KL, PVx=self.PVx, PVy=self.PVy, climv=climv,
                                                axcb=None, colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF',
                                                axis_off=axis_off, periodic_linestyles=periodic_linestyles,
                                                periodic_bondcolor=periodic_bondcolor,
                                                ptsize=ptsize, ptcolor=ptcolor, **kwargs)
        else:
            raise RuntimeWarning('Could not save BW plot since BL is empty (no bonds)!')

        return [ax, axcb]

    def plot_WB_lat(self, fig=None, ax=None, meshfn='./', exten='.pdf', save=True, close=True, axis_off=False,
                    title='auto', **kwargs):
        """Plot (and save if desired) a black and white image of the lattice with connectivity.
        If save is False, displays the lattice if show is True. Otherwise, adds lattice to axis.

        Parameters
        ----------
        kwargs : keyword arguments for nvis.movie_plot_2D()
        """
        # Register cmap
        if 'BlueBlackRed' not in plt.colormaps():
            plt.register_cmap(name='BlueBlackRed', cmap=cmaps.BlueBlackRed)

        print('Plotting as black and white...')
        if save:
            if meshfn != './' and meshfn != 'none':
                dio.ensure_dir(dio.prepdir(meshfn))
            fname = meshfn + '/' + meshfn.split('/')[-1] + '_WB' + exten
        else:
            fname = 'none'

        if title == 'auto':
            title = self.lp['lattice_exten'] + ' ' + str(self.lp['NH']) + ' x ' + str(self.lp['NV'])
        # xlimv = max(max(self.lp['NH'] * 0.5 + 5, self.lp['NV'] * 0.5 + 5),
        # xlimv = max((np.max(self.xy[:, 0]) * 1.05, np.max(self.xy[:, 1]) * 1.05))
        # ylimv = xlimv
        climv = 0.1
        if self.BL.size > 0:
            # if save:
            [ax, axcb] = nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), fname, title, fig=fig, ax=ax,
                                            NL=self.NL, KL=self.KL, PVx=self.PVx, PVy=self.PVy, climv=climv,
                                            axcb=None, colorz=False, colormap='seismic', bgcolor='#000000',
                                            axis_off=False, **kwargs)
            # else:
            #     le.display_lattice_2D(self.xy, self.BL, title=title, NL=self.NL, KL=self.KL,
            #                           PVx=self.PVx, PVy=self.PVy, xlimv=xlimv, ylimv=ylimv, climv=climv,
            #                           colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=axis_off,
            #                           close=close)
        else:
            raise RuntimeWarning('Could not save BW plot since BL is empty (no bonds)!')

        return [ax, axcb]

    def plot_boundary(self, boundary=None, attribute=False, show=True, save=False, outdir=None, outname=None,
                      b0color=None, b1color=None):
        """Plot the boundary(ies) of the sample overlaid on a black and white network"""
        axx = self.plot_BW_lat(save=False, close=False, axis_off=True, title='network boundary')
        ax = axx[0]
        if boundary is None:
            bndry = self.get_boundary(attribute=attribute)
        else:
            bndry = boundary

        # Obtain default colors
        if b0color is None:
            b0color = cmaps.green()

        if isinstance(bndry, tuple):
            # Obtain default color for the second boundary
            if b1color is None:
                b1color = cmaps.violet()

            ax.plot(self.xy[bndry[0], 0], self.xy[bndry[0], 1], 'o', color=b0color)
            ax.plot(self.xy[bndry[1], 0], self.xy[bndry[1], 1], 'o', color=b1color)
        else:
            ax.plot(self.xy[bndry, 0], self.xy[bndry, 1], 'o', color=b0color)
        if outdir is None:
            outdir = dio.prepdir(self.get_meshfn())
        if outname is None:
            outname = dio.prepdir(self.get_meshfn()).split('/')[-1] + '_boundary.png'
        if save:
            plt.savefig(outdir + outname)
        if show:
            plt.show()

    def get_gxy_gr(self, attribute=False):
        if self.gxy is not None and self.gr is not None:
            gxy = self.gxy
            gr = self.gr
        else:
            gxy, gr = self.calc_gxy_gr()

            if attribute:
                self.gxy = gxy
                self.gr = gr

        return gxy, gr

    def plot_numbered(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : keyword arguments for self.plot_BW_lat()
        """
        [ax, axcb] = self.plot_BW_lat(meshfn=None, save=False, close=True, **kwargs)
        for ii in range(len(self.xy)):
            ax.text(self.xy[ii, 0] + 0.3, self.xy[ii, 1], str(ii))

        return [ax, axcb]

    def summarize_structure(self):
        """If customization in the plots characterizing structure are desired,
        first use methods of the lattice passed to this method. As an example, to use
        custom arguments in the gxy_gr calculation, run:
        lattice.calc_gxy_gr( custom arguments here... )
        lattice.summarize_structure()
        """
        plt.close('all')
        cmaps.ensure_cmaps()
        if 'meshfn' in self.lp:
            outdir = dio.prepdir(self.lp['meshfn'])
            dio.ensure_dir(outdir)
            fname = outdir + 'structure_summary.png'  # self.lp['meshfn'].split('/')[-1] + '.pdf'
        else:
            fname = './structure_summary.png'

        # Set up figure
        Wfig = 180
        x0s = 10
        y0s = 10
        ws = Wfig * 0.22
        hs = 3. / 4. * ws
        # space between figures
        vspace = 10
        # space above top figure
        tspace = 10
        wB = ws * 0.8
        hB = hs * 0.8
        wss = wB * .3
        hss = wss * 3. / 4.
        # label space
        lbs = ws * 0.15
        Hfig = y0s + ws * 2 + vspace + tspace

        # create figure
        fig = sps.figure_in_mm(Wfig, Hfig)
        label_params = dict(size=12, fontweight='normal')

        # a: scatter, b: bond hist, c: gxy, d: gr, dINSET: grinset e: variance, f: Skxy, g: Skr
        axes = [
            sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
            for x0, y0, width, height, part in (
                [x0s, y0s + hs + vspace, wB, wB, 'a'],  # snippet
                [x0s + ws, y0s + hs + vspace + lbs, wB, hB, 'b'],  # bond histogram
                [x0s + ws + ws, y0s + hs + vspace + lbs, hB, hB, 'c'],  # g(x,y)
                [x0s + ws + ws + ws, y0s + hs + vspace + lbs, wB, hB, 'd'],  # g(r)
                [x0s + ws + ws + ws + wss * 2.5, y0s + hs + vspace + hss * 3.0, wss, hss, ''],  # g(r) inset
                [x0s, y0s, wB, hB, 'e'],  # nvariance
                [x0s + ws + ws, y0s, hB, hB, 'f'],  # S(kx,ky)
                [x0s + ws + ws + ws, y0s, wB, hB, 'g'],  # S(k)
            )
            ]

        try:
            gxyGrid = self.gxy
            grVec = self.gr
        except:
            gxyGrid, grVec = self.calc_gxy_gr()  # outdir=None, dr=0.1, eps=1e-7, check=False)

        try:
            (varNvec, a_regr) = self.nvariance
        except:
            (varNvec, a_regr) = self.number_variance()  # outdir=None, check=False)

        try:
            SkxyGrid, SkrVec = self.Skxy, self.Skr
        except:
            SkxyGrid, SkrVec = self.structure_factor()

        # Plot snippet of lattice
        keep = np.where(np.logical_and(self.xy[:, 0] < 15, self.xy[:, 1] < 15))[0]
        xysnip, NLsnip, KLsnip, BLsnip = le.remove_pts(keep, self.xy, self.BL)
        xlimv = 10
        ylimv = xlimv
        if BLsnip.size > 0:
            '''Plot snippet of lattice'''
            if len(keep) == len(self.xy):
                nvis.movie_plot_2D(self.xy, self.BL, 0 * (self.BL[:, 0]), fname='none', title='', NL=self.NL,
                                   KL=self.KL,
                                   PVx=self.PVx, PVy=self.PVy, ax=axes[0], fig=fig, axcb='none', xlimv=xlimv,
                                   ylimv=ylimv,
                                   colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=True)
            else:
                nvis.movie_plot_2D(xysnip, np.abs(BLsnip), 0 * (BLsnip[:, 0]), fname='none', title='', NL=NLsnip,
                                   KL=KLsnip,
                                   PVx=None, PVy=None, ax=axes[0], fig=fig, axcb='none', xlimv=xlimv, ylimv=ylimv,
                                   colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=True)

        ind = 1

        # Histogram bond list
        self.bond_length_histogram(fig=fig, ax=axes[ind], outdir=None, check=False, savetxt=False)
        ind += 1

        # Plot gxy as heatmap
        # axes[ind].pcolor( gxyGrid[0], gxyGrid[1], gxyGrid[2], cmap='viridis', vmin=0.0) #vmax=4.0)
        imgxy = axes[ind].pcolormesh(gxyGrid[0], gxyGrid[1], gxyGrid[2], cmap='viridis', vmin=0.0)
        divider = make_axes_locatable(axes[ind])
        # Append axes to the right of ax3, with 20% width of ax3
        cdiv = divider.append_axes("right", size="05%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks` and the format of the ticklabels with kwarg `format`
        cb = plt.colorbar(imgxy, cax=cdiv, format="%.2f")
        # cb.set_label(r'$ g(x,y)$')
        axes[ind].axis('image')
        titlestr = r'$g(\mathbf{x})$'  # with system size=({0:0.3f}'.format(2.* np.min(np.abs(self.lp['BBox'])))+')'
        # print 'title = ', titlestr
        axes[ind].set_title(titlestr)
        axes[ind].set_xlim(-5, 5)
        axes[ind].set_ylim(-5, 5)
        ind += 1

        # Plot g(r)
        axes[ind].plot(grVec[:, 0], grVec[:, 1], 'r.-')
        axes[ind].set_xlim(0, 5)
        axes[ind].set_xlabel(r'$r$')
        axes[ind].set_ylabel(r'$g(r)$')
        ind += 1

        # Plot g(r) inset
        axes[ind].plot(grVec[:, 0], grVec[:, 1], 'r-')
        ind += 1

        # Plot number variance
        axes[ind].plot(varNvec[:, 0], varNvec[:, 1], 'r.-')
        axes[ind].set_xlabel(r'$R$')
        axes[ind].set_ylabel(r'$\sigma^2(R)$')
        ind += 1

        # Plot S(kx,ky) as heatmap
        imgSk = axes[ind].pcolor(SkxyGrid[0], SkxyGrid[1], SkxyGrid[2], cmap='viridis', vmin=0.0)
        divider = make_axes_locatable(axes[ind])
        cdiv = divider.append_axes("right", size="05%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks` and the format of the ticklabels with kwarg `format`
        cb = plt.colorbar(imgSk, cax=cdiv, format="%.0f")
        # cb.set_label(r'$ g(x,y)$')
        axes[ind].axis('image')
        titlestr = r'$S(\mathbf{k})$'  # with system size='+'({0:0.3f}'.format(2.* np.min(np.abs(self.lp['BBox'])))+')'
        # print 'title = ', titlestr
        axes[ind].set_title(titlestr)
        ind += 1

        # Plot S(k)
        # print 'SkrVec = ', SkrVec
        SkrVec = SkrVec[~np.isnan(SkrVec[:, 1]), :]
        axes[ind].plot(SkrVec[:, 0], SkrVec[:, 1], 'r.-')
        axes[ind].axis('tight')
        axes[ind].set_xlim(0, np.max(SkrVec[:, 0]))
        axes[ind].set_ylim(0, 5)
        axes[ind].set_xlabel(r'$k$')
        axes[ind].set_ylabel(r'$S(k)$')

        plt.savefig(fname)

        if self.lp['check']:
            plt.show()


if __name__ == '__main__':
    '''Perform an example of using the lattice class'''

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')

    # Task
    parser.add_argument('-NNNanglehist', '--NNNanglehist', help='Make a sequence of NNN angle histograms',
                        action='store_true')
    parser.add_argument('-NNNvary_param', '--NNNvary_param', help='Parameter to vary in NNN angle visualization',
                        type=str, default='delta')
    parser.add_argument('-loadlat', '--loadlat', help='Demonstrate loading a lattice from meshfn string',
                        action='store_true')
    parser.add_argument('-view_lattice', '--view_lattice', help='Preview a saved lattice by loading it',
                        action='store_true')
    parser.add_argument('-view_numbered', '--view_numbered', help='Preview a saved lattice and label particles',
                        action='store_true')
    parser.add_argument('-summarize', '--summarize', help='Summarize the structure of a lattice',
                        action='store_true')
    parser.add_argument('-redo_polygons', '--redo_polygons', help='Redo calculation of polygons',
                        action='store_true')
    parser.add_argument('-review_polygons', '--review_polygons', help='Review the polygons (loaded or calculated)',
                        action='store_true')
    parser.add_argument('-redo_gxy', '--redo_gxy', help='Redo calculation of g(x,y)',
                        action='store_true')
    parser.add_argument('-maxgxy', '--maxgxy', help='Max plotted value for g(x,y)', type=float, default=0.04)
    parser.add_argument('-load_save_gxy', '--load_save_gxy', help='Load and plot g(x,y)', action='store_true')
    parser.add_argument('-pointset', '--pointset', help='Plot the lattice point set', action='store_true')
    parser.add_argument('-boundaries_strip', '--boundaries_strip',
                        help='Show the two boundaries for a periodic strip sample', action='store_true')
    parser.add_argument('-intnn_info', '--intnn_info',
                        help='Obtain the intnn-th-nearest-neighbor information', action='store_true')
    parser.add_argument('-intnn', '--intnn',
                        help='use intnn for getting intnn-th-nearest-neighbor information', type=int, default=3)

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

    if args.NNNanglehist:
        cmaps.register_colormaps()
        outroot = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/NNNangles/'
        outdir = outroot + lp['LatticeTop'] + '/'
        if lp['LatticeTop'] == 'hexagonal':
            dio.ensure_dir(outdir)
            if args.NNNvary_param == 'delta':
                for delta in np.arange(0.66666, 1.23, 0.02):
                    lp['delta'] = delta * np.pi
                    lat = Lattice(lp)
                    lat.build()
                    # lat.plot_BW_lat(meshfn= outroot + lp['LatticeTop'], exten='_delta'+sf.float2pstr(delta)+'pi.png')
                    fname = 'N' + '{0:03d}'.format(max(lp['NH'], lp['NP_load'])) + \
                            'NNNangles_phi' + sf.float2pstr(lp['phi'] / np.pi) + 'pi' + sf.float2pstr(
                        delta / np.pi) + 'pi.png'
                    lat.plot_NNNangle_hist(outdir=outdir, fname=fname, show=False)
            elif args.NNNvary_param == 'phi':
                for phi in np.arange(0.0, 0.5, 0.0075):
                    lp['phi'] = phi * np.pi
                    lat = Lattice(lp)
                    lat.build()
                    # lat.plot_BW_lat(meshfn= outroot + lp['LatticeTop'], exten='_delta'+sf.float2pstr(delta)+'pi.png')
                    fname = 'N' + '{0:03d}'.format(max(lp['NH'], lp['NP_load'])) + \
                            'NNNangles_phi' + sf.float2pstr(phi) + 'pi' + '_delta' + sf.float2pstr(
                        lp['delta'] / np.pi) + 'pi.png'
                    lat.plot_NNNangle_hist(outdir=outdir, fname=fname, show=False)
        else:
            dio.ensure_dir(outdir)
            lat = Lattice(lp)
            lat.load()
            lat.plot_BW_lat(meshfn=outroot + lp['LatticeTop'], exten='.pdf')
            fname = 'N' + '{0:03d}'.format(max(lp['NH'], lp['NP_load'])) + 'NNNangles_' + lp['LatticeTop'] + '.png'
            lat.plot_NNNangle_hist(outdir=outdir, fname=fname, show=False)
            lat.plot_BW_lat(meshfn=None, includeNNN=True)

    if args.loadlat:
        # Check loading
        meshfn = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/hucentroid/hucentroid_square_periodic_d01_NP000020'
        lat = Lattice()
        lat.load(meshfn=meshfn)
        lat.PVx, lat.PVy = le.PVxydict2PVxPVy(lat.PVxydict, lat.NL, lat.KL)
        BM = le.NL2BM(lat.xy, lat.NL, lat.KL, PVx=lat.PVx, PVy=lat.PVy)
        bL = le.BM2bL(lat.NL, BM, lat.BL)
        plt.hist(bL)
        plt.show()

    if args.view_lattice:
        # pointset
        lat = Lattice(lp)
        lat.load()
        lat.plot_BW_lat(fig=None, ax=None, meshfn=None, exten='.pdf', save=False, close=True, axis_off=False,
                        title='auto')
        plt.show()

    if args.view_numbered:
        lat = Lattice(lp)
        lat.load()
        lat.plot_numbered()
        plt.show()

    if args.summarize:
        lat.summarize_structure()

    if args.redo_gxy:
        lat = Lattice(lp)
        lat.load(load_polygons=False)
        lat.calc_gxy_gr(outdir=dio.prepdir(lp['meshfn']), dr=0.1, eps=1e-7, check=False, maxgxy=args.maxgxy)

    if args.review_polygons:
        '''Plot'''
        lat = Lattice(lp)
        lat.load(load_polygons=True)
        PPC = le.polygons2PPC(lat.xy, lat.polygons, BL=lat.BL, PVxydict=lat.PVxydict)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = PatchCollection(PPC, alpha=0.5)
        colors = 100 * np.random.rand(len(PPC))
        p.set_array(np.array(colors))
        ax.add_collection(p)
        plt.plot(lat.xy[:, 0], lat.xy[:, 1], 'b.')
        xlim = max(abs(lat.xy[:, 0])) + 5
        ylim = max(abs(lat.xy[:, 1])) + 5
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        plt.show()
        plt.clf()

    if args.redo_polygons:
        lat = Lattice(lp)
        lat.load(load_polygons=False)
        lat.save_polygons(attribute=True, check=args.check)

    if args.load_save_gxy:
        lat = Lattice(lp)
        lat.load(load_polygons=False)
        lat.save_gxy_gr()

    if args.nice_plot:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = Lattice(lp)
        lat.load()
        print 'lattice_class: Saving nice BW plot...'
        lat.plot_BW_lat(meshfn=lat.lp['meshfn'], lw=1)
        lat.plot_WB_lat(meshfn=lat.lp['meshfn'], lw=2)

    if args.pointset:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = Lattice(lp)
        lat.load()
        print 'lattice_functions: \nPerforming image of points...'

        lat.pointset(outdir=lat.lp['meshfn'], wsfrac=0.8)

    if args.boundaries_strip:
        """Example usage:
        python lattice_class.py -LT hexagonal -NH 14 -NV 7 -periodic_strip -boundaries_strip -shape square
        """
        if not lp['periodic_strip']:
            raise RuntimeError('must be periodic strip to show strip boundaries')

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = Lattice(lp)
        lat.load()
        print 'lattice_functions: \nDrawing two periodic edges...'
        axx = lat.plot_BW_lat(save=False)
        ax, axcb = axx[0], axx[1]
        bb0, bb1 = lat.get_boundary()
        ax.plot(lat.xy[bb0, 0], lat.xy[bb0, 1], 'ro')
        ax.plot(lat.xy[bb1, 0], lat.xy[bb1, 1], 'go')
        plt.title('Showing two boundaries')
        plt.savefig(dio.prepdir(lat.lp['meshfn']) + 'stripboundaries.png')

    if args.intnn_info:
        """Example usage:
        python lattice_class.py -LT hexagonal -N 3 -periodic -intnn_info -intnn 3
        """
        import lepm.plotting.network_visualization as netvis
        import lepm.plotting.plotting as leplt
        lat = Lattice(lp=lp)
        lat.load()
        NLnns, KLnns, pvxnns, pvynns = lat.get_intnn_info(args.intnn, attribute=True)
        for ii in range(args.intnn):
            print 'lattice_functions: NL = ', NLnns[ii]
            print 'lattice_functions: KL = ', KLnns[ii]
            print 'lattice_functions: pvx  = ', pvxnns[ii]
            print 'lattice_functions: pvy  = ', pvynns[ii]

            # BL = le.NL2BL(NLnns[ii], KLnns[ii])
            # print 'BL = ', BL
            title = str(ii) + 'th-nn connections'
            # Visualize the network manually
            fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=4)
            ax.plot(lat.xy[:, 0], lat.xy[:, 1], 'k.')

            for jj in range(len(lat.xy)):
                for kk in range(len(NLnns[ii][jj])):
                    if abs(KLnns[ii][jj, kk]) > 0:
                        nei = NLnns[ii][jj, kk]
                        if KLnns[ii][jj, kk] > 0:
                            plt.plot([lat.xy[jj, 0], lat.xy[nei, 0]],
                                     [lat.xy[jj, 1], lat.xy[nei, 1]], 'k-')
                            print 'REAL BOND: xy0, xy1 = ', (lat.xy[jj], lat.xy[nei])
                        else:
                            # get periodic info
                            vx, vy = pvxnns[ii][jj, kk], pvynns[ii][jj, kk]
                            plt.plot([lat.xy[jj, 0], lat.xy[nei, 0] + vx],
                                     [lat.xy[jj, 1], lat.xy[nei, 1] + vy], 'k-')
                            print 'lattice_functions: vx, vy = ', (vx, vy)
                            print 'lattice_functions: xy0, xy1 = ', (lat.xy[jj], lat.xy[nei] + np.array([vx, vy]))
                            plt.pause(0.2)

            plt.axis('scaled')
            outfn = dio.prepdir(lat.lp['meshfn']) + 'nnn' + str(ii) + '.pdf'
            print 'lattice_functions: saving to ', outfn
            plt.savefig(outfn)
            plt.show()
            plt.close('all')
            # netvis.movie_plot_2D(lat.xy, BL, bs=BL[:, 0] * 0., fname='none', title=title,
            #                      NL=NLnns[ii], KL=KLnns[ii], BLNNN=[], NLNNN=[], KLNNN=[],
            #                      PVx=pvxnns[ii], PVy=pvynns[ii],
            #                      PVxydict={}, nljnnn=None, kljnnn=None, klknnn=None,
            #                      ax=ax, fig=fig, axcb='auto', cbar_ax=None, cbar_orientation='vertical',
            #                      xlimv='auto', ylimv='auto', climv=0.1, colorz=True, ptcolor=None,
            #                      figsize='auto', colorpoly=False,
            #                      bondcolor='k', colormap='seismic', bgcolor=None, axis_off=False,
            #                      axis_equal=True, text_topleft=None, lw=-1.,
            #                      ptsize=10, negative_NNN_arrows=False, show=True, arrow_alpha=1.0,
            #                      fontsize=8, cax_label='Strain', zorder=0)

