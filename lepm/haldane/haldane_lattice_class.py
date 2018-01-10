import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.haldane.haldane_lattice_functions as hlatfns
import lepm.lattice_functions as lfns
import lepm.stringformat as sf
import lepm.plotting.plotting as leplt
import lepm.plotting.movies as lemov
import lepm.plotting.colormaps as cmaps
import lepm.plotting.haldane_lattice_plotting_functions as hlatpfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmap
import os
import sys
import argparse
import glob
import copy
import h5py
try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm
import lepm.haldane.haldane_infinite_functions as hinffns
import lepm.hdf5io as h5io


'''
Generate Haldane model lattices using the HaldaneLattice class: real NN hoppings, complex NNN hoppings.
t1 : the real NN hopping in the dynamical matrix
t2 : the pure imaginary component of the universally-applied NNN hopping
t2a : the real part of the universally-applied NNN hopping
t2angles : use the geometry of the NNN angle to determine the NNN hopping (real + imaginary components kept)
pureimNNN : specify that only the imaginary part of the NNN hoppings is kept. This should only be used if t2angles is True

For haldane networks with delta-correlated (random) disorder, the pin configuration number is always listed if the
disorder is nonzero, so an example output txt file to save the pinning could be
pin_haldane_pin0p00_pinV0p10_conf0000.txt, even though the pin configuration # is zero. This is different than the gyro
network convention, where pin config #s are only printed if > 1.
Note that in a chern computation, the zero pin configuration tag is not included.
'''


class HaldaneLattice:
    """Create a Haldane network from an instance of the lattice class.
    Attributes of the haldane_lattice are:
        lattice : lattice class instance (has xy, NL, KL, etc). Can be empty so that it can be loaded or built
        t1: N x max#NN float array
            NN hoppings, like hlat.lattice.KL
        t2: a simple float, formed from lp alone, not on its own
            scale factor for NNN hopping or complex part of NNN hopping if t2angles is False
        t2a: a simple float, formed from lp alone, not on its own
            real part of NNN hopping
        pin: N x 1 float array
            pinning frequencies
        matrix : dynamical matrix, or None
        eigvect : eigvenvectors of dynamical matrix, or None if too bulky to keep in RAM
        eigval : eigenvalues of dynamical matrix, or None if too bulky to keep in RAM
        IPM: inverse participation ratio of the modes, or None if too bulky to keep in RAM
        ldos : local density of states
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K

    Attributes of the lattice are:
        xy, BL, NL, KL, LVUC, LV, UC, Pvxydict, LL, BBox, and lp
    lp is a dictionary ('lattice parameters') which can contain:
        LatticeTop, shape, NH, NV, pureimNNN,
        rootdir, meshfn, lattice_exten, phi, delta, theta, eta, x1, x2, x3, z, source, loadlattice_number,
        check, Ndefects, Bvec, dislocation_xy, target_z, make_slit, cutz_method, cutLfrac
        V0_pin_gauss, V0_spring_gauss, dcdisorder

    V0_pin_gauss: float
        stdev of distribution of delta-correlated pin disorder
    V0_pin_gauss: float
        stdev of distribution of delta-correlated spring disorder
    dcdisorder: bool
        Whether there is delta-function correlated disorder in the physics of the network (pin, spring)

    """
    def __init__(self, lattice, lp, t1=None, pin=None, NLNNN=None, KLNNN=None, matrix=None, eigvect=None,
                 eigval=None, ipr=None, prpoly=None, ldos=None, localization=None, edge_localization=None):
        """Create a HaldaneLattice instance.
        NN hopping term is t1 as NP x NP float array
        pin is the on-diagonal pinning strength as NP x NP float array
        NNN hopping is +/- i*t2
        self.t1 is a full NP x max#NN float array, while self.lp['t1'] is float
        self.t2 and self.lp['t2'] are simple floats, not arrays
        self.lp['t2a'] is simple float, not array
        self.lp['theta_twist'] is the twist angle of the x-axis periodic boundary condition in units of pi
        self.lp['phi_twist'] is the twist angle of the y-axis periodic boundary condition in units of pi

        Properties
        ----------
        lattice : lattice class instance (has xy, NL, KL, etc)
        lp : dict
            lattice parameters
        matrix : NP x NP complex array or None
            dynamical matrix
        eigvect : NP x NP complex array or None
            eigvenvectors of dynamical matrix
        eigval : NP x 1 complex or real array or None
            eigenvalues of dynamical matrix
        """
        self.lattice = lattice
        self.lp = lp
        if 'ignore_tris' not in lp:
            self.lp['ignore_tris'] = False

        eps = 1e-7

        # Add lattice properties to haldane_lattice_properties, by convention, but don't overwrite params
        for key in self.lattice.lp:
            if key not in self.lp:
                self.lp[key] = self.lattice.lp[key]

        self.lp['meshfn_exten'] = '_haldane'

        # decide if NNN hoppings are modulated by the opening angle
        if 't2angles' in self.lp:
            if self.lp['t2angles']:
                self.lp['meshfn_exten'] += '_t2angles'
        else:
            self.lp['t2angles'] = False

        # decide if NNN hoppings are purely imaginary, or if they have a real component
        if 'pureimNNN' in self.lp:
            if self.lp['pureimNNN']:
                self.lp['meshfn_exten'] += '_pureimNNN'
        else:
            self.lp['pureimNNN'] = False

        print 'pureim-->', lp['pureimNNN']
        # sys.exit()

        # Form t1 (NP x max#NN array)312
        if t1 == 'auto' or t1 is None:
            if 't1' in self.lp:
                print 'haldane_lattice_class: using t1 from lp...'
                self.t1 = self.lp['t1'] * np.abs(self.lattice.KL)
            else:
                print 'giving t1 the default value of -1s...'
                self.lp['t1'] = -1.0
                self.t1 = -1.0 * np.abs(self.lattice.KL)
        else:
            # Suppled t1 is assumed to be np.shape(self.lattice.KL)
            self.t1 = t1
            if 't1' in self.lp:
                if self.t1 != self.lp['t1'] * np.abs(self.lattice.KL):
                    self.lp['meshfn_exten'] += '_t1spec'
                    self.lp['t1'] = -5000
            else:
                # Check if the values of all elements are identical
                kinds = np.nonzero(self.t1)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in t1. Check if all the same
                    value = self.t1[kinds[0][0], kinds[1][0]]
                    if (t1[kinds] == value).all():
                        self.lp['t1'] = value
                    else:
                        self.lp['t1'] = -5000
                else:
                    self.lp['t1'] = 0.0

        self.lp['meshfn_exten'] += '_t1_' + sf.float2pstr(self.lp['t1'])

        # Form t2 (just a simple float, specified only from lp, not on its own)
        if 't2' in self.lp:
            print 'haldane_lattice_class: using t2 from lp...'
            self.t2 = self.lp['t2']
        else:
            print 'giving t2 the default value of 0.1...'
            # self.t2 is a simple float, not of shape np.shape(self.lattice.KL)
            self.t2 = 0.1
            self.lp['t2'] = self.t2

        self.lp['meshfn_exten'] += '_t2_' + sf.float2pstr(self.lp['t2'])

        if 't2a' in self.lp:
            if abs(self.lp['t2a']) > 1e-7:
                self.lp['meshfn_exten'] += '_t2a_' + sf.float2pstr(self.lp['t2a'])
        else:
            self.lp['t2a'] = 0.0

        self.t2a = self.lp['t2a']

        # Form pin
        if pin is None:
            if 'pin' in self.lp:
                self.pin = self.lp['pin'] * np.ones_like(self.lattice.xy[:, 0])
            else:
                print 'giving pin the default value of zeros...'
                self.pin = np.zeros_like(self.lattice.xy[:, 0])
                self.lp['pin'] = 0.0
        else:
            self.pin = pin
            if 'pin' in self.lp:
                if self.pin != self.lp['pin'] * np.ones_like(self.lattice.xy):
                    self.lp['meshfn_exten'] += '_pinspec'
                    self.lp['pin'] = -5000
            else:
                # Check if the values of all elements are identical
                kinds = np.nonzero(self.pin)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in pin. Check if all the same
                    value = self.pin[kinds[0]]
                    if (pin[kinds] == value).all():
                        self.lp['pin'] = value
                    else:
                        self.lp['pin'] = -5000
                else:
                    self.lp['pin'] = 0.0

        self.lp['meshfn_exten'] += '_pin' + sf.float2pstr(self.lp['pin'])

        if 'ABDelta' in self.lp:
            if self.lp['ABDelta'] > 0:
                self.lp['meshfn_exten'] += '_ABd{0:0.3f}'.format(self.lp['ABDelta']).replace('.', 'p')
        else:
            self.lp['ABDelta'] = 0.
        if 'V0_pin_gauss' in self.lp and 'V0_spring_gauss' in self.lp:
            if self.lp['V0_pin_gauss'] > eps or self.lp['V0_spring_gauss'] > eps:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_pinV' + sf.float2pstr(self.lp['V0_pin_gauss'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_pinconf{0:04d}'.format(self.lp['pinconf'])

                self.lp['meshfn_exten'] += '_sprV' + sf.float2pstr(self.lp['V0_spring_gauss'])
            else:
                self.lp['dcdisorder'] = False
        else:
            self.lp['V0_pin_gauss'] = 0.
            self.lp['V0_spring_gauss'] = 0.
            self.lp['dcdisorder'] = False

        # Note: theta_twist and phi_twist are in units of pi!
        # Append thetatwist and/or phitwist to meshfn_exten
        if 'theta_twist' in lp:
            if abs(lp['theta_twist']) > 1e-15:
                self.lp['meshfn_exten'] += '_thetatw' + sf.float2pstr(lp['theta_twist'], ndigits=3)
        if 'phi_twist' in lp:
            if abs(lp['phi_twist']) > 1e-15:
                self.lp['meshfn_exten'] += '_phitw' + sf.float2pstr(lp['phi_twist'], ndigits=3)

        print 'self.lp[ABDelta] = ', self.lp['ABDelta']
        if self.lp['ABDelta'] > eps or self.lp['V0_pin_gauss'] > eps or self.lp['V0_spring_gauss'] > eps:
            # In order to load the random (V0) or alternating (AB) pinning sites, look for a txt file with the pinnings
            # that also has specifications in its meshfn exten, but IGNORE portion of meshfnexten with t1, t2, and t2a
            # Form abbreviated meshfn exten
            print "self.lp['ABDelta'] > 0 -->", self.lp['ABDelta'] > 0
            print "self.lp['V0_pin_gauss'] -->", self.lp['V0_pin_gauss']
            print "self.lp['V0_spring_gauss'] > 0 -->", self.lp['V0_spring_gauss'] > 0
            pinmfe = self.get_pinmeshfn_exten()

            print 'HaldaneLattice.init: Trying to load offset/disorder to pinning frequencies: '
            print dio.prepdir(self.lp['meshfn']) + 'pin' + pinmfe + '.txt'
            # Attempt to load from file
            try:
                self.load_pinning(meshfn=self.lp['meshfn'], pinmfe=pinmfe)
                print 'Loaded ABDelta and/or dcdisordered pinning frequencies...'
            except IOError:
                if lp['NH'] > 30:
                    print 'HaldaneLattice.__init__: exiting for troubleshoot to not overwrite...'
                    sys.exit()
                print 'hlat.init(): Could not find ' + dio.prepdir(self.lp['meshfn']) + 'pin' + pinmfe + '.txt'
                # sys.exit()
                # Make pin and t1 from scratch
                if self.lp['ABDelta'] > 0:
                    asites, bsites = hlatfns.ascribe_absites(self.lattice)
                    self.pin[asites] += self.lp['ABDelta']
                    self.pin[bsites] -= self.lp['ABDelta']
                if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_spring_gauss'] > 0:
                    self.add_dcdisorder()

                # Save non-standard pin
                if 'save_pinning_to_hdf5' in self.lp:
                    if self.lp['save_pinning_to_hdf5']:
                        force_hdf5pin = True
                    else:
                        force_hdf5pin = False
                else:
                    force_hdf5pin = False
                self.save_pin(force_hdf5=force_hdf5pin)
                # np.savetxt(dio.prepdir(self.lp['meshfn']) + 'pin' + pinmfe + '.txt', self.pin)
                # self.plot_pin()

        self.matrix = matrix
        self.eigvect = eigvect
        self.eigval = eigval
        self.ipr = ipr
        self.prpoly = prpoly
        self.ldos = ldos
        self.localization = localization
        self.edge_localization = edge_localization

        print 'meshfn_exten --> ', self.lp['meshfn_exten']

    def __hash__(self):
        return hash(self.lattice)

    def __eq__(self, other):
        return hasattr(other, 'lattice') and self.lattice == other.lattice

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def load_pinning(self, meshfn=None, pinmfe=None):
        """Load the Omg vector for this instance of GyroLattice"""
        # First try to load from hdf5 file, if it exists
        # If hdf5 file does not exist or contain the pinning for this meshfn_exten, attempt to load from txt file
        if meshfn is None:
            meshfn = self.lp['meshfn']
        if pinmfe is None:
            pinmfe = self.get_pinmeshfn_exten()
        pinning_name = 'pin' + pinmfe
        pinfn = dio.prepdir(meshfn) + 'pin_configs.hdf5'
        if glob.glob(pinfn):
            with h5py.File(dio.prepdir(meshfn) + 'pin_configs.hdf5', "r") as fi:
                inhdf5 = pinning_name in fi.keys()
                if inhdf5:
                    self.pin = fi[pinning_name][:]
                    load_from_txt = False
                else:
                    load_from_txt = True
        else:
            load_from_txt = True

        if load_from_txt:
            print 'could not find pinning config from hdf5, opening pinning configs from txt...'
            try:
                self.pin = np.loadtxt(dio.prepdir(self.lp['meshfn']) + 'pin' + pinmfe + '.txt')
            except:
                # At one point I accidentally saved pins as 'pinpin' + pinmfe...
                self.pin = np.loadtxt(dio.prepdir(self.lp['meshfn']) + 'pinpin' + pinmfe + '.txt')

    def load(self, meshfn='auto', loadDOS=False, load_ipr=False):
        """
        Load a saved lattice into the lattice attribute of the haldane_lattice instance.
        If meshfn is specified, loads that lattice.
        Otherwise, attempts to load lattice based on parameter self.lp['meshfn']. If that is also unavailable,
        loads from lp[rootdir]/networks/self.LatticeTop/self.lp[lattice_exten]_NH_x_NV.
        """
        if meshfn == 'auto':
            fn = self.lattice.lp['meshfn']
        else:
            fnglob = sorted(glob.glob(meshfn))
            is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
            fn = fnglob[is_a_dir[0]]
            print 'fn = ', fn
            if np.size(is_a_dir) > 1:
                print 'Found multiple lattices matching meshfn in lattice.load(). Using the first matching lattice.'
                fn = fn[0]
            self.lattice.lp['meshfn'] = fn

        if len(self.lattice.xy) == 0:
            print 'Lattice is empty lattice instance, loading...'
            self.lattice.load()

        if self.pin is None:
            raise RuntimeError('self.pin should already be loaded from file or created from scratch')
            # if self.lp['V0_pin_gauss'] > 0 or self.lp['ABDelta'] > 0:
            #     self.pin = np.loadtxt(dio.prepdir(fn) + 'pin_mean' + sf.float2pstr(self.lp['pin']) +
            #                           self.lp['meshfn_exten'] + '.txt')
            # else:
            #     self.pin = self.lp['pin'] * np.ones_like(self.lattice.xy[:, 0])

        if self.lp['V0_spring_gauss'] > 0:
            print 'todo: This is not finished'
            sys.exit()

        if loadDOS:
            print 'Loading eigval/vect...'
            self.load_eigval_eigvect(attribute=True)

        if load_ipr:
            print 'Loading ipr...'
            try:
                self.load_ipr(attribute=True)
            except IOError:
                print 'ipr.pkl not found! Calculating and saving ipr for this network...'
                self.calc_ipr(attribute=True)
                self.save_ipr(save_images=True)

    def build(self):
        import lepm.make_lattice
        self.lattice = lepm.make_lattice.build_lattice(self.lattice)

    def get_pinmeshfn_exten(self):
        """Assumes that meshfn_exten has already been built. Get the part of the meshfn associated with """
        if 'ABd' in self.lp['meshfn_exten']:
            abmfe_list = self.lp['meshfn_exten'].split('ABd')
            pinmfe_list = self.lp['meshfn_exten'].split('pin')
            pinmfe = pinmfe_list[0][0:9] + 'pin' + pinmfe_list[1][0:5]
            if 'pinV' in self.lp['meshfn_exten']:
                pinmfe += 'pin' + pinmfe_list[2][0:6]

            pinmfe += 'ABd' + abmfe_list[1][0:5] + '_'
            # print 'pinmfe = ', pinmfe
            # print 'hlat: Is that pin_meshfunction_extension (pinmfe) correct?'
            # sys.exit()
        else:
            # pinV0 must be specified
            pinmfe_list = self.lp['meshfn_exten'].split('pin')
            pinmfe = pinmfe_list[0][0:9] + 'pin' + pinmfe_list[1][0:5] + 'pin' + pinmfe_list[2][0:5]

        # if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_spring_gauss'] > 0:
        if 'pinconf' in self.lp:
            print 'self.lp[pinconf] = ', self.lp['pinconf']
            pinmfe += '_conf{0:04d}'.format(self.lp['pinconf'])
        else:
            self.lp['pinconf'] = 0
            pinmfe += '_conf{0:04d}'.format(self.lp['pinconf'])
        print 'pinmfe = ', pinmfe

        return pinmfe

    def add_dcdisorder(self):
        """Add gaussian noise to pinning or spring energies (delta-correlated disorder)"""
        # Add gaussian noise to pinning energies
        if self.lp['V0_pin_gauss'] > 0:
            self.pin += self.lp['V0_pin_gauss']*np.random.randn(len(self.lattice.xy))
        if self.lp['V0_spring_gauss'] > 0:
            print 'hlat: This is not done correctly here'
            sys.exit()
            self.t1 += self.lp['V0_spring_gauss'] * np.random.randn(np.shape(self.lattice.KL)[0],
                                                                    np.shape(self.lattice.KL)[1])
            sys.exit()

    def infinite_dispersion(self, kx=None, ky=None, save=True, title='Dispersion relation', outdir=None):
        """Compute dispersion relation for infinite sample over a grid of kx ky values

        Parameters
        ----------
        kx : float array or None
            The wavenumber values in x direction to use
        ky : float array or None
            The wavenumber values in y direction to use
        save : bool
            whether to save the results of the dispersion calculation
        title : str
            the title of the plot of the dispersion
        outdir : str or None
            path to the dir where results are saved, if save==True. If None, uses lp['meshfn'] for hlat.lattice

        Returns
        -------
        omegas, kx, ky
        """
        omegas, kx, ky = hinffns.infinite_dispersion(self, kx=kx, ky=ky, save=save, title=title, outdir=outdir)
        return omegas, kx, ky

    def calc_matrix(self, attribute=True, check=False, sparse=False):
        """

        Parameters
        ----------
        attribute : bool
            Attribute calculated matrix to self
        check : bool
            Display intermediate results
        sparse : bool
            build a sparse matrix version of the dynamical matrix

        Returns
        -------
        matrix : N x N complex array
            The dynamical matrix
        """
        print 'HaldaneLattice.calc_matrix: calculating dynamical matrix...'
        # Use t1, t2, and self.pin
        # pureimNNN = 'pureimNNN' in self.lp['meshfn_exten']
        if 'theta_twist' not in self.lp:
            self.lp['theta_twist'] = None
            self.lp['phi_twist'] = None
        NLNNN, KLNNN, matrix = hlatfns.haldane_matrix(self.lattice, self.lp['t2'], t1=self.lp['t1'], t2a=self.lp['t2a'],
                                                      pin=self.pin,
                                                      t2angles=self.lp['t2angles'], pureimNNN=self.lp['pureimNNN'],
                                                      thetatwist=self.lp['theta_twist'], phitwist=self.lp['phi_twist'],
                                                      ignore_tris=self.lp['ignore_tris'],
                                                      sparse=sparse)

        if self.lattice.NLNNN is None:
            self.lattice.NLNNN = NLNNN
        if self.lattice.KLNNN is None:
            self.lattice.KLNNN = KLNNN
        if check:
            le.plot_complex_matrix(matrix, show=True)

        if attribute:
            self.matrix = matrix
        return matrix

    def calc_eigvals(self, matrix=None, sort='real', attribute=True):
        """
        Finds the eigenvalues of dynamical matrix.

        Parameters
        ----------
        matrix : N x M matrix
            matrix to diagonalize

        Returns
        ----------
        eigval_out : 2*N x 1 complex array
            eigenvalues of the matrix, sorted by order of imaginary components
        """
        if matrix is None:
            matrix = self.get_matrix()

        eigval, eigvect = np.linalg.eig(matrix)
        # use imaginary part to get ascending order of eigvals
        if sort == 'imag':
            si = np.argsort(np.imag(eigval))
            eigval_out = eigval[si]
        elif sort == 'real':
            si = np.argsort(np.real(eigval))
            eigval_out = eigval[si]
        else:
            eigval_out = eigval

        if attribute:
            self.eigval = eigval_out

        return eigval_out

    def eig_vals_vects(self, matrix=None, attribute=True, check=False):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        if matrix is None:
            matrix = self.get_matrix(attribute=attribute)
            if check:
                le.plot_complex_matrix(matrix, show=True)

        eigval, eigvect = le.eig_vals_vects_hermitian(matrix, sort='real')
        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect
        return eigval, eigvect

    def load_eigval_eigvect(self, calc_if_not_saved=False, attribute=True):
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        print 'fn_evl = ', fn_evl
        fn_evt = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn_evl) and glob.glob(fn_evt):
            # with open(fn_evl, "r") as f:
            #     eigval = pickle.load(f)

            pklin = open(fn_evl, "rb")
            eigval = pickle.load(pklin)
            pklin.close()

            # with open(fn_evt, "r") as f:
            #    eigvect = pickle.load(f)
            pklin = open(fn_evt, "rb")
            eigvect = pickle.load(pklin)
            pklin.close()
        elif calc_if_not_saved:
            print 'HaldaneLattice.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            eigval, eigvect = self.eig_vals_vects()
            print '... saving eigval/vects'
            self.save_eigval_eigvect()
        else:
            raise RuntimeError('eigval and eigvect are not saved.')

        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect
        return eigval, eigvect

    def load_eigval(self, calc_if_not_saved=False, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        print 'HaldaneLattice.load_eigval: loading ', fn
        try:
            with open(fn, "r") as f:
                eigval = pickle.load(f)
        except IOError:
            if calc_if_not_saved:
                print 'HaldaneLattice.load_eigval: Could not load eigval, computing it...'
                eigval, eigvect = self.eig_vals_vects()
            else:
                raise RuntimeError('Could not load eigval since it is not saved.')

        if attribute:
            self.eigval = eigval
        return eigval

    def load_eigvect(self, calc_if_not_saved=False, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"
        print 'HaldaneLattice.load_eigvect: loading ', fn
        try:
            with open(fn, "r") as f:
                eigvect = pickle.load(f)

        except IOError:
            if calc_if_not_saved:
                print 'HaldaneLattice.load_eigvect: Could not load eigval, computing it...'
                eigval, eigvect = self.eig_vals_vects()
            else:
                raise RuntimeError('Could not load eigval since it is not saved.')
        if attribute:
            self.eigvect = eigvect
        return eigvect

    def load_localization(self, attribute=True):
        """Load eigvect from disk: first try hdf5, then look for pickle"""
        # Make localization name
        locz_name = "localization" + self.lp['meshfn_exten']
        # First look in localization_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_haldane.hdf5"
        saved_to_hdf5 = h5io.dset_in_hdf5(locz_name, h5fn)

        # If not there, look for pkl files
        fn_locz = dio.prepdir(self.lp['meshfn']) + "localization" + self.lp['meshfn_exten'] + ".pkl"

        if saved_to_hdf5:
            locz = h5io.extract_dset_hdf5(locz_name, h5fn)
        elif glob.glob(fn_locz):
            with open(fn_locz, "rb") as f:
                locz = pickle.load(f)
        else:
            return None

        if attribute:
            self.localization = locz

        return locz

    def load_edge_localization(self, attribute=True):
        """Load edge localization from disk: first try hdf5, then look for pickle

        Returns
        -------
        edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        """
        # Make localization name
        locz_name = "localization_edge" + self.lp['meshfn_exten']
        # First look in localization_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_edge_haldane.hdf5"
        saved_to_hdf5 = h5io.dset_in_hdf5(locz_name, h5fn)

        # If not there, look for pkl files
        fn_locz = dio.prepdir(self.lp['meshfn']) + locz_name + ".pkl"

        if saved_to_hdf5:
            elocz = h5io.extract_dset_hdf5(locz_name, h5fn)
        elif glob.glob(fn_locz):
            with open(fn_locz, "rb") as f:
                elocz = pickle.load(f)
        else:
            return None

        if attribute:
            self.edge_localization = elocz

        return elocz

    def load_ipr(self, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "ipr" + self.lp['meshfn_exten'] + ".pkl"
        with open(fn, "r") as f:
            ipr = pickle.load(f)

        if attribute:
            self.ipr = ipr
        return ipr

    def load_prpoly(self, attribute=False):
        fn = dio.prepdir(self.lp['meshfn']) + "prpoly" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn):
            with open(fn, "rb") as f:
                prpoly = pickle.load(f)

            if attribute:
                self.prpoly = prpoly

            return prpoly
        else:
            return None

    def get_matrix(self, attribute=False, sparse=False):
        """Obtain the dynamical matrix of the Haldane Lattice"""
        if self.matrix is None:
            return self.calc_matrix(attribute=attribute, sparse=sparse)
        else:
            return self.matrix

    def get_eigval_eigvect(self, attribute=False):
        """Return eigval and eigvect, obtaining them by (1) calling from self, (2) loading them, or (3) calculating"""
        # First attempt to return, then attempt to load, then calculate if unavailable
        if self.eigval is not None and self.eigvect is not None:
            eigval = self.eigval
            eigvect = self.eigvect
        else:
            try:
                # Try to load eigval and eigvect
                print 'Attempting to load eigval/vect...'
                eigval = self.load_eigval(attribute=attribute)
                eigvect = self.load_eigvect(attribute=attribute)
                print 'loaded!'
            except RuntimeError:
                print 'HaldaneLattice.get_eigval_eigvect: Could not load eigval/vect, calculating...'
                # calculate eigval and eigvect
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=False)
                else:
                    matrix = self.matrix
                eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)

        return eigval, eigvect

    def get_eigval(self, attribute=False):
        """Return eigval, obtaining it by (1) calling from self, (2) loading it, or (3) calculating"""
        # First attempt to return, then attempt to load, then calculate if unavailable
        if self.eigval is not None:
            eigval = self.eigval
        else:
            try:
                # Try to load eigval and eigvect
                eigval = self.load_eigval(attribute=attribute)
            except:
                # calculate eigval and eigvect
                matrix = self.get_matrix(attribute=False)
                eigval = self.calc_eigvals(matrix=matrix, attribute=attribute)

        return eigval

    def get_ipr(self, attribute=False, attrib_eigvalvect=False):
        """"""
        # Check that ipr is available, and load or calculate it if not
        if self.ipr is not None:
            ipr = self.ipr
        else:
            try:
                print 'Loading ipr...'
                ipr = self.load_ipr(attribute=attribute)
            except:
                print 'Calculating ipr...'
                ipr = self.calc_ipr(eigvect=self.eigvect, attribute=attribute, attrib_eigvalvect=attrib_eigvalvect)
        return ipr

    def get_prpoly(self, attribute=False):
        if self.prpoly is None:
            # try:
            print 'Attempting to load prpoly from ', self.lp['meshfn']
            prpoly = self.load_prpoly(attribute=attribute)
            if prpoly is None:
                print 'calculating prpoly...'
                print 'attribute=', attribute, 'but for some reason getting error when putting in attribute here...'
                prpoly = self.calc_pr_polygons(self, eigvect=None)
            else:
                print 'hlatclass: loaded prpoly.'
        else:
            prpoly = self.prpoly
        return prpoly

    def get_ldos(self, eps=None, attribute=False):
        """Obtain the local density of states for the HaldaneLattice

        Parameters
        ----------
        eps : float
            How many energy spacings to broaden the Lorentzian by
        attribute : bool
            Attribute the ldos to self (self.ldos = ldos) after computation, if not already stored
        """
        if eps is None:
            if 'eps' not in self.lp:
                self.lp['eps'] = 5.0
        else:
            self.lp['eps'] = eps

        if self.ldos is None:
            pklfn = dio.prepdir(self.lp['meshfn']) + 'ldos' + self.lp['meshfn_exten'] + '_eps' + \
                    sf.float2pstr(self.lp['eps']) + '.pkl'
            print 'seeking pklfn file ', pklfn
            if glob.glob(pklfn):
                print 'loading ldos...'
                with open(pklfn) as fn:
                    ldos = pickle.load(fn)
            else:
                print 'calculating ldos...'
                ldos = self.calc_ldos(eps=self.lp['eps'], attribute=attribute)
        else:
            ldos = self.ldos
        return ldos

    def get_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                         save_eigval=False, attribute_eigv=False):
        """Obtain the localization of eigenvectors of the HaldaneLattice (fits to 1d exponential decay)

        Parameters
        ----------
        attribute : bool
            make localization an attribute of HaldaneLatticeClass instance
        eigval : N x 1 float array or None
            eigenvalues of dynamical matrix, or None if too bulky to keep in RAM

        Returns
        -------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
        """
        if self.localization is not None:
            localization = self.localization
        else:
            fn = dio.prepdir(self.lp['meshfn']) + 'localization' + self.lp['meshfn_exten'] + '.txt'
            print 'seeking pklfn file ', fn
            if glob.glob(fn):
                print 'loading localization from txt file...'
                localization = np.loadtxt(fn, delimiter=',')
            else:
                print 'calculating localization...'
                localization = self.calc_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                                      save_eigvect_eigval=save_eigvect_eigval, save_eigval=save_eigval,
                                                      attribute_eigv=attribute_eigv)
        return localization

    def get_edge_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                              locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False, force_hdf5=True):
        """Obtain the edge localization of eigenvectors of the HaldaneLattice (fits to 1d exponential decay in distance
        from boundary) according to |psi| ~ A * exp(K * np.sqrt((x - x_edge)**2 + (y - y_edge)**2)) where
        (x_edge, y_edge) is the nearest interpolated point on the boundary

        Returns
        -------
        edge_localization : NP x 5 float array (ie, len(eigval) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting at most negative frequency and increasing in frequency
        """
        if self.edge_localization is not None:
            edge_localization = self.edge_localization
        else:
            # fn = dio.prepdir(self.lp['meshfn']) + 'localization_edge' + self.lp['meshfn_exten'] + '.txt'
            # print 'hlat.HaldaneLattice(): seeking pklfn file ', fn
            # if glob.glob(fn):
            #     print 'loading localization from txt file...'
            #     edge_localization = np.loadtxt(fn, delimiter=',')
            # else:
            edge_localization = self.load_edge_localization(attribute=attribute)
            if edge_localization is None:
                print 'calculating localization...'
                edge_localization = self.calc_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                                                save_eigvect_eigval=save_eigvect_eigval,
                                                                locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                save_eigval=save_eigval,
                                                                attribute_eigv=attribute_eigv)
        return edge_localization

    def get_ill(self, attribute=False, eigval=None, eigvect=None):
        localization = self.get_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        ill = localization[:, 3]
        return ill

    def get_edge_ill(self, attribute=False, eigval=None, eigvect=None):
        edge_localization = self.get_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        ill = - edge_localization[:, 1]
        return ill

    def ensure_eigval_eigvect(self, eigval=None, eigvect=None, attribute=True):
        """Make sure that eigval and eigvect are both saved to disk. If eigval and eigvect are supplied and a file
        exists on disk, do nothing. To obtain each, proceed by
        (1) returning from supplied eigval/eigvect,
        (2) calling from self, (3) loading it, or
        (4) calculating it.

        Parameters
        ----------
        eigval
        eigvect
        attribute

        Returns
        -------

        """
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        print 'fn_evl = ', fn_evl
        fn_evt = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn_evl) and glob.glob(fn_evt) and eigval is not None and eigvect is not None:
            pass
        elif glob.glob(fn_evl) and glob.glob(fn_evt):
            if self.eigval is not None and self.eigvect is not None:
                eigval = self.eigval
                eigvect = self.eigvect
            else:
                # Try to load eigval and eigvect
                print 'Attempting to load eigval/vect...'
                eigval = self.load_eigval(attribute=attribute)
                eigvect = self.load_eigvect(attribute=attribute)
        else:
            print 'hlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if eigval is None or eigvect is None:
                eigval, eigvect = self.eig_vals_vects()
            print '... saving eigval/vects'
            self.save_eigval_eigvect()

        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect
        return eigval, eigvect

    def ensure_eigval(self, eigval=None, attribute=False):
        """Return eigval and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied eigval, (2) calling from self, (3) loading it, or
        (4) calculating it.
        """
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn_evl) and eigval is not None:
            # performs (1), simply return the supplied eigval, since already saved
            pass
        elif glob.glob(fn_evl):
            # attempts (2) or (3)
            if self.eigval is not None:
                eigval = self.eigval
            else:
                # Try to load eigval and eigvect
                print 'Attempting to load eigval/vect...'
                eigval = self.load_eigval(attribute=attribute)
        else:
            # attempts (4)
            print 'hlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if eigval is None:
                eigval, eigvect = self.eig_vals_vects()
            print '... saving eigval/vects'
            self.save_eigval()

        if attribute:
            self.eigval = eigval
        return eigval

    def ensure_edge_localization(self, elocz=None, attribute=False, force_hdf5=False, save_images=False, save_im=False):
        """Return elocz and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied locz, (2) calling from self, (3) loading it, or
        (4) calculating it.

        Parameters
        ----------
        elocz : NP x 5 float array (ie, len(eigval) x 5 float array)
            edge_localization fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting at most negative frequency and increasing in frequency
        attribute : bool
            whether to attribute the edge localization parameters to self
        force_hdf5 : bool
            Save the localization into hdf5, not to text file
        save_images : bool
            if edge_localization is not already saved,output one image for every eigenmode to check that fitting (shown
            as heatmap) is appropriate
        save_im : bool
            if edge_localization is not already saved, save a single image of the edge localization decay param as a
            function of eigenmode frequency

        Returns
        -------
        elocz : NP x 5 float array (ie, len(eigval) x 5 float array)
            edge localization fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting at most negative frequency and increasing in frequency
        """
        # Make locz name
        locz_name = "localization_edge" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_edge_haldane.hdf5"
        saved_to_file = h5io.dset_in_hdf5(locz_name, h5fn)

        # If not there, look for pkl files
        if not saved_to_file:
            fn_locz = dio.prepdir(self.lp['meshfn']) + locz_name + ".txt"
            saved_to_file = glob.glob(fn_locz)

        if saved_to_file and elocz is not None:
            # performs (1), simply return the supplied elocz, since already saved
            pass
        elif saved_to_file:
            # attempts (2) or (3)
            if self.edge_localization is not None:
                locz = self.edge_localization
            else:
                # Try to load locz
                print 'Attempting to load locz...'
                elocz = self.load_edge_localization(attribute=attribute)
        else:
            # attempts (4)
            print 'hlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if elocz is None:
                elocz = self.calc_edge_localization(attribute=attribute)
            print '... saving eigval/vects'
            self.save_edge_localization(force_hdf5=force_hdf5, save_images=save_images, save_im=save_im)

        if attribute:
            self.edge_localization = elocz
        return elocz

    def save_eigval(self, eigval=None, infodir='auto', attribute=True):
        """Save eigenvalues for this HaldaneLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval, eigvect
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)

        Returns
        -------

        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if eigval is None:
            eigval = self.get_eigval(attribute=attribute)

        eigvalfn = infodir + 'eigval' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvalfn, 'wb')
        pickle.dump(eigval, output)
        output.close()

        fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'haldane')
        plt.savefig(infodir + 'eigval_hist' + self.lp['meshfn_exten'] + '.png')
        plt.clf()
        print 'Saved haldane eigval to ' + eigvalfn

    def save_eigval_eigvect(self, eigval=None, eigvect=None, infodir='auto', attribute=True):
        """Save eigenvalues and eigenvectors for this HaldaneLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval, eigvect
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)

        Returns
        -------

        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=attribute)

        eigvalfn = infodir + 'eigval' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvalfn, 'wb')
        pickle.dump(eigval, output)
        output.close()

        eigvectfn = infodir + 'eigvect' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvectfn, 'wb')
        pickle.dump(eigvect, output)
        output.close()

        fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'haldane')
        plt.savefig(infodir + 'eigval_hist' + self.lp['meshfn_exten'] + '.png')
        plt.clf()
        print 'Saved haldane DOS to ' + eigvalfn + '\n and ' + eigvectfn

    def save_pin(self, infodir='auto', attribute=True, histogram=False, force_hdf5=False):
        """Save pinning energies (on-site energies) for this HaldaneLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval, eigvect
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
        histogram : bool
            If saving to a txt file, save a png of the pinning distribution
        force_hdf5 : bool
            save the pinning configuration to an hdf5 file rather than a text file
        """
        # if self.lp['V0_pin_gauss'] > 1e-7 and self.lp['NH'] > 12:
        #     raise RuntimeError('Debugging --> preventing save pin')
        #     sys.exit()
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if self.pin is not None:
            pinmfe = self.get_pinmeshfn_exten()
            pinning_name = 'pin' + pinmfe
            # sys.exit()
            # When running jobs in series (NOT in parallel), can save pinning directly to hdf5
            if force_hdf5:
                h5fn = dio.prepdir(self.lp['meshfn']) + 'pin_configs.hdf5'
                if glob.glob(h5fn):
                    rw = "r+"
                else:
                    rw = "w"

                with h5py.File(h5fn, rw) as fi:
                    keys = fi.keys()
                    # is this pinning configuration already in the hdf5 file?
                    if pinning_name not in keys:
                        # add pinning to the hdf5 file
                        print 'saving pinning in hdf5...'
                        fi.create_dataset(pinning_name, shape=np.shape(self.pin), data=self.pin, dtype='float')
                    else:
                        raise RuntimeError('Pinning config already exists in hdf5, exiting...')
            else:
                fn = dio.prepdir(self.lp['meshfn']) + pinning_name + '.txt'
                np.savetxt(fn, self.pin, header="Pinning strengths in Haldane model (on-diag elements)")
                if histogram:
                    plt.clf()
                    fig, hist_ax = leplt.initialize_histogram(self.pin, xlabel=r'On-site energies', ylabel='Occurrence')
                    plt.savefig(infodir + 'pin_hist' + pinning_name + '.png')
                    plt.clf()
                print 'Saved pinning energies to ' + fn
        else:
            raise RuntimeError('self.pin is None, so cannot save it!')

    def save_ipr(self, infodir='auto', attribute=True, save_images=True, show=False):
        """
        Parameters
        ----------
        infodir: str
            Directory in which to save ipr as ipr.pkl
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)

        Returns
        -------

        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        ipr = self.get_ipr(attribute=attribute)
        fn = infodir + 'ipr' + self.lp['meshfn_exten'] + '.pkl'
        print 'Saving ipr as ' + fn
        pickle.dump(ipr, file(fn, 'wb'))

        if attribute:
            self.ipr = ipr

        if save_images:
            # save IPR as png
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for Haldane network',
                              fname='ipr_haldane_hist'+self.lp['meshfn_exten'],
                              alpha=1.0, FSFS=12, inverse_PR=True, show=show)
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for Haldane network',
                              fname='pr_haldane_hist'+self.lp['meshfn_exten'],
                              alpha=1.0, FSFS=12, inverse_PR=False, show=show)
        print 'Saved haldane ipr to ' + infodir + 'ipr' + self.lp['meshfn_exten'] + '.pkl'

    def save_prpoly(self, infodir='auto', attribute=True, save_plot=True):
        """
        Parameters
        ----------
        infodir: str
            Directory in which to save ipr as ipr.pkl
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
        save_plot : bool
            Save a png of the participation of polygons in the DOS

        Returns
        -------

        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        prpoly = self.get_prpoly(attribute=attribute)
        fn = infodir + 'prpoly' + self.lp['meshfn_exten'] + '.pkl'
        print 'prpoly = ', prpoly
        print 'Saving prpoly as ' + fn
        with open(fn, "wb") as fn:
            pickle.dump(prpoly, fn)

        if save_plot:
            # save prpoly as png
            self.plot_prpoly(outdir=infodir, show=False, shaded=False)

        print 'Saved haldane prpoly to ' + infodir + 'prpoly' + self.lp['meshfn_exten'] + '.pkl'

    def save_ldos(self, infodir='auto', attribute=True, save_images=True):
        """
        Parameters
        ----------
        infodir: str
            Directory in which to save ipr as ipr.pkl
        attribute: bool
            Whether to attribute the local density of states to self (ie self.ldos = ldos)
        save_images : bool
            Save a movie of the localization fits

        Returns
        -------
        ldos : float array?
            local density of states
        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])  # + 'ldos' + self.lp['meshfn_exten'] + '/'

        ldos = self.get_ldos(attribute=attribute)
        fn = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' + sf.float2pstr(self.lp['eps']) + '.pkl'
        print 'Saving ldos as ' + fn
        pickle.dump(ldos, file(fn, 'wb'))

        if save_images:
            eigval = self.get_eigval()
            # save LDOS as png
            ldos_infodir = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' +\
                           sf.float2pstr(self.lp['eps']) + '/'
            dio.ensure_dir(ldos_infodir)
            self.plot_ldos(eigval=eigval, ldos=ldos, outdir=ldos_infodir, FSFS=12)

            # Make movie
            imgname = ldos_infodir + 'ldos_site'
            movname = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' +\
                      sf.float2pstr(self.lp['eps']) + '_sites'
            lemov.make_movie(imgname, movname, indexsz='08', imgdir=ldos_infodir, rm_images=True, save_into_subdir=True)

        print 'Saved haldane ldos to ' + fn

    def save_localization(self, eigval=None, infodir='auto', attribute=True, save_images=False,
                          save_eigvect_eigval=False, save_eigval=False):
        """Get and save localization measure for all eigenvectors of the HaldaneLattice

        Parameters
        ----------
        eigval : N x 1 float array or None
            eigenvalues of dynamical matrix, or None if too bulky to keep in RAM
        infodir : str
            path to where lattice is stored (meshfn). If 'auto', function uses lp['meshfn']
        attribute : bool
        save_images : bool
            Plot the localization as heat maps
        save_eigvect_eigval : bool
            Save the eigvect and eigval to disk if self.localization is None
        save_eigval : bool
            If save_eigvect_eigval is False and this arg is True, save just the eigenvalues, not the eigvect
        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        locz = self.get_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval,
                                     save_eigval=save_eigval)
        fn = infodir + 'localization' + self.lp['meshfn_exten'] + '.txt'
        print 'Saving localization as ' + fn
        header = "Localization of eigvects: fitted to A*exp(K*sqrt((x-xc)**2 + (y-yc)**2)): " +\
                 "xc, yc, A, K, uncA, covAK, uncK"
        np.savetxt(fn, locz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if eigval is None:
            eigval = self.get_eigval()

        fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.6, hsfrac=0.6 * 0.75, fontsize=10, tspace=3)
        evals = np.real(eigval)
        ax.plot(evals, -locz[:, 3], '-', color='#334A5A')
        ax.fill_between(evals, -locz[:, 3] - np.sqrt(locz[:, 6]), -locz[:, 3] + np.sqrt(locz[:, 6]), color='#89BBDB')
        # ax.set_xlim(np.min(evals)-0.1, np.max(evals)+0.1)
        title = r'Localization length $\lambda$ for $|\psi| \sim e^{-r / \lambda}$'
        plt.text(0.5, 1.08, title, horizontalalignment='center', fontsize=10, transform=ax.transAxes)
        ax.set_ylabel(r'Inverse localization length, $1/\lambda$')
        ax.set_xlabel(r'Oscillation frequency, $\omega/\Omega_g$')
        plt.savefig(fn[:-4] + '.pdf', dpi=300)

        if save_images:
            # save localization fits as pngs
            loclz_infodir = infodir + 'localization' + self.lp['meshfn_exten'] + '/'
            dio.ensure_dir(loclz_infodir)
            self.plot_localization(localization=locz, outdir=loclz_infodir, fontsize=12)

            # Make movie
            imgname = loclz_infodir + 'localization' + self.lp['meshfn_exten'] + '_'
            movname = infodir + 'localization' + self.lp['meshfn_exten']
            lemov.make_movie(imgname, movname, indexsz='06', framerate=4, imgdir=loclz_infodir, rm_images=True,
                             save_into_subdir=True)

        print 'Saved haldane localization to ' + fn
        return locz

    def save_edge_localization(self, eigval=None, infodir='auto', attribute=True, save_images=False,
                               save_eigvect_eigval=False, save_im=False, force_hdf5=False):
        """Get and save localization measure for all eigenvectors of the HaldaneLattice

        Parameters
        ----------
        eigval
        infodir
        attribute
        save_images
        save_eigvect_eigval
        save_im
        force_hdf5

        Returns
        -------

        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        elocz = self.get_edge_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        locz_name = 'localization_edge' + self.lp['meshfn_exten']
        if force_hdf5:
            hdf5fn = infodir + 'localization_edge_haldane.hdf5'
            print 'hlat.HaldaneLattice(): saving edge_localization dataset in ' + hdf5fn
            h5io.save_dset_hdf5(elocz, locz_name, hdf5fn)
            fn = infodir + locz_name
        else:
            fn = infodir + locz_name + '.txt'
            print 'Saving edge localization as ' + fn
            header = "Localization of eigvects to edge: fitted to A*exp(K*sqrt((x-xb)**2 + (y-yb)**2)): " + \
                     "A, K, uncA, covAK, uncK. xb and yb are nearest points along the boundary. " + \
                     "The modes examined range from 0 to len(eigval)."
            np.savetxt(fn, elocz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if save_im:
            if eigval is None:
                eigval = self.get_eigval()

            fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.6, hsfrac=0.6 * 0.75, fontsize=10, tspace=3)
            evals = np.real(eigval)
            ax.plot(evals, -elocz[:, 3], '-', color='#334A5A')
            ax.fill_between(evals, -elocz[:, 1] - np.sqrt(elocz[:, 4]), -elocz[:, 1] + np.sqrt(elocz[:, 4]),
                            color='#89BBDB')
            if abs(self.lp['Omg']) == 1 and abs(self.lp['Omk']) == 1 and not self.lp['dcdisorder']:
                ax.set_xlim(1.0, 4.0)
            title = r'Edge localization length $\lambda$ for $|\psi| \sim e^{-r / \lambda}$'
            plt.text(0.5, 1.08, title, horizontalalignment='center', fontsize=10, transform=ax.transAxes)
            ax.set_ylabel(r'Inverse localization length, $1/\lambda$')
            ax.set_xlabel(r'Oscillation frequency, $\omega$')
            plt.savefig(fn[:-4] + '.pdf', dpi=300)

        if save_images:
            # save localization fits as pngs
            loclz_infodir = infodir + locz_name + '/'
            dio.ensure_dir(loclz_infodir)
            self.plot_edge_localization(elocz=elocz, outdir=loclz_infodir, fontsize=12)

            # Make movie
            imgname = loclz_infodir + locz_name + '_'
            movname = infodir + locz_name
            lemov.make_movie(imgname, movname, indexsz='06', framerate=4, imgdir=loclz_infodir, rm_images=True,
                             save_into_subdir=True)

        print 'Saved haldane edge localization to ' + fn
        return elocz

    def save_DOSmovie(self, infodir='auto', attribute=True, save_DOS_if_missing=True):
        if infodir == 'auto':
            infodir = self.lattice.lp['meshfn'] + '/'
        exten = self.lp['meshfn_exten']

        # Obtain eigval and eigvect, and matrix if necessary
        if self.eigval is None or self.eigvect is None:
            # eigal, eigvect = self.load_eigval_eigvect(calc_if_not_saved=True)
            # check if we can load the DOS info
            eigvalfn = infodir + 'eigval' + exten + '.pkl'
            eigvectfn = infodir + 'eigvect' + exten + '.pkl'
            if glob.glob(eigvalfn) and glob.glob(eigvectfn):
                print "Loading eigval and eigvect from " + self.lattice.lp['meshfn'] + "eig*_haldane..."
                with open(eigvalfn, "rb") as f:
                    eigval = pickle.load(f)
                with open(eigvectfn, "rb") as f:
                    eigvect = pickle.load(f)
            else:
                print 'HaldaneLattice.save_DOSmovie: Could not find pkl: ', eigvalfn
                # sys.exit()
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=attribute)
                    eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
                else:
                    eigval, eigvect = self.eig_vals_vects(attribute=attribute)

                if save_DOS_if_missing:
                    output = open(eigvalfn, 'wb')
                    pickle.dump(eigval, output)
                    output.close()

                    output = open(eigvectfn, 'wb')
                    pickle.dump(eigvect, output)
                    output.close()

                    print 'Saved haldane DOS to ' + infodir + 'eigvect(val)' + exten + '.pkl\n'
        else:
            eigval = self.eigval

        if not glob.glob(infodir + 'eigval_haldane_hist' + exten + '.png'):
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'haldane')
            plt.savefig(infodir + 'eigval_haldane_hist' + exten + '.png')
            plt.clf()

        if self.lp['periodicBC']:
            colorstr = 'ill'
        else:
            colorstr = 'pr'

        lemov.save_normal_modes_haldane(self, datadir=infodir, rm_images=True, gapims_only=False,
                                        save_into_subdir=False, overwrite=True, color=colorstr)

    def calc_ipr(self, eigvect=None, attribute=True, attrib_eigvalvect=False, check=False):
        """Calculate the inverse participation ratio for the lattice.
        Parameters
        ----------
        outdir: str
            path for saving
        eigvect: None or numpy float array
            The eigenvectors for the network. If None, this method loads or calculates the eigvect.
        attribute: bool
            Whether to attribute ipr to self
        attrib_eigvalvect: bool
            Attribute eigval and eigvect to self if eigvect is not supplied
        """
        # Calculate the inverse participation ratio defined by the modulus of the "wavefunction" at each site, |psi|
        if eigvect is None:
            print 'Loading eigvect/eigval for ipr calculation without attribution...'
            eigval, eigvect = self.get_eigval_eigvect(attribute=attrib_eigvalvect)

        # ipr --> N sum(|x|^4)/ sum(|x|^2)^2
        ipr = len(eigvect) * np.sum(np.abs(eigvect)**4, axis=1) / np.sum(np.abs(eigvect)**2, axis=1)**2

        # ipr --> sum(|x|^4)/ sum(|x|^2)^2
        # ipr = np.sum(np.abs(eigvect)**4,axis=1) / np.sum(np.abs(eigvect)**2, axis=1)**2

        # ipr --> N sum(|x|^4)/ sum(|x|^2)^2
        # ipr = len(eigvect) * np.sum(np.abs(eigvect)**2,axis=1) / np.sum(np.abs(eigvect), axis=1)**2

        if attribute:
            self.ipr = ipr
            return self.ipr
        else:
            return ipr

    def calc_pr_polygons(self, attribute=True, eigvect=None, check=False):
        """
        Parameters
        ----------
        eigvect: None or 2*NP x 2*NP complex array
            Lets user supply eigvect to avoid sluggish loading or computation

        Returns
        -------
        prpoly : dict (keys are int, values are 2NP x 1 float arrays)
            For each number of sides (polygon type), what percentage of each excitation is attributable to that #sides.
            keys are integer number of sides of possible polygons, beginning with zero.
        """
        if self.lattice.polygons is not None:
            polygons = self.lattice.polygons
        else:
            polygons = self.lattice.load_polygons()

        # number of polygon sides
        Pno = np.array([len(polyg) - 1 for polyg in self.lattice.polygons], dtype=int)
        # print 'Pno = ', Pno

        # Check that the very large polygons are not an error
        # for inds in np.where(Pno>9)[0]:
        #     PPC = le.polygons2PPC(self.lattice.xy, [polygons[inds]], check=check)

        # Create dictionary for each particle: which polygons does the particle participate in (be on boundary for)
        # pnod is 'polygon number dictionary', with each key being a particle number (int) and value being list of
        # polygon numbers participated in by this particle
        pnod = {}
        for ii in range(len(self.lattice.xy)):
            jj = 0
            pnod[ii] = []
            for polyg in polygons:
                if ii in polyg:
                    pnod[ii].append(Pno[jj])
                jj += 1

        # Contribution to each kind of polygon from each particle (psides)
        # psides : NP x NP float array
        #   the (i,j)th element has the percentage of the ith particle attributable to polygons with j sides
        maxNsides = np.max(Pno)
        psides = np.zeros((2.*len(self.lattice.xy), maxNsides+1), dtype=float)
        for ii in range(len(self.lattice.xy)):
            for jj in range(maxNsides+1):
                if len(pnod[ii]) > 0:
                    # print 'pnod[ii] = ', pnod[ii]
                    # print 'np.sum(np.array(pnod[', ii, ']) == ', jj, ') =', np.sum(np.array(pnod[ii]) == jj )
                    psides[ii, jj] = float(np.sum(np.array(pnod[ii]) == jj)) / float(len(pnod[ii]))
                    # psides[ii, jj] = float(np.sum(np.array(pnod[ii]) == jj)) / float(len(pnod[ii]))
                    # psides[ii + len(self.lattice.xy), jj] = psides[2*ii, jj]
                else:
                    print 'Warning! no polygons matching particle: ', ii

        # Calculate the inverse participation ratio weighted by each polygon type
        if eigvect is None:
            print 'Loading eigvect/eigval for prpoly calculation without attribution...'
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)

        # pr --> sum(|x|^2)^2 / [ N sum(|x|^4) ]
        prpoly = {}
        # Pedestrian version
        for ii in range(maxNsides + 1):
            pside = psides[:, ii]
            prpoly[ii] = np.zeros(len(eigvect))
            # prpoly[ii] lists the contributions for every eigenmode attributable to polygons with ii sides
            for jj in range(len(eigvect)):
                prpoly[ii][jj] = np.sum(np.abs(eigvect[jj])**2 * pside)

        # This method below isn't quite right, but would be faster
        # for ii in range(maxNsides+1):
        #     # MM is 2NP x 2NP float array, with (i,j)th element being
        #     MM = np.dstack(np.array([psides[:, ii].tolist()]*np.shape(eigvect)[1]))[0].T
        #     print 'psides[:, ', ii, '] = ', psides[:, ii]
        #     print 'MM = ', MM
        #     print 'np.shape(MM) = ', np.shape(MM)
        #     print 'np.abs(eigvect)**2 = ', np.abs(eigvect)**2
        #     print 'np.abs(eigvect)**2 * MM= ', np.abs(eigvect)**2 * MM
        #     prpoly[ii] = np.sum(np.abs(eigvect)**2 * MM, axis=1)**2 / (len(eigvect) *
        #                                                                     np.sum(np.abs(eigvect)**4, axis=1))
        #     # prpoly[ii] = np.sum(float(ii == 6) * MM, axis=1)**2 / \
        #     #                   (len(eigvect) * np.sum(np.abs(eigvect)**4, axis=1))
        #     print 'prpoly[', ii, '] = ', prpoly[ii]

        if attribute:
            self.prpoly = prpoly

        return prpoly

    def calc_ldos(self, eps=None, attribute=False, eigval=None, eigvect=None):
        if eps is None:
            if 'eps' in self.lp:
                eps = lp['eps']
            else:
                eps = 5.0
        self.lp['eps'] = eps

        if eigval is None or eigvect is None:
            print 'Loading eigvect/eigval for ldos calculation without attribution...'
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        ldos = hlatfns.calc_ldos(eigval, eigvect, eps=eps)
        if attribute:
            self.ldos = ldos

        return ldos

    def calc_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                          save_eigval=False,  locutoffd=None, hicutoffd=None, attribute_eigv=False,
                          force_hdf5=False):
        """Fit each eigenvector excitation to an exponential decay centered about the excitation's COM:
        A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Returns
        -------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
            where fit is A * exp(-K * np.sqrt((x - x_center)**2 + (y - y_center)**2))
        attribute : bool
            attribute localization to self
        save_eigvect_eigval : bool
            Save eigvect and eigval to disk
        save_eigval : bool
            If save_eigvect_eigval is False and save_eigval is True, just saves eigval
        locutoffd : float or None
            minimum distance from max localization point to start using data to fit to exp tail
        hicutoffd : float or None
            maximum distance from max localization point to use data to fit to exp tail
        attribute_eigv : bool
            Attribute eigval and eigvect to self

        Returns
        -------
        localization : NP x 7 float array (ie, len(eigval) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=attribute_eigv)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current hlat if not already saved...'
                self.ensure_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False)
            elif save_eigval:
                print 'Saving eigenvalues for current hlat if not already saved...'
                self.ensure_eigval(eigval=eigval, attribute=False)
        if self.lattice.lp['periodicBC']:
            if 'periodic_strip' in self.lattice.lp:
                if self.lattice.lp['periodic_strip']:
                    perstrip = True
                else:
                    perstrip = False
            else:
                perstrip = False

            if perstrip:
                # edge_localization = hlatfns.fit_edgedecay_periodicstrip(self.lattice.xy, eigval, eigvect,
                #                                                         cutoffd=cutoffd, check=self.lp['check'])
                localization = hlatfns.fit_eigvect_to_exponential_1dperiodic(self.lattice.xy, eigval, eigvect,
                                                                             self.lattice.lp['LL'],
                                                                             locutoffd=cutoffd, hicutoffd=hicutoffd,
                                                                             check=self.lp['check'])
            else:
                localization = hlatfns.fit_eigvect_to_exponential_periodic(self.lattice.xy, eigval, eigvect,
                                                                           self.lattice.lp['LL'],
                                                                           locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                           check=self.lp['check'])
        else:
            localization = hlatfns.fit_eigvect_to_exponential(self.lattice.xy, eigval, eigvect,
                                                              hicutoffd=hicutoffd, check=self.lp['check'])

        if attribute:
            self.localization = localization

        if attribute_eigv:
            self.eigval = eigval
            self.eigvect = eigvect

        return localization

    def calc_edge_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                               save_eigval=False, locutoffd=None, hicutoffd=None, attribute_eigv=False, check=False):
        """Measure the localization length of the edge modes (look for exponential falloff from the edge of the sample)

        Parameters
        ----------
        attribute : bool
            attribute localization to self
        eigvect : 2N x 2N complex array
            The eigenvectors of the haldane network
        save_eigvect_eigval : bool
            Save eigvect and eigval to disk
        save_eigval : bool
            If save_eigvect_eigval is False and save_eigval is True, just saves eigval
        attribute_eigv : bool
            Attribute eigval and eigvect to self
        cutoffd
        locutoffd
        hicutoffd
        attribute_eigv

        Returns
        -------
        edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency

        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current hlat...'
                self.save_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False)
            elif save_eigval:
                print 'hlat.calc_localization(): saving eigval only...'
                self.ensure_eigval(eigval=eigval, attribute=False)

        boundary = self.lattice.get_boundary()

        if not check:
            check = self.lp['check']

        if self.lattice.lp['periodicBC']:
            if 'periodic_strip' in self.lattice.lp:
                if self.lattice.lp['periodic_strip']:
                    perstrip = True
                else:
                    perstrip = False
            else:
                perstrip = False

            if perstrip:
                edge_localization = hlatfns.fit_eigvect_edge_periodicstrip(self.lattice.xy, boundary,
                                                                           PVx=self.lattice.PVx,
                                                                           PVy=self.lattice.PVy,
                                                                           eigvect=eigvect, eigval=eigval,
                                                                           locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                           check=check)
            else:
                raise RuntimeError('No reason to fit a fully periodic sample to edge localization.')
        else:
            edge_localization = hlatfns.fit_eigvect_to_exponential_edge(self.lattice.xy, boundary, eigvect, eigval,
                                                                        locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                        check=check)

        if attribute:
            self.edge_localization = edge_localization

        return edge_localization

    def plot_pin(self, ptsz=10, dpi=300):
        """Plot the pinning sites by coloring sites according to their pinning strengths"""
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(wsfrac=0.5,)
        self.lattice.plot_BW_lat(fig=fig, ax=ax, save=False, close=False, axis_off=False, title='')
        sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=ptsz,
                        c=self.pin, cmap='coolwarm', edgecolors='none')
        ticks = [np.min(self.pin), self.lp['pin'], np.max(self.pin)]
        plt.colorbar(mappable=sc, cax=cbar_ax, label=r'$\Omega_g$', orientation='horizontal', ticks=ticks)
        cbar_ax.xaxis.labelpad = -25
        plt.savefig(dio.prepdir(self.lp['meshfn']) + 'pin_mean' + sf.float2pstr(self.lp['pin']) +
                    self.lp['meshfn_exten'] + '.png', dpi=300)
        plt.clf()
        # plt.show()

    def plot_prpoly(self, outdir=None, title=r'Polygonal contributions to normal mode excitations',
                    fname='prpoly_haldane_hist', fontsize=8, show=True, global_alpha=1.0, shaded=False, save=True):
        """Plot Inverse Participation Ratio of the Haldane network

        Parameters
        ----------
        outdir: None or str
            If not None, outputs plot to this directory. If None and save==True, outputs to self.lp['meshfn']
        title: str
            The title of the plot
        fname: str
            the name of the file to save as png
        FSFS: float or int
            fontsize
        show: bool
            Whether to show the plot after creating it
        global_alpha: float (default=1.0)
            Overall factor by which to reduce the opacity of the bars in the histogram

        Returns
        -------
        ax: tuple of matplotlib axis instance handles
            handles for the axes of the histograms
        """
        prpoly = self.get_prpoly()

        nrows = 1
        n_ax = int(np.ceil(len(prpoly)/float(nrows)))-3
        fig, ax, cbar_ax = leplt.initialize_axis_stack(n_ax, Wfig=90, Hfig=120, make_cbar=True, hfrac=None,
                                                       wfrac=0.6, x0frac=None, y0frac=0.12, cbarspace=5, tspace=10,
                                                       vspace=2, cbar_orientation='horizontal')

        self.add_prpoly_to_plot(fig, ax, cbar_ax=cbar_ax, prpoly=prpoly, global_alpha=global_alpha, shaded=shaded)
        ii = 0
        ylims = ax[0].get_ylim()
        for axis in ax:
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.yaxis.set_ticks_position('left')
            axis.xaxis.set_ticks_position('bottom')
            axis.yaxis.set_ticks([int(ylims[1] * 0.1) * 10])
            if ii < len(ax) - 1:
                axis.set_xticklabels([])
            ii += 1

        ax[len(ax) - 1].set_xlabel(r'Oscillation frequency, $\omega/\Omega_g$')
        ax[len(ax) - 1].annotate('Density of states, $D(\omega)$', xy=(.001, .5), xycoords='figure fraction',
                                 horizontalalignment='left', verticalalignment='center',
                                 fontsize=fontsize, rotation=90)

        if title is not None:
            plt.suptitle(title, fontsize=fontsize)

        if save:
            if outdir is None:
                outdir = dio.prepdir(self.lp['meshfn'])

            # pickle.dump(fig, file(outdir+fname+'.pkl','w'))
            plt.savefig(outdir + fname + self.lp['meshfn_exten'] + '.png')

        if show:
            print 'Displaying figure.'
            plt.show()
        else:
            return fig, ax

    def add_prpoly_to_plot(self, fig, ax, cbar_ax=None, prpoly=None, global_alpha=1.0, shaded=False, fontsize=8,
                           vmax=None):
        """
        Add prpoly to existing plot with many subplots, skipping 1-gons and 2-gons

        ax: list of axes instances (subplots)
        """
        print 'global_alpha = ', global_alpha
        if prpoly is None:
            print 'getting prpoly...'
            prpoly = self.get_prpoly()
        eigval = self.get_eigval()

        # Make sure there are enough subplots in ax for all n-gons with n>2
        naxes = len(prpoly) - 3
        if len(ax) < naxes:
            print 'Insufficient # axes, making new fig...'
            # new_fig = plt.figure()
            # NOT CONVINCED THIS WORKS...
            fig, ax = leplt.change_axes_geometry_stack(fig, ax, naxes)
            # fig, axes = leplt.plot_axes_on_fig(ax[ii], fig=new_fig, geometry=(naxes,1,ii))
            print 'ax = ', ax

        if vmax is None:
            vmax = 0.0
            for ii in range(len(prpoly)):
                vmax = max(np.max(prpoly[ii]), vmax)

        print 'overlaying prpoly to axis stack...'
        for ii in range(len(prpoly)-3):
            # print 'ax[', ii, '] = ', ax[ii], ' of ', len(prpoly)
            if shaded:
                DOS_ax, cbar_ax, cbar, n, bins = \
                    leplt.shaded_DOS_plot(eigval, ax[ii], 'haldane',
                                          alpha=prpoly[ii + 3] * global_alpha, facecolor='#80D080',
                                          fontsize=fontsize, cbar_ax=None, vmin=0.0, vmax=vmax, linewidth=0,
                                          cax_label='', make_cbar=False, climbars=True, xlabel=xlabel, ylabel=str(ii+3))
            else:
                leplt.colored_DOS_plot(eigval, ax[ii], 'haldane', alpha=global_alpha, colorV=prpoly[ii + 3],
                                       colormap='CMRmap_r', norm=None, nbins=75, fontsize=fontsize, cbar_ax=cbar_ax,
                                       vmin=0.0, vmax=vmax, linewidth=0.,
                                       make_cbar=True, climbars=True, xlabel='Oscillation frequency, $\omega/\Omega_g$',
                                       xlabel_pad=None, ylabel=str(ii+3), ylabel_pad=None,
                                       cax_label=r'$\sum_i |\psi_i|^2$ in $n$-gons',
                                       cbar_labelpad=-37, ticks=None, cbar_nticks=3, cbar_tickfmt='%0.2f',
                                       orientation='vertical', cbar_orientation='horizontal',
                                       invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False,
                                       ylabel_rot=90)

            ax[ii].yaxis.set_major_locator(MaxNLocator(nbins=3))

        return fig, ax

    def add_ipr_to_ax(self, ax, ipr=None, alpha=1.0, inverse_PR=True, **kwargs):
        """Add a DOS colored by (Inverse) Participation Ratio of the Haldane network to an axis

        Parameters
        ----------
        alpha: float
            The opacity of the bars in the histogram
        inverse_PR: bool
            Whether to plot the IPR or the PR
        **kwargs : keyword arguments for colored_DOS_plot()
            colormap='viridis', norm=None, facecolor='#80D080', nbins = 75, fontsize=12, cbar_ax=None,
            vmin=None, vmax=None, linewidth=1, cax_label='', make_cbar=True, climbars=True

        Returns
        -------
        DOS_ax: matplotlib axis instance handle
            handle for the axis of the histogram
        """
        # First make sure all eigvals are accounted for
        try:
            self.eigval[0]
            eigval = self.eigval
            # Also make sure ipr is defined
            if ipr is None:
                if self.ipr is None:
                    # This works whether self.eigvect==None (will calc. it on the fly) or already attributed correctly
                    ipr = self.get_ipr(attribute=False)
                else:
                    ipr = self.ipr
        except TypeError:
            # Don't attribute the eigenvectors and eigenvalues for a method that simply plots them
            eigval = self.get_eigval(attribute=False)
            # Also make sure ipr is defined
            if ipr is None:
                if self.ipr is None:
                    ipr = self.get_ipr(attribute=False)
                else:
                    ipr = self.ipr

        # Register cmap if necessary
        if inverse_PR:
            if 'viridis' not in plt.colormaps():
                cmaps.register_colormaps()
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'haldane', alpha=alpha,
                                                                colorV=ipr, colormap='viridis', **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()
            print 'len(ipr) = ', len(ipr)
            print 'len(eigval) = ', len(eigval)
            print 'len(xy) = ', len(self.lattice.xy)
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'haldane', alpha=alpha,
                                                                colorV=1./ipr, colormap='viridis_r', **kwargs)
        return ax

    def plot_ipr_DOS(self, outdir=None, title=r'$D(\omega)$ for Haldane network', fname='ipr_haldane_hist',
                     alpha=1.0, FSFS=12, show=True, inverse_PR=True, save=True):
        """Plot Inverse Participation Ratio of the Haldane network

        Parameters
        ----------
        outdir: None or str
            If not None, outputs plot to this directory
        title: str
            The title of the plot
        fname: str
            the name of the file to save as png
        alpha: float
            The opacity of the bars in the histogram
        FSFS: float or int
            fontsize
        show: bool
            Whether to show the plot after creating it
        inverse_PR: bool
            Whether to plot the IPR or the PR

        Returns
        -------
        DOS_ax: matplotlib axis instance handle
            handle for the axis of the histogram
        """
        # First make sure all eigvals are accounted for
        # Don't attribute the eigenvectors and eigenvalues for a method that simply plots them
        eigval = self.get_eigval(attribute=False)

        # Also make sure ipr is defined
        if self.ipr is None:
            ipr = self.calc_ipr(attribute=False)
        else:
            ipr = self.ipr

        # Register cmap if necessary
        if inverse_PR:
            if 'viridis' not in plt.colormaps():
                cmaps.register_colormaps()

            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'haldane', alpha=alpha,
                                                                           colorV=ipr, colormap='viridis', linewidth=0,
                                                                           cax_label=r'$p^{-1}$', climbars=True)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()

            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'haldane', alpha=alpha,
                                                                           colorV=1./ipr, colormap='viridis_r',
                                                                           linewidth=0, cax_label=r'$p$',
                                                                           climbars=False)

        plt.title(title, fontsize=FSFS)
        if save:
            if outdir is None:
                outdir = dio.prepdir(self.lp['meshfn'])
            plt.savefig(outdir + fname + self.lp['meshfn_exten'] + '.png')

        if show:
            plt.show()
        else:
            return DOS_ax

    def plot_ill_dos(self, save=True, show=False, dos_ax=None, cbar_ax=None, alpha=1.0, vmin=None, vmax=None, **kwargs):
        """
        Parameters
        ----------
        save : bool
        show : bool
        dos_ax : axis instance
        cbar_ax : axis instance
        alpha : float
            opacity of the colored bars on the histogram
        vmin : float or None
            minimum value for coloring the histogram bars
        vmax : float or None
            maximum value for coloring the histogram bars
        **kwargs : keyword arguments for colored_DOS_plot
        """
        dos_ax, cbar_ax = hlatpfns.plot_ill_dos(self, dos_ax=dos_ax, cbar_ax=cbar_ax, alpha=alpha, vmin=vmin,
                                                vmax=vmax, climbars=True, **kwargs)
        if save:
            fn = dio.prepdir(self.lp['meshfn']) + 'ill_dos' + lp['meshfn_exten'] + '.png'
            print 'saving ill to: ' + fn
            plt.savefig(fn, dpi=600)
        if show:
            plt.show()
        return dos_ax, cbar_ax

    def plot_DOS(self, outdir=None, title=r'$D(\omega)$ for Haldane network', fname='eigval_haldane_hist',
                 alpha=None, show=True, dos_ax=None, **kwargs):
        """
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        **kwargs : keyword arguments for colored_DOS_plot()
        """
        # First make sure all eigvals are accounted for
        eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        if dos_ax is None:
            fig, dos_ax = leplt.initialize_DOS_plot(eigval, 'haldane', alpha=alpha)
        else:
            ipr = self.get_ipr()
            leplt.colored_DOS_plot(eigval, dos_ax, 'haldane', **kwargs)
        dos_ax.set_title(title)
        if outdir is not None:
            # pickle.dump(fig, file(outdir+fname+'.pkl','w'))
            plt.savefig(outdir + fname + '.png')
        if show:
            plt.show()

    def plot_ldos(self, eigval=None, ldos=None, outdir=None, FSFS=12):
        """Plot the local density of states for the system.

        """
        if ldos is None:
            try:
                self.ldos[0]
                ldos = self.ldos
            except IndexError:
                # Don't attribute the ldos for a method that simply plots it
                ldos = self.get_ldos(attribute=False)
        if eigval is None:
            try:
                self.eigval[0]
                eigval = self.eigval
            except IndexError:
                # Don't attribute the eigenvectors and eigenvalues for a method that simply plots them
                eigval, eigvect = self.eig_vals_vects(attribute=False)

        print 'saving to outdir = ', outdir
        fig, DOS_ax = hlatpfns.draw_ldos_plots(eigval, ldos, hlat, outdir=outdir, FSFS=FSFS)

    def plot_localization(self, localization=None, eigval=None, eigvect=None,
                          outdir=None, fname='localization', alpha=None, fontsize=12):
        """Plot eigenvector normal modes of system with overlaid fits of exponential localization.

        Parameters
        ----------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
            where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))
        eigval :
        eigvect :
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        fname : str
            Name of the image file to store, aside from index number formed from the index of the normal mode
        alpha : float
            opacity of the localized fit heatmap overlay
        FSFS : int
            Font size for the text in the plot
        """
        # First make sure all eigvals are accounted for
        if eigval is None and eigvect is None:
            # Don't attribute the eigenvectors and eigenvalues for a method that simply plots them
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        if localization is None:
            localization = self.get_localization(attribute=False)
        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn']) + 'localization' + self.lp['meshfn_exten'] + '/'
        fig, DOS_ax, ax = hlatpfns.draw_localization_plots(self, localization, eigval, eigvect, outdir=outdir,
                                                           alpha=alpha, fontsize=fontsize)

    def plot_edge_localization(self, elocz=None, eigval=None, eigvect=None,
                               outdir=None, fname='localization', alpha=None, fontsize=12):
        """Plot eigenvector normal modes of system with overlaid fits of exponential localization.
        where fit is A * exp(K * np.sqrt((x - x_boundary)**2 + (y - y_boundary)**2))

        Parameters
        ----------
        elocz : NP x 5 float array (ie, len(eigval) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        eigval : 2*N x 1 complex array
            eigenvalues of the matrix, sorted by order of imaginary components
        eigvect : typically 2*N x 2*N complex array
            eigenvectors of the matrix, sorted by order of imaginary components of eigvals
            Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
            x0, y0, x1, y1, ... xNP, yNP.
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        fname : str
            Name of the image file to store, aside from index number formed from the index of the normal mode
        alpha : float
            opacity of the localized fit heatmap overlay
        fontsize : int
            Font size for the text in the plot
        """
        # First make sure all eigvals are accounted for
        if eigval is None and eigvect is None:
            # Don't attribute the eigenvectors and eigenvalues for a method that simply plots them
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        if elocz is None:
            elocz = self.get_edge_localization(attribute=False)
        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn']) + 'localization_edge' + self.lp['meshfn_exten'] + '/'
        fig, DOS_ax, ax = hlatpfns.draw_edge_localization_plots(self, elocz, eigval, eigvect,
                                                                outdir=outdir, alpha=alpha, fontsize=fontsize)
        # Turn images into a movie
        imgname = outdir + 'localization_edge' + self.lp['meshfn_exten'] + '_'
        movname = dio.prepdir(self.lp['meshfn']) + 'localization_edge' + self.lp['meshfn_exten']
        lemov.make_movie(imgname, movname, indexsz='06', framerate=7)

    def sum_amplitudes_band_spectrum(self, omegac=None, outdir=None, cmap='viridis_r', deform_difference=False,
                                     deformation='NNNhop'):
        """Sum the normal mode amplitudes at each site over all modes below cutoff frequency.
        If deform_difference is True, then look at how local charges change when conducting deformation specified by
        deformation argument.
        """
        eigval, eigvect = self.get_eigval_eigvect()

        if omegac is None:
            omegac = np.array([np.min(np.real(eigval))-0.01, 0.])

        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn']) + 'charge_haldane' + self.lp['meshfn_exten'] + '/'

        dio.ensure_dir(outdir)
        if cmap not in plt.colormaps():
            lecmap.register_colormaps()

        cmp = plt.get_cmap(cmap)

        Wfig = 224

        jj = 0
        tot_amp = []
        in_sum_list = []
        for omc in omegac:
            in_sum = np.real(eigval) > omc
            print 'np.shape(np.abs(eigvect[in_sum, :])**2 = ', np.shape(np.abs(eigvect[in_sum, :])**2)
            total_amp = np.sum(np.abs(eigvect[in_sum, :])**2, axis=0)
            # plot result
            fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
            [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
            print 'max = ', np.max(total_amp)
            sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp/np.max(total_amp),
                            c=total_amp, cmap=cmp, edgecolor='none')
            maxval = np.max(total_amp)
            minval = np.min(total_amp)
            ticks = [minval, maxval]
            cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', ticks=ticks)
            plt.suptitle(r'Haldane network: $\sum_{\omega > \omega_c} |\psi_i|^2$, with $\omega_c = $' +
                         '{0:0.02f}'.format(omc))
            plt.savefig(outdir + 'charge_' + '{0:05d}'.format(jj) + '.png', dpi=150)
            plt.close('all')
            tot_amp.append(total_amp)
            in_sum_list.append(in_sum)
            jj += 1

        # Make movie (turned off for now)
        if False:
            imgname = outdir + 'charge_'
            imgdir = outdir
            movname = dio.prepdir(self.lp['meshfn']) + 'charge_haldane_omegac' + self.lp['meshfn_exten']
            lemov.make_movie(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False,
                             save_into_subdir=False)

        if deform_difference:
            ######################################
            # DEFORM: NNNhop changes a single NNN hopping term, like a delta function
            ######################################
            xy = self.lattice.xy
            # Shrink a single bond to compute charge difference between these two
            cboxind = np.where(np.logical_and(np.abs(xy[:, 0]) < 0.5, np.abs(xy[:, 1] < 0.5)))[0]
            cind = cboxind[np.argmin(np.abs(xy[cboxind, 1]))]

            # Look for next nearest neighbor with smallest opening angle
            if self.lattice.KLNNN is None:
                self.lattice.get_NNN_info(attribute=True)

            neighbors = self.lattice.NLNNN[cind][np.where(np.abs(self.lattice.KLNNN[cind]))[0]]
            thetas = np.arctan2(xy[neighbors, 1] - xy[cind, 1], xy[neighbors, 0] - xy[cind, 0])
            nind = np.argmin(np.mod(thetas - np.pi * 0.4999, 2 * np.pi))
            nind = neighbors[nind]
            matrix = self.get_matrix()
            if deformation == 'NNNhop':
                # check
                # print('self.lattice.NLNNN[cind][np.where(np.abs(self.lattice.KLNNN[cind]))[0]] = ',
                #       self.lattice.NLNNN[cind][np.where(np.abs(self.lattice.KLNNN[cind]))[0]])
                # print 'neighbors = ', neighbors
                # print 'thetas = ', thetas
                # print 'lp[t2] = ', self.lp['t2']
                # print 'nind = ', nind
                # plt.plot(self.lattice.xy[:, 0], self.lattice.xy[:, 1], 'b.')
                # for ii in range(len(self.lattice.xy[:, 0])):
                #     plt.text(self.lattice.xy[ii, 0]+0.1, self.lattice.xy[ii, 1]+0.1, str(ii))
                #
                # bar = [cind, nind]
                # plt.plot(self.lattice.xy[bar, 0], self.lattice.xy[bar, 1], 'ro-')
                # plt.show()

                # Change NNN hopping angle between cind and nind
                if nind > 0:
                    # This means we haven't somehow grabbed a periodic bond
                    print 'hopping = ', matrix[cind, nind]
                    print 'back hopping = ', matrix[nind, cind]
                    matrix[cind, nind] *= 10.
                    matrix[nind, cind] *= 10.
                    print 'new hopping = ', matrix[cind, nind]
                    print 'new back hopping = ', matrix[nind, cind]
                else:
                    raise RuntimeError('Grabbed a periodic bond for the NNN hopping deformation. Handle this case')
                    # todo: handle periodic bonds in NNN hopping tuning

            elif deformation == 'pin100':
                matrix[cind, cind] += 100.

            print '\nComputing new eigvals and eigvects with altered matrix...'
            eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=True)

            new_amp = []
            jj = 0
            for in_sum in in_sum_list:
                total_amp = np.sum(np.abs(eigvect[in_sum, :])**2, axis=0)
                # plot result
                fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
                [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
                print 'max = ', np.max(total_amp)
                sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp/np.max(total_amp),
                                c=total_amp, cmap=cmp, edgecolor='none')
                maxval = np.max(total_amp)
                minval = np.min(total_amp)
                ticks = [minval, maxval]
                cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', ticks=ticks)
                plt.suptitle(r'Haldane network: $\sum_{\omega > \omega_c} |\psi_i|^2$, with $\omega_c = $' +
                             '{0:0.02f}'.format(omegac[jj]))
                plt.savefig(outdir + 'newcharge_' + '{0:05d}'.format(jj) + '.png', dpi=150)
                plt.close('all')
                new_amp.append(total_amp)
                jj += 1

            # plot difference before and after deforming lattice
            diff_amp = []
            jj = 0
            cmp = plt.get_cmap('coolwarm')
            for kk in range(len(omegac)):
                total_amp = new_amp[kk] - tot_amp[kk]
                plt.clf()
                fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0,
                                                                     tspace=10)
                [ax, _] = self.lattice.plot_BW_lat(ax=ax, save=False, title='', cbar_ax=None)
                print 'max = ', np.max(total_amp)
                netchange = np.sum(total_amp)
                sizes = 200 * np.abs(total_amp) / np.max(np.abs(total_amp))
                maxval = np.max(np.abs(total_amp))
                ticks = [-maxval, maxval]
                sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=sizes, edgecolors='none', c=total_amp,
                                cmap=cmp, vmin=ticks[0], vmax=ticks[1], zorder=1000)

                cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', ticks=ticks)
                plt.suptitle(r'Haldane network: $\Delta \left[\sum_{\omega > \omega_c} |\psi_i|^2 \right]$, ' +
                             r'with $\omega_c = $' + '{0:0.02f}'.format(omegac[kk - len(omegac)]) +
                             '\nnet change={0:0.3e}'.format(netchange))
                plt.savefig(outdir + 'dcharge_' + '{0:05d}'.format(jj) + '.png', dpi=150)
                # Highlight negative contributions
                negind = np.where(total_amp < 0.0)[0]
                ax.scatter(self.lattice.xy[negind, 0], self.lattice.xy[negind, 1], s=sizes[negind],
                                edgecolors='none', c='b', zorder=1000)
                print 'saving figure: ', outdir + 'dcharge_' + '{0:05d}'.format(jj) + '_negatives.png'
                plt.savefig(outdir + 'dcharge_' + '{0:05d}'.format(jj) + '_negatives.png', dpi=150)
                plt.close('all')
                diff_amp.append(total_amp)
                jj += 1

            return [total_amp, new_amp, diff_amp]
        else:
            return total_amp

    def plot_localized_state_polygons(self, thres=None, frange=None, save=True, show=False, outdir=None):
        """
        Parameters
        ----------
        thres : None or float array or float
            threshold for inverse localization length for plotting a mode
        save : bool
            Save the images, rather than just plot them
        show : bool
            Pause for 5 seconds on displaying each image (after saving, if save is True)
        """
        # Get eigval, eigvect, and localization. Convert localization to ill
        eigval, eigvect = self.get_eigval_eigvect()
        localization = self.get_localization(eigval=eigval, eigvect=eigvect)
        ill = localization[:, 3]

        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn'])

        if frange is not None:
            addstr = '_frange{0:0.3f}'.format(frange[0]).replace('.', 'p')
            addstr += '_{0:0.3f}'.format(frange[1]).replace('.', 'p')
            freqs = np.imag(eigval[int(len(eigval)*0.5):len(eigval)])
            ev_in_range = np.logical_and(freqs > frange[0], freqs < frange[1])
        else:
            addstr = ''

        imgdir = outdir + 'illpolygons' + self.lp['meshfn_exten'] + addstr + '/'
        dio.ensure_dir(imgdir)

        if thres is None:
            if frange is not None:
                thres = np.arange(0.08, np.max(ill[ev_in_range]), 0.01)
            else:
                thres = np.arange(0.08, np.max(ill), 0.01)
        elif isinstance(thres, float):
            thres = np.array([thres])

        fig, dos_ax, ax = \
            leplt.initialize_eigvect_DOS_header_plot(eigval, self.lattice.xy, sim_type='haldane',
                                                     preset_cbar=False, orientation='portrait',
                                                     cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                                     colorV=ill, colormap='viridis',
                                                     linewidth=0, cax_label=r'$\lambda^{-1}$')

        self.lattice.plot_BW_lat(fig=fig, ax=ax, save=False, close=False, axis_off=False, title='', ptcolor=None)

        # get max number of polygons
        self.lattice.get_polygons()
        maxpno = np.max(np.array([len(polyg) - 1 for polyg in self.lattice.polygons], dtype=int))
        cmap = plt.get_cmap('jet')

        # plot unusual polygons
        for nsides in [4, 5] + range(7, maxpno + 1):
            lfns.plot_polygons_with_nsides(lat, nsides, ax, color=cmap(float(nsides)/maxpno), alpha=0.5)

        kk = 0
        for thr in thres:
            print 'Plotting excitations for ill threshold = ' + str(thr) + '...'
            if frange is not None:
                todo = np.where(np.logical_and(ill > thr, ev_in_range))[0]
            else:
                todo = np.where(ill > thr)[0]
            pplist = []
            f_marklist = []
            lineslist = []
            scatlist = []
            dmyk = 0
            for ii in todo:
                if dmyk % 50 == 0:
                    print 'adding excitation #' + str(dmyk) + ' of ' + str(len(todo))
                fig, [scat_fg, pp, f_mark, lines12_st] = \
                    leplt.plot_eigvect_excitation(self.lattice.xy, fig, dos_ax, ax, eigval, eigvect,
                                                  ii+len(self.lattice.xy), marker_num=0,
                                                  black_t0lines=False, mark_t0=False, title=None)
                pplist.append(pp)
                f_marklist.append(f_mark)
                dmyk += 1

            if frange is not None:
                frangestr = r' for $\omega$ $\in$ [' + '{0:0.3f}'.format(frange[0]) + \
                            ', {0:0.3f}'.format(frange[1]) + ']'
            else:
                frangestr = ''

            ax.set_title(r'Excitations with $\lambda^{-1} >$' + '{0:0.3f}'.format(thr) + frangestr)

            if save:
                print 'saving figure...'
                plt.savefig(imgdir + 'illpolygons_{0:07d}'.format(kk) + '.png')
            if show:
                plt.pause(5)
            # cleanup:
            # Note that need not remove scat_fg because it is returned as empty list (mark_t0 is False)
            # Note that need not remove lines12_st because it is returned as empty list (black_t0lines is False)
            for pp in pplist:
                pp.remove()
                del pp
            for f_mark in f_marklist:
                f_mark.remove()
                del f_mark
            if lineslist:
                for lines12_st in lineslist:
                    lines12_st.remove()
                    del lines12_st
            if scatlist:
                for scat_fg in scatlist:
                    scat_fg.remove()
                    del scat_fg
            kk += 1

        if save:
            imgname = imgdir + 'illpolygons_'
            movname = outdir + 'ill_polygons' + self.lp['meshfn_exten'] + addstr
            lemov.make_movie(imgname, movname, indexsz='07', framerate=05)

        return fig, ax


if __name__ == '__main__':
    '''Use the HaldaneLattice class to create a density of states, compute participation ratio binned by polygons,
    or build a lattice with the DOS

    Example usage:
    python run_series.py -pro haldane_lattice_class -opts LT/hexagonal/-shape/hexagon/-N/8/-pin/0.0/-Vpin/0.1/-load_and_resave -var t2 0.0:0.05:0.4
    python haldane_lattice_class.py -LT hexagonal -shape hexagon -N 8 -pin 0.0 -Vpin 0.75 -load_and_resave
    python haldane_lattice_class.py -save_lattice -LT hexagonal -shape hexagon -periodic -N 10 -pin 1.0
    python haldane_lattice_class.py -load_and_resave -LT hexagonal -shape hexagon -periodic -N 10 -pin 10.0
    python haldane_lattice_class.py -save_ipr -LT hexagonal -shape hexagon -N 20
    python haldane_lattice_class.py -gap_scaling -LT hexagonal -shape hexagon -periodic -N 10

    To make the NNN hoppings based on bond angles but only keep imaginary part, use:
    python haldane_lattice_class.py -load_and_resave -LT hexagonal -shape hexagon -N 10 -t2angles -pureimNNN

    '''
    import lepm.lattice_class as lattice_class
    import lepm.plotting.colormaps as lecmaps
    from matplotlib.collections import LineCollection

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify parameters HaldaneLattice instance creation.')
    parser.add_argument('-test_hlat', '--test_hlat',
                        help='Construct HaldaneLattice to test that it works/save pinning distribution',
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
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
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
          'periodicBC': args.periodicBC or args.periodic_strip,
          'periodic_strip': args.periodic_strip,
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
          }

    # Loading lattice example:
    if args.nice_plot:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        print 'Saving nice BW plot...'
        lat.plot_BW_lat(meshfn=lat.lp['meshfn'], ptcolor='k', ptsize=15)

    if args.save_prpoly:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        hlat = HaldaneLattice(lat, lp)
        hlat.load()
        # hlat.calc_pr_polygons(check=False)
        hlat.save_prpoly(save_plot=True)

    if args.plot_pin:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'Saving pinning scatterplot...'
        hlat.plot_pin()

    if args.plot_charge:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        hlat = HaldaneLattice(lat, lp)
        hlat.load()
        hlat.sum_amplitudes_band_spectrum(omegac=None, deform_difference=True)

    if args.plot_localized_states:
        """Load a periodic lattice from file, provide physics, and plot the modes which are localized, along with
        polygons.

        Example Usage:
        python haldane_lattice_class.py -LT randomcent -shape square -N 20 -conf 02 -periodic -illpoly -frange 0.0/3.5
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        if args.freq_range != '0/0':
            frange = le.string_to_array(args.freq_range, dtype=float)
        else:
            frange = None
        hlat.plot_localized_state_polygons(save=True, frange=frange)

    if args.dcdisorder:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        print 'loading lattice...'
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        # hlat.load()
        print 'Computing eigvals/vects...'
        hlat.get_eigval_eigvect(attribute=True)
        hlat.save_eigval_eigvect()
        hlat.save_pin()
        print 'Saving DOS movie...'
        hlat.save_DOSmovie()

    if args.make_DOSmovie:
        print 'lp[periodic_strip] = ', lp['periodic_strip']
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'loading lattice...'
        hlat.load()
        print 'Saving DOS movie...'
        hlat.save_DOSmovie()

    if args.save_ipr:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'saving ipr...'
        hlat.save_ipr(show=args.show)
        # hlat.plot_ipr_DOS(save=True)

    if args.load_lattice_resave_physics:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'saving eigvals/vects...'
        hlat.save_eigval_eigvect()
        print 'saving ipr...'
        hlat.save_ipr(show=args.show)
        # print 'Saving DOS movie...'
        # hlat.save_DOSmovie()

    if args.test_hlat:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'Tested lattice. Done!'

    if args.save_lattice:
        """Construct a network and save lattice and the physics"""
        lat = lattice_class.Lattice(lp)
        print 'Building lattice...'
        lat.build()
        hlat = HaldaneLattice(lat, lp, t1='auto', pin='auto')
        print 'Saving haldane lattice...'
        hlat.lattice.save()
        print 'Computing eigvals/vects...'
        hlat.eig_vals_vects(attribute=True)
        hlat.save_eigval_eigvect()
        print 'Saving DOS movie...'
        hlat.save_DOSmovie()

    if args.load_calc_ldos:
        """Load a (Haldane) lattice from file, calc the local density of states and save it
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        hlat.save_ldos()

    if args.localization:
        """Load a periodic lattice from file, provide physics, and seek exponential localization of modes

        Example usage:
        python run_series.py -pro haldane_lattice_class -opts LT/hucentroid/-periodic/-NP/20/-localization/-save_eig -var AB 0.0:0.05:1.0
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        hlat.save_localization(attribute=True, save_images=args.save_images, save_eigvect_eigval=args.save_eig)
        hlat.plot_ill_dos()

    if args.gap_scaling:
        """Visualize the eigvectors near the gap as we approach the strong pinning limit
        (originally done for gyro lattices)"""
        par1 = 'delta'
        par1v = np.pi * np.arange(0.667, 1.25, 0.3)
        par2 = 'pin'
        par2v = np.arange(0.0, 101., 25.)**2
        if args.N == 10:
            print 'set N == 10: choosing ev1, ev2'
            ev1 = 299
            ev2 = 300
        elif args.N == 8:
            print 'set N == 8: choosing ev1, ev2'
            ev1 = 191
            ev2 = 192
        elif args.N == 6:
            print 'set N == 6: choosing ev1, ev2'
            ev1 = 107
            ev2 = 108

        gapsz = np.zeros((len(par1v), len(par2v)), dtype=float)
        eigvals = np.zeros((2, len(par2v)), dtype=float)
        kk = 0
        for p1 in par1v:
            lp[par1] = p1
            lat = lattice_class.Lattice(lp)
            print 'Building lattice...'
            lat.build()

            diff = np.zeros(len(par2v))
            jj = 0
            for p2 in par2v:
                lp[par2] = p2
                hlat = HaldaneLattice(lat, lp)
                eigval, eigvect = hlat.eig_vals_vects(attribute=False)
                print 'eigval[ev1] = ', eigval[ev1]
                print 'eigval[ev2] = ', eigval[ev2]
                sys.exit()
                eigvals[0, jj] = np.imag(eigval[ev1])
                eigvals[1, jj] = np.imag(eigval[ev2])
                diff[jj] = np.abs(np.imag(eigval[ev1] - eigval[ev2]))
                jj += 1

            gapsz[kk] = diff
            plt.plot(par2v, np.log10(abs(diff)), '.-', label=str(par1v))
            plt.xlabel(r'$\Omega_g$')
            plt.ylabel(r'Gap width $\log_{10} W$')
            # plt.pause(0.01)
            plt.show()

            plt.clf()
            print 'par2v = ', par2v
            print 'eigvals = ', eigvals
            plt.plot(par2v, eigvals[0], '.-')
            plt.plot(par2v, eigvals[1], '.-')
            plt.show()

            kk += 1

        plt.show()

    # Building lattice example:
    # lat.build()
    # hlat = HaldaneLattice(lat,lat.lp)
    # ipr = hlat.calc_ipr(attribute=True)
    # hlat.plot_ipr_DOS(outdir='/Users/npmitchell/Desktop/')
    # prpoly = hlat.calc_pr_polygons(check=False)

    # for ii in prpoly:
    #     fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(hlat.eigval, 'haldane', pin=-5000, alpha=1.0,
    #                                                                    colorV=prpoly[ii]*ipr,
    #                                                                    colormap='viridis', linewidth=0,
    #                                                                    cax_label=r'$p^{-1}$')
    #     plt.savefig('/Users/npmitchell/Desktop/prpoly_'+str(ii)+'.png')
    #     plt.close('all')

    # hlat.plot_ipr_DOS(inverse_PR=False)
    # plt.plot(np.imag(hlat.eigval))
    # plt.show()
    # plt.plot(np.imag(hlat.eigval), hlat.ipr, 'b.')
    # plt.title('Inverse participation Ratio')
    # plt.ylabel('ipr')
    # plt.xlabel(r'$\omega$')
    # plt.show()

    if args.twistbcs:
        # Make sure bcs are periodic
        """Load a periodic lattice from file, twist the BCs by phases theta_twist and phi_twist with vals finely spaced
        between 0 and 2pi. Then compute the berry curvature associated with |alpha(theta, phi)>

        Example usage:
        python haldane_lattice_class.py -twistbcs -N 3 -LT hexagonal -shape square -periodic
        """
        lp['periodicBC'] = True
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        # Make a big array of the eigvals: N x N x thetav x phiv
        # thetav = np.arange(0., 2. * np.pi, 0.5)
        # phiv = np.arange(0., 2. * np.pi, 0.5)

        # First just test for two values of theta and two of phi
        thetav = [0., 0.01]
        phiv = [0., 0.01]
        eigvects = {}
        ii = 0
        for theta in thetav:
            eigvects[ii] = {}
            jj = 0
            for phi in phiv:
                lpnew = copy.deepcopy(lp)
                lpnew['theta_twist'] = theta
                lpnew['phi_twist'] = phi
                hlat = HaldaneLattice(lat, lpnew)
                ev, evec = hlat.get_eigval_eigvect(attribute=True)
                if ii == 0 and jj == 0:
                    eigval = copy.deepcopy(ev)
                    ill = hlat.get_ill()
                eigvects[ii][jj] = evec
                jj += 1
            ii += 1

        # Dot the derivs of eigvects together
        # Ensure that there is a nonzero-amplitude wannier with matching phase
        dtheta = eigvects[1][0] - eigvects[0][0]
        dphi = eigvects[0][1] - eigvects[0][0]

        thetamov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
        if not glob.glob(thetamov_fn):
            # Plot differences
            fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                           colorV=ill, colormap='viridis',
                                                                           linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                           cbar_nticks=2,
                                                                           xlabel_pad=10, ylabel_pad=10,
                                                                           cbar_tickfmt='%0.3f')
            DOS_ax.set_title(r'$\partial_\phi \psi$')
            hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
            outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '/'
            dio.ensure_dir(outdir)
            for ii in range(len(ev)):
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                                dtheta, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                                marker_num=0, color_scheme='default', normalization=1.)
                print 'saving theta ', ii
                plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()

            plt.close('all')

            imgname = outdir + 'dos_ev'
            movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev'
            lemov.make_movie(imgname, movname, rm_images=False)

        phimov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
        if not glob.glob(phimov_fn):
            fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                           colorV=ill, colormap='viridis',
                                                                           linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                           cbar_nticks=2,
                                                                           xlabel_pad=10, ylabel_pad=10,
                                                                           cbar_tickfmt='%0.3f')
            DOS_ax.set_title(r'$\partial_\phi \psi$')
            outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '/'
            dio.ensure_dir(outdir)
            for ii in range(len(ev)):
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                                dphi, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                                marker_num=0, color_scheme='default', normalization=1.)
                print 'saving phi ', ii
                plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()

            plt.close('all')

            imgname = outdir + 'dos_ev'
            movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev'
            lemov.make_movie(imgname, movname, rm_images=False)

        # Check
        # print 'shape(dtheta) = ', np.shape(dtheta)
        # print 'shape(dphi) = ', np.shape(dphi)
        # le.plot_complex_matrix(dtheta, show=True, name='dtheta')
        # le.plot_complex_matrix(dphi, show=True, name='dphi')

        fig, ax = leplt.initialize_nxmpanel_fig(4, 1, wsfrac=0.6, x0frac=0.3)
        # < dphi | dtheta >
        dpdt = np.einsum('ij...,ij...->i...', dtheta, dphi.conj())
        # < dtheta | dphi >
        dtdp = np.einsum('ij...,ij...->i...', dphi, dtheta.conj())
        print 'dtdp = ', dtdp
        ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
        ax[1].plot(np.arange(len(dpdt)), dpdt, '-')
        hc = 2. * np.pi * 1j * (dpdt - dtdp)
        ax[2].plot(np.arange(len(lat.xy)), hc, '.-')
        # Plot cumulative sum
        sumhc = np.cumsum(hc)
        ax[3].plot(np.arange(len(lat.xy)), sumhc, '.-')

        ax[2].set_xlabel(r'Eigvect number')
        ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
        ax[2].set_xlabel(r'Eigvect number')
        ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
        ax[2].set_xlabel(r'Eigvect number')
        ax[2].set_ylabel(r'$c_H$')
        ax[3].set_xlabel(r'Eigvect number')
        ax[3].set_ylabel(r'$\sum_{E_\alpha < E_c} c_H$')
        outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/twistbc_test/'
        dio.ensure_dir(outdir)
        print 'saving ', outdir + 'test' + hlat.lp['meshfn_exten'] + '.png'
        plt.savefig(outdir + 'test' + hlat.lp['meshfn_exten'] + '.png')

        #### Now do the same thing but with different values of theta, phi
        # First just test for two values of theta and two of phi
        thetav = [1., 1.01]
        phiv = [1., 1.01]
        eigvects = {}
        ii = 0
        for theta in thetav:
            eigvects[ii] = {}
            jj = 0
            for phi in phiv:
                lpnew = copy.deepcopy(lp)
                lpnew['theta_twist'] = theta
                lpnew['phi_twist'] = phi
                hlat = HaldaneLattice(lat, lpnew)
                ev, evec = hlat.get_eigval_eigvect(attribute=True)
                if ii == 0 and jj == 0:
                    eigval = copy.deepcopy(ev)
                    ill = hlat.get_ill()
                eigvects[ii][jj] = evec
                jj += 1
            ii += 1

        # Dot the derivs of eigvects together
        dtheta = eigvects[1][0] - eigvects[0][0]
        dphi = eigvects[0][1] - eigvects[0][0]

        # Plot differences
        thetamov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
        if not glob.glob(thetamov_fn):
            fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                           colorV=ill, colormap='viridis',
                                                                           linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                           cbar_nticks=2,
                                                                           xlabel_pad=10, ylabel_pad=10,
                                                                           cbar_tickfmt='%0.3f')
            DOS_ax.set_title(r'$\partial_\theta \psi$')
            hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
            outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '/'
            dio.ensure_dir(outdir)
            for ii in range(len(ev)):
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                                dtheta, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                                marker_num=0, color_scheme='default', normalization=1.)
                print 'saving theta ', ii
                plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()

            plt.close('all')

            imgname = outdir + 'dos_ev'
            movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev'
            lemov.make_movie(imgname, movname, rm_images=False)

        # Now do phi
        phimov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
        if not glob.glob(phimov_fn):
            fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                           colorV=ill, colormap='viridis',
                                                                           linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                           cbar_nticks=2,
                                                                           xlabel_pad=10, ylabel_pad=10,
                                                                           cbar_tickfmt='%0.3f')
            DOS_ax.set_title(r'$\partial_\phi \psi$')
            hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
            outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '/'
            dio.ensure_dir(outdir)
            for ii in range(len(ev)):
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                                dphi, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                                marker_num=0, color_scheme='default', normalization=1.)
                print 'saving phi ', ii
                plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()

            plt.close('all')

            imgname = outdir + 'dos_ev'
            movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev'
            lemov.make_movie(imgname, movname, rm_images=False)

        # Check
        # print 'shape(dtheta) = ', np.shape(dtheta)
        # print 'shape(dphi) = ', np.shape(dphi)
        # le.plot_complex_matrix(dtheta, show=True, name='dtheta')
        # le.plot_complex_matrix(dphi, show=True, name='dphi')

        fig, ax = leplt.initialize_nxmpanel_fig(4, 1, wsfrac=0.6, x0frac=0.3)
        # < dphi | dtheta >
        dpdt = np.einsum('ij...,ij...->i...', dtheta, dphi.conj())
        # < dtheta | dphi >
        dtdp = np.einsum('ij...,ij...->i...', dphi, dtheta.conj())
        print 'dtdp = ', dtdp
        ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
        ax[1].plot(np.arange(len(dpdt)), dpdt, '-')
        hc = 2. * np.pi * 1j * (dpdt - dtdp)
        ax[2].plot(np.arange(len(lat.xy)), hc, '.-')
        # Plot cumulative sum
        sumhc = np.cumsum(hc)
        ax[3].plot(np.arange(len(lat.xy)), sumhc, '.-')

        ax[2].set_xlabel(r'Eigvect number')
        ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
        ax[2].set_xlabel(r'Eigvect number')
        ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
        ax[2].set_xlabel(r'Eigvect number')
        ax[2].set_ylabel(r'$c_H$')
        ax[3].set_xlabel(r'Eigvect number')
        ax[3].set_ylabel(r'$\sum_{E_\alpha < E_c} c_H$')
        outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/twistbc_test/'
        dio.ensure_dir(outdir)
        plt.savefig(outdir + 'test' + hlat.lp['meshfn_exten'] + '_theta1p0.png')




        sys.exit()
        ########################################
        # Now test for more theta values, more phi values
        thetav = np.arange(0., 0.14, 0.1)
        phiv = np.arange(0., 0.14, 0.1)
        eigvects = np.zeros((len(lat.xy), len(lat.xy), len(thetav), len(phiv)), dtype=complex)
        ii = 0
        for theta in thetav:
            jj = 0
            for phi in phiv:
                lpnew = copy.deepcopy(lp)
                lpnew['theta_twist'] = theta
                lpnew['phi_twist'] = phi
                hlat = HaldaneLattice(lat, lpnew)
                ev, evec = hlat.get_eigval_eigvect()
                eigvects[:, :, ii, jj] = evec
                jj += 1
            ii += 1

        # Dot the derivs of eigvects together
        print 'eigvects = ', eigvects
        dtheta = np.diff(eigvects, axis=2)
        dphi = np.diff(eigvects, axis=3)
        print 'dtheta = ', dtheta

        print 'shape(dtheta) = ', np.shape(dtheta)
        print 'shape(dphi) = ', np.shape(dphi)
        le.plot_complex_matrix(dtheta[:, :, 0, 0], show=True)
        le.plot_complex_matrix(dphi[:, :, 0, 0], show=True)

        dtheta = dtheta[:, :, :, 0:np.shape(dtheta)[3] - 1]
        dphi = dphi[:, :, 0:np.shape(dphi)[2] - 1, :]
        print 'shape(dtheta) = ', np.shape(dtheta)
        print 'shape(dphi) = ', np.shape(dphi)

        for ii in range(np.shape(dtheta)[-1]):
            le.plot_complex_matrix(dtheta[:, :, ii, 0], show=True)

        fig, ax = leplt.initialize_nxmpanel_fig(3, 1)
        for ii in range(np.shape(dphi)[-1]):
            for jj in range(np.shape(dphi)[-1]):
                # < dphi | dtheta >
                dpdt = np.dot(dtheta[:, :, ii, jj], dphi[:, :, ii, jj].conj().T)
                # < dtheta | dphi >
                dtdp = np.dot(dphi[:, :, ii, jj], dtheta[:, :, ii, jj].conj().T)
                print 'np.shape(dpdt) = ', np.shape(dpdt)
                print 'np.shape(dtdp) = ', np.shape(dtdp)
                ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
                ax[1].plot(np.arange(len(dpdt)), dpdt, '-')

                hc = 2. * np.pi * 1j * (dpdt - dtdp)
                ax[2].plot(np.arange(len(lat.xy)), hc, '.-')

        ax[0].set_xlabel(r'$\theta$')
        ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
        ax[1].set_xlabel(r'$\phi$')
        ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
        ax[2].set_xlabel(r'Eigvect number')
        ax[2].set_ylabel(r'$\partial_{\phi} \alpha_i$')
        plt.show()
        # hc = hlat.hallconductance_twist()

    if args.dispersion:
        """Example usage:
        python ./haldane/haldane_lattice_class.py -LT hexagonal -shape square -periodic -N 3 -dispersion
        python ./haldane/haldane_lattice_class.py -LT hexagonal -shape square -periodic_strip -N 5 -dispersion
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        hlat.infinite_dispersion()

    if args.twiststrip:
        """Example usage:
        python ./haldane/haldane_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 14 -NV 7 -twiststrip
        """
        print 'lp[periodic_strip] = ', lp['periodic_strip']
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lpmaster = copy.deepcopy(lp)
        lat = lattice_class.Lattice(lp)
        lat.load()
        # Set vmax to be 2/L, where L is the full width of the sample in y (the strip width)
        lsize = 2. * np.max(np.abs(lat.xy[:, 1]))
        vmax = 2. / lsize
        eigvals = []
        loczs = []
        # Note that theta is in units of pi
        thetavals = np.linspace(0, 2., 26)
        print 'thetavals = ', thetavals

        for thetatwist in thetavals:
            lpmaster['theta_twist'] = thetatwist
            hlat = HaldaneLattice(lat, lpmaster)
            print 'loading haldane_lattice...'
            hlat.load()
            print 'Saving DOS movie...'
            hlat.ensure_eigval_eigvect(attribute=True)
            eigvals.append(hlat.get_eigval())
            # get the localization of these eigenvalues
            locz = hlat.get_edge_ill(attribute=True)
            loczs.append(locz)

        # Plot the spectra as function of thetatwist
        fig, ax, cax = leplt.initialize_1panel_cbar_fig()
        lecmaps.register_colormaps()
        cmap = plt.get_cmap('viridis')
        vmin = 0.
        ind = 0
        ii = 0
        # Get the spacing between x values
        dval = abs(thetavals[1] - thetavals[0])
        # Prepare to tally to find the largest frequency on the plot
        maxfreq = 0.
        minfreq = 0.
        for val in thetavals:
            ep0 = zip(val * np.ones(len(eigvals[ii])), eigvals[ii])
            ep1 = zip((val + dval) * np.ones(len(eigvals[ii])), eigvals[ii])
            lines = [list(a) for a in zip(ep0, ep1)]

            maxfreq = max(maxfreq, np.max(np.real(eigvals[ii])))
            minfreq = min(minfreq, np.min(np.real(eigvals[ii])))

            # Define colors based on inverse localization length
            colors = cmap(loczs[ii] / float(vmax))
            print 'np.shape(colors) = ', np.shape(colors)

            # print 'colors = ', colors
            lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))

            # lc.set_array(colors)
            ax.add_collection(lc)
            ii += 1

        ax.set_ylim(minfreq - 0.1, maxfreq + 0.1)
        ax.set_xlim(np.min(thetavals), np.max(thetavals) + dval)
        ax.set_ylabel('frequency, $\omega$')
        ax.set_xlabel(leplt.param2description('thetatwist'))

        sm = leplt.empty_scalar_mappable(vmin, vmax, cmap)
        cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 1. / lsize, 2. / lsize])
        cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
        cb.set_label(r'$\xi^{-1}$', labelpad=10, rotation=0, fontsize=8, va='center')

        fname = dio.prepdir(lat.lp['meshfn']) + 'haldane_thetatwistsweep_nthetas' + str(len(thetavals))
        plt.suptitle(r'Spectra for $N = $' + str(lp['NH']) + ' ' + lp['LatticeTop'] + '\n' +
                     hlatfns.nnn_hopping_description(hlat))
        print 'saving figure: ' + fname + '.png'
        plt.savefig(fname + '.png', dpi=300)
        plt.close('all')

    if args.edgelocalization:
        """Example usage
        python ./haldane/haldane_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 14 -NV 7 \
            -edgelocalization
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        hlat = HaldaneLattice(lat, lp)
        print 'loading haldane_lattice...'
        hlat.load()
        # get the localization of these eigenvalues
        locz = hlat.plot_edge_localization()
