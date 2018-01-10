import numpy as np
import lepm.lattice_elasticity as le
import lepm.gyro_lattice_functions as glatfns
import lepm.gyro_lattice_functions_localization as glocfns
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.hdf5io as h5io
import lepm.lattice_functions as lfns
import lepm.plotting.plotting as leplt
import lepm.plotting.movies as lemov
import lepm.plotting.colormaps as cmaps
import lepm.plotting.gyro_lattice_plotting_functions as glatpfns
import lepm.plotting.science_plot_style as sps
import lepm.plotting.colormaps as lecmap
import lepm.plotting.movies as lemov
import os
import sys
import argparse
import glob
import copy
import lepm.stringformat as sf
import lepm.gyro_lattice_kspace_functions as glatkspacefns
try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm
import h5py


'''
Generate a GyroLattices using the GyroLattice class. This takes a lattice (points and connections) as input, as well as
a dictionary of parameters to define the physics on that lattice.

If OmK is not none, lp['OmKspec'] will be defined accordingly.
If the supplied OmK == float * KL, then Omkspec is not appended to meshfn_exten.
If lp['OmKspec'] is defined and no OmK is supplied,
then the OmKspec specifier will be translated into an OmK.
Similarly, for Omg.

To save network data in hdf5, use names:
omg_configs.hdf5
eigvals_gyro.hdf5
'''


class GyroLattice:
    """Create a gyroscopic network from an instance of the lattice class: gyroscopes connected by springs.
    Note that the spring rest length may be tuned here (not in the Lattice class), by specifying lp['bl0'] (float) or
    bL (float array). Non-standard physics such as this updates the lp['meshfn_exten'] string, which specifies
    an 'extension' to the 'mesh file name'.

    Attributes
    ----------
    lattice : lattice class instance (has xy, NL, KL, etc). Can be empty so that it can be loaded or built
    bL : bond rest length array, default is None if spring rest lengths are equal to distances between particles
    OmK: spring constant connectivity
    Omg: pinning frequencies
    matrix : dynamical matrix, or None
    eigvect : eigvenvectors of dynamical matrix, or None if too bulky to keep in RAM
    eigval : eigenvalues of dynamical matrix, or None if too bulky to keep in RAM
    IPM: inverse participation ratio of the gyroscopic modes, or None if too bulky to keep in RAM
    ldos : local density of states
    localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K

    Attributes of the lattice are:
        xy, BL, NL, KL, LVUC, LV, UC, Pvxydict, LL, BBox, and lp
    lp is a dictionary ('lattice parameters') which can contain:
        LatticeTop, shape, NH, NV, 
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
    def __init__(self, lattice, lp, OmK=None, Omg=None, bL=None, matrix=None, eigvect=None, eigval=None,
                 magevecs=None, ipr=None, prpoly=None, ldos=None,
                 localization=None, edge_localization=None, unit_cell=None, eps=1e-8):
        """Create a lattice instance.

        Properties
        ----------
        lattice : lattice class instance (has xy, NL, KL, etc)
        matrix : dynamical matrix
        OmK :
        Omg :
        bL :
        matrix :
        eigvect : typically 2*N x 2*N complex array
            eigenvectors of the matrix, sorted by order of imaginary components of eigvals
            Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
            x0, y0, x1, y1, ... xNP, yNP.
        eigval : 2*N x 1 complex array
            eigenvalues of the matrix, sorted by order of imaginary components
        ipr :
        propoly :
        ldos
        localization
        edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        unit_cell :

        Returns
        -------
        self : GyroLattice class instance

        """
        self.lattice = lattice
        self.lp = lp

        # Add lattice properties to gyro_lattice_properties, by convention, but don't overwrite params
        for key in self.lattice.lp:
            if key not in self.lp:
                self.lp[key] = self.lattice.lp[key]

        # print 'gryo_lattice_class: self.lp  = ', self.lp
        # print 'gryo_lattice_class: self.lp[Omk]  = ', self.lp['Omk']
        # print 'gryo_lattice_class: self.lp[Omg]  = ', self.lp['Omg']

        self.lp['meshfn_exten'] = ''

        # Check if boundary is fixed in place
        if 'immobile_boundary' in self.lp:
            if self.lp['immobile_boundary']:
                self.lp['meshfn_exten'] += '_immobilebnd'

        #############################################################################
        # Form bL. If bL is None and lp['bl0'] == -5000, then just uses distances btwn particles as rest position
        # If bL is None, uses lp['bl0'] to create self.bL.
        # By default, self.bL is kept as None to save RAM.
        if bL is None:
            if 'bl0' in self.lp:
                # The value of -5000 signifies that some other property determines bL (in this case xy)
                if self.lp['bl0'] > -5000:
                    self.bL = np.ones_like(self.lattice.BL[:, 0], dtype=float) * self.lp['bl0']
                    self.lp['meshfn_exten'] += '_bl0' + sf.float2pstr(lp['bl0'], ndigits=3)
                else:
                    # setting self.bL to None, as lp['bl0'] is zero.
                    self.bL = bL
            else:
                # setting self.bL to None
                self.bL = bL
        else:
            # Using supplied bL as self.bL
            # Check if all the elements are the same --> if so, record value in lp['bl0']
            if (bL == bL[0]).all():
                self.lp['bl0'] = bL[0]
                lp['meshfn_exten'] += '_bl0' + sf.float2pstr(lp['bl0'], ndigits=3)
            else:
                # The value of -5000 signifies that some other property determines bL (in this case self.bL)
                self.lp['bl0'] = -5000
                self.lp['meshfn_exten'] += '_bl0spec'
            self.bL = bL

        #############################################################################
        # Form OmK
        self.OmK, lp_Omk, omk_meshfn_exten = glatfns.build_OmK(self.lattice, self.lp, OmK)
        self.lp['Omk'] = lp_Omk
        self.lp['meshfn_exten'] += omk_meshfn_exten
        print self.lp['Omk']
        print self.lp['meshfn_exten']

        #############################################################################
        # Form Omg
        if Omg == 'auto' or Omg is None:
            if 'Omg' in self.lp:
                self.Omg = self.lp['Omg'] * np.ones_like(self.lattice.xy[:, 0])
            else:
                print 'giving Omg the default value of -1s...'
                self.Omg = -1.0 * np.ones_like(self.lattice.xy[:, 0])
            if self.lp['Omg'] != -1.0:
                self.lp['meshfn_exten'] += '_Omg' + sf.float2pstr(self.lp['Omg'])
        else:
            self.Omg = Omg
            if 'Omg' in self.lp:
                if (self.Omg != self.lp['Omg'] * np.ones(np.shape(self.lattice.xy)[0])).any():
                    self.lp['meshfn_exten'] += '_Omgspec'
                    self.lp['Omg'] = -5000
            else:
                # Check if the values of all elements are identical
                kinds = np.nonzero(self.Omg)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in Omg. Check if all the same
                    value = self.Omg[kinds[0]]
                    if (Omg[kinds] == value).all():
                        self.lp['Omg'] = value
                    else:
                        self.lp['Omg'] = -5000
                else:
                    self.lp['Omg'] = 0.0
        if 'ABDelta' in self.lp:
            print 'glat: here: abdelta = ', self.lp['ABDelta']
            if abs(self.lp['ABDelta']) > 0:
                self.lp['meshfn_exten'] += '_ABd{0:0.3f}'.format(self.lp['ABDelta']).replace('.', 'p').replace('-', 'n')
        else:
            self.lp['ABDelta'] = 0.

        #############################################################################
        # Create meshfn exten for disorder in pin or spring frequencies
        if 'V0_pin_gauss' in self.lp and 'V0_spring_gauss' in self.lp:
            if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_spring_gauss'] > 0:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_pinV' + sf.float2pstr(self.lp['V0_pin_gauss'])
                self.lp['meshfn_exten'] += '_sprV' + sf.float2pstr(self.lp['V0_spring_gauss'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_conf{0:04d}'.format(self.lp['pinconf'])
            else:
                self.lp['dcdisorder'] = False
        else:
            self.lp['V0_pin_gauss'] = 0.
            self.lp['V0_spring_gauss'] = 0.
            self.lp['pinconf'] = 0
            self.lp['dcdisorder'] = False

        # Note: theta_twist and phi_twist are in units of pi!
        # Append thetatwist and/or phitwist to meshfn_exten
        if 'theta_twist' in lp:
            if abs(lp['theta_twist']) > 1e-15:
                self.lp['meshfn_exten'] += '_thetatw' + sf.float2pstr(lp['theta_twist'], ndigits=5)
        if 'phi_twist' in lp:
            if abs(lp['phi_twist']) > 1e-15:
                self.lp['meshfn_exten'] += '_phitw' + sf.float2pstr(lp['phi_twist'], ndigits=5)

        #############################################################################
        # Load or save disorder
        if abs(self.lp['ABDelta']) > eps or abs(self.lp['V0_pin_gauss'] > eps) \
                or abs(self.lp['V0_spring_gauss'] > eps):
            # In order to load the random (V0) or alternating (AB) pinning sites, look for a txt file with the pinnings
            # that also has specifications in its meshfn exten, but IGNORE other parts of meshfnexten, if they exist.
            # Form abbreviated meshfn exten
            pinmfe = self.get_pinmeshfn_exten()

            print 'Trying to load offset/disorder to pinning frequencies:'
            print dio.prepdir(self.lp['meshfn']) + pinmfe
            # Attempt to load from file
            try:
                self.load_pinning(meshfn=self.lp['meshfn'])
                # self.Omg = np.loadtxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt')
                print 'Loaded ABDelta and/or dcdisordered pinning frequencies.'
            except IOError:
                print 'Could not load ABDelta and/or dcdisordered pinning frequencies, defining them here...'
                # Make Omg from scratch
                if np.abs(self.lp['ABDelta']) > eps:
                    asites, bsites = glatfns.ascribe_absites(self.lattice)
                    self.Omg[asites] += self.lp['ABDelta']
                    self.Omg[bsites] -= self.lp['ABDelta']
                if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_spring_gauss'] > 0:
                    self.add_dcdisorder()

                # Save non-standard Omg
                if 'save_pinning_to_hdf5' in self.lp:
                    if self.lp['save_pinning_to_hdf5']:
                        force_hdf5pin = True
                    else:
                        force_hdf5pin = False
                else:
                    force_hdf5pin = False
                self.save_Omg(infodir=self.lp['meshfn'], histogram=False, force_hdf5=force_hdf5pin)
                print 'glat: saved pinning!'
                # sys.exit()
                # self.plot_Omg()

        #############################################################################
        # Non-essential attributes, updated later if needed
        self.matrix = matrix
        self.eigvect = eigvect
        self.eigval = eigval
        self.magevecs = magevecs
        # there is no real reason to save kspace matrix --> it is fast to create compared to diagonalization
        # self.matrixk = matrixk
        self.ipr = ipr
        self.prpoly = prpoly
        self.ldos = ldos
        self.localization = localization
        self.edge_localization = edge_localization
        self.unit_cell = unit_cell

        # print 'meshfn_exten --> ', self.lp['meshfn_exten']

    def __hash__(self):
        return hash(self.lattice)

    def __eq__(self, other):
        return hasattr(other, 'lattice') and self.lattice == other.lattice

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def load(self, meshfn='auto', loadDOS=False, load_ipr=False):
        """Load a saved lattice into the lattice attribute of the gyro_lattice instance.
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

        if self.Omg is None:
            # SHOULD ALREADY BY LOADED FROM FILE OR CREATED FROM SCRATCH
            if self.lp['V0_pin_gauss'] > 0 or self.lp['ABDelta'] > 0:
                self.load_pinning(meshfn=meshfn)
            else:
                self.Omg = self.lp['Omg'] * np.ones_like(self.lattice.xy[:, 0])

        if self.lp['V0_spring_gauss'] > 0:
            print 'todo: This is not finished'
            sys.exit()
        else:
            self.OmK = self.lp['Omk'] * np.abs(self.lattice.KL)

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

    def load_pinning(self, meshfn=None):
        """Load the Omg vector for this instance of GyroLattice"""
        # First try to load from hdf5 file, if it exists
        # If hdf5 file does not exist or contain the pinning for this meshfn_exten, attempt to load from txt file
        if meshfn is None:
            meshfn = self.lp['meshfn']
        pinning_name = self.get_pinmeshfn_exten()
        pinfn = dio.prepdir(meshfn) + 'omg_configs.hdf5'
        if glob.glob(pinfn):
            with h5py.File(dio.prepdir(meshfn) + 'omg_configs.hdf5', "r") as fi:
                inhdf5 = pinning_name in fi.keys()
                if inhdf5:
                    self.Omg = fi[pinning_name][:]
                    load_from_txt = False
                else:
                    load_from_txt = True
        else:
            load_from_txt = True

        if load_from_txt:
            print 'could not find pinning config from hdf5, opening pinning configs from txt...'
            self.Omg = np.loadtxt(dio.prepdir(meshfn) + pinning_name + '.txt')

    def build(self):
        import lepm.make_lattice
        self.lattice = lepm.make_lattice.build_lattice(self.lattice)

    def get_pinmeshfn_exten(self):
        """Return the name of the file or dataset that stores the pinning frequencies of this network"""
        pinmfe = 'Omg_mean' + sf.float2pstr(self.lp['Omg']) + self.lp['meshfn_exten']
        return pinmfe

    def add_dcdisorder(self):
        """Add gaussian noise to pinning or spring energies (delta-correlated disorder)"""
        # Add gaussian noise to pinning energies
        if self.lp['V0_pin_gauss'] > 0:
            self.Omg += self.lp['V0_pin_gauss']*np.random.randn(len(self.lattice.xy))
        if self.lp['V0_spring_gauss'] > 0:
            print 'This is not done correctly here'
            self.OmK += self.lp['V0_spring_gauss'] * np.random.randn(np.shape(self.lattice.KL)[0],
                                                                     np.shape(self.lattice.KL)[1])
            sys.exit()

    def load_eigval_eigvect(self, attribute=True):
        """Load eigval and eigvect from disk: first try hdf5, then look for pickle"""
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        eigvect_name = "eigvect" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_gyro.hdf5"
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_gyro.hdf5"
        evl_saved_to_hdf5 = h5io.dset_in_hdf5(eigval_name, h5_evl)
        evt_saved_to_hdf5 = h5io.dset_in_hdf5(eigvect_name, h5_evt)

        # If not there, look for pkl files
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        fn_evt = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"

        if evl_saved_to_hdf5 and evt_saved_to_hdf5:
            eigval = h5io.extract_dset_hdf5(eigval_name, h5_evl)
            eigvect = h5io.extract_dset_hdf5(eigvect_name, h5_evt)
        elif glob.glob(fn_evl) and glob.glob(fn_evt):
            with open(fn_evl, "rb") as f:
                eigval = pickle.load(f)
            with open(fn_evt, "rb") as f:
                eigvect = pickle.load(f)
        else:
            return None

        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect

        return eigval, eigvect

    def load_eigval(self, attribute=True):
        """Load eigval from disk: first try hdf5, then look for pickle"""
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_gyro.hdf5"
        evl_saved_to_hdf5 = h5io.dset_in_hdf5(eigval_name, h5_evl)

        # If not there, look for pkl files
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"

        print 'glat.load_eigval(): evl_saved_to_hdf5 = ', evl_saved_to_hdf5

        if evl_saved_to_hdf5:
            eigval = h5io.extract_dset_hdf5(eigval_name, h5_evl)
        elif glob.glob(fn_evl):
            with open(fn_evl, "rb") as f:
                eigval = pickle.load(f)
        else:
            return None

        if attribute:
            self.eigval = eigval

        return eigval

    def load_eigvect(self, attribute=True):
        """Load eigvect from disk: first try hdf5, then look for pickle"""
        # Make eigval name
        eigvect_name = "eigvect" + self.lp['meshfn_exten']
        # First look in eigvects_gyro.hdf5
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_gyro.hdf5"
        evt_saved_to_hdf5 = h5io.dset_in_hdf5(eigvect_name, h5_evt)

        # If not there, look for pkl files
        fn_evt = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"

        if evt_saved_to_hdf5:
            eigvect = h5io.extract_dset_hdf5(eigvect_name, h5_evt)
        elif glob.glob(fn_evt):
            with open(fn_evt, "rb") as f:
                eigvect = pickle.load(f)
        else:
            return None

        if attribute:
            self.eigvect = eigvect

        return eigvect

    def load_magevecs(self):
        """Load magnitude of eigvectors at each site from disk: first try hdf5, then look for pickle"""
        # Make eigval name
        magevecs_name = "magevecs" + self.lp['meshfn_exten']
        # First look in eigvects_gyro.hdf5
        h5_mevt = dio.prepdir(self.lp['meshfn']) + "magevecs_gyro.hdf5"
        print 'looking for ', magevecs_name, ' in ', h5_mevt
        mevt_saved_to_hdf5 = h5io.dset_in_hdf5(magevecs_name, h5_mevt)

        print 'mevt_saved_to_hdf5: ', mevt_saved_to_hdf5

        # If not there, look for pkl files
        fn_mevt = dio.prepdir(self.lp['meshfn']) + "magevecs" + self.lp['meshfn_exten'] + ".pkl"

        if mevt_saved_to_hdf5:
            magevecs = h5io.extract_dset_hdf5(magevecs_name, h5_mevt)
        elif glob.glob(fn_mevt):
            with open(fn_mevt, "rb") as f:
                magevecs = pickle.load(f)
        else:
            return None

        return magevecs

    def load_localization(self, attribute=True):
        """Load eigvect from disk: first try hdf5, then look for pickle"""
        # Make localization name
        locz_name = "localization" + self.lp['meshfn_exten']
        # First look in localization_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_gyro.hdf5"
        saved_to_hdf5 = h5io.dset_in_hdf5(locz_name, h5fn)

        # If not there, look for pkl files
        fn_locz = dio.prepdir(self.lp['meshfn']) + "localization" + self.lp['meshfn_exten'] + ".txt"

        if saved_to_hdf5:
            locz = h5io.extract_dset_hdf5(locz_name, h5fn)
        elif glob.glob(fn_locz):
            locz = np.loadtxt(fn_locz, delimiter=',')
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
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_edge_gyro.hdf5"
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
        if glob.glob(fn):
            with open(fn, "r") as f:
                ipr = pickle.load(f)

            if attribute:
                self.ipr = ipr
            return ipr
        else:
            return None

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

    def get_matrix(self, attribute=False, basis=None):

        if self.matrix is None:
            return self.calc_matrix(attribute=attribute, basis=basis)
        else:
            return self.matrix

    def get_eigval_eigvect(self, attribute=False, basis=None):
        """Return eigval and eigvect, obtaining them by (1) calling from self, (2) loading them, or (3) calculating"""
        # First attempt to return, then attempt to load, then calculate if unavailable
        if self.eigval is not None and self.eigvect is not None:
            eigval = self.eigval
            eigvect = self.eigvect
        else:
            # Try to load eigval and eigvect
            print 'Attempting to load eigval/vect...'
            eigval = self.load_eigval(attribute=attribute)
            eigvect = self.load_eigvect(attribute=attribute)

            if eigval is None or eigvect is None:
                print 'glat.get_eigval_eigvect: Could not load eigval/vect, calculating...'
                # calculate eigval and eigvect
                # Define matrix first to avoid attributing matrix to self
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=False, basis=basis)
                else:
                    matrix = self.matrix
                eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
            else:
                print 'loaded!'
        return eigval, eigvect

    def get_eigval(self, attribute=False):
        """Return eigval, obtaining it by (1) calling from self, (2) loading it, or (3) calculating"""
        # First attempt to return, then attempt to load, then calculate if unavailable
        if self.eigval is not None:
            eigval = self.eigval
        else:
            # Try to load eigval and eigvect
            eigval = self.load_eigval(attribute=attribute)
            if eigval is None:
                print 'glat.get_eigval() Cannot load eigval, computing it...'
                # calculate eigval and eigvect
                matrix = self.get_matrix(attribute=False)
                eigval = self.calc_eigvals(matrix=matrix, attribute=attribute)
            else:
                print 'glat.get_eigval() Loaded eigval...'
        return eigval

    def get_eigvect(self, attribute=False, basis=None):
        """Return eigvect, obtaining them by (1) calling from self, (2) loading them, or (3) calculating"""
        # First attempt to return, then attempt to load, then calculate if unavailable
        if self.eigvect is not None:
            eigvect = self.eigvect
        else:
            # Try to load eigval and eigvect
            print 'Attempting to load eigvect...'
            eigvect = self.load_eigvect(attribute=attribute)
            if eigvect is None:
                print 'glat.get_eigvect: Could not load eigvect, calculating...'
                # calculate eigval and eigvect
                # Define matrix first to avoid attributing matrix to self
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=False, basis=basis)
                else:
                    matrix = self.matrix
                eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
            else:
                print 'loaded!'

        return eigvect

    def get_magevecs(self, eigvect=None):
        print 'glat: get_magevecs()'
        magevecs = self.load_magevecs()
        if magevecs is None:
            magevecs = self.calc_magevecs(eigvect=eigvect)
        return magevecs

    def get_ipr(self, attribute=False, attrib_eigvalvect=False):
        """"""
        # Check that ipr is available, and load or calculate it if not
        if self.ipr is not None:
            ipr = self.ipr
        else:
            print 'Loading ipr...'
            ipr = self.load_ipr(attribute=attribute)
            if ipr is None:
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
                print 'glatclass: loaded prpoly.'
        else:
            prpoly = self.prpoly
        return prpoly

    def get_ldos(self, eps=None, attribute=False):
        """Obtain the local density of states for the GyroLattice

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
            print 'glat.GyroLattice(): seeking pklfn file ', pklfn
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

    def get_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False, save=False,
                         locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False, force_hdf5=True):
        """Obtain the localization of eigenvectors of the GyroLattice (fits to 1d exponential decay)
        according to |psi| ~ A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Returns
        -------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
        """
        if self.localization is not None:
            localization = self.localization
        else:
            # fn = dio.prepdir(self.lp['meshfn']) + 'localization' + self.lp['meshfn_exten'] + '.txt'
            # print 'glat.GyroLattice(): seeking pklfn file ', fn
            # if glob.glob(fn):
            #     print 'loading localization from txt file...'
            #     localization = np.loadtxt(fn, delimiter=',')
            # else:
            localization = self.load_localization(attribute=attribute)
            if localization is None:
                print 'GyroLattice: calculating localization...'
                localization = self.calc_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                                      save_eigvect_eigval=save_eigvect_eigval, locutoffd=locutoffd,
                                                      hicutoffd=hicutoffd, save_eigval=save_eigval,
                                                      attribute_eigv=attribute_eigv, force_hdf5=force_hdf5)

        if save:
            self.save_localization(force_hdf5=force_hdf5)

        return localization

    def get_edge_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                              locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False, force_hdf5=True):
        """Obtain the edge localization of eigenvectors of the GyroLattice (fits to 1d exponential decay in distance
        from boundary) according to |psi| ~ A * exp(K * np.sqrt((x - x_edge)**2 + (y - y_edge)**2)) where
        (x_edge, y_edge) is the nearest interpolated point on the boundary

        Returns
        -------
        edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        """
        if self.edge_localization is not None:
            edge_localization = self.edge_localization
        else:
            # fn = dio.prepdir(self.lp['meshfn']) + 'localization_edge' + self.lp['meshfn_exten'] + '.txt'
            # print 'glat.GyroLattice(): seeking pklfn file ', fn
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
        """Obtain inverse localization length for all modes"""
        localization = self.get_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        # Note: this line below should say localization[:, 3], not [:, 2] as it did before.
        ill = localization[:, 3]
        ill_full = np.zeros(2 * len(ill), dtype=float)
        ill_full[0:len(ill)] = ill[::-1]
        ill_full[len(ill):2*len(ill)] = ill

        return ill_full

    def get_edge_ill(self, attribute=False, eigval=None, eigvect=None):
        """Obtain inverse localization length for exponential decay from boundary for all modes"""
        edge_localization = self.get_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        ill = -edge_localization[:, 1]
        eill_full = np.zeros(2 * len(ill), dtype=float)
        eill_full[0:len(ill)] = ill[::-1]
        eill_full[len(ill):2 * len(ill)] = ill
        return eill_full

    def get_topbottom_edgelocz(self, attribute=False, eigval=None, eigvect=None):
        """Obtain inverse localization length for exponential decay from boundary for all modes"""
        edge_localization = self.get_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        topbot = edge_localization[:, 5]
        topbot_full = np.zeros(2 * len(topbot), dtype=float)
        topbot_full[0:len(topbot)] = topbot[::-1]
        topbot_full[len(topbot):2 * len(topbot)] = topbot
        return topbot_full

    def get_topbottom_edgelocz_dispersion(self, kx=None, ky=None, nkxvals=50, nkyvals=20,
                                          save=True, save_plot=True, title='Dispersion relation',
                                          save_dos_compare=False, outdir=None, name=None, ax=None,
                                          check=False, checkdir=None):
        """"""
        omegas, kx, ky, elocz = glocfns.topbottom_edgelocz_dispersion(self,
            kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals, save=save, save_plot=save_plot, title=title,
            save_dos_compare=save_dos_compare, outdir=outdir, name=name, ax=ax, check=check, checkdir=checkdir)
        return omegas, kx, ky, elocz


    def ensure_eigval_eigvect(self, eigval=None, eigvect=None, attribute=True, load=True, force_hdf5=False):
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
        eigval
        eigvect
        """
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        eigvect_name = "eigvect" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_gyro.hdf5"
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_gyro.hdf5"
        evl_saved_to_file = h5io.dset_in_hdf5(eigval_name, h5_evl)
        evt_saved_to_file = h5io.dset_in_hdf5(eigvect_name, h5_evt)

        # If not there, look for pkl files
        if not evl_saved_to_file:
            fn_evl = dio.prepdir(self.lp['meshfn']) + eigval_name + ".pkl"
            print 'fn_evl = ', fn_evl
            evl_saved_to_file = glob.glob(fn_evl)

        if not evt_saved_to_file:
            fn_evt = dio.prepdir(self.lp['meshfn']) + eigvect_name + ".pkl"
            evt_saved_to_file = glob.glob(fn_evt)

        # print 'glat: evl_saved_to_file = ', evl_saved_to_file

        if not (evl_saved_to_file and evt_saved_to_file and eigval is not None and eigvect is not None):
            if evl_saved_to_file and evt_saved_to_file:
                if self.eigval is not None and self.eigvect is not None:
                    eigval = self.eigval
                    eigvect = self.eigvect
                else:
                    # Try to load eigval and eigvect, if load or attribute are true
                    if load or attribute:
                        print 'Attempting to load eigval/vect...'
                        eigval = self.load_eigval(attribute=attribute)
                        eigvect = self.load_eigvect(attribute=attribute)
            else:
                print 'glat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
                if eigval is None or eigvect is None:
                    eigval, eigvect = self.eig_vals_vects()
                print '... saving eigval/vects'
                self.save_eigval_eigvect(force_hdf5=force_hdf5)

        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect
        return eigval, eigvect

    def ensure_eigval(self, eigval=None, attribute=False, force_hdf5=False):
        """Return eigval and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied eigval, (2) calling from self, (3) loading it, or
        (4) calculating it.
        """
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_gyro.hdf5"
        evl_saved_to_file = h5io.dset_in_hdf5(eigval_name, h5_evl)

        # If not there, look for pkl files
        if not evl_saved_to_file:
            fn_evl = dio.prepdir(self.lp['meshfn']) + eigval_name + ".pkl"
            print 'fn_evl = ', fn_evl
            evl_saved_to_file = glob.glob(fn_evl)

        if evl_saved_to_file and eigval is not None:
            # performs (1), simply return the supplied eigval, since already saved
            pass
        elif evl_saved_to_file:
            # attempts (2) or (3)
            if self.eigval is not None:
                eigval = self.eigval
            else:
                # Try to load eigval and eigvect
                print 'Attempting to load eigval/vect...'
                eigval = self.load_eigval(attribute=attribute)
        else:
            # attempts (4)
            print 'glat.load_eigval_eigvect: Could not load eigval, computing it and saving it...'
            if eigval is None:
                eigval, eigvect = self.eig_vals_vects()
            print '... saving eigval/vects'
            self.save_eigval(force_hdf5=force_hdf5)

        if attribute:
            self.eigval = eigval
        return eigval

    def ensure_magevecs(self, magevecs=None, attribute=False, force_hdf5=False):
        """Return eigval and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied eigval, (2) calling from self, (3) loading it, or
        (4) calculating it.

        Returns
        -------
        magevecs : #particles x #particles float array
            The magnitude of the upper half of eigenvectors at each site. magevecs[i, j] is the magnitude of the i+NP
            normal mode at site j.
        """

        # Make eigval name
        magevecs_name = "magevecs" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "magevecs_gyro.hdf5"
        mevl_saved_to_file = h5io.dset_in_hdf5(magevecs_name, h5_evl)

        # If not there, look for pkl files
        if not mevl_saved_to_file:
            fn_mevl = dio.prepdir(self.lp['meshfn']) + magevecs_name + ".pkl"
            print 'fn_evl = ', fn_mevl
            mevl_saved_to_file = glob.glob(fn_mevl)

        if mevl_saved_to_file and magevecs is not None:
            # performs (1), simply return the supplied magevecs, since already saved
            pass
        elif mevl_saved_to_file:
            # attempts (2) or (3)
            if self.magevecs is not None:
                magevecs = self.magevecs
            else:
                # Try to load eigval and eigvect
                print 'Attempting to load eigval/vect...'
                magevecs = self.load_magevecs()
        else:
            # attempts (4)
            print 'glat.ensure_magevecs: Could not load magevecs, computing it and saving it...'
            if magevecs is None:
                magevecs = self.get_magevecs()
            print '... saving magevecs'
            self.save_magevecs(force_hdf5=force_hdf5)

        if attribute:
            self.magevecs = magevecs

        return magevecs

    def ensure_localization(self, locz=None, attribute=False, force_hdf5=False):
        """Return locz and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied locz, (2) calling from self, (3) loading it, or
        (4) calculating it.
        """
        # Make locz name
        locz_name = "localization" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_gyro.hdf5"
        saved_to_file = h5io.dset_in_hdf5(locz_name, h5fn)

        # If not there, look for pkl files
        if not saved_to_file:
            fn_locz = dio.prepdir(self.lp['meshfn']) + locz_name + ".pkl"
            saved_to_file = glob.glob(fn_locz)

        if saved_to_file and locz is not None:
            # performs (1), simply return the supplied locz, since already saved
            pass
        elif saved_to_file:
            # attempts (2) or (3)
            if self.localization is not None:
                locz = self.localization
            else:
                # Try to load locz
                print 'Attempting to load locz...'
                locz = self.load_localization(attribute=attribute)
        else:
            # attempts (4)
            print 'glat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if locz is None:
                locz = self.calc_localization()
            print '... saving eigval/vects'
            self.save_localization(force_hdf5=force_hdf5)

        if attribute:
            self.localization = locz
        return locz

    def ensure_edge_localization(self, elocz=None, attribute=False, force_hdf5=False, save_images=False, save_im=False):
        """Return elocz and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied locz, (2) calling from self, (3) loading it, or
        (4) calculating it.

        Parameters
        ----------
        elocz : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            edge_localization fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
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
        elocz : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            edge localization fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        """
        # Make locz name
        locz_name = "localization_edge" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_edge_gyro.hdf5"
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
            print 'glat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if elocz is None:
                elocz = self.calc_edge_localization(attribute=attribute)
            print '... saving eigval/vects'
            self.save_edge_localization(force_hdf5=force_hdf5, save_images=save_images, save_im=save_im)

        if attribute:
            self.edge_localization = elocz
        return elocz

    def ensure_ipr(self, ipr=None, attribute=False):
        """Return ipr (inverse participation ratio) and save it to disk if not saved already.
        To obtain ipr, proceed by (1) returning from supplied ipr, (2) calling from self, (3) loading it, or
        (4) calculating it.
        """
        fn_ipr = dio.prepdir(self.lp['meshfn']) + "ipr" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn_ipr) and ipr is not None:
            # performs (1), simply return the supplied eigval, since already saved
            if attribute:
                self.ipr = ipr

        elif glob.glob(fn_ipr):
            # attempts (2) or (3)
            if self.ipr is not None:
                ipr = self.ipr
            else:
                # Try to load eigval and eigvect
                print 'Attempting to load ipr...'
                ipr = self.load_ipr(attribute=attribute)
        else:
            # attempts (4)
            print 'glat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if ipr is None:
                ipr = self.calc_ipr(attribute=attribute)
            print '... saving eigval/vects'
            self.save_ipr()

        return ipr

    def save_eigval_eigvect(self, eigval=None, eigvect=None, infodir='auto', attribute=True, force_hdf5=False,
                            save_png=False):
        """Save eigenvalues and eigenvectors for this GyroLattice

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

        # Naming
        basis_str = glatfns.get_basis_str(self)
        eigval_name = 'eigval' + self.lp['meshfn_exten'] + basis_str
        eigvect_name = 'eigvect' + self.lp['meshfn_exten'] + basis_str
        if force_hdf5:
            eigvalfn = dio.prepdir(self.lp['meshfn']) + 'eigvals_gyro' + basis_str + '.hdf5'
            h5io.save_dset_hdf5(eigval, eigval_name, eigvalfn)

            eigvectfn = dio.prepdir(self.lp['meshfn']) + 'eigvects_gyro' + basis_str + '.hdf5'
            h5io.save_dset_hdf5(eigvect, eigvect_name, eigvectfn)
        else:
            eigvalfn = infodir + eigval_name + '.pkl'
            output = open(eigvalfn, 'wb')
            pickle.dump(eigval, output)
            output.close()

            eigvectfn = infodir + eigvect_name + '.pkl'
            output = open(eigvectfn, 'wb')
            pickle.dump(eigvect, output)
            output.close()

        if save_png:
            fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'gyro')
            plt.savefig(infodir + 'eigval_gyro_hist' + self.lp['meshfn_exten'] + basis_str + '.png')
            plt.clf()
        print 'Saved gyro DOS to ' + eigvalfn + '\n and ' + eigvectfn

    def save_eigval(self, eigval=None, infodir='auto', attribute=True, force_hdf5=False, save_png=False):
        """Save eigenvalues for this GyroLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval
        attribute: bool
            Whether to attribute the eigvals to self (ie self.eigval = eigval, etc)

        Returns
        -------

        """
        # print 'glat.save_eigval(): eigval = ', eigval
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if eigval is None:
            eigval = self.get_eigval(attribute=attribute)

        # Naming
        basis_str = glatfns.get_basis_str(self)
        eigval_name = 'eigval' + self.lp['meshfn_exten'] + basis_str
        if force_hdf5:
            eigvalfn = dio.prepdir(self.lp['meshfn']) + 'eigvals_gyro' + basis_str + '.hdf5'
            h5io.save_dset_hdf5(eigval, eigval_name, eigvalfn)
        else:
            eigvalfn = infodir + eigval_name + '.pkl'
            output = open(eigvalfn, 'wb')
            pickle.dump(eigval, output)
            output.close()

        if save_png:
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro')
            plt.savefig(infodir + 'eigval_gyro_hist' + self.lp['meshfn_exten'] + '.png')
            plt.clf()
        print 'Saved gyro eigvals to ' + eigvalfn

    def save_magevecs(self, magevecs=None, infodir='auto', force_hdf5=False):
        """Save magnitude of eigenvector element for each site for this GyroLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval

        Returns
        -------

        """
        # print 'glat.save_eigval(): eigval = ', eigval
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if magevecs is None:
            magevecs = self.get_magevecs()

        # Naming
        magevecs_name = 'magevecs' + self.lp['meshfn_exten']
        if force_hdf5:
            magevecsfn = dio.prepdir(self.lp['meshfn']) + 'magevecs_gyro.hdf5'
            h5io.save_dset_hdf5(magevecs, magevecs_name, magevecsfn)
        else:
            magevecsfn = infodir + magevecs_name + '.pkl'
            output = open(magevecsfn, 'wb')
            pickle.dump(magevecs, output)
            output.close()

        print 'Saved gyro magevecs to ' + magevecsfn

    def save_Omg(self, infodir='auto', histogram=False, attribute=True, force_hdf5=False):
        """Save Omk pinning frequencies for this GyroLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval, eigvect
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
        histogram : bool
            If saving to a txt file, save a png of the Omg distribution
        force_hdf5 : bool
            save the pinning configuration to an hdf5 file rather than a text file
        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])
        if self.Omg is not None:
            pinning_name = self.get_pinmeshfn_exten()
            # When running jobs in series (NOT in parallel), can save pinning directly to hdf5
            if force_hdf5:
                h5fn = dio.prepdir(self.lp['meshfn']) + 'omg_configs.hdf5'
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
                        fi.create_dataset(pinning_name, shape=np.shape(self.Omg), data=self.Omg, dtype='float')
                    else:
                        raise RuntimeError('Pinning config already exists in hdf5, exiting...')
            else:
                # Otherwise perform standard save of a text file for the pinning configuration
                print 'saving pinning in txt...'
                fn = dio.prepdir(self.lp['meshfn']) + pinning_name + '.txt'
                np.savetxt(fn, self.Omg, header="Pinning frequencies Omg")
                if histogram:
                    plt.clf()
                    fig, hist_ax = leplt.initialize_histogram(self.Omg, xlabel=r'Pinning frequencies, $\Omega_g$')
                    histfn = 'Omg_hist_mean' + sf.float2pstr(self.lp['Omg']) + self.lp['meshfn_exten']
                    plt.savefig(infodir + histfn + '.png')
                    plt.clf()
                print 'Saved Omg to ' + fn
        else:
            raise RuntimeError('self.Omg is None, so cannot save it!')

    def save_ipr(self, infodir='auto', attribute=True, save_images=True, show=False, **kwargs):
        """
        Parameters
        ----------
        infodir: str
            Directory in which to save ipr as ipr.pkl
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
        save_images : bool
            Save a png of ipr DOS and pr DOS
        show : bool
            Display the plot after plotting
        **kwargs: keyword arguments for lepm.plotting.colored_DOS_plot()
            alpha=1.0, colorV=None, colormap='viridis', norm=None,
            facecolor='#80D080', nbins=75, fontsize=12, cbar_ax=None, vmin=None, vmax=None, linewidth=1,
            make_cbar=True, climbars=True, xlabel='Oscillation frequency, $\omega/\Omega_g$',
            xlabel_pad=16, ylabel=r'$D(\omega)$', ylabel_pad=10, ylabel_ha='center', ylabel_va='center',
            cax_label='', cbar_labelpad=10, ticks=None, cbar_nticks=None, cbar_tickfmt=None,
            cbar_ticklabels=None,
            orientation='vertical', cbar_orientation='vertical',
            invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False, ylabel_rot=90,
            DOSexcite=None, DOSexcite_color='r', histrange=None

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
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for gyroscopic network',
                              fname='ipr_gyro_hist',
                              alpha=1.0, FSFS=12, inverse_PR=True, show=show, **kwargs)
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for gyroscopic network',
                              fname='pr_gyro_hist',
                              alpha=1.0, FSFS=12, inverse_PR=False, show=show, **kwargs)
        print 'Saved gyro ipr to ' + infodir + 'ipr' + self.lp['meshfn_exten'] + '.pkl'

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

        print 'Saved gyro prpoly to ' + infodir + 'prpoly' + self.lp['meshfn_exten'] + '.pkl'

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
            ldos_infodir = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' + sf.float2pstr(self.lp['eps']) + '/'
            dio.ensure_dir(ldos_infodir)
            self.plot_ldos(eigval=eigval, ldos=ldos, outdir=ldos_infodir, FSFS=12)

            # Make movie
            imgname = ldos_infodir + 'ldos_site'
            movname = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' + sf.float2pstr(self.lp['eps']) + '_sites'
            lemov.make_movie(imgname, movname, indexsz='08', imgdir=ldos_infodir, rm_images=True, save_into_subdir=True)

        print 'Saved gyro ldos to ' + fn

    def save_localization(self, eigval=None, infodir='auto', attribute=True, save_images=False,
                          save_eigvect_eigval=False, save_im=False, force_hdf5=False):
        """Get and save localization measure for all eigenvectors of the GyroLattice"""
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        locz = self.get_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        locz_name = 'localization' + self.lp['meshfn_exten']
        if force_hdf5:
            hdf5fn = infodir + 'localization_gyro.hdf5'
            h5io.save_dset_hdf5(locz, locz_name, hdf5fn)
        else:
            fn = infodir + locz_name + '.txt'
            print 'Saving localization as ' + fn
            header = "Localization of eigvects: fitted to A*exp(K*sqrt((x-xc)**2 + (y-yc)**2)): " + \
                     "xc, yc, A, K, uncA, covAK, uncK. The modes examined range from int(len(eigval)*0.5) " \
                     "to len(eigval)."
            np.savetxt(fn, locz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if save_im:
            if eigval is None:
                eigval = self.get_eigval()

            fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.6, hsfrac=0.6 * 0.75, fontsize=10, tspace=3)
            evals = np.imag(eigval[int(len(eigval) * 0.5):])
            ax.plot(evals, -locz[:, 3], '-', color='#334A5A')
            ax.fill_between(evals, -locz[:, 3] - np.sqrt(locz[:, 6]), -locz[:, 3] + np.sqrt(locz[:, 6]), color='#89BBDB')
            if abs(self.lp['Omg']) == 1 and abs(self.lp['Omk']) == 1 and not self.lp['dcdisorder']:
                ax.set_xlim(1.0, 4.0)
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

        print 'Saved gyro localization to ' + fn
        return locz

    def save_edge_localization(self, eigval=None, infodir='auto', attribute=True, save_images=False,
                               save_eigvect_eigval=False, save_im=False, force_hdf5=False):
        """Get and save localization measure for all eigenvectors of the GyroLattice"""
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        elocz = self.get_edge_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        locz_name = 'localization_edge' + self.lp['meshfn_exten']
        if force_hdf5:
            hdf5fn = infodir + 'localization_edge_gyro.hdf5'
            print 'glat.GyroLattice(): saving edge_localization dataset in ' + hdf5fn
            h5io.save_dset_hdf5(elocz, locz_name, hdf5fn)
            fn = infodir + locz_name
        else:
            fn = infodir + locz_name + '.txt'
            print 'Saving edge localization as ' + fn
            header = "Localization of eigvects to edge: fitted to A*exp(K*sqrt((x-xb)**2 + (y-yb)**2)): " + \
                     "A, K, uncA, covAK, uncK. xb and yb are nearest points along the boundary. " + \
                     "The modes examined range from int(len(eigval)*0.5) to len(eigval)."
            np.savetxt(fn, elocz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if save_im:
            if eigval is None:
                eigval = self.get_eigval()

            fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.6, hsfrac=0.6 * 0.75, fontsize=10, tspace=3)
            evals = np.imag(eigval[int(len(eigval) * 0.5):])
            ax.plot(evals, -elocz[:, 1], '-', color='#334A5A')
            ax.fill_between(evals, -elocz[:, 1] - np.sqrt(elocz[:, 4]), -elocz[:, 1] + np.sqrt(elocz[:, 4]),
                            color='#89BBDB')
            if abs(self.lp['Omg']) == 1 and abs(self.lp['Omk']) == 1 and not self.lp['dcdisorder']:
                ax.set_xlim(1.0, 4.0)
            title = r'Edge localization length $\lambda$ for $|\psi| \sim e^{-r / \lambda}$'
            plt.text(0.5, 1.08, title, horizontalalignment='center', fontsize=10, transform=ax.transAxes)
            ax.set_ylabel(r'Inverse localization length, $1/\lambda$')
            ax.set_xlabel(r'Oscillation frequency, $\omega/\Omega_g$')
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

        print 'Saved gyro localization to ' + fn
        return elocz

    def save_DOSmovie(self, infodir='auto', attribute=True, save_DOS_if_missing=True, basis=None):
        """"""
        if infodir == 'auto':
            infodir = self.lattice.lp['meshfn'] + '/'
        exten = self.lp['meshfn_exten']

        # Obtain eigval and eigvect, and matrix if necessary
        if self.eigval is None or self.eigvect is None:
            # check if we can load the DOS info
            if glob.glob(infodir + 'eigval' + exten + '.pkl') and glob.glob(infodir + 'eigvect' + exten + '.pkl'):
                print "Loading eigval and eigvect from " + self.lattice.lp['meshfn']
                with open(infodir + "eigval" + exten + '.pkl', "r") as f:
                    eigval = pickle.load(f)
                with open(infodir + "eigvect" + exten + '.pkl') as f:
                    print 'loading eigvect from pkl file...'
                    eigvect = pickle.load(f)
            else:
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=attribute, basis=basis)
                    eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
                else:
                    eigval, eigvect = self.eig_vals_vects(attribute=attribute)

                if save_DOS_if_missing:
                    output = open(infodir + 'eigval' + exten + '.pkl', 'wb')
                    pickle.dump(eigval, output)
                    output.close()

                    output = open(infodir + 'eigvect' + exten + '.pkl', 'wb')
                    pickle.dump(eigvect, output)
                    output.close()

                    print 'Saved gyro DOS to ' + infodir + 'eigvect(val)' + exten + '.pkl\n'

        if not glob.glob(infodir + 'eigval_gyro_hist' + exten + '.png'):
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro')
            plt.savefig(infodir + 'eigval_gyro_hist' + exten + '.png')
            plt.clf()

        if self.lp['periodicBC'] and len(self.lattice.xy) > 2:
            colorstr = 'ill'
        else:
            colorstr = 'pr'

        lemov.save_normal_modes_Nashgyro(self, datadir=infodir, dispersion=[], sim_type='gyro',
                                         rm_images=True, gapims_only=False, save_into_subdir=False, overwrite=True,
                                         color=colorstr)

    def infinite_dispersion(self, kx=None, ky=None, nkxvals=50, nkyvals=20,
                            save=True, save_plot=True, title='Dispersion relation',
                            save_dos_compare=False, outdir=None, name=None, ax=None, lwscale=1.0):
        """Compute dispersion relation for infinite sample over a grid of kx ky values

        Parameters
        ----------
        kx : float array or None
            The wavenumber values in x direction to use
        ky : float array or None
            The wavenumber values in y direction to use
        nkxvals : int
            If kx is unspecified, then nkxvals determines how many kvectors are sampled in x dimension.
        nkyvals : int
            If ky is unspecified and if network is not a periodic_strip, then nkyvals determines how
            many kvectors are sampled in y dimension.
        save : bool
            whether to save the results of the dispersion calculation
        save_plot : bool
            whether to save the results of the dispersion calculation as a png
        title : str
            the title of the plot of the dispersion
        save_dos_compare : bool
            Compare the projection of the dispersion onto the omega axis with the DOS of the GyroLattice
        outdir : str or None
            path to the dir where results are saved, if save==True. If None, uses lp['meshfn'] for hlat.lattice
        name : str or None
            The name of the file (whether pickle or png) to save in outdir
        ax : matplotlib axis instance or None
            THe axis on which to plot the dispersion, if save_plot is True. If None, creates a
            new 90mm figure with single axis.

        Returns
        -------
        omegas, kx, ky
        """
        # Note: the function called below has not been finished
        omegas, kx, ky = glatkspacefns.infinite_dispersion(self, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                                           save=save, save_plot=save_plot, title=title, outdir=outdir,
                                                           name=name, ax=ax, lwscale=lwscale)

        if save_dos_compare:
            # Save DOS from projection
            if outdir is None:
                outdir = dio.prepdir(self.lp['meshfn'])
            else:
                outdir = dio.prepdir(outdir)
            name = outdir + 'dispersion_gyro' + self.lp['meshfn_exten'] + '_nx' + str(len(kx)) + '_ny' + str(len(ky))
            name += '_maxkx{0:0.3f}'.format(np.max(np.abs(kx))).replace('.', 'p')
            name += '_maxky{0:0.3f}'.format(np.max(np.abs(ky))).replace('.', 'p')

            # initialize figure
            fig, ax = leplt.initialize_1panel_centered_fig()
            ax2 = ax.twinx()
            ax.hist(omegas.ravel(), bins=1000)

            eigval = np.imag(glat.get_eigval())
            ax2.hist(eigval[eigval > 0], bins=50, color=lecmap.green(), alpha=0.2)
            ax.set_title('DOS from dispersion')
            xlims = ax.get_xlim()
            ax.set_xlim(0, xlims[1])
            plt.savefig(name + '_dos.png', dpi=300)

        return omegas, kx, ky

    def calc_matrix(self, attribute=True, check=False, basis=None):
        """

        Parameters
        ----------
        attribute : bool
            attribute the matrix to self.matrix
        check : bool
            plot the matrix after computation
        basis : None or str ('XY', 'psi')
            The basis to use to construct the dynamical matrix

        Returns
        -------

        """
        matrix = glatfns.dynamical_matrix_gyros(self, basis=basis)
        if check:
            le.plot_real_matrix(matrix, show=True)

        if attribute:
            self.matrix = matrix
        return matrix

    def calc_eigvals(self, matrix=None, sort='imag', attribute=True):
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

    def calc_magevecs(self, eigvect=None):
        if eigvect is None:
            eigvect = self.get_eigvect()

        magevec = glatfns.calc_magevecs(eigvect)

        return magevec

    def eig_vals_vects(self, matrix=None, attribute=True, check=False):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        if matrix is None:
            print 'glat.eig_vals_vects: getting matrix...'
            matrix = self.get_matrix(attribute=attribute)
            if check:
                le.plot_complex_matrix(matrix, show=True)

        print 'glat.eig_vals_vects: computing eigval, eigvect...'
        eigval, eigvect = le.eig_vals_vects(matrix)
        if attribute:
            self.eigval = eigval
            self.eigvect = eigvect
        return eigval, eigvect

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
        # psides : 2NP x 2NP float array
        #   the (2i,j)th element has the percentage of the ith particle attributable to polygons with j sides
        #   the (2i + 1,j)th element also has the percentage of the ith particle attributable to polygons with j sides
        maxNsides = np.max(Pno)
        psides = np.zeros((2.*len(self.lattice.xy), maxNsides+1), dtype=float)
        for ii in range(len(self.lattice.xy)):
            for jj in range(maxNsides+1):
                if len(pnod[ii]) > 0:
                    # print 'pnod[ii] = ', pnod[ii]
                    # print 'np.sum(np.array(pnod[', ii, ']) == ', jj, ') =', np.sum(np.array(pnod[ii]) == jj )
                    psides[2*ii, jj] = float(np.sum(np.array(pnod[ii]) == jj)) / float(len(pnod[ii]))
                    psides[2*ii+1, jj] = psides[2*ii, jj]
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

        if attribute:
            self.prpoly = prpoly

        return prpoly

    def calc_ldos(self, eps=None, attribute=False, eigval=None, eigvect=None):
        if eps is None:
            if 'eps' in self.lp:
                eps = self.lp['eps']
            else:
                eps = 5.0
        self.lp['eps'] = eps

        if eigval is None or eigvect is None:
            print 'Loading eigvect/eigval for ldos calculation without attribution...'
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        ldos = glatfns.calc_ldos(eigval, eigvect, eps=eps)
        if attribute:
            self.ldos = ldos

        return ldos

    def calc_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                          save_eigval=False, locutoffd=None, hicutoffd=None, attribute_eigv=False, force_hdf5=True):
        """For each eigvector excitation with a positive frequency, fit excitation to an exponential decay centered
        about the excitation's COM.
        where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Parameters
        ----------
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
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency
        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current glat...'
                self.save_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False, force_hdf5=force_hdf5)
            elif save_eigval:
                print 'glat.calc_localization(): saving eigval only...'
                self.ensure_eigval(eigval=eigval, attribute=False, force_hdf5=force_hdf5)

        if self.lattice.lp['periodicBC']:
            if 'periodic_strip' in self.lattice.lp:
                if self.lattice.lp['periodic_strip']:
                    perstrip = True
                else:
                    perstrip = False
            else:
                perstrip = False

            if perstrip:
                # edge_localization = glocfns.fit_edgedecay_periodicstrip(self.lattice.xy, eigval, eigvect,
                #                                                         cutoffd=cutoffd, check=self.lp['check'])
                localization = glocfns.fit_eigvect_to_exponential_1dperiodic(self.lattice.xy, eigval, eigvect,
                                                                             self.lattice.lp['LL'],
                                                                             locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                             check=self.lp['check'])
            else:
                localization = glocfns.fit_eigvect_to_exponential_periodic(self.lattice.xy, eigval, eigvect,
                                                                           self.lattice.lp['LL'],
                                                                           locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                           check=self.lp['check'])
        else:
            localization = glocfns.fit_eigvect_to_exponential(self.lattice.xy, eigval, eigvect, check=self.lp['check'])

        if attribute:
            self.localization = localization

        return localization

    def calc_edge_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                               save_eigval=False, locutoffd=None, hicutoffd=None, attribute_eigv=False, check=False):
        """Measure the localization length of the edge modes (look for exponential falloff from the edge of the sample)

        Parameters
        ----------
        attribute : bool
            attribute localization to self
        eigvect : 2N x 2N complex array
            The eigenvectors of the magnetic gyroscopic network
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
        edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
            fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
            for modes with positive frequency, starting near zero frequency and increasing in frequency

        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current glat...'
                self.save_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False)
            elif save_eigval:
                print 'glat.calc_localization(): saving eigval only...'
                self.ensure_eigval(eigval=eigval, attribute=False)

        boundary = self.lattice.get_boundary()
        if self.lattice.lp['periodicBC']:
            if 'periodic_strip' in self.lattice.lp:
                if self.lattice.lp['periodic_strip']:
                    perstrip = True
                else:
                    perstrip = False
            else:
                perstrip = False

            if perstrip:
                edge_localization = glocfns.fit_eigvect_edge_boundaries(self.lattice.xy, boundary, eigvect,
                                                                        self.lattice.PVxydict,
                                                                        eigval=eigval,
                                                                        locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                        check=self.lp['check'])
            else:
                raise RuntimeError('No reason to fit a fully periodic sample to edge localization.')
        else:
            if not check:
                check = self.lp['check']
            if 'annulus' in self.lp['LatticeTop'] or self.lp['shape'] == 'annulus':
                # Note that here PVxydict will be None, so it is effectively ignored
                edge_localization = glocfns.fit_eigvect_edge_boundaries(self.lattice.xy, boundary, eigvect,
                                                                        self.lattice.PVxydict,
                                                                        eigval=eigval,
                                                                        locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                        check=self.lp['check'])
            else:
                edge_localization = glocfns.fit_eigvect_to_exponential_edge(self.lattice.xy, boundary, eigvect, eigval,
                                                                            locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                            check=check)

        if attribute:
            self.edge_localization = edge_localization

        return edge_localization

    def plot_Omg(self):
        """Plot the network colored by pinning frequency"""
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(wsfrac=0.5,)
        self.lattice.plot_BW_lat(fig=fig, ax=ax, save=False, close=False, axis_off=False, title='')
        sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], c=self.Omg, cmap='coolwarm', edgecolors='none')
        ticks = [np.min(self.Omg), self.lp['Omg'], np.max(self.Omg)]
        plt.colorbar(mappable=sc, cax=cbar_ax, label=r'$\Omega_g$', orientation='horizontal', ticks=ticks)
        cbar_ax.xaxis.labelpad = -25
        plt.savefig(dio.prepdir(self.lp['meshfn']) + 'Omg_mean' + sf.float2pstr(self.lp['Omg']) +
                    self.lp['meshfn_exten'] + '.png')
        plt.clf()
        # plt.show()

    def plot_OmK(self, axis_off=True):
        """Plot the bonds colored by characteristic spring frequency"""
        fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(wsfrac=0.5,)
        # sc = self.lattice.plot_BW_lat(fig=fig, ax=ax, save=False, close=False, axis_off=False, title='')
        OmKv = le.KL2kL(self.lattice.NL, self.OmK, self.lattice.BL)
        [ax, axcb] = le.movie_plot_2D(self.lattice.xy, self.lattice.BL, OmKv, None, None, fig=fig, ax=ax,
                                      NL=self.lattice.NL, KL=self.lattice.KL,
                                      PVx=self.lattice.PVx, PVy=self.lattice.PVy, climv=(np.min(OmKv), np.max(OmKv)),
                                      axcb='auto', cbar_ax=cbar_ax, cbar_orientation='horizontal',
                                      colorz=False, colormap='BlueBlackRed', bgcolor='#FFFFFF',
                                      axis_off=axis_off, cax_label=r'$\Omega_k$')
        # plt.show()
        # ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], c='k', edgecolors='none')
        ticks = [np.min(OmKv), np.max(OmKv)]
        print 'OmKv = ', OmKv
        cbar_ax.xaxis.set_labelpad = -10
        axcb.set_ticks(ticks)
        plt.savefig(dio.prepdir(self.lp['meshfn']) + 'OmK' + self.lp['meshfn_exten'] + '.png', dpi=150)
        plt.clf()
        # plt.show()

    def plot_prpoly(self, outdir=None, title=r'Polygonal contributions to normal mode excitations',
                    fname='prpoly_gyro_hist', fontsize=8, show=True, global_alpha=1.0, shaded=False, save=True):
        """Plot Inverse Participation Ratio of the gyroscopic network

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
        print 'glat.add_prpoly_to_plot(): global_alpha = ', global_alpha
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

        if vmax is None:
            vmax = 0.0
            for ii in range(len(prpoly)):
                vmax = max(np.max(prpoly[ii]), vmax)

        print 'glat.add_prpoly_to_plot(): overlaying prpoly to axis stack...'
        for ii in range(len(prpoly)-3):
            # print 'ax[', ii, '] = ', ax[ii], ' of ', len(prpoly)
            if shaded:
                DOS_ax, cbar_ax, cbar, n, bins = \
                    leplt.shaded_DOS_plot(eigval, ax[ii], 'gyro',
                                          alpha=prpoly[ii + 3] * global_alpha, facecolor='#80D080',
                                          fontsize=fontsize, cbar_ax=None, vmin=0.0, vmax=vmax, linewidth=0,
                                          cax_label='', make_cbar=False, climbars=True, xlabel=xlabel, ylabel=str(ii+3))
            else:
                leplt.colored_DOS_plot(eigval, ax[ii], 'gyro', alpha=global_alpha, colorV=prpoly[ii + 3],
                                       colormap='CMRmap_r', norm=None, nbins=75, fontsize=fontsize, cbar_ax=cbar_ax,
                                       vmin=0.0, vmax=vmax, linewidth=0.,
                                       make_cbar=True, climbars=True, xlabel='Oscillation frequency, $\omega/\Omega_g$',
                                       xlabel_pad=12, ylabel=str(ii+3), ylabel_pad=None,
                                       cax_label=r'$\sum_i |\psi_i|^2$ in $n$-gons',
                                       cbar_labelpad=-28, ticks=None, cbar_nticks=3, cbar_tickfmt='%0.2f',
                                       orientation='vertical', cbar_orientation='horizontal',
                                       invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False,
                                       ylabel_rot=90)

            ax[ii].yaxis.set_major_locator(MaxNLocator(nbins=3))

        return fig, ax

    def add_ipr_to_ax(self, ax, ipr=None, alpha=1.0, inverse_PR=True, **kwargs):
        """Add a DOS colored by (Inverse) Participation Ratio of the gyroscopic network to an axis

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
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'gyro', alpha=alpha,
                                                                colorV=ipr, colormap='viridis', **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()
            print 'len(ipr) = ', len(ipr)
            print 'len(eigval) = ', len(eigval)
            print 'len(xy) = ', len(self.lattice.xy)
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'gyro', alpha=alpha,
                                                                colorV=1./ipr, colormap='viridis_r', **kwargs)
        return ax

    def plot_ipr_DOS(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='ipr_gyro_hist',
                     alpha=1.0, FSFS=12, show=True, inverse_PR=True, save=True, **kwargs):
        """Plot Inverse Participation Ratio of the gyroscopic network

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
        **kwargs: keyword arguments for lepm.plotting.colored_DOS_plot()
            alpha=1.0, colorV=None, colormap='viridis', norm=None,
            facecolor='#80D080', nbins=75, fontsize=12, cbar_ax=None, vmin=None, vmax=None, linewidth=1,
            make_cbar=True, climbars=True, xlabel='Oscillation frequency, $\omega/\Omega_g$',
            xlabel_pad=16, ylabel=r'$D(\omega)$', ylabel_pad=10, ylabel_ha='center', ylabel_va='center',
            cax_label='', cbar_labelpad=10, ticks=None, cbar_nticks=None, cbar_tickfmt=None,
            cbar_ticklabels=None,
            orientation='vertical', cbar_orientation='vertical',
            invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None, ylabel_right=False, ylabel_rot=90,
            DOSexcite=None, DOSexcite_color='r', histrange=None

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

            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro', alpha=alpha,
                                                                           colorV=ipr, colormap='viridis', linewidth=0,
                                                                           cax_label=r'$p^{-1}$', climbars=True,
                                                                           **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()
            print '1/ipr = ', 1./ipr
            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro', alpha=alpha,
                                                                           colorV=1./ipr, colormap='viridis_r',
                                                                           linewidth=0, cax_label=r'$p$',
                                                                           climbars=False, **kwargs)

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
        dos_ax, cbar_ax = glatpfns.plot_ill_dos(self, dos_ax=dos_ax, cbar_ax=cbar_ax, alpha=alpha, vmin=vmin,
                                                vmax=vmax, climbars=True, **kwargs)
        if save:
            fn = dio.prepdir(self.lp['meshfn']) + 'ill_dos' + self.lp['meshfn_exten'] + '.png'
            print 'saving ill to: ' + fn
            plt.savefig(fn, dpi=600)
        if show:
            plt.show()
        return dos_ax, cbar_ax

    def plot_DOS(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='eigval_gyro_hist',
                 alpha=None, show=True, dos_ax=None, **kwargs):
        """
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        **kwargs : keyword arguments for colored_DOS_plot()
        """
        # First make sure all eigvals are accounted for
        eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        if dos_ax is None:
            fig, dos_ax = leplt.initialize_DOS_plot(eigval, 'gyro', alpha=alpha)
        else:
            ipr = self.get_ipr()
            leplt.colored_DOS_plot(eigval, dos_ax, 'gyro', **kwargs)
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
        fig, DOS_ax = glatpfns.draw_ldos_plots(eigval, ldos, glat, outdir=outdir, FSFS=FSFS)

    def plot_localization(self, localization=None, eigval=None, eigvect=None,
                          outdir=None, fname='localization', alpha=None, fontsize=12):
        """Plot eigenvector normal modes of system with overlaid fits of exponential localization.
        where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Parameters
        ----------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
            where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))
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
        fig, DOS_ax, ax = glatpfns.draw_localization_plots(self, localization, eigval, eigvect, outdir=outdir,
                                                           alpha=alpha, fontsize=fontsize)

    def plot_edge_localization(self, elocz=None, eigval=None, eigvect=None,
                               outdir=None, fname='localization', alpha=None, fontsize=12):
        """Plot eigenvector normal modes of system with overlaid fits of exponential localization.
        where fit is A * exp(K * np.sqrt((x - x_boundary)**2 + (y - y_boundar)**2))

        Parameters
        ----------
        elocz : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
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
        fig, DOS_ax, ax = glatpfns.draw_edge_localization_plots(self, elocz, eigval, eigvect,
                                                                outdir=outdir, alpha=alpha, fontsize=fontsize)

    def sum_amplitudes_band_spectrum(self, omegac=None, outdir=None, cmap='viridis_r', deform_difference=False,
                                     deformation='shrink'):
        """Sum the normal mode amplitudes at each site over all modes below cutoff frequency.
        If deform_difference is True, then look at how local charges change when conducting deformation specified by
        deformation argument
        """
        eigval, eigvect = self.get_eigval_eigvect()

        if omegac is None:
            omegac = np.array([0., 2.25])

        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn']) + 'charge' + self.lp['meshfn_exten'] + '/'

        dio.ensure_dir(outdir)
        if cmap not in plt.colormaps():
            lecmap.register_colormaps()

        cmp = plt.get_cmap(cmap)

        Wfig = 224

        jj = 0
        tot_amp = []
        in_sum_list = []
        for omc in omegac:
            in_sum = np.imag(eigval) > omc
            print 'np.shape(np.abs(eigvect[in_sum, :])**2 = ', np.shape(np.abs(eigvect[in_sum, :])**2)
            tmp = np.sum(np.abs(eigvect[in_sum, :])**2, axis=0)
            print 'np.shape(tmp) = ', np.shape(tmp)
            print 'tmp[0:10] = ', tmp[0:10]
            total_amp = np.array([np.sqrt(tmp[2*ii] + tmp[2*ii + 1]) for ii in range(int(len(tmp)*0.5))])
            # plot result
            fig, ax = leplt.initialize_1panel_centered_fig(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
            [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
            print 'max = ', np.max(total_amp)
            ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp/np.max(total_amp),
                       c=total_amp, cmap=cmp, edgecolor='none')  # cmp(np.max(total_amp)))
            plt.suptitle(r'$\sum_{\omega > \omega_c} |\psi_i|^2$, with $\omega_c = $' + '{0:0.02f}'.format(omc))
            plt.savefig(outdir + 'charge_' + '{0:05d}'.format(jj) + '.png', dpi=150)
            plt.close('all')
            tot_amp.append(total_amp)
            in_sum_list.append(in_sum)
            jj += 1

        # Make movie (turned off for now)
        if False:
            imgname = outdir + 'charge_'
            imgdir = outdir
            movname = dio.prepdir(self.lp['meshfn']) + 'charge_omegac' + self.lp['meshfn_exten']
            lemov.make_movie(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False,
                             save_into_subdir=False)

        if deform_difference:
            ######################################
            # DEFORM: shrink will shrink a bond, deltaB makes a delta function in the NNN hopping
            ######################################
            # twist or shrink lattice a little bit
            fig, ax = leplt.initialize_1panel_centered_fig(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
            xy = copy.deepcopy(self.lattice.xy)
            NL = self.lattice.NL

            if deformation == 'shrink':
                # Shrink a single bond to compute charge difference between these two
                cboxind = np.where(np.logical_and(np.abs(xy[:, 0]) < 0.5, np.abs(xy[:, 1] < 0.5)))[0]
                cind = cboxind[np.argmin(np.abs(xy[cboxind, 1]))]
                neighbors = NL[cind][np.where(self.lattice.KL[cind])[0]]
                thetas = np.arctan2(xy[neighbors, 1] - xy[cind, 1], xy[neighbors, 0] - xy[cind, 0])
                # find neighbor closest to directly above the particle and move particle closer to that neighbor
                nind = np.argmin(np.mod(thetas - np.pi*0.4999, np.pi))
                xy[cind, :] = 0.5 * (xy[cind, :] + xy[neighbors[nind], :])
                self.lattice.xy = xy
                eigval, eigvect = self.eig_vals_vects(attribute=True)
            elif deformation == 'twist':
                pass
            elif deformation == 'deltaB':
                pass

            new_amp = []
            jj = 0
            for in_sum in in_sum_list:
                tmp = np.sum(np.abs(eigvect[in_sum, :])**2, axis=0)
                print 'np.shape(tmp) = ', np.shape(tmp)
                print 'tmp[0:10] = ', tmp[0:10]
                total_amp = np.array([np.sqrt(tmp[2*ii] + tmp[2*ii + 1]) for ii in range(int(len(tmp)*0.5))])
                # plot result
                fig, ax = leplt.initialize_1panel_centered_fig(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
                [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
                print 'max = ', np.max(total_amp)
                ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp/np.max(total_amp),
                           c=total_amp, cmap=cmp, edgecolor='none')
                plt.suptitle(r'$\sum_{\omega > \omega_c} |\psi_i|^2$, with $\omega_c = $' +
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
                fig, ax, cbar_ax = leplt.initialize_1panel_cbar_cent(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
                [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
                print 'max = ', np.max(total_amp)
                netchange = np.sum(total_amp)
                sizes = 200 * np.abs(total_amp) / np.max(np.abs(total_amp))
                sc = ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=sizes, edgecolors='none', c=total_amp,
                                cmap=cmp, vmin=-np.max(np.abs(total_amp)), vmax=np.max(np.abs(total_amp)), zorder=1000)
                maxval = float(np.floor(np.max(np.abs(total_amp) * 1000)) * 0.001)
                ticks = [-maxval, 0, maxval]
                cbar = plt.colorbar(sc, cax=cbar_ax, orientation='horizontal', ticks=ticks)
                plt.suptitle(r'$\Delta \left[\sum_{\omega > \omega_c} |\psi_i|^2 \right]$, with $\omega_c = $' +
                             '{0:0.02f}'.format(omegac[kk - len(omegac)]) + '\nnet change={0:0.3e}'.format(netchange))
                plt.savefig(outdir + 'dcharge_' + '{0:05d}'.format(jj) + '.png', dpi=150)
                # Highlight negative contributions
                negind = np.where(total_amp < 0.0)[0]
                print 'negind = ', negind
                print 'self.lattice.xy[negind, 0] =', self.lattice.xy[negind, 0]
                print 'sizes[negind] =', sizes[negind]
                sc = ax.scatter(self.lattice.xy[negind, 0], self.lattice.xy[negind, 1], s=sizes[negind],
                                edgecolors='none', c='b', zorder=1000)
                plt.savefig(outdir + 'dcharge_' + '{0:05d}'.format(jj) + '_negatives.png', dpi=150)
                plt.close('all')
                diff_amp.append(total_amp)
                jj += 1

            return (total_amp, new_amp, diff_amp)
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
        ill = localization[:, 2]
        ill_full = np.zeros(len(eigval), dtype=float)
        ill_full[0:int(len(eigval)*0.5)] = ill[::-1]
        ill_full[int(len(eigval)*0.5):len(eigval)] = ill

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
            leplt.initialize_eigvect_DOS_header_plot(eigval, self.lattice.xy, sim_type='gyro',
                                                     preset_cbar=False, orientation='portrait',
                                                     cbar_pos=[0.79, 0.80, 0.012, 0.15],
                                                     colorV=ill_full, colormap='viridis',
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

    def plot_eigvect_excitation(self, en, eigval=None, eigvect=None, ax=None, plot_lat=True, bwlw=None, **kwargs):
        """Plot the excitation of a normal mode on axis ax. If plot_lat==True, also draw the network as BW network.

        Parameters
        ----------
        en : int
            The index of the eigenvalue to plot
        plot_lat : bool
        **kwargs : keyword arguments for leplt.plot_eigvect_excitation()
            marker_num=0, black_t0lines=False, mark_t0=True, title='auto', normalization=1., alpha=0.6, lw=1, zorder=10,
            cmap='isolum_rainbow'

        Returns
        -------
        fig, ax : matplotlib.pyplot.figure and matplotlib.pyplot.axis instances
            The figure and axis on which the excitation are plotted
        [scat_fg, pp, f_mark, lines12_st] : matplotlib object handles
            the handles for the plotted excitations, lines, dots, etc
        """
        if ax is None:
            fig, ax = leplt.initialize_1panel_cbar_fig()
        else:
            fig = plt.gcf()

        if plot_lat:
            self.lattice.plot_BW_lat(fig=fig, ax=ax, save=False, close=False, axis_off=False, title='', lw=bwlw)

        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect()

        fig, [scat_fg, pp, f_mark, lines12_st] = leplt.plot_eigvect_excitation(self.lattice.xy, fig, None, ax, eigval,
                                                                               eigvect, en, **kwargs)

        return fig, ax, [scat_fg, pp, f_mark, lines12_st]


if __name__ == '__main__':
    '''Use the GyroLattice class to create a density of states, compute participation ratio binned by polygons,
    or build a lattice with the DOS

    Example usage:
    python gyro_lattice_class.py -save_lattice -LT hexagonal -shape hexagon -periodic -N 10 -Omg -1.0
    python gyro_lattice_class.py -load_and_resave -LT hexagonal -shape hexagon -periodic -N 10 -Omg -10.0
    python gyro_lattice_class.py -save_ipr -LT hexagonal -shape hexagon -N 20
    python gyro_lattice_class.py -gap_scaling -LT hexagonal -shape hexagon -periodic -N 10
    python run_series.py -pro gyro_lattice_class -opts periodic/-LT/hexagonal/-shape/square/-NV/5/-NH/40/-localization -var Vpin 0.1:1.0
    '''
    import lepm.lattice_class as lattice_class
    import lepm.gyro_lattice_class_scripts as glatscripts
    import lepm.gyro_lattice_class_scripts_twistbc as glattwistscripts
    import lepm.gyro_lattice_class_scripts_twist2d as glattwist2dscripts
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create GyroLattice class instance,' +
                                                 ' with options to save or compute attributes of the class.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-dispersion', '--dispersion', help='Draw infinite/semi-infinite dispersion relation',
                        action='store_true')
    parser.add_argument('-disp_abtrans', '--dispersion_abtransition',
                        help='Draw infinite/semi-infinite dispersion relation through AB transition',
                        action='store_true')
    parser.add_argument('-plot_matrix', '--plot_matrix', help='Plot the dynamical matrix',
                        action='store_true')
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
    parser.add_argument('-elocz', '--edge_localization',
                        help='Compute localization of modes to the boundary and plot the fits', action='store_true')
    parser.add_argument('-ldos', '--load_calc_ldos', help='Compute local density of states', action='store_true')
    parser.add_argument('-gap_scaling', '--gap_scaling', help='Study scaling of the numerical gap', action='store_true')
    parser.add_argument('-localization', '--localization', help='Seek exponential localization', action='store_true')
    parser.add_argument('-save_eig', '--save_eig', help='Save eigvect/val during get_localization', action='store_true')
    parser.add_argument('-save_images', '--save_images', help='Save movie for localization', action='store_true')
    parser.add_argument('-charge', '--plot_charge', help='Sum amplitudes of modes in band', action='store_true')
    parser.add_argument('-save_vpins', '--save_vpins', help='Save 100 guassian pin realizations', action='store_true')
    parser.add_argument('-pin2hdf5', '--pin2hdf5',
                        help='Enforce saving pinning as hdf5 rather than as txt', action='store_true')
    parser.add_argument('-omegac', '--omegac', help='Cutoff (upper) freq for summing charge', type=float, default=0.0)
    parser.add_argument('-illpoly', '--plot_localized_states',
                        help='Plot all localized states and show non-hex polygons', action='store_true')
    parser.add_argument('-frange', '--freq_range', help='Range of freqs to analyze in illpoly', type=str, default='0/0')
    parser.add_argument('-OmKspec', '--OmKspec', help='string specifier for OmK bond frequency matrix',
                        type=str, default='')
    # examples of OmKspec:
    #   'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
    #      'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') +\
    #      'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n') --> dicing lattice
    #   'Vpin0p10' --> a specific, single random configuration of potentials
    #   'Vpin0p10theta0p00phi0p00' --> a specific, single random configuration of potentials with twisted BCs
    parser.add_argument('-dice_glat', '--dice_glat',
                        help='Create a glat with bonds that are weakened in a gridlike fashion (bonds crossing '
                             'gridlines are weak)',
                        action='store_true')
    parser.add_argument('-dice_eigval', '--dice_eigval', help='Compute eigvals/vects if dice_glat is True',
                        action='store_true')
    parser.add_argument('-gridspacing', '--gridspacing', help='Spacing of gridlines for dice_glat', type=float,
                        default=7.5)
    parser.add_argument('-weakbond_val', '--weakbond_val',
                        help='Spring frequency of bonds intersecting dicing lines for dice_glat', type=float,
                        default=-0.5)
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
    parser.add_argument('-basis', '--basis', help='basis for computing eigvals', type=str, default='XY')
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

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
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

    print 'theta = ', theta
    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': args.rootdir,
          'phi_lattice': args.phi_lattice,
          'delta_lattice': args.delta_lattice,
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
          'conf': args.realization_number,
          'subconf': args.sub_realization_number,
          'periodicBC': args.periodicBC or args.periodic_strip,
          'periodic_strip': args.periodic_strip,
          'loadlattice_z': args.loadlattice_z,
          'alph': args.alph,
          'eta_alph': args.eta_alph,
          'origin': np.array([0., 0.]),
          'basis': args.basis,
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
          'immobile_boundary': args.immobile_boundary,
          'bl0': args.bl0,  # this is for prestrain in the bonds (the bonds are stretched to their current positions)
          'save_pinning_to_hdf5': args.pin2hdf5,
          'kicksz': args.kicksz,
          'theta_twist': args.thetatwist,
          'phi_twist': args.phitwist,
          'aratio': args.aratio,
          }

    # Loading lattice example:
    if args.nice_plot:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        print 'Saving nice BW plot...'
        lat.plot_BW_lat(meshfn=lat.lp['meshfn'], ptcolor='k', ptsize=15)

    if args.dispersion:
        """Example usage:
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -dispersion -basis psi
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -dispersion
        # make the strip
        python ./build/make_lattice.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -skip_polygons -skip_gyroDOS
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        glat = GyroLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        glat.save_eigval_eigvect(attribute=True, save_png=True)
        glat.infinite_dispersion(save=False, nkxvals=50, nkyvals=25)

    if args.dispersion_abtransition:
        """Example usage:
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -disp_abtrans -basis psi
        """
        glatscripts.dispersion_abtransition(lp)

    if args.save_prpoly:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        glat = GyroLattice(lat, lp)
        glat.load()
        # glat.calc_pr_polygons(check=False)
        glat.save_prpoly(save_plot=True)

    if args.plot_charge:
        """Sum amplitude of all modes in a supplied band (specified by omegac)

        Example usage:
        python gyro_lattice_class.py -LT hexagonal -shape hexagon -N 5 charge
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        glat = GyroLattice(lat, lp)
        glat.load()
        glat.sum_amplitudes_band_spectrum(omegac=None, deform_difference=True)

    if args.plot_localized_states:
        """Load a periodic lattice from file, provide physics, and plot the modes which are localized, along with
        polygons.

        Example Usage:
        python gyro_lattice_class.py -LT randomcent -shape square -N 20 -conf 02 -periodic -illpoly -frange 0.0/3.5
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        if args.freq_range != '0/0':
            frange = le.string_to_array(args.freq_range, dtype=float)
        else:
            frange = None
        glat.plot_localized_state_polygons(save=True, frange=frange)

    if args.dice_glat:
        glatscripts.dice_glat(lp, args)

    if args.dcdisorder:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        print 'loading lattice...'
        lat.load()
        glat = GyroLattice(lat, lp)
        # glat.load()
        print 'Computing eigvals/vects...'
        glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect()
        # glat.save_OmK_Omg()
        print 'Saving DOS movie...'
        glat.save_DOSmovie()

    if args.make_DOSmovie:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        print 'loading lattice...'
        glat.load()
        print 'Saving DOS movie...'
        glat.save_DOSmovie()

    if args.save_ipr:
        '''Example usage
        python run_series.py -pro gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-save_ipr -var Vpin 0.1/0.5/1.0/2.0

        python run_series.py -pro gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-DOSmovie -var Vpin 0.1/5/1.0/2.0
        '''

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        print 'saving ipr...'
        glat.save_ipr(vmax=8.0, xlim=(0., 10.), cbar_labelpad=15)
        # This would save an image of the ipr-colored DOS, but this is already done in save_ipr if save_images=True
        # glat.plot_ipr_DOS(save=True, vmax=5)

    if args.load_lattice_resave_physics:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        print 'saving eigvals/vects...'
        glat.save_eigval_eigvect()
        print 'saving ipr...'
        glat.save_ipr(show=False)
        print 'saving localization...'
        glat.save_localization(save_images=False)
        # print 'Saving DOS movie...'
        # glat.save_DOSmovie()

    if args.save_lattice:
        lat = lattice_class.Lattice(lp)
        print 'Building lattice...'
        lat.build()
        glat = GyroLattice(lat, lp, OmK='auto', Omg='auto')
        print 'Saving gyro lattice...'
        glat.lattice.save()
        print 'Computing eigvals/vects...'
        glat.eig_vals_vects(attribute=True)
        glat.save_eigval_eigvect()
        print 'Saving DOS movie...'
        glat.save_DOSmovie()

    if args.load_calc_ldos:
        """Load a (gyro) lattice from file, calc the local density of states and save it
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        glat.save_ldos()

    if args.localization:
        glatscripts.localization(lp, args)

    if args.gap_scaling:
        glatscripts.gap_scaling(lp, args)

    if args.save_vpins:
        '''Example usage:
        python gyro_lattice_class.py -LT randorg_gammakick0p50_cent -periodic -shape square -spreading_time 0.3 -NP 2500 \
            -N 50 -conf 1 -save_vpins -pin2hdf5 -Vpin 0.4
        
        
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/1/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/2/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/3/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/4/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/5/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/6/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/7/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/8/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/9/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        python run_series.py -pro gyro_lattice_class -opts LT/randorg_gammakick0p50_cent/-periodic/-shape/square/-spreading_time/0.3/-NP/2500/-N/50/-conf/10/-save_vpins/-pin2hdf5 -var Vpin 0.1:0.05:1.4
        '''
        lpmaster = copy.deepcopy(lp)
        lat = lattice_class.Lattice(lpmaster)
        lat.load()
        for pconf in np.arange(11):
            lp = copy.deepcopy(lpmaster)
            lp['pinconf'] = pconf
            glat = GyroLattice(lat, lp)

    if args.twistbcs:
        # Make sure bcs are periodic
        """Load a periodic lattice from file, twist the BCs by phases theta_twist and phi_twist with vals finely spaced
        between 0 and 2pi. Then compute the berry curvature associated with |alpha(theta, phi)>

        Example usage:
        python haldane_lattice_class.py -twistbcs -N 3 -LT hexagonal -shape square -periodic
        """
        glattwist2dscripts.twistbcs(lp)

    if args.edge_localization:
        '''Example usage:
        python gyro_lattice_class.py -elocz -LT hexagonal -shape hexagon -N 4 -AB 0.1 -edge_localization
        python gyro_lattice_class.py -elocz -LT hexagonal -shape hexagon -NH 14 -NV 7 -edge_localization
        '''
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        glat.plot_edge_localization()

    if args.twiststrip:
        """Example usage:
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 14 -NV 7 -twiststrip
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 10 -NP 50 -twiststrip 
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 15 -NP 50 -twiststrip
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 15 -NP 50 -twiststrip -conf 2
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 20 -NV 17 -NP 20 -twiststrip
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 30 -NP 50 -twiststrip
        python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 30 -alph 0.3 -twiststrip
        python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 40 -alph 0.25 -twiststrip
        python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 50 -alph 0.3 -twiststrip -conf 3
         
        # to make more periodicstrips
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 20 -NV 17 -NP 20 -skip_polygon -skip_gyroDOS
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 15 -NP 50 -skip_polygon -skip_gyroDOS -conf 2
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 15 -NP 50 -skip_polygon -skip_gyroDOS -conf 2

        # to make annuli
        python ./build/make_lattice.py -LT hucentroid_annulus -N 50 -alph 0.3 -skip_polygons -skip_gyroDOS -conf 2
        python ./build/make_lattice.py -LT hucentroid_annulus -N 40 -alph 0.25 -skip_polygons -skip_gyroDOS
        python ./build/make_lattice.py -LT hucentroid_annulus -N 30 -alph 0.2 -skip_polygons -skip_gyroDOS
        """
        glattwistscripts.twiststrip(lp)

    if args.twistmodes:
        """Example usage:
        python gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 5 -NV 5 -twistmodes
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 10 -NP 50 -twistmodes
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 20 -NP 50 -twistmodes
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 30 -NP 50 -twistmodes -conf 3
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 20 -NV 17 -NP 20 -twistmodes
        python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 50 -alph 0.3 -twistmodes -conf 3

        # to make more periodicstrips
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 20 -NV 17 -NP 20 -skip_polygon -skip_gyroDOS
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 30 -NP 50 -skip_polygon -skip_gyroDOS
        python ./build/make_lattice.py -LT hucentroid -periodic_strip -NH 50 -NV 20 -NP 50 -skip_polygon -skip_gyroDOS -conf 3
        python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 8 -NV 5 -skip_polygon -skip_gyroDOS
        python ./build/make_lattice.py -LT hexagonal -periodic_strip -NH 5 -NV 5 -skip_polygon -skip_gyroDOS

        # to make annuli
        python ./build/make_lattice.py -LT hucentroid_annulus -N 50 -alph 0.3 -skip_polygons -skip_gyroDOS -conf 2
        """
        glattwistscripts.twistmodes(lp)

    if args.twistmodes_spiral:
        """Example usage:
        python gyro_lattice_class.py -LT hucentroid -shape square -periodic_strip -NH 50 -NV 30 -NP 50 \
            -twistmodes_spiral -startfreq 2.07
        """
        if args.twistmodes_thres0 < 0:
            thres0 = None
        else:
            thres0 = args.twistmodes_thres0
        glattwistscripts.twistmodes_spiral(lp, nrungs=args.twistmodes_nrungs, startfreq=args.twistmodes_startfreq,
                                           springsubplot=args.twistmodes_springax, thres0=thres0)

    if args.edgelocalization:
        """Example usage
        python gyro_lattice_class.py -LT hexagonal -periodic_strip -NH 14 -NV 7 -edgelocalization -thetatwist 0.4
        python gyro_lattice_class.py -LT hexagonal -periodic_strip -NH 8 -NV 5 -edgelocalization -thetatwist 0.4
        python gyro_lattice_class.py -LT hucentroid_annulus -shape annulus -N 30 -alph 0.3 -edgelocalization -thetatwist 0.4
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = GyroLattice(lat, lp)
        print 'loading haldane_lattice...'
        glat.load()
        # get the localization of these eigenvalues
        locz = glat.plot_edge_localization()
        plt.close('all')

    if args.plot_matrix:
        """Example usage:
        python gyro_lattice_class.py -LT stackedrhombic -N 1 -periodic -plot_matrix -phi 0.200
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        glat = GyroLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        matrix = glat.get_matrix()
        le.plot_complex_matrix(matrix, show=False, close=False)
        print 'here'
        plt.title('Dynamical matrix:' + lp['meshfn_exten'])
        outfn = dio.prepdir(lp['meshfn']) + 'matrix_' + lp['meshfn_exten'] + '.png'
        print 'saving ' + outfn
        plt.savefig(outfn)
        aa, bb = np.linalg.eig(matrix)
        print 'eigvals =', aa
        glat.get_eigval(attribute=True)
        print 'glat.eigvals = ', glat.eigval

    # Building lattice example:
    # lat.build()
    # glat = GyroLattice(lat,lat.lp)
    # ipr = glat.calc_ipr(attribute=True)
    # glat.plot_ipr_DOS(outdir='/Users/npmitchell/Desktop/')
    # prpoly = glat.calc_pr_polygons(check=False)

    # for ii in prpoly:
    #     fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(glat.eigval, 'gyro', alpha=1.0,
    #                                                                    colorV=prpoly[ii]*ipr,
    #                                                                    colormap='viridis', linewidth=0,
    #                                                                    cax_label=r'$p^{-1}$')
    #     plt.savefig('/Users/npmitchell/Desktop/prpoly_'+str(ii)+'.png')
    #     plt.close('all')

    # glat.plot_ipr_DOS(inverse_PR=False)
    # plt.plot(np.imag(glat.eigval))
    # plt.show()
    # plt.plot(np.imag(glat.eigval), glat.ipr, 'b.')
    # plt.title('Inverse participation Ratio')
    # plt.ylabel('ipr')
    # plt.xlabel(r'$\omega$')
    # plt.show()
