import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.magnetic_gyro_functions as mgfns
import lepm.stringformat as sf
import lepm.plotting.plotting as leplt
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import lepm.gyro_lattice_functions as glatfns
import os
import glob
import lepm.plotting.gyro_lattice_plotting_functions as glatpfns
import lepm.magnetic_gyro_kspace_functions as mglatkspacefns
import lepm.plotting.movies as lemov
import lepm.plotting.magnetic_gyro_movies as magmov
import copy
import h5py
import lepm.hdf5io as h5io
import lepm.gyro_lattice_functions_localization as glatlzfns
import lepm.magnetic_gyro_lattice_functions_localization as mgloczfns
import lepm.data_handling as dh

'''file that contains the Magnetic_Gyro_Lattice class'''


class MagneticGyroLattice:
    def __init__(self, lattice, lp, xytup=None, NL=None, KL=None, PVx=None, PVy=None,
                 BL_nm=None, NL_nm=None, KL_nm=None,
                 Omg=None, OmK=None, matrix=None, eigval=None, eigvect=None, eps=1e-8, overwrite=False):
        """Initializes the class.
        Rout is the full lattice including the outer particles which are fixed. Rin is all the inner gyroscopes not
        fixed. Free parameters for physics are Omg Omk aoverl
        If supplied, lp['roi'] differentiates inner, mobile particles from all other sites (#vertices x 2 float array),
        but otherwise the boundary is designated as the stationary, outer portion of the lattice
        
        Parameters
        ----------
        lattice : Lattice class instance
            instance of lepm.lattice_class.Lattice. lattice.xy are defaulted to be the entire set of points
        lp : dict
            containing parameters about the magnetic gyro lattice, including roi, Omg, Omk, free_bc
        xytup : tuple of (M x 2 float array, N x 2 float array)
            Equilibrium positions of inner gyros, all gyros
        NL : N x N int array, where N is # total sites (including outer sites)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
        KL : N x N int array, where N is # total sites (including outer sites)
            Coupling array, where KL[i,j] describes the connetion of particle i to particle NL[i,j].
            1 is a true connection, 0 signifies that there is not a connection, -1 is periodic connection
        NL_nm : M x (max#NN) int array or None
            non magnetic Neighbor list, for plotting. Elements index self.xy_inner. This is the same as the lattice NL
        KL_nm : M x (max#NN) int array or None
            non magnetic bond list, for plotting. Elements match NL_nm, which indexes self.xy_inner.
            This is the same as the lattice KL.
        Omg : M x 1 float array or None
            pinning frequencies for the mobile particles
        OmK : N x N float array or None (currently not implemented)
            scaling for magnetic interaction strength between given pairs of magnets (mobile and immobile)
        matrix : M x M float array or None
            dynamical matrix for the mobile gyros
        eigval : M x 1 complex array or None
            eigenvalues of the dynamical matrix
        eigvect : M x M complex array or None
            eigenvectors of the dynamical matrix

        Class members
        ----------
        lattice :
        inner_indices : # mobile gyros x 1 int array
            Note that for some networks, we need to define inner indices manually in a saved file in meshfn
        outer_indices : # immobile gyros x 1 int array
        xy_inner : # mobile gyros x 2 float array
        xy : tuple of (Mx3 float array, N x 3 float array)
            Equilibrium positions of all the gyroscopes, equilibrium xy of gyros and xy of immobile magnet sites
        inner_boundary : M x 1 int array
            the idices of mglat.xy that mark the boundary of mglat.xy_inner
        inner_boundary_inner : M x 1 int array
            the idices of mglat.xy_inner that mark the boundary of mglat.xy_inner
        NL : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
        KL : matrix of dimension n x (max number of neighbors)
            Correponds to NL matrix.  1 corresponds to a true connection while 0 signifies that there is not a
            connection
        spring : float
            spring constant
        pin : float
            gravitational spring constant
        matrix : matrix of dimension 2*(len(mglat.xy_inner)) x 2*(len(mglat.xy_inner))
            Linearized matrix for finding normal modes of system
        eigval : array of dimension 2*(len(mglat.xy_inner)) x 1
            Eigenvalues of self.matrix
        eigvect : array of dimension 2*(len(mglat.xy_inner)) x 2*(len(mglat.xy_inner))
            Eigenvectors of self.matrix
        """
        # xytup is a tuple of inner points, outer points. Define self.xy to be all sites, including immobile ones
        self.lattice = lattice
        # lp contains Omg, Omk, als. With weights, Omg = 1.5, otherwise Omg = 1.0
        # Omk = k_m l^2 / I omega -> 0.86 in original experiments
        # aoverl = 0.0305/.038
        self.lp = lp
        self.inner_indices = None
        self.outer_indices = None

        if xytup is None:
            # Create self.xy and self.xy_inner
            self.xy = lattice.xy
            if 'roi' in self.lp:
                self.xy_inner = le.pts_in_polygon(self.xy, self.lp['roi'])
            else:
                # get boundary of the lattice and denote these particles as the outer particles
                # boundary = le.extract_boundary(self.xy, self.lattice.NL, self.lattice.KL, self.lattice.BL,
                #                                check=False)
                boundary = self.lattice.get_boundary()
                if boundary is None:
                    self.xy_inner = self.xy
                elif isinstance(boundary, tuple):
                    # There are multiple boundaries. Exclude all of them from inner_indices
                    inners = np.arange(len(self.xy))
                    outers = []
                    for bndy in boundary:
                        inners = np.setdiff1d(inners, np.array(bndy))
                        outers.append(bndy)

                    self.inner_indices = np.hstack(inners)
                    self.outer_indices = np.hstack(outers)
                    self.xy_inner = self.xy[self.inner_indices]
                else:
                    self.inner_indices = np.setdiff1d(np.arange(len(self.xy)), np.array(boundary))
                    self.outer_indices = np.array(boundary)
                    self.xy_inner = self.xy[self.inner_indices]
        else:
            self.xy = xytup[1]
            self.xy_inner = xytup[0]

        if NL is not None and KL is not None:
            self.NL = NL
            self.KL = KL
            # determine if shortrange or longrange or intermediate interaction_range from supplied NL
            # if lp['intrange'] == 1:
            #     nl_meshfn_exten = '_shortrange'
            if lp['intrange'] > 0:
                nl_meshfn_exten = '_intrange{0:04d}'.format(mglat.lp['interaction_range'])
            else:
                nl_meshfn_exten = ''
            # Get PVxy if not supplied
            if PVx is not None and PVy is not None:
                self.PVx, self.PVy = PVx, PVy
            else:
                if (KL > -eps).all():
                    self.PVx = np.zeros_like(NL, dtype=float)
                    self.PVy = np.zeros_like(NL, dtype=float)
                else:
                    raise RuntimeError('periodic KL is supplied, but PVx and PVy are not supplied')
        else:
            self.NL, self.KL, self.PVx, self.PVy, nl_meshfn_exten = mgfns.create_NL_KL_pv(self)

        # print 'self.xy_inner = ', np.shape(self.xy_inner)
        # print 'self.xy = ', np.shape(self.xy)
        if self.inner_indices is None or self.outer_indices is None:
            self.outer_indices, self.inner_indices, self.total2inner = self.outer_inner(self.xy, self.xy_inner)

        # Add lattice properties to gyro_lattice_properties, by convention, but don't overwrite params
        for key in self.lattice.lp:
            if key not in self.lp:
                self.lp[key] = self.lattice.lp[key]

        ########################################################################
        self.lp['meshfn_exten'] = '_magnetic' + nl_meshfn_exten + '_aol' + sf.float2pstr(self.lp['aoverl'])
        # For magnetic class, OmK is bond dependent: OmK = 3 mu_0 M^2/ (pi a^5) for each bond
        if self.lp['Omk'] != -1.0:
            self.lp['meshfn_exten'] += '_Omk' + sf.float2pstr(self.lp['Omk'])

        # Build OmK array for couplings
        self.OmK, lp_Omk, omk_meshfn_exten = mgfns.build_OmK(self, self.lp, None)

        # Form Omg
        if Omg == 'auto' or Omg is None:
            if 'Omg' in self.lp:
                self.Omg = self.lp['Omg'] * np.ones_like(self.xy_inner[:, 0])
            else:
                print 'giving Omg the default value of -1s...'
                self.Omg = -1.0 * np.ones_like(self.xy_inner[:, 0])
            if self.lp['Omg'] != -1.0:
                self.lp['meshfn_exten'] += '_Omg' + sf.float2pstr(self.lp['Omg'])
        else:
            self.Omg = Omg
            if 'Omg' in self.lp:
                if (self.Omg != self.lp['Omg'] * np.ones(np.shape(self.xy_inner)[0])).any():
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
            if np.abs(self.lp['ABDelta']) > eps:
                self.lp['meshfn_exten'] += '_ABd{0:0.3f}'.format(self.lp['ABDelta']).replace('.', 'p').replace('-', 'n')
        else:
            self.lp['ABDelta'] = 0.

        # First check for gaussian disorder
        if 'V0_pin_gauss' in self.lp and 'V0_spring_gauss' in self.lp:
            if abs(self.lp['V0_pin_gauss']) > eps or (self.lp['V0_spring_gauss']) > eps:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_pinV' + sf.float2pstr(self.lp['V0_pin_gauss'])
                self.lp['meshfn_exten'] += '_sprV' + sf.float2pstr(self.lp['V0_spring_gauss'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_conf{0:04d}'.format(self.lp['pinconf'])
            else:
                self.lp['dcdisorder'] = False

        # If no gaussian disorder check for flat disorder
        if 'V0_pin_flat' in self.lp and 'V0_spring_flat' in self.lp and not self.lp['dcdisorder']:
            if abs(self.lp['V0_pin_flat']) > eps or (self.lp['V0_spring_flat']) > eps:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_pinVflat' + sf.float2pstr(self.lp['V0_pin_flat'])
                self.lp['meshfn_exten'] += '_sprVflat' + sf.float2pstr(self.lp['V0_spring_flat'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_conf{0:04d}'.format(self.lp['pinconf'])
            else:
                self.lp['dcdisorder'] = False

        # If neither gaussian nor flat, input no disorder
        if not self.lp['dcdisorder']:
            self.lp['V0_pin_gauss'] = 0.
            self.lp['V0_spring_gauss'] = 0.
            self.lp['V0_pin_flat'] = 0.
            self.lp['V0_spring_flat'] = 0.
            self.lp['pinconf'] = 0
            self.lp['dcdisorder'] = False

        vpin_has_gaussian = self.lp['V0_pin_gauss'] > eps or self.lp['V0_spring_gauss'] > eps
        vpin_has_flat = self.lp['V0_pin_flat'] > eps or self.lp['V0_spring_flat'] > eps
        if abs(self.lp['ABDelta']) > eps or vpin_has_gaussian or vpin_has_flat:
            # In order to load the random (V0) or alternating (AB) pinning sites, look for a txt file with the pinnings
            # that also has specifications in its meshfn exten, but IGNORE other parts of meshfnexten, if they exist.
            # Form abbreviated meshfn exten
            pinmfe = self.get_pinmeshfn_exten()

            print 'Trying to load offset/disorder to pinning frequencies: '
            print dio.prepdir(self.lp['meshfn']) + pinmfe
            # Attempt to load from file
            if not overwrite:
                try:
                    self.load_pinning(meshfn=self.lp['meshfn'])
                    # self.Omg = np.loadtxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt')
                    # raise RuntimeError('This means an erronous Omg was saved --where was it saved?')
                    print 'Loaded ABDelta and/or dcdisordered pinning frequencies.'
                    define_Omg_now = False
                except IOError:
                    define_Omg_now = True
            else:
                define_Omg_now = True

            if define_Omg_now:
                print 'Could not load ABDelta and/or dcdisordered pinning frequencies, defining them here...'
                # Make Omg from scratch
                if abs(self.lp['ABDelta']) > eps:
                    asites, bsites = glatfns.ascribe_absites(self.lattice)
                    self.Omg[asites[self.inner_indices]] += self.lp['ABDelta']
                    self.Omg[bsites[self.inner_indices]] -= self.lp['ABDelta']
                if vpin_has_flat or vpin_has_gaussian:
                    self.add_dcdisorder()

                # Save non-standard Omg
                if 'save_pinning_to_hdf5' in self.lp:
                    if self.lp['save_pinning_to_hdf5']:
                        force_hdf5pin = True
                    else:
                        force_hdf5pin = False
                else:
                    force_hdf5pin = False
                self.save_Omg(infodir=self.lp['meshfn'], histogram=False, force_hdf5=force_hdf5pin, overwrite=overwrite)
                # self.plot_Omg()
                # np.savetxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt', self.Omg)
                # self.plot_Omg()

        ########################################################################

        # Non magnetic --> nm: taken from lat.NL, lat.KL, these are the inner particles (xy_in) connectivity M x M
        if BL_nm is not None:
            self.BL_nm = BL_nm
            if NL_nm is not None and KL_nm is not None:
                self.NL_nm = NL_nm
                self.KL_nm = KL_nm
            else:
                self.NL_nm, self.KL_nm = le.BL2NLandKL(self.BL_nm)
        else:
            # Remove self.outer_indices from lattice, and use resulting BL to get NL_nm and KL_nm
            # print 'magnetic_gyro_lattice_class: BL = ', self.lattice.BL
            # print 'magnetic_gyro_lattice_class: keep = ', self.inner_indices
            xytmp, self.NL_nm, self.KL_nm, self.BL_nm, self.PVxydict = \
                le.remove_pts(self.inner_indices, self.lattice.xy, self.lattice.BL, PVxydict=self.lattice.PVxydict,
                              PV=self.lattice.PV)

        # print 'lp[meshfn_exten] = ', self.lp['meshfn_exten']
        # Get inner boundary
        self.inner_boundary = mgfns.calc_boundary_inner(self, check=False)
        # Use dh to get indices of xy_inner that are on the 'inner' network's boundary
        # Note match_values returns inds such that vals[inds] ~= arr
        if self.inner_boundary is not None:
            self.inner_boundary_inner = dh.match_values(self.inner_boundary, self.inner_indices)
        else:
            self.inner_boundary_inner = None
        # print 'self.inner_indices = ', self.inner_indices
        # print 'self.inner_boundary = ', self.inner_boundary
        # print 'self.inner_boundary_inner = ', self.inner_boundary_inner
        # sys.exit()

        # Other attributes that can be calculated
        self.matrix = matrix
        self.eigval = eigval
        self.eigvect = eigvect
        self.localization = None
        self.edge_localization = None
        self.ipr = None

    def __hash__(self):
        return hash(self.lattice)

    def __eq__(self, other):
        return hasattr(other, 'lattice') and self.lattice == other.lattice

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def load_pinning(self, meshfn=None):
        """Load the Omg vector for this instance of GyroLattice"""
        # First try to load from hdf5 file, if it exists
        # If hdf5 file does not exist or contain the pinning for this meshfn_exten, attempt to load from txt file

        # self.Omg = np.loadtxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt')
        #     print 'Loaded ABDelta and/or dcdisordered pinning frequencies.'

        if meshfn is None:
            meshfn = self.lp['meshfn']
        pinning_name = self.get_pinmeshfn_exten()
        pinfn = dio.prepdir(meshfn) + 'omg_configs.hdf5'
        if glob.glob(pinfn):
            print('Loading pinning config from ' + pinfn)
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

    def outer_inner(self, xy, xyin):
        """Given two xy float arrays, return the indices of sites (index as in xy) which are inside (ie also appear
        in xyin) and outside (do not appear in xyin)

        Parameters
        ----------
        xy : N x 2 float array
            All sites
        xyin : M x 2 float array
            sites that are to be considered mobile in the MagneticGyroLattice class

        Returns
        -------
        out_index : int list
            indices of the immobile boundary particles
        in_index : int list
            indices of the inner mobile particles
        """
        inRx = np.in1d(xy[:, 0], xyin[:, 0])
        inRy = np.in1d(xy[:, 1], xyin[:, 1])
        in_index = np.where(np.logical_and(inRx, inRy))[0]
        out_index = np.setdiff1d(np.arange(len(xy), dtype=int), in_index)
        total2inner = {}
        ii = 0
        for ind in in_index:
            total2inner[ind] = ii
            ii += 1
        return out_index, in_index, total2inner

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
            if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_pin_flat'] > 0 or self.lp['ABDelta'] > 0:
                self.load_pinning(meshfn=meshfn)
            else:
                self.Omg = self.lp['Omg'] * np.ones_like(self.lattice.xy[:, 0])

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

    def get_pinmeshfn_exten(self):
        """Return the name of the file or dataset that stores the pinning frequencies of this network"""
        pinmfe = 'Omg_mean' + sf.float2pstr(self.lp['Omg']) + self.lp['meshfn_exten']
        return pinmfe

    def add_dcdisorder(self):
        """Add gaussian noise to pinning or spring energies (delta-correlated disorder)"""
        # Add gaussian noise to pinning energies
        if self.lp['V0_pin_gauss'] > 0:
            self.Omg += self.lp['V0_pin_gauss']*np.random.randn(len(self.xy_inner))
            raise RuntimeError('Adding gaussian disorder: are you sure you want to proceed?')
        if self.lp['V0_spring_gauss'] > 0:
            print 'This is not done correctly here'
            self.OmK += self.lp['V0_spring_gauss'] * np.random.randn(np.shape(self.lattice.KL)[0],
                                                                     np.shape(self.lattice.KL)[1])
            sys.exit()

        if self.lp['V0_pin_flat'] > 0 or self.lp['V0_spring_flat'] > 0:
            # Note that we multiply by two so that V0_pin_flat is the HALF width of the distribution
            flat_disorder = (np.random.rand(len(self.xy_inner))) * 2 - 1.0

        if self.lp['V0_pin_flat'] > 0:
            self.Omg += self.lp['V0_pin_flat'] * flat_disorder
            if self.lp['Omg'] < 0:
                self.Omg[self.Omg > 0] = 0.
            elif self.lp['Omg'] > 0:
                self.Omg[self.Omg < 0] = 0.

            print('magnetic_gyro_lattice_class.py: V0_pin_flat=', self.lp['V0_pin_flat'])
            print(self.Omg)

        if self.lp['V0_spring_flat'] > 0:
            to_add = self.lp['V0_spring_flat'] * flat_disorder[:, np.newaxis] * np.ones_like(self.OmK)
            self.OmK[np.abs(self.OmK) > 0] += to_add[np.abs(self.OmK) > 0]
            if self.lp['Omk'] < 0:
                self.OmK[self.OmK > 0] = 0.
            elif self.lp['Omk'] > 0:
                self.OmK[self.OmK < 0] = 0.

            print('magnetic_gyro_lattice_class.py: OmK after flat disorder= ')
            print(self.OmK)
            # sys.exit()

    def load_eigval_eigvect(self, attribute=True):
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        eigvect_name = "eigvect" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_magnetic_gyro.hdf5"
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_magnetic_gyro.hdf5"
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
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_magnetic_gyro.hdf5"
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
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_magnetic_gyro.hdf5"
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
            matrix = self.calc_matrix(attribute=attribute, basis=basis)
            return matrix
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
                print 'mlat.get_eigval_eigvect: Could not load eigval/vect, calculating...'
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
                print 'mlat.get_eigval() Cannot load eigval, computing it...'
                # calculate eigval and eigvect
                matrix = self.get_matrix(attribute=False)
                eigval = self.calc_eigvals(matrix=matrix, attribute=attribute)
            else:
                print 'mlat.get_eigval() Loaded eigval...'
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
                print 'mlat.get_eigvect: Could not load eigvect, calculating...'
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
                print 'mlatclass: loaded prpoly.'
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

    def get_localization(self, attribute=False, eigval=None, eigvect=None, save=True, save_eigvect_eigval=False,
                         locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False,
                         force_hdf5=True, overwrite=False):
        """Obtain the localization of eigenvectors of the GyroLattice (fits to 1d exponential decay)

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
                # try to load from hdf5
                hdf5fn = dio.prepdir(self.lp['meshfn']) + 'localization_magneticgyro.hdf5'
                locz_name = 'localization' + self.lp['meshfn_exten']
                print 'mglat: extracting dset...'
                localization = h5io.extract_dset_hdf5(locz_name, hdf5fn)
                print 'localization = ', localization
                if localization is None:
                    # sys.exit()
                    print 'calculating localization...'
                    localization = self.calc_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                                          save_eigvect_eigval=save_eigvect_eigval, locutoffd=locutoffd,
                                                          hicutoffd=hicutoffd, save_eigval=save_eigval,
                                                          attribute_eigv=attribute_eigv)
                    if save:
                        self.save_localization(force_hdf5=force_hdf5, overwrite=overwrite)
        return localization

    def get_ill(self, attribute=False, eigval=None, eigvect=None, save=False, overwrite=False):
        localization = self.get_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                             save=save, overwrite=overwrite)
        ill = localization[:, 2]
        ill_full = np.zeros(2 * len(ill), dtype=float)
        ill_full[0:len(ill)] = ill[::-1]
        ill_full[len(ill):2 * len(ill)] = ill

        return ill_full

    def get_edge_ill(self, attribute=False, eigval=None, eigvect=None, save=False, force_hdf5=True,
                              overwrite=False, save_eigvect_eigval=False):
        localization = self.get_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect, save=save,
                                                  force_hdf5=force_hdf5, overwrite=overwrite,
                                                  save_eigvect_eigval=save_eigvect_eigval)
        ill = localization[:, 2]
        edge_ill = np.zeros(2 * len(ill), dtype=float)
        edge_ill[0:len(ill)] = ill[::-1]
        edge_ill[len(ill):2 * len(ill)] = ill

        return edge_ill

    def get_edge_localization(self, attribute=False, eigval=None, eigvect=None, save=False, force_hdf5=True,
                              overwrite=False, save_eigvect_eigval=False):
        """"""
        # try to load it
        if self.edge_localization is not None:
            edge_localization = self.edge_localization
        else:
            fn = dio.prepdir(self.lp['meshfn']) + 'localization_edge' + self.lp['meshfn_exten'] + '.txt'
            print 'seeking pklfn file ', fn
            if glob.glob(fn):
                print 'loading edge_localization from txt file...'
                edge_localization = np.loadtxt(fn, delimiter=',')
            else:
                # try to load from hdf5
                hdf5fn = dio.prepdir(self.lp['meshfn']) + 'localization_edge_magneticgyro.hdf5'
                locz_name = 'localization_edge' + self.lp['meshfn_exten']
                print 'mglat: extracting dset...'
                edge_localization = h5io.extract_dset_hdf5(locz_name, hdf5fn)
                print 'edge_localization = ', edge_localization
                if edge_localization is None:
                    # sys.exit()
                    print 'calculating localization...'
                    edge_localization = self.calc_edge_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)

                    if save:
                        self.edge_localization = edge_localization
                        self.save_edge_localization(force_hdf5=force_hdf5, overwrite=overwrite,
                                                    save_eigvect_eigval=save_eigvect_eigval)
        return edge_localization

    def ensure_eigval_eigvect(self, eigval=None, eigvect=None, attribute=True, load=True, force_hdf5=True):
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
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        eigvect_name = "eigvect" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_magnetic_gyro.hdf5"
        h5_evt = dio.prepdir(self.lp['meshfn']) + "eigvects_magnetic_gyro.hdf5"
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

    def ensure_eigval(self, eigval=None, attribute=False, load=True, force_hdf5=True):
        """Return eigval and save it to disk if not saved already.
        To obtain eigval, proceed by (1) returning from supplied eigval, (2) calling from self, (3) loading it, or
        (4) calculating it.
        """
        # Make eigval name
        eigval_name = "eigval" + self.lp['meshfn_exten']
        # First look in eigvals_gyro.hdf5
        h5_evl = dio.prepdir(self.lp['meshfn']) + "eigvals_magnetic_gyro.hdf5"
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

    def calc_matrix(self, attribute=False, basis=None):
        """calculates the matrix for finding the normal modes of the system

        Parameters
        ----------
        attribute : bool
            attribute the matrix to self.matrix
        basis : None or str ('XY', 'psi')
            The basis to use to construct the dynamical matrix
        """
        matrix = mgfns.dynamical_matrix_magnetic_gyros(self, basis=basis)
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

    def eig_vals_vects(self, matrix=None, attribute=False, attribute_matrix=False):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        if matrix is None:
            matrix = self.get_matrix(attribute=attribute_matrix)
        eigval, eigvect = np.linalg.eig(matrix)
        si = np.argsort(np.imag(eigval))
        eigvect = np.array(eigvect)
        eigvect = eigvect.T[si]
        eigval = eigval[si]

        if attribute:
            self.eigvect = eigvect
            self.eigval = eigval

        # print 'np.shape(eigvect) = ', np.shape(eigvect)
        # sys.exit()
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
        ipr = len(eigvect) * np.sum(np.abs(eigvect) ** 4, axis=1) / np.sum(np.abs(eigvect) ** 2, axis=1) ** 2

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
        psides = np.zeros((2. * len(self.lattice.xy), maxNsides + 1), dtype=float)
        for ii in range(len(self.lattice.xy)):
            for jj in range(maxNsides + 1):
                if len(pnod[ii]) > 0:
                    # print 'pnod[ii] = ', pnod[ii]
                    # print 'np.sum(np.array(pnod[', ii, ']) == ', jj, ') =', np.sum(np.array(pnod[ii]) == jj )
                    psides[2 * ii, jj] = float(np.sum(np.array(pnod[ii]) == jj)) / float(len(pnod[ii]))
                    psides[2 * ii + 1, jj] = psides[2 * ii, jj]
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
                prpoly[ii][jj] = np.sum(np.abs(eigvect[jj]) ** 2 * pside)

        # This method below isn't quite right
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
                          save_eigval=False, cutoffd=None, locutoffd=None, hicutoffd=None, attribute_eigv=False):
        """For each eigvector excitation with a positive frequency, fit excitation to an exponential decay centered
        about the excitation's COM.
        where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Returns
        -------
        localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
            fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
        attribute : bool
            attribute localization to self
        save_eigvect_eigval : bool
            Save eigvect and eigval to disk
        save_eigval : bool
            If save_eigvect_eigval is False and save_eigval is True, just saves eigval
        attribute_eigv : bool
            Attribute eigval and eigvect to self
        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current glat...'
                self.save_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False)
            elif save_eigval:
                print 'glat.calc_localization(): saving eigval only...'
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
                # edge_localization = glatfns.fit_edgedecay_periodicstrip(self.lattice.xy, eigval, eigvect,
                #                                                         cutoffd=cutoffd, check=self.lp['check'])
                localization = glatlzfns.fit_eigvect_to_exponential_1dperiodic(self.lattice.xy, eigval, eigvect,
                                                                               self.lattice.lp['LL'],
                                                                               locutoffd=cutoffd, hicutoffd=cutoffd,
                                                                               check=self.lp['check'])
            else:
                localization = glatlzfns.fit_eigvect_to_exponential_periodic(self.lattice.xy, eigval, eigvect,
                                                                             self.lattice.lp['LL'],
                                                                             locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                             check=self.lp['check'])
        else:
            localization = glatlzfns.fit_eigvect_to_exponential(self.lattice.xy, eigval, eigvect, cutoffd=cutoffd,
                                                                check=self.lp['check'])

        if attribute:
            self.localization = localization

        return localization

    def calc_edge_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                               save_eigval=False, locutoffd=None, hicutoffd=None, attribute_eigv=False):
        """

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
        attribute_eigv : bool
            Attribute eigval and eigvect to self
        cutoffd
        locutoffd
        hicutoffd
        attribute_eigv

        Returns
        -------

        """
        if eigval is None or eigvect is None:
            eigval, eigvect = self.get_eigval_eigvect(attribute=False)
            if save_eigvect_eigval:
                print 'Saving eigenvalues and eigenvectors for current glat...'
                self.save_eigval_eigvect(eigval=eigval, eigvect=eigvect, attribute=False)
            elif save_eigval:
                print 'glat.calc_localization(): saving eigval only...'
                self.ensure_eigval(eigval=eigval, attribute=False)

        # obtain boundary to be the particles just inside the immobile sites, if there are any -- ie if not all sites
        # are inner sites, choose the boundary particles to be the ones just inside the outer sites
        # check it
        # plt.plot(self.lattice.xy[:, 0], self.lattice.xy[:, 1], '.')
        # plt.plot(self.lattice.xy[self.inner_indices, 0], self.lattice.xy[self.inner_indices, 1], 'bo')
        # plt.plot(self.xy[self.inner_boundary, 0], self.xy[self.inner_boundary, 1], 'ro')
        # plt.plot(self.xy_inner[self.inner_boundary_inner, 0],
        #          self.xy_inner[self.inner_boundary_inner, 1], 'm^')
        # print 'np.shape(self.inner_boundary_inner = ', np.shape(self.inner_boundary_inner)
        # print 'np.shape(self.inner_boundary = ', np.shape(self.inner_boundary)
        # plt.show()
        # sys.exit()

        if self.lattice.lp['periodicBC']:
            if 'periodic_strip' in self.lattice.lp:
                if self.lattice.lp['periodic_strip']:
                    perstrip = True
                else:
                    perstrip = False
            else:
                perstrip = False

            if perstrip:
                edge_localization = mgloczfns.fit_edgedecay_periodicstrip(self.xy_inner, self.inner_indices,
                                                                          self.inner_boundary_inner,
                                                                          eigvect, eigval=eigval,
                                                                          locutoffd=locutoffd,
                                                                          hicutoffd=hicutoffd, check=self.lp['check'])
            else:
                raise RuntimeError('No reason to fit a fully periodic sample to edge localization.')
        else:
            edge_localization = mgloczfns.fit_eigvect_to_exponential_edge(self.xy_inner,
                                                                          self.inner_boundary_inner,
                                                                          eigvect, eigval=eigval,
                                                                          locutoffd=locutoffd,
                                                                          hicutoffd=hicutoffd, check=self.lp['check'])

        if attribute:
            self.edge_localization = edge_localization

        return edge_localization

    def save_eigval_eigvect(self, eigval=None, eigvect=None, infodir='auto', attribute=True,
                            force_hdf5=False, save_png=False):
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
            eigvalfn = dio.prepdir(self.lp['meshfn']) + 'eigvals_magnetic_gyro' + basis_str + '.hdf5'
            h5io.save_dset_hdf5(eigval, eigval_name, eigvalfn)

            eigvectfn = dio.prepdir(self.lp['meshfn']) + 'eigvects_magnetic_gyro' + basis_str + '.hdf5'
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
            plt.savefig(infodir + 'eigval_magnetic_gyro_hist' + self.lp['meshfn_exten'] + basis_str + '.png')
            plt.clf()
        print 'Saved maggyro DOS to ' + eigvalfn + '\n and ' + eigvectfn

    def save_eigval(self, eigval=None, infodir='auto', attribute=True, force_hdf5=True, save_png=False):
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
        # print 'mlat.save_eigval(): eigval = ', eigval
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        if eigval is None:
            eigval = self.get_eigval(attribute=attribute)

        # Naming
        basis_str = glatfns.get_basis_str(self)
        eigval_name = 'eigval' + self.lp['meshfn_exten'] + basis_str
        if force_hdf5:
            eigvalfn = dio.prepdir(self.lp['meshfn']) + 'eigvals_magnetic_gyro' + basis_str + '.hdf5'
            h5io.save_dset_hdf5(eigval, eigval_name, eigvalfn)
        else:
            eigvalfn = infodir + eigval_name + '.pkl'
            output = open(eigvalfn, 'wb')
            pickle.dump(eigval, output)
            output.close()

        if save_png:
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro')
            plt.savefig(infodir + 'eigval_magnetic_gyro_hist' + self.lp['meshfn_exten'] + '.png')
            plt.clf()
        print 'Saved gyro DOS to ' + eigvalfn

    def save_Omg(self, infodir='auto', histogram=True, attribute=True, force_hdf5=False, overwrite=False):
        """Save Omk pinning frequencies for this MagneticGyroLattice

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
        if infodir == 'auto' or infodir is None:
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
                    elif overwrite:
                        data = fi[pinning_name]       # load the data
                        data[...] = self.Omg          # assign new values to data
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
        fn = infodir + 'ipr' + self.lp['meshfn_exten'] +'.pkl'
        print 'Saving ipr as ' + fn
        pickle.dump(ipr, file(fn, 'wb'))

        if attribute:
            self.ipr = ipr

        # print 'magnetic_gyro_lattice_class: self.lp['meshfn_exten'] = ', self.lp['meshfn_exten']
        if save_images:
            # save IPR as png
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for gyroscopic network',
                              fname='ipr_mgyro_hist' + self.lp['meshfn_exten'],
                              alpha=1.0, FSFS=12, inverse_PR=True, show=show, **kwargs)
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for gyroscopic network',
                              fname='pr_mgyro_hist' + self.lp['meshfn_exten'],
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
                          save_eigvect_eigval=False, save_im=False, force_hdf5=False, overwrite=False):
        """Get and save localization measure for all eigenvectors of the GyroLattice"""
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        # locz = self.get_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        # fn = infodir + 'localization' + self.lp['meshfn_exten'] + '.txt'
        # print 'Saving localization as ' + fn
        # header = "Localization of eigvects: fitted to A*exp(K*sqrt((x-xc)**2 + (y-yc)**2)): " +\
        #          "xc, yc, A, K, uncA, covAK, uncK. The modes examined range from int(len(eigval)*0.5) to len(eigval)."
        # np.savetxt(fn, locz, delimiter=',', header=header)

        locz = self.get_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval, save=False)
        locz_name = 'localization' + self.lp['meshfn_exten']
        if force_hdf5:
            fn = infodir + 'localization_magneticgyro.hdf5'
            h5io.save_dset_hdf5(locz, locz_name, fn, overwrite=overwrite)
        else:
            fn = infodir + locz_name + '.txt'
            print 'Saving localization as ' + fn
            header = "Localization of eigvects: fitted to A*exp(K*sqrt((x-xc)**2 + (y-yc)**2)): " + \
                     "xc, yc, A, K, uncA, covAK, uncK. The modes examined range from int(len(eigval)*0.5) " \
                     "to len(eigval)."
            np.savetxt(fn, locz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if eigval is None:
            eigval = self.get_eigval()

        if save_im:
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
                               save_eigvect_eigval=False, save_im=False, force_hdf5=False, overwrite=False):
        """Get and save localization measure for all eigenvectors of the GyroLattice"""
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        locz = self.get_edge_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        locz_name = 'localization_edge' + self.lp['meshfn_exten']
        if force_hdf5:
            hdf5fn = infodir + 'localization_edge_magneticgyro.hdf5'
            h5io.save_dset_hdf5(locz, locz_name, hdf5fn, overwrite=overwrite)
            fn = hdf5fn
        else:
            fn = infodir + locz_name + '.txt'
            print 'Saving edge localization as ' + fn
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
            ax.fill_between(evals, -locz[:, 1] - np.sqrt(locz[:, 4]), -locz[:, 1] + np.sqrt(locz[:, 4]),
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
            loclz_infodir = infodir + 'localization' + self.lp['meshfn_exten'] + '/'
            dio.ensure_dir(loclz_infodir)
            self.plot_edge_localization(localization=locz, outdir=loclz_infodir, fontsize=12)

            # Make movie
            imgname = loclz_infodir + 'localization' + self.lp['meshfn_exten'] + '_'
            movname = infodir + 'localization' + self.lp['meshfn_exten']
            lemov.make_movie(imgname, movname, indexsz='06', framerate=4, imgdir=loclz_infodir, rm_images=True,
                             save_into_subdir=True)

        print 'Saved gyro localization to ' + fn
        return locz

    def save_DOSmovie(self, infodir='auto', attribute=True, save_DOS_if_missing=True):
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
                    matrix = self.calc_matrix(attribute=attribute)
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

        if not glob.glob(infodir + 'eigval_magnetic_gyro_hist' + exten + '.png'):
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro')
            plt.savefig(infodir + 'eigval_magnetic_gyro_hist' + exten + '.png')
            plt.clf()

        if self.lp['periodicBC']:
            colorstr = 'ill'
        else:
            colorstr = 'pr'

        magmov.save_normal_modes_maggyro(self, datadir=infodir, sim_type='gyro',
                                         rm_images=True, gapims_only=False, save_into_subdir=False, overwrite=True,
                                         color=colorstr)

    def infinite_dispersion(self, kx=None, ky=None, nkxvals=50, nkyvals=20,
                            save=True, title='Dispersion relation', save_plot=True,
                            save_dos_compare=False, outdir=None):
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
        title : str
            the title of the plot of the dispersion
        save_dos_compare : bool
            Compare the projection of the dispersion onto the omega axis with the DOS of the MagneticGyroLattice
        outdir : str or None
            path to the dir where results are saved, if save==True. If None, uses lp['meshfn'] for hlat.lattice

        Returns
        -------
        omegas, kx, ky
        """
        # Note: the function called below has not been finished
        omegas, kx, ky = mglatkspacefns.infinite_dispersion(self, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                                            save=save, save_plot=save_plot, title=title, outdir=outdir)

        if save_dos_compare:
            mglatkspacefns.compare_dispersion_to_dos(omegas, kx, ky, self, outdir=outdir)

        return omegas, kx, ky

    def plot_ipr_DOS(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='ipr_magneticgyro_hist',
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
            print '1/ipr = ', 1. / ipr
            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'gyro', alpha=alpha,
                                                                           colorV=1. / ipr, colormap='viridis_r',
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
        # Note that plot_ill_dos() works just fine for magnetic gyro lattice
        dos_ax, cbar_ax = glatpfns.plot_ill_dos(self, dos_ax=dos_ax, cbar_ax=cbar_ax, alpha=alpha, vmin=vmin,
                                                vmax=vmax, climbars=True, **kwargs)
        if save:
            fn = dio.prepdir(self.lp['meshfn']) + 'ill_dos' + self.lp['meshfn_exten'] + '.png'
            print 'saving ill to: ' + fn
            plt.savefig(fn, dpi=600)
        if show:
            plt.show()
        return dos_ax, cbar_ax

    def plot_DOS(self, outdir=None, title=r'$D(\omega)$ for gyroscopic network', fname='eigval_magnetic_gyro_hist',
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
        fig, DOS_ax, ax = glatpfns.draw_localization_plots(self, localization, eigval, eigvect, outdir=outdir,
                                                           alpha=alpha, fontsize=fontsize)

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
            print 'np.shape(np.abs(eigvect[in_sum, :])**2 = ', np.shape(np.abs(eigvect[in_sum, :]) ** 2)
            tmp = np.sum(np.abs(eigvect[in_sum, :]) ** 2, axis=0)
            print 'np.shape(tmp) = ', np.shape(tmp)
            print 'tmp[0:10] = ', tmp[0:10]
            total_amp = np.array([np.sqrt(tmp[2 * ii] + tmp[2 * ii + 1]) for ii in range(int(len(tmp) * 0.5))])
            # plot result
            fig, ax = leplt.initialize_1panel_centered_fig(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
            [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
            print 'max = ', np.max(total_amp)
            ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp / np.max(total_amp),
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
                nind = np.argmin(np.mod(thetas - np.pi * 0.4999, np.pi))
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
                tmp = np.sum(np.abs(eigvect[in_sum, :]) ** 2, axis=0)
                print 'np.shape(tmp) = ', np.shape(tmp)
                print 'tmp[0:10] = ', tmp[0:10]
                total_amp = np.array([np.sqrt(tmp[2 * ii] + tmp[2 * ii + 1]) for ii in range(int(len(tmp) * 0.5))])
                # plot result
                fig, ax = leplt.initialize_1panel_centered_fig(Wfig, Wfig * 0.75, wsfrac=0.55, vspace=0, tspace=10)
                [ax, axcb] = self.lattice.plot_BW_lat(ax=ax, save=False, title='')
                print 'max = ', np.max(total_amp)
                ax.scatter(self.lattice.xy[:, 0], self.lattice.xy[:, 1], s=100 * total_amp / np.max(total_amp),
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
        ill_full[0:int(len(eigval) * 0.5)] = ill[::-1]
        ill_full[int(len(eigval) * 0.5):len(eigval)] = ill

        if outdir is None:
            outdir = dio.prepdir(self.lp['meshfn'])

        if frange is not None:
            addstr = '_frange{0:0.3f}'.format(frange[0]).replace('.', 'p')
            addstr += '_{0:0.3f}'.format(frange[1]).replace('.', 'p')
            freqs = np.imag(eigval[int(len(eigval) * 0.5):len(eigval)])
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
            lfns.plot_polygons_with_nsides(lat, nsides, ax, color=cmap(float(nsides) / maxpno), alpha=0.5)

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
                                                  ii + len(self.lattice.xy), marker_num=0,
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


if __name__ == "__main__":
    '''Use the GyroLattice class to create a density of states, compute participation ratio binned by polygons,
    or build a lattice with the DOS

    Example usage:
    python magnetic_gyro_lattice_class.py -save_lattice -LT hexagonal -shape hexagon -periodic -N 10 -Omg -1.0
    python magnetic_gyro_lattice_class.py -load_and_resave -LT hexagonal -shape hexagon -periodic -N 10 -Omg -10.0
    python magnetic_gyro_lattice_class.py -save_ipr -LT hexagonal -shape hexagon -N 20
    python magnetic_gyro_lattice_class.py -gap_scaling -LT hexagonal -shape hexagon -periodic -N 10
    python run_series.py -pro magnetic_gyro_lattice_class -opts periodic/-LT/hexagonal/-shape/square/-NV/5/-NH/40/-localization -var Vpin 0.1:1.0

    # AB site DOS's
    python magnetic_gyro_lattice_class.py -save_ipr -LT hexagonal -shape hexagon -periodic -N 4 -AB 0.2
    '''
    import lepm.lattice_class as lattice_class
    import magnetic_gyro_lattice_scripts as mglatscripts

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
    parser.add_argument('-disp_abbounds', '--dispersion_abtransition_gapbounds',
                        help='Draw infinite/semi-infinite dispersion relation through AB transition and plot bounds',
                        action='store_true')
    parser.add_argument('-gap_sweep', '--gap_sweep',
                        help='Sweep through parameters of Omg, Omk, and aoverl to look for good gap',
                        action='store_true')
    parser.add_argument('-glatparam_sweep', '--glatparam_sweep',
                        help='Sweep through a gyrolattice parameter and plot spectrum vs param', action='store_true')
    parser.add_argument('-glatparam', '--glatparam',
                        help='The gyrolattice parameter to vary', type=str, default='ABDelta')
    parser.add_argument('-glatvals', '--glatvals',
                        help='The gyrolattice parameter values to use for varying', type=str, default='0:0.1:1.5')
    parser.add_argument('-save_prpoly', '--save_prpoly',
                        help='Create dict and hist of excitation participation, grouped by polygonal contributions',
                        action='store_true')
    parser.add_argument('-dcdisorder', '--dcdisorder', help='Construct DOS with delta correlated disorder and view ipr',
                        action='store_true')
    parser.add_argument('-save_ipr', '--save_ipr', help='Load MagneticGyroLattice and save ipr',
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
    parser.add_argument('-OmKspec', '--OmKspec', help='string specifier for OmK bond frequency matrix',
                        type=str, default='')
    # examples of OmKspec:
    #   'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
    #      'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') +\
    #      'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n') --> dicing lattice
    #   'Vpin0p10' --> a specific, single random configuration of potentials
    #   'Vpin0p10theta0p00phi0p00' --> a specific, single random configuration of potentials with twisted BCs
    parser.add_argument('-edgelocalization', '--edgelocalization',
                        help='Check localization properties to the boundary of the sample', action='store_true')
    parser.add_argument('-savepintxt', '--save_pinning_to_txt',
                        help='when creating a new array of pinning frequencies, save to hdf5 instead of txt',
                        action='store_true')

    # Geometry and physics arguments
    parser.add_argument('-intrange', '--intrange',
                        help='Consider magnetic couplings only to nth nearest neighbors (if ==2, then NNNs, for ex)',
                        type=int, default=1)
    parser.add_argument('-aol', '--aoverl',
                        help='interparticle distance divided by length of pendulum from pivot to center of mass',
                        type=float, default=0.60)
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vpinf', '--V0_pin_flat',
                        help='Half width of flat distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vsprf', '--V0_spring_flat',
                        help='Half width of flat distribution of delta-correlated bond disorder',
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
    parser.add_argument('-spreading_time', '--spreading_time',
                        help='Amount of time for spreading to take place in uniformly random pt sets ' +
                             '(with 1/r potential)',
                        type=float, default=0.0)

    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-periodic_strip', '--periodic_strip',
                        help='Enforce strip periodic boundary condition in horizontal dim', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.667')
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-pinconf', '--pinconf',
                        help='Lattice disorder realization number', type=int, default=01)
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
    dcdisorder1 = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0
    dcdisorder2 = args.V0_pin_flat > 0 or args.V0_spring_flat > 0
    dcdisorder = dcdisorder1 or dcdisorder2

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
          'basis': args.basis,
          'Omk': float((args.Omk).replace('n', '-').replace('p', '.')),
          'Omg': float((args.Omg).replace('n', '-').replace('p', '.')),
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'V0_pin_flat': args.V0_pin_flat,
          'V0_spring_flat': args.V0_spring_flat,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta': args.ABDelta,
          'thres': args.thres,
          'pinconf': args.pinconf,
          'OmKspec': args.OmKspec,
          'spreading_time': args.spreading_time,
          'intparam': args.intparam,
          'aoverl': args.aoverl,
          'interaction_range': args.intrange,
          'save_pinning_to_hdf5': not args.save_pinning_to_txt,
          }

    if lp['periodicBC']:
        lp['bcs'] = 'periodic'
    else:
        lp['bcs'] = 'free'

    if args.make_DOSmovie:
        '''Example usage:
        python magnetic_gyro_lattice_class.py -N 5 -LT kagome -shape square -periodic -DOSmovie -Vpin 0.1
        python magnetic_gyro_lattice_class.py -N 4 -LT hexagonal -shape hexagon -DOSmovie -Vpin 0.1
        python run_series.py -pro magnetic_gyro_lattice_class -opts N/4/-LT/hexagonal/-shape/hexagon/-DOSmovie/-Vpin/0.1 \
            -var AB 1.1/1.2/1.3/1.4
        0.0/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/1.0
        '''
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MagneticGyroLattice(lat, lp)
        print 'loading lattice...'
        mlat.load()
        print 'Saving DOS movie...'
        mlat.save_DOSmovie()

    if args.dispersion:
        """Example usage:
        python magnetic_gyro_lattice_class.py -N 1 -LT hexagonal -shape square -periodic -dispersion -basis psi -aol 1.0 -Omk n0.67
        python magnetic_gyro_lattice_class.py -N 1 -LT hexagonal -shape square -periodic -dispersion -basis psi -aol 1.0 -Omk n0.67 -intrange 3
        python magnetic_gyro_lattice_class.py -N 1 -LT hexagonal -shape square -delta 0p900 -periodic -dispersion -basis psi -aol 1.0 -Omk n0.67
        python magnetic_gyro_lattice_class.py -N 1 -LT hexagonal -shape square -periodic -dispersion -basis psi -aol 0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -dispersion -aol 0.6

        # make the strip
        python ./build/make_lattice.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -skip_polygons -skip_gyroDOS
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        glat = MagneticGyroLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        glat.eig_vals_vects(attribute=True)
        omegas, kx, ky = glat.infinite_dispersion(save=False, nkxvals=100, nkyvals=25, save_dos_compare=False)


        # make nice plot of dispersion

        band1color = '#70a6ff' # blue
        band2color = '#ff7777'  # red
        # band1color = '#DF813B'  # orange
        # band2color = '#31A2C4'  # blue
        plt.style.use('dark_background')
        fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.5, tspace=0)
        for jj in range(len(ky)):
            for kk in range(len(omegas[0, jj, :])):
                if (omegas[:, jj, kk] > 1.0).any():
                    ax.plot(kx, omegas[:, jj, kk], '-', color=band1color,
                            lw=max(1, 5. / (len(kx) * len(ky))))
                else:
                    ax.plot(kx, omegas[:, jj, kk], '-', color=band2color,
                            lw=max(1, 5. / (len(kx) * len(ky))))
        ax.set_ylim(0, np.pi)
        ax.xaxis.set_ticks([-np.pi, 0, np.pi])
        ax.xaxis.set_ticklabels([r'$-\pi$', 0, r'$\pi$'])
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$\omega$')
        print 'magnetic_gyro_kspace_functions: saving image to '
        fn = dio.prepdir(lp['meshfn']) + 'dispersion_still_2d' + lp['meshfn_exten'] + '.png'
        print 'fn = ', fn
        plt.savefig(fn, dpi=300)
        plt.close('all')

        sys.exit()
        # Plot in 3D
        fig = plt.gcf()
        ax = fig.add_subplot(projection='3d')  # 111,
        # rows will be kx, cols wll be ky
        kyv = np.array([[ky[i].tolist()] * len(kx) for i in range(len(ky))]).ravel()
        kxv = np.array([[kx.tolist()] * len(ky)]).ravel()
        # print 'kyv = ', np.shape(kyv)
        # print 'kxv = ', np.shape(kxv)
        # for kk in range(len(omegas[0, 0, :])):
        #     ax.trisurf(kxv, kyv, omegas[:, :, kk].ravel())

        fontsize = 14
        tick_fontsize = fontsize
        # fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=180, wsfrac=0.7, fontsize=fontsize)
        # ax = fig.gca(projection='3d')

        # fig = plt.gcf()
        # ax = fig.add_subplot(projection='3d')

        from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # from matplotlib import cm
        # from matplotlib.ticker import LinearLocator, FormatStrFormatter

        # fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=180, wsfrac=0.7, fontsize=fontsize)
        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        # # Reshape colors, energies
        # b1color = cmap((berry1.reshape(np.shape(km[0])) / (2. * np.pi)) / (vmax - vmin) + 0.5)
        # b2color = cmap((berry2.reshape(np.shape(km[0])) / (2. * np.pi)) / (vmax - vmin) + 0.5)
        # energy1 = energy1.reshape(np.shape(km[0]))
        # energy2 = energy2.reshape(np.shape(km[0]))
        # print 'b1color = ', b1color

        # Plot surfaces
        print 'ax = ', ax
        print 'shape(kxv) = ', np.shape(kxv)
        kxv = kxv.reshape(np.shape(omegas[:, :, 0]))
        kyv = kyv.reshape(np.shape(omegas[:, :, 0]))
        print 'shape(omegas) = ', np.shape(omegas)
        surf = ax.plot_surface(kxv, kyv, omegas[:, :, 0], rstride=1, cstride=1,
                               # facecolors=band1color,
                               # vmin=vmin, vmax=vmax, cmap=cmap,
                               linewidth=1, antialiased=False)
        surf = ax.plot_surface(kxv, kyv, omegas[:, :, 1], rstride=1, cstride=1,
                               # facecolors=band2color,
                               # vmin=vmin, vmax=vmax, cmap=cmap,
                               linewidth=1, antialiased=False)

        ax.set_ylabel(r'$k_y$', labelpad=15, fontsize=fontsize)
        ax.yaxis.set_ticks([-np.pi, 0, np.pi])
        ax.yaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_fontsize)
        ax.set_zlabel(r'$\omega$', labelpad=15, fontsize=fontsize, rotation=90)
        # ax.zaxis.set_ticks([-np.pi, 0, np.pi])
        # ax.zaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(tick_fontsize)
        ax.axis('scaled')
        ax.set_zlim(-np.pi, np.pi)

        ax.set_xlabel(r'$k_x$', labelpad=15, fontsize=fontsize)
        ax.xaxis.set_ticks([-np.pi, 0, np.pi])
        ax.xaxis.set_ticklabels([r'-$\pi$', 0, r'$\pi$'], fontsize=tick_fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_fontsize)
        ax.axis('scaled')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)

        # ax.view_init(elev=0, azim=0.)
        plt.savefig('/Users/npmitchell/Desktop/band_structure_3d.png')


    if args.dispersion_abtransition:
        """Example usage:
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 1.0 -Omk -0.67 -delta 0p800
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 4 -disp_abtrans -basis psi -aol 1.2 -Omk -0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 1.2 -Omk -0.5
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.85 -Omk -0.75
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.8 -Omk -0.344
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.6 -Omk 0.128
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -disp_abtrans -basis psi
        """
        mglatscripts.dispersion_abtransition(lp, fullspectrum=False)


    if args.dispersion_abtransition_gapbounds:
        """Example usage:
        # Note: shortrange makes no difference for N=1 samples (single unit cell)
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abbounds -basis psi -aol 0.795 -Omk -0.344
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abbounds -basis psi -aol 0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -disp_abbounds -basis psi

        python run_series.py -pro magnetic_gyro_lattice_class -opts LT/hexagonal/-shape/square/-periodic/-N/1/-disp_abbounds/-basis/psi -var aol 0.25:0.05:0.9
        """
        gapbounds = mglatscripts.dispersion_abtransition_gapbounds(lp)


    if args.gap_sweep:
        '''Example usage:
        python magnetic_gyro_lattice_class.py -N 5 -shape square -periodic -gap_sweep -Vpin 0.1 -LT kagome
        python magnetic_gyro_lattice_class.py -N 4 -shape hexagon -gap_sweep -Vpin 0.1 -LT hexagonal -force_hdf5
        '''
        from matplotlib.collections import LineCollection
        import lepm.plotting.colormaps as lecmaps
        # keep Omg = 1.0
        # Omk = k_m l^2 / I omega -> 0.86 in original experiments
        # aoverl = 0.0305/.038
        # lp['periodicBC'] = True
        aolv = np.arange(0.6, 1.0, 0.1)
        omkv = -np.arange(0.4, 1.8, 0.2)

        res = {}
        vmax = 0.
        for aol in aolv:
            res[sf.float2pstr(aol)] = {}
            for omk in omkv:
                lpnew = copy.deepcopy(lp)
                lpnew['aoverl'] = aol
                lpnew['Omk'] = omk

                lat = lattice_class.Lattice(lpnew)
                lat.load()
                if vmax == 0.:
                    if lp['periodicBC']:
                        vmax = 1. / (np.max(np.abs(lat.xy.ravel())))
                    else:
                        vmax = 3.
                    lsize = 2. * np.max(np.abs(lat.xy.ravel()))

                mlat = MagneticGyroLattice(lat, lpnew)
                # eigval = mlat.ensure_eigval()
                eigval = mlat.get_eigval()

                eigval_upper = np.abs(np.imag(eigval[0:int(len(eigval) * 0.5)][::-1]))

                if lp['periodicBC']:
                    locz = mlat.get_localization(attribute=True)

                    # First time around
                    loc_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'localization' +
                                           mlat.lp['meshfn_exten'] + '.txt')
                    if not loc_exists:
                        print '\nLocalization file is not already saved. Creating it...'
                        plt.close('all')
                        mlat.save_localization(attribute=True, save_images=False, save_eigvect_eigval=False)
                        mlat.plot_ill_dos()
                        plt.close('all')
                    ill_upper = locz[:, 2]
                    res[sf.float2pstr(aol)][sf.float2pstr(omk)] = [eigval_upper, ill_upper]
                else:
                    ipr = mlat.get_ipr(attribute=True)
                    # First time around
                    ipr_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'ipr' +
                                           mlat.lp['meshfn_exten'] + '.txt')
                    if not ipr_exists:
                        print '\nLocalization file is not already saved. Creating it...'
                        plt.close('all')
                        mlat.save_ipr(attribute=True, save_images=False)
                        plt.close('all')
                    # print 'ipr = ', ipr
                    # print 'np.shape(ipr) = ', np.shape(ipr)
                    # print 'np.shape(eigval_upper) = ', np.shape(eigval_upper)
                    ipr_upper = ipr[int(len(ipr)*0.5):]
                    ill_upper = ipr_upper

                res[sf.float2pstr(aol)][sf.float2pstr(omk)] = [eigval_upper, ill_upper]
                # vmax = max(vmax, np.max(ill_upper))

        # Plot eigvals
        aol0 = sf.float2pstr(aolv[0])
        fig, ax, cax = leplt.initialize_nxmpanel_cbar_fig(1, len(aolv), Wfig=180, x0frac=0.06,
                                                          wsfrac=0.7 / float(len(aolv)),
                                                          cbar_placement='right_right', orientation='vertical')
        lecmaps.register_colormaps()
        cmap = plt.get_cmap('viridis')
        if lp['periodicBC']:
            vmin = 0.
        else:
            vmin = 1.0
        ind = 0
        ii = 0
        domk = abs(omkv[1] - omkv[0])
        for aols in res:
            for omks in res[aols]:
                print 'aols = ', aols
                print 'omks = ', omks
                ep0 = zip(np.abs(sf.str2float(omks)) * np.ones(len(res[aols][omks][0])), res[aols][omks][0])
                ep1 = zip((np.abs(sf.str2float(omks)) + domk) * np.ones(len(res[aols][omks][0])), res[aols][omks][0])
                lines = [list(a) for a in zip(ep0, ep1)]

                # print 'res[aols][omks][1] / float(vmax) = ', res[aols][omks][1] / float(vmax)
                print 'np.shape(res[aols][omks][1] / float(vmax)) = ', np.shape(res[aols][omks][1] / float(vmax))
                print 'np.shape(ones) = ', np.shape(np.ones_like(res[aols][omks][1] / float(vmax)))
                colors = cmap(res[aols][omks][1] / float(vmax))
                print 'res[aols][omks][1] = ', res[aols][omks][1]
                print 'vmax = ', vmax
                # print 'colors = ', colors
                lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
                # lc.set_array(colors)
                ax[ii].add_collection(lc)
            ii += 1

        for jj in range(ii):
            ax[jj].set_title(r'$a/l= $' + '{0:0.2f}'.format(aolv[jj]))
            ax[jj].set_xlim(np.min(np.abs(omkv)) - 0.2, np.max(np.abs(omkv)) + 0.1)
            ax[jj].set_ylim(-0.1, 5.5)
            ax[jj].xaxis.set_ticks([0, 1, 2])
            ax[jj].yaxis.set_ticks([0, 2, 4])
            if jj == 0:
                ax[jj].set_ylabel('frequency, $\omega/\Omega$')
            ax[jj].set_xlabel(r'$\Omega_k$')

        sm = leplt.empty_scalar_mappable(vmin, vmax, cmap)
        if lp['periodicBC']:
            cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 1. / lsize, 2. / lsize])
            cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
            cb.set_label(r'$\lambda^{-1}$', labelpad=23, rotation=0, fontsize=8, va='center')
        else:
            cb = plt.colorbar(sm, cax=cax, orientation='vertical')  # , ticks=[0, 1. / lsize, 2. / lsize])
            # cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
            cb.set_label(r'$p^{-1}$', labelpad=23, rotation=0, fontsize=8, va='center')
        plt.suptitle('Mobility gaps for magnetic networks')
        fname = dio.prepdir(lat.lp['meshfn']) + 'magnetic_sweep_minaol' + sf.float2pstr(np.min(aolv)) + \
                '_maxaol' + sf.float2pstr(np.max(aolv)) + \
                '_minomk' + sf.float2pstr(np.min(omkv)) + \
                '_maxomk' + sf.float2pstr(np.max(omkv)) + \
                '_nconfs{0:03d}'.format(ind)
        if abs(lp['V0_pin_flat']) > 0:
            fname += '_Vpinflat' + sf.float2pstr(lp['V0_pin_flat'])
        else:
            fname += '_Vpin' + sf.float2pstr(lp['V0_pin_gauss'])

        plt.suptitle(r'Spectra for $N = $' + str(lp['NH']) + ' ' + lp['LatticeTop'])
        print 'saving figure: ' + fname + '.png'
        plt.savefig(fname + '.png', dpi=300)
        plt.close('all')

        print 'done'

    if args.glatparam_sweep:
        '''Example usage:
        python magnetic_gyro_lattice_class.py -N 4 -shape hexagon -glatparam_sweep -Vpin 0.1 -LT hexagonal \
            -glatparam ABDelta -glatvals 0:0.1:1.5
        '''
        from matplotlib.collections import LineCollection
        import lepm.plotting.colormaps as lecmaps

        # keep Omg = 1.0
        # Omk = k_m l^2 / I omega -> 0.86 in original experiments
        # aoverl = 0.0305/.038
        # lp['periodicBC'] = True

        res = {}
        vmax = 0.
        values = sf.string_sequence_to_numpy_array(args.glatvals, dtype=float)
        print 'values = ', values
        for val in values:
            lpnew = copy.deepcopy(lp)
            lpnew[args.glatparam] = val

            lat = lattice_class.Lattice(lpnew)
            lat.load()
            if vmax == 0.:
                if lp['periodicBC']:
                    vmax = 1. / (np.max(np.abs(lat.xy.ravel())))
                else:
                    vmax = 3.
                lsize = 2. * np.max(np.abs(lat.xy.ravel()))

            mlat = MagneticGyroLattice(lat, lpnew)
            eigval = mlat.get_eigval()

            eigval_upper = np.abs(np.imag(eigval[0:int(len(eigval) * 0.5)][::-1]))

            if lp['periodicBC']:
                locz = mlat.get_localization(attribute=True)
                # First time around, save ill
                # loc_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'localization' +
                #                        mlat.lp['meshfn_exten'] + '.txt')
                # if not loc_exists:
                #     print '\nLocalization file is not already saved. Creating it...'
                #     plt.close('all')
                #     mlat.save_localization(attribute=True, save_images=False, save_eigvect_eigval=False)
                #     mlat.plot_ill_dos()
                #     plt.close('all')
                ill_upper = locz[:, 2]
            else:
                ipr = mlat.get_ipr(attribute=True)
                # First time around, save ipr
                # ipr_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'ipr' +
                #                        mlat.lp['meshfn_exten'] + '.txt')
                # if not ipr_exists:
                #     print '\nLocalization file is not already saved. Creating it...'
                #     plt.close('all')
                #     mlat.save_ipr(attribute=True, save_images=False)
                #     plt.close('all')
                # # print 'ipr = ', ipr
                # # print 'np.shape(ipr) = ', np.shape(ipr)
                # # print 'np.shape(eigval_upper) = ', np.shape(eigval_upper)
                ipr_upper = ipr[int(len(ipr) * 0.5):]
                ill_upper = ipr_upper

            res[sf.float2pstr(val)] = [eigval_upper, ill_upper]
            # vmax = max(vmax, np.max(ill_upper))

        # Plot eigvals
        fig, ax, cax = leplt.initialize_1panel_cbar_fig(x0frac=0.1, wsfrac=0.4)  # , orientation='vertical')
        lecmaps.register_colormaps()
        cmap = plt.get_cmap('viridis')
        if lp['periodicBC']:
            vmin = 0.
        else:
            vmin = 1.0
        ind = 0
        ii = 0
        # Get the spacing between x values
        dval = abs(values[1] - values[0])
        # Prepare to tally to find the largest frequency on the plot
        maxfreq = 0.
        for val in res:
            ep0 = zip(np.abs(sf.str2float(val)) * np.ones(len(res[val][0])), res[val][0])
            ep1 = zip((np.abs(sf.str2float(val)) + dval) * np.ones(len(res[val][0])), res[val][0])
            lines = [list(a) for a in zip(ep0, ep1)]

            maxfreq = max(maxfreq, np.max(ep1))
            # print 'res[aols][omks][1] / float(vmax) = ', res[aols][omks][1] / float(vmax)
            colors = cmap(res[val][1] / float(vmax))
            print 'vmax = ', vmax
            # print 'colors = ', colors
            lc = LineCollection(lines, colors=colors, linewidths=0.5, cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
            # lc.set_array(colors)
            ax.add_collection(lc)

        # ax[jj].set_title(r'$a/l= $' + '{0:0.2f}'.format(aolv[jj]))
        ax.set_xlim(np.min(np.abs(values)) - 0.05, np.max(np.abs(values)) + 0.15)
        ax.set_ylim(-0.1, maxfreq + 0.1)
        # ax.xaxis.set_ticks([0, 1, 2])
        # ax.yaxis.set_ticks([0, 2, 4])
        ax.set_ylabel('frequency, $\omega/\Omega$')
        ax.set_xlabel(leplt.param2description(args.glatparam))

        sm = leplt.empty_scalar_mappable(vmin, vmax, cmap)
        if lp['periodicBC']:
            cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 1. / lsize, 2. / lsize])
            cax.yaxis.set_ticklabels([0, r'$1/L$', r'$2/L$'])
            cb.set_label(r'$\lambda^{-1}$', labelpad=23, rotation=0, fontsize=8, va='center')
        else:
            cb = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[1, 2, 3])
            cb.set_label(r'$p^{-1}$', labelpad=15, rotation=0, fontsize=8, va='center')
        fname = dio.prepdir(lat.lp['meshfn']) + 'magnetic_glatparamsweep_' + args.glatparam + \
                '_min' + sf.float2pstr(np.min(values)) + '_max' + sf.float2pstr(np.max(values)) + \
                '_Vpin' + sf.float2pstr(lp['V0_pin_gauss'])
        plt.suptitle(r'Spectra for magnetic $N = $' + str(lp['NH']) + ' ' + lp['LatticeTop'] + '\n'
                     r' ($a/l=$' + '{0:0.2f}'.format(lp['aoverl']) +
                     r', $\Omega_k/\Omega_g$ = ' + '{0:0.2f}'.format(lp['Omk'] / lp['Omg']) + ')')
        print 'saving figure: ' + fname + '.png'
        plt.savefig(fname + '.png', dpi=300)
        plt.close('all')

        print 'done'

    if args.save_ipr:
        '''Example usage
        python run_series.py -pro magnetic_gyro_lattice_class -opts LT/hexagonal/-shape/hexagon/-periodic/-N/4/-save_ipr -var AB 0.1:0.2:1.0
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape hexagon -periodic -N 4 -save_ipr -AB 0.1
        '''
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = MagneticGyroLattice(lat, lp)
        print 'saving ipr...'
        glat.save_ipr(vmax=8.0, xlim=(0., 10.), cbar_labelpad=15)
        # This would save an image of the ipr-colored DOS, but this is already done in save_ipr if save_images=True
        # glat.plot_ipr_DOS(save=True, vmax=5)

    if args.localization:
        """Load a periodic lattice from file, provide physics, and seek exponential localization of modes

        Example usage:
        python run_series.py -pro magnetic_gyro_lattice_class -opts LT/hucentroid/-periodic/-NP/20/-localization/-save_eig -var AB 0.0:0.05:1.0
        python run_series.py -pro magnetic_gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-localization/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0
        python run_series.py -pro magnetic_gyro_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-DOSmovie/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0

        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        extent = 2 * max(np.max(lat.xy[:, 0]), np.max(lat.xy[:, 1]))
        glat = MagneticGyroLattice(lat, lp)
        glat.save_localization(attribute=True, save_images=args.save_images, save_eigvect_eigval=args.save_eig)
        glat.plot_ill_dos(vmax=4./extent, xlim=(0., 14.), ticks=[0, 2./extent, 4./extent],
                          cbar_ticklabels=[0, r'$2/L$', r'$4/L$'], cbar_labelpad=15)

    if args.edgelocalization:
        """Get the edge mode localization length for states in the middle of a finite-size sample's spectrum"""
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        extent = 2 * max(np.max(lat.xy[:, 0]), np.max(lat.xy[:, 1]))
        glat = MagneticGyroLattice(lat, lp)
        locz = glat.edge_localization()
