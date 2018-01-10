import numpy as np
import lepm.twisty.twisty_functions as tfns
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.stringformat as sf
import lepm.twisty.twisty_simulation_fns as tsimfns
import lepm.twisty.twisty_functions_localization as tlocfns
import lepm.plotting.plotting as leplt
import lepm.twisty.plotting.twisty_movies as tmov
import lepm.plotting.colormaps as lecmap
import scipy.optimize
import matplotlib.pyplot as plt
import glob
import os
import cPickle as pkl
import lepm.hdf5io as h5io
import lepm.twisty.twisty_kspace_functions as tlatkspacefns
import lepm.brillouin_zone_functions as bzf

'''
Description
===========
Generate a TwistyLattice class instance. This takes a lattice (points and connections) as input, as well as
a dictionary of parameters to define the physics on that lattice.
'''


class TwistyLattice:
    """Create a network made of twisty springs from an instance of the lattice class.
    Attributes of the TwistyLattice are:
        lattice : lattice class instance (has xy, NL, KL, etc). Can be empty so that it can be loaded or built
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
        V0_pin_gauss, V0_gauss, dcdisorder

    V0_pin_gauss: float
        stdev of distribution of delta-correlated pin disorder
    V0_pin_gauss: float
        stdev of distribution of delta-correlated spring disorder
    dcdisorder: bool
        Whether there is delta-function correlated disorder in the physics of the network (pin, spring)

    """
    def __init__(self, lattice, lp, xytup=None, xytup_meshfnexten=None,
                 KK=None, GG=None, CC=None, matrix=None,
                 NL=None, KL=None, BL=None, PVx=None, PVy=None,
                 NL_t=None, KL_t=None, BL_t=None, PVx_t=None, PVy_t=None,
                 eigval=None, eigvect=None, bL=None,
                 eps=1e-9):
        """Create a TwistyLattice instance. This has an inherent contraint in every bond: extension is coupled to twist.
        If the bond is extended by a distance given by self.pitch, it twists by 2pi radians.
        Normals are tracked at every vertex.

        Properties
        ----------
        lattice : lattice class instance (has xy, NL, KL, etc)
        xytup : tuple of (M x 2 float array, N x 2 float array)
            Equilibrium positions of inner gyros, all gyros --> so (inner, all)
        kk : float or #bonds x 1 float array
            spring contant for extension
        bb : float or #bonds x 1 float array
            bending constant for each spring
        pitch : float or #bonds x 1 float array
            the distance a spring must stretch to complete a full revolution
        """
        # xytup is a tuple of inner points, outer points. Define self.xy to be all sites, including immobile ones
        self.lattice = lattice
        self.lp = lp
        self.bL = bL

        # Determine inner_indices and outer_indices
        # By default, look for roi that selects inner particles. If no roi in lp,
        # make all particles mobile (so inner_indices is all of the indices
        if xytup is None:
            # Create self.xy and self.xy_inner
            self.xy = lattice.xy
            if 'roi' in self.lp:
                self.xy_inner = le.pts_in_polygon(self.xy, self.lp['roi'])
            elif 'immobile_boundary' in self.lp:
                # get boundary of the lattice and denote these particles as the outer particles
                # boundary = le.extract_boundary(self.xy, self.lattice.NL, self.lattice.KL, self.lattice.BL,
                #                                check=False)
                boundary = self.lattice.get_boundary()
                if boundary is None:
                    self.xy_inner = self.xy
                elif isinstance(boundary, tuple):
                    # There are multiple boundaries. Exclude all or one of them from inner_indices
                    inners = np.arange(len(self.xy))
                    outers = []
                    if 'immobile_boundary0' in self.lp or 'immobile_boundary1' in self.lp:
                        # If one of the keys is missing, ensure that it is false
                        if 'immobile_boundary0' not in self.lp:
                            self.lp['immobile_boundary0'] = False
                        if 'immobile_boundary1' not in self.lp:
                            self.lp['immobile_boundary1'] = False

                        # Decide on inner and outer indices
                        if self.lp['immobile_bounary0']:
                            # Exclude only boundary0 from inner_indices
                            boundary = boundary[0]
                            inners = np.setdiff1d(inners, np.array(bndy))
                            outers.append(boundary)
                            self.lp['immobile_boundary1'] = False
                            xytup_meshfnexten = '_immobilebnd0'
                        elif self.lp['immobile_bounary1']:
                            # Exclude only boundary0 from inner_indices
                            boundary = boundary[1]
                            inners = np.setdiff1d(inners, np.array(bndy))
                            outers.append(boundary)
                            self.lp['immobile_boundary0'] = False
                            xytup_meshfnexten = '_immobilebnd1'
                        elif self.lp['immobile_boundary']:
                            # Exclude all boundaries from inner_indices
                            for bndy in boundary:
                                inners = np.setdiff1d(inners, np.array(bndy))
                                outers.append(bndy)
                            xytup_meshfnexten = '_immobilebnd'
                    elif self.lp['immobile_boundary']:
                        # Exclude all boundaries from inner_indices
                        for bndy in boundary:
                            inners = np.setdiff1d(inners, np.array(bndy))
                            outers.append(bndy)
                        xytup_meshfnexten = '_immobilebnd'
                    else:
                        # no immobile boundary
                        inners = np.arange(len(self.xy))
                        outers = np.array([])
                        self.xy_inner = self.xy
                        self.xy_outer = np.array([])
                        # default xytup, so ignore xytup_meshfn_exten argument
                        xytup_meshfnexten = ''

                    self.inner_indices = np.hstack(inners)
                    self.outer_indices = np.hstack(outers)
                    self.xy_inner = self.xy[self.inner_indices]
                else:
                    self.inner_indices = np.setdiff1d(np.arange(len(self.xy)), np.array(boundary))
                    self.outer_indices = np.array(boundary)
                    self.xy_inner = self.xy[self.inner_indices]
                    xytup_meshfnexten = '_immobilebnd'
            else:
                self.xy_inner = self.xy
                self.xy_outer = np.array([])
                # default xytup, so ignore xytup_meshfn_exten argument
                xytup_meshfnexten = ''
        else:
            self.xy = xytup[1]
            self.xy_inner = xytup[0]
            self.xy_outer = np.setdiff1d(xytup[1], xytup[0])
            if xytup_meshfnexten is None:
                raise RuntimeError('Must supply xytup_meshfnexten when supply xytup')

        # Determine NL and KL
        if NL is not None and KL is not None:
            self.NL = NL
            self.KL = KL
            # determine if shortrange or longrange or intermediate interaction_range from supplied NL
            # if lp['intrange'] == 1:
            #     nl_meshfn_exten = '_shortrange'
            if lp['intrange'] == 1:
                nl_meshfn_exten = ''
            elif lp['intrange'] > 1:
                nl_meshfn_exten = '_intrange{0:04d}'.format(lp['interaction_range'])
            elif lp['intrange'] == 0:
                nl_meshfn_exten = '_infiniterange'
            else:
                raise RuntimeError('What is intrange?')

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
            self.NL, self.KL, self.PVx, self.PVy, nl_meshfn_exten = tfns.create_NL_KL_pv(self)

        print 'self.xy_inner = ', np.shape(self.xy_inner)
        print 'self.xy = ', np.shape(self.xy)
        self.outer_indices, self.inner_indices, self.total2inner = self.outer_inner(self.xy, self.xy_inner)

        # Add lattice properties to gyro_lattice_properties, by convention, but don't overwrite params
        for key in self.lattice.lp:
            if key not in self.lp:
                self.lp[key] = self.lattice.lp[key]

        ########################################################################
        if self.lp['scale_interactions']:
            scale_exten = ''
        else:
            scale_exten = '_fixscale'

        self.lp['meshfn_exten'] = '_twisty' + xytup_meshfnexten + nl_meshfn_exten + scale_exten

        # Build KK array for stretch couplings
        self.KK, lp['kk'], omk_meshfn_exten = tfns.build_KK(self, self.lp, KK)
        self.lp['meshfn_exten'] += omk_meshfn_exten
        # Form GG array
        self.GG, lp['gg'], omk_meshfn_exten = tfns.build_GG(self, GG)
        self.lp['meshfn_exten'] += omk_meshfn_exten
        # Form CC array
        self.CC, lp['cc'], omk_meshfn_exten = tfns.build_CC(self, CC)
        self.lp['meshfn_exten'] += omk_meshfn_exten

        ########################################################################
        if 'V0_gauss_c' in self.lp and 'V0_gauss_k' in self.lp and 'V0_gauss_g' in self.lp:
            if abs(self.lp['V0_gauss_c']) > eps or (self.lp['V0_gauss_k']) > eps \
                    or (self.lp['V0_gauss_g']) > eps:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_Vk' + sf.float2pstr(self.lp['V0_gauss_k'])
                self.lp['meshfn_exten'] += '_Vg' + sf.float2pstr(self.lp['V0_gauss_g'])
                self.lp['meshfn_exten'] += '_Vc' + sf.float2pstr(self.lp['V0_gauss_c'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_conf{0:04d}'.format(self.lp['pinconf'])
            else:
                self.lp['dcdisorder'] = False
        else:
            self.lp['V0_pin_gauss'] = 0.
            self.lp['V0_gauss'] = 0.
            self.lp['pinconf'] = 0
            self.lp['dcdisorder'] = False

        if self.lp['dcdisorder']:
            # Form abbreviated meshfn exten
            pinmfe = self.get_pinmeshfn_exten()

            print 'Trying to load offset/disorder to pinning frequencies: '
            print dio.prepdir(self.lp['meshfn']) + pinmfe
            # Attempt to load from file
            try:
                self.load_pinning(meshfn=self.lp['meshfn'])
                # self.Omg = np.loadtxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt')
                print 'Loaded ABDelta and/or dcdisordered pinning frequencies.'
            except IOError:
                print 'Could not load ABDelta and/or dcdisordered pinning frequencies, defining them here...'
                # Make Omg from scratch
                if abs(self.lp['ABDelta']) > eps:
                    asites, bsites = glatfns.ascribe_abbonds(self.lattice)
                    raise RuntimeError('Make sure this works')
                    self.KK[abonds] += self.lp['ABDelta']
                    self.KK[bbonds] -= self.lp['ABDelta']
                if self.lp['V0_pin_gauss'] > 0 or self.lp['V0_gauss'] > 0:
                    self.add_dcdisorder()

                # Save non-standard Omg
                if 'save_pinning_to_hdf5' in self.lp:
                    if self.lp['save_pinning_to_hdf5']:
                        force_hdf5pin = True
                    else:
                        force_hdf5pin = False
                else:
                    force_hdf5pin = False
                self.save_couplings(infodir=self.lp['meshfn'], force_hdf5=force_hdf5pin)

        ########################################################################
        # NL_t, KL_t, BL_t are the connectivity of the inner particles (xy_in) connectivity M x M
        if BL_t is not None:
            self.BL_t = BL_t
            if NL_t is not None and KL_t is not None:
                self.NL_t, self.KL_t = NL_t, KL_t
                self.PVx_t, self.PVy_t = PVx_t, PVy_t
            else:
                self.NL_t, self.KL_t = le.BL2NLandKL(self.BL_t)
                self.PVx_t, self.PVy_t = PVx_t, PVy_t
                self.PVxydict_t = PVxydict_t
        else:
            # Remove self.outer_indices from lattice, and use resulting BL to get NL_nm and KL_nm
            # Note that we supply inner_indices as the particles to keep
            xytmp, self.NL_t, self.KL_t, self.BL_t, self.PVxydict_t = \
                le.remove_pts(self.inner_indices, self.lattice.xy, self.lattice.BL, PVxydict=self.lattice.PVxydict,
                              PV=self.lattice.PV)
            # self.PVx_t, self.PVy_t = self.lattice.PVx, self.lattice.PVy
            self.PVx_t, self.PVy_t = le.PVxydict2PVxPVy(self.PVxydict_t, self.NL_t, self.KL_t)

        # Other attributes that can be calculated
        self.matrix = matrix
        self.eigval = eigval
        self.eigvect = eigvect
        self.localization = None
        self.ipr = None
        self.bo = le.bond_length_list(self.lattice.xy, self.lattice.BL, NL=self.lattice.NL, KL=self.lattice.KL,
                                      PVx=self.lattice.PVx, PVy=self.lattice.PVy)

        # print 'self.lp[kk] = ', self.lp['kk']
        # sys.exit()

    def __hash__(self):
        return hash(self.lattice)

    def __eq__(self, other):
        return hasattr(other, 'lattice') and self.lattice == other.lattice

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def outer_inner(self, xy, xyin):
        """Given two xy float arrays, return the indices of sites (index as in xy) which are inside (ie also appear
        in xyin) and outside (do not appear in xyin)

        Parameters
        ----------
        xy : N x 2 float array
            All sites
        xyin : M x 2 float array
            sites that are to be considered mobile in the TwistyLattice class

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
        """Load a saved lattice into the lattice attribute of the TwistyLattice instance.
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

    def load_eigval(self, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        print 'mlat: looking for eigval: ', fn
        if glob.glob(fn):
            with open(fn, "r") as f:
                eigval = pkl.load(f)
            if attribute:
                self.eigval = eigval
            return eigval
        else:
            return None

    def load_eigvect(self, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn):
            print 'loading ', fn
            with open(fn, "r") as f:
                eigvect = pkl.load(f)

            if attribute:
                self.eigvect = eigvect
            return eigvect
        else:
            return None

    def load_ipr(self, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "ipr" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn):
            with open(fn, "r") as f:
                ipr = pkl.load(f)

            if attribute:
                self.ipr = ipr
            return ipr
        else:
            return None

    def load_localization(self, attribute=True):
        """Load eigvect from disk: first try hdf5, then look for pickle"""
        # Make localization name
        locz_name = "localization" + self.lp['meshfn_exten']
        # First look in localization_gyro.hdf5
        h5fn = dio.prepdir(self.lp['meshfn']) + "localization_twisty.hdf5"
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

    def get_ill(self, attribute=False, eigval=None, eigvect=None):
        """Obtain inverse localization length for all modes"""
        localization = self.get_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        # Note: this line below should say localization[:, 3], not [:, 2] as it did before.
        ill = localization[:, 3]
        return ill

    def get_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False, save=False,
                         locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False, force_hdf5=True):
        """Obtain the localization of eigenvectors of the TwistyLattice (fits to 1d exponential decay)
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
            # print 'glat.TwistyLattice(): seeking pklfn file ', fn
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
        matrix = tfns.dynamical_matrix(self, basis=basis)
        if check:
            le.plot_real_matrix(matrix, show=True)

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

    def calc_magevecs(self, eigvect=None):
        if eigvect is None:
            eigvect = self.get_eigvect()

        magevec = tfns.calc_magevecs(eigvect)

        return magevec

    def eig_vals_vects(self, matrix=None, attribute=True, check=False):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        if matrix is None:
            print 'glat.eig_vals_vects: getting matrix...'
            matrix = self.get_matrix(attribute=attribute)
            if check:
                le.plot_complex_matrix(matrix, show=True)

        print 'glat.eig_vals_vects: computing eigval, eigvect...'
        eigval, eigvect = le.eig_vals_vects(matrix, sort='real')

        # Check it
        # print 'eigvect = ', eigvect
        # leplt.plot_complex_matrix(eigvect, show=True)
        # print 'sums = ', np.sum(np.abs(eigvect**2), axis=1)
        # sys.exit()
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
        ipr = len(eigvect) * np.sum(np.abs(eigvect) ** 4, axis=1) / np.sum(np.abs(eigvect) ** 2, axis=1) ** 2

        if attribute:
            self.ipr = ipr
            return self.ipr
        else:
            return ipr

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
                # edge_localization = tlocfns.fit_edgedecay_periodicstrip(self.lattice.xy, eigval, eigvect,
                #                                                         cutoffd=cutoffd, check=self.lp['check'])
                localization = tlocfns.fit_eigvect_to_exponential_1dperiodic(self.lattice.xy, eigval, eigvect,
                                                                             self.lattice.lp['LL'],
                                                                             locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                             check=self.lp['check'])
            else:
                localization = tlocfns.fit_eigvect_to_exponential_periodic(self.lattice.xy, eigval, eigvect,
                                                                           self.lattice.lp['LL'],
                                                                           locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                           check=self.lp['check'])
        else:
            localization = tlocfns.fit_eigvect_to_exponential(self.lattice.xy, eigval, eigvect,
                                                              check=self.lp['check'])

        if attribute:
            self.localization = localization

        return localization

    def build_constitutive_relation(self):
        """

        Returns
        -------

        """
        return tsimfns.constitutive_relation(self)

    def elastic_energy(self):
        energy = 0
        return energy

    def relax(self, fixed, displace, tol=1e-7):
        """

        Parameters
        ----------
        pair : particles whose positions to fix during elastic relaxation

        Returns
        -------

        """
        xyz = np.dstack((self.lattice.xy, np.zeros(len(self.lattice.xy[:, 0]))))[0]
        npts = len(xyz[:, 0])
        if len(fixed) == 2:
            pair = fixed
            bounds = [[None, None]] * (2 * (pair[0])) + \
                      np.c_[xyz[pair[0], :].ravel(), xyz[pair[0], :].ravel()].tolist() + \
                     [[None, None]] * (2 * (pair[1] - pair[0] - 1)) + \
                      np.c_[xyz[pair[1], :].ravel(), xyz[pair[1], :].ravel()].tolist() + \
                     [[None, None]] * (2 * (npts - pair[1] - 1))

        xyzR = scipy.optimize.minimize(self.energy, xyz.ravel(),
                                       method='L-BFGS-B',
                                       bounds=bounds, tol=tol * 0.1).x.reshape((-1, 2))
        return xyzR

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
                    eigval = pkl.load(f)
                with open(infodir + "eigvect" + exten + '.pkl') as f:
                    print 'loading eigvect from pkl file...'
                    eigvect = pkl.load(f)
            else:
                if self.matrix is None:
                    matrix = self.calc_matrix(attribute=attribute)
                    eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
                else:
                    eigval, eigvect = self.eig_vals_vects(attribute=attribute)

                if save_DOS_if_missing:
                    output = open(infodir + 'eigval' + exten + '.pkl', 'wb')
                    pkl.dump(eigval, output)
                    output.close()

                    output = open(infodir + 'eigvect' + exten + '.pkl', 'wb')
                    pkl.dump(eigvect, output)
                    output.close()

                    print 'Saved gyro DOS to ' + infodir + 'eigvect(val)' + exten + '.pkl\n'

        if not glob.glob(infodir + 'eigval_hist' + exten + '.png'):
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'xywz')
            plt.savefig(infodir + 'eigval_hist' + exten + '.png')
            plt.clf()

        if self.lp['periodicBC']:
            colorstr = 'ill'
        else:
            colorstr = 'pr'

        tmov.save_normal_modes_twisty(self, datadir=infodir, sim_type='xywz',
                                      rm_images=True, save_into_subdir=False, overwrite=True, color=colorstr)

    def infinite_dispersion(self, kx=None, ky=None, nkxvals=50, nkyvals=20,
                            save=True, plot=True, save_plot=True, title='Dispersion relation',
                            save_dos_compare=False, outdir=None, name=None, ax=None, xaxis='kx'):
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
        xaxis : str ('kx' or 'ky')
            whether to use kx or ky as the x axis in the dispersion plot

        Returns
        -------
        omegas, kx, ky
        """
        # Note: the function called below has not been finished
        omegas, kx, ky = tlatkspacefns.infinite_dispersion(self, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                                           save=save, plot=plot, save_plot=save_plot,
                                                           title=title, outdir=outdir, name=name, ax=ax, xaxis=xaxis)

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

            eigval = np.real(self.get_eigval())
            ax2.hist(eigval[eigval > 0], bins=50, color=lecmap.green(), alpha=0.2)
            ax.set_title('DOS from dispersion')
            xlims = ax.get_xlim()
            ax.set_xlim(0, xlims[1])
            plt.savefig(name + '_dos.png', dpi=300)

        return omegas, kx, ky

    def lowest_eigenmode(self, nkxy=25, save=False, save_plots=True, name=None, outdir=None, imtype='png'):
        """Obtain and/or plot the lowest eigvects of the TwistyLattice's spectrum

        Parameters
        ----------
        save_plot : bool
            whether to save the plot as a png or pdf
        nkxy : int
        save : bool
        save_plot : bool
        name : str or None
        outdir : str or None
        imtype : str specifier ('png' 'pdf')

        Returns
        -------
        modes :
            normal modes of the lowest eigenvalue states in the BZ evaluated at kxy
        omegas :
        kxy :
        vtx :
        """
        modes, omegas, kxy, symmetryinds, vtcs = tlatkspacefns.lowest_eigenmode(self, nkxy=nkxy, save=save,
                                                                                save_plots=save_plots,
                                                                                name=name, outdir=outdir, imtype=imtype)
        return modes, omegas, kxy, vtcs

    def lowest_mode(self, nkxy=25, save=False, save_plot=True, name=None, outdir=None, imtype='png'):
        """Obtain and/or plot the lowest eigval of the TwistyLattice's spectrum

        Parameters
        ----------
        save_plot : bool
            whether to save the plot as a png or pdf
        nkxy : int
        save : bool
        save_plot : bool
        name : str or None
        outdir : str or None
        imtype : str specifier ('png' 'pdf')

        Returns
        -------
        omegas, vtx, vty
        """
        omegas, kxy, vtcs = tlatkspacefns.lowest_mode(self, nkxy=nkxy, save=save, save_plot=save_plot, name=name,
                                                      outdir=outdir, imtype=imtype)
        return omegas, kxy, vtcs

    def plot_lowest_mode(self, nkxy=25, fig=None, ax=None, cax=None,
                         save=False, name=None, outdir=None, imtype='png'):
        """Obtain and/or plot the lowest eigval of the TwistyLattice's spectrum

        Parameters
        ----------
        save : bool
            whether to save the data as a pickle file
        save_plot : bool
            whether to save the plot as a png or pdf
        nkxy : int
        save : bool
        save_plot : bool
        name : str or None
        outdir : str or None
        imtype : str specifier ('png' 'pdf')

        Returns
        -------
        omegas, vtx, vty
        """
        omegas, kxy, vtcs = tlatkspacefns.lowest_mode(self, nkxy=nkxy, save=save, save_plot=False, name=name,
                                                      outdir=outdir, imtype=imtype)
        if outdir is not None and name is not None:
            outpath = outdir + name + '.' + imtype
        else:
            outpath = None
        fig, ax, cax = tlatkspacefns.plot_lowest_mode(omegas, kxy, vtcs, nkxy=nkxy, fig=fig, ax=ax, cax=cax,
                                                      outpath=outpath)
        return fig, ax, cax, omegas, kxy, vtcs


if __name__ == "__main__":
    '''Use the TwistyLattice class to create a density of states, compute participation ratio binned by polygons,
    or other such things.
    For other example usage, see the script wedge_dispersions.py for making a deformed kagome wedge lattice and
    looking at the dispersions with one free boundary at a time.

    Example usage:
    python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -dispersion -x1 -0.1 -x2 0.1 -x3 0.1
    python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -DOSmovie -x1 -0.1 -x2 0.1 -x3 0.1 -cc 0. -gg 0. -kk 1.0
    python ./twisty/twisty_lattice.py -N 4 -LT square -DOSmovie
    python ./twisty/twisty_lattice.py -N 2 -LT linear -DOSmovie
    python ./twisty/twisty_lattice.py -N 3 -LT linear -DOSmovie
    python ./twisty/twisty_lattice.py -N 5 -LT hexagonal -DOSmovie
    '''
    import argparse
    import copy
    import lepm.lattice_class as lattice_class
    import twisty_scripts as tlatscripts

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create GyroLattice class instance,' +
                                                 ' with options to save or compute attributes of the class.')
    parser.add_argument('-rootdir', '--rootdir', help='Path to networks folder containing lattices/networks',
                        type=str, default='/Users/npmitchell/Dropbox/Soft_Matter/GPU/')
    parser.add_argument('-dispersion', '--dispersion', help='Draw infinite/semi-infinite dispersion relation',
                        action='store_true')
    parser.add_argument('-plot_matrix', '--plot_matrix', help='Plot the dynamical matrix',
                        action='store_true')
    parser.add_argument('-lowest_mode', '--lowest_mode',
                        help='Draw lowest mode of infinite, kspace dispersion relation over the BZ',
                        action='store_true')
    parser.add_argument('-lowest_eigenmode', '--lowest_eigenmode',
                        help='Draw lowest eigenmodes of infinite, kspace dispersion relation over path in the BZ',
                        action='store_true')
    parser.add_argument('-disp_abtrans', '--dispersion_abtransition',
                        help='Draw infinite/semi-infinite dispersion relation through AB transition',
                        action='store_true')
    parser.add_argument('-disp_abbounds', '--dispersion_abtransition_gapbounds',
                        help='Draw infinite/semi-infinite dispersion relation through AB transition and plot bounds',
                        action='store_true')
    parser.add_argument('-gap_sweep', '--gap_sweep',
                        help='Sweep through parameters of kk, gg, cc to look for good features', action='store_true')
    parser.add_argument('-glatparam_sweep', '--glatparam_sweep',
                        help='Sweep through a gyrolattice parameter and plot spectrum vs param', action='store_true')
    parser.add_argument('-glatparam', '--glatparam',
                        help='The twistylattice parameter to vary', type=str, default='ABDelta_k')
    parser.add_argument('-glatvals', '--glatvals',
                        help='The gyrolattice parameter values to use for varying', type=str, default='0:0.1:1.5')
    parser.add_argument('-save_prpoly', '--save_prpoly',
                        help='Create dict and hist of excitation participation, grouped by polygonal contributions',
                        action='store_true')
    parser.add_argument('-dcdisorder', '--dcdisorder', help='Construct DOS with delta correlated disorder and view ipr',
                        action='store_true')
    parser.add_argument('-save_ipr', '--save_ipr', help='Load TwistyLattice and save ipr',
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
    parser.add_argument('-KKspec', '--KKspec', help='string specifier for OmK bond frequency matrix',
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
    parser.add_argument('-basis', '--basis', help='basis for computing eigvals', type=str, default='XY')
    parser.add_argument('-kk', '--kk', help='Spring frequency', type=str, default='1.0')
    parser.add_argument('-gg', '--gg', help='twist-stretch coupling', type=str, default='1.0')
    parser.add_argument('-cc', '--cc', help='Twist-twist coupling', type=str, default='1.0')
    parser.add_argument('-fix_interactions', '-fix_interactions',
                        help='Do not scale the interaction strength by the length of each bond',
                        action='store_true')
    parser.add_argument('-intrange', '--intrange',
                        help='Consider couplings only to nth nearest neighbors (if ==2, then NNNs, '
                             'for intrange=0, consider infinite range (all coupled), etc)',
                        type=int, default=1)
    parser.add_argument('-Vg', '--V0_gauss_g',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)
    parser.add_argument('-Vk', '--V0_gauss_k',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)
    parser.add_argument('-Vc', '--V0_gauss_c',
                        help='St.deviation of distribution of twist-coupling disorder', type=float, default=0.0)

    # Lattice loading arguments
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
    parser.add_argument('-ABk', '--ABDelta_k', help='Difference in stetch coefficient for AB bonds', type=float,
                        default=0.)
    parser.add_argument('-ABg', '--ABDelta_g', help='Difference in twist-stetch coupling for AB bonds', type=float,
                        default=0.)
    parser.add_argument('-ABc', '--ABDelta_c', help='Difference in twist coefficient for AB bonds', type=float,
                        default=0.)

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
    parser.add_argument('-x1', '--x1',
                        help='1st Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x2', '--x2',
                        help='2nd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-x3', '--x3',
                        help='3rd Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-zz', '--zz',
                        help='4th Deformation parameter for deformed_kagome', type=float, default=0.00)
    parser.add_argument('-nkx', '--nkx',
                        help='Number of kx values in dispersion', type=int, default=50)
    parser.add_argument('-nky', '--nky',
                        help='Number of ky values in dispersion', type=int, default=50)

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
    dcdisorder = args.V0_gauss_k > 0 or args.V0_gauss_g > 0 or args.V0_gauss_c > 0

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
          'kk': float((args.kk).replace('n', '-').replace('p', '.')),
          'gg': float((args.gg).replace('n', '-').replace('p', '.')),
          'cc': float((args.cc).replace('n', '-').replace('p', '.')),
          'V0_gauss_k': args.V0_gauss_k,
          'V0_gauss_g': args.V0_gauss_g,
          'V0_gauss_c': args.V0_gauss_c,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta_k': args.ABDelta_k,
          'ABDelta_g': args.ABDelta_g,
          'ABDelta_c': args.ABDelta_c,
          'thres': args.thres,
          'pinconf': args.pinconf,
          'KKspec': args.KKspec,
          'spreading_time': args.spreading_time,
          'intparam': args.intparam,
          'interaction_range': args.intrange,
          'save_pinning_to_hdf5': not args.save_pinning_to_txt,
          'scale_interactions': not args.fix_interactions,
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
        tlat = TwistyLattice(lat, lp)
        print 'loading lattice...'
        tlat.load()
        print 'Saving DOS movie...'
        tlat.save_DOSmovie()

    if args.dispersion:
        """Example usage:
        python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -dispersion -x1 -0.1 -x2 0.1 -x3 0.1 -gg 0.1 -cc 1. -nky 50
        # make the strip
        python ./build/make_lattice.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -skip_polygons -skip_gyroDOS
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        tlat = TwistyLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        tlat.eig_vals_vects(attribute=True)
        tlat.infinite_dispersion(save=False, nkxvals=args.nkx, nkyvals=args.nky, save_dos_compare=False)

    if args.lowest_mode:
        """Example usage:
        python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 -x3 0.1 -gg 0.1 -cc 1. -nky 50 -fix_interactions
        # build the lattice
        python build/make_lattice.py -LT deformed_kagome -N 1 -periodic -x1 -0.1 -x2 0.3 -x3 0.1 -skip_polygons
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        tlat = TwistyLattice(lat, lp)
        tlat.lowest_mode(nkxy=args.nkx)

    if args.dispersion_abtransition:
        """Example usage:
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 4 -disp_abtrans -basis psi -aol 1.2 -Omk -0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 1.2 -Omk -0.5
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.85 -Omk -0.75
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.8 -Omk -0.344
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.6 -Omk 0.128
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic -N 1 -disp_abtrans -basis psi -aol 0.6
        python magnetic_gyro_lattice_class.py -LT hexagonal -shape square -periodic_strip -NH 1 -NV 5 -disp_abtrans -basis psi
        """
        mglatscripts.dispersion_abtransition(lp, fullspectrum=False)

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

            tlat = TwistyLattice(lat, lpnew)
            eigval = tlat.get_eigval()

            eigval_upper = np.abs(np.imag(eigval[0:int(len(eigval) * 0.5)][::-1]))

            if lp['periodicBC']:
                locz = tlat.get_localization(attribute=True)
                # First time around, save ill
                # loc_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'localization' +
                #                        tlat.lp['meshfn_exten'] + '.txt')
                # if not loc_exists:
                #     print '\nLocalization file is not already saved. Creating it...'
                #     plt.close('all')
                #     tlat.save_localization(attribute=True, save_images=False, save_eigvect_eigval=False)
                #     tlat.plot_ill_dos()
                #     plt.close('all')
                ill_upper = locz[:, 2]
            else:
                ipr = tlat.get_ipr(attribute=True)
                # First time around, save ipr
                # ipr_exists = glob.glob(dio.prepdir(lat.lp['meshfn']) + 'ipr' +
                #                        tlat.lp['meshfn_exten'] + '.txt')
                # if not ipr_exists:
                #     print '\nLocalization file is not already saved. Creating it...'
                #     plt.close('all')
                #     tlat.save_ipr(attribute=True, save_images=False)
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

    if args.plot_matrix:
        """Example usage:
        python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -plot_matrix -x1 -0.1 -x2 0.1 -x3 0.1 -gg 0. -cc 0.
        """
        from lepm.mass_lattice_class import MassLattice
        lp0 = copy.deepcopy(lp)
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        tlat = TwistyLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        matrix = tlat.get_matrix()
        ind = 2 * len(tlat.lattice.xy)
        fig, (ax1, ax2), (cbar1, cbar2) = le.plot_complex_matrix(matrix[0:ind, 0:ind], show=False, close=False)
        plt.title('Dynamical matrix:' + lp['meshfn_exten'])
        outfn = dio.prepdir(lp['meshfn']) + 'matrix_' + lp['meshfn_exten'] + '.png'
        print 'saving ' + outfn
        plt.savefig(outfn)
        aa, bb = np.linalg.eig(matrix)
        print 'eigvals =', aa

        # Compare to mass lattice
        lp = copy.deepcopy(lp0)
        lp['mass'] = 1.0
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        tlat = MassLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        matrix = tlat.get_matrix()
        le.plot_complex_matrix(matrix, show=False, close=False)
        plt.title('Dynamical matrix:' + lp['meshfn_exten'])
        outfn = dio.prepdir(lp['meshfn']) + 'matrix_' + lp['meshfn_exten'] + '.png'
        print 'saving ' + outfn
        plt.savefig(outfn)
        aa, bb = np.linalg.eig(matrix)
        print 'eigvals =', aa

    if args.lowest_eigenmode:
        """Example usage:
        python twisty/twisty_lattice.py -LT deformed_kagome -N 1 -periodic -lowest_eigenmode -x1 -0.1 -x2 0.1 -x3 0.1 \
            -gg 0.1 -cc .48 -nky 50 -fix_interactions

        python twisty/twisty_lattice.py -LT deformed_kagome -NH 1 -NV 10 -periodic_strip -lowest_eigenmode \
            -x1 -0.05 -x2 0.05 -x3 0.05 -gg 0.1 -cc .48 -nky 50 -fix_interactions -theta 0.0

        # build the lattice
        python build/make_lattice.py -LT deformed_kagome -N 1 -periodic -x1 -0.1 -x2 0.3 -x3 0.1 -skip_polygons
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])

        if lat.lp['periodic_strip']:
            bnd = lat.boundary
            ii = 0
            # do each boundary, indexed by ii
            for by in bnd:
                outer = by
                inner = np.setdiff1d(np.arange(len(lat.xy)), outer)
                xytup = (lat.xy[inner], lat.xy)
                if ii == 0:
                    xytup_meshfnexten = '_clamptop' + str(len(by))
                else:
                    xytup_meshfnexten = '_clampbottom' + str(len(by))

                tlat = TwistyLattice(lat, lp, xytup=xytup, xytup_meshfnexten=xytup_meshfnexten)
                tlat.lowest_eigenmode(nkxy=args.nkx, save=True)
                ii += 1
        else:
            # just do typical periodic case
            tlat = TwistyLattice(lat, lp)
            tlat.lowest_eigenmode(nkxy=args.nkx)




