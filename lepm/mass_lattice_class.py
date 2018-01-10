import numpy as np
import cPickle as pkl
import lepm.plotting.plotting as leplt
import matplotlib.pyplot as plt
import lepm.lattice_elasticity as le
from lepm.timer import Timer
import argparse
import copy
import lepm.stringformat as sf
import lepm.dataio as dio
import glob
import os
import lepm.plotting.movies as lemov
import lepm.mass_lattice_functions as mlatfns
import lepm.plotting.colormaps as lecmaps
from mpl_toolkits.mplot3d import Axes3D
import sys
import mass_lattice_kspace_functions as mlatkspacefns

'''Defines the MassLattice class: masses + springs

Example usage:
python mass_lattice_class.py -LT hexagonal -shape hexagon -N 5 -DOSmovie
python mass_lattice_class.py -LT deformed_kagome -N 1 -periodic -x1 -0.1 -x2 0.1 -x3 0.1 -DOSmovie


'''


class MassLattice:
    """Masses (with variable mass possible via tuning V0_mass_flat) and """
    def __init__(self, lattice, lp, xytup=None, xytup_meshfnexten=None, kk=None, mass=None, bL=None,
                 kpin=None, matrix=None, eigvect=None, eigval=None, ipr=None,
                 prpoly=None, ldos=None, localization=None):
        """Initializes the class

        Parameters
        ----------
        val_array :
            contains [spring, pin, other things?] Currently just holds spring and pinning values, but you could add
            more things here, which would fall under self.labels

        Class members
        ----------
        lattice : lepm.lattice_class.Lattice() instance
        xytup : tuple of (M x 2 float array, N x 2 float array)
            Equilibrium positions of inner gyros, all gyros --> so (inner, all)
        spring : float
            spring constant
        pin : float
            gravitational spring constant
        matrix : matrix of dimension 2n x 2n
            Linearized matrix for finding normal modes of system
        eigval : array of dimension 2nx1
            sqrt(-Eigenvalues of self.matrix)
        eigvect : array of dimension 2n x 2n
            Eigenvectors of self.matrix
        """
        self.lattice = lattice
        self.lp = lp
        eps = 1e-7

        # Add lattice properties to self.lp, by convention, but don't overwrite params
        for key in self.lattice.lp:
            if key not in self.lp:
                self.lp[key] = self.lattice.lp[key]

        # print 'mass_lattice_class: self.lp  = ', self.lp

        self.lp['meshfn_exten'] = '_massspring'

        # Check if boundary is fixed in place
        if xytup is None:
            # Create self.xy and self.xy_inner
            self.xy = lattice.xy
            if 'roi' in self.lp:
                self.xy_inner = le.pts_in_polygon(self.xy, self.lp['roi'])
            elif 'immobile_boundary' in self.lp:
                if self.lp['immobile_boundary']:
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
                            if not 'immobile_boundary0' in self.lp:
                                self.lp['immobile_boundary0'] = False
                            if not 'immobile_boundary1' in self.lp:
                                self.lp['immobile_boundary1'] = False

                            # Decide on inner and outer indices
                            if self.lp['immobile_bounary0']:
                                # Exclude only boundary0 from inner_indices
                                boundary = boundary[0]
                                inners = np.setdiff1d(inners, np.array(boundary))
                                outers.append(boundary)
                                self.lp['immobile_boundary1'] = False
                                xytup_meshfnexten = '_immobilebnd0'
                            elif self.lp['immobile_bounary1']:
                                # Exclude only boundary0 from inner_indices
                                boundary = boundary[1]
                                inners = np.setdiff1d(inners, np.array(boundary))
                                outers.append(boundary)
                                self.lp['immobile_boundary0'] = False
                                xytup_meshfnexten = '_immobilebnd1'
                            else:
                                # Exclude all boundaries from inner_indices
                                for bndy in boundary:
                                    inners = np.setdiff1d(inners, np.array(bndy))
                                    outers.append(bndy)
                                xytup_meshfnexten = '_immobilebnd'
                        else:
                            # Exclude all boundaries from inner_indices
                            for bndy in boundary:
                                inners = np.setdiff1d(inners, np.array(bndy))
                                outers.append(bndy)

                        self.inner_indices = np.hstack(inners)
                        self.outer_indices = np.hstack(outers)
                        self.xy_inner = self.xy[self.inner_indices]
                        xytup_meshfnexten = '_immobilebnd'
                    else:
                        self.inner_indices = np.setdiff1d(np.arange(len(self.xy)), np.array(boundary))
                        self.outer_indices = np.array(boundary)
                        self.xy_inner = self.xy[self.inner_indices]
                        xytup_meshfnexten = '_immobilebnd'
                else:
                    self.xy_inner = self.xy
                    self.xy_outer = np.array([])
                    xytup_meshfnexten = ''
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

        self.lp['meshfn_exten'] += xytup_meshfnexten

        print 'self.xy_inner = ', np.shape(self.xy_inner)
        print 'self.xy = ', np.shape(self.xy)
        self.outer_indices, self.inner_indices, self.total2inner = self.outer_inner(self.xy, self.xy_inner)
        ############################################################
        # Form spring constant matrix kk
        self.kk, lp_kk, kk_meshfn_exten = mlatfns.build_kk(self.lattice, self.lp, kk)
        self.lp['kk'] = lp_kk
        self.lp['meshfn_exten'] += kk_meshfn_exten
        print 'meshfn_exten -> ', self.lp['meshfn_exten']

        ############################################################
        # Form mass
        if mass == 'auto' or mass is None:
            if 'mass' in self.lp:
                self.mass = self.lp['mass'] * np.ones_like(self.lattice.xy[:, 0])
            else:
                print 'giving mass the default value of 1s...'
                self.mass = 1.0 * np.ones_like(self.lattice.xy[:, 0])
            if self.lp['mass'] != 1.0:
                self.lp['meshfn_exten'] += '_mass' + sf.float2pstr(self.lp['mass'])
        else:
            self.mass = mass
            if 'mass' in self.lp:
                if (self.mass != self.lp['mass'] * np.ones(np.shape(self.lattice.xy)[0])).any():
                    self.lp['meshfn_exten'] += '_massspec'
                    self.lp['mass'] = -5000
            else:
                # Check if the values of all elements are identical
                kinds = np.nonzero(self.mass)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in mass. Check if all the same
                    value = self.mass[kinds[0]]
                    if (mass[kinds] == value).all():
                        self.lp['mass'] = value
                    else:
                        self.lp['mass'] = -5000
                else:
                    self.lp['mass'] = 0.0

        ############################################################
        # Store the rest bond lengths for all springs
        self.bL = bL

        ############################################################
        # Form bonds that pin each mass to its rest site
        if kpin == 'auto' or kpin is None:
            if 'kpin' in self.lp:
                self.kpin = self.lp['kpin'] * np.ones_like(self.lattice.xy[:, 0])
            else:
                print 'giving kpin the default value of zeros...'
                self.kpin = 0. * np.ones_like(self.lattice.xy[:, 0])
                self.lp['kpin'] = 0.0
            if self.lp['kpin'] != 0.0:
                self.lp['meshfn_exten'] += '_kpin' + sf.float2pstr(self.lp['kpin'])
        else:
            self.kpin = kpin
            if 'kpin' in self.lp:
                if (self.kpin != self.lp['kpin'] * np.ones(np.shape(self.lattice.xy)[0])).any():
                    self.lp['meshfn_exten'] += '_kpinspec'
                    self.lp['kpin'] = -5000
            else:
                # Check if the values of all elements are identical
                kinds = np.nonzero(self.kpin)
                if len(kinds[0]) > 0:
                    # There are some nonzero elements in kpin. Check if all the same
                    value = self.kpin[kinds[0]]
                    if (kpin[kinds] == value).all():
                        self.lp['kpin'] = value
                    else:
                        self.lp['kpin'] = -5000
                else:
                    self.lp['kpin'] = 0.0

        #########################################################################################
        # Naming for inhomogeneity in pinning or masses or springs, adding ABDelta offsets
        #########################################################################################
        if 'ABDelta' in self.lp:
            if self.lp['ABDelta'] > 0:
                self.lp['meshfn_exten'] += '_ABd{0:0.3f}'.format(self.lp['ABDelta']).replace('.', 'p')
        else:
            self.lp['ABDelta'] = 0.

        print 'meshfn_exten --> ', self.lp['meshfn_exten']

        if 'V0_mass_flat' in self.lp or 'V0_spring_flat' in self.lp or 'V0_pin_flat' in self.lp:
            if 'V0_mass_flat' not in self.lp:
                lp['V0_mass_flat'] = 0.
            if 'V0_spring_flat' not in self.lp:
                lp['V0_spring_flat'] = 0.
            if 'V0_pin_flat' not in self.lp:
                lp['V0_pin_flat'] = 0.

            if self.lp['V0_mass_flat'] > 0 or self.lp['V0_spring_flat'] > 0 or self.lp['V0_pin_flat'] > 0:
                self.lp['dcdisorder'] = True
                self.lp['meshfn_exten'] += '_massVf' + sf.float2pstr(self.lp['V0_mass_flat'])
                self.lp['meshfn_exten'] += '_sprVf' + sf.float2pstr(self.lp['V0_spring_flat'])
                self.lp['meshfn_exten'] += '_pinVf' + sf.float2pstr(self.lp['V0_pin_flat'])
                if 'pinconf' not in self.lp:
                    self.lp['pinconf'] = 0
                elif self.lp['pinconf'] > 0:
                    self.lp['meshfn_exten'] += '_conf{0:04d}'.format(self.lp['pinconf'])
            else:
                self.lp['dcdisorder'] = False
        else:
            self.lp['V0_mass_flat'] = 0.
            self.lp['V0_spring_flat'] = 0.
            self.lp['V0_pin_flat'] = 0.
            self.lp['pinconf'] = 0
            self.lp['dcdisorder'] = False

        if self.lp['ABDelta'] > 0 or self.lp['V0_mass_flat'] > 0 or self.lp['V0_spring_flat'] > 0 \
                or self.lp['V0_pin_flat'] > 0:
            # In order to load the random (V0) or alternating (AB) pinning sites, look for a txt file with the pinnings
            # that also has specifications in its meshfn exten, but IGNORE other parts of meshfnexten, if they exist.
            # Form abbreviated meshfn exten
            pinmfe = self.get_pinmeshfn_exten()

            print 'Trying to load offset/disorder to pinning frequencies: '
            print dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt'

            # Attempt to load from file
            try:
                self.mass = np.loadtxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt')
                print 'Loaded ABDelta and/or dcdisordered pinning frequencies.'
            except IOError:
                print 'Could not load ABDelta and/or dcdisordered pinning frequencies, defining them here...'
                # Make mass from scratch
                if self.lp['ABDelta'] > 0:
                    asites, bsites = mlatfns.ascribe_absites(self.lattice)
                    self.mass[asites] += self.lp['ABDelta']
                    self.mass[bsites] -= self.lp['ABDelta']
                if self.lp['V0_mass_flat'] > 0 or self.lp['V0_spring_flat'] > 0 or self.lp['V0_pin_flat'] > 0:
                    self.add_dcdisorder()

                # Save non-standard mass
                self.save_mass(infodir=self.lp['meshfn'], histogram=False)
                np.savetxt(dio.prepdir(self.lp['meshfn']) + pinmfe + '.txt', self.mass)
                # self.plot_mass()

        # things that have to be calculated
        self.matrix = matrix
        self.eigvect = eigvect
        self.eigval = eigval
        self.ipr = ipr
        self.prpoly = prpoly
        self.ldos = ldos
        self.localization = localization

        print 'meshfn_exten --> ', self.lp['meshfn_exten']

    def __hash__(self):
        return hash(self.lattice)

    def __eq__(self, other):
        return hasattr(other, 'lattice') and self.lattice == other.lattice

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

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
        """Load a saved lattice into the lattice attribute of the MassLattice instance.
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

        if self.lp['V0_spring_flat'] > 0:
            print 'todo: This is not finished'
            sys.exit()
        else:
            self.kk = self.lp['kk'] * np.abs(self.lattice.KL)

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
        print 'grabbing pinmeshfn_exten...'
        pinmfe = 'mass_mean' + sf.float2pstr(self.lp['mass']) + self.lp['meshfn_exten']
        return pinmfe

    def add_dcdisorder(self):
        """Add noise to masses or spring constants (delta-correlated disorder)"""
        # Add noise to pinning energies
        if self.lp['V0_mass_flat'] > 0:
            self.mass += self.lp['V0_mass_flat'] * (np.random.rand(len(self.lattice.xy)) - 0.5) * 2.0
        if self.lp['V0_spring_flat'] > 0:
            raise RuntimeError('This is not done correctly here')
            self.kk += self.lp['V0_spring_flat'] * np.random.randn(np.shape(self.lattice.KL)[0],
                                                                     np.shape(self.lattice.KL)[1])

    def load_eigval_eigvect(self, attribute=True):
        fn_evl = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        print 'fn_evl = ', fn_evl
        fn_evt = dio.prepdir(self.lp['meshfn']) + "eigvect" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn_evl) and glob.glob(fn_evt):
            # with open(fn_evl, "r") as f:
            #     eigval = pkl.load(f)

            pklin = open(fn_evl, "rb")
            eigval = pkl.load(pklin)
            pklin.close()

            # with open(fn_evt, "r") as f:
            #    eigvect = pkl.load(f)
            pklin = open(fn_evt, "rb")
            eigvect = pkl.load(pklin)
            pklin.close()

            if attribute:
                self.eigval = eigval
                self.eigvect = eigvect
        else:
            return None

        return eigval, eigvect

    def load_eigval(self, attribute=True):
        fn = dio.prepdir(self.lp['meshfn']) + "eigval" + self.lp['meshfn_exten'] + ".pkl"
        # print 'mlat: looking for eigval: ', fn
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

    def load_prpoly(self, attribute=False):
        fn = dio.prepdir(self.lp['meshfn']) + "prpoly" + self.lp['meshfn_exten'] + ".pkl"
        if glob.glob(fn):
            with open(fn, "rb") as f:
                prpoly = pkl.load(f)

            if attribute:
                self.prpoly = prpoly

            return prpoly
        else:
            return None

    def get_matrix(self, attribute=False):
        if self.matrix is None:
            print 'masslat: about to calculate matrix'
            return self.calc_matrix(attribute=attribute)
        else:
            return self.matrix

    def get_eigval_eigvect(self, attribute=False):
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
                    matrix = self.calc_matrix(attribute=False)
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

    def get_eigvect(self, attribute=False):
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
                    matrix = self.calc_matrix(attribute=False)
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
                    ldos = pkl.load(fn)
            else:
                print 'calculating ldos...'
                ldos = self.calc_ldos(eps=self.lp['eps'], attribute=attribute)
        else:
            ldos = self.ldos
        return ldos

    def get_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                         locutoffd=None, hicutoffd=None, save_eigval=False, attribute_eigv=False):
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
                print 'calculating localization...'
                localization = self.calc_localization(attribute=attribute, eigval=eigval, eigvect=eigvect,
                                                      save_eigvect_eigval=save_eigvect_eigval, locutoffd=locutoffd,
                                                      hicutoffd=hicutoffd, save_eigval=save_eigval,
                                                      attribute_eigv=attribute_eigv)
        return localization

    def get_ill(self, attribute=False, eigval=None, eigvect=None):
        localization = self.get_localization(attribute=attribute, eigval=eigval, eigvect=eigvect)
        ill = localization[:, 2]
        return ill

    def ensure_eigval_eigvect(self, eigval=None, eigvect=None, attribute=True, load=True):
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
                # Try to load eigval and eigvect, if load or attribute are true
                if load or attribute:
                    print 'Attempting to load eigval/vect...'
                    eigval = self.load_eigval(attribute=attribute)
                    eigvect = self.load_eigvect(attribute=attribute)
        else:
            print 'mlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
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
            print 'mlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if eigval is None:
                eigval, eigvect = self.eig_vals_vects()
            print '... saving eigval/vects'
            self.save_eigval()

        if attribute:
            self.eigval = eigval
        return eigval

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
            print 'mlat.load_eigval_eigvect: Could not load eigval/vect, computing it and saving it...'
            if ipr is None:
                ipr = self.calc_ipr(attribute=attribute)
            print '... saving eigval/vects'
            self.save_ipr()

        return ipr

    def save_eigval_eigvect(self, eigval=None, eigvect=None, infodir='auto', attribute=True):
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

        eigvalfn = infodir + 'eigval' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvalfn, 'wb')
        pkl.dump(eigval, output)
        output.close()

        eigvectfn = infodir + 'eigvect' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvectfn, 'wb')
        pkl.dump(eigvect, output)
        output.close()

        fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'mass')
        plt.savefig(infodir + 'eigval_hist' + self.lp['meshfn_exten'] + '.png')
        plt.clf()
        print 'Saved mass DOS to ' + eigvalfn + '\n and ' + eigvectfn

    def save_eigval(self, eigval=None, infodir='auto', attribute=True):
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

        eigvalfn = infodir + 'eigval' + self.lp['meshfn_exten'] + '.pkl'
        output = open(eigvalfn, 'wb')
        pkl.dump(eigval, output)
        output.close()

        fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'mass')
        plt.savefig(infodir + 'eigval_mass_hist' + self.lp['meshfn_exten'] + '.png')
        plt.clf()
        print 'Saved mass DOS to ' + eigvalfn

    def save_mass(self, infodir='auto', histogram=True, attribute=True):
        """Save kk pinning frequencies for this GyroLattice

        Parameters
        ----------
        infodir : str (default = 'auto')
            The path where to save eigval, eigvect
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
        """
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])
        if self.mass is not None:
            fn = dio.prepdir(self.lp['meshfn']) + 'mass_mean' + sf.float2pstr(self.lp['mass']) + \
                 self.lp['meshfn_exten'] + '.txt'
            np.savetxt(fn, self.mass, header="Pinning frequencies Omg/mass")
            plt.clf()
            if histogram:
                # Note: Keep Omg label here for loading purposes
                fig, hist_ax = leplt.initialize_histogram(self.mass, xlabel=r'Pinning frequencies, $\Omega_g$, ' +
                                                                            r'or masses, $m$')
                histfn = 'Omg_hist_mean' + sf.float2pstr(self.lp['mass']) + self.lp['meshfn_exten']
                plt.savefig(infodir + histfn + '.png')
                plt.clf()
            print 'Saved Omg to ' + fn

    def save_ipr(self, infodir='auto', attribute=True, save_images=True, show=False, **kwargs):
        """
        Parameters
        ----------
        infodir: str
            Directory in which to save ipr as ipr.pkl
        attribute: bool
            Whether to attribute the matrix, eigvals, and eigvects to self (ie self.eigval = eigval, etc)
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
        pkl.dump(ipr, file(fn, 'wb'))

        if attribute:
            self.ipr = ipr

        if save_images:
            # save IPR as png
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for mass-spring network',
                              fname='ipr_mass_hist',
                              alpha=1.0, FSFS=12, inverse_PR=True, show=show, **kwargs)
            plt.close('all')
            self.plot_ipr_DOS(outdir=infodir, title=r'$D(\omega)$ for mass-spring network',
                              fname='pr_mass_hist',
                              alpha=1.0, FSFS=12, inverse_PR=False, show=show, **kwargs)
        print 'Saved mass ipr to ' + infodir + 'ipr' + self.lp['meshfn_exten'] + '.pkl'

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
            pkl.dump(prpoly, fn)

        if save_plot:
            # save prpoly as png
            self.plot_prpoly(outdir=infodir, show=False, shaded=False)

        print 'Saved mass prpoly to ' + infodir + 'prpoly' + self.lp['meshfn_exten'] + '.pkl'

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
        pkl.dump(ldos, file(fn, 'wb'))

        if save_images:
            eigval = self.get_eigval()
            # save LDOS as png
            ldos_infodir = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' + sf.float2pstr(self.lp['eps']) + '/'
            dio.ensure_dir(ldos_infodir)
            self.plot_ldos(eigval=eigval, ldos=ldos, outdir=ldos_infodir, FSFS=12)

            # Make movie
            imgname = ldos_infodir + 'ldos_site'
            movname = infodir + 'ldos' + self.lp['meshfn_exten'] + '_eps' + sf.float2pstr(self.lp['eps']) + '_sites'
            lemov.make_movie(imgname, movname, indexsz='08', imgdir=ldos_infodir, rm_images=True,
                             save_into_subdir=True)

        print 'Saved mass ldos to ' + fn

    def save_localization(self, eigval=None, infodir='auto', attribute=True, save_images=False,
                          save_eigvect_eigval=False):
        """Get and save localization measure for all eigenvectors of the GyroLattice"""
        if infodir == 'auto':
            infodir = dio.prepdir(self.lattice.lp['meshfn'])

        locz = self.get_localization(attribute=attribute, save_eigvect_eigval=save_eigvect_eigval)
        fn = infodir + 'localization' + self.lp['meshfn_exten'] + '.txt'
        print 'Saving localization as ' + fn
        header = "Localization of eigvects: fitted to A*exp(K*sqrt((x-xc)**2 + (y-yc)**2)): " + \
                 "xc, yc, A, K, uncA, covAK, uncK. The modes examined range from int(len(eigval)*0.5) to len(eigval)."
        np.savetxt(fn, locz, delimiter=',', header=header)

        # Save summary plot of exponential decay param
        if eigval is None:
            eigval = self.get_eigval()

        fig, ax = leplt.initialize_1panel_centered_fig(wsfrac=0.6, hsfrac=0.6 * 0.75, fontsize=10, tspace=3)
        evals = np.real(eigval[int(len(eigval) * 0.5):])
        ax.plot(evals, -locz[:, 3], '-', color='#334A5A')
        ax.fill_between(evals, -locz[:, 3] - np.sqrt(locz[:, 6]), -locz[:, 3] + np.sqrt(locz[:, 6]),
                        color='#89BBDB')
        # if abs(self.lp['Omg']) == 1 and abs(self.lp['kk']) == 1 and not self.lp['dcdisorder']:
        #     ax.set_xlim(1.0, 4.0)
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

        print 'Saved mass localization to ' + fn
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

                    print 'Saved mass DOS to ' + infodir + 'eigvect(val)' + exten + '.pkl\n'

        if not glob.glob(infodir + 'eigval_mass_hist' + exten + '.png'):
            fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'mass')
            plt.savefig(infodir + 'eigval_mass_hist' + exten + '.png')
            plt.clf()

        if self.lp['periodicBC']:
            colorstr = 'ill'
        else:
            colorstr = 'pr'

        lemov.save_normal_modes_mass(self, datadir=infodir, rm_images=True, save_into_subdir=False, overwrite=True,
                                     color=colorstr)

    def calc_matrix(self, attribute=False):
        """calculates the matrix for finding the normal modes of the system"""
        matrix = mlatfns.dynamical_matrix_mass(self)
        if attribute:
            self.matrix = matrix
        print 'masslat: returning matrix'
        return matrix

    def calc_eigvals(self, matrix=None, attribute=True):
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
        # # For mass system, take negative and sqrt
        eigval = np.sqrt(-eigval)
        # use real part to get ascending order of eigvals
        si = np.argsort(np.real(eigval))
        eigval_out = eigval[si]

        if attribute:
            self.eigval = eigval_out

        return eigval_out

    def eig_vals_vects(self, matrix=None, attribute=False, check=True):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        if matrix is None:
            print 'mlat.eig_vals_vects: getting matrix...'
            matrix = self.get_matrix(attribute=attribute)
            if check:
                le.plot_complex_matrix(matrix, show=True)

        print 'mlat.eig_vals_vects: computing eigval, eigvect...'
        eigval, eigvect = le.eig_vals_vects(matrix, sort='real')
        # take eigval to negative itself since we have -omega^2 = ...
        eigval = -eigval
        # Take square root to get frequencies
        eigval = np.sqrt(eigval)
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
        ldos = mlatfns.calc_ldos(eigval, eigvect, eps=eps)
        if attribute:
            self.ldos = ldos

        return ldos

    def calc_localization(self, attribute=False, eigval=None, eigvect=None, save_eigvect_eigval=False,
                          save_eigval=False, cutoffd=None, locutoffd=None, hicutoffd=None, attribute_eigv=False):
        """For each eigvector excitation, fit excitation to an exponential decay centered
        about the excitation's COM.
        where fit is A * exp(K * np.sqrt((x - x_center)**2 + (y - y_center)**2))

        Returns
        -------
        localization : NP x 7 float array (ie, len(eigval) x 7 float array)
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
                # edge_localization = mlatfns.fit_edgedecay_periodicstrip(self.lattice.xy, eigval, eigvect,
                #                                                         cutoffd=cutoffd, check=self.lp['check'])
                localization = mlatfns.fit_eigvect_to_exponential_1dperiodic(self.lattice.xy, eigval, eigvect,
                                                                             self.lattice.lp['LL'],
                                                                             locutoffd=cutoffd, hicutoffd=cutoffd,
                                                                             check=self.lp['check'])
            else:
                localization = mlatfns.fit_eigvect_to_exponential_periodic(self.lattice.xy, eigval, eigvect,
                                                                           self.lattice.lp['LL'],
                                                                           locutoffd=locutoffd, hicutoffd=hicutoffd,
                                                                           check=self.lp['check'])
        else:
            localization = mlatfns.fit_eigvect_to_exponential(self.lattice.xy, eigval, eigvect, cutoffd=cutoffd,
                                                              check=self.lp['check'])

        if attribute:
            self.localization = localization

        return localization

    def plot_prpoly(self, outdir=None, title=r'Polygonal contributions to normal mode excitations',
                    fname='prpoly_mass_hist', fontsize=8, show=True, global_alpha=1.0, shaded=False, save=True):
        """Plot Inverse Participation Ratio of the mass-spring network

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
        n_ax = int(np.ceil(len(prpoly) / float(nrows))) - 3
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

        ax[len(ax) - 1].set_xlabel(r'Oscillation frequency, $\omega$')
        ax[len(ax) - 1].annotate('Density of states, $D(\omega)$', xy=(.001, .5), xycoords='figure fraction',
                                 horizontalalignment='left', verticalalignment='center',
                                 fontsize=fontsize, rotation=90)

        if title is not None:
            plt.suptitle(title, fontsize=fontsize)

        if save:
            if outdir is None:
                outdir = dio.prepdir(self.lp['meshfn'])

            # pkl.dump(fig, file(outdir+fname+'.pkl','w'))
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
        for ii in range(len(prpoly) - 3):
            # print 'ax[', ii, '] = ', ax[ii], ' of ', len(prpoly)
            if shaded:
                DOS_ax, cbar_ax, cbar, n, bins = \
                    leplt.shaded_DOS_plot(eigval, ax[ii], 'mass',
                                          alpha=prpoly[ii + 3] * global_alpha, facecolor='#80D080',
                                          fontsize=fontsize, cbar_ax=None, vmin=0.0, vmax=vmax, linewidth=0,
                                          cax_label='', make_cbar=False, climbars=True, xlabel=xlabel,
                                          ylabel=str(ii + 3))
            else:
                leplt.colored_DOS_plot(eigval, ax[ii], 'mass', alpha=global_alpha, colorV=prpoly[ii + 3],
                                       colormap='CMRmap_r', norm=None, nbins=75, fontsize=fontsize, cbar_ax=cbar_ax,
                                       vmin=0.0, vmax=vmax, linewidth=0.,
                                       make_cbar=True, climbars=True,
                                       xlabel='Oscillation frequency, $\omega$',
                                       xlabel_pad=12, ylabel=str(ii + 3), ylabel_pad=None,
                                       cax_label=r'$\sum_i |\psi_i|^2$ in $n$-gons',
                                       cbar_labelpad=-28, ticks=None, cbar_nticks=3, cbar_tickfmt='%0.2f',
                                       orientation='vertical', cbar_orientation='horizontal',
                                       invert_xaxis=False, yaxis_tickright=False, yaxis_ticks=None,
                                       ylabel_right=False,
                                       ylabel_rot=90)

            ax[ii].yaxis.set_major_locator(MaxNLocator(nbins=3))

        return fig, ax

    def add_ipr_to_ax(self, ax, ipr=None, alpha=1.0, inverse_PR=True, **kwargs):
        """Add a DOS colored by (Inverse) Participation Ratio of the mass-spring network to an axis

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
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'mass', alpha=alpha,
                                                                colorV=ipr, colormap='viridis', **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()
            print 'len(ipr) = ', len(ipr)
            print 'len(eigval) = ', len(eigval)
            print 'len(xy) = ', len(self.lattice.xy)
            ax, cbar_ax, cbar, n, bins = leplt.colored_DOS_plot(eigval, ax, 'mass', alpha=alpha,
                                                                colorV=1. / ipr, colormap='viridis_r', **kwargs)
        return ax

    def plot_ipr_DOS(self, outdir=None, title=r'$D(\omega)$ for mass-spring network', fname='ipr_mass_hist',
                     alpha=1.0, FSFS=12, show=True, inverse_PR=True, save=True, **kwargs):
        """Plot Inverse Participation Ratio of the mass-spring network

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

            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'mass', alpha=alpha,
                                                                           colorV=ipr, colormap='viridis',
                                                                           linewidth=0,
                                                                           cax_label=r'$p^{-1}$', climbars=True,
                                                                           **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                cmaps.register_colormaps()
            print '1/ipr = ', 1. / ipr
            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'mass', alpha=alpha,
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

    def plot_ill_dos(self, save=True, show=False, dos_ax=None, cbar_ax=None, alpha=1.0, vmin=None, vmax=None,
                     **kwargs):
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

    def plot_DOS(self, outdir=None, title=r'$D(\omega)$ for mass-spring network', fname='eigval_mass_hist',
                 alpha=None, show=True, dos_ax=None, **kwargs):
        """
        outdir : str or None
            File path in which to save the overlay plot of DOS from collection
        **kwargs : keyword arguments for colored_DOS_plot()
        """
        # First make sure all eigvals are accounted for
        eigval, eigvect = self.get_eigval_eigvect(attribute=False)
        if dos_ax is None:
            fig, dos_ax = leplt.initialize_DOS_plot(eigval, 'mass', alpha=alpha)
        else:
            ipr = self.get_ipr()
            leplt.colored_DOS_plot(eigval, dos_ax, 'mass', **kwargs)
        dos_ax.set_title(title)
        if outdir is not None:
            # pkl.dump(fig, file(outdir+fname+'.pkl','w'))
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
            leplt.initialize_eigvect_DOS_header_plot(eigval, self.lattice.xy, sim_type='mass',
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

    def plot_eigval_hist(self, infodir=None, show=False):
        print 'mass: self.eigval = ', self.eigval
        fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'mass', pin=-5000)
        if infodir is not None:
            infodir = le.prepdir(infodir)
            plt.savefig(infodir + 'eigval_mass_hist.png')
        if show:
            plt.show()
        plt.clf()

    def plot_ipr_DOS(self, outdir=None, title=r'$D(\omega)$ for mass-spring network', fname='ipr_mass_hist',
                     alpha=1.0, FSFS=12, show=True, inverse_PR=True, save=True, **kwargs):
        """Plot Inverse Participation Ratio of the mass-spring network

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
                lecmaps.register_colormaps()

            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'mass', alpha=alpha,
                                                                           colorV=ipr, colormap='viridis', linewidth=0,
                                                                           cax_label=r'$p^{-1}$', climbars=True,
                                                                           **kwargs)
        else:
            if 'viridis_r' not in plt.colormaps():
                lecmaps.register_colormaps()
            print 'plot_ipr_DOS(): 1/ipr[0:5] = ', 1. / ipr[0:5]
            fig, DOS_ax, cbar_ax, cbar = leplt.initialize_colored_DOS_plot(eigval, 'mass', alpha=alpha,
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
            Compare the projection of the dispersion onto the omega axis with the DOS of the TwistyLattice
        outdir : str or None
            path to the dir where results are saved, if save==True. If None, uses lp['meshfn'] for hlat.lattice

        Returns
        -------
        omegas, kx, ky
        """
        # Note: the function called below has not been finished
        omegas, kx, ky = mlatkspacefns.infinite_dispersion(self, kx=kx, ky=ky, nkxvals=nkxvals, nkyvals=nkyvals,
                                                           save=save, save_plot=save_plot, title=title,
                                                           outdir=outdir)

        if save_dos_compare:
            mlatkspacefns.compare_dispersion_to_dos(omegas, kx, ky, self, outdir=outdir)

        return omegas, kx, ky

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
        omegas, kxy, vtcs = mlatkspacefns.lowest_mode(self, nkxy=nkxy, save=save, save_plot=save_plot, name=name,
                                                      outdir=outdir, imtype=imtype)
        return omegas, kxy, vtcs


if __name__ == '__main__':
    '''Use the MassLattice class to create a density of states, compute participation ratio binned by polygons,
    or build a lattice with the DOS

    Example usage:
    python mass_lattice_class.py -save_ipr -LT hexagonal -shape hexagon -N 5
    '''
    import lepm.lattice_class as lattice_class

    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create MassLattice class instance,' +
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
    parser.add_argument('-save_prpoly', '--save_prpoly',
                        help='Create dict and hist of excitation participation, grouped by polygonal contributions',
                        action='store_true')
    parser.add_argument('-dcdisorder', '--dcdisorder', help='Construct DOS with delta correlated disorder and view ipr',
                        action='store_true')
    parser.add_argument('-save_ipr', '--save_ipr', help='Load MassLattice and save ipr',
                        action='store_true')
    parser.add_argument('-DOSmovie', '--make_DOSmovie', help='Load the mass lattice and make DOS movie of normal modes',
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
    parser.add_argument('-kkspec', '--kkspec', help='string specifier for kk bond strength matrix',
                        type=str, default='')
    # examples of kkspec:
    #   'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') +
    #      'strong{0:0.3f}'.format(lp['kk']).replace('.', 'p').replace('-', 'n') +\
    #      'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n') --> dicing lattice
    #   'Vpin0p10' --> a specific, single random configuration of potentials
    #   'Vpin0p10theta0p00phi0p00' --> a specific, single random configuration of potentials with twisted BCs

    parser.add_argument('-dice_mlat', '--dice_mlat',
                        help='Create a mlat with bonds that are weakened in a gridlike fashion (bonds crossing '
                             'gridlines are weak)',
                        action='store_true')
    parser.add_argument('-dice_eigval', '--dice_eigval', help='Compute eigvals/vects if dice_mlat is True',
                        action='store_true')
    parser.add_argument('-gridspacing', '--gridspacing', help='Spacing of gridlines for dice_mlat', type=float,
                        default=7.5)
    parser.add_argument('-weakbond_val', '--weakbond_val',
                        help='Spring frequency of bonds intersecting dicing lines for dice_mlat', type=float,
                        default=-0.5)

    # Geometry and physics arguments
    parser.add_argument('-Vfmass', '--V0_mass_flat',
                        help='Half-width of flat distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vfspr', '--V0_spring_flat',
                        help='Half-width of flat distribution of delta-correlated bond disorder',
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
    parser.add_argument('-kk', '--kk', help='Spring frequency', type=str, default='1.0')
    parser.add_argument('-mass', '--mass', help='Pinning frequency', type=str, default='1.0')
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
    dcdisorder = args.V0_mass_flat > 0 or args.V0_spring_flat > 0

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
          'kk': float((args.kk).replace('n', '-').replace('p', '.')),
          'mass': float((args.mass).replace('n', '-').replace('p', '.')),
          'V0_mass_flat': args.V0_mass_flat,
          'V0_spring_flat': args.V0_spring_flat,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'ABDelta': args.ABDelta,
          'thres': args.thres,
          'pinconf': args.pinconf,
          'kkspec': args.kkspec,
          'spreading_time': args.spreading_time,
          'intparam': args.intparam,
          'immobile_boundary': args.immobile_boundary,
          }

    if args.dispersion:
        """Example usage:
        python mass_lattice_class.py -LT hexagonal -shape square -periodic -N 3 -dispersion
        """
        print 'loading/computing infinite dispersion'
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MassLattice(lat, lp)
        mlat.infinite_dispersion(kx=None, ky=None, nkxvals=args.nkx, nkyvals=args.nky, save=False)

    if args.save_prpoly:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        mlat = MassLattice(lat, lp)
        mlat.load()
        # mlat.calc_pr_polygons(check=False)
        mlat.save_prpoly(save_plot=True)

    if args.plot_charge:
        """Sum amplitude of all modes in a supplied band (specified by omegac)

        Example usage:
        python mass_lattice_class.py -LT hexagonal -shape hexagon -N 5 charge
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        lat.get_polygons(attribute=True, save_if_missing=True)
        mlat = MassLattice(lat, lp)
        mlat.load()
        mlat.sum_amplitudes_band_spectrum(omegac=None, deform_difference=True)

    if args.plot_localized_states:
        """Load a periodic lattice from file, provide physics, and plot the modes which are localized, along with
        polygons.

        Example Usage:
        python mass_lattice_class.py -LT randomcent -shape square -N 20 -conf 02 -periodic -illpoly -frange 0.0/3.5
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MassLattice(lat, lp)
        if args.freq_range != '0/0':
            frange = le.string_to_array(args.freq_range, dtype=float)
        else:
            frange = None
        mlat.plot_localized_state_polygons(save=True, frange=frange)

    if args.dice_mlat:
        '''make the kk arrays to load when computing cherns for diced networks
        Example usage:
        python run_series.py -pro mass_lattice_class -opts LT/hexagonal/-N/11/-shape/square/-dice_mlat/-gridspacing/3.0 -var weakbond_val n1.0:0.1:0.05
        '''
        import lepm.line_segments as lsegs
        # Create a mlat with bonds that are weakened in a gridlike fashion (bonds crossing gridlines are weak)
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()

        # Find where bonds cross gridlines
        gridspacing = args.gridspacing
        lp['kkspec'] = 'gridlines{0:0.2f}'.format(gridspacing).replace('.', 'p') + \
                        'strong{0:0.3f}'.format(lp['kk']).replace('.', 'p').replace('-', 'n') +\
                        'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')
        maxval = max(np.max(np.abs(lat.xy[:, 0])), np.max(np.abs(lat.xy[:, 1]))) + 1
        gridright = np.arange(gridspacing, maxval, gridspacing)
        gridleft = -gridright
        gridvals = np.hstack((gridleft, 0, gridright))
        # Draw grid
        gridlinesH = np.array([[-maxval, gridv, maxval, gridv] for gridv in gridvals])
        gridlinesV = np.array([[gridv, -maxval, gridv, maxval] for gridv in gridvals])
        gridsegs = np.vstack((gridlinesH, gridlinesV))
        print 'np.shape(gridlines) = ', np.shape(gridsegs)
        # Make the bond linesegments
        xy = lat.xy
        bondsegs = np.array([[xy[b[0], 0], xy[b[0], 1], xy[b[1], 0], xy[b[1], 1]] for b in lat.BL])
        # get crossings
        print 'gridsegs = ', gridsegs
        does_intersect = lsegs.linesegs_intersect_linesegs(bondsegs, gridsegs)
        tmp_mlat = MassLattice(lat, lp)
        kk = copy.deepcopy(tmp_mlat.kk)
        print 'Altering weak bonds --> ', args.weakbond_val
        for bond in lat.BL[does_intersect]:
            kk[bond[0], np.where(lat.NL[bond[0]] == bond[1])] = args.weakbond_val
        mlat = MassLattice(lat, lp, kk=kk)
        mlat.plot_kk()
        if args.dice_eigval:
            mlat.save_eigval_eigvect()
            mlat.save_DOSmovie()

    if args.dcdisorder:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        print 'loading lattice...'
        lat.load()
        mlat = Lattice(lat, lp)
        # mlat.load()
        print 'Computing eigvals/vects...'
        mlat.get_eigval_eigvect(attribute=True)
        # mlat.save_eigval_eigvect()
        # mlat.save_kk_mass()
        print 'Saving DOS movie...'
        mlat.save_DOSmovie()

    if args.make_DOSmovie:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        print '0: self.lp[V0_mass_flat] = ', lp['V0_mass_flat']
        mlat = MassLattice(lat, lp)
        print 'loading lattice...'
        mlat.load()
        print 'Saving DOS movie...'
        mlat.save_DOSmovie()

    if args.save_ipr:
        '''Example usage
        python run_series.py -pro mass_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-save_ipr -var Vpin 0.1/0.5/1.0/2.0

        python run_series.py -pro mass_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-DOSmovie -var Vpin 0.1/5/1.0/2.0
        '''

        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MassLattice(lat, lp)
        print 'saving ipr...'
        mlat.save_ipr(vmax=8.0, xlim=(0., 10.), cbar_labelpad=15)
        # This would save an image of the ipr-colored DOS, but this is already done in save_ipr if save_images=True
        # mlat.plot_ipr_DOS(save=True, vmax=5)

    if args.load_lattice_resave_physics:
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MassLattice(lat, lp)
        print 'saving eigvals/vects...'
        mlat.save_eigval_eigvect()
        print 'saving ipr...'
        mlat.save_ipr(show=False)
        print 'saving localization...'
        mlat.save_localization(save_images=False)
        # print 'Saving DOS movie...'
        # mlat.save_DOSmovie()

    if args.save_lattice:
        lat = lattice_class.Lattice(lp)
        print 'Building lattice...'
        lat.build()
        mlat = MassLattice(lat, lp, kk='auto', mass='auto')
        print 'Saving mass lattice...'
        mlat.lattice.save()
        print 'Computing eigvals/vects...'
        mlat.eig_vals_vects(attribute=True)
        mlat.save_eigval_eigvect()
        print 'Saving DOS movie...'
        mlat.save_DOSmovie()

    if args.load_calc_ldos:
        """Load a (mass) lattice from file, calc the local density of states and save it
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        mlat = MassLattice(lat, lp)
        mlat.save_ldos()

    if args.localization:
        """Load a periodic lattice from file, provide physics, and seek exponential localization of modes

        Example usage:
        python run_series.py -pro mass_lattice_class -opts LT/hucentroid/-periodic/-NP/20/-localization/-save_eig -var AB 0.0:0.05:1.0
        python run_series.py -pro mass_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-localization/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0
        python run_series.py -pro mass_lattice_class -opts LT/hyperuniform/-periodic/-NP/50/-DOSmovie/-save_eig -var Vpin 0.1/0.5/1.0/2.0/4.0/6.0

        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        extent = 2 * max(np.max(lat.xy[:, 0]), np.max(lat.xy[:, 1]))
        mlat = MassLattice(lat, lp)
        mlat.save_localization(attribute=True, save_images=args.save_images, save_eigvect_eigval=args.save_eig)
        mlat.plot_ill_dos(vmax=4./extent, xlim=(0., 14.), ticks=[0, 2./extent, 4./extent],
                          cbar_ticklabels=[0, r'$2/L$', r'$4/L$'], cbar_labelpad=15)

    if args.plot_matrix:
        """Example usage:
        python mass_lattice_class.py -LT deformed_kagome -N 1 -periodic -plot_matrix -x1 -0.1 -x2 0.1 -x3 0.1
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        mlat = MassLattice(lat, lp)
        # glat.get_eigval_eigvect(attribute=True)
        # glat.save_eigval_eigvect(attribute=True, save_png=True)
        matrix = mlat.get_matrix()
        print 'here'
        le.plot_complex_matrix(matrix, show=False, close=False)
        print 'here'
        plt.title('Dynamical matrix:' + lp['meshfn_exten'])
        outfn = dio.prepdir(lp['meshfn']) + 'matrix_' + lp['meshfn_exten'] + '.png'
        print 'saving ' + outfn
        plt.savefig(outfn)
        aa, bb = np.linalg.eig(matrix)
        print 'eigvals =', aa
        mlat.get_eigval(attribute=True)
        print 'mlat.eigvals = ', mlat.eigval

    if args.lowest_mode:
        """Example usage:
        python mass_lattice_class.py -LT deformed_kagome -N 1 -periodic -lowest_mode -x1 -0.1 -x2 0.1 -x3 0.1
        """
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load(check=lp['check'])
        mlat = MassLattice(lat, lp)
        mlat.lowest_mode(nkxy=args.nkx, save=False)


