import numpy as np
import lepm.lattice_elasticity as le
import lepm.structure as lestructure
import lepm.plotting.colormaps as cmaps
import lepm.plotting.science_plot_style as sps
import lepm.kitaev.kitaev_functions as kfns
import lepm.kitaev.kitaev_chern_class as kitaev_chern_class
import lepm.kitaev.kitaev_collection_functions as kcollfns
import lepm.gyro_lattice_functions as glatfns
import lepm.gyro_collection as gyro_collection
import socket
import matplotlib.pyplot as plt
import argparse
import lepm.lattice_class as lattice_class
import lepm.gyro_lattice_class as gyro_lattice_class
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as cmaps
import lepm.plotting.kitaev_plotting_functions as kpfns
import lepm.plotting.kitaev_collection_plotting_functions as kcollpfns
import lepm.stringformat as sf
import glob
import copy
import lepm.dataio as dio
import pdb
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl
import os

'''
Collections of Chern number measurements made via the kitaev realspace method for gyro+spring networks.
Note: methods with 'vary_glatparam' cycle through the same geometric lattice with different physics on that lattice,
      while methods with 'vary_lpparam' cycle through different geometries of networks.

Example usage:
for march meeting 2017:
python ./kitaev/kitaev_collection.py -LT hexkagcframe -alph 0.3 -varyloc -N 30 -shape circle
'''


class KitaevCollection:
    """Create a collection of chern measurements for gyroscopic spring networks.
    Attributes of the class can exist in memory, on hard disk, or both.
    self.cherns is a list of tuples of dicts: self.cherns = [chern1, chern2,...]
    where chern is a class with attributes cp, chern_finsize array, params_regs dict.
        cp : dict
            keys : str
                'meshfn', 'omegac', 'poly_offset', 'ksize_frac_arr', 'regalph', 'regbeta', 'reggamma', 'polyT',
                'poly_offset'
        chern_finsize : len(ksize) x 5 float array
            contains [Nreg1, ksize_frac, ksize, ksys_size (note this is 2*NP_summed), ksys_frac, nu for Chern calculation]
        params_regs : dict
            a nested dictionary with key,value pairs given by
                keys : str
                    each key is a string element of ksize: '{0:0.3f}'.format(ksize)
                values : dict
                    dictionary with key,value pairs given by
                        reg1_xy : int list
                            list of self.gyro_lattice.lattice.xy indices in reg1 of kitaev sum region
                        reg2_xy : int list
                            list of self.gyro_lattice.lattice.xy indices in reg2 of kitaev sum region
                        reg3_xy : int list
                            list of self.gyro_lattice.lattice.xy indices in reg3 of kitaev sum region
                        polygon1 : #vertices x 2 float array
                            coordinates of vertices of polygon enclosing reg1
                        polygon2 : #vertices x 2 float array
                            coordinates of vertices of polygon enclosing reg2
                        polygon3 : #vertices x 2 float array
                            coordinates of vertices of polygon enclosing reg3
                        reg1 : int list
                            list of self.gyro_lattice.lattice.xy indices in reg1 of kitaev sum region
                        reg2 : int list
                            list of self.gyro_lattice.lattice.xy indices in reg2 of kitaev sum region
                        reg3 : int list
                            list of self.gyro_lattice.lattice.xy indices in reg3 of kitaev sum region

    Attributes
    ----------
    self.gyro_collection : list of gyro_lattice instances
        list of instances of lattice_class.lattice() corresponding to each network
    self.meshfns : list of strings
        string paths to the location of the gyro_lattices in the gyro_collection
    self.cherns : dict
        keys are glat_names (strings), values are lists of KitaevChern instances
    self.glat_names : list of strings
        the names of the gyro_lattices in the chern collection
    """
    def __init__(self, gyro_collection, cp=None):
        """Create an instance of a lattice_collection."""
        self.gyro_collection = gyro_collection
        self.meshfns = gyro_collection.meshfns
        self.cherns = {}
        if cp is None:
            cp = {}
        if 'ksize_frac_arr' not in cp:
            cp['ksize_frac_arr'] = np.arange(0.0, 1.10, 0.01)
        if 'omegac' not in cp:
            cp['omegac'] = np.array([2.25])
        if 'regalph' not in cp:
            cp['regalph'] = np.pi*(11./6.)
            cp['regbeta'] = np.pi*(7./6.)
            cp['reggamma'] = np.pi*0.5
        if 'polyT' not in cp:
            cp['polyT'] = False
        if 'poly_offset' not in cp:
            cp['poly_offset'] = 'none'

        self.cp = cp
        self.glat_names = []
        print 'instantiated KitaevCollection with cp = ', self.cp

    def chern_is_saved(self, glat, cp=None, verbose=False):
        if verbose:
            print '\nChecking if chern is already saved...'
        if cp is None:
            cp = self.cp
        chern = kitaev_chern_class.KitaevChern(glat, cp)
        if glob.glob(chern.cp['cpmeshfn'] + 'chern_finsize.txt'):
            return True
        else:
            return False

    def add_chern(self, glat, cp=None, proj=None, skip_paramsregs=True, attribute_evs=False, verbose=False):
        """Add a KitaevChern instance to the KitaevCollection, and append the glat_name to self.glat_names if not
        already in that list"""
        if cp is None:
            cp = self.cp
        chern = kitaev_chern_class.KitaevChern(glat, cp)
        chern.get_kitaev_chern(proj=proj, skip_paramsregs=skip_paramsregs, attribute_evs=attribute_evs, verbose=verbose)
        glat_name = glat.lp['meshfn'] + glat.lp['meshfn_exten']
        if glat_name not in self.cherns:
            self.cherns[glat_name] = []

        self.cherns[glat_name].append(chern)
        if glat_name not in self.glat_names:
            self.glat_names.append(glat_name)
        return chern

    def get_cherns(self, cp=None, verbose=False, omegac_glat_func=None, omegac_glatparam=None, reverse=False):
        """Retrieve cherns for each glat in self.gyro_collection.gyro_lattices matching the supplied/attributed cp

        Parameters
        ----------
        cp : none or dict
            chern parameters dictionary
        verbose : bool
            Print more statements to commandline out
        omegac_glat_func : None or function
            function to find omegac based on features of the gyro_lattice (for example, a function to get the middle of
            a band gap based on the deformation angle of a lattice). Must be used in conjunction with glatparam used as
            input to determine omegac value.
        glatparam : None or any type (key for lp)
            input for function omegac_glat_func to determine omegac (which is output of omegac_glat_func(glatparam).
        """
        if cp is None:
            cp_orig = self.cp
        else:
            cp_orig = cp

        if reverse:
            glatstodo = self.gyro_collection.gyro_lattices[::-1]
        else:
            glatstodo = self.gyro_collection.gyro_lattices
        for glat in glatstodo:
            cp = copy.deepcopy(cp_orig)
            cp.pop('cpmeshfn', None)
            # print '\n\n\nkitaev_collection: get_cherns: glat = ', glat.lp['meshfn']
            # if verbose:
            print 'Adding chern for glat =', glat.lp['meshfn']

            # The optional omegac_glat_func function can map the current glatparam to a desired cutoff freq --
            # for ex, if the middle of the gap is a function of the gyrolattice parameter that is varying
            if omegac_glat_func is not None:
                    cp['omegac'] = omegac_glat_func(glat.lp[omegac_glatparam])
                    if len(cp['omegac']) > 0:
                        for ii in range(len(cp['omegac'])):
                            cpii = copy.deepcopy(cp)
                            cpii['omegac'] = cp['omegac'][ii]
                            self.add_chern(glat, cpii, verbose=verbose)
                    else:
                        raise RuntimeError('Supplied omegac_glat_func returned empty array')
            else:
                if len(cp['omegac']) > 0:
                    for ii in range(len(cp['omegac'])):
                        cpii = copy.deepcopy(cp)
                        cpii['omegac'] = cp['omegac'][ii]
                        self.add_chern(glat, cpii, verbose=verbose)
                else:
                    self.add_chern(glat, cp=cp, verbose=verbose)

    def calc_cherns_omegac(self, omegacV=None, reverse=False):
        if omegacV is None:
            omegacV = self.cp['omegac']
        else:
            self.cp['omegac'] = omegacV

        if reverse:
            omegacV = omegacV[::-1]

        # Add chern calculation for each omegac value
        print 'going to get/calc/add cherns for glat = ', self.gyro_collection.gyro_lattices
        for glat in self.gyro_collection.gyro_lattices:
            print 'going to get/calc/add cherns for omegac values = ', omegacV
            for omegac in omegacV:
                cp = copy.deepcopy(self.cp)
                print 'omegac = ', omegac
                cp['omegac'] = omegac
                self.add_chern(glat, cp, attribute_evs=True)

        return self.cherns

    def get_cherns_omegac(self, omegacV=None, reverse=False):
        self.calc_cherns_omegac(omegacV=omegacV, reverse=reverse)
        return self.cherns

    def get_maxchern_omegac(self, max_boxfrac=-1., max_boxsize=-1., singleksz_frac=0.5, singleksz=-1., singleksyssz=-1):
        """Considering all omegac in the kcoll, get the extremal chern value obtained for ksize_frac=singleksz_frac

        Parameters
        ----------
        max_boxfrac : float
            If positive, used as fraction of spatial extent of the sample to use as maximum bound over which to search
            for extremal chern. Ignored if max_boxsize, singleksz_frac, singleksz, or singleksyssz are positive.
        max_boxsize : float
            If positive, uses max_boxfrac * spatial extent of the sample as max_boxsize over which to search for
            extremal chern. Ignored if singleksz_frac, singleksz, or singleksyssz are positive
        singleksz_frac : float
            if positive, plots the spatially-resolved chern number for a single kitaev summation region size, closest to
            the supplied value in fraction of the system size. Otherwise, draws many rectangles of different sizes
            for different ksizes. If positive, IGNORES max_boxfrac and max_boxsize arguments.
            Ignored if singleksz or singleksyssz are positive
        singleksz : float
            if positive, plots the spatially-resolved chern number for a single kitaev summation region size, closest to
            the supplied value in actual size. Otherwise, draws many rectangles of different sizes for different ksizes.
            If positive, IGNORES singleksz_frac, max_boxfrac, and max_boxsize arguments
            Ignored if singleksyssz is positive.
        singleksyssz : int
            if positive, obtains extremal chern number over all omegac values for a single kitaev summation region size,
            closest to the supplied value in number of dof (2 x # sites considered).
            Otherwise, uses either singlksz (2st choice) or singleksz_frac (2nd choice) or max_boxsize (3rd) or
            max_boxfrac (4th choice).
            If positive, IGNORES singleksz_frac, and singleksz_frac, max_boxfrac, and max_boxsize arguments
        """
        print 'Obtaining extremal chern over all omegac values'
        nuoutlist = []
        glat_name_list = []
        for glat_name in self.cherns:
            print 'glat_name = ', glat_name
            val = 0.
            # get val when ksys_frac = 0.5
            for chernii in self.cherns[glat_name]:
                # Grab small, medium, and large circles
                nu = chernii.chern_finsize[:, -1]
                if singleksyssz > 0:
                    ksyssize = chernii.chern_finsize[:, 3]
                    ind = np.argmin(np.abs(ksyssize - singleksyssz))
                elif singleksz > 0:
                    ksize = chernii.chern_finsize[:, 2]
                    ind = np.argmin(np.abs(ksize - singleksz))
                elif singleksz_frac > 0:
                    ksize_frac = chernii.chern_finsize[:, 1]
                    ind = np.argmin(np.abs(ksize_frac - singleksz_frac))
                elif max_boxsize > 0:
                    # todo: Consider all ksizes less than max_boxsize
                    raise RuntimeError("Haven't finished this")
                elif max_boxfrac > 0:
                    # todo: Consider all ksizes less than max_boxfrac * spatial extent
                    raise RuntimeError("Haven't finished this")
                else:
                    raise RuntimeError("One argument must be positive in KitaevCollection.get_maxchern_omegac()")

                newnu = nu[ind]
                if np.abs(newnu) > np.abs(val):
                    val = newnu

            nuoutlist.append(val)
            glat_name_list.append(glat_name)

        return glat_name_list, nuoutlist

    def get_cherns_varyloc(self, max_ksize_frac=None, max_ksize=None, step=1., fracsteps=False, locV=None,
                           reverse=False, verbose=False):
        self.calc_cherns_varyloc(max_ksize_frac=max_ksize_frac, max_ksize=max_ksize, step=step,
                                 fracsteps=fracsteps, reverse=reverse, verbose=verbose, locV=locV)
        return self.cherns

    def calc_cherns_varyloc(self, max_ksize_frac=None, max_ksize=None, step=0.5, fracsteps=False, locV=None,
                            reverse=False, verbose=False):
        """Compute chern number for fixed omegac (self.cp['omegac'][0]), but vary the location of the kitaev sum.
        Stop increasing the sum region size when the edge hits the edge of the sample.

        Parameters
        ----------
        max_ksize_frac : float
        max_ksize : float or None
            Maximum real-coordinate-space size of a kitaev sum region. Overrides max_ksize_frac if not None
        step : float
            Half spacing between centers of kitaev regions. If fracsteps=None, this is a real-coordinate-space size.
        fracsteps : bool
            Whether to use the supplied step parameter as a fraction of the maximum extent of the lattice rather than as
            a real-coordinate-space step size
        """
        # prepare xy locations as fraction of system size
        print 'Looping over all gyro lattices...'
        print('--> glats to do:', [key for key in self.gyro_collection.gyro_lattices])
        for glat in self.gyro_collection.gyro_lattices:
            print '\n\nkitaev_collection: Getting cherns for GyroLattice ', glat,  '...'
            print ' of name ', glat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)

            proj = None
            if glat.lp['shape'] == 'square':
                # get extent of the network from Bounding box
                Radius = np.abs(glat.lp['BBox'][0, 0])
            elif glat.lp['shape'] in ['hexagon']:
                # get extent of the network from Bounding box
                Radius = max(np.max(glat.lp['BBox'][:, 0]) - np.min(glat.lp['BBox'][:, 0]),
                             np.max(glat.lp['BBox'][:, 1]) - np.min(glat.lp['BBox'][:, 1]))
            elif glat.lp['shape'] in ['circle']:
                # get extent of the network from Bounding box
                Radius = 0.5 * max(np.max(glat.lp['BBox'][:, 0]) - np.min(glat.lp['BBox'][:, 0]),
                                   np.max(glat.lp['BBox'][:, 1]) - np.min(glat.lp['BBox'][:, 1]))
            else:
                # todo: allow different geometries
                pass
            print 'Radius = ', Radius

            if locV is not None:
                print 'setting locV from argument...'
                xxV = locV
                yyV = locV
            else:
                xxV = np.arange(step, Radius + 2 * step, step)
                xxV = np.hstack((-xxV[::-1], np.array([0]), xxV))
                yyV = copy.deepcopy(xxV)

            if fracsteps:
                xxV = np.arange(-1.0 + step, 1.0, step)
                yyV = np.arange(-1.0 + step, 1.0, step)
                # translate into xy coordinates
                xxV *= Radius
                yyV *= Radius
            print 'xxV = ', xxV

            if reverse:
                xxVtodo = xxV[::-1]
                yyVtodo = yyV
            else:
                xxVtodo = xxV
                yyVtodo = yyV[::-1]

            for yy in yyVtodo:
                for xx in xxVtodo:
                    cp = copy.deepcopy(self.cp)
                    cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                    if max_ksize is not None:
                        maxsz = max(np.max(glat.lattice.xy[:, 0]) - np.min(glat.lattice.xy[:, 0]),
                                    np.max(glat.lattice.xy[:, 1]) - np.min(glat.lattice.xy[:, 1]))
                        cp['ksize_frac_arr'] = cp['ksize_frac_arr'][cp['ksize_frac_arr'] * maxsz < max_ksize]
                        cp['ksize_frac_arr'] = np.hstack((cp['ksize_frac_arr'], np.array([max_ksize / maxsz])))
                        print 'limited ksize_frac_arr to ', cp['ksize_frac_arr']
                    elif max_ksize_frac is not None:
                        cp['ksize_frac_arr'] = cp['ksize_frac_arr'][cp['ksize_frac_arr'] < max_ksize_frac]
                        print 'limited ksize_frac_arr to ', cp['ksize_frac_arr']

                    # Test whether projector should be constructed --> if multiple cherns are computed with same
                    # projector, then its best to compute projector only once.
                    if not self.chern_is_saved(glat, cp=cp) and proj is None:
                        print '\nkcoll: Chern is not saved, computing projector...'
                        print '... for cpmeshfn = ', cp['cpmeshfn']
                        proj = kfns.calc_projector(glat, self.cp['omegac'][0])

                    self.add_chern(glat, cp=cp, proj=proj, skip_paramsregs=True, verbose=verbose)

        return self.cherns

    def calc_cherns_vary_glatparam(self, glatparam, paramV, max_ksize_frac=None, max_ksize=None, reverse=False,
                                 varyloc=False, verbose=False, varyloc_range=5.0, step=1.0):
        """For a single Lattice instance, vary a GyroLattice parameter and compute the Chern number for each value.
        When supplied, a single gyro_lattice instance is required for each physical lattice, but each new glat
        (same lattice, different lp) will be appended to self.gyro_collection.gyro_lattices when lp is updated

        Parameters
        ----------
        glatparam : str key for lp dictionary
            the string specifier for the parameter to change FOR THE SAME LATTICE, for each value in paramV
        paramV : list or 1d numpy array
            the values to assign to glat.lp[glatparam] for each Chern calculation
        varyloc : bool
            Also vary the position of the vertex for each glat with modified glatparam
        varyloc_range : float or int
            If varyloc is True, this is the halfwidth of the window on which to compute the spatially-resolved cherns

        """
        # prepare xy locations as fraction of system size
        print 'Looping over all gyro lattices in the gyro collection. ' + \
              'For each one, cycling through glat lpparams in paramV..'
        print('--> glats to do:', [key for key in self.gyro_collection.gyro_lattices])
        glatstodo = [key for key in self.gyro_collection.gyro_lattices]
        for glat in glatstodo:
            print '\n\nkitaev_collection: Getting cherns for GyroLattice ', glat,  '...'
            print ' of name ', glat.lp['meshfn']
            print 'Removing cpmeshfn from cp...'
            self.cp.pop('cpmeshfn', None)

            proj = None
            if glat.lp['shape'] == 'square':
                # get extent of the network from Bounding box
                Radius = np.abs(glat.lp['BBox'][0, 0])
            elif glat.lp['shape'] in ['hexagon', 'circle']:
                # get extent of the network from Bounding box
                Radius = max(np.max(glat.lp['BBox'][:, 0]) - np.min(glat.lp['BBox'][:, 0]),
                             np.max(glat.lp['BBox'][:, 1]) - np.min(glat.lp['BBox'][:, 1]))
            else:
                # todo: allow different geometries
                pass

            print 'paramV = ', paramV
            if reverse:
                paramtodo = paramV[::-1]
            else:
                paramtodo = paramV

            lp_master = copy.deepcopy(glat.lp)
            lat = copy.deepcopy(glat.lattice)
            for val in paramtodo:
                cp = copy.deepcopy(self.cp)
                lp = copy.deepcopy(lp_master)
                lp[glatparam] = val
                print 'glatparam = ', glatparam
                # print 'lp[OmKspec]=', lp['OmKspec']
                glat = gyro_lattice_class.GyroLattice(lat, lp)
                print 'glat.lp[meshfn]=', glat.lp['meshfn']
                self.gyro_collection.gyro_lattices.append(glat)

                if max_ksize is not None:
                    maxsz = max(np.max(glat.lattice.xy[:, 0]) - np.min(glat.lattice.xy[:, 0]),
                                np.max(glat.lattice.xy[:, 1]) - np.min(glat.lattice.xy[:, 1]))
                    cp['ksize_frac_arr'] = cp['ksize_frac_arr'][cp['ksize_frac_arr'] * maxsz < max_ksize]
                    cp['ksize_frac_arr'] = np.hstack((cp['ksize_frac_arr'], np.array([max_ksize / maxsz])))
                    print 'limited ksize_frac_arr to ', cp['ksize_frac_arr']
                elif max_ksize_frac is not None:
                    cp['ksize_frac_arr'] = cp['ksize_frac_arr'][cp['ksize_frac_arr'] < max_ksize_frac]
                    print 'limited ksize_frac_arr to ', cp['ksize_frac_arr']

                # Take one approach if examining multiple sites (varyloc)
                if varyloc:
                    proj = None
                    cp.pop('cpmeshfn', None)
                    cp_master_vl = copy.deepcopy(cp)
                    # Add one chern instance for each location, given that the projector is computed
                    xxV = np.arange(step, varyloc_range + 2 * step, step)
                    xxV = np.hstack((-xxV[::-1], np.array([0]), xxV))
                    yyV = copy.deepcopy(xxV)
                    print 'xxV = ', xxV

                    if reverse:
                        xxVtodo = xxV[::-1]
                        yyVtodo = yyV[::-1]
                    else:
                        xxVtodo = xxV
                        yyVtodo = yyV

                    for xx in xxVtodo:
                        for yy in yyVtodo:
                            cp = copy.deepcopy(cp_master_vl)
                            cp['poly_offset'] = '{0:03f}'.format(xx) + '/' + '{0:0.3f}'.format(yy)
                            print 'kcoll: varyloc is true: ', xx, ' and ', yy
                            if not self.chern_is_saved(glat, cp=cp) and proj is None:
                                print '\nkcoll: Chern is not saved, computing projector...'
                                print '... for cpmeshfn = ', cp['cpmeshfn']
                                proj = kfns.calc_projector(glat, self.cp['omegac'][0])

                            self.add_chern(glat, cp=cp, proj=proj, skip_paramsregs=True, verbose=verbose)
                else:
                    # Test whether projector should be constructed --> if multiple cherns are computed with same
                    # projector, then its best to compute projector only once.
                    if not self.chern_is_saved(glat, cp=cp):
                        print '\nkcoll: Chern is not saved, computing projector...'
                        print '... for cpmeshfn = ', cp['cpmeshfn']
                        proj = kfns.calc_projector(glat, self.cp['omegac'][0])

                    # Simply add the one chern to the collection
                    self.add_chern(glat, cp=cp, proj=proj, skip_paramsregs=True, verbose=verbose)

        return self.cherns

    def plot_cherns_varyloc(self, title='auto', filename='chern_varyloc', exten='.png',
                            rootdir='auto', outdir=None,
                            max_boxfrac=None, max_boxsize=None,
                            xlabel=None, ylabel=None, step=1.0, fracsteps=False, ax=None, cbar_ax=None,
                            singleksz_frac=-1, singleksz=-1, maxchern=False,
                            save=True, make_cbar=True, colorz=False, dpi=600, rasterized=False):
        """Plot the chern as a function of space for each gyro_lattice examined. Note that this allows to add the
        spatially resolved chern number to an axis with ax and cbar_ax args.

        Parameters
        ----------
        outdir : str
            File path for the image. If specified, then rootdir is ignored
        max_boxfrac : float
            Fraction of spatial extent of the sample to use as maximum bound for kitaev sum
        max_boxsize : float or None
            If None, uses max_boxfrac * spatial extent of the sample as max_boxsize over which to consider cherns
        singleksz_frac : float
            if positive, plots the spatially-resolved chern number for a single kitaev summation region size, closest to
            the supplied value in fraction of the system size. Otherwise, draws many rectangles of different sizes
            for different ksizes. If positive, IGNORES max_boxfrac and max_boxsize arguments
        singleksz : float
            if positive, plots the spatially-resolved chern number for a single kitaev summation region size, closest to
            the supplied value in actual size. Otherwise, draws many rectangles of different sizes for different ksizes.
            If positive, IGNORES singleksz_frac, max_boxfrac, and max_boxsize arguments
        maxchern : bool
            if True, plots the EXTREMAL spatially-resolved chern number for each voxel --> ie the maximum (signed absolute)
            value it reaches. Otherwise, draws many rectangles of different sizes for different ksizes.
            If True, the function IGNORES singleksz, singleksz_frac, max_boxfrac, and max_boxsize arguments
        **kwargs : keyword arguments for plot_cherns_varyloc,
            used only if no cherns are yet stored in self.cherns
        """
        if len(self.cherns) == 0:
            self.calc_cherns_varyloc(step=step, fracsteps=fracsteps)

        if title == 'auto':
            title = r'Spatially-resolved Chern number, with $\omega_c = $' + \
                    '{0:0.2f}'.format(self.cp['omegac'][0]) + r'$\Omega_g$'

        if rootdir is None or rootdir == 'auto':
            rootdir = self.cp['rootdir']

        kcollpfns.plot_cherns_varyloc(self, title=title, filename=filename, rootdir=rootdir, outdir=outdir,
                                      exten=exten, max_boxfrac=max_boxfrac, max_boxsize=max_boxsize,
                                      xlabel=xlabel, ylabel=ylabel, step=step, fracsteps=fracsteps,
                                      ax=ax, cbar_ax=cbar_ax, singleksz_frac=singleksz_frac,
                                      singleksz=singleksz, maxchern=maxchern,
                                      save=save, make_cbar=make_cbar, colorz=colorz, dpi=dpi, rasterized=rasterized)

    def movie_cherns_varyloc(self, title='auto', filename='chern_varyloc', exten='.png',
                             max_boxfrac=None, max_boxsize=None, xlabel=None, ylabel=None,
                             step=1.0, fracsteps=False, framerate=3, **kwargs):
        """Plot the chern as a function of space for each gyro_lattice examined

        Parameters
        ----------
        title : str
            title of the movie
        filename : str
            the name of the files to save
        exten : str (.png, .jpg, etc)
            file type extension
        max_boxfrac : float
            Fraction of spatial extent of the sample to use as maximum bound for kitaev sum
        max_boxsize : float or None
            If None, uses max_boxfrac * spatial extent of the sample asmax_boxsize
        xlabel : str
            label for x axis
        ylabel : str
            label for y axis
        step : float (default=1.0)
            how far apart to sample kregion vertices in varyloc
        fracsteps : bool
        framerate : int
            The framerate at which to save the movie
        kwargs : keyword arguments for calc_cherns_varyloc,
            used only if no cherns are yet stored in self.cherns
        """
        if len(self.cherns) == 0:
            self.calc_cherns_varyloc(step=step, fracsteps=fracsteps, **kwargs)

        if title == 'auto':
            title = r'Spatially-resolved Chern number, with $\omega_c = $' + \
                    '{0:0.2f}'.format(self.cp['omegac'][0]) + r'$\Omega_g$'

        kcollpfns.movie_cherns_varyloc(self, title=title, filename=filename, rootdir=self.cp['rootdir'],
                                       exten=exten, max_boxfrac=max_boxfrac,
                                       max_boxsize=max_boxsize, xlabel=xlabel, ylabel=ylabel,
                                       step=step, fracsteps=fracsteps, framerate=framerate)

    def movie_cherns_varyloc_moving_kregion(self, boxsize=5, title='auto', filename='chern_varyloc', exten='.png',
                                            xlabel=None, ylabel=None,
                                            step=1.0, fracsteps=False, framerate=45, randomloc=False, **kwargs):
        """Plot the chern as a function of space for each gyro_lattice examined

        Parameters
        ----------
        title : str
            title of the movie
        filename : str
            the name of the files to save
        exten : str (.png, .jpg, etc)
            file type extension
        boxsize : float or None
            Size of kitaev summation region to probe at each site. Uses closest value to boxsize for the movie
        xlabel : str
            label for x axis
        ylabel : str
            label for y axis
        step : float (default=1.0)
            how far apart to sample kregion vertices in varyloc
        fracsteps : bool
            Whether to choose the interval between chern calculation kreg locations as a fraction of the measured
            system size
        framerate : int
            The framerate at which to save the movie
        randomloc : bool
            If True, makes movie where the kreg is placed in seemingly random subsequent places, and more frames
            devoted to the first few evaluations
        kwargs : keyword arguments for calc_cherns_varyloc,
            used only if no cherns are yet stored in self.cherns
        """
        if len(self.cherns) == 0:
            self.calc_cherns_varyloc(step=step, fracsteps=fracsteps, **kwargs)

        if title == 'auto':
            title = r'Spatially-resolved Chern number, with $\omega_c = $' + \
                    '{0:0.2f}'.format(self.cp['omegac'][0]) + r'$\Omega_g$'

        kcollpfns.movie_cherns_varyloc_moving_kregion(self, boxsize=boxsize, title=title, filename=filename,
                                                      rootdir=self.cp['rootdir'],
                                                      exten=exten, xlabel=xlabel, ylabel=ylabel,
                                                      step=step, fracsteps=fracsteps, framerate=framerate,
                                                      randomloc=randomloc)

    def plot_cherns_omegac(self, ngrid=None, filename='omegac_overlay', vmin=-1.0, vmax=1.0,
                           title='Chern number calculation', xlabel=r'Fraction of particles in sum',
                           ylabel=r'Cutoff frequency $\omega_c$'):
        """Plot a chern spectrum with accompanying DOS on the side, one for each gyro_lattice considered in the
        KitaevCollection."""
        plt.close('all')
        def_each_n = ngrid is None
        for glat_name in self.cherns:
            # Grab a pointer to the gyro_lattice
            glat = self.cherns[glat_name][0].gyro_lattice

            print 'Opening glat_name = ', glat_name
            if def_each_n:
                ngrid = len(self.cherns[glat_name][0].chern_finsize)

            # Assume all ksize elements are the same size for now
            ksys_frac = self.cherns[glat_name][0].chern_finsize[:, -2]*0.5
            ksys_fracM = np.array([ksys_frac for i in range(len(self.cherns[glat_name]))])
            # print 'ksys_fracM =  ', ksys_fracM

            # Build omegacV
            omegacV = np.zeros(len(self.cherns[glat_name]))
            nuM = np.zeros((len(self.cherns[glat_name]), len(ksys_frac)))
            # print 'nuM = ', nuM
            for ind in range(len(self.cherns[glat_name])):
                print 'ind = ', ind
                cp_ii = self.cherns[glat_name][ind].cp
                omegacV[ind] = cp_ii['omegac']
                nuM[ind, :] = self.cherns[glat_name][ind].chern_finsize[:, -1]
                # print 'nuM = ', nuM

            omegacM = omegacV.reshape(len(nuM), 1) * np.ones_like(nuM)
            print 'omegacM = ', omegacM

            # Save the plot
            glat_cmesh = kfns.get_cmeshfn(self.cherns[glat_name][ind].gyro_lattice.lp,
                                                      rootdir=self.cherns[glat_name][ind].cp['rootdir'])
            print 'saving plot to ' + glat_cmesh + filename
            kpfns.plot_spectrum_with_dos(glat, ksys_fracM, omegacM, nuM, ngrid,
                                         xlabel=xlabel, ylabel=ylabel, title=title)

            plt.savefig(glat_cmesh + filename + '.pdf')
            plt.savefig(glat_cmesh + filename + '.png', dpi=300)
            plt.clf()

    def plot_chern_spectrum_on_axis(self, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, **kwargs):
        return kcollpfns.plot_chern_spectrum_on_axis(self, ax, dos_ax, cbar_ipr_ax, cbar_nu_ax, **kwargs)

    def collect_cherns_vary_lpparam(self, param='percolation_density', omegac_glat_func=None, omegac_glatparam=None,
                                      reverse=False):
        """Plot a 2D colormap with x axis being kitaev sum region size and y axis being the parameter which is varying
        between gyro lattices.
        If there are multiple chern calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all chern calculations on a particular gyro_lattice.

        """
        if self.cherns == {}:
            self.get_cherns(omegac_glat_func=omegac_glat_func, omegac_glatparam=omegac_glatparam, reverse=reverse)

        params = []
        nuList = []
        ksysList = []
        max_nksize = 0
        for glat_name in self.cherns:
            print 'adding params from glat_name = ', glat_name
            first = True
            for kchern in self.cherns[glat_name]:
                if first:
                    # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                    # the chern instance
                    params.append(kchern.gyro_lattice.lp[param])
                    print 'kcoll: added param for varying glat_param: ', param, ' --> ', kchern.gyro_lattice.lp[param]
                    # For now assume that ksize_frac_arr (and regions, etc) are uniform across them all
                    ksys_size = kchern.chern_finsize[:, -2]
                    max_nksize = max(len(ksys_size), max_nksize)
                    nu = kchern.chern_finsize[:, -1]
                    nu = nu.reshape(len(nu), 1)
                    print 'nu = ', nu
                    first = False
                else:
                    # Collate the other Chern calculations
                    if (kchern.chern_finsize[:, -2] == ksys_size).all():
                        params.append(kchern.gyro_lattice.lp[param])
                        nu_tmp = kchern.chern_finsize[:, -1]
                        nu_tmp = nu_tmp.reshape(len(nu_tmp), 1)
                        nu = np.hstack((nu, nu_tmp))
                    else:
                        print 'kcoll: There is a mismatch in ksys_size vectors for glat_name = ', glat_name
                        print '       ...with lp[param] = ', kchern.gyro_lattice.lp[param]
                        print '       ...with cp[cpmeshfn] = ', kchern.cp['cpmeshfn']
                        bad = np.where(np.abs(kchern.chern_finsize[:, -2] - ksys_size) > 1e-3)[0]
                        print 'kcoll: kchern.chern_finsize[bad, :] = ', kchern.chern_finsize[bad, :]
                        print 'kcoll: ksys_size[bad] = ', ksys_size[bad]
                        plt.plot(np.arange(len(ksys_size)), ksys_size, 'bo-')
                        plt.plot(np.arange(len(ksys_size)), kchern.chern_finsize[:, -2], 'r.-')
                        plt.show()
                        raise RuntimeError('ksize_frac_arr must be uniform for all chern ' +\
                                           'measurements associated with a particular gyro_lattice!')

            # If there were multiple, average over them all
            if len(self.cherns[glat_name]) > 1:
                # Average over them all
                nu = np.mean(nu, axis=1)

            ksysList.append(ksys_size)
            nuList.append(nu)

        # Build output array
        print 'params = ', params
        print 'nuList = ', nuList
        print 'ksysList = ', ksysList
        for ii in range(len(params)):
            param = params[ii]
            ksys = ksysList[ii]
            nu = nuList[ii]
            if ii == 0:
                sz_param_nu = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
            else:
                add_array = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
                sz_param_nu = np.vstack((sz_param_nu, add_array))

        return sz_param_nu, max_nksize

    def collect_cherns_vary_glatparam(self, param='ABDelta', omegac_glat_func=None, omegac_glatparam=None,
                                      reverse=False):
        """Plot a 2D colormap with x axis being kitaev sum region size and y axis being the parameter which is varying
        between gyro lattices.
        If there are multiple chern calculations of a particular GyroLattice, average nu over them all.
        The ksize_frac_array need not be the same for all chern calculations on a particular gyro_lattice. Note that
        this may overlook errors in which the frac_arrays are different now.

        """
        if self.cherns == {}:
            self.get_cherns(omegac_glat_func=omegac_glat_func, omegac_glatparam=omegac_glatparam, reverse=reverse)

        params = []
        nuList = []
        ksysList = []
        max_nksize = 0
        for glat_name in self.cherns:
            print 'adding params from glat_name = ', glat_name
            first = True
            for kchern in self.cherns[glat_name]:
                if first:
                    # Grab the param associated with this gyro_lattice, which is stored as an attribute of
                    # the chern instance
                    paramval = kcollfns.retrieve_param_value(kchern.gyro_lattice.lp[param])
                    params.append(paramval)
                    print 'kcoll: added param for varying glat_param: ', param, ' --> ', kchern.gyro_lattice.lp[param]
                    # For now assume that ksize_frac_arr (and regions, etc) are uniform across them all
                    ksys_size = kchern.chern_finsize[:, -2]
                    max_nksize = max(len(ksys_size), max_nksize)
                    nu = kchern.chern_finsize[:, -1]
                    print 'np.shape(nu) = ', np.shape(nu)
                    nuList.append(nu)
                    first = False
                else:
                    # Collate the other Chern calculations
                    if (kchern.chern_finsize[:, -2] == ksys_size).all():
                        params.append(kchern.gyro_lattice.lp[param])
                        nu_tmp = kchern.chern_finsize[:, -1]
                        # print 'kcoll: ksys_size vectors for glat_name = ', glat_name
                        # print '       ...with lp[param] = ', kchern.gyro_lattice.lp[param]
                        # print '       ...with cp[cpmeshfn] = ', kchern.cp['cpmeshfn']
                        # print 'np.shape(nu) = ', np.shape(nu)
                        # print 'np.shape(nu_tmp) = ', np.shape(nu_tmp)
                        nuList.append(nu_tmp)
                    else:
                        show_mismatch = False
                        if show_mismatch:
                            print 'kcoll: There is a mismatch in ksys_size vectors for glat_name = ', glat_name
                            print '       ...with lp[param] = ', kchern.gyro_lattice.lp[param]
                            print '       ...with cp[cpmeshfn] = ', kchern.cp['cpmeshfn']
                            bad = np.where(np.abs(kchern.chern_finsize[:, -2] - ksys_size) > 1e-3)[0]
                            print 'kcoll: kchern.chern_finsize[bad, :] = ', kchern.chern_finsize[bad, :]
                            print 'kcoll: ksys_size[bad] = ', ksys_size[bad]
                            plt.plot(np.arange(len(ksys_size)), ksys_size, 'bo-', label='')
                            plt.plot(np.arange(len(ksys_size)), kchern.chern_finsize[:, -2], 'r.-')
                            print('ksize_frac_arr must be uniform for all chern ' + \
                                  'measurements associated with a particular gyro_lattice!')
                            plt.pause(0.0001)

                        # mismatch_ok = input('Continue? (y/n or 1/0)')
                        # if mismatch_ok in ['y', 1]:
                        params.append(kchern.gyro_lattice.lp[param])
                        nu_tmp = kchern.chern_finsize[:, -1]
                        nuList.append(nu_tmp)
                        # else:
                        #     raise RuntimeError('ksize_frac_arr must be uniform for all chern ' +\
                        #                        'measurements associated with a particular gyro_lattice!')

                ksysList.append(ksys_size)

        # Build output array
        print 'params = ', params
        print 'shape(params) = ', np.shape(params)
        print 'nuList = ', np.shape(nuList)
        print 'ksysList = ', np.shape(ksysList)
        for ii in range(len(params)):
            print 'ii = ', ii
            param = kcollfns.retrieve_param_value(params[ii])
            ksys = ksysList[ii]
            nu = nuList[ii]
            if ii == 0:
                print 'param = ', param
                print 'param * np.ones(len(nu)) = ', np.shape(param * np.ones(len(nu)))
                sz_param_nu = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
            else:
                add_array = np.dstack((param * np.ones(len(nu)), ksys, nu))[0]
                sz_param_nu = np.vstack((sz_param_nu, add_array))

        return sz_param_nu, max_nksize

    def plot_cherns_vary_param(self, param_type='glat', sz_param_nu=None, omegac_glat_func=None, omegac_glatparam=None,
                               reverse=False, param='percolation_density',
                               ngrid=None, title='Chern number calculation', xlabel=r'Fraction of particles in sum',
                               ylabel=None, varyloc=False):
        """Plot a 2D colormap with x axis being kitaev sum region size and y axis being the parameter which is varying
        between gyro lattices.
        If there are multiple chern calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all chern calculations on a particular gyro_lattice.

        Parameters
        ----------
        param_type : str ('glat' or 'lat')
            Whether we are varying a lattice parameter (different lattices) or a gyrolattice parameter (same lattice,
            different physics)
        sz_param_nu : None or ...
        omegac_glat_func : None or python function
            function taking lp[omegac_glatparam] as input and supplying numpy array of omegac values
        omegac_glatparam : None or string
            key for glat.lp[omegac_glatparam], to be used to identify numpy array of omegac values
        reverse : bool
            Compute cherns for GyroLattice instances in self.gyro_collection in reverse order
        param : string
            string for GyroLattice parameter to vary
        varyloc : bool
            Whether many different sites were probed --> in which case we average over all the sites
        """
        if sz_param_nu is None:
            if param_type == 'lat' or param_type == 'lp':
                sz_param_nu, max_nksize = \
                    self.collect_cherns_vary_lpparam(param=param, omegac_glat_func=omegac_glat_func,
                                                     omegac_glatparam=omegac_glatparam, reverse=reverse)
            elif param_type == 'glat':
                sz_param_nu, max_nksize = \
                    self.collect_cherns_vary_glatparam(param=param, omegac_glat_func=omegac_glat_func,
                                                       omegac_glatparam=omegac_glatparam, reverse=reverse)
            else:
                raise RuntimeError("param_type argument passed is not 'glat' or 'lat/lp'")

        # Plot it as colormap
        plt.close('all')

        if ngrid is None:
            ngrid = max(max_nksize, len(self.cherns)) + 7

        ksz = sz_param_nu[:, 1] * 0.5
        print 'ksz = ', ksz
        paramV = sz_param_nu[:, 0]
        nu = sz_param_nu[:, 2]

        # Make figure
        FSFS = 8
        Wfig = 90
        x0 = round(Wfig * 0.15)
        y0 = round(Wfig * 0.1)
        ws = round(Wfig * 0.4)
        hs = ws
        wsDOS = ws * 0.3
        hsDOS = hs
        wscbar = wsDOS
        hscbar = wscbar*0.1
        vspace = 8  # vertical space btwn subplots
        hspace = 8  # horizonl space btwn subplots
        tspace = 10  # space above top figure
        fig = sps.figure_in_mm(Wfig, y0+hs+vspace+hscbar+tspace)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [Wfig*0.5 - ws * 0.5, y0, ws, hs, ''],    # Chern vary glatparam vs ksize
                [Wfig*0.5 - wscbar, y0 + hs + vspace, wscbar*2, hscbar, '']  # cbar for chern
              )]

        if ylabel is None:
            ylabel = param.replace('_', ' ')
        leplt.plot_pcolormesh(ksz, paramV, nu, ngrid, ax=ax[0], cax=ax[1], method='nearest',
                              cmap=cmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1.0, vmax=1.0, title='', xlabel=xlabel, ylabel=ylabel,  cax_label=r'$\nu$',
                              cbar_labelpad=3, cbar_orientation='horizontal', ticks=[-1, 0, 1], fontsize=FSFS)

        # Add title
        ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                       horizontalalignment='center', verticalalignment='center')
        # Match axes
        ax[1].xaxis.set_label_position("top")

        # Save the plot
        if param_type == 'glat':
            outdir = rootdir + 'cherns/kitaev_glatparam/' + param + '/'
            outd_ex = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn_exten']
            # If the parameter name is part of the meshfn_exten, replace its value with XXX in
            # the meshfnexten part of outdir.
            mfestr = glatfns.param2meshfnexten_name(param)
            if mfestr in outd_ex:
                'param is in meshfn_exten, splitting...'
                # split the outdir by the param string
                od_split = outd_ex.split(mfestr)
                # split the second part by the value of the param string and the rest
                od2val_rest = od_split[1].split('_')
                odrest = od_split[1].split(od2val_rest[0])[1]
                print 'odrest = ', odrest
                print 'od2val_rest = ', od2val_rest
                outd_ex = od_split[0] + param + 'XXX'
                outd_ex += odrest
                print 'outd_ex = ', outd_ex
            else:
                outd_ex += '_' + param + 'XXX'
            # pdb.set_trace()
        elif param_type == 'lat':
            outdir = rootdir + 'cherns/kitaev_lpparam/' + param + '/'
            outd_ex = self.cherns[self.cherns.items()[0][0]][0].gyro_lattice.lp['meshfn_exten']

        dio.ensure_dir(outdir)

        fname = outdir + self.gyro_collection.gyro_lattices[0].lp['LatticeTop']
        fname += '_Nks' + str(int(len(self.cherns[self.glat_names[0]][0].cp['ksize_frac_arr'])))
        fname += '_' + self.cherns[self.glat_names[0]][0].cp['shape']
        fname += '_kitaev_' + param + '_Ncoll' + '{0:03d}'.format(len(self.gyro_collection.gyro_lattices))
        fname += outd_ex
        print 'saving to ' + fname + '.png'
        plt.savefig(fname + '.png')
        plt.clf()

    def collect_chernspectra_vary_glatparam(self, param='percolation_density'):
        """Plot a 2D colormap with x axis being kitaev sum region size and y axis being the parameter which is varying
        between gyro lattices.
        If there are multiple chern calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all chern calculations on a particular gyro_lattice.

        """
        if self.cherns == {}:
            self.get_cherns()

        params = []
        nuList = []
        omegaList = []
        max_nomegac = 0
        for glat_name in self.cherns:
            print 'adding params from glat_name = ', glat_name
            nu = []
            omega = []
            # Collate the other Chern calculations
            print 'self.cherns[glat_name] = ', self.cherns[glat_name]
            for kchern in self.cherns[glat_name]:
                # Add this value of the lattice param being varied
                params.append(kchern.gyro_lattice.lp[param])
                print 'params = ', params
                # Use maximum abs() chern value for this omegac (looking over all ksizes used)
                nu_tmp = kchern.chern_finsize[:, -1]
                nu_add = nu_tmp[np.argmax(np.abs(nu_tmp))]
                nu.append(nu_add)
                omega.append(kchern.cp['omegac'])
                print 'len(params) = ', len(params)
                print 'len(nu) = ', len(nu)
                print 'len(omega) = ', len(omega)

            max_nomegac = max(len(omega), max_nomegac)
            omegaList.append(omega)
            nuList.append(nu)
            print 'nuList = ', nuList
            print 'omegaList = ', omegaList
            print 'len(nuList) = ', len(nuList)
            print 'len(omegaList) = ', len(omegaList)

        omegaV = np.array(omegaList).ravel()
        nuV = np.array(nuList).ravel()
        print 'len(nuList) = ', len(nuV)
        print 'len(omegaList) = ', len(omegaV)

        # Build output array
        print 'params = ', params
        param_om_nu = np.dstack((np.array(params), omegaV, nuV))[0]

        return param_om_nu, max_nomegac

    def plot_chernspectra_vary_glatparam(self, param_om_nu=None, param='percolation_density', ngrid=None,
                                         title='Chern number calculation', xlabel=r'$\omega_c/\Omega_g$',
                                         ylabel=None):
        """Plot a 2D colormap with x axis being kitaev sum region size and y axis being the parameter which is varying
        between gyro lattices.
        If there are multiple chern calculations of a particular lattice, average nu over them all.
        The ksize_frac_array must be the same for all chern calculations on a particular gyro_lattice.

        Parameters
        ----------
        """
        if param_om_nu is None:
            param_om_nu, max_nomegac = self.collect_chernspectra_vary_glatparam(param=param)

        # Plot it as heatmap
        plt.close('all')

        if ngrid is None:
            ngrid = max(max_nomegac, len(self.cherns)) + 7

        om = param_om_nu[:, 1]
        paramV = param_om_nu[:, 0]
        nu = param_om_nu[:, 2]

        # Make figure
        FSFS = 8
        Wfig = 90
        x0 = round(Wfig * 0.15)
        y0 = round(Wfig * 0.12)
        ws = round(Wfig * 0.4)
        hs = ws
        wsDOS = ws * 0.3
        hsDOS = hs
        wscbar = wsDOS
        hscbar = wscbar*0.1
        vspace = 8  # vertical space btwn subplots
        hspace = 8  # horizonl space btwn subplots
        tspace = 10  # space above top figure
        fig = sps.figure_in_mm(Wfig, y0 + hs + vspace + hscbar + tspace)
        label_params = dict(size=FSFS, fontweight='normal')
        ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
              for x0, y0, width, height, part in (
                [Wfig*0.5-ws*0.5, y0 , ws, hs, ''],    # Chern vary glatparam vs ksize
                [Wfig*0.5-wscbar, y0 + hs + vspace, wscbar*2, hscbar, '']  # cbar for chern
              )]

        if ylabel is None:
            ylabel = param.replace('_', ' ')
        leplt.plot_pcolormesh(om, paramV, nu, ngrid, ax=ax[0], cax=ax[1], method='nearest',
                              cmap=cmaps.diverging_cmap(250, 10, l=30),
                              vmin=-1.0, vmax=1.0, title='', xlabel=xlabel, ylabel=ylabel,  cax_label=r'$\nu$',
                              cbar_labelpad=3, cbar_orientation='horizontal', ticks=[-1, 0, 1], fontsize=FSFS)

        # Add title
        ax[0].annotate(title, xy=(0.5, .95), xycoords='figure fraction',
                       horizontalalignment='center', verticalalignment='center')
        # Match axes
        ax[1].xaxis.set_label_position("top")

        # Save the plot
        outdir = rootdir + 'cherns/kitaev_glatparam/' + self.cherns + param + '/'
        le.ensure_dir(outdir)
        fname = outdir + 'kiataev_spectra_' + param + '_Ncoll' +\
                '{0:03d}'.format(len(self.gyro_collection.gyro_lattices)) + '.png'
        print 'saving to ', fname
        plt.savefig(fname)
        plt.clf()


if __name__ == '__main__':
    '''Perform an example of using the kitaev_collection class.
    To store visualization of the chern calculations, set save_ims == True.

    Example usage to create chern spectrum:
    python kitaev_collection.py -LT kagome -N 16 -vary_omegac -omegac 1.0:0.1:4.0
    python ./kitaev/kitaev_collection.py -LT accordionkag -N 5 -shape square -vary_omegac -omegac 1.0:.1:4.0 -Nks 110 -intparam 2 -alph 1.0
    python ./kitaev/kitaev_collection.py -LT accordionkag -N 6 -shape square -vary_omegac -omegac 1.0:.1:4.0 -Nks 110 -intparam 2 -alph 0.85

    Example usage to vary glat parameter:
    python kitaev_collection.py -vary_lpparam -conf2 02 -LT kagper_hucent
    python kitaev_collection.py -LT kagper_hucent -perd 0.5 -varyloc -Nks 201 -N 40
    python kitaev_collection.py -vary_lpparam -glatparam alpha_kagframe -LT hex_kagframe -N 11 -ksize_array_frac 0.0:0.001:1.10 -visua
    python kitaev_collection.py -vary_glatparam -glatparam ABDelta -LT hucentroid -N 30
    '''
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Create collection of kitaev chern calculations.')
    # Script options
    parser.add_argument('-vary_omegac', '--calc_cherns_omegac',
                        help='Compute chern number using a range of omegac values for each glat in the collection' +
                             ' -- ie, create a chern spectrum',
                        action='store_true')
    parser.add_argument('-omegac_reverse', '--omegac_reverse',
                        help='When computing chern number for multiple omegac cutoffs, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-varyloc', '--calc_cherns_varyloc',
                        help='Compute chern number using a grid of kitaev region locations for each gyro lattice ' +
                             'in the collection', action='store_true')
    parser.add_argument('-varyloc_movies', '--varyloc_movies', help='Make movies of the varyloc chern calcs',
                        action='store_true')
    parser.add_argument('-varyloc_reverse', '--varyloc_reverse',
                        help='When computing chern number using a grid of locations, reverse the order of computing' +
                             ' on the grid', action='store_true')
    parser.add_argument('-edge_localization', '--edge_localization', help='Measure localization to regs of big grad nu',
                        action='store_true')
    parser.add_argument('-vary_lpparam', '--vary_lpparam',
                        help='Plot chern number calcs for many GyroLattices build on DISTINCT Lattice instances ' +
                             'with a parameter varying between them',
                        action='store_true')
    parser.add_argument('-spectra_vary_glatparam', '--spectra_vary_glatparam',
                        help='Plot chern number spectra (so for range of cutoff frequencies) for many gyrolattices ' + \
                             'with a parameter varying between them vs omegac', action='store_true')
    parser.add_argument('-vary_glatparam', '--vary_glatparam',
                        help='Plot chern number calcs for many GyroLattices build on THE SAME Lattice instance ' +
                             'with a parameter varying between them',
                        action='store_true')
    parser.add_argument('-vary_lpparam_varyloc', '--vary_lpparam_varyloc',
                        help='While plotting chern number calcs for many GyroLattices build on THE SAME Lattice ' +
                             'instance with a parameter varying between them, also vary the location of the kitaev ' +
                             'vertex',
                        action='store_true')
    parser.add_argument('-vary_lpparam_vary_glatparam', '--vary_lpparam_vary_glatparam',
                        help='Do multiple chern number calcs for many GyroLattices build on THE SAME Lattice ' +
                             'instance with a parameter varying between them, then also vary which lattice to use, ' +
                             'by varying glatparam',
                        action='store_true')
    parser.add_argument('-chernreg_fps', '--chernreg_fps', help='Framerate for movies of scanning Chern calculation',
                        type=int, default=90)

    # Options for script method choice
    parser.add_argument('-Nks', '--Nks',
                        help='How many kitaev region sizes to sample at each site if -calc_cherns_varyloc',
                        type=int, default=401)
    parser.add_argument('-step', '--step', help='Step size to take while sampling the regions of the network via chern',
                        type=float, default=1.0)
    parser.add_argument('-glatparam', '--glatparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='percolation_density')
    parser.add_argument('-glatparam_reverse', '--glatparam_reverse',
                        help='When computing chern number varying a gyrolattice param, reverse the order of computing',
                        action='store_true')
    parser.add_argument('-lpparam', '--lpparam',
                        help='String specifier for which parameter is varied across gyro lattices in collection',
                        type=str, default='ABDelta')
    parser.add_argument('-paramV', '--paramV',
                        help='Sequence of values to assign to lp[param] if vary_lpparam or vary_glatparam is True',
                        type=str, default='0.0:0.1:2.0')
    parser.add_argument('-paramVdtype', '-paramVdtype',
                        help='The data type for the numpy array formed from paramV',
                        type=str, default='float')
    parser.add_argument('-chern_collection', '--chern_collection',
                        help='Whether to collect cherns by LT and analyze their cherns',
                        action='store_true')
    parser.add_argument('-annulus_chern', '--annulus_chern',
                        help='Whether to collect cherns computed with an annular summation region',
                        action='store_true')
    parser.add_argument('-proj_site_glatparam', '--proj_site_glatparam',
                        help='Compare the projector elements near a point proj_XY for a collection of glats',
                        action='store_true')
    parser.add_argument('-proj_XY', '--proj_XY',
                        help='If args.characterize_projector, calc projector vals as function of dist ' +
                             'relative to the point closest to this location',
                        type=str, default='0.0/0.0')

    # chern parameters
    parser.add_argument('-ksize_frac_array', '--ksize_frac_array',
                        help='Array of fractional sizes to make the kitaev region, specified with /s', type=str,
                        default='0.0:0.01:1.10')
    parser.add_argument('-omegac', '--omegac', help='cutoff frequency for projector, specified as string with /s',
                        type=str, default='2.25')
    parser.add_argument('-regalph', '--regalph', help='largest angle dividing kitaev region',
                        type=float, default=np.pi * (11. / 6.))
    parser.add_argument('-regbeta', '--regbeta', help='middle angle dividing kitaev region',
                        type=float, default=np.pi * (7. / 6.))
    parser.add_argument('-reggamma', '--reggamma', help='smallest angle dividing kitaev region',
                        type=float, default=np.pi * 0.5)
    parser.add_argument('-polyT', '--polyT', help='whether to transpose the kitaev region', action='store_true')
    parser.add_argument('-poly_offset', '--poly_offset', help='coordinates to translate the kitaev region, as string',
                        type=str, default='none')
    parser.add_argument('-basis', '--basis', help='basis for performing kitaev calculation (XY, psi)',
                        type=str, default='XY')
    parser.add_argument('-modsave', '--modsave',
                        help='How often to output an image of the kitaev region and calculation result',
                        type=int, default=40)
    parser.add_argument('-save_ims', '--save_ims', help='Whether to save images of the calculations',
                        action='store_true')
    parser.add_argument('-title', '--title', help='Title of the chern calculation saved images', type=str, default='')

    # Geometry arguments for the lattices to load
    parser.add_argument('-Vpin', '--V0_pin_gauss',
                        help='St.deviation of distribution of delta-correlated pinning disorder',
                        type=float, default=0.0)
    parser.add_argument('-Vspr', '--V0_spring_gauss',
                        help='St.deviation of distribution of delta-correlated bond disorder',
                        type=float, default=0.0)
    parser.add_argument('-pinconf', '--pinconf', help='Pinning distribution realization number',
                        type=int, default=0)
    parser.add_argument('-Omk', '--Omk', help='Spring frequency', type=str, default='-1.0')
    parser.add_argument('-Omg', '--Omg', help='Pinning frequency', type=str, default='-1.0')
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-OmKspec', '--OmKspec', help='string specifier for spring frequencies', type=str, default='')
    parser.add_argument('-dice', '--dice', help='Weaken bonds along a grid in the Lattice instance',
                        action='store_true')
    parser.add_argument('-gridspacing', '--gridspacing', help='Spacing of gridlines for dice_glat', type=float,
                        default=7.5)
    parser.add_argument('-weakbond_val', '--weakbond_val',
                        help='Spring frequency of bonds intersecting dicing lines for dice_glat', type=float,
                        default=-0.5)
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings' +
                                          ' (leave blank to specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load', help='Specify to nonzero int to load a network of a particular size' +
                                                 ' in its entirety, without cropping. Will override NH and NV',
                        type=int, default=0)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=20)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                    'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
    parser.add_argument('-shape', '--shape', help='Shape of the overall mesh geometry', type=str, default='square')
    parser.add_argument('-Nhist', '--Nhist', help='Number of bins for approximating S(k)', type=int, default=50)
    parser.add_argument('-kr_max', '--kr_max', help='Upper bound for magnitude of k in calculating S(k)',
                        type=int, default=30)
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
    parser.add_argument('-cut_z', '--cut_z', help='Declare whether or not to cut bonds to obtain target' +
                                                  'coordination number z', type=bool, default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-perd', '--percolation_density', help='Fraction of vertices to decorate', type=float,
                        default=0.5)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default=1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy', help='Position of single dislocation, if not centered' +
                                                              ' at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')
    
    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-delta', '--delta_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.667')
    parser.add_argument('-phi', '--phi_lattice', help='for hexagonal lattice, measure of deformation in radians/pi',
                        type=str, default='0.000')
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-eta_alph', '--eta_alph', help='parameter for percent system randomized', type=float,
                        default=0.00)
    parser.add_argument('-conf', '--realization_number', help='Lattice realization number', type=int, default=01)
    parser.add_argument('-conf2', '--sub_realization_number', help='Decoration realization number', type=int, default=1)
    parser.add_argument('-intparam', '--intparam',
                        help='Integer-valued parameter for building networks (ex # subdivisions in accordionization)',
                        type=int, default=1)
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
    lattice_type = args.LatticeTop

    # phi = np.pi* args.phi
    # delta = np.pi* args.delta

    strain = 0.00  # initial
    # z = 4.0 #target z
    if lattice_type == 'linear':
        shape = 'line'
    else:
        shape = args.shape

    make_slit = args.make_slit
    # deformed kagome params
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    z = 0.0

    # Define description of the network topology
    if args.LatticeTop == 'iscentroid':
        description = 'voronoized jammed'
    elif args.LatticeTop == 'kagome_isocent':
        description = 'kagomized jammed'
    elif args.LatticeTop == 'hucentroid':
        description = 'voronoized hyperuniform'
    elif args.LatticeTop == 'kagome_hucent':
        description = 'kagomized hyperuniform'
    elif args.LatticeTop == 'kagper_hucent':
        description = 'kagome decoration percolation'

    if socket.gethostname()[0:6] == 'midway':
        rootdir = '/home/npmitchell/scratch-midway/'
        cprootdir = rootdir
    elif socket.gethostname()[0:8] == 'Messiaen' or socket.gethostname()[0:5] == 'cvpn-':
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
        if not glob.glob(cprootdir):
            print 'could not glob cprootdir, changing to research2TB...'
            sys.exit()
            cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
            if not glob.glob(cprootdir):
                cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
                if not glob.glob(cprootdir):
                    if not glob.glob(cprootdir):
                        dio.ensure_dir(cprootdir)
    else:
        rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
        cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
    print 'cprootdir = ', cprootdir

    outdir = rootdir + 'experiments/DOS_scaling/' + args.LatticeTop + '/'
    dio.ensure_dir(outdir)

    dcdisorder = args.V0_pin_gauss > 0 or args.V0_spring_gauss > 0

    lp = {'LatticeTop': args.LatticeTop,
          'shape': shape,
          'NH': NH,
          'NV': NV,
          'NP_load': args.NP_load,
          'rootdir': rootdir,
          'phi_lattice': args.phi_lattice,
          'delta_lattice': args.delta_lattice,
          'theta': args.theta,
          'eta': args.eta,
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
          'origin': np.array([0., 0.]),
          'Omk': float(args.Omk.replace('n', '-').replace('p', '.')),
          'Omg': float(args.Omg.replace('n', '-').replace('p', '.')),
          'V0_pin_gauss': args.V0_pin_gauss,
          'V0_spring_gauss': args.V0_spring_gauss,
          'pinconf': args.pinconf,
          'dcdisorder': dcdisorder,
          'percolation_density': args.percolation_density,
          'viewmethod': False,
          'ABDelta': args.ABDelta,
          'intparam': args.intparam,
          'eta_alph': args.eta_alph,
          }

    if args.OmKspec is not '':
        lp['OmKspec'] = args.OmKspec
    elif args.dice:
        lp['OmKspec'] = 'gridlines{0:0.2f}'.format(args.gridspacing).replace('.', 'p') + \
                        'strong{0:0.3f}'.format(lp['Omk']).replace('.', 'p').replace('-', 'n') + \
                        'weak{0:0.3f}'.format(args.weakbond_val).replace('.', 'p').replace('-', 'n')

    ksize_frac_arr = sf.string_sequence_to_numpy_array(args.ksize_frac_array, dtype=float)

    print 'ksize_frac_arr = ', ksize_frac_arr
    cp = {'ksize_frac_arr': ksize_frac_arr,
          'omegac': sf.string_sequence_to_numpy_array(args.omegac, dtype=float),
          'regalph': args.regalph,
          'regbeta': args.regbeta,
          'reggamma': args.reggamma,
          'shape': args.shape,
          'polyT': args.polyT,
          'poly_offset': args.poly_offset,
          'basis': args.basis,
          'modsave': args.modsave,
          'save_ims': args.save_ims,
          'rootdir': cprootdir
          }

    if args.Nks < 400:
        cp, fps = kfns.translate_Nks(args.Nks, cp)

    print '\n\n cprootdir = ', cprootdir

    if args.title == '':
        title = None
    else:
        print 'title = ', eval("r'" + args.title.replace('_', ' ') + "'")
        titlelist = args.title.split('newline')
        print 'titlelist = ', titlelist
        title = ''
        for tt in titlelist:
            title += eval("r'" + tt.replace('_', ' ') + "'") + '\n'
        print 'title = ', title

        # title = eval("r'" + args.title.replace('_', ' ') + "'")

    if args.calc_cherns_omegac:
        # For a single lattice, collate cherns with varying omegac
        meshfn, xyffind = le.build_meshfn(lp)
        print 'meshfn = ', meshfn
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        try:
            lat.load()
        except IOError:
            lat.build()
            lat.save()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()
        gc.add_gyro_lattice(glat)
        print 'Creating chern collection from gyro_collection...'
        kcoll = KitaevCollection(gc, cp=cp)
        print 'Getting chern calculations with varying omegac:\n omegac=', cp['omegac']
        kcoll.calc_cherns_omegac(reverse=args.omegac_reverse)
        kcoll.plot_cherns_omegac()

    if args.calc_cherns_varyloc:
        """Makes movie of spatially-resolved Chern number

        Example usage:
        python kitaev_collection.py --calc_cherns_varyloc -LT kagper_hucent -perd 0.25 -Nks 211 -N 40
        python kitaev_collection.py --calc_cherns_varyloc -LT kagper_hucent -perd 0.25 -Nks 201 -N 40 -conf 02 -title Spatially-resolved_Chern_number_for_25%_kagomized_amorphous_gyroscopic_network
        python ./kitaev/kitaev_collection.py -varyloc -LT hucent_kagframe -N 20 -alph 0.5 -title Spatially-resolved_Chern_number_for_amorphous_gyroscopic_network
        python ./kitaev/kitaev_collection.py -varyloc -LT hucent_kagcframe -N 20 -shape circle -alph 0.3 -Nks 51 -title Spatially-resolved_Chern_number_for_amorphous_gyroscopic_network
        python ./kitaev/kitaev_collection.py -varyloc -LT hucentroid -N 30 -shape square -Nks 201 -title Spatially-resolved_Chern_number_for_amorphous_gyroscopic_network

        # to run spatially resolved chern calcs with weakened bonds (diced)
        python run_series.py -pro kitaev_collection -opts LT/hexagonal/-N/11/-shape/square/-varyloc/-dice/-gridspacing/7.5 -var weakbond_val n1.00:0.1:0.05
        python run_series.py -pro kitaev_collection -opts LT/hexagonal/-N/11/-shape/square/-varyloc/-dice/-gridspacing/5.0 -var weakbond_val n1.00:0.1:0.05
        """
        # For a single lattice, collate cherns with varying location of the kitaev summation region vertex
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        # try:
        lat.load()
        # except IOError:
        #     lat.build()
        #     lat.save()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()
        gc.add_gyro_lattice(glat)
        print 'Creating chern collection from gyro_collection...'
        cp, fps = kfns.translate_Nks(args.Nks, cp)
        kcoll = KitaevCollection(gc, cp=cp)
        print "kcoll.cp['ksize_frac_arr'] = ", kcoll.cp['ksize_frac_arr']
        print 'Getting chern calculations with varying position, holding omegac constant:\n omegac=', cp['omegac'][0]
        # kcoll.calc_cherns_varyloc(step=args.step, reverse=args.varyloc_reverse)
        kcoll.plot_cherns_varyloc(step=args.step, title=title, singleksz=5.0)
        kcoll.plot_cherns_varyloc(step=args.step, title=title, singleksz=7.5)
        kcoll.plot_cherns_varyloc(step=args.step, title=title, singleksz=10.0)
        kcoll.plot_cherns_varyloc(step=args.step, title=title, singleksz=15.0)
        if args.varyloc_movies:
            # kcoll.movie_cherns_varyloc_moving_kregion(step=args.step, framerate=args.chernreg_fps, boxsize=5.0,
            #                                           randomloc=True)
            kcoll.movie_cherns_varyloc_moving_kregion(step=args.step, framerate=args.chernreg_fps, boxsize=7.5,
                                                      randomloc=True)
            kcoll.movie_cherns_varyloc_moving_kregion(step=args.step, framerate=args.chernreg_fps, boxsize=5.0)
            kcoll.movie_cherns_varyloc_moving_kregion(step=args.step, framerate=args.chernreg_fps, boxsize=7.5)
            # Plot full sequence of kregion sizes as movie
            kcoll.movie_cherns_varyloc(step=args.step, framerate=fps, title=title)

        # Plot full sequence of kregion sizes as still image
        kcoll.plot_cherns_varyloc(step=args.step, title=title)

    if args.vary_glatparam:
        """Vary an lp parameter and, if vary_lpparam_varyloc==True, compute the chern number in a 6x6 grid

        python kitaev_collection.py -vary_glatparam -glatparam ABDelta -LT hucentroid -N 30 -paramV 0.0:0.1:1.0
        # One disorder slice of the Topological Anderson Insulator (TAI)
        python kitaev_collection.py -Vpin 0.1 -vary_glatparam -glatparam ABDelta -LT hexagonal -shape hexagon -N 10 -paramV 0.0:0.1:1.0
        # Many slices in series
        python run_series.py -pro ./kitaev/kitaev_collection -opts vary_glatparam/-glatparam/ABDelta/-LT/hexagonal/-shape/hexagon/-N/6/-paramV/0.0:0.1:1.0 -var Vpin 0:0.1:1.0

        # decreasing connectivity between meshlike regions
        python kitaev_collection.py -vary_glatparam -LT hexagonal -N 11 -glatparam OmKspec -paramVdtype str -paramV gridlines7p50strongn1p00weakn1p00/gridlines7p50strongn1p00weakn0p80/gridlines7p50strongn1p00weakn0p60/gridlines7p50strongn1p00weakn0p40/gridlines7p50strongn1p00weakn0p20/gridlines7p50strongn1p00strongn1p00weakn0p00
        python kitaev_collection.py -vary_glatparam -LT hexagonal -N 11 -glatparam OmKspec -paramVdtype str -paramV n1.0:0.1:0.05
        python kitaev_collection.py -vary_glatparam -LT hexagonal -N 11 -glatparam OmKspec -paramVdtype str -paramV n1.0:0.1:0.05 -gridspacing 3.0
        python kitaev_collection.py -vary_glatparam -LT hexagonal -N 11 -glatparam OmKspec -paramVdtype str -paramV n1.0:0.1:0.05 -gridspacing 5.0
        python kitaev_collection.py -vary_glatparam -LT hexagonal -N 11 -vary_lpparam_varyloc -glatparam OmKspec -paramVdtype str -paramV n1.0:0.1:0.05

        alternatively, to run spatially resolved chern calcs with diced networks
        python run_series.py -pro kitaev_collection -opts LT/hexagonal/-N/14/-shape/square/-varyloc -var OmKspec gridlines7p50strongn1p00weakn1p00/gridlines7p50strongn1p00weakn0p80/gridlines7p50strongn1p00weakn0p60/gridlines7p50strongn1p00weakn0p40/gridlines7p50strongn1p00weakn0p20/gridlines7p50strongn1p00strongn1p00weakn0p00
        """
        # Collate cherns for one lattice with a gyro_lattice parameter that varies between instances of that lattice
        meshfn = le.find_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        lat.load()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()
        gc.add_gyro_lattice(glat)
        print 'Creating chern collection from single-lattice gyro_collection...'
        kcoll = KitaevCollection(gc, cp=cp)
        if args.paramVdtype == 'str':
            # if sf.is_number_sequence(args.paramV):
            #     paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype, corrections=False)
            # else:
            if args.glatparam == 'OmKspec':
                OmKspec_list = glatfns.build_OmKspec_list(args.gridspacing, lp['Omk'], args.paramV)
                print 'OmKspec_list = ', OmKspec_list
                paramV = np.array(OmKspec_list)
            else:
                raise RuntimeError('Need to make exception for this paramV option')
        else:
            paramV = sf.string_sequence_to_numpy_array(args.paramV, dtype=args.paramVdtype)
        kcoll.calc_cherns_vary_glatparam(args.glatparam, paramV, max_ksize_frac=None, max_ksize=None, reverse=False,
                                         verbose=False, varyloc=args.vary_lpparam_varyloc)
        kcoll.plot_cherns_vary_param(param=args.glatparam, param_type='glat', varyloc=args.vary_lpparam_varyloc)

    if args.vary_lpparam:
        """Example usage:
        # topological transition hexagonal --> bowtie
        python kitaev_collection.py -vary_lpparam -LT hexagonal -N 15 -lpparam delta -shape hexagon
        """
        # Collate cherns for many lattices with a gyro_lattice parameter that varies between them
        gc = gyro_collection.GyroCollection()
        if args.lpparam == 'delta' and lp['LatticeTop'] == 'hexagonal':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_' + lp['shape'] + '_delta*_phi0p000_' +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            gc.ensure_all_gyro_lattices()
            kcoll = KitaevCollection(gc, cp=cp)
            midgaps = kcollfns.gap_midpoints_honeycomb
            kcoll.plot_cherns_vary_param(param=args.lpparam, omegac_glat_func=midgaps, omegac_glatparam='delta',
                                             reverse=args.glatparam_reverse)
        if args.lpparam == 'percolation_density':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*perd*_' + 'r{0:02d}_'.format(lp['subconf']) +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            gc.ensure_all_gyro_lattices()
            kcoll = KitaevCollection(gc, cp=cp)
            kcoll.plot_cherns_vary_param()
        # To find the relevant locality size to compute chern number, decorate outside the kitaev region with varying
        # alphas (distances from the center that the frame is decorated).
        if args.lpparam == 'alpha_kagframe':
            # Assume that LatticeTop was specified as either hex_kagframe, hucent_kagframe, or similar
            if lp['LatticeTop'] in ['hex_kagframe', 'hex_kagcframe', 'hucent_kagframe']:
                print 'looking for meshfns of type hex_*frames...'
                gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_square_delta' + le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) + '_alph*0_' +
                              '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))
            elif lp['LatticeTop'] in ['hex_kagperframe']:
                print 'kitaev_collection: looking for meshfns...'
                gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_square_delta' + le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) +
                              '_perd' + le.float2pstr(lp['percolation_density'], ndigits=2) +
                              '_alph*0_' + '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))
            gc.ensure_all_gyro_lattices()
            kcoll = KitaevCollection(gc, cp=cp)
            kcoll.plot_cherns_vary_param(param='alph')

    if args.proj_site_glatparam:
        '''Compare projectors near a particular point in a collection of GyroLattices.
        Example usage:
        python kitaev_collection.py -proj_site_glatparam -glatparam alpha_kagframe -LT hex_kagframe -NH 11 -NV 12
        python kitaev_collection.py -proj_site_glatparam -glatparam alpha_kagframe -LT hex_kagcframe -shape circle -N 30
        '''
        gc = gyro_collection.GyroCollection()
        if args.glatparam == 'alpha_kagframe':
            if lp['LatticeTop'] in ['hex_kagframe', 'hex_kagcframe', 'hucent_kagframe']:
                gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_' + lp['shape'] + '_delta' +
                              le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) + '_alph*0_' +
                              '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))

            elif lp['LatticeTop'] in ['hex_kagperframe']:
                print 'kitaev_collection: looking for meshfns...'
                gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                              args.LatticeTop + '_' + lp['shape'] + '_delta' +
                              le.float2pstr(float(lp['delta_lattice']), ndigits=3) +
                              '_phi' + le.float2pstr(float(lp['phi_lattice']), ndigits=3) +
                              '_perd' + le.float2pstr(lp['percolation_density'], ndigits=2) +
                              '_alph*0_' + '{0:06d}'.format(lp['NH'], ndigits=3) + '_x_{0:06d}'.format(lp['NV']))
        gc.ensure_all_gyro_lattices()
        print 'comparing locality of projectors with omegac = ', cp['omegac'][0]
        proj_XY = sf.string_sequence_to_numpy_array(args.proj_XY, dtype=float)
        kfns.projector_site_vs_dist_glatparam(gc, cp['omegac'][0], proj_XY, plot_mag=True,
                                                          plot_diff=True, save_plt=True, alpha=0.2, reverse_order=True)

    if args.spectra_vary_glatparam:
        # Collate cherns for many lattices with a gyro_lattice parameter that varies between them
        gc = gyro_collection.GyroCollection()
        if args.glatparam == 'percolation_density':
            gc.add_meshfn(rootdir + 'networks/'+args.LatticeTop+'/'+
                          args.LatticeTop+'_square_d*perd*0_' + 'r{0:02d}_'.format(lp['subconf']) +
                          '{0:06d}'.format(lp['NH']) + '_x_{0:06d}'.format(lp['NV']))
            gc.ensure_all_gyro_lattices()
            kcoll = KitaevCollection(gc, cp=cp)
            kcoll.plot_chernspectra_vary_glatparam()

    if args.chern_collection:
        # Collate cherns for many lattices
        gc = gyro_collection.gyro_collection()
        if args.LatticeTop == 'iscentroid':
            gc.add_meshfn('networks/' + args.LatticeTop+'/' +
                          args.LatticeTop + '_square_periodic_hexner_size' + str(args.NP_load) +
                          '_conf*_NP*' + str(args.NP_load))
        elif args.LatticeTop == 'kagome_isocent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop+'_square_periodic_hexner_size' + str(args.NP_load) +
                          '_conf*_NP*'+str(args.NP_load))
        elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_periodic_d*'+'_NP*' + str(args.NP_load))
        elif args.LatticeTop == 'kagper_hucent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*' + '_' + '{0:06d}'.format(NH))

        title = r'$D(\omega)$ for ' + description + ' networks'
        kcoll = KitaevCollection(gc)
        kcoll.plot_cherns_vary_param()

    if args.annulus_chern:
        # Compute the chern number given an annulus with outer diameter larger than system size ksize=1.5
        cp['outerH'] = 1.5

    if args.edge_localization:
        """Look at if eigenvectors have bias to regions with large gradients in calculated chern number.
        This is performed for all available networks matching the meshfn search string (see below).

        Example usage:
        python kitaev_collection.py -edge_localization -LT kagper_hucent -perd 0.25 -N 20 -Nks 31
        python kitaev_collection.py -edge_localization -LT kagper_hucent -perd 0.25 -N 40 -Nks 201
        python kitaev_collection.py -edge_localization -LT kagper_hucent -perd 0.50 -N 40 -Nks 201
        python kitaev_collection.py -edge_localization -LT hucentroid -N 20 -Nks 76
        python kitaev_collection.py -edge_localization -LT kagsplit_hex -NH 18 -NV 10 -Nks 31 -alph 0.500
        """
        # Look at localization of excitation around gradients of computed chern value nu.
        #
        # original code (which compared many different methods) is dumped in
        # lepm/lepm/kitaev_collection_nugrad_analysis_original.py
        check = False
        # Measure localization to where gradients in chern values are high after loading varyloc cherns
        cmaps.register_colormaps()
        # For a single lattice, collate cherns with varying omegac
        meshfn, xyffind = le.build_meshfn(lp)
        lp['meshfn'] = meshfn
        lat = lattice_class.Lattice(lp)
        # try:
        lat.load()
        # except IOError:
        #     lat.build()
        #     lat.save()
        glat = gyro_lattice_class.GyroLattice(lat, lp)
        glat.load()
        gc = gyro_collection.GyroCollection()

        if args.LatticeTop == 'iscentroid':
            # unfinished
            gc.add_meshfn('networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_hexner_size!!!')
        elif args.LatticeTop == 'kagome_isocent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*' + '{0:06d}'.format(NH) + '*_' + '{0:06d}'.format(NV))
        elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d01*' + '{0:06d}'.format(NH) + '*_' + '{0:06d}'.format(NV))
        elif args.LatticeTop == 'kagper_hucent':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_d*perd' + le.float2pstr(args.percolation_density, ndigits=2) +
                          '*_' + '{0:06d}'.format(NH))
            chernsoutdir = rootdir + 'gnupsi2/' + lp['LatticeTop'] + '_square_perd' +\
                           le.float2pstr(args.percolation_density, ndigits=2) +\
                           '_NH' + '{0:06d}'.format(NH) + '_Nks' + str(args.Nks) + '/'
            dio.ensure_dir(chernsoutdir)
        elif args.LatticeTop == 'kagsplit_hex':
            gc.add_meshfn(rootdir + 'networks/' + args.LatticeTop + '/' +
                          args.LatticeTop + '_square_' +
                          'delta' + le.float2pstr(lp['delta']/np.pi, ndigits=3) + '_' +
                          'phi' + le.float2pstr(lp['phi']/np.pi, ndigits=3) + '_' +
                          'alph' + le.float2pstr(lp['alph'], ndigits=2) + '_*' +
                          '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV))
            chernsoutdir = rootdir + 'gnupsi2/' + lp['LatticeTop'] + '_square_' +\
                           'delta' + le.float2pstr(lp['delta']/np.pi, ndigits=3) + '_' +\
                           'phi' + le.float2pstr(lp['phi']/np.pi, ndigits=3) + '_' +\
                           'alph' + le.float2pstr(lp['alph'], ndigits=2) +\
                           '_NH' + '{0:06d}'.format(NH) + '_NV' + '{0:06d}'.format(NV) + \
                           '_Nks' + str(args.Nks) + '/'
            dio.ensure_dir(chernsoutdir)

        # gc.add_gyro_lattice(glat)
        print 'Creating chern collection from gyro_collection...'
        if args.Nks == 211:
            cp['ksize_frac_arr'] = np.arange(0.0, 1.0500001, 0.005)
            fps = 20
        elif args.Nks == 201:
            cp['ksize_frac_arr'] = np.arange(0.0, 0.5000001, 0.0025)
            fps = 20
        elif args.Nks == 121:
            cp['ksize_frac_arr'] = np.arange(0.0, 0.3000001, 0.0025)
            fps = 12
        elif args.Nks == 76:
            cp['ksize_frac_arr'] = np.arange(0.0, 0.76, 0.01)
            fps = 7
        elif args.Nks == 30:
            cp['ksize_frac_arr'] = np.arange(0.0, 0.30, 0.01)
            fps = 3
        elif args.Nks == 31:
            cp['ksize_frac_arr'] = np.arange(0.0, 0.31, 0.01)
            fps = 3

        gc.ensure_all_gyro_lattices()
        kcoll = KitaevCollection(gc, cp=cp)
        print "kcoll.cp['ksize_frac_arr'] = ", kcoll.cp['ksize_frac_arr']
        print 'Getting chern calculations with varying position, holding omegac constant:\n omegac=', cp['omegac'][0]

        # Load or compute all cherns
        print '\nLoading cherns....'
        kcoll.calc_cherns_varyloc(step=args.step, verbose=False)

        # For each glat, for every ksize, compute gradients. Then look at evects.
        plot_gnusum_images = True
        first_glat = True
        print '\n\n\nkcoll.cherns contains:'
        for key in kcoll.cherns:
            print '\n\nkitaev_collection: key = ', key
            print 'kcoll.cherns[key] = ', kcoll.cherns[key]

        for glat_name in kcoll.cherns:
            print('Looking for cherns for: ', glat_name)
            print('kcoll --> now glat = ', kcoll.cherns[glat_name][0].gyro_lattice)
            glat = kcoll.cherns[glat_name][0].gyro_lattice

            outdir = kfns.get_ccpath(cp, kcoll.cherns[glat_name][0].gyro_lattice.lp,
                                                 rootdir='/Users/npmitchell/Dropbox/Soft_Matter/GPU/', method='varyloc')
            dio.ensure_dir(le.prepdir(outdir))
            outdir = le.prepdir(outdir) + glat.lp['LatticeTop']
            outfn = outdir + '_gnu_dict.pkl'
            if glob.glob(outfn):
                print 'Loading collected gnu results instead of computing/loading cherns...'
                with open(outfn, 'rb') as fn:
                    gnudict = pickle.load(fn)

                gnu_psi2 = gnudict['gnu_psi2']
                xyvec = gnudict['xyvec']
                ksizes = gnudict['ksizes']
                kszf = gnudict['kszf']
                nugrids = gnudict['nugrids']
                nugsum = gnudict['nugsum']
                eigval = gnudict['eigval']
            else:
                gnu_psi2, xyvec, ksizes, kszf, nugrids, nugsum, eigval = \
                    kcollfns.nu_gradient_excitation(kcoll, glat_name, outdir=outdir, check=False)

            # Now plot results
            print 'Plotting results...'
            print 'set outdir = ', outdir
            dio.ensure_dir(outdir)

            if plot_gnusum_images:
                # pick a representative index for a later stage of ksize
                repi = int(len(ksizes[0]) * 0.9)
                repi2 = int(len(ksizes[0]) * 0.4)
                plt.clf()
                kcollpfns.plot_nugrid_chern(kcoll.cherns[glat_name][0], glat, xyvec, nugrids, repi, outdir=outdir,
                                            name_exten='')
                plt.clf()
                kcollpfns.plot_gradnusum_chern(kcoll.cherns[glat_name][0], glat, xyvec, nugsum, repi, outdir=outdir,
                                               name_exten='_early')
                plt.clf()
                top = range(len(gnu_psi2[0]))
                kcollpfns.plot_various_gnu_psi2(glat, gnu_psi2, eigval, top, repi, repi2, sz=1, outdir=outdir)

            # Store the curve in memory
            if first_glat:
                ev_vect = copy.deepcopy(np.abs(np.imag(eigval[0:int(0.5 * len(eigval))])))
                gnupsi2_vect = copy.deepcopy(gnu_psi2)
                first_glat = False
                print '\n\n\nfirst glat: np.shape(ev_vect) = ', np.shape(ev_vect)
            else:
                ev_vect = np.hstack((ev_vect, np.abs(np.imag(eigval[0:int(0.5 * len(eigval))].ravel()))))
                gnupsi2_vect = np.hstack((gnupsi2_vect, gnu_psi2))
                print '\n\n\nnp.shape(ev_vect) = ', np.shape(ev_vect)

        # Sort eigval and sort gnupsi2_vect by eigval
        sort = np.argsort(ev_vect)
        ev_v = ev_vect[sort]
        gnupsi2_v = gnupsi2_vect[:, sort]

        # Save result
        print 'saving ev_v and gnupsi2_v as pickles: ' + chernsoutdir + '###.pkl'
        print 'np.shape(ev_v) = ', np.shape(ev_v)
        with open(chernsoutdir + 'ev_v.pkl', "wb") as fn:
            pickle.dump(ev_v, fn)
        with open(chernsoutdir + 'gnupsi2_v.pkl', "wb") as fn:
            pickle.dump(gnupsi2_v, fn)

        # Plot the averaged curves
        for repi_frac in [0.2, 0.4, 0.6, 0.8, 0.9]:
            repi = int(np.shape(gnupsi2_v)[0] * repi_frac)
            plt.clf()
            kcollpfns.plot_kcoll_gnusum_chern(ev_v, gnupsi2_v, outdir=chernsoutdir, fn_exten='', fexten='.png')

        # Average curve to smooth
        avgpt = [5, 10, 20, 50, 100]
        for ap in avgpt:
            # eigval vector averaged
            evva = le.running_mean(ev_v, ap)
            # grad nu psi^2 vector averaged
            gpva = le.running_mean(gnupsi2_v, ap)

            # Save result
            fn_exten = '_avg{0:04d}'.format(ap)
            with open(chernsoutdir + 'ev_v' + fn_exten + '.pkl', "wb") as fn:
                pickle.dump(evva, fn)
            with open(chernsoutdir + 'gnupsi2_v' + fn_exten + '.pkl', "wb") as fn:
                pickle.dump(gpva, fn)

            # Plot the averaged curves
            for repi_frac in [0.2, 0.4, 0.6, 0.8, 0.9]:
                repi = int(np.shape(gpva)[0] * repi_frac)
                kcollpfns.plot_kcoll_gnusum_chern(evva, gpva, outdir=chernsoutdir, fn_exten=fn_exten)
