import numpy as np
import lepm.lattice_elasticity as le
import lepm.dataio as dio
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.science_plot_style as sps
import lepm.stringformat as sf
import socket
import glob
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import argparse
import copy
import sys
import lepm.polygon_functions as polyfns
import lepm.data_handling as dh
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

'''
Compute Chern number measurements made via the kitaev realspace method.
Note that since the gyroscope system has two dof at each site, each region (for ex, reg1) contains indices for the
projector to be summed over that include 2*i and 2*i+1 for particle i, corresponding to that particle's band-projected
x and y displacement. Note that if the psi basis is used (rather than XY basis) then indices will be i and i+N for
particle i, corresponding to that particles left and right circularly polarized displacements.

For example usage, use if name==main clause at end of document.

'''


def translate_Nks(Nks, cp):
    if Nks == 401 or Nks == 110:
        cp['ksize_frac_arr'] = np.arange(0.0, 1.1, 0.01)
        fps = 10
        # raise RuntimeError('Not sure what ksize_frac_arr 401 corresponds to --> find out and finish this case.')
    elif Nks == 211:
        cp['ksize_frac_arr'] = np.arange(0.0, 1.0500001, 0.005)
        fps = 20
    elif Nks == 201:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.5000001, 0.0025)
        fps = 20
    elif Nks == 121:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.3000001, 0.0025)
        fps = 12
    elif Nks == 76:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.76, 0.01)
        fps = 7
    elif Nks == 51:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.51, 0.01)
        fps = 5
    elif Nks == 30:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.30, 0.01)
        fps = 3
    elif Nks == 31:
        cp['ksize_frac_arr'] = np.arange(0.0, 0.31, 0.01)
        fps = 3
    else:
        raise RuntimeError('Nks must take a value from [30, 31, 51, 76, 110, 121, 201, 211, 401]')
    return cp, fps


def calc_projector(gyro_lattice, omegac, attribute=False):
    """Given a gyroscopic network and a cutoff frequency, create the projection operator taking states below that freq
    to zero and above to themselves

    Parameters
    ----------
    gyro_lattice : GyroLattice instance
        The network on which to find the projector
    omegac : float
        cutoff frequency

    Returns
    -------
    proj : len(gyro_lattice.lattice.xy)*2 x len(gyro_lattice.lattice.xy)*2 complex array
        projection operator
    """
    # print '\n\n\nkfns: gyro_lattice.eigval= ', gyro_lattice.eigval, '\n\n\n\n'
    eigval, eigvect = gyro_lattice.get_eigval_eigvect(attribute=attribute)

    U = eigvect.transpose()
    U1 = la.inv(U)
    MM = np.zeros((len(U), len(U)), dtype=complex)

    for ii in range(len(eigval)):
        if np.imag(eigval[ii]) > omegac:
            MM[ii, ii] = 1.0 + 0.0*1j

    # MM = copy.deepcopy(D)
    # MM[MM.imag < omegac] = 0
    # MM[MM.imag > omegac] = 1
    # le.plot_complex_matrix(MM, show=True)
    # sys.exit()
    proj = np.dot(U, np.dot(MM, U1))
    return proj


def characterize_projector(proj, gyro_lattice, index_dist=False, outdir=None, check=False):
    """
    Parameters
    ----------
    proj : 2*NP x 2*NP complex array
        projection operator
    index_dist : bool
        Plot the distances based just on the indices as well, not on physical distances
    outdir : str or None
        If not None, saves dists and magproj as pickles in outdir
    check : bool
        Display intermediate results

    Returns
    -------
    dists : NP x NP float array
        Euclidean distances between points. Element i,j is the distance between particle i and j
    magproj : NP x NP float array
        Element i,j gives the magnitude of the projector connecting site i to particle j
    evxymag : 2*NP x NP float array
        Same as magproj, but with x and y components separate. So, element 2*i,j gives the magnitude of the x component
        of the projector connecting site i to the full xy of particle j and element 2*i+1,j gives the
        magnitude of the y component of the projector connecting site i to the full xy of particle j.
    """
    if index_dist:
        for ii in range(len(proj)):
            print 'potting ', ii
            plt.scatter(np.abs(np.arange(len(proj[ii])) - ii), np.abs(proj[ii, :]), color='k', alpha=0.01)
        plt.title('Locality of projector by index')
        plt.xlabel('Index distance $|i-j|$')
        plt.ylabel('$|P_{ij}|$')
        plt.show()

    dists = le.dist_pts(gyro_lattice.lattice.xy, gyro_lattice.lattice.xy)

    # evxymag has dims len(evect) x NP
    evxymag = np.array([np.sqrt(np.abs(proj[:, ind])**2 + np.abs(proj[:, ind + 1])**2) for
                        ind in np.arange(0, len(proj[0]), 2)])
    magproj = np.array([np.sqrt(evxymag[:, 2*ind].ravel()**2 + evxymag[:, 2*ind + 1].ravel()**2) for
                        ind in range(len(dists))])
    if check:
        print 'np.shape(dists) = ', np.shape(dists)
        print 'len(proj[0]) = ', len(proj[0])
        print 'np.shape(evxymag) = ', np.shape(evxymag)
        print 'len(evxymag) =', len(evxymag)
        print 'np.shape(magproj) = ', np.shape(magproj)

    if outdir is not None:
        print 'saving dists and magproj as pickles...'
        # save dists and magproj as pkl
        with open(outdir + gyro_lattice.lp['LatticeTop'] + "_dists.pkl", "wb") as fn:
            pickle.dump(dists, fn)
        with open(outdir + gyro_lattice.lp['LatticeTop'] + "_magproj.pkl", "wb") as fn:
            pickle.dump(magproj, fn)

    return dists, magproj, evxymag


def evxyproj2magproj(evxyproj):
    """Convert evxyproj (piece of projector associated with a particle to magproj (magnitude of projector elements
    associated with that particle.

    Parameters
    ----------
    evxyproj :

    Returns
    ----------
    magproj :
    """
    tmp = np.sqrt(np.abs(evxyproj[0]).ravel() ** 2 + np.abs(evxyproj[1]).ravel() ** 2)
    magproj = np.array([np.sqrt(tmp[2 * ind].ravel() ** 2 + tmp[2 * ind + 1].ravel() ** 2)
                        for ind in range(int(0.5 *len(tmp)))])
    return magproj


def projector_site_differences(evxyproj0, evxyproj, xy0, xy):
    """Return magnitude of difference between elements of projector and old projector for sites that are the in the same
     location between two networks.

    Parameters
    ----------
    evxyproj0 : 2 x 2*NP complex array
        original projector component associated with a particular particle. First row if for x, second for y
    evxyproj0 : 2 x 2*NP complex array
        new projector component associated with a particular particle. First row if for x, second for y

    Returns
    -------
    magdiff :
    magfdiff :
    same_inds : 2-tuple of #(identical xy points) x 1 float arrays
        Array matching points in xy to pts in xy0. First row indexes new points preserved.
        Second row indexes old points. same_inds[1][ii] has become --> same_inds[0][ii] when going from old to new
    """
    # get magnitude of original for later computing fractional change
    tmp = np.sqrt(np.abs(evxyproj[0]).ravel() ** 2 + np.abs(evxyproj[1]).ravel() ** 2)
    mag0 = np.array([np.sqrt(tmp[2 * ind].ravel() ** 2 + tmp[2 * ind + 1].ravel() ** 2)
                     for ind in range(len(xy0))])

    # find particles that are the same as in the reference network
    same_inds = np.where(le.dist_pts(xy, xy0) == 0)
    # Order them by distance
    current = np.array([[2 * same_inds[0][ii], 2 * same_inds[0][ii] + 1] for ii in range(len(same_inds[0]))])
    current = current.ravel()
    orig = np.array([[2 * same_inds[1][ii], 2 * same_inds[1][ii] + 1] for ii in range(len(same_inds[0]))])
    orig = orig.ravel()

    if len(current) > 0:
        evxydiff = evxyproj[:, current] - evxyproj0[:, orig]
        tmp = np.sqrt(np.abs(evxydiff[0]).ravel() ** 2 + np.abs(evxydiff[1]).ravel() ** 2)
        magdiff = np.array([np.sqrt(tmp[2 * ind].ravel() ** 2 + tmp[2 * ind + 1].ravel() ** 2)
                            for ind in range(len(same_inds[0]))]).ravel()
        # magnitude of fractional difference
        print 'same_inds[0] = ', same_inds[0]
        magfdiff = np.array([magdiff[ii] / mag0[same_inds[1][ii]] for ii in range(len(same_inds[0]))]).ravel()
        print 'magfdiff = ', magfdiff
        return magdiff, magfdiff, same_inds
    else:
        return np.array([]), np.array([])


def projector_site_vs_dist_glatparam(gcoll, omegac, proj_XY, plot_mag=True, plot_diff=False, save_plt=True,
                                     alpha=1.0, maxdistlines=True, reverse_order=False, check=True):
    """
    Compare the projector values at distances relative to a particular site (gyro at location proj_XY).

    Parameters
    ----------
    gcoll : GyroCollection instance
        The collection of gyro_lattices for which to compare projectors as fn of distance from a given site.
    omegac : float
        Cutoff frequency for the projector
    proj_XY : 2 x 1 float numpy array
        The location at which to find the nearest gyro and consider projector elements relative to this site.
    plot : bool
        Whether to plot magproj vs dist for all GyroLattices in gcoll
    check : bool
        Display intermediate results

    Returns
    -------
    dist_list : list of NP x NP float arrays
        Euclidean distances between points. Element i,j is the distance between particle i and j
    proj_list : list of evxyprojs (2 x 2*NP float arrays)
        Each element is like magproj, but with x and y components separate.
        So, element 0,2*j gives the magnitude of the x component of the projector connecting the site in question
        to the x component of particle j and element 1,2*j+1 gives the
        magnitude of the y component of the projector connecting the site in question to the y component of particle j.
    """
    proj_list = []
    dist_list = []
    kk = 0
    if plot_mag:
        # magnitude plot
        mfig, magax, mcbar = leplt.initialize_1panel_cbar_fig()
    if plot_diff:
        # difference plot
        dfig, dax, dcbar = leplt.initialize_1panel_cbar_fig()
        # difference fraction plot
        dffig, dfax, dfcbar = leplt.initialize_1panel_cbar_fig()
        if maxdistlines:
            maxdist = []

    sat = 1
    light = 0.6
    colors = lecmaps.husl_palette(n_colors=len(gcoll.gyro_lattices), s=sat, l=light)
    if reverse_order:
        glat_list = gcoll.gyro_lattices[::-1]
    else:
        glat_list = gcoll.gyro_lattices

    for glat in glat_list:
        proj_ind = ((glat.lattice.xy - proj_XY)[:, 0]**2 + (glat.lattice.xy - proj_XY)[:, 1]**2).argmin()
        outdir = dio.prepdir(glat.lp['meshfn'].replace('networks', 'projectors'))
        outfn = outdir + glat.lp['LatticeTop'] + "_dist_singlept_{0:06d}".format(proj_ind) + ".pkl"

        if check:
            print 'identified proj_ind = ', proj_ind
            newfig = plt.figure()
            glat.lattice.plot_numbered(ax=plt.gca())

        # attempt to load
        if glob.glob(outfn):
            with open(outfn, "rb") as fn:
                dist = pickle.load(fn)

            outfn = outdir + glat.lp['LatticeTop'] + "_evxyproj_singlept_{0:06d}".format(proj_ind) + ".pkl"
            with open(outfn, "rb") as fn:
                evxyproj = pickle.load(fn)
        else:
            # compute dists and evxyproj_proj_ind
            xydiff = glat.lattice.xy - glat.lattice.xy[proj_ind]
            dist = np.sqrt(xydiff[:, 0]**2 + xydiff[:, 1]**2)
            print 'calculating projector...'
            proj = calc_projector(glat, omegac, attribute=False)

            outdir = dio.prepdir(glat.lp['meshfn'].replace('networks', 'projectors'))
            le.ensure_dir(outdir)
            # save dist as pickle
            with open(outfn, "wb") as fn:
                pickle.dump(dist, fn)

            # evxyproj has dims 2 xlen(evect)
            evxyproj = np.dstack((proj[2 * proj_ind, :], proj[2 * proj_ind + 1, :]))[0].T
            # save evxyproj as pickle
            outfn = outdir + glat.lp['LatticeTop'] + "_evxyproj_singlept_{0:06d}".format(proj_ind) + ".pkl"
            with open(outfn, "wb") as fn:
                pickle.dump(evxyproj, fn)

        proj_list.append(evxyproj)
        dist_list.append(dist)

        if plot_mag:
            tmp = np.sqrt(np.abs(evxyproj[0]).ravel()**2 + np.abs(evxyproj[1]).ravel()**2)
            magproj = np.array([np.sqrt(tmp[2*ind].ravel()**2 + tmp[2*ind + 1].ravel()**2) for ind in range(len(dist))])
            magax.scatter(dist, np.log10(magproj), s=1, color=colors[kk], alpha=alpha)

        if plot_diff:
            if kk > 0:
                # find particles that are the same as in the reference network
                same_inds = np.where(le.dist_pts(glat.lattice.xy, xy0) == 0)
                # Order them by distance
                current = np.array([[2*same_inds[0][ii], 2*same_inds[0][ii] + 1] for ii in range(len(same_inds[0]))])
                current = current.ravel()
                orig = np.array([[2 * same_inds[1][ii], 2*same_inds[1][ii] + 1] for ii in range(len(same_inds[0]))])
                orig = orig.ravel()

                # if check:
                #     origfig = plt.figure()
                #     origax = origfig.gca()
                #     [origax, origaxcb] = glat.lattice.plot_numbered(fig=origfig, ax=origax, axis_off=False,
                #                                                     title='Original lattice for proj comparison')
                #     origax.scatter(glat.lattice.xy[same_inds[0], 0], glat.lattice.xy[same_inds[0], 1])
                #     plt.pause(5)
                #     plt.close()
                if len(current) > 0:
                    evxydiff = evxyproj[:, current] - evxyproj0[:, orig]
                    tmp = np.sqrt(np.abs(evxydiff[0]).ravel()**2 + np.abs(evxydiff[1]).ravel()**2)
                    magdiff = np.array([np.sqrt(tmp[2*ind].ravel()**2 + tmp[2*ind + 1].ravel()**2)
                                        for ind in range(len(same_inds[0]))])
                    # magnitude of fractional difference
                    magfdiff = np.array([magdiff[same_inds[0][ii]] / mag0[same_inds[1][ii]]
                                         for ii in range(len(same_inds[0]))])
                    dax.scatter(dist[same_inds[0]], magdiff, s=1, color=colors[kk], alpha=alpha)
                    dfax.scatter(dist[same_inds[0]], magfdiff, s=1, color=colors[kk], alpha=alpha)
                    if maxdistlines:
                        maxdist.append(np.max(dist[same_inds[0]]))

            else:
                origfig = plt.figure()
                origax = origfig.gca()
                [origax, origaxcb] = glat.lattice.plot_numbered(fig=origfig, ax=origax, axis_off=False,
                                                                title='Original lattice for proj comparison')
                origfig.show()
                plt.close()
                evxyproj0 = copy.deepcopy(evxyproj)
                xy0 = copy.deepcopy(glat.lattice.xy)
                tmp = np.sqrt(np.abs(evxyproj[0]).ravel()**2 + np.abs(evxyproj[1]).ravel()**2)
                mag0 = np.array([np.sqrt(tmp[2*ind].ravel()**2 + tmp[2*ind + 1].ravel()**2)
                                 for ind in range(len(dist))])

        kk += 1

    # Save plots
    outsplit = dio.prepdir(glat.lp['meshfn'].replace('networks', 'projectors')).split('/')
    outdir = '/'
    for sdir in outsplit[0:-2]:
        outdir += sdir + '/'
    outdir += '/'

    if plot_mag:
        if maxdistlines and plot_diff:
            ylims = magax.get_ylim()
            print 'ylims = ', ylims
            ii = 1
            for dd in maxdist:
                magax.plot([dd, dd], np.array([ylims[0], ylims[1]]), '-', color=colors[ii])
                ii += 1

        magax.set_title('Magnitude of projector vs distance')
        magax.set_xlabel(r'$|\mathbf{x}_i - \mathbf{x}_0|$')
        magax.set_ylabel(r'$|P_{i0}|$')
        # make a scalar mappable for colorbar'
        husl_cmap = lecmaps.husl_cmap(s=sat, l=light)
        sm = plt.cm.ScalarMappable(cmap=husl_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, cax=mcbar, ticks=[0, 1])
        cbar.ax.set_ylabel(r'$\alpha$', rotation=0)
        if save_plt:
            print 'kfns: saving magnitude comparison plot for gcoll (usually run from kitaev_collection)...'
            mfig.savefig(outdir + glat.lp['LatticeTop'] + "_magproj_singlept.png", dpi=300)
        else:
            plt.show()
    if plot_diff:
        dax.set_title('Projector differences')
        dax.set_xlabel(r'$|\mathbf{x}_i - \mathbf{x}_0|$')
        dax.set_ylabel(r'$|\Delta P_{i0}|$')
        # make a scalar mappable for colorbar'
        husl_cmap = lecmaps.husl_cmap(s=sat, l=light)
        sm = plt.cm.ScalarMappable(cmap=husl_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = plt.colorbar(sm, cax=dcbar, ticks=[0, 1])
        cbar.ax.set_ylabel(r'$\alpha$', rotation=0)
        if maxdistlines:
            # Grab ylimits for setting after adding lines
            ylims = dax.get_ylim()
            df_ylims = dfax.get_ylim()
            ii = 1
            for dd in maxdist:
                dax.plot([dd, dd], np.array([-0.05, -0.005]), '-', color=colors[ii])
                dfax.plot([dd, dd], np.array([df_ylims[0], -0.01]), '-', color=colors[ii])
                ii += 1
            dax.set_ylim(-0.05, ylims[1])
            dfax.set_ylim(df_ylims[0], df_ylims[1])
        # now do fractional difference magnitude plot
        dfax.set_title('Fractional projector differences')
        dfax.set_xlabel(r'$|\mathbf{x}_i - \mathbf{x}_0|$')
        dfax.set_ylabel(r'$|\Delta P_{i0}|/|P_{i0}|$')
        # make a scalar mappable for colorbar'
        cbar = plt.colorbar(sm, cax=dfcbar, ticks=[0, 1])
        cbar.ax.set_ylabel(r'$\alpha$', rotation=0)
        if save_plt:
            print 'kfns: saving diff plot for gcoll projecctors (usually run from kitaev_collection)...'
            dfig.savefig(outdir + glat.lp['LatticeTop'] + "_diffproj_singlept.png", dpi=300)
            dffig.savefig(outdir + glat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff.png", dpi=300)
            dfax.set_ylim(-0.1, 1)
            dffig.savefig(outdir + glat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff_zoom.png", dpi=300)
            dfax.set_ylim(-0.02, 0.1)
            dffig.savefig(outdir + glat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff_extrazoom.png", dpi=300)
        elif not plot_mag:
            plt.show()

    return dist_list, proj_list


def calc_kitaev_chern(gyro_lattice, cp, pp=None, check=False, contributions=False, verbose=False, vis_exten='.png',
                      contrib_exten='.pdf', attribute_evs=False, title=None):
    """Compute the chern number for a gyro_lattice

    Parameters
    ----------
    eigval : 2NP x 1 complex array
        Eigenvalues of the system
    eigvect : 2NP x 2NP complex array
        Eigenvectors of the system
    pp : len(gyro_lattice.lattice.xy)*2 x len(gyro_lattice.lattice.xy)*2 complex array (optional)
        projection operator, if already calculated previously. If None, this function calculates this.
    check : bool (optional)
        Display intermediate results
    contributions : bool (optional)
        Compute the contribution of each individual particle to the chern result for the given summation regions
    verbose : bool (optional)
        Print more output on command line
    vis_exten : str ('.png', '.pdf', '.jpg', etc, default = '.png')
        Extension for the plotted output, if cp['save_ims'] == True
    contrib_exten : str ('.png', '.pdf', '.jpg', etc, default = '.pdf')
        Extension for the plotted contributions of each particle, if cp['save_ims'] == True and contributions==True
    attribute_evs : bool (optional)
        Attribute the eigval and eigvect to the GyroLattice instance gyro_lattice, if projector pp is not supplied

    Returns
    -------
    chern_finsize : len(cp['ksize_frac_arr']) x 6 complex array
        np.dstack((Nreg1V, ksize_frac_arr, ksize_V, ksys_sizeV, ksys_fracV, nuV))[0]
    params_regs : dict
        For each kitaev region size (ksize), there is a key '{0:0.3f}'.format(ksize) and value pair, of the form
        params_regs['{0:0.3f}'.format(ksize)] = {'reg1': reg1, 'reg2': reg2, 'reg3': reg3,
                                                 'polygon1': polygon1, 'polygon2': polygon2, 'polygon3': polygon3,
                                                 'reg1_xy': reg1_xy, 'reg2_xy': reg2_xy, 'reg3_xy': reg3_xy}
    contribs : dict or None
        If contributions == True, contribs is a dictionary with values storing the contributions to the chern result for
        each particle in reg1, reg2, and reg3. Contribs has keys which are '{0:0.3f}'.format(ksize) for each ksize, and
        each value of contribs['{0:0.3f}'.format(ksize)] is itself a dictionary with keys 'reg1', 'reg2', 'reg3' and
        values as the contributions of each particle, for particles indexed by reg1, 2, 3.
        ie, contribs['{0:0.3f}'.format(ksize)] = {'reg1': cb1, 'reg2': cb2, 'reg3': cb3}
        Here cb1,2,3 are (# particles in region n) x 1 complex arrays -- contributions of each particle in each region
        to the total result (when summed over the other two regions)
    """
    save_ims = cp['save_ims']
    modsave = cp['modsave']
    shape = cp['shape']
    ksize_frac_arr = cp['ksize_frac_arr']
    omegac = cp['omegac']
    if save_ims:
        # Register colormaps
        lecmaps.register_colormaps()
        imagedir = cp['cpmeshfn'] + 'visualization/'
        le.ensure_dir(imagedir)

    xy = gyro_lattice.lattice.xy
    NP = len(xy)

    if pp is None:
        print 'Computing projector...'
        pp = calc_projector(gyro_lattice, omegac, attribute=attribute_evs)

    # Initialize empty region index arrays for speedup by comparing with prev iteration
    reg1 = np.array([])
    reg2 = np.array([])
    reg3 = np.array([])
    nu = 0.0 + 0.0 * 1j
    epskick = 0.001 * np.random.rand(len(xy), 2)

    # Preallocate arrays
    nuV = np.zeros(len(ksize_frac_arr))
    Nreg1V = np.zeros(len(ksize_frac_arr))
    ksize_V = np.zeros(len(ksize_frac_arr))
    ksys_sizeV = np.zeros(len(ksize_frac_arr))
    ksys_fracV = np.zeros(len(ksize_frac_arr))
    params_regs = {}
    if contributions:
        contribs = {}
    else:
        contribs = None

    # Get max(width, height) of network
    maxsz = max(np.max(xy[:, 0]) - np.min(xy[:, 0]), np.max(xy[:, 1]) - np.min(xy[:, 1]))

    # If we want to look at individual contributions from each gyro, find h first
    if contributions and NP < 500:
        method = '2current'
        print 'NP = ', NP
        print 'constructing 2-current h_ijk...'
        hh = np.einsum('jk,kl,lj->jkl', pp, pp, pp) - np.einsum('jl,lk,kj->jkl', pp, pp, pp)
        # hh = np.zeros((len(pp), len(pp), len(pp)), dtype=complex)
        # for j in range(len(pp)):
        #     for k in range(len(pp)):
        #         for l in range(len(pp)):
        #             hh[j, k, l] = pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j]
        hh *= 12 * np.pi * 1j
    else:
        print 'skipping 2-current h_ijk to do calc piece by piece (projector method)...'
        method = 'projector'

    jj = 0
    # for each ksize_frac, perform sum
    for kk in range(len(ksize_frac_arr)):
        ksize_frac = ksize_frac_arr[kk]
        ksize = ksize_frac * maxsz
        if verbose:
            print 'ksize = ', ksize

        if 'delta' in gyro_lattice.lattice.lp:
            # Note: by convention, lp['delta'] is in radians, whereas lp['delta_lattice'] is in radians/pi
            delta = gyro_lattice.lattice.lp['delta']/np.pi
        else:
            if gyro_lattice.lattice.lp['delta_lattice'] in ['0.667', '0p667']:
                delta = 2./3.
            else:
                if isinstance(gyro_lattice.lattice.lp['delta_lattice'], str):
                    delta = float(gyro_lattice.lattice.lp['delta_lattice'].replace('p', '.'))
                else:
                    delta = float(gyro_lattice.lattice.lp['delta_lattice'])

        polygon1, polygon2, polygon3 = get_kitaev_polygons(shape, cp['regalph'], cp['regbeta'], cp['reggamma'],
                                                           ksize, delta_pi=delta, outerH=cp['outerH'])
        if cp['polyT']:
            polygon1 = np.fliplr(polygon1)
            polygon2_tmp = np.fliplr(polygon3)
            polygon3 = np.fliplr(polygon2)
            polygon2 = polygon2_tmp

        if cp['poly_offset'] != 'none' and cp['poly_offset'] is not None:
            if '/' in cp['poly_offset']:
                splitpo = cp['poly_offset'].split('/')
            else:
                splitpo = cp['poly_offset'].split('_')
            # print 'split_po = ', splitpo
            poly_offset = np.array([sf.str2float(splitpo[0]), sf.str2float(splitpo[1])])
            polygon1 += poly_offset
            polygon2 += poly_offset
            polygon3 += poly_offset

        # Save the previous reg1,2,3
        r1old = reg1
        r2old = reg2
        r3old = reg3

        reg1_xy = dh.inds_in_polygon(xy + epskick, polygon1)
        reg2_xy = dh.inds_in_polygon(xy + epskick, polygon2)
        reg3_xy = dh.inds_in_polygon(xy + epskick, polygon3)

        if cp['basis'] == 'XY':
            reg1 = np.sort(np.vstack((2 * reg1_xy, 2 * reg1_xy + 1)).ravel())
            reg2 = np.sort(np.vstack((2 * reg2_xy, 2 * reg2_xy + 1)).ravel())
            reg3 = np.sort(np.vstack((2 * reg3_xy, 2 * reg3_xy + 1)).ravel())
        elif cp['basis'] == 'psi':
            if verbose:
                print 'stacking regions with right-moving selves...'
            reg1 = np.sort(np.vstack((reg1, NP+reg1)).ravel())
            reg2 = np.sort(np.vstack((reg2, NP+reg2)).ravel())
            reg3 = np.sort(np.vstack((reg3, NP+reg3)).ravel())

        if contributions:
            if method == '2current':
                nu, [cb1, cb2, cb3] = sum_kitaev_with_contributions(reg1, reg2, reg3, r1old, r2old, r3old,
                                                                    hh, nu, verbose=False)
            else:
                nu, [cb1, cb2, cb3] = sum_kitaev_with_contributions_projector(reg1, reg2, reg3, r1old, r2old, r3old,
                                                                              pp, nu, verbose=False)
            contribs['{0:0.3f}'.format(ksize)] = {'reg1': cb1, 'reg2': cb2, 'reg3': cb3}
        else:
            nu = sum_kitaev_projector(reg1, reg2, reg3, r1old, r2old, r3old, pp, nu, verbose=verbose)

        # print 'nu = ', nu
        nuV[kk] = np.real(nu)
        Nreg1V[kk] = len(reg1)
        ksize_V[kk] = ksize
        ksys_sizeV[kk] = len(reg1) + len(reg2) + len(reg3)
        # ksys_fracV is the fractional number of particles in the kitaev sum wrt the total
        ksys_fracV[kk] = ksys_sizeV[kk]/len(xy)

        # Save regions
        params_regs['{0:0.3f}'.format(ksize)] = {'reg1': reg1, 'reg2': reg2, 'reg3': reg3,
                                                 'polygon1': polygon1, 'polygon2': polygon2, 'polygon3': polygon3,
                                                 'reg1_xy': reg1_xy, 'reg2_xy': reg2_xy, 'reg3_xy': reg3_xy}
        if save_ims and (kk % modsave == 0 or kk == (len(ksize_frac_arr)-1)):
            plt.close('all')
            filename = 'division_lattice_regions_{0:06d}'.format(jj) + vis_exten
            # title = r'Division of lattice: $\nu = ${0:0.3f}'.format(nu.real)
            # Commented out: plot just the regs and the title
            # plot_chern_realspace(gyro_lattice, reg1_xy, reg2_xy, reg3_xy, polygon1, polygon2, polygon3,
            #                     ax=None, outdir=imagedir, filename=filename, title=title, check=check)
            # New way: plot the regs with title plus curve of nu vs ksize
            plot_chern_realspace_2panel(gyro_lattice, ksys_fracV, nuV, kk, reg1_xy, reg2_xy, reg3_xy,
                                        polygon1, polygon2, polygon3,
                                        outdir=imagedir, filename=filename, title=title, check=check)

            if contributions:
                # Save plot of contributions from individual gyroscopes
                plt.close('all')
                filename = 'contributions_ksize{0:0.3f}'.format(ksize_frac) + contrib_exten
                plot_chern_contributions(gyro_lattice, reg1, reg2, reg3, cb1, cb2, cb3, polygon1, polygon2, polygon3,
                                         basis=cp['basis'], outdir=imagedir, filename=filename)

            jj += 1

    if save_ims:
        movname = cp['cpmeshfn'] + 'visualization'
        imgname = imagedir + 'division_lattice_regions_'
        print 'glob.glob(imgname) = ', glob.glob(imgname + '*')
        print 'len(glob.glob(imgname)[0]) = ', len(glob.glob(imgname + '*'))
        framerate = float(len(glob.glob(imgname + '*')))/7.0
        print 'framerate = ', framerate
        if socket.gethostname()[0:6] != 'midway':
            subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname+'%6d.png',
                             movname+'.mov', '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12',
                             '-threads', '0', '-r', '1', '-pix_fmt', 'yuv420p'])

    chern_finsize = np.dstack((Nreg1V, ksize_frac_arr, ksize_V, ksys_sizeV, ksys_fracV, nuV))[0]
    return chern_finsize, params_regs, contribs


def get_kitaev_polygons(shape, regalph, regbeta, reggamma, ksize, delta_pi=2./3., teps=1e-7, verbose=False,
                        outerH=None, check=False):
    """Create polygons that match the supplied shape for summing up the Kitaev 2-current

    Parameters
    ----------
    shape : str
        identifier for shape of summation region: circle, square, hexagon, square_annulus
        If shape == square_annulus, outerH must be specified (as float, not None)
    regalph : float
        largest angle dividing kitaev region
    regbeta : float
        middle angle dividing kitaev region
    reggamma : float
        smallest angle dividing kitaev region

    """
    # Divide plane into three, A->B->C counterclockwise
    # small offset of region boundaries is teps
    if shape == 'hexagon':
        # ksize = Ns*ksize_frac*dist
        delta = np.pi * float(delta_pi)
        # Use the largest possible side length as NH and NV by using a side length for regular hexagon with width ksize
        # Then the width of the hexagon is ksize = a0 + 2 a1 * cos(pi/3) = a0 + a1.
        # Assuming equal sides, we then decide to use ksize * 0.5.
        # Effectively, this ensures that ksize is the side length of the hexagon (for a regular hexagon).
        a1, a2 = polyfns.deformed_hexcell_to_hexagonal_sidelengths(delta, ksize*0.5, ksize*0.5)
        hexagon = polyfns.deformed_hex_polygon(a1, a2)
        # plt.clf()
        # plt.plot(hexagon[:,0],hexagon[:,1],'r-')
        # plt.show()
        polygon1, polygon2, polygon3 = polyfns.divide_hexagon_by_sidelengths(hexagon, eps=1e-8)
        if verbose or check:
            print 'delta = ', delta, ' so delta/pi = ', delta / np.pi
            print 'hexagon = ', hexagon
            print 'polygon1 = ', polygon1
            print 'polygon2 = ', polygon2
            print 'polygon3 = ', polygon3
            print 'a1, a2 = ', a1, a2

    elif shape == 'square':
        ksizeH = ksize
        ksizeV = ksize
        apH = ksizeH*0.5
        apV = ksizeV*0.5
        # length of rays that come out at angles
        angleL = apH/np.cos(regalph)
        # print 'ksize = ', ksize
        polygon1 = np.array([[0., 0.],
                             [apH, angleL*np.sin(regalph)],
                             [apH, apV],
                             [0., apV]])
        polygon2 = np.array([[0., 0.],
                             [np.cos(np.pi*0.5+teps), apV],
                             [-apH, apV],
                             [-apH, angleL*np.sin(regbeta)]])
        polygon3 = np.array([[0., 0.],
                             [-apH, angleL * np.sin(regbeta+teps)],
                             [-apH, -apV],
                             [apH, -apV],
                             [apH, angleL * np.sin(regalph-teps)]])

    elif shape == 'circle':
        tt = np.linspace(0.0, 2.*np.pi, 100)
        # print 'tt = ', tt
        # print 'np.dstack((np.cos(tt), np.sin(tt))) = ', np.dstack((np.cos(tt), np.sin(tt)))
        # print 'ksize = ', ksize
        circlepoly = ksize * 0.5 * np.dstack((np.cos(tt), np.sin(tt)))[0]
        polygon1, polygon2, polygon3 = polyfns.slice_polygon_regions(circlepoly, reggamma, regbeta, regalph)
        if verbose or check:
            print 'polygon1 = ', polygon1
            print 'polygon2 = ', polygon2
            print 'polygon3 = ', polygon3

    elif shape == 'square_annulus':
        ksizeH = ksize
        ksizeV = ksize
        apH = ksizeH * 0.5
        apV = ksizeV * 0.5
        oh = outerH * 0.5
        # length of rays that come out at angles
        angleL = apH/np.cos(regalph)
        aleng = oh/np.cos(regalph)
        # print 'ksize = ', ksize
        polygon1 = np.array([[apH, angleL*np.sin(regalph)],
                             [oh, aleng * np.sin(regalph)],
                             [oh, oh],
                             [0., oh],
                             [0., apV],
                             [apH, apV]])
        polygon2 = np.array([[0., apV],
                             [np.cos(np.pi*0.5+teps), oh],
                             [-oh, oh],
                             [-oh, aleng * np.sin(regbeta)],
                             [-apH, angleL * np.sin(regbeta)],
                             [-apH, apV]])
        polygon3 = np.array([[-apH, angleL * np.sin(regbeta+teps)],
                             [-oh, aleng * np.sin(regbeta + teps)],
                             [-oh, -oh],
                             [oh, -oh],
                             [oh, aleng * np.sin(regalph - teps)],
                             [apH, angleL * np.sin(regalph - teps)],
                             [apH, -apV],
                             [-apH, -apV]])
    else:
        RuntimeError('This shape is not yet supported!')

    if check:
        plt.clf()
        plt.plot(np.hstack((polygon1[:, 0], polygon1[0, 0])), np.hstack((polygon1[:, 1], polygon1[0, 1])), 'r-')
        plt.plot(np.hstack((polygon2[:, 0], polygon2[0, 0])), np.hstack((polygon2[:, 1], polygon2[0, 1])), 'g-')
        plt.plot(np.hstack((polygon3[:, 0], polygon3[0, 0])), np.hstack((polygon3[:, 1], polygon3[0, 1])), 'b-')
        plt.show()

    return polygon1, polygon2, polygon3


def sum_kitaev_projector(reg1, reg2, reg3, r1old, r2old, r3old, pp, nu, verbose=False):
    """
    Given the previous regions summed, the previous result nu, and the new regions and Projector pp, calculate realspace
    chern number.

    reg1 :
    reg2 :
    reg3 :
    r1old :
        elements previous reg3 --> indices already accounted for in sum
    r2old :
        elements previous reg3 --> indices already accounted for in sum
    r3old :
        elements previous reg3 --> indices already accounted for in sum
    pp :
        projector operator taking eigenstates to zero or themselves, based on cutoff frequency
    nu :
        previous result from kitaev chern sum
    verbose : bool
        whether to print statements about progress and current values
    """
    # Add onto the previous nu already computed from previous sum iff reg1_old in reg1, etc
    # First check that new reg1,2,3 contain ALL of elements in reg1_old, reg2_old, reg3_old
    r1ok = len(np.setdiff1d(r1old, reg1)) == 0
    r2ok = len(np.setdiff1d(r2old, reg2)) == 0
    r3ok = len(np.setdiff1d(r3old, reg3)) == 0
    if verbose:
        print 'regions are ok = ', r1ok and r2ok and r3ok
    if r1ok and r2ok and r3ok:
        if verbose:
            print 'Continuing sum from last iteration...'
        reg1star = np.setdiff1d(reg1, r1old)
        reg2star = np.setdiff1d(reg2, r2old)
        reg3star = np.setdiff1d(reg3, r3old)
        nusum_cont = True
    else:
        if verbose:
            print 'Restarting sum from scratch...'
        reg1star = reg1
        reg2star = reg2
        reg3star = reg3
        nu = 0. + 0. * 1j
        nusum_cont = False

    if verbose:
        print 'Summing up h values...'
    dmyi = 0
    for j in reg1star:
        if dmyi % 200 == 0 and verbose: print 'sum: '+str(dmyi) + '/'+str(len(reg1star))
        for k in reg2:
            for l in reg3:
                # print 'jkl = ', j, ',', k, ',', l
                # nu += h[j,k,l]
                nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
        dmyi += 1

    # Do Astar - Bstar
    if nusum_cont:
        dmyi = 0
        for j in r1old:
            if dmyi % 200 == 0 and verbose: print 'aux1 sum: ' + str(dmyi) + '/' + str(len(r1old))
            for k in reg2star:
                for l in reg3:
                    nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
            dmyi += 1

        dmyi = 0
        for j in r1old:
            if dmyi % 200 == 0 and verbose: print 'aux2 sum: ' + str(dmyi) + '/' + str(len(r1old))
            for k in r2old:
                for l in reg3star:
                    nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
            dmyi += 1

    return nu


def sum_kitaev_2current(reg1, reg2, reg3, r1old, r2old, r3old, hh, nu, verbose=False):
    """
    Given the previous regions summed, the previous result nu, and the new regions and Projector pp, calculate realspace
    chern number.

    reg1 :
    reg2 :
    reg3 :
    r1old :
        elements previous reg3 --> indices already accounted for in sum
    r2old :
        elements previous reg3 --> indices already accounted for in sum
    r3old :
        elements previous reg3 --> indices already accounted for in sum
    pp :
        projector operator taking eigenstates to zero or themselves, based on cutoff frequency
    nu :
        previous result from kitaev chern sum
    verbose : bool
        whether to print statements about progress and current values
    """
    # Add onto the previous nu already computed from previous sum iff reg1_old in reg1, etc
    # First check that new reg1,2,3 contain ALL of elements in reg1_old, reg2_old, reg3_old
    r1ok = len(np.setdiff1d(r1old, reg1)) == 0
    r2ok = len(np.setdiff1d(r2old, reg2)) == 0
    r3ok = len(np.setdiff1d(r3old, reg3)) == 0
    if verbose:
        print 'regions are ok = ', r1ok and r2ok and r3ok
    if r1ok and r2ok and r3ok:
        if verbose:
            print 'Continuing sum from last iteration...'
        reg1star = np.setdiff1d(reg1, r1old)
        reg2star = np.setdiff1d(reg2, r2old)
        reg3star = np.setdiff1d(reg3, r3old)
        nusum_cont = True
    else:
        if verbose:
            print 'Restarting sum from scratch...'
        reg1star = reg1
        reg2star = reg2
        reg3star = reg3
        nu = 0. + 0. * 1j
        nusum_cont = False

    if verbose: print 'Summing up h values...'
    dmyi = 0
    for j in reg1star:
        if dmyi % 200 == 0 and verbose: print 'sum: '+str(dmyi) + '/'+str(len(reg1star))
        for k in reg2:
            for l in reg3:
                # print 'jkl = ', j, ',', k, ',', l
                nu += hh[j, k, l]
                # nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
        dmyi +=1

    # Do Astar - Bstar
    if nusum_cont:
        dmyi = 0
        for j in r1old:
            if dmyi % 200 == 0 and verbose: print 'aux1 sum: ' + str(dmyi) + '/' + str(len(r1old))
            for k in reg2star:
                for l in reg3:
                    nu += hh[j, k, l]
                    # nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
            dmyi += 1

        dmyi = 0
        for j in r1old:
            if dmyi % 200 == 0 and verbose: print 'aux2 sum: ' + str(dmyi) + '/' + str(len(r1old))
            for k in r2old:
                for l in reg3star:
                    nu += hh[j, k, l]
                    # nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
            dmyi += 1

    return nu


def sum_kitaev_with_contributions_projector(reg1, reg2, reg3, r1old, r2old, r3old, pp, nu, verbose=False):
    """
    Keep track of contributions from individual gyroscopes as sum 2-current of the projector
    todo: make this more efficient by using r1old, r2old, r3old so as not to redo and to add to prev contrib

    Parameters
    ----------
    reg1 : (#particles in region 1) x 1 int array
        indices of particles in first region of the kitaev summation regions
    reg2 : (#particles in region 2) x 1 int array
        indices of particles in second region of the kitaev summation regions
    reg3 : (#particles in region 3) x 1 int array
        indices of particles in third region of the kitaev summation regions
    r1old : currently unused, should be used for speedup
    r2old : currently unused, should be used for speedup
    r3old : currently unused, should be used for speedup
    pp : len(gyro_lattice.lattice.xy)*2 x len(gyro_lattice.lattice.xy)*2 complex array (optional)
        projection operator
    nu : currently unused, should be used for speedup
    verbose : bool
        Output more to the command line as computation progresses

    Returns
    -------
    nu : complex number
        The chern number result for this computation
    [nu_sum1, nu_sum2, nu_sum3] : list of (# particles in region n) x 1 complex arrays
        contributions of each particle in each region to the total result (when summed over the other two regions)
    """
    print 'Summing up h values using the h object (very slow, for checking)...'
    dmyi = 0
    nu = 0. + 0. * 1j
    # nu_sum12 would look at the contribution for each pair of particles in regA and regB as a matrix
    # nu_sum12 = np.zeros((len(reg1), len(reg2)), dtype='complex')
    # print 'shape(nu_sum12) = ', np.shape(nu_sum12)

    # nu_sum1 is the contribution to nu for each particle in region 1 when region 2 and 3 are summed
    nu_sum1 = np.zeros(len(reg1), dtype='complex')
    nu_sum2 = np.zeros(len(reg2), dtype='complex')
    nu_sum3 = np.zeros(len(reg3), dtype='complex')
    # nu_last12 = 0.
    nu_last1 = 0.
    jind = 0
    for j in reg1:
        # print 'jind = ', jind, ' of ', len(reg1)
        kind = 0
        for k in reg2:
            for l in reg3:
                nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
                dmyi += 1
            # nu_sum12[jind, kind] = nu - nu_last12
            # nu_last12 = nu
            # kind += 1
        nu_sum1[jind] = nu - nu_last1
        nu_last1 = nu
        jind += 1

    print 'nu = ', nu

    print 'sum_kitaev_with_contributions_projector(): Performing other partial sums...'
    nu_last2 = 0.
    jind = 0
    dmyi = 0
    nu = 0. + 0.*1j
    for j in reg2:
        for k in reg3:
            for l in reg1:
                nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
                dmyi += 1
        nu_sum2[jind] = nu - nu_last2
        nu_last2 = nu
        jind += 1

    print 'sum_kitaev_with_contributions_projector(): Performing last other partial sums...'
    nu_last3 = 0.
    jind = 0
    dmyi = 0
    nu = 0. + 0.*1j
    for j in reg3:
        for k in reg1:
            for l in reg2:
                nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
                dmyi += 1
        nu_sum3[jind] = nu - nu_last3
        nu_last3 = nu
        jind += 1

    return nu, [nu_sum1, nu_sum2, nu_sum3]


def sum_kitaev_with_contributions(reg1, reg2, reg3, r1old, r2old, r3old, hh, nu, verbose=False):
    """
    Keep track of contributions from individual gyroscopes as sum 2-current of the projector
    todo: make this more efficient by using r1old, r2old, r3old so as not to redo and to add to prev contrib
    """
    print 'sum_kitaev_with_contributions(): Summing up h values using the h object (very slow, for checking)...'
    dmyi = 0
    nu = 0. + 0. * 1j
    # nu_sum12 would look at the contribution for each pair of particles in regA and regB as a matrix
    # nu_sum12 = np.zeros((len(reg1), len(reg2)), dtype='complex')
    # print 'shape(nu_sum12) = ', np.shape(nu_sum12)

    # nu_sum1 is the contribution to nu for each particle in region 1 when region 2 and 3 are summed
    nu_sum1 = np.zeros(len(reg1), dtype='complex')
    nu_sum2 = np.zeros(len(reg2), dtype='complex')
    nu_sum3 = np.zeros(len(reg3), dtype='complex')
    # nu_last12 = 0.
    nu_last1 = 0.
    jind = 0
    for j in reg1:
        # print 'jind = ', jind, ' of ', len(reg1)
        kind = 0
        for k in reg2:
            for l in reg3:
                # print 'jkl = ', j, ',', k, ',', l
                nu += hh[j, k, l]
                # nu += 12 * np.pi * 1j * (pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j])
                dmyi += 1
            # nu_sum12[jind, kind] = nu - nu_last12
            # nu_last12 = nu
            # kind += 1
        nu_sum1[jind] = nu - nu_last1
        nu_last1 = nu
        jind += 1

    print 'nu = ', nu

    print 'sum_kitaev_with_contributions(): Performing other partial sums...'
    nu_last2 = 0.
    jind = 0
    dmyi = 0
    nu = 0. + 0.*1j
    for j in reg2:
        for k in reg3:
            for l in reg1:
                nu += hh[j, k, l]
                dmyi += 1
        nu_sum2[jind] = nu - nu_last2
        nu_last2 = nu
        jind += 1

    print 'sum_kitaev_with_contributions(): Performing last other partial sums...'
    nu_last3 = 0.
    jind = 0
    dmyi = 0
    nu = 0. + 0.*1j
    for j in reg3:
        for k in reg1:
            for l in reg2:
                nu += hh[j, k, l]
                dmyi += 1
        nu_sum3[jind] = nu - nu_last3
        nu_last3 = nu
        jind += 1

    return nu, [nu_sum1, nu_sum2, nu_sum3]


def plot_chern_realspace(gyro_lattice, reg1_xy, reg2_xy, reg3_xy, polygon1, polygon2, polygon3,
                         ax=None, outdir=None, filename='division_lattice.png', title='', legend=False, check=False):
    """Plot the network or point set used (depending on the instance type of the input gyro_lattice) and plot the
    kitaev summation regions as colored circles, squares, and triangles on top of the sites, as well as the polygons
    associated with the 3 summation regions.

    Parameters
    ----------
    gyro_lattice : GyroLattice instance or NP x 2 float array
        The gyroscopic network on which to compute the chern number (a GyroLattice instance), or just the points from
        that network (an NP x 2 float array)
    reg1_xy : (#particles in region 1) x 1 int array
        The indices of the particles in region 1 of the summation regions
    reg2_xy : (#particles in region 2) x 1 int array
        The indices of the particles in region 2 of the summation regions
    reg3_xy : (#particles in region 3) x 1 int array
        The indices of the particles in region 3 of the summation regions
    polygon1 : #vertices x 2 float array
        The polygon delineating region 1 of the summation regions
    polygon2 : #vertices x 2 float array
        The polygon delineating region 2 of the summation regions
    polygon3 : #vertices x 2 float array
        The polygon delineating region 3 of the summation regions
    ax : matpplotlib axis instance or none
        If not None, plot onto this axis
    outdir : str or None
        Path to store plot, if not None
    filename : str
        filename as which to save plot, if outdir is not None
    title : str
        title for the plot
    legend : bool
        Whether to make a legend
    check : bool
        Display intermediate results

    Returns
    -------
    ax : matplotlib axis instance
        The axis with the plotted network and regions
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(gyro_lattice, np.ndarray):
        xy = gyro_lattice
        ax.scatter(xy[:, 0], xy[:, 1], c='k')
    else:
        xy = gyro_lattice.lattice.xy
        ax = le.display_lattice_2D(xy, gyro_lattice.lattice.BL,
                                   PVxydict=gyro_lattice.lattice.PVxydict,
                                   NL=gyro_lattice.lattice.NL, KL=gyro_lattice.lattice.KL,
                                   bs='none', close=False, colorz=False, colormap='BlueBlackRed',
                                   bgcolor='#FFFFFF', axis_off=True, ax=ax)

    ax.scatter(xy[reg1_xy, 0], xy[reg1_xy, 1], c='r', s=20, marker='o', alpha=0.3, label='A', zorder=100)
    ax.scatter(xy[reg2_xy, 0], xy[reg2_xy, 1], c='g', s=20, marker='^', alpha=0.3, label='B', zorder=101)
    ax.scatter(xy[reg3_xy, 0], xy[reg3_xy, 1], c='b', s=20, marker='s', alpha=0.3, label='C', zorder=102)
    patchList = []
    patchList.append(patches.Polygon(polygon1, color='r'))
    patchList.append(patches.Polygon(polygon2, color='g'))
    patchList.append(patches.Polygon(polygon3, color='b'))
    p = PatchCollection(patchList, cmap=cm.jet, alpha=0.2, zorder=99)
    colors = np.linspace(0, 1, 3)[::-1]
    p.set_array(np.array(colors))
    ax.add_collection(p)
    if legend:
        ax.legend()
    ax.set_title(title)
    ax.axis('equal')

    xlimv = (np.max(np.abs(xy[:, 0]))) * 1.1 + 2
    ylimv = (np.max(np.abs(xy[:, 1]))) * 1.1 + 2
    ax.set_xlim(-xlimv, xlimv)
    ax.set_ylim(-ylimv, ylimv)
    if outdir is not None:
        plt.savefig(outdir + filename)
    if check:
        plt.show()

    return ax


def plot_chern_realspace_2panel(gyro_lattice, ksys_fracV, nuV, kk, reg1_xy, reg2_xy, reg3_xy,
                                polygon1, polygon2, polygon3, cmap='bbr0',
                                outdir=None, filename='division_lattice_nu.png', title=None,
                                Wfig=360, Hfig=270, fontsize=12, wsfrac=0.4, wssfrac=0.3,
                                x0frac=0.1, y0frac=0.1, check=False):
    """Plots the computation of the realspace chern number, with a panel for the network on the left, and a panel for
    a plot of chern vs kitaev size on the right.

    Parameters
    ----------
    gyro_lattice : GyroLattice instance or NP x 2 float array
        The gyroscopic network on which to compute the chern number (a GyroLattice instance), or just the points from
        that network (an NP x 2 float array)
    ksys_fracV : 1d float array
        fractional number of particles in the kitaev sum wrt the total
    nuV :
    kk :
    reg1_xy : (#particles in region 1) x 1 int array
        The indices of the particles in region 1 of the summation regions
    reg2_xy : (#particles in region 2) x 1 int array
        The indices of the particles in region 2 of the summation regions
    reg3_xy : (#particles in region 3) x 1 int array
        The indices of the particles in region 3 of the summation regions
    polygon1 : #vertices x 2 float array
        The polygon delineating region 1 of the summation regions
    polygon2 : #vertices x 2 float array
        The polygon delineating region 2 of the summation regions
    polygon3 : #vertices x 2 float array
        The polygon delineating region 3 of the summation regions
    cmap : str specifier
        colormap to use in the plot for the chern number curve (discretely sampled, at each ksys_fracV element)
    outdir : str
        Path of directory in which to store the output
    filename : str
        name of the plot
    title : str
        Master title, currently unused
    Wfig : int or float
        Width of the figure
    Hfig : int or float
        Height of the figure
    FSFS : int
        Font size for text in figure
    wsfrac : float between 0 and 1
        fractional width of the subplot relative to Wfig
    wssfrac : float between 0 and 1
        fractional width of the subsubplot relative to Wfig (for the chern vs ksize plot)
    x0frac : float less than 1
        fractional width of the whitespace on the left
    y0frac : float less than 1
        fractional width of the whitespace on the bottom of the plot
    check : bool
        Display intermediate results
    """
    fig = sps.figure_in_mm(Wfig, Hfig)
    ws = wsfrac * Wfig
    hs = ws
    wss = wssfrac * Wfig
    hss = wss * 3./4.
    x0 = x0frac * Wfig
    y0 = y0frac * Wfig
    label_params = dict(size=fontsize, fontweight='normal')
    ax = [sps.axes_in_mm(x0, y0, width, height, label=part, label_params=label_params)
          for x0, y0, width, height, part in (
                  [x0, (Hfig - hs) * 0.5, ws, hs, ''],  # network and kitaev regions
                  [Wfig - wss - x0, (Hfig - hss) * 0.5, wss, hss, '']  # plot for chern
          )]

    # Note that plot_chern_realspace accepts a simple xy as the gyro_lattice argument
    plot_chern_realspace(gyro_lattice, reg1_xy, reg2_xy, reg3_xy, polygon1, polygon2, polygon3,
                         ax=ax[0], outdir=None, filename='', title='', check=False)

    if cmap not in plt.colormaps():
        lecmaps.register_colormaps()

    ax[1].scatter(ksys_fracV[0:kk+1]*0.5, nuV[0:kk+1], c=nuV[0:kk+1], edgecolor="none",
                  cmap='bbr0', vmin=-1.0, vmax=1.0)
    ax[1].set_xlim(0, 1.0)
    ax[1].set_ylim(-1., 1.0)
    ax[1].set_xlabel(r'Fraction of gyros in sum')
    ax[1].set_ylabel(r'Chern number, $\nu$')

    if title is not None:
        ax[1].text(0.5, 0.8, title, horizontalalignment='center', fontsize=fontsize, transform=fig.transFigure)

    title = 'Chern number results\n' + r'$\nu =$' + '{0:0.3f}'.format(nuV[kk])
    ax[1].text(0.5, 1.08, title, horizontalalignment='center', fontsize=fontsize, transform=ax[1].transAxes)
    if outdir is not None:
        plt.savefig(outdir + filename)
    if check:
        plt.show()


def plot_chern_contributions(gyro_lattice, reg1, reg2, reg3, nu_sum1, nu_sum2, nu_sum3, polygon1, polygon2, polygon3,
                             basis='XY', ax=None, outdir=None, filename='spatial_contributions.pdf',
                             negcolor='#4479BA', poscolor='#D93B46'):
    """Plots the contributions of each particle to the chern number computation

    Parameters
    ----------
    gyro_lattice : GyroLattice instance or NP x 2 float array
        If GyroLattice instance, the network (with physics) to draw. If NP x 2 float array, the point set to draw.
    reg1 : #gyros in reg1 x 1 array
        indices of eigvector/eigvals for gyros in reg3 (2*index for x, 2*index + 1 for y d.o.f.)
    reg2 : #gyros in reg1 x 1 array
        indices of eigvector/eigvals for gyros in reg3 (2*index for x, 2*index + 1 for y d.o.f.)
    reg3 : #gyros in reg1 x 1 array
        indices of eigvector/eigvals for gyros in reg3 (2*index for x, 2*index + 1 for y d.o.f.)
    nu_sum1 :
        partial contributions from each gyro in reg1
    nu_sum2 :
        partial contributions from each gyro in reg2
    nu_sum3 :
        partial contributions from each gyro in reg3
    polygon1 : #vertices x 2 float array or None
        polygon enclosing polygon1
    polygon2 : #vertices x 2 float array or None
        polygon enclosing polygon2
    polygon3 : #vertices x 2 float array or None
        polygon enclosing polygon3
    basis : str
        'XY' or 'psi', the basis in which the computation was performed. If XY, particle 0 corresponds to rows 0 (x)
        and 1 (y) of an eigenvector. If psi, corresponds to rows 0 (right moving) and NP (left moving).
    ax : axis instance
        axis on which to plot the contributions
    outdir : str or None
        If not None, saves plot to this directory
    filename : str
        If outdir != None, saves plot as this name in outdir
    """
    if ax is None:
        ax = plt.gca()
    # Display partial sums spatially
    print 'Plotting partial sums spatially...'
    pos1 = np.where(np.real(nu_sum1) > 0)[0]
    pos2 = np.where(np.real(nu_sum2) > 0)[0]
    pos3 = np.where(np.real(nu_sum3) > 0)[0]
    neg1 = np.where(np.real(nu_sum1) < 0)[0]
    neg2 = np.where(np.real(nu_sum2) < 0)[0]
    neg3 = np.where(np.real(nu_sum3) < 0)[0]

    if basis == 'XY':
        r1pt = np.floor(reg1 * 0.5).astype(int)
        r2pt = np.floor(reg2 * 0.5).astype(int)
        r3pt = np.floor(reg3 * 0.5).astype(int)
    else:
        raise RuntimeError("Haven't handled this yet")

    if isinstance(gyro_lattice, np.ndarray):
        xy = gyro_lattice
        ax.scatter(xy[:, 0], xy[:, 1], c='k')
    else:
        xy = gyro_lattice.lattice.xy
        ax = le.display_lattice_2D(xy, gyro_lattice.lattice.BL,
                                   PVxydict=gyro_lattice.lattice.PVxydict,
                                   NL=gyro_lattice.lattice.NL, KL=gyro_lattice.lattice.KL,
                                   bs='none', close=False, colorz=False, colormap='BlueBlackRed',
                                   bgcolor='#FFFFFF', axis_off=True, ax=ax, linewidth=0.2)

    # plt.legend(loc='best')
    patchList = []
    if polygon1 is not None:
        patchList.append(patches.Polygon(polygon1, color='r'))
    if polygon2 is not None:
        patchList.append(patches.Polygon(polygon2, color='g'))
    if polygon3 is not None:
        patchList.append(patches.Polygon(polygon3, color='b'))
    p = PatchCollection(patchList, cmap=cm.jet, alpha=0.05)
    colors = np.linspace(0, 1, 3)[::-1]
    p.set_array(np.array(colors))
    ax.add_collection(p)

    plt.scatter(xy[r1pt, 0][pos1], xy[r1pt, 1][pos1], s=np.abs(nu_sum1[pos1]) * 400, facecolors=poscolor,
                edgecolors=poscolor, marker='o', label='reg1')
    plt.scatter(xy[r1pt, 0][neg1], xy[r1pt, 1][neg1], s=np.abs(nu_sum1[neg1]) * 400, facecolors=negcolor,
                edgecolors=negcolor, marker='o')
    plt.scatter(xy[r2pt, 0][pos2], xy[r2pt, 1][pos2], s=np.abs(nu_sum2[pos2]) * 400, facecolors=poscolor,
                edgecolors=poscolor, marker='^', label='reg2')
    plt.scatter(xy[r2pt, 0][neg2], xy[r2pt, 1][neg2], s=np.abs(nu_sum2[neg2]) * 400, facecolors=negcolor,
                edgecolors=negcolor, marker='^')
    plt.scatter(xy[r3pt, 0][pos3], xy[r3pt, 1][pos3], s=np.abs(nu_sum3[pos3]) * 400, facecolors=poscolor,
                edgecolors=poscolor, marker='s', label='reg3')
    plt.scatter(xy[r3pt, 0][neg3], xy[r3pt, 1][neg3], s=np.abs(nu_sum3[neg3]) * 400, facecolors=negcolor,
                edgecolors=negcolor, marker='s')
    plt.title('$\sum_{kl} h_{jkl}$ Each sum separate')
    ax.axis('equal')

    if outdir is not None:
        plt.savefig(outdir + filename)

    return ax


def get_cmeshfn(lp, rootdir=None):
    """Prepare the path where the cherns for a particular lattice are stored. If rootdir is specified, use that for the
    base of the path. Otherwise, adopt the base of the path from the lattice_params dict (lp) of the lattice.

    Parameters
    ----------
    lp : dict
        lattice parameters dictionary
    rootdir : str or None

    Returns
    -------
    cmeshfn : str
    """
    meshfn_split = lp['meshfn'].split('/')
    ind = np.where(np.array(meshfn_split) == 'networks')[0][0]
    cmeshfn = ''
    if rootdir is None:
        for strseg in meshfn_split[0:ind]:
            cmeshfn += strseg + '/'
    else:
        cmeshfn = dio.prepdir(rootdir)
    cmeshfn += 'chern/'
    for strseg in meshfn_split[(ind + 1):]:
        cmeshfn += strseg + '/'

    # Form physics subdir for Omk, Omg, V0_pin_gauss, V0_spring_gauss
    eps = 1e-7
    if 'ABDelta' in lp:
        if lp['ABDelta'] > eps:
            cmeshfn += 'ABd' + '{0:0.4f}'.format(lp['ABDelta']).replace('.', 'p') + '/'
    if 'pinconf' in lp:
        if lp['pinconf'] > 0:
            cmeshfn += 'pinconf' + '{0:04d}'.format(lp['pinconf']) + '/'

    if 'OmKspec' in lp:
        cmeshfn += lp['OmKspec']
    else:
        cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'])
    cmeshfn += '_Omg' + sf.float2pstr(lp['Omg'])
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        cmeshfn += '_pinV' + sf.float2pstr(lp['V0_pin_gauss'])
        if np.abs(lp['V0_spring_gauss']) < eps:
            if 'pinconf' in lp:
                if lp['pinconf'] > 0:
                    # Add pinconf tag first, then spring disorder strength
                    cmeshfn += '_pinconf' + '{0:04d}'.format(lp['pinconf'])
                    cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
                else:
                    cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
            else:
                cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
        else:
            # Add spring disorder strength FIRST, then configuration tag
            cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])
            cmeshfn += '_pinconf' + '{0:04d}'.format(lp['pinconf'])

    if 'ABDelta' in lp:
        if lp['ABDelta'] > 0:
            cmeshfn += '_ABd' + sf.float2pstr(lp['ABDelta'])
    return cmeshfn


def get_cpmeshfn(cp, lp):
    """Get the path for the specific chern calculation that uses the chern parameter dict cp on a glat with lattice
    parameters lp.
    """
    if 'rootdir' in cp:
        cpmeshfn = get_cmeshfn(lp, rootdir=cp['rootdir'])
        print '\n kfns: get_cpmeshfn(): rootdir is found in cp:'
        print 'cpmeshfn ==> ', cpmeshfn, '\n'
    else:
        print '\n kfns: get_cpmeshfn(): rootdir is NOT found in cp!'
        cpmeshfn = get_cmeshfn(lp)

    # Form cp subdir
    if isinstance(cp['omegac'], np.ndarray):
        if 'check' in cp:
            if cp['check']:
                print "Warning: cp['omegac'] is numpy array, using first element to get cpmeshfn..."
        cpmeshfn += '_omc' + sf.float2pstr(cp['omegac'][0])
    else:
        cpmeshfn += '_omc' + sf.float2pstr(cp['omegac'])
    cpmeshfn += '_Nks' + str(int(len(cp['ksize_frac_arr'])))
    cpmeshfn += '_' + cp['shape']
    cpmeshfn += '_a' + sf.float2pstr(cp['regalph'])
    cpmeshfn += '_b' + sf.float2pstr(cp['regbeta'])
    cpmeshfn += '_g' + sf.float2pstr(cp['reggamma'])
    cpmeshfn += '_polyT' + sf.float2pstr(cp['polyT'])
    # fix poly_offset string if it does not conform to 6 decimal convention
    if cp['poly_offset'] != 'none':
        tmp = cp['poly_offset'].split('/')
        poly_offset = sf.float2pstr(sf.str2float(tmp[0]), ndigits=6) + '/' + \
                      sf.float2pstr(sf.str2float(tmp[1]), ndigits=3)
    else:
        poly_offset = cp['poly_offset']

    cpmeshfn += '_polyoff' + sf.prepstr(poly_offset)
    cpmeshfn += '_' + cp['basis'] + '/'
    return cpmeshfn


def get_ccpath(cp, lp, rootdir=None, method='varyloc'):
    """Get the path for outputting info on a collection of chern calculations that uses the chern parameter dict cp.

    Parameters
    ----------
    cp : dict
        parameters for chern collection
    lp : dict
        lattice parameters for the gyro lattice for which many cherns are computed
    rootdir : str
        Root path where info for collections of cherns are stored
    method : str
        String describing how to categorize the many cherns

    Returns
    -------
    cpmesh : str
        cppath
    """
    meshfn_split = lp['meshfn'].split('/')
    ind = np.where(np.array(meshfn_split) == 'networks')[0][0]
    ccmeshfn = ''
    if rootdir is None:
        for strseg in meshfn_split[0:ind]:
            ccmeshfn += strseg + '/'
    else:
        ccmeshfn = dio.prepdir(rootdir)
    ccmeshfn += 'cherns/'
    for strseg in meshfn_split[(ind + 1):]:
        ccmeshfn += strseg + '/'

    ccmeshfn += method
    # Form physics subdir for Omk, Omg, V0_pin_gauss, V0_spring_gauss
    if lp['Omk'] != -1.0:
        ccmeshfn += '_Omk' + sf.float2pstr(lp['Omk'])
    if lp['Omg'] != -1.0:
        ccmeshfn += '_Omg' + sf.float2pstr(lp['Omg'])
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        ccmeshfn += '_pinV' + sf.float2pstr(lp['V0_pin_gauss'])
        ccmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])

    # Form cp subdir
    if method != 'omegac':
        if isinstance(cp['omegac'], np.ndarray):
            print "Warning: cp['omegac'] is numpy array, using first element to get cpmeshfn..."
            ccmeshfn += '_omc' + sf.float2pstr(cp['omegac'][0])
        else:
            ccmeshfn += '_omc' + sf.float2pstr(cp['omegac'])

    ccmeshfn += '_Nks' + str(int(len(cp['ksize_frac_arr'])))
    ccmeshfn += '_' + cp['shape']
    ccmeshfn += '_a' + sf.float2pstr(cp['regalph'])
    ccmeshfn += '_b' + sf.float2pstr(cp['regbeta'])
    ccmeshfn += '_g' + sf.float2pstr(cp['reggamma'])
    ccmeshfn += '_polyT' + sf.float2pstr(cp['polyT'])
    if method != 'varyloc':
        ccmeshfn += '_polyoff' + sf.prepstr(cp['poly_offset']).replace('/', '_')
        ccmeshfn += '_' + cp['basis'] + '/'

    return ccmeshfn


######################################################
######################################################
######################################################
######################################################
if __name__ == '__main__':
    '''Perform an example of using the lattice_collection class'''
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
    parser.add_argument('-prpoly_overlay', '--prpoly_overlay', help='Make overlaid DOS plots', action='store_true')
    parser.add_argument('-ipr_overlay', '--ipr_overlay', help='Overlay participation ratio DOS plots',
                        action='store_true')
    parser.add_argument('-ipr_stack', '--ipr_stack', help='Stack subplots of participation-ratio-colored DOS',
                        action='store_true')

    # Geometry arguments for the lattices to load
    parser.add_argument('-AB', '--ABDelta', help='Difference in pinning frequency for AB sites', type=float, default=0.)
    parser.add_argument('-N', '--N', help='Mesh width AND height, in number of lattice spacings (leave blank to ' +
                                         'specify separate dims)', type=int, default=-1)
    parser.add_argument('-NP', '--NP_load',
                        help='Specify to nonzero int to load a network of a particular size in its entirety, without' +
                             ' cropping. Will override NH and NV',
                        type=int, default=20)
    parser.add_argument('-NH', '--NH', help='Mesh width, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-NV', '--NV', help='Mesh height, in number of lattice spacings', type=int, default=50)
    parser.add_argument('-LT', '--LatticeTop', help='Lattice topology: linear, hexagonal, triangular, ' +
                                                    'deformed_kagome, hyperuniform, circlebonds, penroserhombTri',
                        type=str, default='hucentroid')
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
                        help='Declare whether or not to cut bonds to obtain target coordination number z',
                        type=bool, default=False)
    parser.add_argument('-cutz_method', '--cutz_method',
                        help='Method for cutting z from initial loaded-lattice value to target_z (highest or random)',
                        type=str, default='none')
    parser.add_argument('-z', '--target_z', help='Coordination number to enforce', type=float, default=-1)
    parser.add_argument('-Ndefects', '--Ndefects', help='Number of defects to introduce', type=int, default =1)
    parser.add_argument('-Bvec', '--Bvec', help='Direction of burgers vectors of dislocations (random, W, SW, etc)',
                        type=str, default='W')
    parser.add_argument('-dislocxy', '--dislocation_xy',
                        help='XY of single dislocation, if not centered at (0,0), as strings sep by / (ex: 1/4.4)',
                        type=str, default='none')
    
    # Global geometric params
    parser.add_argument('-periodic', '--periodicBC', help='Enforce periodic boundary conditions', action='store_true')
    parser.add_argument('-slit', '--make_slit', help='Make a slit in the mesh', action='store_true')
    parser.add_argument('-phi', '--phi', help='Shear angle for hexagonal (honeycomb) lattice in radians/pi', type=float,
                        default=0.0)
    parser.add_argument('-delta', '--delta', help='Deformation angle for hexagonal (honeycomb) lattice in radians/pi',
                        type=float, default=120./180.)
    parser.add_argument('-eta', '--eta', help='Randomization/jitter in lattice', type=float, default=0.000)
    parser.add_argument('-theta', '--theta', help='Overall rotation of lattice', type=float, default=0.000)
    parser.add_argument('-alph', '--alph', help='Twist angle for twisted_kagome, max is pi/3', type=float, default=0.00)
    parser.add_argument('-huno', '--hyperuniform_number', help='Hyperuniform realization number',
                        type=str, default='01')
    parser.add_argument('-skip_gr', '--skip_gr', help='Skip calculation of g(r) correlation function for the lattice',
                        action='store_true')
    parser.add_argument('-skip_gxy', '--skip_gxy',
                        help='Skip calculation of g(x,y) 2D correlation function for the lattice', action='store_true')
    parser.add_argument('-skip_sigN', '--skip_sigN', help='Skip calculation of variance_N(R)', action='store_true')
    parser.add_argument('-fancy_gr', '--fancy_gr',
                        help='Perform careful calculation of g(r) correlation function for the ENTIRE lattice',
                        action='store_true')
    args = parser.parse_args()
    
    print 'args.delta  = ', args.delta
    if args.N >0:
        NH = args.N
        NV = args.N
    else:
        NH = args.NH
        NV = args.NV
    lattice_type = args.LatticeTop

    phi = np.pi* args.phi
    delta = np.pi* args.delta

    strain =0.00  # initial
    # z = 4.0 #target z
    if lattice_type == 'linear':
        shape = 'line'
    else:
        shape = args.shape
    
    theta = args.theta
    eta = args.eta
    transpose_lattice=0
    
    make_slit = args.make_slit
    # deformed kagome params
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    z  = 0.0

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

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    outdir = rootdir+'experiments/DOS_scaling/'+args.LatticeTop+'/'
    le.ensure_dir(outdir)
    lp = { 'LatticeTop' : args.LatticeTop,
          'NH' : NH,
          'NV' : NV,
          'rootdir' : rootdir,
          'periodicBC' : True,
          }

    # Collate DOS for many lattices
    gc = gyro_collection.GyroCollection()
    if args.LatticeTop == 'iscentroid':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/'+args.LatticeTop+'/'+\
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'kagome_isocent':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
        gc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
            args.LatticeTop + '_square_periodic_d*'+'_NP*' + str(args.NP_load))
    elif args.LatticeTop == 'kagper_hucent':
        gc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
                      args.LatticeTop + '_square_d*' + '_' + '{0:06d}'.format(NH))

    title = r'$D(\omega)$ for ' + description + ' networks'

    if args.prpoly_overlay:
        gc.ensure_all_ipr_saved()
        gc.plot_prpoly_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_prpoly_overlay',title=title)

    if args.ipr_overlay:
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_ipr_overlay',title=title, inverse_PR=True)
        gc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_pr_overlay',title=title, inverse_PR=False)
        gc.plot_DOS_overlay(outdir=outdir, fname= 'NP'+str(args.NP_load)+'_eigval_hist_overlay',title=title)

    if args.ipr_stack:
        # Get ylabels from meshfn names
        ylabels = []
        ii = 0
        for meshfn in gc.meshfns:
            pstring = meshfn[meshfn.index('perd')+4:meshfn.index('perd')+8].replace('p','.')
            ylabels.append(pstring)
            ii += 1

        gc.plot_ipr_DOS_stack(outdir=outdir, fname='N'+str(args.N)+'_eigval_ipr_stack', title=title,
                              ylabels=ylabels, inverse_PR=False, vmin=0.0, vmax=0.5)