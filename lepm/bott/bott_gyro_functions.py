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
import numpy.linalg as la
from scipy.linalg import logm
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import argparse
import copy
import sys
import h5py
import lepm.polygon_functions as polyfns
import lepm.data_handling as dh
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

'''
Compute Bott Index measurements made via the Loring & Hastings realspace method.
Note that since the gyroscope system has two dof at each site, each region (for ex, reg1) contains indices for the
projector to be summed over that include 2*i and 2*i+1 for particle i, corresponding to that particle's band-projected
x and y displacement. Note that if the psi basis is used (rather than XY basis) then indices will be i and i+N for
particle i, corresponding to that particles left and right circularly polarized displacements.

For example usage, use if name==main clause at end of document.

'''


def data_on_disk(bott):
    """Look to see if bott index is stored in hdf5 file

    Returns
    -------
    bool
        whether the data is stored
    """
    print 'bott.cp[h5fn] = ', bott.cp['h5fn']
    if glob.glob(bott.cp['h5fn']):
        fi = h5py.File(bott.cp['h5fn'], "r")
        if bott.cp['h5subgroup'] in fi:
            fi.close()
            return True
        else:
            fi.close()
            return False
    else:
        return False


def data_on_disk_txt(bott):
    """Look to see if bott index is stored as txt file

    Returns
    -------
    bool
        whether the data is stored
    """
    if glob.glob(dio.prepdir(bott.cp['cpmeshfn']) + 'bott.txt'):
        return True
    else:
        return False


def calc_small_projector(haldane_lattice, omegac, attribute=False):
    """Given a haldane network and a cutoff frequency, create the projection operator taking states below that freq
    to zero and above to themselves

    Parameters
    ----------
    haldane_lattice : HaldaneLattice instance
        The network on which to find the projector
    omegac : float
        cutoff frequency

    Returns
    -------
    proj : len(haldane_lattice.lattice.xy)*2 x len(haldane_lattice.lattice.xy)*2 complex array
        projection operator
    """
    # print '\n\n\nkfns: haldane_lattice.eigval= ', haldane_lattice.eigval, '\n\n\n\n'
    eigval, eigvect = haldane_lattice.get_eigval_eigvect(attribute=attribute)
    # eigvect is stored as NModes x NP array, make projector nmodes above omegac x NP array
    return eigvect[np.imag(eigval) > omegac]


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
    """THIS HAS NOT BEEN UPDATED FOR BOTT INDEX SPECIFIC CALC
    Return magnitude of difference between elements of projector and old projector for sites that are the in the same
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
    """THIS HAS NOT BEEN UPDATED FOR BOTT INDEX SPECIFIC CALC
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


def calc_bott(gyro_lattice, cp, psub=None, check=False, verbose=False, attribute_evs=False):
    """Compute the bott index for a gyro_lattice

    Parameters
    ----------
    gyro_lattice : GyroLattice instance
        The gyro network of which to compute the Bott index
    psub : (#states with freq > omegac) x len(gyro_lattice.lattice.xy)*2 complex array (optional)
        projection operator, if already calculated previously. If None, this function calculates this.
    check : bool (optional)
        Display intermediate results
    verbose : bool (optional)
        Print more output on command line
    attribute_evs : bool (optional)
        Attribute the eigval and eigvect to the GyroLattice instance gyro_lattice, if projector pp is not supplied

    Returns
    -------
    bott : float
        The computed Bott index for this system
    """
    lat = gyro_lattice.lattice
    xy = lat.xy
    lat.get_PV(attribute=True)
    omegac = cp['omegac']

    if psub is None:
        print 'Computing projector...'
        psub = calc_small_projector(gyro_lattice, omegac, attribute=attribute_evs)

    # print 'psub = ', psub
    # sys.exit()

    #######################################################
    # if lp['check']:
    #     print 'min diff eigval = ', np.abs(np.diff(eigval)).min()
    #     close = np.where(np.abs(np.diff(eigval)) < 1e-14)[0]
    #     print 'diff eigval = ', np.diff(eigval)
    #     print 'close -> ', close
    #     plt.plot(np.arange(len(eigval)), np.real(eigval), 'ro-', label=r'Re($e_i$)')
    #     plt.plot(np.arange(len(eigval)), np.imag(eigval), 'b.-', label=r'Im($e_i$)')
    #     plt.legend()
    #     plt.xlabel('eigenvalue index', fontsize=fsfs)
    #     plt.ylabel('eigenvalue', fontsize=fsfs)
    #     plt.title('Eigenvalues of the Haldane Model', fontsize=fsfs)
    #     plt.show()
    #
    #     # Look at the matrix of eigenvectors
    #     ss = eigvect
    #     sum0 = np.abs(np.sum(np.abs(ss ** 2), axis=0))
    #     sum1 = np.abs(np.sum(np.abs(ss ** 2), axis=1))
    #     print 'sum0 = ', sum0
    #     print 'sum1 = ', sum1
    #     ss1 = ss.conj().T
    #     nearI = np.dot(ss, ss1)
    #######################################################

    # Get U = P exp(iX) P and V = P exp(iY) P
    # double the position vectors in x and y
    xx = np.repeat(xy[:, 0], 2)
    yy = np.repeat(xy[:, 1], 2)

    # rescale the position vectors in x and y
    xsize = np.sqrt(lat.PV[0][0] ** 2 + lat.PV[0][1] ** 2)
    ysize = np.sqrt(lat.PV[1][0] ** 2 + lat.PV[1][1] ** 2)
    theta = (xx - np.min(xy[:, 0])) / xsize * 2. * np.pi
    phi = (yy - np.min(xy[:, 1])) / ysize * 2. * np.pi

    print 'np.shape(theta) = ', np.shape(theta)
    print 'np.shape(phi) = ', np.shape(phi)
    print 'np.shape(psub) = ', np.shape(psub)

    uu = np.dot(psub, np.dot(np.exp(1j * theta) * np.identity(len(xx)), psub.conj().transpose()))
    vv = np.dot(psub, np.dot(np.exp(1j * phi) * np.identity(len(yy)), psub.conj().transpose()))

    # This is the way using the full projector P = S M S^{-1}
    #     uu = np.dot(pp, np.dot(np.exp(1j * theta) * np.identity(len(xx)), pp))
    #     vv = np.dot(pp, np.dot(np.exp(1j * phi) * np.identity(len(xx)), pp))

    # Could multiply the eigenvalues of VUUtVt to get the determinant of the product,
    # but instead add logs of evs.
    # Log Det = Trace Log
    # if Log e^A = A, then this holds. Wikipedia says this holds for Lie groups
    # Wikipedia Matrix Exponential > Jacobi's Formula says that
    # det e^A = e^{tr(A)}

    # Compute the Bott index
    if verbose:
        print 'diagonalizing (V U Vt Ut)...'

    ev = la.eigvals(np.dot(vv, np.dot(uu, np.dot(vv.conj().T, uu.conj().T))))

    # Consider only eigvals near the identity -- should always automatically
    # be satisfied given that all included states are orthonormal.
    # ev = ev[np.abs(ev) > 1e-9]

    # Perhaps sensitive to errors during product, so this is commented out
    # tr = np.prod(ev)

    bott = np.imag(np.sum(np.log(ev))) / (2. * np.pi)
    if verbose:
        print 'bott = ', bott

    return bott


def get_cmeshfn(lp, rootdir=None):
    """Prepare the path where the calculations for a particular lattice are stored. If rootdir is specified, use that
    for the base of the path. Otherwise, adopt the base of the path from the lattice_params dict (lp) of the lattice.

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
    cmeshfn += 'bott_gyro/'
    for strseg in meshfn_split[(ind + 1):]:
        cmeshfn += strseg + '/'

    # Form physics subdir for Omk, Omg, V0_pin_gauss, V0_spring_gauss
    eps = 1e-7
    if 'ABDelta' in lp:
        if lp['ABDelta'] > eps:
            cmeshfn += 'ABd' + '{0:0.4f}'.format(lp['ABDelta']).replace('.', 'p') + '/'
    if 'OmKspec' in lp:
        if lp['OmKspec'] != '':
            cmeshfn += lp['OmKspec']
        else:
            cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'], ndigits=3)
    else:
        cmeshfn += 'Omk' + sf.float2pstr(lp['Omk'], ndigits=3)
    cmeshfn += '_Omg' + sf.float2pstr(lp['Omg'], ndigits=3) + '/'

    if 'pinconf' in lp:
        if lp['pinconf'] > 0:
            cmeshfn += 'pinconf' + '{0:04d}'.format(lp['pinconf']) + '/'

    ####################################
    # Now start final part of path's name
    print 'lp[V0_pin_gauss] = ', lp['V0_pin_gauss']
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        dcdisorder = True
        cmeshfn += 'pinV' + sf.float2pstr(lp['V0_pin_gauss'], ndigits=4)
        cmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'], ndigits=4)
    else:
        dcdisorder = False

    if 'ABDelta' in lp:
        if lp['ABDelta'] > 0:
            if dcdisorder:
                cmeshfn += '_'
            cmeshfn += 'ABd' + sf.float2pstr(lp['ABDelta'], ndigits=4)

    return cmeshfn


def get_cpmeshfn(cp, lp):
    """Get the path for the specific bott calculation that uses the calc parameter dict cp on a glat with lattice
    parameters lp.
    """
    if 'rootdir' in cp:
        cpmeshfn = get_cmeshfn(lp, rootdir=cp['rootdir'])
        print '\n kfns: get_cpmeshfn(): rootdir is found in cp:'
        print 'cpmeshfn ==> ', cpmeshfn, '\n'
    else:
        print '\n kfns: get_cpmeshfn(): rootdir is NOT found in cp!'
        cpmeshfn = get_cmeshfn(lp)

    cpmeshfn = dio.prepdir(cpmeshfn)

    # Form cp subdir
    if isinstance(cp['omegac'], np.ndarray):
        if 'check' in cp:
            if cp['check']:
                print "Warning: cp['omegac'] is numpy array, using first element to get cpmeshfn..."
        cpmeshfn += 'omc' + sf.float2pstr(cp['omegac'][0], ndigits=4) + '/'
    else:
        cpmeshfn += 'omc' + sf.float2pstr(cp['omegac'], ndigits=4) + '/'

    # cpmeshfn += cp['basis'] + '/'

    # print 'cpmeshfn = ', cpmeshfn
    # sys.exit()
    return cpmeshfn


def get_bcpath(cp, lp, rootdir=None, method='varyloc'):
    """Get the path for outputting info on a collection of bott calculations that uses the calc parameter dict cp.

    Parameters
    ----------
    cp : dict
        parameters for bott collection
    lp : dict
        lattice parameters for the gyro lattice for which many botts are computed
    rootdir : str
        Root path where info for collections of botts are stored
    method : str
        String describing how to categorize the many botts

    Returns
    -------
    bcmeshfn : str
        path for the bott collection
    """
    meshfn_split = lp['meshfn'].split('/')
    ind = np.where(np.array(meshfn_split) == 'networks')[0][0]
    bcmeshfn = ''
    if rootdir is None:
        for strseg in meshfn_split[0:ind]:
            bcmeshfn += strseg + '/'
    else:
        bcmeshfn = dio.prepdir(rootdir)
    bcmeshfn += 'botts/'
    for strseg in meshfn_split[(ind + 1):]:
        bcmeshfn += strseg + '/'

    bcmeshfn += method
    # Form physics subdir for Omk, Omg, V0_pin_gauss, V0_spring_gauss
    if lp['Omk'] != -1.0:
        bcmeshfn += '_Omk' + sf.float2pstr(lp['Omk'])
    if lp['Omg'] != -1.0:
        bcmeshfn += '_Omg' + sf.float2pstr(lp['Omg'])
    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        bcmeshfn += '_pinV' + sf.float2pstr(lp['V0_pin_gauss'])
        bcmeshfn += '_sprV' + sf.float2pstr(lp['V0_spring_gauss'])

    # Form cp subdir
    if method != 'omegac':
        if isinstance(cp['omegac'], np.ndarray):
            print "Warning: cp['omegac'] is numpy array, using first element to get cpmeshfn..."
            bcmeshfn += '_omc' + sf.float2pstr(cp['omegac'][0])
        else:
            bcmeshfn += '_omc' + sf.float2pstr(cp['omegac'])

    bcmeshfn += '_Nks' + str(int(len(cp['ksize_frac_arr'])))
    bcmeshfn += '_' + cp['shape']
    bcmeshfn += '_a' + sf.float2pstr(cp['regalph'])
    bcmeshfn += '_b' + sf.float2pstr(cp['regbeta'])
    bcmeshfn += '_g' + sf.float2pstr(cp['reggamma'])
    bcmeshfn += '_polyT' + sf.float2pstr(cp['polyT'])
    if method != 'varyloc':
        bcmeshfn += '_polyoff' + sf.prepstr(cp['poly_offset']).replace('/', '_')
        bcmeshfn += '_' + cp['basis'] + '/'

    return bcmeshfn


######################################################
######################################################
######################################################
######################################################
if __name__ == '__main__':
    '''Perform an example of using bott_gyro_functions to compute the bott index'''
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')

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
