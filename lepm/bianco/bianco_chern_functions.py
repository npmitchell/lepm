import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.science_plot_style as sps
from lepm.haldane import haldane_collection
import socket
import glob
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import argparse
import copy
import sys
import polygon_functions as polyfns
import lepm.kitaev.kitaev_functions as kfns
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

'''
Compute Chern number measurements made via the kitaev realspace method.

For example usage, use if name==main clause at end of document.

'''


def translate_Nks(Nks, cp):
    cp, fps = kfns.translate_Nks(Nks, cp)
    return cp, fps


def calc_projector(haldane_lattice, omegac, attribute=False):
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

    U = eigvect.transpose()
    U1 = la.inv(U)
    MM = np.zeros((len(U), len(U)), dtype=float)

    for ii in range(len(eigval)):
        if eigval[ii] > omegac:
            MM[ii, ii] = 1.0

    # Check
    le.plot_complex_matrix(MM, show=False, close=False)
    proj = np.dot(U, np.dot(MM, U1))
    return proj


def characterize_projector(proj, haldane_lattice, index_dist=False, outdir=None, check=False):
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
    """
    if index_dist:
        for ii in range(len(proj)):
            print 'potting ', ii
            plt.scatter(np.abs(np.arange(len(proj[ii])) - ii), np.abs(proj[ii, :]), color='k', alpha=0.01)
        plt.title('Locality of projector by index')
        plt.xlabel('Index distance $|i-j|$')
        plt.ylabel('$|P_{ij}|$')
        plt.show()

    dists = le.dist_pts(haldane_lattice.lattice.xy, haldane_lattice.lattice.xy)

    # evxymag has dims len(evect) x NP (ie, NP x NP)
    evxymag = np.abs(proj)
    magproj = evxymag

    if outdir is not None:
        print 'saving dists and magproj as pickles...'
        # save dists and magproj as pkl
        with open(outdir + haldane_lattice.lp['LatticeTop'] + "_dists.pkl", "wb") as fn:
            pickle.dump(dists, fn)
        with open(outdir + haldane_lattice.lp['LatticeTop'] + "_magproj.pkl", "wb") as fn:
            pickle.dump(magproj, fn)

    return dists, magproj


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
    raise RuntimeError("Haven't finished this code yet!")
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


def projector_site_vs_dist_hlatparam(hcoll, omegac, proj_XY, plot_mag=True, plot_diff=False, save_plt=True,
                                     alpha=1.0, maxdistlines=True, reverse_order=False, check=True):
    """
    Compare the projector values at distances relative to a particular site (site at location proj_XY).

    Parameters
    ----------
    hcoll : HaldaneCollection instance
        The collection of haldane_lattices for which to compare projectors as fn of distance from a given site.
    omegac : float
        Cutoff frequency for the projector
    proj_XY : 2 x 1 float numpy array
        The location at which to find the nearest site and consider projector elements relative to this site.
    plot : bool
        Whether to plot magproj vs dist for all HaldaneLattices in hcoll
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
    raise RuntimeError("Haven't finished this code yet! Need to adapt to Haldane lattice structure")
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
    colors = lecmaps.husl_palette(n_colors=len(hcoll.haldane_lattices), s=sat, l=light)
    if reverse_order:
        hlat_list = hcoll.haldane_lattices[::-1]
    else:
        hlat_list = hcoll.haldane_lattices

    for hlat in hlat_list:
        proj_ind = ((hlat.lattice.xy - proj_XY)[:, 0]**2 + (hlat.lattice.xy - proj_XY)[:, 1]**2).argmin()
        outdir = le.prepdir(hlat.lp['meshfn'].replace('networks', 'projectors'))
        outfn = outdir + hlat.lp['LatticeTop'] + "_dist_singlept_{0:06d}".format(proj_ind) + ".pkl"

        if check:
            print 'identified proj_ind = ', proj_ind
            newfig = plt.figure()
            hlat.lattice.plot_numbered(ax=plt.gca())

        # attempt to load
        if glob.glob(outfn):
            with open(outfn, "rb") as fn:
                dist = pickle.load(fn)

            outfn = outdir + hlat.lp['LatticeTop'] + "_evxyproj_singlept_{0:06d}".format(proj_ind) + ".pkl"
            with open(outfn, "rb") as fn:
                evxyproj = pickle.load(fn)
        else:
            # compute dists and evxyproj_proj_ind
            xydiff = hlat.lattice.xy - hlat.lattice.xy[proj_ind]
            dist = np.sqrt(xydiff[:, 0]**2 + xydiff[:, 1]**2)
            print 'calculating projector...'
            proj = calc_projector(hlat, omegac, attribute=False)

            outdir = le.prepdir(hlat.lp['meshfn'].replace('networks', 'projectors'))
            le.ensure_dir(outdir)
            # save dist as pickle
            with open(outfn, "wb") as fn:
                pickle.dump(dist, fn)

            # evxyproj has dims 2 xlen(evect)
            evxyproj = np.dstack((proj[2 * proj_ind, :], proj[2 * proj_ind + 1, :]))[0].T
            # save evxyproj as pickle
            outfn = outdir + hlat.lp['LatticeTop'] + "_evxyproj_singlept_{0:06d}".format(proj_ind) + ".pkl"
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
                same_inds = np.where(le.dist_pts(hlat.lattice.xy, xy0) == 0)
                # Order them by distance
                current = np.array([[2*same_inds[0][ii], 2*same_inds[0][ii] + 1] for ii in range(len(same_inds[0]))])
                current = current.ravel()
                orig = np.array([[2 * same_inds[1][ii], 2*same_inds[1][ii] + 1] for ii in range(len(same_inds[0]))])
                orig = orig.ravel()

                # if check:
                #     origfig = plt.figure()
                #     origax = origfig.gca()
                #     [origax, origaxcb] = hlat.lattice.plot_numbered(fig=origfig, ax=origax, axis_off=False,
                #                                                     title='Original lattice for proj comparison')
                #     origax.scatter(hlat.lattice.xy[same_inds[0], 0], hlat.lattice.xy[same_inds[0], 1])
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
                [origax, origaxcb] = hlat.lattice.plot_numbered(fig=origfig, ax=origax, axis_off=False,
                                                                title='Original lattice for proj comparison')
                origfig.show()
                plt.close()
                evxyproj0 = copy.deepcopy(evxyproj)
                xy0 = copy.deepcopy(hlat.lattice.xy)
                tmp = np.sqrt(np.abs(evxyproj[0]).ravel()**2 + np.abs(evxyproj[1]).ravel()**2)
                mag0 = np.array([np.sqrt(tmp[2*ind].ravel()**2 + tmp[2*ind + 1].ravel()**2)
                                 for ind in range(len(dist))])

        kk += 1

    # Save plots
    outsplit = le.prepdir(hlat.lp['meshfn'].replace('networks', 'projectors')).split('/')
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
            print 'kfns: saving magnitude comparison plot for hcoll (usually run from kitaev_collection)...'
            mfig.savefig(outdir + hlat.lp['LatticeTop'] + "_magproj_singlept.png", dpi=300)
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
            print 'kfns: saving diff plot for hcoll projecctors (usually run from kitaev_collection)...'
            dfig.savefig(outdir + hlat.lp['LatticeTop'] + "_diffproj_singlept.png", dpi=300)
            dffig.savefig(outdir + hlat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff.png", dpi=300)
            dfax.set_ylim(-0.1, 1)
            dffig.savefig(outdir + hlat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff_zoom.png", dpi=300)
            dfax.set_ylim(-0.02, 0.1)
            dffig.savefig(outdir + hlat.lp['LatticeTop'] + "_diffproj_singlept_fractionaldiff_extrazoom.png", dpi=300)
        elif not plot_mag:
            plt.show()

    return dist_list, proj_list


def calc_bianco_chern(haldane_lattice, cp, xxt=None, yyt=None, pp=None, check=False, verbose=False,
                      vis_exten='.png', attribute_evs=False, title=None):
    """Compute the chern number for a haldane_lattice using the method of Bianco and Resta (2011)

    Parameters
    ----------
    haldane_lattice : HaldaneLattice instance
        The lattice with Haldane model physics on which to do the calculation
    cp : dict
        Chern parameter calculation dictionary
    pp : len(haldane_lattice.lattice.xy)*2 x len(haldane_lattice.lattice.xy)*2 complex array (optional)
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
        Attribute the eigval and eigvect to the HaldaneLattice instance haldane_lattice, if projector pp is not supplied

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

    # Register colormaps
    lecmaps.register_colormaps()
    imagedir = le.prepdir(cp['cpmeshfn'])
    le.ensure_dir(imagedir)

    xy = haldane_lattice.lattice.xy
    NP = len(xy)

    if pp is None:
        print 'hcfns: Computing projector...'
        pp = calc_projector(haldane_lattice, omegac, attribute=attribute_evs)

    xxt = np.zeros_like(pp, dtype=complex)
    yyt = np.zeros_like(pp, dtype=complex)
    for ii in np.arange(len(xy)):
        for jj in np.arange(len(xy)):
            xxt[ii, jj] = np.sum(pp[ii, :] * xy[:, 0] * pp[:, jj])
            yyt[ii, jj] = np.sum(pp[ii, :] * xy[:, 1] * pp[:, jj])

    # Define: tilde{X}_ij (xxt) is the sum over k of P_ik x_k P_kj
    # xxt = np.outer(np.dot(pp, xy[:, 0].T), pp.sum(axis=0))
    # yyt = np.outer(np.dot(pp, xy[:, 1].T), pp.sum(axis=0))
    # nu = -2. * np.pi * 1j * (np.dot(xxt, yyt.T) - np.dot(yyt, xxt.T))

    nu = np.zeros_like(xy[:, 0], dtype=complex)
    for ii in range(len(xxt)):
        nu[ii] = -2.*np.pi * 1j * np.dot(xxt[:, ii], yyt[ii] - np.dot(yyt[:, ii], xxt[ii, :]))

    print 'np.shape(nu) = ', np.shape(nu)
    plt.clf()
    le.plot_complex_matrix(xxt, show=True)
    le.plot_complex_matrix(yyt, show=True)
    plt.plot(np.arange(len(nu)), np.imag(nu), 'r-')
    plt.plot(np.arange(len(nu)), np.real(nu), 'b-')
    plt.show()

    # Save plot of contributions from individual haldane sites
    plt.close('all')
    filename = 'local_chern' + vis_exten
    plot_chern_local(haldane_lattice, nu, outdir=imagedir, filename=filename)

    plt.clf()
    plt.plot(np.sqrt(xy[:, 0]**2 + xy[:, 1]**2), np.real(nu), 'bo')
    plt.ylim(-1.1, 1.1)
    plt.title(r'Local Chern number, $\nu(r)$')
    plt.xlabel(r'Position, $r$')
    plt.xlabel(r'Local Chern number, $\nu$')
    plt.savefig(imagedir + 'chern_position.png')
    sys.exit()
    return nu


def get_bianco_polygon(shape, regalph, regbeta, reggamma, ksize, delta_pi=2./3., teps=1e-7, verbose=False,
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


def plot_chern_realspace(haldane_lattice, reg1_xy, reg2_xy, reg3_xy, polygon1, polygon2, polygon3,
                         ax=None, outdir=None, filename='division_lattice.png', title='', legend=False, check=False):
    """Plot the network or point set used (depending on the instance type of the input haldane_lattice) and plot the
    kitaev summation regions as colored circles, squares, and triangles on top of the sites, as well as the polygons
    associated with the 3 summation regions.

    Parameters
    ----------
    haldane_lattice : HaldaneLattice instance or NP x 2 float array
        The haldane network on which to compute the chern number (a HaldaneLattice instance), or just the points from
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

    if isinstance(haldane_lattice, np.ndarray):
        xy = haldane_lattice
        ax.scatter(xy[:, 0], xy[:, 1], c='k')
    else:
        xy = haldane_lattice.lattice.xy
        ax = le.display_lattice_2D(xy, haldane_lattice.lattice.BL,
                                   PVxydict=haldane_lattice.lattice.PVxydict,
                                   NL=haldane_lattice.lattice.NL, KL=haldane_lattice.lattice.KL,
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


def plot_chern_local(haldane_lattice, nu, ax=None, outdir=None, filename='spatial_chern.pdf'):
    """Plots the contributions of each particle to the chern number computation

    Parameters
    ----------
    haldane_lattice : HaldaneLattice instance or NP x 2 float array
        If HaldaneLattice instance, the network (with physics) to draw. If NP x 2 float array, the point set to draw.
    ax : axis instance
        axis on which to plot the contributions
    outdir : str or None
        If not None, saves plot to this directory
    filename : str
        If outdir != None, saves plot as this name in outdir
    """
    if ax is None:
        ax = plt.gca()
    # Display local chern number for each particle
    if isinstance(haldane_lattice, np.ndarray):
        xy = haldane_lattice
    else:
        xy = haldane_lattice.lattice.xy

    ax = le.display_lattice_2D(xy, haldane_lattice.lattice.BL,
                               PVxydict=haldane_lattice.lattice.PVxydict,
                               NL=haldane_lattice.lattice.NL, KL=haldane_lattice.lattice.KL,
                               bs='none', close=False, colorz=False, ptcolor=None, colormap='BlueBlackRed',
                               bgcolor='#FFFFFF', axis_off=True, ax=ax, linewidth=0.2)

    sm = ax.scatter(xy[:, 0], xy[:, 1], c=np.real(nu), edgecolor="none", cmap='bbr0', vmin=-1.0, vmax=1.0)
    plt.colorbar(sm)
    plt.title(r'Local Chern number, $C(\mathbf{r})$')
    ax.axis('equal')

    if outdir is not None:
        plt.savefig(outdir + filename)

    return ax


def get_cmeshfn(lp, rootdir=None):
    """Prepare the path where the cherns for a particular lattice are stored. If rootdir is specified, use that for the
    base of the path. Otherwise, adopt the base of the path from the lattice_params dict (lp) of the lattice.
    """
    meshfn_split = lp['meshfn'].split('/')
    ind = np.where(np.array(meshfn_split) == 'networks')[0][0]
    cmeshfn = ''
    if rootdir is None:
        for strseg in meshfn_split[0:ind]:
            cmeshfn += strseg + '/'
    else:
        cmeshfn = le.prepdir(rootdir)
    cmeshfn += 'chern_bianco/'
    for strseg in meshfn_split[(ind + 1):]:
        cmeshfn += strseg + '/'

    # Form physics subdir for t1, t2, V0_pin_gauss, V0_spring_gauss
    # Subdirs here
    if np.abs(lp['t2a']) > 1e-7:
        cmeshfn += 't2a_real/'
    elif 'pureimNNN' in lp:
        if lp['pureimNNN']:
            cmeshfn += 'pureimNNN/'
    if 'pinconf' in lp:
        if lp['pinconf'] > 0:
            cmeshfn += 'pinconf' + '{0:04d}'.format(lp['pinconf']) + '/'

    if 'pureimNNN' in lp:
        if lp['pureimNNN']:
            cmeshfn += 'pureimNNN' + '_'
    if 't2angles' in lp:
        if lp['t2angles']:
            cmeshfn += 't2angles' + '_'
    cmeshfn += 'pin' + le.float2pstr(lp['pin'])
    cmeshfn += '_t1_' + le.float2pstr(lp['t1'])
    cmeshfn += '_t2_' + le.float2pstr(lp['t2'])
    if np.abs(lp['t2a']) > 1e-7:
        cmeshfn += '_t2a_' + le.float2pstr(lp['t2a'])

    if lp['V0_pin_gauss'] > 0 or lp['V0_spring_gauss'] > 0:
        cmeshfn += '_pinV' + le.float2pstr(lp['V0_pin_gauss'])
        if lp['pinconf'] > 0:
            cmeshfn += '_pinconf' + '{0:04d}'.format(lp['pinconf'])

        cmeshfn += '_sprV' + le.float2pstr(lp['V0_spring_gauss'])
    if 'ABDelta' in lp:
        if lp['ABDelta'] > 0:
            cmeshfn += '_ABd' + le.float2pstr(lp['ABDelta'])
    return cmeshfn


def get_cpmeshfn(cp, lp):
    """Get the path for the specific haldane model's chern calculation that uses the chern parameter dict cp.
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
        cpmeshfn += '_omc' + le.float2pstr(cp['omegac'][0])
    else:
        cpmeshfn += '_omc' + le.float2pstr(cp['omegac'])
    cpmeshfn += '_Nks' + str(int(len(cp['ksize_frac_arr'])))
    cpmeshfn += '_' + cp['shape']
    cpmeshfn += '_a' + le.float2pstr(cp['regalph'])
    cpmeshfn += '_b' + le.float2pstr(cp['regbeta'])
    cpmeshfn += '_g' + le.float2pstr(cp['reggamma'])
    cpmeshfn += '_polyT' + le.float2pstr(cp['polyT'])
    cpmeshfn += '_polyoff' + le.prepstr(cp['poly_offset'])
    cpmeshfn += '_' + cp['basis'] + '/'
    return cpmeshfn


def get_ccpath(cp, lp, rootdir=None, method='varyloc'):
    """Get the path for outputting info on a collection of chern calculations that uses the chern parameter dict cp.

    Parameters
    ----------
    cp : dict
        parameters for chern collection
    lp : dict
        lattice parameters for the haldane lattice for which many cherns are computed
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
        ccmeshfn = le.prepdir(rootdir)
    ccmeshfn += 'cherns/'
    for strseg in meshfn_split[(ind + 1):]:
        ccmeshfn += strseg + '/'

    ccmeshfn += method
    # Form physics subdir for t1, t2, V0_pin_gauss, V0_spring_gauss
    if 'pureimNNN' in lp:
        if lp['pureimNNN']:
            cmeshfn += 'pureimNNN' + '_'
    if 't2angles' in lp:
        if lp['t2angles']:
            cmeshfn += 't2angles' + '_'
    cmeshfn += 'pin' + le.float2pstr(lp['pin'])
    cmeshfn += '_t1_' + le.float2pstr(lp['t1'])
    cmeshfn += '_t2_' + le.float2pstr(lp['t2'])
    if np.abs(lp['t2a']) > 1e-7:
        cmeshfn += '_t2a_' + le.float2pstr(lp['t2a'])

    # Form cp subdir
    if method != 'omegac':
        if isinstance(cp['omegac'], np.ndarray):
            print "Warning: cp['omegac'] is numpy array, using first element to get cpmeshfn..."
            ccmeshfn += '_omc' + le.float2pstr(cp['omegac'][0])
        else:
            ccmeshfn += '_omc' + le.float2pstr(cp['omegac'])

    ccmeshfn += '_Nks' + str(int(len(cp['ksize_frac_arr'])))
    ccmeshfn += '_' + cp['shape']
    ccmeshfn += '_a' + le.float2pstr(cp['regalph'])
    ccmeshfn += '_b' + le.float2pstr(cp['regbeta'])
    ccmeshfn += '_g' + le.float2pstr(cp['reggamma'])
    ccmeshfn += '_polyT' + le.float2pstr(cp['polyT'])
    if method != 'varyloc':
        ccmeshfn += '_polyoff' + le.prepstr(cp['poly_offset']).replace('/', '_')
        ccmeshfn += '_' + cp['basis'] + '/'

    return ccmeshfn


######################################################
######################################################
######################################################
######################################################
if __name__ == '__main__':
    '''Perform an example of using the lattice_collection class'''
    
    # check input arguments for timestamp (name of simulation is timestamp)
    parser = argparse.ArgumentParser(description='Specify parameters for haldane lattice funtion demos.')
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

    phi = np.pi * args.phi
    delta = np.pi * args.delta

    strain = 0.00  # initial
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

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    outdir = rootdir+'experiments/DOS_scaling/'+args.LatticeTop+'/'
    le.ensure_dir(outdir)
    lp = {'LatticeTop': args.LatticeTop,
          'NH': NH,
          'NV': NV,
          'rootdir': rootdir,
          'periodicBC': True,
          }

    # Collate DOS for many lattices
    hc = haldane_collection.HaldaneCollection()
    if args.LatticeTop == 'iscentroid':
        hc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/'+args.LatticeTop+'/'+\
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'kagome_isocent':
        hc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
            args.LatticeTop+'_square_periodic_hexner_size'+str(args.NP_load)+'_conf*_NP*'+str(args.NP_load))
    elif args.LatticeTop == 'hucentroid' or args.LatticeTop == 'kagome_hucent':
        hc.add_meshfn(
            '/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
            args.LatticeTop + '_square_periodic_d*'+'_NP*' + str(args.NP_load))
    elif args.LatticeTop == 'kagper_hucent':
        hc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/' + args.LatticeTop + '/' + \
                      args.LatticeTop + '_square_d*' + '_' + '{0:06d}'.format(NH))

    title = r'$D(\omega)$ for ' + description + ' networks'

    if args.prpoly_overlay:
        hc.ensure_all_ipr_saved()
        hc.plot_prpoly_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_prpoly_overlay',title=title)

    if args.ipr_overlay:
        hc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_ipr_overlay', title=title,
                                inverse_PR=True)
        hc.plot_ipr_DOS_overlay(outdir=outdir, fname='NP'+str(args.NP_load)+'_eigval_pr_overlay', title=title,
                                inverse_PR=False)
        hc.plot_DOS_overlay(outdir=outdir, fname= 'NP'+str(args.NP_load)+'_eigval_hist_overlay', title=title)

    if args.ipr_stack:
        # Get ylabels from meshfn names
        ylabels = []
        ii = 0
        for meshfn in hc.meshfns:
            pstring = meshfn[meshfn.index('perd')+4:meshfn.index('perd')+8].replace('p','.')
            ylabels.append(pstring)
            ii += 1

        hc.plot_ipr_DOS_stack(outdir=outdir, fname='N'+str(args.N)+'_eigval_ipr_stack', title=title,
                              ylabels=ylabels, inverse_PR=False, vmin=0.0, vmax=0.5)