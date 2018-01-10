import numpy as np
import lepm.lattice_elasticity as le
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.science_plot_style as sps
import lepm.gyro_collection
import lepm.kitaev_functions as kfns
##################
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
try:
    import cPickle as pickle
    import pickle as pl
except ImportError:
    import pickle
    import pickle as pl

"""
"""


def calc_projector_from_evs(eigval, eigvect, omegac):
    """Given eigenvalues, eigenvectors, and a cutoff frequency, create the projection operator taking states below that
    freq to zero and above to themselves

    Parameters
    ----------
    eigval : 2NP x 1 complex array
        Eigenvalues of the system
    eigvect : 2NP x 2NP complex array
        Eigenvectors of the system
    omegac : float
        cutoff frequency

    Returns
    -------
    proj : len(gyro_lattice.lattice.xy)*2 x len(gyro_lattice.lattice.xy)*2 complex array
        projection operator
    """
    U = eigvect.transpose()
    U1 = la.inv(U)
    D = np.zeros((len(U), len(U)), dtype=complex)

    for ii in range(len(eigval)):
        ev = eigval[ii]
        D[ii, ii] = ev

    MM = copy.deepcopy(D)

    MM[MM.imag < omegac] = 0
    MM[MM.imag > omegac] = 1
    proj = np.dot(U, np.dot(MM, U1))
    # h = np.zeros((len(P),len(P),len(P)),dtype=complex)
    return proj


def calc_kitaev_chern_from_evs(xy, eigval, eigvect, cp, pp=None, check=False, contributions=False, verbose=False,
                               vis_exten='.png', contrib_exten='.pdf', delta=2./3.):
    """Compute the chern number for a gyro_lattice

    Parameters
    ----------
    xy : NP x 2 float array
        points of the gyro network
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

    NP = len(xy)

    if pp is None:
        print 'Computing projector...'
        pp = calc_projector_from_evs(eigval, eigvect, omegac)

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
    if contributions and NP < 800:
        method = '2current'
        print 'constructing 2-current h_ijk...'
        hh = np.einsum('jk,kl,lj->jkl', pp, pp, pp) - np.einsum('jl,lk,kj->jkl', pp, pp, pp)
        # hh = np.zeros((len(pp), len(pp), len(pp)), dtype=complex)
        # for j in range(len(pp)):
        #     for k in range(len(pp)):
        #         for l in range(len(pp)):
        #             hh[j, k, l] = pp[j, k] * pp[k, l] * pp[l, j] - pp[j, l] * pp[l, k] * pp[k, j]
        hh *= 12 * np.pi * 1j
    else:
        method = 'projector'

    jj = 0
    # for each ksize_frac, perform sum
    for kk in range(len(ksize_frac_arr)):
        ksize_frac = ksize_frac_arr[kk]
        ksize = ksize_frac * maxsz
        if verbose:
            print 'ksize = ', ksize

        polygon1, polygon2, polygon3 = kfns.get_kitaev_polygons(shape, cp['regalph'], cp['regbeta'], cp['reggamma'],
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
            poly_offset = np.array([float(splitpo[0]), float(splitpo[1])])
            polygon1 += poly_offset
            polygon2 += poly_offset
            polygon3 += poly_offset

        # Save the previous reg1,2,3
        r1old = reg1
        r2old = reg2
        r3old = reg3

        reg1_xy = le.inds_in_polygon(xy+epskick, polygon1)
        reg2_xy = le.inds_in_polygon(xy+epskick, polygon2)
        reg3_xy = le.inds_in_polygon(xy+epskick, polygon3)

        if cp['basis'] == 'XY':
            reg1 = np.sort(np.vstack((2*reg1_xy, 2*reg1_xy+1)).ravel())
            reg2 = np.sort(np.vstack((2*reg2_xy, 2*reg2_xy+1)).ravel())
            reg3 = np.sort(np.vstack((2*reg3_xy, 2*reg3_xy+1)).ravel())
        elif cp['basis'] == 'psi':
            if verbose:
                print 'stacking regions with right-moving selves...'
            reg1 = np.sort(np.vstack((reg1, NP+reg1)).ravel())
            reg2 = np.sort(np.vstack((reg2, NP+reg2)).ravel())
            reg3 = np.sort(np.vstack((reg3, NP+reg3)).ravel())

        if contributions:
            if method == '2current':
                nu, [cb1, cb2, cb3] = kfns.sum_kitaev_with_contributions(reg1, reg2, reg3, r1old, r2old, r3old,
                                                                         hh, nu, verbose=verbose)
            else:
                nu, [cb1, cb2, cb3] = kfns.sum_kitaev_with_contributions_projector(reg1, reg2, reg3, r1old, r2old,
                                                                                   r3old, pp, nu, verbose=verbose)
            contribs['{0:0.3f}'.format(ksize)] = {'reg1': cb1, 'reg2': cb2, 'reg3': cb3}
        else:
            nu = kfns.sum_kitaev_projector(reg1, reg2, reg3, r1old, r2old, r3old, pp, nu, verbose=verbose)

        # print 'nu = ', nu
        nuV[kk] = np.real(nu)
        Nreg1V[kk] = len(reg1)
        ksize_V[kk] = ksize
        ksys_sizeV[kk] = len(reg1) + len(reg2) + len(reg3)
        ksys_fracV[kk] = ksys_sizeV[kk]/len(xy)

        # Save regions
        params_regs['{0:0.3f}'.format(ksize)] = {'reg1': reg1, 'reg2': reg2, 'reg3': reg3,
                                                 'polygon1': polygon1, 'polygon2': polygon2, 'polygon3': polygon3,
                                                 'reg1_xy': reg1_xy, 'reg2_xy': reg2_xy, 'reg3_xy': reg3_xy}
        if save_ims and (kk % modsave == 0 or kk == (len(ksize_frac_arr)-1)):
            plt.clf()
            filename = 'division_lattice_regions_{0:06d}'.format(jj) + vis_exten
            # title = r'Division of lattice: $\nu = ${0:0.3f}'.format(nu.real)
            # Commented out: plot just the regs and the title
            # plot_chern_realspace(gyro_lattice, reg1_xy, reg2_xy, reg3_xy, polygon1, polygon2, polygon3,
            #                     ax=None, outdir=imagedir, filename=filename, title=title, check=check)
            # New way: plot the regs with title plus curve of nu vs ksize
            kfns.plot_chern_realspace_2panel(xy, ksys_fracV, nuV, kk, reg1_xy, reg2_xy, reg3_xy,
                                             polygon1, polygon2, polygon3,
                                             outdir=imagedir, filename=filename, title='', check=check)

            if contributions:
                # Save plot of contributions from individual gyroscopes
                plt.clf()
                filename = 'contributions_ksize{0:0.3f}'.format(ksize_frac) + contrib_exten
                kfns.plot_chern_contributions_experiment(xy, reg1, reg2, reg3, cb1, cb2, cb3, polygon1,
                                                         polygon2, polygon3, basis=cp['basis'], outdir=imagedir,
                                                         filename=filename)

            jj += 1

    if save_ims:
        movname = cp['cpmeshfn'] + 'visualization'
        imgname = imagedir + 'division_lattice_regions_'
        print 'glob.glob(imgname) = ', glob.glob(imgname + '*')
        print 'len(glob.glob(imgname)[0]) = ', len(glob.glob(imgname + '*'))
        framerate = float(len(glob.glob(imgname + '*')))/7.0
        print 'framerate = ', framerate
        subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname+'%6d.png',
                         movname+'.mov', '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12',
                         '-threads', '0', '-r', '1', '-pix_fmt', 'yuv420p'])

    chern_finsize = np.dstack((Nreg1V, ksize_frac_arr, ksize_V, ksys_sizeV, ksys_fracV, nuV))[0]
    return chern_finsize, params_regs, contribs

