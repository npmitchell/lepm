import numpy as np
import lepm.stringformat as sf
import lepm.dataio as dio
import lepm.data_handling as dh
import lepm.line_segments as linsegs
try:
    import scipy.interpolate as interpolate
except:
    print 'Could not import scipy.interpolate!'
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
import glob
import copy
import os
import pylab
import ilpm.vector as vector
import lepm.line_segments as lseg
import lepm.plotting.plotting as leplt
import subprocess
import pickle
import cPickle
import matplotlib
from matplotlib.path import Path
import matplotlib.cm as cm
import matplotlib.patches as patches
from scipy.spatial import Delaunay
from itertools import tee, izip

try:
    import scipy
except:
    print 'WARNING: Cannot import scipy!'
import sys

'''
Description
===========
Module with auxiliary functions for creating and evolving lattices (of springs, masses, gyros)

A. Normal Mode Analysis
B. Data Handling
C. Physical Observables
D. Lattice Creation
E. Plotting
F. Saving Data
G. Files, Folders, and Directory Structure
H. Runge-Kutta functions

Common Variables
================
xy : N x 2 float array
    2D positions of points (positions x,y). Row i is the x,y position of the ith particle.
NL : array of dimension #pts x max(#neighbors)
    The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
    maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
KL : NP x max(#neighbors) int array
    spring connection/constant list, where 1 corresponds to a true connection,
    0 signifies that there is not a connection, -1 signifies periodic bond
BM : array of length #pts x max(#neighbors)
    The (i,j)th element is the bond length of the bond connecting the ith particle to its jth neighbor (the particle with index NL[i,j]).
BL : array of dimension #bonds x 2
    Each row is a bond and contains indices of connected points. Negative values denote particles connected through periodic bonds.
bL : array of length #bonds
    The ith element is the length of of the ith bond in BL
LL : tuple of 2 floats
    Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    These give the dimensions of the network in x and y, for S(k) measurements and other periodic things.
BBox : #vertices x 2 float array
    bounding polygon for the network, usually a rectangle
lp : dict
    The lattice parameters dictionary, with all keys needed for specifying path (these params vary depending on the
    value of the LatticeTop key).
eigval : 2*N x 1 complex array
    eigenvalues of the matrix, sorted by order of imaginary components
eigvect : typically 2*N x 2*N complex array
    eigenvectors of the matrix, sorted by order of imaginary components of eigvals
    Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
    x0, y0, x1, y1, ... xNP, yNP.
polygons : list of int lists
    indices of xy points defining polygons.
NLNNN : array of length #pts x max(#next-nearest-neighbors)
    Next-nearest-neighbor array: The ith row contains indices for the next nearest neighbors for the ith point.
KLNNN : array of length #pts x max(#next-nearest-neighbors)
    Next-nearest-neighbor connectivity/orientation array:
    The ith row states whether a next nearest neighbors is counterclockwise (1) or clockwise (-1)
PVxydict : dict
    dictionary of periodic bonds (keys) to periodic vectors (values)
    If key = (i,j) and val = np.array([ 5.0, 2.0]), then particle i sees particle j at xy[j]+val
    --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as 
    seen by particle i
    If network is a small periodic unit like a unit cell, such that one particle i sees particle j more than once,
    then PVxydict[(i, j)] has multiple vectors specifying each image, like
    ex. PVxydict = {(i, j): np.array([first_pv, second_pv, ...])
    In this case, denote that this is a unitcell with lp['unitcell'] == True to catch exceptions
PVx : NP x NN float array (optional, for periodic lattices)
    ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i 
    Note that shape(PVx) == shape(NL)
PVy : NP x NN float array (optional, for periodic lattices)
    ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    Note that shape(PVy) == shape(NL)
PVxnp : NP x NP float array
    ijth element of PVy is the x-component of the vector taking j = NL[i,k] to its image as seen by particle i
    Note that shape(PVxnp) == (len(xy), len(xy))
PVynp : NP x NP float array
    ijth element of PVy is the y-component of the vector taking j = NL[i,k] to its image as seen by particle i
    Note that shape(PVynp) == (len(xy), len(xy))
nljnnn : #pts x max(#NNN) int array
    nearest neighbor array matching NLNNN and KLNNN. nljnnn[i, j] gives the neighbor of i such that NLNNN[i, j] is
    the next nearest neighbor of i through the particle nljnnn[i, j]
kljnnn : #pts x max(#NNN) int array
    bond array describing periodicity of bonds matching NLNNN and KLNNN. kljnnn[i, j] describes the bond type
    (bulk -> +1, periodic --> -1) of bond connecting i to nljnnn[i, j]
klknnn : #pts x max(#NNN) int array
    bond array describing periodicity of bonds matching NLNNN and KLNNN. klknnn[i, j] describes the bond type
    (bulk -> +1, periodic --> -1) of bond connecting nljnnn[i, j] to NLNNN[i, j]
pvxnnn : NP x max(#NNN) float array
    ijth element of pvxnnn is the x-component of the vector taking NLNNN[i,j] to its image as seen by particle i
pvynnn : NP x max(#NNN) float array
    ijth element of pvynnn is the y-component of the vector taking NLNNN[i,j] to its image as seen by particle i
PV : 2 x 2 float array
    periodic lattice vectors, with x-dominant vector first, y-dominant vector second.
gxy : tuple of NX x NY arrays
    two-point correlation function in the positions of particles as function of vector distance x,y
gr :
    two-point correlation function in the positions of particles as function of scalar distance

GyroLattice attributes
======================
localization : NP x 7 float array (ie, int(len(eigval)*0.5) x 7 float array)
        fit details: x_center, y_center, A, K, uncertainty_A, covariance_AK, uncertainty_K
        for modes with positive frequency, starting near zero frequency and increasing in frequency
edge_localization : NP x 5 float array (ie, int(len(eigval)*0.5) x 5 float array)
        fit details: A, K, uncertainty_A, covariance_AK, uncertainty_K
        for modes with positive frequency, starting near zero frequency and increasing in frequency


Boundary conditions
===================
tug : give particles initial velocity
pull : displace particles continually with pullrate
offset : displace particles at boundaries
randomize : initially displace particles everywhere
fixed : displace particles at boundaries and keep fixed

For periodic boundary conditions, KL elements are taken to be -1, NL elements are normally-valued, and a periodic bond
between i and j will appear in BL as [-i, -j].
'''


########################
# Normal Mode Analysis
########################
def save_normal_modes_Nashgyro_psirep(datadir, R, NL, KL, OmK, Omg, params={}, dispersion=[], sim_type='gyro'):
    """Compute, plot, and save the normal modes of a coupled gyros with fixed pivot points.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    OmK : float or NP x NN array
        OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw'
    Omg : float or NP x 1 array
        gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw
    params : dict
        parameters dictionary
    dispersion : array or list
        dispersion relation of...
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    """
    NP = len(R)
    matrix = dynamical_matrix_psi(R, NL, KL, OmK, Omg, params, dispersion=[], sublattice_labels=[])

    plt.imshow(np.imag(matrix), interpolation='none')

    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = eig_vals_vects(matrix)

    #####################################
    # SAVE eigenvals/vects as txt file
    #####################################
    output = open(datadir + 'eigval_psi.pkl', 'wb')
    pickle.dump(eigval, output)
    output.close()

    output = open(datadir + 'eigvect_psi.pkl', 'wb')
    pickle.dump(eigvect, output)
    output.close()

    # FIND LEFT EIGENVECTORS
    AT = matrix.transpose()

    print 'Finding eigenvals/vects of dynamical matrix...'
    eigvalT, eigvectT = eig_vals_vects(AT)

    #####################################
    # SAVE eigenvals/vects as txt file
    #####################################
    output = open(datadir + 'eigval_psiT.pkl', 'wb')
    pickle.dump(eigvalT, output)
    output.close()

    output = open(datadir + 'eigvect_psiT.pkl', 'wb')
    pickle.dump(eigvectT, output)
    output.close()

    return eigvect, eigval, matrix


def plot_pcolormesh_scalar(x, y, C, outpath, title, xlabel=None, ylabel=None, title2='', subtext='', subsubtext='',
                           vmin='auto', vmax='auto', cmap="coolwarm", show=False, close=True, axis_on=True, FSFS=20):
    """This function is mirrored from lepm.plotting.plotting.
    Save a single-panel plot of a scalar quantity C as colored pcolormesh

    Parameters
    ----------
    x, y : NxN mesh arrays
        the x and y positions of the points evaluated to Cx, Cy
    C : NxN arrays
        values for the plotted quantity C evaluated at points (x,y)
    outpath : string
        full name with file path
    title : string
        title of the plot
    title2 : string
        placed below title
    subtext : string
        placed below plot
    subsubtext : string
        placed at bottom of image
    vmin, vmax : float
        minimum, maximum value of C for colorbar; default is range of values in C
    cmap : matplotlib colormap
    show : bool
        whether to display the plot for interactive viewing
    close : bool
        whether to close the plot at end of function
    axis_on : bool
        if False, axis labels will be removed
    """
    import lepm.plotting.plotting as leplt
    return leplt.plot_pcolormesh_scalar(x, y, C, outpath, title, xlabel=None, ylabel=None, title2='', subtext='',
                                        subsubtext='', vmin='auto', vmax='auto', cmap="coolwarm", show=False,
                                        close=True, axis_on=True, FSFS=20)


def plot_real_matrix(M, name='', outpath=None, fig='auto', climv=None, cmap="coolwarm", show=False, close=True,
                     fontsize=None):
    """This function is mirrored from lepm.plotting.plotting.
    Plot matrix as colored subplot, with red positive and blue negative.

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    clear : bool (default == True)
        Whether to clear the plot after saving or showing
    Returns
    ----------

    """
    import lepm.plotting.plotting as leplt
    return leplt.plot_real_matrix(M, name='', outpath=None, fig='auto', climv=None, cmap="coolwarm", show=False, close=True,
                     fontsize=None)


def plot_complex_matrix(M, name='', outpath=None, fig='auto', climvs=[], show=False, close=True, fontsize=None):
    """This function is mirrored from lepm.plotting.plotting.
    Plot real and imaginary parts of matrix as two subplots

    Parameters
    ----------
    M : complex array
        matrix to plot
    name : string
        name to save plot WITHOUT extension (png)
    outpath : string (default='none' -> no saving)
        Directory and name of file as which to save plot. If outpath is None or 'none', does not save plot.
    fig : matplotlib figure instance
        The figure to use for the plots
    clims : list of two lists
        Real and imaginary plot colorlimits, as [[real_lower, real_upper], [imag_lower, imag_upper]]
    show : bool (default == False)
        Whether to show the plot (and force user to close it to continue)
    close : bool (default == True)
        Whether to clear the plot after saving or showing
    fontsize : int
        The font size for the title, if name is not empty

    Returns
    ----------

    """
    # unpack or set colorlimit values
    import lepm.plotting.plotting as leplt
    fig, (ax1, ax2), (cbar1, cbar2) = leplt.plot_complex_matrix(M, name=name, outpath=outpath, fig=fig,
                                                                climvs=climvs, show=show,
                                                                close=close, fontsize=fontsize)
    return fig, (ax1, ax2), (cbar1, cbar2)


def plot_projection_complex_vects(P, eigvect, eigval, inds2proj, fig='auto', outpath=None, show=True, close=True):
    """
    Plot the projection of complex vectors, plotting real and imaginary parts separately of originals and projections.
    """
    if fig == 'auto':
        fig = plt.gcf()
        plt.clf()

    a1 = fig.add_subplot(2, 2, 1)
    a1.set_title('Imaginary part')
    a2 = fig.add_subplot(2, 2, 2)
    a2.set_title('Real part')
    a3 = fig.add_subplot(2, 2, 3)
    a4 = fig.add_subplot(2, 2, 4)
    for i in inds2proj:
        proj = np.dot(P, eigvect[i].transpose())
        a4.plot(np.arange(len(eigvect[i])), np.real(eigvect[i]),
                label=r'$\omega=$' + '{0:0.3f}'.format(eigval[i].real) + '+ i({0:0.3f})'.format(eigval[i].imag))
        a2.plot(np.arange(len(eigvect[i])), np.real(proj),
                label=r'$P e_{\omega}$, $\omega=$' + '{0:0.3f}'.format(eigval[i].real) + '+ i({0:0.3f})'.format(
                    eigval[i].imag))
        a3.plot(np.arange(len(eigvect[i])), np.imag(eigvect[i]),
                label=r'$\omega=$' + '{0:0.3f}'.format(eigval[i].real) + '+ i({0:0.3f})'.format(eigval[i].imag))
        a1.plot(np.arange(len(eigvect[i])), np.imag(proj),
                label=r'$P e_{\omega}$, $\omega=$' + '{0:0.3f}'.format(eigval[i].real) + '+ i({0:0.3f})'.format(
                    eigval[i].imag))

    # Shrink current axis by 20%
    box = a1.get_position()
    a1.set_position([box.x0 - box.width * 0.2, box.y0, box.width * 0.8, box.height])
    box = a2.get_position()
    a2.set_position([box.x0 - box.width * 0.4, box.y0, box.width * 0.8, box.height])
    box = a3.get_position()
    a3.set_position([box.x0 - box.width * 0.2, box.y0, box.width * 0.8, box.height])
    box = a4.get_position()
    a4.set_position([box.x0 - box.width * 0.4, box.y0, box.width * 0.8, box.height])

    a2.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, fancybox=True, fontsize=9)
    a4.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, fancybox=True, fontsize=9)

    if outpath != None and outpath != 'none':
        print 'outputting complex matrix image to ', outpath
        plt.savefig(outpath + '.png')
    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, PVxydict=None, params={}, dispersion=[],
                                    sim_type='gyro', save_pkl=True, rm_images=True, save_ims=True, gapims_only=True,
                                    eigval=None, eigvect=None, matrix=None, lattice_color='k'):
    """Compute, plot, and save the normal modes of a coupled gyros with fixed pivot points.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)
    This function does not require a GyroLattice instance, but instead relies on all its necessary attributes.

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    OmK : float or NP x NN array
        OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw'
    Omg : float or NP x 1 array
        gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw
    params : dict
        parameters dictionary, optional, currently not used since normal_modes_gyros doesn't use it either
    dispersion : array or list
        dispersion relation of...
    sim_type : str
        'gyro' --> indicates that we are using the gyro DM construction
    save_pkl : bool
        save the DOS as pickle
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    save_ims : bool
        Save the DOS eigenvect images (could be deleted after making a movie of the images if rm_images == True)
    gapims_only : bool
        Only plot eigenvectors with eigvalues near the middle of the spectrum
    eigval : 2NP x 1 complex array or None
        If not None, uses this as the eigenvalues of the system (skips the computation)
    eigvect : 2NP x 2NP complex array or None
        If not None, uses this as the eigenvectors of the system (skips the computation)
    matrix : 2NP x 2NP complex or float array
        The dynamical matrix, if already known. If eigval and eigvect are supplied, then this is not needed (only
        needed to pass to output).
    """
    NP = len(R)
    if eigval is None and eigvect is None:
        matrix = dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params=params, PVxydict=PVxydict, dispersion=[],
                                        sublattice_labels=[])
        # plot_complex_matrix(matrix, show=True)
        print 'Finding eigenvals/vects of dynamical matrix...'
        eigval, eigvect = eig_vals_vects(matrix)

    # Plot histogram of eigenvalues if saving at all
    if datadir is not None:
        fig, DOS_ax = leplt.initialize_DOS_plot(eigval, 'gyro', pin=-5000)
        plt.savefig(datadir + 'eigval_hist.png')

    # prepare DOS output dir
    if save_ims:
        dio.ensure_dir(datadir + 'DOS/')

    #####################################
    # SAVE eigenvals/vects as txt file
    #####################################
    if save_pkl:
        print 'Saving eigvals/vects as txt files...'
        output = open(datadir + 'eigval.pkl', 'wb')
        pickle.dump(eigval, output)
        output.close()

        output = open(datadir + 'eigvect.pkl', 'wb')
        pickle.dump(eigvect, output)
        output.close()
        # np.savetxt()

    if save_ims:
        print 'Saving DOS images...'
        #####################################
        # Prepare for plotting
        #####################################
        # Determine b,c
        omk = OmK[np.where(abs(OmK) > 0.)[0], np.where(abs(OmK) > 0)[1]]
        omg = Omg.ravel()[np.where(abs(Omg.ravel()) > 0.)[0]]
        if (omg == 0).all():
            '''Omg is purely zeros'''
            omg = Omg.ravel()[0:NP]
            omg_purely_zero = True
        else:
            omg_purely_zero = False

        # omk > 0, omg > 0: b=0,c=1
        # omk > 0, omg < 0: b=1,c=0
        # omk < 0, omg > 0: b=1,c=1
        # omk < 0, omg < 0: b=0,c=0
        # Check if uniform/homogenous omk and omg
        # and make string for labelling spin direction and whether
        # pendulum is standing or hanging.
        if (omk == omk[0]).all() and not omg_purely_zero:
            if (omg == omg[0]).all():
                do_bc = True
                # Find values for b and c
                if omk[0] > 0:
                    if omg[0] > 0:
                        bstr = 'b=0'
                        cstr = 'c=1'
                        btmp = 0
                        ctmp = 1
                    elif omg[0] < 0:
                        bstr = 'b=1'
                        cstr = 'c=0'
                        btmp = 1
                        ctmp = 0
                elif omk[0] < 1:
                    if omg[0] > 0:
                        bstr = 'b=1'
                        cstr = 'c=1'
                        btmp = 1
                        ctmp = 1
                    elif omg[0] < 0:
                        bstr = 'b=0'
                        cstr = 'c=0'
                        btmp = 0
                        ctmp = 0
            else:
                do_bc = False
                bstr = 'b and/or c are mixed sign'
                cstr = ''
        else:
            do_bc = False
            bstr = 'b and/or c are mixed sign or Omg=0.'
            cstr = ''

        bcstr = bstr + ', ' + cstr

        if do_bc:
            # temporarily store the homogeneous pin frequency as pin
            pin = omg[0]
        else:
            pin = -5000

        print 'plotting...'
        fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, R, sim_type=sim_type)
        leplt.lattice_plot(R, NL, KL, eig_ax, linecolor=lattice_color)

        # Make strings for spring, pin, k, and g values
        if do_bc:
            springstr_Hz = '{0:.03f}'.format(omk[0] / (2. * np.pi))
            pinstr_Hz = '{0:.03f}'.format(omg[0] / (2. * np.pi))
        else:
            springstr_Hz = ''
            pinstr_Hz = ''

        # If small system, find analytic eigval solns
        if len(R) < 5:
            try:
                exact_eigvals = check_explicit_eigvals(len(R), b[0], c[0], params={})
                exactstr = '\n' + str(exact_eigvals)
            except:
                print 'Function check_explicit_eigvals is not written yet...'
                exactstr = ''
        else:
            exactstr = ''

        text2show = 'spring = ' + springstr_Hz + ' Hz,  pin = ' + pinstr_Hz + ' Hz\n' + \
                    exactstr + '\n' + bcstr
        fig.text(0.4, 0.1, text2show, horizontalalignment='center', verticalalignment='center')

        # Add schematic of hanging/standing top spinning with dir
        if do_bc:
            schem_ax = plt.axes([0.85, 0.0, .025 * 5, .025 * 7], axisbg='w')
            # drawing
            schem_ax.plot([0., 0.2], [1 - btmp, btmp], 'k-')
            schem_ax.scatter([0.2], [btmp], s=150, c='k')
            schem_ax.arrow(0.2, btmp, -(-1) ** ctmp * 0.06, 0.3 * (-1) ** (btmp + ctmp), \
                           head_width=0.3, head_length=0.1, fc='b', ec='b')
            wave_x = np.arange(-0.07 * 5, 0.0, 0.001)
            wave_y = 0.1 * np.sin(wave_x * 100) + 1. - btmp
            schem_ax.plot(wave_x, wave_y, 'k-')
            schem_ax.set_xlim(-0.1 * 5, .21 * 5)
            schem_ax.set_ylim(-0.1 * 7, .21 * 7)
            # schem_ax.axis('equal')
            schem_ax.axis('off')

        #####################################
        # SAVE eigenvals/vects as images
        #####################################

        done_pngs = len(glob.glob(datadir + 'DOS/DOS_*.png'))
        # check if normal modes have already been done
        if not done_pngs:
            totN = len(eigval)
            if done_pngs < totN:
                # decide on which eigs to plot
                if gapims_only:
                    middle = int(round(totN * 0.25))
                    ngap = int(round(np.sqrt(totN)))
                    todo = range(middle - ngap, middle + ngap)
                else:
                    todo = range(int(round(len(eigval) * 0.5)))

                dmyi = 0
                for ii in todo:
                    if np.mod(ii, 50) == 0:
                        print 'plotting eigvect ', ii, ' of ', len(eigval)
                    fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                        leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type, NL, KL,
                                                         marker_num=0, color_scheme='default', sub_lattice=-1, )
                    plt.savefig(datadir + 'DOS/DOS_' + '{0:05}'.format(dmyi) + '.png')
                    scat_fg.remove()
                    scat_fg2.remove()
                    p.remove()
                    f_mark.remove()
                    lines_12_st.remove()
                    dmyi += 1

        fig.clf()
        plt.close('all')

        ######################
        # Save DOS as movie
        ######################
        imgname = datadir + 'DOS/DOS_'
        names = datadir.split('/')[0:-1]
        # Construct movie name from datadir path string
        movname = ''
        for ii in range(len(names)):
            if ii < len(names) - 1:
                movname += names[ii] + '/'
            else:
                movname += names[ii]

        movname += '_DOS'

        subprocess.call(
            ['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v', 'main',
             '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

        if rm_images:
            # Delete the original images
            print 'Deleting folder ' + datadir + 'DOS/'
            subprocess.call(['rm', '-r', datadir + 'DOS/'])

    return eigvect, eigval, matrix


def plot_movie_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, dispersion=[], sim_type='gyro',
                                     rm_images=True, gapims_only=True, save_into_subdir=False):
    """
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.
    Note that b and c are built INTO the signs of OmK and Omg.
    --> b (0 -> 'hang',  1 -> 'stand').
    --> c (0 -> aligned with a,  1 -> aligned with a)

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    OmK : float or NP x NN array
        OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw'
    Omg : float or NP x 1 array
        gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw
    params : dict
        parameters dictionary
    dispersion : array or list
        dispersion relation of...
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    gapims_only : bool
        Whether to just plot modes near the middle of the DOS frequency range
    save_into_subdir: bool
        Whether to save the movie in the same sudir where the DOS/ directory (containing the frames) is placed.
    """
    NP = len(R)
    print 'Loading eigenvals/vects of dynamical matrix...'
    # Find eigval
    with open(datadir + 'eigval.pkl', "rb") as input_file:
        eigval = cPickle.load(input_file)
    with open(datadir + 'eigvect.pkl', "rb") as input_file:
        eigvect = cPickle.load(input_file)

    # prepare DOS output dir
    dio.ensure_dir(datadir + 'DOS/')

    #####################################
    # Prepare for plotting
    #####################################
    print 'Preparing plot settings...'
    # Determine b,c
    omk = OmK[np.where(abs(OmK) > 0.)[0], np.where(abs(OmK) > 0)[1]]
    omg = Omg.ravel()[np.where(abs(Omg.ravel()) > 0.)[0]]

    # omk > 0, omg > 0: b=0,c=1
    # omk > 0, omg < 0: b=1,c=0
    # omk < 0, omg > 0: b=1,c=1
    # omk < 0, omg < 0: b=0,c=0
    # Check if uniform/homogenous omk and omg
    # and make string for labelling spin direction and whether
    # pendulum is standing or hanging.
    if (omk == omk[0]).all():
        if (omg == omg[0]).all():
            do_bc = True
            # Find values for b and c
            if omk[0] > 0:
                if omg[0] > 0:
                    bstr = 'b=0'
                    cstr = 'c=1'
                    btmp = 0
                    ctmp = 1
                elif omg[0] < 0:
                    bstr = 'b=1'
                    cstr = 'c=0'
                    btmp = 1
                    ctmp = 0
            elif omk[0] < 1:
                if omg[0] > 0:
                    bstr = 'b=1'
                    cstr = 'c=1'
                    btmp = 1
                    ctmp = 1
                elif omg[0] < 0:
                    bstr = 'b=0'
                    cstr = 'c=0'
                    btmp = 0
                    ctmp = 0
        else:
            do_bc = False
            bstr = 'b and/or c are mixed sign'
            cstr = ''
    else:
        do_bc = False
        bstr = 'b and/or c are mixed sign'
        cstr = ''

    bcstr = bstr + ', ' + cstr

    if do_bc:
        # temporarily store the homogeneous pin frequency as pin
        pin = omg[0]
    else:
        pin = -5000

    print 'plotting...'
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, R, NL, KL, sim_type=sim_type, pin=pin)

    # Make strings for spring, pin, k, and g values
    if do_bc:
        springstr_Hz = '{0:.03f}'.format(omk[0] / (2. * np.pi))
        pinstr_Hz = '{0:.03f}'.format(omg[0] / (2. * np.pi))
    else:
        springstr_Hz = ''
        pinstr_Hz = ''

    text2show = 'spring = ' + springstr_Hz + ' Hz,  pin = ' + pinstr_Hz + ' Hz\n' + '\n' + bcstr
    fig.text(0.4, 0.1, text2show, horizontalalignment='center', verticalalignment='center')

    # Add schematic of hanging/standing top spinning with dir
    if do_bc:
        schem_ax = plt.axes([0.85, 0.0, .025 * 5, .025 * 7], axisbg='w')
        # drawing
        schem_ax.plot([0., 0.2], [1 - btmp, btmp], 'k-')
        schem_ax.scatter([0.2], [btmp], s=150, c='k')
        schem_ax.arrow(0.2, btmp, -(-1) ** ctmp * 0.06, 0.3 * (-1) ** (btmp + ctmp), \
                       head_width=0.3, head_length=0.1, fc='b', ec='b')
        wave_x = np.arange(-0.07 * 5, 0.0, 0.001)
        wave_y = 0.1 * np.sin(wave_x * 100) + 1. - btmp
        schem_ax.plot(wave_x, wave_y, 'k-')
        schem_ax.set_xlim(-0.1 * 5, .21 * 5)
        schem_ax.set_ylim(-0.1 * 7, .21 * 7)
        # schem_ax.axis('equal')
        schem_ax.axis('off')

    #####################################
    # SAVE eigenvals/vects as images
    #####################################

    done_pngs = len(glob.glob(datadir + 'DOS/DOS_*.png'))
    # check if normal modes have already been done
    if not done_pngs:
        # decide on which eigs to plot
        totN = len(eigval)
        if gapims_only:
            middle = int(round(totN * 0.25))
            ngap = int(round(np.sqrt(totN)))
            todo = range(middle - ngap, middle + ngap)
        else:
            todo = range(int(round(len(eigval) * 0.5)))

        if done_pngs < len(todo):
            dmyi = 0
            for ii in todo:
                if np.mod(ii, 50) == 0:
                    print 'plotting eigvect ', ii, ' of ', len(eigval)
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type, NL, KL,
                                                     marker_num=0, color_scheme='default', sub_lattice=-1)
                plt.savefig(datadir + 'DOS/DOS_' + '{0:05}'.format(dmyi) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()
                dmyi += 1

    fig.clf()
    plt.close('all')

    ######################
    # Save DOS as movie
    ######################
    imgname = datadir + 'DOS/DOS_'
    names = datadir.split('/')[0:-1]
    # Construct movie name from datadir path string
    movname = ''
    for ii in range(len(names)):
        if ii < len(names) - 1:
            movname += names[ii] + '/'
        else:
            if save_into_subdir:
                movname += names[ii] + '/' + names[ii] + '_DOS'
            else:
                movname += names[ii]
                movname += '_DOS'

    subprocess.call(
        ['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v', 'main', '-crf',
         '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    if rm_images:
        # Delete the original images
        print 'Deleting folder ' + datadir + 'DOS/'
        subprocess.call(['rm', '-r', datadir + 'DOS/'])


# def plot_movie_normal_modes_mass(datadir, R, NL, KL, OmK, Omg, sim_type='mass', rm_images=True,
#                                  save_ims=True, gapims_only=True):
#     """
#     Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.
#     Note that b and c are built INTO the signs of OmK and Omg.
#     --> b (0 -> 'hang',  1 -> 'stand').
#     --> c (0 -> aligned with a,  1 -> aligned with a)
#
#     Parameters
#     ----------
#     datadir: string
#         directory where simulation data is stored
#     R : NP x dim array
#         position array in 2d (3d might work with 3rd dim ignored)
#     NL : NP x NN array
#         Neighbor list
#     KL : NP x NN array
#         spring connectivity array
#     OmK : float or NP x NN array
#         OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw'
#     Omg : float or NP x 1 array
#         gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw
#     params : dict
#         parameters dictionary
#     dispersion : array or list
#         dispersion relation of...
#     rm_images : bool
#         Whether or not to delete all the images after a movie has been made of the DOS
#     """
#     NP = len(R)
#     print 'Loading eigenvals/vects of harmonic dynamical matrix...'
#     # Find eigval
#     with open(datadir + 'eigval_mass.pkl', "rb") as input_file:
#         eigval = cPickle.load(input_file)
#     with open(datadir + 'eigvect_mass.pkl', "rb") as input_file:
#         eigvect = cPickle.load(input_file)
#
#     # prepare DOS output dir
#     dio.ensure_dir(datadir + 'DOS_mass/')
#
#     #####################################
#     # Prepare for plotting
#     #####################################
#     print 'Preparing plot settings...'
#     # Determine b,c
#     omk = OmK[np.where(abs(OmK) > 0.)[0], np.where(abs(OmK) > 0)[1]]
#     omg = Omg.ravel()[np.where(abs(Omg.ravel()) > 0.)[0]]
#
#     print 'plotting...'
#     fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, R, NL, KL, sim_type=sim_type, pin=-5000)
#
#     #####################################
#     # SAVE eigenvals/vects as images
#     #####################################
#
#     done_pngs = len(glob.glob(datadir + 'DOS_mass/DOS_*.png'))
#     # check if normal modes have already been done
#     if not done_pngs:
#         # decide on which eigs to plot
#         totN = len(eigval)
#         if gapims_only:
#             middle = int(round(totN * 0.25))
#             ngap = int(round(np.sqrt(totN)))
#             todo = range(middle - ngap, middle + ngap)
#         else:
#             todo = range(int(round(len(eigval) * 0.5)))
#
#         if done_pngs < len(todo):
#             dmyi = 0
#             for ii in todo:
#                 if np.mod(ii, 50) == 0:
#                     print 'plotting eigvect ', ii, ' of ', len(eigval)
#                 fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
#                     leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type, NL, KL,
#                                                      marker_num=0, color_scheme='default', sub_lattice=-1)
#                 plt.savefig(datadir + 'DOS_mass/DOS_' + '{0:05}'.format(dmyi) + '.png')
#                 scat_fg.remove()
#                 scat_fg2.remove()
#                 p.remove()
#                 f_mark.remove()
#                 lines_12_st.remove()
#                 dmyi += 1
#
#     fig.clf()
#     plt.close('all')
#
#     ######################
#     # Save DOS as movie
#     ######################
#     imgname = datadir + 'DOS_mass/DOS_'
#     movname = datadir + 'DOS_mass'
#     subprocess.call(['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v',
#                      'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])
#
#     if rm_images:
#         # Delete the original images
#         print 'Deleting folder ' + datadir + 'DOS_mass/'
#         subprocess.call(['rm', '-r', datadir + 'DOS_mass/'])


def plot_movie_normal_modes_mass(datadir, R, NL, KL, kk, mass, sim_type='mass', rm_images=True,
                                 save_ims=True, gapims_only=True):
    """New version 20170506, with no pendulum component -- uses just masses and spring constants
    Plot the normal modes of a coupled gyros with fixed pivot points and make a movie of them.

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    kk : float or NP x NN array
        spring constants for each spring
    mass : float or NP x 1 array
        masses of each site
    params : dict
        parameters dictionary
    dispersion : array or list
        dispersion relation of...
    rm_images : bool
        Whether or not to delete all the images after a movie has been made of the DOS
    """
    NP = len(R)
    print 'Loading eigenvals/vects of harmonic dynamical matrix...'
    # Find eigval
    with open(datadir + 'eigval_massspring.pkl', "rb") as input_file:
        eigval = cPickle.load(input_file)
    with open(datadir + 'eigvect_massspring.pkl', "rb") as input_file:
        eigvect = cPickle.load(input_file)

    # prepare DOS output dir
    dio.ensure_dir(datadir + 'DOS_massspring/')

    #####################################
    # Prepare for plotting
    #####################################
    print 'plotting...'
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, R, NL, KL, sim_type=sim_type, pin=-5000)

    #####################################
    # SAVE eigenvals/vects as images
    #####################################

    done_pngs = len(glob.glob(datadir + 'DOS_massspring/DOS_*.png'))
    # check if normal modes have already been done
    if not done_pngs:
        # decide on which eigs to plot
        totN = len(eigval)
        if gapims_only:
            middle = int(round(totN * 0.25))
            ngap = int(round(np.sqrt(totN)))
            todo = range(middle - ngap, middle + ngap)
        else:
            todo = range(int(round(len(eigval) * 0.5)))

        if done_pngs < len(todo):
            dmyi = 0
            for ii in todo:
                if np.mod(ii, 50) == 0:
                    print 'plotting eigvect ', ii, ' of ', len(eigval)
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type, NL, KL,
                                                     marker_num=0, color_scheme='default', sub_lattice=-1)
                plt.savefig(datadir + 'DOS_massspring/DOS_' + '{0:05}'.format(dmyi) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()
                dmyi += 1

    fig.clf()
    plt.close('all')

    ######################
    # Save DOS as movie
    ######################
    imgname = datadir + 'DOS_massspring/DOS_'
    movname = datadir + 'DOS_massspring'
    subprocess.call(['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v',
                     'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    if rm_images:
        # Delete the original images
        print 'Deleting folder ' + datadir + 'DOS_massspring/'
        subprocess.call(['rm', '-r', datadir + 'DOS_massspring/'])


def heavy_fixCM_eigvals(NP, b, c, params):
    """If the system is small, return the exact eigenvalues and print the exact DOS matrix, for checking.

    Parameters
    ----------
    NP : int
        number of points
    b : int (0 or 1)
        hanging (0) or standing (1) gyro
    c : int (0 or 1)
        L aligned with a or anti-aligned with a
        if w3<0 c=0, if w3>0 c=1
    params : dict
        parameters for this simulation
    """
    l = params['l']
    k = params['k']
    I3 = params['I3']
    # Here, omega_3 is just the MAGNITUDE, not signed
    w3 = np.abs(params['w3'][0])
    gn = params['Mm'] * params['g']

    # Check output if small system
    print 'gn = ', gn
    print 'b = ', b
    print 'c = ', c

    if NP == 1:
        pass
    elif NP == 2:
        matrix = -np.array([[0., (-1) ** (1 + c) * l * gn / (I3 * w3), 0., 0.],
                            [(-1) ** (1 + c) * (-l * gn + (-1) ** (1 + b) * l ** 2 * k) / (I3 * w3), 0.,
                             (-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0.],
                            [0., 0., 0., (-1) ** (1 + c) * l * gn / (I3 * w3)],
                            [(-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0.,
                             (-1) ** (1 + c) * (-l * gn + (-1) ** (1 + b) * l ** 2 * k) / (I3 * w3), 0.]
                            ])
        print 'exact matrix = ', matrix
        eigvals = np.array([
            1j * l * gn / (I3 * w3),
            -1j * l * gn / (I3 * w3),
            l * np.sqrt(gn) * np.sqrt(0j - 2. * l * k * (-1) ** (b) - gn) / (I3 * w3),
            -l * np.sqrt(gn) * np.sqrt(0j - 2. * l * k * (-1) ** (b) - gn) / (I3 * w3)
        ])
        print 'exact_eigvals are =', eigvals
        return eigvals
    elif NP == 3:
        matrix = -np.array([[0., (-1) ** (1 + c) * l * gn / (I3 * w3), 0., 0., 0., 0.],
                            [(-1) ** (1 + c) * (-l * gn + (-1) ** (1 + b) * l ** 2 * k) / (I3 * w3), 0.,
                             (-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0., 0., 0.],
                            [0., 0., 0., (-1) ** (1 + c) * l * gn / (I3 * w3), 0., 0.],
                            [(-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0.,
                             (-1) ** (1 + c) * (-l * gn - 2. * (-1) ** (b) * l ** 2 * k) / (I3 * w3), 0., \
                             (-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0.],
                            [0., 0., 0., 0., 0., (-1) ** (1 + c) * l * gn / (I3 * w3)],
                            [0., 0., (-1) ** (1 + b + c) * l ** 2 * k / (I3 * w3), 0.,
                             (-1) ** (1 + c) * (-l * gn + (-1) ** (1 + b) * l ** 2 * k) / (I3 * w3), 0.]
                            ])
        print 'exact matrix = ', matrix

        eigvals = np.array([
            1j * l * gn / (I3 * w3),
            # -1j*l*gn/(I3*w3),
            l * np.sqrt(gn) * np.sqrt(0j - 3. * l * k * (-1) ** (b) - gn) / (I3 * w3),
            # -l*np.sqrt(gn)*np.sqrt(0j-3.*l*k*(-1)**(b) - gn)/(I3*w3),
            l * np.sqrt(gn) * np.sqrt(0j - l * k * (-1) ** (b) - gn) / (I3 * w3),
            # -l*np.sqrt(gn)*np.sqrt(0j -  l*k*(-1)**(b) - gn)/(I3*w3)
        ])
        return eigvals
    else:
        return np.array([])


def plot_save_normal_modes(datadir, R, NL, KL, params, dispersion=[], spin_dir=[], b='hang',
                           spring='auto', pin='auto', sim_type='gHST_massive', rm_images=False):
    """Compute, plot, and save the normal modes of a coupled gliding Heavy Symmetric Top.

    Parameters
    ----------
    datadir: string
        directory where simulation data is stored
    R : NP x dim array
        position array in 2d (3d might work with 3rd dim ignored)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    params : dict
        parameters dictionary
    dispersion : array or list
    spin_dir :
        If empty or 1, assumes spin direction is aligned with body axis 3 (antialigned with a),
        if -1, sends spin direction to be antialigned with body axis 3 (aligned with a)
        if spin_dir ==1, c=-1. Elif spin_dir==-1, c=0
    b : string or int (default is 'hang')
        parameter which determines if gyros are standing or hanging: ('hang'->0 'stand'->1).
        Used to define exponent of (-1): zero or one for theta ~ pi or theta ~ 0, respectively.
    spring : float or NP x NN array
        k*l**2/I3*omega3, if 'auto', then computes quantity from params
    pin : float or NP x 1 array
        l*gn/I3*omega3, if 'auto', then computes quantity from params
    """
    NP = len(R)

    if spring == 'auto':
        # FIX THIS TO USE A MATRIX OF SPRING CONSTANTS NOT A VECTOR
        spring = params['k'] * params['l'] ** 2 / (params['I3'] * np.abs(params['w3']))
        # If there is more than one particle, and if the speeds vary from particle to particle,
        # then make spring the same length as a dynamical matrix column
        if len(spring) > 0:
            # Check if all nonzero elems are identical.
            # Assumes that spring[0,0] is nonzero! --> could use np.where to get first nonzero elem instead
            checkMat = spring - spring[0]
            # set checkMat=0 for all zero entries of spring
            checkMat[checkMat == - spring[0]] = 0
            if (abs(checkMat) > 1e-9).any():
                # The rotation rates vary from particle to particle, so reshape
                spring_new = np.zeros_like(spring)
                dmyi = 0  # a new index ('dummy i')
                for ii in range(NP):
                    # Since 2 dof for position of pivot of gHST, double the size
                    spring_new[dmyi] = spring[ii]
                    spring_new[dmyi + 1] = spring[ii]
                    dmyi += 2
            else:
                # the elements are all identical, so just keep the first one
                spring = spring[0]

    if pin == 'auto':
        gn = params['Mm'] * params['g']
        pin = params['l'] * gn / (params['I3'] * np.abs(params['w3']))
        # If there is more than one particle, and if the speeds vary from particle to particle,
        # then make pin the same length as a dynamical matrix column
        if len(pin) > 0:
            if (abs(pin - pin[0]) > 1e-9).any():
                # The rotation rates vary from particle to particle, so reshape
                pin_new = np.zeros_like(pin)
                dmyi = 0  # a new index ('dummy i')
                for ii in range(NP):
                    # Since 2 dof for position of pivot of gHST, double the size
                    pin_new[dmyi] = pin[ii]
                    pin_new[dmyi + 1] = pin[ii]
                    dmyi += 2
            else:
                # the elements are all identical, so just keep the first one
                pin = pin[0]

    # define c =0,1 for aligned, antialigned with a
    if not spin_dir:
        spin_dir = params['w3'] > 0
        spin_dir[spin_dir == 0] = -1
    elif isinstance(spin_dir, int) or isinstance(spin_dir, float):
        if spin_dir == 1:
            '''aligned with body axis 3 --> c=1'''
            spin_dir = np.ones(NP)
        elif spin_dir == -1:
            '''antialigned with body axis 3 --> c=0'''
            spin_dir = - np.ones(NP)

    if b == 'hang':
        print 'Defining b as zeros (NP x 1 array)...'
        b = np.zeros(NP)
    elif b == 'stand':
        print 'Defining b as ones (NP x 1 array)...'
        b = np.ones(NP)

    matrix = normal_modes_gHST(R, NL, KL, params, dispersion=dispersion, spin_dir=spin_dir, b=b, spring=spring, pin=pin)

    # check output if  small system
    if len(matrix) < 10:
        print 'matrix = ', matrix

    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = eig_vals_vects(matrix)
    print 'plotting...'
    fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, R, NL, KL, sim_type=sim_type, pin=pin)

    # Make strings for spring, pin, k, and g values
    if isinstance(spring, np.float64):
        springstr_Hz = '{0:.03f}'.format(spring / (2. * np.pi))
    else:
        springstr_Hz = '{0:.03f}'.format(spring[0] / (2. * np.pi))
    if isinstance(pin, np.float64):
        pinstr_Hz = '{0:.03f}'.format(pin / (2. * np.pi))
    else:
        pinstr_Hz = '{0:.03f}'.format(pin[0] / (2. * np.pi))
    if isinstance(params['k'], float):
        kstr = '{0:.03f}'.format(params['k'])
    else:
        kstr = '{0:.03f}'.format(params['k'][0])
    if isinstance(params['g'], float):
        gstr = '{0:.03f}'.format(params['g'])
    else:
        gstr = '{0:.03f}'.format(params['g'][0])
    if isinstance(params['Mm'], float):
        mstr = '{0:.03f}'.format(params['Mm'])
    else:
        mstr = '{0:.03f}'.format(params['Mm'][0])
    if isinstance(params['l'], float):
        lstr = '{0:.03f}'.format(params['l'])
    else:
        lstr = '{0:.03f}'.format(params['l'][0])

    c = np.round(0.5 * spin_dir + 0.5).astype(int)

    # check output if  small system
    if len(spin_dir) < 5:
        print '\n\n\nspin_dir = ', spin_dir
        print 'so \n c =', c

    # If small system, find analytic eigval solns
    if len(R) < 5:
        exact_eigvals = heavy_fixCM_eigvals(len(R), b[0], c[0], params)
        exactstr = '\n' + str(exact_eigvals)
    else:
        exactstr = ''

    # make string for labelling spin direction and whether pendulum is standing or hanging
    if b.size > 1:
        if (b == b[0]).all():
            bstr = 'b=' + str(b[0])
            do_b = True  # to do schematic, based on uniform b value
            if b[0] == 0:
                b_tmp = 0
            else:
                b_tmp = 1
        else:
            bstr = 'b=mix'
            do_b = False
    else:
        bstr = ' c=' + str(b)
        do_b = True
        b_tmp = b[0]

    if c.size > 1:
        if (c == c[0]).all():
            cstr = ' c=' + str(c[0])
            if c[0] == 0:
                c_tmp = 0
            else:
                print 'c[0] =', c[0], '  ---> 1'
                c_tmp = 1
            do_c = True  # to do schematic, based on uniform c value
        else:
            cstr = ' c=mix'
            do_c = False
    else:
        cstr = ' c=' + str(c)
        do_c = True
        c_tmp = c[0]

    bcstr = bstr + cstr

    text2show = 'spring = ' + springstr_Hz + ' Hz,  pin = ' + pinstr_Hz + ' Hz\n' + \
                'k = ' + kstr + ', g = ' + gstr + ', m = ' + mstr + ', l = ' + lstr + \
                exactstr + '\n' + bcstr
    fig.text(0.4, 0.1, text2show, horizontalalignment='center', verticalalignment='center')

    # Add schematic of hanging/standing top spinning with dir
    if do_b and do_c:
        schem_ax = plt.axes([0.85, 0.0, .025 * 5, .025 * 7], axisbg='w')
        # drawing
        schem_ax.plot([0., 0.2], [1 - b_tmp, b_tmp], 'k-')
        schem_ax.scatter([0.2], [b_tmp], s=150, c='k')
        schem_ax.arrow(0.2, b_tmp, -(-1) ** c_tmp * 0.06, 0.3 * (-1) ** (b_tmp + c_tmp),
                       head_width=0.3, head_length=0.1, fc='b', ec='b')
        wave_x = np.arange(-0.07 * 5, 0.0, 0.001)
        wave_y = 0.1 * np.sin(wave_x * 100) + 1. - b_tmp
        schem_ax.plot(wave_x, wave_y, 'k-')
        schem_ax.set_xlim(-0.1 * 5, .21 * 5)
        schem_ax.set_ylim(-0.1 * 7, .21 * 7)
        # schem_ax.axis('equal')
        schem_ax.axis('off')

    # prepare DOS output dir
    dio.ensure_dir(datadir + 'DOS/')

    #####################################
    # SAVE eigenvals/vects as txt file
    #####################################
    output = open(datadir + 'eigval.pkl', 'wb')
    pickle.dump(eigval, output)
    output.close()

    output = open(datadir + 'eigvect.pkl', 'wb')
    pickle.dump(eigvect, output)
    output.close()

    #####################################
    # SAVE eigenvals/vects as images
    #####################################
    done_pngs = len(glob.glob(datadir + 'DOS/DOS_*.png'))
    # check if normal modes have already been done
    if not done_pngs:
        if done_pngs < len(eigval):
            for ii in range(int(round(len(eigval) * 0.5))):
                if np.mod(ii, 50) == 0:
                    print 'plotting eigvect ', ii, ' of ', len(eigval)
                fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                    leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type,
                                                     NL, KL, marker_num=0, color_scheme='default', sub_lattice=-1)
                plt.savefig(datadir + 'DOS/DOS_' + '{0:05}'.format(ii) + '.png')
                scat_fg.remove()
                scat_fg2.remove()
                p.remove()
                f_mark.remove()
                lines_12_st.remove()

    fig.clf()
    plt.close('all')

    ######################
    # Save DOS as movie
    ######################
    imgname = datadir + 'DOS/DOS_'
    names = datadir.split('/')[0:-1]
    # Construct movie name from datadir path string
    movname = ''
    for ii in range(len(names)):
        if ii < len(names) - 1:
            movname += names[ii] + '/'
        else:
            movname += names[ii]

    movname += '_DOS'

    subprocess.call(
        ['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v', 'main', '-crf',
         '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    if rm_images:
        # Delete the original images
        subprocess.call(['rm', '-r', datadir + 'DOS/'])

    # CHECK FOR REAL EIGENVALUES, if exist, make plots and movie without them as well
    if (np.abs(eigval.real) > 1e-9).any():
        # make indices for just the eigenvalues with imaginary components
        imIND_tmp = np.ones_like(eigval)
        imIND_tmp[np.abs(eigval.real) > 1e-9] = 0
        imIND = np.where(imIND_tmp)[0]

        if len(imIND) != len(eigval):
            eigval = eigval[imIND]
            eigvect = eigvect[imIND]

            print 'plotting just imaginary eigval plots...'
            fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_plot(eigval, R, NL, KL, sim_type=sim_type, pin=pin)

            # Make strings for spring, pin, k, and g values
            if isinstance(spring, np.float64):
                springstr_Hz = '{0:.03f}'.format(spring / (2. * np.pi))
            else:
                springstr_Hz = '{0:.03f}'.format(spring[0] / (2. * np.pi))
            if isinstance(pin, np.float64):
                pinstr_Hz = '{0:.03f}'.format(pin / (2. * np.pi))
            else:
                pinstr_Hz = '{0:.03f}'.format(pin[0] / (2. * np.pi))
            if isinstance(params['k'], float):
                kstr_Hz = '{0:.03f}'.format(params['k'])
            else:
                kstr_Hz = '{0:.03f}'.format(params['k'][0])
            if isinstance(params['g'], float):
                gstr_Hz = '{0:.03f}'.format(params['g'])
            else:
                gstr_Hz = '{0:.03f}'.format(params['g'][0])

            text2show = 'spring = ' + springstr_Hz + ' Hz,  pin = ' + pinstr_Hz + ' Hz\n' + \
                        'k = ' + kstr_Hz + ', g = ' + gstr_Hz
            fig.text(0.5, 0.05, text2show, horizontalalignment='center', verticalalignment='center')

            # prepare DOS output dir
            dio.ensure_dir(datadir + 'DOS_imag/')

            #####################################
            # SAVE eigenvals/vects as txt file
            #####################################
            output = open(datadir + 'eigval_imag.pkl', 'wb')
            pickle.dump(eigval, output)
            output.close()

            output = open(datadir + 'eigvect_imag.pkl', 'wb')
            pickle.dump(eigvect, output)
            output.close()

            #####################################
            # SAVE eigenvals/vects as images
            #####################################
            done_pngs = len(glob.glob(datadir + 'DOS_imag/DOS_*.png'))
            # check if normal modes have already been done
            if not done_pngs:
                if done_pngs < len(eigval):
                    for ii in range(len(eigval)):
                        if np.mod(ii, 50) == 0:
                            print 'plotting eigvect ', ii, ' of ', len(eigval)
                        fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                            leplt.construct_eigvect_DOS_plot(R, fig, DOS_ax, eig_ax, eigval, eigvect, ii, sim_type,
                                                             NL, KL, marker_num=0, color_scheme='default',
                                                             sub_lattice=-1)
                        plt.savefig(datadir + 'DOS_imag/DOS_' + '{0:05}'.format(ii) + '.png')
                        scat_fg.remove()
                        scat_fg2.remove()
                        p.remove()
                        f_mark.remove()
                        lines_12_st.remove()

            fig.clf()
            plt.close('all')

            ######################
            # Save DOS as movie
            ######################
            imgname = datadir + 'DOS_imag/DOS_'
            names = datadir.split('/')[0:-1]
            # Construct movie name from datadir path string
            movname = ''
            for ii in range(len(names)):
                if ii < len(names) - 1:
                    movname += names[ii] + '/'
                else:
                    movname += names[ii]

            movname += '_DOS_imag'

            subprocess.call(
                ['./ffmpeg', '-i', imgname + '%05d.png', movname + '.mov', '-vcodec', 'libx264', '-profile:v', 'main',
                 '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

            if rm_images:
                # Delete the original images
                subprocess.call(['rm', '-r', datadir + 'DOS_imag/'])


def normal_modes_gHST(R, NL, KL, params, dispersion=[], spin_dir=[], sublattice_labels=[], b='hang', spring='auto',
                      pin='auto'):
    """Calculates the matrix for finding the normal modes of the system
    Assumes that all bonds are unit length.

    Parameters
    ----------
    R : NP x dim array
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    dispersion : array or list
    spin_dir :
        If empty or 1, assumes spin direction is aligned with body axis 3 (antialigned with a),
        if -1, sends spin direction to be antialigned with body axis 3 (aligned with a)
    b : string (default is 'hang')
        parameter which determines if gyros are standing or hanging: ('hang' 'stand').
        Used to define exponent of (-1): zero or one for theta ~ pi or theta ~ 0, respectively.
    spring : float or NP x 1 array
        k*l**2/I3*omega3, if 'auto', then computes quantity from params
    pin : float or NP x 1 array
        l*gn/I3*omega3, if 'auto', then computes quantity from params
    """
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle.'''
        NP = 1
        NN = 0

    M1 = np.zeros((2 * NP, 2 * NP))
    M2 = np.zeros((2 * NP, 2 * NP))
    if spring == 'auto':
        spring = params['k'] * params['l'] ** 2 / (params['I3'] * np.abs(params['w3']))
        # If there is more than one particle, and if the speeds vary from particle to particle,
        # then make spring the same length as a dynamical matrix column
        if len(spring) > 0:
            if (abs(spring - spring[0]) > 1e-9).any():
                # The rotation rates vary from particle to particle, so reshape
                spring_new = np.zeros_like(spring)
                dmyi = 0  # a new index ('dummy i')
                for ii in range(NP):
                    # Since 2 dof for position of pivot of gHST, double the size
                    spring_new[dmyi] = spring[ii]
                    spring_new[dmyi + 1] = spring[ii]
                    dmyi += 2
            else:
                # the elements are all identical, so just keep the first one
                spring = spring[0]

    if pin == 'auto':
        gn = params['Mm'] * params['g']
        pin = params['l'] * gn / (params['I3'] * np.abs(params['w3']))
        # If there is more than one particle, and if the speeds vary from particle to particle,
        # then make pin the same length as a dynamical matrix column
        if len(pin) > 0:
            if (abs(pin - pin[0]) > 1e-9).any():
                # The rotation rates vary from particle to particle, so reshape
                pin_new = np.zeros_like(pin)
                dmyi = 0  # a new index ('dummy i')
                for ii in range(NP):
                    # Since 2 dof for position of pivot of gHST, double the size
                    pin_new[dmyi] = pin[ii]
                    pin_new[dmyi + 1] = pin[ii]
                    dmyi += 2
            else:
                # the elements are all identical, so just keep the first one
                pin = pin[0]

    m2_shape = np.shape(M2)

    if b == 'hang':
        b = np.zeros(NP)
    elif b == 'stand':
        b = np.ones(NP)

    if spin_dir == []:
        '''Assume antialigned with a, aligned with body axis 3'''
        spin_dir = np.ones(NP)

    print 'Constructing dynamical matrix...'
    for i in range(NP):
        for nn in range(NN):

            ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
            k = KL[i, nn]  # true connection?

            if len(dispersion) > 1:
                disp = 1. / (1. + dispersion[i])
            else:
                disp = 1.

            diffx = R[ni, 0] - R[i, 0]
            diffy = R[ni, 1] - R[i, 1]
            alphaij = 0.

            rij_mag = np.sqrt(diffx ** 2 + diffy ** 2)

            if k != 0:
                alphaij = np.arctan2(diffy, diffx)

            # for periodic systems, KL is -1 for particles on opposing boundaries
            if KL[i, nn] == -1:
                alphaij = (np.pi + alphaij) % (2 * pi)

                # What is this for?
            if KL[i, nn] == -2:  # will only happen on first or last gyro in a line
                if i == 0 or i == (NP - 1):
                    print i, '--> NL=-2 for this particle'
                    yy = np.where(KL[i] == 1)
                    dx = R[NL[i, yy], 0] - R[NL[i, yy], 0]
                    dy = R[NL[i, yy], 1] - R[NL[i, yy], 1]
                    al = (np.arctan2(dy, dx)) % (2 * pi)
                    alphaij = np.pi - al
                    if i == 1:
                        alphaij = np.pi - ((90 / 2) * np.pi / 180.)
                    else:
                        alphaij = - ((90 / 2) * np.pi / 180.)

            Cos = np.cos(alphaij)
            Sin = np.sin(alphaij)

            if abs(Cos) < 10E-8:
                Cos = 0.0

            if abs(Sin) < 10E-8:
                Sin = 0

            Cos2 = Cos ** 2
            Sin2 = Sin ** 2
            CosSin = Cos * Sin

            # -1 for aligned with a, 1 for aligned with 3.
            # dir factor :== 1/(-1)^c = (-1)^c
            dir_factor = spin_dir[i]

            if len(sublattice_labels) > 0:
                if sublattice_labels[i] == 1:
                    extra_factor = 1. * del_A_B
                    # print self.del_A_B
                elif sublattice_labels[i] == 0:
                    extra_factor = 1.
                else:
                    extra_factor = 1.
            else:
                extra_factor = 1.

            M1[2 * i, 2 * i] += -disp * k * CosSin * ((-1) ** b[i]) * dir_factor  # dxi - dxi
            M1[2 * i, 2 * i + 1] += -disp * k * Sin2 * ((-1) ** b[i]) * dir_factor  # dxi - dyi
            M1[2 * i, 2 * ni] += disp * k * CosSin * ((-1) ** b[i]) * dir_factor  # dxi - dxj
            M1[2 * i, 2 * ni + 1] += disp * k * Sin2 * ((-1) ** b[i]) * dir_factor  # dxi - dyj

            # (y components)
            M1[2 * i + 1, 2 * i] += disp * k * Cos2 * ((-1) ** b[i]) * dir_factor  # dyi - dxi
            M1[2 * i + 1, 2 * i + 1] += disp * k * CosSin * ((-1) ** b[i]) * dir_factor  # dyi - dyi
            M1[2 * i + 1, 2 * ni] += -disp * k * Cos2 * ((-1) ** b[i]) * dir_factor  # dyi - dxj
            M1[2 * i + 1, 2 * ni + 1] += -disp * k * CosSin * ((-1) ** b[i]) * dir_factor  # dyi - dyj

            # if i==0:
            #    print '\n --- \n added M1[2*i+1, 2*i] = ',disp*k*Cos2   *((-1)**b[i]) *dir_factor
            #    print 'dir_factor = ', dir_factor
            #    print 'k = ', k
            #    print 'else =', ((-1)**b[i]) *dir_factor

            # pinning/gravitational matrix
            M2[2 * i, 2 * i + 1] = (1.) * disp * dir_factor * extra_factor
            M2[2 * i + 1, 2 * i] = -(1.) * disp * dir_factor * extra_factor

            # self.pin_array.append(2*pi*1*extra_factor)
    # Assumes:
    # (-1)**c adot =  - spring* (-1)**b SUM{ z x nij*(nij.(dri-drj)) } + pin
    matrix = - (-spring * M1 + pin * M2)

    return matrix


def eig_vals_vects(matrix, sort='imag', not_hermitian=False, verbose=False):
    """
    Finds the eigenvalues and eigenvectors of dynamical matrix.

    Parameters
    ----------
    matrix : N x M matrix
        matrix to diagonalize
    sort : str
        string specifier for how to sort the output eigenvector array and eigvals (by ascending 'imag' or 'real'
        components)
    not_hermitian : bool
        if False, will check for shortcuts in diagonalization routine by being hermitian

    Returns
    ----------
    eigval_out : 2*N x 1 complex array
        eigenvalues of the matrix, sorted by order of imaginary components
    eigvect_out : typically 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    """
    # if len(matrix) < 10:
    #     print '\nFinding eigvals, matrix = ', matrix

    # check if hermitian:
    if not_hermitian:
        eigval, eigvect = np.linalg.eig(matrix)
    else:
        if (matrix == matrix.conj().T).all():
            print 'Shortcut eigvect/vals since matrix is hermitian...'
            eigval, eigvect = np.linalg.eigh(matrix)
        else:
            if verbose:
                print 'matrix is not hermitian...'
            eigval, eigvect = np.linalg.eig(matrix)

    # use imaginary part to get ascending order of eigvals
    if sort == 'imag':
        si = np.argsort(np.imag(eigval))
    elif sort == 'real':
        si = np.argsort(np.real(eigval))
    else:
        si = np.arange(len(eigval))

    eigvect = np.array(eigvect)
    eigvect_out = eigvect.T[si]
    eigval_out = eigval[si]

    # if len(eigval_out) < 10:
    #     print 'eigvals return as =', eigval_out

    return eigval_out, eigvect_out


def eig_vals_vects_hermitian(matrix, sort='imag'):
    """Finds the eigenvalues and eigenvectors of hermitian matrix

    Parameters
    ----------
    matrix : N x M matrix
        matrix to diagonalize

    Returns
    ----------
    eigval_out : 2*N x 1 complex array
        eigenvalues of the matrix, sorted by order of imaginary components
    eigvect_out : typically 2*N x 2*N complex array
        eigenvectors of the matrix, sorted by order of imaginary components of eigvals
        Eigvect is stored as NModes x NP*2 array, with x and y components alternating, like:
        x0, y0, x1, y1, ... xNP, yNP.
    """
    # if len(matrix) < 10:
    #     print '\nFinding eigvals, matrix = ', matrix
    eigval, eigvect = np.linalg.eig(matrix)
    # use imaginary part to get ascending order of eigvals
    if sort == 'imag':
        si = np.argsort(np.imag(eigval))
    elif sort == 'real':
        si = np.argsort(np.real(eigval))
    else:
        si = np.arange(len(eigval))

    eigvect = np.array(eigvect)
    eigvect_out = eigvect.T[si]
    eigval_out = eigval[si]
    if len(eigval_out) < 10:
        print 'eigvals return as =', eigval_out
    return eigval_out, eigvect_out


##########################################
# Data Handling
##########################################
# Functions for handling boundaries

def remove_dangling_points(xy, NL, KL, BL, check=False):
    """

    Parameters
    ----------
    xy
    NL
    KL
    BL

    Returns
    -------

    """
    dangles = np.where(~KL.any(axis=1))[0]
    if len(dangles) > 0:
        print 'le: extract_boundary: Removing dangling points: dangles = ', dangles
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.plot(xy[dangles, 0], xy[dangles, 1], 'ro')
            plt.title('Original point indices, before removing dangles. Dangles circled in red.')
            plt.show()

        NP = len(xy)

        nondangles = np.setdiff1d(np.arange(NP), dangles)
        # Note that remove_pts can handle periodic BL
        xy, NL, KL, BL, PVxydict = remove_pts(nondangles, xy, BL)

        # Remove bonds which were periodic.
        pbonds = np.where(KL.ravel() < 0)[0]
        print 'le: pbonds = ', pbonds
        if pbonds:
            print 'le: Found periodic bonds in extract_boundary(), clearing...'
            KLr = KL.ravel()
            KLr[pbonds] = 0
            KL = KLr.reshape(np.shape(KL))
            print 'le: pbonds = ', pbonds

        if check:
            print 'le: NL = ', NL
            display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Removed points in extract_boundary()')

        # xy = xy[nondangles]
        # NL = NL[nondangles]
        # KL = KL[nondangles]

        # translation converts indices of long old xy to small new xy
        # backtrans converts indices of small, new xy to indices of long, old xy
        #      .1                                 .0
        #   .0           trans ----->
        #       . 2      <----- backtrans           .1
        #  .3                                 .2
        translation = np.arange(NP, dtype=int)
        for IND in dangles:
            translation[IND:] -= 1
            # mark the removed point by -5
            translation[IND] = -5

        backtrans = np.where(translation > -1)[0]
        if check:
            print 'le: backtrans = ', backtrans
            print 'le: translation = ', translation
    else:
        backtrans = None

    return dangles, xy, NL, KL, BL, backtrans


def bond_angles_wrt_bond(current, next, xy, NL, KL):
    """Return the angles of bonds connected to particle indexed by next.

    Parameters
    ----------
    current : int
        The index of particle (indexing xy) for whom we measure bond angles
    next : int
        The particle for whom we measure bond angles
    NL :

    KL :

    Returns
    -------
    angles : #neighbors x 1 float array
        the bond angles of bond emanating from next to its neighbors
    neighbors : #neighbors x 1 int array
        The neighbors of next
    """
    n_tmp = NL[next, np.argwhere(KL[next].ravel())]
    if len(n_tmp) == 1:
        print 'le: The bond is a lone bond, not part of a triangle, so returning neighbor as next particle'
        neighbors = n_tmp
    else:
        neighbors = np.delete(n_tmp, np.where(n_tmp == current)[0])
    # print 'n_tmp = ', n_tmp
    # print 'neighbors = ', neighbors
    angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[next, 1],
                               xy[neighbors, 0] - xy[next, 0]).ravel() -
                    np.arctan2(xy[current, 1] - xy[next, 1],
                               xy[current, 0] - xy[next, 0]).ravel(),
                    2 * np.pi)
    return angles, neighbors


def extract_boundary(xy, NL, KL, BL, check=False):
    """Extract the boundary of a 2D network (xy,NL,KL). If periodic, discards this information, so this returns the
    openBC boundary.

    Parameters
    ----------
    xy : NP x 2 float array
        point set in 2D
    BL : #bonds x 2 int array
        Bond list
    NL : NP x NN int array
        Neighbor list. The ith row contains the indices of xy that are the bonded pts to the ith pt.
        Nonexistent bonds are replaced by zero.
    KL : NP x NN int array
        spring connection/constant array, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    check: bool
        Whether to show intermediate results

    Returns
    ----------
    boundary : #points on boundary x 1 int array
        indices of points living on boundary of the network
    """
    # Clear periodic bonds from KL
    pbonds = np.where(KL.ravel() < 0)[0]
    if len(pbonds) > 0:
        print 'le: Found periodic bonds in le.extract_boundary(), clearing...'
        KLr = KL.ravel()
        KLr[pbonds] = 0
        KL = KLr.reshape(np.shape(KL))
        print 'le: pbonds = ', pbonds

    # If there are dangling points, remove them for now and adjust indices later
    dangles = np.where(~KL.any(axis=1))[0]
    if len(dangles) > 0:
        print 'le: extract_boundary: Removing dangling points: dangles = ', dangles
        if check:
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.plot(xy[dangles, 0], xy[dangles, 1], 'ro')
            plt.title('Original point indices, before removing dangles. Dangles circled in red.')
            plt.show()

        translate_at_end = True

        NP = len(xy)

        nondangles = np.setdiff1d(np.arange(NP), dangles)
        # Note that remove_pts can handle periodic BL
        xy, NL, KL, BL, PVxydict = remove_pts(nondangles, xy, BL)

        # Remove bonds which were periodic.
        pbonds = np.where(KL.ravel() < 0)[0]
        print 'le: pbonds = ', pbonds
        if pbonds:
            print 'le: Found periodic bonds in extract_boundary(), clearing...'
            KLr = KL.ravel()
            KLr[pbonds] = 0
            KL = KLr.reshape(np.shape(KL))
            print 'le: pbonds = ', pbonds

        if check:
            print 'le: NL = ', NL
            display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Removed points in extract_boundary()')

        # xy = xy[nondangles]
        # NL = NL[nondangles]
        # KL = KL[nondangles]

        # translation converts indices of long old xy to small new xy
        # backtrans converts indices of small, new xy to indices of long, old xy
        #      .1                                 .0
        #   .0           trans ----->
        #       . 2      <----- backtrans           .1
        #  .3                                 .2
        translation = np.arange(NP, dtype=int)
        for IND in dangles:
            translation[IND:] -= 1
            # mark the removed point by -5
            translation[IND] = -5

        backtrans = np.where(translation > -1)[0]
        if check:
            print 'le: backtrans = ', backtrans
            print 'le: translation = ', translation

            # translation = np.where()

    else:
        translate_at_end = False

    # Initialize the list of boundary indices to be larger than necessary
    bb = np.zeros(2 * len(xy), dtype=int)

    # Start with the rightmost point, which is guaranteed to be
    # at the convex hull and thus also at the outer edge.
    # Then take the first step to be along the minimum angle bond
    rightIND = np.where(xy[:, 0] == max(xy[:, 0]))[0]
    # If there are more than one rightmost point, choose one
    if rightIND.size > 1:
        rightIND = rightIND[0]

    if check:
        print 'le.extract_boundary(): Found rightmost pt: ', rightIND
        print 'le.extract_boundary():   with neighbors: ', NL[rightIND]
        print 'le.extract_boundary():   with connectns: ', KL[rightIND]
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'bo')
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
        plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'ro')
        plt.pause(0.01)

    # Grab the true neighbors of this starting point
    print 'le.extract_boundary(): NL[rightIND, :] = ', NL[rightIND, :]
    neighbors = NL[rightIND, np.argwhere(KL[rightIND].ravel()).ravel()]
    print 'le.extract_boundary(): neighbors = ', neighbors
    print 'le.extract_boundary(): rightIND = ', rightIND

    # Compute the angles of the neighbor bonds
    angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[rightIND, 1], xy[neighbors, 0] - xy[rightIND, 0]).ravel(),
                    2 * np.pi)
    if check:
        print 'KL[rightIND] = ', KL[rightIND]
        print 'KL[rightIND,0] = ', KL[rightIND, 0]
        print 'KL[rightIND,0] ==0 ', KL[rightIND, 0] == 0
        print 'np.argwhere(KL[rightIND]) = ', np.argwhere(KL[rightIND])
        print 'np.argwhere(KL[rightIND].ravel())= ', np.argwhere(KL[rightIND].ravel())
        print 'neighbors = ', neighbors
        print 'angles = ', angles

    # Take the second particle to be the one with the lowest bond angle (will be >= pi/2)
    # print ' angles==min--> ', angles==min(angles)
    nextIND = neighbors[angles == min(angles)][0]
    bb[0] = rightIND

    dmyi = 1
    # as long as we haven't completed the full outer edge/boundary, add nextIND
    while nextIND != rightIND:
        # print '\n nextIND = ', nextIND
        # print 'np.argwhere(KL[nextIND]) = ', np.argwhere(KL[nextIND]).ravel()
        bb[dmyi] = nextIND
        angles, neighbors = bond_angles_wrt_bond(bb[dmyi - 1], nextIND, xy, NL, KL)
        nextIND = neighbors[angles == min(angles)][0]
        # print 'nextIND = ', nextIND

        if check:
            # plt.plot(xy[:,0],xy[:,1],'k.')
            XY = np.vstack([xy[bb[dmyi], :], xy[nextIND, :]])
            plt.plot(XY[:, 0], XY[:, 1], 'r-')
            # for i in range(len(xy)):
            #    plt.text(xy[i,0]+0.2,xy[i,1],str(i))
            plt.gca().set_aspect('equal')
            plt.pause(0.01)

        dmyi += 1

    # Truncate the list of boundary indices
    boundary = bb[0:dmyi]

    # Since some points were removed from the boundary identification, translate
    # indices back to indices of original xy
    if translate_at_end:
        print 'le.extract_boundary(): Translating boundary points back into original indices...'
        # print 'boundary = ', boundary
        # print 'translation = ', translation
        # print 'backtrans = ', backtrans
        boundary = backtrans[boundary]

    return boundary


def extract_inner_boundary(xy, NL, KL, BL, check=False):
    """Extract the boundary on the interior of an annular sample or the polygon at the center, and return the indices
    of the particles composing this boundary in clockwise order (note that opposite orientation from outer boundary).
    If there are periodic boundary conditions, this function discards that information.

    Parameters
    ----------
    xy : NP x 2 float array
        point set in 2D
    BL : #bonds x 2 int array
        Bond list
    NL : NP x NN int array
        Neighbor list. The ith row contains the indices of xy that are the bonded pts to the ith pt.
        Nonexistent bonds are replaced by zero.
    KL : NP x NN int array
        Connectivity list. The jth column of the ith row ==1 if pt i is bonded to pt NL[i,j].
        The jth column of the ith row ==0 if pt i is not bonded to point NL[i,j].
    check: bool
        Whether to show intermediate results

    Returns
    ----------
    boundary : #points on inner boundary x 1 int array
        indices of points living on inner boundary of the network
    """
    # Clear periodic bonds from KL
    pbonds = np.where(KL.ravel() < 0)[0]
    if len(pbonds) > 0:
        print 'le: Found periodic bonds in le.extract_inner_boundary(), clearing...'
        KLr = KL.ravel()
        KLr[pbonds] = 0
        KL = KLr.reshape(np.shape(KL))
        print 'le: pbonds = ', pbonds

    # If there are dangling points, remove them for now and adjust indices later
    dangles, xy, NL, KL, BL, backtrans = remove_dangling_points(xy, NL, KL, BL, check=check)
    translate_at_end = len(dangles) > 0

    # Initialize the list of boundary indices to be larger than necessary
    bb = np.zeros(2 * len(xy), dtype=int)

    # Start with the centermost point that is on the right side of the y axis, which is guaranteed to be
    # at the convex hull for an annular sample and thus also at the inner edge.
    # Then take the first step to be along the minimum angle bond
    # Compute radial distance of each particle
    distr2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
    xpositive = np.where(xy[:, 0] > 0)[0]
    if translate_at_end:
        # avoid choosing a dangling particle with no bonds
        selection = np.intersect1d(xpositive, nodangles)
        rightIND = np.where(distr2 == np.min(distr2[selection]))[0]
    else:
        rightIND = np.where(distr2 == np.min(distr2[xpositive]))[0]
    # print 'rightIND = ', rightIND
    # plt.plot(xy[:, 0], xy[:, ])
    # for ii in range(len(xy)):
    #     plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
    # plt.show()
    # sys.exit()
    # If there are more than one rightmost point, choose one
    if rightIND.size > 1:
        rightIND = rightIND[0]

    if check:
        print 'le.extract_inner_boundary(): Found innermost pt: ', rightIND
        print 'le.extract_inner_boundary():   with neighbors: ', NL[rightIND]
        print 'le.extract_inner_boundary():   with connectns: ', KL[rightIND]
        plt.plot(xy[:, 0], xy[:, 1], 'k.')
        plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'bo')
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
        plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'ro')
        plt.pause(0.1)

    # Grab the true neighbors of this starting point
    print 'le.extract_inner_boundary(): NL[rightIND, :] = ', NL[rightIND, :]
    neighbors = NL[rightIND, np.argwhere(KL[rightIND].ravel()).ravel()]
    print 'le.extract_inner_boundary(): neighbors = ', neighbors
    print 'le.extract_inner_boundary(): rightIND = ', rightIND

    # Take the second particle to be the one with the smallest bond angle above pi (might be <= 3pi/2, but not
    # necessarily).
    # Compute the angles of the neighbor bonds and add pi
    angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[rightIND, 1], xy[neighbors, 0] - xy[rightIND, 0]).ravel() + np.pi,
                    2 * np.pi)
    nextIND = neighbors[angles == min(angles)][0]
    bb[0] = rightIND
    dmyi = 1

    if check:
        print 'KL[rightIND] = ', KL[rightIND]
        print 'KL[rightIND,0] = ', KL[rightIND, 0]
        print 'KL[rightIND,0] ==0 ', KL[rightIND, 0] == 0
        print 'np.argwhere(KL[rightIND]) = ', np.argwhere(KL[rightIND])
        print 'np.argwhere(KL[rightIND].ravel())= ', np.argwhere(KL[rightIND].ravel())
        print 'neighbors = ', neighbors
        print 'angles = ', angles

    # This part, commented out, was a red herring
    # It is possible for the first particle to be attached to only one other site. If this is the case, then we need to
    # add its neighbor to the bb array and take the next max angle with respect to that bond instead of the min angle.
    # while len(angles) == 1:
    #     print 'le.extract_inner_boundary(): there is only one neighbor for the first identified boundary particle'
    #     bb[dmyi] = nextIND
    #     angles, neighbors = bond_angles_wrt_bond(bb[dmyi - 1], nextIND, xy, BL, KL)
    #     nextIND = neighbors[angles == max(angles)][0]
    #     # print 'nextIND = ', nextIND

    print 'bb = ', bb
    # sys.exit()
    # as long as we haven't completed the full outer edge/boundary, add nextIND
    while nextIND != rightIND:
        # print '\n nextIND = ', nextIND
        # print 'np.argwhere(KL[nextIND]) = ', np.argwhere(KL[nextIND]).ravel()
        bb[dmyi] = nextIND
        angles, neighbors = bond_angles_wrt_bond(bb[dmyi - 1], nextIND, xy, NL, KL)
        nextIND = neighbors[angles == min(angles)][0]
        # print 'nextIND = ', nextIND

        if check:
            plt.plot(xy[:,0],xy[:,1],'k.')
            XY = np.vstack([xy[bb[dmyi], :], xy[nextIND, :]])
            plt.plot(XY[:, 0], XY[:, 1], 'r-')
            for i in range(len(xy)):
               plt.text(xy[i,0] + 0.2, xy[i, 1], str(i))
            plt.gca().set_aspect('equal')
            plt.show()

        dmyi += 1

    # Truncate the list of boundary indices
    inner_boundary = bb[0:dmyi]

    # Since some points were removed from the boundary identification, translate
    # indices back to indices of original xy
    if translate_at_end:
        print 'le.extract_boundary(): Translating boundary points back into original indices...'
        inner_boundary = backtrans[inner_boundary]

    return inner_boundary


def extract_1d_boundaries(xy, NL, KL, BL, PVx, PVy, check=False):
    """Extract the boundary of a partially periodic 2D network (xy,NL,KL). Assume periodic in x dimension only, so that
    max and min in y are on the top and bottom boundaries.
    Note that this function might not work if particle i is connected to particle j twice or more, through both
    at least one periodic bond and/or a bulk bond.

    Parameters
    ----------
    xy : NP x 2 float array
        point set in 2D
    BL : #bonds x 2 int array
        Bond list
    NL : NP x NN int array
        Neighbor list. The ith row contains the indices of xy that are the bonded pts to the ith pt.
        Nonexistent bonds are replaced by zero.
    KL : NP x NN int array
        Connectivity list. The jth column of the ith row ==1 if pt i is bonded to pt NL[i,j].
        The jth column of the ith row ==0 if pt i is not bonded to point NL[i,j].
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i

    check: bool
        Whether to show intermediate results

    Returns
    ----------
    boundary : tuple of two (#points on boundary x 1) int arrays
        indices of points living on each boundary of the periodic_strip network
    """
    if PVx is None and PVy is None:
        raise RuntimeError('Not designed to allow openBC networks.')
        # PVx = np.zeros_like(KL, dtype=float)
        # PVy = np.zeros_like(KL, dtype=float)

    # If there are dangling points, remove them for now and adjust indices later
    dangles, xy, NL, KL, BL, backtrans = remove_dangling_points(xy, NL, KL, BL, check=check)
    # If no dangling bonds, no need to translate indices at the end
    translate_at_end = len(dangles) > 0

    # Initialize the list of boundary indices to be larger than necessary
    boundaries = []
    for boundaryloc in ['top', 'bottom']:
        # Initialize the boundary list to be as long as possible (will truncate later)
        bb = np.zeros(2 * len(xy), dtype=int)
        if boundaryloc == 'top':
            # Start with the topmost point, which is guaranteed to be
            # at the convex hull and thus also at the top outer edge.
            # Then take the first step to be along the minimum angle bond
            rightIND = np.where(xy[:, 1] == np.max(xy[:, 1]))[0]
            # If there are more than one rightmost point, choose one
            if rightIND.size > 1:
                rightIND = rightIND[0]
        else:
            # Start with the bottom most point, which is guaranteed to be
            # at the convex hull and thus also at the bottom outer edge.
            # Then take the first step to be along the minimum angle bond
            rightIND = np.where(xy[:, 1] == np.min(xy[:, 1]))[0]
            # If there are more than one rightmost point, choose one
            if rightIND.size > 1:
                rightIND = rightIND[0]

        if check:
            print 'le.extract_1d_boundaries(): Found extremal pt: ', rightIND
            print 'le.extract_1d_boundaries():   with neighbors: ', NL[rightIND]
            print 'le.extract_1d_boundaries():   with connectns: ', KL[rightIND]
            plt.plot(xy[:, 0], xy[:, 1], 'k.')
            plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'bo')
            for ii in range(len(xy)):
                plt.text(xy[ii, 0] + 0.1, xy[ii, 1], str(ii))
            plt.plot(xy[rightIND, 0], xy[rightIND, 1], 'ro')
            plt.pause(0.01)

        # Grab the true neighbors of this starting point
        # print 'le.extract_boundary(): NL[rightIND, :] = ', NL[rightIND, :]
        connect = np.argwhere(np.abs(KL[rightIND]).ravel()).ravel()
        neighbors = NL[rightIND, connect]
        if check:
            print 'le.extract_1d_boundaries(): neighbors = ', neighbors
            print 'le.extract_1d_boundaries(): rightIND = ', rightIND

        # Compute the angles of the neighbor bonds
        angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[rightIND, 1] + PVy[rightIND, connect],
                                   xy[neighbors, 0] - xy[rightIND, 0] + PVx[rightIND, connect]).ravel(),
                        2 * np.pi)
        if check:
            print 'le.extract_1d_boundaries(): KL[rightIND] = ', KL[rightIND]
            print 'le.extract_1d_boundaries(): KL[rightIND,0] = ', KL[rightIND, 0]
            print 'le.extract_1d_boundaries(): KL[rightIND,0] ==0 ', KL[rightIND, 0] == 0
            print 'le.extract_1d_boundaries(): np.argwhere(KL[rightIND]) = ', np.argwhere(KL[rightIND])
            print 'le.extract_1d_boundaries(): np.argwhere(KL[rightIND].ravel())= ', np.argwhere(KL[rightIND].ravel())
            print 'le.extract_1d_boundaries(): neighbors = ', neighbors
            print 'le.extract_1d_boundaries(): angles = ', angles

        # Assign this pvx and pvy as pvx_prev and pvy_prev for next time around.
        # Note that this must preceed the redefinition of nextIND
        pvx_prev = PVx[rightIND, connect[angles == min(angles)][0]]
        pvy_prev = PVy[rightIND, connect[angles == min(angles)][0]]

        # Take the second particle to be the one with the lowest bond angle (will be >= pi/2)
        nextIND = neighbors[angles == min(angles)][0]
        bb[0] = rightIND

        dmyi = 1
        # as long as we haven't completed the full outer edge/boundary, add nextIND
        while nextIND != rightIND:
            # print '\n nextIND = ', nextIND
            # print 'np.argwhere(KL[nextIND]) = ', np.argwhere(KL[nextIND]).ravel()
            bb[dmyi] = nextIND
            connect = np.argwhere(np.abs(KL[nextIND]).ravel())
            n_tmp = NL[nextIND, connect]

            # Get position in row of NL where NL == bb[dmyi - 1] (the previous boundary particle/site)
            # and where the PVx and PVy are opposite of the last used PVx and PVy values (to make sure we
            # are looking backwards along the boundary). We will use this to get the 'backward angle' -- the
            # angle of the previous bond in the boundary
            # Note that bb[dmyi - 1] may have been index 0, so there could be multiple matches
            nlpos = np.where(np.logical_and(NL[nextIND] == bb[dmyi - 1],
                                            np.abs(KL[nextIND]).ravel().astype(bool)))[0]
            if len(nlpos) > 1:
                # There is more than one connection to the previous particle. Check for where PVx and PVy
                # values are opposite the previously used values.
                ind_nlpos = np.where(np.logical_and(PVx[nextIND, nlpos] == -pvx_prev,
                                                    PVy[nextIND, nlpos] == -pvy_prev))[0]
                print 'ind_nlpos = ', ind_nlpos
                nlpos = nlpos[ind_nlpos]

            # Exclude previous boundary particle (the copy of that particle in the nlpos position)
            # from the neighbors array, UNLESS IT IS THE ONLY ONE,
            # since its angle with itself is zero!

            # Used to remove previous particle, but this assumes that boundary is more than 2
            # particles long, which might not be true for periodic_strip bcs
            if len(n_tmp) == 1:
                print 'le: The bond is a lone bond, not part of a triangle.'
                neighbors = n_tmp
            else:
                print 'n_tmp = ', n_tmp
                neighbors = np.delete(n_tmp, nlpos)
                connect = np.delete(connect, nlpos)
            print 'n_tmp = ', n_tmp
            print 'neighbors = ', neighbors

            # print 'le: nlpos = ', nlpos
            forward_angles = np.arctan2(xy[neighbors, 1] - xy[nextIND, 1] + PVy[nextIND, connect],
                                        xy[neighbors, 0] - xy[nextIND, 0] + PVx[nextIND, connect]).ravel()
            backward_angle = np.arctan2(xy[bb[dmyi - 1], 1] - xy[nextIND, 1] + PVy[nextIND, nlpos],
                                         xy[bb[dmyi - 1], 0] - xy[nextIND, 0] + PVx[nextIND, nlpos]).ravel()
            if check:
                print 'le: connect = ', connect
                print 'le: forward_angles = ', forward_angles
                print 'le: backward_angle = ', backward_angle

            angles = np.mod(forward_angles - backward_angle, 2 * np.pi)
            if check:
                print 'le: angles = ', angles
                print 'le: angles==min--> ', angles == min(angles)
                print 'le: neighbors = ', neighbors
                print 'le.extract_1d_boundaries(): angles==min--> ', angles == min(angles)
                print 'le.extract_1d_boundaries(): neighbors[angles == min(angles)] --> ', neighbors[angles == min(angles)]

            # Assign this pvx and pvy as pvx_prev and pvy_prev for next time around.
            # Note that this must preceed the redefinition of nextIND
            pvx_prev = PVx[nextIND, connect[angles == min(angles)][0]]
            pvy_prev = PVy[nextIND, connect[angles == min(angles)][0]]
            # Redefine nextIND to be the new boundary index
            nextIND = neighbors[angles == min(angles)][0]
            # print 'nextIND = ', nextIND

            if check:
                # plt.plot(xy[:,0],xy[:,1],'k.')
                XY = np.vstack([xy[bb[dmyi], :], xy[nextIND, :]])
                plt.plot(XY[:, 0], XY[:, 1], 'r-')
                # for i in range(len(xy)):
                #    plt.text(xy[i,0]+0.2,xy[i,1],str(i))
                plt.gca().set_aspect('equal')
                plt.pause(0.01)

            dmyi += 1

        # Truncate the list of boundary indices
        boundary = bb[0:dmyi]

        # Since some points were removed from the boundary identification, translate
        # indices back to indices of original xy
        if translate_at_end:
            print 'le.extract_boundary(): Translating boundary points back into original indices...'
            # print 'boundary = ', boundary
            # print 'translation = ', translation
            # print 'backtrans = ', backtrans
            boundary = backtrans[boundary]

        boundaries.append(boundary)

    return tuple(boundaries)


def distance_from_boundary(xy, boundary, interp_n=None, check=False):
    """Return the distance of each point in xy from the boundary, defined as ordered indices of xy.
    If interp_n is not None, use linear interpolation of the boundary points xy[boundary] to define the boundary of the
    sample and take minimum distance of each xy from the boundary interpolation. This is less precise but may be faster
    if the number of points is very large. It might not be much faster, I'm not sure yet, since the interp_n=None method
    is quite optimized.
    If interp_n is None, uses an exact formulation to get the distance from each linesegment and takes the minimum.

    Parameters
    ----------
    xy : N x 2 float array
        The points for which to find distances from the boundary
    boundary : N x 1 int array
        The indices of xy defining the boundary points xy[boundary]
    interp_n : int or None
        The number of interpolation points between each pair of boundary points xy[boundary]
    check : bool
        view the interpolation as function is executed

    Returns
    -------
    dists : N x 1 float array
        The distance of each point from the (possibly interpolated) boundary
    """
    if interp_n is None:
        # Use exact calculation of minimum distance by finding nearest point on each boundary linesegment
        boundarypts = xy[boundary]
        linesegs = linsegs.array_to_linesegs(boundarypts)
        dists = linsegs.mindist_from_multiple_linesegs(xy, linesegs)
    else:
        # Use approximation (usually done for speed)
        xb = []
        yb = []
        nn = len(boundary)
        for ii in range(nn):
            xb.append(np.linspace(xy[boundary[ii], 0], xy[boundary[(ii + 1) % nn], 0], interp_n + 2).tolist())
            yb.append(np.linspace(xy[boundary[ii], 1], xy[boundary[(ii + 1) % nn], 1], interp_n + 2).tolist())

        xb = np.array(xb).ravel()
        yb = np.array(yb).ravel()
        if check:
            print 'xnew = ', xb
            print 'ynew = ', yb
            plt.scatter(xb, yb, alpha=0.3, c='r')
            plt.plot(xy[:, 0], xy[:, 1])
            plt.show()

        boundarypts = np.dstack((xb, yb))[0]

        d_all = dh.dist_pts(xy, boundarypts)
        dists = np.min(d_all, axis=1)
        # print 'd_all = ', d_all
        # print 'dists = ', dists

    return dists


def distance_from_boundaries_openbc(xy, boundaries, interp_n=None, check=False):
    """Return the distance of each point in xy from either boundary (given in tuple boundaries), each defined as ordered
    indices of xy. If interp_n is not None, use linear interpolation of the boundary points xy[boundary] to define the
    boundary of the sample and take minimum distance of each xy from the boundary interpolation. This function assumes
    open boundary conditions (no periodicity)

    Parameters
    ----------
    xy : N x 2 float array
        The points for which to find distances from the boundary
    boundaries : tuple of (M x 1) int arrays
        The indices of xy defining the boundary points xy[boundary] for top and bottom boundaries
    interp_n : int or None
        The number of interpolation points between each pair of boundary points xy[boundary]
    check : bool
        view the interpolation as function is executed

    Returns
    -------
    dists : tuple of N x 1 float arrays, one array for each boundary
        The distance of each point from the (possibly interpolated) boundary
    """
    dists = []
    for boundary in boundaries:
        these_dists = distance_from_boundary(xy, boundary, interp_n=interp_n, check=check)
        dists.append(these_dists)

    return tuple(dists)


def distance_from_boundaries(xy, boundaries, PVxydict, interp_n=None, check=False):
    """Return the distance of each point in xy from either boundary (given in tuple boundaries), each defined as ordered
    indices of xy. If interp_n is not None, use linear interpolation of the boundary points xy[boundary] to define the
    boundary of the sample and take minimum distance of each xy from the boundary interpolation.
    This is designed for periodicstrip boundary conditions but can handle any

    Parameters
    ----------
    xy : N x 2 float array
        The points for which to find distances from the boundary
    boundaries : tuple of (M x 1) int arrays
        The indices of xy defining the boundary points xy[boundary] for top and bottom boundaries
    PVxydict : dict or None
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
        If None, then we have something like an annulus which has two boundaries but no periodic bonds, or something
        else like open boundary conditions, with no periodicity. PVxydict will effectively be ignored in these cases.
    interp_n : int or None
        The number of interpolation points between each pair of boundary points xy[boundary]
    check : bool
        view the interpolation as function is executed

    Returns
    -------
    dists : tuple of N x 1 float arrays
        The distance of each point from each (possibly interpolated) boundary, as a float array
    """
    jj = 0
    dists = []
    # print 'dists = ', dists
    # print 'boundaries = ', boundaries
    null = np.array([0., 0.])
    if not isinstance(boundaries, tuple):
        if len(np.shape(boundaries)) > 1:
            print 'boundaries = ', boundaries
            print 'np.shape(boundaries) = ', np.shape(boundaries)
            raise RuntimeError('assuming boundary has been stored as M x #boundaries array, where each boundary has '
                               + 'M elements, but it should be given as tuple or single array')
        else:
            boundary = tuple([boundaries])

    # If there is no periodicity, then make PVxydict an empty dict
    if PVxydict is None:
        PVxydict = {}

    for boundary in boundaries:
        if interp_n is None:
            boundarypts = xy[boundary]
        else:
            xb = []
            yb = []
            nn = len(boundary)
            for ii in range(nn):
                # Get periodic vector by which
                if (ii, nn) in PVxydict:
                    pvii = PVxydict[(ii, nn)]
                elif (nn, ii) in PVxydict:
                    pvii = - PVxydict[(nn, ii)]
                else:
                    pvii = null
                xb.append(np.linspace(xy[boundary[ii], 0],
                                      xy[boundary[(ii + 1) % nn], 0] + pvii[0], interp_n + 2).tolist())
                yb.append(np.linspace(xy[boundary[ii], 1],
                                      xy[boundary[(ii + 1) % nn], 1] + pvii[1], interp_n + 2).tolist())

            xb = np.array(xb).ravel()
            yb = np.array(yb).ravel()
            if check:
                print 'xnew = ', xb
                print 'ynew = ', yb
                plt.scatter(xb, yb, alpha=0.3, c='r')
                plt.plot(xy[:, 0], xy[:, 1])
                plt.show()

            boundarypts = np.dstack((xb, yb))[0]

        d_all = dh.dist_pts(xy, boundarypts)
        # print 'd_all = ', d_all
        dists.append(np.min(d_all, axis=1))
        jj += 1

    return tuple(dists)


def distance_periodic(xy, com, LL, dim=-1):
    """Compute the distance of xy points from com given periodic rectangular BCs with extents LL

    Parameters
    ----------
    xy : NP x 2 float array
        Positions to evaluate distance in periodic rectangular domain
    com : 1 x 2 float array
        position to compute distances with respect to
    LL : tuple of floats
        spatial extent of each periodic dimension (ie x(LL[0]) = x(0), y(LL[1]) = y(0))

    Returns
    -------
    dist : len(xy) x 1 float array
        distance of xy from com given periodic boundaries in 2d"""
    dist2d = np.abs(xy - com)
    dist2d[dist2d[:, 0] > LL[0] * 0.5, 0] -= LL[0]
    dist2d[dist2d[:, 1] > LL[1] * 0.5, 1] -= LL[1]
    if dim == 0:
        output = dist2d[:, 0]
    elif dim == 1:
        output = dist2d[:, 1]
    else:
        output = np.sqrt(dist2d[:, 0] ** 2 + dist2d[:, 1] ** 2)
    return output


def distancey_periodicstrip(xy, com, LL):
    """Compute the distance of xy points from com given periodic rectangular BCs with extents LL

    Parameters
    ----------
    xy : NP x 2 float array
        Positions to evaluate distance in periodic rectangular domain
    com : float or 1 x 2 float array
        position to compute distances with respect to, either y coordinate or both coordinates
    LL : float or tuple of floats
        spatial extent of each periodic dimension (ie x(LL[0]) = x(0), y(LL[1]) = y(0))

    Returns
    -------
    dist2d : len(xy) x 1 float array
        distance of xy from com given periodic boundaries in 2d"""
    if len(LL) == 2:
        lenx = LL[0]
    else:
        lenx = LL
    if len(com) == 2:
        dist2d = np.abs(xy - com)[:, 0]
        dist2d[dist2d > lenx * 0.5] -= lenx
    elif len(com) == 1:
        # assume com is given just by the y coordinate of the center of mass
        dist2d = np.abs(xy[:, 0] - com)
        dist2d[dist2d > lenx * 0.5] -= lenx
    return np.abs(dist2d)


def distancex_periodicstrip(xy, com, LL):
    """Compute the distance of xy points from com given periodic rectangular BCs with extents LL
    Parameters
    ----------
    xy : NP x 2 float array
        Positions to evaluate distance in periodic rectangular domain
    com : float or 1 x 2 float array
        position to compute distances with respect to, either x coordinate or both coordinates
    LL : float tuple of floats
        spatial extent of each periodic dimension (ie x(LL[0]) = x(0), y(LL[1]) = y(0))

    Returns
    -------
    dist : len(xy) x 1 float array
        distance of xy from com given periodic boundaries in 2d"""
    if len(LL) == 2:
        lenx = LL[0]
    else:
        lenx = LL
    if len(com) == 2:
        pos = np.abs(xy - com)[:, 0]
        pos[pos > lenx * 0.5] -= lenx
    elif len(com) == 1:
        # assume com is given just by the x coordinate
        pos = np.abs(xy[:, 0] - com)
        pos[pos > lenx * 0.5] -= lenx
    return np.abs(pos)


def center_of_mass(xy, masses):
    """Compute the center of mass for a 2D finite, non-PBC system.

    Parameters
    ----------
    xy : NP x 2 float array
        The positions of the particles/masses/weighted objects
    masses : NP x 1 float array
        The weight to attribute to each particle.

    Returns
    -------
    com : 2 x 1 float array
        The position in simulation space of the periodic center of mass
    """
    return np.sum(masses.reshape(len(xy), 1) * xy.astype(np.float), axis=0) / float(np.sum(masses))


def com_periodic(xy, LL, masses=1.):
    """Compute the center of mass for a 2D periodic system. When a cluster straddles the periodic boundary, a naive
    calculation of the center of mass will be incorrect. A generalized method for calculating the center of mass for
    periodic systems is to treat each coordinate, x and y, as if it were on a circle instead of a line.
    The calculation takes every particle's x coordinate and maps it to an angle, averages the angle, then maps back.

    Parameters
    ----------
    xy : NP x 2 float array
        The positions of the particles/masses/weighted objects
    LL :
    masses : NP x 1 float array or float
        The weight to attribute to each particle. If masses is a float, it is ignored (all particles weighted equally)

    Returns
    -------
    com : 2 x 1 float array
        The position in simulation space of the periodic center of mass
    """
    # test case:
    # import lepm.lattice_elasticity as le
    # import matplotlib.pyplot as plt
    # import numpy as np
    # xy = np.random.rand(100, 2) - np.array([0.5, 0.5])
    # LL = (1.0, 1.0)
    # plt.scatter(xy[:, 0], xy[:, 1])
    # com = le.com_periodic(xy, LL)
    # plt.plot(com[0], com[1], 'ro')
    # plt.show()

    minxy = np.min(xy, axis=0)
    if isinstance(LL, tuple):
        LL = np.array([LL[0], LL[1]])

    # map to xi and zeta coordinates. Each xi element has x component and y component.
    if isinstance(masses, np.ndarray):
        xi = np.cos(((xy - minxy) / LL) * 2. * np.pi) * masses.reshape(len(masses), 1)
        zeta = np.sin(((xy - minxy) / LL) * 2. * np.pi) * masses.reshape(len(masses), 1)
    else:
        xi = np.cos(((xy - minxy) / LL) * 2. * np.pi)
        zeta = np.sin(((xy - minxy) / LL) * 2. * np.pi)

    # average to get center of mass on each circle
    xibar = np.mean(xi, axis=0)
    zetabar = np.mean(zeta, axis=0)

    thetabar = np.arctan2(-zetabar, -xibar) + np.pi
    com = LL * thetabar / (2. * np.pi) + minxy

    return com


def com_periodicstrip(xy, LL, masses=1., check=False):
    """Compute the center of mass for a 2D periodic system. When a cluster straddles the periodic boundary, a naive
    calculation of the center of mass will be incorrect. A generalized method for calculating the center of mass for
    periodic systems is to treat each coordinate, x and y, as if it were on a circle instead of a line.
    The calculation takes every particle's x coordinate and maps it to an angle, averages the angle, then maps back.

    Parameters
    ----------
    xy : NP x 2 float array
        The positions of the particles/masses/weighted objects
    LL : float or tuple of floats or 2x1 array of floats
        spatial extent of the system
    masses : NP x 1 float array or float
        The weight to attribute to each particle. If masses is a float, it is ignored (all particles weighted equally)

    Returns
    -------
    com : 2 x 1 float array
        The position in simulation space of the periodic center of mass
    """
    # test case:
    # import lepm.lattice_elasticity as le
    # import matplotlib.pyplot as plt
    # import numpy as np
    # xy = np.random.rand(100, 2) - np.array([0.5, 0.5])
    # LL = (1.0, 1.0)
    # plt.scatter(xy[:, 0], xy[:, 1])
    # com = le.com_periodic(xy, LL)
    # plt.plot(com[0], com[1], 'ro')
    # plt.show()
    if len(LL) == 2:
        lenx = LL[0]

    minx = np.min(xy[:, 0])
    # map to xi and zeta coordinates. Each xi element has x component and y component.
    print 'np.shape(masses) =', np.shape(masses)

    if isinstance(masses, np.ndarray):
        xi = np.cos(((xy[:, 0] - minx) / lenx) * 2. * np.pi) * masses
        zeta = np.sin(((xy[:, 0] - minx) / lenx) * 2. * np.pi) * masses
    else:
        raise RuntimeError('Debug: masses should not be equal for my current debugging program')
        xi = np.cos(((xy[:, 0] - minx) / lenx) * 2. * np.pi)
        zeta = np.sin(((xy[:, 0] - minx) / lenx) * 2. * np.pi)

    # average to get center of mass on each circle
    xibar = np.mean(xi)
    zetabar = np.mean(zeta)

    thetabar = np.arctan2(-zetabar, -xibar) + np.pi
    comx = lenx * thetabar / (2. * np.pi) + minx

    # Check it
    angles = np.arctan2(-zeta, -xi) + np.pi
    print 'le: np.shape(angles) = ', np.shape(angles)
    print 'le: np.min(angles) = ', np.min(angles)
    print 'le: np.max(angles) = ', np.max(angles)
    print 'le: thetabar = ', thetabar

    if check:
        print 'le: check=', check
        plt.plot(np.cos(angles), np.sin(angles), alpha=0.05)
        plt.plot(np.cos(thetabar), np.sin(thetabar), 'ro')
        plt.show()
        plt.clf()

    com_nonper = center_of_mass(xy, masses)
    com = np.array([comx, com_nonper[1]])
    return com


def count_NN(KL):
    """Count how many nearest neighbors each particle has.

    Parameters
    ----------
    KL : NP x max(#neighbors) array
        Connectivity matrix. All nonzero elements (whether =1 or simply !=0) are interpreted as connections.

    Returns
    ----------
    zvals : NP x 1 int array
        The ith element gives the number of neighbors for the ith particle
    """
    zvals = (KL != 0).sum(1)
    return zvals


def NL2NLdict(NL, KL):
    """Convert NL into a dictionary which gives neighbors (values) of the particle index (key).

    Parameters
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL : NP x max(#neighbors) array
        Connectivity matrix. All nonzero elements (whether =1 or simply !=0) are interpreted as connections.

    Returns
    ----------
    NLdict : dict
        NLdict[i] returns an array listing neighbors of the ith particle, where i is an integer.
    """
    ii = 0
    NLdict = {}
    for row in NL:
        connections = np.where(KL[ii] != 0)[0]
        NLdict[int(ii)] = row[connections]
        ii += 1

    return NLdict


def Tri2BL(TRI):
    """Convert triangulation array (#tris x 3) to bond list (#bonds x 2) for 2D lattice of triangulated points.

    Parameters
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.

    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points"""
    BL1 = TRI[:, [0, 1]]
    BL2 = np.vstack((BL1, TRI[:, [0, 2]]))
    BL3 = np.vstack((BL2, TRI[:, [1, 2]]))
    BLt = np.sort(BL3, axis=1)
    # select unique rows of BL
    # this method of making unique rows doesn't work on Jiayi's computer
    # BL = np.unique(BLt.view(np.dtype((np.void, BLt.dtype.itemsize *
    #                                   BLt.shape[1])))).view(BLt.dtype).reshape(-1, BLt.shape[1])
    # Use this method instead, for now
    BL = dh.unique_rows(BLt)
    return BL


def BL2TRI(BL, xy):
    """Convert bond list (#bonds x 2) to Triangulation array (#tris x 3) (using dictionaries for speedup and scaling)

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points

    Returns
    ----------
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.
    """
    d = {}
    # preallocate for speed
    tri = np.zeros((len(BL), 3), dtype=np.int)
    # c is dmy index to fill up and cut off tri
    c = 0
    for i in BL:
        # reorder row if [big, small]
        if i[0] > i[1]:
            t = i[0]
            i[0] = i[1]
            i[1] = t
        # Check if small val in row is key of dict d.
        # If not, then initialize the key, value pair.
        if (i[0] in d):
            d[i[0]].append(i[1])
        else:
            d[i[0]] = [i[1]]

    # From dict d, make TRI
    for key in d:
        for n in d[key]:
            for n2 in d[key]:
                if (n > n2) or n not in d:
                    continue
                if n2 in d[n]:
                    tri[c, :] = [key, n, n2]
                    c += 1
    tri = tri[0:c]

    # Check for points inside each triangle. If they exist, remove that triangle
    keep = np.ones(len(tri), dtype=bool)
    index = 0
    for row in tri:
        mask = np.ones(len(xy), dtype=bool)
        mask[row] = False
        remove = where_points_in_triangle(xy[mask, :], xy[row[0], :], xy[row[1], :], xy[row[2], :])
        if remove.any():
            keep[index] = False
            # if check:
            #     plt.triplot(xy[:,0],xy[:,1], tri, 'g.-')
            #     plt.plot(xy[row,0], xy[row,1],'ro')
            #     plt.show()

        index += 1

    TRI = tri[keep]

    return TRI


def where_points_in_triangle(pts, v1, v2, v3):
    """Determine whether point pt is inside triangle with vertices v1,v2,v3, in 2D.
    """
    b1 = cross_pts_triangle(pts, v1, v2) < 0.0
    b2 = cross_pts_triangle(pts, v2, v3) < 0.0
    b3 = cross_pts_triangle(pts, v3, v1) < 0.0

    return np.logical_and((b1 == b2), (b2 == b3))


def cross_pts_triangle(p1, p2, p3):
    """Return the cross product for arrays of triangles or triads of points
    """
    return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])


def memberIDs(a, b):
    """Return array (c) of indices where elements of a are members of b.
    If ith a elem is member of b, ith elem of c is index of b where a[i] = b[index].
    If ith a elem is not a member of b, ith element of c is 'None'.
    The speed is O(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value


def ismember(a, b):
    """Return logical array (c) testing where elements of a are members of b.
    The speed is Order(len(a)+len(b)), so it's fast.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = True
    return np.array([bind.get(itm, False) for itm in a])  # None can be replaced by any other "not in b" value


def rows_matching(BL, indices):
    """Find rows with elements matching elements of the supplied array named 'indices'.
    Right now just set up for N x 2 int arrays. --> todo: generalize to n x m
    Returns row indices of BL whose row elements are ALL contained in array named 'indices'.
    """
    inBL0 = np.in1d(BL[:, 0], indices)
    # print 'BLUC[:,0] = ', BL[:,0]
    # print 'indices = ', indices
    # print 'inBL0 = ', inBL0
    inBL1 = np.in1d(BL[:, 1], indices)
    matches = np.where(np.logical_and(inBL0, inBL1))[0]
    return matches


def row_is_in_array(row, array):
    """Check if row ([x,y]) is an element of an array ([[x1,y1],[x2,y2],...]) """
    return any((array[:] == row).all(1))


def minimum_distance_between_pts(R):
    """
    Parameters
    ----------
    R : NP x ndim array
        the points over which to find the min distance between points

    Returns
    ----------
    min_dist : float
        the minimum distance between points
    """
    distsq = dh.dist_pts(R, R, square_norm=True).ravel()
    distsq_min = np.min(distsq[np.where(distsq > 0.)])
    # Take the root of the smallest value
    min_dist = np.sqrt(distsq_min)

    # Slow method
    # for ind in range(len(R)):
    #     mask = np.ones(len(R), dtype=bool)
    #     mask[ind] = 0
    #     row = R[ind]
    #     # dist = scipy.spatial.distance.cdist(R,row)
    #     dist = np.sqrt(np.sum((R-row)**2, axis=1))
    #     if ind == 0:
    #         min_dist = np.min(dist[mask])
    #     else:
    #         min_dist = min(min_dist, np.min(dist[mask]))
    # print 'min_dist slow method -->', min_dist
    # print 'Do they match? If so, erase the slow method!'
    # sys.exit()
    return min_dist


def find_nearest(array, value):
    """return the value in array that is nearest to the supplied value"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def NL2BL(NL, KL):
    """Convert neighbor list and KL (NPxNN) to bond list (#bonds x 2) for lattice of bonded points.

    Parameters
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL : NP x max(#neighbors) array
        Connectivity matrix. All nonzero elements (whether =1 or simply !=0) are interpreted as connections.

    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points. Values are negative for periodic BCs.
    """
    # reshape to get BL array of particle pairs. Below, 't' stands for 'temporary'.
    BLttnormal = np.asarray([[i, NL[i, j]] for i in range(len(KL)) for j in np.where(KL[i, :] > 0)[0]])
    BLt = np.sort(BLttnormal)

    if (KL.ravel() == -1).any():
        # print 'detected periodic BCs...'
        BLperiodic = np.asarray([[-i, -NL[i, j]] for i in range(len(KL)) for j in np.where(KL[i, :] == -1)[0]])
        # print 'BLperiodic = ', BLperiodic
        BL2 = np.sort(BLperiodic)
        BLt = np.vstack((BLt, np.dstack((BL2[:, 1], BL2[:, 0]))[0]))

    # select unique rows of BL
    # BL = np.unique(BLt.view(np.dtype((np.void, BLt.dtype.itemsize*BLt.shape[1])))).view(BLt.dtype).reshape(-1, BLt.shape[1])
    BL = dh.unique_rows(BLt)
    return BL.astype(np.intc)


def KL2kL(NL, KL, BL):
    """Convert spring constant matrix to spring constant list (#bonds x 1) for lattice of bonded points.

    Parameters
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where nonzero corresponds to a true connection while 0 signifies that there is not a
        connection. Here, periodic bonds are specified by negative rows in BL, NOT in KL, since KL can be spring
        constants of arbitary value.
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points

    Returns
    ----------
    kL : array of length #bonds
        The ith element is the spring constant of the ith bond in BL, taken from KL
    """
    # cycle through BL, finding matching inds in NL and thus KL
    # for row in BL, get KL value in row BL[i,0] and col where(NL[BL[i,0],:]==BL[i,1])[0]
    if (BL < 0).any():
        aBL = np.abs(BL)
        kL = np.array([KL[aBL[i, 0], np.where(NL[aBL[i, 0], :] == aBL[i, 1])[0]][0] for i in range(len(aBL))])
        kL2 = np.array([KL[aBL[i, 1], np.where(NL[aBL[i, 1], :] == aBL[i, 0])[0]][0] for i in range(len(aBL))])
        if np.abs(kL - kL2).any() > 1e-6:
            raise RuntimeError('KL is not properly symmetric! KL[i, j neighbor] != KL[j neighbor, i]')
    else:
        kL = np.array([KL[BL[i, 0], np.where(NL[BL[i, 0], :] == BL[i, 1])[0]][0] for i in range(len(BL))])
    return kL


def BL2KL(BL, NL):
    """Convert the bond list (#bonds x 2) to connectivity list using NL for its structure.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.

    Returns
    ----------
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where 1 corresponds to a true connection while 0 signifies that there is not a connection.
    """
    # NP, NN = np.shape(NL)
    # check for periodic bonds
    if (BL.ravel() < 0).any():
        # periodic bonds exist
        KL = np.zeros_like(NL)
        for row in BL:
            if (row < 0).any():
                KL[row[0], np.where(NL[abs(row[0])] == abs(row[1]))[0][0]] = -1
                KL[row[1], np.where(NL[abs(row[1])] == abs(row[0]))[0][0]] = -1
            else:
                KL[row[0], np.where(NL[abs(row[0])] == abs(row[1]))[0][0]] = 1
                KL[row[1], np.where(NL[abs(row[1])] == abs(row[0]))[0][0]] = 1
                # print 'KL =', KL
    else:
        # all regular bonds
        KL = np.zeros_like(NL)
        for row in BL:
            KL[row[0], np.where(NL[row[0]] == row[1])[0][0]] = 1
            KL[row[1], np.where(NL[row[1]] == row[0])[0][0]] = 1
            # print 'KL =', KL

    return KL


def BL2NLandKL(BL, NP='auto', NN='min'):
    """Convert bond list (#bonds x 2) to neighbor list (NL) and connectivity list (KL) for lattice of bonded points.
    Returns KL as ones where there is a bond and zero where there is not.
    (Even if you just want NL from BL, you have to compute KL anyway.)
    Note that this makes no attempt to preserve any previous version of NL, which in the philosophy of these simulations should remain constant during a simulation.
    If NL is known, use BL2KL instead, which creates KL according to the existing NL.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    NP : int
        number of points (defines the length of NL and KL)
    NN : int
        maximum number of neighbors (defines the width of NL and KL)

    Returns
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where 1 corresponds to a true connection while 0 signifies that there is not a connection.
    """
    if NN == 'min':
        # Start with many many nearest neighbors, and cut columns until its ok
        NN = 20  # pick NN so that it is inconceviable to have more than NN neighbors
        trimNN = True
    else:
        trimNN = False

    if NP == 'auto':
        if BL.size > 0:
            NL = np.zeros((max(BL.ravel()) + 1, NN), dtype=np.intc)
            KL = np.zeros((max(BL.ravel()) + 1, NN), dtype=np.intc)
        else:
            raise RuntimeError('ERROR: there is no BL to use to define NL and KL, so cannot run BL2NLandKL()')
    else:
        NL = np.zeros((NP, NN), dtype=np.intc)
        KL = np.zeros((NP, NN), dtype=np.intc)

    # print 'shape(NL) =', np.shape(NL)
    # print 'shape(KL) =', np.shape(KL)
    if BL.size > 0:
        if (BL.ravel() < 0).any():
            # periodic bonds exist
            for row in BL:
                col = np.where(KL[abs(row[0]), :] == 0)[0][0]
                NL[abs(row[0]), col] = abs(row[1])
                if (row < 0).any():
                    KL[abs(row[0]), col] = -1
                else:
                    KL[abs(row[0]), col] = 1
                col = np.where(KL[abs(row[1]), :] == 0)[0][0]
                NL[abs(row[1]), col] = abs(row[0])
                if (row < 0).any():
                    KL[abs(row[1]), col] = -1
                else:
                    KL[abs(row[1]), col] = 1
        else:
            for row in BL:
                # print 'row = ', row
                # print 'KL = ', KL
                # print np.where(KL[row[0],:]==0)[0]
                # print KL
                # print NL
                col = np.where(KL[row[0], :] == 0)[0][0]
                NL[row[0], col] = row[1]
                KL[row[0], col] = 1
                col = np.where(KL[row[1], :] == 0)[0][0]
                NL[row[1], col] = row[0]
                KL[row[1], col] = 1

    if trimNN:
        NLout, KLout = trim_cols_NLandKL(NL, KL)
    else:
        NLout, KLout = NL, KL

    return NLout, KLout


def NL2BM(xy, NL, KL, PVx=None, PVy=None):
    """Convert bond list (#bonds x 2) to bond matrix (#pts x #nn) for lattice of bonded points.

    Parameters
    ----------
    xy : array of dimension nxd
        2D lattice of points (positions x,y(,z) )
    NL : int array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL : array of dimension #pts x (max number of neighbors)
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i

    Returns
    ----------
    BM : array of length #pts x max(#neighbors)
        The (i,j)th element is the bond length of the bond connecting the ith particle to its jth neighbor (the particle with index NL[i,j]).
    """
    BM = np.zeros_like(NL).astype(float)
    NL = NL.astype(int)
    if np.shape(xy)[1] > 3:
        print '\n\n#####################################################' + \
              'WARNING! xy supplied to NL2BM() has dim > 3.' + \
              'Using only first TWO dimensions to measure distances.' + \
              '#####################################################'
        xy = xy[:, 0:2]

    for i in range(len(NL)):
        # print 'i = ', i
        # print 'np.shape(NL) = ', np.shape(NL)
        # print 'np.shape(xy) = ', np.shape(xy)
        # print 'vector.mag(xy[NL[i], :] - xy[i, :]) = ', vector.mag(xy[NL[i], :] - xy[i, :])
        BM[i] = KL[i] * vector.mag(xy[NL[i], :] - xy[i, :])

    # Check for periodic bonds/boundaries. Then calc bond lengths
    if (KL < 0).any():
        try:
            # There are periodic bonds
            # print 'np.where(KL < 0) = ', np.where(KL < 0)
            for (i, j) in zip(np.where(KL < 0)[0], np.where(KL < 0)[1]):
                BM[i, j] = vector.mag(xy[NL[i, j]] + np.array([PVx[i, j], PVy[i, j]]) - xy[i, :])
        except TypeError:
            raise RuntimeError('When calling NL2BM() on a network with periodic BCs, you must specify PVx and PVy!')

    return BM


def NL2BMxy(xy, NL, KL, PVx=None, PVy=None):
    """Convert bond list (#bonds x 2) to bond distance matrices (each #pts x #nn, one for each dimension)
     for lattice of bonded points.

    Parameters
    ----------
    xy : array of dimension nxd
        2D lattice of points (positions x,y(,z) )
    NL : int array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    KL : array of dimension #pts x (max number of neighbors)
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    PVx : NP x NN float array (optional, for periodic lattices)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array (optional, for periodic lattices)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i

    Returns
    ----------
    BMx : #pts x max(#neighbors) float array
        The (i,j)th element is the x-axis projection of the bond length of the bond connecting the ith particle to
        its jth neighbor (the particle with index NL[i,j]).
    BMy : #pts x max(#neighbors) float array
        The (i,j)th element is the y-axis projection of the bond length of the bond connecting the ith particle to
        its jth neighbor (the particle with index NL[i,j]).
    """
    BMx = np.zeros_like(NL).astype(float)
    BMy = np.zeros_like(NL).astype(float)
    NL = NL.astype(int)
    if np.shape(xy)[1] > 3:
        print '\n\n#####################################################' + \
              'WARNING! xy supplied to NL2BM() has dim > 3.' + \
              'Using only first TWO dimensions to measure distances.' + \
              '#####################################################'
        xy = xy[:, 0:2]

    for i in range(len(NL)):
        # print 'i = ', i
        # print 'np.shape(NL) = ', np.shape(NL)
        # print 'np.shape(xy) = ', np.shape(xy)
        # print 'vector.mag(xy[NL[i], :] - xy[i, :]) = ', vector.mag(xy[NL[i], :] - xy[i, :])
        BMx[i] = KL[i] * xy[NL[i], 0] - xy[i, 0]
        BMy[i] = KL[i] * xy[NL[i], 1] - xy[i, 1]

    # Check for periodic bonds/boundaries. Then calc bond lengths
    if (KL < 0).any():
        try:
            # There are periodic bonds
            # print 'np.where(KL < 0) = ', np.where(KL < 0)
            for (i, j) in zip(np.where(KL < 0)[0], np.where(KL < 0)[1]):
                BMx[i, j] = xy[NL[i, j], 0] + PVx[i, j] - xy[i, 0]
                BMy[i, j] = xy[NL[i, j], 1] + PVy[i, j] - xy[i, 1]
        except TypeError:
            raise RuntimeError('When calling NL2BMxy() on a network with periodic BCs, you must specify PVx and PVy!')

    return BMx, BMy


def BL2BM(xy, NL, BL, KL=None, PVx=None, PVy=None):
    """Convert bond list (#bonds x 2) to bond matrix (#pts x #nn) for lattice of bonded points.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points

    Returns
    ----------
    BM : array of length #pts x max(#neighbors)
        The (i,j)th element is the bond length of the bond connecting the ith particle to its jth neighbor (the particle with index NL[i,j]).
    """
    bL = bond_length_list(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)

    # clear periodicity information
    BL = np.abs(BL)

    BM = np.zeros_like(NL).astype(float)
    for i in range(len(bL)):
        BM[BL[i, 0], np.where(NL[BL[i, 0], :] == BL[i, 1])[0][0]] = bL[i]
        BM[BL[i, 1], np.where(NL[BL[i, 1], :] == BL[i, 0])[0][0]] = bL[i]

    # print '\n\n Made BM =', BM
    return BM


def BM2bL(NL, BM, BL):
    """Convert bond list (#bonds x 2) to bond length list (#bonds x 1) for lattice of bonded points.
    Assumes that BM has correctly accounted for periodic BCs, if applicable.

    Parameters
    ----------
    NL : array of dimension #pts x max(#neighbors)
        The ith row contains indices for the neighbors for the ith point.
    BM : array of length #pts x max(#neighbors)
        The (i,j)th element is the bond length of the bond connecting the ith particle to its jth neighbor (the particle with index NL[i,j]).
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points

    Returns
    ----------
    bL : array of length #bonds
        The ith element is the length of of the ith bond in BL.
    """
    # clear periodicity information
    BL = np.abs(BL)

    # cycle through BL, finding matching inds in NL and thus KL
    # for row in BL, get KL value in row BL[i,0] and col where(NL[BL[i,0],:]==BL[i,1])[0]
    # bL = np.array([BM[BL[i, 0], np.where(NL[BL[i, 0], :] == BL[i, 1])[0]][0] for i in range(len(BL))])
    # Rewrite the above to ensure that having two bonds between i and j is ok (one bulk, one periodic, or
    # two periodic, for ex).
    BLunique_sorted, order, ui = dh.args_unique_rows_threshold(BL, 1e-3)
    bL = np.zeros_like(BL[:, 0], dtype=float)
    if len(BLunique_sorted) == len(BL):
        bL = np.array([BM[BL[i, 0], np.where(NL[BL[i, 0], :] == BL[i, 1])[0]][0] for i in range(len(BL))])
    else:
        # Sort BL to keep track of whether we have used a copy of each row already
        BLs = BL[order]
        for ii in range(len(BLs)):
            # check if this row is a unique row, or the first of its kind
            if ui[ii]:
                index = 0
            else:
                index += 1
            bL[ii] = BM[BLs[ii, 0], np.where(NL[BLs[ii, 0], :] == BLs[ii, 1])[0]][index]
        # Finally, rearrange so that iith element of BLs matches iith element of BL
        bL[order] = bL
    return bL

# fake data for testing
# BL = np.array([[0, 1], [0, 1], [0, 2], [1, 2], [1, 2], [1, 2]])
# BLunique_sorted, order, ui = dh.args_unique_rows_threshold(BL, 1e-3)
# uorder, indices = np.unique(order, return_index=True, return_inverse=False)
# print 'indices = ', indices


def bL2BM(bL, BL, NL, KL):
    """Use bL to create a matrix of bond lengths with the same shape as NL and KL.

    Parameters
    ----------
    bL : #bonds x 1 float array
        The ith element is the length of of the ith bond in BL
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x max(#neighbors) int array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond

    Returns
    -------
    BM : NP x max #NN float array
        the rest lengths of each bond, as provided by bL
    """
    # Clear periodicity information from BL
    BL = np.abs(BL)

    # Pre-allocate BM
    BM = np.zeros_like(NL, dtype=float)

    ind = 0
    for bond in BL:
        val = bL[ind]
        # Check KL if the particle we seek is indexed as zero
        if 0 in bond:
            okinds = np.where(np.abs(KL[bond[0]]) > 0)[0]
            col = np.where(NL[bond[0], okinds] == bond[1])[0][0]
            BM[bond[0], okinds[col]] = val
            okinds = np.where(np.abs(KL[bond[1]]) > 0)[0]
            col = np.where(NL[bond[1], okinds] == bond[0])[0][0]
            BM[bond[1], okinds[col]] = val
        else:
            col = np.where(NL[bond[0]] == bond[1])[0][0]
            BM[bond[0], col] = val
            col = np.where(NL[bond[1]] == bond[0])[0][0]
            BM[bond[1], col] = val
        ind += 1

    return BM


def BM2BSM(xy, NL, KL, BM0):
    """Calc strain in each bond, reported in NP x NN array called BSM ('bond strain matrix')
    """
    # Check if 3D or 2D
    # np.sqrt( (xy[NL[i,0],0]-xy[BL[:,1],0])**2+(xy[BL[:,0],1]-xy[BL[:,1],1])**2) ]
    '''this isn't finished....'''


def bond_length_list(xy, BL, NL=None, KL=None, PVx=None, PVy=None):
    """Convert bond list (#bonds x 2) to bond length list (#bonds x 1) for lattice of bonded points. (BL2bL)

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.

    Returns
    ----------
    bL : array of dimension #bonds x 1
        Bond lengths, in order of BL (lowercase 'b' denotes 1D array)
    """
    if (BL < 0).any():
        if PVx is None or PVy is None or NL is None or KL is None:
            raise RuntimeError('PVx and PVy and NL and KL are required if periodic bonds exist.')
            # PVxydict = BL2PVxydict(BL, xy, PV)
        else:
            # import lepm.plotting.network_visualization as nvis
            # print 'xy ->', np.shape(xy)
            # print 'BL ->', np.shape(BL)
            # print 'NL ->', np.shape(NL)
            # print 'KL ->', np.shape(KL)
            # nvis.movie_plot_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy)
            # plt.show()
            BM = NL2BM(xy, NL, KL, PVx=PVx, PVy=PVy)
            # print 'le: BM = ', BM
            bL = BM2bL(NL, BM, BL)
    else:
        bL = np.array([np.sqrt(np.dot(xy[int(BL[i, 1]), :] - xy[int(BL[i, 0]), :],
                                      xy[int(BL[i, 1]), :] - xy[int(BL[i, 0]), :])) for i in range(len(BL))])
    return bL


def bond_strain_list(xy, BL, bo):
    """Convert neighbor list to bond list (#bonds x 2) for lattice of bonded points. (makes bs)

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.
    bo : array of dimension #bonds x 1
        Rest lengths of bonds, in order of BL, lowercase to denote 1D array.

    Returns
    ----------
    bs : array of dimension #bonds x 1
        The strain of each bond, in order of BL
    """
    bL = bond_length_list(xy, BL)
    # print 'len(bL) = ', len(bL)
    # print 'len(bo) = ', len(bo)
    bs = (bL - bo) / bo
    return bs


def xyandTRI2centroid(xy, TRI):
    """Convert xy and TRI to centroid xy array
    """

    centxy = np.zeros((len(TRI), 2), dtype=float)
    for ii in range(len(TRI)):
        row = TRI[ii]
        centxy[ii] = (xy[row[0]] + xy[row[1]] + xy[row[2]]) / 3.

    # Check
    # plt.triplot(xy[:,0], xy[:,1], TRI, 'go-')
    # print 'TRI = ', TRI
    # plt.scatter(centxy[:,0], centxy[:,1], s=100,c='r')
    # plt.show()

    return centxy


def TRI2centroidNLandKL(TRI):
    """Convert triangular rep of lattice (such as triangulation) to neighbor array and connectivity array of the triangle centroids.

    Parameters
    ----------
    TRI : Ntris x 3 int array
        Triangulation of a point set. Each row gives indices of vertices of single triangle.

    Returns
    ----------
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array (optional, for speed)
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    """
    BL = TRI2centroidBL(TRI)
    NL, KL = BL2NLandKL(BL, NP='auto', NN='min')
    return NL, KL


def TRI2centroidNLandKLandBL(TRI):
    """Convert triangular rep of lattice (such as triangulation) to neighbor array, connectivity array, and bond list of the triangle centroids.

    Parameters
    ----------
    TRI : Ntris x 3 int array
        Triangulation of a point set. Each row gives indices of vertices of single triangle.

    Returns
    ----------
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array (optional, for speed)
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    """
    BL = TRI2centroidBL(TRI)
    NL, KL = BL2NLandKL(BL, NP='auto', NN='min')
    return NL, KL, BL


def TRI2centroidBL(TRI):
    """Convert triangular rep of lattice (such as triangulation) to bond list of the triangle centroids

    Parameters
    ----------
    TRI : Ntris x 3 int array
        Triangulation of a point set. Each row gives indices of vertices of single triangle.

    Returns
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    """
    # For each row of TRI, find rows that share a bond (up to 3 of these)
    dic = {}
    # nbrs = np.zeros((len(BL),3), dtype=np.int)
    BL = np.zeros((len(TRI) * 3, 2), dtype=np.int)
    dmyi = 0
    # make sure each row has indices increasing in each row
    TRI.sort()

    # Compose dictionary with pairs of neighboring triangles
    for row in TRI:
        if (row[0], row[1]) in dic:
            dic[(row[0], row[1])].append(dmyi)
        else:
            dic[(row[0], row[1])] = [dmyi]
        if (row[0], row[2]) in dic:
            dic[(row[0], row[2])].append(dmyi)
        else:
            dic[(row[0], row[2])] = [dmyi]
        if (row[1], row[2]) in dic:
            dic[(row[1], row[2])].append(dmyi)
        else:
            dic[(row[1], row[2])] = [dmyi]
        dmyi += 1

    # print 'dic = ', dic
    # Compose BL from the pairs of indices in each dict entry
    dmyi = 0
    for key in dic:
        # if the bond is not an edge of the lattice or
        # a dangling bond, add to BL.
        # print 'key = ', key, '  -> d[key] = ', dic[key]
        if len(dic[key]) == 2:
            BL[dmyi] = dic[key]
            dmyi += 1

    return BL[0:dmyi]


def TRI2NLandKL(TRI, remove_negatives=False):
    """
    Convert triangulation index array (Ntris x 3) to Neighbor List (Nbonds x 2) array and Connectivity array.

    Parameters
    ----------
    TRI : Ntris x 3 int array
        Triangulation of a point set. Each row gives indices of vertices of single triangle.
    remove_negatives : bool
        If any of the indices are -1 in TRI, remove those bonds; this is for triangular reps of the lattice that have non-triangular elements (ex dangling bonds)

    Returns
    ----------
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array (optional, for speed)
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    """
    BL = TRI2BL(TRI, remove_negatives=remove_negatives)
    NL, KL = BL2NLandKL(BL, NP='auto', NN='min')
    return NL, KL


def TRI2BL(TRI, remove_negatives=False):
    """
    Convert triangulation index array (Ntris x 3) to Bond List (Nbonds x 2) array.

    Parameters
    ----------
    TRI : Ntris x 3 int array
        Triangulation of a point set. Each row gives indices of vertices of single triangle.
    remove_negatives : bool
        If any of the indices are -1 in TRI, remove those bonds; this is for triangular reps of the lattice that have non-triangular elements (ex dangling bonds)

    Returns
    ----------
    BL : Nbonds x 2 int array
        Bond list

    """
    # each edge is shared by 2 triangles unless at the boundary.
    # each row contains 3 edges.
    # An upper bound on the number bonds is 3*len(TRI)
    BL = np.zeros((3 * len(TRI), 2), dtype=int)

    dmyi = 0
    for row in TRI:
        BL[dmyi] = [row[0], row[1]]
        BL[dmyi + 1] = [row[1], row[2]]
        BL[dmyi + 2] = [row[0], row[2]]
        dmyi += 3

    # Sort each row to be ascending
    BL_sort = np.sort(BL, axis=1)
    BLtrim = dh.unique_rows(BL_sort)

    if remove_negatives:
        # If any of the indices are -1, remove those bonds
        BL = BL[np.where(np.logical_and(BL[:, 0] >= 0, BL[:, 1] >= 0))[0]]

    return BLtrim


def PVxy2PVxynp(PVx, PVy, NL):
    """Convert periodic vector arrays that have shape of NL (PVx and PVy) into arrays PVxnp and PVynp
    that have shape len(xy) x len(xy).

    Parameters
    ----------
    PVx
    PVy
    NL

    Returns
    -------
    PVxnp : NP x NP float array
        ijth element of PVy is the x-component of the vector taking j = NL[i,k] to its image as seen by particle i
        Note that shape(PVxnp) == (len(xy), len(xy))
    PVynp : NP x NP float array
        ijth element of PVy is the y-component of the vector taking j = NL[i,k] to its image as seen by particle i
        Note that shape(PVynp) == (len(xy), len(xy))
    """
    PVxnp = np.zeros((len(NL), len(NL)), dtype=float)
    PVynp = np.zeros((len(NL), len(NL)), dtype=float)
    wheres = np.where(np.logical_or(np.abs(PVx) > 0, np.abs(PVy) > 0))
    print 'wheres = ', wheres
    raise RuntimeError('Have not written...')
    return PVxnp, PVynp


def PVxydict2PV(PVxydict, check=False):
    """Obtain the 2D lattice vectors (PV) from a dictionary of periodic displacements

    Parameters
    ----------
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    check : bool
        print intermediate and final output

    Returns
    -------
    PV : n x 2 float array, where usually n=2
        The basis of periodic vectors, usually [[Lx, 0], [0, Ly]]
    """
    # Make a flattened list of periodic vectors so that all elements are 2 x 1
    pvlist = []
    for key in PVxydict:
        tmp = PVxydict[key]
        if len(np.shape(tmp)) == 1:
            pvlist.append(tmp)
        else:
            for row in tmp:
                pvlist.append(row)

    print 'pvlist = ', pvlist
    pvarr = np.array(pvlist)
    pvs = dh.unique_rows(pvarr)
    if check:
        print 'pvs = ', pvs
    if len(pvs) > 0:
        # maxx = np.argmax(pvs[:, 0])
        # maxy = np.argmax(pvs[:, 1])
        minax = np.argmin(np.abs(pvs[:, 0]))
        minay = np.argmin(np.abs(pvs[:, 1]))
        # Order the vectors by x-dominant, y-dominant, with both dominant vector components > 0, ie
        # PV -> [[+Lx, small], [small, +Ly]]
        PV = np.vstack((pvs[minax], pvs[minay]))
        if check:
            print 'le.PVxydict2PV: PV = ', PV
    else:
        PV = pvs

    return PV


def PVxydict2PVxPVy(PVxydict, NL, KL, check=False):
    """Convert dictionary of periodic bonds (keys) to periodic vectors (values)
    denoting periodic 'virtual' displacement for bond (i,j) of particle j as viewed by i.
    Note: added KL argument on 11-6-2017 to allow multiple bonds between a given pair of nodes.
    todo: Note that I'm fixing this so that two particles can be connected by both a real and periodic bond, or
    even multiple periodic bonds (need to test this though)

    Parameters
    ----------
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have
        the maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : #pts x max(#neighbors) int array
        spring connection/constant list, where 1 corresponds to a true connection,
        0 signifies that there is not a connection, -1 signifies periodic bond
    check : bool
        view intermediate results

    Returns
    ----------
    PVx : NP x NN float array
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    """
    if NL is None or np.size(NL) == 0:
        raise RuntimeError('Must supply non-empty NL to PVxydict2PVxPVy.')
    PVx = np.zeros_like(NL, dtype='float')
    PVy = np.zeros_like(NL, dtype='float')
    unused = np.ones_like(NL, dtype=bool)

    if PVxydict is None:
        print 'Warning: supplied PVxydict is None, so passing PVx and PVy as eros'
        PVxydict = {}

    for key in PVxydict:
        if len(np.shape(PVxydict[key])) > 1:
            for val in PVxydict[key]:
                if check:
                    print 'key = ', key
                    print 'NL[key[0]] = ', NL[key[0]]

                # Add ii, jj element
                col = np.argwhere(np.logical_and(unused[key[0]],
                                                 np.logical_and(NL[key[0]] == key[1], KL[key[0]] < 0)))[0]
                PVx[key[0], col] = val[0]
                PVy[key[0], col] = val[1]
                unused[key[0], col] = False
                # Add jj, ii element
                col = np.argwhere(np.logical_and(unused[key[1]],
                                                 np.logical_and(NL[key[1]] == key[0], KL[key[1]] < 0)))[0]
                PVx[key[1], col] = -val[0]
                PVy[key[1], col] = -val[1]
                unused[key[1], col] = False

        else:
            if check:
                print 'key = ', key
                print 'NL[key[0]] = ', NL[key[0]]
            # Add ii, jj element
            col = np.argwhere(np.logical_and(unused[key[0]],
                                             np.logical_and(NL[key[0]] == key[1], KL[key[0]] < 0)))[0]
            PVx[key[0], col] = PVxydict[key][0]
            PVy[key[0], col] = PVxydict[key][1]
            unused[key[0], col] = False
            # Add jj, ii element
            col = np.argwhere(np.logical_and(unused[key[1]],
                                             np.logical_and(NL[key[1]] == key[0], KL[key[1]] < 0)))[0]
            PVx[key[1], col] = -PVxydict[key][0]
            PVy[key[1], col] = -PVxydict[key][1]
            unused[key[1], col] = False

    return PVx, PVy


def flexible_PVxydict2PVxPVy(PVxydict, NL, KL, check=False):
    """Flexibly (defined later) convert dictionary of periodic bonds (keys) to periodic vectors (values)
    denoting periodic 'virtual' displacement for bond (i,j) of particle j as viewed by i.
    The flexibility is in that if there is no matching neighbor for a periodic bond, it is destroyed:
    ie if col = np.argwhere(NL[key[0]] == key[1])[0][0] returns IndexError, delete key from PVxydict and don't add
    anything for that bond to PVx and PVy

    Parameters
    ----------
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have
        the maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    check : bool
        view intermediate results

    Returns
    ----------
    PVx : NP x NN float array
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    """
    if NL is None or np.size(NL) == 0:
        raise RuntimeError('Must supply non-empty NL to PVxydict2PVxPVy.')
    PVx = np.zeros_like(NL, dtype='float')
    PVy = np.zeros_like(NL, dtype='float')
    PVxydict_out = {}
    for key in PVxydict:
        if check:
            print 'le: key = ', key
            print 'le: NL[key[0]] = ', NL[key[0]]
        try:
            col = np.argwhere(NL[key[0]] == key[1])[0][0]
            PVx[key[0], col] = PVxydict[key][0]
            PVy[key[0], col] = PVxydict[key][1]
            col = np.argwhere(NL[key[1]] == key[0])[0]
            PVx[key[1], col] = -PVxydict[key][0]
            PVy[key[1], col] = -PVxydict[key][1]
            PVxydict_out[key] = PVxydict[key]
        except IndexError:
            print 'le: skipping this key: ', key


def PVxy2PVxydict(PVx, PVy, NL, KL=None, eps=1e-7, check=False):
    """Convert periodic vector arrays PVx and PVy into a dictionary with keys for each periodic
    bond and values of the vector

    Parameters
    ----------
    PVx : NP x NN float array
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
    PVy : NP x NN float array
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array (optional, for speed, could be None)
        Connectivity list, with -1 entries for periodic bonds
    eps : float
        Threshold norm value for periodic boundary vector to be recognized as such

    Returns
    ----------
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    """
    if KL is None:
        todo = np.where(abs(PVx) > eps)[0]
        todo = np.where(abs(PVy) > eps)[0]
        """finish this by taking unique intersection"""
        raise RuntimeError('havenot written this case yet')
    else:
        rows, cols = np.where(KL == -1)

    PVxydict = {}
    kk = 0
    for ii in rows:
        # Get jj from NL
        jj = NL[ii, cols[kk]]
        print 'pair is ', (ii, jj)
        print 'check0 =', (ii, jj) not in PVxydict and (jj, ii) not in PVxydict,  \
              (ii, jj) in PVxydict and (jj, ii) in PVxydict,  \
              (ii, jj) in PVxydict,  \
              (jj, ii) in PVxydict
        # make sure we haven't already added the bond in reverse order
        if (ii, jj) not in PVxydict and (jj, ii) not in PVxydict:
            PVxydict[(ii, jj)] = np.array([PVx[ii, cols[kk]], PVy[ii, cols[kk]]])
        elif (ii, jj) in PVxydict and (jj, ii) in PVxydict:
            msg = 'Both (%d, %d) and (%d, %d) in PVxydict!' % (ii, jj, jj, ii)
            raise RuntimeError(msg)
        elif (ii, jj) in PVxydict:
            print 'stacking pvv to PVxydict(%d, %d)' % (ii, jj)
            pvv = np.array([PVx[ii, cols[kk]], PVy[ii, cols[kk]]])
            if (PVxydict[(ii, jj)] != pvv).any():
                PVxydict[(ii, jj)] = np.vstack((PVxydict[(ii, jj)], pvv))
        elif (jj, ii) in PVxydict:
            # Here we assume that the interaction is reciprocal, so only include if it is novel
            pvv = np.array([PVx[ii, cols[kk]], PVy[ii, cols[kk]]])
            if (PVxydict[(jj, ii)] != -pvv).any():
                PVxydict[(jj, ii)] = np.vstack((-PVxydict[(jj, ii)], pvv))
        kk += 1

    # Check it
    if check:
        print 'le: PVxydict = ', PVxydict

    return PVxydict


def BL2PVxydict(BL, xy, PV):
    """Extract dictionary of periodic boundary condition vectors from bond list, particle positions

    Parameters
    ----------
    BL : #bonds x 2 int array
        Bond list array, with negative-valued rows denoting periodic bonds
    xy : NP x 2 float array
        positions of the particles in 2D
    PV : #(periodic sides) x 2 float array
        array of vectors taking each periodic side to its matching opposite side

    Returns
    ----------
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    """
    # The ijth element of PVx is the xcomponent of the vector taking NL[i,j] to its image as seen by particle i.
    PVxydict = {}
    # check both directions along each periodic vector
    PVtmp = np.vstack((PV, -PV))

    # For each bond that is a periodic bond, determine its periodic boundary vector (a row of the array PV)
    pBs = np.unique(np.where(BL < 0)[0])
    print 'le: BL[pBs] = ', BL[pBs]
    print 'le: pBs = ', pBs
    for ind in pBs:
        # Find the PV (periodic vector) that brings the second particle (j) closest to the first (i).
        # This will be PVxydict[(i,j)], since particle i sees j at xy[j]+PVxydict[(i,j)]
        a1 = xy[np.abs(BL[ind, 0])]
        a2 = xy[np.abs(BL[ind, 1])]
        distxy = a2 + PVtmp - a1
        dist = distxy[:, 0] ** 2 + distxy[:, 1] ** 2
        # print 'a1, a2 = ', a1, a2
        # print 'distxy = ', distxy
        # print 'PV = ', PV
        # print 'dist = ', dist
        if np.argmin(dist) > len(PV) - 1:
            PVxydict[(np.abs(BL[ind, 0]), np.abs(BL[ind, 1]))] = -PV[np.argmin(dist) % len(PV)]
        else:
            PVxydict[(np.abs(BL[ind, 0]), np.abs(BL[ind, 1]))] = PV[np.argmin(dist) % len(PV)]

    print 'le: PVxydict = ', PVxydict
    return PVxydict


def extract_phase(eigvector, point_arr=[]):
    """
    Extract phase information from an eigenvector.

    Parameters
    ----------
    eigvector : 2 x NP complex array
        First row is X component, second is Y.
    point_arr : M x 1 int array or empty for all points
        array of indices for which to order the output, in desired order.
        If empty, gets phase info for all elements/points.

    Returns
    ----------
    phase : NP x 1 float array
        The phase of the array
    """
    pa = point_arr
    if np.size(pa) == 0:
        pa = np.arange(len(evY))

    evX = eigvector[2 * pa]
    evY = eigvector[2 * pa + 1]
    phase = np.arctan2(evY.real, evX.real)
    # print 'evY[0] =', evY[0]
    # print 'evX[0] =', evX[0]
    # print 'phase[0] = ', phase[0]
    return phase


# def argTRI_delaunay_cut_unnatural_boundary(xy,NL,KL,BL,TRI, thres):
#     '''Keep cutting skinny triangles on the boundary until no more skinny ones.
#     Cuts boundary tris of a triangulation until all have reasonable height/base values.
#
#     Parameters
#     ----------
#     xy : NP x 2 float array
#         The point set
#     BL : Nbonds x 2 int array
#         Bond list for the lattice (can include bonds that aren't triangulated)
#     TRI : Ntris x 3 int array
#         The triangulation of the lattice/mesh
#     thres : float
#         threshold value for height/base, below which to cut the boundary tri
#
#     Returns
#     ----------
#     keepTRI : Ntris x 1 bool array
#         whether to keep (True) or cut (False) each triangle based on how squished it is,
#         for trimming bad boundary tris after triangulation
#     '''
#     NP = len(xy)
#     NN = np.shape(NL)[1]
#     print ' delaunay_cut_unnatural_boundary : extract boundary...'
#     boundary = extract_boundary(xy,NL,KL, BL)
#     Ncut = 1
#     dmyi = 0
#     while Ncut>0:
#         print 'cutting pass '+str(dmyi)
#         BL, TRItrim, Ncut = delaunay_cut_unnatural_boundary_singlepass(xy,BL,TRI,boundary,thres)
#         TRI = BL2TRI(BL)
#         NL, KL = BL2NLandKL(BL,NP=NP,NN=NN)
#         #print ' --> extract new boundary...'
#         boundary = extract_boundary(xy,NL,KL, BL)
#         dmyi += 1
#
#     return keepTRI


def delaunay_cut_unnatural_boundary(xy, NL, KL, BL, TRI, thres, check=False):
    """Keep cutting skinny triangles on the boundary until no more skinny ones.
    Cuts boundary tris of a triangulation until all have reasonable height/base values.

    Parameters
    ----------
    xy : NP x 2 float array
        The point set
    NL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Neighbor list. The ith row has neighbors of the ith particle, padded with zeros
    KL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Connectivity list. The ith row has ones where ith particle is connected to NL[i,j]
    BL : Nbonds x 2 int array
        Bond list for the lattice (can include bonds that aren't triangulated)
    TRI : Ntris x 3 int array
        The triangulation of the lattice/mesh
    thres : float
        threshold value for base/height, below which to cut the boundary tri
    check: bool
        Display intermediate results

    Returns
    ----------
    NL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Neighbor list. The ith row has neighbors of the ith particle, padded with zeros
    KL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Connectivity list. The ith row has ones where ith particle is connected to NL[i,j]
    BL : Nbonds x 2 int array
        Bond list for the lattice (can include bonds that aren't triangulated)
    TRI : (Ntris - Ncut) x 3 int array
        The new, trimmed triangulation
    """
    # Computes-->
    # boundary : # points on boundary x 1 int array
    #     The indices of the points that live on the boundary
    NP = len(xy)
    NN = np.shape(NL)[1]
    print ' delaunay_cut_unnatural_boundary : extract boundary...'
    boundary = extract_boundary(xy, NL, KL, BL, check=check)
    Ncut = 1
    dmyi = 0
    while Ncut > 0:
        print 'cutting pass ' + str(dmyi)
        BL, Ncut = delaunay_cut_unnatural_boundary_singlepass(xy, BL, TRI, boundary, thres, check=check)
        TRI = BL2TRI(BL, xy)
        NL, KL = BL2NLandKL(BL, NP=NP, NN=NN)
        # print ' --> extract new boundary...'
        boundary = extract_boundary(xy, NL, KL, BL)
        dmyi += 1

    return NL, KL, BL, TRI


def delaunay_cut_unnatural_boundary_singlepass(xy, BL, TRI, boundary, thres=4.0, check=False):
    """
    Algorithm: For each row of TRI containing at least one boundary pt,
    if contains 2 boundary pts, cut the base (edge) if base/height > threshold.
    If the tri has all three vertices on the boundary, keep it.
    Two boundary triangles may share an edge (bond), and removing both will leave
    a dangling point. To avoid this, check for edges shared between simplices.
    This allows bonds which are not part of the triangulation.
    Haven't inspected for periodicity issues.

    Parameters
    ----------
    xy : NP x 2 float array
        The point set
    BL : Nbonds x 2 int array
        Bond list for the lattice (can include bonds that aren't triangulated)
    TRI : Ntris x 3 int array
        The triangulation of the lattice/mesh
    boundary : # points on boundary x 1 int array
        The indices of the points that live on the boundary
    thres : float
        threshold value for base/height, below which to cut the boundary tri
    check: bool
        Display intermediate results

    Returns
    ----------
    TRItrim : (Ntris - Ncut) x 3 int array
        The new, trimmed triangulation
    Ncut : int
        The number of rows of TRI that have been cut (# boundary tris cut)

    """
    # btri are indices of TRI rows that are boundary triangles
    btri = boundary_triangles(TRI, boundary)

    # CHECK
    if check:
        zfaces = np.zeros(len(TRI), dtype=float)
        zfaces[btri] = 1.
        # Color the boundary triangles in a plot
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tripcolor(xy[:, 0], xy[:, 1], TRI, facecolors=zfaces, edgecolors='k')
        plt.colorbar()
        for i in range(len(xy)):
            plt.text(xy[i, 0] - 0.2, xy[i, 1], str(i))
        plt.title('tripcolor() showing boundary tris')
        plt.show()

    # preallocate for speed --> to cut from BL, TRI
    rows2cut = np.zeros(len(BL), dtype=bool)
    # rowsTRI2cut = np.zeros(len(TRI),dtype=bool)

    for ii in range(len(btri)):
        # Check if base/height >thres for this row
        row = TRI[btri[ii]]
        # print 'row = ', row, ' btri[ii] = ', btri[ii]
        onB_bool = np.array([row[i] in boundary for i in range(len(row))])
        onB = np.where(onB_bool)[0]

        # Two possibilities:
        # (1) The tri has two vertices on the boundary --> keep it if aspect ratio is ok
        # (2) The tri has all three vertices on the boundary --> keep it
        # print 'onB = ', onB
        # print 'onB_bool.all()  =', onB_bool.all()

        if not onB_bool.all():
            # identify the point that lies across from the base of the scalene tri
            pt3 = xy[row[np.setdiff1d(np.arange(3), onB)], :][0]
            # identify the base points
            base = np.array([xy[row[onB[0]], :], xy[row[onB[1]], :]])
            # identify the point closest to pt3 on lineseg
            p = lseg.closest_pt_along_line(pt3, base[0], base[1])

            # calc base and height
            baseL = float(np.linalg.norm(base[0] - base[1]))
            height = float(np.linalg.norm(pt3 - p))

            if (baseL / height) > thres:
                # if too skinny, kill the row of BL corresponding to the base
                BLp0 = min(row[onB])
                BLp1 = max(row[onB])

                # print 'row[onB] = ', row[onB]
                # print 'BL0 = ', BLp0
                # print 'BL1 = ', BLp1

                # Mark for BL removal
                indBL2cut = np.where((BL == (BLp0, BLp1)).all(axis=1))[0][0]
                rows2cut[indBL2cut] = True

                # Mark for TRI removal--> not natural since indices of TRI are changes
                # rowsTRI2cut[btri[ii]] = True

                # print 'boundary =', boundary
                # print 'onB =', onB
                # print 'xy ind = ', row[np.setdiff1d(np.arange(3),onB)]
                # print 'pt3 = ', pt3
                # print 'base = ', base
                # print 'p = ', p
                # print 'baseL = ', baseL
                # print 'height = ', height
                # print 'baseL/height = ', baseL/height

                # check
                # keepTRI = np.ones(len(TRI),dtype=bool)
                # keepTRI[btri[rows2cut]] = False
                # TRItest = TRI[keepTRI]
                # plt.triplot(xy[:,0], xy[:,1], TRItest, 'bo-')
                # plt.plot([p[0],pt3[0]],[p[1],pt3[1]],'k.-')
                # plt.show()

    # Not natural since indices of TRI are changes!
    # keepTRI = np.ones(len(TRI),dtype=bool)
    # keepTRI[rowsTRI2cut] = False
    # TRItrim = TRI[keepTRI]

    keepBL = np.ones(len(BL), dtype=bool)
    keepBL[rows2cut] = False
    BLtrim = BL[keepBL]

    # check
    # print 'Checking in delaunay_cut_unnatural_boundary ...'
    # display_lattice_2D(xy,BLtrim)
    # plt.triplot(xy[:,0], xy[:,1], TRItrim, 'ro-')
    # plt.show()

    Ncut = len(np.where(rows2cut)[0])
    return BLtrim, Ncut


def boundary_triangles(TRI, boundary):
    """Identify triangles of triangulation that live on the boundary
    (ie share one edge with no other simplices)
    """
    # Look for triangles in TRI that contain 2 elements on the boundary
    # (ie they have a boundary edge in the triangle)
    inb0 = np.where(np.in1d(TRI[:, 0], boundary))[0]
    inb1 = np.where(np.in1d(TRI[:, 1], boundary))[0]
    inb2 = np.where(np.in1d(TRI[:, 2], boundary))[0]
    inb_all = np.hstack((inb0, inb1, inb2)).ravel()
    # print 'inb_all = ', inb_all

    # Look for indices that appear twice in cat( inb0,inb1,inb2).
    s = np.sort(inb_all, axis=None)
    btris = s[s[1:] == s[:-1]]

    # If any values are repeated in btri, that means all three vertices are boundary.
    # Keep these. Also, remove from the list any tris that share two points with one of these tris.
    # --> this is because this means an edge (not a boundary edge) connects two boundary particles,
    # and cuts off another particle.
    btri_repeats = btris[btris[1:] == btris[:-1]]
    # print 'TRI = ', TRI
    # print 'btris = ', btris
    # print 'btri_repeats = ', btri_repeats

    # btri = np.setdiff1d(btris,btri_repeats)
    btris = np.unique(btris)

    # If any btri triangles share an edge with a btri_repeats (they share 2 points),
    # kill the btri triangle.
    mask = np.ones(len(btris), dtype=bool)
    for ii in range(len(btris)):
        # if this one isn't itself a repeat, check against all brtri_repeats
        if not np.in1d(btris[ii], btri_repeats):
            tri0 = TRI[btris[ii]]
            for btr in btri_repeats:
                tri1 = TRI[btr]
                if len(np.intersect1d(tri0, tri1, assume_unique=True)) > 1:
                    # print 'matching = ', np.intersect1d(tri0,tri1,assume_unique=True)
                    mask[ii] = False
    btri = btris[mask]

    return btri


def extract_polygons_lattice(xy, BL, NL=None, KL=None, PVx=None, PVy=None, PVxydict=None, viewmethod=False,
                             check=False):
    """ Extract polygons from a lattice of points.
    Note that dangling bonds are removed, but no points are removed. This allows correct indexing for PVxydict keys, if supplied.

    Parameters
    ----------
    xy : NP x 2 float array
        points living on vertices of dual to triangulation
    BL : Nbonds x 2 int array
        Each row is a bond and contains indices of connected points
    NL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Neighbor list. The ith row has neighbors of the ith particle, padded with zeros
    KL : NP x NN int array (optional, speeds up calc if it is known there are no dangling bonds)
        Connectivity list. The ith row has ones where ith particle is connected to NL[i,j]
    PVx : NP x NN float array (optional, for periodic lattices and speed)
        ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen by particle i
        If PVx and PVy are specified, PVxydict need not be specified.
    PVy : NP x NN float array (optional, for periodic lattices and speed)
        ijth element of PVy is the y-component of the vector taking NL[i,j] to its image as seen by particle i
        If PVx and PVy are specified, PVxydict need not be specified.
    PVxydict : dict (optional, for periodic lattices)
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    viewmethod: bool
        View the results of many intermediate steps
    check: bool
        Check the initial and final result

    Returns
    ----------
    polygons : list of lists of ints
        list of lists of indices of each polygon
    """
    NP = len(xy)

    if KL is None or NL is None:
        NL, KL = BL2NLandKL(BL, NP=NP, NN='min')
        if (BL < 0).any():
            if len(PVxydict) > 0:
                PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL)
            else:
                raise RuntimeError('Must specify either PVxydict or KL and NL in extract_polygons_lattice()' +
                                   ' when periodic bonds exist!')
    elif (BL < 0).any():
        if PVx is None or PVy is None:
            if PVxydict is None:
                raise RuntimeError('Must specify either PVxydict or PVx and PVy in extract_polygons_lattice()' +
                                   ' when periodic bonds exist!')
            else:
                PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL)

    NN = np.shape(KL)[1]
    # Remove dangling bonds
    # dangling bonds have one particle with only one neighbor
    finished_dangles = False
    while not finished_dangles:
        dangles = np.where([np.count_nonzero(row) == 1 for row in KL])[0]
        if len(dangles) > 0:
            # Check if need to build PVxy dictionary from PVx and PVy before changing NL and KL
            if (BL < 0).any() and len(PVxydict) == 0:
                PVxydict = PVxy2PVxydict(PVx, PVy, NL, KL=KL)

            # Make sorted bond list of dangling bonds
            dpair = np.sort(np.array([[d0, NL[d0, np.where(KL[d0] != 0)[0]]] for d0 in dangles]), axis=1)
            # Remove those bonds from BL
            BL = dh.setdiff2d(BL, dpair.astype(BL.dtype))
            # print 'dpair = ', dpair
            # print 'ending BL = ', BL
            NL, KL = BL2NLandKL(BL, NP=NP, NN=NN)

            # Now that NL and KL rebuilt (changed), (re)build PVx and PVy if periodic bcs
            if (BL < 0).any():
                if len(PVxydict) > 0:
                    PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL)
        else:
            finished_dangles = True

    if viewmethod or check:
        print 'Plotting result after chopped dangles, if applicable...'
        display_lattice_2D(xy, BL, NL=NL, KL=KL, PVx=PVx, PVy=PVy, PVxydict=PVxydict,
                           title='Result after chopping dangling bonds', close=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
        plt.show()

    # bond markers for counterclockwise, clockwise
    used = np.zeros((len(BL), 2), dtype=bool)
    polygons = []
    finished = False
    if viewmethod:
        f, (ax1, ax2) = plt.subplots(1, 2)

    # For periodicity, remember which bonds span periodic boundary
    periB = np.array([(row < 0).any() for row in BL])

    if periB.any() and PVxydict is None and (PVx is None or PVy is None):
        raise RuntimeError('Periodic boundaries have been detected, but no periodic vectors supplied to ' +
                           'extract_polygons_lattice()')

    if not periB.any():
        print 'no PBCs, calculating polygons...'
        while not finished:
            # Check if all bond markers are used in order A-->B
            # print 'Checking AB (A-->B): '
            todoAB = np.where(~used[:, 0])[0]
            # print 'len(todoAB) = ', len(todoAB)
            # print 'used = ', used
            # print 'todoAB = ', todoAB
            if len(todoAB) > 0:
                bond = BL[todoAB[0]]

                # bb will be list of polygon indices
                # Start with orientation going from bond[0] to bond[1]
                nxt = bond[1]
                bb = [bond[0], nxt]
                dmyi = 1

                ###############
                # check
                if viewmethod:
                    ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                    ax1.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                 xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                 arrowprops=dict(arrowstyle="->",
                                                 color="r",
                                                 shrinkA=5, shrinkB=5,
                                                 patchA=None,
                                                 patchB=None,
                                                 connectionstyle="arc3,rad=0.2", ), )
                    for i in range(len(xy)):
                        ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
                    ax2.imshow(used)
                    ax1.set_aspect('equal')
                ###############

                # as long as we haven't completed the full outer polygon, add next index
                while nxt != bond[0]:
                    n_tmp = NL[nxt, np.argwhere(KL[nxt]).ravel()]
                    # Exclude previous boundary particle from the neighbors array, unless its the only one
                    # (It cannot be the only one, if we removed dangling bonds)
                    if len(n_tmp) == 1:
                        '''The bond is a lone bond, not part of a triangle.'''
                        neighbors = n_tmp
                    else:
                        neighbors = np.delete(n_tmp, np.where(n_tmp == bb[dmyi - 1])[0])

                    angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[nxt, 1], xy[neighbors, 0] - xy[nxt, 0]).ravel() \
                                    - np.arctan2(xy[bb[dmyi - 1], 1] - xy[nxt, 1],
                                                 xy[bb[dmyi - 1], 0] - xy[nxt, 0]).ravel(), 2 * np.pi)
                    nxt = neighbors[angles == max(angles)][0]
                    bb.append(nxt)

                    ###############
                    # # Check
                    # if viewmethod:
                    #     plt.annotate("", xy=(xy[bb[dmyi],0],xy[bb[dmyi],1] ), xycoords='data',
                    #             xytext=(xy[nxt,0], xy[nxt,1]), textcoords='data',
                    #             arrowprops=dict(arrowstyle="->",
                    #                             color="r",
                    #                             shrinkA=5, shrinkB=5,
                    #                             patchA=None,
                    #                             patchB=None,
                    #                             connectionstyle="arc3,rad=0.2",),  )
                    #
                    ###############

                    # Now mark the current bond as used
                    thisbond = [bb[dmyi - 1], bb[dmyi]]
                    # Get index of used matching thisbond
                    mark_used = np.where((np.logical_or(BL == bb[dmyi - 1], BL == bb[dmyi])).all(axis=1))
                    # mark_used = np.where((BL == thisbond).all(axis=1))
                    if not used[mark_used, 0]:
                        # print 'marking bond [', thisbond, '] as used'
                        used[mark_used, 0] = True
                    else:
                        # Get index of used matching reversed thisbond (this list boolean is directional)
                        # mark_used = np.where((BL == thisbond[::-1]).all(axis=1))
                        # Used this bond in reverse order
                        used[mark_used, 1] = True
                    # print 'used = ', used
                    dmyi += 1

                polygons.append(bb)
                ###############
                # Check new polygon
                if viewmethod:
                    ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                    for i in range(len(xy)):
                        ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
                    for dmyi in range(len(bb)):
                        nxt = bb[np.mod(dmyi + 1, len(bb))]
                        ax1.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                     xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                     arrowprops=dict(arrowstyle="->",
                                                     color="r",
                                                     shrinkA=5, shrinkB=5,
                                                     patchA=None,
                                                     patchB=None,
                                                     connectionstyle="arc3,rad=0.2", ), )
                    ax2.cla()
                    ax2.imshow(used)
                    plt.pause(0.00001)
                    ###############

            else:
                # Check for remaining bonds unused in reverse order (B-->A)
                # print 'CHECKING REVERSE (B-->A): '
                todoBA = np.where(~used[:, 1])[0]
                # print 'len(todoBA) = ', len(todoBA)
                if len(todoBA) > 0:
                    bond = BL[todoBA[0]]

                    ###############
                    # # check
                    # if viewmethod:
                    #     plt.annotate("", xy=(xy[bb[dmyi],0],xy[bb[dmyi],1] ), xycoords='data',
                    #             xytext=(xy[nxt,0], xy[nxt,1]), textcoords='data',
                    #             arrowprops=dict(arrowstyle="->",
                    #                         color="b",
                    #                         shrinkA=5, shrinkB=5,
                    #                         patchA=None,
                    #                         patchB=None,
                    #                         connectionstyle="arc3,rad=0.6",),  )
                    # ###############

                    # bb will be list of polygon indices
                    # Start with orientation going from bond[0] to bond[1]
                    nxt = bond[0]
                    bb = [bond[1], nxt]
                    dmyi = 1

                    # as long as we haven't completed the full outer polygon, add nextIND
                    while nxt != bond[1]:
                        n_tmp = NL[nxt, np.argwhere(KL[nxt]).ravel()]
                        # Exclude previous boundary particle from the neighbors array, unless its the only one
                        # (It cannot be the only one, if we removed dangling bonds)
                        if len(n_tmp) == 1:
                            '''The bond is a lone bond, not part of a triangle.'''
                            neighbors = n_tmp
                        else:
                            neighbors = np.delete(n_tmp, np.where(n_tmp == bb[dmyi - 1])[0])

                        angles = np.mod(np.arctan2(xy[neighbors, 1] - xy[nxt, 1], xy[neighbors, 0] - xy[nxt, 0]).ravel() \
                                        - np.arctan2(xy[bb[dmyi - 1], 1] - xy[nxt, 1],
                                                     xy[bb[dmyi - 1], 0] - xy[nxt, 0]).ravel(), 2 * np.pi)
                        nxt = neighbors[angles == max(angles)][0]
                        bb.append(nxt)

                        ###############
                        # Check
                        # if viewmethod:
                        #     plt.annotate("", xy=(xy[bb[dmyi],0],xy[bb[dmyi],1] ), xycoords='data',
                        #         xytext=(xy[nxt,0], xy[nxt,1]), textcoords='data',
                        #         arrowprops=dict(arrowstyle="->",
                        #                     color="b",
                        #                     shrinkA=5, shrinkB=5,
                        #                     patchA=None,
                        #                     patchB=None,
                        #                     connectionstyle="arc3,rad=0.6", #connectionstyle,
                        #                     ),  )
                        ###############

                        # Now mark the current bond as used --> note the inversion of the bond order to match BL
                        thisbond = [bb[dmyi], bb[dmyi - 1]]
                        # Get index of used matching [bb[dmyi-1],nxt]
                        mark_used = np.where((BL == thisbond).all(axis=1))
                        if len(mark_used) > 0:
                            used[mark_used, 1] = True
                        else:
                            raise RuntimeError(
                                'Cannot mark polygon bond as used: this bond was already used in its attempted orientation. (All bonds in first column should already be marked as used.)')

                        dmyi += 1

                    polygons.append(bb)

                    # Check new polygon
                    if viewmethod:
                        ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                        for i in range(len(xy)):
                            ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
                        for dmyi in range(len(bb)):
                            nxt = bb[np.mod(dmyi + 1, len(bb))]
                            ax1.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                         xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                         arrowprops=dict(arrowstyle="->",
                                                         color="b",
                                                         shrinkA=5, shrinkB=5,
                                                         patchA=None,
                                                         patchB=None,
                                                         connectionstyle="arc3,rad=0.6", ), )
                        ax2.cla()
                        ax2.imshow(used)
                        plt.pause(0.00001)
                        ###############

                else:
                    # All bonds have been accounted for
                    finished = True
    else:
        print 'detected periodicity...'
        # get particles on the finite (non-periodic) system's boundary. This allows massive speedup.
        KLfin = np.zeros_like(KL)
        KLfin[KL > 0] = 1
        # Create BLfin to pass to extract_boundary()
        prows = np.where(BL < 0)[0]
        nprows = np.setdiff1d(np.arange(len(BL)), prows)
        if check:
            print 'rows of BL that are periodic: ', prows
            print 'BL[prows] = ', BL[prows]
        BLfin = BL[nprows]
        finbd = extract_boundary(xy, NL, KLfin, BLfin, check=check)

        # If there were dangling points in the non-periodic representation, then we need to add those to finbd because
        # they will have periodic bonds attached to them.
        dangles = np.where(~KLfin.any(axis=1))[0]
        print 'dangles = ', dangles
        if len(dangles) > 0:
            print 'Found dangling points in the finite/non-periodic representation. Adding to finbd...'
            finbd = np.hstack((finbd, np.array(dangles)))

        if check:
            print 'finite boundary: finbd = ', finbd
            plt.clf()
            display_lattice_2D(xy, BL, NL=NL, KL=KLfin, PVx=PVx, PVy=PVy, PVxydict=PVxydict,
                               title='Identified finite boundary', close=False)
            for i in range(len(xy)):
                plt.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
            plt.plot(xy[finbd, 0], xy[finbd, 1], 'ro')
            plt.show()
        first_check = True

        # Then erase periodicity in BL
        BL = np.abs(BL)

        while not finished:
            if len(polygons) % 20 == 0:
                print 'constructed ', len(polygons), ' polygons...'
            # Check if all bond markers are used in order A-->B
            # print 'Checking AB (A-->B): '
            todoAB = np.where(~used[:, 0])[0]
            # print 'len(todoAB) = ', len(todoAB)
            # print 'used = ', used
            # print 'todoAB = ', todoAB
            if len(todoAB) > 0:
                bond = BL[todoAB[0]]

                # bb will be list of polygon indices
                # Start with orientation going from bond[0] to bond[1]
                nxt = bond[1]
                bb = [bond[0], nxt]
                dmyi = 1

                # define 'previous angle' as backwards of current angle -- ie angle(prev-current_pos)
                # Must include effect of PV on this angle -- do in ref frame of nxt particle
                PVind = np.argwhere(NL[nxt] == bond[0])[0][0]
                addx = PVx[nxt, PVind]
                addy = PVy[nxt, PVind]
                xyb0 = xy[bond[0], :] + np.array([addx, addy])
                prev_angle = np.arctan2(xyb0[1] - xy[nxt, 1], xyb0[0] - xy[nxt, 0]).ravel()

                ###############
                # check
                if viewmethod:
                    if first_check:
                        ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                        for i in range(len(xy)):
                            ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))
                        first_check = False

                    ax1.annotate("", xy=(xy[bb[dmyi - 1], 0], xy[bb[dmyi - 1], 1]), xycoords='data',
                                 xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                 arrowprops=dict(arrowstyle="->",
                                                 color="r",
                                                 shrinkA=5, shrinkB=5,
                                                 patchA=None,
                                                 patchB=None,
                                                 connectionstyle="arc3,rad=0.2", ), )
                    ax2.imshow(used, aspect=1. / len(used), interpolation='none')
                    ax1.set_aspect('equal')
                ###############

                # as long as we haven't completed the full outer polygon, add next index
                while nxt != bond[0]:
                    # print nxt
                    #            o     o neighbors
                    #             \   /
                    #              \ /
                    #               o nxt
                    #             /
                    #           /
                    #         o  bb[dmyi-1]
                    #
                    n_tmp = NL[nxt, np.argwhere(KL[nxt]).ravel()]
                    # Exclude previous boundary particle from the neighbors array, unless its the only one
                    # (It cannot be the only one, if we removed dangling bonds)
                    if len(n_tmp) == 1:
                        '''The bond is a lone bond, not part of a triangle/polygon.'''
                        neighbors = n_tmp
                    else:
                        neighbors = np.delete(n_tmp, np.where(n_tmp == bb[dmyi - 1])[0])

                    # check if neighbors CAN be connected across periodic bc--
                    #  ie if particle on finite boundary (finbd)
                    if nxt in finbd:
                        # Since on finite system boundary, particle could have periodic bonds
                        # Find x values to add to neighbors, by first getting indices of row of
                        # PV (same as of NL) matching neighbors
                        PVinds = [np.argwhere(NL[nxt] == nnn)[0][0] for nnn in neighbors]
                        addx = PVx[nxt, PVinds]
                        addy = PVy[nxt, PVinds]

                        xynb = xy[neighbors, :] + np.dstack([addx, addy])[0]
                        xynxt = xy[nxt, :]
                        current_angles = np.arctan2(xynb[:, 1] - xynxt[1], xynb[:, 0] - xynxt[0]).ravel()
                        angles = np.mod(current_angles - prev_angle, 2 * np.pi)

                        if check:
                            print '\n'
                            print 'particle ', nxt, ' is on finbd'
                            print 'nxt = ', nxt
                            print 'neighbors = ', neighbors
                            print 'xy[neighbors,:] =', xy[neighbors, :]
                            print 'addxy = ', np.dstack([addx, addy])[0]
                            print 'xynb = ', xynb
                            print 'xynxt = ', xynxt
                            print 'current_angles = ', current_angles
                            print 'prev_angle = ', prev_angle
                            print 'angles = ', angles
                            print 'redefining nxt = ', neighbors[angles == max(angles)][0]

                        # redefine previous angle as backwards of current angle -- ie angle(prev-current_pos)
                        prev_angletmp = np.arctan2(xynxt[1] - xynb[:, 1], xynxt[0] - xynb[:, 0]).ravel()
                        prev_angle = prev_angletmp[angles == max(angles)][0]

                        # print 'prev_angletmp = ', prev_angletmp
                        # print 'prev_angle = ', prev_angle
                        # print 'NL[nxt] = ', NL[nxt]
                        # print 'bb = ', bb

                        # CHECK
                        # ax1 = plt.gca()
                        # ax1.plot(xy[:,0],xy[:,1],'k.')
                        # for i in range(len(xy)):
                        #    ax1.text(xy[i,0]+0.2,xy[i,1],str(i))
                        # plt.show()


                    else:
                        current_angles = np.arctan2(xy[neighbors, 1] - xy[nxt, 1],
                                                    xy[neighbors, 0] - xy[nxt, 0]).ravel()
                        angles = np.mod(current_angles - prev_angle, 2 * np.pi)
                        # redefine previous angle as backwards of current angle -- ie angle(prev-current_pos)
                        # prev_angle = np.arctan2(xy[bb[dmyi-1],1] - xynxt[1], xy[bb[dmyi-1],0] - xynxt[0] ).ravel()
                        xynxt = xy[nxt, :]
                        xynb = xy[neighbors, :]
                        prev_angletmp = np.arctan2(xynxt[1] - xy[neighbors, 1], xynxt[0] - xy[neighbors, 0]).ravel()
                        prev_angle = prev_angletmp[angles == max(angles)][0]

                    nxt = neighbors[angles == max(angles)][0]
                    bb.append(nxt)

                    ###############
                    # # Check bond
                    if viewmethod:
                        # Check individually
                        # ax1 = plt.gca()
                        # ax1.plot(xy[:,0],xy[:,1],'k.')
                        if first_check:
                            for i in range(len(xy)):
                                ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))

                        plt.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                     xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                     arrowprops=dict(arrowstyle="->",
                                                     color="r",
                                                     shrinkA=5, shrinkB=5,
                                                     patchA=None,
                                                     patchB=None,
                                                     connectionstyle="arc3,rad=0.2", ), )

                    ###############

                    # Now mark the current bond as used
                    # thisbond = [bb[dmyi-1], bb[dmyi]]
                    # Get index of used matching thisbond
                    mark_used = np.where((np.logical_or(BL == bb[dmyi - 1], BL == bb[dmyi])).all(axis=1))[0]
                    # mark_used = np.where((BL == thisbond).all(axis=1))
                    # print 'mark_used = ', mark_used
                    # I think we need to adjust the line below to allow multiple entries in mark_used
                    if not used[mark_used, 0]:
                        # print 'marking bond [', thisbond, '] as used'
                        used[mark_used, 0] = True
                    else:
                        # Get index of used matching reversed thisbond (this list boolean is directional)
                        # mark_used = np.where((BL == thisbond[::-1]).all(axis=1))
                        # Used this bond in reverse order
                        used[mark_used, 1] = True
                    # print 'used = ', used
                    dmyi += 1
                    if check:
                        print 'bb = ', bb

                polygons.append(bb)
                ###############
                # Check new polygon
                if viewmethod:
                    if first_check:
                        ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                        for i in range(len(xy)):
                            ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))

                    for dmyi in range(len(bb)):
                        nxt = bb[np.mod(dmyi + 1, len(bb))]
                        ax1.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                     xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                     arrowprops=dict(arrowstyle="->",
                                                     color="r",
                                                     shrinkA=5, shrinkB=5,
                                                     patchA=None,
                                                     patchB=None,
                                                     connectionstyle="arc3,rad=0.2", ), )
                    ax2.cla()
                    ax2.imshow(used, aspect=1. / len(used), interpolation='none')
                    print 'polygons = ', polygons
                    # plt.show()
                    plt.pause(0.00001)
                    ###############

            else:
                # Check for remaining bonds unused in reverse order (B-->A)
                # print 'CHECKING REVERSE (B-->A): '
                todoBA = np.where(~used[:, 1])[0]
                # print 'len(todoBA) = ', len(todoBA)
                if len(todoBA) > 0:
                    bond = BL[todoBA[0]]

                    ###############
                    # # check
                    if viewmethod:
                        plt.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                     xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                     arrowprops=dict(arrowstyle="->",
                                                     color="b",
                                                     shrinkA=5, shrinkB=5,
                                                     patchA=None,
                                                     patchB=None,
                                                     connectionstyle="arc3,rad=0.6", ), )
                    # ###############

                    # bb will be list of polygon indices
                    # Start with orientation going from bond[0] to bond[1]
                    nxt = bond[0]
                    bb = [bond[1], nxt]
                    dmyi = 1

                    # define 'previous angle' as backwards of current angle -- ie angle(prev-current_pos)
                    # Must include effect of PV on this angle -- do in ref frame of nxt particle
                    PVind = np.argwhere(NL[nxt] == bond[1])[0][0]
                    addx = PVx[nxt, PVind]
                    addy = PVy[nxt, PVind]
                    xyb0 = xy[bond[1], :] + np.array([addx, addy])
                    prev_angle = np.arctan2(xyb0[1] - xy[nxt, 1], xyb0[0] - xy[nxt, 0])  # .ravel()

                    # print '\n---------\n'
                    # print 'bb start = ', bb
                    # print 'xy[nxt] = ', xy[nxt]
                    # print 'addx = ', addx
                    # print 'addy = ', addy
                    # print 'xyb0 = ', xyb0
                    # print 'prev_angle = ', prev_angle/np.pi
                    # print 'type(prev_angle) = ', type(prev_angle)

                    # as long as we haven't completed the full outer polygon, add nextIND
                    while nxt != bond[1]:
                        n_tmp = NL[nxt, np.argwhere(KL[nxt]).ravel()]
                        # Exclude previous boundary particle from the neighbors array, unless its the only one
                        # (It cannot be the only one, if we removed dangling bonds)
                        if len(n_tmp) == 1:
                            '''The bond is a lone bond, not part of a triangle.'''
                            neighbors = n_tmp
                        else:
                            neighbors = np.delete(n_tmp, np.where(n_tmp == bb[dmyi - 1])[0])

                        ########

                        # check if neighbors CAN be connected across periodic bc-- ie if particle on finite boundary (finbd)
                        if nxt in finbd:
                            # Since on finite system boundary, particle could have periodic bonds
                            # Find x values to add to neighbors, by first getting indices of row of PV (same as of NL) matching neighbors
                            # ALL CALCS in frame of reference of NXT particle
                            PVinds = [np.argwhere(NL[nxt] == nnn)[0][0] for nnn in neighbors]
                            addx = PVx[nxt, PVinds]
                            addy = PVy[nxt, PVinds]

                            xynb = xy[neighbors, :] + np.dstack([addx, addy])[0]
                            xynxt = xy[nxt, :]
                            # print '\n'
                            # print 'nxt = ', nxt
                            # print 'neighbors = ', neighbors
                            # print 'xy[neighbors,:] =', xy[neighbors,:]
                            # print 'addxy = ', np.dstack([addx, addy])[0]
                            # print 'xynb = ', xynb
                            # print 'xynxt = ', xynxt
                            current_angles = np.arctan2(xynb[:, 1] - xynxt[1], xynb[:, 0] - xynxt[0]).ravel()
                            angles = np.mod(current_angles - prev_angle, 2 * np.pi)
                            selectIND = np.where(angles == max(angles))[0][0]
                            # print 'selectIND = ', selectIND
                            # print 'current_angles = ', current_angles/np.pi
                            # print 'prev_angle = ', prev_angle/np.pi
                            # print 'angles = ', angles/np.pi

                            # redefine previous angle as backwards of current angle -- ie angle(nxt - neighbor )
                            prev_angletmp = np.arctan2(xynxt[1] - xynb[:, 1], xynxt[0] - xynb[:, 0]).ravel()
                            prev_angle = prev_angletmp[selectIND]

                            # print 'new prev_angle = ', prev_angle/np.pi
                            # print 'NL[nxt] = ', NL[nxt]
                            # print 'bb = ', bb
                            # # CHECK
                            # ax1 = plt.gca()
                            # ax1.plot(xy[:,0],xy[:,1],'k.')
                            # for i in range(len(xy)):
                            #   ax1.text(xy[i,0]+0.2,xy[i,1],str(i))
                            # plt.arrow(xynxt[0], xynxt[1], np.cos(angles[selectIND]), np.sin(angles[selectIND]),fc='r', ec='r')
                            # plt.arrow(xynb[selectIND,0], xynb[selectIND,1], np.cos(prev_angle), np.sin(prev_angle),fc='b', ec='b')
                            # plt.show()


                        else:
                            current_angles = np.arctan2(xy[neighbors, 1] - xy[nxt, 1],
                                                        xy[neighbors, 0] - xy[nxt, 0]).ravel()
                            angles = np.mod(current_angles - prev_angle, 2 * np.pi)
                            # redefine previous angle as backwards of current angle -- ie angle(prev-current_pos)
                            xynxt = xy[nxt, :]
                            xynb = xy[neighbors, :]
                            prev_angletmp = np.arctan2(xynxt[1] - xynb[:, 1], xynxt[0] - xynb[:, 0]).ravel()
                            selectIND = np.where(angles == max(angles))[0][0]
                            # print '\n'
                            # print 'nxt = ', nxt
                            # print 'bb = ', bb
                            # print 'neighbors = ', neighbors
                            # print 'current_angles = ', current_angles/np.pi
                            # print 'prev_angle = ', prev_angle/np.pi
                            # print 'angles = ', angles/np.pi
                            # print 'selectIND = ', selectIND
                            # print('xynxt[1] - xynb[:,1], xynxt[0] - xynb[:,0] = ', xynxt[1] - xynb[:,1],
                            #       xynxt[0] - xynb[:,0])
                            # print('np.arctan2(xynxt[1] - xynb[:,1], xynxt[0] - xynb[:,0]) = ',
                            #       np.arctan2(xynxt[1] - xynb[:,1], xynxt[0] - xynb[:,0]))
                            # print 'prev_angletmp = ', prev_angletmp/np.pi

                            prev_angle = prev_angletmp[selectIND]
                            # print 'new prev_angle = ', prev_angle/np.pi

                        ###############
                        nxt = neighbors[angles == max(angles)][0]
                        bb.append(nxt)

                        ###############
                        # Check
                        if viewmethod:
                            # If checking individual bonds
                            # ax1 = plt.gca()
                            # ax1.plot(xy[:,0],xy[:,1],'k.')
                            # for i in range(len(xy)):
                            #    ax1.text(xy[i,0]+0.2,xy[i,1],str(i))

                            plt.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                         xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                         arrowprops=dict(arrowstyle="->",
                                                         color="b",
                                                         shrinkA=5, shrinkB=5,
                                                         patchA=None,
                                                         patchB=None,
                                                         connectionstyle="arc3,rad=0.6",
                                                         ), )
                            # plt.show()
                        ###############

                        # Now mark the current bond as used --> note the inversion of the bond order to match BL
                        thisbond = [bb[dmyi], bb[dmyi - 1]]
                        # Get index of used matching [bb[dmyi-1],nxt]
                        mark_used = np.where((BL == thisbond).all(axis=1))
                        if len(mark_used) > 0:
                            used[mark_used, 1] = True
                        else:
                            messg = 'Cannot mark polygon bond as used: this bond was already used in its attempted' + \
                                    ' orientation. (All bonds in first column should already be marked as used.)'
                            raise RuntimeError(messg)

                        dmyi += 1

                    polygons.append(bb)
                    # print 'added polygon = ', bb

                    # Check new polygon
                    if viewmethod:
                        if first_check:
                            ax1.plot(xy[:, 0], xy[:, 1], 'k.')
                            for i in range(len(xy)):
                                ax1.text(xy[i, 0] + 0.2, xy[i, 1], str(i))

                        for dmyi in range(len(bb)):
                            nxt = bb[np.mod(dmyi + 1, len(bb))]
                            ax1.annotate("", xy=(xy[bb[dmyi], 0], xy[bb[dmyi], 1]), xycoords='data',
                                         xytext=(xy[nxt, 0], xy[nxt, 1]), textcoords='data',
                                         arrowprops=dict(arrowstyle="->",
                                                         color="b",
                                                         shrinkA=5, shrinkB=5,
                                                         patchA=None,
                                                         patchB=None,
                                                         connectionstyle="arc3,rad=0.6", ), )
                        ax2.cla()
                        ax2.imshow(used)
                        # plt.show()
                        plt.pause(0.0001)
                        ###############

                else:
                    # All bonds have been accounted for
                    print 'all finished with finding polygons...'
                    finished = True
    # check
    if viewmethod:
        plt.show()

    # Check for duplicates (up to cyclic permutations and inversions) in polygons
    # Note that we need to ignore the last element of each polygon (which is also starting pt)
    keep = np.ones(len(polygons), dtype=bool)
    for ii in range(len(polygons)):
        polyg = polygons[ii]
        for p2 in polygons[ii + 1:]:
            if is_cyclic_permutation(polyg[:-1], p2[:-1]):
                keep[ii] = False

    polygons = [polygons[i] for i in np.where(keep)[0]]

    # Remove duplicates via inversion (maybe not necessary?)

    # Remove the polygon which is the entire lattice boundary, except dangling bonds
    if not periB.any():
        print 'Removing entire lattice boundary from list of polygons...'
        boundary = extract_boundary(xy, NL, KL, BL)
        # print 'boundary = ', boundary
        keep = np.ones(len(polygons), dtype=bool)
        for ii in range(len(polygons)):
            polyg = polygons[ii]
            if is_cyclic_permutation(polyg[:-1], boundary.tolist()):
                keep[ii] = False
            elif is_cyclic_permutation(polyg[:-1], boundary[::-1].tolist()):
                keep[ii] = False

        polygons = [polygons[i] for i in np.where(keep)[0]]

    # Check order of each polygon so that it is oriented counterclockwise
    # for polys in polygons:
    #     angle_poly = 0
    #     # Make sure that oriented counterclockwise
    #     print 'polys = ', polys
    #     for i in range(len(polys)):
    #         p0 = polys[ np.mod(i-1, len(polys)-1)]
    #         p1 = polys[i]
    #         p2 = polys[ np.mod(i+1,len(polys)-1) ]
    #         print 'p0,p1,p2 = ', p0, p1, p2
    #         angle_tmp = np.mod(np.arctan2(xy[p2,1]-xy[p1,1], xy[p2,0]-xy[p1,0]) - np.arctan2( xy[p1,1]-xy[p0,1],
    #                            xy[p1,0]-xy[p0,0] ), 2*np.pi)
    #         print 'angle_tmp = ', angle_tmp
    #         angle_poly += angle_tmp
    #
    #     print 'angle = ', angle_poly/6.

    if check:
        polygons2PPC(xy, polygons, BL=BL, PVxydict=PVxydict, check=True)

    return polygons


def pairwise(iterable):
    """Convert list into tuples with adjacent items tupled, like:
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Parameters
    ----------
    iterable : list or other iterable
        the list to tuple

    Returns
    -------
    out : list of tuples
        adjacent items tupled
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def periodic_polygon_indices2xy(poly, xy, BLdbl, PVxydict):
    """Convert a possibly-periodic polygon object that lists indices of vertices/points into an array of vertex
    coordinates. This is nontrivial only because of the existence of periodic boundary conditions. This function returns
    the polygon as it would appear to the zero-index particle (the first particle indexed in the variable 'poly'.

    Parameters
    ----------
    poly : list of ints
        The indices of xy, indexing the polygon in question. The first index may or may not be repeated as the last ---
        it seems to work either way
    xy : #pts x 2 float array
        the coordinates of the vertices of all polygons in the network
    BLdbl : #bonds x 2 signed int array
        Bond list reapeated twice, with the second time being a copy of the first, but flipped (the convention of
        doubling BL is a matter of speedup). ie BLdbl = np.vstack((BL, np.fliplr(BL)))
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i

    Returns
    -------
    xypoly : list of coordinates (each a list of 2 floats)
        The polygon as it appears to the first particle/vertex in the input list poly
    periodicpoly : bool
        Whether the polygon traverses a periodic boundary
    """
    periodicpoly = False
    tups = pairwise(poly)
    xypoly = []
    pervec = np.array([0., 0.])
    # Add first point to coordinate list
    xypoly.append((xy[tups[0][0], :] + pervec).tolist())
    for tup in tups:
        # Check if the matching row of BL is all positive --> if so, then not periodic bond
        # NOTE: If tup is positive, and bond is periodic, then will not register a match!
        match = (BLdbl[:, 0] == tup[0]) & (BLdbl[:, 1] == tup[1])
        if match.any() and (BLdbl[match, :] > -0.5).all():
            xypoly.append((xy[tup[1], :] + pervec).tolist())
        else:
            # # Check if the matching row of BL flippedlr is all positive --> if so, then not periodic bond
            # match2 = (BL[:, 0] == tup[1]) & (BL[:, 1] == tup[0])
            # if match2.any() and (BL[match2, :] > -0.5).all():
            #     xypoly.append((xy[tup[0], :] + pervec).tolist())
            #     xypoly.append((xy[tup[1], :] + pervec).tolist())
            # else:

            # Declare that this polygon exists on at least two sides
            periodicpoly = True
            # Add periodic vector (PVx, PVy) to forming polygon
            try:
                pervec += PVxydict[tup]
            except KeyError:
                pervec += -PVxydict[(tup[1], tup[0])]
            xypoly.append((xy[tup[1], :] + pervec).tolist())

    return xypoly, periodicpoly


def polygons2PPC(xy, polygons, BL=None, PVxydict=None, check=False):
    """Create list of polygon patches from polygons indexing xy. If the network is periodic, BL, PVx, PVy are required.

    Parameters
    ----------
    xy : NP x 2 float array
        coordinates of particles
    polygons: list of lists of ints
        list of polygons, each of which are a closed list of indices of xy
    check: bool
        whether to plot the polygon patch collection

    Returns
    -------
    PPC: list of patches
        list for a PatchCollection object
    """
    # Prepare a polygon patch collection plot
    if PVxydict is not None and PVxydict != {}:
        BLdbl = np.vstack((BL, np.fliplr(BL)))

    PPC = []
    for poly in polygons:
        if PVxydict is not None and PVxydict != {}:
            xypoly, periodicpoly = periodic_polygon_indices2xy(poly, xy, BLdbl, PVxydict)

            # Add to list of polygon path patches
            pp = Path(np.array(xypoly), closed=True)
            ppp = patches.PathPatch(pp, lw=2)
            PPC.append(ppp)

            # If polygon was periodic, get other permutations of the polygon
            if periodicpoly:
                # print 'Dealing with periodic polygon here...'
                # make sure that polygon doesn't have repeated index
                # print 'poly = ', poly
                if poly[-1] == poly[0]:
                    poly = poly[0:len(poly) - 1]

                oldpolys = [xypoly[0:len(xypoly) - 1]]
                for ii in range(len(poly)):
                    # permute polygon, check if it is a cyclic permutation for any previously-plotted polygons
                    poly = np.roll(poly, 1)
                    # print 'rolled poly = ', poly
                    newxyp, trash = periodic_polygon_indices2xy(poly, xy, BLdbl, PVxydict)
                    # print 'oldxyp[:, 0] = ', np.array(oldpolys[0])[:, 0]
                    # print 'newxyp[:, 0] = ', np.array(newxyp)[:, 0]
                    xcyclic = np.array([is_cyclic_permutation(np.array(oldp)[:, 0].tolist(),
                                                              np.array(newxyp)[:, 0].tolist()) for oldp in oldpolys])
                    ycyclic = np.array([is_cyclic_permutation(np.array(oldp)[:, 1].tolist(),
                                                              np.array(newxyp)[:, 1].tolist()) for oldp in oldpolys])
                    if not xcyclic.any() or not ycyclic.any():
                        # print '\n\n\n\n\n adding new periodic polygon! \n\n\n\n'
                        pp = Path(np.array(np.vstack((np.array(newxyp), np.array(newxyp)[0, :]))), closed=True)
                        ppp = patches.PathPatch(pp, lw=2)
                        PPC.append(ppp)
                        oldpolys.append(newxyp)
        else:
            pp = Path(xy[poly], closed=True)
            ppp = patches.PathPatch(pp, lw=2)
            PPC.append(ppp)

    if check:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = PatchCollection(PPC, cmap=cm.jet, alpha=0.5)
        colors = 100 * np.random.rand(len(PPC))
        p.set_array(np.array(colors))
        ax.add_collection(p)
        xlim = max(abs(xy[:, 0]))
        ylim = max(abs(xy[:, 1]))
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        plt.show()
        plt.clf()

    return PPC


def is_cyclic_permutation(A, B):
    """Check if list A is a cyclic permutation of list B.

    Parameters
    ----------
    A : list
    B : list

    Returns
    ----------
    bool
        Whether A is a cylic permutation of B
    """
    # Check if same length
    if len(A) != len(B):
        return False
    # Check that contain the same elements
    if set(A) == set(B):
        longlist = A + A
        if contains_sublist(longlist, B):
            return True
        else:
            return False
    else:
        return False


def is_cyclic_permutation_numpy(A, B):
    """Check if 1d numpy array A is a cyclic permutation of 1d numpy array B.
    HAVENT TESTED THIS BUT SHOULD WORK?

    Parameters
    ----------
    A : 1d numpy array
    B : 1d numpy array

    Returns
    ----------
    bool
        Whether A is a cylic permutation of B
    """
    # Check if same length
    if len(A) != len(B):
        return False
    # Check that contain the same elements
    if set(A) == set(B):
        longlist = np.hstack((A, A))
        n = len(A)
        if any((A == longlist[i:i + n]).all() for i in xrange(len(longlist) - n + 1)):
            return True
        else:
            return False
    else:
        return False


def contains_sublist(lst, sublst):
    """Check if a list contains a sublist (same order)

    Parameters
    ----------
    lst, sublst : lists
        The list and sublist to test

    Returns
    ----------
    bool
        Whether sublst is a sublist of lst.
    """
    n = len(sublst)
    return any((sublst == lst[i:i + n]) for i in xrange(len(lst) - n + 1))


def trim_cols_NLandKL(NL, KL):
    # Now trim the # columns down--> any column with all zeros gets cut
    done_cutting = False
    while not done_cutting:
        # print 'np.where(KL[:,-1])  =', np.where(KL[:,-1])
        if np.where(KL[:, -1])[0].size > 0:
            done_cutting = True
            # print 'Chopped last column, now shape = ', np.shape(KL)
        else:
            KLnew = copy.deepcopy(KL[:, 0:-1])
            KL = KLnew

    NL = NL[:, 0:np.shape(KL)[1]]
    return NL, KL


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile, by Jessica R. Lu.

    Parameters
    ----------
    image : N x M numpy array
        The 2D image
    center : 1 x 2 numpy float (or int) array
        The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    Returns
    -----------
    radial_prof : 1 x N numpy float array
        The values of the image, ordered by radial distance
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def remove_pts(keep, xy, BL, NN='min', check=False, PVxydict=None, PV=None):
    """
    Remove particles not indexed by keep from xy, BL, then rebuild NL, and KL.
    Zeros out inds where they appear in NL and KL and reorders rows to put zeros last.

    Parameters
    ----------
    keep : NP_0 x 1 int array or bool array
        indices (of xy, of particles) to keep
    xy : NP_0 x 2 float array
        points living on vertices of dual to triangulation
    BL : Nbonds_0 x 2 int array or empty list
        Each row is a bond and contains indices of connected points
    NN : int or string 'min'
        Number of nearest neighbors in each row of KL, NL
    check : boolean
        Whether to view
    PVxydict: dict or None
        dictionary of periodic bonds (keys) to periodic vectors (values)
    PV : 2 x 2 float array or None (required only if periodic and check==True)
        periodic lattice vectors, only used if check is True

    Returns
    ----------
    xy : NP x 2 float array
        points living on vertices of dual to triangulation
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BL : Nbonds x 2 int array
        Each row is a bond and contains indices of connected points
    """
    NP = len(xy)
    # print 'NP = ', NP
    # print 'len(keep) = ', len(keep)

    if check:
        print 'PVxydict = ', PVxydict
        display_lattice_2D(xy, BL, PVxydict=PVxydict, colorz=True, title='Input to remove points', check=True,
                           close=False)
        for ii in range(len(xy)):
            plt.text(xy[ii, 0] + 0.5 * np.random.rand(1)[0], xy[ii, 1] + 0.5 * np.random.rand(1)[0], str(ii))
        plt.show()

    # ensure that keep is int array of indices, not bool
    if keep.dtype == 'bool':
        print 'converting bool keep to int array...'
        keep = np.where(keep)[0]
    else:
        keep = np.sort(keep)

    if check:
        print 'keep = ', keep

    remove = np.setdiff1d(np.arange(NP), keep)
    # print 'keep = ', keep
    xyout = xy[keep, :]
    # print 'BL = ', BL

    if BL is not None and BL != []:
        # Make BLout
        # Find rows of BL for which both elems are in keep
        inBL0 = np.in1d(np.abs(BL[:, 0]), keep)
        inBL1 = np.in1d(np.abs(BL[:, 1]), keep)
        keepBL = np.logical_and(inBL0, inBL1)
        BLt = BL[keepBL, :]

        if check:
            print 'Removed bonds with removed particle as endpt:'
            print 'BLt = ', BLt

        # Make xyout
        # Reorder BLout to match new coords by making map from old to new
        # (Lower elements of NL by #particles removed)
        BL_r = copy.deepcopy(BLt)  # BL to reorder
        if (BL < 0).any():
            for ind in remove:
                BL_r[np.abs(BLt) > ind] = np.sign(BL_r[np.abs(BLt) > ind]) * (np.abs(BL_r[np.abs(BLt) > ind]) - 1)
        else:
            for ind in remove:
                BL_r[BLt > ind] = (BL_r[BLt > ind] - 1)
                # print 'max(BL_r) = ', max(BL_r.ravel())
                # print 'BL = ', BL_r

        print '\nRemoved ', len(remove), ' particles...'
        # BLout = np.sort(BL_r, axis=1)
        # BLtrim = dh.unique_rows(BLout)
        BLtrim = np.sort(BL_r, axis=1)

        # print 'BLtrim = ', BLtrim
        NL, KL = BL2NLandKL(BLtrim, NN=NN)
    else:
        NL = []
        KL = []
        BLtrim = []

    if check:
        if (BLtrim < -0.5).any() or PVxydict is not None:
            print 'PVxydict = ', PVxydict
            if PV is None:
                raise RuntimeError('Must supply PV when check==True and bonds are periodic')
            print 'PV = ', PV
            PVxydict = BL2PVxydict(BLtrim, xyout, PV)
            print 'le: PVxydict = ', PVxydict
        display_lattice_2D(xyout, BLtrim, PVxydict=PVxydict, colorz=True,
                           title='Network after removing points (called from remove_pts())', check=True)

    if PVxydict is not None:
        # trim the undesired particles from the periodic vector dictionary
        pvd_out = {}
        for key in PVxydict:
            if not key[0] in remove and not key[1] in remove:
                if (key[0] > remove).any() or (key[1] > remove).any():
                    # Lower the key indices
                    down0 = len(np.where(remove < key[0])[0])
                    down1 = len(np.where(remove < key[1])[0])
                    key_out = (key[0] - down0, key[1] - down1)
                else:
                    key_out = key
                pvd_out[key_out] = PVxydict[key]
    else:
        pvd_out = None

    return xyout, NL, KL, BLtrim, pvd_out


def compute_bulk_z(xy, NL, KL, BL):
    """Compute the average coordination number of the bulk particles in a lattice/network.

    Parameters
    ----------
    xy : NP x dim array
        positions of particles
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        connectivity list

    Returns
    ----------
    z : float
        average coordination number of the bulk particles in the network
    """
    NP = len(xy)
    boundary = extract_boundary(xy, NL, KL, BL)
    bulk = np.setdiff1d(np.arange(NP), boundary)
    # Compute the starting z in the bulk
    countKL = [KL[jj] for jj in bulk]
    # print 'found = ', np.count_nonzero(countKL), ' connections for ', NP_bulk, ' bulk particles...'
    z = float(np.count_nonzero(countKL)) / float(len(bulk))
    return z


def cut_bonds_z_random(xy, NL, KL, BL, target_z, min_coord=2, bulk_determination='Triangulation', check=False):
    """Cut bonds in network so that average bulk coordination is z +/- 1 bond/system size.
    Note that 'boundary' is not a unique array if there are dangling bonds.

    Parameters
    ----------
    xy : NP x dim array
        positions of particles
    NL : NP x NN int array
        neighbor list
    KL : NP x NN int array
        connectivity list
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    target_z : float
        target average bulk coordination, to be approximately enforced by cutting bonds
    bulk_determination : string ('Triangulation' 'Cutter' 'Endpts')
        How to determine which bonds are in the bulk and which are on the boundary

    Returns
    -----------
    NLout : NL x NN int array
        neighbor list
    KLout : NL x NN int array
        connectivity list
    BLout : #bonds x 2 int array
        Each row is a bond and contains indices of connected points
    """
    print ' Cutting bonds z...'
    NP = len(xy)
    NN = np.shape(NL)[1]

    # Identify boundary pts, bulk pts
    print ' cut_bonds_z : extract boundary...'
    boundary = extract_boundary(xy, NL, KL, BL)
    # print 'boundary = ', boundary
    bulk = np.setdiff1d(np.arange(NP), boundary)
    NP_bulk = len(bulk)
    NP_bound = len(np.unique(boundary))
    print 'NP_bound = ', NP_bound
    print 'NP_bulk = ', NP_bulk

    if bulk_determination == 'Triangulation':
        # Form indices of BL in bulk. Bulk bonds appear in two simplices.
        # CHANGE THIS TO TEST IF BOND TWO SIMPLICES
        TRI = BL2TRI(BL, xy)
        Binds_list = []
        for ii in range(len(BL)):
            row = BL[ii]
            # get rows of TRI where each elem of row lives
            is_a = np.where(TRI == row[0])[0]
            is_b = np.where(TRI == row[1])[0]
            # The intersection of those rows gives where both live
            simplices = np.intersect1d(is_a, is_b)
            # print 'simplices = ', simplices
            # print 'np.size(simplices) = ', np.size(simplices)
            # If more than one simplex, bulk bond
            if np.size(simplices) < 2:
                # add to boundary list
                Binds_list.append(ii)
                # print ' --> Binds = ', Binds_list

        Binds = np.array(Binds_list).ravel()
        # Get the BL indices of bulk bonds --> (binds)
        binds = np.setdiff1d(np.arange(len(BL)), Binds)

    elif bulk_determination == 'Endpts':
        # Define bulk bonds as connecting at least one bulk particle
        is_a = np.in1d(BL[:, 0], bulk)
        is_b = np.in1d(BL[:, 1], bulk)
        binds = np.where(np.logical_or(is_a, is_b))[0]
        Binds = np.setdiff1d(np.arange(len(BL)), binds)
    else:
        raise RuntimeError('ERROR: argument <bulk_determination> did not match known method!')

    # print 'binds = ', binds
    # print 'Binds = ', Binds
    print 'len(binds) = ', len(binds)
    print 'len(Binds) = ', len(Binds)

    # Check
    if check:
        # plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')
        for bii in binds:
            XX = xy[BL[bii], 0]
            YY = xy[BL[bii], 1]
            plt.plot(XX, YY, 'b-')
        for Bii in Binds:
            XX = xy[BL[Bii], 0]
            YY = xy[BL[Bii], 1]
            plt.plot(XX, YY, 'r-')
        # for i in range(len(xy)):
        #    plt.text(xy[i,0]+0.2,xy[i,1],str(i))
        plt.gca().set_aspect('equal')
        plt.show()

    # Compute the starting z in the bulk
    countKL = [KL[jj] for jj in bulk]
    # print 'found = ', np.count_nonzero(countKL), ' connections for ', NP_bulk, ' bulk particles...'
    z_start = float(np.count_nonzero(countKL)) / float(NP_bulk)
    print 'z_start = ', z_start
    print 'target_z = ', target_z

    # number of bonds to cut in the bulk
    # Be sure to divide the number of bonds by 2, since each bond double counts
    nbulk2cut = int(max([0, round((z_start - target_z) * 0.5 * float(NP_bulk))]))
    print 'nbulk2cut = ', nbulk2cut
    # number of bonds to cut in the boundary = nbulk2cut * (# boundary bonds)/(#bulk bonds)
    nB2cut = int(round(nbulk2cut * float(len(Binds)) / float(len(binds))))
    print 'nB2cut = ', nB2cut

    # CUT RANDOM BONDS

    ############################################
    ## DO BOUNDARY FIRST --> to avoid dangling particles
    # Choose nB2cut randomly from bulk
    # Shuffle bulk in-place
    np.random.shuffle(Binds)
    # Now work slowly towards selecting nbulk2cut: of the bonds,
    # but ensure that never leave a particle dangling without bonds
    done_cutting = False
    dmyi = 0
    # Set up mask for BL
    mask = np.ones(len(BL), dtype=bool)

    #################################
    # # Check :
    # plt.figure()
    # plt.gca().set_aspect('equal')
    # for ii in range(len(BL)):
    #     XX = xy[BL[ii],0]
    #     YY = xy[BL[ii],1]
    #     plt.plot(XX, YY, 'b-')
    #     plt.text(np.mean(XX), np.mean(YY), str(ii))
    # plt.show()
    #################################

    while not done_cutting:
        if len(np.where(mask == False)[0]) == nB2cut:
            done_cutting = True
        else:
            if np.mod(dmyi, 200) == 1:
                print 'cutting boundary bond: pass ', dmyi, ' (need to cut', nB2cut, ')'
            # consider adding dmyi element of bind to cut (make a test list)
            test = copy.deepcopy(mask)
            test[Binds[dmyi]] = False
            BLtmp = BL[test]
            # Check that BL leads to no dangling particles
            KLtmp = BL2KL(BLtmp, NL)
            # if all the rows in KLtmp have at least one nonzero bond, add dmyi to cut
            # print 'KLtmp.any(axis=1) = ', KLtmp.any(axis=1)
            if (np.where(~KLtmp.any(axis=1))[0]).size > 0:
                dmyi += 1
            else:
                mask[Binds[dmyi]] = False
                dmyi += 1

    ############################################
    # Choose nbulk2cut randomly from bulk
    # Shuffle bulk in-place
    np.random.shuffle(binds)
    # print 'binds = ', binds
    # Now work slowly towards selecting nbulk2cut: of the bonds,
    # but ensure that never leave a particle dangling without bonds
    done_cutting = False
    dmyi = 0
    while not done_cutting:
        if len(np.where(mask == False)[0]) == nB2cut + nbulk2cut:
            done_cutting = True
        else:
            if np.mod(dmyi, 200) == 1:
                print 'cutting bulk bond: pass ', dmyi, ' (need to cut', nbulk2cut, ')'
            # consider adding dmyi element of bind to cut (make a test list)
            test = copy.deepcopy(mask)
            test[binds[dmyi]] = False
            BLtmp = BL[test]
            # Check that BL leads to no dangling particles
            KLtmp = BL2KL(BLtmp, NL)
            # print 'KL = ', KLtmp
            # print 'np.where(~KLtmp.any(axis=1))[0] = ', np.where(~KLtmp.any(axis=1))[0]
            # if all the rows in KLtmp have at least one nonzero bond, add dmyi to cut
            if (np.where(~KLtmp.any(axis=1))[0]).size > min_coord - 1:
                dmyi += 1
            else:
                mask[binds[dmyi]] = False
                dmyi += 1

                # drop the nbulk2cut + nB2cut rows from total Bond List
    BL = BL[mask]
    # print 'BLout = ', BLout
    NL, KL = BL2NLandKL(BL, NN=NN)
    if check:
        display_lattice_2D(xy, BL)

    print '\nReturning lattice with ', len(BL), ' bonds for ', NP, ' particles...'
    print 'KL[bulk] = ', KL[bulk]

    return NL, KL, BL


def cut_bonds_z_highest(xy, NL, KL, BL, target_z, check=False):
    """Cut bonds in network so that average bulk coordination is z +/- 1 bond/system size, using iterative procedure based on connectivity.
    Note that 'boundary' is not a unique array if there are dangling bonds. The boundary is not treated at all, since it is very hard to treat it correctly.
    Therefore, one MUST crop out the desired region after lattice creation.

    Parameters
    ----------
    xy : NP x dim array
        positions of particles
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NL x NN array
        connectivity list
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    target_z : float
        target average bulk coordination, to be approximately enforced by cutting bonds

    Returns
    -----------
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NL x NN int array
        connectivity list
    BL : #bonds x 2 int array
        Each row is a bond and contains indices of connected points
    """
    print ' Cutting bonds z...'
    NP = len(xy)
    NN = np.shape(NL)[1]

    # Identify boundary pts, bulk pts
    print ' cut_bonds_z : extract boundary...'
    boundary = extract_boundary(xy, NL, KL, BL)
    # print 'boundary = ', boundary
    bulk = np.setdiff1d(np.arange(NP), boundary)
    NP_bulk = len(bulk)
    NP_bound = len(np.unique(boundary))
    print 'NP_bound = ', NP_bound
    print 'NP_bulk = ', NP_bulk

    # Define bulk bonds as connecting at least one bulk particle
    is_a = np.in1d(BL[:, 0], bulk)
    is_b = np.in1d(BL[:, 1], bulk)
    binds = np.where(np.logical_or(is_a, is_b))[0]
    Binds = np.setdiff1d(np.arange(len(BL)), binds)
    BLbulk = BL[binds]
    BLboun = BL[Binds]

    # bBinds bonds connect bulk to boundary
    # Treat these as is connecting bulk(z) to bulk(z)
    bBinds = np.where(np.logical_xor(is_a, is_b))[0]
    BLbB = BL[bBinds]

    print 'len(binds) = ', len(binds)
    print 'len(Binds) = ', len(Binds)

    # Check
    if check:
        # plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')
        for bii in binds:
            XX = xy[BL[bii], 0]
            YY = xy[BL[bii], 1]
            plt.plot(XX, YY, 'b-')

        for Bii in Binds:
            XX = xy[BL[Bii], 0]
            YY = xy[BL[Bii], 1]
            plt.plot(XX, YY, 'r-')

        # for i in range(len(xy)):
        #    plt.text(xy[i,0]+0.2,xy[i,1],str(i))
        plt.gca().set_aspect('equal')
        plt.show()

    # number of bonds to cut in the bulk
    # Be sure to divide the number of bonds by 2, since each bond double counts
    # Can write in terms of bonds? 2have = zt
    # nbulk2cut = int(max([0,round((z_start - target_z)*0.5*float(NP_bulk))]))
    # nbulk2have = len(binds) - nbulk2cut
    # print 'nboun2have = ', nboun2have
    # print 'nbulk2have = ', nbulk2have

    # CUT BONDS FROM HIGHEST Z NODES (sum of endpts)
    # Unfortunately, this has to be done iteratively.
    # Algorithm: find zvals of all bonds. For all bonds with zval = max(zval),
    # cut all the bonds that don't share endpts with any of the other bonds.
    # Find these by going through in-place-randomized B2cut and cross off if later bonds share indices.
    # Let boundary bonds be cut, or not, and pay no attention to them, since lattice will be cropped.

    # First cut most coordinated, whether on bulk or boundary, but keep track of which.
    # Get bonds with highest z pairs of nodes
    NN = np.shape(KL)[1]
    zz = np.sum(KL, axis=1)
    # print 'zz = ', zz
    zbulk = float(np.sum(zz[bulk])) / float(len(bulk))
    print 'zbulk so far = ', zbulk

    # As long as we haven't cut enough bonds, cut some more
    while zbulk > target_z:
        print 'zbulk = ', zbulk
        zb = zz[BL[:, 0]] + zz[BL[:, 1]]
        zcut = np.where(zb == max(zb))[0]
        np.random.shuffle(zcut)
        B2cut = BL[zcut]
        # print 'B2cut = ', B2cut

        # Check --> show bond numbers and bond to cut
        if check:
            display_lattice_2D(xy, BL, close=False)
            # for ii in range(len(BL)):
            # plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(ii))
            # plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(zb[ii]))
            for row in B2cut:
                plt.plot([xy[row[0], 0], xy[row[1], 0]], [xy[row[0], 1], xy[row[1], 1]], 'r-')
            plt.title('Initial counting marks these')
            plt.pause(0.01)
            plt.clf()

        # print 'B2cut = ', B2cut
        # Cross off if later bonds share indices
        keep = np.ones(len(B2cut), dtype=bool)
        for ii in range(len(B2cut)):
            row = B2cut[ii]
            if row[0] in B2cut[ii + 1:, :].ravel():
                # print 'found ', row[0], 'in rest of array '
                # print '   --> len BL[ii+1:,:] = ', len(B2cut[ii+1:,:] )
                keep[ii] = False
            elif row[1] in B2cut[ii + 1:, :].ravel():
                keep[ii] = False

        # print 'keep = ', keep
        # print 'keep.any() = ', keep.any()
        if keep.any():
            B2cut = B2cut[keep]
        else:
            print 'The highest nodes are all connected to at least one other. Killing one bond...'
            B2cut = B2cut[0:1]

        # Only interested in the bulk bonds for measurement, but cutting boundary
        # bonds will get us out of a situation where bulk is less coordinated than
        # boundary so don't do --> B2cut = intersect2d(B2cut,BLbulk)

        N2cut = len(B2cut)

        # See what would happen if we cut all of these
        BLt = dh.setdiff2d(BL, B2cut)
        NLt, KLt = BL2NLandKL(BLt, NP=NP, NN=NN)
        zzt = np.sum(KLt, axis=1)
        zbulk = np.float(np.sum(zzt[bulk])) / float(len(bulk))

        # If we can cut all of these, do that. Otherwise, cut only as many as needed after shuffling.
        if len(np.where(zzt == 0)[0]) > 0:
            print 'There are dangling points. Removing bonds2cut that would make these...'
            # There are dangling points.
            # Remove the bonds that make zzt elems zero from the bonds to cut list
            # and recalculate.
            dangle_pts = np.where(zzt == 0)[0]
            # protect dangle points --> there is only one bond to find since we have run a "keep" search on B2cut
            inb0 = np.where(np.in1d(B2cut[:, 0], dangle_pts))[0]
            inb1 = np.where(np.in1d(B2cut[:, 1], dangle_pts))[0]
            keep = np.setdiff1d(np.arange(len(B2cut)), inb0)
            keep = np.setdiff1d(keep, inb1)
            print 'Protecting dangling bond: keep for dangle =', keep

            # Check --> show bond numbers and bond to cut and protect (dangles)
            if check:
                display_lattice_2D(xy, BL, close=False)
                for ii in range(len(BL)):
                    # plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(ii))
                    plt.text((xy[BL[ii, 0], 0] + xy[BL[ii, 1], 0]) * 0.5, (xy[BL[ii, 0], 1] + xy[BL[ii, 1], 1]) * 0.5,
                             str(zb[ii]))
                for row in B2cut:
                    plt.plot([xy[row[0], 0], xy[row[1], 0]], [xy[row[0], 1], xy[row[1], 1]], 'r-')
                plt.plot([xy[B2cut[keep, 0], 0], xy[B2cut[keep, 1], 0]], [xy[B2cut[keep, 0], 1], xy[B2cut[keep, 1], 1]],
                         'b-', lw=5)
                plt.show()
                plt.clf()

            B2cut = B2cut[keep]
            N2cut = len(B2cut)

            BLt = dh.setdiff2d(BL, B2cut)
            NLt, KLt = BL2NLandKL(BLt, NP=NP, NN=NN)
            zzt = np.sum(KLt, axis=1)
            zbulk = np.float(np.sum(zzt[bulk])) / float(len(bulk))

            # If we end up in a place where these are the only bonds to cut, raise exception
            # --> means target_z is just too low for our given lattice.
            if np.size(B2cut) == 0:
                raise RuntimeError('target_z is too low for the given lattice! Cutting bonds led to dangling points.')

        if zbulk > target_z:
            print 'Still above: zbulk = ', zbulk

            # Check --> show bond numbers and bond to cut
            if check:
                display_lattice_2D(xy, BL, close=False)
                # for ii in range(len(BL)):
                # plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(ii))
                #    plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(zb[ii]))
                for row in B2cut:
                    plt.plot([xy[row[0], 0], xy[row[1], 0]], [xy[row[0], 1], xy[row[1], 1]], 'r-')

                plt.pause(0.01)
                plt.clf()

            # move pointers
            BL, BLt = BLt, BL
            NL, NLt = NLt, NL
            KL, KLt = KLt, KL
            zz, zzt = zzt, zz
        else:
            print 'Approaching z = ', target_z, ' tuning one bond at a time...'
            # Cut a bond unless there is only one to cut
            # (in which case we are within threshold)
            if N2cut == 1:
                zbulk = 0.
                # move pointers
                BL, BLt = BLt, BL
                NL, NLt = NLt, NL
                KL, KLt = KLt, KL
                zz, zzt = zzt, zz
            else:
                # Check --> show bond numbers and bond to cut
                if check:
                    display_lattice_2D(xy, BL, close=False)
                    for ii in range(len(BL)):
                        # plt.text((xy[BL[ii,0],0]+xy[BL[ii,1],0])*0.5,(xy[BL[ii,0],1]+xy[BL[ii,1],1])*0.5,str(ii))
                        plt.text((xy[BL[ii, 0], 0] + xy[BL[ii, 1], 0]) * 0.5,
                                 (xy[BL[ii, 0], 1] + xy[BL[ii, 1], 1]) * 0.5, str(zb[ii]))
                    for row in B2cut:
                        plt.plot([xy[row[0], 0], xy[row[1], 0]], [xy[row[0], 1], xy[row[1], 1]], 'r-')
                    plt.pause(0.01)
                    plt.clf()

                BL = dh.setdiff2d(BL, B2cut[0:1])
                NL, KL = BL2NLandKL(BL, NP=NP, NN=NN)
                zz = np.sum(KLt, axis=1)
                print 'zz = ', zz
                zbulk = np.float(np.sum(zz[bulk])) / float(len(bulk))

    # IGNORE BOUNDARY: MUST CUT OUT DESIRED REGION. OTHERWISE, IT'S JUST TOO HARD TO MAKE IT RIGHT.
    # Only interested in the boundary bonds now
    # number of bonds to cut in the boundary = nbulkcut * (# boundary bonds)/(#bulk bonds)
    # nB2cut = int(round(nbulk2cut * float(len(Binds))/float(len(binds))))
    # nboun2have = len(Binds) - nB2cut
    #
    # while nboun > nboun2have:
    #     zz = np.sum(KL, axis=1)
    #     zb = zz[BL[:,0]] + zz[BL[:,1]]
    #     zcut = np.where(zb== max(zb))[0]
    #     np.random.shuffle(zcut)
    #     B2cut = BL[zcut]
    #     # Only interested in the boundary bonds now
    #     B2cut = intersect2d(B2cut,BLboun)
    #     # Cross off if later bonds share indices
    #     keep = np.ones(len(B2cut),dtype = bool)
    #     for ii in range(len(B2cut)):
    #         row = B2cut[ii]
    #         if row[0] in BL[ii+1,:].ravel():
    #             keep[ii] = False
    #     B2cut = B2cut[keep]
    #     # Cut only as many as needed
    #     nboun2cut = min([nboun - nboun2have, len(B2cut)])
    #     BL = dh.setdiff2d(BL,B2cut[0:nboun2cut])
    #     nboun = len(intersect2d(BL,BLboun))
    #     print 'nbound so far =', nboun
    #     NL, KL = BL2NLandKL(BL,NP=NP,NN=NN)

    zz = np.sum(KL, axis=1)
    zbulk = np.float(np.sum(zz[bulk])) / float(len(bulk))
    print 'Tuned to zbulk = ', zbulk

    if check:
        display_lattice_2D(xy, BL, close=False)
        plt.show()

    print '\nReturning lattice with ', len(BL), ' bonds for ', NP, ' particles...'

    return NL, KL, BL


def intersect2d(A, B):
    """Return row elements in A that are in B.

    Parameters
    ----------
    A : N1 x M array
        Array to take rows of that are in B (could be BL, for ex)
    B : N2 x M
        Array whose rows to compare to those of A

    Returns
    ----------
    C : P x M array (of type int or float)
        Rows in A that are in B.
    """
    # print 'A = ', A
    # print 'B = ', B
    a1_rows = A.view([('', A.dtype)] * A.shape[1])
    a2_rows = B.view([('', B.dtype)] * B.shape[1])
    # Now trim those bonds from BL
    C = np.intersect1d(a1_rows, a2_rows).view(A.dtype).reshape(-1, A.shape[1])
    return C


def cut_bonds_strain(xy, NL, KL, BM0, bstrain):
    """Cut bonds from KL (set elems of KL to zero) based on the bond strains.
     This is not finished since for now seems ok to convert to BL --> cut --> NL,KL

    Parameters
    ----------
    xy : NP x dim array
        positions of particles
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NL x NN array
        connectivity list
    BM0 : NL x NN array
        bond rest lengths

    Return
    ----------
    KL, BLtrim, bL0trim
    """
    NP, NN = np.shape(NL)
    BL = NL2BL(NL, KL)
    bL0 = BM2bL(NL, BM0, BL)
    BLtrim, bL0trim = cut_bonds_strain_BL(BL, xy, bL0, bstrain)
    KL = BL2KL(BLtrim, NL)
    # i2cut = (np.sqrt((xy[BL[:,0],0]-xy[BL[:,1],0])**2+(xy[BL[:,0],1]-xy[BL[:,1],1])**2) - bL0) < bstrain*bL0
    return KL, BLtrim, bL0trim


def cut_bonds(BL, xy, thres):
    """Cuts bonds with LENGTHS greater than cutoff value.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    thres : float
        cutoff length between points

    Returns
    ----------
    BLtrim : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points, contains no bonds longer than thres"""
    i2cut = (xy[BL[:, 0], 0] - xy[BL[:, 1], 0]) ** 2 + (xy[BL[:, 0], 1] - xy[BL[:, 1], 1]) ** 2 < thres ** 2
    BLtrim = BL[i2cut]
    return BLtrim


def cut_bonds_strain_BL(BL, xy, bL0, bstrain):
    """Cuts bonds with STRAIN greater than cutoff value based on BL, and return trimmed BL.

    Parameters
    ----------
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    bstrain : float
        breaking strain -- bonds strained above this amount are cut

    Returns
    ----------
    BLtrim : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points, contains no bonds longer than thres
    bL0trim : array of dimension #bonds x 1
        unstrained bond length list: Each element is the length of the reference (unstrained) bond, in same order as rows of BL
        """
    i2cut = (np.sqrt(
        (xy[BL[:, 0], 0] - xy[BL[:, 1], 0]) ** 2 + (xy[BL[:, 0], 1] - xy[BL[:, 1], 1]) ** 2) - bL0) < bstrain * bL0
    bL0trim = bL0[i2cut]
    BLtrim = BL[i2cut]
    return BLtrim, bL0trim


def bonds_are_in_region(NL, KL, reg1, reg2):
    """Discern if a bond is connecting particles in the same region (reg1),
    or else: connecting reg1 to reg2 or reg2 to reg2.
    Returns KL-like and kL-like objects with 1 for reg1-reg1 connections and 0 for else.

    Parameters
    ----------
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN array
        Connectivity array. 1 for true connection, 0 for no connection, -1 for periodic connection
    reg1 : (fraction)*NP x 1 array
        Indices of the portion of the lattice in question
    reg2 : (1-fraction)*NP x 1 array
        Indices of the rest of the lattice

    Returns
    --------
    KL_reg1 : NP x NN array
        Region 1 connectivity array. Same as input KL, but with reg1-reg2 and reg2-reg2 connections zeroed out.

    """
    # Kill all KL rows of reg2 particles
    KL[reg2, :] = 0
    # Find where reg1 row of NL contains element from reg2
    # The side searched for elems of row must not contain index 0.
    for ii in reg1:
        row = NL[ii]
        # print 'row = ', row
        if np.in1d(row, reg2).any():
            tuned = np.in1d(row, reg2)
            # print 'tuned = ', tuned
            # There is at least one element in row that is on the right side of the sample
            # Change spring const of that bond.
            # Current particle is indexed as reg1[ii]
            KL[ii, tuned] = 0
            # print 'OmK[reg1[ii],tuned] = ', OmK[ii,tuned]
    KL_reg1 = KL
    return KL_reg1


##########################################
# Physical Observables
##########################################
def particle_energies_Nashgyro(xyv, NL, KL, BM_rest, OmK, Omg):
    """Compute the energy of all particles in the Nash gyro lattice.
    Assumes that Iw=l=g=1, so that OmK=k and Omg=m.

    Parameters
    ----------
    xyv : array of dimension NP x 4
        2D lattice of points (positions x,y) and velocities (in x,y)


    Returns
    ----------
    """
    # Split xyv
    xy = xyv[:, 0:2]
    v = xyv[:, 2:4]

    # Potential energy
    BL = NL2BL(NL, KL)
    bo = BM2bL(NL, BM_rest, BL)
    bL = bond_length_list(xy, BL)
    kL = KL2kL(NL, OmK, BL)
    U = 0.5 * abs(kL) * (bL - bo) ** 2
    # # Check
    # print 'KL = ', KL
    # print 'BL = ', BL
    # print 'bo = ', bo
    # print 'kL = ', kL
    # print 'U = ', U

    # Kinetic energy
    speed_squared = v[:, 0] ** 2 + v[:, 1] ** 2
    KE = 0.5 * (abs(Omg) * speed_squared)

    # Check
    if (U < 0).any() or (KE < 0).any():
        print 'KE = ', KE
        print 'U = ', U
        print 'kL*(bL-bo)**2 = ', kL * (bL - bo) ** 2
        print 'kL = ', kL
        raise RuntimeError('NEGATIVE ENERGY!')

    return U, KE


def potential_energy(xy, BL, bo=1., kL=1.):
    """Calculate the potential energy of the lattice bonds assuming linear springs.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.
    bo : array of dimension #bonds x 1 (default is 1.0)
        Rest lengths of bonds, in order of BL, lowercase to denote 1D array.
    kL : float array of length #bonds (default is 1.0) or single float
        The ith element is the spring constant of the ith bond in BL (lowercase denotes 1D array)

    Returns
    ----------
    bU : float
        Sum of potential energy in bonds
    """
    bL = bond_length_list(xy, BL)
    bU = 0.5 * sum(kL * (bL - bo) ** 2)
    return bU


def potential_energy_bins(xy, BL, bins, bo=1.0, kL=1.0):
    """Calculate the potential energy of the lattice bonds associated with particles in each supplied bin, assuming
    linear springs.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points.
    bins : list of int arrays (each any length)
        bins[kk] is an array of indices of particles in the kk'th bin
    bo : #bonds x 1 float array or single float (default is 1.0)
        Rest lengths of bonds, in order of BL, lowercase to denote 1D array.
    kL : #bonds x 1 float array or single float (default is 1.0)
        The ith element is the spring constant of the ith bond in BL (lowercase denotes 1D array)

    Returns
    ----------
    pe_bins : #bins x 1 float array
        Sum of potential energy in bonds in each bin
    """
    bL = bond_length_list(xy, BL)
    pe = 0.5 * kL * (bL - bo) ** 2
    # for each bin bins[kk], find incidices i of bL for which BL[i] is connected to at least one particle in bins[kk].
    pe_bins = np.zeros(len(bins), dtype=float)
    kk = 0
    for bin in bins:
        mask = np.logical_or(np.in1d(BL[:, 0], bin), np.in1d(BL[:, 1], bin))
        pe_bins[kk] = np.sum(pe[mask])
        kk += 1
    return pe_bins


def kinetic_energy(v, Mm=1.):
    """Calculate the kinetic energy of the lattice particles.

    Parameters
    ----------
    v : array of dimension NP x nd
        velocities of the points given by xy
    Mm : array of dimension NP x 1
        Masses of each point particle.

    Returns
    ----------
    KE : float
        Sum of kinetic energy in particles
    """
    speed_squared = v[:, 0] ** 2 + v[:, 1] ** 2
    # timeit.timeit('vt[:,0]**2+vt[:,1]**2', setup='import numpy as np; vt = np.random.rand(10000,2)', number=1000)
    KE = 0.5 * sum(Mm * speed_squared)
    return KE


def kinetic_energy_bins(v, bins, Mm=1.0):
    """Calculate the kinetic energy of the particles in each supplied bin.

    Parameters
    ----------
    v : array of dimension NP x nd
        velocities of the points given by xy
    bins : list of int lists/arrays
        indices of particles in each bin
    Mm : array of dimension NP x 1
        Masses of each point particle.

    Returns
    ----------
    ke_bin : #bins x 1 float array
        Sum of kinetic energies of particles in each bin
    """
    speed_squared = v[:, 0] ** 2 + v[:, 1] ** 2
    ke = 0.5 * Mm * speed_squared
    # for each bin bins[kk], sum kinetic energies of particles in bin
    ke_bins = np.zeros(len(bins), dtype=float)
    kk = 0
    for bin in bins:
        ke_bins[kk] = np.sum(ke[bin])
        kk += 1
    return ke_bins


def kinetic_energy_rigidbody(theta, phi, vX, vY, vtheta, vphi, vpsi, Mm, params):
    """Calculate the kinetic energy of a rigid body with euler angles theta, phi, psi.

    Parameters
    ----------
    vX : array of dimension NP x 1
        velocities of the pivot points in lab frame (x component)
    vY : array of dimension NP x 1
        velocities of the pivot points in lab frame (y component)
    theta : array of dimension NP x 1
        second Euler angle (inclination wrt z) of the rigid bodies
    phi : array of dimension NP x 1
        first Euler angle (rotation about z) of the rigid bodies
    psi : array of dimension NP x 1
        third Euler angle (rotation about 3hat) of the rigid bodies
    vtheta : array of dimension NP x 1
        rotation rate of second Euler angle (inclination wrt z) of the rigid bodies
    vphi : array of dimension NP x 1
        rotation rate of first Euler angle (rotation about z) of the rigid bodies
    vpsi : array of dimension NP x 1
        rotation rate of third Euler angle (rotation about 3hat) of the rigid bodies
    Mm : array of dimension NP x 1
        Masses of each point particle.
    params : dictionary
        must contain key-vals of I1, I3, l

    Returns
    ----------
    KE : float
        Sum of kinetic energy in particles
    """
    l = params['l']
    I3 = params['I3']
    I1star = params['I1'] + Mm * l ** 2

    # gw3 = vpsi + vphi* np.cos(theta)
    w3 = params['w3']

    v_sq = vX ** 2 + vY ** 2
    vXprod = vX * (vtheta * np.cos(theta) * np.cos(phi) - vphi * np.sin(theta) * np.sin(phi))
    vYprod = vY * (vtheta * np.cos(theta) * np.sin(phi) + vphi * np.sin(theta) * np.cos(phi))
    T1 = 0.5 * Mm * (v_sq)
    T2 = Mm * l * (vXprod + vYprod)
    T3 = 0.5 * I1star * (vphi ** 2 * np.sin(theta) ** 2 + vtheta ** 2)
    T4 = 0.5 * I3 * w3 ** 2

    KEvec = T1 + T2 + T3 + T4
    KE = sum(KEvec)
    if 'BIND' in params:
        if len(params['BIND']) > 0:
            KEnonboundary = KE - sum(KEvec[params['BIND']])
        else:
            KEnonboundary = 0 * KE
    else:
        KEnonboundary = 0 * KE

    return KE, KEvec, KEnonboundary, sum(T1), sum(T2), sum(T3), sum(T4)


def xy2v(xnew, xold, h):
    """Calculate velocity from position and positions at prev time step.

    Parameters
    ----------
    xnew : array NP x nd
        position at time t (current positions)
    xold : array NP x nd
        position at time t-h (one time step before)
    h : float
        time step

    Returns
    ----------
    v : array NP x nd
        estimated velocity
    """
    v = (xnew - xold) / h
    return v


###########################
# LATTICE CREATION
###########################
def delaunay_lattice_from_pts(xy, trimbound=True, target_z=-1, max_bond_length=-1, thres=4.0, zmethod='random',
                              minimum_bonds=-1, check=False):
    """
    Convert 2D pt set to lattice (xy, NL, KL, BL, BM) via traingulation.
    The order of operations is very important:
    A) Trim boundary triangles (needs to be a triangulation to work).
    B) Cut long bonds.
    C) Tune average coordination.
    The nontrivial part of this program kills edges which are unnatural according to the following definition:
    1. The outside edge is the hypotenuse of the triangle
    2. Call the hypotenuse the "base" and determine the "height" of the triangle.
    Then if base/height > x (where x is some criteria value, say 4), delete that
    edge/triangle (depending on data structure/object definitions).

    Parameters
    ----------
    xy : #pts to be triangulated x 2 array
        xy points from which to construct lattice
    trimbound : bool (optional)
        Whether to trim the boundary triangles based on aspect ratio
    target_z: float
        Average coordinate to which the function will tune the network, if specified to be > 0.
    max_bond_length : float
        cut bonds longer than this value, if specified to be > 0
    thres : float
        cut boundary triangles with height/base longer than this
    zmethod : string ('random' 'highest')
        Whether to cut randomly or cut bonds from nodes with highest z, if target_z > 0
    minimum_bonds: int or None (default=-1)
        If minimum_bonds>0, removes all points with fewer than minimum_bonds bonds
    check: bool
        View intermediate results

    Returns
    ----------
    xy : NP x 2 float array
        triangulated points
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BL : Nbonds x 2 int array
        Bond list
    BM : NP x NN float array
        unstretched (reference) bond length matrix

    """
    NP = len(xy)
    tri = Delaunay(xy)
    TRI = tri.vertices

    # check
    # plt.triplot(xy[:,0], xy[:,1], TRI, 'go-')
    # plt.show()

    BL = TRI2BL(TRI)
    NL, KL = BL2NLandKL(BL, NP=NP, NN='min')

    if trimbound:
        # Cut unnatural edge bonds (ones that are long and skinny)
        NL, KL, BL, TRI = delaunay_cut_unnatural_boundary(xy, NL, KL, BL, TRI, thres)

        # check
        if check:
            plt.clf()
            plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
            plt.show()

    # Cut bonds longer than max allowed length
    if max_bond_length > 0:
        print 'Cutting bonds longer than max_bond_length...'
        BL = cut_bonds(BL, xy, max_bond_length)
        if check:
            display_lattice_2D(xy, BL, title='In delaunay_lattice_from_pts(), removed long bonds.')
        NL, KL = BL2NLandKL(BL, NN='min')

    if minimum_bonds > 0:
        # Remove any points with no bonds
        print 'Removing points without any bonds...'
        if minimum_bonds == 1:
            keep = KL.any(axis=1)
        else:
            keep = np.sum(KL, axis=1) > minimum_bonds
            # keep = np.array([np.count_nonzero(KL[i]) > minimum_bonds for i in range(len(KL))])
        xy, NL, KL, BL, PVxydict = remove_pts(keep, xy, BL, NN='min')
        if check:
            display_lattice_2D(xy, BL, NL=NL, KL=KL, title='In delaunay_lattice_from_pts(), removed pts without bonds.')

    # Cut bonds to tune average coordination
    if target_z > 0:
        print 'Cutting bonds to tune average coordination...'
        if zmethod == 'random':
            NL, KL, BL = cut_bonds_z_random(xy, NL, KL, BL, target_z)
        elif zmethod == 'highest':
            NL, KL, BL = cut_bonds_z_highest(xy, NL, KL, BL, target_z)

    print 'Constructing BM...'
    BM = NL2BM(xy, NL, KL)

    if check:
        display_lattice_2D(xy, BL, NL=NL, KL=KL, title='Checking output lattice in delaunay_lattice_from_pts()')
    # vc = cc[:,tri.neighbors]
    # # kill edges at infinity, plotting those would need more work...
    # vc[:,tri.neighbors == -1] = np.nan
    #
    # lines = []
    # lines.extend(zip(cc.T, vc[:,:,0].T))
    # lines.extend(zip(cc.T, vc[:,:,1].T))
    # lines.extend(zip(cc.T, vc[:,:,2].T))
    return xy, NL, KL, BL, BM


def voronoi_lattice_from_pts(points, polygon=None, NN=3, kill_outliers=True, check=False):
    """Convert 2D pt set to dual lattice (xy, NL, KL, BL) via Voronoi construction: a Wigner-Seitz construction)

    Parameters
    ----------
    points : #triangulated pts x 2 array
        xy points from a Delaunay triangulation
    polygon: n x 2 float array
        a bounding polygon to kill outlying points
    NN : int (default=3)
        the number of rows of KL, NL to make
    kill_outliers : bool

    check : bool

    Returns
    ----------
    xy : NP x 2 float array
        points living on vertices of dual to triangulation
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BL : Nbonds x 2 int array
        Bond list
    """
    tri = Delaunay(points)
    p = tri.points[tri.vertices]

    # Triangle vertices
    A = p[:, 0, :].T
    B = p[:, 1, :].T
    C = p[:, 2, :].T

    # See http://en.wikipedia.org/wiki/Circumscribed_circle#Circumscribed_circles_of_triangles
    # The following is just a direct transcription of the formula there
    a = A - C
    b = B - C

    def dot2(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def cross2(u, v, w):
        """u x (v x w)"""
        return dot2(u, w) * v - dot2(u, v) * w

    def ncross2(u, v):
        """|| u x v ||^2"""
        return sq2(u) * sq2(v) - dot2(u, v) ** 2

    def sq2(u):
        return dot2(u, u)

    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2 * ncross2(a, b)) + C

    # Construct NL from the triangulation object
    print 'Constructing NL, KL...'
    if NN > 3:
        NP = len(cc[0])
        NL = np.zeros(NP, NN).astype(int)
        NL[:, 0:3] = tri.neighbors
    else:
        NL = tri.neighbors

    # Here we have used 1-indexing to make the array, now we move to zero-indexing
    # (so that the first point is zero instead of 1)
    KL = (NL > -0.5).astype(int)
    NL[NL == -1] = 0

    print 'Constructing xy, BL...'
    xy = np.array([[cc[0][ii], cc[1][ii]] for ii in range(len(cc[0]))])
    BL = NL2BL(NL, KL)

    if kill_outliers:
        # Only keep xy inside the bounds of the supplied points
        # points -= np.mean(points, axis=0)
        keep = np.where(np.logical_and(abs(xy[:, 0]) < max(points[:, 0]), abs(xy[:, 1]) < max(points[:, 1])))[0]
        xy, NL, KL, BL, PVxydict = remove_pts(keep, xy, BL)

    if polygon == 'auto' or polygon is None:
        xycent = xy
    else:
        print 'cropping polygon = ', polygon
        pth = Path(polygon, closed=False)
        keep = pth.contains_points(xy)
        if check:
            print 'xy = ', xy
            print 'keep = ', keep
            plt.plot(xy[:, 0], xy[:, 1], 'b.')
            plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
            plt.show()

        xy, NL, KL, BL, PVxydict = remove_pts(keep, xy, BL, NN=3)

    # vc = cc[:,tri.neighbors]
    # # kill edges at infinity, plotting those would need more work...
    # vc[:,tri.neighbors == -1] = np.nan
    #
    # lines = []
    # lines.extend(zip(cc.T, vc[:,:,0].T))
    # lines.extend(zip(cc.T, vc[:,:,1].T))
    # lines.extend(zip(cc.T, vc[:,:,2].T))
    return xy, NL, KL, BL


def voronoi_rect_periodic_from_pts(xy, LL, BBox='auto', dist=7., check=False):
    """Convert 2D pt set to dual lattice (xy, NL, KL, BL) via Voronoi construction

    Parameters
    ----------
    points : #triangulated pts x 2 array
        xy points from a Delaunay triangulation
    polygon

    Returns
    ----------
    xy : NP x 2 float array
        points living on vertices of dual to triangulation
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BL : Nbonds x 2 int array
        Bond list
    """
    xytmp = buffer_points_for_rectangular_periodicBC(xy, LL, dist=dist)
    xy, NL, KL, BL = voronoi_lattice_from_pts(xytmp, polygon=None, kill_outliers=True)
    xytrim, NL, KL, BLtrim, PVxydict = buffered_pts_to_periodic_network(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def delaunay_centroid_lattice_from_pts(xy, polygon=None, trimbound=True, thres=2.0, shear=-1, check=False):
    """
    Convert 2D pt set to lattice (xy, NL, KL, BL, BM) via traingulation.
    Performs:
    A) Triangulate the point set.
    B) If trimbound==True, trim boundary triangles (needs to be a triangulation to work)
    C) Find centroids
    D) Connect centroids in z=3 lattice
    Part of this program kills edges which are unnatural according to the following definition:
    1. The outside edge is the hypotenuse of the triangle
    2. Call the hypotenuse the "base" and determine the "height" of the triangle.
    Then if base/height > x (where x is some criteria value, say 4), delete that
    edge/triangle (depending on data structure/object definitions).

    Parameters
    ----------
    xy : #pts to be triangulated x 2 array
        xy points from which to construct lattice
    polygon : numpy float array [[x0,y0],...,[xN,yN]], or string 'auto'
        polygon to cut out from centroid lattice. If 'auto', cuts out a box with a buffer distance of 5%
    trimbound : bool
        Whether to trim off high-aspect-ratio triangles from edges of the triangulation before performing centroid
        operation
    thres : float (default = 2.0)
        cut boundary triangles with height/base longer than this value
    shear : float (ignored if negative)
        shear inverse slope (run/rise) to apply to xy in order to prevent degeneracy in triangulation (breaks symmetry)
    check : bool
        Whether to plot results as they are computed

    Returns
    ----------
    xycent : NP x 2 float array
        Centroids of triangulation of xy
    NL : NP x NN int array
        Neighbor list (of centroids of triangulation of xy).
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list (of centroids of triangulation of xy)
    BL : Nbonds x 2 int array
        Bond list (of centroids of triangulation of xy)
    BM : NP x NN float array
        unstretched (reference) bond length matrix (of centroids of triangulation of xy)

    """
    ###############################
    # A) Triangulate the point set.
    ###############################
    NP = len(xy)
    if shear > 0:
        xytmp = np.dstack((xy[:, 0] + shear * xy[:, 1], xy[:, 1]))[0]
        tri = Delaunay(xytmp)
    else:
        tri = Delaunay(xy)
    TRI = tri.vertices
    print 'max(TRI) = ', max(TRI.ravel())

    # check
    if check:
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
        plt.show()

    # centxy = xyandTRI2centroid(xy,TRI)
    # BL = TRI2BL(TRI)

    if trimbound:
        ###############################
        # B) Trim boundary triangles
        ###############################
        print 'converting TRI to NL, KL, BL...'
        BL = TRI2BL(TRI)
        NL, KL = BL2NLandKL(BL, NP='auto', NN='min')
        if check:
            display_lattice_2D(xy, BL, title='TRI lattice before cutting edge TRIs, before centroid')
        # Cut unnatural edge bonds (ones that are long and skinny)
        NL, KL, BL, TRI = delaunay_cut_unnatural_boundary(xy, NL, KL, BL, TRI, thres)
        if check:
            display_lattice_2D(xy, BL, title='TRI lattice after cutting edge TRIs, before centroid')
    else:
        BL = TRI2BL(TRI)

    # check
    # plt.clf()
    # plt.triplot(xy[:,0], xy[:,1], TRI, 'go-')
    # plt.show()

    ######################
    # C) Find centroids
    ######################
    centxy = xyandTRI2centroid(xy, TRI)
    if check:
        display_lattice_2D(xy, BL, title='TRI lattice after cutting edge TRIs, before centroid', colorz=False,
                           close=False)
        plt.plot(centxy[:, 0], centxy[:, 1], 'b.')
        plt.show()

    # Now make TRIout
    # NL = tri.neighbors
    # KL = np.zeros_like(NL, dtype=int)
    # KL[NL >0 ] = 1
    # BL = NL2BL(NL,KL)

    # Now make TRIout
    NL, KL, BL = TRI2centroidNLandKLandBL(TRI)

    if check:
        display_lattice_2D(centxy, BL, title='centroid lattice before cropping', close=False)
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
        plt.show()
        print 'centroid BL = ', BL
        print 'centroid max(BL) =', np.max(BL.ravel())
        print 'centroid len(centxy) = ', len(centxy)

    # The chopping below really isn't necessary for the centroid method
    if polygon == 'auto' or polygon is None:
        xycent = centxy
    else:
        print 'cropping polygon = ', polygon
        pth = Path(polygon, closed=False)
        keep = pth.contains_points(centxy)
        if check:
            print 'centxy = ', centxy
            print 'keep = ', keep
            plt.plot(centxy[:, 0], centxy[:, 1], 'b.')
            plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
            plt.show()

        xycent, NL, KL, BL, PVxydict = remove_pts(keep, centxy, BL, NN=3)

    if check:
        display_lattice_2D(xycent, BL, title='centroid lattice after cropping', close=False)
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
        plt.show()

    # print 'Constructing BM...'
    # BM = NL2BM(xycent, NL,KL)

    # vc = cc[:,tri.neighbors]
    # # kill edges at infinity, plotting those would need more work...
    # vc[:,tri.neighbors == -1] = np.nan
    #
    # lines = []
    # lines.extend(zip(cc.T, vc[:,:,0].T))
    # lines.extend(zip(cc.T, vc[:,:,1].T))
    # lines.extend(zip(cc.T, vc[:,:,2].T))

    return xycent, NL, KL, BL


def delaunay_centroid_rect_periodic_network_from_pts(xy, LL, BBox='auto', check=False):
    """Convert 2D pt set to lattice (xy, NL, KL, BL, BM, PVxydict) via traingulation, handling periodic BCs.
    Performs:
    A) Triangulate an enlarged version of the point set.
    B) Find centroids
    C) Connect centroids in z=3 lattice
    D) Crops to original bounding box and connects periodic BCs
    Note: Normally BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = buffer_points_for_rectangular_periodicBC(xy, LL)
    xy, NL, KL, BL = delaunay_centroid_lattice_from_pts(xytmp, polygon=None, trimbound=False, check=check)
    xytrim, NL, KL, BLtrim, PVxydict = buffered_pts_to_periodic_network(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def delaunay_centroid_periodicstrip_from_pts(xy, LL, BBox='auto', check=False):
    """Convert 2D pt set to lattice (xy, NL, KL, BL, BM, PVxydict) via traingulation, handling periodic BCs.
    Performs:
    A) Triangulate an enlarged version of the point set.
    B) Find centroids
    C) Connect centroids in z=3 lattice
    D) Crops to original bounding box and connects periodic BCs
    Note: Normally BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------
    xytrim : NP x 2 float array
        xy points which are centroids of the triangulation of the input xy
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BLtrim : Nbonds x 2 int array
        Bond list
    PVxydict : dict
        dictionary of periodic bonds (keys) to periodic vectors (values)
        If key = (i,j) and val = np.array([ 5.0,2.0]), then particle i sees particle j at xy[j]+val
        --> transforms into:  ijth element of PVx is the x-component of the vector taking NL[i,j] to its image as seen
        by particle i
    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = buffer_points_for_rectangular_periodicBC(xy, LL)
    xy, NL, KL, BL = delaunay_centroid_lattice_from_pts(xytmp, polygon=None, trimbound=False, check=check)
    xytrim, NL, KL, BLtrim, PVxydict = buffered_pts_to_periodicstrip(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def delaunay_rect_periodic_network_from_pts(xy, LL, BBox='auto', check=False, target_z=-1, max_bond_length=-1,
                                            zmethod='random', minimum_bonds=-1, dist=7.0):
    """Buffers the true point set with a mirrored point set across each boundary, for a rectangular boundary.

    Parameters
    ----------
    xy : NP x 2 float array
        Particle positions
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : 4 x 2 float array or 'auto'
        The bounding box to use as the periodic boundary box
    check : bool
        Whether to display intermediate results
    target_z: float
        Average coordinate to which the function will tune the network, if specified to be > 0.
    max_bond_length : float
        cut bonds longer than this value
    zmethod : string ('random' 'highest')
        Whether to cut randomly or cut bonds from nodes with highest z
    minimum_bonds: int
        If >0, remove pts with fewer bonds
    dist : float
        minimum depth of the buffer on each side

    Returns
    -------
    xytrim : (~NP) x 2 float array
        triangulated point set with periodic BCs on right, left, above, and below
    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = buffer_points_for_rectangular_periodicBC(xy, LL, dist=dist)
    xy, NL, KL, BL, BM = delaunay_lattice_from_pts(xytmp, trimbound=False, target_z=target_z,
                                                   max_bond_length=max_bond_length,
                                                   zmethod=zmethod, minimum_bonds=minimum_bonds,
                                                   check=check)
    xytrim, NL, KL, BLtrim, PVxydict = buffered_pts_to_periodic_network(xy, BL, LL, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def buffer_points_for_rectangular_periodicBC(xy, LL, dist=7.0):
    """Buffers the true point set with a mirrored point set across each boundary, for a rectangular boundary.

    Parameters
    ----------
    xy : NP x 2 float array
        Particle positions
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    dist : float
        minimum depth of the buffer on each side

    Returns
    -------
    xyout : (>= NP) x 2 float array
        Buffered point set with edges of sample tiled on the right, left, above and below
    """
    # Copy some of lattice to north, south, east, west and corners
    print 'le: xy = ', xy
    print 'le: np.min(xy[:, 0]) = ', np.min(xy[:, 0])
    print 'np.sort(xy[:, 0]) = ', np.sort(xy[:, 0])
    west = np.where(xy[:, 0] < (np.nanmin(xy[:, 0]) + dist))[0]
    sout = np.where(xy[:, 1] < (np.nanmin(xy[:, 1]) + dist))[0]
    east = np.where(xy[:, 0] > (np.nanmax(xy[:, 0]) - dist))[0]
    nort = np.where(xy[:, 1] > (np.nanmax(xy[:, 1]) - dist))[0]
    swest = np.intersect1d(sout, west)
    seast = np.intersect1d(sout, east)
    neast = np.intersect1d(nort, east)
    nwest = np.intersect1d(nort, west)
    Epts = xy[west] + np.array([LL[0], 0.0])
    Npts = xy[sout] + np.array([0.0, LL[1]])
    Wpts = xy[east] + np.array([-LL[0], 0.0])
    Spts = xy[nort] + np.array([0.0, -LL[1]])
    NEpts = xy[swest] + np.array([LL[0], LL[1]])
    NWpts = xy[seast] + np.array([-LL[0], LL[1]])
    SWpts = xy[neast] + np.array([-LL[0], -LL[1]])
    SEpts = xy[nwest] + np.array([LL[0], -LL[1]])
    # print 'extrapts = ', Epts, NEpts, Npts, NWpts
    # print '...and more'
    xyout = np.vstack((xy, Epts, NEpts, Npts, NWpts, Wpts, SWpts, Spts, SEpts))

    return xyout


def buffered_pts_to_periodic_network(xy, BL, LL, BBox=None, check=False):
    """Crops to original bounding box and connects periodic BCs of rectangular sample
    Note: Default BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points with buffer points, so xy are in the triangular representation
    LL : tuple of 2 floatsf
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    if BBox is None or isinstance(BBox, str):
        # Assuming that BBox is centered and has width, height of LL[0], LL[1]
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        keep = np.where(np.logical_and(abs(xy[:, 0]) < LL[0] * 0.5, abs(xy[:, 1]) < LL[1] * 0.5))[0]
    else:
        bpath = mplpath.Path(BBox)
        keep = np.where(bpath.contains_points(xy))[0]
        if check:
            print 'checking that keep is not a logical ==> '
            print ' this would be bool keep = ', bpath.contains_points(xy)
            print ' and this is keep = ', keep

    minX = np.min(BBox[:, 0])
    maxX = np.max(BBox[:, 0])
    minY = np.min(BBox[:, 1])
    maxY = np.max(BBox[:, 1])
    PVdict = {'e': np.array([LL[0], 0.0]),
              'n': np.array([0.0, LL[1]]),
              'w': np.array([-LL[0], 0.0]),
              's': np.array([0.0, -LL[1]]),
              'ne': np.array([LL[0], LL[1]]),
              'nw': np.array([-LL[0], LL[1]]),
              'sw': np.array([-LL[0], -LL[1]]),
              'se': np.array([LL[0], -LL[1]])}

    # Create a kd tree of the points
    tree = scipy.spatial.KDTree(xy)

    # Find bonds that will be cut. For each bond, match to other particle and add pair to BL and PVxydict
    BLcut, cutIND = find_cut_bonds(BL, keep)

    if check:
        plt.scatter(xy[:, 0], xy[:, 1], c='g', marker='x')
        plt.scatter(xy[keep, 0], xy[keep, 1], c='b', marker='o')
        highlight_bonds(xy, BL, ax=plt.gca(), color='b', show=False)
        highlight_bonds(xy, BLcut, ax=plt.gca(), color='r', lw=1, show=False)
        xxtmp = np.hstack((BBox[:, 0], np.array(BBox[:, 0])))
        print 'xxtmp = ', xxtmp
        yytmp = np.hstack((BBox[:, 1], np.array(BBox[:, 1])))
        print 'yytmp = ', yytmp
        plt.plot(xxtmp, yytmp, 'k-', lw=2)
        plt.title('Showing bonds that are cut, btwn original and mirrored network')
        plt.show()

    # preallocate BL2add and PVs
    BL2add = np.zeros((len(BLcut), 2), dtype=int)
    PVd = {}  # = np.zeros((len(BLcut),2), dtype=float)
    kk = 0
    for bond in BLcut:
        # which endpt is outside?
        ptA = bond[0]
        ptB = bond[1]
        # mpt is short for 'mirror point', the point outside the bounding box
        if ptA not in keep:
            mpt, kpt = ptA, ptB
        else:
            mpt, kpt = ptB, ptA
        if xy[mpt, 0] < minX:
            if xy[mpt, 1] < minY:
                # Mirror particle is SW
                PV = PVdict['sw']
            elif xy[mpt, 1] > maxY:
                # Mirror particle is NW
                PV = PVdict['nw']
            else:
                # Mirror particle is West
                PV = PVdict['w']
        elif xy[mpt, 0] > maxX:
            if xy[mpt, 1] < minY:
                # Mirror particle is SE
                PV = PVdict['se']
            elif xy[mpt, 1] > maxY:
                # Mirror particle is NE
                PV = PVdict['ne']
            else:
                # Mirror particle is East
                PV = PVdict['e']
        elif xy[mpt, 1] < minY:
            # Mirror particle is South
            PV = PVdict['s']
        else:
            # Mirror particle is North
            PV = PVdict['n']

        # Get index of the particle that resides a vector -PV away from mirror particle
        dist, ind = tree.query(xy[mpt] - PV)
        BL2add[kk] = np.array([-kpt, -ind])
        PVd[(kpt, ind)] = PV
        kk += 1

    if check:
        print 'PVd = ', PVd
        display_lattice_2D(xy, np.abs(BL), title="showing extended lattice (w/o PBCs)")

    # Crop network, and add back cut bonds as periodic ones
    BL = np.vstack((BL, BL2add))
    xytrim, NL, KL, BLtrim, PVxydict = remove_pts(keep, xy, BL)
    # Adjusting BL2add to account for smaller #npts (post-cropping) is already done in remove_pts
    # Adjust PVs to account for smaller #npts (post-cropping)
    remove = np.setdiff1d(np.arange(len(xy)), keep)

    # PVxydict should be correct as is, from output of remove_pts...
    PVxydict_check = {}
    for key in PVd:
        # adjust key to lower indices
        # count how many pts in remove are lower than key[0] and key[1], respectively
        lower0 = np.sum(remove < key[0])
        lower1 = np.sum(remove < key[1])
        newkey = (key[0] - lower0, key[1] - lower1)
        PVxydict_check[newkey] = PVd[key]
    print 'PVxydict = ', PVxydict
    print 'PVxydict_check = ', PVxydict_check
    raise RuntimeError('Are these PVxydicts the same?')

    if check:
        # Plot lattice without PBCs
        display_lattice_2D(xytrim, np.abs(BLtrim), title="showing lattice connectivity w/o PBCs")
        display_lattice_2D(xytrim, BLtrim, PVxydict=PVxydict, title="showing lattice connectivity with PBCs")

    return xytrim, NL, KL, BLtrim, PVxydict


def buffered_pts_to_periodicstrip(xy, BL, LL, BBox='auto', check=False):
    """Crops to original bounding box and connects periodic BCs of rectangular sample
    Note: Default BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points with buffer points, so xy are in the triangular representation
    LL : tuple of 2 floatsf
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    if BBox == 'auto':
        # Assuming that BBox is centered and has width, height of LL[0], LL[1]
        BBox = 0.5 * np.array([[-LL[0], -LL[1]], [LL[0], -LL[1]], [LL[0], LL[1]], [-LL[0], LL[1]]])
        keep = np.where(np.logical_and(abs(xy[:, 0]) < LL[0] * 0.5, abs(xy[:, 1]) < LL[1] * 0.5))[0]
    else:
        bpath = mplpath.Path(BBox)
        keep = np.where(bpath.contains_points(xy))[0]
        if check:
            print 'checking that keep is not a logical ==> '
            print ' this would be bool keep = ', bpath.contains_points(xy)
            print ' and this is keep = ', keep

    minX = np.min(BBox[:, 0])
    maxX = np.max(BBox[:, 0])
    minY = np.min(BBox[:, 1])
    maxY = np.max(BBox[:, 1])
    PVdict = {'e': np.array([LL[0], 0.0]),
              'n': np.array([0.0, LL[1]]),
              'w': np.array([-LL[0], 0.0]),
              's': np.array([0.0, -LL[1]]),
              'ne': np.array([LL[0], LL[1]]),
              'nw': np.array([-LL[0], LL[1]]),
              'sw': np.array([-LL[0], -LL[1]]),
              'se': np.array([LL[0], -LL[1]])}

    # Create a kd tree of the points
    tree = scipy.spatial.KDTree(xy)

    # Find bonds that will be cut. For each bond, match to other particle and add pair to BL and PVxydict
    BLcut, cutIND = find_cut_bonds(BL, keep)

    if check:
        plt.scatter(xy[:, 0], xy[:, 1], c='g', marker='x')
        plt.scatter(xy[keep, 0], xy[keep, 1], c='b', marker='o')
        highlight_bonds(xy, BL, ax=plt.gca(), color='b', show=False)
        highlight_bonds(xy, BLcut, ax=plt.gca(), color='r', lw=1, show=False)
        xxtmp = np.hstack((BBox[:, 0], np.array(BBox[:, 0])))
        print 'xxtmp = ', xxtmp
        yytmp = np.hstack((BBox[:, 1], np.array(BBox[:, 1])))
        print 'yytmp = ', yytmp
        plt.plot(xxtmp, yytmp, 'k-', lw=2)
        plt.title('Showing bonds that are cut, btwn original and mirrored network')
        plt.show()

    # preallocate BL2add and PVs
    BL2add = np.zeros((len(BLcut), 2), dtype=int)
    PVd = {}  # = np.zeros((len(BLcut),2), dtype=float)
    kk = 0
    for bond in BLcut:
        # which endpt is outside?
        ptA = bond[0]
        ptB = bond[1]
        # mpt is short for 'mirror point', the point outside the bounding box
        if ptA not in keep:
            mpt, kpt = ptA, ptB
        else:
            mpt, kpt = ptB, ptA

        # Assume that the bond should remain broken unless the PV is 'e' or 'w' (east or west)
        ok_stripbc = False
        if xy[mpt, 0] < minX:
            if xy[mpt, 1] < minY:
                # Mirror particle is SW
                PV = PVdict['sw']
            elif xy[mpt, 1] > maxY:
                # Mirror particle is NW
                PV = PVdict['nw']
            else:
                # Mirror particle is West
                PV = PVdict['w']
                ok_stripbc = True
        elif xy[mpt, 0] > maxX:
            if xy[mpt, 1] < minY:
                # Mirror particle is SE
                PV = PVdict['se']
            elif xy[mpt, 1] > maxY:
                # Mirror particle is NE
                PV = PVdict['ne']
            else:
                # Mirror particle is East
                PV = PVdict['e']
                ok_stripbc = True
        elif xy[mpt, 1] < minY:
            # Mirror particle is South
            PV = PVdict['s']
        else:
            # Mirror particle is North
            PV = PVdict['n']

        if ok_stripbc:
            # Get index of the particle that resides a vector -PV away from mirror particle
            dist, ind = tree.query(xy[mpt] - PV)
            BL2add[kk] = np.array([-kpt, -ind])
            PVd[(kpt, ind)] = PV
            kk += 1

    BL2add = BL2add[0:kk]

    if check:
        print 'PVd = ', PVd
        display_lattice_2D(xy, np.abs(BL), title="showing extended lattice (w/o strip PBCs)")

    # Crop network, and add back cut bonds as periodic ones
    BL = np.vstack((BL, BL2add))
    xytrim, NL, KL, BLtrim, PVxydict = remove_pts(keep, xy, BL)
    # Adjusting BL2add to account for smaller #npts (post-cropping) is already done in remove_pts
    # Adjust PVs to account for smaller #npts (post-cropping)
    remove = np.setdiff1d(np.arange(len(xy)), keep)
    PVxydict = {}
    for key in PVd:
        # adjust key to lower indices
        # count how many pts in remove are lower than key[0] and key[1], respectively
        lower0 = np.sum(remove < key[0])
        lower1 = np.sum(remove < key[1])
        newkey = (key[0] - lower0, key[1] - lower1)
        PVxydict[newkey] = PVd[key]

    if check:
        # Plot lattice without PBCs
        display_lattice_2D(xytrim, np.abs(BLtrim), title="showing lattice connectivity w/o strip PBCs")
        display_lattice_2D(xytrim, BLtrim, PVxydict=PVxydict, title="showing lattice connectivity with strip PBCs")

    return xytrim, NL, KL, BLtrim, PVxydict


def delaunay_periodic_network_from_pts(xy, PV, BBox='auto', check=False, target_z=-1, max_bond_length=-1,
                                       zmethod='random', minimum_bonds=-1, ensure_periodic=False):
    """Buffers the true point set with a mirrored point set across each boundary, for a rectangular boundary.

    Parameters
    ----------
    xy : NP x 2 float array
        Particle positions
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : 4 x 2 float array or 'auto'
        The bounding box to use as the periodic boundary box
    check : bool
        Whether to display intermediate results
    target_z: float
        Average coordinate to which the function will tune the network, if specified to be > 0.
    max_bond_length : float
        cut bonds longer than this value
    zmethod : string ('random' 'highest')
        Whether to cut randomly or cut bonds from nodes with highest z
    minimum_bonds: int
        If >0, remove pts with fewer bonds
    dist : float
        minimum depth of the buffer on each side

    Returns
    -------
    xytrim : (~NP) x 2 float array
        triangulated point set with periodic BCs on right, left, above, and below
    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = buffer_points_for_periodicBC(xy, PV)
    if check:
        plt.show()
        plt.plot(xytmp[:, 0], xytmp[:, 1], 'b.')
        plt.title('Buffered points')
        plt.show()
    xy, NL, KL, BL, BM = delaunay_lattice_from_pts(xytmp, trimbound=False, target_z=target_z,
                                                   max_bond_length=max_bond_length,
                                                   zmethod=zmethod, minimum_bonds=minimum_bonds,
                                                   check=check)
    if ensure_periodic:
        BL = ensure_periodic_connectivity(xy, NL, KL, BL)
        NL, KL = BL2NLandKL(BL)

    # todo: allow for other shapes of periodic boundaries other than parallelogram
    xytrim, NL, KL, BLtrim, PVxydict = \
        buffered_pts_to_periodic_network_parallelogram(xy, BL, PV, BBox=BBox, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def ensure_periodic_connectivity(xy, NL, KL, BL):
    """For each paticle, look at bonds, and look at the bonds of the same particles elsewhere in a lattice
    according to the periodic vectors PV. If there are bonds that have been triangulated differently than they
    in different parts of a periodic network, which can happen due to roundoff error, fix these bonds"""
    raise RuntimeError('This function is not built yet')
    return BL


def delaunay_centroid_periodic_network_from_pts(xy, PV, BBox='auto', flex_pvxy=False, shear=-1., check=False):
    """Convert 2D pt set to lattice (xy, NL, KL, BL, BM, PVxydict) via traingulation, handling periodic BCs.
    Performs:
    A) Triangulate an enlarged version of the point set.
    B) Find centroids
    C) Connect centroids in z=3 lattice
    D) Crops to original bounding box and connects periodic BCs
    Note: Normally BBox is centered such that original BBox is [-LL[0]*0.5, -LL[1]*0.5], [LL[0]*0.5, -LL[1]*0.5], etc.

    Parameters
    ----------
    xy : NP x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    BBox : #vertices x 2 numpy float array
        bounding box for the network. Here, this MUST be rectangular, and the side lengths should be taken to be
        LL[0], LL[1] for it to be sensible.
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    # Algorithm for handling boundaries:
    #  - Copy parts of lattice to buffer up the edges
    #  - Cut the bonds with the bounding box of the loaded configuration
    #  - For each cut bond, match the outside endpt with its corresponding mirror particle
    xytmp = buffer_points_for_periodicBC(xy, PV, check=check)
    xy, NL, KL, BL = delaunay_centroid_lattice_from_pts(xytmp, polygon=None, trimbound=False, shear=shear, check=check)
    xytrim, NL, KL, BLtrim, PVxydict = \
        buffered_pts_to_periodic_network_parallelogram(xy, BL, PV, BBox=BBox, flex_pvxy=flex_pvxy, check=check)
    return xytrim, NL, KL, BLtrim, PVxydict


def buffer_points_for_periodicBC(xy, PV, check=False):
    """Buffers the true point set with a mirrored point set across each boundary, for a parallelogram (or evenutally a
    more argitrary boundary).

    Parameters
    ----------
    xy : NP x 2 float array
        Particle positions
    PV : 2 x 2 float array
        Periodic vectors: the first has x and y components, the second has only positive y component.
    dist : float
        minimum depth of the buffer on each side

    Returns
    -------
    xyout : (>= NP) x 2 float array
        Buffered point set with edges of sample tiled on the right, left, above and below
    """
    Epts = xy + PV[0]
    Npts = xy + PV[1]
    Wpts = xy - PV[0]
    Spts = xy - PV[1]
    NEpts = xy + PV[0] + PV[1]
    NWpts = xy - PV[0] + PV[1]
    SWpts = xy - PV[0] - PV[1]
    SEpts = xy + PV[0] - PV[1]
    xyout = np.vstack((xy, Epts, NEpts, Npts, NWpts, Wpts, SWpts, Spts, SEpts))
    if check:
        eps = 0.1
        plt.scatter(xy[:, 0], xy[:, 1], c='r', edgecolor='none')
        plt.scatter(Epts[:, 0] + eps, Epts[:, 1], c='y', edgecolor='none')
        plt.scatter(NEpts[:, 0] + eps, NEpts[:, 1] + eps, c='g', edgecolor='none')
        plt.scatter(Npts[:, 0], Npts[:, 1] + eps, c='b', edgecolor='none')
        plt.scatter(NWpts[:, 0] - eps, NWpts[:, 1] + eps, c='w')
        plt.scatter(Wpts[:, 0] - eps, Wpts[:, 1], c='m', edgecolor='none')
        plt.scatter(SWpts[:, 0] - eps, SWpts[:, 1] - eps, c='k', edgecolor='none')
        plt.scatter(Spts[:, 0], Spts[:, 1] - eps, c='lightgrey', edgecolor='none')
        plt.scatter(SEpts[:, 0] - eps, SEpts[:, 1] - eps, c='c', edgecolor='none')
        plt.show()
    return xyout


def buffered_pts_to_periodic_network_parallelogram(xy, BL, PV, BBox='auto', flex_pvxy=False, check=False):
    """Crops to original bounding box and connects periodic BCs of sample in a parallelogram
    Note: Default BBox is such that original BBox is [-PV[0, 0]*0.5, -(PV[1, 1] + PV[0, 1])*0.5],
    [PV[0, 0]*0.5, (-PV[1, 1] + PV[0, 1])*0.5], etc.
    Presently this only allows parallelograms with vertical sides.

       /|PV[0] + PV[1]
      / |
     /  |
    |   | PV[0]
    |  /
    | /
    |/ (0,0)

    Parameters
    ----------
    xy : NP x 2 float array
        xy points with buffer points, so xy are in the triangular representation
    BL : #bonds x 2 int array
        Bond list for the network: a row with [i, j] means i and j are connected. If negative, across periodic boundary
    PV : 2 x 2 float array
        Periodic vectors: the first has x and y components, the second has only positive y component.
    BBox : 4 x 2 numpy float array
        bounding box for the network. Here, this must be a parallelogram.
        The first point must be the lower left point!
    check : bool
        Whether to view intermediate results

    Returns
    -------

    """
    if BBox == 'auto':
        # Assuming that BBox is centered and has width, height of LL[0], LL[1]
        BBox = 0.5 * np.array([[-PV[0, 0], -PV[1, 1] - PV[0, 1]],
                               [PV[0, 0], -PV[1, 1] + PV[0, 1]],
                               [PV[0, 0], PV[1, 1] + PV[0, 1]],
                               [-PV[0, 0], PV[1, 1] - PV[0, 1]]])

    bpath = mplpath.Path(BBox)
    keep = np.where(bpath.contains_points(xy))[0]
    if check:
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        plt.plot(BBox[:, 0], BBox[:, 1], 'r.-')
        plt.title('le.buffered_pts_to_periodic_network_parallelogram BBox')
        plt.show()

    minX = np.min(BBox[:, 0])
    maxX = np.max(BBox[:, 0])
    # Note that minY is the minY on the LEFT of the system, not total
    # Similarly, note that minY is the maxY on the RIGHT of the system, not total
    slope = PV[0, 1] / (maxX - minX)
    minY = BBox[0, 1]
    maxY = minY + PV[1, 1] + PV[0, 1]

    def lowerY(x):
        return minY + slope * (x - minX)

    def upperY(x):
        return minY + PV[1, 1] + slope * (x - minX)

    PVdict = {'e': PV[0],
              'n': PV[1],
              'w': -PV[0],
              's': -PV[1],
              'ne': PV[0] + PV[1],
              'nw': -PV[0] + PV[1],
              'sw': -PV[0] - PV[1],
              'se': PV[0] - PV[1]}

    # Create a kd tree of the points
    tree = scipy.spatial.KDTree(xy)

    # Find bonds that will be cut. For each bond, match to other particle and add pair to BL and PVxydict
    BLcut, cutIND = find_cut_bonds(BL, keep)

    if check:
        plt.scatter(xy[:, 0], xy[:, 1], c='g', marker='x')
        plt.scatter(xy[keep, 0], xy[keep, 1], c='b', marker='o')
        highlight_bonds(xy, BL, ax=plt.gca(), color='b', show=False)
        highlight_bonds(xy, BLcut, ax=plt.gca(), color='r', lw=1, show=False)
        xxtmp = np.hstack((BBox[:, 0], np.array(BBox[:, 0])))
        print 'xxtmp = ', xxtmp
        yytmp = np.hstack((BBox[:, 1], np.array(BBox[:, 1])))
        print 'yytmp = ', yytmp
        plt.plot(xxtmp, yytmp, 'k-', lw=2)
        plt.title('Showing bonds that are cut, btwn original and mirrored network')
        plt.show()
        plt.scatter(xy[:, 0], xy[:, 1], c='g', marker='x')
        plt.scatter(xy[keep, 0], xy[keep, 1], c='b', marker='o')
        highlight_bonds(xy, BL, ax=plt.gca(), color='b', show=False)
        highlight_bonds(xy, BLcut, ax=plt.gca(), color='r', lw=1, show=False)
        xxtmp = np.hstack((BBox[:, 0], np.array(BBox[:, 0])))
        print 'xxtmp = ', xxtmp
        yytmp = np.hstack((BBox[:, 1], np.array(BBox[:, 1])))
        print 'yytmp = ', yytmp
        plt.plot(xxtmp, yytmp, 'k-', lw=2)
        plt.title('Showing bonds that are cut, with pt #s')
        for ind in range(len(xy)):
            plt.text(xy[ind, 0] + 0.1, xy[ind, 1] - 0.1, str(ind))
        plt.show()

        # Prepare image to display NSWE scattered on top
        highlight_bonds(xy, BL, ax=plt.gca(), color='lightgrey', show=False)
        highlight_bonds(xy, BLcut, ax=plt.gca(), color='dimgray', lw=1, show=False)
        print 'preparing image....'

    # preallocate BL2add and PVs
    BL2add = np.zeros((len(BLcut), 2), dtype=int)
    PVd = {}
    kk = 0
    for bond in BLcut:
        # which endpt is outside?
        ptA = bond[0]
        ptB = bond[1]
        # mpt is short for 'mirror point', the point outside the bounding box
        if ptA not in keep:
            mpt, kpt = ptA, ptB
        else:
            mpt, kpt = ptB, ptA
        if xy[mpt, 0] < minX:
            # Mirror particle is to the left of the system (West)
            if xy[mpt, 1] < lowerY(xy[mpt, 0]):
                # Mirror particle is SW
                bPV = PVdict['sw']
            elif xy[mpt, 1] > upperY(xy[mpt, 0]):
                # Mirror particle is NW
                bPV = PVdict['nw']
            else:
                # Mirror particle is West
                bPV = PVdict['w']
        elif xy[mpt, 0] > maxX:
            # Mirror particles is the right of the system (East)
            if xy[mpt, 1] < lowerY(xy[mpt, 0]):
                # Mirror particle is SE
                bPV = PVdict['se']
            elif xy[mpt, 1] > upperY(xy[mpt, 0]):
                # Mirror particle is NE
                bPV = PVdict['ne']
            else:
                # Mirror particle is East
                bPV = PVdict['e']
        elif xy[mpt, 1] < lowerY(xy[mpt, 0]):
            # Mirror particle is South
            bPV = PVdict['s']
        else:
            # Mirror particle is North
            bPV = PVdict['n']

        if check:
            print 'adding pt...'
            if (bPV == PVdict['sw']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='r', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['w']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='g', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['nw']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='y', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['n']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='b', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['ne']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='c', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['e']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='m', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['se']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='k', edgecolor='none', zorder=9999)
            elif (bPV == PVdict['s']).all():
                plt.scatter(xy[mpt, 0], xy[mpt, 1], c='w', edgecolor='none', zorder=9999)

        # Link keep point (kpt) to the particle that resides a vector -PV away from mirror particle
        dist, ind = tree.query(xy[mpt] - bPV)
        BL2add[kk] = np.array([-kpt, -ind])
        PVd[(kpt, ind)] = bPV
        kk += 1

    if check:
        plt.plot(np.hstack((BBox[:, 0], np.array([BBox[0, 0]]))),
                 np.hstack((BBox[:, 1], np.array([BBox[0, 1]]))), 'r-')
        plt.show()

    if check:
        print 'PVd = ', PVd
        xyshake = xy + 0.1 * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        display_lattice_2D(xyshake, np.abs(BL), title="showing extended lattice (w/o PBCs)")

    # Crop network, and add back cut bonds as periodic ones
    BL = np.vstack((BL, BL2add))
    xytrim, NL, KL, BLtrim, PVxydict = remove_pts(keep, xy, BL)
    # Adjusting BL2add to account for smaller #npts (post-cropping) is already done in remove_pts
    # Adjust PVs to account for smaller #npts (post-cropping)
    remove = np.setdiff1d(np.arange(len(xy)), keep)

    if check:
        print 'PVd = ', PVd
        xyshake = xy + 0.1 * np.random.rand(np.shape(xy)[0], np.shape(xy)[1])
        display_lattice_2D(xyshake, np.abs(BL), title="showing extended lattice with BL2add", close=False)
        plt.scatter(xy[remove, 0], xy[remove, 1], c='c', zorder=999999)
        plt.show()

    # Use PVd (which included buffered pts) to make PVxydict
    PVxydict = {}
    for key in PVd:
        # adjust key to lower indices
        # count how many pts in remove are lower than key[0] and key[1], respectively
        lower0 = np.sum(remove < key[0])
        lower1 = np.sum(remove < key[1])
        newkey = (key[0] - lower0, key[1] - lower1)
        PVxydict[newkey] = PVd[key]
        # if lower0 > 0 or lower1 > 0:
        #     print 'key =', key
        #     print 'newkey =', newkey
        #     print 'lower0 =', lower0
        #     print 'lower1 =', lower1

    if check:
        # Plot lattice without PBCs
        display_lattice_2D(xytrim, np.abs(BLtrim), title="showing lattice connectivity w/o PBCs", close=False)
        for ind in range(len(xytrim)):
            plt.text(xytrim[ind, 0], xytrim[ind, 1], str(ind))

        plt.show()
        print 'PVxydict = ', PVxydict
        NL, KL = BL2NLandKL(BLtrim)
        PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL, check=check)
        print 'PVx = ', PVx
        display_lattice_2D(xytrim, BLtrim, PVxydict=PVxydict, PVx=PVx, PVy=PVy,
                           title="showing lattice connectivity with PBCs")

    return xytrim, NL, KL, BLtrim, PVxydict


def highlight_bonds(xy, BL, ax=None, color='r', lw=1, show=True):
    """Plot bonds specified in BL on specified axis.

    Parameters
    ----------
    xy
    BL
    ax
    color
    show

    Returns
    -------
    ax
    """
    if ax is None:
        ax = plt.gca()
    for bond in BL:
        ax.plot([xy[bond[0], 0], xy[bond[1], 0]], [xy[bond[0], 1], xy[bond[1], 1]], '-', color=color, lw=lw)
    if show:
        plt.show()
    return ax


def find_cut_bonds(BL, keep):
    """Identify which bonds are cut by the supplied mask 'keep'.

    Parameters
    ----------
    BL
    keep

    Returns
    -------
    BLcut : #cut bonds x 2 int array
        The cut bonds
    cutIND : #bonds x 1 int array
        indices of BL of cut bonds
    """
    # ensure that keep is int array of indices, not bool
    if keep.dtype == 'bool':
        print 'converting bool keep to int array...'
        keep = np.where(keep)[0]

    # Make output BLcut and the indices of BL that are cut (cutIND)
    # Find rows of BL for which both elems are in keep
    inBL0 = np.in1d(np.abs(BL[:, 0]), keep)
    inBL1 = np.in1d(np.abs(BL[:, 1]), keep)
    cutIND = np.logical_xor(inBL0, inBL1)
    BLcut = BL[cutIND, :]

    return BLcut, cutIND


def PBCmap_for_rectangular_buffer(xy, LL, dist=7.0):
    """Buffers the true point set with a mirrored point set across each boundary, for a rectangular boundary, and
    creates a dict mapping each mirror (outer) particle to its true (inner) particle.
    NOTE: This code is UNTESTED!

    Parameters
    ----------
    xy : NP x 2 float array
        Particle positions
    LL : tuple of 2 floats
        Horizontal and vertical extent of the bounding box (a rectangle) through which there are periodic boundaries.
    dist : float
        maximum depth of the buffer on each side

    Returns
    -------
    xyout :
    PBCmap : dict with int keys and tuple of (int, numpy 1x2 float array) as values
        Periodic boundary condition map, takes (key) index of xyout to (value) its reference/true point
        For example, say that particle 2 looks to particle 1 as if beyond a boundary, translated by (LL[0],0).
        Since this function buffers the true point set with a mirrored point set across each boundary,
        then PBCmap maps the index of the mirror point to its true particle position inside the bounding box of the
        sample, tupled with the true particle's periodic vector (the vector mapping the true point to the mirror pt).
    """
    # Copy some of lattice to north, south, east, west and corners
    west = np.where(xy[:, 0] < (np.min(xy[:, 0]) + dist))[0]
    sout = np.where(xy[:, 1] < (np.min(xy[:, 1]) + dist))[0]
    east = np.where(xy[:, 0] > (np.max(xy[:, 0]) - dist))[0]
    nort = np.where(xy[:, 1] > (np.max(xy[:, 1]) - dist))[0]
    swest = np.intersect1d(sout, west)
    seast = np.intersect1d(sout, east)
    neast = np.intersect1d(nort, east)
    nwest = np.intersect1d(nort, west)
    Epts = xy[west] + np.array([LL[0], 0.0])
    Npts = xy[sout] + np.array([0.0, LL[1]])
    Wpts = xy[east] + np.array([-LL[0], 0.0])
    Spts = xy[nort] + np.array([0.0, -LL[1]])
    NEpts = xy[swest] + np.array([LL[0], LL[1]])
    NWpts = xy[seast] + np.array([-LL[0], LL[1]])
    SWpts = xy[neast] + np.array([-LL[0], -LL[1]])
    SEpts = xy[nwest] + np.array([LL[0], -LL[1]])
    xyout = np.vstack((xy, Epts, NEpts, Npts, NWpts, Wpts, SWpts, Spts, SEpts))
    groupLister = ['w'] * len(west) + ['s'] * len(sout) + ['e'] * len(east) + ['n'] * len(nort) + \
                  ['sw'] * len(swest) + ['se'] * len(seast) + ['ne'] * len(neast) + ['nw'] * len(nwest)
    # PVdict maps the location of the true point to its mirror (outside bbox) point
    PVdict = {'w': np.array([LL[0], 0.0]),
              's': np.array([0.0, LL[1]]),
              'e': np.array([-LL[0], 0.0]),
              'n': np.array([0.0, -LL[1]]),
              'sw': np.array([LL[0], LL[1]]),
              'se': np.array([-LL[0], LL[1]]),
              'ne': np.array([-LL[0], -LL[1]]),
              'nw': np.array([LL[0], -LL[1]])}

    # Now form dictionary taking (key) index of xyout to (value) its reference/mirror point.
    # Note that order matters!!
    PBCmap = {}
    ind = len(xy)
    for group in [west, sout, east, nort, swest, seast, neast, nwest]:
        PV = PVdict[groupLister[ind - len(xy)]]
        for ii in group:
            PBCmap[ind] = (ii, PV)
            ind += 1

    return xyout, PBCmap


def centroid_lattice_from_TRI(xy, TRI, check=False):
    """Convert triangular representation (such as a triangulation) of lattice to xy, NL, KL of centroid lattice.

    Parameters
    ----------
    xy : #pts x 2 float array
        xy points from which to find centroids, so xy are in the triangular representation
    TRI : array of dimension #tris x 3
        Each row contains indices of the 3 points lying at the vertices of the tri.

    Returns
    ----------
    xy : #pts x 2 float array
        centroids for lattice
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : NP x NN int array
        Connectivity list
    BL : Nbonds x 2 int array
        Bond list
    BM : NP x NN float array
        unstretched (reference) bond length matrix
    """
    # Check if any elements of TRI are negative, and remove those rows of TRI
    # DO THIS

    # Compute centroids
    centxy = xyandTRI2centroid(xy, TRI)

    sizes = np.random.rand(len(centxy)) * 100
    colors = np.random.rand(len(centxy))
    if check:
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
        plt.scatter(centxy[:, 0], centxy[:, 1], s=sizes, c=colors, alpha=0.2)
        plt.show()

    # Now make NL and KL from TRI
    NL, KL = TRI2centroidNLandKL(TRI)
    BL = NL2BL(NL, KL)
    BM = NL2BM(centxy, NL, KL)

    return centxy, NL, KL, BL, BM


##########################################
# Plotting
##########################################
def collect_lines(xy, BL, bs, climv):
    """Creates collection of line segments, colored according to an array of values.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points
    bs : array of dimension #bonds x 1
        Strain in each bond
    climv : float or tuple
        Color limit for coloring bonds by bs

    Returns
    ----------
    line_segments : matplotlib.collections.LineCollection
        Collection of line segments
    """
    lines = [zip(xy[BL[i, :], 0], xy[BL[i, :], 1]) for i in range(len(BL))]
    line_segments = LineCollection(lines,  # Make a sequence of x,y pairs
                                   linewidths=1.,  # could iterate over list
                                   linestyles='solid',
                                   cmap='coolwarm',
                                   norm=plt.Normalize(vmin=-climv, vmax=climv))
    line_segments.set_array(bs)
    print(lines)
    return line_segments


def draw_lattice(ax, lat, bondcolor='k', colormap='BlueBlackRed', lw=1, climv=0.1):
    """Add a network to an axis instance (draw the bonds of the network)

    Parameters
    ----------
    ax
    xy
    BL
    bondcolor
    colormap
    lw
    climv

    Returns
    -------

    """
    if (lat.BL < 0).any():
        if lat.PVxydict is None:
            raise RuntimeError('PVxydict must be supplied to draw_lattice when periodic BCs exist!')
        else:
            PVx, PVy = PVxydict2PVxPVy(lat.PVxydict, lat.NL, lat.KL)
            # get indices of periodic bonds
            perINDS = np.unique(np.where(lat.BL < 0)[0])
            perBL = np.abs(lat.BL[perINDS])
            # # Check
            # print 'perBL = ', perBL
            # plt.plot(xy[:,0], xy[:,1],'b.')
            # for i in range(len(xy)):
            #     plt.text(xy[i,0]+0.05, xy[i,1],str(i))
            # plt.show()
            normINDS = np.setdiff1d(np.arange(len(lat.BL)), perINDS)
            BLtmp = lat.BL[normINDS]
            lines = [zip(lat.xy[BLtmp[i, :], 0], lat.xy[BLtmp[i, :], 1]) for i in range(len(BLtmp))]

            xy_add = np.zeros((4, 2))

            # Add periodic bond lines to image
            for row in perBL:
                # print 'NL[row[0]] = ', NL[row[0]]
                colA = np.argwhere(lat.NL[row[0]] == row[1])[0][0]
                colB = np.argwhere(lat.NL[row[1]] == row[0])[0][0]
                xy_add[0] = lat.xy[row[0]]
                xy_add[1] = lat.xy[row[1]] + np.array([PVx[row[0], colA], PVy[row[0], colA]])
                xy_add[2] = lat.xy[row[1]]
                xy_add[3] = lat.xy[row[0]] + np.array([PVx[row[1], colB], PVy[row[1], colB]])
                # print 'first line : ', zip(xy_add[0:2,0], xy_add[0:2,1])
                # print 'second line : ', zip(xy_add[2:4,0], xy_add[2:4,1])
                lines += zip(xy_add[0:2, 0], xy_add[0:2, 1]), zip(xy_add[2:4, 0], xy_add[2:4, 1])

                # CHECK
                # line_segments = LineCollection(lines, # Make a sequence of x,y pairs
                #                 linewidths    = lw, #could iterate over list
                #                 linestyles = 'solid',
                #                 cmap='seismic',
                #                 norm=plt.Normalize(vmin=-climv,vmax=climv))
                # ax.add_collection(line_segments)
                # for i in range(len(xy)):
                #     ax.text(xy[i,0] + 0.05, xy[i,1],str(i))
                # plt.pause(.01)
    else:
        if np.shape(BL)[0] > 1:
            lines = [zip(lat.xy[lat.BL[i, :], 0], lat.xy[lat.BL[i, :], 1]) for i in range(np.shape(lat.BL)[0])]
        else:
            lines = [zip(lat.xy[lat.BL[i][0]], lat.xy[lat.BL[i][1]]) for i in range(np.shape(lat.BL)[0])]

    if bondcolor is None:
        line_segments = LineCollection(lines,  # Make a sequence of x,y pairs
                                       linewidths=lw,  # could iterate over list
                                       linestyles='solid',
                                       cmap=colormap,
                                       norm=plt.Normalize(vmin=-climv, vmax=climv))
        line_segments.set_array(bs)
    else:
        line_segments = LineCollection(lines, linewidths=lw, linestyles='solid', colors=bondcolor)

    ax.add_collection(line_segments)
    return ax


# NOTE: timestep_plot() was moved to plotting.plotting


def construct_timestep_and_decomp_plot(fig, pos_ax, decomp_ax, time_step, R, eigval, eigvect, Ni, Nk, factor, amp,
                                       title_label=''):
    pylab.sca(pos_ax)
    Rx = R[:, 0]
    Ry = R[:, 1]
    pylab.xlim(Rx.min() - 1, Rx.max() + 1)
    pylab.ylim(Ry.min() - 1, Ry.max() + 1)
    pos_ax.set_autoscale_on(False)
    s = absolute_sizer()
    ppu = get_points_per_unit()

    # v1 = decomp_plot(fig, time_step-R, eigvect, eigval, title_label, decomp_ax)
    v2 = leplt.timestep_plot(time_step, R, Ni, Nk, pos_ax, factor=factor, amp=amp, title=title_label)
    return fig, v2


def get_points_per_unit(ax=None):
    """Use the size of the matplotlib axis to measure the number of points per unit"""
    if ax is None:
        ax = plt.gca()
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    return ax.bbox.width / abs(x1 - x0)


def normalize(vector):
    N = len(vector)
    tot = 0
    for i in range(N):
        tot += abs(vector[i]) ** 2

    return vector / np.sqrt(tot)


def gHST_plot(xy, NL, KL, BM, params, factor=1, climv=1., title=''):
    """makes plot in position space for time step. (Similar to timestep_plot, but for gHST without time-dependence)

    Parameters
    ----------
    xy: array NP x 5
        current positions of the gyros (x,y of pivot, theta, phi, psi)
    Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BM: NP x NN array
        rest bond lenth matrix, as NP x NN array
    factor : float (optional):
        factor to multiply displacements by for drawing ( to see them better)
    amp : float(optional):
        amplitude of maximum displacement
    title : string
        title of the plot

    Returns
    ----------
    [scat_fg, p, lines_st]:
        things to be cleared before next time step is drawn
    """
    try:
        NP, NN = np.shape(NL)
    except:
        print 'There is only one particle to plot.'
        NP = len(xy)
        NN = 0

    # Extract the euler angles
    theta = xy[:, 2]
    phi = xy[:, 3]
    # The pivot points are called R_p
    R_p = xy[:, 0:2]
    # pylab.sca(ax)
    ax = plt.gca()
    # ax.set_axis_bgcolor('#E8E8E8')
    Rx = R_p[:, 0]
    Ry = R_p[:, 1]

    l = params['l']
    diffx = l * np.sin(xy[:, 2]) * np.cos(xy[:, 3])
    diffy = l * np.sin(xy[:, 2]) * np.sin(xy[:, 3])

    # Make the circles
    patch = [None] * NP
    mag = np.sqrt(diffx ** 2 + diffy ** 2)
    mag[mag == 0] = 1
    # angles= np.arccos(diffx/mag)
    # angles[diffy<0] = 2*np.pi-angles[diffy<0]
    angles = np.mod(phi, 2 * np.pi)

    # the displayed points
    scat_x = Rx + factor * diffx
    scat_y = Ry + factor * diffy

    # the actual points
    ss_x = Rx + diffx
    ss_y = Ry + diffy

    # the circles
    patch = [patches.Circle((Rx[i], Ry[i]), radius=factor * mag[i]) for i in range(len(Rx))]

    z = np.zeros(len(scat_x))

    # Initialize streches vector to be longer than necessary
    inc = 0
    stretches = np.zeros(3 * len(R_p))

    test = list(stretches)
    for i in range(len(R_p)):
        # for j, k in zip(Ni[i], Nk[i]):
        if NN > 0:
            for j, k, q in zip(NL[i], KL[i], BM[i]):
                if i < j and abs(k) > 0:
                    # the distance between the actual points
                    n1 = float(np.linalg.norm(R_p[i] - R_p[j]))
                    stretches[inc] = n1 - q
                    test[inc] = [R_p[(i, j), 0], R_p[(i, j), 1]]
                    inc += 1

    test = test[0:inc]
    # print 'test = ', test
    lines = [zip(x, y) for x, y in test]
    stretch = np.array(stretches[0:inc])
    # print 'stretch = ', stretch

    # LINE Segments based on STretch --> lines_st
    lines_st = LineCollection(lines, array=stretch, cmap='seismic', linewidth=4)
    lines_st.set_clim([-climv, climv])
    if np.mean(theta) < np.pi * 0.5:
        print ''
        lines_st.set_zorder(0)
    else:
        lines_st.set_zorder(3)

    p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.6)

    p.set_array(P.array(angles))
    p.set_clim([0, 2 * np.pi])
    p.set_zorder(1)

    ax.add_collection(p)
    ax.add_collection(lines_st)

    fig = plt.gcf()
    axcb = fig.colorbar(lines_st)
    axcb.set_label('Strain')
    axcb.set_clim(vmin=-climv, vmax=climv)

    # Plot masses
    ax.set_aspect('equal')
    s = absolute_sizer()
    scat_fg = ax.scatter(scat_x, scat_y, s=s(0.02), c=angles, vmin=0., vmax=2. * np.pi, cmap='isolum_rainbow',
                         alpha=1, zorder=2)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    pylab.title(title)

    return [scat_fg, lines_st, p]


def gHST_plot_PL(xy, NL, KL, BM, params, factor=1, climv=1., title=''):
    """makes plot in position space for time step. Assumes b=0 (hanging gyros) for zorder of plotted collections (lines above circles).

    Parameters
    ----------
    xy: array NP x 4
        current positions of the gyros (x,y of pivot, dX, dY vector to center of mass projection in plane of pivot)
    NL : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    BM: NP x NN array
        rest bond lenth matrix, as NP x NN array
    factor : float (optional):
        factor to multiply displacements by for drawing ( to see them better)
    amp : float(optional):
        amplitude of maximum displacement
    title : string
        title of the plot

    Returns
    ----------
    [scat_fg, p, lines_st]:
        things to be cleared before next time step is drawn
    """
    try:
        NP, NN = np.shape(NL)
    except:
        print 'There is only one particle to plot.'
        NP = 1
        NN = 0

    # The pivot points are called R_p
    R_p = xy[:, 0:2]
    # pylab.sca(ax)
    ax = plt.gca()
    ax.set_axis_bgcolor('#d9d9d9')  # '#E8E8E8')
    # Pivot positions
    Rx = R_p[:, 0]
    Ry = R_p[:, 1]

    l = params['l']
    diffx = xy[:, 2]
    diffy = xy[:, 3]

    # Make the circles
    patch = [None] * NP
    mag = np.sqrt(diffx ** 2 + diffy ** 2)
    mag[mag == 0] = 1

    # angles= np.arccos(diffx/mag)
    # angles[diffy<0] = 2*np.pi-angles[diffy<0]
    angles = np.mod(np.arctan2(diffy, diffx), 2. * np.pi)

    # the displayed points
    scat_x = Rx + factor * diffx
    scat_y = Ry + factor * diffy

    # the actual points
    ss_x = Rx + diffx
    ss_y = Ry + diffy

    # the circles
    patch = [patches.Circle((Rx[i], Ry[i]), radius=factor * mag[i]) for i in range(len(Rx))]

    z = np.zeros(len(scat_x))

    # Initialize streches vector to be longer than necessary
    inc = 0
    stretches = np.zeros(3 * len(R_p))

    test = list(stretches)
    for i in range(len(R_p)):
        if NN > 0:
            # for j, k in zip(Ni[i], Nk[i]):
            for j, k, q in zip(NL[i], KL[i], BM[i]):
                if i < j and abs(k) > 0:
                    # the distance between the actual points
                    n1 = float(np.linalg.norm(R_p[i] - R_p[j]))
                    stretches[inc] = n1 - q
                    test[inc] = [R_p[(i, j), 0], R_p[(i, j), 1]]
                    inc += 1

    test = test[0:inc]
    lines = [zip(x, y) for x, y in test]
    stretch = np.array(stretches[0:inc])

    # LINE Segments based on STretch --> lines_st
    lines_st = LineCollection(lines, array=stretch, cmap='coolwarm', linewidth=4)
    lines_st.set_clim([-climv, climv])
    lines_st.set_zorder(3)

    p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.6)

    p.set_array(P.array(angles))
    p.set_clim([0, 2 * np.pi])
    p.set_zorder(1)

    ax.add_collection(p)
    ax.add_collection(lines_st)

    fig = plt.gcf()
    axcb = fig.colorbar(lines_st)
    axcb.set_label('Strain')
    axcb.set_clim(vmin=-climv, vmax=climv)

    # Plot masses
    ax.set_aspect('equal')
    s = absolute_sizer()
    scat_fg = ax.scatter(scat_x, scat_y, s=s(0.02), c=angles, vmin=0., vmax=2. * np.pi, cmap='isolum_rainbow',
                         alpha=1, zorder=2)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(title)

    return [scat_fg, lines_st, p]


def data2energy_plot_rigidbody(params, datadir, ylimmax, plotBIND=True):
    """
    Plot energy of a single rigid body (like a gHST) over time

    Parameters
    ----------
    plotBIND : boolean
        if True, plots the different energies over time for only those particles that are not on the boundary
        as a separate plot and saves this too
    """
    I1 = params['I1']
    I3 = params['I3']
    w3 = params['w3']
    g = params['g']
    l = params['l']
    h = params['h']

    # get dirs
    xypath = sorted(glob.glob(datadir + 'xyv/'))[0]
    KLpath = sorted(glob.glob(datadir + 'KL/'))[0]
    print 'found KL dir : ', KLpath
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))
    KLfiles = sorted(glob.glob(KLpath + '*.txt'))
    # load setup
    xy0file = sorted(glob.glob(datadir + 'xy.txt'))[0]
    xyv0file = sorted(glob.glob(datadir + 'xyv0.txt'))[0]

    # Note the inversion of the naming system below
    # xy file is mesh, store as xy0
    xy0 = np.loadtxt(xy0file, delimiter=',', usecols=(0, 1))

    # xyv0 file is initial condition, store as xy and v, etc
    xy = np.loadtxt(xyv0file, delimiter=',', usecols=(0, 1))
    theta, phi, vX, vY, vtheta, vphi, vpsi = \
        np.loadtxt(xyv0file, delimiter=',', usecols=(2, 3, 5, 6, 7, 8, 9), unpack=True)

    NLfile = sorted(glob.glob(datadir + 'NL.txt'))[0]
    NL = np.loadtxt(NLfile, dtype='int', delimiter=',')

    # Load masses, from file, or else from params, or else define it as unity
    if os.path.exists(datadir + 'Mm.txt'):
        Mfile = sorted(glob.glob(datadir + 'Mm.txt'))[0]
        Mm = np.loadtxt(Mfile)
        print 'Loaded mass list from txt file...'
    else:
        try:
            Mm = params['Mm']
            print 'Loaded mass list from params dict...'
        except:
            Mm = np.ones_like(xy0[:, 0])
            print 'WARNING: Could not find mass array, defined as unity...'

            # Calculate or load initial setup --> assumes KL at h=0 is same as initial
    KL = np.loadtxt(KLfiles[0], delimiter=',')
    BL = NL2BL(NL, KL)
    bo = bond_length_list(xy0, BL)

    kL = KL2kL(NL, KL, BL) * params['k']
    nzcount = np.count_nonzero(KL)
    NP = len(xy0[:, 0])

    # Define single particle to track (pp)
    BIND = params['BIND']
    bulkIND = np.setdiff1d(np.arange(len(xy[:, 0])), BIND)
    pp = bulkIND[0]

    fig = plt.figure(1)
    plt.clf()
    ax = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)

    # Calc and plot energy
    T1 = 0.5 * (vX ** 2 + vY ** 2)
    tmp1 = vtheta * np.cos(theta) * np.cos(phi) - vphi * np.sin(theta) * np.sin(phi)
    tmp2 = vtheta * np.cos(theta) * np.sin(phi) + vphi * np.sin(theta) * np.cos(phi)
    T2 = Mm * l * (vX * tmp1 + vY * tmp2)
    T3 = 0.5 * (I1 + Mm * l ** 2) * (vphi ** 2 * np.sin(theta) ** 2 + vtheta ** 2)
    T4 = 0.5 * I3 * w3 ** 2
    T4alt = 0.5 * I3 * (vpsi + vphi * np.cos(theta)) ** 2
    KE_v = T1 + T2 + T3 + T4
    KE = KE_v[pp]
    gU_v = Mm * l * g * np.cos(theta)
    gU = gU_v[pp]
    # bU = 0.5*k*
    totE = KE + gU

    ax.plot(-h, totE - T4[pp], 'k.')
    # ax2.plot(-h,KE-T4[pp],'r.')
    ax2.plot(-h, T1[pp] + T2[pp], 'r.')
    ax3.plot(-h, T3[pp], 'r.')
    ax4.plot(-h, T4alt[pp], 'r.')
    ax5.plot(-h, gU, 'g.')

    # Run through saved data, calc and plot energy
    print('Plotting energy vs t...')
    doall = range(0, len(xyfiles))
    for i in doall:
        KL = np.loadtxt(KLfiles[i], delimiter=',')
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]
        t = float(iterind) * h

        x, y, theta, phi = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1, 2, 3), unpack=True)
        vX, vY, vtheta, vphi, vpsi = np.loadtxt(xyfiles[i], delimiter=',', usecols=(5, 6, 7, 8, 9), unpack=True)
        xy = np.dstack((x, y))[0]

        # redo calculations of BL, bo, kL if bonds were cut
        if np.count_nonzero(KL) != nzcount:
            BL = NL2BL(NL, KL)
            bo = bond_length_list(xy0, BL)
            kL = KL2kL(NL, KL, BL) * params['k']
            nzcount = np.count_nonzero(KL)

        T1 = 0.5 * Mm * (vX ** 2 + vY ** 2)
        tmp1 = vtheta * np.cos(theta) * np.cos(phi) - vphi * np.sin(theta) * np.sin(phi)
        tmp2 = vtheta * np.cos(theta) * np.sin(phi) + vphi * np.sin(theta) * np.cos(phi)
        T2 = Mm * l * (vX * tmp1 + vY * tmp2)
        T3 = 0.5 * (I1 + Mm * l ** 2) * (vphi ** 2 * np.sin(theta) ** 2 + vtheta ** 2)
        T4 = 0.5 * I3 * w3 ** 2
        T4alt = 0.5 * I3 * (vpsi + vphi * np.cos(theta)) ** 2
        KE_v = T1 + T2 + T3 + T4
        KE = KE_v[pp]
        gU_v = Mm * l * g * np.cos(theta)
        gU = gU_v[pp]
        # bU = 0.5*k*
        totE = KE + gU

        ax.plot(t, totE - T4[pp], 'k.')
        # ax2.plot(t,KE-T4[pp],'r.')
        ax2.plot(t, T1[pp] + T2[pp], 'r.')
        ax3.plot(t, T3[pp], 'r.')
        ax4.plot(t, T4alt[pp], 'r.')
        ax5.plot(t, gU, 'g.')

    # ax.legend()
    title = 'Energy Conservation -- Single Gyro'
    ax.set_title(title)
    ax5.set_xlabel('time')
    ax.set_ylabel(r'Total - $\frac{I_3 \omega_3^2}{2} $')
    ax2.set_ylabel('Translational')
    ax3.set_ylabel('Rotational')
    ax4.set_ylabel('Fast Rot.')
    ax5.set_ylabel('Gravitational')

    ax.set_xlim([-h, t])
    ax2.set_xlim([-h, t])
    ax3.set_xlim([-h, t])
    ax4.set_xlim([-h, t])
    ax5.set_xlim([-h, t])

    ax.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])

    outname = datadir + 'Energy_vs_t_single.png'
    plt.savefig(outname)


def plot_track_rigidbody(params, datadir, pp=4, zoom=None):
    """Track a rigid body's position and orientation in space and plot the result.

    Parameters
    ----------
    params : dict
        dictionary of parameters for simulation
    datadir : string
        path for simulation data
    pp : ind
        particles to track and plot
    zoom : float or None
        if float, saves figure from time t=-h to t=zoom
    """
    h = params['h']

    # get dirs
    xypath = sorted(glob.glob(datadir + 'xyv/'))[0]
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))

    plt.clf()
    ax1 = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)
    doall = range(0, len(xyfiles))
    for i in doall:
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]
        t = float(iterind) * h

        x, y, theta, phi, psi = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1, 2, 3, 4), unpack=True)
        # vX,vY,vtheta,vphi,vpsi = np.loadtxt(xyfiles[i],delimiter=',', usecols=(5,6,7,8,9), unpack=True)

        ax1.plot(t, x[pp], 'k.')
        ax2.plot(t, y[pp], 'r.')
        ax3.plot(t, theta[pp], 'r.')
        ax4.plot(t, phi[pp], 'r.')
        ax5.plot(t, psi[pp], 'g.')

    title = 'Tracking Single Gyro'
    ax1.set_title(title)
    ax5.set_xlabel('time')
    ax1.set_ylabel('X')
    ax2.set_ylabel('Y')
    ax3.set_ylabel(r'$\theta$')
    ax4.set_ylabel(r'$\phi$')
    ax5.set_ylabel(r'$\psi$')

    ax1.set_xlim([-h, t])
    ax2.set_xlim([-h, t])
    ax3.set_xlim([-h, t])
    ax4.set_xlim([-h, t])
    ax5.set_xlim([-h, t])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])

    outname = datadir + 'Track.png'
    plt.savefig(outname)

    if zoom != None:
        ax1.set_xlim([-h, zoom])
        ax2.set_xlim([-h, zoom])
        ax3.set_xlim([-h, zoom])
        ax4.set_xlim([-h, zoom])
        ax5.set_xlim([-h, zoom])

        outname = datadir + 'Track_zoom.png'
        plt.savefig(outname)


def data2energy_plot(params, datadir, ylimmax='auto', plotBIND=False):
    """Plots potential, kinetic, and total energy as function of time for gyros or HSTs.
    Data can be 5d (HST), 3d (HST-planar limit), or 2d (pinned, fast-spinning limit)

    Parameters
    ----------
    params : dict
        parameters associated with the simulation
    datadir : string
        The directory where the data is stored (contains subdirs for xyv, KL) and to which the plot is saved
    ylimmax : float
        Optional, maximum energy per particle for y axis limit of top subplot. If 'auto', auto limits are used.
    plotBIND : boolean
        Whether or not to plot the bulk particles as well, separately, as squares on each plot

    Returns
    ----------
    """
    # get dirs
    xypath = sorted(glob.glob(datadir + 'xyv/'))[0]
    KLpath = sorted(glob.glob(datadir + 'KL/'))[0]
    print 'found KL dir : ', KLpath
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))
    KLfiles = sorted(glob.glob(KLpath + '*.txt'))
    # load setup
    xy0file = sorted(glob.glob(datadir + 'xy.txt'))[0]
    xyv0file = sorted(glob.glob(datadir + 'xyv0.txt'))[0]

    # Note the inversion of the naming system below!
    # xy file is mesh, here stored as the variable xy0
    xy0 = np.loadtxt(xy0file, delimiter=',', usecols=(0, 1))

    # xyv0 file is initial condition, store as xy and v, etc
    xy = np.loadtxt(xyv0file, delimiter=',', usecols=(0, 1))
    try:
        '''Data is 5D (x,y,theta,phi,psi)'''
        theta, phi, vx, vy, vtheta, vphi, vpsi = \
            np.loadtxt(xyv0file, delimiter=',', usecols=(2, 3, 5, 6, 7, 8, 9), unpack=True)
        dimension = 5
    except:
        try:
            '''Data is 3D'''
            v0 = np.loadtxt(xyv0file, delimiter=',', usecols=(3, 4))
            dimension = 3
        except:
            '''Data is 2D'''
            v0 = np.loadtxt(xyv0file, delimiter=',', usecols=(2, 3))
            dimension = 2

    NLfile = sorted(glob.glob(datadir + 'NL.txt'))[0]
    NL = np.loadtxt(NLfile, dtype='int', delimiter=',')
    # hfile = sorted(glob.glob(datadir+'h.txt'))[0]
    # h = np.loadtxt(hfile)
    h = params['h']

    # Load masses, from file, or else from params, or else define it as unity
    if os.path.exists(datadir + 'Mm.txt'):
        Mfile = sorted(glob.glob(datadir + 'Mm.txt'))[0]
        Mm = np.loadtxt(Mfile)
        print 'Loaded mass list from txt file...'
    else:
        try:
            Mm = params['Mm']
            print 'Loaded mass list from params dict...'
        except:
            Mm = np.ones_like(xy0[:, 0])
            print 'le: WARNING: Could not find mass array, defined as unity...'

    # Calculate or load initial setup --> assumes KL at h=0 is same as initial
    KL = np.loadtxt(KLfiles[0], delimiter=',')
    BL = NL2BL(NL, KL)
    bo = bond_length_list(xy0, BL)

    kL = KL2kL(NL, KL, BL) * params['k']
    nzcount = np.count_nonzero(KL)
    NP = len(xy0[:, 0])
    print('NP = ' + str(NP))

    fig = plt.figure(1)
    plt.clf()
    if dimension < 5:
        ax = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
    else:
        ax = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2)
        ax3 = plt.subplot(4, 1, 3)
        ax4 = plt.subplot(4, 1, 4)

    # Calc and plot initial vals
    bU = potential_energy(xy, BL, bo, kL)
    if dimension < 5:
        KE = kinetic_energy(v0, Mm)
        totE = bU + KE
    else:
        KE, KEvec, KE_bulk, T1, T2, T3, T4 = \
            kinetic_energy_rigidbody(theta, phi, vx, vy, vtheta, vphi, vpsi, Mm, params)
        print 'KE = ', KE
        # Gravitational potential energy
        gU_v = Mm * params['g'] * params['l'] * np.cos(theta)
        gU = np.sum(gU_v)
        totE = bU + KE + gU
        ax4.plot(-h, gU, 'g.')  # label='Gravitational')
        # ax3.plot(-h,totE-np.sum(0.5*params['I3']*(vpsi+vphi*np.cos(theta))**2),'k.')

    ax.plot(-h, totE - T4, 'k.')  # ,label='Total')
    ax3.plot(-h, bU, 'b.')  # ,label='Elastic')
    ax2.plot(-h, KE - T4, 'r.')  # ,label='Kinetic')
    ax2.plot(-h, T3, 'rx')  # ,label='Kinetic')

    if plotBIND:
        if 'BIND' in params:
            if len(params['BIND']) > 0:
                BIND = params['BIND']
                fig2 = plt.figure(2)
                plt.clf()
                ax21 = plt.subplot(4, 1, 1)
                ax22 = plt.subplot(4, 1, 2)
                ax23 = plt.subplot(4, 1, 3)
                ax24 = plt.subplot(4, 1, 4)
                # ax.plot(-h,bU,'b.',label='Bulk Potential')
                bulkIND = np.setdiff1d(np.arange(len(xy[:, 0])), BIND)
                gU_bulk = np.sum(gU_v[bulkIND])
                print 'bulkIND = ', bulkIND
                print 'gU_particle4 = ', gU_bulk
                totE_bulk = KE_bulk + bU + gU_bulk
                ax21.plot(-h, totE_bulk, 's', color=[0.3, .3, .3])
                ax22.plot(-h, KE_bulk, 's', color=[1.0, .5, .5])  # ,label='Bulk Kinetic')
                ax23.plot(-h, bU, 'bs')
                ax24.plot(-h, gU_bulk, 's', color=[0.5, 1.0, .5], markeredgecolor='none')

                ax21.set_ylabel('Total')
                ax24.set_xlabel('time')
                ax22.set_ylabel('Kinetic')
                ax23.set_ylabel('Elastic')
                ax24.set_ylabel('Gravitational')

    # make list of indices to plot
    doall = range(0, len(xyfiles))

    print('Plotting energy vs t...')
    for i in doall:
        KL = np.loadtxt(KLfiles[i], delimiter=',')
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]
        t = float(iterind) * h

        if dimension < 5:
            xy = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1))
            v = np.loadtxt(xyfiles[i], delimiter=',', usecols=(2, 3))
            KE = kinetic_energy(v, Mm)
        else:
            x, y, theta, phi = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1, 2, 3), unpack=True)
            vx, vy, vtheta, vphi, vpsi = np.loadtxt(xyfiles[i], delimiter=',', usecols=(5, 6, 7, 8, 9), unpack=True)
            xy = np.dstack((x, y))[0]
            KE, KEvec, KE_bulk, T1, T2, T3, T4 = kinetic_energy_rigidbody(theta, phi, vx, vy, vtheta, vphi, vpsi, Mm,
                                                                          params)
            # Gravitational potential energy
            gU_v = Mm * params['g'] ** params['l'] * np.cos(theta)
            gU = np.sum(gU_v)

        # redo calculations of BL, bo, kL if bonds were cut
        if np.count_nonzero(KL) != nzcount:
            BL = NL2BL(NL, KL)
            bo = bond_length_list(xy0, BL)
            kL = KL2kL(NL, KL, BL) * params['k']
            nzcount = np.count_nonzero(KL)

        if 'shrinkrate' in params:
            if 'prestrain' in params:
                bU = potential_energy(xy, BL, bo * (1. - params['prestrain'] - params['shrinkrate'] * t), kL)
            else:
                bU = potential_energy(xy, BL, bo * (1. - params['shrinkrate'] * t), kL)
        else:
            bU = potential_energy(xy, BL, bo, kL)

        if dimension < 5:
            totE = bU + KE
        else:
            totE = bU + KE + gU
            ax4.plot(t, gU, 'g.')
            # ax3.plot(t,totE-np.sum(0.5*params['I3']*(vpsi+vphi*np.cos(theta))**2),'k.')
            # ax3.plot(t,totE-np.sum(0.5*params['I3']*params['w3']**2),'k.')

        ax.plot(t, totE - T4, 'k.')
        ax2.plot(t, KE - T4, 'r.')
        # ax2.plot(t,T1,'ro') #,label='Kinetic')
        # ax2.plot(t,T2,'r^') #,label='Kinetic')
        # ax2.plot(t,T3,'rx') #,label='Kinetic')
        ax3.plot(t, bU, 'b.')

        if plotBIND:
            if 'BIND' in params:
                if len(params['BIND']) > 0:
                    # ax.plot(-h,bU,'b.',label='Bulk Potential')
                    ax22.plot(t, KE_bulk, 's', color=[1.0, .5, .5], markeredgecolor='none')  # 'Bulk Kinetic'
                    gU_bulk = np.sum(gU_v[bulkIND])
                    totE_bulk = KE_bulk + bU + gU_bulk
                    ax24.plot(t, gU_bulk, 's', color=[0.5, 1.0, .5], markeredgecolor='none')
                    ax23.plot(t, bU, 'bs')
                    ax21.plot(t, totE_bulk, 's', color=[0.5, .3, .3], markeredgecolor='none')  # 'Bulk Total'

    title = 'Energy Conservation'
    ax.set_title(title)
    ax.set_ylabel(r'Total - $\frac{I_3 \omega_3^2}{2} $')
    ax3.set_xlabel('time')
    ax2.set_ylabel(r'Kinetic - $\frac{I_3  \omega_3^2}{2}$')
    ax3.set_ylabel('Elastic')

    ax.set_xlim([-h, t])
    ax2.set_xlim([-h, t])
    ax3.set_xlim([-h, t])

    if dimension == 5:
        ax4.set_ylabel('Gravitational')
        ax4.set_xlim([-h, t])
        ax3.set_xlabel('')
        ax4.set_xlabel('time')
        ax3.set_xticks([])

    # ylimit of first axis is either the last total energy or a large number
    if ylimmax != 'auto':
        ax.set_ylim(0, NP * ylimmax)  # np.nanmin((totE,NP*ylimmax)))
        print('ylim_max = ' + str(np.nanmin((totE, NP * ylimmax))))

    ax.set_xticks([])
    ax2.set_xticks([])

    outname = datadir + 'Energy_vs_t.png'
    plt.figure(1)
    plt.savefig(outname)

    if plotBIND:
        if 'BIND' in params:
            if len(params['BIND']) > 0:
                plt.figure(2)
                ax21.set_xlim([-h, t])
                ax22.set_xlim([-h, t])
                ax23.set_xlim([-h, t])
                ax24.set_xlim([-h, t])

                ax21.set_xticks([])
                ax22.set_xticks([])
                ax23.set_xticks([])

                outname = datadir + 'Energy_vs_t_BIND.png'
                plt.savefig(outname)

                # ax.set_xlim([-h,5])
                # ax2.set_xlim([-h,5])
                # ax3.set_xlim([-h,5])
                # ax4.set_xlim([-h,5])
                #
                # ax.set_xticks([])
                # ax2.set_xticks([])
                #
                # outname = datadir+'Energy_vs_t_zoom.png'
                # #plt.show()
                # plt.figure(1)
                # plt.savefig(outname)


def plotposition_2particle_linear(datadir, ylimmax=0, fixedL=0):
    """Plots positions of two particles and theoretical predictions as function of time.

    Parameters
    ----------
    datadir : string
        The output directory for the data (contains subdirs for xyv, KL)
    ylimmax : float
        Optional, maximum energy per particle for y axis limit of top subplot. If zero, auto limits are used.
    fixedL : int
        Optional, if nonzero, then treats the left particle as a fixed wall

    Returns
    ----------
    """
    # PLOT DAMPED MOTION
    # get dirs
    xypath = sorted(glob.glob(datadir + 'xyv/'))[0]
    KLpath = sorted(glob.glob(datadir + 'KL/'))[0]
    # list files
    xyfiles = sorted(glob.glob(xypath + '*.txt'))
    KLfiles = sorted(glob.glob(KLpath + '*.txt'))
    # load setup
    xyv0file = sorted(glob.glob(datadir + 'xyv0.txt'))[0]
    xy0 = np.loadtxt(xyv0file, delimiter=',', usecols=(0, 1))
    try:
        '''Data is 3D'''
        # x0,y0,z0,vx0,vy0,vz0 = np.loadtxt(xyv0file,delimiter=',', unpack=True)
        v0 = np.loadtxt(xyv0file, delimiter=',', usecols=(3, 4))
    except:
        '''Data is 2D'''
        # x0,y0,vx0,vy0 = np.loadtxt(xyv0file,delimiter=',', unpack=True)
        v0 = np.loadtxt(xyv0file, delimiter=',', usecols=(2, 3))

    NLfile = sorted(glob.glob(datadir + 'NL.txt'))[0]
    NL = np.loadtxt(NLfile, dtype='int', delimiter=',')
    hfile = sorted(glob.glob(datadir + 'h.txt'))[0]
    h = np.loadtxt(hfile)
    betafile = sorted(glob.glob(datadir + 'beta.txt'))[0]
    beta = np.loadtxt(betafile)

    # Load masses, from file, or else from params, or else define it as unity
    if os.path.exists(datadir + 'Mm.txt'):
        Mfile = sorted(glob.glob(datadir + 'Mm.txt'))[0]
        Mm = np.loadtxt(Mfile)
        print 'Loaded mass list from txt file...'
    else:
        try:
            Mm = params['Mm']
            print 'Loaded mass list from params dict...'
        except:
            Mm = np.ones_like(xy0[:, 0])
            print 'le: WARNING: Could not find mass array, defined as unity...'

    # Calculate or load initial setup --> assumes KL at h=0 is same as initial
    KL = np.loadtxt(KLfiles[0], delimiter=',')
    BL = NL2BL(NL, KL)
    bo = bond_length_list(xy0, BL)
    kL = KL2kL(NL, KL, BL) * params['k']
    nzcount = np.count_nonzero(KL)
    NP = len(x0)
    # print('NP = '+str(NP))

    plt.clf()
    ax = plt.subplot(1, 1, 1)
    # Calc and plot initial vals
    bU = potential_energy(xy0, BL, bo, kL)
    KE = kinetic_energy(v0, Mm)
    totE = bU + KE
    ax.plot(-h, 0, 'b.', label='particle 1')
    ax.plot(-h, 0, 'r.', label='particle 2')

    # make list of indices to plot
    doall = range(0, len(xyfiles))

    print('Plotting position vs t...')
    for i in doall:
        xy = np.loadtxt(xyfiles[i], delimiter=',', usecols=(0, 1))
        KL = np.loadtxt(KLfiles[i], delimiter=',')
        iterind = (xyfiles[i].split('_')[-1]).split('.')[0]
        t = float(iterind) * h

        # redo calculations of BL, bo, kL if bonds were cut
        if np.count_nonzero(KL) != nzcount:
            BL = NL2BL(NL, KL)
            bo = bond_length_list(xy0, BL)
            kL = KL2kL(NL, KL, BL)
            nzcount = np.count_nonzero(KL)

        if i == 0:
            # print(xy)
            if fixedL == 0:
                print('Assuming equal and opposite displacements...')
                # amplitude of oscillation
                A = (abs(xy[1, 0] - xy0[1, 0]) + abs(xy[0, 0] - xy0[0, 0])) / 2.
            else:
                print('Using left particle as fixed wall, with pt0 having 0 displ...')
                A = xy[1, 0] - xy0[1, 0]

        ax.plot(t, xy[0, 0] - xy0[0, 0], 'b.')
        ax.plot(t, xy[1, 0] - xy0[1, 0], 'r.')

    # plot theory
    time = np.arange(0, t, t / 100.)
    m = 1.
    k = 1.
    if fixedL == 0:
        meff = (m * m) / (m + m);
        omega = np.sqrt(k / meff - beta ** 2 / (4. * meff ** 2))
        x = A * np.exp(-beta * time / (2 * meff)) * np.cos(omega * time)
        ax.plot(time, -x, 'k.')
        ax.plot(time, x, 'k.')
    else:
        print('Using left particle as fixed wall...')
        omega = np.sqrt(k / m - beta ** 2 / (4. * m ** 2))
        x = A * np.exp(-beta * time / (2 * m)) * np.cos(omega * time)
        ax.plot(time, np.zeros_like(x), 'k--')
        ax.plot(time, x, 'k--')

    ax.legend()
    title = 'Particle Motion'
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('position')
    # ylimit of first axis is either the last total energy or a large number
    if ylimmax != 0:
        ax.set_ylim(0, NP * ylimmax)
        print('ylim_max = ' + str(np.nanmin((totE, NP * ylimmax))))
    outname = datadir + 'position_vs_t.png'
    plt.savefig(outname)


def display_lattice_2D(xy, BL, NL=[], KL=[], BLNNN=[], NLNNN=[], KLNNN=[], PVxydict={}, PVx=[], PVy=[], bs='none',
                       title='', xlimv=None, ylimv=None, climv=0.1, colorz=True, ptcolor=None, ptsize=10,
                       close=True, colorpoly=False, viewmethod=False, labelinds=False,
                       colormap='seismic', bgcolor='#d9d9d9', axis_off=False, fig=None, ax=None, linewidth=0.0,
                       edgecolors=None, check=False):
    # another popular choice: 'BlueBlackRed' , #FFFFFF True
    """Plots and displays a 2D image of the lattice with colored bonds.

    Parameters
    ----------
    xy : array of dimension nx2
        2D lattice of points (positions x,y)
    BL : array of dimension #bonds x 2
        Each row is a bond and contains indices of connected points. Negative values denote periodic BCs
    bs : array of dimension #bonds x 1 (like np.shape(BL[:, 0])
        Strain in each bond
    fname : string
        Full path including name of the file (.png, etc)
    title : string
        The title of the frame
    climv : float or tuple
        Color limit for coloring bonds by bs
    colorz : bool
        Whether to color the particles by their coordination number
    close : bool
        Whether or not to leave the plot hanging to force it to be closed
    colorpoly : bool
        Whether to color polygons by the number of edges
    BLNNN : NP x NNNN int array
        Bond list for next nearest neighbor couplings.

    Returns
    ----------
    ax : matplotlib axis instance or None
        If close==True, returns None. Otherwise returns the axis with the network plotted on it.
    """
    if bs == 'none':
        bs = np.zeros_like(BL[:, 0])

    if ax is None:
        plt.clf()
        ax = plt.axes()

    NP = len(xy)
    if linewidth == 0.0:
        if NP < 10000:
            lw = 2.
        else:
            lw = (30 / np.sqrt(len(xy)))
    else:
        lw = linewidth

    if NL == [] or KL == []:
        if colorz or colorpoly:
            NL, KL = BL2NLandKL(BL, NP=NP, NN='min')
            if (BL < 0).any():
                if PVxydict is None:
                    raise RuntimeError('PVxydict must be supplied to display_lattice_2D() when periodic BCs exist,' +
                                       ' if NL and KL not supplied!')
                elif len(PVxydict) == 0:
                    raise RuntimeError('PVxydict must be supplied to display_lattice_2D() when periodic BCs exist,' +
                                       ' if NL and KL not supplied!')
                else:
                    PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL, check=check)

    if colorz:
        zvals = (KL != 0).sum(1)
        zmed = np.median(zvals)
        # print 'zmed = ', zmed
        under1 = np.logical_and(zvals < zmed - 0.5, zvals > zmed - 1.5)
        over1 = np.logical_and(zvals > zmed + 0.5, zvals < zmed + 1.5)
        Cz = np.zeros((len(xy), 3), dtype=int)
        # far under black // under blue // equal white // over red // far over green
        Cz[under1] = [0. / 255, 50. / 255, 255. / 255]
        Cz[zvals == zmed] = [255. / 255, 255. / 255, 255. / 255]
        Cz[over1] = [255. / 255, 0. / 255, 0. / 255]
        Cz[zvals > zmed + 1.5] = [0. / 255, 255. / 255, 50. / 255]
        # Cz[zvals<zmed-1.5] = [0./255,255./255,150./255] #leave these black

        s = leplt.absolute_sizer()
        sval = min([.005, .12 / np.sqrt(len(xy))])
        sizes = np.zeros(NP, dtype=float)
        sizes[zvals > zmed + 0.5] = sval
        sizes[zvals < zmed - 0.5] = sval

        # topinds = zvals!=zmed
        # print 'coloring by coordination: Cz=', Cz
        # plt.plot(np.arange(len(sizes)), sizes, 'b.-')
        # plt.show()
        # sys.exit()
        ax.scatter(xy[:, 0], xy[:, 1], s=s(sizes), c=Cz, edgecolor='none', zorder=10)
        ax.axis('equal')
    elif ptcolor is not None:
        if NP < 1000:
            # if smallish #pts, plot them
            # print 'xy = ', xy
            # plt.plot(xy[:,0],xy[:,1],'k.')
            s = leplt.absolute_sizer()
            ax.scatter(xy[:, 0], xy[:, 1], s=ptsize, alpha=0.5, color=ptcolor, zorder=10, edgecolors=edgecolors)

    # Efficiently plot many lines in a single set of axes using LineCollection
    # First check if there are periodic bonds
    if (BL < 0).any():
        print 'displaying with periodic bonds...'
        if len(PVx) == 0 or len(PVy) == 0:
            if len(PVxydict) == 0:
                raise RuntimeError('PVx and PVy or PVxydict must be supplied when periodic BCs exist!')
            else:
                PVx, PVy = PVxydict2PVxPVy(PVxydict, NL, KL)

        # get indices of periodic bonds
        perINDS = np.unique(np.where(BL < 0)[0])
        perBL = np.abs(BL[perINDS])
        normINDS = np.setdiff1d(np.arange(len(BL)), perINDS)
        BLtmp = BL[normINDS]
        bstmp = bs[normINDS]
        lines = [zip(xy[BLtmp[i, :], 0], xy[BLtmp[i, :], 1]) for i in range(len(BLtmp))]

        xy_add = np.zeros((4, 2))
        # Build new strain list bs_out by storing bulk lines first, then recording the strain twice
        # for each periodic bond since the periodic bond is two lines in the plot
        bs_out = np.zeros(len(normINDS) + 2 * len(perINDS), dtype=float)
        bs_out[0:len(normINDS)] = bstmp

        # Add periodic bond lines to image
        kk = 0
        for row in perBL:
            colA = np.argwhere(NL[row[0]] == row[1])[0][0]
            colB = np.argwhere(NL[row[1]] == row[0])[0][0]
            xy_add[0] = xy[row[0]]
            xy_add[1] = xy[row[1]] + np.array([PVx[row[0], colA], PVy[row[0], colA]])
            xy_add[2] = xy[row[1]]
            xy_add[3] = xy[row[0]] + np.array([PVx[row[1], colB], PVy[row[1], colB]])
            lines += zip(xy_add[0:2, 0], xy_add[0:2, 1]), zip(xy_add[2:4, 0], xy_add[2:4, 1])
            bs_out[2 * kk + len(normINDS)] = bs[perINDS[kk]]
            bs_out[2 * kk + 1 + len(normINDS)] = bs[perINDS[kk]]
            kk += 1

        # replace bs by the new bs (bs_out)
        bs = bs_out
    else:
        lines = [zip(xy[BL[i, :], 0], xy[BL[i, :], 1]) for i in range(len(BL))]

    if isinstance(climv, tuple):
        vmin = climv[0]
        vmax = climv[1]
    else:
        vmin = -climv
        vmax = climv

    # If the colormap is not already registered, register it here
    if not colormap in plt.colormaps():
        import lepm.plotting.colormaps as lecmaps
        lecmaps.register_colormaps(colormap)

    line_segments = LineCollection(lines,  # Make a sequence of x,y pairs
                                   linewidths=lw,  # could iterate over list
                                   linestyles='solid',
                                   cmap=colormap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
    if bs is not 'none':
        line_segments.set_array(bs)

    ax.add_collection(line_segments)
    ax.set_axis_bgcolor(bgcolor)  # [214./255.,214./255.,214./255.] ) #'#E8E8E8')

    # POLYGONS
    if colorpoly:
        # Color the polygons based on # sides
        print 'Extracting polygons from lattice...'
        print 'NL = ', NL
        polygons = extract_polygons_lattice(xy, BL, NL, KL, viewmethod=viewmethod, PVx=PVx, PVy=PVy,
                                            PVxydict=PVxydict)
        PolyPC = polygons2PPC(polygons)
        ax = plt.gca()
        # number of polygon sides
        Pno = np.array([len(polyg) for polyg in polygons], dtype=int)
        print 'Pno = ', Pno
        print 'medPno = ', np.floor(np.median(Pno))
        medPno = np.floor(np.median(Pno))
        uIND = np.where(Pno == medPno - 1)[0]
        mIND = np.where(Pno == medPno)[0]
        oIND = np.where(Pno == medPno + 1)[0]
        loIND = np.where(Pno < medPno - 1.5)[0]
        hiIND = np.where(Pno > medPno + 1.5)[0]
        print ' uIND = ', uIND
        print ' oIND = ', oIND
        print ' loIND = ', loIND
        print ' hiIND = ', hiIND
        if len(uIND) > 0:
            PPCu = [PolyPC[i] for i in uIND]
            pu = PatchCollection(PPCu, color='b', alpha=0.5)
            ax.add_collection(pu)
        if len(mIND) > 0:
            PPCm = [PolyPC[i] for i in mIND]
            pm = PatchCollection(PPCm, color=[0.5, 0.5, 0.5], alpha=0.5)
            ax.add_collection(pm)
        if len(oIND) > 0:
            PPCo = [PolyPC[i] for i in oIND]
            po = PatchCollection(PPCo, color='r', alpha=0.5)
            ax.add_collection(po)
        if len(loIND) > 0:
            PPClo = [PolyPC[i] for i in loIND]
            plo = PatchCollection(PPClo, color='k', alpha=0.5)
            ax.add_collection(plo)
        if len(hiIND) > 0:
            PPChi = [PolyPC[i] for i in hiIND]
            phi = PatchCollection(PPChi, color='g', alpha=0.5)
            ax.add_collection(phi)

    if not bs == 'none':
        if len(np.nonzero(bs)) == 0:
            axcb = plt.colorbar(line_segments)
            axcb.set_label('Strain')
            axcb.set_clim(vmin=vmin, vmax=vmax)

    # set limits
    if isinstance(xlimv, tuple):
        ax.set_xlim(xlimv)
    elif xlimv:
        # print 'setting xlimv here...'
        ax.set_xlim(-xlimv, xlimv)
    else:
        # print 'setting auto limits...'
        # print 'max(xy[:,0]) = ', max(xy[:,0])
        # print 'min(xy[:,0]) = ', min(xy[:,0])
        extent = max(xy[:, 0]) - min(xy[:, 0])
        # print 'extent = ', extent
        ax.set_xlim(min(xy[:, 0]) - extent * 0.1, max(xy[:, 0]) + extent * 0.1)

    if isinstance(ylimv, tuple):
        ax.set_ylim(ylimv)
    elif ylimv:
        ax.set_ylim(-ylimv, ylimv)
    else:
        extent = max(xy[:, 1]) - min(xy[:, 1])
        ax.set_ylim(min(xy[:, 1]) - extent * 0.1, max(xy[:, 1]) + extent * 0.1)

    ax.axis('equal')

    if len(BLNNN) > 0:
        # Efficiently plot many lines in a single set of axes using LineCollection
        if (BLNNN < 0).any():
            raise RuntimeError('Finish getting clockwise and counterclockwise NNN BL segments here. ' +
                               'Also need to allow for periodic bonds in this part!')
            # Get rows where BL is clockwise (positive definite) <-- this is not right yet
            BLNNNblue = BLNNN[np.where(BLNNN > 0.5)]
            blines = [zip(xy[BLNNN[i, :], 0], xy[BLNNN[i, :], 1]) for i in range(len(BLNNN))]
            blinesNNN = LineCollection(blines,  # Make a sequence of x,y pairs
                                       linewidths=lw,  # could iterate over list
                                       linestyles='dashed',
                                       color='blue',
                                       zorder=100)
            ax.add_collection(linesNNN)
            # Get rows where BL is counter clockwise (negative definite) <-- this is not right yet
            BLNNNred = BLNNN[np.where(BLNNN) < 0.5]  # just want the rows....
            rlines = [zip(xy[BLNNNred[i, :], 0], xy[BLNNNred[i, :], 1]) for i in range(len(BLNNNred))]
            rlinesNNN = LineCollection(rlines,  # Make a sequence of x,y pairs
                                       linewidths=lw,  # could iterate over list
                                       linestyles='dashed',
                                       color='red',
                                       zorder=100)
            ax.add_collection(rlinesNNN)
        else:
            lines = [zip(xy[BLNNN[i, :], 0], xy[BLNNN[i, :], 1]) for i in range(len(BLNNN))]
            linesNNN = LineCollection(lines,  # Make a sequence of x,y pairs
                                      linewidths=lw,  # could iterate over list
                                      linestyles='dashed',
                                      color='blue',
                                      zorder=100)
            ax.add_collection(linesNNN)

    if len(NLNNN) > 0 and len(KLNNN) > 0:
        if (BL < 0).any():
            print 'plotting periodic NNN...'
            for i in range(NP):
                todo = np.where(KLNNN[i, :] > 1e-12)[0]
                for ind in NLNNN[i, todo]:
                    ax.arrow(xy[i, 0], xy[i, 1], (xy[ind, 0] - xy[i, 0]) * 0.98, (xy[ind, 1] - xy[i, 1]) * 0.98,
                             head_width=0.1, head_length=0.2, fc='b', ec='b')
        else:
            # amount to offset clockwise nnn arrows
            for i in range(NP):
                todo = np.where(KLNNN[i, :] > 1e-12)[0]
                for ind in NLNNN[i, todo]:
                    offset = (xy[ind, :] - xy[i, :]) * 0.5
                    ax.arrow(xy[i, 0] + offset[0], xy[i, 1] + offset[1],
                             (xy[ind, 0] - xy[i, 0]) * 0.3, (xy[ind, 1] - xy[i, 1]) * 0.3,
                             head_width=0.1, head_length=0.2, fc='b', ec='b')

                todo = np.where(KLNNN[i, :] < -1e-12)[0]
                for ind in NLNNN[i, todo]:
                    offset = (xy[ind, :] - xy[i, :]) * 0.5
                    ax.arrow(xy[i, 0] + offset[0], xy[i, 1] + offset[1], (xy[ind, 0] - xy[i, 0]) * 0.3,
                             (xy[ind, 1] - xy[i, 1]) * 0.3,
                             head_width=0.1, head_length=0.2, fc='r', ec='r')

    if labelinds:
        for i in range(NP):
            ax.text(xy[i, 0] + 0.1, xy[i, 1], str(i))

    ax.set_title(title)

    print '...ending the display lattice function '
    if axis_off:
        ax.axis('off')
    if close:
        plt.show()
        return None
    else:
        return ax


##########################################
# Saving data
##########################################
def write_initdata(xy0, v0, NL, BND, h, beta, outdir):
    """Writes initialization variables (xy0, NL and h) for the lattice to outdir.

    Parameters
    ----------
    xy0 : array of dimension NP x nd
        Initial positions of 2D lattice
    v0 : array of dimension NP x nd
        Initial velocities of the points given by xy
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    h : float
        The time step of the simulation
    beta : float
        Dissipation damping constant
    BND : array of dimension NP x 1
        Boolean for Boundary particle identification
    outdir : string
        The output directory for the initial data
    """
    dio.ensure_dir(outdir)
    M = np.hstack((xy0, v0))
    np.savetxt(outdir + 'NL.txt', NL, fmt='%i', delimiter=',', header='NL (Neighbor List)')
    np.savetxt(outdir + 'BND.txt', BND, fmt='%i', header='BND (Boundary List)')
    np.savetxt(outdir + 'xyv0.txt', M, delimiter=',', header='xy0 (initial positions) v0 (initial velocities)')
    with open(outdir + 'h.txt', "w") as hfile:
        hfile.write("# h (time step) \n{0:4f}".format(h))
    if beta != 'none':
        with open(outdir + 'beta.txt', "w") as betafile:
            betafile.write("# beta (damping coeff) \n{0:4f}".format(beta))


def write_initparams(params, outdir, padding_var=7, paramsfn='parameters', skiplat=False, skipglat=False):
    """Writes initialization variables (xy0, NL, h, etc) for the lattice to outdir.
    Initialization variables must be stored in dictionary 'params'.

    Parameters
    ----------
    params : dict
        dictionary containing initial parameters for the run, with keys as strings; must include xy0, v0, NL.
    outdir : string
        The output directory for the initial data
    padding_var : int
        how much white space to leave between key names column and values column
    paramsfn : string
        file name with which to save txt file
    """
    paramfile = outdir + paramsfn + '.txt'
    with open(paramfile, 'w') as myfile:
        myfile.write('# Parameters\n')

    dio.ensure_dir(outdir)
    for key in params:
        if key == 'reg1' or key == 'reg2' or key == 'reg3':
            np.savetxt(outdir + key + '.txt', params[key], fmt='%d', delimiter=',', header=key + ' particle IDs')
        if key == 'xyv0':
            np.savetxt(outdir + 'xyv0.txt', params['xyv0'], delimiter=',',
                       header='xy0 (initial positions) v0 (initial velocities)')
        elif key == 'xy':
            if not skiplat:
                np.savetxt(outdir + 'xy.txt', params['xy'], delimiter=',',
                           header='xy0 (undeformed lattice positions from mesh)')
        elif key == 'KL':
            if not skiplat:
                np.savetxt(outdir + 'KL.txt', params['KL'], fmt='%i', delimiter=',',
                           header='KL (Bond Connectivity List)')
        elif key == 'NL':
            if not skiplat:
                np.savetxt(outdir + 'NL.txt', params['NL'], fmt='%i', delimiter=',', header='NL (Neighbor List)')
        elif key == 'BND':
            np.savetxt(outdir + 'BND.txt', params['BND'], fmt='%i', header='BND (Boundary List)')
        elif key == 'OmK':
            if not skipglat:
                np.savetxt(outdir + 'OmK.txt', params['OmK'], fmt='%f', delimiter=',',
                           header='OmK (spring frequency array, for Nash limit: (-1)^(c+b)kl^2/Iw')
        elif key == 'OmG':
            if not skipglat:
                np.savetxt(outdir + 'Omg.txt', params['OmG'], fmt='%f', delimiter=',',
                           header='Omg (gravitational frequency array, for Nash limit: (-1)^(c+1)mgl/Iw')
        elif key == 'LVUC':
            if not skiplat:
                np.savetxt(outdir + 'LVUC.txt', params['LVUC'], fmt='%i', delimiter=',',
                           header='Lattice Vector and Unit cell vector coordinates')
        else:
            with open(paramfile, 'a') as myfile:
                # print 'Writing param ', str(key)
                # print ' with value ', str(params[key])
                # print ' This param is of type ', type(params[key])

                if isinstance(params[key], str):
                    myfile.write('{{0: <{}}}'.format(padding_var).format(key) + \
                                 '= ' + params[key] + '\n')
                elif isinstance(params[key], np.ndarray):
                    # print params[key].dtype
                    if key == 'BIND':
                        print 'BIND = ', str(params[key]).replace('\n', '')

                    myfile.write('{{0: <{}}}'.format(padding_var).format(key) + \
                                 '= ' + ", ".join(np.array_str(params[key]).split()).replace('[,', '[') + '\n')
                    # if params[key].dtype == 'float64':
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ np.array_str(params[key]).replace('\n','').replace('  ',',') +'\n')
                    # elif params[key].dtype == 'int32':
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ str(params[key]).replace('\n','').replace(' ',',') +'\n')
                    # else:
                    #    myfile.write('{{0: <{}}}'.format(padding_var).format(key)+\
                    #             '= '+ str(params[key]).replace('\n','').replace(' ',',') +'\n')
                elif isinstance(params[key], list):
                    myfile.write('{{0: <{}}}'.format(padding_var).format(key) + \
                                 '= ' + str(params[key]) + '\n')
                else:
                    # print key, ' = ', params[key]
                    myfile.write('{{0: <{}}}'.format(padding_var).format(key) + \
                                 '= ' + '{0:.12e}'.format(params[key]) + '\n')

                    # elif key == 'LV':
                    #     np.savetxt(outdir+'LV.txt',params['LV'], fmt='%18e',delimiter=',', header='Lattice Vector coordinates')
                    # elif key == 'UC':
                    #     np.savetxt(outdir+'UC.txt',params['UC'], fmt='%18e',delimiter=',', header='Unit cell vector coordinates')
                    #
                    # elif key == 'h':
                    #    with open(outdir+'h.txt', "w") as hfile:
                    #        hfile.write("# h (time step) \n{0:5e}".format(h) )
                    # elif key == 'beta':
                    #    with open(outdir+'beta.txt', "w") as betafile:
                    #        betafile.write("# beta (damping coeff) \n{0:5e}".format(beta) )


def load_evaled_dict(datadir, filename):
    """Load a dictionary from a txt file , for keys of the dictionary that are evaluated rather than saved as strings
    (as would be done in a parameters file).

    Parameters
    ----------
    datadir: str
        the directory in which to find the txt file to be loaded
    filename: str
        the name of the file. If this string does not include the extension '.txt', it is added

    Returns
    -------
    ddict: dict
    """
    ddict = {}
    datadir = dio.prepdir(datadir)
    if filename[-4:] != '.txt':
        filename += '.txt'
    with open(datadir + filename) as f:
        for line in f:
            if '# ' not in line:
                (k, val) = line.split('=')
                key = k.strip()
                # print 'key = ', key
                # print 'val = ', val
                if key == 'date':
                    val = val[:-1].strip()
                    print '\nloading params for: date= ', val
                elif sf.is_number(val):
                    # val is a number, so convert to a float
                    val = float(val[:-1].strip())
                else:
                    '''This should handle tuples without a problem'''
                    try:
                        # If val has both [] and , --> then it is a numpy array
                        # (This is a convention choice.)
                        if '[' in val and ',' in val:
                            make_ndarray = True

                        # val might be a list, so interpret it as such using ast
                        # val = ast.literal_eval(val.strip())
                        exec ('val = %s' % (val.strip()))

                        # Make array if found '[' and ','
                        if make_ndarray:
                            val = np.array(val)

                    except:
                        # print 'type(val) = ', type(val)
                        # val must be a string
                        try:
                            # val might be a list of strings?
                            val = val[:-1].strip()
                        except:
                            '''val is a list with a single number'''
                            val = val

                ddict[eval(key)] = val
                # print val
    return ddict


def load_params(outdir, paramsfn='parameters', ignore=None):
    """Load params (dictionary) from parameters.txt file in outdir.

    Parameters
    ----------
    outdir: str
        The path to the dictionary txt file. If ends in '.txt', then we will split outdir into outdir and paramsfn
        accordingly. Thus, if paramsfn is supplied, outdir cannot be a (directory) string ending with '.txt'
    paramsfn: str
        The name of the txt file with the dictionary's key-value pairs to load, if outdir supplied is not the full path
    ignore : list of str
        keys to ignore in the loading of the parameters. For example, when loading a lattice_parameters.txt file for
        creating a network, we want to ignore any physics, like interactions, pinning strength, etc. So we'd have
        ignore = ['VO_pin_gauss', 'pin', 'Omg', etc]

    Returns
    -------
    params: dict
        A dictionary, with all datatypes preserved as best as possible from reading in txt file

    """
    params = {}
    # If outdir is the entire path, it has .txt at end, and so use this to split it up into dir and filename
    if outdir[-4:] == '.txt':
        outsplit = outdir.split('/')
        outdir = ''
        for tmp in outsplit[:-1]:
            outdir += tmp + '/'
        paramsfn = outsplit[-1]

    if '*' in outdir:
        print 'outdir specified with wildcard, searching and taking first result...'
        outdir = glob.glob(outdir)[0]

    outdir = dio.prepdir(outdir)
    if paramsfn[-4:] != '.txt':
        paramsfn += '.txt'
    with open(outdir + paramsfn) as f:
        # for line in f:
        #     print line
        for line in f:
            if '# ' not in line:
                (k, val) = line.split('=')
                key = k.strip()
                if key == 'date':
                    val = val[:-1].strip()
                    print '\nloading params for: date= ', val
                elif sf.is_number(val):
                    # val is a number, so convert to a float
                    val = float(val[:-1].strip())
                else:
                    '''This should handle tuples without a problem'''
                    try:
                        # If val has both [] and , --> then it is a numpy array
                        # (This is a convention choice.)
                        if '[' in val and ',' in val:
                            make_ndarray = True

                        # val might be a list, so interpret it as such using ast
                        # val = ast.literal_eval(val.strip())
                        exec ('val = %s' % (val.strip()))

                        # Make array if found '[' and ','
                        if make_ndarray:
                            val = np.array(val)

                    except:
                        # print 'type(val) = ', type(val)
                        # val must be a string
                        try:
                            # val might be a list of strings?
                            val = val[:-1].strip()
                        except:
                            '''val is a list with a single number'''
                            val = val
                if ignore is None:
                    params[key] = val
                elif key not in ignore:
                    params[key] = val

                # print val

    return params


def write_data(xy, v, KL, iteration, h, outdir, fname):
    """Writes xy and KL data for the lattice to outdir.

    Parameters
    ----------
    xy : array of dimension NP x nd
        Current (for example: x,y or x,y,z or x,y,theta,phi,psi)
    v : array of dimension NP x nd or empty numpy array
        velocities of the points given by xy--> if empty, only saves position data
    KL :  array of dimension #pts x (max number of neighbors)
        Spring constant list, where 1 corresponds to a true connection while 0 signifies that there is not a connection.
    iteration : int
        The iteration of the simulation (current time/time step)
    outdir : string
        The output directory for the data (contains subdirs for xyv, KL)
    fname : string
        Description of the simulation being run (base name for files saved)
    """
    if v.size > 0:
        M = np.hstack((xy, v))
        xyvdir = outdir + 'xyv/'
    else:
        M = xy
        xyvdir = outdir + 'xy/'

    itstr = '%08d' % iteration
    dio.ensure_dir(xyvdir)
    dio.ensure_dir(outdir + 'KL/')
    if np.shape(M)[1] == 2:
        # data is 2D, just positions
        np.savetxt(xyvdir + fname + '_xy_' + itstr + '.txt', M, fmt='%.18e', delimiter=',',
                   header='x,y (t=' + str(h * iteration) + ')')
    elif np.shape(M)[1] == 4:
        # data is 2D with velocities
        np.savetxt(xyvdir + fname + '_xyv_' + itstr + '.txt', M, fmt='%.18e', delimiter=',',
                   header='x,y,vx,vy (t=' + str(h * iteration) + ')')
    elif np.shape(M)[1] == 6:
        # data is 3D with velocities
        np.savetxt(xyvdir + fname + '_xyv_' + itstr + '.txt', M, fmt='%.18e', delimiter=',',
                   header='x,y,z,vx,vy,vz (t=' + str(h * iteration) + ')')
    elif np.shape(M)[1] == 8:
        # data is 2D with moving rest/pivor positions and displacements
        np.savetxt(xyvdir + fname + '_xyv_' + itstr + '.txt', M, fmt='%.18e', delimiter=',',
                   header='X,Y,dX,dY,vX,vY,vdX,vdY (t=' + str(h * iteration) + ')')
    elif np.shape(M)[1] == 10:
        # data is 2D plus euler angles
        np.savetxt(xyvdir + fname + '_xyv_' + itstr + '.txt', M, fmt='%.18e', delimiter=',',
                   header='x,y,theta,phi,psi,vx,vy,vtheta,vphi,vpsi (t=' + str(h * iteration) + ')')

    if KL.size > 0:
        '''KL may be changing for timestep to timestep, so save it for each one'''
        np.savetxt(outdir + 'KL/' + fname + '_KL_' + itstr + '.txt', KL, fmt='%.18e', delimiter=',',
                   header='KL (t=' + str(h * iteration) + ')')
    else:
        '''KL is static. Refer to KL.txt saved in the datadir'''
        pass


def write_xy0(xy0, iteration, hh, outdir, fname):
    """Writes reference lattice/rest positions (xy0) to outdir.

    Parameters
    ----------
    xy0 : array of dimension NP x nd
        Current (for example: x,y or x,y,z or x,y,theta,phi,psi)
    iteration : int
        The iteration of the simulation (current time/time step)
    outdir : string
        The output directory for the data (contains subdirs for xyv, KL)
    fname : string
        Description of the simulation being run (base name for files saved)
    """
    xy0dir = outdir + 'xy0/'
    itstr = '%08d' % iteration
    dio.ensure_dir(xy0dir)
    np.savetxt(xy0dir + fname + '_xy0_' + itstr + '.txt', xy0, fmt='%.18e', delimiter=',',
               header='x0,y0 (t=' + str(hh * iteration) + ')')


##########################################
# Files, Folders, and Directory Structure
##########################################
def build_meshfn(lp):
    """Build the path string where a lattice/network would be saved or should be saved. This does not actually create
    the path on the hard disk

    See Also
    --------
    find_meshfn(lattice_params)

    Parameters
    ----------
    lp : dict
        The lattice parameters dictionary, with all keys needed for specifying path (these params vary depending on the
        value of the LatticeTop key).

    Returns
    -------
    ffind : str
        The path to where the lattice should be stored or would be stored
    """
    # Place values assoc with keys of lattice_params as their defaults if not specified
    LatticeTop = lp['LatticeTop']
    shape = lp['shape']
    rootdir = lp['rootdir']
    NH = lp['NH']
    NV = lp['NV']
    if 'cutLstr' in lp:
        cutLstr = lp['cutLstr']
    else:
        cutLstr = ''

    if 'delta_lattice' in lp:
        delta_lattice = lp['delta_lattice']
    elif 'delta' in lp:
        delta_lattice = '{0:0.3f}'.format(lp['delta'] / np.pi).replace('.', 'p')
    else:
        delta_lattice = ''

    if 'phi_lattice' in lp:
        phi_lattice = lp['phi_lattice'].replace('.', 'p')
    else:
        if 'phi' in lp:
            phi_lattice = '{0:0.3f}'.format(lp['phi']).replace('.', 'p')
        else:
            phi_lattice = '0p000'

    if 'theta_lattice' in lp:
        theta_lattice = lp['theta_lattice']
    else:
        theta_lattice = ''
    if 'eta' in lp:
        eta = lp['eta']
    else:
        eta = ''
    if 'huID' in lp:
        huID = lp['huID']
    elif 'conf' in lp:
        huID = '{0:02d}'.format(int(lp['conf']))
    else:
        huID = '01'
    if 'zkagome' in lp:
        zkagome = lp['zkagome']
    else:
        zkagome = -1
    if 'z' in lp:
        z = str(lp['z'])
    else:
        z = -1
    if 'origin' in lp:
        print 'lp[origin] = ', lp['origin']
        print "(np.abs(lp['origin']) > 1e-7) = ", (np.abs(lp['origin']) > 1e-7)
        if (np.abs(lp['origin']) > 1e-7).any():
            originstr = '_originX' + '{0:0.2f}'.format(lp['origin'][0]).replace('.', 'p') + \
                        'Y' + '{0:0.2f}'.format(lp['origin'][1]).replace('.', 'p')
        else:
            originstr = ''
    else:
        originstr = ''

    if 'periodic_strip' not in lp:
        lp['periodic_strip'] = False

    print '\n\n\noriginstr = ', originstr
    print 'Searching for ' + LatticeTop + ' lattice...'

    # make sure rootdir ends with /
    rootdir = dio.prepdir(rootdir)
    ########################################################################################
    ########################################################################################
    print 'LatticeTop =', LatticeTop
    if LatticeTop == 'square':
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodicBC'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        etatheta_str = etastr + thetastr
        print '\n\n', etatheta_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                etatheta_str + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['hexagonal', 'hexmeanfield']:
        print '... forming hexagonal meshfn...'
        print 'le: again, lp[periodic_strip] = ', lp['periodic_strip']
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodicBC'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr + thetastr
        print '\n\n', delta_phi_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                delta_phi_str + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'hexannulus':
        # correct NV if it equals NH --> this would never be possible, and so if NV isn't specified (ie N=NH=NV is
        #  specified), then use alph to determine the thickness of the annulus
        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr
        alphstr = '_alph{0:0.2f}'.format(lp['alph']).replace('.', 'p')
        print '\n\n', delta_phi_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_circle' + \
                delta_phi_str + alphstr + '_' + '{0:06d}'.format(NH) + '_x_' + '*' + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'selregion' in LatticeTop:
        # Assume there is only one instance of this selregion LatticeTop with a given NP size
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '*NP{0:06d}'.format(lp['NP_load']) + '_xy.txt'
    elif LatticeTop == 'triangular':
        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta)

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        extrastr = etastr + thetastr
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + extrastr + '_' + \
                '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'jammed' or LatticeTop == 'isostatic':
        if lp['periodicBC']:
            if LatticeTop == 'jammed':
                periodicstr = '_periodicBC'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if lp['source'] == 'ulrich':
            hustr = '_homog_z' + '{0:0.03f}'.format(lp['target_z']) + '_conf' + huID + '_zmethod' + lp['cutz_method']
        elif lp['source'] == 'hexner':
            if lp['NP_load'] > 0:
                hustr = periodicstr + '_hexner' + '_z*_conf' + huID + '_zmethod' + lp['cutz_method']
            else:
                print '---> here <----'
                if float(z) > 0:
                    zstr = '{0:0.03f}'.format(float(z))
                else:
                    zstr = '*'

                hustr = '_hexner' + periodicstr + '_z' + zstr + '_conf' + huID + '_zmethod' + lp['cutz_method']
        if lp['NP_load'] > 0:
            print '{0:06d}'.format(lp['NP_load'])
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + '_NP' + \
                    '{0:06d}'.format(lp['NP_load']) + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + '_' + \
                    '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'deformed_kagome' or LatticeTop == 'deformed_martini':
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if np.abs(lp['theta']) > 1e-9:
            thetastr = '_theta{0:0.3f}'.format(np.round(lp['theta'] * 1000) * 0.001).replace('.', 'p')
        else:
            thetastr = ''

        paramstr = '_x1_' + '{0:0.4f}'.format(lp['x1']).replace('.', 'p').replace('-', 'n') + \
                   '_x2_' + '{0:0.4f}'.format(lp['x2']).replace('.', 'p').replace('-', 'n') + \
                   '_x3_' + '{0:0.4f}'.format(lp['x3']).replace('.', 'p').replace('-', 'n') + \
                   '_z_' + '{0:0.4f}'.format(lp['z']).replace('.', 'p').replace('-', 'n')
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr +\
                thetastr + paramstr + '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'twisted_kagome':
        paramstr = '_alph_' + '{0:0.4f}'.format(lp['alph'])
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + paramstr + '_{0:06d}'.format(
            NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'hyperuniform' in LatticeTop:
        # hyperuniform ID string
        hustr = '_d' + huID + '_z{0:0.3f}'.format(lp['target_z']).replace('.', 'p').replace('-', 'n')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''

        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + '_' + \
                    '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['hucentroid', 'huvoronoi']:
        # hyperuniform ID string
        hustr = '_d' + huID
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
                stripnhnv = '_NH{0:06d}'.format(lp['NH']) + '_NV{0:06d}'.format(lp['NV'])
            else:
                periodicstr = '_periodic'
                stripnhnv = ''
        else:
            periodicstr = ''
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + stripnhnv + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['kagome_hucent', 'kagome_huvor']:
        # hyperuniform ID string
        hustr = '_d' + huID
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_periodic' + hustr \
                    + '_NP{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + \
                    '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    # elif 'isostatic' in LatticeTop :
    #    # hyperuniform ID string --> isostatic ID string
    #    hustr = '_homog_zindex001'+'_conf'+huID
    #    ffind = rootdir+'networks/'+LatticeTop+'/'+LatticeTop+'_'+shape+hustr+'_'+'{0:06d}'.format(NH)+'_x_'+
    #            '{0:06d}'.format(NV)+cutLstr+'_xy.txt'
    #    print 'searching for ', ffind
    elif LatticeTop in ['iscentroid', 'isvoronoi']:
        # isostatic ID string
        if lp['NP_load'] > 0:
            hustr = '_hexner_size' + str(lp['NP_load']) + '_conf' + huID
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_periodic' + hustr + \
                    '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            if lp['source'] == 'ulrich':
                hustr = '_homog_zindex001' + '_conf' + huID
            elif lp['source'] == 'hexner':
                if NH > 10 or NV > 10:
                    hustr = '_hexner_size8192_conf' + huID
                else:
                    hustr = '_hexner_size0512_conf' + huID
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + '_' + '{0:06d}'.format(
                NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['kagome_isocent', 'kagome_isovor']:
        # isostatic ID string
        if lp['source'] == 'ulrich':
            hustr = '_ulrich_homog_zindex001' + '_conf' + huID
        elif lp['source'] == 'hexner':
            if lp['periodicBC'] and lp['NP_load'] > 0:
                hustr = '_hexner_size' + str(lp['NP_load']) + '_conf' + huID
            elif NH > 13 or NV > 13:
                hustr = '_hexner_size8192_conf' + huID
            else:
                hustr = '_hexner_size0512_conf' + huID
        if lp['periodicBC'] and lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_periodic' + hustr + \
                    '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['hucentroid_annulus', 'kagome_hucent_annulus']:
        # hyperuniform ID string
        lp['shape'] = 'annulus'
        shape = lp['shape']
        hustr = '_d' + huID
        if lp['periodicBC'] or lp['periodic_strip']:
            raise RuntimeError('Network is labeled as periodic but is also an annulus.')

        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + hustr + \
                '_alph' + sf.float2pstr(lp['alph']) + \
                originstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'linear':
        etastr = '{0:.3f}'.format(lp['eta']).replace('.', 'p')
        thetastr = '{0:.3f}'.format(lp['theta']).replace('.', 'p')
        if lp['periodicBC']:
            periodicstr = '_periodic'
        else:
            periodicstr = ''
        exten = periodicstr + '_line_theta' + thetastr + 'pi_eta' + etastr
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + exten + '_{0:06d}'.format(NH) + \
                '_x_' + '{0:06d}'.format(1) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'circlebonds':
        # circle of particles connected in a periodic line
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_{0:06d}'.format(NH) + \
                '_x_' + '{0:06d}'.format(1) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'dislocated':
        Ndefects = str(lp['Ndefects'])
        Bvec = lp['Bvec']
        dislocxy = lp['dislocxy']  # specifies the position of a single defect, if not centered, as tuple of strings
        if dislocxy == 'none':
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_Ndefects' + Ndefects + \
                    '_Bvec' + Bvec + '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_Ndefects' + Ndefects + \
                    '_Bvec' + Bvec + '_dislocxy_' + str(dislocxy[0]) + '_' + str(dislocxy[1]) + '_{0:06d}'.format(NH) + \
                    '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'dislocatedRand':
        Ndefects = str(lp['Ndefects'])
        Bvec = lp['Bvec']
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_Ndefects' + Ndefects + \
                '_Bvec' + Bvec + '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'triangularz':
        zmethodstr = lp['cutz_method']
        zstr = str(lp['z'])
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_zmethod' + zmethodstr + \
                '_z' + zstr + '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'penroserhombTri':
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            perstr = ''
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + perstr + '_' + shape + \
                '_div*_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'penroserhombTricent':
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            perstr = ''
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + perstr + '_' + shape + \
                '_div*_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'kagome_penroserhombTricent':
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_div*_{0:06d}'.format(
            NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'random_organization_gamma' in LatticeTop:
        hustr = '_d' + huID
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + '_' + \
                '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'kagper_hucent':
        print '\n\n sub-realization number (for given hu realization, which decoration?): lp[subconf] = ', lp[
            'subconf'], '\n'
        # hyperuniform ID string
        hustr = '_d' + huID
        perdstr = '_perd' + '{0:0.2f}'.format(lp['percolation_density']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + perdstr + \
                    '_r' + '{0:02d}'.format(int(lp['subconf'])) + \
                    '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + perdstr + \
                    '_r' + '{0:02d}'.format(int(lp['subconf'])) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['hucent_kagframe', 'kaghu_centframe', 'hucent_kagcframe']:
        # hyperuniform ID string
        hustr = '_d' + huID
        alphstr = '_alph' + '{0:0.2f}'.format(lp['alph']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + alphstr + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + alphstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(
                NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['isocent_kagframe', 'isocent_kagcframe']:
        # isostatic ID string
        if lp['source'] == 'ulrich':
            hustr = '_ulrich_homog_zindex001' + '_conf' + huID
        elif lp['source'] == 'hexner':
            if lp['periodicBC'] and lp['NP_load'] > 0:
                hustr = '_hexner_size' + str(lp['NP_load']) + '_conf' + huID
            elif NH > 80 or NV > 80:
                hustr = '_hexner_size128000_conf' + huID
            elif NH > 15 or NV > 15:
                hustr = '_hexner_size8192_conf' + huID
            else:
                hustr = '_hexner_size0512_conf' + huID
        perdstr = '_alph' + '{0:0.2f}'.format(lp['alph']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + perdstr + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + perdstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(
                NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'hex_kagframe' or LatticeTop == 'hex_kagcframe':
        alphstr = '_alph' + '{0:0.2f}'.format(lp['alph']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')
            if 'eta_alph' not in lp:
                print 'did not find eta_alph in lp, using alph value as eta_alph...'
                lp['eta_alph'] = lp['alph']

            etastr += '_etaalph' + sf.float2pstr(lp['eta_alph'], ndigits=3)

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + \
                        phi_lattice.replace('.', 'p') + thetastr

        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + delta_phi_str + \
                    originstr + alphstr + etastr + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + delta_phi_str + \
                    originstr + alphstr + etastr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + \
                    cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'kagsplit_hex':
        alphstr = '_alph' + '{0:0.2f}'.format(lp['alph']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodicBC'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta)
        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr + thetastr
        print '\n\n', delta_phi_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + delta_phi_str + \
                alphstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop == 'kagper_hex':
        perdstr = '_perd' + '{0:0.2f}'.format(lp['percolation_density']).replace('.', 'p')
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodicBC'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta)

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        delta_phi_str = '_delta' + delta_lattice + '_phi' + phi_lattice + etastr + thetastr
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                    delta_phi_str + perdstr + \
                    '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                    delta_phi_str + perdstr + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['randomcent', 'kagome_randomcent']:
        if lp['periodicBC']:
            if lp['periodic_strip']:
                perstr = '_periodicstrip'
            else:
                perstr = '_periodic'
        else:
            perstr = ''
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + perstr + '_r' + \
                '{0:02d}'.format(int(lp['conf'])) + \
                '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['randomspreadcent', 'kagome_randomspreadcent']:
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            perstr = ''
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + perstr + '_r' + \
                '{0:02d}'.format(int(lp['conf'])) + \
                '_spreadt{0:0.3f}'.format(lp['spreading_time']).replace('.', 'p') + \
                '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['uofc_hucent', 'uofc_kaglow_hucent', 'uofc_kaghi_hucent',
                        'kaghi_hucent_curvys', 'kaglow_hucent_curvys']:
        hustr = '_d' + huID
        if 'thres' not in lp:
            lp['thres'] = 1.0

        if 'curvys' in LatticeTop:
            aratiostr = '_aratio{0:0.3f}'.format(lp['aratio']).replace('.', 'p')
        else:
            aratiostr = ''

        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + \
                '_thres' + sf.float2pstr(lp['thres'], ndigits=1) + aratiostr + \
                '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif LatticeTop in ['uofc_isocent', 'uofc_kaglow_isocent', 'uofc_kaghi_isocent', 'chicago_kaglow_isocent',
                        'chicago_kaghi_isocent', 'kaghi_isocent_chern', 'kaghi_hucent_chern',
                        'csmf_kaghi_isocent', 'kaghi_isocent_thanks',
                        'kaghi_isocent_curvys', 'kaglow_isocent_curvys']:
        if lp['source'] == 'ulrich':
            hustr = '_ulrich_homog_zindex001' + '_conf' + huID
        elif lp['source'] == 'hexner':
            if lp['periodicBC'] and lp['NP_load'] > 0:
                hustr = '_hexner_size' + str(lp['NP_load']) + '_conf' + huID
            elif NH > 80.5 or NV > 80.5:
                hustr = '_hexner_size128000_conf' + huID
            elif NH > 9 or NV > 9:
                hustr = '_hexner_size8192_conf' + huID
            else:
                hustr = '_hexner_size0512_conf' + huID

        if 'curvys' in LatticeTop:
            aratiostr = '_aratio{0:0.3f}'.format(lp['aratio']).replace('.', 'p')
        else:
            aratiostr = ''

        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        if 'thres' not in lp:
            lp['thres'] = 1.0

        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + \
                    '_thres' + sf.float2pstr(lp['thres'], ndigits=1) + aratiostr + \
                    '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + hustr + \
                    originstr + \
                    '_thres' + sf.float2pstr(lp['thres'], ndigits=1) + aratiostr + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'kaghi_randorg_gammakick' in LatticeTop and 'cent_curvys' in LatticeTop:
        # kaghi_randorg_gammakick1p60_cent_curvys
        # For cover optios in Nature Physics paper
        aratiostr = '_aratio{0:0.3f}'.format(lp['aratio']).replace('.', 'p')
        kickszstr = '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=3)
        spreadtstr = '_spreadt' + sf.float2pstr(lp['spreading_time'], ndigits=3)
        dtstr = '_dt' + sf.float2pstr(lp['spreading_dt'], ndigits=3)
        # for ensuring that no points are too close
        # alphstr =
        if lp['periodic_strip']:
            lp['NP_load'] = lp['NH'] * lp['NV']
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_periodicstrip' + kickszstr + spreadtstr + dtstr +  \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + \
                    aratiostr + '_NP{0:06d}'.format(lp['NP_load']) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + '_xy.txt'
        elif lp['periodicBC']:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_periodic' + kickszstr + spreadtstr + dtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + aratiostr + \
                    '_NP{0:06d}'.format(lp['NP_load']) + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + kickszstr + spreadtstr + dtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + \
                    aratiostr + '_NP{0:06d}'.format(lp['NP_load']) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'

        print 'searching for ', ffind
    elif LatticeTop == 'kagome':
        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr + thetastr

        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodic'
        else:
            periodicstr = ''
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                delta_phi_str + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'randorg_gammakick' in LatticeTop:
        kickszstr = '_kicksz' + sf.float2pstr(lp['kicksz'], ndigits=3)
        spreadtstr = '_spreadt' + sf.float2pstr(lp['spreading_time'], ndigits=3)
        if lp['periodic_strip']:
            lp['NP_load'] = lp['NH'] * lp['NV']
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_periodicstrip' + kickszstr + spreadtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + '_NP{0:06d}'.format(lp['NP_load']) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + '_xy.txt'
        elif lp['periodicBC']:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_periodic' + kickszstr + spreadtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + '_NP{0:06d}'.format(lp['NP_load']) + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + kickszstr + spreadtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + '_NP{0:06d}'.format(lp['NP_load']) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'

        print 'searching for ', ffind
    elif 'randorg_gamma' in LatticeTop:
        # NOTE THAT WE USE RANDORG_GAMMAKICK NOW
        raise RuntimeError('We use randorg_gammakick now instead of randorg_gamma.')
        spreadtstr = 'spreadt' + sf.float2pstr(lp['spreading_time'], ndigits=3)
        if lp['NP_load'] > 0:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_periodic_' + spreadtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + '_NP{0:06d}'.format(lp['NP_load']) + '_xy.txt'
        else:
            ffind = rootdir + 'networks/' + LatticeTop + '/' + \
                    lp['LatticeTop'] + '_' + lp['shape'] + '_' + spreadtstr + \
                    '_d' + '{0:02d}'.format(int(lp['conf'])) + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'accordion' in LatticeTop:
        if 'hucent' in LatticeTop:
            # hyperuniform ID string
            hustr = '_d' + huID
            alphstr = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])
            if lp['NP_load'] > 0:
                ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_periodic' + hustr \
                        + alphstr + '_NP{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
            else:
                ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr \
                        + alphstr + '_{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        elif 'isocent' in LatticeTop:
            # accordionkag_isocent or accordionhex_isocent
            alphstr = 'alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam']) + '_'
            # isostatic ID string
            if lp['NP_load'] > 0:
                hustr = '_hexner_size' + str(lp['NP_load']) + '_conf' + huID
                ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + '_periodic' + hustr + \
                        alphstr + '_NP' + '{0:06d}'.format(lp['NP_load']) + cutLstr + '_xy.txt'
            else:
                if lp['source'] == 'ulrich':
                    hustr = '_homog_zindex001' + '_conf' + huID
                elif lp['source'] == 'hexner':
                    if NH > 10 or NV > 10:
                        hustr = '_hexner_size8192_conf' + huID
                    else:
                        hustr = '_hexner_size0512_conf' + huID
                ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + hustr + '_' + \
                        alphstr + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        elif LatticeTop in ['accordionhex', 'accordionkag']:
            if lp['periodicBC']:
                if lp['periodic_strip']:
                    periodicstr = '_periodicstrip'
                else:
                    periodicstr = '_periodicBC'
            else:
                periodicstr = ''

            if eta == 0. or eta == '':
                etastr = ''
            else:
                etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

            if theta_lattice == 0. or theta_lattice == '':
                thetastr = ''
            else:
                thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

            alphstr = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])

            if 'eta_alph' in lp:
                if lp['eta_alph'] > 0:
                    alphstr += '_etaalph' + sf.float2pstr(lp['eta_alph'], ndigits=2)

            delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr + thetastr
            print '\n\n', delta_phi_str
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                    delta_phi_str + alphstr + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'spindle' in LatticeTop:
        if LatticeTop=='spindle':
            if lp['periodicBC']:
                if lp['periodic_strip']:
                    periodicstr = '_periodicstrip'
                else:
                    periodicstr = '_periodicBC'
            else:
                periodicstr = ''

            if eta == 0. or eta == '':
                etastr = ''
            else:
                etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

            if theta_lattice == 0. or theta_lattice == '':
                thetastr = ''
            else:
                thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

            alphstr = '_alph' + sf.float2pstr(lp['alph'], ndigits=4)

            if 'eta_alph' in lp:
                if lp['eta_alph'] > 0:
                    alphstr += '_etaalph' + sf.float2pstr(lp['eta_alph'], ndigits=2)

            delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr + thetastr
            print '\n\n', delta_phi_str
            ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                    delta_phi_str + alphstr + \
                    '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        else:
            raise RuntimeError('only spindley lattice coded in le is spindle itself')
        print 'searching for ', ffind
    elif LatticeTop == 'stackedrhombic':
        if lp['periodicBC']:
            if lp['periodic_strip']:
                periodicstr = '_periodicstrip'
            else:
                periodicstr = '_periodicBC'
        else:
            periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')

        if theta_lattice == 0. or theta_lattice == '':
            thetastr = ''
        else:
            thetastr = '_theta{0:.3f}'.format(theta_lattice).replace('.', 'p') + 'pi'

        stacknum = '_stack' + str(lp['intparam'])

        if 'phi_lattice' not in lp:
            lp['phi_lattice'] = sf.float2pstr(lp['phi'] / np.pi, ndigits=3)

        phi_str = '_phi' + lp['phi_lattice'].replace('.', 'p') + 'pi' + etastr + thetastr
        print '\n\n', phi_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + periodicstr + \
                stacknum + phi_str + \
                '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind
    elif 'junction' in LatticeTop:
        # hexjunction or kagjunction
        periodicstr = ''

        if eta == 0. or eta == '':
            etastr = ''
        else:
            etastr = '_eta{0:.3f}'.format(eta).replace('.', 'p')
        alphstr = '_alph' + sf.float2pstr(lp['alph']) + '_nzag{0:02d}'.format(lp['intparam'])

        delta_phi_str = '_delta' + delta_lattice.replace('.', 'p') + '_phi' + phi_lattice + etastr
        print '\n\n', delta_phi_str
        ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + periodicstr + delta_phi_str + \
                alphstr + '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
        print 'searching for ', ffind

    # else:
    #     theta_eta_str = '_theta' + '{0:.3f}'.format(theta_lattice / np.pi).replace('.','p') +\
    #                     'pi' + '_eta{0:.3f}'.format(eta).replace('.', 'p')
    #     ffind = rootdir + 'networks/' + LatticeTop + '/' + LatticeTop + '_' + shape + theta_eta_str +
    #             '_' + '{0:06d}'.format(NH) + '_x_' + '{0:06d}'.format(NV) + cutLstr + '_xy.txt'
    #     print 'searching for ', ffind
    xyffind = ffind
    meshfn = ffind[0:-7]
    print 'le.build_meshfn(): returning meshfn = ', meshfn
    return meshfn, xyffind


def find_meshfn(lp):
    """Find the filename of a lattice to load, given input parameters.

    Parameters
    ----------
    lp : dict
        dictionary of parameters, including:
        LatticeTop : string
            local topology of the network
        shape : string
            overall geometry of the lattice (boundary shape)
        rootdir : string
            directory containing 'networks' directory
        NH : int
            Number of cells/pts in horizontal dimension
        NV : int
            Number of cells/pts in vertical dimension
        cutLstr (str) [What size slit to cut out of the lattice]
        delta_lattice (str) [For LT==hexagonal, delta is the opening angle at the top of a hexagon]
        phi_lattice (str) [Shear angle phi]
        theta_lattice (float) [Overall rotation of the lattice]
        eta (float) [randomization of the lattice points]
        huID (string) [For LT==hyperuniform(dual), identified which hyperuniform source sample the network is based on]
        z (float) [average coordination. If -1, then uses default]

    Returns
    ----------
    meshfn : string
        mesh file name
    """
    try:
        trash, ffind = build_meshfn(lp)
        print 'ffind = ', ffind
        print 'found --> ', sorted(glob.glob(ffind))[0]
        meshfn = sorted(glob.glob(ffind))[0][0:-7]
    except IndexError:
        raise RuntimeError('ERROR: Could not find specified meshfn!')
    return meshfn


def meshfn_is_used(meshfn):
    """Determine if a network has been saved in directory 'meshfn'

    Parameters
    ----------
    meshfn : str
        The path where the network would be saved

    Returns
    -------
    bool
        Whether the meshfn is used or not
    """
    if glob.glob(meshfn):
        return True
    else:
        return False


def find_lattice(lattice_params):
    """Find the filename of a lattice to load, given input parameters.

    Parameters
    ----------
    LatticeTop : string
        local topology of the network
    shape : string
        overall geometry of the lattice (boundary shape)
    rootdir : string
        directory containing 'networks' directory
    NH : int
        Number of cells/pts in horizontal dimension
    NV : int
        Number of cells/pts in vertical dimension
    lattice_params : dict
        dictionary of parameters, including:
        cutLstr (str) [What size slit to cut out of the lattice]
        delta_lattice (str) [For LT==hexagonal, delta is the opening angle at the top of a hexagon]
        phi_lattice (str) [Shear angle phi]
        theta_lattice (float) [Overall rotation of the lattice]
        eta (float) [randomization of the lattice points]
        huID (string) [For LT==hyperuniform(dual), identified which hyperuniform source sample the network is based on]
        z (float) [average coordination. If -1, then uses default]

    Returns
    ----------
    xyload, NLload, KLload,
    meshfn : string
        mesh file name
    """
    meshfn = find_meshfn(lattice_params)
    xyload = np.loadtxt(meshfn + '_xy.txt', delimiter=',', skiprows=1, usecols=(0, 1))  # , unpack=True)
    NLload = np.loadtxt(meshfn + '_NL.txt', delimiter=',', skiprows=1)
    KLload = np.loadtxt(meshfn + '_KL.txt', delimiter=',', skiprows=1)
    return xyload, NLload, KLload, meshfn


##########################################
# Runge-Kutta functions
##########################################
def rk4_1stO(Xn, R, dt, f, Ni, Nk, spring, pin, b_l=1):
    """Calculates the change in position for a lattice of gyroscopes according to 1st order eqn ('Nash evolution').
    calc_forces_and_cross(X,R, NL, KL, k=1., pin=1., bond_length = 1.)

    Parameters
    ----------
    Xn : nx3 (n = number of gyros) array
        Current positions of the gyroscopes
    dt : float
        simulation time step
    f : function
        function which calculates the right hand side of the differential equation.  Equation of motion
    Ni : n x (max number of neighbors) array
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : dimension n x (max number of neighbors) int array
        Correponds to Ni array.  1 corresponds to a true connection while 0 signifies that there is not a connection
    spring : float
        spring constant
    pin : float
        gravitational spring constant
    R : nx3 float array
        Equilibrium positions of all the gyroscopes

    Returns
    ----------
    dx1,dx2,dx3,dx4,xout : arrays of dimension nx3
        displacements in this time step """
    dx1 = dt * f(Xn, R, Ni, Nk, spring, pin, bond_length=b_l)
    dx2 = dt * f(Xn + dx1 / 2, R, Ni, Nk, spring, pin, bond_length=b_l)
    dx3 = dt * f(Xn + dx2 / 2, R, Ni, Nk, spring, pin, bond_length=b_l)
    dx4 = dt * f(Xn + dx3, R, Ni, Nk, spring, pin, bond_length=b_l)
    xout = Xn + (dx1 + 2. * (dx2 + dx3) + dx4) / 6.
    return dx1, dx2, dx3, dx4, xout


def rk4_damp(xy, v, NL, KL, BM, Mm, beta, h):
    """Performs Runge-Kutta 4th order time step increment for spring system with damping.

    Parameters
    ----------
    x : array NP x nd
        initial position (extension of spring)
    v : array NP x nd
        initial velocity
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    BM : array NP x nn
        unstretched (reference) bond length array
    beta : float
        damping coefficient
    h : float
        time step
    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    dx1 = h * v
    dv1 = h * fdspring(xy, v, NL, KL, BM, Mm, beta)
    dx2 = h * (v + dv1 / 2.)
    dv2 = h * fdspring(xy + dx1 / 2., v + dv1 / 2., NL, KL, BM, Mm, beta)  # xy,NL,KL,BM, Mm, beta, NP,nn
    dx3 = h * (v + dv2 / 2.)
    dv3 = h * fdspring(xy + dx2 / 2., v + dv2 / 2., NL, KL, BM, Mm, beta)
    dx4 = h * (v + dv3)
    dv4 = h * fdspring(xy + dx3, v + dv3, NL, KL, BM, Mm, beta)
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    # print 'rk BM = ', BM
    return dx1, dv1, dx2, dv2, dx3, dv3, dx4, dv4, xout, vout


def rk4_damp_constvBC(xy, v, NL, KL, BM, Mm, beta, h, BND):
    """Performs Runge-Kutta 4th order time step increment for spring system with damping, while moving boundary
    particles (BND index particles) at a prescribed velocity (prescribed in v, by v[BND])

    Parameters
    ----------
    x : array NP x nd
        initial position (extension of spring)
    v : array NP x nd
        initial velocity
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    BM : array NP x nn
        unstretched (reference) bond length array
    beta : float
        damping coefficient
    h : float
        time step
    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    dx1 = h * v
    dv1 = h * fdspring(xy, v, NL, KL, BM, Mm, beta)
    dx1[BND] = h * v[BND]
    dv1[BND] = 0.
    dx2 = h * (v + dv1 / 2.)
    dv2 = h * fdspring(xy + dx1 / 2., v + dv1 / 2., NL, KL, BM, Mm, beta)  # xy,NL,KL,BM, Mm, beta, NP,nn
    dv2[BND] = h * v[BND]
    dx3 = h * (v + dv2 / 2.)
    dv3 = h * fdspring(xy + dx2 / 2., v + dv2 / 2., NL, KL, BM, Mm, beta)
    dv3[BND] = h * v[BND]
    dx4 = h * (v + dv3)
    dv4 = h * fdspring(xy + dx3, v + dv3, NL, KL, BM, Mm, beta)
    dv4[BND] = h * v[BND]
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    # print 'rk BM = ', BM
    return dx1, dv1, dx2, dv2, dx3, dv3, dx4, dv4, xout, vout


def fglidingHST_exact(xy, v, NL, KL, BM, Mm, params):
    """Increment velocity for lattice of gliding Heavy Symmetric Tops (HSTs)
    """
    I1 = params['I1']
    I3 = params['I3']
    l = params['l']
    g = params['g']
    k = params['k']

    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle'''
        NP = 1
        NN = 0

    # print 'xy = ', xy
    # print 'v = ', v

    x = xy[:, 0].ravel()  # .reshape(NP,1);
    y = xy[:, 1].ravel()  # .reshape(NP,1);
    theta = xy[:, 2].ravel()  # .reshape(NP,1);
    phi = xy[:, 3].ravel()  # .reshape(NP,1);
    psi = xy[:, 4].ravel()  # .reshape(NP,1);
    vx = v[:, 0].ravel()  # .reshape(NP,1);
    vy = v[:, 1].ravel()  # .reshape(NP,1);
    vtheta = v[:, 2].ravel()  # .reshape(NP,1);
    vphi = v[:, 3].ravel()  # .reshape(NP,1);
    vpsi = v[:, 4].ravel()  # .reshape(NP,1)

    # if theta is very nearly pi, push it back
    close_pi = 3.1415
    # xout[xy[:,2] > close_pi,2] = close_pi
    theta[theta > close_pi] = close_pi

    # w3 = vpsi + vphi*np.cos(theta)
    w3 = params['w3']
    # if not isinstance(w3,np.ndarray):
    #    print 'w3 --> ndarray'
    #    w3 = np.array(w3)

    # SPRING FORCE
    vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(NN)] for i in range(NP)])
    vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(NN)] for i in range(NP)])
    mag = np.sqrt(vecx ** 2 + vecy ** 2)
    # KLnoz = KL.copy() #no zeros
    # KLnoz[KLnoz ==0] = 1. #same value as mag[mag==0], so that stretch=0 for those
    stretch = mag - BM
    mag[mag == 0.] = 1.  # avoid divide by zero error
    # print(stretch)
    springx = k * np.sum(stretch * vecx / mag, axis=-1)
    springy = k * np.sum(stretch * vecy / mag, axis=-1)
    # print 'stretch = ', stretch

    # add them up
    FX = - springx.ravel()  # .reshape(NP,1)
    FY = - springy.ravel()  # .reshape(NP,1)

    # Set force on fixed particles to zero
    if 'BIND' in params:
        if len(params['BIND']) > 0:
            FX[params['BIND']] = 0.
            FY[params['BIND']] = 0.

    # Transform into A frame
    Fx = FX * np.cos(phi) + FY * np.sin(phi)
    Fy = -FX * np.sin(phi) + FY * np.cos(phi)

    # print '\n Fx =', Fx

    # VERTICAL REACTION FORCE
    # print 'T1 = ', Mm*g*I1
    # print 'T2 =', - Mm*l*(I1*np.cos(theta)*(vtheta**2 + vphi**2*np.sin(theta)**2))
    # print 'T3a = ', I3*w3
    # print 'T3b = ', vphi*np.sin(theta)**2
    # print 'T3 = ', I3*w3*vphi*np.sin(theta)**2
    # print 'T4 = ', - l* np.sin(theta)*np.cos(theta)*Fx
    # print 'denom = ', I1 + Mm*l**2*np.sin(theta)**2
    gn = (Mm * g * I1 - Mm * l * (I1 * np.cos(theta) * (vtheta ** 2 + vphi ** 2 * np.sin(theta) ** 2) - \
                                  I3 * w3 * vphi * (np.sin(theta) ** 2) - l * np.sin(theta) * np.cos(theta) * Fx)) / (
         I1 + Mm * l ** 2 * np.sin(theta) ** 2)

    # print 'gn_ term 1 = ', Mm*g*I1
    # print 'gn_ denominator = ', (I1 + Mm*l**2*np.sin(theta)**2)
    # print 'gn_ denom term 2 = ', Mm*l**2
    print 'gn_exact = ', gn
    # print 'gn = ', gn

    # EULER EQUATIONS
    # print 'denominator = ',I1*np.sin(theta)
    dvphi = (I3 * w3 * vtheta - 2 * I1 * vphi * vtheta * np.cos(theta) - l * Fy) / (I1 * np.sin(theta))
    # print 'dvtheta -- term 1:', l*gn[4]*np.sin(theta[4])
    # print 'dvtheta -- term 2:', -l*Fx[4]*np.cos(theta[4])
    # print 'dvtheta -- term 3:', I1*vphi[4]**2*np.sin(theta[4])*np.cos(theta[4])
    # print 'dvtheta -- term 4:', I3*w3[4]*vphi[4]*np.sin(theta[4])
    dvtheta = (1. / I1) * (l * gn * np.sin(theta) - l * Fx * np.cos(theta) + I1 * vphi ** 2 * np.sin(theta) * np.cos(
        theta) - I3 * w3 * vphi * np.sin(theta))
    dvpsi = - dvphi * np.cos(theta) + vphi * np.sin(theta) * vtheta

    # print 'shape(dvphi)=', np.shape(dvphi)
    # print 'shape(Fx)=', np.shape(Fx)

    # SPRING EQUATIONS
    # print 'dvtheta =', dvtheta
    wx = l * (dvtheta * np.cos(theta) - vtheta ** 2 * np.sin(theta) - vphi ** 2 * np.sin(theta))
    wy = l * (dvphi * np.sin(theta) + 2 * vphi * vtheta * np.cos(theta))
    wX = wx * np.cos(phi) - wy * np.sin(phi)
    wY = wx * np.sin(phi) + wy * np.cos(phi)
    dvX = (FX / Mm) - wX
    dvY = (FY / Mm) - wY

    # print 'shapes = ', np.shape(dvX), np.shape(dvY),np.shape(dvtheta),np.shape(dvphi),np.shape(dvpsi)
    ftx = np.dstack((dvX, dvY, dvtheta, dvphi, dvpsi))[0]
    # print 'Resulting second derivative: ', ftx[1,:]

    if 'BIND' in params:
        if len(params['BIND']) > 0:
            ftx[params['BIND'], 0:2] = [0., 0.]
            # ftx[params['BIND']] = [0.,0.,0.,0.,0.]

    # print 'ftx = ', ftx[4,:]
    # gn_check = Mm*g - Mm*l*(dvtheta*np.sin(theta) + vtheta**2*np.cos(theta))
    # print 'gn_check = ', gn_check-gn
    # if sum(abs(gn_check -gn)) > 1e-8:
    #    print 'gn vertical reaction force does not match up!'
    #    print 'gn_check - gn = ', gn_check-gn
    print 'ftx = ', ftx

    return ftx


def rk4_glidingHST_exact(xy, v, NL, KL, BM, Mm, params):
    """Performs Runge-Kutta 4th order time step increment for gliding heavy symmetric top (Rutstam-like evolution)

    Parameters
    ----------
    xy : array NP x 5
        current position of COM of gyroscopes: x, y, theta, phi, psi
    v : array NP x 5
        current velocity of the pivot points: vx, vy, dtheta/dt, dphi/dt, dpsi/dt
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    BM : array NP x nn
        unstretched (reference) bond length matrix
    h : float
        time step

    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    h = params['h']
    dx1 = h * v
    dv1 = h * fglidingHST_exact(xy, v, NL, KL, BM, Mm, params)
    dx2 = h * (v + dv1 / 2.)
    dv2 = h * fglidingHST_exact(xy + dx1 / 2., v + dv1 / 2., NL, KL, BM, Mm, params)
    dx3 = h * (v + dv2 / 2.)
    dv3 = h * fglidingHST_exact(xy + dx2 / 2., v + dv2 / 2., NL, KL, BM, Mm, params)
    dx4 = h * (v + dv3)
    dv4 = h * fglidingHST_exact(xy + dx3, v + dv3, NL, KL, BM, Mm, params)
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    # print 'dv1 = ', dv1[:,3][10]
    # print 'rk BM = ', BM

    # Ensure theta remains positive
    # xout[:,2] = np.abs(xout[:,2])

    if params['BCtype'] == 'excite':
        d = params['amplitude']
        l = params['l']
        freq = params['frequency']
        x0_BIND = params['x0_BIND']
        y0_BIND = params['y0_BIND']
        BIND = params['BIND']
        w3 = params['w3'][BIND]

        phidot = freq
        phi_new = (xy[BIND, 3] + phidot * h)[0]
        psidot = w3 - phidot * np.cos(xy[BIND, 2])

        # print 'T1 = ', x0_BIND+d*np.cos(phi_new)
        # print 'T2 = ', y0_BIND+d*np.sin(phi_new)
        # print 'T3 = ', xy[BIND,2]
        # print 'T4 = ', xy[BIND,3]+phidot*h
        # print 'T5 = ', xy[BIND,4]+psidot*h

        xout[BIND, :] = np.array([x0_BIND + d * np.cos(phi_new), y0_BIND + d * np.sin(phi_new),
                                  xy[BIND, 2], xy[BIND, 3] + phidot * h, xy[BIND, 4] + psidot * h]).reshape(1, 5)
        vout[BIND, :] = np.array([-d * np.sin(phi_new) * phidot, d * np.cos(phi_new) * phidot,
                                  0.0, phidot, psidot]).reshape(1, 5)

    xout[np.isnan(xout)] = xy[np.isnan(xout)]
    vout[np.isnan(vout)] = v[np.isnan(vout)]

    # Modulo phi
    xout[:, 3] = np.mod(xout[:, 3], 2. * np.pi)
    # Modulo theta
    # xout[:,2] = np.mod(xout[:,4],np.pi)

    return xout, vout


def fglidingHST_PL(xy, v, NL, KL, BM, Mm, params):
    """Increment velocity for lattice of gliding Heavy Symmetric Tops (HSTs) in Planar Limit
    """
    I1 = params['I1']
    I3 = params['I3']
    l = params['l']
    g = params['g']
    k = params['k']

    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle'''
        NP = 1
        NN = 0

    X = xy[:, 0].ravel()  # .reshape(NP,1);
    Y = xy[:, 1].ravel()  # .reshape(NP,1);
    dX = xy[:, 2].ravel()  # .reshape(NP,1);
    dY = xy[:, 3].ravel()  # .reshape(NP,1);
    vX = v[:, 0].ravel()  # .reshape(NP,1);
    vX = v[:, 1].ravel()  # .reshape(NP,1);
    vdX = v[:, 2].ravel()  # .reshape(NP,1);
    vdY = v[:, 3].ravel()  # .reshape(NP,1);

    phi = np.arctan2(dY, dX)
    # print 'xy = ', xy
    # print 'v = ', v

    # Note: w3 = vpsi + vphi*np.cos(theta)
    w3 = params['w3']

    # SPRING FORCE
    vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(NN)] for i in range(NP)])
    vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(NN)] for i in range(NP)])
    mag = np.sqrt(vecx ** 2 + vecy ** 2)
    # KLnoz = KL.copy() #no zeros
    # KLnoz[KLnoz ==0] = 1. #same value as mag[mag==0], so that stretch=0 for those
    stretch = mag - BM
    mag[mag == 0.] = 1.  # avoid divide by zero error
    # print(stretch)
    springx = k * np.sum(stretch * vecx / mag, axis=-1)
    springy = k * np.sum(stretch * vecy / mag, axis=-1)
    # print 'stretch = ', stretch

    # add them up
    FX = - springx.ravel()  # .reshape(NP,1)
    FY = - springy.ravel()  # .reshape(NP,1)

    # Set force on fixed particles to zero
    if 'BIND' in params:
        if len(params['BIND']) > 0:
            FX[params['BIND']] = 0.
            FY[params['BIND']] = 0.

    # Transform into A frame
    Fx = FX * np.cos(phi) + FY * np.sin(phi)
    Fy = -FX * np.sin(phi) + FY * np.cos(phi)

    # print '\n Fx =', Fx

    # POLAR COORDINATES (delta, phi)
    delta = np.sqrt(dX ** 2 + dY ** 2)
    v_delta = vdX * np.cos(phi) + vdY * np.sin(phi)
    v_phi = -vdX * np.sin(phi) + vdY * np.cos(phi)

    # VERTICAL REACTION FORCE
    gn = Mm * (g * l * I1 + I1 * (vdX ** 2 + vdY ** 2) \
               + I3 * w3 * v_phi * delta \
               - l ** 2 * delta * Fx) / (l * I1 + Mm * l * delta ** 2)

    # print 'gn = ', gn

    # EULER EQUATIONS
    dv_phi = (1. / I1) * (-l ** 2 * Fy - I3 * w3 * v_delta)
    dv_delta = (1. / I1) * (-l * gn * delta - l ** 2 * Fx + I3 * w3 * v_phi)

    d_vdX = dv_delta * np.cos(phi) - dv_phi * np.sin(phi)
    d_vdY = dv_delta * np.sin(phi) + dv_phi * np.cos(phi)

    # SPRING EQUATIONS
    # print 'dvtheta =', dvtheta
    qx = dv_delta - v_delta ** 2 * delta / l ** 2
    qy = dv_phi
    qX = qx * np.cos(phi) - qy * np.sin(phi)
    qY = qx * np.sin(phi) + qy * np.cos(phi)
    d_vX = (FX / Mm) - qX
    d_vY = (FY / Mm) - qY

    # print 'check d_vX = ', d_vX

    if params['BCtype'] == 'excite':
        if params['excite_continue']:
            # print 'exciting'
            d = params['amplitude']
            freq = params['frequency']
            x0_BIND = params['x0_BIND']
            y0_BIND = params['y0_BIND']
            BIND = params['BIND']
            w3 = params['w3'][BIND]

            nu = freq
            phi_BIND = (np.arctan2(dY[BIND], dX[BIND]) + nu * params['h'])[0]
            # print 'phi_BIND =', phi_BIND

            d_vX[BIND] = d * nu ** 2 * np.cos(phi_BIND)
            d_vY[BIND] = d * nu ** 2 * np.sin(phi_BIND)
            d_vdX[BIND] = -d * nu ** 2 * np.cos(phi_BIND)
            d_vdY[BIND] = -d * nu ** 2 * np.sin(phi_BIND)

    elif 'BIND' in params:
        if len(params['BIND']) > 0:
            # ftx[params['BIND'],0:2] = [0.,0.]
            d_vX[params['BIND']] = 0.
            d_vY[params['BIND']] = 0.

    # print 'shapes = ', np.shape(dvX), np.shape(dvY),np.shape(dvtheta),np.shape(dvphi),np.shape(dvpsi)
    ftx = np.dstack((d_vX, d_vY, d_vdX, d_vdY))[0]
    # print 'Resulting second derivative: ', ftx[1,:]
    # ftx_exact = fglidingHST_exact(xy, v, NL, KL, BM, Mm, params)
    # print 'gn = ', gn
    # print 'ftx = ', ftx
    # print 'v_delta = ', v_delta
    # print 'v_phi = ', v_phi
    # print 'dv_delta = ', dv_delta
    # print 'dv_phi = ', dv_phi
    # print 'qx = ', qx
    # print 'qy = ', qy
    # print 'ftx_exact = ', ftx_exact

    return ftx


def rk4_glidingHST_PL(xy, v, NL, KL, BM, Mm, params):
    """Performs Runge-Kutta 4th order time step increment for gliding heavy symmetric top in Planar Limit (hanging gyros only)
    Note that b=0 here (hanging gyros)!

    Parameters
    ----------
    xy : array NP x 4
        current position of COM of gyroscopes: X, Y, deltaX, deltaY
    v : array NP x 4
        current velocity of the pivot points: vX, vY, vdeltaX, vdeltaY
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    BM : array NP x nn
        unstretched (reference) bond length matrix
    h : float
        time step
    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    h = params['h']
    dx1 = h * v
    dv1 = h * fglidingHST_PL(xy, v, NL, KL, BM, Mm, params)
    dx2 = h * (v + dv1 * 0.5)
    dv2 = h * fglidingHST_PL(xy + dx1 * 0.5, v + dv1 * 0.5, NL, KL, BM, Mm, params)
    dx3 = h * (v + dv2 * 0.5)
    dv3 = h * fglidingHST_PL(xy + dx2 * 0.5, v + dv2 * 0.5, NL, KL, BM, Mm, params)
    dx4 = h * (v + dv3)
    dv4 = h * fglidingHST_PL(xy + dx3, v + dv3, NL, KL, BM, Mm, params)
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    # print 'rk BM = ', BM

    if params['BCtype'] == 'excite':
        if params['excite_continue']:
            # print 'exciting...'
            d = params['amplitude']
            l = params['l']
            freq = params['frequency']
            x0_BIND = params['x0_BIND']
            y0_BIND = params['y0_BIND']
            BIND = params['BIND']
            w3 = params['w3'][BIND]

            phidot = float(freq)

            phi_new = (np.arctan2(xy[BIND, 3], xy[BIND, 2]) + phidot * h)[0]
            # print 'xy[BIND,:]=', xy[BIND,:]
            # print 'np.arctan2(xy[BIND,3],xy[BIND,2]) = ', np.arctan2(xy[BIND,3],xy[BIND,2])
            # print 'phidot = ', phidot
            # print 'h = ', h
            # print 'np.sin(phi_new) = ', np.sin(phi_new)

            try:
                xout[BIND, :] = np.array([x0_BIND - d * np.cos(phi_new), y0_BIND - d * np.sin(phi_new), \
                                          d * np.cos(phi_new), d * np.sin(phi_new)])
                vout[BIND, :] = np.array([d * np.sin(phi_new) * phidot, -d * np.cos(phi_new) * phidot,
                                          - d * np.sin(phi_new) * phidot, d * np.cos(phi_new) * phidot])

            except:
                xout[BIND, :] = np.hstack((x0_BIND - d * np.cos(phi_new), y0_BIND - d * np.sin(phi_new), \
                                           d * np.cos(phi_new), d * np.sin(phi_new)))[0]
                vout[BIND, :] = np.hstack((d * np.sin(phi_new) * phidot, -d * np.cos(phi_new) * phidot,
                                           -d * np.sin(phi_new) * phidot, d * np.cos(phi_new) * phidot))[0]

                # print 'xout[BIND,:] = ', xout[BIND,:]

    # Ensure theta remains positive
    # xout[:,2] = np.abs(xout[:,2])
    # print xout

    xout[np.isnan(xout)] = xy[np.isnan(xout)]
    vout[np.isnan(vout)] = v[np.isnan(vout)]

    return xout, vout


def calc_forces_and_cross(X, R, NL, KL, k=1., pin=1., bond_length=1.):
    """Calculates the forces on gyroscopes from springs and pinning and does a cross product for EOM .
    Currently CANNOT handle inhomogeneous bond_lengths or spring constants. FIX THIS.
    CANNOT handle correctly the case when bond_length !=1 either. FIX THIS TOO.

    Parameters
    ----------
    X : matrix of dimensions nx3 (n = number of gyros)
        Current positions of the gyroscopes
    R : matrix of dimension nx3
        Equilibrium positions of all the gyroscopes
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : int array of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    k : float or NP x 1 array
        spring constant array. Default value = 1
    pin : float or NP array
        gravitational spring constant = lmg/Iw. Default value = 1
    bond_length : float
        Length of unstretched springs.  Default value = 1

    Returns
    ----------
    crossed : matrix of dimension nx3
        cross product of forces and zhat
    """
    try:
        NP, NN = np.shape(NL)
    except:
        '''There is only one particle'''
        NP = 1
        NN = 0

    dX = X - R
    Nk2 = np.reshape(KL, [NP, NN, 1])
    Nk1 = KL.copy()
    Nk1 = np.reshape(Nk1, [NP, NN, 1])
    Nk1[np.where(Nk1 != 0)] = 1
    pedge = np.where(Nk2 < 0)
    Xni = X[NL]  # positions of neighbors
    vecs = np.array([Nk1[i] * (X[i] - Xni[i]) for i in range(len(Xni))])  # vector from gyro to neighboring gyro

    mags = np.sum(abs(vecs) ** 2, axis=-1) ** (1 / 2.)  # magnitude of vectors
    # set zero magnitude vectors to bond length (avoids divide by zero error).
    # Also sets 'stretch' = 0 if we had something that wasn't a connection
    # print 'shape(mags) = ', np.shape(mags)
    # print 'shape(bond_length) = ', np.shape(bond_length)

    # Note that is is ok to have mags improperly normalized for elems=0, since KL is zero in those places
    mags[mags == 0] = 1.

    # find the amount each bond is stretched/compressed by
    stretch = KL * (mags - bond_length)
    mags = np.reshape(mags, [NP, NN, 1])
    vec_hat = vecs / mags  # unit length vectors
    stretch = np.reshape(stretch, [NP, NN, 1])
    # print 'Nk',  Nk
    force = -stretch * vec_hat  # assuming k = homogenous constant added later
    force_v = np.sum(force, axis=1)

    crossed = np.zeros([NP, 2])

    # print 'force_v = ', force_v
    # CHECKING
    # tmp1 =  k*force_v.T[1]
    # tmp2 = np.array(pin, dtype = float)* dX.T[1]
    # print 'shape(term1) = ', np.shape(tmp1)
    # print 'shape(term2) = ', np.shape(tmp2)

    crossed[:, 0] = (k * force_v.T[1] - np.array(pin, dtype=float) * dX.T[1])
    crossed[:, 1] = (-k * force_v.T[0] + np.array(pin, dtype=float) * dX.T[0])

    return crossed


def calc_forces_and_cross_magnetic(X, Ni, Nk, spring, pin, R, bond_length=1, out_index=1):
    """Calculates the forces on gyroscopes from magnetic interactions and pinning and does a cross product for EOM .
    This is too crude -- update to include repulsive perpendicular component!

    Parameters
    ----------
    X : matrix of dimensions 2nx3 (n = number of gyros)
        Current positions of the gyroscopes
    Ni : matrix of dimension n x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    Nk : matrix of dimension n x (max number of neighbors)
        Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
    spring : float
        spring constant
    pin : float
        gravitational spring constant
    R : matrix of dimension 2nx3
        Equilibrium positions of all the gyroscopes
    bond_length : float
        Length of unstretched springs.  Default value = 1

    Returns
    ----------
    crossed : matrix of dimension nx3
        cross product of forces and zhat """

    NP, NN = np.shape(Ni)

    dX = X - R  # delta x
    Nk2 = np.reshape(Nk, [NP, NN, 1])

    Xni = X[Ni]  # positions of neighbors

    vecs = np.array([Nk2[i] * (X[i] - Xni[i]) for i in range(len(Xni))])  # vector from neighboring gyro to gyro

    mags = sum(abs(vecs) ** 2, axis=-1) ** (1 / 2.)  # magnitude of vectors

    mags[mags == 0] = bond_length  # set zero magnitude vectors to bond length (avoids divide by zero error).

    mags = np.reshape(mags, [NP, NN, 1])
    vec_hat = vecs / mags  # unit length vectors

    force = vecs / mags ** 5
    force_v = sum(force, axis=1)

    crossed = np.zeros([NP, 3])
    crossed[:, 0] = (spring * force_v.T[1] - pin * dX.T[1])
    crossed[:, 1] = (- spring * force_v.T[0] + pin * dX.T[0])
    crossed[out_index] = 0
    crossed[np.where(abs(crossed) < 10 ** -5)] = 0
    return crossed


def fdspring(xy, v, NL, KL, BM, Mm, beta):
    """Computes dv/dt for massive particles connected by damped springs (in 2D or 3D).

    Parameters
    ----------
    xy : matrix of dimensions Nx2 (N = number of masses/gyros)
        Current positions of the masses/gyroscopes
    v : array NP x spatial dimensions
        velocity
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : matrix of dimension N x (max number of neighbors)
        Spring constant -- like NL matrix, with spring const values (k) corresponding to a true connection while 0 signifies that there is not a connection
    BM : NP x NN array or float
        Rest bond lengths, arranged like KL
    Mm : matrix of dimension Nx1
        Masses of each particle
    beta : float
        damping coefficient

    Returns
    ----------
    ftx : martix of dimension N x 2
        dv/dt for all particles
    """
    NP, nn = np.shape(NL)
    if np.shape(xy)[1] == 2:
        '''2D version'''
        vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
        vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
        mag = np.sqrt(vecx ** 2 + vecy ** 2)
        # KLnoz = KL.copy() #no zeros
        # KLnoz[KLnoz ==0] = 1. --> same value as mag[mag==0], so that stretch=0 for those
        stretch = mag - BM
        mag[mag == 0.] = 1.  # avoid divide by zero error
        # print(stretch)
        dxvec = np.sum(stretch * vecx / mag, axis=-1) / Mm
        dyvec = np.sum(stretch * vecy / mag, axis=-1) / Mm
        # damping term
        damp_dv = np.array([beta / Mm[i] * v[i] for i in range(NP)])
        # add them up
        ftx = -np.hstack((dxvec.reshape(NP, 1), dyvec.reshape(NP, 1))) - damp_dv
    else:
        '''3D version'''
        vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
        vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
        vecz = np.array([[KL[i, j] * (xy[i, 2] - xy[NL[i, j], 2]) for j in range(nn)] for i in range(NP)])
        mag = np.sqrt(vecx ** 2 + vecy ** 2 + vecz ** 2)
        # KLnoz = KL.copy() #no zeros
        # KLnoz[KLnoz ==0] = 1. #same value as mag[mag==0], so that stretch=0 for those
        stretch = mag - BM
        mag[mag == 0.] = 1.  # avoid divide by zero error
        dxvec = np.sum(stretch * vecx / mag, axis=-1) / Mm
        dyvec = np.sum(stretch * vecy / mag, axis=-1) / Mm
        dzvec = np.sum(stretch * vecz / mag, axis=-1) / Mm
        # damping term
        damp_dv = np.array([beta / Mm[i] * v[i] for i in range(NP)])
        # add them up
        ftx = -np.hstack((dxvec.reshape(NP, 1), dyvec.reshape(NP, 1), dyvec.reshape(NP, 1))) - damp_dv
    return ftx


def fspring(xy, NL, KL, BM, Mm):
    """Computes dv/dt for massive particles connected by undamped springs (in 2D or 3D).

    Parameters
    ----------
    t : float
        simulation time
    xy : matrix of dimensions Nx2 (N = number of masses/gyros)
        Current positions of the masses/gyroscopes
    f : function
        function which calculates the right hand side of the differential equation.  Equation of motion
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : matrix of dimension N x (max number of neighbors)
        Spring constant -- like NL matrix, with spring const values (k) corresponding to a true connection while 0 signifies that there is not a connection
    BM : NP x NN array or float
        Rest bond lengths, arranged like KL
    Mm : matrix of dimension Nx1
        Masses of each particle

    Returns
    ----------
    ftx : martix of dimension N x 2
        dv/dt for all particles
    """
    NP, nn = np.shape(NL)
    if np.shape(xy)[1] == 2:
        '''2D version'''
        vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
        vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
        mag = np.sqrt(vecx ** 2 + vecy ** 2)
        # KLnoz = KL.copy() #no zeros
        # KLnoz[KLnoz ==0] = 1. #same value as mag[mag==0], so that stretch=0 for those
        stretch = mag - BM
        mag[mag == 0.] = 1.  # avoid divide by zero error
        # print(stretch)
        dxvec = np.sum(stretch * vecx / mag, axis=-1) / Mm
        dyvec = np.sum(stretch * vecy / mag, axis=-1) / Mm
        # add them up
        ftx = -np.hstack((dxvec.reshape(NP, 1), dyvec.reshape(NP, 1)))
    else:
        '''3D version'''
        vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
        vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
        vecz = np.array([[KL[i, j] * (xy[i, 2] - xy[NL[i, j], 2]) for j in range(nn)] for i in range(NP)])
        mag = np.sqrt(vecx ** 2 + vecy ** 2 + vecz ** 2)
        # KLnoz = KL.copy() #no zeros
        # KLnoz[KLnoz ==0] = 1. #same value as mag[mag==0], so that stretch=0 for those
        stretch = mag - BM
        mag[mag == 0.] = 1.  # avoid divide by zero error
        # print(stretch)
        dxvec = np.sum(stretch * vecx / mag, axis=-1) / Mm
        dyvec = np.sum(stretch * vecy / mag, axis=-1) / Mm
        dzvec = np.sum(stretch * vecz / mag, axis=-1) / Mm
        # add them up
        ftx = -np.hstack((dxvec.reshape(NP, 1), dyvec.reshape(NP, 1), dyvec.reshape(NP, 1)))
    return ftx


def fspring_perp(xy, NL, KL, KGdivKL, Mm, NP, nn):
    """Computes dv/dt for all particles in 2D at time t, with a perpendicular force between particles.

    Parameters
    ----------
    t : float
        simulation time
    xy : matrix of dimensions Nx2 (N = number of masses/gyros)
        Current positions of the masses/gyroscopes
    f : function
        function which calculates the right hand side of the differential equation.  Equation of motion
    NL : matrix of dimension N x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension N x (max number of neighbors)
        Spring constant -- like NL matrix, with spring const values (k) corresponding to a true connection while 0 signifies that there is not a connection
    Mm : matrix of dimension Nx1
        Masses of each particle

    Returns
    ----------
    ftx : martix of dimension N x 2
        dv/dt for all particles, in 2D
    """
    vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
    vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
    mag = np.sqrt(vecx ** 2 + vecy ** 2)
    mag[mag == 0.] = 1.  # avoid divide by zero error
    KLnoz = KL.copy()  # no zeros
    KLnoz[KLnoz == 0] = 1.  # same value as mag[mag==0], so that stretch=0 for those
    stretch = mag - KL
    # print(stretch)
    vechx = vecx / mag
    vechy = vecy / mag
    dxvec = np.sum(stretch * vechx, axis=-1) / Mm
    dyvec = np.sum(stretch * vechy, axis=-1) / Mm
    # Add perp force
    dxvecP = KGdivKL * np.sum(stretch * (-vechy), axis=-1) / Mm
    dyvecP = KGdivKL * np.sum(stretch * vechx, axis=-1)
    ftx = -np.hstack((dxvec.reshape(NP, 1) + dxvecP.reshape(NP, 1), dyvec.reshape(NP, 1) + dyvecP.reshape(NP, 1)))
    return ftx


def rk4_perp(xy, v, NL, KL, KGdivKL, Mm, NP, nn, h):
    """Performs Runge-Kutta 4th order time step increment for spring system with perpendicular force btwn particles.

    Parameters
    ----------
    x : array NP x nd
        initial position (extension of spring)
    v : array NP x nd
        initial velocity
    NL : array NP x nn
        ith row contains indices of neighbors for ith particle
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    h : float
        time step
    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    dx1 = h * v
    dv1 = h * fspring_perp(xy, NL, KL, KGdivKL, Mm, NP, nn)
    dx2 = h * (v + dv1 / 2.)
    dv2 = h * fspring_perp(xy + dx1 / 2., NL, KL, KGdivKL, Mm, NP, nn)
    dx3 = h * (v + dv2 / 2.)
    dv3 = h * fspring_perp(xy + dx2 / 2., NL, KL, KGdivKL, Mm, NP, nn)
    dx4 = h * (v + dv3)
    dv4 = h * fspring_perp(xy + dx3, NL, KL, KGdivKL, Mm, NP, nn)
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    return dx1, dv1, dx2, dv2, dx3, dv3, dx4, dv4, xout, vout


def fgyro_Nash(xy, xy0, NL, theta, OmK, Omg, Mm, NP, nn):
    """Computes dx/dt for gyros in 2D with linearized approximation (Schrodinger-like).

    Parameters
    ----------
    xy : matrix of dimensions Nx2 (N = number of masses/gyros)
        Current positions of the masses/gyroscopes
    xy0 : array NP x nd
        initial position (extension of spring)
    NL : matrix of dimension N x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    theta : array NP x nn
        matrix of angles between particles
    OmK : matrix of dimension N x (max number of neighbors)
        Spring constant -- like NL matrix, with spring const values (k l^2 /I omega) corresponding to a true connection while 0 signifies that there is not a connection
    Mm : matrix of dimension Nx1
        Masses of each particle

    Returns
    ----------
    ftx : martix of dimension N x 2
        dv/dt for all particles, in 2D
    """
    xgrav = Omg * (xy[:, 1] - xy0[:, 1])
    vecx = np.array([[OmK[i, j] * 0.5 * (xy[i, 1] - xy0[i, 1] - xy[NL[i, j], 1] + xy0[NL[i, j], 1] +
                                         np.cos(2 * theta[i, j]) * (
                                         -xy[i, 1] + xy0[i, 1] + xy[NL[i, j], 1] - xy0[NL[i, j], 1]) +
                                         np.sin(2 * theta[i, j]) * (
                                         xy[i, 0] - xy0[i, 0] - xy[NL[i, j], 0] + xy0[NL[i, j], 0]))
                      for j in range(nn)] for i in range(NP)])

    ygrav = -Omg * (xy[:, 0] - xy0[:, 0])
    vecy = np.array([[-OmK[i, j] * 0.5 * (xy[i, 0] - xy0[i, 0] - xy[NL[i, j], 0] + xy0[NL[i, j], 0] +
                                          np.cos(2 * theta[i, j]) * (
                                          xy[i, 0] - xy0[i, 0] - xy[NL[i, j], 0] + xy0[NL[i, j], 0]) +
                                          np.sin(2 * theta[i, j]) * (
                                          xy[i, 1] - xy0[i, 1] - xy[NL[i, j], 1] + xy0[NL[i, j], 1]))
                      for j in range(nn)] for i in range(NP)])

    dx = np.sum(vecx, axis=-1)
    dy = np.sum(vecy, axis=-1)
    ftx = np.hstack(((dx + xgrav).reshape(NP, 1), (dy + ygrav).reshape(NP, 1)))
    return ftx


def fBrunGyro(xy, xy0, NL, KL, KG, Mm, NP, nn):
    """Computes dv/dt for all gyros in 2D at time t in steady state frequency determined by KG=I omega^2/l^2,
    according to the model of Brun et al 2012.

    Parameters
    ----------
    t : float
        simulation time
    xy : matrix of dimensions Nx2 (N = number of masses/gyros)
        Current positions of the masses/gyroscopes
    f : function
        function which calculates the right hand side of the differential equation.  Equation of motion
    NL : matrix of dimension N x (max number of neighbors)
        Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
    KL : matrix of dimension N x (max number of neighbors)
        Spring constant -- like NL matrix, with spring const values (k) corresponding to a true connection while 0 signifies that there is not a connection
    KG : array of dimension N x 1
        Gyro constant 'rotational force'
    Mm : matrix of dimension Nx1
        Masses of each particle

    Returns
    ----------
    ftx : martix of dimension N x 2
        dv/dt for all particles, in 2D
    """
    vecx = np.array([[KL[i, j] * (xy[i, 0] - xy[NL[i, j], 0]) for j in range(nn)] for i in range(NP)])
    vecy = np.array([[KL[i, j] * (xy[i, 1] - xy[NL[i, j], 1]) for j in range(nn)] for i in range(NP)])
    mag = np.sqrt(vecx ** 2 + vecy ** 2)
    mag[mag == 0.] = 1.  # avoid divide by zero error
    KLnoz = KL.copy()  # no zeros
    KLnoz[KLnoz == 0] = 1.  # same value as mag[mag==0], so that stretch=0 for those
    stretch = mag - KL
    # Add 'rotational force'
    Uvec = np.array([KG[i] * (xy[i, :] - xy0[i, :]) for i in range(len(KG))])
    Umag = np.sqrt(Uvec[:, 0] * Uvec[:, 0] + Uvec[:, 1] * Uvec[:, 1])
    Umag[Umag == 0] = 1.
    Uvech = np.array([Uvec[i, :] / Umag[i] for i in range(len(Umag))])
    Uvech_tr = np.dstack((-Uvech[:, 1], Uvech[:, 0]))[0]
    dvecG = np.array([KG[i] * Umag[i] * Uvech_tr[i, :] for i in range(len(Umag))])
    # Sum total forces
    dxvec = np.sum(stretch * vecx / mag, axis=-1) / Mm
    dyvec = np.sum(stretch * vecy / mag, axis=-1) / Mm
    ftx = -np.hstack((dxvec.reshape(NP, 1), dyvec.reshape(NP, 1)))
    dxyG = np.array([dvecG[i, :] / Mm[i] for i in range(len(Mm))])
    dv = ftx + dxyG
    return dv


def rk4_BrunGyro(xy, xy0, v, NL, KL, KG, Mm, NP, nn, hh):
    """Performs Runge-Kutta 4th order time step increment for spring-gyro system in steady state frequency determined by KG=I omega^2/l^2,
    according to the model of Brun et al 2012.

    Parameters
    ----------
    x : array NP x nd
        initial position (extension of spring)
    v : array NP x nd
        initial velocity
    NL : #pts x max(#neighbors) int array
        The ith row contains indices for the neighbors for the ith point, buffered by zeros if a particle does not have the
        maximum # nearest neighbors. KL can be used to discriminate a true 0-index neighbor from a buffered zero.
    KL : array NP x nn
        bond strength for each bond (same format as NL)
    h : float
        time step
    Returns
    ----------
    dx1,dv1,dx2,dv2,dx3,dv3,dx4,dv4,xout,vout : arrays NP x nd
        all parameters for integration
    """
    dx1 = hh * v
    dv1 = hh * fBrunGyro(xy, xy0, NL, KL, KG, Mm, NP, nn)
    dx2 = hh * (v + dv1 / 2.)
    dv2 = hh * fBrunGyro(xy + dx1 / 2., xy0, NL, KL, KG, Mm, NP, nn)
    dx3 = hh * (v + dv2 / 2.)
    dv3 = hh * fBrunGyro(xy + dx2 / 2., xy0, NL, KL, KG, Mm, NP, nn)
    dx4 = hh * (v + dv3)
    dv4 = hh * fBrunGyro(xy + dx3, xy0, NL, KL, KG, Mm, NP, nn)
    xout = xy + (dx1 + 2. * dx2 + 2. * dx3 + dx4) / 6.
    vout = v + (dv1 + 2. * dv2 + 2. * dv3 + dv4) / 6.

    return dx1, dv1, dx2, dv2, dx3, dv3, dx4, dv4, xout, vout


def print_output_1stO(dx1, dx2, dx3, dx4, xyout):
    """Print the output of a 1st order Runge-Kutta 4th order evaluation."""
    print('dx1 = ' + str(dx1) + '\n')
    print('dx2 = ' + str(dx2) + '\n')
    print('dx3 = ' + str(dx3) + '\n')
    print('dx4 = ' + str(dx4) + '\n')
    print('xyout = ' + str(xyout) + '\n')


def print_output_test_actual_1stO(dx1t, dx1, dx2t, dx2, dx3t, dx3, dx4t, dx4, xyoutt, xyout):
    """Print the test (numpy) and actual (OpenCL) output of a 1st order Runge-Kutta 4th order evaluation."""
    print('dx1tes = ' + str(dx1t) + '\n')
    print('dx1 = ' + str(dx1) + '\n')
    print('dx2test = ' + str(dx2t) + '\n')
    print('dx2 = ' + str(dx2) + '\n')
    print('dx3test = ' + str(dx3t) + '\n')
    print('dx3 = ' + str(dx3) + '\n')
    print('dx4test = ' + str(dx4t) + '\n')
    print('dx4 = ' + str(dx4) + '\n')
    print('xyouttest = ' + str(xyoutt) + '\n')
    print('xyout = ' + str(xyout) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Demonstrate some functions in lattice_elasticity')
    parser.add_argument('-all_demos', '--all_demos', help='Do all demos', action='store_true')
    parser.add_argument('-demo_dirgym', '--demo_dirgym', help='Directory gymnastics demo', action='store_true')
    parser.add_argument('-demo_kagome', '--demo_kagome', help='kagome lattice demo', action='store_true')
    parser.add_argument('-demo_cutbonds', '--demo_cutbonds', help='cutting bonds demo', action='store_true')
    parser.add_argument('-demo_boundary', '--demo_boundary', help='do demo for handling boundary', action='store_true')
    parser.add_argument('-demo_coordination', '--demo_coordination', help='do demo for tuning coordination',
                        action='store_true')
    parser.add_argument('-demo_centroid', '--demo_centroid', help='do demo for decorating lattice as centroid',
                        action='store_true')
    parser.add_argument('-demo_polygons', '--demo_polygons', help='do demo for extracting and coloring polygons',
                        action='store_true')
    parser.add_argument('-demo_haldane', '--demo_haldane', help='do demo for computing haldane model',
                        action='store_true')
    parser.add_argument('-demo_unique_rows', '--demo_unique_rows',
                        help='do demo for getting unique rows in a numpy array', action='store_true')
    parser.add_argument('-demo_chern', '--demo_chern',
                        help='do demo calculating chern number of disordered gyro lattice', action='store_true')
    parser.add_argument('-demo_BL', '--demo_BL', help='Do demo handling bond list', action='store_true')
    parser.add_argument('-demo_dislocation', '--demo_dislocation', help='do demo inputting a dislocation in a lattice',
                        action='store_true')
    parser.add_argument('-demo_triangulation', '--demo_triangulation', help='do demo with triangulating points',
                        action='store_true')
    parser.add_argument('-demo_distance_from_boundary', '--demo_distance_from_boundary',
                        help='do demo with computing distance of each point from boundary',
                        action='store_true')
    args = parser.parse_args()

    if args.all_demos:
        args.demo_dirgym = True
        args.demo_kagome = True
        args.demo_cutbonds = True
        args.demo_boundary = True
        args.demo_coordination = True
        args.demo_centroid = True
        args.demo_polygons = True
        args.demo_haldane = True
        args.demo_unique_rows = True
        args.demo_chern = True
        args.demo_BL = True
        args.demo_dislocation = True
        args.demo_triangulation = True
        args.demo_intersections = True

    if args.demo_boundary:
        xy = np.array([[0.1512926, -0.37403114],
                       [0.66390186, -1.21998997],
                       [1.17480764, 0.27231353],
                       [-3.81844152, -0.30166257],
                       [0.93878695, 0.87060133],
                       [0.53435557, 1.30176558],
                       [1.31181627, -0.91816195],
                       [1.05345549, -0.01980385],
                       [-0.91245103, -0.36542935],
                       [-0.80641536, 1.06187272],
                       [-0.3041718, 1.05616468],
                       [-0.31971738, -1.20429765],
                       [-0.52918824, -0.1159254],
                       [-0.03666739, -0.85863593],
                       [-0.108136, -1.4940799],
                       [2.56295229, -1.92179549],
                       [0.16828214, -0.63639772]])

        # Show what just a simple triangulation would do:
        xy, NL, KL, BL, BM = delaunay_lattice_from_pts(xy, trimbound=False, max_bond_length=1.4,
                                                       minimum_bonds=2, thres=10.0, check=True)
        TRI = BL2TRI(BL, xy)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'bo-')
        plt.title('Resulting triangulation from delaunay_lattice_from_pts')
        plt.show()

        # Show how to kill unnatural boundaries (trimbound)
        xy, NL, KL, BL, BM = delaunay_lattice_from_pts(xy, trimbound=True, max_bond_length=1.4, thres=10.0, check=True)
        TRI = BL2TRI(BL, xy)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'go-')
        plt.show()

        # Extract the new boundary of a bonded point set
        print ' demo : extract boundary...'
        boundary = extract_boundary(xy, NL, KL, BL, check=True)

        fig = plt.figure()
        plt.clf()
        ax = plt.axes()
        ax.set_aspect('equal')
        ax.plot(xy[:, 0], xy[:, 1], 'b.')
        ax.plot(xy[boundary, 0], xy[boundary, 1], 'b-')
        ax.plot(xy[[boundary[-1], boundary[0]], 0], xy[[boundary[-1], boundary[0]], 1], 'b-')
        plt.title('Resulting boundary from extract_boundary()')
        plt.show()

        # Make BL to make TRI
        BL = NL2BL(NL, KL)
        TRI = BL2TRI(BL, xy)

        # # Show how to identify the boundary triangles
        # btri = boundary_triangles(TRI,boundary)
        # print 'Identified the boundary triangles as:', TRI[btri]
        # zfaces = np.zeros(len(TRI),dtype=float)
        # zfaces[btri] = 1.
        # # Color the boundary triangles in a plot
        # plt.figure()
        # plt.gca().set_aspect('equal')
        # plt.tripcolor(xy[:,0], xy[:,1], TRI, facecolors=zfaces, edgecolors='k')
        # plt.colorbar()
        # for i in range(len(xy)):
        #     plt.text(xy[i,0]-0.2, xy[i,1], str(i))
        # plt.title('tripcolor() showing boundary tris')
        # plt.show()
        #
        # display_lattice_2D(xy,BL,title='',close=False)
        # for i in range(len(xy)):
        #     plt.text(xy[i,0]+0.05,xy[i,1],str(i))
        # plt.show()

    if args.demo_coordination:
        xy = np.array([[1.39690013, -3.03634367],
                       [-0.14636921, 0.18539301],
                       [0.34083439, 2.34870316],
                       [0.74877208, -0.68229828],
                       [-1.37567611, -0.72132123],
                       [-0.59368019, 2.93538677],
                       [0.54114647, -0.97306515],
                       [-2.04741577, -0.34073868],
                       [-2.70216019, -0.02370007],
                       [0.43653183, 0.60133754],
                       [1.49008235, -0.84843801],
                       [0.8493978, -2.98065358],
                       [1.43130423, 1.76312731],
                       [0.31831829, -1.01562674],
                       [0.13901971, 0.58589103],
                       [-0.19681738, 0.13672238],
                       [0.98078747, 1.6857505]])

        # Now tune coordination, after automatically computing the starting coordination
        # print 'CUT RANDOM BONDS...'
        # NL,KL,BL,BM = delaunay_lattice_from_pts(xy,trimbound=True,target_z=4.0,zmethod='random')
        # TRI = BL2TRI(BL,xy)
        # fig = plt.figure()
        # plt.gca().set_aspect('equal')
        # #plt.triplot(xy[:,0], xy[:,1], TRI, 'bo-')
        # display_lattice_2D(xy,BL,title='',close=False)
        # for i in range(len(xy)):
        #     plt.text(xy[i,0]+0.05,xy[i,1],str(i))
        # #print 'NL = ', NL
        # #print 'KL = ', KL
        # #boundary = extract_boundary(xy,NL,KL, BL)
        # #bulk = np.setdiff1d(np.arange(len(xy)), boundary)
        # #print 'bulk z = ', float(np.count_nonzero(KL[bulk]))/float(len(bulk))
        # #print 'boundary z = ', float(np.count_nonzero(KL[boundary]))/float(len(boundary))
        # plt.show()

        fig = plt.figure()
        plt.gca().set_aspect('equal')
        # Now tune coordination, after automatically computing the starting coordination
        print 'CUT HIGHEST z BONDS...'
        NL, KL, BL, BM = delaunay_lattice_from_pts(xy, trimbound=True, target_z=2.5, zmethod='highest')
        TRI = BL2TRI(BL, xy)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        display_lattice_2D(xy, BL, title='', close=False)
        for i in range(len(xy)):
            plt.text(xy[i, 0] + 0.05, xy[i, 1], str(i))

    if args.demo_dirgym:
        maindir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/experiments/'
        subdir, subsubdir = dio.find_subsubdirectory('20151226*', maindir)
        print 'subdir = ', subdir
        print 'subsubdir = ', subsubdir
        print '\n\n len(subdir)= ', len(subdir)
        print '\n\n len(subsubdir)= ', len(subsubdir)

    if args.demo_cutbonds:
        xy = np.array([[-1., 0.],
                       [-0.5, -0.8660254], [-0.5, 0.8660254],
                       [0., 0.], [0.5, -0.8660254],
                       [0.5, 0.8660254], [1., 0.]])
        NL = np.array([[1, 2, 3, 0, 0, 0],
                       [0, 3, 4, 0, 0, 0], [0, 3, 5, 0, 0, 0],
                       [0, 1, 2, 4, 5, 6], [1, 3, 6, 0, 0, 0],
                       [2, 3, 6, 0, 0, 0], [3, 4, 5, 0, 0, 0]])
        KL = np.array([[1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
        BL = np.array([[0, 1], [0, 2],
                       [0, 3], [1, 3],
                       [2, 3], [1, 4],
                       [3, 4], [2, 5],
                       [3, 5], [3, 6],
                       [4, 6], [5, 6]])
        BM0 = NL2BM(xy, NL, KL)
        BM0 -= (np.random.rand(np.shape(BM0)[0], np.shape(BM0)[1]) - 0.5) * 0.1

        bstrain = 0.04
        bL0 = BM2bL(NL, BM0, BL)
        bs = bond_strain_list(xy, BL, bL0)
        display_lattice_2D(xy, BL, bs, title='', xlimv=2.0, ylimv=-2.0, climv=bstrain)
        KL, BLtrim, bL0trim = cut_bonds_strain(xy, NL, KL, BM0, bstrain)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BLtrim = ', BLtrim
        print 'bL0trim = ', bL0trim
        bs = bond_strain_list(xy, BLtrim, bL0trim)
        display_lattice_2D(xy, BLtrim, bs, title='', xlimv=2.0, ylimv=-2.0, climv=bstrain)

        bstrain = 0.02
        bL0 = BM2bL(NL, BM0, BLtrim)
        KL, BLtrim, bL0trim = cut_bonds_strain(xy, NL, KL, BM0, bstrain)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BLtrim = ', BLtrim
        print 'bL0trim = ', bL0trim
        bs = bond_strain_list(xy, BLtrim, bL0trim)
        BLtest = NL2BL(NL, KL)
        print 'BLtrim - BLtest = ', BLtrim - BLtest
        display_lattice_2D(xy, BLtrim, bs, title='', xlimv=2.0, ylimv=-2.0, climv=bstrain)

        bstrain = 0.005
        bL0 = BM2bL(NL, BM0, BLtrim)
        KL, BLtrim, bL0trim = cut_bonds_strain(xy, NL, KL, BM0, bstrain)
        print 'NL = ', NL
        print 'KL = ', KL
        print 'BLtrim = ', BLtrim
        print 'bL0trim = ', bL0trim
        BLtest = NL2BL(NL, KL)
        print 'BLtrim - BLtest = ', BLtrim - BLtest
        bs = bond_strain_list(xy, BLtrim, bL0trim)
        display_lattice_2D(xy, BLtrim, bs, title='', xlimv=2.0, ylimv=-2.0, climv=bstrain)

    if args.demo_kagome:
        '''Create a lattice by xy, NL, KL, and BL'''
        xy = np.array([[-0.39444444, -0.81791288],
                       [0.60555556, -0.81791288], [-0.34444444, -0.38490018],
                       [0.65555556, -0.38490018], [-0.84444444, -0.38490018],
                       [0.15555556, -0.38490018], [1.15555556, -0.38490018],
                       [-1.34444444, -0.38490018], [0.10555556, 0.04811252],
                       [1.10555556, 0.04811252], [-0.89444444, 0.04811252],
                       [-0.84444444, 0.48112522], [0.15555556, 0.48112522],
                       [1.15555556, 0.48112522], [-1.34444444, 0.48112522],
                       [-0.34444444, 0.48112522], [0.65555556, 0.48112522],
                       [-0.39444444, 0.91413793], [0.60555556, 0.91413793]])
        NL = np.array([[2., 4., 0., 0.],
                       [3., 5., 0., 0.], [0., 4., 5., 8.],
                       [1., 5., 6., 9.], [0., 2., 7., 10.],
                       [1., 2., 3., 8.], [3., 9., 0., 0.],
                       [4., 10., 0., 0.], [2., 5., 12., 15.],
                       [3., 6., 13., 16.], [4., 7., 11., 14.],
                       [10., 14., 15., 17.], [8., 15., 16., 18.],
                       [9., 16., 0., 0.], [10., 11., 0., 0.],
                       [8., 11., 12., 17.], [9., 12., 13., 18.],
                       [11., 15., 0., 0.], [12., 16., 0., 0.]])
        KL = np.array([[1., 1., 0., 0.],
                       [1., 1., 0., 0.], [1., 1., 1., 1.],
                       [1., 1., 1., 1.], [1., 1., 1., 1.],
                       [1., 1., 1., 1.], [1., 1., 0., 0.],
                       [1., 1., 0., 0.], [1., 1., 1., 1.],
                       [1., 1., 1., 1.], [1., 1., 1., 1.],
                       [1., 1., 1., 1.], [1., 1., 1., 1.],
                       [1., 1., 0., 0.], [1., 1., 0., 0.],
                       [1., 1., 1., 1.], [1., 1., 1., 1.],
                       [1., 1., 0., 0.], [1., 1., 0., 0.]])
        BL = np.array([[0, 2],
                       [0, 4], [2, 4],
                       [2, 5], [2, 8],
                       [3, 5], [3, 6],
                       [3, 9], [4, 7],
                       [4, 10], [5, 8],
                       [6, 9], [7, 10],
                       [8, 12], [8, 15],
                       [9, 13], [9, 16],
                       [10, 11], [10, 14],
                       [11, 14], [11, 15],
                       [11, 17], [12, 15],
                       [12, 16], [12, 18],
                       [13, 16], [15, 17],
                       [16, 18], [1, 3],
                       [1, 5]])

        ''' Determine BM from the lattice and display the lattice'''
        BM = NL2BM(xy, NL, KL)
        print 'BM = ', BM
        bo = 0.5
        bs = bond_strain_list(xy, BL, bo)
        display_lattice_2D(xy, BL, bs, title='', xlimv=2.0, ylimv=-2.0, climv=0.1)

    # TESTING
    if args.demo_centroid:
        xy0 = np.array([[2.724274e+00, -2.97643558474658384e+00], [2.182848335803861950e+00, -3.402075703300047049e+00],
                        [-3.92344420015747855e+00, -3.46379685512046676e+00],
                        [2.113057428073286115e+00, 4.546926452878380154e+00],
                        [2.302451831394968895e+00, 3.490762764967130671e+00],
                        [9.697986117896510994e-01, -3.680146892264366532e-01],
                        [-2.63263309004209349e+00, 3.755965455515874130e+00],
                        [-5.086571399355268586e-01, 6.176592495147481543e-02],
                        [2.340939735579837810e-01, -6.58868530075730297e-01],
                        [2.022524068028572763e-01, -9.540877874370963241e-01],
                        [-5.26791567374263323e-01, -1.98106312859869501e+00],
                        [-2.190395732290378206e+00, 1.471025339312911573e+00],
                        [-4.34140034587249300e+00, 3.233124827116466093e+00],
                        [-3.318352638435415169e+00, 3.265408160648248792e+00],
                        [5.851094832679919477e-01, 4.140675320548108829e+00],
                        [7.433178188089029081e-01, 2.955629726028079940e+00],
                        [3.962228141441888063e-01, 3.616045069395356748e+00],
                        [-8.943742198507069752e-01, 1.064360364521021785e+00],
                        [-1.62462739175278481e+00, 1.525904473532839534e+00],
                        [-1.414243508313497877e+00, 3.590822289317727289e+00],
                        [-1.32195642573613328e+00, 3.677056133764459389e+00],
                        [1.614826379443214721e+00, 1.007009093094507257e+00],
                        [7.617299309174689892e-01, 2.462751656633878738e+00],
                        [7.693535981790642353e-01, 2.483824209870186372e+00],
                        [3.032298585035444027e+00, -2.12417386010545479e+00],
                        [1.927324360676074599e+00, -5.527197521114308731e-01],
                        [-2.35081630380943020e+00, -8.41253275266633318e-01],
                        [-3.053078481905877162e+00, -2.697928957418050278e+00],
                        [-1.94578136887151674e+00, -2.79580097806940169e+00],
                        [-1.905567142224744881e+00, -2.767568539675566264e+00],
                        [-1.90429603092965615e+00, -2.76575277457979479e+00],
                        [-3.546114155348067953e+00, 1.320863649898669223e+00],
                        [-2.51316886058362298e+00, 1.311798238466580280e+00],
                        [-2.953651565222117803e+00, -5.855745758574000259e-01],
                        [-4.36822851543015040e+00, -9.65093701556713045e-01]])

        polygon = np.array([[-5., -5.], [-5., 5.], [5., 5.], [5., -5.]])
        xy, NL, KL, BL = delaunay_centroid_lattice_from_pts(xy0, polygon=polygon)
        display_lattice_2D(xy, BL, close=True)

        xy0 = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5],
                        [1, 1], [2, 1], [3, 1], [4, 1], [0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5],
                        [1, 2], [2, 2], [3, 2], [4, 2], [0.5, 2.5], [1.5, 2.5], [2.5, 2.5], [3.5, 2.5]])
        # xy0 = np.array([[1,0], [2,0],[3,0],[0.5,0.5],[1.5,0.5],[2.5,0.5],
        #                 [1,1], [2,1],[3,1],[0.5,1.5],[1.5,1.5],[2.5,1.5],
        #                 [1,2], [2,2],[3,2],[0.5,2.5],[1.5,2.5],[2.5,2.5]])

        tri = Delaunay(xy0)
        TRI = tri.vertices
        plt.triplot(xy0[:, 0], xy0[:, 1], TRI, 'go-');
        plt.show()
        BL = TRI2centroidBL(TRI)
        centxy = xyandTRI2centroid(xy0, TRI)

        display_lattice_2D(centxy, BL, close=False)
        plt.triplot(xy0[:, 0], xy0[:, 1], TRI, 'go-')
        for i in range(len(centxy)):
            plt.text(centxy[i, 0] + 0.01, centxy[i, 1] + 0.01, str(i))
        for i in range(len(xy0)):
            plt.text(xy0[i, 0] + 0.01, xy0[i, 1] + 0.01, str(i))
        plt.show()

        # Alternatively, make it like this:
        xy, NL, KL, BL = centroid_lattice_from_TRI(xy0, TRI)
        display_lattice_2D(xy, BL)

    if args.demo_polygons:
        xy = np.array(
            [[2.724217488627969974e+00, -2.97643984e+00], [2.182848335803861950e+00, -3.40207503300047049e+00],
             [-3.923444200157479855e+00, -3.463796855120466756e+00],
             [2.113057428073286115e+00, 4.546926452878380154e+00],
             [2.302451831394968895e+00, 3.490762764967130671e+00], [9.697986117896510994e-01, -3.68014682264366532e-01],
             [-2.632633090042079349e+00, 3.755965455515874130e+00],
             [-5.086571399355268586e-01, 6.17659249147481543e-02],
             [2.340939735579837810e-01, -6.588685300757304297e-01],
             [2.022524068028572763e-01, -9.54087787470963241e-01],
             [-5.267915673742963323e-01, -1.981063128598679501e+00],
             [-2.190395732290378206e+00, 1.47102533932911573e+00],
             [-4.341400345873249300e+00, 3.233124827116466093e+00],
             [-3.318352638435415169e+00, 3.26540816064248792e+00],
             [5.851094832679919477e-01, 4.140675320548108829e+00], [7.433178188089029081e-01, 2.955629726028079940e+00],
             [3.962228141441888063e-01, 3.616045069395356748e+00], [-8.943742198507069752e-01, 1.06436036452121785e+00],
             [-1.624627391735278481e+00, 1.525904473532839534e+00],
             [-1.414243508313497877e+00, 3.59082228931777289e+00],
             [-1.321956425073613328e+00, 3.677056133764459389e+00],
             [1.614826379443214721e+00, 1.007009093094507257e+00],
             [7.617299309174689892e-01, 2.462751656633878738e+00], [7.693535981790642353e-01, 2.483824209870186372e+00],
             [3.032298585035444027e+00, -2.124173860105345479e+00],
             [1.927324360676074599e+00, -5.52719752111430831e-01],
             [-2.350816308380943020e+00, -8.412532752668633318e-01],
             [-3.053078481905877162e+00, -2.6979289574180078e+00],
             [-1.945781306887151674e+00, -2.795800978086940169e+00]])
        polygon = np.array([[-5., -5.], [-5., 5.], [5., 5.], [5., -5.]])
        xy, NL, KL, BL = delaunay_centroid_lattice_from_pts(xy, polygon=polygon, check=False)
        display_lattice_2D(xy, BL, close=True, colorpoly=False)
        display_lattice_2D(xy, BL, close=True, colorpoly=True, viewmethod=False)

    if args.demo_haldane:
        xy = np.array([[-1.687049508094787598e+00, -3.986734628677368164e+00],
                       [2.830811403691768646e-02, -4.038495540618896484e+00],
                       [1.742334723472595215e+00, -4.06428839343261719e+00],
                       [-2.62065720558665039e+00, -3.53836846356235352e+00],
                       [-8.249260783195495605e-01, -3.4991567718505859e+00],
                       [8.473153114318847656e-01, -3.50249814987186172e+00],
                       [2.580250978469848633e+00, -3.48031938476562500e+00],
                       [-2.61470127105728906e+00, -2.51269888877686523e+00],
                       [-8.180426359176635742e-01, -2.5701206206665039e+00],
                       [8.893306255340576172e-01, -2.53642010688978173e+00],
                       [2.594556808471679688e+00, -2.55153298379907227e+00],
                       [-3.49462151527447852e+00, -1.99566042423242910e+00],
                       [-1.671632170677185059e+00, -1.9792637821220703e+00],
                       [-2.17138528823825391e-02, -2.01269245147705037e+00],
                       [1.768977165222167969e+00, -1.75879073143005371e+00],
                       [3.515128374099731445e+00, -2.06668496131969727e+00],
                       [-3.414126873016357422e+00, -1.3129720687866109e+00],
                       [-1.72077190876070801e+00, -1.06094861035786133e+00],
                       [-1.182176358997821808e-02, -1.0347372055057109e+00],
                       [1.703027367591857910e+00, -1.07166421413216309e+00],
                       [3.480890750885009766e+00, -1.06953888320922852e+00],
                       [-4.35105991365253906e+00, -5.41745126247060059e-01],
                       [-2.532985925674438477e+00, -4.9026458520892822e-01],
                       [-8.16166341304790527e-01, -4.84445780515707764e-01],
                       [8.684157133102416992e-01, -5.54679533729553223e-01],
                       [2.576100349426269531e+00, -4.87423539161681289e-01],
                       [4.365602970123291016e+00, -5.14180794570922852e-01],
                       [-4.30348348615537109e+00, 4.442375898361206055e-01],
                       [-2.545343399047851562e+00, 5.06929874201660156e-01],
                       [-8.23982536727551270e-01, 4.710277915000915527e-01],
                       [8.823772072792053223e-01, 4.382138550281524658e-01],
                       [2.635555982589721680e+00, 5.075737833976745605e-01],
                       [4.316730976104736328e+00, 4.415661394596099854e-01],
                       [-3.43612027682739258e+00, 9.245750308036804199e-01],
                       [-1.755252599716186523e+00, 9.85430836677512695e-01],
                       [-1.31688015354738235e-02, 1.022574186325073242e+00],
                       [1.778118729591369629e+00, 9.445834159851074219e-01],
                       [3.476804256439208984e+00, 9.436782598495483398e-01],
                       [-3.447193384170532227e+00, 1.98589217628112793e+00],
                       [-1.74849010394287109e+00, 1.997837901115417480e+00],
                       [-1.436182949692010880e-02, 1.96901965112963867e+00],
                       [1.709827899932861328e+00, 1.946695446968078613e+00],
                       [3.524006128311157227e+00, 2.004672527313232422e+00],
                       [-2.56480083938598633e+00, 2.477272033691406250e+00],
                       [-8.241441845893859863e-01, 2.47764205932171875e+00],
                       [8.585451245307922363e-01, 2.492717504501342773e+00],
                       [2.631927251815795898e+00, 2.499896287918090820e+00],
                       [-2.55175162734985352e+00, 3.505367040634155273e+00],
                       [-8.665223717689514160e-01, 3.49772286411000977e+00],
                       [8.820865750312805176e-01, 3.462744235992431641e+00],
                       [2.608837127685546875e+00, 3.434029817581176758e+00],
                       [-1.66453692054748535e+00, 3.924802541732788086e+00],
                       [5.434467922896146774e-03, 3.956029891967773438e+00],
                       [1.786471843719482422e+00, 3.994060754776000977e+00]])
        NL = np.array([[3, 4, 0, 0, 0, 0],
                       [5, 4, 0, 0, 0, 0], [5, 6, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0], [8, 0, 1, 0, 0, 0],
                       [9, 2, 1, 0, 0, 0], [10, 2, 0, 0, 0, 0],
                       [12, 3, 11, 0, 0, 0], [13, 4, 12, 0, 0, 0], [5, 14, 13, 0, 0, 0], [15, 6, 14, 0, 0, 0],
                       [16, 7, 0, 0, 0, 0], [7, 17, 8, 0, 0, 0], [8, 9, 18, 0, 0, 0],
                       [9, 19, 10, 0, 0, 0], [20, 10, 0, 0, 0, 0], [11, 21, 22, 0, 0, 0], [12, 23, 22, 0, 0, 0],
                       [23, 24, 13, 0, 0, 0], [25, 14, 24, 0, 0, 0],
                       [15, 26, 25, 0, 0, 0], [16, 27, 0, 0, 0, 0], [16, 28, 17, 0, 0, 0], [18, 17, 29, 0, 0, 0],
                       [18, 19, 30, 0, 0, 0], [19, 20, 31, 0, 0, 0],
                       [20, 32, 0, 0, 0, 0], [33, 21, 0, 0, 0, 0], [33, 34, 22, 0, 0, 0], [35, 34, 23, 0, 0, 0],
                       [24, 35, 36, 0, 0, 0], [36, 37, 25, 0, 0, 0],
                       [26, 37, 0, 0, 0, 0], [28, 27, 38, 0, 0, 0], [29, 28, 39, 0, 0, 0], [29, 40, 30, 0, 0, 0],
                       [31, 41, 30, 0, 0, 0], [42, 31, 32, 0, 0, 0],
                       [43, 33, 0, 0, 0, 0], [43, 34, 44, 0, 0, 0], [44, 35, 45, 0, 0, 0], [46, 45, 36, 0, 0, 0],
                       [37, 46, 0, 0, 0, 0], [39, 38, 47, 0, 0, 0],
                       [40, 48, 39, 0, 0, 0], [49, 40, 41, 0, 0, 0], [50, 41, 42, 0, 0, 0], [51, 43, 0, 0, 0, 0],
                       [44, 52, 51, 0, 0, 0], [45, 53, 52, 0, 0, 0],
                       [46, 53, 0, 0, 0, 0], [47, 48, 0, 0, 0, 0], [49, 48, 0, 0, 0, 0], [50, 49, 0, 0, 0, 0]])
        KL = np.array(
            [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0]])

        # xy = np.array([[-9.381941874331420905e-01,-1.625000000000000000e+00],
        #     [7.938566201357353247e-01,-1.625000000000000000e+00],
        #     [-1.804219591217580687e+00,-1.125000000000000000e+00],
        #     [-7.216878364870349394e-02,-1.125000000000000000e+00],
        #     [1.659882023920173699e+00,-1.125000000000000000e+00],
        #     [-1.804219591217580687e+00,-1.250000000000000000e-01],
        #     [-7.216878364870349394e-02,-1.250000000000000000e-01],
        #     [1.659882023920173699e+00,-1.250000000000000000e-01],
        #     [-9.381941874331420905e-01,3.750000000000000000e-01],
        #     [7.938566201357353247e-01,3.750000000000000000e-01],
        #     [-9.381941874331420905e-01,1.375000000000000000e+00],
        #     [7.938566201357353247e-01,1.375000000000000000e+00],
        #     [-7.216878364870349394e-02,1.875000000000000000e+00] ])
        # NL = np.array([[2,3,0],[3,4,0],[0,5,0],[0,1,6],[1,7,0],[2,8,0],[3,8,9],[4,9,0],[5,6,10],
        #     [6,7,11],[8,12,0],[9,12,0],[10,11,0]])
        # KL = np.array([[1,1,0],[1,1,0],[1,1,0],[1,1,1],[1,1,0],[1,1,0],[1,1,1],[1,1,0],[1,1,1],
        #     [1,1,1],[1,1,0],[1,1,0],[1,1,0]])
        # BL2 = np.array([ [4,7],[1,3],[6,8],[10,12],[6,9],[1,4],[0,2],[9,11],[3,6],[8,10],[2,5],[11,12],
        #     [0,3],[5,8],[7,9] ])

        BL = NL2BL(NL, KL)
        # NLNNN, KLNNN =  calc_NLNNN_and_KLNNN(xy,BL,NL,KL)
        # BLNNN = NL2BL(NLNNN,np.abs(KLNNN))
        # display_lattice_2D(xy, BL, title='haldane NNN plot',colorz=True,colorpoly=True,close=True, BLNNN=BLNNN)
        eigvect, eigval, matrix = save_normal_modes_haldane('none', xy, NL, KL, BL)
        plt.imshow(np.imag(matrix), interpolation='none')
        plt.show()

    if args.demo_unique_rows:
        R = np.array([[0.00000000e+00, 0.00000000e+00],
                      [1.12763114e+00, 4.10424172e-01],
                      [-6.04419663e-01, -5.89575828e-01],
                      [1.12763114e+00, -1.58957583e+00],
                      [9.84807753e-01, -1.73648178e-01],
                      [-7.47243055e-01, 8.26351822e-01],
                      [1.73205081e+00, 1.00000000e+00],
                      [1.22464680e-16, 2.00000000e+00],
                      [9.84807753e-01, 1.82635182e+00],
                      [1.73205081e+00, -1.00000000e+00],
                      [0.00000000e+00, 0.00000000e+00],
                      [1.12763114e+00, 4.10424172e-01],
                      [-6.04419663e-01, -5.89575828e-01],
                      [1.12763114e+00, -1.58957583e+00],
                      [9.84807753e-01, -1.73648178e-01],
                      [-7.47243055e-01, 8.26351822e-01],
                      [1.73205081e+00, 1.00000000e+00],
                      [1.22464680e-16, 2.00000000e+00],
                      [9.84807753e-01, 1.82635182e+00],
                      [1.73205081e+00, -1.00000000e+00],
                      [1.22464680e-16, 2.00000000e+00],
                      [1.12763114e+00, 2.41042417e+00],
                      [-6.04419663e-01, 1.41042417e+00],
                      [1.12763114e+00, 4.10424172e-01],
                      [9.84807753e-01, 1.82635182e+00],
                      [-7.47243055e-01, 2.82635182e+00],
                      [1.73205081e+00, 3.00000000e+00],
                      [2.44929360e-16, 4.00000000e+00],
                      [9.84807753e-01, 3.82635182e+00],
                      [1.73205081e+00, 1.00000000e+00],
                      [-1.73205081e+00, 1.00000000e+00],
                      [-6.04419663e-01, 1.41042417e+00],
                      [-2.33647047e+00, 4.10424172e-01],
                      [-6.04419663e-01, -5.89575828e-01],
                      [-7.47243055e-01, 8.26351822e-01],
                      [-2.47929386e+00, 1.82635182e+00],
                      [4.44089210e-16, 2.00000000e+00],
                      [-1.73205081e+00, 3.00000000e+00],
                      [-7.47243055e-01, 2.82635182e+00],
                      [6.66133815e-16, 1.33226763e-15],
                      [-1.73205081e+00, 3.00000000e+00],
                      [-6.04419663e-01, 3.41042417e+00],
                      [-2.33647047e+00, 2.41042417e+00],
                      [-6.04419663e-01, 1.41042417e+00],
                      [-7.47243055e-01, 2.82635182e+00],
                      [-2.47929386e+00, 3.82635182e+00],
                      [5.66553890e-16, 4.00000000e+00],
                      [-1.73205081e+00, 5.00000000e+00],
                      [-7.47243055e-01, 4.82635182e+00],
                      [7.88598495e-16, 2.00000000e+00]])

        sizevals = np.arange(len(R)) + 5
        colorvals = np.linspace(0, 1, len(R))
        plt.scatter(R[:, 0], R[:, 1], s=sizevals, c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()
        Rout = dh.unique_rows_threshold(R, 0.001)
        sizevals = np.arange(len(Rout)) + 5
        colorvals = np.linspace(0, 1, len(Rout))
        plt.scatter(Rout[:, 0], Rout[:, 1], s=sizevals, c=colorvals, cmap='afmhot', alpha=0.2)
        plt.show()

    if args.demo_chern:
        import scipy.linalg as la

        check = False
        # Parameters
        shape = 'hexagon'
        NH = 9
        NV = 9
        delta = np.pi * 120. / 180.
        eta = 0.
        rot = 0.
        periodicBC = False
        phi = 0.
        theta = 0.
        viewmethod = False
        compute_type = 'skeleton'

        # Make Lattice
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = \
            makeL.generate_honeycomb_lattice(shape, NH, NV, delta, phi, eta=eta, rot=theta, periodicBC=periodicBC)

        polygons = extract_polygons_lattice(xy, BL, NL, KL, viewmethod=viewmethod, PVxydict=PVxydict)
        PolyPC = polygons2PPC(polygons)
        print 'polygons = ', polygons
        NLNNN, KLNNN = calc_NLNNN_and_KLNNN(xy, BL, NL, KL, PVx=PVx, PVy=PVy)
        print 'KLNNN = ', KLNNN
        display_lattice_2D(xy, BL, NL=NL, KL=KL, BLNNN=[], NLNNN=NLNNN, KLNNN=KLNNN, PVxydict={}, PVx=PVx, PVy=PVy,
                           bs='none', title='', xlimv=None, ylimv=None, climv=0.1, colorz=True,
                           close=True, colorpoly=False, viewmethod=False, labelinds=False)
        print 'NLNNN = ', NLNNN

        datadir = './test_chern_haldane/'
        epsilon = 0.1

        # COMPUTE DYNAMICAL MATRIX AND EIGENVALUES
        if compute_type == 'skeleton':
            eigvect, eigval, DM = save_normal_modes_haldane(datadir, xy, NL, KL, BL, epsilon=epsilon,
                                                            save_ims=True, PVx=PVx, PVy=PVy)
        elif compute_type == 'barebones':
            NLNNN, KLNNN = calc_NLNNN_and_KLNNN(xy, BL, NL, KL, PVx=PVx, PVy=PVy)

            NP, NN = np.shape(NL)
            M1 = np.zeros((NP, NP), dtype=complex)
            M2 = np.zeros((NP, NP), dtype=complex)

            print 'Constructing Haldane dynamical matrix...'
            eps = 1e-14
            for i in range(NP):
                for nn in range(NN):
                    ni = NL[i, nn]  # the number of the gyroscope i is connected to (particle j)
                    k = KL[i, nn]  # true connection?

                    if abs(k) > eps:
                        # (psi components)
                        M1[i, ni] += 1  # psi_j

            NNNN = np.shape(KLNNN)[1]
            for i in range(NP):
                for nn in range(NNNN):
                    ni = NLNNN[i, nn]
                    k = KLNNN[i, nn]

                    if k > eps:
                        # There is a true NNN connection, so update dynamical matrix
                        M2[i, ni] += 1j * epsilon
                    if k < -eps:
                        # There is a true NNN connection, so update dynamical matrix
                        # print 'i = ', i, ' ni= ', ni
                        M2[i, ni] -= 1j * epsilon
                        # print 'M2[0,:] = ', M2[:,0]

            DM = M1 + M2
            eigval, eigvect = np.linalg.eig(DM)

        ################################
        if check:
            for jjj in range(len(DM)):
                row = DM[jjj]
                print 'NN = ', np.where(row == 1)[0]
                print '+NNN = ', np.where(row == 1j * epsilon)[0]
                print '-NNN = ', np.where(row == -1j * epsilon)[0]

            plot_complex_matrix(DM, name='Dynamical Matrix', outpath='none', show=True, close=True)
            display_lattice_2D(xy, BL, NL=NL, KL=KL, BLNNN=[], NLNNN=NLNNN, KLNNN=KLNNN, PVxydict={},
                               PVx=PVx, PVy=PVy, bs='none', title='', xlimv=None, ylimv=None, climv=0.1,
                               colorz=True, close=False, colorpoly=False, viewmethod=False, labelinds=False)
            for ii in range(len(xy)):
                plt.text(xy[ii, 0], xy[ii, 1], str(ii))
            plt.show()
        ################################

        U = eigvect.transpose()
        U1 = la.inv(U)
        D = np.zeros((len(U), len(U)), dtype=complex)
        for ii in range(len(eigval)):
            ev = eigval[ii]
            D[ii, ii] = ev

        M = copy.deepcopy(D)
        omegac = 0.
        M[M.real < omegac] = 0
        M[M.real > omegac] = 1
        P = np.dot(U, np.dot(M, U1))

        # CHECK
        if check:
            plot_complex_matrix(D, name='Eigenvalue Matrix', outpath='none', show=True, close=True)
            plot_complex_matrix(M, name='Cut-Off Eigval Matrix', outpath='none', show=True, close=True)
            plot_complex_matrix(P, name='Projector', outpath='none', show=True, close=True)
            # P is hermitian
            plot_complex_matrix(P - P.transpose().conjugate(), name=r'$P -P^{dagger}$', outpath='none',
                                show=True, close=True)
            # Check that P_ij is correct by transforming an eigenvector to itself or to zero
            inds2proj = [1, round(len(eigval) * 0.5), round(len(eigval) * 0.96)]
            plot_projection_complex_vects(P, eigvect, eigval, inds2proj, fig='auto', outpath='none',
                                          show=True, close=True)

        # h = np.zeros((len(P),len(P),len(P)),dtype=complex)
        # for j in range(len(P)):
        #     for k in range(len(P)):
        #         for l in range(len(P)):
        #             h[j,k,l] = P[j,k]*P[k,l]*P[l,j] - P[j,l]*P[l,k]*P[k,j]
        #
        # h *= 12*np.pi*1j

        # # CHECK h
        # if check:
        #     plot_complex_matrix(h[0],name='h[0]',outpath='none',show=True,close=True)
        #     plot_complex_matrix(np.sum(h,axis=1),name='$(\partial h)_{kl} \equiv \sum_{j} h_{jkl}$',
        #                         outpath='none',show=True,close=True)

        # Divide plane into three, A->B->C counterclockwise
        method = 'equal'
        if method == 'equal':
            theta1 = -np.pi / 6.
            polygon1 = 6. * np.array([[0., 0.], [np.cos(theta1), np.sin(theta1)], [0., 1.]])
            theta2 = np.pi * (7. / 6.)
            teps = 0.001
            polygon2 = 6. * np.array([[0., 0.], [0., 1.], [np.cos(theta2), np.sin(theta2)]])
            polygon3 = 6. * np.array([[0., 0.], [np.cos(theta2 + teps), np.sin(theta2 + teps)],
                                      [np.cos(theta1 - teps), np.sin(theta1 - teps)]])
            reg1 = inds_in_polygon(xy, polygon1)
            reg2 = inds_in_polygon(xy, polygon2)
            reg3 = inds_in_polygon(xy, polygon3)
        else:
            # First divide into two
            NP = len(xy)
            reg1 = np.where(np.logical_and(xy[:, 0] > -0.34, xy[:, 1] > -0.45))[0]
            regB = np.setdiff1d(np.arange(NP), reg1)
            reg2IND = np.where(xy[regB, 1] > -0.45)[0]
            reg2 = regB[reg2IND]
            reg3 = np.setdiff1d(regB, reg2)

        # CHECK division
        if check:
            print 'reg1 = ', reg1
            print 'reg2 = ', reg2
            print 'reg3 = ', reg3
            print 'polygon1 = ', polygon1
            print 'polygon2 = ', polygon2
            print 'polygon3 = ', polygon3
            ax = plt.gca()
            ax.scatter(xy[reg1, 0], xy[reg1, 1], c='r', s=400, marker='o', alpha=0.5, label='reg1')
            ax.scatter(xy[reg2, 0], xy[reg2, 1], c='g', s=300, marker='^', alpha=0.5, label='reg2')
            ax.scatter(xy[reg3, 0], xy[reg3, 1], c='b', s=200, marker='s', alpha=0.5, label='reg3')
            patchList = []
            patchList.append(patches.Polygon(polygon1, color='r'))
            patchList.append(patches.Polygon(polygon2, color='g'))
            patchList.append(patches.Polygon(polygon3, color='b'))
            p = PatchCollection(patchList, cmap=cm.jet, alpha=0.2)
            colors = np.linspace(0, 1, 3)[::-1]
            p.set_array(np.array(colors))
            ax.add_collection(p)
            plt.legend()
            plt.title('Division of lattice into 3 parts')
            plt.show()

        print 'Summing up h values...'
        dmyi = 0
        nu = 0. + 0. * 1j
        nu_sum12 = np.zeros((len(reg1), len(reg2)), dtype='complex')
        nu_sum1 = np.zeros(len(reg1), dtype='complex')
        nu_sum2 = np.zeros(len(reg2), dtype='complex')
        nu_sum3 = np.zeros(len(reg3), dtype='complex')
        nu_last12 = 0.
        nu_last1 = 0.
        jind = 0
        kind = 0
        for j in reg1:
            kind = 0
            for k in reg2:
                for l in reg3:
                    print 'jkl = ', j, ',', k, ',', l
                    # nu += h[j,k,l]
                    nu += 12 * np.pi * 1j * (P[j, k] * P[k, l] * P[l, j] - P[j, l] * P[l, k] * P[k, j])
                    dmyi += 1
                    # nu_sum12[jind,kind] = nu - nu_last12
                    # nu_last12 = nu
                    # kind += 1
                    # nu_sum1[jind] = nu - nu_last1
                    # nu_last1 = nu
                    # jind += 1

        print 'nu = ', nu

        # Display partial sums
        plot_complex_matrix(nu_sum12, name='$\sum_l h_{jkl}$ (regA,B)', outpath='none', show=True, close=True)
        plt.plot(np.arange(len(nu_sum1)), nu_sum1)
        plt.title('$\sum_{kl} h_{jkl}$ Each sum separate (leaving regA inds)')

    if args.demo_BL:
        # Parameters
        shape = 'hexagon'
        NH = 2
        NV = 2
        delta = np.pi * 120. / 180.
        eta = 0.
        rot = 0.
        periodicBC = True
        phi = 0.
        theta = 0.
        viewmethod = True

        # Make lattice
        xy, NL, KL, BL, LVUC, LV, UC, PVxydict, PVx, PVy, lattice_exten = \
            makeL.generate_honeycomb_lattice(shape, NH, NV, delta, phi, eta=eta, rot=theta, periodicBC=periodicBC)

        # Get polygons
        polygons = extract_polygons_lattice(xy, BL, NL, KL, viewmethod=viewmethod, PVxydict=PVxydict)
        PolyPC = polygons2PPC(polygons)
        print 'polygons = ', polygons

        # Next nearest neighbors
        NLNNN, KLNNN = calc_NLNNN_and_KLNNN(xy, BL, NL, KL, PVx=PVx, PVy=PVy)
        print 'NLNNN = ', NLNNN
        print 'KLNNN = ', KLNNN
        display_lattice_2D(xy, BL, NL=NL, KL=KL, BLNNN=[], NLNNN=NLNNN, KLNNN=KLNNN, PVxydict={}, PVx=PVx, PVy=PVy,
                           bs='none', title='', xlimv=None, ylimv=None, climv=0.1, colorz=True,
                           close=True, colorpoly=True, viewmethod=False, labelinds=False)

    if args.demo_dislocation:
        NH = 9
        NV = 9
        shape = 'hexagon'
        pt = np.array([[0., -4.]])
        xy, NL, KL, BL, lattice_exten = makeL.generate_dislocated_hexagonal_lattice(shape, NH, NV, pt, check=False,
                                                                                    Bvecs=[])

    if args.demo_triangulation:
        # import lepm.lattice_elasticity as le
        # import numpy as np
        xy = np.random.rand(20, 2)
        NL, KL, BL, BM = delaunay_lattice_from_pts(xy, trimbound=True, target_z=-1, max_bond_length=-1,
                                                   thres=4.0, zmethod='random', check=True)
        TRI = BL2TRI(BL, xy)
        boundary = extract_boundary(xy, NL, KL, BL)

        # Add some spurious points
        xyadd = np.random.rand(5, 2) + np.array([1., 1.])
        xy = np.vstack((xy, xyadd))

        plt.triplot(xy[:, 0], xy[:, 1], TRI, 'g.-')
        plt.plot(xy[:, 0], xy[:, 1], 'b.')
        closedb = np.hstack((boundary, boundary[0]))
        plt.plot(xy[closedb, 0], xy[closedb, 1], 'k-')
        plt.show()

    if args.demo_distance_from_boundary:
        '''Demonstrate calculation of distance from a boundary of a collection of 2d coordinates'''
        xy0 = np.meshgrid(np.arange(5), np.arange(5))
        xy = np.dstack((xy0[0].ravel(), xy0[1].ravel()))[0]
        xy = np.vstack((xy, np.random.rand(1000, 2) * 4.0))
        boundary = np.array([0, 1, 2, 3, 4, 9, 14, 19, 24,
                             23, 22, 21, 20, 15, 10, 5])

        # Show that boundary is identified
        plt.plot(xy[:, 0], xy[:, 1], '.')
        plt.plot(xy[boundary, 0], xy[boundary, 1], 'ro-')
        for ii in range(len(boundary)):
            plt.text(xy[boundary[ii], 0] + 0.2, xy[boundary[ii], 1], str(ii))
        plt.show()

        # Use distance from boundary without interpolating boundary
        dists = distance_from_boundary(xy, boundary, interp_n=None)
        plt.scatter(xy[:, 0], xy[:, 1], c=dists)
        plt.colorbar()
        plt.show()

        # Use distance from boundary with interpolated boundary
        boundary = np.array([0, 4, 24, 20])

        dists = distance_from_boundary(xy, boundary, interp_n=1)
        plt.scatter(xy[:, 0], xy[:, 1], c=dists)
        plt.colorbar()
        plt.show()
