import numpy as np
import lepm.lattice_elasticity as le
from lepm.lattice_class import Lattice
from lepm.gyro_lattice_class import GyroLattice
import lepm.plotting.plotting as leplt
from lepm.chern.kchern_gyro import KChernGyro
from lepm.chern.kchern import KChern
import matplotlib.pyplot as plt
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov
import copy
import dataio as dio
from lepm.haldane.haldane_lattice_class import HaldaneLattice
import glob

'''See if the hamiltonian of the gyro lattice is (block) Hermitian or can be cast in some form like that.
Look at projector, squared projector, etc
'''


def plot_spectrum_hist(eigval, fig):
    ax = fig.add_axes([0.4, 0.83, 0.2, 0.14])
    ax.hist(eigval, bins=120, color=lecmaps.green(), lw=0)
    ax.set_xlabel('frequency, $\omega/\Omega_g$')
    ax.yaxis.set_ticks([])
    ylim = ax.get_ylim()
    ax.plot([eigval[ii], eigval[ii]], [ylim[0], ylim[1]], 'r-')
    ax.set_ylim(ylim)
    return ax


x0, x1 = 0.0, 0.5
sz = 0.5
wcbar = 0.3
ax1loc = [x0, 0.2, sz, sz]
ax2loc = [x1, 0.2, sz, sz]
cax1loc = [x0 + (sz - wcbar) * 0.5, 0.1, wcbar, 0.025]
cax2loc = [x1 + (sz - wcbar) * 0.5, 0.1, wcbar, 0.025]
fr = 30
cmap = lecmaps.ensure_cmap('rwb0')
outdir = '/Users/npmitchell/Dropbox/Soft_Matter/PAPER/review_paper/'
outdir += 'test_symmetries_ari_largesystem/'
# IrvineLabCodes/noah/noah/lepm/lepm/test_symmetries/'

# open boundary condition hexagonal
latticetop = 'iscentroid'
haldane = False
if haldane:
    outdir += 'haldane_'
nn = 64
npload = 64
outdir += latticetop + '_nn{0:03d}'.format(nn)
if npload > 0:
    outdir += '_npload{0:03d}'.format(npload) + '/'
else:
    outdir += '/'

dio.ensure_dir(outdir)
show, close = False, False

# periodic boundary amorphous
# latticetop = 'randorg_gammakick0p50_cent'
# nn = 8
# npload = 64
spreadingt = 0.

if haldane:
    lp = {'LatticeTop': latticetop,
          'shape': 'square',
          'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
          'delta_lattice': '0.667',
          'phi_lattice': '0.000',
          'NH': nn,
          'NV': nn,
          'NP_load': nn,
          'theta': 0.,
          'theta_twist': 0.,
          'phi_twist': 0.,
          'eta': 0.,
          'x1': 0.,
          'x2': 0.,
          'x3': 0.,
          'z': 0.,
          'source': 'hexner',
          'loadlattice_number': 01,
          'check': False,
          'Ndefects': 0,
          'Bvec': 'W',
          'dislocation_xy': '0/0',
          'target_z': 0.,
          'make_slit': False,
          'cutz_method': 'none',
          'cutLfrac': 0.0,
          'conf': 01,
          'subconf': 01,
          'periodicBC': npload > 0,
          'loadlattice_z': '001',
          'alph': 1.0,
          'origin': np.array([0., 0.]),
          't1': -1.0,
          't2': 0.1,
          'pin': 0.0,
          't2funcdescription': None,
          'V0_pin_gauss': 0.0,
          'V0_spring_gauss': 0.,
          'percolation_density': 0.5,
          'save_pinning_to_hdf5': True,
          'spreading_time': spreadingt,
          'kicksz': -1.5
          }
else:
    lp = {'LatticeTop': latticetop,
          'shape': 'square',
          'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
          'delta_lattice': '0.667',
          'phi_lattice': '0.000',
          'NH': nn,
          'NV': nn,
          'NP_load': nn,
          'theta': 0.,
          'eta': 0.,
          'x1': 0.,
          'x2': 0.,
          'x3': 0.,
          'z': 0.,
          'source': 'hexner',
          'loadlattice_number': 01,
          'check': False,
          'Ndefects': 0,
          'Bvec': 'W',
          'dislocation_xy': '0/0',
          'target_z': 0.,
          'make_slit': False,
          'cutz_method': 'none',
          'cutLfrac': 0.0,
          'conf': 01,
          'subconf': 01,
          'periodicBC': npload > 0,
          'loadlattice_z': '001',
          'alph': 1.0,
          'origin': np.array([0., 0.]),
          'Omk': -1.0,
          'Omg': -1.0,
          'V0_pin_gauss': 0.0,
          'V0_spring_gauss': 0.,
          'percolation_density': 0.5,
          'save_pinning_to_hdf5': True,
          'spreading_time': spreadingt,
          'kicksz': -1.5
          }

lat = Lattice(lp=lp)
lat.load()
lat.plot_BW_lat(save=False, close=close)
plt.savefig(outdir + 'lattice.png')
plt.clf()

if haldane:
    glat = HaldaneLattice(lat, lp=lp)
else:
    glat = GyroLattice(lat, lp=lp)
    print 'Omg = ', glat.Omg

mat = glat.calc_matrix(attribute=True)

leplt.plot_complex_matrix(mat, name='Dynamical matrix', show=show, close=close, cmap=cmap)
plt.savefig(outdir + 'dynamical_matrix.png')
plt.clf()

matdag = mat.T.conj()
leplt.plot_complex_matrix(mat - matdag)
leplt.plot_complex_matrix(mat - matdag, name=r'$D - D^{\dagger}$', show=show, close=close, cmap=cmap)
plt.savefig(outdir + 'DminusDdagger.png')
plt.clf()

# Check orthogonality of eigenvectors
eigvect = glat.get_eigvect()
leplt.plot_complex_matrix(eigvect, name=r'Eigenvectors at $k=(0, 0)$', show=show, close=close, cmap=cmap)
plt.savefig(outdir + 'eigenvectors_gammapt.png')
plt.clf()

eigTeig = np.dot(eigvect.T, eigvect)
leplt.plot_complex_matrix(eigTeig, name=r'$\vec{e} \cdot \vec{e}$', show=show, cmap=cmap)

eigdeig = np.dot(np.conjugate(eigvect).T, eigvect)
leplt.plot_complex_matrix(eigdeig, name=r'$\vec{e}^\dagger \cdot \vec{e}$', show=show, close=close, cmap=cmap)
plt.savefig(outdir + 'eigdeig_gammapt.png')
plt.clf()

eigdeig = np.dot(np.conjugate(eigvect).T, eigvect) - np.diag(np.ones_like(eigvect[:, 0]))
leplt.plot_complex_matrix(eigdeig, name=r'$\vec{e}^\dagger \cdot \vec{e} - 1$', show=show, close=close,
                          climvs=[[-5e-2, 5e-2], [-5e-2, 5e-2]], cmap=cmap)
plt.savefig(outdir + 'eigdeig_minus1.png')
plt.clf()

# Check projector
lp['basis'] = 'psi'
lat = Lattice(lp=lp)
lat.load()
if haldane:
    hlat = HaldaneLattice(lat, lp=lp)
    kchern = KChern(hlat)
else:
    glat = GyroLattice(lat, lp=lp)
    kchern = KChernGyro(glat)

for series in ['gammapt', 'kpt']:
    if series == 'gammapt':
        # Plot the projectors
        proj, mat, eigval, eigvect = kchern.get_projector(kvec=np.array([0.0, 0.0]))
    if series == 'kpt':
        ####################################
        # Redo at a nonzero wavenumber
        ####################################
        # Plot the eigenvectors at kpt
        proj, mat, eigval, eigvect = kchern.get_projector(kvec=np.array([0.5, 0.5]))

    if haldane:
        eigval = np.real(eigval)
    else:
        eigval = np.real(1j * eigval)

    leplt.plot_complex_matrix(eigvect.T, name=r'Eigenvectors at $k=(1/2, 1/2)$', show=show, close=close, cmap=cmap)
    plt.savefig(outdir + 'eigenvectors_' + series + '.png')
    eigdeig = (np.dot(np.conjugate(eigvect).T, eigvect) - np.diag(np.ones_like(eigvect[:, 0]))).T
    name = r'$\vec{e}^\dagger \cdot \vec{e} - 1$ at $k=(1/2, 1/2)$'
    leplt.plot_complex_matrix(eigdeig, name=name, show=show, close=close, cmap=cmap)
    plt.savefig(outdir + 'eigenvectors_' + series + '_minus1.png')
    plt.clf()

    # Now that these are formed, plot P^2 and other things about the projector
    print 'proj = ', proj
    print 'np.shape(proj) = ', np.shape(proj)

    done = len(glob.glob(outdir + 'proj/' + series + '/proj_band*.png')) == len(proj)
    if not done:
        maxsumpp2 = 0
        for (pp, ii) in zip(proj, range(len(proj))):
            fig, axes, caxes = leplt.plot_complex_matrix(pp, name=r'projector $P$ for band ' + str(ii),
                                                         show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ' + outdir + 'proj/' + series + '/'
            dio.ensure_dir(outdir + 'proj/' + series + '/')
            plt.savefig(outdir + 'proj/' + series + '/proj_band{0:03d}'.format(ii) + '.png')

            # check p**2 - p
            pp2 = np.dot(pp, pp) - pp
            fig, axes, caxes = leplt.plot_complex_matrix(pp2, name=r'$P^2 - P$ for band ' + str(ii),
                                                         show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ', outdir
            dio.ensure_dir(outdir + 'proj_p2minusp/' + series + '/')
            plt.savefig(outdir + 'proj_p2minusp/' + series + '/p2minusp_{0:03d}_'.format(ii) + series + '.png')
            plt.clf()

            if ii == 0:
                sumproj = copy.deepcopy(pp)
            else:
                sumproj += pp

            sumstr = 'bands0through{0:03d}'.format(ii)

            # Check if the sum of projectors squares to itself or not
            dio.ensure_dir(outdir + 'sumproj_p2minusp/' + series + '/')
            pp2 = np.dot(sumproj, sumproj) - sumproj
            name = r'$P^2 - P$ for projecting onto $\Sigma_{i}^N$ band$_i$, N=' + str(ii)
            fig, axes, caxes = leplt.plot_complex_matrix(pp2, name=name, show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ', outdir
            dio.ensure_dir(outdir + 'sumproj_p2minusp/' + series + '/')
            plt.savefig(outdir + 'sumproj_p2minusp/' + series + '/sumproj_p2minusp_' + sumstr)
            plt.clf()

            # Check that the sum of projectors takes eigvects to themselves or zero
            dio.ensure_dir(outdir + 'sumproj/' + series + '/')
            prod = sumproj
            name = r'$P$ for projecting onto $\Sigma_{i}^N$ band$_i$, N=' + str(ii)
            fig, axes, caxes = leplt.plot_complex_matrix(prod, name=name, show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ', outdir
            dio.ensure_dir(outdir + 'sumproj/' + series + '/')
            plt.savefig(outdir + 'sumproj/' + series + '/sumproj_' + sumstr)
            plt.clf()

            # Check that the sum of projectors takes eigvects to themselves or zero
            dio.ensure_dir(outdir + 'sumproj_dot_e/' + series + '/')
            prod = np.dot(sumproj, eigvect.T).T
            name = r'$P \cdot \vec{e}$ for projecting onto $\Sigma_{i}^N$ band$_i$, N=' + str(ii)
            name += '\n(cols are eigvects)'
            fig, axes, caxes = leplt.plot_complex_matrix(prod, name=name, show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ', outdir
            dio.ensure_dir(outdir + 'sumproj_dot_e/' + series + '/')
            plt.savefig(outdir + 'sumproj_dot_e/' + series + '/sumproj_dot_eigvect_' + sumstr)
            plt.clf()

            # Check that the sum of projectors takes eigvects to themselves or zero
            dio.ensure_dir(outdir + 'sumproj_dot_e_minus_e/' + series + '/')
            prod = np.dot(sumproj, eigvect.T).T - eigvect
            name = r'$P \cdot \vec{e} -\vec{e}$ for projecting onto $\Sigma_{i}^N$ band$_i$, N=' + str(ii)
            # name += '\n(cols are eigvects)'
            fig, axes, caxes = leplt.plot_complex_matrix(prod, name=name, show=show, close=close, cmap=cmap,
                                                         ax1loc=ax1loc, ax2loc=ax2loc, cax1loc=cax1loc, cax2loc=cax2loc)
            ax = plot_spectrum_hist(eigval, fig)
            print 'saving to ', outdir
            dio.ensure_dir(outdir + 'sumproj_dot_e_minus_e/' + series + '/')
            plt.savefig(outdir + 'sumproj_dot_e_minus_e/' + series +
                        '/sumproj_dot_eigvect_minus_e_' + sumstr)
            plt.clf()

            # Save sumproj for later checking if it squares to unity
            pp2 = np.dot(sumproj, sumproj) - sumproj
            maxsumpp2 = max(maxsumpp2, np.max(np.abs(pp2)))
            plt.clf()

    # Make movies
    imgname = outdir + 'proj/' + series + '/proj_band'
    movname = outdir + 'proj_' + series
    lemov.make_movie(imgname, movname, indexsz='03', framerate=fr)

    imgname = outdir + 'proj_p2minusp/' + series + '/p2minusp_'
    movname = outdir + 'proj_p2minusp_' + series
    lemov.make_movie(imgname, movname, indexsz='03', framerate=fr)

    imgname = outdir + 'sumproj_dot_e/' + series + '/sumproj_dot_eigvect_bands0through'
    movname = outdir + 'sumproj_dot_e_' + series
    lemov.make_movie(imgname, movname, indexsz='03', framerate=fr)

    imgname = outdir + 'sumproj_dot_e_minus_e/' + series + '/sumproj_dot_eigvect_minus_e_bands0through'
    movname = outdir + 'sumproj_dot_e_minus_e_' + series
    lemov.make_movie(imgname, movname, indexsz='03', framerate=fr)

    imgname = outdir + 'sumproj_p2minusp/' + series + '/sumproj_p2minusp_bands0through'
    movname = outdir + 'sumproj_p2minusp_' + series
    lemov.make_movie(imgname, movname, indexsz='03', framerate=fr)

# Notes:
# P^2 -P DOES equal zero, and states are not orthogonal for both disordered and clean lattices
