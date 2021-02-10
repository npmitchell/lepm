from pympler import asizeof
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import lepm.plotting.plotting as leplt
import lepm.plotting.colormaps as lecmap
import scipy
from lepm.lattice_class import Lattice
from lepm.haldane.haldane_lattice_class import HaldaneLattice
import lepm.timing as timing
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import dataio as dio
import cPickle as pkl

"""Test the memory footprint and timing of diagonalizing test dynamical matrices"""

nvec = np.array([5, 10, 15, 20, 25])  # , 30, 35])
iterations = 1
eigfrac = 1.0
alpha = 0.3
mksz = 3
outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/computation/eigval_decomp_test/'
dio.ensure_dir(outdir)
t_la = []
t_sla = []
t_lil = []
t_csr = []
sz_mat = []
sz_lil = []
sz_csr = []
nnlist = []

for nh in nvec:
    try:
        with open(outdir + 'results_nh' + str(int(nh)) + 'pkl', "rb") as fn:
            res = pkl.load(fn)

        t_la.append(res['s_la'])
        t_sla.append(res['s_sla'])
        t_lil.append(res['s_lil'])
        t_csr.append(res['s_csr'])

        sz_mat.append(res['sz_mat'])
        sz_lil.append(res['sz_lil'])
        sz_csr.append(res['sz_csr'])

        nnlist.append(res['nn'])

    # if iterations == 1:
    except IOError:
        lp = {'LatticeTop': 'hexagonal',
              'shape': 'square',
              'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
              'delta_lattice': '0.667',
              'phi_lattice': '0.000',
              'NH': nh,
              'NV': nh,
              'periodicBC': False,
              'pin': 0.0,
              }

        lat = Lattice(lp)
        lat.load()
        nn = len(lat.xy[:, 0])
        hlat = HaldaneLattice(lat, lp)
        mat = hlat.get_matrix()
        szmat = asizeof.asizeof(mat)
        print '\nmat -->', szmat
        # timing.timefunc(hlat.eig_vals_vects, matrix=mat, attribute=False)

        print 'numpy.linalg.eig()'
        rla, s_la, l_la = timing.timefunc(la.eig, mat)
        print 'scipy.linalg.eig()'
        rsla, s_sla, l_sla = timing.timefunc(sla.eig, mat, iterations=iterations)

        mat_lil = scipy.sparse.lil_matrix(mat)
        # mat_lil = hlat.get_matrix(sparse=True)
        szlil = asizeof.asizeof(mat_lil)
        print '\nmat_lil -->', szlil
        print 'scipy.sparse.lil_matrix()'
        kk = min(int(mat_lil.shape[0] * eigfrac), mat_lil.shape[0] - 2)
        print '(finding ', kk, ' eigs out of ', mat_lil.shape[0], ')'
        rlil, s_lil, l_lil = timing.timefunc(spla.eigs, mat_lil, kk, iterations=iterations)

        mat_csr = scipy.sparse.csr_matrix(mat)
        szcsr = asizeof.asizeof(mat_csr)
        print '\nmat_csr -->', szcsr
        print 'scipy.sparse.lil_matrix()'
        print '(finding ', kk, ' eigs out of ', mat_lil.shape[0], ')'
        rcsr, s_csr, l_csr = timing.timefunc(spla.eigs, mat_csr, kk, iterations=iterations)

        # Store results for compilation
        t_la.append(s_la)
        t_sla.append(s_sla)
        t_lil.append(s_lil)
        t_csr.append(s_csr)

        sz_mat.append(szmat)
        sz_lil.append(szlil)
        sz_csr.append(szcsr)

        nnlist.append(nn)

        # Look at spectra
        fig, ax = leplt.initialize_1panel_centered_fig()
        rla = np.sort(rla[0])
        rsla = np.sort(rsla[0])
        rlil = np.sort(rlil[0])
        rcsr = np.sort(rcsr[0])
        print 'rla = ', rla
        print 'np.shape(rla) = ', np.shape(rla)
        markers = leplt.get_markerstyles(4)
        colors = lecmap.husl_palette(4)
        ax.plot(np.arange(len(rla)), rla, '.-', color=colors[0], label='la', alpha=alpha, markersize=mksz)
        ax.plot(np.arange(len(rsla)), rsla, '^-', color=colors[1], label='sla', alpha=alpha, markersize=mksz)
        ax.plot(np.arange(len(rlil)), rlil, 'o-', color=colors[2], label='LIL', alpha=alpha, markersize=mksz)
        ax.plot(np.arange(len(rcsr)), rcsr, 's-', color=colors[3], label='CSR', alpha=alpha, markersize=mksz)
        ax.legend()
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue')
        ax.text(0.5, 1.1, 'Matrix Spectra', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(outdir + 'script_test_sparse_matrices_scaling_sz' + str(int(nh)) + '.png')
        plt.close('all')

        # save data
        res = {'s_la': s_la,
               's_sla': s_sla,
               's_lil': s_lil,
               's_csr': s_csr,
               'sz_mat': szmat,
               'sz_lil': szlil,
               'sz_csr': szcsr,
               'rla': rla,
               'rsla': rsla,
               'rlil': rlil,
               'rcsr': rcsr,
               'nn': nn,
               }
        with open(outdir + 'results_nh' + str(int(nh)) + 'pkl', "wb") as fn:
            pkl.dump(res, fn)

t_la = np.array(t_la)
t_sla = np.array(t_sla)
t_lil = np.array(t_lil)
t_csr = np.array(t_csr)
times = np.vstack((t_la, t_sla, t_lil, t_csr))
labels = ['la', 'sla', 'LIL', 'CSR']
print 'np.shape(times) = ', np.shape(times)

fig, ax = leplt.initialize_1panel_centered_fig()
markers = leplt.get_markerstyles(len(times))
colors = lecmap.husl_palette(len(times))
ii = 0
for timev in times:
    # ax.semilogy(nvec, timev, markers[ii] + '-', color=colors[ii])
    ax.loglog(nvec, timev, markers[ii] + '-', color=colors[ii])
    ii += 1

ax.legend(labels, loc='best')
title = 'Scaling of eigenvalue solvers'

ax.set_xlabel('Linear dimension of the network')
ax.set_ylabel('Time [s]')
ax.text(0.5, 1.1, title, ha='center', va='center', transform=ax.transAxes)
plt.savefig(outdir + 'script_test_sparse_matrices_scaling.png', dpi=300)
plt.close('all')

############################
# Look at matrix sizes
############################
sz_mat = np.array(sz_mat) / 1000.
sz_lil = np.array(sz_lil) / 1000.
sz_csr = np.array(sz_csr) / 1000.
szs = [sz_mat, sz_lil, sz_csr]

# Create figure
fig, ax = leplt.initialize_1panel_centered_fig()
labels = ['dense', 'LIL', 'CSR']
ii = 0
for ii in [0, 1, 2]:
    ax.loglog(nvec, szs[ii], markers[ii] + '-', color=colors[ii], label=labels[ii])

ax.legend(labels, loc='best')
title = 'Scaling of matrix size'
labels = ['dense', 'LIL', 'CSR']

ax.set_xlabel('Linear dimension of the network')
ax.set_ylabel('Size [kb]')
ax.text(0.5, 1.1, title, ha='center', va='center', transform=ax.transAxes)
plt.savefig(outdir + 'script_test_sparse_matrices_scaling_size.png', dpi=300)
plt.close('all')
