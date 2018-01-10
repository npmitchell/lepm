import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import numpy.linalg as la
import lepm.bott.bott_gyro_functions as bgfns
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
import lepm.gyro_lattice_class as gyro_lattice_class
from lepm.bott.bott_gyro import GyroBottIndex
import lepm.kitaev.kitaev_functions as kfns
import socket
import os
import copy
import lepm.plotting.plotting as leplt
import sys

'''Test out improvements to the BottIndex code in script format here'''

# check = True
check = False
addstr = 'randorg_gammakick0p50_cent'

# Use shorthand for U, V or compute from P = S M S^{-1}?
alternative = True
# alternative = False

# nsize = 64
nsize = 100
hostname = socket.gethostname()
if hostname[0:6] == 'midway':
    print '\n\nWe are on Midway!\n\n\n\n'
    rootdir = '/home/npmitchell/scratch-midway/'
    cprootdir = '/home/npmitchell/scratch-midway/'
elif hostname[0:10] == 'nsit-dhcp-':
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    cprootdir = '/Volumes/research4TB/Soft_Matter/GPU/'
elif hostname == 'Messiaen.local' or hostname[0:8] == 'wireless':
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
    if not os.path.isdir(cprootdir):
        cprootdir = '/Users/npmitchell/Desktop/data_local/GPU/'
elif hostname[0:5] == 'cvpn-':
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    cprootdir = '/Volumes/research2TB/Soft_Matter/GPU/'
else:
    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

lp0 = {'LatticeTop': 'randorg_gammakick0p50_cent',  # 'hexagonal',
       'shape': 'square',
       'NH': nsize,
       'NV': nsize,
       'NP_load': nsize,
       'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
       'phi_lattice': '0.000',
       'delta_lattice': '0.667',
       'theta': 0.,
       'eta': 0.,
       'x1': 0.,
       'x2': 0.,
       'x3': 0.,
       'z': 0.,
       'source': 'hexner',
       'loadlattice_number': 01,
       'check': check,
       'Ndefects': 0,
       'periodicBC': True,
       'V0_pin_gauss': 0.1,
       'V0_spring_gauss': 0.,
       'theta_twist': 0.0,
       'phi_twist': 0.0,
       'Omg': -1.0,
       'Omk': -1.0,
       'kicksz': -1.5,
       'spreading_time': 0.,
       'conf': 1,
       }

omegac = 2.25

cp = {'omegac': omegac,
      'basis': 'XY',
      'rootdir': cprootdir,
      }

fsfs = 20
tickfs = fsfs - 8

# check system size convergence
# testvals = [3, 5, 11, 13, 19, 21]
# check ABDelta
if lp0['LatticeTop'] != 'hexagonal':
    testvals = np.arange(0, 1.0, 0.05)
    # testvals = np.arange(0, 0.15, 0.1)
    # testvals = np.hstack((testvals, np.arange(0.15, 0.18, 0.01)))
    # testvals = np.hstack((testvals, np.arange(0.18, 0.25, 0.004)))
    # testvals = np.hstack((testvals, np.arange(0.3, 1.0, 0.2)))
else:
    testvals = np.arange(0, 0.15, 0.1)
    testvals = np.hstack((testvals, np.arange(0.15, 0.3, 0.005)))
    testvals = np.hstack((testvals, np.arange(0.3, 1.0, 0.2)))
print 'testvals = ', testvals

result = np.zeros((len(testvals), 2), dtype=float)
print 'result = ', result
kk = 0

# check system size:
# for ntest in testvals:
#     lp['NH'] = ntest
#     lp['NV'] = ntest
#     lp['N'] = ntest
#     lp['NP_load'] = ntest

# check ABDelta
for ntest in testvals:
    plt.close('all')
    lp = copy.deepcopy(lp0)

    lp['ABDelta'] = ntest

    # Calc bott index for specified network
    # try:
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lat = lattice_class.Lattice(lp)
    lat.load()
    lat.get_PV(attribute=True)
    # print 'lat.lp[PV] = ', lat.lp['PV']
    print 'lat.lp[periodicBC] = ', lat.lp['periodicBC']
    print 'lat.PV = ', lat.PV
    # except RuntimeError:
    #     print '\n\n Could not find lattice --> creating it!'
    #     meshfn, trash = le.build_meshfn(lp)
    #     lp['meshfn'] = meshfn
    #     lat = lattice_class.Lattice(lp)
    #     lat.build()
    #     lat.save()
    glat = gyro_lattice_class.GyroLattice(lat, lp)
    glat.load()
    print 't1 = ', glat.lp['Omg']
    print 't2 = ', glat.lp['Omk']
    print '\n\n\n', glat.lattice.lp, '\n'
    bott = GyroBottIndex(glat, cp=cp)

    ######################
    # Testing
    omegac = cp['omegac']
    xy = glat.lattice.xy

    # if pp is None:
    print 'Computing projector...'
    mat = glat.get_matrix(attribute=True)

    # Check that the hamiltonian is hermitian
    # mmtc = mat - mat.conj().T
    # print 'mmtc = ', mmtc
    # le.plot_complex_matrix(mmtc, show=True, name=r'$|H|^2 - I$')

    pp = kfns.calc_projector(glat, omegac, attribute=True)
    eigval, eigvect = glat.get_eigval_eigvect(attribute=True)

    #######################################################
    if lp['check']:
        print 'min diff eigval = ', np.abs(np.diff(eigval)).min()
        close = np.where(np.abs(np.diff(eigval)) < 1e-14)[0]
        print 'diff eigval = ', np.diff(eigval)
        print 'close -> ', close
        plt.plot(np.arange(len(eigval)), np.real(eigval), 'ro-', label=r'Re($e_i$)')
        plt.plot(np.arange(len(eigval)), np.imag(eigval), 'b.-', label=r'Im($e_i$)')
        plt.legend()
        plt.xlabel('eigenvalue index', fontsize=fsfs)
        plt.ylabel('eigenvalue', fontsize=fsfs)
        plt.title('Eigenvalues of the gyro network', fontsize=fsfs)
        plt.show()

        # Look at the matrix of eigenvectors
        ss = eigvect
        sum0 = np.abs(np.sum(np.abs(ss ** 2), axis=0))
        sum1 = np.abs(np.sum(np.abs(ss ** 2), axis=1))
        print 'sum0 = ', sum0
        print 'sum1 = ', sum1
        ss1 = ss.conj().T
        nearI = np.dot(ss, ss1)
        ii = 0
        eec = np.zeros(len(ss))
        eiej = np.zeros_like(ss)
        for evec in ss:
            eec[ii] = np.abs(np.sum(evec * evec.conj()))
            jj = 0
            for evec2 in ss:
                eiej[ii, jj] = np.abs(np.sum(evec * evec2.conj()))
                jj += 1
            ii += 1

        plt.plot(np.arange(len(eec)), eiej, '.')
        plt.xlabel('Eigenvector number', fontsize=fsfs)
        plt.ylabel(r'$\langle e_i | e_j \rangle$', fontsize=fsfs)
        plt.title(r'Orthogonality of eigenvectors $\langle e_i | e_j \rangle$', fontsize=fsfs)
        # plt.title(r'$e_i e^*_i$')
        plt.show()

        plt.plot(np.arange(len(sum0)), sum0, 'b.-', label='sum of cols')
        plt.plot(np.arange(len(sum0)), sum1, 'r.-', label='sum of rows')
        plt.title(r'$\sum_{i} |e_i^2|$', fontsize=fsfs)
        plt.legend(fontsize=fsfs)
        plt.show()

        le.plot_complex_matrix(ss, show=True, name=r'$S$', fontsize=fsfs)
        le.plot_complex_matrix(nearI - np.identity(len(ss)), show=True, name=r'$S S^{\dagger} - I$',
                               fontsize=fsfs)
    #######################################################

    # Check that projector is mapping states above to themselves
    # for ii in range(len(eigval)):
    #     if eigval[ii] < omegac:
    #         plt.plot(np.arange(len(eigvect[:, 0])), np.dot(pp, eigvect[ii, :]), 'b')
    #         plt.pause(0.01)
    #     else:
    #         plt.plot(np.arange(len(eigvect[:, 0])), np.dot(pp, eigvect[ii, :]) - eigvect[ii, :], 'r')
    #         plt.pause(0.01)
    #
    # plt.show()

    if lp['check']:
        # Check in a simpler way --> just look at magnitudes of the projected states
        print 'shape of sum(pp . eigvect) = ', np.shape(np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0))
        print 'shape of eigval = ', np.shape(np.abs(eigval))
        plt.plot(np.imag(eigval), np.sum(np.abs(eigvect), axis=1), 'bo', label=r'$| |e \rangle |$')
        plt.plot(np.imag(eigval), np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0), 'r.', label=r'$| P|e \rangle |$')
        plt.xlabel(r'$| e_i |$', fontsize=fsfs)
        plt.ylabel(r'$||e_i \rangle |$', fontsize=fsfs)
        plt.legend(loc='lower right', fontsize=fsfs)
        plt.title(r'Check that $P |e \rangle \rightarrow |e \rangle$ or $|0\rangle$', fontsize=fsfs)
        plt.show()

        # Flip included and non-included states to check
        pp = np.identity(len(pp)) - pp

        # Check in a simpler way --> just look at magnitudes of the projected states
        print 'shape of sum(pp . eigvect) = ', np.shape(np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0))
        print 'shape of eigval = ', np.shape(np.abs(eigval))
        plt.plot(np.imag(eigval), np.sum(np.abs(eigvect), axis=1), 'bo', label=r'$| |e\rangle |$')
        plt.plot(np.imag(eigval), np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0), 'r.', label=r'$P|e \rangle $')
        plt.title(r'Check that $(1-P) |e \rangle \rightarrow |e \rangle$ or $|0\rangle$', fontsize=fsfs)
        plt.legend(loc='lower right', fontsize=fsfs)
        plt.show()

        # flip back to regular projector
        pp = np.identity(len(pp)) - pp

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


    if alternative:
        # Alternative definition of U and V: just the mapped subspace
        ###############################################################
        # Get simple projector that only includes occupied states
        # (M x N)(N x N)
        # eigval, eigvect = glat.get_eigval_eigvect(attribute=True)

        # U = eigvect.transpose().conj()
        psub = eigvect[np.imag(eigval) > omegac]

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
        # if verbose:
        print 'diagonalizing (V U Vt Ut)...'

        if check:
            # Check equivalence of uu and uu0
            ueigs = la.eigvals(uu)
            si = np.argsort(np.real(ueigs))
            ueigs = ueigs[si]
            plt.plot(np.arange(len(uu)), ueigs, 'ro', label=r'$U \equiv \overline{e}_{\omega>E_F} e^{i \Theta} \overline{e}_'
                                                            r'{\omega>E_F}^{\dagger}$')

            # Get usual definitions and compare them by comparing eigenvalues
            uu0 = np.dot(pp, np.dot(np.exp(1j * theta) * np.identity(len(xx)), pp))
            vv0 = np.dot(pp, np.dot(np.exp(1j * phi) * np.identity(len(xx)), pp))

            u0eigs = la.eigvals(uu0)
            si = np.argsort(np.real(u0eigs))
            u0eigs = u0eigs[si]
            plt.plot(np.arange(len(uu0)), u0eigs, 'b.', label='$U \equiv P e^{i \Theta} P$')
            plt.ylabel(r'Eigenvalues of $U$', fontsize=fsfs)
            plt.xlabel(r'Eigenvalue Index', fontsize=fsfs)
            plt.legend(fontsize=fsfs, loc='lower right')
            plt.title('Comparison of shorthand $U$ vs full $U$', fontsize=fsfs)
            plt.tick_params(axis='both', labelsize=tickfs)
            plt.show()
    else:
        uu = np.dot(pp, np.dot(np.exp(1j * theta) * np.identity(len(xx)), pp))
        vv = np.dot(pp, np.dot(np.exp(1j * phi) * np.identity(len(xx)), pp))

    # Check
    if lp['check']:
        print 'shape U -> ', np.shape(uu)
        print 'shape V -> ', np.shape(vv)
        print 'shape P -> ', np.shape(pp)
        print 'PV = ', lat.PV
        print 'lat.PVx = ', lat.PVx
        lat.get_PVxyij(attribute=True)

        draw_hoppings = False
        if draw_hoppings:
            for ii in range(len(theta)):
                plt.plot(theta[ii], phi[ii], 'b.')
                plt.text(theta[ii] - 0.1, phi[ii] - 0.1, str(ii))
                # Draw hoppings from projector
                for jj in range(len(pp[ii])):
                    plt.plot([theta[ii], theta[ii] + lat.PVxij[ii, jj]], [phi[ii], phi[ii] + lat.PVyij[ii, jj]],
                             'k-', lw=np.abs(pp[ii, jj]))

            plt.show()

        le.plot_complex_matrix(pp, show=True, name='P', fontsize=fsfs)
        le.plot_complex_matrix(uu, show=True, name='U', fontsize=fsfs)
        le.plot_complex_matrix(vv, show=True, name='V', fontsize=fsfs)

        pp2 = np.dot(pp, pp) - pp
        print 'np.sum(np.abs(pp2)) = ', np.sum(np.abs(pp2))
        le.plot_complex_matrix(pp2, show=True, name=r'$P^2- P$', fontsize=fsfs)

        # dump into a pickle for troubleshooting
        data = {'pp': pp, 'uu': uu, 'vv': vv, 'xy': xy, 'eigval': eigval, 'eigvect': eigvect,
                'PV': lat.PV, 'PVxij': lat.PVxij, 'PVyij': lat.PVyij}
        fn = '/Users/npmitchell/Desktop/troubleshoot.pkl'
        with open(fn, 'w') as fn:
            pickle.dump(data, fn)

    # extract the lower diagonal part of the matrix?

    # Compute the Bott index

    # multiply the eigenvalues of VUUtVt to get the determinant of the product
    # Log Det = Trace Log
    # if Log e^A = A, then this holds. Wikipedia says this holds for Lie groups
    # Wikipedia Matrix Exponential > Jacobi's Formula says that
    # det e^A = e^{tr(A)}

    # Compute the Bott index
    mat = np.dot(vv, np.dot(uu, np.dot(vv.conj().T, uu.conj().T)))
    ev = la.eigvals(mat)

    if lp['check']:
        # Scatterplot the eigenvalues
        ax = plt.gca()
        ax.scatter(np.real(ev), np.imag(ev), alpha=0.3)
        ax.axis('scaled')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.plot([0, 0], ax.get_xlim(), 'k--')
        ax.plot(ax.get_xlim(), [0, 0], 'k--')
        ax.set_title('Eigenvalues of $VUV^{\dagger}U^{\dagger}$', fontsize=fsfs)
        ax.set_xlabel(r'Re$(e)$', fontsize=fsfs)
        ax.set_ylabel(r'Im$(e)$', fontsize=fsfs)
        plt.show()

    if lp['check']:
        plt.plot(np.arange(len(ev)), np.real(ev), 'ro', label=r'Re $e$')
        plt.plot(np.arange(len(ev)), np.imag(ev), 'b.', label=r'Im $e$')
        plt.title(r'Eigenvalues of $VUV^{\dagger}U^{\dagger}$', fontsize=fsfs)
        plt.legend()
        plt.show()

    # Consider only eigvals near the identity
    ev = ev[np.abs(ev) > 1e-9]

    if lp['check']:
        # Scatterplot the eigenvalues
        ax = plt.gca()
        ax.scatter(np.real(ev), np.imag(ev), alpha=0.3)
        ax.axis('scaled')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.plot([0, 0], ax.get_xlim(), 'k--')
        ax.plot(ax.get_xlim(), [0, 0], 'k--')
        ax.set_title('Eigenvalues of $VUV^{\dagger}U^{\dagger}$ (near zero removed)', fontsize=fsfs)
        ax.set_xlabel(r'Re$(e)$', fontsize=fsfs)
        ax.set_ylabel(r'Im$(e)$', fontsize=fsfs)
        plt.show()

    # Perhaps sensitive to errors during product
    # tr = np.prod(ev)
    tr = np.sum(np.log(ev))
    bott = np.imag(tr)
    # print 'tr = ', tr
    # print '2 pi bott = ', bott

    bott /= (2. * np.pi)
    print 'bott = ', bott

    result[kk, 0] = ntest
    result[kk, 1] = bott

    # from scipy.linalg import logm
    # print 'total bott = ', np.trace(logm(np.dot(vv, np.dot(uu, np.dot(vv.conj().T, uu.conj().T)))))
    # bott = np.imag(np.trace(logm(np.dot(vv, np.dot(uu, np.dot(vv.conj().T, uu.conj().T))))))
    ######################

    kk += 1

plt.close('all')
# print 'result = ', result
# sys.exit()
fig, ax = leplt.initialize_1panel_centered_fig()
ax.plot(result[:, 0], result[:, 1], 'o-', markersize=1, markeredgecolor=None)

# check system size
# plt.xlabel(r'System size $L$', fontsize=fsfs)
# plt.title('Convergence?', fontsize=fsfs)

# check ABDelta
ax.set_xlabel(r'$\Delta_{AB}$')
ax.set_title(r'Topological phase transition, $N=$' + str(lp['NH']))

ax.set_ylabel('Bott Index')
# plt.tick_params(axis='both', labelsize=tickfs)

addstr += '_vpin{0:0.3f}'.format(lp['V0_pin_gauss']).replace('.', 'p')
if lp['V0_pin_gauss'] > 1e-6:
    gyrodirty = 'gyro_dirty'
else:
    gyrodirty = 'gyro'

if alternative:
    name = '/Users/npmitchell/Desktop/temp/gyro_dirty/' + gyrodirty + '{0:02d}'.format(nsize) + '_shorthand' + addstr
else:
    name = '/Users/npmitchell/Desktop/temp/gyro_dirty/' + gyrodirty + '{0:02d}'.format(nsize) + addstr

plt.savefig(name + '_abtransition_n' + str(lp['NH']) + '.png')
if lp['LatticeTop'] == 'hexagonal':
    ax.set_xlim(0.15, 0.3)
    ax.xaxis.set_ticks([0.2, 0.3])
else:
    ax.set_xlim(0.15, 0.3)
    ax.xaxis.set_ticks([0.2, 0.5])

plt.savefig(name + '_vpin0p10_abtransition_zoom_n' + str(lp['NH']) + '.png')
# plt.show()
#
# bott.calc_bott(check=lp['check'], verbose=True)
# bott.save_bott()
