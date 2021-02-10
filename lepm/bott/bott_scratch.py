'''Notes for calculating the bott index'''

import matplotlib.pyplot as plt

fsfs = 20
testvals = [11, 13, 15, 19, 21]
result = np.zeros((len(testvals), 2), dtype=float)
print 'result = ', result

kk = 0
for ntest in testvals:
    lp['NH'] = ntest
    lp['NV'] = ntest
    lp['N'] = ntest
    lp['NP_load'] = ntest

    # Calc bott index for specified network
    # try:
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lat = lattice_class.Lattice(lp)
    lat.load()
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
    hlat = haldane_lattice_class.HaldaneLattice(lat, lp)
    hlat.load()
    print '\n\n\n', hlat.lattice.lp, '\n'
    bott = BottIndex(hlat, cp=cp)

    ######################
    # Testing
    omegac = cp['omegac']
    xy = hlat.lattice.xy

    # if pp is None:
    print 'Computing projector...'
    mat = hlat.get_matrix(attribute=True)

    # Check that the hamiltonian is hermitian
    # mmtc = mat - mat.conj().T
    # print 'mmtc = ', mmtc
    # le.plot_complex_matrix(mmtc, show=True, name=r'$|H|^2 - I$')

    pp = bfns.calc_projector(hlat, omegac, attribute=True)
    eigval, eigvect = hlat.get_eigval_eigvect(attribute=True)

    #######################################################
    if args.check:
        print 'min diff eigval = ', np.abs(np.diff(eigval)).min()
        close = np.where(np.abs(np.diff(eigval)) < 1e-14)[0]
        print 'diff eigval = ', np.diff(eigval)
        print 'close -> ', close
        plt.plot(np.arange(len(eigval)), np.real(eigval), 'ro-', label=r'Re($e_i$)')
        plt.plot(np.arange(len(eigval)), np.imag(eigval), 'b.-', label=r'Im($e_i$)')
        plt.legend()
        plt.xlabel('eigenvalue index', fontsize=fsfs)
        plt.ylabel('eigenvalue', fontsize=fsfs)
        plt.title('Eigenvalues of the Haldane Model', fontsize=fsfs)
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

    if args.check:
        # Check in a simpler way --> just look at magnitudes of the projected states
        print 'shape of sum(pp . eigvect) = ', np.shape(np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0))
        print 'shape of eigval = ', np.shape(np.abs(eigval))
        plt.plot(eigval, np.sum(np.abs(eigvect), axis=1), 'bo', label=r'$| |e \rangle |$')
        plt.plot(eigval, np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0), 'r.', label=r'$| P|e \rangle |$')
        plt.xlabel(r'$| e_i |$', fontsize=fsfs)
        plt.ylabel(r'$||e_i \rangle |$', fontsize=fsfs)
        plt.legend()
        plt.title(r'Check that $P |e \rangle \rightarrow |e \rangle$ or $|0\rangle$', fontsize=fsfs)
        plt.show()

        # Flip included and non-included states to check
        pp = np.identity(len(pp)) - pp

        # Check in a simpler way --> just look at magnitudes of the projected states
        print 'shape of sum(pp . eigvect) = ', np.shape(np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0))
        print 'shape of eigval = ', np.shape(np.abs(eigval))
        plt.plot(eigval, np.sum(np.abs(eigvect), axis=1), 'bo', label=r'$| |e\rangle |')
        plt.plot(eigval, np.sum(np.abs(np.dot(pp, eigvect.T)), axis=0), 'r.', label=r'$P|e \rangle $')
        plt.title(r'Check that $(1-P) |e \rangle \rightarrow |e \rangle$ or $|0\rangle$', fontsize=fsfs)
        plt.legend(fontsize=fsfs)
        plt.show()

        # flip back to regular projector
        pp = np.identity(len(pp)) - pp

    # Get U = P exp(iX) P and V = P exp(iY) P
    # rescale the position vectors in x and y
    xx = xy[:, 0] - np.min(xy[:, 0])
    yy = xy[:, 1] - np.min(xy[:, 1])
    xsize = np.sqrt(lat.PV[0][0] ** 2 + lat.PV[0][1] ** 2)
    ysize = np.sqrt(lat.PV[1][0] ** 2 + lat.PV[1][1] ** 2)
    theta = xx / xsize * 2. * np.pi
    phi = yy / ysize * 2. * np.pi

    if args.check:
        print 'theta = ', theta
        print 'exp theta = ', np.exp(1j * theta) * np.identity(len(xx))

    uu = np.dot(pp, np.dot(np.exp(1j * theta) * np.identity(len(xx)), pp))
    vv = np.dot(pp, np.dot(np.exp(1j * phi) * np.identity(len(xx)), pp))

    # Check
    if args.check:
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

    if args.check:
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

    if args.check:
        plt.plot(np.arange(len(ev)), np.real(ev), 'ro', label=r'Re $e$')
        plt.plot(np.arange(len(ev)), np.imag(ev), 'b.', label=r'Im $e$')
        plt.title(r'Eigenvalues of $VUV^{\dagger}U^{\dagger}$', fontsize=fsfs)
        plt.legend()
        plt.show()

    # Consider only eigvals near the identity
    ev = ev[np.abs(ev) > 1e-9]

    if args.check:
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

plt.plot(result[:, 0], result[:, 1], 'o-')
plt.xlabel(r'System size $L$', fontsize=fsfs)
plt.ylabel('Bott Index', fontsize=fsfs)
plt.title('Convergence?', fontsize=fsfs)
ax = plt.gca()
plt.tick_params(axis='both', labelsize=fsfs - 8)
plt.show()
