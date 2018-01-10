import lepm.lattice_elasticity as le
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import lepm.plotting.colormaps as lecmaps
import lepm.lattice_class
import lepm.plotting.plotting as leplt
import lepm.gyro_lattice_class
import glob
import sys
import copy
import subprocess
import cPickle as pkl
import lepm.plotting.movies as lemov
import lepm.dataio as dio

"""This script searches for ways of reducing the problem of finding the edges of a gap in a network to
finding the eigenvalues of a very small network.
"""

rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'

single_kag_eig = False
if single_kag_eig:
    # Test simgle mobile particle in kagome configuration
    # Numerical mean field theory, with A free to move (particles in unit cell are A,B,C
    #      C.
    #      / \
    #     /   \
    #    /     \
    # A .-------. B
    #
    datadir = rootdir + 'meanfield/kagome_A_Omk0p5/'
    if not glob.glob(datadir):
        dio.ensure_dir(datadir)

    R = np.array([[4.062499999999998890e-01, -1.605755436183646800e+00],
                  [-5.937500000000002220e-01, -1.605755436183646800e+00],
                  [4.062499999999998890e-01, 1.262953713852306425e-01],
                  [-5.937500000000002220e-01, 1.262953713852306425e-01],
                  [-9.375000000000001388e-02, -7.397300323992080928e-01]])

    NL = np.array([[4, 0, 0, 0],
                   [4, 0, 0, 0],
                   [4, 0, 0, 0],
                   [4, 0, 0, 0],
                   [3, 0, 1, 2]])

    KL = np.array([[1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 1, 1, 1]])

    OmK = -0.5 * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    # the one free particle is particle 4, so zero out 0 through (2*4-1)
    matrix[:8, :] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    # plt.plot(np.arange(len(eigval)), np.imag(eigval), 'b.-')
    # plt.show()
    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

trio_kag_eig = True
if trio_kag_eig:
    # Test trio of mobile particle in kagome configuration
    # Numerical mean field theory, with A free to move (particles in unit cell are A,B,C
    #           \     /
    #            \   /
    #             \ /
    #             C.
    #             / \
    #            /   \
    #           /     \
    # .----- A .-------. B -----.
    #         /         \
    #        /           \
    #       /             \
    #      .               .

    datadir = rootdir + 'meanfield/kagome_ABC_Omkn1p00/'
    if not glob.glob(datadir):
        dio.ensure_dir(datadir)

    R = np.array([[-0.5, 0.],
                  [0.5, 0.],
                  [0., np.sqrt(3)*0.5],
                  [-1.5, 0.],
                  [-1.0, -np.sqrt(3) * 0.5],
                  [1.0, -np.sqrt(3) * 0.5],
                  [2., 0.],
                  [-0.5, np.sqrt(3)],
                  [0.5, np.sqrt(3)]
                  ])

    NL = np.array([[1, 2, 3, 4],
                   [0, 2, 5, 6],
                   [0, 1, 7, 8],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [2, 0, 0, 0],
                   [3, 0, 0, 0]])

    KL = np.array([[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0]])

    OmK = -1.00 * KL
    Omg = -1.00 * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    # the one free particle is particle 4, so zero out 0 through (2*4-1)
    matrix[6:, :] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    # plt.plot(np.arange(len(eigval)), np.imag(eigval), 'b.-')
    # plt.show()
    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

sys.exit()

########################################################################################################################
########################################################################################################################
# Make a series of small periodic networks through bowtie transition, storing their eigenvals
########################################################################################################################
########################################################################################################################
small_periodic = False
if small_periodic:
    nlist = [1, 2, 3, 4, 6, 8, 10]
    kk = 0
    outdir = rootdir + 'meanfield/periodic_system_gaps/'
    le.ensure_dir(outdir)
    ymin = -1
    ymax = 4.1
    xmin = 0.5
    xmax = 1.5
    for N in nlist:
        plt.clf()
        delta_v = np.pi * np.arange(2./3., 1. + 1./3., 0.02)
        lp = {'LatticeTop': 'hexagonal',
              'NH': N,
              'NV': N,
              'shape': 'hexagon',
              'Omk': -1,
              'Omg': -1,
              'phi': 0.0,
              'eta': 0.0,
              'theta': 0.0,
              'make_slit': False,
              'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
              'check': False,
              'periodicBC': True}

        ii = 0
        eigvals = 'not_yet_defined'
        fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=190*0.75, wsfrac=0.75, hsfrac=0.75*0.75)

        pklout = outdir + 'eigvals_N{0:02d}'.format(N) + '.pkl'
        if glob.glob(pklout):
            with open(pklout, 'rb') as fn:
                eigvals = pkl.load(fn)
            # Draw lattices on plot to denote the deformation -- this could be sped up by not creating each lattice
            todo = [0, np.floor(len(delta_v) * 0.5) - 1, len(delta_v) - 1]
            for delta in delta_v[todo]:
                lp['delta'] = delta
                lat = lepm.lattice_class.Lattice(lp=lp)
                lat.build()

                lat2 = copy.deepcopy(lat)
                lat2.xy[:, 0] *= 0.07 / (float(N) + 1.)
                lat2.xy[:, 1] *= 0.07 / (float(N) + 1.) * ((ymax - ymin) / (xmax - xmin))
                for key in lat2.PVxydict:
                    lat2.PVxydict[key][0] *= 0.07 / (float(N) + 1.)
                    lat2.PVxydict[key][1] *= 0.07 / (float(N) + 1.) * ((ymax - ymin) / (xmax - xmin))
                lat2.xy += np.array([delta / np.pi, 0])
                le.draw_lattice(ax[0], lat2, bondcolor='k', colormap='BlueBlackRed', lw=0.2, climv=0.1)

        else:
            # Compute eigenvalues for each value of delta and draw a few lattices on plot to denote the deformation
            for delta in delta_v:
                lp['delta'] = delta
                lat = lepm.lattice_class.Lattice(lp=lp)
                lat.build()
                print 'lat.xy = ', lat.xy
                if ii in [0, np.floor(len(delta_v) * 0.5) - 1, len(delta_v) - 1]:
                    lat2 = copy.deepcopy(lat)
                    lat2.xy[:, 0] *= 0.07 / (float(N) + 1.)
                    lat2.xy[:, 1] *= 0.07 / (float(N) + 1.) / ((ymax - ymin) / (xmax - xmin))
                    for key in lat2.PVxydict:
                        lat2.PVxydict[key][0] *= 0.07 / (float(N) + 1.)
                        lat2.PVxydict[key][1] *= 0.07 / (float(N) + 1.) * ((ymax - ymin) / (xmax - xmin))

                    lat2.xy += np.array([delta/np.pi, 0])
                    le.draw_lattice(ax[0], lat2, bondcolor='k', colormap='BlueBlackRed', lw=0.2, climv=0.1)

                glat = lepm.gyro_lattice_class.GyroLattice(lat, lp=lp)
                eigval = glat.get_eigval()
                if eigvals == 'not_yet_defined':
                    eigvals = np.zeros((len(delta_v), int(len(eigval) * 0.5)))
                eigvals[ii, :] = np.imag(eigval)[np.imag(eigval) > 0]
                ii += 1

            # Save the eigenvalues
            with open(pklout, "wb") as fn:
                pkl.dump(eigvals, fn)

        ax[0].plot(delta_v/np.pi, eigvals, 'g-')
        ax[0].set_title('Eigenvalues of ' + str(N) + '-cell periodic system')
        ax[0].set_xlabel(r'Hexagon opening angle $\delta/\pi$')
        ax[0].set_ylabel(r'Eigenvalues $\omega/\Omega_g$')
        ax[0].set_xlim(xmin, xmax)
        ax[0].xaxis.set_ticks([0.5, 1.0, 1.5])
        ax[0].set_ylim(ymin, ymax)
        plt.savefig(outdir + 'frames/bowtie_transition_' + '{0:02d}'.format(kk) + '.png', dpi=300)
        kk += 1

    framerate = 1
    imgname = outdir + 'frames/bowtie_transition_'
    index_sz = 2
    movname = outdir + 'eigvals_bowtie_transition_smallperiodic'
    mov_exten = ''
    subprocess.call(['./ffmpeg', '-framerate', str(framerate), '-i', imgname + '%' + str(index_sz) + 'd.png',
                     movname + mov_exten + '.mov', '-vcodec',
                     'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '30',
                     '-pix_fmt', 'yuv420p'])


########################################
########################################
# Do hexagonal to bowtie transition as meanfield (2 particles)
########################################
########################################
bowtie_mf = True
if bowtie_mf:
    datadir = rootdir + 'meanfield/hexagonal_bowtie_transition/'
    le.ensure_dir(datadir)
    step = 0.001
    alph_vec = np.arange(-1./6., 1./6. + step, step)
    ii = 0
    eigvals = np.zeros((len(alph_vec), 1), dtype=float)
    for alphopi in alph_vec:
        alph = alphopi * np.pi
        beta = np.pi - alph
        gamma = np.pi + alph
        delta = 2. * np.pi - alph

        R = np.array([[0., 0.5],
                      [0., -0.5],
                      [np.cos(alph), 0.5 + np.sin(alph)],
                      [np.cos(beta), 0.5 + np.sin(beta)],
                      [np.cos(gamma), -0.5 + np.sin(gamma)],
                      [np.cos(delta), -0.5 + np.sin(delta)]])
        NL = np.array([[1, 2, 3],
                       [0, 4, 5],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])
        KL = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])

        OmK = -1. * KL
        Omg = -1. * np.ones_like(R[:, 0])
        matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

        matrix[4:, :] = 0
        matrix[:, 4:] = 0

        # Get eigvals, vects
        eigval, eigvect = le.eig_vals_vects(matrix)
        eigvals[ii] = np.min(np.imag(eigval[np.imag(eigval) > 0]))
        ii += 1

    plt.plot(1. - 2. * alph_vec, eigvals, '.-', color='k')

    vect = np.dstack((1. - 2. * alph_vec, eigvals.ravel()))[0]
    print 'vect = ', vect
    with open(datadir + 'hexagonalAB_bowtie_transition_lowerbound.pkl', "wb") as fn:
        pkl.dump(vect, fn)

    # Now add upper bound to plot
    ii = 0
    eigvals = np.zeros((len(alph_vec), 1), dtype=float)
    for alphopi in alph_vec:
        alph = alphopi * np.pi
        beta = np.pi - alph
        gamma = np.pi + alph
        delta = 2. * np.pi - alph
        R = np.array([[0., 0.],
                      [0., -1.],
                      [np.cos(alph), np.sin(alph)],
                      [np.cos(beta), np.sin(beta)]])
        NL = np.array([[1, 2, 3],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
        KL = np.array([[1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])

        OmK = -1. * KL
        Omg = -1. * np.ones_like(R[:, 0])
        matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)
        matrix[2:, :] = 0
        matrix[:, 2:] = 0

        # Get eigvals, vects
        eigval, eigvect = le.eig_vals_vects(matrix)
        eigvals[ii, :] = np.imag(eigval[np.imag(eigval) > 0])
        ii += 1

    plt.plot(1. - 2. * alph_vec, eigvals, '.-', color='k')

    plt.ylim(1.7, 2.6)
    plt.xlabel(r'Opening angle $\delta / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Bounds for band gap using mean field arguments')
    plt.savefig(datadir + 'hexagonalAB_bowtie_transition_bounds.pdf')
    plt.show()

    vect = np.dstack((1. - 2. * alph_vec, eigvals.ravel()))[0]
    print 'vect = ', vect
    with open(datadir + 'hexagonalAB_bowtie_transition_upperbound.pkl', "wb") as fn:
        pkl.dump(vect, fn)

########################################################################
########################################################################
# Check if mean field argument gives same bounds in bowtie transition as small periodic
########################################################################
########################################################################
compare_smallper_mf = False
if compare_smallper_mf:
    datadir = rootdir + 'meanfield/hexagonal_bowtie_transition/'
    # Do small periodic system (one cell)
    plt.clf()
    N = 1
    delta_v = np.pi * np.arange(2./3., 1. + 1./3., 0.02)
    lp = {'LatticeTop': 'hexagonal',
          'NH': N,
          'NV': N,
          'shape': 'hexagon',
          'Omk': -1,
          'Omg': -1,
          'phi': 0.0,
          'eta': 0.0,
          'theta': 0.0,
          'make_slit': False,
          'rootdir': '/Users/npmitchell/Dropbox/Soft_Matter/GPU/',
          'check': False,
          'periodicBC': True}

    ii = 0
    eigvals = 'not_yet_defined'
    fig, ax = leplt.initialize_1panel_centered_fig(Wfig=180, Hfig=190*0.75, wsfrac=0.75, hsfrac=0.75*0.75)

    # Compute eigenvalues for each value of delta
    for delta in delta_v:
        lp['delta'] = delta
        lat = lepm.lattice_class.Lattice(lp=lp)
        lat.build()
        glat = lepm.gyro_lattice_class.GyroLattice(lat, lp=lp)
        eigval = glat.get_eigval()
        if eigvals == 'not_yet_defined':
            eigvals = np.zeros((len(delta_v), int(len(eigval) * 0.5)))
        eigvals[ii, :] = np.imag(eigval)[np.imag(eigval) > 0]
        ii += 1

    ax[0].plot(delta_v/np.pi, eigvals, 'g-', lw=3)

    # Load pickles of upper/lower bounds via mean field argument
    with open(datadir + 'hexagonalAB_bowtie_transition_lowerbound.pkl', "rb") as fn:
        lowerbound = pkl.load(fn)
    with open(datadir + 'hexagonalAB_bowtie_transition_upperbound.pkl', "rb") as fn:
        upperbound = pkl.load(fn)

    plt.plot(lowerbound[:, 0], lowerbound[:, 1], 'k-')
    plt.plot(upperbound[:, 0], upperbound[:, 1], 'k-')

    ax[0].set_title('Eigenvalues of ' + str(N) + '-cell periodic system')
    ax[0].set_xlabel(r'Hexagon opening angle $\delta/\pi$')
    ax[0].set_ylabel(r'Eigenvalues $\omega/\Omega_g$')
    ax[0].xaxis.set_ticks([0.5, 1.0, 1.5])
    plt.savefig(datadir + 'compare_bowtie_transition_meanfield_smallperiodic.png', dpi=300)

########################################
########################################
# Do hexagonal to bowtie transition as meanfield (2 particles) with LINEAR CONSTRAINT
########################################
########################################
hex2bowtie_transition_2particle = False
if hex2bowtie_transition_2particle:
    datadir = rootdir + 'meanfield/hexagonal_bowtie_transition_linear_constraint/'
    le.ensure_dir(datadir)
    step = 0.001
    alph_vec = np.arange(-1./6., 1./6. + step, step)
    ii = 0
    eigvals = np.zeros((len(alph_vec), 1), dtype=float)
    for alphopi in alph_vec:
        alph = alphopi * np.pi
        beta = np.pi - alph
        gamma = np.pi + alph
        delta = 2. * np.pi - alph

        R = np.array([[0., 0.5],
                      [0., -0.5],
                      [np.cos(alph), 0.5 + np.sin(alph)],
                      [np.cos(beta), 0.5 + np.sin(beta)],
                      [np.cos(gamma), -0.5 + np.sin(gamma)],
                      [np.cos(delta), -0.5 + np.sin(delta)]])
        NL = np.array([[1, 2, 3],
                       [0, 4, 5],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])
        KL = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])

        OmK = -1. * KL
        Omg = -1. * np.ones_like(R[:, 0])
        matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

        matrix[4:, :] = 0
        matrix[:, 4:] = 0
        matrix[2, :] = 0
        matrix[:, 2] = 0

        # Get eigvals, vects
        eigval, eigvect = le.eig_vals_vects(matrix)
        eigvals[ii] = np.min(np.imag(eigval[np.imag(eigval) > 0]))
        ii += 1

        if np.mod(ii, np.round(len(alph_vec)*0.2)) == 0:
            outdir = datadir + 'linear_constr' + '_' + sf.float2pstr(alphopi) + '/'
            le.ensure_dir(outdir)
            le.plot_save_normal_modes_Nashgyro(outdir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                               rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    plt.plot(1. - 2. * alph_vec, eigvals, '.-', color='k')

    plt.xlabel(r'Opening angle $\delta / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Lower bounds for band gap using linear contraint (bowtie trans)')
    plt.savefig(datadir + 'hexagonalAB_bowtie_transition_linear_constraint.pdf')
    plt.show()

    sys.exit()

########################################################################
########################################################################
########################################################################
# Some test cases for analytical mean field theory
########################################################################
########################################################################
########################################################################
numerical_mean_field = False
if numerical_mean_field:
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)
    ca = np.cos(alph)
    sa = np.sin(alph)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    cd = np.cos(delta)
    sd = np.sin(delta)

    dm = np.array([[ca*sa + cb*sb, sa**2 + sb**2 + 2, 0, -1],
                   [-ca**2 - cb**2 - 1, -ca*sa - cb*sb, 0, 0],
                   [0, -1, cg*sg + cd*sd, sg**2 + sd**2 + 2],
                   [0, 0, -cg**2 - cd**2 - 1, -cg*sg - cd*sd]])
    eval, evect = la.eig(dm)
    print 'eval = ', eval
    # print 'evect = ', evect

    dm = np.array([[ca * sa + cb * sb, sa ** 2 + sb ** 2 + 2, 0, -1],
                   [-ca ** 2 - cb ** 2 - 1, -ca * sa - cb * sb, 0, 0],
                   [0, -1, cg * sg + cd * sd, sg ** 2 + sd ** 2 + 2],
                   [0, 0, -cg ** 2 - cd ** 2 - 1, -cg * sg - cd * sd]])
    eval, evect = la.eig(dm)
    print 'eval = ', eval
    # print 'evect = ', evect

    ##############################
    # DOS movie Numerical mean field theory for bowtie transition, with A, B free to move
    alph = np.pi * 0.
    beta = np.pi - alph
    gamma = np.pi + alph
    delta = np.pi * 2 - alph

    abstr = 'a' + sf.float2pstr(alph/np.pi) + '_b' + sf.float2pstr(beta/np.pi)
    gdstr = '_g' + sf.float2pstr(gamma/np.pi) + '_d' + sf.float2pstr(delta/np.pi)
    datadir = rootdir + 'meanfield/hexagonalAB_' + abstr + gdstr + '/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0.5],
                  [0., -0.5],
                  [np.cos(alph), 0.5 + np.sin(alph)],
                  [np.cos(beta), 0.5 + np.sin(beta)],
                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                  [np.cos(delta), -0.5 + np.sin(delta)]])
    NL = np.array([[1, 2, 3],
                   [0, 4, 5],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    matrix[4:, :] = 0
    matrix[:, 4:] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    ##################################################
    ##################################################
    # DOS movie Numerical mean field theory for bowtie transition, with A free to move
    alph = np.pi * 0.
    beta = np.pi - alph
    gamma = np.pi + alph
    delta = np.pi * 2 - alph

    abstr = 'a' + sf.float2pstr(alph/np.pi) + '_b' + sf.float2pstr(beta/np.pi)
    gdstr = '_g' + sf.float2pstr(gamma/np.pi) + '_d' + sf.float2pstr(delta/np.pi)
    datadir = rootdir + 'meanfield/hexagonalAz3_' + abstr + gdstr + '/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0],
                  [0., -1.],
                  [np.cos(alph), np.sin(alph)],
                  [np.cos(beta), np.sin(beta)]])
    NL = np.array([[1, 2, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    matrix[2:, :] = 0
    matrix[:, 2:] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    sys.exit()

    ##############################
    # Numerical mean field theory, with A, B free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    abstr = 'a' + sf.float2pstr(alph/np.pi) + '_b' + sf.float2pstr(beta/np.pi)
    gdstr = '_g' + sf.float2pstr(gamma/np.pi) + '_d' + sf.float2pstr(delta/np.pi)
    datadir = rootdir + 'meanfield/hexagonalAB_' + abstr + gdstr + '/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0.5],
                  [0., -0.5],
                  [np.cos(alph), 0.5 + np.sin(alph)],
                  [np.cos(beta), 0.5 + np.sin(beta)],
                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                  [np.cos(delta), -0.5 + np.sin(delta)]])
    NL = np.array([[1, 2, 3],
                   [0, 4, 5],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    matrix[4:, :] = 0
    matrix[:, 4:] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    ##############################
    # Numerical mean field theory, with A, B free to move and neighbors free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    datadir = rootdir + 'meanfield/hexagonalABz3free_' + abstr + gdstr + '/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0.5],
                  [0., -0.5],
                  [np.cos(alph), 0.5 + np.sin(alph)],
                  [np.cos(beta), 0.5 + np.sin(beta)],
                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                  [np.cos(delta), -0.5 + np.sin(delta)]])
    NL = np.array([[1, 2, 3],
                   [0, 4, 5],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    ##############################
    # Numerical mean field theory, with particle A free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    datadir = rootdir + 'meanfield/hexagonalAz3_' + abstr + '_g1p50/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0.],
                  [0., -1.],
                  [np.cos(alph), np.sin(alph)],
                  [np.cos(beta), np.sin(beta)]])
    NL = np.array([[1, 2, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    matrix[2:, :] = 0
    matrix[:, 2:] = 0
    print 'matrix = ', matrix

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    ##############################
    # Numerical mean field theory, with Az3 free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    datadir = rootdir + 'meanfield/hexagonalAz3free_' + abstr + '_g1p50/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    R = np.array([[0., 0.],
                  [0., -1.],
                  [np.cos(alph), np.sin(alph)],
                  [np.cos(beta), np.sin(beta)]])
    NL = np.array([[1, 2, 3],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    KL = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

    OmK = -1. * KL
    Omg = -1. * np.ones_like(R[:, 0])
    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

    # Get eigvals, vects
    print 'Finding eigenvals/vects of dynamical matrix...'
    eigval, eigvect = le.eig_vals_vects(matrix)
    # print 'eigvect = ', eigvect
    # print 'eigval = ', eigval

    le.plot_save_normal_modes_Nashgyro(datadir, R, NL, KL, OmK, Omg, params={}, sim_type='gyro',
                                       rm_images=False, gapims_only=False, eigval=eigval, eigvect=eigvect)

    ##############################
    # VARY ANGLES: Numerical mean field theory, with Az3 free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    datadir = rootdir + 'meanfield/hexagonalAz3_vary_angles/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    alph_vec = np.arange(-0.5, 0.500001, 0.05)
    beta_add_vec = np.arange(0., 1.0001, 0.05)
    lecmaps.register_colormaps()
    colormap = plt.get_cmap('husl_qual')

    kk = 0
    for beta_add in beta_add_vec:
        ii = 0
        eigvals = np.zeros((len(alph_vec), 1), dtype=float)
        for alphopi in alph_vec:
            alph = alphopi * np.pi
            ca = np.cos(alph)
            sa = np.sin(alph)

            beta = beta_add * (1.5 - alphopi) * np.pi + alph
            cb = np.cos(beta)
            sb = np.sin(beta)

            R = np.array([[0., 0.],
                          [0., -1.],
                          [np.cos(alph), np.sin(alph)],
                          [np.cos(beta), np.sin(beta)]])
            NL = np.array([[1, 2, 3],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
            KL = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]])

            OmK = -1. * KL
            Omg = -1. * np.ones_like(R[:, 0])
            matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)
            matrix[2:, :] = 0
            matrix[:, 2:] = 0

            # Get eigvals, vects
            eigval, eigvect = le.eig_vals_vects(matrix)
            eigvals[ii, :] = np.imag(eigval[np.imag(eigval) > 0])
            ii += 1

        color = colormap(float(kk)/len(beta_add_vec))
        plt.plot(alph_vec, eigvals, '.-', color=color)
        kk += 1

    plt.ylim(1.0, 3.0)
    plt.xlabel(r'Angle to first neighbor $\alpha / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Upper bound for gap')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(beta_add_vec), vmax=np.max(beta_add_vec)))
    # fake up the array of the scalar mappable.
    sm._A = []

    plt.colorbar(sm, label=r'$\beta/\pi$')
    print 'saving figure ' + datadir + 'hexagonalABz3_vary_angles' + '_vs_alpha.pdf'
    plt.savefig(datadir + 'hexagonalAz3_vary_angles' + '_vs_alpha.pdf')
    plt.show()


    plt.clf()

    # VARY BETA INSTEAD
    kk = 0
    for alphopi in alph_vec:
        alph = alphopi * np.pi
        ca = np.cos(alph)
        sa = np.sin(alph)

        ii = 0
        eigvals = np.zeros((len(beta_add_vec), 1), dtype=float)
        for beta_add in beta_add_vec:
            beta = beta_add * (1.5 - alphopi) * np.pi + alph
            cb = np.cos(beta)
            sb = np.sin(beta)

            R = np.array([[0., 0.],
                          [0., -1.],
                          [np.cos(alph), np.sin(alph)],
                          [np.cos(beta), np.sin(beta)]])
            NL = np.array([[1, 2, 3],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
            KL = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]])

            OmK = -1. * KL
            Omg = -1. * np.ones_like(R[:, 0])
            matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)
            matrix[2:, :] = 0
            matrix[:, 2:] = 0

            # Get eigvals, vects
            eigval, eigvect = le.eig_vals_vects(matrix)
            eigvals[ii, :] = np.imag(eigval[np.imag(eigval) > 0])
            ii += 1

        color = colormap(float(kk)/len(alph_vec))
        plt.plot(beta_add_vec, eigvals, '.-', color=color)
        kk += 1

    plt.ylim(1.0, 3.0)
    plt.xlabel(r'Angle to second neighbor $\beta / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Upper bound for gap')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(alph_vec), vmax=np.max(alph_vec)))
    # fake up the array of the scalar mappable.
    sm._A = []

    plt.colorbar(sm, label=r'$\alpha/\pi$')
    plt.savefig(datadir + 'hexagonalAz3_vary_angles' + '_vs_beta.pdf')
    plt.show()


    ##############################
    # VARY ANGLES AB: Numerical mean field theory, with ABz3 free to move
    alph = np.pi * 1./6.
    beta = np.pi * (1 - 1./6.)
    gamma = np.pi * (1. + 1./6.)
    delta = np.pi * (2 - 1./6.)

    datadir = rootdir + 'meanfield/hexagonalABz3_vary_angles/'
    if not glob.glob(datadir):
        le.ensure_dir(datadir)

    alph_vec = np.arange(-0.5, 0.5000001, 0.05)
    beta_add_vec = np.arange(0., 1.00001, 0.2)
    gamma_vec = np.arange(0.5, 2.50001, 0.2)
    delta_add_vec = np.arange(0., 1.00001, 0.2)
    lecmaps.register_colormaps()
    colormap = plt.get_cmap('husl_qual')

    kk = 0
    for delta_add in delta_add_vec:
        print 'finished with ', delta_add, ' of ', np.max(delta_add_vec)
        for gammaopi in gamma_vec:
            gamma = gammaopi * np.pi
            delta = delta_add * (2.5 - gammaopi)* np.pi + gamma
            for beta_add in beta_add_vec:
                ii = 0
                eigvals = np.zeros((len(alph_vec), 1), dtype=float)
                for alphopi in alph_vec:
                    alph = alphopi * np.pi
                    beta = beta_add * (1.5 - alphopi) * np.pi + alph

                    R = np.array([[0., 0.5],
                                  [0., -0.5],
                                  [np.cos(alph), 0.5 + np.sin(alph)],
                                  [np.cos(beta), 0.5 + np.sin(beta)],
                                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                                  [np.cos(delta), -0.5 + np.sin(delta)]])
                    NL = np.array([[1, 2, 3],
                                   [0, 4, 5],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])
                    KL = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])

                    OmK = -1. * KL
                    Omg = -1. * np.ones_like(R[:, 0])
                    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

                    matrix[4:, :] = 0
                    matrix[:, 4:] = 0

                    # Get eigvals, vects
                    eigval, eigvect = le.eig_vals_vects(matrix)
                    eigvals[ii] = np.min(np.imag(eigval[np.imag(eigval) > 0]))
                    ii += 1

                color = colormap(float(kk) / len(delta_add_vec))
                plt.plot(alph_vec, eigvals, '-', color=color)
        kk += 1

    plt.ylim(1.0, 3.0)
    plt.xlabel(r'Angle to first neighbor $\alpha / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Lower bound for gap')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(beta_add_vec), vmax=np.max(beta_add_vec)))
    # fake up the array of the scalar mappable.
    sm._A = []

    plt.colorbar(sm, label=r'$\beta/\pi$')
    print 'saving figure ' + datadir + 'hexagonalABz3_vary_angles' + '_vs_alpha.pdf'
    plt.savefig(datadir + 'hexagonalAz3_vary_angles' + '_vs_alpha.pdf')
    plt.show()


    plt.clf()

    # VARY BETA INSTEAD (ABz3)
    alph_vec = np.arange(-0.5, 0.5000001, 0.2)
    beta_add_vec = np.arange(0., 1.00001, 0.05)

    kk = 0
    for delta_add in delta_add_vec:
        print 'finished with ', delta_add, ' of ', np.max(delta_add_vec)
        for gammaopi in gamma_vec:
            gamma = gammaopi * np.pi
            delta = delta_add * (2.5 - gammaopi)* np.pi + gamma
            for alphopi in alph_vec:
                alph = alphopi * np.pi
                ca = np.cos(alph)
                sa = np.sin(alph)

                ii = 0
                eigvals = np.zeros((len(beta_add_vec), 1), dtype=float)
                for beta_add in beta_add_vec:
                    beta = beta_add * (1.5 - alph) * np.pi * 0.5 + alph
                    cb = np.cos(beta)
                    sb = np.sin(beta)

                    R = np.array([[0., 0.5],
                                  [0., -0.5],
                                  [np.cos(alph), 0.5 + np.sin(alph)],
                                  [np.cos(beta), 0.5 + np.sin(beta)],
                                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                                  [np.cos(delta), -0.5 + np.sin(delta)]])
                    NL = np.array([[1, 2, 3],
                                   [0, 4, 5],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])
                    KL = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])

                    OmK = -1. * KL
                    Omg = -1. * np.ones_like(R[:, 0])
                    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

                    matrix[4:, :] = 0
                    matrix[:, 4:] = 0

                    # Get eigvals, vects
                    eigval, eigvect = le.eig_vals_vects(matrix)
                    eigvals[ii] = np.min(np.imag(eigval[np.imag(eigval) > 0]))
                    ii += 1

                color = colormap(float(kk)/len(delta_add_vec))
                plt.plot(beta_add_vec, eigvals, '-', color=color)
        kk += 1

    plt.ylim(1.0, 3.0)
    plt.xlabel(r'Angle to second neighbor $\beta / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Lower bound for gap')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(alph_vec), vmax=np.max(alph_vec)))
    # fake up the array of the scalar mappable.
    sm._A = []

    plt.colorbar(sm, label=r'$\gamma/\pi$')
    plt.savefig(datadir + 'hexagonalABz3_vary_angles' + '_vs_beta.pdf')
    plt.clf()

########################################
########################################
# Color by alpha and do both upper and lower bounds
########################################
########################################
get_alpha_bounds = False
if get_alpha_bounds:
    alph_vec = np.arange(-0.5, 0.5000001, 0.2)
    beta_add_vec = np.arange(0., 1.00001, 0.05)
    gamma_vec = np.arange(0.5, 2.50001, 0.3)
    delta_add_vec = np.arange(0., 1.00001, 0.25)

    for delta_add in delta_add_vec:
        print 'finished with ', delta_add, ' of ', np.max(delta_add_vec)
        for gammaopi in gamma_vec:
            gamma = gammaopi * np.pi
            delta = delta_add * (2.5 - gammaopi) * np.pi + gamma
            kk = 0
            for alphopi in alph_vec:
                alph = alphopi * np.pi
                ca = np.cos(alph)
                sa = np.sin(alph)

                ii = 0
                eigvals = np.zeros((len(beta_add_vec), 1), dtype=float)
                for beta_add in beta_add_vec:
                    beta = beta_add * (1.5 - alph) * np.pi * 0.5 + alph
                    cb = np.cos(beta)
                    sb = np.sin(beta)

                    R = np.array([[0., 0.5],
                                  [0., -0.5],
                                  [np.cos(alph), 0.5 + np.sin(alph)],
                                  [np.cos(beta), 0.5 + np.sin(beta)],
                                  [np.cos(gamma), -0.5 + np.sin(gamma)],
                                  [np.cos(delta), -0.5 + np.sin(delta)]])
                    NL = np.array([[1, 2, 3],
                                   [0, 4, 5],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])
                    KL = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0],
                                   [1, 0, 0]])

                    OmK = -1. * KL
                    Omg = -1. * np.ones_like(R[:, 0])
                    matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)

                    matrix[4:, :] = 0
                    matrix[:, 4:] = 0

                    # Get eigvals, vects
                    eigval, eigvect = le.eig_vals_vects(matrix)
                    eigvals[ii] = np.min(np.imag(eigval[np.imag(eigval) > 0]))
                    ii += 1

                color = colormap(float(kk)/len(alph_vec))
                plt.plot(beta_add_vec, eigvals, '-', color=color)
                kk += 1

    # Now add upper bound to plot
    alph_vec = np.arange(-0.5, 0.50001, 0.05)
    beta_add_vec = np.arange(0., 1.00001, 0.05)
    kk = 0
    for alphopi in alph_vec:
        alph = alphopi * np.pi
        ca = np.cos(alph)
        sa = np.sin(alph)

        ii = 0
        eigvals = np.zeros((len(beta_add_vec), 1), dtype=float)
        for beta_add in beta_add_vec:
            beta = beta_add * (1.5 - alphopi) * np.pi + alph
            cb = np.cos(beta)
            sb = np.sin(beta)

            R = np.array([[0., 0.],
                          [0., -1.],
                          [np.cos(alph), np.sin(alph)],
                          [np.cos(beta), np.sin(beta)]])
            NL = np.array([[1, 2, 3],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
            KL = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0]])

            OmK = -1. * KL
            Omg = -1. * np.ones_like(R[:, 0])
            matrix = le.dynamical_matrix_gyros(R, NL, KL, OmK, Omg, params={}, PVxydict=None)
            matrix[2:, :] = 0
            matrix[:, 2:] = 0

            # Get eigvals, vects
            eigval, eigvect = le.eig_vals_vects(matrix)
            eigvals[ii, :] = np.imag(eigval[np.imag(eigval) > 0])
            ii += 1

        color = colormap(float(kk)/len(alph_vec))
        plt.plot(beta_add_vec, eigvals, '.-', color=color)
        kk += 1

    plt.ylim(1.0, 3.0)
    plt.xlabel(r'Angle to second neighbor $\beta / \pi$')
    plt.ylabel(r'Eigenvalues $\omega/\Omega_g$')
    plt.title(r'Bounds for band gap using mean field arguments')
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(alph_vec), vmax=np.max(alph_vec)))
    # fake up the array of the scalar mappable.
    sm._A = []

    plt.colorbar(sm, label=r'$\alpha/\pi$')
    plt.savefig(datadir + 'hexagonalAz3_vary_angles' + '_bounds.pdf')
    plt.show()

########################################
########################################
# Do hexagonal to bowtie transition as meanfield (2 particles)
########################################
########################################
hinge_mf = False
fsz = 18
if hinge_mf:
    colors = lecmaps.husl_palette(3, h=0.01, s=0.9, l=0.5)
    Omglist = [1., 10., 1000.]
    for omgval in Omglist:
        datadir = rootdir + 'meanfield/hinge/'
        energydatdir = rootdir + 'meanfield/hinge/energy_vs_angle_Omg{0:03.2f}'.format(omgval).replace('.', 'p') + '/'
        phasedatdir = rootdir + 'meanfield/hinge/Omg{0:03.2f}'.format(omgval).replace('.', 'p') + 'eigvectors_varyangle'
        le.ensure_dir(datadir)
        le.ensure_dir(energydatdir)
        step = 0.05
        theta_vec = np.arange(0., 2.*np.pi + step, step)
        ii = 0
        eigvals = np.zeros((len(theta_vec), 3), dtype=float)

        tstep = 0.002
        tt = np.arange(0, 2*np.pi + tstep, tstep)
        expi = np.exp(1j*tt)
        energyevecs = np.zeros((len(theta_vec), 6, len(expi)), dtype=float)
        phi1 = np.zeros((len(theta_vec), 3), dtype=float)
        phi2 = np.zeros((len(theta_vec), 3), dtype=float)
        # phi2 = np.zeros((len(theta_vec), 3), dtype=float)
        for theta in theta_vec:
            xy = np.array([[0., 0.],
                          [1., 0.],
                          [np.cos(theta), np.sin(theta)]])
            NL = np.array([[1, 2],
                           [0, 0],
                           [0, 0]])
            KL = np.array([[1, 1],
                           [1, 0],
                           [1, 0]])
            BL = le.NL2BL(NL, KL)
            OmK = -1. * KL
            Omg = -omgval * np.ones_like(xy[:, 0])
            matrix = le.dynamical_matrix_gyros(xy, NL, KL, OmK, Omg, params={}, PVxydict=None)

            # Get eigvals, vects
            eigval, eigvect = le.eig_vals_vects(matrix)
            ev = np.imag(eigval[np.imag(eigval) < 0])
            eigvals[ii] = np.abs(ev)

            # Make outdirs for eigenvector phase tracking
            phaseoutdirs = []
            for en in range(3):
                outdir = phasedatdir + '_' + str(en) + '/'
                le.ensure_dir(outdir)
                phaseoutdirs.append(outdir)

            # Compute the energy averaged over each cycle of the eigenvector
            jj = 0
            for evect in eigvect[0:3]:
                tmp = np.real(expi * evect.reshape(-1, 1))
                tmp += np.array([xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1]]).reshape(6, 1)
                dx1 = tmp[2, :] - tmp[0, :]
                dx2 = tmp[4, :] - tmp[0, :]
                dy1 = tmp[3, :] - tmp[1, :]
                dy2 = tmp[5, :] - tmp[1, :]
                l1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
                l2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
                nrg = ((l1 - 1) / l1)**2 + ((l2 - 1) / l2)**2
                energyevecs[ii, jj, :] = nrg

                # get angles of each gyro
                phase0 = np.mod(np.arctan2(np.real(evect[1]), np.real(evect[0])), 2. * np.pi)
                phase1 = np.mod(np.arctan2(np.real(evect[3]), np.real(evect[2])), 2. * np.pi)
                phase2 = np.mod(np.arctan2(np.real(evect[5]), np.real(evect[4])), 2. * np.pi)
                phi1[ii, jj] = np.mod(phase1 - phase0, 2. * np.pi)
                phi2[ii, jj] = np.mod(phase2 - phase0, 2. * np.pi)

                jj += 1

            if not glob.glob(energydatdir + 'energy_*' + '{0:04d}'.format(ii) + '.png'):
                # Plot energy over cycle
                plt.plot(tt / np.pi, energyevecs[ii].T, '.-')
                plt.gca().set_xlim(0, 2.)
                plt.xlabel(r'Angle in cycle, $\phi/\pi$', fontsize=fsz)
                plt.ylabel(r'Energy stored in bonds, $\omega$', fontsize=fsz)
                plt.title(r'Energy of a gyro hinge', fontsize=fsz)
                plt.savefig(energydatdir + 'energy_vs_theta_omg{0:03.3f}_'.format(-Omg[0]).replace('.', 'p') +
                            '{0:04d}'.format(ii) + '.png')
                plt.clf()

            # Save image sequence of the eigenvectors
            for en in range(3):
                if not glob.glob(phaseoutdirs[en] + 'eigvects_{0:05d}'.format(ii) + '.png'):
                    fig, dos_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(eigval, xy, sim_type='gyro')
                    le.movie_plot_2D(xy, BL, 0 * (BL[:, 0]), None, None, fig=fig, ax=eig_ax, NL=NL,
                                     KL=KL, PVx=None, PVy=None, climv=0.1, axcb=None, colorz=False,
                                     colormap='BlueBlackRed', bgcolor='#FFFFFF', axis_off=False)
                    leplt.construct_eigvect_DOS_plot(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, 'gyro',
                                                     NL, KL, marker_num=0, color_scheme='default', sub_lattice=-1)
                    dos_ax.set_xlim(omgval, omgval + 2.)
                    # leplt.plot_eigvect_excitation(xy, fig, dos_ax, eig_ax, eigval, eigvect, en, marker_num=0,
                    #                               black_t0lines=True, mark_t0=True, title='auto', normalization=1.,
                    #                               alpha=0.6, lw=1, zorder=10)
                    plt.savefig(phaseoutdirs[en] + 'eigvects_{0:05d}'.format(ii) + '.png')
                    plt.clf()

            ii += 1

        # Plot eigenvalues as function of angle
        for en in range(3):
            plt.plot(theta_vec/np.pi, eigvals[:, en], '.-', color=colors[en])
        plt.gca().set_xlim(0, 2.)
        plt.gca().set_ylim(-Omg[0]-1, -Omg[0] + 3*(-OmK[0, 0]))
        plt.xlabel(r'Hinge opening angle, $\theta/\pi$', fontsize=fsz)
        plt.ylabel(r'Eigenvalues, $\omega$', fontsize=fsz)
        plt.title(r'Eigenvalues of a gyro hinge', fontsize=fsz)
        plt.savefig(datadir + 'eigvalues_vs_theta_omg{0:03.3f}'.format(-Omg[0]).replace('.', 'p') + '.pdf')
        plt.clf()

        # Plot mean energy over a cycle as function of angle
        for en in range(3):
            plt.plot(theta_vec / np.pi, np.mean(energyevecs[:, en], axis=1), '.-', color=colors[en])
        plt.gca().set_xlim(0, 2.)
        plt.gca().set_ylim(ymin=-0.2)
        # plt.gca().set_ylim(-Omg[0] - 1, -Omg[0] + 3 * (-OmK[0, 0]))
        plt.xlabel(r'Hinge opening angle, $\theta/\pi$', fontsize=fsz)
        plt.ylabel(r'Average energy over a cycle, $\langle\omega \rangle$', fontsize=fsz)
        plt.title(r'Spring energies of a gyro hinge', fontsize=fsz)
        plt.savefig(datadir + 'energies_vs_theta_omg{0:03.3f}'.format(-Omg[0]).replace('.', 'p') + '.pdf')
        plt.clf()

        # Plot mean energy over a cycle as function of angle
        for en in range(3):
            plt.plot(theta_vec / np.pi, phi1[:, en], '.', color=colors[en])
            plt.plot(theta_vec / np.pi, phi2[:, en], '.', color=colors[en])

        plt.gca().set_xlim(0, 2.)
        # plt.gca().set_ylim(-Omg[0] - 1, -Omg[0] + 3 * (-OmK[0, 0]))
        plt.xlabel(r'Hinge opening angle, $\theta/\pi$', fontsize=fsz)
        plt.ylabel(r'Phase difference, $\Delta \phi$', fontsize=fsz)
        plt.title(r'Gyro phase differences', fontsize=fsz)
        plt.savefig(datadir + 'phases_vs_theta_omg{0:03.3f}'.format(-Omg[0]).replace('.', 'p') + '.pdf')
        plt.clf()

        # Make phase images into a movie
        for en in range(3):
            imgname = phaseoutdirs[en] + 'eigvects_'
            movname = phasedatdir + '_' + str(en)
            print 'Making movie: ', movname
            lemov.make_movie(imgname, movname, indexsz='05', framerate=10)