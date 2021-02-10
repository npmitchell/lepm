import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import lepm.lattice_elasticity as le
import lepm.lattice_class as lattice_class
from gyro_lattice_class import GyroLattice
import lepm.plotting.plotting as leplt
import lepm.dataio as dio
import lepm.plotting.colormaps as lecmaps
import lepm.plotting.movies as lemov
from lepm.gyro_data_handling import phase_minimizes_difference_complex_displacments
import glob
import subprocess
import sys

'''Auxiliary script-like functions called by gyro_lattice_class.py when __name__=='__main__' for samples twisted in
two periodic dimensions.
'''


def twistbcs(lp):
    """Load a periodic lattice from file, twist the BCs by phases theta_twist and phi_twist with vals finely spaced
    between 0 and 2pi. Then compute the berry curvature associated with |alpha(theta, phi)>

    Example usage:
    python haldane_lattice_class.py -twistbcs -N 3 -LT hexagonal -shape square -periodic

    Parameters
    ----------
    lp

    Returns
    -------

    """
    lp['periodicBC'] = True
    meshfn = le.find_meshfn(lp)
    lp['meshfn'] = meshfn
    lat = lattice_class.Lattice(lp)
    lat.load()
    # Make a big array of the eigvals: N x N x thetav x phiv
    # thetav = np.arange(0., 2. * np.pi, 0.5)
    # phiv = np.arange(0., 2. * np.pi, 0.5)

    # First just test for two values of theta and two of phi
    thetav = [0., 0.01]
    phiv = [0., 0.01]
    eigvects = {}
    ii = 0
    for theta in thetav:
        eigvects[ii] = {}
        jj = 0
        for phi in phiv:
            lpnew = copy.deepcopy(lp)
            lpnew['theta_twist'] = theta
            lpnew['phi_twist'] = phi
            hlat = HaldaneLattice(lat, lpnew)
            ev, evec = hlat.get_eigval_eigvect(attribute=True)
            if ii == 0 and jj == 0:
                eigval = copy.deepcopy(ev)
                ill = hlat.get_ill()
            eigvects[ii][jj] = evec
            jj += 1
        ii += 1

    # Dot the derivs of eigvects together
    # Ensure that there is a nonzero-amplitude wannier with matching phase
    dtheta = eigvects[1][0] - eigvects[0][0]
    dphi = eigvects[0][1] - eigvects[0][0]

    thetamov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
    if not glob.glob(thetamov_fn):
        # Plot differences
        fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                       colorV=ill, colormap='viridis',
                                                                       linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                       cbar_nticks=2,
                                                                       xlabel_pad=10, ylabel_pad=10,
                                                                       cbar_tickfmt='%0.3f')
        DOS_ax.set_title(r'$\partial_\phi \psi$')
        hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
        outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '/'
        dio.ensure_dir(outdir)
        for ii in range(len(ev)):
            fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                            dtheta, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                            marker_num=0, color_scheme='default', normalization=1.)
            print 'saving theta ', ii
            plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
            scat_fg.remove()
            scat_fg2.remove()
            p.remove()
            f_mark.remove()
            lines_12_st.remove()

        plt.close('all')

        imgname = outdir + 'dos_ev'
        movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev'
        lemov.make_movie(imgname, movname, rm_images=False)

    phimov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
    if not glob.glob(phimov_fn):
        fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                       colorV=ill, colormap='viridis',
                                                                       linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                       cbar_nticks=2,
                                                                       xlabel_pad=10, ylabel_pad=10,
                                                                       cbar_tickfmt='%0.3f')
        DOS_ax.set_title(r'$\partial_\phi \psi$')
        outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '/'
        dio.ensure_dir(outdir)
        for ii in range(len(ev)):
            fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                            dphi, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                            marker_num=0, color_scheme='default', normalization=1.)
            print 'saving phi ', ii
            plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
            scat_fg.remove()
            scat_fg2.remove()
            p.remove()
            f_mark.remove()
            lines_12_st.remove()

        plt.close('all')

        imgname = outdir + 'dos_ev'
        movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev'
        lemov.make_movie(imgname, movname, rm_images=False)

    # Check
    # print 'shape(dtheta) = ', np.shape(dtheta)
    # print 'shape(dphi) = ', np.shape(dphi)
    # le.plot_complex_matrix(dtheta, show=True, name='dtheta')
    # le.plot_complex_matrix(dphi, show=True, name='dphi')

    fig, ax = leplt.initialize_nxmpanel_fig(4, 1, wsfrac=0.6, x0frac=0.3)
    # < dphi | dtheta >
    dpdt = np.einsum('ij...,ij...->i...', dtheta, dphi.conj())
    # < dtheta | dphi >
    dtdp = np.einsum('ij...,ij...->i...', dphi, dtheta.conj())
    print 'dtdp = ', dtdp
    ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
    ax[1].plot(np.arange(len(dpdt)), dpdt, '-')
    hc = 2. * np.pi * 1j * (dpdt - dtdp)
    ax[2].plot(np.arange(len(lat.xy)), hc, '.-')
    # Plot cumulative sum
    sumhc = np.cumsum(hc)
    ax[3].plot(np.arange(len(lat.xy)), sumhc, '.-')

    ax[2].set_xlabel(r'Eigvect number')
    ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
    ax[2].set_xlabel(r'Eigvect number')
    ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
    ax[2].set_xlabel(r'Eigvect number')
    ax[2].set_ylabel(r'$c_H$')
    ax[3].set_xlabel(r'Eigvect number')
    ax[3].set_ylabel(r'$\sum_{E_\alpha < E_c} c_H$')
    outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/twistbc_test/'
    dio.ensure_dir(outdir)
    print 'saving ', outdir + 'test' + hlat.lp['meshfn_exten'] + '.png'
    plt.savefig(outdir + 'test' + hlat.lp['meshfn_exten'] + '.png')

    # ### Now do the same thing but with different values of theta, phi
    # First just test for two values of theta and two of phi
    thetav = [1., 1.01]
    phiv = [1., 1.01]
    eigvects = {}
    ii = 0
    for theta in thetav:
        eigvects[ii] = {}
        jj = 0
        for phi in phiv:
            lpnew = copy.deepcopy(lp)
            lpnew['theta_twist'] = theta
            lpnew['phi_twist'] = phi
            hlat = HaldaneLattice(lat, lpnew)
            ev, evec = hlat.get_eigval_eigvect(attribute=True)
            if ii == 0 and jj == 0:
                eigval = copy.deepcopy(ev)
                ill = hlat.get_ill()
            eigvects[ii][jj] = evec
            jj += 1
        ii += 1

    # Dot the derivs of eigvects together
    dtheta = eigvects[1][0] - eigvects[0][0]
    dphi = eigvects[0][1] - eigvects[0][0]

    # Plot differences
    thetamov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
    if not glob.glob(thetamov_fn):
        fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                       colorV=ill, colormap='viridis',
                                                                       linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                       cbar_nticks=2,
                                                                       xlabel_pad=10, ylabel_pad=10,
                                                                       cbar_tickfmt='%0.3f')
        DOS_ax.set_title(r'$\partial_\theta \psi$')
        hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
        outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '/'
        dio.ensure_dir(outdir)
        for ii in range(len(ev)):
            fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                            dtheta, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                            marker_num=0, color_scheme='default', normalization=1.)
            print 'saving theta ', ii
            plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
            scat_fg.remove()
            scat_fg2.remove()
            p.remove()
            f_mark.remove()
            lines_12_st.remove()

        plt.close('all')

        imgname = outdir + 'dos_ev'
        movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_theta' + hlat.lp['meshfn_exten'] + '_dos_ev'
        lemov.make_movie(imgname, movname, rm_images=False)

    # Now do phi
    phimov_fn = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev.mov'
    if not glob.glob(phimov_fn):
        fig, DOS_ax, eig_ax = leplt.initialize_eigvect_DOS_header_plot(ev, hlat.lattice.xy, sim_type='haldane',
                                                                       colorV=ill, colormap='viridis',
                                                                       linewidth=0, cax_label=r'$\xi^{-1}$',
                                                                       cbar_nticks=2,
                                                                       xlabel_pad=10, ylabel_pad=10,
                                                                       cbar_tickfmt='%0.3f')
        DOS_ax.set_title(r'$\partial_\phi \psi$')
        hlat.lattice.plot_BW_lat(fig=fig, ax=eig_ax, save=False, close=False, axis_off=False, title='')
        outdir = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '/'
        dio.ensure_dir(outdir)
        for ii in range(len(ev)):
            fig, [scat_fg, scat_fg2, p, f_mark, lines_12_st], cw_ccw = \
                hlatpfns.construct_haldane_eigvect_DOS_plot(hlat.lattice.xy, fig, DOS_ax, eig_ax, eigval,
                                                            dphi, ii, hlat.lattice.NL, hlat.lattice.KL,
                                                            marker_num=0, color_scheme='default', normalization=1.)
            print 'saving phi ', ii
            plt.savefig(outdir + 'dos_ev' + '{0:05}'.format(ii) + '.png')
            scat_fg.remove()
            scat_fg2.remove()
            p.remove()
            f_mark.remove()
            lines_12_st.remove()

        plt.close('all')

        imgname = outdir + 'dos_ev'
        movname = dio.prepdir(hlat.lp['meshfn']) + 'twistbc_phi' + hlat.lp['meshfn_exten'] + '_dos_ev'
        lemov.make_movie(imgname, movname, rm_images=False)

    # Check
    # print 'shape(dtheta) = ', np.shape(dtheta)
    # print 'shape(dphi) = ', np.shape(dphi)
    # le.plot_complex_matrix(dtheta, show=True, name='dtheta')
    # le.plot_complex_matrix(dphi, show=True, name='dphi')

    fig, ax = leplt.initialize_nxmpanel_fig(4, 1, wsfrac=0.6, x0frac=0.3)
    # < dphi | dtheta >
    dpdt = np.einsum('ij...,ij...->i...', dtheta, dphi.conj())
    # < dtheta | dphi >
    dtdp = np.einsum('ij...,ij...->i...', dphi, dtheta.conj())
    print 'dtdp = ', dtdp
    ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
    ax[1].plot(np.arange(len(dpdt)), dpdt, '-')
    hc = 2. * np.pi * 1j * (dpdt - dtdp)
    ax[2].plot(np.arange(len(lat.xy)), hc, '.-')
    # Plot cumulative sum
    sumhc = np.cumsum(hc)
    ax[3].plot(np.arange(len(lat.xy)), sumhc, '.-')

    ax[2].set_xlabel(r'Eigvect number')
    ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
    ax[2].set_xlabel(r'Eigvect number')
    ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
    ax[2].set_xlabel(r'Eigvect number')
    ax[2].set_ylabel(r'$c_H$')
    ax[3].set_xlabel(r'Eigvect number')
    ax[3].set_ylabel(r'$\sum_{E_\alpha < E_c} c_H$')
    outdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/twistbc_test/'
    dio.ensure_dir(outdir)
    plt.savefig(outdir + 'test' + hlat.lp['meshfn_exten'] + '_theta1p0.png')

    sys.exit()
    ########################################
    # Now test for more theta values, more phi values
    thetav = np.arange(0., 0.14, 0.1)
    phiv = np.arange(0., 0.14, 0.1)
    eigvects = np.zeros((len(lat.xy), len(lat.xy), len(thetav), len(phiv)), dtype=complex)
    ii = 0
    for theta in thetav:
        jj = 0
        for phi in phiv:
            lpnew = copy.deepcopy(lp)
            lpnew['theta_twist'] = theta
            lpnew['phi_twist'] = phi
            hlat = HaldaneLattice(lat, lpnew)
            ev, evec = hlat.get_eigval_eigvect()
            eigvects[:, :, ii, jj] = evec
            jj += 1
        ii += 1

    # Dot the derivs of eigvects together
    print 'eigvects = ', eigvects
    dtheta = np.diff(eigvects, axis=2)
    dphi = np.diff(eigvects, axis=3)
    print 'dtheta = ', dtheta

    print 'shape(dtheta) = ', np.shape(dtheta)
    print 'shape(dphi) = ', np.shape(dphi)
    le.plot_complex_matrix(dtheta[:, :, 0, 0], show=True)
    le.plot_complex_matrix(dphi[:, :, 0, 0], show=True)

    dtheta = dtheta[:, :, :, 0:np.shape(dtheta)[3] - 1]
    dphi = dphi[:, :, 0:np.shape(dphi)[2] - 1, :]
    print 'shape(dtheta) = ', np.shape(dtheta)
    print 'shape(dphi) = ', np.shape(dphi)

    for ii in range(np.shape(dtheta)[-1]):
        le.plot_complex_matrix(dtheta[:, :, ii, 0], show=True)

    fig, ax = leplt.initialize_nxmpanel_fig(3, 1)
    for ii in range(np.shape(dphi)[-1]):
        for jj in range(np.shape(dphi)[-1]):
            # < dphi | dtheta >
            dpdt = np.dot(dtheta[:, :, ii, jj], dphi[:, :, ii, jj].conj().T)
            # < dtheta | dphi >
            dtdp = np.dot(dphi[:, :, ii, jj], dtheta[:, :, ii, jj].conj().T)
            print 'np.shape(dpdt) = ', np.shape(dpdt)
            print 'np.shape(dtdp) = ', np.shape(dtdp)
            ax[0].plot(np.arange(len(dtdp)), dtdp, '-')
            ax[1].plot(np.arange(len(dpdt)), dpdt, '-')

            hc = 2. * np.pi * 1j * (dpdt - dtdp)
            ax[2].plot(np.arange(len(lat.xy)), hc, '.-')

    ax[0].set_xlabel(r'$\theta$')
    ax[0].set_ylabel(r'$\langle \partial_{\theta} \alpha_i | \partial_{\phi} \alpha_i \rangle$')
    ax[1].set_xlabel(r'$\phi$')
    ax[1].set_ylabel(r'$\langle \partial_{\phi} \alpha_i | \partial_{\theta} \alpha_i \rangle$')
    ax[2].set_xlabel(r'Eigvect number')
    ax[2].set_ylabel(r'$\partial_{\phi} \alpha_i$')
    plt.show()
    # hc = hlat.hallconductance_twist()